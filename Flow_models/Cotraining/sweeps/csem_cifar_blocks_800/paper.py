#!/usr/bin/env python3
"""Six CIFAR CSEM cells: original 1/2/3 blocks plus three band-corrected runs, 800 epochs."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback


def manifest(base):
    with (Path(base)/'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if ([int(r['cell_id']) for r in rows] != list(range(6)) or
            [int(r['blocks']) for r in rows] != [1, 2, 3, 1, 2, 2] or
            any(r['dataset'] != 'CIFAR' or int(r['epochs_joint']) != 800 for r in rows)):
        raise ValueError('Expected six CIFAR-only 800-epoch cells, IDs 0 through 5')
    for row in rows:
        expected_band = 'matched' if int(row['cell_id']) >= 3 else ''
        expected_variant = 'optimized' if int(row['cell_id']) == 5 else ''
        if row.get('tmin_band_weight', '') != expected_band or row.get('hierarchy_variant', '') != expected_variant:
            raise ValueError('Unexpected band/architecture settings in manifest')
    return rows


def split_ints(value):
    return [int(v) for v in value.split(':')]


def config_hash(row):
    science = {k: v for k, v in row.items() if k not in {'notes', 'result_name', 'outputs', 'source_v5_result_name'}}
    # Empty extension columns preserve the original rows' existing status hashes.
    for key in ('tmin_band_weight', 'hierarchy_variant', 'block_spatial_sizes'):
        if not science.get(key):
            science.pop(key, None)
    return hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()


def config_for(row, base, data_root=None):
    from suite import resolve_config
    cfg = resolve_config(row, Path(base))
    if data_root is not None:
        cfg['data_root'] = str(Path(data_root).resolve())
    if int(row['blocks']) > 1:
        cfg.update(block_channels=split_ints(row['block_channels']),
                   score_depths=split_ints(row['score_depths']), hierarchy_version=1,
                   training_engine='conditional_csem_hierarchy_v1')
    if row.get('tmin_band_weight') == 'matched':
        cfg.update(tmin_band_weight='matched', resolved_tmin_band_weight=float(cfg['score_w_vae']))
        if int(row['blocks']) == 1:
            cfg['training_engine'] = 'flat_csem_with_matched_band_v1'
    if row.get('hierarchy_variant') == 'optimized':
        cfg.update(hierarchy_variant='optimized', block_spatial_sizes=split_ints(row['block_spatial_sizes']),
                   hierarchy_version=2, training_engine='conditional_residual_multiscale_csem_v2',
                   innovation_channels=[8, 6], latent_coordinates=512)
    return cfg


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    os.replace(tmp, path)


def status_path(base, row):
    return Path(base)/'status'/f'cell_{int(row["cell_id"]):03d}.json'


def status(base, row):
    p = status_path(base, row)
    return json.loads(p.read_text()) if p.is_file() else {}


def read_evaluation(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def fid_column(columns, *, level, steps=20, method='rk4'):
    prefix = f'fid_{method}_{int(steps)}_randtok_cfg{str(float(level)).replace(".", "_")}_temp1_initgaussianT'
    cols = [k for k in columns if k.startswith(prefix) and 'initgaussianTK' not in k]
    if len(cols) != 1:
        raise ValueError(f'Expected one full-T Gaussian FID at CFG={level}, {method}-{steps}: {cols}')
    return cols[0]


def validate_evaluation(row, values):
    if not values:
        raise ValueError('Missing evaluation history')
    endpoint = int(row['epochs_joint'])
    data = [v for v in values if v.get('tag') == 'LSI_Diff' and v.get('stage') == 'cotrain']
    seen = {int(float(v['epoch'])) for v in data}
    expected = set(range(int(row['eval_every']), endpoint+1, int(row['eval_every']))) | {endpoint}
    if not expected.issubset(seen):
        raise ValueError(f'Missing evaluated epochs: {sorted(expected-seen)}')
    final = max(data, key=lambda r: int(float(r['epoch'])))
    if int(float(final['epoch'])) != endpoint:
        raise ValueError('Run has not reached its 800-epoch endpoint')
    for level in (2.5, 3.0):
        for method, steps in (('rk4', 20), ('rk4', 25), ('heun', 20)):
            col = fid_column(final, level=level, steps=steps, method=method)
            if not math.isfinite(float(final[col])):
                raise ValueError(f'Nonfinite endpoint metric: {col}')
    if int(row['blocks']) > 1:
        for level in (2.5, 3.0):
            key = f'fid_encoded_root_generated_children_cfg{str(level).replace(".", "_")}'
            if not math.isfinite(float(final.get(key, 'nan'))):
                raise ValueError(f'Missing descendant-prior diagnostic: {key}')
    return final


def prepare_data(base, data_root=None):
    import core
    from paper_io import prepare_dataset
    root = Path(data_root).resolve() if data_root else Path(base)/'data_cache_v6'
    prepared = prepare_dataset(core.DATASET_INFO['CIFAR']['class'], root, 'CIFAR')
    print(f'CIFAR train/test data prepared: {prepared}', flush=True)
    return root


def run_job(base, cid, *, resume=False, restart=False, data_root=None):
    import core
    from suite import run_cell
    from paper_io import save_csv_atomic
    row = next((r for r in manifest(base) if int(r['cell_id']) == int(cid)), None)
    if row is None:
        raise ValueError(f'Unknown cell {cid}')
    sp = status_path(base, row); sp.parent.mkdir(exist_ok=True)
    with sp.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = status(base, row)
        dest = Path(base)/'results'/row['result_name']
        if st.get('returncode') == 0:
            if st.get('config_hash') != config_hash(row):
                raise ValueError('Completed cell has a different configuration')
            validate_evaluation(row, read_evaluation(dest/'dataframes'/'eval_metrics.csv'))
            print(f'[skip] completed {row["blocks"]}-block run')
            return 0
        ckpt = dest/'checkpoints'/'training_state_latest.pt'
        if dest.exists() and restart:
            archive = Path(base)/'failed_attempts'/f'{row["result_name"]}_{time.time_ns()}'
            archive.parent.mkdir(exist_ok=True); os.replace(dest, archive)
        elif dest.exists() and (not resume or not ckpt.is_file()):
            raise FileExistsError(f'Existing run needs --resume with a checkpoint, or --restart: {dest}')
        payload = dict(row, returncode=None, config_hash=config_hash(row),
                       slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                       started_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(sp, payload)
        try:
            if not core.torch.cuda.is_available():
                raise RuntimeError('CUDA GPU is required for the 800-epoch sweep')
            if not core.LPIPS_AVAILABLE:
                raise RuntimeError('lpips is required to preserve the CIFAR baseline loss')
            prepare_data(base, data_root)
            print(f'[RUN] CIFAR, blocks={row["block_channels"]}, DiT depths={row["score_depths"]}, '
                  '800 cotraining epochs; no frozen-VAE or refinement stage', flush=True)
            if int(row['blocks']) == 1 and data_root is None and not row.get('tmin_band_weight'):
                # Original flat route, original config resolver and original evaluator.
                loss, evaluation, cfg = run_cell(row, Path(base), resume=resume)
            else:
                cfg = config_for(row, base, data_root)
                if dest.exists():
                    cfg['paper_resume_state'] = str(ckpt)
                dest.mkdir(parents=True, exist_ok=True)
                atomic_json(dest/'paper_cell_config.json', cfg)
                core.seed_everything(int(cfg['seed']))
                if int(row['blocks']) == 1:
                    if cfg.get('tmin_band_weight') == 'matched':
                        from band_regularizer import train_flat_band
                        loss, evaluation = train_flat_band(cfg)
                    else:
                        loss, evaluation = core.train_vae_cotrained_cond(cfg)
                else:
                    from hier_engine import train_hierarchy
                    loss, evaluation = train_hierarchy(cfg)
                atomic_json(dest/'paper_cell_config.json', cfg)
            save_csv_atomic(loss, dest/'dataframes'/'loss_history.csv')
            save_csv_atomic(evaluation, dest/'dataframes'/'eval_metrics.csv')
            validate_evaluation(row, evaluation.to_dict('records'))
            payload.update(returncode=0, resolved_config=cfg,
                           loss_rows=len(loss), eval_rows=len(evaluation))
            rc = 0
        except Exception as exc:
            payload.update(returncode=1, error=repr(exc), traceback=traceback.format_exc())
            print(payload['traceback'], file=sys.stderr)
            rc = 1
        payload['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(sp, payload)
        return rc


def select_cells(spec, rows):
    if spec in ('all', 'main', 'missing'):
        return rows
    if spec in ('new', 'band'):
        return [r for r in rows if int(r['cell_id']) >= 3]
    if spec == 'baseline':
        return [r for r in rows if int(r['cell_id']) < 3]
    ids = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = map(int, part.split('-', 1)); ids.update(range(a, b+1))
        else:
            ids.add(int(part))
    if ids - {int(r['cell_id']) for r in rows}:
        raise ValueError('Only cells 0 through 5 exist')
    return [r for r in rows if int(r['cell_id']) in ids]


def parse_slurm_job_id(output):
    """Read the final job-ID record, ignoring site banners and diagnostics."""
    for line in reversed(output.splitlines()):
        match = re.fullmatch(r'(?:Submitted batch job\s+)?([0-9]+)(?:;[^\s;]+)?', line.strip())
        if match:
            return match.group(1)
    raise RuntimeError(f'Cannot parse Slurm job ID: {output!r}. '
                       'sbatch succeeded, so check squeue before submitting again.')


def submit_jobs(base, spec='all', *, dry_run=False, restart=False, data_root=None):
    selected = select_cells(spec, manifest(base))
    pending, active_jobs = [], None
    for row in selected:
        st = status(base, row)
        if st.get('returncode') == 0:
            if st.get('config_hash') != config_hash(row):
                raise ValueError('Completed cell configuration changed')
            validate_evaluation(row, read_evaluation(Path(base)/'results'/row['result_name']/'dataframes'/'eval_metrics.csv'))
            print(f'[skip] completed cell {row["cell_id"]}')
            continue
        if st.get('returncode') is None and st.get('slurm_job_id'):
            if active_jobs is None:
                q = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i'],
                                   capture_output=True, text=True, check=True)
                active_jobs = set(q.stdout.split())
            if str(st['slurm_job_id']) in active_jobs:
                print(f'[skip] active cell {row["cell_id"]}: job {st["slurm_job_id"]}')
                continue
        pending.append(row)
    print(f'Selected {len(pending)} training jobs. Dataset preparation runs inside them under a shared lock.')
    (Path(base)/'slurm_logs').mkdir(exist_ok=True)
    for row in pending:
        exports = {'CELL_ID': row['cell_id'], 'BASE_DIR': str(Path(base).resolve()), 'RESTART': str(int(restart))}
        if data_root:
            exports['CSEM_DATA_ROOT'] = str(Path(data_root).resolve())
        if any(',' in v or '\n' in v for v in exports.values()):
            raise ValueError('Slurm export values cannot contain commas or newlines')
        suffix = ('_optband' if row.get('hierarchy_variant') == 'optimized' else
                  '_band' if row.get('tmin_band_weight') else '')
        cmd = ['sbatch', '--parsable', f'--job-name=cifar_csem_b{row["blocks"]}{suffix}',
               '--export=ALL,'+','.join(f'{k}={v}' for k, v in exports.items()), str(Path(base)/'job.slurm')]
        print(shlex.join(cmd))
        if not dry_run:
            q = subprocess.run(cmd, cwd=base, text=True, capture_output=True, check=True)
            jobid = parse_slurm_job_id(q.stdout)
            current = status(base, row)
            if not (str(current.get('slurm_job_id')) == jobid and current.get('started_utc')):
                atomic_json(status_path(base, row), dict(row, returncode=None,
                            slurm_job_id=jobid, config_hash=config_hash(row),
                            queued_utc=datetime.now(timezone.utc).isoformat()))
    return 0


def report(base, level=2.5):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    compiled = Path(base)/'compiled'; compiled.mkdir(exist_ok=True)
    curves, endpoints = [], []
    for row in manifest(base):
        frames = Path(base)/'results'/row['result_name']/'dataframes'
        path = frames/'eval_metrics.csv'
        if not path.is_file():
            path = frames/'eval_metrics_in_progress.csv'
        values = read_evaluation(path)
        for item in values:
            if item.get('tag') != 'LSI_Diff' or item.get('stage') != 'cotrain':
                continue
            try:
                key = fid_column(item, level=level)
            except ValueError:
                continue
            if not math.isfinite(float(item.get(key, 'nan'))):
                continue
            diag = f'fid_encoded_root_generated_children_cfg{str(float(level)).replace(".", "_")}'
            record = dict(cell_id=int(row['cell_id']), model=row['outputs'], blocks=int(row['blocks']), channels=row['block_channels'],
                          band_weight=float(row['csem_w']) if row.get('tmin_band_weight') else 0.0,
                          hierarchy_variant=row.get('hierarchy_variant') or 'standard',
                          score_depths=row['score_depths'], epoch=int(float(item['epoch'])),
                          fid=float(item[key]), fid_posterior_recon=float(item.get('fid_vae_recon', 'nan')),
                          fid_encoded_root_generated_children=float(item.get(diag, 'nan')),
                          cfg=float(level), steps_per_block=20, seed=int(row['seed']),
                          completed=status(base, row).get('returncode') == 0)
            curves.append(record)
        block_rows = [c for c in curves if c['cell_id'] == int(row['cell_id'])]
        if block_rows:
            endpoints.append(max(block_rows, key=lambda r: r['epoch']))
    df = pd.DataFrame(curves); last = pd.DataFrame(endpoints)
    df.to_csv(compiled/f'curves_cfg{level:g}.csv', index=False)
    last.to_csv(compiled/f'latest_cfg{level:g}.csv', index=False)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for cell, group in df.groupby('cell_id'):
            group = group.sort_values('epoch')
            ax.plot(group.epoch, group.fid, marker='o', ms=3, label=group.iloc[0]['model'])
        ax.set(xlabel='Cotraining epoch', ylabel='Legacy Inception FID',
               title=f'CIFAR-10 CSEM: 512 latent coordinates, CFG={level:g}, RK4-20 per block')
        ax.legend(); fig.tight_layout()
        fig.savefig(compiled/f'fid_cfg{level:g}.png', dpi=180)
        fig.savefig(compiled/f'fid_cfg{level:g}.pdf'); plt.close(fig)
    print(last.to_string(index=False) if not last.empty else 'No measured results yet; no values inferred.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'submit', 'run', 'prepare-data', 'report', 'smoke'):
        p = commands.add_parser(name)
        p.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parent)
        if name in ('submit', 'run', 'prepare-data'):
            p.add_argument('--data-root', type=Path, default=os.environ.get('CSEM_DATA_ROOT') or None)
        if name == 'submit':
            p.add_argument('--cells', default='all'); p.add_argument('--dry-run', action='store_true')
            p.add_argument('--restart-incomplete', action='store_true')
        if name == 'run':
            p.add_argument('--cell-id', type=int, required=True)
            p.add_argument('--resume', action='store_true'); p.add_argument('--restart', action='store_true')
        if name == 'report':
            p.add_argument('--cfg', type=float, choices=(2.5, 3.0), default=2.5)
        if name == 'smoke':
            p.add_argument('--blocks', type=int, choices=(1, 2, 3), default=None)
            p.add_argument('--cell-id', type=int, choices=range(6), default=None)
            p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    a = parser.parse_args(argv); base = a.base_dir.resolve()
    if a.command == 'validate':
        import tests
        return tests.run(base)
    if a.command == 'smoke':
        import tests
        return tests.torch_smoke(base, a.blocks, a.device, cell_id=a.cell_id)
    if a.command == 'prepare-data':
        prepare_data(base, a.data_root); return 0
    if a.command == 'submit':
        return submit_jobs(base, a.cells, dry_run=a.dry_run, restart=a.restart_incomplete, data_root=a.data_root)
    if a.command == 'run':
        return run_job(base, a.cell_id, resume=a.resume, restart=a.restart, data_root=a.data_root)
    return report(base, a.cfg)


if __name__ == '__main__':
    raise SystemExit(main())
