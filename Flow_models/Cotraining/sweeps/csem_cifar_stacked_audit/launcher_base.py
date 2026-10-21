#!/usr/bin/env python3
"""Eight RGB CIFAR direct-child cascade arms, optimized CIFAR recipe, 800 epochs."""
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
    if [int(r['cell_id']) for r in rows] != list(range(8)):
        raise ValueError('Expected eight fresh CIFAR direct-child sweep cells 0..7')
    common = ('dataset','epochs_joint','epochs_refine','T_K','T_full','csem_w','terminal_kl_w',
              'latent_anchor_mode','score_time_weighting','score_head_time_weighting',
              'lr_score_head','eval_every','eval_samples','cfg_scale','rk4_steps','tmin_band_weight')
    for k in common:
        if len({r[k] for r in rows}) != 1: raise ValueError(f'Unplanned recipe variation: {k}')
    for row in rows:
        if row['dataset']!='CIFAR' or int(row['epochs_joint'])!=800 or int(row['epochs_refine'])!=0:
            raise ValueError('Every arm is 800-epoch RGB CIFAR cotraining without refinement')
        if row['tmin_band_weight']!='matched': raise ValueError('All arms require matched B_t')
        if sum(split_ints(row['block_channels']))!=8 or sum(split_ints(row['score_depths']))!=12:
            raise ValueError('Keep 512 latent coordinates and 12 total score trunk layers')
        if int(row['blocks'])==2 and row['hierarchy_variant']!='direct_child_v1':
            raise ValueError('All cascades must use the new direct child baseline')
        if int(row['blocks'])==1 and row['hierarchy_variant']: raise ValueError('Flat control cannot have a hierarchy variant')
        if '/' in row['result_name'] or '..' in row['result_name']: raise ValueError('Invalid result directory name')
    if len({r['result_name'] for r in rows})!=8: raise ValueError('Result names must be unique')
    return rows

def split_ints(value):
    return [int(v) for v in value.split(':')]


def config_hash(row):
    files = ('core.py','suite.py','paper_io.py','hier_engine.py','hier_models.py',
             'direct_models.py','band_regularizer.py','hier_eval.py')
    root=Path(__file__).resolve().parent
    sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    science={k:v for k,v in row.items() if k not in {'notes','outputs','source_v5_result_name'}}
    return hashlib.sha256(json.dumps(dict(row=science,sources=sources),sort_keys=True).encode()).hexdigest()

def config_for(row, base, data_root=None):
    from suite import resolve_config
    cfg=resolve_config(row,Path(base))
    if data_root is not None: cfg['data_root']=str(Path(data_root).resolve())
    cfg.update(tmin_band_weight='matched',resolved_tmin_band_weight=float(cfg['score_w_vae']),
               latent_coordinates=512,architecture_label=row['architecture_label'],
               block_coordinates=split_ints(row['block_coordinates']))
    if int(row['blocks'])==2:
        cfg.update(block_channels=split_ints(row['block_channels']),score_depths=split_ints(row['score_depths']),
                   root_size=int(row['root_size']),block_spatial_sizes=split_ints(row['block_spatial_sizes']),
                   hierarchy_version=4,hierarchy_variant='direct_child_v1',
                   stochastic_channels=split_ints(row['stochastic_channels']),
                   training_engine='direct_child_conditional_csem_v1',
                   child_coordinates='direct Z2: posterior = diffused = regularized = decoded',
                   reconstruction_root='noisy; every decoder root feature recomputed')
    else: cfg['training_engine']='flat_csem_with_matched_band_v1'
    cfg['bundle_recipe_hash']=config_hash(row)
    return cfg

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    os.replace(tmp, path)


def status_path(base,row):
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
        if st.get('config_hash') not in (None,config_hash(row)) and (st.get('returncode')==0 or not restart):
            raise ValueError('Source/manifest changed; use --restart-incomplete for an unfinished cell')
        if st.get('returncode') == 0:
            if st.get('config_hash') != config_hash(row):
                raise ValueError('Completed cell has a different configuration')
            validate_evaluation(row, read_evaluation(dest/'dataframes'/'eval_metrics.csv'))
            print(f'[skip] completed {row["blocks"]}-block run')
            return 0
        if st.get('config_hash') not in (None,config_hash(row)) and not restart:
            raise ValueError('Source/manifest changed; restart incomplete cell into a fresh archived attempt')
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


def select_cells(spec,rows):
    if spec in ('all','main','missing'): return rows
    if spec in ('baseline','cascade'): return rows[:2]
    if spec=='controls': return rows[2:4]
    if spec=='paired': return rows[:4]
    if spec in ('new','variants'): return rows[4:]
    ids=set()
    for part in spec.split(','):
        if '-' in part:
            a,b=map(int,part.split('-',1));ids.update(range(a,b+1))
        else: ids.add(int(part))
    if ids-{int(r['cell_id']) for r in rows}: raise ValueError('Only cells 0..7 exist')
    return [r for r in rows if int(r['cell_id']) in ids]

def parse_slurm_job_id(output):
    """Read the final job-ID record, ignoring site banners and diagnostics."""
    for line in reversed(output.splitlines()):
        match = re.fullmatch(r'(?:Submitted batch job\s+)?([0-9]+)(?:;[^\s;]+)?', line.strip())
        if match:
            return match.group(1)
    raise RuntimeError(f'Cannot parse Slurm job ID: {output!r}. '
                       'sbatch succeeded, so check squeue before submitting again.')


def submit_jobs(base, spec='all', *, dry_run=False, restart=False, data_root=None,
                time_limit='24:00:00', partition='gh', account=None):
    selected = select_cells(spec, manifest(base))
    pending, active_jobs = [], None
    for row in selected:
        st = status(base, row)
        if st.get('config_hash') not in (None,config_hash(row)) and (st.get('returncode')==0 or not restart):
            raise ValueError('Source/manifest changed; use --restart-incomplete for an unfinished cell')
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
        cmd=['sbatch','--parsable',f'--job-name=cifar_direct_c{row["cell_id"]}_s{row["seed"]}',
             f'--time={time_limit}',f'--partition={partition}',f'--chdir={Path(base).resolve()}',
             '--export=ALL,'+','.join(f'{k}={v}' for k,v in exports.items())]
        if account: cmd.append(f'--account={account}')
        cmd.append(str(Path(base)/'job.slurm'))
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


def report(base,strict=False):
    return subprocess.call([sys.executable,str(Path(base)/'compile_results.py'),'--base-dir',str(base)]
                           + (['--strict'] if strict else []))

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
            p.add_argument('--time',default='24:00:00')
            p.add_argument('--partition',default='gh')
            p.add_argument('--account',default=None)
        if name == 'run':
            p.add_argument('--cell-id', type=int, required=True)
            p.add_argument('--resume', action='store_true'); p.add_argument('--restart', action='store_true')
        if name == 'report':
            p.add_argument('--strict',action='store_true')
        if name == 'smoke':
            p.add_argument('--blocks', type=int, choices=(1, 2), default=None)
            p.add_argument('--cell-id', type=int, choices=(0, 1, 3, 4, 5), default=None)
            p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    a = parser.parse_args(argv); base = a.base_dir.resolve()
    if a.command == 'validate':
        import tests
        return tests.run(base)
    if a.command == 'smoke':
        import tests
        return tests.torch_smoke(base,a.blocks,a.device,cell_id=a.cell_id)
    if a.command == 'prepare-data':
        prepare_data(base, a.data_root); return 0
    if a.command == 'submit':
        return submit_jobs(base,a.cells,dry_run=a.dry_run,restart=a.restart_incomplete,data_root=a.data_root,
                           time_limit=a.time,partition=a.partition,account=a.account)
    if a.command == 'run':
        return run_job(base, a.cell_id, resume=a.resume, restart=a.restart, data_root=a.data_root)
    return report(base,a.strict)


if __name__ == '__main__':
    raise SystemExit(main())
