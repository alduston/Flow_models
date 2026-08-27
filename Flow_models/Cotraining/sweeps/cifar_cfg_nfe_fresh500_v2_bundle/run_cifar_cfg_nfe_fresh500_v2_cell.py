#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep')
MANIFEST = 'cifar_cfg_nfe_fresh500_v2_manifest.csv'
TARGET = 'csem_cfg_nfe_fresh500_v2.py'
RESULTS_ROOT = 'cifar_cfg_nfe_fresh500_v2_results'
LOG_ROOT = 'cifar_cfg_nfe_fresh500_v2_logs'
STATUS_ROOT = 'cifar_cfg_nfe_fresh500_v2_status'

def utc_now(): return datetime.now(timezone.utc).isoformat()

def load_row(path: Path, cell_id: int):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    hits = [r for r in rows if int(r['cell_id']) == cell_id]
    if len(hits) != 1:
        raise RuntimeError(f'Expected exactly one manifest row for cell {cell_id}, found {len(hits)}')
    return hits[0]

def tee(cmd, log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic(); started = utc_now()
    with log_path.open('w', buffering=1) as log:
        log.write('COMMAND:\n' + shlex.join(cmd) + '\n\n')
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            log.write(line); sys.stdout.write(line); sys.stdout.flush()
        rc = p.wait()
    return rc, time.monotonic() - t0, started, utc_now()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell-id', type=int, required=True)
    ap.add_argument('--base-dir', type=Path, default=BASE)
    ap.add_argument('--python', default=None, help='Python executable; defaults to current interpreter.')
    args = ap.parse_args()

    base = args.base_dir.resolve()
    row = load_row(base / MANIFEST, args.cell_id)
    target = base / TARGET
    result_root = base / RESULTS_ROOT / row['result_name']
    expected = result_root / 'run_terminal_kl' / 'dataframes' / f"deployment_grid_ep{row['epochs']}.csv"
    status_path = base / STATUS_ROOT / f"{row['result_name']}.json"

    if expected.is_file() and status_path.is_file():
        try:
            if int(json.loads(status_path.read_text()).get('returncode', 999)) == 0:
                print(f'[skip] complete: {row["result_name"]}')
                return 0
        except Exception:
            pass
    if result_root.exists():
        raise SystemExit(
            f'[blocked] Partial/existing result directory: {result_root}\n'
            'Rename/remove only this seed result before retrying. The runner never overwrites a partial 500-epoch training.'
        )

    py = args.python or sys.executable
    cmd = [
        py, '-u', str(target),
        '--dataset', 'CIFAR',
        '--model-preset', 'auto',
        '--arms', 'terminal_kl',
        '--seed', row['seed'],
        '--score-time-weighting', 'canonical',
        '--score-head-time-weighting', 'unweighted-eps',
        '--csem-w', row['csem_w'],
        '--terminal-kl-w', row['terminal_kl_w'],
        '--score-head-loss-w', '1.0',
        '--T-terminal', row['T_K'],
        '--T', row['T_full'],
        '--epochs', row['epochs'],
        '--refine-epochs', row['refine_epochs'],
        '--eval-every', '0',
        '--eval-samples', row['eval_samples'],
        '--cfg-strength', '3.0',
        '--canonical-lr-scale', '1.0',
        '--encoder-score-warmup-epochs', '0',
        '--csem-ramp-epochs', '0',
        '--score-tracking-steps', '0',
        '--grad-diagnostics-every', '0',
        '--logvar-min=-30.0',
        '--logvar-max', '20.0',
        '--no-fail-on-nonfinite',
        '--no-bespoke-fid-classifier',
        # Critical design: deployment-only changes the final evaluator, but because
        # NO --evaluation-only-checkpoint-dir is supplied, this invocation trains
        # from random initialization for 500 epochs and evaluates the in-memory EMA.
        '--deployment-only',
        '--deployment-cfg-grid', row['cfg_grid'],
        '--deployment-temperature-grid', row['temperature'],
        '--deployment-rk4-step-grid', row['rk4_steps'],
        '--skip-lsi-gap',
        '--no-save-eval-sample-panels',
        '--no-eval-oracle-diagnostics',
        '--no-eval-oracle-transport-decomposition',
        '--master-results-dir', str(result_root),
    ]

    print('=' * 100)
    print('CIFAR CFG x NFE FRESH-TRAIN AUDIT V2')
    print(f"cell/seed            = {row['cell_id']} / {row['seed']}")
    print(f"best geometry        = T_K={row['T_K']}, T={row['T_full']}, DeltaT={row['delta_T']}")
    print(f"weights              = wC={row['csem_w']}, wK={row['terminal_kl_w']}")
    print(f"training             = {row['epochs']} joint epochs, {row['refine_epochs']} refine")
    print(f"evaluation samples   = {row['eval_samples']} per config")
    print(f"CFG grid             = {row['cfg_grid']}")
    print(f"RK4 steps / NFE      = {row['rk4_steps']} / {row['nfe_grid']}")
    print(f"temperature          = {row['temperature']} (fixed; NOT swept)")
    print('checkpoint reload    = NONE; final grid uses fresh in-memory EMA')
    print('=' * 100)

    rc, elapsed, started, finished = tee(cmd, base / LOG_ROOT / f"{row['result_name']}.log", base)
    payload = {
        'returncode': rc,
        'cell_id': int(row['cell_id']), 'seed': int(row['seed']),
        'T_K': float(row['T_K']), 'T_full': float(row['T_full']), 'delta_T': float(row['delta_T']),
        'csem_w': float(row['csem_w']), 'terminal_kl_w': float(row['terminal_kl_w']),
        'epochs': int(row['epochs']), 'refine_epochs': int(row['refine_epochs']),
        'eval_samples': int(row['eval_samples']), 'cfg_grid': row['cfg_grid'],
        'rk4_steps': row['rk4_steps'], 'nfe_grid': row['nfe_grid'], 'temperature': float(row['temperature']),
        'result_dir': str(result_root), 'expected_table': str(expected),
        'elapsed_seconds': elapsed, 'started_utc': started, 'finished_utc': finished,
        'command': cmd,
        'checkpoint_reload_used': False,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2) + '\n')
    if rc == 0 and not expected.is_file():
        print(f'[error] process returned 0 but expected deployment table is missing: {expected}')
        return 3
    return rc

if __name__ == '__main__':
    raise SystemExit(main())
