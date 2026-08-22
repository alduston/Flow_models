#!/usr/bin/env python3
"""Run one frozen-checkpoint CIFAR oracle-score NFE mechanism cell."""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE_DIR = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0')
MANIFEST = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_manifest.csv'
TARGET = 'csem_oracle_nfe_mechanism_TK1p2_T1p6_v1.py'
RESULTS_ROOT = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_results'
LOG_ROOT = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_config_logs'
STATUS_ROOT = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_status'
N_CELLS = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cell(path: Path, cell_id: int) -> dict[str, str]:
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    hit = [r for r in rows if int(r['cell_id']) == cell_id]
    if len(hit) != 1:
        raise RuntimeError(f'Expected exactly one manifest row for cell {cell_id}, found {len(hit)}')
    return hit[0]


def completed(status_path: Path, result_dir: Path) -> bool:
    if not status_path.is_file() or not result_dir.is_dir():
        return False
    try:
        d = json.loads(status_path.read_text())
        return int(d.get('returncode', 999999)) == 0
    except Exception:
        return False


def tee_process(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', buffering=1) as log:
        log.write('COMMAND:\n')
        log.write(shlex.join(command) + '\n\n')
        proc = subprocess.Popen(
            command, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell-id', type=int, required=True)
    ap.add_argument('--base-dir', type=Path, default=BASE_DIR)
    args = ap.parse_args()

    if args.cell_id < 0 or args.cell_id >= N_CELLS:
        raise SystemExit(f'cell-id must be in 0..{N_CELLS - 1}, got {args.cell_id}')

    base = args.base_dir.resolve()
    manifest = base / MANIFEST
    target = base / TARGET
    if not manifest.is_file():
        raise SystemExit(f'Missing manifest: {manifest}')
    if not target.is_file():
        raise SystemExit(f'Missing target: {target}')
    row = load_cell(manifest, args.cell_id)

    source_root = base / row['source_results_root'] / row['source_result_name']
    source_ckpt = source_root / 'run_terminal_kl' / 'checkpoints'
    required_ckpts = [source_ckpt / 'vae_cotrained.pt', source_ckpt / 'unet_lsi.pt']
    missing = [str(p) for p in required_ckpts if not p.is_file()]
    if missing:
        raise SystemExit(
            '[error] Frozen source representation is unavailable. This mechanism sweep '\
            'does not retrain or silently substitute random weights. Missing:\n  ' + '\n  '.join(missing)
        )

    results_root = base / RESULTS_ROOT
    log_root = base / LOG_ROOT
    status_root = base / STATUS_ROOT
    for d in (results_root, log_root, status_root):
        d.mkdir(parents=True, exist_ok=True)

    result_dir = results_root / row['result_name']
    log_path = log_root / f"{row['result_name']}.log"
    status_path = status_root / f"{row['result_name']}.json"

    if completed(status_path, result_dir):
        print(f'[skip] cell {args.cell_id} already completed successfully.')
        return 0
    if result_dir.exists():
        raise SystemExit(
            f'[blocked] Partial/existing result directory: {result_dir}\n'
            'Rename/remove it explicitly before retrying this cell.'
        )

    print('=' * 100)
    print('CIFAR ORACLE-SCORE NFE MECHANISM SWEEP — FROZEN CHECKPOINT')
    print(f"cell                 = {args.cell_id}")
    print(f"representation       = {row['rep_id']}")
    print(f"role                 = {row['rep_role']}")
    print(f"source               = {source_root}")
    print(f"T_K / T              = {row['T_K']} / {row['T_full']}")
    print(f"csem_w / terminal KL = {row['csem_w']} / {row['terminal_kl_w']}")
    print(f"RK4 steps / NFE      = {row['rk4_steps']} / {row['rk4_nfe']}")
    print('mechanism modes      = oracle vs learned × q_h vs Gaussian × h in {T_K,T}')
    print('oracle field         = exact empirical class-conditional score, no CFG')
    print('training             = NONE; checkpoint is frozen')
    print('=' * 100)

    command = [
        sys.executable, '-u', str(target),
        '--dataset', 'CIFAR',
        '--model-preset', 'auto',
        '--arms', 'terminal_kl',
        '--score-time-weighting', 'canonical',
        '--score-head-time-weighting', 'unweighted-eps',
        '--csem-w', row['csem_w'],
        '--terminal-kl-w', row['terminal_kl_w'],
        '--score-head-loss-w', '1.0',
        '--T-terminal', row['T_K'],
        '--T', row['T_full'],
        # Parser contract requires epochs>=1, but evaluation-only mode returns
        # before the first optimizer step.
        '--epochs', '1',
        '--refine-epochs', '0',
        '--eval-every', '0',
        '--eval-samples', '1000',
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
        '--eval-oracle-diagnostics',
        '--eval-oracle-full-train-reference',
        '--oracle-profile-query-samples', row['oracle_profile_query_samples'],
        '--oracle-profile-time-points', row['oracle_profile_time_points'],
        '--oracle-profile-batch-size', '16',
        '--oracle-reference-batch-size', '2048',
        '--oracle-sampling-samples', row['oracle_sampling_samples'],
        '--oracle-sampling-batch-size', '16',
        '--oracle-sampling-steps', row['rk4_steps'],
        '--eval-oracle-transport-decomposition',
        '--no-eval-oracle-standard-samplers',
        '--evaluation-only-checkpoint-dir', str(source_ckpt),
        '--oracle-eval-epoch-label', '600',
        '--master-results-dir', str(result_dir),
    ]

    started = utc_now()
    t0 = time.monotonic()
    rc = tee_process(command, log_path, base)
    elapsed = time.monotonic() - t0

    payload = {
        'cell_id': args.cell_id,
        'rep_id': row['rep_id'],
        'rep_role': row['rep_role'],
        'source_results_root': row['source_results_root'],
        'source_result_name': row['source_result_name'],
        'source_checkpoint_dir': str(source_ckpt),
        'T_K': float(row['T_K']),
        'T_full': float(row['T_full']),
        'csem_w': float(row['csem_w']),
        'terminal_kl_w': float(row['terminal_kl_w']),
        'rk4_steps': int(row['rk4_steps']),
        'rk4_nfe': int(row['rk4_nfe']),
        'result_name': row['result_name'],
        'results_dir': str(result_dir),
        'log_path': str(log_path),
        'returncode': rc,
        'elapsed_seconds': elapsed,
        'started_utc': started,
        'finished_utc': utc_now(),
        'command': command,
    }
    status_path.write_text(json.dumps(payload, indent=2) + '\n')

    if rc == 0:
        print(f'[ok] cell {args.cell_id} completed in {elapsed/3600:.2f} h')
    else:
        print(f'[failed] cell {args.cell_id} rc={rc}')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
