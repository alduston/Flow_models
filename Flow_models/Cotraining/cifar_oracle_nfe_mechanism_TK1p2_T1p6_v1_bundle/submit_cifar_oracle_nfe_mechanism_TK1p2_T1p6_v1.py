#!/usr/bin/env python3
"""Submit the 20 oracle-NFE evaluation cells in groups of up to three.

Important: the comma-separated cell list is injected into sbatch's environment
and propagated with --export=ALL.  It is intentionally NOT embedded inside
--export, which would reproduce the earlier Slurm comma-parsing bug.
"""

from __future__ import annotations
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

BASE_DIR = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0')
MANIFEST = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_manifest.csv'
GRID = 'generate_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.py'
SLURM = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_cell_job.slurm'
TARGET = 'csem_oracle_nfe_mechanism_TK1p2_T1p6_v1.py'
LOG_DIR = 'slurm_logs_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1'
RECEIPT_DIR = 'submission_receipts_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1'
N_CELLS = 20
RUNS_PER_JOB = 3
ENV_NAME = 'CSEM_ORACLE_NFE_CELL_IDS'


def parse_cells(spec: str) -> list[int]:
    if spec.strip().lower() == 'all':
        return list(range(N_CELLS))
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            lo, hi = sorted((int(a), int(b)))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    cells = sorted(out)
    if not cells or any(c < 0 or c >= N_CELLS for c in cells):
        raise ValueError(f'Cells must be a nonempty subset of 0..{N_CELLS-1}; got {cells}')
    return cells


def chunked(values: list[int], n: int) -> list[list[int]]:
    return [values[i:i+n] for i in range(0, len(values), n)]


def parse_job_id(text: str) -> str | None:
    hits = re.findall(r'Submitted\s+batch\s+job\s+(\d+)', text, flags=re.I)
    if hits:
        return hits[-1]
    for line in reversed(text.splitlines()):
        m = re.fullmatch(r'\s*(\d+)(?:;[^\s]+)?\s*', line)
        if m:
            return m.group(1)
    return None


def ensure_manifest(base: Path) -> Path:
    path = base / MANIFEST
    if not path.is_file():
        proc = subprocess.run([sys.executable, str(base / GRID)], cwd=str(base))
        if proc.returncode:
            raise RuntimeError('Manifest generation failed.')
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != N_CELLS:
        raise RuntimeError(f'Expected {N_CELLS} manifest rows, found {len(rows)}')
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE_DIR)
    ap.add_argument('--cells', default='all', help='all, 0-19, or e.g. 0,2,5-8')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-source-preflight', action='store_true', help='Skip frozen-checkpoint existence checks (testing only).')
    args = ap.parse_args()

    base = args.base_dir.resolve()
    if not base.is_dir():
        raise SystemExit(f'Base directory not found: {base}')
    for name in (TARGET, SLURM):
        if not (base / name).is_file():
            raise SystemExit(f'Missing required file: {base / name}')

    manifest_path = ensure_manifest(base)
    cells = parse_cells(args.cells)

    # Fail before submitting anything if a requested frozen representation is
    # missing.  This is an evaluation sweep: silently retraining or falling back
    # to random weights would invalidate the mechanism comparison.
    if not args.skip_source_preflight:
        with manifest_path.open(newline='') as f:
            manifest_rows = list(csv.DictReader(f))
        selected = [r for r in manifest_rows if int(r['cell_id']) in set(cells)]
        missing = []
        checked = set()
        for row in selected:
            key = (row['source_results_root'], row['source_result_name'])
            if key in checked:
                continue
            checked.add(key)
            ckpt = base / row['source_results_root'] / row['source_result_name'] / 'run_terminal_kl' / 'checkpoints'
            for name in ('vae_cotrained.pt', 'unet_lsi.pt'):
                path = ckpt / name
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            raise SystemExit(
                '[error] Refusing to submit: frozen source checkpoint(s) missing:\n  ' +
                '\n  '.join(missing)
            )
        print(f'Frozen-checkpoint preflight OK for {len(checked)} representation(s).')

    groups = chunked(cells, RUNS_PER_JOB)

    log_dir = base / LOG_DIR
    receipt_dir = base / RECEIPT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipts = []
    print(
        f'Submitting {len(cells)} oracle-NFE cells as {len(groups)} ordinary jobs '
        f'(up to {RUNS_PER_JOB} sequential evaluations/job); no arrays.'
    )

    for group_idx, group in enumerate(groups):
        cell_spec = ','.join(str(c) for c in group)
        range_tag = f'{group[0]:02d}_{group[-1]:02d}'
        out = log_dir / f'oracle_nfe_cells_{range_tag}_%j.out'

        # CRITICAL: pass the comma-containing value through the process
        # environment.  Slurm's --export parser treats commas as separators.
        submit_env = os.environ.copy()
        submit_env[ENV_NAME] = cell_spec
        cmd = [
            'sbatch',
            '-J', f'confe{range_tag}',
            '-o', str(out),
            '--export=ALL',
            str(base / SLURM),
        ]
        display_cmd = f'{ENV_NAME}={cell_spec} ' + ' '.join(cmd)
        print('\n$ ' + display_cmd)

        if args.dry_run:
            job_id, rc, raw = 'DRY_RUN', 0, ''
        else:
            proc = subprocess.run(
                cmd, cwd=str(base), text=True, capture_output=True, env=submit_env
            )
            raw = (proc.stdout or '') + (proc.stderr or '')
            if raw:
                print(raw, end='' if raw.endswith('\n') else '\n')
            rc = proc.returncode
            if rc != 0:
                raise RuntimeError(f'sbatch failed for cells {cell_spec}: rc={rc}')
            job_id = parse_job_id(raw)
            if job_id is None:
                raise RuntimeError(
                    'sbatch returned success but job ID could not be parsed. '
                    'Refusing to continue so the sweep cannot be double-submitted.'
                )

        receipts.append({
            'group_index': group_idx,
            'cell_ids': cell_spec,
            'n_cells': len(group),
            'job_id': job_id,
            'returncode': rc,
            'submitted_at': datetime.now().isoformat(),
            'command': display_cmd,
        })
        print(f'Recorded cells {cell_spec} -> {job_id}')

    receipt = receipt_dir / ('submission_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
    with receipt.open('w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['group_index','cell_ids','n_cells','job_id','returncode','submitted_at','command'],
        )
        w.writeheader()
        w.writerows(receipts)
    print(f'\nWrote {receipt}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
