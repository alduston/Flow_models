#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

from csem_paper_suite_v1 import run_cell

MANIFEST = 'csem_paper_final_v1_manifest.csv'
RESULT_ROOT = 'csem_paper_results_v1'
STATUS_ROOT = 'csem_paper_status_v1'
LOG_ROOT = 'csem_paper_config_logs_v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell-id', type=int, required=True)
    ap.add_argument('--base-dir', type=Path, default=Path.cwd())
    a = ap.parse_args()
    base = a.base_dir.resolve()
    with (base / MANIFEST).open(newline='') as f:
        rows = list(csv.DictReader(f))
    by_id = {int(r['cell_id']): r for r in rows}
    if a.cell_id not in by_id:
        raise SystemExit(f'Unknown cell id {a.cell_id}; valid={sorted(by_id)}')
    row = by_id[a.cell_id]
    status_dir = base / STATUS_ROOT
    log_dir = base / LOG_ROOT
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"cell_{a.cell_id:03d}_{row['result_name']}.json"
    result_dir = base / RESULT_ROOT / row['result_name']

    if status_path.is_file() and result_dir.is_dir():
        try:
            old = json.loads(status_path.read_text())
            if old.get('returncode') == 0:
                print(f"[skip] completed cell {a.cell_id}: {row['result_name']}")
                return 0
        except Exception:
            pass
    if result_dir.exists():
        raise SystemExit(f"Refusing to overwrite partial/existing result dir: {result_dir}")

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    payload = dict(row)
    payload['cell_id'] = int(row['cell_id'])
    payload['started_utc'] = started
    payload['returncode'] = None
    try:
        print('='*100)
        print(f"CSEM PAPER FINAL V1 | cell {row['cell_id']} | {row['family']} | {row['result_name']}")
        print(json.dumps(row, indent=2))
        print('='*100)
        loss_df, eval_df, cfg = run_cell(row, base)
        payload['returncode'] = 0
        payload['loss_rows'] = int(len(loss_df))
        payload['eval_rows'] = int(len(eval_df))
        payload['resolved_config'] = cfg
        rc = 0
    except Exception as e:
        payload['returncode'] = 1
        payload['error'] = repr(e)
        payload['traceback'] = traceback.format_exc()
        print(payload['traceback'], file=sys.stderr)
        rc = 1
    payload['elapsed_seconds'] = time.time() - t0
    payload['finished_utc'] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    return rc

if __name__ == '__main__':
    raise SystemExit(main())
