#!/usr/bin/env python3
"""Run one CIFAR CSEM/KL weight-sweep cell."""

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
MANIFEST = 'cifar_weight_sweep_TK1p2_T1p6_v1_manifest.csv'
TARGET = 'csem_split_new_weight_sweep_TK1p2_T1p6_v1.py'
RESULTS_ROOT = 'cifar_weight_sweep_TK1p2_T1p6_v1_results'
LOG_ROOT = 'cifar_weight_sweep_TK1p2_T1p6_v1_config_logs'
STATUS_ROOT = 'cifar_weight_sweep_TK1p2_T1p6_v1_status'
N_CELLS = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cell(path: Path, cell_id: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    hit = [r for r in rows if int(r["cell_id"]) == cell_id]
    if len(hit) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest row for cell {cell_id}, found {len(hit)}"
        )
    return hit[0]


def completed(status_path: Path, result_dir: Path) -> bool:
    if not status_path.is_file() or not result_dir.is_dir():
        return False
    try:
        d = json.loads(status_path.read_text())
        return int(d.get("returncode", 999999)) == 0
    except Exception:
        return False


def tee_process(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        log.write("COMMAND:\n")
        log.write(shlex.join(command) + "\n\n")
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", type=int, required=True)
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    args = ap.parse_args()

    cell_id = args.cell_id
    if cell_id < 0 or cell_id >= N_CELLS:
        raise SystemExit(f"cell-id must be in 0..{N_CELLS - 1}, got {cell_id}")

    base = args.base_dir.resolve()
    manifest = base / MANIFEST
    target = base / TARGET
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")
    if not target.is_file():
        raise SystemExit(f"Missing target: {target}")

    row = load_cell(manifest, cell_id)

    results_root = base / RESULTS_ROOT
    log_root = base / LOG_ROOT
    status_root = base / STATUS_ROOT
    for d in (results_root, log_root, status_root):
        d.mkdir(parents=True, exist_ok=True)

    result_dir = results_root / row["result_name"]
    log_path = log_root / f"{row['result_name']}.log"
    status_path = status_root / f"{row['result_name']}.json"

    if completed(status_path, result_dir):
        print(f"[skip] cell {cell_id} already completed successfully.")
        return 0
    if result_dir.exists():
        raise SystemExit(
            f"[blocked] Partial/existing result directory: {result_dir}\n"
            "Rename/remove it explicitly before retrying this cell."
        )

    print("=" * 96)
    print("CIFAR CSEM / TERMINAL-KL WEIGHT SWEEP")
    print(f"cell                 = {cell_id}")
    print(f"T_K                  = {row['T_K']}")
    print(f"T                    = {row['T_full']}")
    print(f"csem_w               = {row['csem_w']}")
    print(f"terminal_kl_w        = {row['terminal_kl_w']}")
    print("outer/head metric    = canonical / unweighted-eps")
    print("cotrain/refine       = 500 / 100")
    print("score-head LR        = CIFAR preset default (1e-4)")
    print("evaluation           = every 50, 10000 samples")
    print("4-way eval           = oracle/Gaussian at both T_K and T; matched NFE density")
    print("CFG                  = 3.0")
    print("=" * 96)

    command = [
        sys.executable, "-u", str(target),
        "--dataset", "CIFAR",
        "--model-preset", "auto",
        "--arms", "terminal_kl",
        "--score-time-weighting", row["outer_time_weighting"],
        "--score-head-time-weighting", row["head_time_weighting"],
        "--csem-w", row["csem_w"],
        "--terminal-kl-w", row["terminal_kl_w"],
        "--score-head-loss-w", row["score_head_loss_w"],
        "--T-terminal", row["T_K"],
        "--T", row["T_full"],
        "--epochs", row["epochs"],
        "--refine-epochs", row["refine_epochs"],
        "--eval-every", row["eval_every"],
        "--eval-samples", row["eval_samples"],
        "--cfg-strength", row["cfg_strength"],
        "--canonical-lr-scale", "1.0",
        "--encoder-score-warmup-epochs", "0",
        "--csem-ramp-epochs", "0",
        "--score-tracking-steps", "0",
        "--grad-diagnostics-every", "0",
        "--logvar-min=-30.0",
        "--logvar-max", "20.0",
        "--no-fail-on-nonfinite",
        "--no-bespoke-fid-classifier",
        "--master-results-dir", str(result_dir),
    ]

    started = utc_now()
    t0 = time.monotonic()
    rc = tee_process(command, log_path, base)
    elapsed = time.monotonic() - t0

    payload = {
        "cell_id": cell_id,
        "T_K": float(row["T_K"]),
        "T_full": float(row["T_full"]),
        "csem_w": float(row["csem_w"]),
        "terminal_kl_w": float(row["terminal_kl_w"]),
        "result_name": row["result_name"],
        "results_dir": str(result_dir),
        "log_path": str(log_path),
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "started_utc": started,
        "finished_utc": utc_now(),
        "command": command,
    }
    status_path.write_text(json.dumps(payload, indent=2) + "\n")

    if rc == 0:
        print(f"[ok] cell {cell_id} completed in {elapsed/3600:.2f} h")
    else:
        print(f"[failed] cell {cell_id} rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
