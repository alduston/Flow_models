#!/usr/bin/env python3
"""Run one T_K group: three csem_w cells sequentially on one GPU."""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
MANIFEST = "cifar_TKxCSEM_fine_T1p75_v1_manifest.csv"
TARGET = "csem_split_metric_TKxCSEM_fine_cifar_v1.py"
RESULTS_ROOT = "cifar_TKxCSEM_fine_T1p75_v1_results"
LOG_ROOT = "cifar_TKxCSEM_fine_T1p75_v1_config_logs"
STATUS_ROOT = "cifar_TKxCSEM_fine_T1p75_v1_status"

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_group(path: Path, group_id: int) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    hit = [r for r in rows if int(r["group_id"]) == group_id]
    hit.sort(key=lambda r: float(r["csem_w"]))
    if len(hit) != 3:
        raise RuntimeError(
            f"Expected exactly 3 manifest rows for group {group_id}, found {len(hit)}"
        )
    tks = {r["T_K"] for r in hit}
    if len(tks) != 1:
        raise RuntimeError(f"Group {group_id} contains multiple T_K values: {tks}")
    return hit

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
    ap.add_argument("--group-id", type=int, default=None)
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    group_id = args.group_id
    if group_id is None:
        raw = os.environ.get("CSEM_FINE_GROUP_ID")
        if raw is None:
            raise SystemExit("Need --group-id or CSEM_FINE_GROUP_ID.")
        group_id = int(raw)
    if group_id < 0 or group_id >= 6:
        raise SystemExit(f"group-id must be in 0..5, got {group_id}")

    base = args.base_dir.resolve()
    manifest = base / MANIFEST
    target = base / TARGET
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")
    if not target.is_file():
        raise SystemExit(f"Missing target: {target}")

    group = load_group(manifest, group_id)
    tk = float(group[0]["T_K"])

    results_root = base / RESULTS_ROOT
    log_root = base / LOG_ROOT
    status_root = base / STATUS_ROOT
    for d in (results_root, log_root, status_root):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CIFAR FINE T_K x CSEM SWEEP")
    print(f"group       = {group_id}")
    print(f"T_K         = {tk:.2f}")
    print("T           = 1.75")
    print("csem_w      = 0.075, 0.100, 0.150 (sequential)")
    print("terminal KL = 0.30")
    print("epochs      = 500 cotrain + 0 refinement")
    print("eval        = every 50 epochs, 10000 samples")
    print("outer/head  = canonical / unweighted-eps")
    print("CFG         = 3.0")
    print("score LR    = CIFAR preset default 1e-4")
    print("=" * 100)

    group_started = time.monotonic()

    for j, row in enumerate(group, 1):
        cell_id = int(row["cell_id"])
        rel_result = Path(row["result_name"])
        result_dir = results_root / rel_result
        status_path = status_root / (str(rel_result).replace("/", "__") + ".json")
        log_path = log_root / (str(rel_result).replace("/", "__") + ".log")
        result_dir.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 100)
        print(
            f"GROUP {group_id} RUN {j}/3 | cell={cell_id:02d} | "
            f"T_K={row['T_K']} | csem_w={row['csem_w']}"
        )
        print("-" * 100)

        if completed(status_path, result_dir):
            print(f"[skip] already completed successfully: {result_dir}")
            continue
        if result_dir.exists():
            raise SystemExit(
                f"[blocked] Existing partial/unverified result directory: {result_dir}\n"
                "Move or remove that directory explicitly before retrying this group."
            )

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

        print("$ " + shlex.join(command))
        if args.dry_run:
            continue

        started = utc_now()
        t0 = time.monotonic()
        rc = tee_process(command, log_path, base)
        elapsed = time.monotonic() - t0

        payload = {
            "cell_id": cell_id,
            "group_id": group_id,
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
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, indent=2) + "\n")

        if rc != 0:
            print(f"[failed] cell {cell_id:02d} rc={rc} after {elapsed/3600:.2f} h")
            print("[abort] Stopping this 3-run group to avoid wasting the remaining GPU allocation.")
            return rc

        print(f"[ok] cell {cell_id:02d} completed in {elapsed/3600:.2f} h")

    total = time.monotonic() - group_started
    print(f"\n[ok] group {group_id} finished in {total/3600:.2f} h")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
