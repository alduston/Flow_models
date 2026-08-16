#!/usr/bin/env python3
"""
Run one six-configuration bundle from fmnist_csem120_manifest_v1.csv.

The bundle ID is supplied either as:
    --bundle-id N
or through:
    CSEM_BUNDLE_ID=N

All six configurations are attempted sequentially, even if one exits nonzero.
Completed configurations are skipped on a bundle resubmission when their status
JSON records returncode == 0 and their result directory still exists.

No result directory is ever overwritten automatically.
"""

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

DEFAULT_BASE_DIR = Path(
    "/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
)
MANIFEST_NAME = "fmnist_csem120_manifest_v1.csv"
TARGET_NAME = "csem_split_metric.py"
RESULTS_ROOT_NAME = "fmnist_csem120_results_v1"
CONFIG_LOG_ROOT_NAME = "fmnist_csem120_config_logs_v1"
STATUS_ROOT_NAME = "fmnist_csem120_status_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_bundle(manifest: Path, bundle_id: int) -> list[dict[str, str]]:
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    bundle = [r for r in rows if int(r["bundle_id"]) == bundle_id]
    bundle.sort(key=lambda r: int(r["slot_in_bundle"]))
    if len(bundle) != 6:
        raise RuntimeError(
            f"Expected exactly 6 rows for bundle {bundle_id}, found {len(bundle)}"
        )
    return bundle


def successful_previous_run(status_path: Path, results_dir: Path) -> bool:
    if not status_path.is_file() or not results_dir.is_dir():
        return False
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return False
    return int(status.get("returncode", 999999)) == 0


def tee_process(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        log.write("COMMAND:\n")
        log.write(shlex.join(command) + "\n\n")
        log.flush()

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-id",
        type=int,
        default=None,
        help="Bundle 0..19. Defaults to CSEM_BUNDLE_ID.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing csem_split_metric.py and the sweep files.",
    )
    args = parser.parse_args()

    env_bundle = os.environ.get("CSEM_BUNDLE_ID")
    bundle_id = args.bundle_id
    if bundle_id is None:
        if env_bundle is None:
            raise SystemExit("Need --bundle-id or CSEM_BUNDLE_ID.")
        bundle_id = int(env_bundle)

    if not 0 <= bundle_id < 20:
        raise SystemExit(f"Bundle ID must be in 0..19, got {bundle_id}")

    base_dir = args.base_dir.resolve()
    manifest = base_dir / MANIFEST_NAME
    target = base_dir / TARGET_NAME

    if not base_dir.is_dir():
        raise SystemExit(f"Base directory does not exist: {base_dir}")
    if not manifest.is_file():
        raise SystemExit(
            f"Manifest not found: {manifest}\n"
            f"Run: python3 {base_dir / 'fmnist_csem120_grid_v1.py'}"
        )
    if not target.is_file():
        raise SystemExit(f"Training script not found: {target}")

    results_root = base_dir / RESULTS_ROOT_NAME
    log_root = base_dir / CONFIG_LOG_ROOT_NAME
    status_root = base_dir / STATUS_ROOT_NAME
    results_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    status_root.mkdir(parents=True, exist_ok=True)

    rows = load_bundle(manifest, bundle_id)

    print("=" * 88)
    print(f"CSEM FMNIST 120-cell screening sweep — bundle {bundle_id:02d}")
    print(f"Host: {os.uname().nodename}")
    print(f"Python: {sys.executable}")
    print(f"Base dir: {base_dir}")
    print("Runs in bundle: 6 sequential configurations")
    print("=" * 88)

    failures: list[tuple[str, int]] = []

    for run_number, row in enumerate(rows, start=1):
        cfg_id = row["config_id"]
        result_name = row["result_name"]
        results_dir = results_root / result_name
        log_path = log_root / f"{result_name}.log"
        status_path = status_root / f"{result_name}.json"

        print("\n" + "#" * 88)
        print(
            f"Bundle {bundle_id:02d} | run {run_number}/6 | cfg {cfg_id} | "
            f"outer={row['outer_time_weighting']} | "
            f"head={row['head_time_weighting']} | "
            f"lambda={row['rep_weight']} | "
            f"lr_score_head={row['lr_score_head']}"
        )
        print(f"Results: {results_dir}")
        print(f"Log:     {log_path}")
        print("#" * 88)

        if successful_previous_run(status_path, results_dir):
            print(f"[skip] cfg {cfg_id} already completed successfully.")
            continue

        # The training driver itself refuses to overwrite an existing master root.
        # We detect that situation here too, so a partial run can never be destroyed
        # silently on resubmission.
        if results_dir.exists():
            print(
                f"[blocked] cfg {cfg_id}: results directory already exists but there "
                f"is no matching successful status record:\n  {results_dir}\n"
                "Remove/rename that partial directory explicitly before retrying this cfg."
            )
            failures.append((cfg_id, 90))
            continue

        rep_w = row["rep_weight"]

        command = [
            sys.executable,
            "-u",
            str(target),
            "--dataset", "FMNIST",
            "--model-preset", "auto",

            # Sweep axes.
            "--score-time-weighting", row["outer_time_weighting"],
            "--score-head-time-weighting", row["head_time_weighting"],
            "--csem-w", rep_w,
            "--terminal-kl-w", rep_w,
            "--score-head-loss-w", row["score_head_loss_w"],
            "--lr-score-head", row["lr_score_head"],

            # Fixed screening protocol.
            "--T-terminal", row["T_terminal"],
            "--epochs", row["epochs"],
            "--refine-epochs", row["refine_epochs"],
            "--eval-every", row["eval_every"],
            "--eval-samples", row["eval_samples"],
            "--cfg-strength", row["cfg_strength"],
            "--arms", "terminal_kl",

            # Explicitly lock the reference-run choices.
            "--canonical-lr-scale", "1.0",
            "--encoder-score-warmup-epochs", "0",
            "--csem-ramp-epochs", "0",
            "--score-tracking-steps", "0",
            "--grad-diagnostics-every", "0",
            "--no-bespoke-fid-classifier",
            "--no-fail-on-nonfinite",

            "--master-results-dir", str(results_dir),
        ]

        started = utc_now()
        t0 = time.monotonic()
        returncode = tee_process(command, log_path, base_dir)
        elapsed = time.monotonic() - t0
        finished = utc_now()

        status = {
            "config_id": cfg_id,
            "bundle_id": bundle_id,
            "slot_in_bundle": int(row["slot_in_bundle"]),
            "result_name": result_name,
            "results_dir": str(results_dir),
            "log_path": str(log_path),
            "started_utc": started,
            "finished_utc": finished,
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "outer_time_weighting": row["outer_time_weighting"],
            "head_time_weighting": row["head_time_weighting"],
            "rep_weight": float(row["rep_weight"]),
            "lr_score_head": float(row["lr_score_head"]),
            "command": command,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")

        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        if returncode == 0:
            print(f"[ok] cfg {cfg_id} finished in {h}h {m}m {s}s")
        else:
            print(f"[failed] cfg {cfg_id} rc={returncode} after {h}h {m}m {s}s")
            failures.append((cfg_id, returncode))

    print("\n" + "=" * 88)
    if failures:
        print("Bundle completed with failed/blocked configurations:")
        for cfg_id, rc in failures:
            print(f"  cfg {cfg_id}: rc={rc}")
        print("All other configurations in the bundle were still attempted.")
        return 1

    print(f"Bundle {bundle_id:02d} completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
