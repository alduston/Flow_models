#!/usr/bin/env python3
"""
Run one reconstruction-matched CSEM diagnostic pair.

Each pair contains exactly two 120-epoch FMNIST runs:
  slot 0: unweighted outer, lambda=0.60
  slot 1: canonical outer,  lambda=0.10

The score-head metric is fixed to unweighted-eps.  The four pair jobs differ
only in score-head LR (1e-4, 2e-4, 4e-4, 8e-4).

Every run performs endpoint-only exact aggregate-score/oracle diagnostics using
the full FashionMNIST training-set posterior mixture.

Completed runs are skipped safely on resubmission. Partial result directories
are never overwritten automatically.
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
MANIFEST_NAME = "fmnist_csem_why_manifest_v2.csv"
TARGET_NAME = "csem_split_metric_why_v2.py"
RESULTS_ROOT_NAME = "fmnist_csem_why_results_v2"
CONFIG_LOG_ROOT_NAME = "fmnist_csem_why_config_logs_v2"
STATUS_ROOT_NAME = "fmnist_csem_why_status_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pair(manifest: Path, pair_id: int) -> list[dict[str, str]]:
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    pair = [r for r in rows if int(r["pair_id"]) == pair_id]
    pair.sort(key=lambda r: int(r["slot_in_pair"]))
    if len(pair) != 2:
        raise RuntimeError(
            f"Expected exactly 2 rows for pair {pair_id}, found {len(pair)}"
        )
    return pair


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
        "--pair-id",
        type=int,
        default=None,
        help="Pair 0..3. Defaults to CSEM_WHY_PAIR_ID.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
    )
    args = parser.parse_args()

    pair_id = args.pair_id
    if pair_id is None:
        env_id = os.environ.get("CSEM_WHY_PAIR_ID")
        if env_id is None:
            raise SystemExit("Need --pair-id or CSEM_WHY_PAIR_ID.")
        pair_id = int(env_id)
    if not 0 <= pair_id < 4:
        raise SystemExit(f"Pair ID must be in 0..3, got {pair_id}")

    base_dir = args.base_dir.resolve()
    manifest = base_dir / MANIFEST_NAME
    target = base_dir / TARGET_NAME

    if not manifest.is_file():
        raise SystemExit(
            f"Missing {manifest}. Run: python3 {base_dir / 'fmnist_csem_why_grid_v1.py'}"
        )
    if not target.is_file():
        raise SystemExit(f"Missing diagnostic training script: {target}")

    results_root = base_dir / RESULTS_ROOT_NAME
    log_root = base_dir / CONFIG_LOG_ROOT_NAME
    status_root = base_dir / STATUS_ROOT_NAME
    results_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    status_root.mkdir(parents=True, exist_ok=True)

    rows = load_pair(manifest, pair_id)
    lr = rows[0]["lr_score_head"]

    print("=" * 92)
    print(f"CSEM WHY SWEEP — matched reconstruction pair {pair_id}")
    print(f"Host: {os.uname().nodename}")
    print(f"Python: {sys.executable}")
    print(f"Score-head LR: {lr}")
    print("Score-head metric: unweighted-eps (fixed)")
    print("Runs: U(lambda=.60) then C(lambda=.10)")
    print("=" * 92)

    failures: list[tuple[str, int]] = []

    for run_number, row in enumerate(rows, start=1):
        cfg_id = row["config_id"]
        result_name = row["result_name"]
        results_dir = results_root / result_name
        log_path = log_root / f"{result_name}.log"
        status_path = status_root / f"{result_name}.json"

        print("\n" + "#" * 92)
        print(
            f"Pair {pair_id} | run {run_number}/2 | cfg {cfg_id} | "
            f"outer={row['outer_time_weighting']} | lambda={row['rep_weight']} | "
            f"score LR={row['lr_score_head']} | "
            f"prior recon FID={row['prior_recon_fid']}"
        )
        print("#" * 92)

        if successful_previous_run(status_path, results_dir):
            print(f"[skip] cfg {cfg_id} already completed successfully.")
            continue

        if results_dir.exists():
            print(
                f"[blocked] Existing partial result directory:\n  {results_dir}\n"
                "Rename/remove it explicitly before retrying this configuration."
            )
            failures.append((cfg_id, 90))
            continue

        rep_w = row["rep_weight"]
        command = [
            sys.executable, "-u", str(target),
            "--dataset", "FMNIST",
            "--model-preset", "auto",

            "--score-time-weighting", row["outer_time_weighting"],
            "--score-head-time-weighting", "unweighted-eps",
            "--csem-w", rep_w,
            "--terminal-kl-w", rep_w,
            "--score-head-loss-w", row["score_head_loss_w"],
            "--lr-score-head", row["lr_score_head"],

            "--T-terminal", row["T_terminal"],
            "--epochs", row["epochs"],
            "--refine-epochs", row["refine_epochs"],
            "--eval-every", row["eval_every"],
            "--eval-samples", row["eval_samples"],
            "--cfg-strength", row["cfg_strength"],
            "--arms", "terminal_kl",

            # Focused exact-oracle mechanism protocol.
            "--eval-oracle-diagnostics",
            "--eval-oracle-full-train-reference",
            "--oracle-profile-query-samples", row["oracle_profile_query_samples"],
            "--oracle-profile-time-points", row["oracle_profile_time_points"],
            "--oracle-profile-batch-size", row["oracle_profile_batch_size"],
            "--oracle-reference-batch-size", row["oracle_reference_batch_size"],
            "--oracle-sampling-samples", row["oracle_sampling_samples"],
            "--oracle-sampling-batch-size", row["oracle_sampling_batch_size"],
            "--oracle-sampling-steps", row["oracle_sampling_steps"],
            "--eval-oracle-transport-decomposition",
            "--no-eval-oracle-standard-samplers",

            # Preserve parent-sweep training choices.
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
            "pair_id": pair_id,
            "slot_in_pair": int(row["slot_in_pair"]),
            "result_name": result_name,
            "results_dir": str(results_dir),
            "log_path": str(log_path),
            "started_utc": started,
            "finished_utc": finished,
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "outer_time_weighting": row["outer_time_weighting"],
            "head_time_weighting": "unweighted-eps",
            "rep_weight": float(row["rep_weight"]),
            "lr_score_head": float(row["lr_score_head"]),
            "prior_recon_fid": float(row["prior_recon_fid"]),
            "command": command,
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")

        if returncode == 0:
            print(f"[ok] cfg {cfg_id} completed ({elapsed/3600:.2f} h)")
        else:
            print(f"[failed] cfg {cfg_id}: rc={returncode}")
            failures.append((cfg_id, returncode))

    print("\n" + "=" * 92)
    if failures:
        print("Pair completed with failures/blocked cells:")
        for cfg_id, rc in failures:
            print(f"  cfg {cfg_id}: rc={rc}")
        return 1
    print(f"Pair {pair_id} completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
