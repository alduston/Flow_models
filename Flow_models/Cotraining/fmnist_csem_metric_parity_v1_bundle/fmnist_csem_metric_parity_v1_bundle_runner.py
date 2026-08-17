#!/usr/bin/env python3
"""Run one two-cell bundle for fmnist_csem_metric_parity_v1."""

from __future__ import annotations
import argparse, csv, json, os, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep")
MANIFEST = "fmnist_csem_metric_parity_v1_manifest.csv"
TARGET = "csem_split_metric_metric_parity_v1.py"
RESULTS_ROOT = "fmnist_csem_metric_parity_v1_results"
LOG_ROOT = "fmnist_csem_metric_parity_v1_config_logs"
STATUS_ROOT = "fmnist_csem_metric_parity_v1_status"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_bundle(path: Path, bundle_id: int):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = [r for r in rows if int(r["bundle_id"]) == bundle_id]
    out.sort(key=lambda r: int(r["slot_in_bundle"]))
    if not out:
        raise RuntimeError(f"No rows for bundle {bundle_id}")
    if len(out) > 2:
        raise RuntimeError(f"Bundle {bundle_id} unexpectedly has {len(out)} rows")
    return out


def completed(status_path: Path, result_dir: Path) -> bool:
    if not status_path.is_file() or not result_dir.is_dir():
        return False
    try:
        s = json.loads(status_path.read_text())
        return int(s.get("returncode", 999)) == 0
    except Exception:
        return False


def tee(command, log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        log.write("COMMAND:\n" + shlex.join(command) + "\n\n")
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
    ap.add_argument("--bundle-id", type=int, default=None)
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    args = ap.parse_args()

    bundle_id = args.bundle_id
    if bundle_id is None:
        value = os.environ.get("CSEM_METRIC_PARITY_BUNDLE_ID")
        if value is None:
            raise SystemExit("Need --bundle-id or CSEM_METRIC_PARITY_BUNDLE_ID.")
        bundle_id = int(value)

    base = args.base_dir.resolve()
    manifest = base / MANIFEST
    target = base / TARGET
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")
    if not target.is_file():
        raise SystemExit(f"Missing target: {target}")

    result_root = base / RESULTS_ROOT
    log_root = base / LOG_ROOT
    status_root = base / STATUS_ROOT
    for d in (result_root, log_root, status_root):
        d.mkdir(parents=True, exist_ok=True)

    rows = load_bundle(manifest, bundle_id)
    failures = []

    print("="*96)
    print(f"Metric-parity bundle {bundle_id} | {len(rows)} configuration(s)")
    print(f"Python: {sys.executable}")
    print("="*96)

    for row in rows:
        cid = row["config_id"]
        name = row["result_name"]
        result_dir = result_root / name
        log_path = log_root / f"{name}.log"
        status_path = status_root / f"{name}.json"

        print("\n" + "#"*96)
        print(
            f"cfg={cid} role={row['role']} outer={row['outer_time_weighting']} "
            f"csem_w={row['csem_w']} terminal_kl_w={row['terminal_kl_w']} "
            f"lr_score={row['lr_score_head']}"
        )
        print("#"*96)

        if completed(status_path, result_dir):
            print(f"[skip] cfg {cid} already completed.")
            continue
        if result_dir.exists():
            print(f"[blocked] Partial result directory exists: {result_dir}")
            failures.append((cid, 90))
            continue

        cmd = [
            sys.executable, "-u", str(target),
            "--dataset", "FMNIST",
            "--model-preset", "auto",
            "--arms", "terminal_kl",

            "--score-time-weighting", row["outer_time_weighting"],
            "--score-head-time-weighting", "unweighted-eps",
            "--csem-w", row["csem_w"],
            "--terminal-kl-w", row["terminal_kl_w"],
            "--score-head-loss-w", row["score_head_loss_w"],
            "--lr-score-head", row["lr_score_head"],

            "--T-terminal", row["T_terminal"],
            "--epochs", row["epochs"],
            "--refine-epochs", row["refine_epochs"],
            "--eval-every", row["eval_every"],
            "--eval-samples", row["eval_samples"],
            "--cfg-strength", row["cfg_strength"],

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

            "--canonical-lr-scale", "1.0",
            "--encoder-score-warmup-epochs", "0",
            "--csem-ramp-epochs", "0",
            "--score-tracking-steps", "0",
            "--grad-diagnostics-every", "0",
            "--no-bespoke-fid-classifier",
            "--no-fail-on-nonfinite",

            "--master-results-dir", str(result_dir),
        ]

        started = utc_now()
        t0 = time.monotonic()
        rc = tee(cmd, log_path, base)
        elapsed = time.monotonic() - t0
        finished = utc_now()

        payload = dict(
            config_id=cid,
            bundle_id=bundle_id,
            result_name=name,
            results_dir=str(result_dir),
            log_path=str(log_path),
            started_utc=started,
            finished_utc=finished,
            elapsed_seconds=elapsed,
            returncode=rc,
            role=row["role"],
            outer_time_weighting=row["outer_time_weighting"],
            csem_w=float(row["csem_w"]),
            terminal_kl_w=float(row["terminal_kl_w"]),
            lr_score_head=float(row["lr_score_head"]),
            command=cmd,
        )
        status_path.write_text(json.dumps(payload, indent=2) + "\n")

        if rc:
            failures.append((cid, rc))
            print(f"[failed] cfg {cid} rc={rc}")
        else:
            print(f"[ok] cfg {cid} elapsed={elapsed/3600:.2f} h")

    if failures:
        print("Failures:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
