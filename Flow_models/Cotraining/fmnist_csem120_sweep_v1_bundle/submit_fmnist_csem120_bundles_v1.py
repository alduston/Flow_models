#!/usr/bin/env python3
"""
Submit the 20 ordinary CSEM sweep bundles to TACC Vista.

No Slurm job arrays are used. Each sbatch invocation submits one normal job
with CSEM_BUNDLE_ID exported into the generic bundle Slurm script.

The parser is intentionally tolerant of TACC's login/banner text and extracts
job IDs from lines such as:
    Submitted batch job 910163
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys

BASE_DIR = Path(
    "/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
)
MANIFEST_NAME = "fmnist_csem120_manifest_v1.csv"
SLURM_NAME = "fmnist_csem120_bundle_job_v1.slurm"
GRID_SCRIPT_NAME = "fmnist_csem120_grid_v1.py"
SLURM_LOG_DIR_NAME = "slurm_logs_csem120_v1"
RECEIPT_DIR_NAME = "submission_receipts_csem120_v1"


def parse_bundle_spec(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(20))
    out: set[int] = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(piece))
    bundles = sorted(out)
    bad = [b for b in bundles if not 0 <= b < 20]
    if bad:
        raise ValueError(f"Bundle IDs must be in 0..19; bad values: {bad}")
    if not bundles:
        raise ValueError("No bundles selected.")
    return bundles


def parse_job_id(text: str) -> str | None:
    # Primary TACC/Slurm form, robust to arbitrary banner text before it.
    matches = re.findall(r"Submitted\s+batch\s+job\s+(\d+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1]

    # Fallback for `sbatch --parsable`-style output.
    for line in reversed(text.splitlines()):
        m = re.fullmatch(r"\s*(\d+)(?:;[^\s]+)?\s*", line)
        if m:
            return m.group(1)
    return None


def ensure_manifest(base_dir: Path) -> Path:
    manifest = base_dir / MANIFEST_NAME
    if not manifest.is_file():
        grid_script = base_dir / GRID_SCRIPT_NAME
        if not grid_script.is_file():
            raise FileNotFoundError(f"Missing grid builder: {grid_script}")
        print(f"Manifest missing; generating it with {grid_script.name}")
        proc = subprocess.run(
            [sys.executable, str(grid_script)],
            cwd=str(base_dir),
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError("Manifest generation failed.")

    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 120:
        raise RuntimeError(f"Manifest must have 120 rows; found {len(rows)}.")
    for b in range(20):
        count = sum(int(r["bundle_id"]) == b for r in rows)
        if count != 6:
            raise RuntimeError(f"Bundle {b} has {count} rows; expected 6.")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Sweep directory on Vista.",
    )
    parser.add_argument(
        "--bundles",
        default="all",
        help="Which ordinary jobs to submit, e.g. all, 0-19, 3,7-9.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.is_dir():
        raise SystemExit(f"Base directory not found: {base_dir}")
    if not (base_dir / "csem_split_metric.py").is_file():
        raise SystemExit(f"Missing {base_dir / 'csem_split_metric.py'}")

    manifest = ensure_manifest(base_dir)
    slurm_script = base_dir / SLURM_NAME
    if not slurm_script.is_file():
        raise SystemExit(f"Missing Slurm script: {slurm_script}")

    bundles = parse_bundle_spec(args.bundles)
    if len(bundles) > 25:
        raise SystemExit("Refusing to submit more than 25 jobs in one invocation.")

    slurm_log_dir = base_dir / SLURM_LOG_DIR_NAME
    receipt_dir = base_dir / RECEIPT_DIR_NAME
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Validated manifest: {manifest}")
    print(f"Submitting {len(bundles)} ordinary jobs: {bundles}")
    print("Each job runs 6 configurations sequentially; no job arrays are used.")

    receipts: list[dict[str, str]] = []

    for bundle_id in bundles:
        output_pattern = slurm_log_dir / f"csem120_bundle_{bundle_id:02d}_%j.out"
        command = [
            "sbatch",
            "-J", f"c120b{bundle_id:02d}",
            "-o", str(output_pattern),
            f"--export=ALL,CSEM_BUNDLE_ID={bundle_id}",
            str(slurm_script),
        ]

        print("\n$ " + " ".join(command))
        if args.dry_run:
            receipts.append(
                {
                    "bundle_id": str(bundle_id),
                    "job_id": "DRY_RUN",
                    "returncode": "0",
                    "submitted_at": datetime.now().isoformat(),
                    "command": " ".join(command),
                }
            )
            continue

        proc = subprocess.run(
            command,
            cwd=str(base_dir),
            text=True,
            capture_output=True,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        if combined:
            print(combined, end="" if combined.endswith("\n") else "\n")

        job_id = parse_job_id(combined)
        receipts.append(
            {
                "bundle_id": str(bundle_id),
                "job_id": job_id or "UNPARSED",
                "returncode": str(proc.returncode),
                "submitted_at": datetime.now().isoformat(),
                "command": " ".join(command),
            }
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed for bundle {bundle_id} with rc={proc.returncode}."
            )
        if job_id is None:
            # Important: do not blindly continue when submission apparently
            # succeeded but the ID cannot be proven; that risks duplicate jobs
            # if the user reruns the submitter.
            raise RuntimeError(
                "sbatch returned success but the job ID could not be parsed. "
                "The raw output is printed above. The parser accepts both "
                "'Submitted batch job NNN' and parsable numeric lines."
            )

        print(f"Recorded bundle {bundle_id:02d} -> job {job_id}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = receipt_dir / f"submission_{stamp}.csv"
    with receipt_path.open("w", newline="") as f:
        fieldnames = ["bundle_id", "job_id", "returncode", "submitted_at", "command"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(receipts)

    print(f"\nWrote submission receipt: {receipt_path}")
    if args.dry_run:
        print("Dry run only: no jobs were submitted.")
    else:
        print(f"Submitted {len(receipts)} jobs, representing {len(receipts) * 6} configurations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
