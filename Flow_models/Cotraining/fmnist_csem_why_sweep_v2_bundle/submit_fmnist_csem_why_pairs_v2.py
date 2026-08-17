#!/usr/bin/env python3
"""Submit the four ordinary matched-pair diagnostic jobs on Vista."""

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
MANIFEST_NAME = "fmnist_csem_why_manifest_v2.csv"
GRID_NAME = "fmnist_csem_why_grid_v2.py"
SLURM_NAME = "fmnist_csem_why_time_pair_job_v2.slurm"
TARGET_NAME = "csem_split_metric_why_v2.py"
LOG_DIR_NAME = "slurm_logs_csem_why_v2"
RECEIPT_DIR_NAME = "submission_receipts_csem_why_v2"


def parse_pair_spec(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(4))
    out = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            lo, hi = sorted((int(a), int(b)))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(piece))
    pairs = sorted(out)
    if not pairs or any(p < 0 or p > 3 for p in pairs):
        raise ValueError(f"Pair IDs must be a nonempty subset of 0..3; got {pairs}")
    return pairs


def parse_job_id(text: str) -> str | None:
    matches = re.findall(r"Submitted\s+batch\s+job\s+(\d+)", text, flags=re.I)
    if matches:
        return matches[-1]
    for line in reversed(text.splitlines()):
        m = re.fullmatch(r"\s*(\d+)(?:;[^\s]+)?\s*", line)
        if m:
            return m.group(1)
    return None


def ensure_manifest(base_dir: Path) -> Path:
    manifest = base_dir / MANIFEST_NAME
    if not manifest.is_file():
        grid = base_dir / GRID_NAME
        if not grid.is_file():
            raise FileNotFoundError(f"Missing grid builder: {grid}")
        proc = subprocess.run([sys.executable, str(grid)], cwd=str(base_dir))
        if proc.returncode:
            raise RuntimeError("Manifest generation failed.")
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 manifest rows, found {len(rows)}.")
    for p in range(4):
        if sum(int(r["pair_id"]) == p for r in rows) != 2:
            raise RuntimeError(f"Pair {p} does not contain exactly two rows.")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--pairs", default="all", help="all, 0-3, or e.g. 0,2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    if not base.is_dir():
        raise SystemExit(f"Base directory not found: {base}")
    for name in (TARGET_NAME, SLURM_NAME):
        if not (base / name).is_file():
            raise SystemExit(f"Missing required file: {base/name}")

    manifest = ensure_manifest(base)
    pairs = parse_pair_spec(args.pairs)

    log_dir = base / LOG_DIR_NAME
    receipt_dir = base / RECEIPT_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipts = []
    print(f"Validated {manifest}")
    print(f"Submitting {len(pairs)} ordinary jobs for pairs {pairs}; no arrays.")

    for pair_id in pairs:
        out = log_dir / f"csem_why_time_pair_{pair_id:02d}_%j.out"
        cmd = [
            "sbatch",
            "-J", f"cwhyT{pair_id}",
            "-o", str(out),
            f"--export=ALL,CSEM_WHY_PAIR_ID={pair_id}",
            str(base / SLURM_NAME),
        ]
        print("\n$ " + " ".join(cmd))
        if args.dry_run:
            job_id, rc, raw = "DRY_RUN", 0, ""
        else:
            proc = subprocess.run(cmd, cwd=str(base), text=True, capture_output=True)
            raw = (proc.stdout or "") + (proc.stderr or "")
            if raw:
                print(raw, end="" if raw.endswith("\n") else "\n")
            rc = proc.returncode
            job_id = parse_job_id(raw)
            if rc != 0:
                raise RuntimeError(f"sbatch failed for pair {pair_id}: rc={rc}")
            if job_id is None:
                raise RuntimeError(
                    "sbatch returned success but job ID could not be parsed. "
                    "Raw output is printed above; refusing to continue to avoid duplicates."
                )
        receipts.append({
            "pair_id": pair_id,
            "job_id": job_id or "UNPARSED",
            "returncode": rc,
            "submitted_at": datetime.now().isoformat(),
            "command": " ".join(cmd),
        })
        print(f"Recorded pair {pair_id} -> {job_id}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt = receipt_dir / f"submission_{stamp}.csv"
    with receipt.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["pair_id","job_id","returncode","submitted_at","command"]
        )
        w.writeheader()
        w.writerows(receipts)
    print(f"\nWrote {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
