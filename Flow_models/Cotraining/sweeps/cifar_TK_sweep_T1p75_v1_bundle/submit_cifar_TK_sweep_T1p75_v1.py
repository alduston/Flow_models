#!/usr/bin/env python3
"""Submit the eight ordinary CIFAR T_K sweep jobs. No Slurm arrays."""

from __future__ import annotations
import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
MANIFEST = "cifar_TK_sweep_T1p75_v1_manifest.csv"
GRID = "generate_cifar_TK_sweep_T1p75_v1.py"
SLURM = "cifar_TK_sweep_T1p75_v1_cell_job.slurm"
TARGET = "csem_split_metric_TK_sweep_cifar_v1.py"
LOG_DIR = "slurm_logs_cifar_TK_sweep_T1p75_v1"
RECEIPT_DIR = "submission_receipts_cifar_TK_sweep_T1p75_v1"


def parse_cells(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(8))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = sorted((int(a), int(b)))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    cells = sorted(out)
    if not cells or any(c < 0 or c > 7 for c in cells):
        raise ValueError(f"Cells must be a nonempty subset of 0..7; got {cells}")
    return cells


def parse_job_id(text: str) -> str | None:
    hits = re.findall(r"Submitted\s+batch\s+job\s+(\d+)", text, flags=re.I)
    if hits:
        return hits[-1]
    for line in reversed(text.splitlines()):
        m = re.fullmatch(r"\s*(\d+)(?:;[^\s]+)?\s*", line)
        if m:
            return m.group(1)
    return None


def ensure_manifest(base: Path) -> Path:
    path = base / MANIFEST
    if not path.is_file():
        proc = subprocess.run([sys.executable, str(base / GRID)], cwd=str(base))
        if proc.returncode:
            raise RuntimeError("Manifest generation failed.")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 manifest rows, found {len(rows)}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--cells", default="all", help="all, 0-7, or e.g. 0,2,5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    if not base.is_dir():
        raise SystemExit(f"Base directory not found: {base}")

    for name in (TARGET, SLURM):
        if not (base / name).is_file():
            raise SystemExit(f"Missing required file: {base / name}")

    ensure_manifest(base)
    cells = parse_cells(args.cells)

    log_dir = base / LOG_DIR
    receipt_dir = base / RECEIPT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipts = []
    print(f"Submitting {len(cells)} ordinary jobs; no arrays.")

    for cell in cells:
        out = log_dir / f"TK_cell_{cell:02d}_%j.out"
        cmd = [
            "sbatch",
            "-J", f"cftk{cell:02d}",
            "-o", str(out),
            f"--export=ALL,CSEM_TK_SWEEP_CELL_ID={cell}",
            str(base / SLURM),
        ]
        print("\n$ " + " ".join(cmd))

        if args.dry_run:
            job_id, rc, raw = "DRY_RUN", 0, ""
        else:
            proc = subprocess.run(
                cmd, cwd=str(base), text=True, capture_output=True
            )
            raw = (proc.stdout or "") + (proc.stderr or "")
            if raw:
                print(raw, end="" if raw.endswith("\n") else "\n")
            rc = proc.returncode
            if rc != 0:
                raise RuntimeError(f"sbatch failed for cell {cell}: rc={rc}")
            job_id = parse_job_id(raw)
            if job_id is None:
                raise RuntimeError(
                    "sbatch returned success but job ID could not be parsed. "
                    "Refusing to continue so the sweep cannot be double-submitted."
                )

        receipts.append({
            "cell_id": cell,
            "job_id": job_id,
            "returncode": rc,
            "submitted_at": datetime.now().isoformat(),
            "command": " ".join(cmd),
        })
        print(f"Recorded cell {cell} -> {job_id}")

    receipt = receipt_dir / (
        "submission_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    with receipt.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cell_id", "job_id", "returncode", "submitted_at", "command"
            ],
        )
        w.writeheader()
        w.writerows(receipts)

    print(f"\nWrote {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
