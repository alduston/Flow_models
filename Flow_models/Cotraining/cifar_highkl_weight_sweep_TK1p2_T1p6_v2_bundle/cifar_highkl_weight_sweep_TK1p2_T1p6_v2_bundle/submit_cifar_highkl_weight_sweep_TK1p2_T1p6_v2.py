#!/usr/bin/env python3
"""Submit CIFAR CSEM/KL weight-sweep cells in sequential groups of up to three."""

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
MANIFEST = 'cifar_highkl_weight_sweep_TK1p2_T1p6_v2_manifest.csv'
GRID = 'generate_cifar_highkl_weight_sweep_TK1p2_T1p6_v2.py'
SLURM = 'cifar_highkl_weight_sweep_TK1p2_T1p6_v2_cell_job.slurm'
TARGET = 'csem_split_new_highkl_weight_sweep_TK1p2_T1p6_v2.py'
LOG_DIR = 'slurm_logs_cifar_highkl_weight_sweep_TK1p2_T1p6_v2'
RECEIPT_DIR = 'submission_receipts_cifar_highkl_weight_sweep_TK1p2_T1p6_v2'
N_CELLS = 20
RUNS_PER_JOB = 3


def parse_cells(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(N_CELLS))
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
    if not cells or any(c < 0 or c >= N_CELLS for c in cells):
        raise ValueError(f"Cells must be a nonempty subset of 0..{N_CELLS - 1}; got {cells}")
    return cells


def chunked(values: list[int], n: int) -> list[list[int]]:
    return [values[i:i+n] for i in range(0, len(values), n)]


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
    if len(rows) != N_CELLS:
        raise RuntimeError(f"Expected {N_CELLS} manifest rows, found {len(rows)}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--cells", default="all", help="all, 0-19, or e.g. 0,2,5-8")
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
    groups = chunked(cells, RUNS_PER_JOB)

    log_dir = base / LOG_DIR
    receipt_dir = base / RECEIPT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipts = []
    print(
        f"Submitting {len(cells)} cells as {len(groups)} ordinary jobs "
        f"(up to {RUNS_PER_JOB} sequential runs/job); no arrays."
    )

    for group_idx, group in enumerate(groups):
        cell_spec = ",".join(str(c) for c in group)
        range_tag = f"{group[0]:02d}_{group[-1]:02d}"
        out = log_dir / f"weight_cells_{range_tag}_%j.out"
        # Do not embed the comma-separated cell list in --export.  Slurm parses
        # commas in --export as separators between environment assignments, so
        # a value like 0,1,2 would be truncated to 0.  Instead, place the full
        # value in sbatch's own environment and export the inherited environment.
        submit_env = os.environ.copy()
        submit_env["CSEM_WEIGHT_SWEEP_CELL_IDS"] = cell_spec

        cmd = [
            "sbatch",
            "-J", f"cfw{range_tag}",
            "-o", str(out),
            "--export=ALL",
            str(base / SLURM),
        ]
        display_cmd = (
            f"CSEM_WEIGHT_SWEEP_CELL_IDS={cell_spec} " + " ".join(cmd)
        )
        print("\n$ " + display_cmd)

        if args.dry_run:
            job_id, rc, raw = "DRY_RUN", 0, ""
        else:
            proc = subprocess.run(
                cmd, cwd=str(base), text=True, capture_output=True, env=submit_env
            )
            raw = (proc.stdout or "") + (proc.stderr or "")
            if raw:
                print(raw, end="" if raw.endswith("\n") else "\n")
            rc = proc.returncode
            if rc != 0:
                raise RuntimeError(f"sbatch failed for cells {cell_spec}: rc={rc}")
            job_id = parse_job_id(raw)
            if job_id is None:
                raise RuntimeError(
                    "sbatch returned success but job ID could not be parsed. "
                    "Refusing to continue so the sweep cannot be double-submitted."
                )

        receipts.append({
            "group_index": group_idx,
            "cell_ids": cell_spec,
            "n_cells": len(group),
            "job_id": job_id,
            "returncode": rc,
            "submitted_at": datetime.now().isoformat(),
            "command": display_cmd,
        })
        print(f"Recorded cells {cell_spec} -> {job_id}")

    receipt = receipt_dir / (
        "submission_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    with receipt.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "group_index", "cell_ids", "n_cells", "job_id", "returncode",
                "submitted_at", "command"
            ],
        )
        w.writeheader()
        w.writerows(receipts)

    print(f"\nWrote {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
