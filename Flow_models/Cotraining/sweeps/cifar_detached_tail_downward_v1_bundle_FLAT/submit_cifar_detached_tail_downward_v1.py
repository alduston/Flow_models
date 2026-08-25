#!/usr/bin/env python3
"""Submit detached-tail downward-extension cells, one Slurm job per training."""
from __future__ import annotations
import argparse, csv, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep")
MANIFEST = "cifar_detached_tail_downward_v1_manifest.csv"
GRID = "generate_cifar_detached_tail_downward_v1.py"
SLURM = "cifar_detached_tail_downward_v1_cell_job.slurm"
RUNNER = "run_cifar_detached_tail_downward_v1_cell.py"
TARGET = "csem_detached_tail_downward_v1.py"
LOG_DIR = "slurm_logs_cifar_detached_tail_downward_v1"
RECEIPT_DIR = "submission_receipts_cifar_detached_tail_downward_v1"
N_CELLS = 32
ENV_NAME = "CSEM_TAIL_DOWN_CELL"


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
        raise ValueError(f"Cells must be subset 0..{N_CELLS-1}: {cells}")
    return cells


def parse_job_id(text: str):
    hits = re.findall(r"Submitted\s+batch\s+job\s+(\d+)", text, flags=re.I)
    if hits:
        return hits[-1]
    for line in reversed(text.splitlines()):
        m = re.fullmatch(r"\s*(\d+)(?:;[^\s]+)?\s*", line)
        if m:
            return m.group(1)
    return None


def ensure_manifest(base: Path):
    p = base / MANIFEST
    if not p.is_file():
        proc = subprocess.run([sys.executable, str(base / GRID)], cwd=str(base))
        if proc.returncode:
            raise RuntimeError("Manifest generation failed")
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != N_CELLS:
        raise RuntimeError(f"Expected {N_CELLS} rows, found {len(rows)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--cells", default="all", help="all, comma list, or ranges e.g. 0-7,12")
    ap.add_argument("--group", choices=("all", "primary", "control"), default="all",
                    help="primary=24 wK=.60 cells; control=8 wK=.40 cells")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    if not base.is_dir():
        raise SystemExit(f"Base directory not found: {base}")
    for name in (TARGET, RUNNER, SLURM):
        if not (base / name).is_file():
            raise SystemExit(f"Missing required file: {base / name}")

    rows = ensure_manifest(base)
    cells = parse_cells(args.cells)
    if args.group == "primary":
        allowed = {int(r["cell_id"]) for r in rows if r["sweep_group"] == "primary"}
        cells = [c for c in cells if c in allowed]
    elif args.group == "control":
        allowed = {int(r["cell_id"]) for r in rows if r["sweep_group"] == "kl_control"}
        cells = [c for c in cells if c in allowed]
    if not cells:
        raise SystemExit("No cells remain after applying --cells/--group filters.")

    log_dir = base / LOG_DIR
    receipt_dir = base / RECEIPT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Submitting {len(cells)} / {N_CELLS} downward-tail trainings; group={args.group}; one cell per Slurm job.")
    receipts = []
    for cell in cells:
        row = next(r for r in rows if int(r["cell_id"]) == cell)
        out = log_dir / f"tail_down_cell_{cell:02d}_%j.out"
        env = os.environ.copy(); env[ENV_NAME] = str(cell)
        cmd = ["sbatch", "-J", f"tdn{cell:02d}", "-o", str(out), "--export=ALL", str(base / SLURM)]
        display = f"{ENV_NAME}={cell} " + " ".join(cmd)
        print("\n$ " + display)
        if args.dry_run:
            job_id, rc, raw = "DRY_RUN", 0, ""
        else:
            proc = subprocess.run(cmd, cwd=str(base), text=True, capture_output=True, env=env)
            raw = (proc.stdout or "") + (proc.stderr or "")
            rc = proc.returncode
            if raw:
                print(raw, end="" if raw.endswith("\n") else "\n")
            if rc != 0:
                raise RuntimeError(f"sbatch failed for cell {cell}: rc={rc}")
            job_id = parse_job_id(raw)
            if job_id is None:
                raise RuntimeError("sbatch succeeded but job ID could not be parsed; stopping to avoid duplicates")
        receipts.append({
            "cell_id": cell, "sweep_group": row["sweep_group"], "rep_id": row["rep_id"],
            "seed": row["seed"], "T_full": row["T_full"], "delta_T": row["delta_T"],
            "T_K": row["T_K"], "terminal_kl_w": row["terminal_kl_w"],
            "job_id": job_id, "returncode": rc,
            "submitted_at": datetime.now().isoformat(), "command": display,
        })
        print(f"Recorded cell {cell:02d} ({row['sweep_group']}, {row['rep_id']}) -> {job_id}")

    receipt = receipt_dir / ("submission_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    fields = ["cell_id", "sweep_group", "rep_id", "seed", "T_full", "delta_T", "T_K",
              "terminal_kl_w", "job_id", "returncode", "submitted_at", "command"]
    with receipt.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(receipts)
    print(f"\nWrote {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
