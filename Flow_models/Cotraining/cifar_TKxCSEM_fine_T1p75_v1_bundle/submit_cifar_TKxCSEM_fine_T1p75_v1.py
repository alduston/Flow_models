#!/usr/bin/env python3
"""Submit six ordinary Slurm jobs; each runs three csem_w values sequentially."""

from __future__ import annotations
import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
MANIFEST = "cifar_TKxCSEM_fine_T1p75_v1_manifest.csv"
GRID = "generate_cifar_TKxCSEM_fine_T1p75_v1.py"
SLURM = "cifar_TKxCSEM_fine_T1p75_v1_group_job.slurm"
TARGET = "csem_split_metric_TKxCSEM_fine_cifar_v1.py"
LOG_DIR = "slurm_logs_cifar_TKxCSEM_fine_T1p75_v1"
RECEIPT_DIR = "submission_receipts_cifar_TKxCSEM_fine_T1p75_v1"

def parse_groups(spec: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(6))
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
    groups = sorted(out)
    if not groups or any(g < 0 or g > 5 for g in groups):
        raise ValueError(f"Groups must be a nonempty subset of 0..5; got {groups}")
    return groups

def parse_job_id(text: str) -> str | None:
    hits = re.findall(r"Submitted\s+batch\s+job\s+(\d+)", text, flags=re.I)
    if hits:
        return hits[-1]
    for line in reversed(text.splitlines()):
        m = re.fullmatch(r"\s*(\d+)(?:;[^\s]+)?\s*", line)
        if m:
            return m.group(1)
    return None

def ensure_manifest(base: Path) -> list[dict[str, str]]:
    path = base / MANIFEST
    if not path.is_file():
        proc = subprocess.run([sys.executable, str(base / GRID)], cwd=str(base))
        if proc.returncode:
            raise RuntimeError("Manifest generation failed.")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 manifest rows, found {len(rows)}")
    for gid in range(6):
        hit = [r for r in rows if int(r["group_id"]) == gid]
        if len(hit) != 3:
            raise RuntimeError(f"Group {gid} has {len(hit)} rows, expected 3.")
    return rows

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--groups", default="all", help="all, 0-5, or e.g. 0,2,5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    if not base.is_dir():
        raise SystemExit(f"Base directory not found: {base}")
    for name in (TARGET, SLURM):
        if not (base / name).is_file():
            raise SystemExit(f"Missing required file: {base / name}")

    rows = ensure_manifest(base)
    groups = parse_groups(args.groups)
    tk_by_group = {
        gid: float(next(r["T_K"] for r in rows if int(r["group_id"]) == gid))
        for gid in range(6)
    }

    log_dir = base / LOG_DIR
    receipt_dir = base / RECEIPT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipts = []
    print(
        f"Submitting {len(groups)} ordinary jobs; "
        "each job runs 3 csem_w cells sequentially. No arrays."
    )

    for gid in groups:
        tk = tk_by_group[gid]
        tk_tag = f"{tk:.2f}".replace(".", "p")
        out = log_dir / f"TK_{tk_tag}_%j.out"
        job_name = f"cfTK{int(round(tk*100)):03d}"
        cmd = [
            "sbatch",
            "-J", job_name,
            "-o", str(out),
            f"--export=ALL,CSEM_FINE_GROUP_ID={gid}",
            str(base / SLURM),
        ]
        print("\n$ " + " ".join(cmd))
        print(f"  group {gid}: T_K={tk:.2f}, csem_w=.075,.10,.15")

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
                raise RuntimeError(f"sbatch failed for group {gid}: rc={rc}")
            job_id = parse_job_id(raw)
            if job_id is None:
                raise RuntimeError(
                    "sbatch returned success but the job ID could not be parsed. "
                    "Stopping immediately to prevent accidental duplicate submissions."
                )

        receipts.append({
            "group_id": gid,
            "T_K": tk,
            "job_id": job_id,
            "returncode": rc,
            "submitted_at": datetime.now().isoformat(),
            "command": " ".join(cmd),
        })
        print(f"Recorded group {gid} / T_K={tk:.2f} -> {job_id}")

    receipt = receipt_dir / (
        "submission_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    with receipt.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "group_id", "T_K", "job_id", "returncode",
                "submitted_at", "command",
            ],
        )
        w.writeheader()
        w.writerows(receipts)

    print(f"\nWrote {receipt}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
