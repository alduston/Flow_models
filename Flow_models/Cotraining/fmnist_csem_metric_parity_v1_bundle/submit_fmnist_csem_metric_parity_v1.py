#!/usr/bin/env python3
"""Submit the 9 ordinary jobs for fmnist_csem_metric_parity_v1. No Slurm arrays."""

from __future__ import annotations
import argparse, csv, re, subprocess, sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep")
MANIFEST = "fmnist_csem_metric_parity_v1_manifest.csv"
GRID = "fmnist_csem_metric_parity_v1_grid.py"
SLURM = "fmnist_csem_metric_parity_v1_bundle_job.slurm"
TARGET = "csem_split_metric_metric_parity_v1.py"
LOG_DIR = "slurm_logs_fmnist_csem_metric_parity_v1"
RECEIPT_DIR = "submission_receipts_fmnist_csem_metric_parity_v1"


def parse_spec(spec: str):
    if spec.strip().lower() == "all":
        return list(range(9))
    out = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a,b = piece.split("-",1)
            lo,hi = sorted((int(a),int(b)))
            out.update(range(lo,hi+1))
        else:
            out.add(int(piece))
    ans = sorted(out)
    if not ans or any(x < 0 or x > 8 for x in ans):
        raise ValueError(f"Bundles must be in 0..8; got {ans}")
    return ans


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
    path = base / MANIFEST
    if not path.is_file():
        proc = subprocess.run([sys.executable, str(base/GRID)], cwd=str(base))
        if proc.returncode:
            raise RuntimeError("Manifest generation failed.")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 17:
        raise RuntimeError(f"Expected 17 manifest rows; got {len(rows)}")
    if len({int(r['bundle_id']) for r in rows}) != 9:
        raise RuntimeError("Expected 9 bundle IDs.")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--bundles", default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    for name in (SLURM, TARGET):
        if not (base/name).is_file():
            raise SystemExit(f"Missing required file: {base/name}")
    ensure_manifest(base)

    bundles = parse_spec(args.bundles)
    logdir = base/LOG_DIR
    recdir = base/RECEIPT_DIR
    logdir.mkdir(parents=True, exist_ok=True)
    recdir.mkdir(parents=True, exist_ok=True)

    receipts = []
    for bid in bundles:
        out = logdir / f"metric_parity_bundle_{bid:02d}_%j.out"
        cmd = [
            "sbatch",
            "-J", f"cmp{bid:02d}",
            "-o", str(out),
            f"--export=ALL,CSEM_METRIC_PARITY_BUNDLE_ID={bid}",
            str(base/SLURM),
        ]
        print("$ " + " ".join(cmd))
        if args.dry_run:
            jid, rc, raw = "DRY_RUN", 0, ""
        else:
            proc = subprocess.run(cmd, cwd=str(base), text=True, capture_output=True)
            raw = (proc.stdout or "") + (proc.stderr or "")
            if raw:
                print(raw, end="" if raw.endswith("\n") else "\n")
            rc = proc.returncode
            if rc:
                raise RuntimeError(f"sbatch failed for bundle {bid}: rc={rc}")
            jid = parse_job_id(raw)
            if jid is None:
                raise RuntimeError(
                    "sbatch succeeded but job ID could not be parsed; refusing to continue."
                )
        receipts.append(dict(
            bundle_id=bid, job_id=jid, returncode=rc,
            submitted_at=datetime.now().isoformat(), command=" ".join(cmd)
        ))

    receipt = recdir / f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with receipt.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bundle_id","job_id","returncode","submitted_at","command"])
        w.writeheader(); w.writerows(receipts)
    print("Wrote", receipt)


if __name__ == "__main__":
    main()
