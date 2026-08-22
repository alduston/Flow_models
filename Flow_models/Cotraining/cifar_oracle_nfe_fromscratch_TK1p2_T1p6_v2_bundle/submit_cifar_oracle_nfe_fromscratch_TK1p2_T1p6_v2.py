#!/usr/bin/env python3
"""Submit five self-contained representation jobs; each yields four NFE curve points."""
from __future__ import annotations
import argparse,csv,os,re,subprocess,sys
from datetime import datetime
from pathlib import Path

BASE_DIR=Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
MANIFEST="cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_manifest.csv"
GRID="generate_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2.py"
SLURM="cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_cell_job.slurm"
RUNNER="run_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_cell.py"
TARGET="csem_oracle_nfe_fromscratch_TK1p2_T1p6_v2.py"
LOG_DIR="slurm_logs_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2"
RECEIPT_DIR="submission_receipts_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2"
N_CELLS=5
ENV_NAME="CSEM_ORACLE_NFE_REP_CELL"


def parse_cells(spec:str)->list[int]:
    if spec.strip().lower()=="all": return list(range(N_CELLS))
    out=set()
    for part in spec.split(","):
        part=part.strip()
        if not part: continue
        if "-" in part:
            a,b=part.split("-",1); lo,hi=sorted((int(a),int(b))); out.update(range(lo,hi+1))
        else: out.add(int(part))
    cells=sorted(out)
    if not cells or any(c<0 or c>=N_CELLS for c in cells): raise ValueError(f"Cells must be subset 0..{N_CELLS-1}: {cells}")
    return cells


def parse_job_id(text:str):
    hits=re.findall(r"Submitted\s+batch\s+job\s+(\d+)",text,flags=re.I)
    if hits:return hits[-1]
    for line in reversed(text.splitlines()):
        m=re.fullmatch(r"\s*(\d+)(?:;[^\s]+)?\s*",line)
        if m:return m.group(1)
    return None


def ensure_manifest(base:Path):
    p=base/MANIFEST
    if not p.is_file():
        proc=subprocess.run([sys.executable,str(base/GRID)],cwd=str(base))
        if proc.returncode: raise RuntimeError("Manifest generation failed")
    with p.open(newline="") as f: rows=list(csv.DictReader(f))
    if len(rows)!=N_CELLS: raise RuntimeError(f"Expected {N_CELLS} rows, found {len(rows)}")
    return rows


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--base-dir",type=Path,default=BASE_DIR); ap.add_argument("--cells",default="all"); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args()
    base=args.base_dir.resolve()
    if not base.is_dir(): raise SystemExit(f"Base directory not found: {base}")
    for n in (TARGET,RUNNER,SLURM):
        if not (base/n).is_file(): raise SystemExit(f"Missing required file: {base/n}")
    rows=ensure_manifest(base); cells=parse_cells(args.cells)
    log_dir=base/LOG_DIR; receipt_dir=base/RECEIPT_DIR; log_dir.mkdir(parents=True,exist_ok=True); receipt_dir.mkdir(parents=True,exist_ok=True)
    receipts=[]
    print(f"Submitting {len(cells)} self-contained representation jobs. Each trains once and produces four NFE points; no arrays.")
    for cell in cells:
        row=next(r for r in rows if int(r["cell_id"])==cell)
        out=log_dir/f"oracle_nfe_rep_{cell:02d}_%j.out"
        env=os.environ.copy(); env[ENV_NAME]=str(cell)
        cmd=["sbatch","-J",f"onfe2{cell:02d}","-o",str(out),"--export=ALL",str(base/SLURM)]
        display=f"{ENV_NAME}={cell} "+" ".join(cmd)
        print("\n$ "+display)
        if args.dry_run: job_id,rc,raw="DRY_RUN",0,""
        else:
            proc=subprocess.run(cmd,cwd=str(base),text=True,capture_output=True,env=env); raw=(proc.stdout or "")+(proc.stderr or ""); rc=proc.returncode
            if raw: print(raw,end="" if raw.endswith("\n") else "\n")
            if rc!=0: raise RuntimeError(f"sbatch failed for cell {cell}: rc={rc}")
            job_id=parse_job_id(raw)
            if job_id is None: raise RuntimeError("sbatch succeeded but job ID could not be parsed; stopping to prevent duplicates")
        receipts.append({"cell_id":cell,"rep_id":row["rep_id"],"job_id":job_id,"returncode":rc,"submitted_at":datetime.now().isoformat(),"command":display})
        print(f"Recorded cell {cell} ({row['rep_id']}) -> {job_id}")
    receipt=receipt_dir/("submission_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".csv")
    with receipt.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["cell_id","rep_id","job_id","returncode","submitted_at","command"]); w.writeheader(); w.writerows(receipts)
    print(f"\nWrote {receipt}")
    return 0

if __name__=="__main__": raise SystemExit(main())
