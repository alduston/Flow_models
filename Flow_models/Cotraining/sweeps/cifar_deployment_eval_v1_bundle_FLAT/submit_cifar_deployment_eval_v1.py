#!/usr/bin/env python3
import argparse,csv,subprocess,sys
from pathlib import Path
BASE=Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep")
MANIFEST='cifar_deployment_eval_v1_manifest.csv'; SLURM='cifar_deployment_eval_v1_cell_job.slurm'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=BASE); ap.add_argument('--group',choices=['primary','robustness','all'],default='primary'); ap.add_argument('--cells',default='all',help='all or comma-separated cell ids'); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args(); base=a.base_dir.resolve()
    with (base/MANIFEST).open(newline='') as f: rows=list(csv.DictReader(f))
    if a.group!='all': rows=[r for r in rows if r['sweep_group']==a.group]
    if a.cells!='all':
        wanted={int(x) for x in a.cells.split(',') if x.strip()}; rows=[r for r in rows if int(r['cell_id']) in wanted]
    if not rows: raise SystemExit('No cells selected')
    print(f"Selected {len(rows)} jobs")
    for r in rows:
        cmd=['sbatch',f"--export=ALL,CELL_ID={r['cell_id']},BASE_DIR={base}",str(base/SLURM)]; print(' '.join(map(str,cmd)),f"# {r['sweep_group']} {r['geometry']} s{r['seed']} NFE={r['nfe']}")
        if not a.dry_run:
            q=subprocess.run(cmd,cwd=base,text=True,capture_output=True)
            if q.stdout: print(q.stdout.strip())
            if q.returncode!=0: print(q.stderr,file=sys.stderr); return q.returncode
    return 0
if __name__=='__main__': raise SystemExit(main())
