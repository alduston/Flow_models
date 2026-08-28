#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,re,subprocess,sys
from pathlib import Path
BASE=Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/tk_anchor_sweep'); MANIFEST='cifar_tk_anchor_compare_v1_manifest.csv'; SLURM='cifar_tk_anchor_compare_v1_cell_job.slurm'
def parse(spec,valid):
    if spec=='all': return sorted(valid)
    out=set()
    for p in spec.split(','):
        p=p.strip()
        if not p: continue
        if '-' in p:
            a,b=map(int,p.split('-',1)); out.update(range(min(a,b),max(a,b)+1))
        else: out.add(int(p))
    bad=out-valid
    if bad: raise SystemExit(f'Unknown cells: {sorted(bad)}')
    return sorted(out)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=BASE); ap.add_argument('--cells',default='all'); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args(); base=a.base_dir.resolve()
    with (base/MANIFEST).open(newline='') as f: rows=list(csv.DictReader(f))
    by={int(r['cell_id']):r for r in rows}; ids=parse(a.cells,set(by)); print(f'Selected {len(ids)} / {len(rows)} jobs: {ids}')
    (base/'slurm_logs_cifar_tk_anchor_compare_v1').mkdir(parents=True,exist_ok=True)
    for cid in ids:
        r=by[cid]; cmd=['sbatch',f'--export=ALL,CELL_ID={cid},BASE_DIR={base}',str(base/SLURM)]
        print(' '.join(map(str,cmd)),f"# {r['arm_id']} seed={r['seed']} anchor={r['anchor_mode']} KL={r['terminal_kl_w']}")
        if a.dry_run: continue
        q=subprocess.run(cmd,cwd=base,text=True,capture_output=True); text=(q.stdout or '')+'\n'+(q.stderr or '')
        if q.returncode: print(text,file=sys.stderr); return q.returncode
        print((q.stdout or '').strip())
    return 0
if __name__=='__main__': raise SystemExit(main())
