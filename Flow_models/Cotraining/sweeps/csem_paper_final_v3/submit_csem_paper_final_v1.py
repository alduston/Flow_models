#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re, subprocess, sys
from pathlib import Path

MANIFEST='csem_paper_final_v1_manifest.csv'
SLURM='csem_paper_final_v1_cell_job.slurm'
DEFAULT_BASE=Path(__file__).resolve().parent

def parse_spec(spec, rows):
    valid={int(r['cell_id']) for r in rows}
    families={r['family'] for r in rows}
    if spec=='all': return sorted(valid)
    if spec in families: return sorted(int(r['cell_id']) for r in rows if r['family']==spec)
    out=set()
    for part in spec.split(','):
        part=part.strip()
        if not part: continue
        if '-' in part:
            a,b=map(int,part.split('-',1)); out.update(range(min(a,b),max(a,b)+1))
        else: out.add(int(part))
    bad=out-valid
    if bad: raise SystemExit(f'Unknown cells {sorted(bad)}')
    return sorted(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-dir',type=Path,default=DEFAULT_BASE)
    ap.add_argument('--cells',default='all',help='all, family name, comma list, or ranges')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(); base=a.base_dir.resolve()
    with (base/MANIFEST).open(newline='') as f: rows=list(csv.DictReader(f))
    by={int(r['cell_id']):r for r in rows}; selected=parse_spec(a.cells,rows)
    (base/'slurm_logs_csem_paper_final_v1').mkdir(parents=True,exist_ok=True)
    print(f'Selected {len(selected)} jobs: {selected}')
    for cid in selected:
        r=by[cid]
        cmd=['sbatch',f'--export=ALL,CELL_ID={cid},BASE_DIR={base}',str(base/SLURM)]
        print(' '.join(map(str,cmd)), f"# {r['family']} {r['dataset']} {r['mode']} seed={r['seed']} {r['result_name']}")
        if a.dry_run: continue
        q=subprocess.run(cmd,cwd=base,text=True,capture_output=True)
        text=(q.stdout or '')+'\n'+(q.stderr or '')
        if q.returncode:
            print(text,file=sys.stderr); return q.returncode
        m=re.search(r'Submitted batch job\s+(\d+)',text)
        print((q.stdout or '').strip() or f'submitted {m.group(1) if m else "?"}')
    return 0
if __name__=='__main__': raise SystemExit(main())
