#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re, subprocess, sys
from pathlib import Path

BASE = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep')
MANIFEST = 'cifar_cfg_nfe_fresh500_v2_manifest.csv'
SLURM = 'cifar_cfg_nfe_fresh500_v2_cell_job.slurm'

def parse_cells(spec: str, valid: set[int]) -> list[int]:
    if spec == 'all': return sorted(valid)
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            a,b = map(int, part.split('-',1)); out.update(range(min(a,b), max(a,b)+1))
        else: out.add(int(part))
    bad = out - valid
    if bad: raise SystemExit(f'Unknown cell ids: {sorted(bad)}')
    return sorted(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE)
    ap.add_argument('--cells', default='all', help='all, comma list, or ranges; cells 0/1 are seeds 42/43')
    ap.add_argument('--dry-run', action='store_true')
    a=ap.parse_args(); base=a.base_dir.resolve()
    with (base/MANIFEST).open(newline='') as f: rows=list(csv.DictReader(f))
    by_id={int(r['cell_id']):r for r in rows}; selected=parse_cells(a.cells,set(by_id))
    print(f'Selected {len(selected)} fresh-training job(s): {selected}')
    (base / 'slurm_logs_cifar_cfg_nfe_fresh500_v2').mkdir(parents=True, exist_ok=True)
    for cid in selected:
        r=by_id[cid]
        cmd=['sbatch', f'--export=ALL,CELL_ID={cid},BASE_DIR={base}', str(base/SLURM)]
        print(' '.join(map(str,cmd)), f"# seed={r['seed']} train500 -> 5 CFG x 3 NFE")
        if a.dry_run: continue
        q=subprocess.run(cmd,cwd=base,text=True,capture_output=True)
        text=(q.stdout or '')+'\n'+(q.stderr or '')
        if q.returncode != 0:
            print(text,file=sys.stderr); return q.returncode
        m=re.search(r'Submitted batch job\s+(\d+)',text)
        print((q.stdout or '').strip() or f'submitted job {m.group(1) if m else "(id not parsed)"}')
    return 0

if __name__=='__main__': raise SystemExit(main())
