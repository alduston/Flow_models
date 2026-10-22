"""Predeclared matched comparison; all recipes inherit the actual 4.3 run."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from original_experiments import COMMON as REFERENCE_COMMON

SEEDS = (42,43,44,45,46)
CONFIRMATION_SEEDS = (44,45,46)
ARMS = (
    ('flat',2,'decoupled',1,'reference_two'),
    ('stack2',2,'full',2,'reference_two'),
    ('flat_three_encoder',3,'decoupled',1,'budget_three'),
    ('stack3',3,'full',3,'budget_three'),
)
COMMON = copy.deepcopy(REFERENCE_COMMON)
COMMON['rep_multiplier'] = .5


def rows():
    result=[]
    for seed in SEEDS:
        for arm,blocks,topology,depth,family in ARMS:
            config=dict(COMMON,seed=seed,blocks=blocks,topology=topology,
                stochastic_blocks=depth,capacity_family=family,arm=arm,
                z2_channels=3 if blocks==2 else 1,z3_channels=0 if blocks==2 else 2)
            result.append(dict(cell_id=len(result),name=f'{arm}_s{seed}',stage='joint',
                source=None,group='core',question='matched-stochastic-depth',config=config))
    return result


def selected(spec):
    rr=rows()
    if spec in ('all','core'):return rr
    if spec=='primary':return [r for r in rr if r['config']['arm']!='flat_three_encoder']
    if spec=='confirmation':return [r for r in rr if r['config']['seed'] in CONFIRMATION_SEEDS]
    if spec=='historical-seeds':return [r for r in rr if r['config']['seed'] in (42,43)]
    if spec in ('flat','stack2','stack3','flat_three_encoder'):
        return [r for r in rr if r['config']['arm']==spec]
    wanted=set()
    for chunk in spec.split(','):
        if '-' in chunk:
            lo,hi=map(int,chunk.split('-'));wanted.update(range(lo,hi+1))
        else:wanted.add(int(chunk))
    if wanted-{r['cell_id'] for r in rr}:raise ValueError('Unknown cell IDs')
    return [r for r in rr if r['cell_id'] in wanted]


def digest(base):
    root=Path(base)
    names=sorted([p.relative_to(root).as_posix() for p in root.glob('*.py')]+
                 [p.relative_to(root).as_posix() for p in (root/'pinned').glob('*.py')]+['job.slurm'])
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def configuration(row,base,data_root=None,device=None):
    return argparse.Namespace(**dict(row['config'],output=str(Path(base)/'results'/row['name']),
        data_root=str(Path(data_root) if data_root else Path(base)/'data_cache_fmnist'),
        stage=row['stage'],source=row['source'],cell_id=row['cell_id'],name=row['name'],
        **({'device':device} if device else {})))


def export(base):
    base=Path(base);rr=rows()
    (base/'manifest.json').write_text(json.dumps(rr,indent=2)+'\n')
    with (base/'manifest.csv').open('w',newline='') as f:
        fields=['cell_id','name','arm','seed','stochastic_blocks','blocks','topology',
                'capacity_family','rep_multiplier','epochs','stage']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rr:writer.writerow({k:r.get(k,r['config'].get(k,'')) for k in fields})
