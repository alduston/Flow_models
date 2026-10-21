"""Small, paired cross-dataset replication; no best-checkpoint selection."""
from pathlib import Path
import csv, json, hashlib

ROOT=Path(__file__).resolve().parent

def rows():
    template=list(csv.DictReader((ROOT/'pinned/direct_cifar_manifest.csv').open()))[2]
    out=[]
    arms=[('flat',1,'full',1),('stack_r0',2,'full',0),('stack_r05',2,'full',.5),
          ('stack_r1',2,'full',1),('stack_r4',2,'full',4),('stack_decoupled',2,'decoupled',1)]
    for name,blocks,topology,r in arms:
        for seed in (42,43):
            row=dict(template,cell_id=str(len(out)),result_name=f'cifar_{name}_s{seed}',seed=str(seed),
                     blocks=str(blocks),topology=topology,rep_multiplier=str(r),stage='joint',source='',teacher='naive',
                     group='core',architecture_label=name,block_channels='8' if blocks==1 else '2:6',
                     block_coordinates='512' if blocks==1 else '128:384',score_depths='12',root_size='4',
                     epochs_joint='800',epochs_refine='0',notes='CIFAR recipe; one joint DiT and reverse solve')
            out.append(row)
    for source in list(out):
        if source['result_name'].rsplit('_s',1)[0] not in ('cifar_stack_r1','cifar_stack_r4'): continue
        for teacher in ('naive','corrected','dsm'):
            out.append(dict(source,cell_id=str(len(out)),result_name=source['result_name']+'_frozen_'+teacher,
                stage='frozen',group='frozen',source=source['cell_id'],teacher=teacher,epochs_joint='50',eval_every='50'))
    return out

def select(spec, manifest=None):
    rr=rows() if manifest is None else manifest
    if spec=='all':return rr
    if spec in ('core','frozen'):return [r for r in rr if r['group']==spec]
    if spec=='baseline':return rr[:2]
    if spec=='pilot':return [rr[i] for i in (0,2,4,6,8,10)]
    ids=set()
    for part in spec.split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));ids.update(range(lo,hi+1))
        else:ids.add(int(part))
    if ids-{int(r['cell_id']) for r in rr}:raise ValueError('Unknown cell')
    return [r for r in rr if int(r['cell_id']) in ids]

def source_hash():
    names=['core.py','suite.py','paper_io.py','direct_models.py','hier_models.py','band_regularizer.py',
           'design.py','rng_state.py','stacked.py','engine.py','audit.py','frozen.py','paper.py','evaluation.py']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}

def config_for(row,base,data_root=None):
    from suite import resolve_config
    cfg=resolve_config(row,Path(base))
    cfg.update(blocks=int(row['blocks']),topology=row['topology'],rep_multiplier=float(row['rep_multiplier']),
               stage=row['stage'],teacher=row['teacher'],source_cell=row['source'],
               block_channels=[8] if row['blocks']=='1' else [2,6],root_size=4,
               tmin_band_weight='matched',fixed_band_weight=.05,latent_coordinates=512,
               skip_lsi_gap=True,eval_tk_vs_t_comparison=False,eval_oracle=False,eval_oracle_diagnostics=False,
               paper_cfg_grid=[1.,2.5,3.],audit_times=[.01,.2,1.05,1.35],audit_samples=256,
               witness_train=2048,witness_test=1024,witness_times=[.2],audit_batch=8,
               training_engine='cifar_joint_stacked_v1',paper_checkpoint_every=10)
    if data_root:cfg['data_root']=str(Path(data_root).resolve())
    # Scaling is at the representation-loss boundary only. B is fixed at .05.
    science={k:v for k,v in row.items() if k not in ('notes','outputs')}
    cfg['bundle_recipe_hash']=hashlib.sha256(json.dumps([science,source_hash()],sort_keys=True).encode()).hexdigest()
    return cfg

def export(base):
    base=Path(base);rr=rows()
    (base/'manifest.json').write_text(json.dumps(rr,indent=2)+'\n')
    with (base/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
