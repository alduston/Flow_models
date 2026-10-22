"""Predefined mechanistic contrasts; never rank/select recipes by best FID."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

COMMON = dict(epochs=500,batch=128,num_workers=2,seed=42,blocks=2,topology='full',
    rep_multiplier=1.,band_mode='fixed',clip_mode='global',joint_width=192,joint_depth=18,
    joint_heads=6,z1_channels=4,z2_channels=3,lr_vae=5e-4,lr_head=2e-4,score_weight=.6,
    terminal_weight=1.,tmin_band_weight='auto',perc_weight=.85,gan_weight=.0025,
    disc_start=25,lr_disc=1e-4,label_dropout=.1,ema=.9997,t_min=2e-5,t_max=2.,
    decode_time=1e-4,cfg=3.,rk4_steps=20,eval_every=50,eval_samples=10000,eval_batch=128,
    feature_batch=128,checkpoint_every=5,archive_epochs=[100,200,300,400,500],
    diagnostic_every=25,diagnostic_samples=64,diagnostic_particles=8,diagnostic_batch=8,
    diagnostic_times=[2e-5,.001,.01,.05,.2,.5,1.,2.],witness_every=100,
    witness_train=4096,witness_test=1024,witness_features=64,witness_ridge=.001,
    sampling_check_every=100,sampling_check_samples=1024,diagnostic_at_start=True,
    device='cuda',cpu_threads=8,max_batches=0,save_array_samples=256)


def rows():
    designs=[]
    for r in (0.,.5,1.,2.,4.):
        designs.append((f'b2_full_r{r:g}',dict(rep_multiplier=r),'core','encoder-pressure'))
    designs.append(('b2_decoupled_r1',dict(topology='decoupled'),'core','zero-bias-control'))
    for name,changes in [
        ('b3_full_r1',dict(blocks=3)),('b3_full_r4',dict(blocks=3,rep_multiplier=4.)),
        ('b3_markov_r1',dict(blocks=3,topology='markov')),
        ('b3_siblings_r1',dict(blocks=3,topology='siblings')),
        ('b3_decoupled_r1',dict(blocks=3,topology='decoupled'))]:
        designs.append((name,changes,'core','depth-and-dependence'))
    for b in (2,3):
        designs.append((f'b{b}_full_r4_band_scaled',dict(blocks=b,rep_multiplier=4.,band_mode='scaled'),'robustness','boundary-penalty'))
    for r in (1.,4.):
        designs.append((f'b2_full_r{r:g}_separate_clip',dict(rep_multiplier=r,clip_mode='separate'),'robustness','clipping-mediation'))
    out=[]
    for name,changes,group,question in designs:
        for seed in (42,43):
            config=dict(COMMON,**changes);config['seed']=seed
            config['z2_channels']=3 if config['blocks']==2 else 1
            config['z3_channels']=0 if config['blocks']==2 else 2
            out.append(dict(cell_id=len(out),name=f'{name}_s{seed}',group=group,
                            question=question,stage='joint',source=None,config=config))
    sources=[r for r in out if r['name'].rsplit('_s',1)[0] in ('b2_full_r1','b2_full_r4','b3_full_r1','b3_full_r4')]
    for source in sources:
        for teacher in ('naive','corrected','dsm'):
            config=dict(source['config'],epochs=50,lr_frozen=2e-5,ema=.999,
                        diagnostic_at_start=True,checkpoint_every=5,archive_epochs=[50],
                        eval_every=50,diagnostic_every=10,witness_every=50,
                        sampling_check_every=50,teacher=teacher)
            out.append(dict(cell_id=len(out),name=source['name']+'_frozen_'+teacher,
                group='frozen',question='bias-versus-teacher-variance',stage='frozen',
                source=source['cell_id'],config=config))
    return out


def selected(spec, manifest=None):
    manifest=rows() if manifest is None else manifest
    if spec=='all':return manifest
    if spec in ('core','robustness','frozen'):return [r for r in manifest if r['group']==spec]
    if spec=='pilot':return [r for r in manifest if r['name'] in
        ('b2_full_r0_s42','b2_full_r1_s42','b2_full_r4_s42','b2_decoupled_r1_s42','b3_full_r1_s42','b3_full_r4_s42')]
    wanted=set()
    for chunk in spec.split(','):
        if '-' in chunk:
            lo,hi=map(int,chunk.split('-'));wanted.update(range(lo,hi+1))
        else:wanted.add(int(chunk))
    if wanted-{r['cell_id'] for r in manifest}:raise ValueError('Unknown cell IDs')
    return [r for r in manifest if r['cell_id'] in wanted]


def digest(base):
    names=['experiments.py','models.py','vae_three.py','trainer.py','measurements.py',
           'witness.py','sampling.py','reference.py','paper.py','paper_io.py','fmnist_reference.py',
           'pinned/stacked_two_block.py']
    return {n:hashlib.sha256((Path(base)/n).read_bytes()).hexdigest() for n in names}


def configuration(row,base,data_root=None,device=None):
    return argparse.Namespace(**dict(row['config'],output=str(Path(base)/'results'/row['name']),
        data_root=str(Path(data_root) if data_root else Path(base)/'data_cache_fmnist'),
        stage=row['stage'],source=row['source'],cell_id=row['cell_id'],name=row['name'],
        **({'device':device} if device else {})))


def export(base):
    base=Path(base)
    (base/'manifest.json').write_text(json.dumps(rows(),indent=2)+'\n')
    with (base/'manifest.csv').open('w',newline='') as f:
        fields=['cell_id','name','stage','group','question','source','blocks','topology','rep_multiplier','band_mode','clip_mode','seed','epochs','teacher']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for r in rows():wr.writerow({k:r.get(k,r['config'].get(k,'')) for k in fields})
