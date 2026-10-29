"""Frozen recipes, deterministic DAG and content-addressed stage identity."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = 'fmnist_flat_comparison_v1'
BETAS = (0., .01, .03, .07, .15, .30)
LRS = (2e-4/.6, 8e-4)
FRESH = (47,48,49,50,51)
REGIMES = ('cotrained_csem','indep_tweedie','indep_csem')

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def object_sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def sources(base=ROOT):
    base=Path(base)
    names=sorted([p.relative_to(base).as_posix() for p in base.glob('*.py')]+
                 [p.relative_to(base).as_posix() for p in (base/'pinned').glob('*.py')]+
                 ['job.slurm','provenance/flat_three_encoder_cell2.json'])
    return {n:sha(base/n) for n in names}

def common(base=ROOT):
    c=read(Path(base)/'provenance/flat_three_encoder_cell2.json')['config']
    # Only evaluation/probe cadence changes; the joint training objective is inherited.
    c.update(archive_epochs=list(range(50,501,50)),diagnostic_at_start=False,
             diagnostic_every=0,witness_every=0,sampling_check_every=0,
             split='train50k_validation10k',eval_split='validation',
             split_seed=20261028,validation_per_class=1000,eval_seed=83042,
             lr_prior=LRS[0],beta0=None,stage='joint',regime='cotrained_csem',
             source=None,source_selection=None,checkpoint_every=1,
             model_version='fmnist_capacity_matched_v1',campaign_version=VERSION,
             kid_subsets=20,kid_subset_size=1000,kid_seed=91000,
             save_array_samples=100,device='cuda',max_batches=0)
    assert c['arm']=='flat_three_encoder' and c['stochastic_blocks']==1
    assert c['blocks']==3 and c['topology']=='decoupled'
    return c

def graph(base=ROOT, include_fast_lr=True):
    nodes=[]
    def add(name,kind='train',deps=(),**kw):
        r=dict(name=name,kind=kind,deps=list(deps),**kw);nodes.append(r);return name
    screen=[]
    screen.append(add('screen_joint_s42',group='screen',stage='joint',seed=42))
    for b in BETAS:
        screen.append(add(f'screen_vae_b{b:g}_s42',group='screen',stage='vae_pretrain',seed=42,beta0=b))
    add('shortlist',kind='shortlist',deps=screen,group='screen')
    dev=[]
    for k in (0,1):
        parent=add(f'dev_vae_k{k}_s43',deps=['shortlist'],group='development',stage='vae_pretrain',seed=43,beta_slot=k)
        for seed in (42,43):
            for li,lr in enumerate(LRS if include_fast_lr else LRS[:1]):
                for stage in ('prior_dsm','prior_csem'):
                    dev.append(add(f'dev_{stage}_k{k}_lr{li}_s{seed}',deps=['shortlist']+([parent] if seed==43 else []),
                                   group='development',stage=stage,seed=seed,beta_slot=k,lr_prior=lr))
    add('freeze',kind='freeze',deps=dev,group='development')
    final=[]
    for seed in FRESH:
        final.append(add(f'final_joint_s{seed}',deps=['freeze'],group='final',stage='joint',seed=seed))
        parent=add(f'final_vae_s{seed}',deps=['freeze'],group='final',stage='vae_pretrain',seed=seed,selected=True)
        final.append(parent)
        for stage in ('prior_dsm','prior_csem'):
            final.append(add(f'final_{stage}_s{seed}',deps=[parent,'freeze'],group='final',stage=stage,seed=seed,selected=True))
    add('report',kind='report',deps=final,group='final')
    return nodes

def campaign(base):
    p=Path(base)/'campaign.json'
    return read(p) if p.exists() else dict(include_fast_lr=True)

def rows(base=ROOT):
    return graph(base,campaign(base)['include_fast_lr'])

def resolve(row,base=ROOT,data_root=None,device=None):
    base=Path(base);c=common(base)
    c.update({k:row[k] for k in ('stage','seed','beta0','lr_prior') if k in row})
    if 'beta_slot' in row:
        selection=read(base/'selections/shortlist.json')
        c['beta0']=selection['betas'][row['beta_slot']]
        c['source_selection']=sha(base/'selections/shortlist.json')
    if row.get('selected'):
        selection=read(base/'selections/final.json')
        c.update(beta0=selection['beta0'],lr_prior=selection['lr_prior'],source_selection=sha(base/'selections/final.json'))
    stage=c['stage'];seed=c['seed']
    c['regime']={'joint':'cotrained_csem','vae_pretrain':'independent_vae',
                 'prior_dsm':'indep_tweedie','prior_csem':'indep_csem'}[stage]
    if stage.startswith('prior'):
        if row.get('selected'):source=f'final_vae_s{seed}'
        elif seed==42:source=f'screen_vae_b{c["beta0"]:g}_s42'
        else:source=f'dev_vae_k{row["beta_slot"]}_s43'
        c['source']=source
    if row.get('group')=='final':c['eval_split']='test'
    c.update(name=row['name'],group=row.get('group','pilot'),output=str(base/'results'/row['name']),
             data_root=str(Path(data_root or os.environ.get('CSEM_DATA_ROOT',base/'data_cache_fmnist')).resolve()))
    if device:c['device']=device
    return argparse.Namespace(**c)

def resolved_contract(c):
    keys=('campaign_version','split','eval_split','split_seed','validation_per_class','eval_samples',
          'eval_seed','rk4_steps','t_min','t_max','decode_time','ema','eval_batch','feature_batch')
    return {k:getattr(c,k) for k in keys}|dict(backend='inception-legacy',temperature=1.,
             start='gaussian',cfg=[1.,3.],class_schedule='reference labels',feature_dimension=2048)

def write_manifests(base=ROOT):
    base=Path(base)
    rr=rows(base)
    atomic(base/'manifests/development.json',[r for r in rr if r['group']!='final'])
    atomic(base/'manifests/final.json',[r for r in rr if r['group']=='final'])
    atomic(base/'manifests/campaign_dag.json',rr)
    return rr
