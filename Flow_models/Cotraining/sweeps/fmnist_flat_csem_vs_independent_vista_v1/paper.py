#!/usr/bin/env python3
"""Fashion-MNIST: co-trained CSEM versus matched two-stage DSM/CSEM."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback
from protocol import ROOT,atomic,read,rows,resolve,sources,write_manifests

def run_task(base,name,data_root=None):
    import campaign
    row=next((r for r in rows(base) if r['name']==name),None)
    if name=='pilot':row=dict(name='pilot',kind='pilot',deps=[])
    if name=='historical_joint_s42':row=dict(name=name,kind='train',group='historical',stage='joint',seed=42)
    if row is None:raise ValueError(f'Unknown task {name}')
    status=base/'status';status.mkdir(exist_ok=True)
    with (status/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if campaign.task_complete(base,row):print('[SKIP complete]',name);return 0
        for parent in row['deps']:
            pr=next(r for r in rows(base) if r['name']==parent)
            if not campaign.task_complete(base,pr):raise ValueError(f'Uncommitted parent: {parent}')
        record=dict(name=name,slurm_job_id=os.environ.get('SLURM_JOB_ID'),source_hashes=sources(base))
        atomic(status/f'{name}.json',record)
        if row['kind']=='shortlist':campaign.shortlist(base);rc=0
        elif row['kind']=='freeze':campaign.freeze(base);rc=0
        elif row['kind']=='report':
            from compile_results import compile
            rc=compile(base,strict=True)
        elif row['kind']=='pilot':rc=pilot(base,data_root)
        else:
            from train_stages import run
            a=resolve(row,base,data_root)
            if row['group']=='historical':a.split='historical_full60k';a.eval_split='test'
            rc=run(a,base,resume=True)
        record['returncode']=rc;atomic(status/f'{name}.json',record);return rc

def pilot(base,data_root):
    import torch
    import checks
    from train_stages import run
    checks.run(base,device='cuda',models_only=False)
    measurements={}
    for stage in ('joint','vae_pretrain','prior_dsm','prior_csem'):
        row=dict(name=f'pilot_{stage}',kind='train',stage=stage,seed=42,beta0=.03,group='pilot')
        a=resolve(row,base,data_root)
        a.epochs=1;a.max_batches=10;a.num_workers=2;a.archive_epochs=[1];a.eval_every=1
        a.disc_start=1  # Time the adversarial image path as well; this is a pilot only.
        if stage.startswith('prior'):a.source='pilot_vae_pretrain'
        rc=run(a,base,resume=True)
        if rc:return rc
        out=Path(a.output);done=read(out/'complete.json');loss=__import__('campaign').csv_rows(out/'loss.csv')[0]
        measurements[stage]=dict(seconds_per_batch=done['timing']['training_seconds']/done['timing']['optimizer_steps'],
            evaluation_seconds=done['timing']['evaluation_seconds'],batches=done['timing']['optimizer_steps'],
            train_seconds=done['timing']['training_seconds'])
    atomic(base/'pilot/timing.json',dict(gpu=torch.cuda.get_device_name(0),stages=measurements,
        notes='Production architecture, real validation10k, both CFGs, GAN active; 10 timed batches per stage. No training-rate estimate from GAD.'))
    print('[PILOT COMPLETE]',json.dumps(measurements));return 0

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir',type=Path,default=ROOT)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init');q.add_argument('--fixed-lr-only',action='store_true')
    sub.add_parser('plan')
    q=sub.add_parser('prepare-data');q.add_argument('--weights',action='store_true');q.add_argument('--data-root',type=Path,default=os.environ.get('CSEM_DATA_ROOT'))
    q=sub.add_parser('check');q.add_argument('--device',choices=('cpu','cuda'),default='cpu');q.add_argument('--models-only',action='store_true')
    q=sub.add_parser('submit');q.add_argument('--phase',choices=('pilot','all','screen','development','final'),default='all')
    q.add_argument('--dry-run',action='store_true');q.add_argument('--account',default=os.environ.get('VISTA_ACCOUNT'))
    q.add_argument('--partition',default='gh');q.add_argument('--time',default='auto')
    q.add_argument('--data-root',type=Path,default=os.environ.get('CSEM_DATA_ROOT'));q.add_argument('--repair-blocked',action='store_true')
    q=sub.add_parser('run');q.add_argument('--task',required=True);q.add_argument('--data-root',type=Path,default=os.environ.get('CSEM_DATA_ROOT'))
    q=sub.add_parser('shortlist');q.add_argument('--betas',help='Optional two positive grid points, chosen using validation diagnostics only')
    sub.add_parser('freeze')
    q=sub.add_parser('report');q.add_argument('--strict',action='store_true')
    sub.add_parser('status')
    a=p.parse_args(argv);base=a.base_dir.resolve()
    if a.command=='init':
        settings=dict(include_fast_lr=not a.fixed_lr_only)
        path=base/'campaign.json'
        if path.exists() and read(path)!=settings:raise ValueError('Campaign already initialized with different settings; use a new directory')
        atomic(path,settings);write_manifests(base);print(json.dumps(settings));return 0
    if a.command=='plan':
        rr=write_manifests(base)
        for r in rr:print(r['name'],r.get('stage',r['kind']),'after',','.join(r['deps']) or 'nothing')
        print(f'{sum(r["kind"]=="train" for r in rr)} training tasks; 20 final tasks, 15 final generative models.');return 0
    if a.command=='prepare-data':
        import prepare_cache
        prepare_cache.prepare_data(a.data_root or base/'data_cache_fmnist')
        if a.weights:prepare_cache.prepare_weights()
        return 0
    if a.command=='check':
        import checks
        return checks.run(base,a.device,a.models_only)
    if a.command=='submit':
        import campaign
        return campaign.submit(base,a.phase,a.dry_run,a.account,a.partition,a.time,a.data_root,a.repair_blocked)
    if a.command=='run':return run_task(base,a.task,a.data_root)
    if a.command=='shortlist':
        import campaign
        campaign.shortlist(base,list(map(float,a.betas.split(','))) if a.betas else None);return 0
    if a.command=='freeze':
        import campaign
        campaign.freeze(base);return 0
    if a.command=='report':
        import compile_results
        return compile_results.compile(base,a.strict)
    import campaign
    for r in rows(base):
        try:s='complete' if campaign.task_complete(base,r) else 'incomplete'
        except (ValueError,FileNotFoundError) as e:s=f'INVALID: {e}'
        print(f'{r["name"]:36s} {s}')
    return 0

if __name__=='__main__':raise SystemExit(main())
