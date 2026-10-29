"""Strict stage completion, validation-only selection and Slurm dependencies."""
from __future__ import annotations
import csv
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from protocol import (atomic,read,sha,object_sha,sources,rows,resolve,BETAS,LRS,FRESH,VERSION)

def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def validate_complete(out,production=False):
    out=Path(out);done=read(out/'complete.json');c=read(out/'config.json')
    if done['version']!=VERSION or done['epoch']!=c['epochs'] or done['recipe_sha256']!=c['recipe_sha256']:
        raise ValueError(f'Invalid completion: {out}')
    if production and done['synthetic']:raise ValueError('Synthetic metrics cannot enter production selection/report')
    if not done['parity']['passed']:raise ValueError('Reload parity has not passed')
    snapshot=out/'snapshots'/f'epoch_{c["epochs"]:04d}.pt'
    if not snapshot.exists():raise FileNotFoundError(snapshot)
    if done.get('snapshot_sha256')!=sha(snapshot):raise ValueError(f'Completed snapshot checksum mismatch: {snapshot}')
    values=csv_rows(out/'eval.csv');keys=set();observed=set()
    expected_backend='synthetic-smoke-only' if done['synthetic'] else 'inception-legacy'
    for r in values:
        key=(int(r['epoch']),float(r['cfg']),r['kind'],r['eval_split'])
        if key in keys:raise ValueError('Duplicate evaluation row')
        keys.add(key)
        if r['backend']!=expected_backend or int(r['num_eval'])!=c['eval_samples'] or r['eval_split']!=c['eval_split']:
            raise ValueError('Mixed evaluation contract')
        if int(r['seed'])!=c['seed'] or r['regime']!=c['regime']:raise ValueError('Evaluation identity drift')
        if int(r['steps'])!=c['rk4_steps'] or float(r['temperature'])!=1. or r['init']!='gaussian' or float(r['decode_time'])!=c['decode_time']:
            raise ValueError('Sampler contract drift')
        for field in ('fid_posterior_recon','training_seconds','end_to_end_training_gpu_hours'):
            if not math.isfinite(float(r[field])):raise ValueError('Nonfinite metric')
        if r['kind']=='generation' and not math.isfinite(float(r['fid_generated'])):raise ValueError('Nonfinite generation FID')
        observed.add((int(r['epoch']),float(r['cfg'])))
    epochs=set(range(c['eval_every'],c['epochs']+1,c['eval_every']))|{c['epochs']}
    cfgs=(0.,) if c['stage']=='vae_pretrain' else (1.,3.)
    if observed!={(e,g) for e in epochs for g in cfgs}:
        raise ValueError(f'Missing/extra checkpoints: {out}; observed={sorted(observed)}; expected={sorted((e,g) for e in epochs for g in cfgs)}; stage={c["stage"]}')
    losses=csv_rows(out/'loss.csv')
    if [int(r['epoch']) for r in losses]!=list(range(1,c['epochs']+1)):raise ValueError('Training history incomplete')
    return done

def endpoint(out,cfg=3.,production=True):
    validate_complete(out,production)
    c=read(Path(out)/'config.json')
    values=[r for r in csv_rows(Path(out)/'eval.csv') if int(r['epoch'])==c['epochs'] and float(r['cfg'])==cfg]
    if len(values)!=1:raise ValueError('Missing unique endpoint')
    return values[0]

def freeze_write(path,payload):
    path=Path(path)
    if path.exists():
        if read(path)!=payload:raise ValueError(f'Selection is already frozen: {path}; use a new bundle directory for a different campaign')
        return
    atomic(path,payload)

def shortlist(base,betas=None):
    base=Path(base);ref=endpoint(base/'results/screen_joint_s42')
    target=float(ref['fid_posterior_recon']);table=[]
    for beta in BETAS:
        out=base/'results'/f'screen_vae_b{beta:g}_s42';r=endpoint(out,0.)
        diag=read(out/'diagnostics_epoch_0500.json')
        table.append(dict(beta0=beta,reconstruction_fid=float(r['fid_posterior_recon']),
                          kl0=diag['kl0'],posterior_variance=diag['posterior_variance'],
                          recipe_sha256=read(out/'complete.json')['recipe_sha256']))
    positive=[r for r in table if r['beta0']>0]
    frontier=[r for r in positive if not any(q['reconstruction_fid']<=r['reconstruction_fid'] and q['kl0']<=r['kl0'] and
        (q['reconstruction_fid']<r['reconstruction_fid'] or q['kl0']<r['kl0']) for q in positive)]
    matched=min(positive,key=lambda r:(abs(math.log(max(r['reconstruction_fid'],1e-9)/max(target,1e-9))),r['beta0']))
    if betas is None:
        if abs(matched['reconstruction_fid']/max(target,1e-9)-1)>.5:
            raise RuntimeError('No positive beta lies within 50% of joint reconstruction FID. Extend the beta screen before final confirmation.')
        pool=[r for r in frontier if r['beta0']!=matched['beta0']]
        if not pool:pool=[r for r in positive if r['beta0']!=matched['beta0']]
        # Prefer the other side of the reconstruction target, then the closest frontier point.
        alternate=min(pool,key=lambda r:((r['reconstruction_fid']-target)*(matched['reconstruction_fid']-target)>0,
                                        abs(math.log(max(r['reconstruction_fid'],1e-9)/max(target,1e-9))),r['beta0']))
        betas=[matched['beta0'],alternate['beta0']]
    if len(betas)!=2 or len(set(betas))!=2 or any(b not in BETAS or b<=0 for b in betas):raise ValueError('Choose exactly two screened positive beta values')
    payload=dict(betas=sorted(betas),development_seeds=[42,43],validation_only=True,
        co_trained_reconstruction_fid=target,reconstruction_matched_beta=matched['beta0'],
        nondominated_positive_betas=[r['beta0'] for r in frontier],screen=table,
        rule='closest positive reconstruction match plus nearest nondominated point, preferring the opposite distortion side; beta=0 is a control',
        source_sha256=object_sha(sources(base)))
    freeze_write(base/'selections/shortlist.json',payload);print(json.dumps(payload,indent=2));return payload

def freeze(base):
    base=Path(base);rr=[r for r in rows(base) if r.get('group')=='development' and r.get('stage','').startswith('prior')]
    grid={};identities={};observed=[]
    for r in rr:
        out=base/'results'/r['name'];ep=endpoint(out);c=read(out/'config.json')
        if c['eval_split']!='validation' or c['seed'] not in (42,43):raise ValueError('Selection must use development validation only')
        key=(c['beta0'],c['lr_prior']);grid.setdefault(key,[]).append(float(ep['fid_generated']))
        pair=(c['beta0'],c['lr_prior'],c['seed'])
        identity=(c['source_sha256'],c['initial_head_sha256'],c['split_metadata']['hash'])
        if pair in identities and identities[pair]!=identity:raise ValueError('Paired priors do not share VAE, initialization and split')
        identities[pair]=identity
        observed.append(dict(name=r['name'],beta0=c['beta0'],lr_prior=c['lr_prior'],seed=c['seed'],stage=c['stage'],
                             fid=float(ep['fid_generated']),recipe_sha256=c['recipe_sha256'],source_sha256=c['source_sha256']))
    if not grid or any(len(v)!=4 for v in grid.values()):raise ValueError('Both objectives and both development seeds are required in every candidate')
    ranked=sorted((sum(v)/len(v),b,lr) for (b,lr),v in grid.items());score,beta,lr=ranked[0]
    if any((base/'results'/r['name']).exists() for r in rows(base) if r['group']=='final' and r['kind']=='train') and not (base/'selections/final.json').exists():
        raise ValueError('Fresh seeds were started before selection was frozen')
    payload=dict(beta0=beta,lr_prior=lr,mean_validation_fid_over_objectives_and_seeds=score,
        criterion='mean CFG3 epoch500 validation FID over DSM/CSEM and seeds42/43; deterministic ties: smaller beta then LR',
        candidates=[dict(mean_validation_fid=s,beta0=b,lr_prior=l) for s,b,l in ranked],observations=observed,
        shortlist_sha256=sha(base/'selections/shortlist.json'),fresh_seeds=list(FRESH),source_sha256=object_sha(sources(base)),
        final_epochs=500,pretraining_epochs=500,training_split='train50k_validation10k',evaluation_split='test',
        reconstruction_matched_beta=read(base/'selections/shortlist.json')['reconstruction_matched_beta'])
    freeze_write(base/'selections/final.json',payload)
    atomic(base/'manifests/final_resolved.json',[dict(name=r['name'],config=vars(resolve(r,base))) for r in rows(base) if r['kind']=='train' and r['group']=='final'])
    print(json.dumps(payload,indent=2));return payload

def signature(row,base):return object_sha(dict(row=row,sources=sources(base)))

def estimate(base,nodes):
    p=Path(base)/'pilot/timing.json'
    if not p.exists():return dict(known=False,message='Run the Vista pilot before production submission')
    pilot=read(p);stages=pilot['stages'];total=0.;by={}
    for r in nodes:
        if r['kind']!='train':continue
        s=stages[r['stage']]
        hours=(s['seconds_per_batch']*(50000//128)*500+s['evaluation_seconds']*10)/3600
        by[r['name']]=hours;total+=hours
    rate=os.environ.get('VISTA_SU_PER_NODE_HOUR')
    return dict(known=True,training_tasks=len(by),estimated_gpu_node_hours=total,
        estimated_SUs=total*float(rate) if rate else None,SU_rate_per_node_hour=float(rate) if rate else None,
        SU_note='Set VISTA_SU_PER_NODE_HOUR to your allocation billing rate; no billing factor is assumed',
        safety_multiplier_for_walltime=1.7,per_task_hours=by,pilot_gpu=pilot['gpu'],
        caveat='Pilot-based estimate includes evaluations, excludes queue time; 500 epochs and 10 evaluation checkpoints')

def submit(base,phase='all',dry=False,account=None,partition='gh',time_limit='48:00:00',data_root=None,repair=False):
    base=Path(base).resolve();(base/'status').mkdir(exist_ok=True);(base/'slurm_logs').mkdir(exist_ok=True)
    selected=[dict(name='pilot',kind='pilot',deps=[])] if phase=='pilot' else rows(base)
    if phase not in ('pilot','all'):selected=[r for r in selected if r['group']==phase]
    cost=estimate(base,selected) if phase!='pilot' else dict(pilot=True,time_limit='02:00:00')
    print('[PROJECTED COST]',json.dumps(cost),flush=True)
    if phase!='pilot' and not cost['known'] and not dry:raise RuntimeError('Production submission requires the completed timed Vista pilot')
    if cost.get('known'):atomic(base/'projected_cost.json',cost)
    ledger_path=base/'status/submissions.json'
    with (base/'status/submit.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        ledger=read(ledger_path) if ledger_path.exists() else {};active={}
        if not dry:
            q=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%r'],check=True,capture_output=True,text=True)
            for line in q.stdout.splitlines():
                a=line.split('|',2)
                if len(a)==3:active[a[0]]=(a[1],a[2])
        queued={}
        for r in selected:
            name=r['name'];sig=signature(r,base);previous=ledger.get(name,{})
            if previous and previous['signature']!=sig:raise ValueError(f'Frozen source/recipe changed: {name}')
            if task_complete(base,r):print('[SKIP complete]',name);continue
            jid=str(previous.get('job_id',''))
            if jid in active:
                if repair and active[jid][0]=='PENDING' and active[jid][1] in ('DependencyNeverSatisfied','Dependency'):
                    if not dry:subprocess.run(['scancel',jid],check=True)
                else:queued[name]=jid;print('[SKIP active]',name,jid);continue
            deps=[]
            for parent in r['deps']:
                pr=next(x for x in rows(base) if x['name']==parent)
                if task_complete(base,pr):continue
                dep=queued.get(parent)
                if dep is None:
                    old_job=ledger.get(parent,{}).get('job_id')
                    if str(old_job) in active:dep=str(old_job)
                if dep is None:raise ValueError(f'Missing successful or queued dependency {parent} for {name}; submit --phase all')
                deps.append(dep)
            exports=dict(BASE_DIR=str(base),TASK=name)
            if data_root:exports['CSEM_DATA_ROOT']=str(Path(data_root).resolve())
            if any(',' in v or '\n' in v for v in exports.values()):raise ValueError('Slurm export contains comma/newline')
            wall=('02:00:00' if time_limit=='auto' else time_limit) if phase=='pilot' else ('48:00:00' if time_limit=='auto' else time_limit) if r['kind']=='train' else '00:30:00'
            if cost.get('known') and r['kind']=='train' and time_limit=='auto':
                minutes=max(60,math.ceil(cost['per_task_hours'][name]*1.7*60));wall=f'{min(minutes//60,48):02d}:{minutes%60 if minutes<2880 else 0:02d}:00'
            cmd=['sbatch','--parsable',f'--job-name=fm3_{name}',f'--partition={partition}',f'--time={wall}',
                 f'--chdir={base}','--export=ALL,'+','.join(f'{k}={v}' for k,v in exports.items())]
            if account:cmd.append(f'--account={account}')
            if deps:cmd.append('--dependency=afterok:'+':'.join(deps))
            cmd.append(str(base/'job.slurm'));print(shlex.join(cmd),flush=True)
            if dry:queued[name]=f'<{name}>';continue
            q=subprocess.run(cmd,check=True,capture_output=True,text=True,cwd=base)
            m=re.fullmatch(r'(\d+)(?:;[^\s]+)?',q.stdout.strip())
            if not m:raise RuntimeError('Unrecognized sbatch response; inspect squeue before retry')
            queued[name]=m[1];ledger[name]=dict(job_id=m[1],signature=sig,submitted_unix=time.time(),row=r)
            atomic(ledger_path,ledger)
    return 0

def task_complete(base,row):
    base=Path(base);kind=row['kind'];name=row['name']
    if kind=='train':
        out=base/'results'/name
        if not (out/'complete.json').exists():return False
        validate_complete(out,True)
        c=read(out/'config.json');a=resolve(row,base,c['data_root'])
        if c.get('source_hashes')!=sources(base):raise ValueError(f'Completed stage sources changed: {name}')
        if {k:c[k] for k in vars(a)}!=vars(a):raise ValueError(f'Completed resolved recipe changed: {name}')
        return True
    path={'pilot':'pilot/timing.json','shortlist':'selections/shortlist.json','freeze':'selections/final.json','report':'compiled/complete.json'}[kind]
    return (base/path).exists()
