"""Seed-aware strict exports and reproducible manuscript/website figures."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.stats import t
from protocol import read,atomic,FRESH,REGIMES,rows,resolved_contract,resolve
from campaign import csv_rows,validate_complete

LABELS={'cotrained_csem':'Co-trained CSEM','indep_tweedie':'Two-stage, DSM','indep_csem':'Two-stage, CSEM'}
COLORS={'cotrained_csem':'#116e6b','indep_tweedie':'#bd7130','indep_csem':'#4853a4'}

def stats(values):
    v=np.asarray(values,dtype=float)
    if not len(v) or not np.isfinite(v).all():raise ValueError('Invalid statistical input')
    n=len(v);sd=float(v.std(ddof=1)) if n>1 else None
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=float(v.mean()),sd=sd,ci95_low=float(v.mean()-half) if half is not None else None,
                ci95_high=float(v.mean()+half) if half is not None else None)

def write_csv(path,records,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not records and fields is None:return
    fields=fields or list(records[0])
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)

def compile(base,strict=False):
    base=Path(base);dest=base/'compiled';dest.mkdir(exist_ok=True)
    (dest/'complete.json').unlink(missing_ok=True)
    raw=[];status=[];timing=[];contracts=[];configs={}
    expected=[r for r in rows(base) if r['kind']=='train' and r['group']=='final']
    for row in expected:
        out=base/'results'/row['name'];complete=(out/'complete.json').exists();cpath=out/'config.json'
        status.append(dict(name=row['name'],stage=row['stage'],seed=row['seed'],complete=complete))
        if not cpath.exists():continue
        c=read(cpath);configs[row['name']]=c
        if c.get('synthetic'):raise ValueError('Refusing synthetic metrics in final report')
        if complete:
            done=validate_complete(out,True)
            timing.append(dict(name=row['name'],stage=c['stage'],seed=c['seed'],**done['timing']))
        elif strict:raise ValueError(f'Incomplete run: {row["name"]}')
        if not (out/'eval.csv').exists():continue
        if c['eval_split']!='test':raise ValueError('Validation data in final report')
        contracts.append(c['evaluation_contract'])
        seen=set()
        for r in csv_rows(out/'eval.csv'):
            key=(r['epoch'],r['cfg'],r['kind'])
            if key in seen:raise ValueError(f'Duplicate row: {row["name"]} {key}')
            seen.add(key)
            if int(r['seed'])!=row['seed'] or r['backend']!='inception-legacy' or r['eval_split']!='test':raise ValueError('Metric identity mismatch')
            if int(r['steps'])!=20 or int(r['num_eval'])!=10000 or float(r['temperature'])!=1.:raise ValueError('Mixed final evaluation contract')
            # Live outputs may include rows beyond latest committed epoch after an interruption.
            if not complete:
                if strict:raise ValueError('Incomplete final output')
                # Only complete training tasks enter public summary curves.
                continue
            raw.append(dict(name=row['name'],**r))
    write_csv(dest/'status.csv',status)
    if strict and not all(s['complete'] for s in status):raise ValueError('All 20 fresh-seed stages are required for strict compilation')
    if contracts and any(c!=contracts[0] for c in contracts):raise ValueError('Cross-arm evaluation contracts differ')
    if not raw:
        atomic(dest/'live_status.json',dict(complete_tasks=sum(s['complete'] for s in status),expected_tasks=20))
        print('No completed fresh-seed runs yet; wrote status.csv');return 0
    if strict:
        frozen=read(base/'selections/final.json')
        if frozen['fresh_seeds']!=list(FRESH):raise ValueError('Fresh-seed contract changed')
        for seed in FRESH:
            parent=configs[f'final_vae_s{seed}']
            a=configs[f'final_prior_dsm_s{seed}'];b=configs[f'final_prior_csem_s{seed}']
            parent_done=read(base/'results'/f'final_vae_s{seed}'/'complete.json')
            if parent_done.get('snapshot_sha256')!=a['source_sha256']:raise ValueError('Independent parent snapshot identity changed')
            if a['source_sha256']!=b['source_sha256'] or a['initial_head_sha256']!=b['initial_head_sha256']:raise ValueError('Broken independent target-swap pair')
            if a['beta0']!=b['beta0'] or a['lr_prior']!=b['lr_prior']:raise ValueError('Target-specific tuning in matched pair')
            if a['beta0']!=frozen['beta0'] or a['lr_prior']!=frozen['lr_prior']:raise ValueError('Frozen final settings changed')
    write_csv(dest/'per_seed_metrics.csv',raw);write_csv(dest/'timing.csv',timing)
    curves=[r for r in raw if r['kind']=='generation' and float(r['cfg'])==3.]
    simple=[dict(regime=r['regime'],seed=int(r['seed']),epoch=int(r['epoch']),fid=float(r['fid_generated'])) for r in curves]
    write_csv(dest/'website_training_curves.csv',simple)
    grouped={}
    for r in curves:
        k=(r['regime'],int(r['epoch']));grouped.setdefault(k,{})
        if int(r['seed']) in grouped[k]:raise ValueError('Duplicate training-seed replicate')
        grouped[k][int(r['seed'])]=r
    summary=[]
    for (regime,epoch),rs in sorted(grouped.items()):
        if strict and set(rs)!=set(FRESH):raise ValueError('Missing prescribed training seed')
        s=stats([float(r['fid_generated']) for r in rs.values()])
        summary.append(dict(regime=regime,epoch=epoch,expected_n=5,training_gpu_hours_mean=float(np.mean([float(r['end_to_end_training_gpu_hours']) for r in rs.values()])),**s))
    write_csv(dest/'curve_summary.csv',summary)
    endpoints=[]
    for regime in REGIMES:
        for cfg in (1.,3.):
            rs=[r for r in raw if r['regime']==regime and r['kind']=='generation' and int(r['epoch'])==500 and float(r['cfg'])==cfg]
            if not rs:continue
            for metric in ('fid_generated','fid_posterior_recon','kid','end_to_end_training_gpu_hours'):
                vs=[float(r[metric]) for r in rs if r[metric]!='']
                if vs:endpoints.append(dict(regime=regime,cfg=cfg,metric=metric,expected_n=5,**stats(vs)))
    write_csv(dest/'endpoint_summary.csv',endpoints)
    paired=[]
    for left,right in (('indep_csem','indep_tweedie'),('cotrained_csem','indep_tweedie'),('cotrained_csem','indep_csem')):
        l=grouped.get((left,500),{});r=grouped.get((right,500),{});seeds=sorted(l.keys()&r.keys())
        if strict and seeds!=list(FRESH):raise ValueError('Incomplete paired contrast')
        if seeds:
            delta=[float(l[s]['fid_generated'])-float(r[s]['fid_generated']) for s in seeds]
            paired.append(dict(contrast=f'{left} minus {right}',seeds=','.join(map(str,seeds)),expected_n=5,**stats(delta)))
    write_csv(dest/'paired_differences.csv',paired)
    common=compute_grid(grouped,strict)
    write_csv(dest/'common_compute_summary.csv',common)
    atomic(dest/'endpoint_summary.json',dict(endpoints=endpoints,paired_fid_differences=paired,
        training_seeds=list(FRESH),interval='pointwise 95% Student-t across independent training seeds',
        KID='raw polynomial MMD^2; 20 fixed subsets of 1000; subset SD in raw metrics is separate from seed SD',
        historical='Supplied plan reports seed42 FID4.3229685350 and observed-seed mean4.5597724371; excluded from confirmation'))
    plot(dest,summary,common,strict)
    # Representative images always use the prespecified seed47.
    for name in ('final_joint_s47','final_prior_dsm_s47','final_prior_csem_s47'):
        source=base/'results'/name/'samples/epoch_0500/cfg3'
        if source.exists():shutil.copytree(source,dest/'representative_seed47'/name,dirs_exist_ok=True)
    text=['# Fresh-seed Fashion-MNIST comparison','',
          'All arms train on the same fixed 50,000-image training subset. Development uses the held-out 10,000 training images; final reporting uses the standard 10,000 test images.',
          '','The independent pipelines receive 500 VAE epochs plus 500 fresh-prior epochs. Co-training receives 500 joint epochs. This is not equal compute.',
          '','CFG 3, Gaussian start at T=2, temperature 1, 20 RK4 integration steps; the raw table records actual score forward calls. Online/frozen VAE with EMA score head.',
          '','Posterior reconstruction FID is a reference, not a theoretical lower bound. Negative paired differences favor the first method.',
          '','Compute curves include the full VAE cost for each independent method. Common-budget summaries linearly interpolate only between observed checkpoints within every seed\'s common compute range.',
          '',f'Completed tasks: {sum(s["complete"] for s in status)}/20. Strict final figure: {strict}.',
          '','| Regime | CFG | Metric | n | Mean | SD | 95% CI |','|---|---:|---|---:|---:|---:|---|']
    for e in endpoints:
        fmt=lambda x:'NA' if x is None else f'{x:.4f}'
        text.append(f'| {LABELS[e["regime"]]} | {e["cfg"]:g} | {e["metric"]} | {e["n"]} | {fmt(e["mean"])} | {fmt(e["sd"])} | [{fmt(e["ci95_low"])}, {fmt(e["ci95_high"])}] |')
    (dest/'report.md').write_text('\n'.join(text)+'\n')
    if strict:atomic(dest/'complete.json',dict(strict=True,seeds=list(FRESH),training_tasks=20))
    print(f'Wrote {dest}');return 0

def compute_grid(grouped,strict):
    arrays={};expected=set(FRESH)
    for regime in REGIMES:
        seeds=sorted({s for (reg,e),rs in grouped.items() if reg==regime for s in rs})
        if strict and set(seeds)!=expected:raise ValueError('Compute comparison lacks all paired seeds')
        for seed in seeds:
            rr=[rs[seed] for (reg,e),rs in sorted(grouped.items()) if reg==regime and seed in rs]
            x=np.array([float(r['end_to_end_training_gpu_hours']) for r in rr]);y=np.array([float(r['fid_generated']) for r in rr])
            if len(x)<2:continue
            if np.any(np.diff(x)<=0):raise ValueError('Training compute must be strictly increasing')
            arrays[regime,seed]=(x,y)
    if not arrays or {k[0] for k in arrays}!=set(REGIMES):return []
    lo=max(x[0] for x,y in arrays.values());hi=min(x[-1] for x,y in arrays.values())
    if hi<=lo:return []
    records=[]
    for budget in np.linspace(lo,hi,20):
        for regime in REGIMES:
            values=[float(np.interp(budget,x,y)) for (r,s),(x,y) in arrays.items() if r==regime]
            records.append(dict(regime=regime,training_gpu_hours=float(budget),expected_n=5,interpolation='linear inside observed range',**stats(values)))
    return records

def plot(dest,summary,common,strict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained')
    for regime in REGIMES:
        rr=sorted([r for r in summary if r['regime']==regime],key=lambda r:r['epoch'])
        if not rr:continue
        counts=sorted({r['n'] for r in rr});label=LABELS[regime]+('' if strict else f' (n={min(counts)}–{max(counts)}/5)')
        for a,key in ((ax[0],'epoch'),):
            x=[r[key] for r in rr];y=[r['mean'] for r in rr]
            a.plot(x,y,label=label,color=COLORS[regime],lw=2)
            if all(r['ci95_low'] is not None for r in rr):a.fill_between(x,[r['ci95_low'] for r in rr],[r['ci95_high'] for r in rr],color=COLORS[regime],alpha=.16)
        compute=sorted([r for r in common if r['regime']==regime],key=lambda r:r['training_gpu_hours'])
        if compute:
            x=[r['training_gpu_hours'] for r in compute];y=[r['mean'] for r in compute]
            ax[1].plot(x,y,label=label,color=COLORS[regime],lw=2)
            if all(r['ci95_low'] is not None for r in compute):ax[1].fill_between(x,[r['ci95_low'] for r in compute],[r['ci95_high'] for r in compute],color=COLORS[regime],alpha=.16)
        else:
            ax[1].plot([r['training_gpu_hours_mean'] for r in rr],[r['mean'] for r in rr],label=label,color=COLORS[regime],lw=2)
    ax[0].set_xlabel('Joint / prior-training epoch');ax[1].set_xlabel('End-to-end training GPU-hours')
    for a in ax:a.set_ylabel('FID ↓');a.grid(alpha=.17)
    ax[0].set_title('500 prior epochs after 500 VAE epochs for two-stage');ax[1].set_title('Common measured range; includes VAE cost' if common else 'Includes VAE cost; no common compute range yet')
    ax[1].legend(frameon=False,fontsize=8)
    for ext in ('pdf','svg','png'):fig.savefig(dest/f'fmnist_comparison.{ext}',dpi=220)
    plt.close(fig)
