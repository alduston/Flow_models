#!/usr/bin/env python3
"""Final scheduled endpoints and paired seed/probe contrasts; keep partials separate."""
from pathlib import Path
import argparse,json,math
import numpy as np
import pandas as pd
from design import rows
from launcher_base import fid_column


def compile(base,strict=False):
    base=Path(base);out=base/'compiled';out.mkdir(exist_ok=True)
    inventory=[];evaluations=[];losses=[];audits=[];endpoints={};missing=[]
    for row in rows():
        root=base/'results'/row['result_name'];stage=row['stage'];target=int(row['epochs_joint'])
        ev=root/('eval_metrics.csv' if stage=='frozen' else 'dataframes/eval_metrics.csv')
        if not ev.exists():ev=ev.with_name('eval_metrics_in_progress.csv')
        df=pd.read_csv(ev) if ev.exists() else pd.DataFrame()
        if not df.empty:
            df=df[df['tag']=='LSI_Diff'].copy();df['cell_id']=int(row['cell_id']);evaluations.append(df)
        final=df[df['epoch']==target] if not df.empty else pd.DataFrame()
        complete=False;error=''
        if not final.empty:
            from paper import validate_evaluation
            try:validate_evaluation(row,df.to_dict('records'));complete=True
            except ValueError as exc:error=str(exc)
        if complete:endpoints[int(row['cell_id'])]=final.iloc[-1].to_dict()
        else:missing.append(int(row['cell_id']))
        item=dict(cell_id=int(row['cell_id']),name=row['result_name'],stage=stage,source=row['source'],seed=int(row['seed']),
                  blocks=int(row['blocks']),topology=row['topology'],r=float(row['rep_multiplier']),teacher=row['teacher'],
                  complete=complete,latest_epoch=float(df['epoch'].max()) if not df.empty else None,error=error)
        if complete:
            for cfg in (1.,2.5,3.):item[f'fid_cfg{cfg:g}']=float(endpoints[int(row['cell_id'])][fid_column(final,level=cfg)])
            item['fid_recon']=endpoints[int(row['cell_id'])].get('fid_vae_recon')
        inventory.append(item)
        lp=root/('loss_history.csv' if stage=='frozen' else 'dataframes/loss_history.csv')
        if lp.exists():
            v=pd.read_csv(lp);v['cell_id']=int(row['cell_id']);losses.append(v)
        for p in sorted((root/'audit').glob('epoch_*.csv')):
            v=pd.read_csv(p);v['cell_id']=int(row['cell_id']);audits.append(v)
    inv=pd.DataFrame(inventory);inv.to_csv(out/'summary.csv',index=False)
    for name,parts in [('all_eval',evaluations),('all_loss',losses),('all_diagnostics',audits)]:
        if parts:pd.concat(parts,ignore_index=True).to_csv(out/(name+'.csv'),index=False)
    contrasts=[]
    def contrast(a,b,label):
        if a not in endpoints or b not in endpoints:return
        for cfg in (1.,2.5,3.):
            aa=endpoints[a];bb=endpoints[b]
            contrasts.append(dict(comparison=label,seed=int(rows()[a]['seed']),a=a,b=b,cfg=cfg,
                                  delta_fid=float(aa[fid_column(aa,level=cfg)])-float(bb[fid_column(bb,level=cfg)])))
    for seed in range(2):
        for a,label in [(2,'r0-minus-flat'),(4,'r05-minus-flat'),(6,'r1-minus-flat'),(8,'r4-minus-r1'),(10,'decoupled-minus-r1')]:
            contrast(a+seed,(6 if a>=8 else 0)+seed,label)
        contrast(4+seed,6+seed,'r05-minus-r1')
    risks=[]
    for a in range(12,24,3):
        source=int(rows()[a]['source'])
        contrast(a,source,'naive-continuation-minus-source')
        for b in (a+1,a+2):
            contrast(b,a,rows()[b]['teacher']+'-minus-naive-frozen')
            p=base/'results'/rows()[a]['result_name']/'audit/epoch_0050_paired.npz'
            q=base/'results'/rows()[b]['result_name']/'audit/epoch_0050_paired.npz'
            if not p.exists() or not q.exists():continue
            with np.load(p) as left,np.load(q) as right:
                for key in left.files:
                    # Source risk on the same draws certifies pairing before a risk contrast.
                    if key.startswith('source|'):
                        if not np.array_equal(left[key],right[key]):raise ValueError('Frozen probe pairing failed: '+key)
                    if not key.startswith('ema|') or not key.endswith(('valid_risk','dsm_risk')):continue
                    d=right[key]-left[key];n=len(d);se=float(d.std(ddof=1)/math.sqrt(n))
                    risks.append(dict(source=source,seed=int(rows()[a]['seed']),teacher=rows()[b]['teacher'],probe=key,
                                      delta_risk=float(d.mean()),se=se,n=n,pointwise_lcb=float(d.mean()-1.96*se),pointwise_ucb=float(d.mean()+1.96*se)))
    pd.DataFrame(contrasts).to_csv(out/'paired_fid_contrasts.csv',index=False)
    pd.DataFrame(risks).to_csv(out/'paired_valid_risk_contrasts.csv',index=False)
    if contrasts:
        pd.DataFrame(contrasts).groupby(['comparison','cfg'])['delta_fid'].agg(['count','mean','std']).to_csv(out/'paired_fid_seed_aggregates.csv')
    lines=['# CIFAR stacked CSEM audit','',f'Complete final endpoints: {len(endpoints)}/24. Missing: {missing}.','',
           'Primary: epoch 800 (joint) or 50 (frozen), 10,000 images, full-T Gaussian start, RK4-20, CFG 2.5.',
           'CFG 1 and 3 and solver checks are separate columns. Negative delta favors the first arm.',
           'Paired image SE describes fixed-model probe uncertainty; it is not across-training uncertainty.',
           'Never equate lower naive loss, pathwise defect, or factorwise KL with full joint score accuracy.','',
           '## Endpoints','',inv.to_markdown(index=False) if __import__('importlib.util').util.find_spec('tabulate') else inv.to_csv(index=False)]
    if contrasts:lines+=['','## Paired FID contrasts','',pd.DataFrame(contrasts).to_csv(index=False)]
    if risks:lines+=['','## Frozen valid-risk contrasts','',pd.DataFrame(risks).to_csv(index=False)]
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    if evaluations:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5))
        for d in evaluations:
            cid=int(d['cell_id'].iloc[0]);row=rows()[cid]
            if row['stage']!='joint':continue
            col=fid_column(d,level=2.5);ax.plot(d['epoch'],d[col],label=row['result_name'])
        ax.set(xlabel='Joint epoch',ylabel='FID (10,000; CFG 2.5; RK4-20)');ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/'fid_trajectories.png',dpi=180);plt.close(fig)
    print(f'Compiled {len(endpoints)}/24 complete endpoints to {out}')
    if strict and missing:raise SystemExit('Missing/invalid final endpoints: '+str(missing))
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--strict',action='store_true');a=p.parse_args();raise SystemExit(compile(a.base_dir,a.strict))
