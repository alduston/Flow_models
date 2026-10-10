#!/usr/bin/env python3
"""One entry point: validate, import, prepare, submit, run, compile and plot."""
from __future__ import annotations
import os, sys, subprocess, traceback, time, hashlib, shutil, fcntl
from datetime import datetime, timezone
from pathlib import Path
from paper_io import normalize_metrics
DEFAULT_BASE=Path(__file__).resolve().parent

#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path
import pandas as pd

MANIFEST='manifest.csv'
RESULT_ROOT='results'
STATUS_ROOT='status'
OUT_ROOT='compiled'


def _read_manifest(base: Path):
    with (base/MANIFEST).open(newline='') as f: return list(csv.DictReader(f))

def _add_meta(df, row):
    if df is None or df.empty: return pd.DataFrame()
    out=df.copy()
    for k,v in row.items(): out[k]=v
    return out

def _last(df):
    if df is None or df.empty: return None
    return df.sort_values('epoch').iloc[-1]

def _find_metric(row, prefix, steps=None, init='initgaussianT', cfg=None):
    import re
    candidates=[]
    for c in row.index:
        if not c.startswith(prefix): continue
        if cfg is not None:
            m=re.search(r'_cfg([0-9]+(?:_[0-9]+)?)_',c)
            if not m or not math.isclose(float(m.group(1).replace('_','.')),float(cfg)): continue
        # T_K names share the string prefix: exclude them explicitly.
        if init and (init not in c or (init=='initgaussianT' and 'initgaussianTK' in c)): continue
        if steps is not None and f'_{int(steps)}_' not in c: continue
        try: value=float(row[c])
        except (TypeError, ValueError): continue
        if not math.isfinite(value): continue
        candidates.append((c,value))
    if not candidates: return (None,float('nan'))
    return sorted(candidates,key=lambda x:(len(x[0]),x[0]))[0]

def _main_regime(eval_df, mode):
    if eval_df.empty: return []
    if mode=='independent_pair':
        rr=[]
        for tag,reg in [('LSI_Diff_Refine','Independent CSEM'), ('Ctrl_Diff_Refine','Independent Tweedie')]:
            sub=eval_df[(eval_df['tag']==tag) & (eval_df['stage']=='refine')].copy()
            if not sub.empty:
                sub['regime']=reg; rr.append(sub)
        return rr
    if mode in {'cotrained_csem','fmnist_aug18_recovered','fmnist_aug13_gaussian','cotrained_csem_norm','raw_terminal_control'}:
        sub=eval_df[eval_df.get('tag','').astype(str).str.fullmatch('LSI_Diff',na=False)].copy()
        if sub.empty:
            sub=eval_df[eval_df.get('tag','').astype(str).str.contains('LSI_Diff',na=False)].copy()
        if not sub.empty: sub['regime']='Co-trained CSEM'
        return [sub] if not sub.empty else []
    return []

def compile_results(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=Path.cwd())
    ap.add_argument('--cifar-treatment',choices=['optimized','raw'],default='optimized')
    ap.add_argument('--cfg-policy', choices=['best-final','recipe','old-paper'], default='best-final')
    a=ap.parse_args(argv); base=a.base_dir.resolve(); out=base/OUT_ROOT; out.mkdir(parents=True,exist_ok=True)
    rows=_read_manifest(base)
    all_loss=[]; all_eval=[]; statuses=[]; completed=[]
    for row in rows:
        cid=int(row['cell_id']); rdir=base/RESULT_ROOT/row['result_name']
        st=read_status(base, row)
        if st.get('returncode') == 0 and st.get('config_hash') != config_hash(row):
            raise ValueError(f"Cell {cid}: completed status has a different manifest configuration")
        row=dict(row, run_complete=st.get('returncode') == 0)
        statuses.append({**row,'returncode':st.get('returncode'),'elapsed_seconds':st.get('elapsed_seconds'),'error':st.get('error','')})
        lp=rdir/'dataframes'/'loss_history.csv'; ep=rdir/'dataframes'/'eval_metrics.csv'
        if row['run_complete'] and (not lp.is_file() or not ep.is_file()):
            raise FileNotFoundError(f'Cell {cid}: completed status is missing final CSV files')
        if not row['run_complete']:
            choices=[p for p in [lp,lp.with_name('loss_history_in_progress.csv')] if p.is_file()]
            if choices: lp=max(choices,key=lambda p:p.stat().st_mtime_ns)
        if not row['run_complete'] or not ep.is_file():
            progress=rdir/'dataframes'/'eval_metrics_in_progress.csv'
            if progress.is_file(): ep=progress
        ldf=read_frame(lp); edf=read_frame(ep)
        if not edf.empty and row['mode']=='fmnist_aug13_gaussian':
            edf,_=normalize_metrics(edf)
        if not ldf.empty: all_loss.append(_add_meta(ldf,row))
        if not edf.empty: all_eval.append(_add_meta(edf,row))
        if not ldf.empty and not edf.empty: completed.append((row,ldf,edf))
    loss=pd.concat(all_loss,ignore_index=True,sort=False) if all_loss else pd.DataFrame()
    ev=pd.concat(all_eval,ignore_index=True,sort=False) if all_eval else pd.DataFrame()
    loss.to_csv(out/'all_loss_history.csv',index=False)
    ev.to_csv(out/'all_eval_records.csv',index=False)
    pd.DataFrame(statuses).to_csv(out/'run_status.csv',index=False)

    def is_main(row):
        if row['dataset']=='FMNIST':return row['family']=='main'
        if row['mode']=='independent_pair':return row['family']=='main'
        if a.cifar_treatment=='optimized':return row['family']=='main' and row['mode']=='cotrained_csem'
        return row['mode']=='raw_terminal_control' and row['family']=='scale_anchor'
    # Select CFG on co-trained endpoints, then freeze it across epochs and priors.
    # All candidates must be Gaussian-start at the EXACT paper RK4-20 budget.
    import re
    selected={}; candidates=[]
    for row,ldf,edf in completed:
        if not row['run_complete'] or not is_main(row) or row['mode']=='independent_pair': continue
        last=_last(edf)
        if last is None: continue
        for c in last.index:
            if not c.startswith('fid_rk4_20_') or 'initgaussianTK' in c or 'initgaussianT' not in c: continue
            m=re.search(r'_cfg([0-9]+(?:_[0-9]+)?)_',c)
            if m and pd.notna(last[c]) and math.isfinite(float(last[c])): candidates.append(dict(dataset=row['dataset'],cfg=float(m.group(1).replace('_','.')),fid=float(last[c])))
    for dataset in ['FMNIST','CIFAR']:
        default=3.0 if dataset=='FMNIST' else 2.5
        if a.cfg_policy=='old-paper': selected[dataset]=1.5 if dataset=='FMNIST' else 3.0
        elif a.cfg_policy=='recipe': selected[dataset]=default
        else:
            cs=pd.DataFrame([r for r in candidates if r['dataset']==dataset])
            selected[dataset]=float(cs.groupby('cfg').fid.mean().idxmin()) if not cs.empty else default
    (out/'selected_cfg.json').write_text(json.dumps(dict(policy=a.cfg_policy,cifar_treatment=a.cifar_treatment,selected=selected,selection_status={d: ('final-grid-selection' if sum(r['run_complete'] and is_main(r) and r['mode']!='independent_pair' and r['dataset']==d for r,_,_ in completed)==2 else 'provisional-or-recipe-default') for d in selected},selection='mean final co-trained FID, RK4-20, Gaussian start; candidate grid only'),indent=2)+'\n')
    # Main comparison long curves + endpoint table.
    curve=[]; endpoints=[]
    for row,ldf,edf in completed:
        if not is_main(row): continue
        for sub in _main_regime(edf,row['mode']):
            steps=20; cfg_value=selected[row['dataset']]
            paper_fid=[]; paper_kid=[]; paper_sw2=[]
            for _, rr in sub.iterrows():
                paper_fid.append(_find_metric(rr,'fid_rk4_',steps,cfg=cfg_value)[1])
                paper_kid.append(_find_metric(rr,'kid_rk4_',steps,cfg=cfg_value)[1])
                paper_sw2.append(_find_metric(rr,'feature_sw2_rk4_',steps,cfg=cfg_value)[1])
            sub=_add_meta(sub,row)
            sub['paper_fid']=paper_fid; sub['paper_kid']=paper_kid; sub['paper_sw2']=paper_sw2
            sub['paper_latent_sw2']=[_find_metric(rr,'sw2_rk4_',steps,cfg=cfg_value)[1] for _,rr in sub.iterrows()]
            sub['paper_csem_gap']=pd.to_numeric(sub.get('csem_gap_score_uncond',float('nan')),errors='coerce')
            sub['paper_cfg']=cfg_value; sub['paper_steps']=steps
            curve.append(sub)
            last=_last(sub)
            if last is None or not row['run_complete']: continue
            steps=20; cfg_value=selected[row['dataset']]
            fid_col,fid=_find_metric(last,'fid_rk4_',steps,cfg=cfg_value)
            kid_col,kid=_find_metric(last,'kid_rk4_',steps,cfg=cfg_value)
            sw_col,sw=_find_metric(last,'feature_sw2_rk4_',steps,cfg=cfg_value)
            gap=float(last.get('csem_gap_score_uncond',float('nan')))
            endpoints.append({
                'dataset':row['dataset'],'seed':int(row['seed']),'regime':last['regime'],
                'epoch':float(last['epoch']),'eval_samples':int(row['eval_samples']),'sampler_steps':steps,'cfg_scale':cfg_value,
                'fid':fid,'kid':kid,'sw2':sw,'csem_gap':gap,
                'fid_column':fid_col,'kid_column':kid_col,'sw2_column':sw_col,
                'recon_fid':float(last.get('fid_vae_recon',float('nan'))),
                'recon_kid':float(last.get('kid_vae_recon',float('nan'))),
                'recon_sw2':float(last.get('feature_sw2_vae_recon',float('nan'))),
            })
    curves=pd.concat(curve,ignore_index=True,sort=False) if curve else pd.DataFrame()
    curves.to_csv(out/'main_epoch_curves_long.csv',index=False)
    enddf=pd.DataFrame(endpoints)
    enddf.to_csv(out/'main_final_metrics_by_seed.csv',index=False)
    agg=pd.DataFrame()
    if not enddf.empty:
        agg=enddf.groupby(['dataset','regime'],as_index=False).agg(
            n=('seed','count'),fid_mean=('fid','mean'),fid_sd=('fid','std'),
            kid_mean=('kid','mean'),kid_sd=('kid','std'),sw2_mean=('sw2','mean'),sw2_sd=('sw2','std'),
            csem_gap_mean=('csem_gap','mean'),recon_fid_mean=('recon_fid','mean'),recon_kid_mean=('recon_kid','mean'),recon_sw2_mean=('recon_sw2','mean'))
    agg.to_csv(out/'main_final_metrics_aggregate.csv',index=False)

    # Scale anchor table: terminal arm is main CIFAR CSEM seed 42; other rows are dedicated cells.
    scale_rows=[]
    for row,ldf,edf in completed:
        if not row['run_complete']: continue
        if row['dataset']!='CIFAR' or int(row['seed'])!=42: continue
        if row['mode']=='cotrained_csem' and row['family']=='main': continue  # OU-partial is a separate architectural intervention
        elif row['mode']=='raw_terminal_control' and row['family']=='scale_anchor': label='Raw mean + terminal K_TK'
        elif row['mode']=='anchor_norm': label='Historical hard GN0'
        elif row['mode']=='anchor_none': label='No anchor'
        else: continue
        le=_last(edf); ll=_last(ldf[ldf.get('stage','')=='cotrain'] if 'stage' in ldf.columns else ldf)
        if le is None or ll is None: continue
        _,fid=_find_metric(le,'fid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
        _,kid=_find_metric(le,'kid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
        scale_rows.append({'scale_treatment':label,'eval_samples':int(row['eval_samples']),'cell_id':int(row['cell_id']),'recon_loss':float(ll.get('recon',float('nan'))),
                           'recon_fid':float(le.get('fid_vae_recon',float('nan'))),'csem_loss':float(ll.get('score_mse_weighted',ll.get('score_lsi',float('nan')))),
                           'terminal_kl':float(ll.get('terminal_kl',float('nan'))),'latent_rms':float(ll.get('latent_rms',float('nan'))),
                           'posterior_var':float(ll.get('posterior_var',float('nan'))),'fid':fid,'kid':kid})
    pd.DataFrame(scale_rows).to_csv(out/'scale_anchor_final.csv',index=False)

    # Compact T and lambda_K sensitivity; include certified center point from main run.
    sens=[]
    for row,ldf,edf in completed:
        if not row['run_complete']: continue
        use=(row['family']=='horizon_anchor') or (row['family']=='main' and row['mode']=='cotrained_csem' and row['dataset']=='CIFAR' and int(row['seed'])==42)
        if not use: continue
        le=_last(edf); ll=_last(ldf[ldf.get('stage','')=='cotrain'] if 'stage' in ldf.columns else ldf)
        if le is None or ll is None: continue
        _,fid=_find_metric(le,'fid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
        _,kid=_find_metric(le,'kid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
        sens.append({'eval_samples':int(row['eval_samples']),'cell_id':int(row['cell_id']),'T_K':float(row['T_K']),'T':float(row['T_full']),'lambda_K':float(row['terminal_kl_w']),
                     'csem_w':float(row['csem_w']),'recon_loss':float(ll.get('recon',float('nan'))),'csem_loss':float(ll.get('score_mse_weighted',float('nan'))),'recon_fid':float(le.get('fid_vae_recon',float('nan'))),
                     'csem_gap':float(le.get('csem_gap_score_uncond',float('nan'))),'terminal_kl':float(ll.get('terminal_kl',float('nan'))),
                     'latent_rms':float(ll.get('latent_rms',float('nan'))),'posterior_var':float(ll.get('posterior_var',float('nan'))),'fid':fid,'kid':kid})
    sensdf=pd.DataFrame(sens)
    if not sensdf.empty: sensdf=sensdf.sort_values(['lambda_K','T'])
    sensdf.to_csv(out/'terminal_horizon_anchor_sensitivity.csv',index=False)

    raw=[]
    for row,ldf,edf in completed:
        if not row['run_complete']: continue
        if row['mode']!='raw_terminal_control' or int(row['seed'])!=42: continue
        le=_last(edf); ll=_last(ldf[ldf.stage=='cotrain'])
        if le is None or ll is None: continue
        raw.append(dict(eval_samples=int(row['eval_samples']),cell_id=int(row['cell_id']),T_K=float(row['T_K']),T=float(row['T_full']),lambda_K=float(row['terminal_kl_w']),
            fid=_find_metric(le,'fid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))[1],
            kid=_find_metric(le,'kid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))[1],
            latent_rms=ll.get('latent_rms'),terminal_kl=ll.get('terminal_kl'),recon_loss=ll.get('recon'),
            csem_loss=ll.get('score_mse_weighted'),csem_gap=le.get('csem_gap_score_uncond')))
    pd.DataFrame(raw).to_csv(out/'raw_anchor_sensitivity.csv',index=False)

    # Naive Tweedie collapse tables and solver-order extraction.
    collapse=[]; solver=[]
    for row,ldf,edf in completed:
        if not row['run_complete']: continue
        if row['mode'] not in {'naive_tweedie_cotrain','cotrained_csem','fmnist_aug13_gaussian','raw_terminal_control'}: continue
        if row['mode']=='cotrained_csem' and not (row['dataset']=='CIFAR' and int(row['seed'])==42 and row['family']=='main'): continue
        if row['mode']=='raw_terminal_control' and row['family']!='scale_anchor':continue
        le=_last(edf)
        if le is None: continue
        if row['mode']=='naive_tweedie_cotrain':
            ll=_last(ldf[ldf.get('stage','')=='cotrain'] if 'stage' in ldf.columns else ldf)
            _,fid=_find_metric(le,'fid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
            _,kid=_find_metric(le,'kid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
            collapse.append({'dataset':row['dataset'],'seed':int(row['seed']),'epoch':float(le['epoch']),'fid':fid,'kid':kid,
                             'latent_rms':float(ll.get('latent_rms',float('nan'))) if ll is not None else float('nan'),
                             'posterior_std':float(ll.get('posterior_std',float('nan'))) if ll is not None else float('nan'),
                             'posterior_var':float(ll.get('posterior_var',float('nan'))) if ll is not None else float('nan')})
        hcol,hfid=_find_metric(le,'fid_heun_',int(row.get('paper_solver_steps',20)),cfg=float(row['cfg_scale']))
        rcol,rfid=_find_metric(le,'fid_rk4_',int(row.get('paper_solver_steps',20)),cfg=float(row['cfg_scale']))
        hkcol,hkid=_find_metric(le,'kid_heun_',int(row.get('paper_solver_steps',20)),cfg=float(row['cfg_scale']))
        rkcol,rkid=_find_metric(le,'kid_rk4_',int(row.get('paper_solver_steps',20)),cfg=float(row['cfg_scale']))
        solver.append({'mode':row['mode'],'dataset':row['dataset'],'seed':int(row['seed']),'eval_samples':int(row['eval_samples']),'steps':int(row.get('paper_solver_steps',20)),
                       'heun_fid':hfid,'heun_kid':hkid,'rk4_fid':rfid,'rk4_kid':rkid,
                       'heun_fid_column':hcol,'rk4_fid_column':rcol})
    pd.DataFrame(collapse).to_csv(out/'naive_tweedie_collapse_final.csv',index=False)
    pd.DataFrame(solver).to_csv(out/'solver_order_comparison.csv',index=False)
    print(f'Compiled outputs -> {out}')
    return 0


#!/usr/bin/env python3
"""Publication PNG/PDF figures for every experimental placeholder."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError
OUT_ROOT='compiled';FIG_ROOT='figures'
STYLES={'Co-trained CSEM':('#1f77b4','-','o'),'Independent CSEM':('#1f77b4','--','o'),'Independent Tweedie':('#d62728','--','s')}
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})

def read(p):
    if not p.is_file(): return pd.DataFrame()
    try:return pd.read_csv(p,float_precision='round_trip')
    except EmptyDataError:return pd.DataFrame()

def save(fig,p):
    fig.tight_layout();fig.savefig(p.with_suffix('.png'),dpi=240,bbox_inches='tight');fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def lines(ax,d,x,y,group,ylabel,main=False):
    ax.set_xlabel('LDM training epoch' if main else x.replace('_',' '));ax.set_ylabel(ylabel);ax.grid(alpha=.22)
    if d.empty or not {x,y,group}.issubset(d.columns):
        ax.text(.5,.5,'Awaiting results',ha='center',transform=ax.transAxes)
        return False
    drawn=False
    for name,g in d.groupby(group,sort=False):
        q=g[[x,y]].copy();q[x]=pd.to_numeric(q[x],errors='coerce');q[y]=pd.to_numeric(q[y],errors='coerce');q=q.dropna()
        if q.empty:continue
        stats=q.groupby(x)[y].agg(['mean','std','count']).sort_index();color,style,marker=STYLES.get(name,(None,'-','o'))
        ax.plot(stats.index,stats['mean'],label=name,color=color,linestyle=style,marker=marker if main else None,markersize=4)
        band=stats[stats['count']>1]
        if len(band):ax.fill_between(band.index,band['mean']-band['std'],band['mean']+band['std'],color=color,alpha=.12)
        drawn=True
    ax.set_xlabel('LDM training epoch' if main else x.replace('_',' '));ax.set_ylabel(ylabel);ax.grid(alpha=.22)
    if drawn:ax.legend(fontsize=8)
    else:ax.text(.5,.5,'Awaiting finite results',ha='center',transform=ax.transAxes)
    return drawn

def plot_results(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--base-dir',type=Path,default=Path.cwd());a=ap.parse_args(argv);base=a.base_dir.resolve();comp=base/OUT_ROOT;figs=base/FIG_ROOT;figs.mkdir(parents=True,exist_ok=True)
    d=read(comp/'main_epoch_curves_long.csv');l=read(comp/'all_loss_history.csv');coverage={};missing=[]
    for dataset in ['FMNIST','CIFAR']:
        q=d[d.dataset==dataset].copy() if 'dataset' in d else pd.DataFrame()
        found=set(q.loc[pd.to_numeric(q.paper_fid,errors='coerce').notna(),'regime']) if {'regime','paper_fid'}.issubset(q) else set()
        missing.extend(f'{dataset}: {r}' for r in STYLES if r not in found)
        for metric,label,tag in [('paper_fid','FID ↓','fid'),('paper_kid','KID ↓','kid'),('paper_sw2','Feature SW2 ↓','sw2'),('paper_latent_sw2','Latent SW2² ↓','latent_sw2'),('paper_csem_gap','CSEM gap (score units) ↓','csem_gap')]:
            fig,ax=plt.subplots(figsize=(6.3,4.3));ok=lines(ax,q,'epoch',metric,'regime',label,True)
            cfg=q.paper_cfg.iloc[0] if 'paper_cfg' in q and len(q) else '?'
            ax.set_title(f'{dataset}: RK4-20, CFG={cfg}');stem=f'main_{dataset.lower()}_{tag}_vs_epoch';save(fig,figs/stem);coverage[stem]=ok
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,dataset,metric,label in [(axs[0],'FMNIST','paper_fid','FID ↓'),(axs[1],'CIFAR','paper_kid','KID ↓')]:
        q=d[d.dataset==dataset] if 'dataset' in d else pd.DataFrame();lines(ax,q,'epoch',metric,'regime',label,True)
        cfg=q.paper_cfg.iloc[0] if 'paper_cfg' in q and len(q) else '?';ax.set_title(f'{dataset}: RK4-20, CFG={cfg}')
    save(fig,figs/'main_old_paper_epoch_comparison')
    # Only the controlled raw/GN/no-anchor treatments enter the central panel.
    if 'mode' in l and 'stage' in l:
        s=l[(l.seed.astype(str)=='42')&(l.family=='scale_anchor')&(l.stage=='cotrain')&l['mode'].isin(['raw_terminal_control','anchor_norm','anchor_none'])].copy()
        s['treatment']=s['mode'].map({'raw_terminal_control':'Raw mean + terminal KL','anchor_norm':'Per-sample mean normalization','anchor_none':'No anchor'})
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for ax,m,label in zip(axs.flat,['recon','score_mse_weighted','terminal_kl','latent_rms'],['Reconstruction loss','CSEM training loss','Terminal K_TK','Mean latent RMS']):
            lines(ax,s,'epoch',m,'treatment',label);ax.set_title(label)
            f,aa=plt.subplots(figsize=(6.3,4.3));coverage['scale_'+m]=lines(aa,s,'epoch',m,'treatment',label);save(f,figs/('scale_anchor_'+m))
        save(fig,figs/'scale_anchor_dynamics')
        n=l[(l['mode']=='naive_tweedie_cotrain')&(l.stage=='cotrain')]
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,m,label in zip(axs,['latent_rms','posterior_var'],['Posterior mean RMS','Mean posterior variance']):lines(ax,n,'epoch',m,'result_name',label)
        save(fig,figs/'tweedie_collapse_mean_variance')
        for m in ['posterior_var','posterior_std','latent_rms']:
            f,ax=plt.subplots(figsize=(6.3,4.3));coverage['collapse_'+m]=lines(ax,n,'epoch',m,'result_name',m.replace('_',' '));save(f,figs/('tweedie_collapse_'+m))
        c=l[(l['mode']=='cotrained_csem')&(l.dataset=='CIFAR')&(l.stage=='cotrain')]
        for m in ['lr_vae_current','lr_score_current','recon','score_mse_weighted']:
            f,ax=plt.subplots(figsize=(6.3,4.3));lines(ax,c,'epoch',m,'result_name',m.replace('_',' '));save(f,figs/('cifar_training_'+m))
    # Sampling horizon T and anchor weight, plus raw-mean anchor-horizon controls.
    for filename,axis,center,center_val,stem in [('terminal_horizon_anchor_sensitivity.csv','T','lambda_K',.6,'sensitivity_T'),('terminal_horizon_anchor_sensitivity.csv','lambda_K','T',1.35,'sensitivity_lambdaK'),('raw_anchor_sensitivity.csv','T_K','lambda_K',.6,'raw_sensitivity_TK'),('raw_anchor_sensitivity.csv','lambda_K','T_K',1.05,'raw_sensitivity_lambdaK')]:
        s=read(comp/filename)
        if {axis,center}.issubset(s):s=s[np.isclose(s[center],center_val)].copy();s['series']=s['eval_samples'].map(lambda n:f'Controls, N={int(n)}') if 'eval_samples' in s else 'Controls'
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for ax,m,label in zip(axs.flat,['fid','kid','latent_rms','terminal_kl','recon_loss','csem_gap'],['FID ↓','KID ↓','Latent RMS','Terminal K_TK','Reconstruction loss','CSEM gap (score units) ↓']):
            lines(ax,s,axis,m,'series',label);ax.set_title(label)
        save(fig,figs/stem);coverage[stem]=not s.empty
    s=read(comp/'solver_order_comparison.csv')
    for _,r in s.iterrows():
        fig,axs=plt.subplots(1,2,figsize=(9,4))
        for ax,m,label in zip(axs,['fid','kid'],['FID ↓','KID ↓']):ax.bar(['Heun SDE','RK4 ODE'],[r.get('heun_'+m,np.nan),r.get('rk4_'+m,np.nan)]);ax.set_ylabel(label)
        fig.suptitle(f"{r['dataset']}: {r['mode']}, {int(r['steps'])} steps (different NFEs)")
        save(fig,figs/f"solver_{r['mode']}_{r['dataset'].lower()}_s{int(r['seed'])}")
    expected=[]
    for row in _read_manifest(base):
        if row['family'] != 'main': continue
        regimes=['Independent CSEM','Independent Tweedie'] if row['mode']=='independent_pair' else ['Co-trained CSEM']
        for regime in regimes:
            qq=d[(d.dataset==row['dataset']) & (pd.to_numeric(d.seed)==int(row['seed'])) & (d.regime==regime)] if {'dataset','seed','regime'}.issubset(d) else pd.DataFrame()
            fields={'dataset':row['dataset'],'seed':int(row['seed']),'regime':regime,'cell_id':int(row['cell_id']),
                    'complete':read_status(base,row).get('returncode') == 0,'finite_fid_epochs':int(qq.paper_fid.notna().sum()) if 'paper_fid' in qq else 0}
            expected.append(fields)
    pd.DataFrame(expected).to_csv(figs/'main_curve_coverage.csv',index=False)
    (figs/'figure_coverage.json').write_text(json.dumps(dict(figures_with_data=coverage,missing_regimes=missing,main_regime_seed_coverage=expected,note='Missing curves are never fabricated; bands are across-seed SD where n>1.'),indent=2)+'\n')
    print(f'Figures -> {figs}; missing main regimes: {missing}')
    return 0



def read_frame(path):
    return read(Path(path))


def read_status(base, row):
    path=base/STATUS_ROOT/f"cell_{int(row['cell_id']):03d}.json"
    if not path.is_file(): return {}
    return json.loads(path.read_text())


def config_hash(row):
    scientific={k:v for k,v in row.items() if k not in {'notes','result_name','source_v5_result_name','outputs','run_complete'}}
    return hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');os.replace(temporary,path)


def validate_evaluation(row, frame, full=True):
    if frame.empty or not {'epoch','stage','tag'}.issubset(frame):
        raise ValueError('Missing evaluation records or stage/tag/epoch metadata')
    if any(any(ord(c)<32 for c in str(name)) for name in frame.columns) or frame.columns.duplicated().any():
        raise ValueError('Invalid metric headers')
    independent=row['mode']=='independent_pair'
    expected_epoch=int(row['epochs_refine'] if independent else row['epochs_joint'])
    tags=['LSI_Diff_Refine','Ctrl_Diff_Refine'] if independent else (['Ctrl_Diff'] if row['mode']=='naive_tweedie_cotrain' else ['LSI_Diff'])
    grid=[float(row['cfg_scale'])]
    main=row['family']=='main'
    if main: grid.append(1.5 if row['dataset']=='FMNIST' else 3.)
    for tag in tags:
        sub=frame[(frame.tag==tag)&(frame.stage==('refine' if independent else 'cotrain'))]
        if sub.empty: raise ValueError(f'Missing trained head {tag}')
        epochs=set(pd.to_numeric(sub.epoch).astype(int))
        expected=set(range(int(row['eval_every']),expected_epoch+1,int(row['eval_every']))) | {expected_epoch}
        if full and main and not expected.issubset(epochs):
            raise ValueError(f'{tag}: missing evaluated epochs {sorted(expected-epochs)}')
        last=_last(sub)
        if full and int(last.epoch)!=expected_epoch:
            raise ValueError(f'{tag}: endpoint {last.epoch} is short of {expected_epoch}')
        for g in grid:
            for prefix in ['fid_rk4_','kid_rk4_','feature_sw2_rk4_'] if main else ['fid_rk4_']:
                value=_find_metric(last,prefix,20 if main else int(row['rk4_steps']),cfg=g)[1]
                if not math.isfinite(value): raise ValueError(f'{tag}: missing finite {prefix} at CFG={g}')
        if main and not math.isfinite(float(last.get('csem_gap_score_uncond',float('nan')))):
            raise ValueError(f'{tag}: missing score-unit CSEM gap')


def parse_cells(spec, rows):
    valid={int(r['cell_id']) for r in rows}
    if spec in {'all','missing'}: return sorted(valid)
    if spec in {r['family'] for r in rows}: return sorted(int(r['cell_id']) for r in rows if r['family']==spec)
    if spec=='independent': return sorted(int(r['cell_id']) for r in rows if r['mode']=='independent_pair')
    out=set()
    for part in spec.split(','):
        if '-' in part:
            a,b=map(int,part.split('-',1));out.update(range(min(a,b),max(a,b)+1))
        else: out.add(int(part))
    if out-valid: raise ValueError(f'Unknown cells {sorted(out-valid)}')
    return sorted(out)


def run_job(base, cid, resume=False, restart=False):
    from suite import run_cell
    row=next((r for r in _read_manifest(base) if int(r['cell_id'])==cid),None)
    if row is None: raise ValueError(f'Unknown cell {cid}')
    status_path=base/STATUS_ROOT/f'cell_{cid:03d}.json'
    status_path.parent.mkdir(parents=True,exist_ok=True)
    with (status_path.parent/f'cell_{cid:03d}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        st=read_status(base,row)
        destination=base/RESULT_ROOT/row['result_name']
        if st.get('returncode')==0:
            if st.get('config_hash')!=config_hash(row): raise ValueError('Completed run has a different manifest configuration')
            validate_evaluation(row,read_frame(destination/'dataframes'/'eval_metrics.csv'))
            print(f'[skip] completed cell {cid}: {row["outputs"]}');return 0
        checkpoint=destination/'checkpoints'/'training_state_latest.pt'
        losses_exist=any((destination/'dataframes'/name).is_file() for name in ['loss_history.csv','loss_history_in_progress.csv'])
        if destination.exists() and (restart or (not checkpoint.is_file() and not losses_exist)):
            archive=base/'failed_attempts'/(row['result_name']+'_'+str(time.time_ns()))
            archive.parent.mkdir(exist_ok=True);os.replace(destination,archive)
            print(f'[restart] archived incomplete attempt: {archive}')
        payload=dict(row,started_utc=datetime.now(timezone.utc).isoformat(),returncode=None,
                     config_hash=config_hash(row),slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        write_json(status_path,payload);started=time.time()
        try:
            print(f'Cell {cid}: {row["dataset"]}, {row["outputs"]}, seed={row["seed"]}')
            loss,evaluation,cfg=run_cell(row,base,resume=resume)
            validate_evaluation(row,evaluation)
            payload.update(returncode=0,loss_rows=len(loss),eval_rows=len(evaluation),resolved_config=cfg)
            rc=0
        except Exception as exc:
            payload.update(returncode=1,error=repr(exc),traceback=traceback.format_exc())
            print(payload['traceback'],file=sys.stderr);rc=1
        payload.update(elapsed_seconds=time.time()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path,payload)
        return rc


def prepare_data(base, datasets):
    import core
    from paper_io import prepare_dataset
    for name in dict.fromkeys(datasets):
        info=core.DATASET_INFO[name]
        path=prepare_dataset(info['class'],base/'data_cache_v6',name,
                             kwargs={'split':info['split']} if name=='EMNIST' else {})
        print(f'Prepared and validated {name}: {path}')
    return 0


def parse_sbatch_job_id(output):
    """Vista adds a site banner even with --parsable; accept only ID lines."""
    import re
    matches=[]
    for line in output.splitlines():
        match=re.fullmatch(r'\s*(?:Submitted batch job\s+)?([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*',line)
        if match: matches.append(match.group(1))
    ids=set(matches)
    if len(ids)!=1:
        raise RuntimeError(f'Cannot uniquely parse submitted Slurm job ID: {output!r}')
    return matches[-1]


def submit_jobs(base, spec, dry_run=False, restart=False, prepare_job_id=None):
    rows=_read_manifest(base);by={int(r['cell_id']):r for r in rows}
    selected=parse_cells(spec,rows)
    # Every command skips completed validated cells; missing is the safe default.
    pending=[]
    active_jobs=None
    for cid in selected:
        row=by[cid];st=read_status(base,row)
        if st.get('returncode')==0:
            if st.get('config_hash')!=config_hash(row): raise ValueError(f'Configuration changed for completed cell {cid}')
            validate_evaluation(row,read_frame(base/RESULT_ROOT/row['result_name']/'dataframes'/'eval_metrics.csv'))
            continue
        if st.get('returncode') is None and st.get('slurm_job_id'):
            if active_jobs is None:
                import getpass
                q=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i'],capture_output=True,text=True)
                if q.returncode: raise RuntimeError('Could not check existing Slurm jobs')
                active_jobs=set(q.stdout.split())
            if str(st['slurm_job_id']) in active_jobs:
                print(f'[skip] active cell {cid}, Slurm job {st["slurm_job_id"]}');continue
        pending.append(cid)
    print(f'Selected {len(pending)} unfinished jobs: {pending}')
    if not pending:return 0
    (base/'slurm_logs').mkdir(exist_ok=True)
    datasets=','.join(dict.fromkeys(by[c]['dataset'] for c in pending))
    prep=['sbatch','--parsable','--job-name=csem_prepare','--time=02:00:00',
          f'--export=ALL,TASK=prepare,DATASETS={datasets.replace(",",":")},BASE_DIR={base}',str(base/'job.slurm')]
    prep_id='PREP_JOB_ID'
    if prepare_job_id is not None:
        prep_id=parse_sbatch_job_id(str(prepare_job_id))
        print(f'[reuse] preparation job {prep_id}; training depends on its successful completion')
    else:
        print(' '.join(prep))
    if not dry_run and prepare_job_id is None:
        q=subprocess.run(prep,cwd=base,text=True,capture_output=True,check=True)
        receipt=dict(stdout=q.stdout,stderr=q.stderr,submitted_utc=datetime.now(timezone.utc).isoformat(),datasets=datasets)
        write_json(base/STATUS_ROOT/'preparation_submission.json',receipt)
        prep_id=parse_sbatch_job_id(q.stdout)
        write_json(base/STATUS_ROOT/'preparation_submission.json',dict(receipt,slurm_job_id=prep_id))
    for cid in pending:
        row=by[cid]
        cmd=['sbatch','--parsable',f'--dependency=afterok:{prep_id}',
             f'--export=ALL,TASK=run,CELL_ID={cid},BASE_DIR={base},RESTART={int(restart)}',str(base/'job.slurm')]
        print(' '.join(cmd),f'# {row["dataset"]}: {row["outputs"]}, seed={row["seed"]}')
        if not dry_run:
            q=subprocess.run(cmd,cwd=base,text=True,capture_output=True,check=True)
            jobid=parse_sbatch_job_id(q.stdout)
            current=read_status(base,row)
            if not (str(current.get('slurm_job_id'))==jobid and current.get('started_utc')):
                write_json(base/STATUS_ROOT/f'cell_{cid:03d}.json',dict(row,returncode=None,slurm_job_id=jobid,prepare_job_id=prep_id,config_hash=config_hash(row),queued_utc=datetime.now(timezone.utc).isoformat()))
    return 0


def import_v5(base, source):
    """Import completed compatible metrics; leave all original results untouched."""
    source=source.resolve()
    if source==base: raise ValueError('Import source and destination must differ')
    combined=source/'csem_paper_compiled_v1'
    if not combined.is_dir():combined=source
    statuses=read_frame(combined/'run_status.csv')
    evals=read_frame(combined/'all_eval_records.csv')
    losses=read_frame(combined/'all_loss_history.csv')
    imported=[];rejected=[];header_recovery={}
    for row in _read_manifest(base):
        cid=int(row['cell_id']);oldname=row['source_v5_result_name']
        st={}
        status_file=source/'csem_paper_status_v1'/f'cell_{cid:03d}_{oldname}.json'
        if status_file.is_file():st=json.loads(status_file.read_text())
        elif 'cell_id' in statuses:
            ss=statuses[pd.to_numeric(statuses.cell_id)==cid]
            if len(ss):st=ss.iloc[-1].to_dict()
        if st.get('returncode')!=0:
            continue
        target=base/RESULT_ROOT/row['result_name']
        if target.exists():
            print(f'[skip] destination already exists for cell {cid}');continue
        oldroot=source/'csem_paper_results_v1'/oldname
        ev=read_frame(oldroot/'dataframes'/'eval_metrics.csv')
        lo=read_frame(oldroot/'dataframes'/'loss_history.csv')
        if ev.empty and 'cell_id' in evals:ev=evals[pd.to_numeric(evals.cell_id)==cid].copy()
        if lo.empty and 'cell_id' in losses:lo=losses[pd.to_numeric(losses.cell_id)==cid].copy()
        # Combined exports contain empty metric columns from other cells/datasets.
        # Remove only columns with no observations in this run before recovery.
        ev=ev.dropna(axis=1,how='all')
        try:
            # Metadata present in combined exports is checked before relabeling.
            for k in ['dataset','mode','seed','epochs_joint','epochs_refine','T_K','T_full','csem_w','terminal_kl_w','lr_score_head','cfg_scale','eval_samples']:
                if k not in st:raise ValueError(f'Import lacks source configuration: {k}')
                wanted=row[k];got=st[k]
                try:equal=math.isclose(float(wanted),float(got),rel_tol=1e-12,abs_tol=1e-15)
                except (TypeError,ValueError):equal=str(wanted)==str(got)
                if not equal:raise ValueError(f'V5 configuration mismatch: {k}={got}, expected {wanted}')
            if lo.empty:raise ValueError('Missing loss history')
            if row['mode']=='fmnist_aug13_gaussian':
                ev,mapping=normalize_metrics(ev,recover_v5=True);header_recovery[str(cid)]=mapping
            validate_evaluation(row,ev)
        except Exception as exc:
            rejected.append({'cell_id':cid,'reason':repr(exc)});print(f'[not imported] cell {cid}: {exc}');continue
        # Strip exported manifest fields so the new compiler supplies consistent metadata.
        metadata=set(row)|{'returncode','elapsed_seconds','error','run_complete'}
        ev=ev.drop(columns=[c for c in ev if c in metadata],errors='ignore')
        lo=lo.drop(columns=[c for c in lo if c in metadata],errors='ignore')
        frames=target/'dataframes';frames.mkdir(parents=True)
        ev.to_csv(frames/'eval_metrics.csv',index=False);lo.to_csv(frames/'loss_history.csv',index=False)
        write_json(target/'imported_from_v5.json',dict(source=str(source),source_result_name=oldname,
                   values_unchanged=True,header_mapping=header_recovery.get(str(cid),{}),checkpoints_copied=False))
        write_json(base/STATUS_ROOT/f'cell_{cid:03d}.json',dict(row,returncode=0,config_hash=config_hash(row),imported_from=str(source),elapsed_seconds=st.get('elapsed_seconds')))
        imported.append(cid)
    write_json(base/'import_report.json',dict(imported_cells=imported,rejected=rejected,header_recovery=header_recovery,
               note='Only completed compatible records imported. Original files untouched. No training states inferred from CSVs.'))
    print(f'Imported {len(imported)} completed cells: {imported}; rejected: {rejected}')
    return 0 if not rejected else 1


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    for name in ['validate','prepare-data','submit','run','import-v5','report','compile','plot']:
        p=sub.add_parser(name)
        p.add_argument('--base-dir',type=Path,default=DEFAULT_BASE)
        if name=='prepare-data':p.add_argument('--datasets',default='CIFAR,FMNIST')
        if name=='submit':
            p.add_argument('--cells',default='missing',help='missing, all, main, independent, family, 2-3,6-7')
            p.add_argument('--dry-run',action='store_true');p.add_argument('--restart-incomplete',action='store_true')
            p.add_argument('--prepare-job-id',help='Reuse an already submitted preparation job covering these datasets')
        if name=='run':
            p.add_argument('--cell-id',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--restart',action='store_true')
        if name=='import-v5':p.add_argument('--source',type=Path,required=True)
        if name in {'report','compile'}:
            p.add_argument('--cfg-policy',choices=['best-final','recipe','old-paper'],default='best-final')
            p.add_argument('--cifar-treatment',choices=['optimized','raw'],default='optimized')
    a=ap.parse_args();base=a.base_dir.resolve()
    if a.command=='validate':
        import tests
        return tests.run(base)
    if a.command=='prepare-data':return prepare_data(base,a.datasets.split(','))
    if a.command=='submit':return submit_jobs(base,a.cells,a.dry_run,a.restart_incomplete,a.prepare_job_id)
    if a.command=='run':return run_job(base,a.cell_id,a.resume,a.restart)
    if a.command=='import-v5':return import_v5(base,a.source)
    if a.command=='plot':return plot_results(['--base-dir',str(base)])
    args=['--base-dir',str(base),'--cfg-policy',a.cfg_policy,'--cifar-treatment',a.cifar_treatment]
    rc=compile_results(args)
    if not rc and a.command=='report':return plot_results(['--base-dir',str(base)])
    return rc

if __name__=='__main__':raise SystemExit(main())
