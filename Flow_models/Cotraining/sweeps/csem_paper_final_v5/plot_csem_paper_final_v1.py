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
OUT_ROOT='csem_paper_compiled_v1';FIG_ROOT='csem_paper_figures_v1'
STYLES={'Co-trained CSEM':('#1f77b4','-','o'),'Independent CSEM':('#1f77b4','--','o'),'Independent Tweedie':('#d62728','--','s')}
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})

def read(p):
    if not p.is_file(): return pd.DataFrame()
    try:return pd.read_csv(p)
    except EmptyDataError:return pd.DataFrame()

def save(fig,p):
    fig.tight_layout();fig.savefig(p.with_suffix('.png'),dpi=240,bbox_inches='tight');fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def lines(ax,d,x,y,group,ylabel,main=False):
    if d.empty or not {x,y,group}.issubset(d.columns):return False
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-dir',type=Path,default=Path.cwd());a=ap.parse_args();base=a.base_dir.resolve();comp=base/OUT_ROOT;figs=base/FIG_ROOT;figs.mkdir(parents=True,exist_ok=True)
    d=read(comp/'main_epoch_curves_long.csv');l=read(comp/'all_loss_history.csv');coverage={};missing=[]
    for dataset in ['FMNIST','CIFAR']:
        q=d[d.dataset==dataset].copy() if 'dataset' in d else pd.DataFrame()
        found=set(q.regime.dropna()) if 'regime' in q else set()
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
        s=l[(l.seed.astype(str)=='42')&(l.stage=='cotrain')&l['mode'].isin(['raw_terminal_control','anchor_norm','anchor_none'])].copy()
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
        if {axis,center}.issubset(s):s=s[np.isclose(s[center],center_val)].copy();s['series']='Matched controls'
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for ax,m,label in zip(axs.flat,['fid','kid','latent_rms','terminal_kl','recon_loss','csem_gap'],['FID ↓','KID ↓','Latent RMS','Terminal K_TK','Reconstruction loss','CSEM gap (score units) ↓']):
            lines(ax,s,axis,m,'series',label);ax.set_title(label)
        save(fig,figs/stem);coverage[stem]=not s.empty
    s=read(comp/'solver_order_comparison.csv')
    for _,r in s.iterrows():
        fig,axs=plt.subplots(1,2,figsize=(9,4))
        for ax,m,label in zip(axs,['fid','kid'],['FID ↓','KID ↓']):ax.bar(['Heun SDE','RK4 ODE'],[r.get('heun_'+m,np.nan),r.get('rk4_'+m,np.nan)]);ax.set_ylabel(label)
        fig.suptitle(f"{r['dataset']}: {r['mode']}, {int(r['steps'])} steps (different NFEs)")
        save(fig,figs/f"solver_{r['mode']}_{r['dataset'].lower()}")
    (figs/'figure_coverage.json').write_text(json.dumps(dict(figures_with_data=coverage,missing_regimes=missing,note='Missing curves are never fabricated; bands are across-seed SD where n>1.'),indent=2)+'\n')
    print(f'Figures -> {figs}; missing main regimes: {missing}')
    return 0
if __name__=='__main__':raise SystemExit(main())
