#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt

OUT_ROOT='csem_paper_compiled_v1'
FIG_ROOT='csem_paper_figures_v1'

def save_line(df,x,y,group,title,ylabel,path):
    if df.empty or y not in df.columns: return
    plt.figure(figsize=(7.2,4.8))
    for name,g in df.groupby(group):
        gg=g.sort_values(x)
        if gg[y].notna().any(): plt.plot(gg[x],gg[y],marker='o',label=str(name))
    plt.xlabel(x.replace('_',' ')); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=220); plt.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=Path.cwd())
    a=ap.parse_args(); base=a.base_dir.resolve(); comp=base/OUT_ROOT; figs=base/FIG_ROOT; figs.mkdir(parents=True,exist_ok=True)

    curves_path=comp/'main_epoch_curves_long.csv'
    if curves_path.is_file():
        try: d=pd.read_csv(curves_path)
        except EmptyDataError: d=pd.DataFrame()
        for dataset in sorted(d['dataset'].dropna().unique()) if 'dataset' in d.columns else []:
            q=d[d['dataset']==dataset].copy()
            # Average seeds at each regime/epoch when multiple seeds exist.
            agg=q.groupby(['regime','epoch'],as_index=False).agg(paper_fid=('paper_fid','mean'),paper_kid=('paper_kid','mean'),paper_sw2=('paper_sw2','mean'))
            save_line(agg,'epoch','paper_fid','regime',f'{dataset}: generation FID vs score-training epoch','FID ↓',figs/f'main_{dataset.lower()}_fid_vs_epoch.png')
            save_line(agg,'epoch','paper_kid','regime',f'{dataset}: generation KID vs score-training epoch','KID ↓',figs/f'main_{dataset.lower()}_kid_vs_epoch.png')
            save_line(agg,'epoch','paper_sw2','regime',f'{dataset}: latent SW2 vs score-training epoch','SW2 ↓',figs/f'main_{dataset.lower()}_sw2_vs_epoch.png')

    loss_path=comp/'all_loss_history.csv'
    if loss_path.is_file():
        try: l=pd.read_csv(loss_path)
        except EmptyDataError: l=pd.DataFrame()
        # CIFAR scale-anchor dynamics: certified terminal arm + dedicated norm/no-anchor cells.
        ids={'cifar_main_csem_s42':'Terminal K_TK','cifar_anchor_norm_s42':'Architectural normalization','cifar_anchor_none_s42':'No anchor'}
        s=l[l['result_name'].isin(ids)].copy() if 'result_name' in l.columns else pd.DataFrame()
        if not s.empty:
            s['scale_treatment']=s['result_name'].map(ids)
            s=s[s['stage']=='cotrain']
            for metric,label,fn in [
                ('recon','reconstruction loss','scale_anchor_reconstruction.png'),
                ('score_mse_weighted','CSEM loss','scale_anchor_csem_loss.png'),
                ('terminal_kl','terminal K_TK','scale_anchor_terminal_kl.png'),
                ('latent_rms','latent RMS','scale_anchor_latent_rms.png'),
                ('posterior_var','posterior variance','scale_anchor_posterior_variance.png'),
            ]:
                if metric in s.columns: save_line(s,'epoch',metric,'scale_treatment',f'Scale-anchor ablation: {label}',label,figs/fn)

        n=l[l['result_name']=='cifar_naive_tweedie_cotrain_s42'].copy() if 'result_name' in l.columns else pd.DataFrame()
        if not n.empty:
            n=n[n['stage']=='cotrain']
            if 'posterior_std' in n.columns: save_line(n,'epoch','posterior_std','result_name','Naive Tweedie co-training: posterior scale','mean posterior std',figs/'tweedie_collapse_posterior_std.png')
            if 'latent_rms' in n.columns: save_line(n,'epoch','latent_rms','result_name','Naive Tweedie co-training: latent RMS','latent RMS',figs/'tweedie_collapse_latent_rms.png')

    sens_path=comp/'terminal_horizon_anchor_sensitivity.csv'
    if sens_path.is_file():
        try: s=pd.read_csv(sens_path)
        except EmptyDataError: s=pd.DataFrame()
        if not s.empty:
            q=s[np.isclose(s['lambda_K'],0.60)].sort_values('T')
            if len(q):
                plt.figure(figsize=(6.6,4.6)); plt.plot(q['T'],q['fid'],marker='o'); plt.xlabel('Full horizon T'); plt.ylabel('FID ↓'); plt.title('Terminal horizon sensitivity (lambda_K=0.60)'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(figs/'sensitivity_T_fid.png',dpi=220); plt.close()
            q=s[np.isclose(s['T'],1.35)].sort_values('lambda_K')
            if len(q):
                plt.figure(figsize=(6.6,4.6)); plt.plot(q['lambda_K'],q['fid'],marker='o'); plt.xlabel('lambda_K'); plt.ylabel('FID ↓'); plt.title('Terminal-anchor weight sensitivity (T=1.35)'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(figs/'sensitivity_lambdaK_fid.png',dpi=220); plt.close()

    solver_path=comp/'solver_order_comparison.csv'
    if solver_path.is_file():
        try: s=pd.read_csv(solver_path)
        except EmptyDataError: s=pd.DataFrame()
        s=s.dropna(subset=['heun_fid','rk4_fid'],how='all')
        for _,r in s.iterrows():
            vals=[r.get('heun_fid',np.nan),r.get('rk4_fid',np.nan)]
            plt.figure(figsize=(5.6,4.4)); plt.bar(['Heun SDE','RK4 ODE'],vals); plt.ylabel('FID ↓'); plt.title(f"{r['mode']} solver comparison ({int(r['steps'])} steps)"); plt.tight_layout(); plt.savefig(figs/f"solver_{r['mode']}_{r['dataset'].lower()}_fid.png",dpi=220); plt.close()

    print(f'Figures -> {figs}')
    return 0
if __name__=='__main__': raise SystemExit(main())
