#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,re
from pathlib import Path
import pandas as pd
BASE=Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/tk_anchor_sweep'); MANIFEST='cifar_tk_anchor_compare_v1_manifest.csv'; ROOT='cifar_tk_anchor_compare_v1_results'; OUT='cifar_tk_anchor_compare_v1_compiled'
def pick_col(cols, prefix, token):
    hits=[c for c in cols if c.startswith(prefix) and token in c]
    if len(hits)!=1: return None
    return hits[0]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=BASE); a=ap.parse_args(); base=a.base_dir.resolve()
    with (base/MANIFEST).open(newline='') as f: rows=list(csv.DictReader(f))
    finals=[]; missing=[]; loss_finals=[]
    for r in rows:
        ddir=base/ROOT/r['result_name']/'run_terminal_kl'/'dataframes'; ep=ddir/'eval_metrics.csv'; lp=ddir/'loss_history.csv'
        if not ep.is_file(): missing.append({**r,'reason':'missing eval_metrics','expected_path':str(ep)}); continue
        e=pd.read_csv(ep)
        if e.empty: missing.append({**r,'reason':'empty eval_metrics','expected_path':str(ep)}); continue
        if 'epoch' in e: e=e.sort_values('epoch')
        er=e.iloc[-1].to_dict(); cols=list(e.columns)
        fid_g=pick_col(cols,'fid_rk4_','initgaussianT'); kid_g=pick_col(cols,'kid_rk4_','initgaussianT'); w2_g=pick_col(cols,'sw2_rk4_','initgaussianT')
        fid_q=pick_col(cols,'fid_rk4_','initoracleqTK'); kid_q=pick_col(cols,'kid_rk4_','initoracleqTK'); w2_q=pick_col(cols,'sw2_rk4_','initoracleqTK')
        rec={**r,'eval_epoch':er.get('epoch'),'fid_recon':er.get('fid_vae_recon'),'kid_recon':er.get('kid_vae_recon'),'w2_recon':er.get('sw2_vae_recon'),
             'fid_gaussian':er.get(fid_g) if fid_g else None,'kid_gaussian':er.get(kid_g) if kid_g else None,'w2_gaussian':er.get(w2_g) if w2_g else None,
             'fid_oracle_qtk':er.get(fid_q) if fid_q else None,'kid_oracle_qtk':er.get(kid_q) if kid_q else None,'w2_oracle_qtk':er.get(w2_q) if w2_q else None,
             'gaussian_fid_col':fid_g,'oracle_qtk_fid_col':fid_q}
        finals.append(rec)
        if lp.is_file():
            l=pd.read_csv(lp)
            if not l.empty:
                if 'epoch' in l: l=l.sort_values('epoch')
                lr=l.iloc[-1]
                keep=['loss','recon','kl','terminal_kl','anchor_penalty','anchor_objective','latent_rms','latent_rms_median','posterior_var','posterior_var_median','posterior_std','logvar_median','score_mse_unweighted','score_mse_weighted','score_mse_head_weighted']
                loss_finals.append({**r,**{k:lr.get(k) for k in keep}})
    out=base/OUT; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(missing).to_csv(out/'missing_cells.csv',index=False)
    if not finals: print('No completed cells yet.'); return 0
    fdf=pd.DataFrame(finals); fdf.to_csv(out/'final_eval_by_seed.csv',index=False)
    ldf=pd.DataFrame(loss_finals); ldf.to_csv(out/'final_training_by_seed.csv',index=False)
    metrics=['fid_gaussian','kid_gaussian','w2_gaussian','fid_oracle_qtk','fid_recon']
    agg_spec={'n_seeds':('seed','nunique')}
    for m in metrics:
        if m in fdf and pd.api.types.is_numeric_dtype(fdf[m]): agg_spec[m+'_mean']=(m,'mean'); agg_spec[m+'_std']=(m,'std')
    agg=fdf.groupby(['arm_id','anchor_mode','terminal_kl_w','ou_visible_anchor_w'],dropna=False).agg(**agg_spec).reset_index()
    if 'fid_gaussian_mean' in agg: agg=agg.sort_values('fid_gaussian_mean')
    agg.to_csv(out/'arm_seed_aggregates.csv',index=False)
    if not ldf.empty:
        lm=[c for c in ['terminal_kl','anchor_penalty','anchor_objective','latent_rms','posterior_var','logvar_median','score_mse_unweighted','score_mse_head_weighted'] if c in ldf]
        las={'n_seeds':('seed','nunique')}
        for m in lm: las[m+'_mean']=(m,'mean'); las[m+'_std']=(m,'std')
        lagg=ldf.groupby(['arm_id','anchor_mode','terminal_kl_w','ou_visible_anchor_w'],dropna=False).agg(**las).reset_index()
        lagg.to_csv(out/'training_seed_aggregates.csv',index=False)
    # Paired factorial contrasts on Gaussian FID for each seed.
    if 'fid_gaussian' in fdf:
        pivot=fdf.pivot_table(index='seed',columns='arm_id',values='fid_gaussian',aggfunc='first')
        contrasts=[]
        pairs=[('A_current_kl','B_unanchored','KL effect / no anchor'),('D_historical_gn0_plus_kl','C_historical_gn0','KL effect / GN0'),('F_ou_visible_tk_plus_kl','E_ou_visible_tk','KL effect / OU-visible'),('H_ou_partial_tk_plus_kl','G_ou_partial_tk','KL effect / OU-partial'),('C_historical_gn0','B_unanchored','GN0 effect / no KL'),('E_ou_visible_tk','B_unanchored','OU-visible effect / no KL'),('G_ou_partial_tk','B_unanchored','OU-partial effect / no KL')]
        for hi,lo,label in pairs:
            if hi in pivot and lo in pivot:
                d=(pivot[hi]-pivot[lo]).dropna()
                contrasts.append({'contrast':label,'arm_minus':hi,'baseline':lo,'n_paired_seeds':len(d),'fid_delta_mean':d.mean(),'fid_delta_std':d.std()})
        pd.DataFrame(contrasts).to_csv(out/'paired_fid_contrasts.csv',index=False)
    print(f'Completed cells: {len(fdf)}/{len(rows)}; arms with aggregates: {len(agg)}')
    print(agg.to_string(index=False)); print(f'Compiled -> {out}')
    return 0
if __name__=='__main__': raise SystemExit(main())
