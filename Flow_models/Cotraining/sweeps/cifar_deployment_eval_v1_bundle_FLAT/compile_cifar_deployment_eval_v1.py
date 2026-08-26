#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
import pandas as pd
BASE=Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep")
MANIFEST='cifar_deployment_eval_v1_manifest.csv'; ROOT='cifar_deployment_eval_v1_results'; OUT='cifar_deployment_eval_v1_compiled'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=BASE); a=ap.parse_args(); base=a.base_dir.resolve()
    with (base/MANIFEST).open(newline='') as f: rows=list(csv.DictReader(f))
    frames=[]; missing=[]
    for r in rows:
        p=base/ROOT/r['result_name']/'run_terminal_kl'/'dataframes'/f"deployment_grid_ep{r['epoch_label']}.csv"
        if not p.is_file(): missing.append({**r,'reason':'missing deployment_grid'}); continue
        d=pd.read_csv(p)
        for k in ['cell_id','sweep_group','geometry','seed','T_full','T_K','delta_T','csem_w','terminal_kl_w','source_result_name']: d[k]=r[k]
        d['cell_id']=int(r['cell_id']); d['seed']=int(r['seed']); d['T_full']=float(r['T_full']); d['T_K']=float(r['T_K']); d['delta_T']=float(r['delta_T']); d['csem_w']=float(r['csem_w']); d['terminal_kl_w']=float(r['terminal_kl_w']); frames.append(d)
    out=base/OUT; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(missing).to_csv(out/'missing_cells.csv',index=False)
    if not frames: print('No completed cells yet.'); return 0
    all_df=pd.concat(frames,ignore_index=True); all_df.to_csv(out/'deployment_eval_all.csv',index=False)
    primary=all_df[all_df['sweep_group']=='primary'].copy(); keys=['geometry','T_full','T_K','delta_T','rk4_steps','nfe','cfg_scale','init_temperature']; metrics=['fid','kid','w2','div']
    agg=primary.groupby(keys,dropna=False).agg(n_seeds=('seed','nunique'),**{f'{m}_mean':(m,'mean') for m in metrics},**{f'{m}_std':(m,'std') for m in metrics}).reset_index().sort_values(['fid_mean','nfe','cfg_scale','init_temperature'])
    agg.to_csv(out/'primary_seed_aggregates.csv',index=False); agg.sort_values('fid_mean').groupby('nfe',as_index=False).first().sort_values('nfe').to_csv(out/'best_cfg_temp_by_nfe.csv',index=False); agg.head(25).to_csv(out/'top25_primary_configs.csv',index=False)
    agg[['nfe','rk4_steps','cfg_scale','init_temperature','fid_mean','fid_std','kid_mean','w2_mean','n_seeds']].sort_values(['nfe','cfg_scale','init_temperature']).to_csv(out/'primary_cfg_temp_nfe_surface.csv',index=False)
    for axis in ['cfg_scale','init_temperature','nfe']:
        g=primary.groupby(axis).agg(fid_mean=('fid','mean'),fid_std=('fid','std'),kid_mean=('kid','mean'),w2_mean=('w2','mean'),n=('fid','size')).reset_index().sort_values(axis); g.to_csv(out/f'primary_marginal_by_{axis}.csv',index=False)
    robust=all_df[all_df['sweep_group']=='robustness'].copy()
    if len(robust):
        rkeys=['geometry','T_full','T_K','delta_T','rk4_steps','nfe','cfg_scale','init_temperature']; ragg=robust.groupby(rkeys).agg(n_seeds=('seed','nunique'),fid_mean=('fid','mean'),fid_std=('fid','std'),kid_mean=('kid','mean'),w2_mean=('w2','mean')).reset_index().sort_values(['geometry','fid_mean']); ragg.to_csv(out/'robustness_seed_aggregates.csv',index=False); ragg.groupby('geometry',as_index=False).first().to_csv(out/'robustness_best_by_geometry.csv',index=False)
    print(f"Completed cells: {len(frames)}/{len(rows)}"); print(f"Rows: {len(all_df)}"); print('\nBest primary configs:'); print(agg.head(12).to_string(index=False)); print(f"\nCompiled -> {out}"); return 0
if __name__=='__main__': raise SystemExit(main())
