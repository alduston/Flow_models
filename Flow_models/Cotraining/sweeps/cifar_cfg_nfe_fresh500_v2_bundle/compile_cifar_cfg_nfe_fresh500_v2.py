#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
import pandas as pd

BASE=Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep')
MANIFEST='cifar_cfg_nfe_fresh500_v2_manifest.csv'
ROOT='cifar_cfg_nfe_fresh500_v2_results'
OUT='cifar_cfg_nfe_fresh500_v2_compiled'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=BASE); a=ap.parse_args(); base=a.base_dir.resolve()
    with (base/MANIFEST).open(newline='') as f: rows=list(csv.DictReader(f))
    frames=[]; missing=[]
    for r in rows:
        p=base/ROOT/r['result_name']/'run_terminal_kl'/'dataframes'/f"deployment_grid_ep{r['epochs']}.csv"
        if not p.is_file():
            missing.append({**r,'reason':'missing deployment_grid','expected_path':str(p)}); continue
        d=pd.read_csv(p)
        d['cell_id']=int(r['cell_id']); d['seed']=int(r['seed']); d['T_K']=float(r['T_K']); d['T_full']=float(r['T_full'])
        d['delta_T']=float(r['delta_T']); d['csem_w']=float(r['csem_w']); d['terminal_kl_w']=float(r['terminal_kl_w'])
        d['train_epochs']=int(r['epochs']); d['checkpoint_reload_used']=False
        frames.append(d)
    out=base/OUT; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(missing).to_csv(out/'missing_seeds.csv',index=False)
    if not frames:
        print('No completed fresh-training seeds yet.'); return 0
    all_df=pd.concat(frames,ignore_index=True)
    all_df.to_csv(out/'cfg_nfe_all.csv',index=False)

    # Temperature is intentionally fixed at 1.0, so the scientific surface is CFG x NFE.
    metrics=[m for m in ['fid','kid','w2','div'] if m in all_df.columns]
    agg_spec={'n_seeds':('seed','nunique')}
    for m in metrics:
        agg_spec[f'{m}_mean']=(m,'mean'); agg_spec[f'{m}_std']=(m,'std')
    agg=(all_df.groupby(['rk4_steps','nfe','cfg_scale'],dropna=False)
         .agg(**agg_spec).reset_index().sort_values(['fid_mean','nfe','cfg_scale']))
    agg.to_csv(out/'cfg_nfe_seed_aggregates.csv',index=False)
    agg.sort_values(['nfe','fid_mean']).groupby('nfe',as_index=False).first().sort_values('nfe').to_csv(out/'best_cfg_by_nfe.csv',index=False)
    agg.sort_values(['cfg_scale','fid_mean']).groupby('cfg_scale',as_index=False).first().sort_values('cfg_scale').to_csv(out/'best_nfe_by_cfg.csv',index=False)
    agg.sort_values('fid_mean').head(15).to_csv(out/'ranked_cfg_nfe_configs.csv',index=False)
    baseline=agg[(agg['cfg_scale'].sub(3.0).abs()<1e-12) & (agg['nfe']==100)]
    baseline.to_csv(out/'baseline_cfg3_nfe100.csv',index=False)

    print(f'Completed seeds: {all_df.seed.nunique()}/{len(rows)}')
    print(f'Raw deployment rows: {len(all_df)}')
    print('\nBest CFG x NFE configs:')
    print(agg.head(15).to_string(index=False))
    if len(baseline):
        print('\nBaseline reproduction (CFG=3, NFE=100):')
        print(baseline.to_string(index=False))
    print(f'\nCompiled -> {out}')
    return 0

if __name__=='__main__': raise SystemExit(main())
