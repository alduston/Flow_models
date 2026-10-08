#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
from csem_paper_suite_v1 import build_base_config, configure_mode

MANIFEST='csem_paper_final_v1_manifest.csv'

def main():
    base=Path(__file__).resolve().parent
    rows=list(csv.DictReader((base/MANIFEST).open(newline='')))
    ids=[int(r['cell_id']) for r in rows]
    assert ids==list(range(len(rows))), f'cell ids must be contiguous: {ids}'
    for r in rows:
        cfg=build_base_config(
            dataset=r['dataset'],seed=int(r['seed']),epochs=int(r['epochs_joint']),refine_epochs=int(r['epochs_refine']),
            T_K=float(r['T_K']),T_full=float(r['T_full']),csem_w=float(r['csem_w']),terminal_kl_w=float(r['terminal_kl_w']),
            eval_every=int(r['eval_every']),eval_samples=int(r['eval_samples']),cfg_scale=float(r['cfg_scale']),
            results_dir=str(base/'_VALIDATION_ONLY'),score_time_weighting=r['score_time_weighting'],
            score_head_time_weighting=r['score_head_time_weighting'],rk4_steps=int(r['rk4_steps']),
            paper_solver_suite=r['paper_solver_suite']=='1',paper_solver_steps=int(r['paper_solver_steps']),
            latent_anchor_mode=r.get('latent_anchor_mode','none'))
        cfg=configure_mode(cfg,r['mode'],beta0=float(r['beta0']))
        assert cfg['t_max'] >= cfg['T_terminal'] >= 0
        if r['mode']=='independent_pair':
            assert cfg['T_terminal']==0 and cfg['freeze_score_in_cotrain'] and cfg['train_tracking_head']
        if r['mode']=='naive_tweedie_cotrain':
            assert cfg['cotrain_head']=='control' and cfg['kl_w']==0 and not cfg['use_latent_norm']
        if r['mode']=='cotrained_csem':
            assert cfg['cotrain_head']=='lsi' and cfg['kl_reg_type']=='terminal' and not cfg['use_latent_norm']
            if r['dataset']=='CIFAR': assert cfg['latent_anchor_mode']=='ou_partial'
        if r['mode']=='sensitivity':
            assert cfg['latent_anchor_mode']=='ou_partial' and cfg['kl_reg_type']=='terminal'
        if r['mode']=='raw_terminal_control':
            assert cfg['latent_anchor_mode']=='none' and cfg['kl_reg_type']=='terminal'
        print(f"OK cell {r['cell_id']:>2} {r['result_name']}")
    print(f'Validated {len(rows)} cells.')
    return 0
if __name__=='__main__': raise SystemExit(main())
