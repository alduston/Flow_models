#!/usr/bin/env python3
"""Continue a CIFAR cotraining run to 800; preserve its original cosine horizon."""
import argparse,csv,json
from pathlib import Path
from csem_paper_suite_v1 import build_base_config,configure_mode,core

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--source-results',type=Path,required=True);ap.add_argument('--seed',type=int,choices=[42,43],required=True)
    ap.add_argument('--allow-weight-only',action='store_true',help='Acknowledge V3 lacks Adam/discriminator/RNG state; warm-start from its saved EMA weights')
    a=ap.parse_args();base=a.base_dir.resolve();src=a.source_results.resolve()
    row=next(r for r in csv.DictReader((base/'csem_paper_final_v1_manifest.csv').open()) if r['family']=='main' and r['dataset']=='CIFAR' and r['mode']=='cotrained_csem' and int(r['seed'])==a.seed)
    dest=base/'csem_paper_results_v1'/row['result_name']
    if dest.exists():raise FileExistsError(f'Refusing to overwrite: {dest}')
    cfg=build_base_config(dataset='CIFAR',seed=a.seed,epochs=800,refine_epochs=0,T_K=1.05,T_full=1.35,csem_w=.05,terminal_kl_w=.6,eval_every=50,eval_samples=10000,cfg_scale=2.5,results_dir=str(dest),latent_anchor_mode='ou_partial',lr_score_head_override=1e-4,paper_solver_suite=True)
    cfg=configure_mode(cfg,'cotrained_csem');cfg.update(data_root=str(base/'data_cache_v2'),paper_cfg_grid=[2.5,3.0],paper_cell_id=int(row['cell_id']),paper_family='main',paper_result_name=row['result_name'])
    # Do not continue unrelated weights just because their architecture matches.
    config_path=src/'paper_cell_config.json'
    if config_path.is_file(): source_cfg=json.loads(config_path.read_text())
    else:
        import ast
        config_path=next((src/p for p in ['config.txt','config_in_progress.txt'] if (src/p).is_file()),None)
        if config_path is None:raise FileNotFoundError('Source resolved configuration is required for continuation')
        source_cfg={}
        for line in config_path.read_text().splitlines():
            if ': ' not in line:continue
            k,v=line.split(': ',1)
            try:source_cfg[k]=ast.literal_eval(v)
            except (ValueError,SyntaxError):source_cfg[k]=v
    for k in ['dataset','seed','T_terminal','t_max','latent_anchor_mode','score_w_vae','kl_w','lr_schedule_epochs']:
        if source_cfg.get(k)!=cfg.get(k):raise ValueError(f'Source configuration mismatch {k}: {source_cfg.get(k)} vs {cfg.get(k)}')
    state=src/'checkpoints'/'training_state_latest.pt'
    if state.is_file():cfg['paper_resume_state']=str(state);kind='Full-state continuation'
    else:
        if not a.allow_weight_only:raise SystemExit('V3 has weight-only checkpoints. Pass --allow-weight-only to explicitly use an approximate continuation; AdamW/discriminator reset and the score starts from saved EMA.')
        cfg.update(load_from_checkpoint=True,ckpt_load_dir=str(src/'checkpoints'),paper_weight_only_continuation=True,paper_start_epoch=500,paper_source_dataframes=str(src/'dataframes'));kind='Weight-only warm start from epoch500; not an exact resume'
        ep=src/'dataframes'/'eval_metrics.csv'
        import pandas as pd
        if not ep.is_file() or int(pd.read_csv(ep).epoch.max())!=500:raise ValueError('Weight-only helper requires a completed epoch500 source')
    print(kind);print('Destination:',dest);print('Original 800-epoch cosine tail, no LR restart')
    core.seed_everything(a.seed);core.train_vae_cotrained_cond(cfg)
    (dest/'paper_cell_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (dest/'CONTINUATION_PROVENANCE.txt').write_text(kind+'\nSource: '+str(src)+'\nOld rows lack the newly added RK4-20 alternate-CFG / feature-SW2 / score-unit gap evaluations.\n')
if __name__=='__main__':main()
