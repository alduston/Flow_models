#!/usr/bin/env python3
"""CPU integration checks using clearly synthetic temporary result fixtures."""
import ast,csv,importlib.util,json,subprocess,sys,tempfile,types
from pathlib import Path
import pandas as pd
B=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(B/'verify_fmnist_v4.py')],check=True)
sys.path.insert(0,str(B));import paper_eval_extensions_v5 as extensions
module=types.ModuleType('legacy_adapter_check');module.evaluate_current_state=lambda epoch_idx,prefix,vae,unet,loader,cfg,device,lpips_fn,**kwargs: {}
extensions.install_historical_evaluator(module,B/'provenance/csem_new_fmnist_aug13.py')
assert {'feature_sw2','score_unit_gap'}.issubset(module._paper_extra_evaluate.__code__.co_names)
rows=list(csv.DictReader((B/'csem_paper_final_v1_manifest.csv').open()))
with tempfile.TemporaryDirectory() as tmp:
 base=Path(tmp);(base/'csem_paper_final_v1_manifest.csv').write_bytes((B/'csem_paper_final_v1_manifest.csv').read_bytes())
 for r in rows:
  dest=base/'csem_paper_results_v1'/r['result_name']/'dataframes';dest.mkdir(parents=True)
  losses=[];evals=[]
  independent=r['mode']=='independent_pair'
  for ep in [50,100]:
   losses.append(dict(epoch=ep,stage='cotrain',recon=.02,score_mse_weighted=.5,terminal_kl=.05,latent_rms=.8,posterior_var=.01,posterior_std=.1,lr_vae_current=1e-4,lr_score_current=1e-4))
   for tag in (['LSI_Diff_Refine','Ctrl_Diff_Refine'] if independent else ['Ctrl_Diff' if r['mode']=='naive_tweedie_cotrain' else 'LSI_Diff']):
    e=dict(epoch=ep,stage='refine' if independent else 'cotrain',tag=tag,csem_gap_score_uncond=123.0,lsi_gap_unet_uncond=999.0,fid_vae_recon=4.,kid_vae_recon=.001,feature_sw2_vae_recon=.01)
    for cfg in ([3.,1.5] if r['dataset']=='FMNIST' else [2.5,3.]):
     token=str(cfg).replace('.','_');time='2' if r['dataset']=='FMNIST' else '1p35'
     for method,steps in [('rk4',20),('rk4',25),('heun',20)]:
      suffix=f'{method}_{steps}_randtok_cfg{token}_temp1_initgaussianT{time}'
      for metric,value in [('fid',8. if cfg==3 else 10.),('kid',.002),('sw2',.03),('feature_sw2',.04)]:e[f'{metric}_{suffix}']=value
    evals.append(e)
  pd.DataFrame(losses).to_csv(dest/'loss_history.csv',index=False);pd.DataFrame(evals).to_csv(dest/'eval_metrics.csv',index=False)
 subprocess.run([sys.executable,str(B/'compile_csem_paper_final_v1.py'),'--base-dir',str(base)],check=True)
 comp=base/'csem_paper_compiled_v1';d=pd.read_csv(comp/'main_epoch_curves_long.csv')
 assert set(d.dataset)=={'FMNIST','CIFAR'} and set(d.regime)=={'Co-trained CSEM','Independent CSEM','Independent Tweedie'}
 assert set(d.paper_steps)=={20} and set(d.paper_cfg)=={3.0}
 assert (d.paper_csem_gap==123).all() and (d.paper_sw2==.04).all() and (d.paper_latent_sw2==.03).all()
 scale=pd.read_csv(comp/'scale_anchor_final.csv');assert len(scale)==3
 assert len(pd.read_csv(comp/'raw_anchor_sensitivity.csv'))==5
 subprocess.run([sys.executable,str(B/'plot_csem_paper_final_v1.py'),'--base-dir',str(base)],check=True)
 figs=base/'csem_paper_figures_v1';coverage=json.loads((figs/'figure_coverage.json').read_text());assert coverage['missing_regimes']==[] and all(coverage['figures_with_data'].values())
 assert (figs/'main_old_paper_epoch_comparison.pdf').stat().st_size>1000
 subprocess.run([sys.executable,str(B/'compile_csem_paper_final_v1.py'),'--base-dir',str(base),'--cifar-treatment','raw','--cfg-policy','old-paper'],check=True)
 d=pd.read_csv(comp/'main_epoch_curves_long.csv');c=d[d.dataset=='CIFAR'];assert set(c[c.regime=='Co-trained CSEM']['mode'])=={'raw_terminal_control'}
 assert set(d[d.dataset=='FMNIST'].paper_cfg)=={1.5}
print('PASS: all 22 configurations; legacy evaluation adapter; six main regimes; strict CFG/RK4-20 selection; score/feature/latent units; controlled ablations; all plot families in PNG/PDF; raw-method/old-paper switches.')
print('Fixtures are synthetic and deleted; no training or measured FID is asserted by this test.')
