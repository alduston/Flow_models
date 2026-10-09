#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import pandas as pd

MANIFEST='csem_paper_final_v1_manifest.csv'
RESULT_ROOT='csem_paper_results_v1'
STATUS_ROOT='csem_paper_status_v1'
OUT_ROOT='csem_paper_compiled_v1'


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
        # V3 uses the exact T_K=0 two-stage protocol: both prior heads are
        # trained during the cotrain epoch clock on detached latents from the
        # same VAE.  Retain Refine tags as a backwards-compatible fallback.
        rr=[]
        specs=[
            ('LSI_Diff','cotrain','Independent CSEM'),
            ('Ctrl_Diff','cotrain','Independent Tweedie'),
            ('LSI_Diff_Refine','refine','Independent CSEM'),
            ('Ctrl_Diff_Refine','refine','Independent Tweedie'),
        ]
        seen=set()
        for tag,stage,reg in specs:
            if reg in seen: continue
            sub=eval_df[(eval_df.get('tag','')==tag) & (eval_df.get('stage','')==stage)].copy()
            if not sub.empty:
                sub['regime']=reg; rr.append(sub); seen.add(reg)
        return rr
    if mode in {'cotrained_csem','fmnist_aug18_recovered','fmnist_aug13_gaussian','cotrained_csem_norm','raw_terminal_control'}:
        sub=eval_df[eval_df.get('tag','').astype(str).str.fullmatch('LSI_Diff',na=False)].copy()
        if sub.empty:
            sub=eval_df[eval_df.get('tag','').astype(str).str.contains('LSI_Diff',na=False)].copy()
        if not sub.empty: sub['regime']='Co-trained CSEM'
        return [sub] if not sub.empty else []
    return []

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,default=Path.cwd())
    ap.add_argument('--cifar-treatment',choices=['optimized','raw'],default='optimized')
    ap.add_argument('--cfg-policy', choices=['best-final','recipe','old-paper'], default='best-final')
    a=ap.parse_args(); base=a.base_dir.resolve(); out=base/OUT_ROOT; out.mkdir(parents=True,exist_ok=True)
    rows=_read_manifest(base)
    all_loss=[]; all_eval=[]; statuses=[]; completed=[]
    for row in rows:
        cid=int(row['cell_id']); rdir=base/RESULT_ROOT/row['result_name']
        status_candidates=list((base/STATUS_ROOT).glob(f'cell_{cid:03d}_*.json'))
        st={}
        if status_candidates:
            try: st=json.loads(status_candidates[0].read_text())
            except Exception: st={}
        statuses.append({**row,'returncode':st.get('returncode'),'elapsed_seconds':st.get('elapsed_seconds'),'error':st.get('error','')})
        lp=rdir/'dataframes'/'loss_history.csv'; ep=rdir/'dataframes'/'eval_metrics.csv'
        if lp.is_file(): all_loss.append(_add_meta(pd.read_csv(lp),row))
        if ep.is_file(): all_eval.append(_add_meta(pd.read_csv(ep),row))
        if lp.is_file() and ep.is_file(): completed.append((row,pd.read_csv(lp),pd.read_csv(ep)))
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
        if not is_main(row) or row['mode']=='independent_pair': continue
        last=_last(edf)
        if last is None: continue
        for c in last.index:
            if not c.startswith('fid_rk4_20_') or 'initgaussianTK' in c or 'initgaussianT' not in c: continue
            m=re.search(r'_cfg([0-9]+(?:_[0-9]+)?)_',c)
            if m and pd.notna(last[c]): candidates.append(dict(dataset=row['dataset'],cfg=float(m.group(1).replace('_','.')),fid=float(last[c])))
    for dataset in ['FMNIST','CIFAR']:
        default=3.0 if dataset=='FMNIST' else 2.5
        if a.cfg_policy=='old-paper': selected[dataset]=1.5 if dataset=='FMNIST' else 3.0
        elif a.cfg_policy=='recipe': selected[dataset]=default
        else:
            cs=pd.DataFrame([r for r in candidates if r['dataset']==dataset])
            selected[dataset]=float(cs.groupby('cfg').fid.mean().idxmin()) if not cs.empty else default
    (out/'selected_cfg.json').write_text(json.dumps(dict(policy=a.cfg_policy,cifar_treatment=a.cifar_treatment,selected=selected,selection='mean final co-trained FID, RK4-20, Gaussian start; candidate grid only'),indent=2)+'\n')
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
            if last is None: continue
            steps=20; cfg_value=selected[row['dataset']]
            fid_col,fid=_find_metric(last,'fid_rk4_',steps,cfg=cfg_value)
            kid_col,kid=_find_metric(last,'kid_rk4_',steps,cfg=cfg_value)
            sw_col,sw=_find_metric(last,'feature_sw2_rk4_',steps,cfg=cfg_value)
            gap=float(last.get('csem_gap_score_uncond',float('nan')))
            endpoints.append({
                'dataset':row['dataset'],'seed':int(row['seed']),'regime':last['regime'],
                'epoch':float(last['epoch']),'sampler_steps':steps,'cfg_scale':cfg_value,
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
    if not enddf.empty:
        agg=enddf.groupby(['dataset','regime'],as_index=False).agg(
            n=('seed','count'),fid_mean=('fid','mean'),fid_sd=('fid','std'),
            kid_mean=('kid','mean'),kid_sd=('kid','std'),sw2_mean=('sw2','mean'),sw2_sd=('sw2','std'),
            csem_gap_mean=('csem_gap','mean'),recon_fid_mean=('recon_fid','mean'),recon_kid_mean=('recon_kid','mean'),recon_sw2_mean=('recon_sw2','mean'))
        agg.to_csv(out/'main_final_metrics_aggregate.csv',index=False)

    # Scale anchor table: terminal arm is main CIFAR CSEM seed 42; other rows are dedicated cells.
    scale_rows=[]
    for row,ldf,edf in completed:
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
        scale_rows.append({'scale_treatment':label,'cell_id':int(row['cell_id']),'recon_loss':float(ll.get('recon',float('nan'))),
                           'recon_fid':float(le.get('fid_vae_recon',float('nan'))),'csem_loss':float(ll.get('score_mse_weighted',ll.get('score_lsi',float('nan')))),
                           'terminal_kl':float(ll.get('terminal_kl',float('nan'))),'latent_rms':float(ll.get('latent_rms',float('nan'))),
                           'posterior_var':float(ll.get('posterior_var',float('nan'))),'fid':fid,'kid':kid})
    pd.DataFrame(scale_rows).to_csv(out/'scale_anchor_final.csv',index=False)

    # Compact T and lambda_K sensitivity; include certified center point from main run.
    sens=[]
    for row,ldf,edf in completed:
        use=(row['family']=='horizon_anchor') or (row['family']=='main' and row['mode']=='cotrained_csem' and row['dataset']=='CIFAR' and int(row['seed'])==42)
        if not use: continue
        le=_last(edf); ll=_last(ldf[ldf.get('stage','')=='cotrain'] if 'stage' in ldf.columns else ldf)
        if le is None or ll is None: continue
        _,fid=_find_metric(le,'fid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
        _,kid=_find_metric(le,'kid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))
        sens.append({'cell_id':int(row['cell_id']),'T_K':float(row['T_K']),'T':float(row['T_full']),'lambda_K':float(row['terminal_kl_w']),
                     'csem_w':float(row['csem_w']),'recon_loss':float(ll.get('recon',float('nan'))),'csem_loss':float(ll.get('score_mse_weighted',float('nan'))),'recon_fid':float(le.get('fid_vae_recon',float('nan'))),
                     'csem_gap':float(le.get('csem_gap_score_uncond',float('nan'))),'terminal_kl':float(ll.get('terminal_kl',float('nan'))),
                     'latent_rms':float(ll.get('latent_rms',float('nan'))),'posterior_var':float(ll.get('posterior_var',float('nan'))),'fid':fid,'kid':kid})
    sensdf=pd.DataFrame(sens)
    if not sensdf.empty: sensdf=sensdf.sort_values(['lambda_K','T'])
    sensdf.to_csv(out/'terminal_horizon_anchor_sensitivity.csv',index=False)

    raw=[]
    for row,ldf,edf in completed:
        if row['mode']!='raw_terminal_control' or int(row['seed'])!=42: continue
        le=_last(edf); ll=_last(ldf[ldf.stage=='cotrain'])
        if le is None or ll is None: continue
        raw.append(dict(cell_id=int(row['cell_id']),T_K=float(row['T_K']),T=float(row['T_full']),lambda_K=float(row['terminal_kl_w']),
            fid=_find_metric(le,'fid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))[1],
            kid=_find_metric(le,'kid_rk4_',int(row['rk4_steps']),cfg=float(row['cfg_scale']))[1],
            latent_rms=ll.get('latent_rms'),terminal_kl=ll.get('terminal_kl'),recon_loss=ll.get('recon'),
            csem_loss=ll.get('score_mse_weighted'),csem_gap=le.get('csem_gap_score_uncond')))
    pd.DataFrame(raw).to_csv(out/'raw_anchor_sensitivity.csv',index=False)

    # Naive Tweedie collapse tables and solver-order extraction.
    collapse=[]; solver=[]
    for row,ldf,edf in completed:
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
        solver.append({'mode':row['mode'],'dataset':row['dataset'],'seed':int(row['seed']),'steps':int(row.get('paper_solver_steps',20)),
                       'heun_fid':hfid,'heun_kid':hkid,'rk4_fid':rfid,'rk4_kid':rkid,
                       'heun_fid_column':hcol,'rk4_fid_column':rcol})
    pd.DataFrame(collapse).to_csv(out/'naive_tweedie_collapse_final.csv',index=False)
    pd.DataFrame(solver).to_csv(out/'solver_order_comparison.csv',index=False)
    print(f'Compiled outputs -> {out}')
    return 0
if __name__=='__main__': raise SystemExit(main())
