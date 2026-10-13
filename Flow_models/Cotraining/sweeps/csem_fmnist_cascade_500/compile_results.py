#!/usr/bin/env python3
"""Compile the eight-cell FMNIST CSEM sweep: CSVs, concise Markdown, PNGs and PDF.

Adapted from the attached CIFAR compile_results.py. Reads native FMNIST loss.csv,
eval.csv and the sampling/child/gradient diagnostics; never alters training output.
The primary comparison is 10000-sample, legacy Inception FID, Gaussian full-T
start, RK4-20 PER BLOCK, CFG=3. Subset checks are kept in a separate table.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

EXPECTED_CELLS = tuple(range(8))
LABELS = {0:'Flat s42',1:'Optimized s42',2:'Flat s43',3:'Optimized s43',
          4:'Matched depth',5:'Balanced allocation',6:'Same resolution',7:'Bottom-up posterior'}
COLORS = {0:'#4c78a8',1:'#f58518',2:'#4c78a8',3:'#f58518',
          4:'#54a24b',5:'#e45756',6:'#b279a2',7:'#9d755d'}


def read_csv(path):
    path=Path(path)
    if not path.is_file() or not path.stat().st_size: return pd.DataFrame()
    try: return pd.read_csv(path,low_memory=False)
    except (pd.errors.EmptyDataError,pd.errors.ParserError) as exc:
        raise ValueError(f'Cannot read {path}: {exc}') from exc


def asnum(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else float('nan')
    except (ValueError,TypeError): return float('nan')


def display(value,precision=3):
    v=asnum(value)
    return f'{v:.{precision}f}' if math.isfinite(v) else '—'


def last_epoch(df):
    return df.sort_values('epoch').iloc[-1].to_dict() if len(df) else {}


def clean_frame(df,*,keys=('epoch',)):
    if df.empty: return df
    if 'epoch' not in df: raise ValueError('CSV is missing epoch')
    df=df.copy();df['epoch']=pd.to_numeric(df.epoch,errors='raise')
    return df.sort_values(list(keys)).drop_duplicates(list(keys),keep='last').reset_index(drop=True)


def tag(df,cid):
    if len(df):
        df=df.copy();df.insert(0,'cell_id',cid);df.insert(1,'cell_name',LABELS[cid])
    return df


def concat(frames,columns=()):
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame(columns=columns)


def primary(frame):
    if frame.empty: return frame
    required=('fid_backend','cfg','rk4_steps','num_eval')
    if any(k not in frame for k in required): raise ValueError('Evaluation is missing protocol metadata')
    ok=(frame.fid_backend=='inception-legacy') & (frame.cfg.astype(float)==3.0) & \
       (frame.rk4_steps.astype(int)==20) & (frame.num_eval.astype(int)==10000)
    if not ok.all(): raise ValueError('Primary eval.csv mixes incompatible FID protocols')
    return frame


def table(frame,columns,headers=None):
    lines=['| '+' | '.join(headers or columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    strings={'name','status','variant','block_coordinates','score_depths','contrast'}
    ints={'cell_id','seed','eval_last_epoch','fid_best_epoch','epoch','n_seeds'}
    for _,row in frame.iterrows():
        values=[]
        for key in columns:
            val=row.get(key,'')
            if key in strings: values.append(str(val).replace('|','/'))
            elif key in ints: values.append(str(int(val)) if math.isfinite(asnum(val)) else '—')
            else: values.append(display(val))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def save_plot(fig,name,figures,pdf):
    fig.savefig(figures/f'{name}.png',dpi=150,bbox_inches='tight')
    pdf.savefig(fig,bbox_inches='tight');plt.close(fig)


def plot_by_cell(ax,frame,col,*,title=None,ylabel=None):
    used=0
    if len(frame) and col in frame:
        for cid in EXPECTED_CELLS:
            part=frame[frame.cell_id==cid].sort_values('epoch')
            x=pd.to_numeric(part.epoch,errors='coerce').to_numpy(dtype=float)
            y=pd.to_numeric(part[col],errors='coerce').to_numpy(dtype=float)
            ok=np.isfinite(x)&np.isfinite(y)
            if ok.any():
                ax.plot(x[ok],y[ok],label=LABELS[cid],color=COLORS[cid],
                        linestyle='--' if cid in (2,3) else '-',marker='o',ms=2.5,lw=1.6)
                used+=1
    ax.set(xlabel='Training epoch',ylabel=ylabel or col,title=title or ylabel or col)
    ax.grid(alpha=.2)
    if not used: ax.text(.5,.5,'No measured results',ha='center',va='center',transform=ax.transAxes)


def panels(frame,options,shape,title,name,figures,pdf):
    fig,axes=plt.subplots(*shape,figsize=(12.5,shape[0]*3.3),constrained_layout=True)
    for ax,(col,label) in zip(np.asarray(axes).flat,options): plot_by_cell(ax,frame,col,ylabel=label)
    handles,labels=np.asarray(axes).flat[0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.025),fontsize=8)
    fig.suptitle(title,weight='bold',fontsize=13);save_plot(fig,name,figures,pdf)


def plot_reports(figures,pdf,overview,evaluations,losses,sampling,usage,pairs):
    panels(evaluations,[('fid_generated','Generated FID ↓'),('fid_posterior_recon','Posterior reconstruction FID'),
                        ('fid_encoded_root_generated_child','Encoded root + generated child FID'),
                        ('generation_decode_ms_per_image','Generation + decoding ms/image')],(2,2),
           'FMNIST: Gaussian full-T, 10000 samples, RK4-20 per block, CFG=3',
           '01_generated_fid_comparison',figures,pdf)
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
    vals=overview['fid_epoch500'].to_numpy(dtype=float)
    axes[0].bar(range(8),np.nan_to_num(vals,nan=0),color=[COLORS[i] for i in range(8)])
    for i,v in enumerate(vals):
        if math.isfinite(v): axes[0].text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
    axes[0].set_xticks(range(8),[LABELS[i] for i in range(8)],rotation=35,ha='right',fontsize=8)
    axes[0].set(ylabel='Epoch-500 FID ↓',title='Fixed endpoint; missing endpoints have no score')
    if len(pairs):
        for seed,part in pairs.groupby('seed'):
            axes[1].plot(part.epoch,part.cascade_minus_flat,marker='o',label=f'Seed {seed}')
        axes[1].axhline(0,color='black',lw=.8);axes[1].legend()
    else: axes[1].text(.5,.5,'No paired results',ha='center',transform=axes[1].transAxes)
    axes[1].set(xlabel='Epoch',ylabel='Cascade FID − flat FID',title='Paired cascade effect; negative is better')
    for ax in axes: ax.grid(axis='y',alpha=.2)
    save_plot(fig,'02_final_fid_leaderboard',figures,pdf)
    panels(losses,[('recon','Reconstruction MSE'),('score','Coordinate-mean CSEM ε loss'),
                   ('terminal','Terminal KL'),('band_objective','Weighted B contribution'),
                   ('lpips','LPIPS'),('seconds','Training seconds/epoch')],(3,2),
           'FMNIST training: shared objective and 500-epoch cosine schedule',
           '03_training_diagnostics',figures,pdf)
    hier=losses[~losses.cell_id.isin((0,2))] if len(losses) else losses
    panels(hier,[('score_1','Root score loss'),('score_2','Child score loss'),('kl_1','Root terminal KL'),
                 ('kl_2','Child terminal KL'),('var_1','Root posterior variance'),('var_2','Child posterior variance')],(3,2),
           'Two-block models: root and child', '04_root_child_diagnostics',figures,pdf)
    if len(usage):
        probes=usage.pivot(index=['cell_id','epoch'],columns='variant',values='mse_to_posterior_recon').reset_index()
    else: probes=pd.DataFrame()
    panels(probes,[('zero_child','Zero child: decoded MSE change'),('shuffled_child','Shuffled child: decoded MSE change'),
                   ('generated_child_1','Generated child 1: decoded MSE change'),
                   ('generated_child_2','Generated child 2: decoded MSE change')],(2,2),
           'Fixed encoded root: child usage and conditional-prior mismatch',
           '05_hierarchy_generation_diagnostics',figures,pdf)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),constrained_layout=True)
    last=sampling[sampling.epoch==500] if len(sampling) else sampling
    if len(last):
        for cid,part in last.groupby('cell_id'):
            part=part[part.cfg==3].sort_values('rk4_steps')
            axes[0].plot(part.total_rk4_stages_per_image,part.fid_generated,marker='o',color=COLORS[cid],label=LABELS[cid])
        axes[0].legend(fontsize=8,ncol=2)
    else: axes[0].text(.5,.5,'No epoch-500 subset checks',ha='center',transform=axes[0].transAxes)
    axes[0].set(xlabel='Total RK4 stages per image',ylabel='1024-sample FID ↓',title='Separate subset: equal sampling-budget control')
    cost=overview[['generation_decode_ms_per_image','fid_epoch500']].dropna()
    for cid,row in cost.iterrows():
        axes[1].scatter(row.generation_decode_ms_per_image,row.fid_epoch500,color=COLORS[cid])
        axes[1].annotate(LABELS[cid],(row.generation_decode_ms_per_image,row.fid_epoch500),fontsize=8)
    axes[1].set(xlabel='Generation + decoding ms/image',ylabel='10000-sample epoch-500 FID',title='Measured quality and runtime')
    for ax in axes: ax.grid(alpha=.2)
    save_plot(fig,'06_sampling_budget_and_timing',figures,pdf)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--strict',action='store_true',help='Require all eight completed 500-epoch arms')
    args=p.parse_args(argv);base=args.base_dir.resolve()
    out=(args.output_dir if args.output_dir else base/'compiled').resolve()
    out.mkdir(parents=True,exist_ok=True);figures=out/'figures';figures.mkdir(exist_ok=True)
    with (base/'manifest.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    if tuple(int(r['cell_id']) for r in rows)!=EXPECTED_CELLS:
        raise ValueError('Expected exactly the eight FMNIST arms 0..7')
    frames={k:[] for k in ('loss','eval','sampling','usage','gradient','routing')};summaries=[];statuses=[]
    for r in rows:
        cid=int(r['cell_id']);dest=base/'results'/r['result_name']
        stpath=base/'status'/f'cell_{cid:03d}.json'
        st=json.loads(stpath.read_text()) if stpath.is_file() else {}
        loss=clean_frame(read_csv(dest/'loss.csv'));evaluation=primary(clean_frame(read_csv(dest/'eval.csv')))
        sources={'loss':loss,'eval':evaluation,
                 'sampling':clean_frame(read_csv(dest/'eval_sampling_checks.csv'),keys=('epoch','cfg','rk4_steps')),
                 'usage':clean_frame(read_csv(dest/'eval_child_usage.csv'),keys=('epoch','variant')),
                 'gradient':clean_frame(read_csv(dest/'eval_gradient_checks.csv'),keys=('epoch','term')),
                 'routing':clean_frame(read_csv(dest/'csem_routing.csv'))}
        if len(sources['sampling']):
            sources['sampling']['total_rk4_stages_per_image']=4*sources['sampling'].rk4_steps*int(r['blocks'])
        for key,frame in sources.items():
            if len(frame): frames[key].append(tag(frame,cid))
        le=last_epoch(loss);ev=last_epoch(evaluation)
        endpoint=evaluation[evaluation.epoch==500] if len(evaluation) else pd.DataFrame()
        ep=last_epoch(endpoint)
        valid_end=bool(ep) and math.isfinite(asnum(ep.get('fid_generated')))
        if valid_end:
            keys=['fid_posterior_recon']+(['fid_encoded_root_generated_child'] if int(r['blocks'])==2 else [])
            valid_end=all(math.isfinite(asnum(ep.get(k))) for k in keys)
        expected=set(range(50,501,50))
        seen=set(evaluation.epoch.astype(int)) if len(evaluation) else set()
        complete=(st.get('returncode')==0 and int(le.get('epoch',0))==500 and valid_end and expected.issubset(seen))
        state='complete' if complete else ('error' if st.get('returncode') not in (None,0) else 'incomplete')
        summary=dict(cell_id=cid,name=LABELS[cid],variant=r['variant'],seed=int(r['seed']),
                     status=state,blocks=int(r['blocks']),block_coordinates=r['block_coordinates'],score_depths=r['score_depths'],
                     train_last_epoch=le.get('epoch',0),eval_last_epoch=ev.get('epoch',0),
                     fid_latest=asnum(ev.get('fid_generated')),fid_epoch500=asnum(ep.get('fid_generated')),
                     fid_epoch300=float('nan'),fid_best=float('nan'),fid_best_epoch=float('nan'))
        if len(evaluation):
            valid=evaluation[np.isfinite(pd.to_numeric(evaluation.fid_generated,errors='coerce'))]
            if len(valid):
                best=valid.loc[valid.fid_generated.idxmin()];summary.update(fid_best=float(best.fid_generated),fid_best_epoch=int(best.epoch))
            at300=evaluation[evaluation.epoch==300]
            if len(at300): summary['fid_epoch300']=asnum(at300.iloc[-1].fid_generated)
        for k in ('fid_posterior_recon','fid_encoded_root_generated_child','generation_decode_ms_per_image','vae_parameters','score_parameters'):
            summary[k]=asnum(ep.get(k))
        summary['improvement_300_to_500']=summary['fid_epoch500']-summary['fid_epoch300']
        summary['fid_final']=summary['fid_epoch500']
        summary['root_prior_fid_gap']=summary['fid_epoch500']-summary['fid_encoded_root_generated_child']
        for k in ('recon','score','terminal','band','band_weight','band_objective','lpips','score_1','score_2','kl_1','kl_2','var_1','var_2'):
            summary['train_'+k]=asnum(le.get(k))
        summaries.append(summary)
        statuses.append(dict(**summary,result_name=r['result_name'],slurm_job_id=st.get('slurm_job_id',''),error=st.get('error','')))
        print(f'cell {cid}: {LABELS[cid]} | {state} | train {le.get("epoch",0)} | eval {ev.get("epoch",0)}')
    if args.strict and any(s['status']!='complete' for s in summaries):
        raise SystemExit('Not all eight arms have complete status, evaluation history and a finite epoch-500 endpoint')
    overview=pd.DataFrame(summaries).set_index('cell_id',drop=False)
    combined={k:concat(v) for k,v in frames.items()};evaluation=combined['eval'];loss=combined['loss']
    overview.reset_index(drop=True).to_csv(out/'leaderboard.csv',index=False)
    pd.DataFrame(statuses).to_csv(out/'run_status.csv',index=False)
    filenames={'loss':'all_loss_history.csv','eval':'all_eval_records.csv','sampling':'sampling_checks.csv',
               'usage':'child_usage_checks.csv','gradient':'gradient_checks.csv','routing':'csem_routing.csv'}
    for k,filename in filenames.items(): combined[k].to_csv(out/filename,index=False)
    curve=evaluation[['cell_id','cell_name','epoch','fid_generated','num_eval','cfg','rk4_steps']].copy() if len(evaluation) else pd.DataFrame(columns=['cell_id','cell_name','epoch','fid_generated','num_eval','cfg','rk4_steps'])
    curve.to_csv(out/'fid_primary_curve.csv',index=False)
    curve.pivot(index='epoch',columns='cell_id',values='fid_generated').to_csv(out/'fid_primary_wide.csv')
    long_metrics=curve.rename(columns={'fid_generated':'value','rk4_steps':'steps'}).copy()
    long_metrics['metric']='fid';long_metrics['method']='rk4'
    long_metrics['horizon_suffix']='2';long_metrics['source_column']='fid_generated'
    long_metrics['fid_backend']='inception-legacy'
    long_metrics.to_csv(out/'gaussian_fullT_metrics_long.csv',index=False)
    pairs=[]
    for flat,cascade,seed in ((0,1,42),(2,3,43)):
        a=curve[curve.cell_id==flat][['epoch','fid_generated']]
        b=curve[curve.cell_id==cascade][['epoch','fid_generated']]
        for _,v in a.merge(b,on='epoch',suffixes=('_flat','_cascade')).iterrows():
            pairs.append(dict(seed=seed,epoch=int(v.epoch),fid_flat=v.fid_generated_flat,
                              fid_cascade=v.fid_generated_cascade,cascade_minus_flat=v.fid_generated_cascade-v.fid_generated_flat))
    pairs=pd.DataFrame(pairs,columns=['seed','epoch','fid_flat','fid_cascade','cascade_minus_flat'])
    pairs.to_csv(out/'paired_seed_contrasts.csv',index=False)
    agg=overview.reset_index(drop=True).groupby('variant').agg(n_seeds=('seed','count'),
          fid_epoch500_mean=('fid_epoch500','mean'),fid_epoch500_std=('fid_epoch500','std'),
          finite_endpoints=('fid_epoch500','count')).reset_index()
    agg.to_csv(out/'architecture_seed_summary.csv',index=False)
    contrasts=[]
    for reference,test,label in ((1,4,'Matched score depth vs optimized'),(1,5,'Balanced allocation vs optimized'),
                                 (1,6,'Same resolution vs optimized'),(1,7,'Bottom-up child inference vs optimized')):
        contrasts.append(dict(contrast=label,epoch=500,delta_fid=overview.loc[test,'fid_epoch500']-overview.loc[reference,'fid_epoch500']))
    pd.DataFrame(contrasts).to_csv(out/'architecture_contrasts.csv',index=False)
    lines=['# FMNIST CSEM: eight-arm, 500-epoch cascade sweep','',
           f'Compiled {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.','',
           'Primary FID uses 10000 test examples, the recovered legacy Inception backend, Gaussian full-T initialization, '
           'CFG=3 and RK4-20 per block. Every arm uses 256 latent coordinates, noisy-root reconstruction and B_t matched to encoder CSEM.','',
           '## Fixed endpoint and learning progress','',
           table(overview.reset_index(drop=True),['cell_id','name','status','eval_last_epoch','fid_epoch300','fid_epoch500','fid_best','fid_best_epoch','fid_posterior_recon'],
                 ['ID','Variant','Status','Last eval','Epoch 300','Epoch 500','Best observed','Best epoch','Recon at 500']), '',
           'Missing epoch-500 values remain missing. Latest partial evaluations are retained in leaderboard.csv. '
           'The best observed value is a descriptive minimum over evaluation checkpoints. The 300 and 500 checkpoints '
           'belong to this new 500-epoch cosine schedule; they do not reproduce an earlier 300-epoch learning-rate schedule.','',
           '## Paired cascade effect','']
    endpoint_pairs=pairs[pairs.epoch==500]
    lines.append(table(endpoint_pairs,['seed','fid_flat','fid_cascade','cascade_minus_flat'],['Seed','Flat FID','Cascade FID','Cascade − flat']))
    if len(endpoint_pairs)==2 and np.isfinite(endpoint_pairs.cascade_minus_flat).all():
        delta=endpoint_pairs.cascade_minus_flat
        lines.append(f'\nMean paired effect: {delta.mean():+.3f} FID; observed range [{delta.min():+.3f}, {delta.max():+.3f}]. Negative favors the cascade.')
    lines.extend(['','Two training seeds give a replication check, not a precise significance claim. '
                  'The four exploratory variants have one seed. The optimized cascade uses 8+8 score layers versus flat 8; '
                  'cell 4 reduces the cascade to 4+4 to assess that capacity contribution.','',
                  '## Structure contrasts at seed 42','',table(pd.DataFrame(contrasts),['contrast','delta_fid'],['Change from optimized cascade','Δ FID at 500']), '',
                  '## Representation and runtime','',table(overview.reset_index(drop=True),
                    ['cell_id','block_coordinates','score_depths','train_terminal','train_band_objective','fid_encoded_root_generated_child','root_prior_fid_gap','generation_decode_ms_per_image'],
                    ['ID','Coordinates','Depths','Terminal KL','Weighted B','Fixed-root FID','Full − fixed-root FID','ms/image']), '',
                  'Fixed-root FID helps localize conditional-prior versus root-prior mismatch; it is a diagnostic rather than a unique causal decomposition. '
                  'Child zero/shuffle probes, per-block variances, and pre-update gradient audits are retained.','',
                  '## Sampling-budget controls','',
                  'sampling_checks.csv contains the separate 1024-example paired subset: CFG 1 versus 3, RK4-20 versus 40 per block, '
                  'and RK4-10 per block for two-block models. Flat RK4-20 and cascade RK4-10 both perform 80 RK4 stages per image. '
                  'Network FLOPs still differ. Do not compare these subset FIDs numerically to 10000-example primary FIDs.','',
                  '## Outputs','',
                  '- leaderboard.csv, run_status.csv and architecture_seed_summary.csv: endpoints, best checkpoints and progress.',
                  '- fid_primary_curve.csv / fid_primary_wide.csv: matched 10000-example trajectories.',
                  '- paired_seed_contrasts.csv and architecture_contrasts.csv: controlled differences.',
                  '- all_loss_history.csv / all_eval_records.csv: complete per-cell tables.',
                  '- sampling_checks.csv, child_usage_checks.csv, gradient_checks.csv and csem_routing.csv: detailed probes.',
                  '- figures/ and eight_cell_review.pdf: six-page visual review.',''])
    (out/'SUMMARY.md').write_text('\n'.join(lines))
    with PdfPages(out/'eight_cell_review.pdf') as pdf:
        plot_reports(figures,pdf,overview,evaluation,loss,combined['sampling'],combined['usage'],pairs)
    print('\n'+table(overview.reset_index(drop=True),['cell_id','name','status','fid_epoch500','fid_best']))
    print(f'\nCompiled report: {out/"SUMMARY.md"}\nVisual review: {out/"eight_cell_review.pdf"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
