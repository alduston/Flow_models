#!/usr/bin/env python3
"""Compile all five runs, all hybrids, per-block risks and gradient diagnostics."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from suite import manifest, validate_manifest

FILES={'block_cfg':'eval_block_cfg.csv','loss':'loss.csv','eval':'eval.csv','combinations':'eval_fid_combinations.csv',
       'risk':'eval_csem_risk.csv','risk_queries':'eval_csem_risk_queries.csv',
       'sampling':'eval_sampling_checks.csv','usage':'eval_child_usage.csv',
       'gradients':'eval_gradient_checks.csv','routing':'csem_routing.csv',
       'reconstruction_routing':'reconstruction_routing.csv'}

def read(path):
    return pd.read_csv(path) if path.is_file() and path.stat().st_size else pd.DataFrame()

def markdown(frame):
    if frame.empty: return 'No results yet.'
    def val(v):
        if isinstance(v,(float,np.floating)): return f'{v:.5g}' if math.isfinite(v) else '—'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(val(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--strict',action='store_true')
    a=p.parse_args(argv);base=a.base_dir.resolve();rows=manifest(base);validate_manifest(rows)
    out=base/'report';out.mkdir(exist_ok=True)
    frames={key:[] for key in FILES};endpoints=[]
    from paper import validate_evaluation
    for cell in rows:
        dest=base/'results'/cell['result_name'];valid=False
        if (dest/'eval.csv').exists():
            try: validate_evaluation(cell,dest);valid=True
            except ValueError:
                if a.strict: raise
        elif a.strict: raise ValueError(f'Missing cell {cell["cell_id"]}')
        per={}
        for key,name in FILES.items():
            frame=read(dest/name)
            if not frame.empty:
                frame=frame.copy();frame['cell_id']=int(cell['cell_id']);frame['cell_name']=cell['variant']
                frame['blocks']=int(cell['blocks']);frame['topology']=cell['conditioning'];frame['seed']=int(cell['seed'])
                frames[key].append(frame);per[key]=frame
        ev=per.get('eval',pd.DataFrame())
        endpoint=dict(cell_id=int(cell['cell_id']),name=cell['variant'],blocks=int(cell['blocks']),
            topology=cell['conditioning'],seed=int(cell['seed']),status='complete' if valid else 'incomplete',
            coordinates=cell['block_coordinates'],score_depths=cell['score_depths'])
        if not ev.empty:
            last=ev.sort_values('epoch').iloc[-1];best=ev.loc[ev.fid_generated.idxmin()]
            endpoint.update(last_epoch=int(last.epoch),best_fid=float(best.fid_generated),best_epoch=int(best.epoch))
            for metric in ['fid_generated','fid_posterior_recon','fid_encoded_root_generated_child','fid_encoded_z1_z2_generated_z3']:
                endpoint[metric]=last.get(metric,np.nan)
        config=dest/'config.json'
        if config.exists():
            cfg=json.loads(config.read_text())
            for key in ('vae_parameters','score_parameters','tmin_band_weight_resolved'): endpoint[key]=cfg.get(key,np.nan)
        endpoints.append(endpoint)
    combined={k:pd.concat(v,ignore_index=True,sort=False) if v else pd.DataFrame() for k,v in frames.items()}
    for key,frame in combined.items(): frame.to_csv(out/f'all_{key}.csv',index=False)
    summary=pd.DataFrame(endpoints);summary.to_csv(out/'summary.csv',index=False)
    contrasts=[];ev=combined['eval']
    if not ev.empty:
        for control in (1,2):
            left=ev[ev.cell_id==control][['epoch','fid_generated']]
            for other in range(5):
                if other==control: continue
                right=ev[ev.cell_id==other][['epoch','fid_generated']]
                for _,r in left.merge(right,on='epoch',suffixes=('_control','_other')).iterrows():
                    contrasts.append(dict(control=control,other=other,epoch=int(r.epoch),
                        fid_control=r.fid_generated_control,fid_other=r.fid_generated_other,
                        other_minus_control=r.fid_generated_other-r.fid_generated_control))
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    lines=['# FMNIST conditioning comparison','',markdown(summary),'',
        'All runs: seed 42, 500 joint epochs, 256 coordinates, 8 layers per head. '
        'Primary FID: 10,000 images, legacy Inception, CFG 3, RK4-20 per generated block. '
        'Depth comparisons have different score capacity and sampling cost; one seed does not establish statistical dominance.','',
        'Three-block inference and decoder are shared; only prior conditioning changes. '
        'E is a fixed encoded posterior sample; G is a generated code using the parents actually present. '
        'Generated-parent/encoded-descendant hybrids are conditioning stress interventions. '
        'FID differences are not additive error contributions.','',
        'Bayes/excess fields are finite-reference estimates in epsilon units, per coordinate, at CFG 1. '
        'Inspect ESS, split-bank disagreement, 64-to-256 bank stability and the orthogonality residual. '
        'Cross-fitted estimates can be negative due to finite-reference error.']
    combos=combined['combinations']
    if not combos.empty:
        last=combos[combos.epoch==combos.groupby('cell_id').epoch.transform('max')]
        lines+=['','## Latest block combinations','',markdown(last[['cell_name','epoch','combination','kind','fid','num_eval']])]
    risk=combined['risk']
    if not risk.empty:
        last=risk[(risk.time_bin==-1)&(risk.head_source=='ema')]
        last=last[last.epoch==last.groupby('cell_id').epoch.transform('max')]
        lines+=['','## Latest finite-reference risks','',markdown(last[['cell_name','epoch','block','bank_size',
            'raw_csem_risk','bayes_risk_proxy','excess_risk_proxy','orthogonality_residual','ess','low_ess','split_oracle_disagreement']])]
    block_cfg=combined['block_cfg']
    if not block_cfg.empty and 'fid' in block_cfg:
        last=block_cfg[block_cfg.epoch==block_cfg.groupby('cell_id').epoch.transform('max')]
        lines+=['','## Latest block-specific guidance probes (paired subset)','',
                markdown(last[['cell_name','epoch','combination','changed_block','cfg_vector','fid','mse_to_default','num_eval']])]
    gradients=combined['gradients']
    if not gradients.empty and 'grad_norm_z2_posterior' in gradients:
        terms=gradients[gradients.term.isin(['csem_2','csem_3'])]
        norm=terms.pivot(index=['cell_id','cell_name','epoch'],columns='term',values='grad_norm_z2_posterior').reset_index()
        if 'csem_2' in norm and 'csem_3' in norm:
            norm['downstream_over_own_z2']=norm['csem_3']/norm['csem_2'].replace(0,np.nan)
            norm.to_csv(out/'z2_gradient_pressure.csv',index=False)
            last=norm[norm.epoch==norm.groupby('cell_id').epoch.transform('max')]
            lines+=['','## Latest middle-posterior gradient pressure','',markdown(last)]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    colors=plt.get_cmap('tab10')
    with PdfPages(out/'comparison.pdf') as pdf:
        def panel(frame,metrics,title):
            if frame.empty: return
            metrics=[m for m in metrics if m in frame and frame[m].notna().any()]
            if not metrics: return
            fig,axes=plt.subplots(len(metrics),1,figsize=(10,3.1*len(metrics)),squeeze=False)
            for ax,metric in zip(axes[:,0],metrics):
                for cid,group in frame.groupby('cell_id'):
                    group=group.sort_values('epoch')
                    ax.plot(group.epoch,group[metric],marker='.',color=colors(cid),label=rows[cid]['variant'])
                ax.set(xlabel='Epoch',ylabel=metric);ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle(title);fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        panel(ev,['fid_generated','fid_posterior_recon','fid_encoded_root_generated_child','fid_encoded_z1_z2_generated_z3'],'Primary and staged FIDs')
        panel(combined['loss'],['total','recon','lpips','score','terminal'],'Training objectives')
        panel(combined['loss'],['score_1','score_2','score_3','var_1','var_2','var_3'],'Per-block objectives and variances')
        if not combos.empty:
            latest=combos[combos.epoch==combos.groupby('cell_id').epoch.transform('max')]
            pivot=latest.pivot(index='cell_name',columns='combination',values='fid')
            data=pivot.to_numpy();fig,ax=plt.subplots(figsize=(10,5))
            im=ax.imshow(np.ma.masked_invalid(data),aspect='auto',cmap='viridis_r')
            ax.set_xticks(range(len(pivot.columns)),pivot.columns);ax.set_yticks(range(len(pivot.index)),pivot.index)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if math.isfinite(data[i,j]): ax.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax,label='FID');ax.set_title('All hybrids at latest evaluation');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        if not risk.empty:
            for block in range(1,4):
                frame=risk[(risk.block==block)&(risk.time_bin==-1)&(risk.head_source=='ema')&(risk.bank_size==risk.bank_size.max())]
                panel(frame,['raw_csem_risk','bayes_risk_proxy','excess_risk_proxy','ess','split_oracle_disagreement'],f'Block {block}: EMA finite-reference risks')
        grads=combined['gradients']
        if not grads.empty: panel(grads[grads.term=='joint'],['cosine_csem2_csem3_z2_posterior','cosine_csem2_csem3_vae'],'Cross-block gradient alignment')
        if all(f.empty for f in combined.values()):
            fig,ax=plt.subplots(figsize=(9,4));ax.axis('off');ax.text(.1,.6,'No training results yet.\nRe-run paper.py report after the cluster jobs.',fontsize=16);pdf.savefig(fig);plt.close(fig)
    print(f'Report written to {out}')
    return 0

if __name__=='__main__': raise SystemExit(main())
