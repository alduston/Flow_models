"""Compile complete AND partial runs without selecting the best FID checkpoint."""
import csv
import json
from pathlib import Path
import pandas as pd
import numpy as np
import experiments


def read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run(base,strict=False):
    base=Path(base);dest=base/'report';dest.mkdir(exist_ok=True)
    summaries=[];tables={k:[] for k in ('eval','loss','diagnostics','witness','reference_audit')}
    for row in experiments.rows():
        out=base/'results'/row['name'];loss=read(out/'loss.csv')
        ep=int(loss.epoch.max()) if len(loss) else 0
        complete=(out/'complete.json').exists() and json.loads((out/'complete.json').read_text())['epoch']==row['config']['epochs']
        record=dict(cell_id=row['cell_id'],name=row['name'],group=row['group'],stage=row['stage'],source=row['source'],
                    seed=row['config']['seed'],blocks=row['config']['blocks'],topology=row['config']['topology'],
                    rep_multiplier=row['config']['rep_multiplier'],epoch=ep,complete=complete)
        ev=read(out/'eval.csv')
        for cfg in (1.,3.):
            if len(ev):
                chosen=ev[(ev.cfg==cfg)&(ev.init=='gaussian')&(ev.num_eval==row['config']['eval_samples'])&(ev.steps==row['config']['rk4_steps'])]
                if len(chosen):
                    last=chosen.sort_values('epoch').iloc[-1]
                    record[f'fid_cfg{cfg:g}']=last.fid_generated;record[f'fid_epoch_cfg{cfg:g}']=int(last.epoch)
        summaries.append(record)
        for name in tables:
            file=out/f'{name}.csv' if name in ('loss','eval') else out/name/'metrics.csv'
            table=read(file)
            if len(table):
                table.insert(0,'cell_name',row['name']);table.insert(0,'cell_id',row['cell_id']);table['seed']=row['config']['seed']
                tables[name].append(table)
        if strict and not complete:raise RuntimeError(f'Incomplete cell: {row["name"]}')
        if strict:
            from paper import validate_complete
            validate_complete(row,out)
    summary=pd.DataFrame(summaries);summary.to_csv(dest/'summary.csv',index=False)
    merged={}
    for name,items in tables.items():
        merged[name]=pd.concat(items,ignore_index=True) if items else pd.DataFrame()
        merged[name].to_csv(dest/f'all_{name}.csv',index=False)
    mechanism=[]
    for kind in ('diagnostics','witness','reference_audit'):
        table=merged[kind]
        if len(table):
            keys=['cell_id','time','conditioning','head','block','metric']
            last=table.sort_values('epoch').groupby(keys,dropna=False,as_index=False).tail(1).copy()
            last['table']=kind;mechanism.append(last)
    if mechanism:
        endpoints=pd.concat(mechanism,ignore_index=True).merge(summary[['cell_id','complete','stage','blocks','topology','rep_multiplier']],on='cell_id')
        endpoints.to_csv(dest/'mechanism_endpoints.csv',index=False)
    # Planned paired contrasts, aligned at COMMON epochs and matched seed.
    contrasts=[];ev=merged['eval']
    planned=[('pressure0','b2_full_r1','b2_full_r0'),('pressure4','b2_full_r1','b2_full_r4'),
             ('decoupled2','b2_full_r1','b2_decoupled_r1'),('pressure4_3','b3_full_r1','b3_full_r4'),
             ('markov3','b3_full_r1','b3_markov_r1'),('siblings3','b3_full_r1','b3_siblings_r1'),
             ('decoupled3','b3_full_r1','b3_decoupled_r1'),('band2','b2_full_r4','b2_full_r4_band_scaled'),
             ('band3','b3_full_r4','b3_full_r4_band_scaled'),
             ('clip1','b2_full_r1','b2_full_r1_separate_clip'),('clip4','b2_full_r4','b2_full_r4_separate_clip')]
    if len(ev):
        for label,control,treatment in planned:
            for seed in (42,43):
                a=ev[ev.cell_name==f'{control}_s{seed}'];b=ev[ev.cell_name==f'{treatment}_s{seed}']
                pairs=a.merge(b,on=['epoch','cfg','steps','init','num_eval','seed'],suffixes=('_control','_treatment'))
                for _,r in pairs.iterrows():contrasts.append(dict(contrast=label,seed=seed,epoch=r.epoch,cfg=r.cfg,steps=r.steps,init=r.init,num_eval=r.num_eval,
                    delta_fid=r.fid_generated_treatment-r.fid_generated_control))
        for parent in [r for r in experiments.rows() if r['stage']=='joint' and any(q['source']==r['cell_id'] for q in experiments.rows())]:
            arms={teacher:ev[ev.cell_name==parent['name']+'_frozen_'+teacher] for teacher in ('naive','corrected','dsm')}
            for teacher in ('corrected','dsm'):
                pairs=arms['naive'].merge(arms[teacher],on=['epoch','cfg','steps','init','num_eval','seed'],suffixes=('_control','_treatment'))
                for _,r in pairs.iterrows():contrasts.append(dict(contrast=parent['name']+'_frozen_'+teacher,seed=r.seed,epoch=r.epoch,cfg=r.cfg,steps=r.steps,init=r.init,num_eval=r.num_eval,
                    delta_fid=r.fid_generated_treatment-r.fid_generated_control))
    pd.DataFrame(contrasts).to_csv(dest/'paired_fid_contrasts.csv',index=False)
    frozen_risk_contrasts(base,dest)
    mechanism_plots(merged,summary,dest)
    body=['# Naive stacked CSEM mechanism study','',
          'Endpoints are the latest observed epochs, never best-checkpoint selections. Partial runs are explicitly labeled.',
          'Two seeds provide paired replication; they are not a precise population uncertainty estimate.','',
          '| Cell | Name | Epoch | Complete | CFG1 FID | CFG3 FID |','|---:|---|---:|---|---:|---:|']
    for r in summaries:body.append(f'| {r["cell_id"]} | {r["name"]} | {r["epoch"]} | {r["complete"]} | {r.get("fid_cfg1","")} | {r.get("fid_cfg3","")} |')
    body += ['', '## Interpretation constraints','',
        '- Dependence/Jacobian/covariance trajectories test partial decoupling; covariance alone cannot establish independence.',
        '- Learned witness lower bounds must be judged on untouched test draws and their SEs. Positive bounds establish a detected projection of bias; negative bounds are inconclusive.',
        '- Pathwise defect energy is an upper bound on squared conditional-mean bias, not its value.',
        '- Raw DSM and corrected/naive target losses have different noise floors. Use the common valid excess-risk probes for head comparisons at the same encoder.',
        '- Numerical reference errors require bank/particle/ESS resolution; screen failure remains visible in all_reference_audit.csv.',
        '- Compare frozen continuations against BOTH the matched naive continuation and their epoch-zero source at CFG1 and CFG3.',
        '- Three-block controls share head/latent budget; their VAE is larger. Within-depth contrasts isolate topology/pressure more cleanly than depth alone.',
        '- Full checkpointed sources, raw image/query observations, and settings remain under each results directory.']
    body += ['', '## Mechanism outputs','',
        '`mechanism_endpoints.csv` retains the latest observed dependence, pressure, score, witness and reference statistics with their actual epoch and completion status.',
        '`dependence_trajectories.png` compares seed-mean trajectories; bands are SE across available seeds, not confidence intervals from two seeds.',
        '`bias_witness_brackets.png` shows raw signed held-out lower bounds and pathwise upper bounds at common available epochs. Negative plotted lower bounds remain visible.',
        '`frozen_valid_risk_contrasts.csv` uses paired individual-image common-valid-risk differences against the naive continuation and unchanged source. It does not compare different teacher MSE floors.']
    (dest/'report.md').write_text('\n'.join(body)+'\n')
    if len(ev):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for cfg,ax in zip((1.,3.),axes):
            for name,group in ev[(ev.cfg==cfg)&(ev.init=='gaussian')&(ev.num_eval==10000)&(ev.steps==20)].groupby('cell_name'):
                if '_frozen_' in name:continue
                g=group.sort_values('epoch');ax.plot(g.epoch,g.fid_generated,label=name)
            ax.set(title=f'CFG {cfg:g}, fixed 10000-sample protocol',xlabel='Epoch',ylabel='FID')
            if ax.lines:ax.legend(fontsize=5)
        fig.tight_layout();fig.savefig(dest/'fid_trajectories.png',dpi=150);plt.close(fig)
    print(dest/'report.md')
    return 0


def frozen_risk_contrasts(base,dest):
    records=[];rows=experiments.rows()
    for row in rows:
        if row['stage']!='frozen':continue
        path=base/'results'/row['name']/'diagnostics'
        control=next(r for r in rows if r['source']==row['source'] and r['stage']=='frozen' and r['config']['teacher']=='naive')
        ctrlpath=base/'results'/control['name']/'diagnostics'
        current=path/f'epoch_{row["config"]["epochs"]:04d}_probes.npz'
        if not current.exists():continue
        comparisons=[('versus_source',path/'epoch_0000_probes.npz')]
        if row['config']['teacher']!='naive':comparisons.append(('versus_naive_continuation',ctrlpath/f'epoch_{control["config"]["epochs"]:04d}_probes.npz'))
        with np.load(current,allow_pickle=False) as a:
            for comparison,other in comparisons:
                if not other.exists():continue
                meta_a=json.loads(current.with_name(current.name.replace('_probes.npz','.json')).read_text())['settings']
                meta_b=json.loads(other.with_name(other.name.replace('_probes.npz','.json')).read_text())['settings']
                if meta_a.get('test_ids')!=meta_b.get('test_ids'):raise ValueError('Paired risk image IDs differ')
                with np.load(other,allow_pickle=False) as b:
                    for key in a.files:
                        if not key.startswith('score/excess_valid|') or key not in b.files:continue
                        aa,bb=a[key],b[key]
                        if aa.shape!=bb.shape:raise ValueError('Paired risk probe counts differ')
                        delta=aa-bb;parts=key.split('|')
                        records.append(dict(cell_id=row['cell_id'],name=row['name'],seed=row['config']['seed'],comparison=comparison,
                            epoch=row['config']['epochs'],time=parts[1].split('=',1)[1],conditioning=parts[2],head=parts[3],block=int(parts[4].split('=')[1]),
                            mean_delta_valid_risk=float(delta.mean()),se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,n=len(delta)))
    pd.DataFrame(records).to_csv(dest/'frozen_valid_risk_contrasts.csv',index=False)


def mechanism_plots(merged,summary,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=summary[['cell_id','stage','group','blocks','topology','rep_multiplier']]
    diag=merged['diagnostics']
    if len(diag):
        diag=diag.merge(config,on='cell_id')
        diag=diag[(diag.stage=='joint')&(diag.group=='core')]
        specs=[(2,2,'representation/conditional_symkl_per_dim',None,'Two-block conditional dependence'),
               (3,3,'representation/conditional_symkl_per_dim',None,'Three-block leaf dependence'),
               (2,2,'representation/local_parent1_mean_jac_energy',None,'Two-block parent mean sensitivity'),
               (3,2,'score/pathwise_defect_energy',.2,'Three-block middle defect upper bound, t=0.2')]
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for ax,(depth,block,metric,tv,title) in zip(axes.flat,specs):
            frame=diag[(diag.blocks==depth)&(diag.block==block)&(diag.metric==metric)].copy()
            if tv is not None:frame=frame[pd.to_numeric(frame.time,errors='coerce').sub(tv).abs()<1e-10]
            frame['arm']=frame.cell_name.str.replace(r'_s[0-9]+$','',regex=True)
            for name,group in frame.groupby('arm'):
                points=group.groupby('epoch')['mean'].agg(['mean','sem']).sort_index()
                ax.plot(points.index,points['mean'],label=name)
                sem=points['sem'].fillna(0);ax.fill_between(points.index,points['mean']-sem,points['mean']+sem,alpha=.12)
            ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=6)
        fig.suptitle('Seed means; bands are SE across available seeds');fig.tight_layout();fig.savefig(dest/'dependence_trajectories.png',dpi=150);plt.close(fig)
    wt=merged['witness']
    if len(wt):
        wt=wt.merge(config,on='cell_id');wt=wt[(wt.stage=='joint')&(wt.group=='core')&(wt.conditioning=='class')]
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for ax,(depth,pressure) in zip(axes.flat,[(2,1.),(2,4.),(3,1.),(3,4.)]):
            frame=wt[(wt.blocks==depth)&(wt.topology=='full')&(wt.rep_multiplier==pressure)].copy()
            epochs=[set(g.epoch) for _,g in frame.groupby('seed')]
            common=set.intersection(*epochs) if epochs else set()
            endpoint=max(common) if common else None
            frame=frame[frame.epoch==endpoint] if endpoint is not None else frame.iloc[:0]
            for block in range(1,depth):
                for metric,style,label in [('witness/bias_squared_lower','--','lower'),('witness/pathwise_bias_upper','-','upper')]:
                    group=frame[(frame.block==block)&(frame.metric==metric)].copy()
                    group['time_numeric']=pd.to_numeric(group.time,errors='coerce')
                    points=group.groupby('time_numeric')['mean'].mean().sort_index()
                    if len(points):ax.plot(points.index,points.values,style,marker='.',label=f'block {block} {label}')
            when=f'epoch {endpoint}, {len(epochs)} seeds' if endpoint is not None else 'awaiting witness data'
            ax.set(xscale='log',yscale='symlog',title=f'{depth} blocks, pressure {pressure:g}, {when}',xlabel='OU time',ylabel='epsilon energy per coordinate')
            ax.set_yscale('symlog',linthresh=1e-6);ax.axhline(0,color='gray',lw=.5);ax.grid(alpha=.2)
            if ax.lines and len(frame):ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(dest/'bias_witness_brackets.png',dpi=150);plt.close(fig)
