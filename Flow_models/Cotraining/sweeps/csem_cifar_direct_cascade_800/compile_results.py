#!/usr/bin/env python3
"""Compile the eight-cell direct-child CIFAR-10 CSEM sweep.

Copy alongside manifest.csv and paper.py, then run:

    python compile_results.py
    python compile_results.py --base-dir /path/to/csem_cifar_blocks_800

Reads existing results; never trains or changes run statuses/checkpoints.
Writes CSV, a concise SUMMARY.md, and PNG/PDF figures under compiled/.
Requires numpy, pandas and matplotlib (the sweep's existing dependencies).

Primary comparison: FULL-T, Gaussian-start, RK4-20, CFG=2.5, temperature=1.
Never substitutes T_K/oracle-start runs. Also compares RK4-25 and CFG=3.0.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


EXPECTED_CELLS = tuple(range(8))
PRIMARY = ('fid', 'rk4', 20, 2.5)
METRIC_RE = re.compile(
    r'^(fid|kid|feature_sw2|sw2)_(rk4|heun)_(\d+)_randtok_cfg'
    r'(\d+_\d+)_temp1_initgaussianT(?!K)([\w.\-]*)$'
)
LABELS = {0:'Direct baseline s42',1:'Direct baseline s43',2:'Flat s42',3:'Flat s43',
          4:'Balanced 256/256',5:'Fine-heavy 64/448',6:'Same resolution',7:'Child depth 4+8'}
COLORS = {0:'#f58518',1:'#f58518',2:'#4c78a8',3:'#4c78a8',
          4:'#54a24b',5:'#e45756',6:'#b279a2',7:'#9d755d'}
TRAIN_FIELDS = [
    'recon', 'perc', 'score_lsi', 'score_mse_weighted', 'score_mse_unweighted',
    'terminal_kl', 'kl', 'posterior_var', 'posterior_var_median',
    'posterior_std', 'latent_rms', 'score_1', 'score_2', 'kl_1', 'kl_2',
    'var_1', 'var_2', 'band', 'band_penalty', 'band_weight', 'band_objective',
    'vae_clip_rate', 'score_clip_rate', 'vae_grad_preclip', 'score_grad_preclip',
]
EXTRA_EVAL = [
    'fid_vae_recon', 'kid_vae_recon', 'feature_sw2_vae_recon',
    'fid_encoded_root_generated_children_cfg2_5',
    'fid_encoded_root_generated_children_cfg3_0',
    'kid_encoded_root_generated_children_cfg2_5',
    'child_use_posterior_mse', 'child_use_zero_mse',
    'child_use_shuffle_mse', 'child_use_noise_mse',
    'child_use_zero_delta', 'child_use_shuffle_delta', 'child_use_noise_delta',
    'child_generated_draws_mse', 'child_generated_vs_posterior_mse',
    'vae_parameters', 'score_parameters', 'root_coordinates', 'child_coordinates',
    'conditional_csem_gap_score_uncond',
    'csem_gap_score_uncond', 'eval_samples',
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f'[WARN] Cannot read {path}: {exc}')
        return pd.DataFrame()


def asnum(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else float('nan')
    except (ValueError, TypeError):
        return float('nan')


def get_num(row, key):
    return asnum(row.get(key, float('nan')))


def display(value, precision=2):
    v = asnum(value)
    return f'{v:.{precision}f}' if math.isfinite(v) else '—'


def choose_file(directory: Path, name: str):
    final = directory / f'{name}.csv'
    progress = directory / f'{name}_in_progress.csv'
    if final.exists():
        return final
    return progress


def extract_gaussian_metrics(frame: pd.DataFrame, cell: int) -> pd.DataFrame:
    """Long table of only matched FULL-T Gaussian start metrics."""
    valid = []
    for name in frame.columns:
        m = METRIC_RE.fullmatch(str(name))
        if m:
            metric, method, steps, cfg, horizon = m.groups()
            valid.append((name, metric, method, int(steps), float(cfg.replace('_', '.')), horizon))
    out = []
    for _, row in frame.iterrows():
        ep = get_num(row, 'epoch')
        for name, metric, method, steps, cfg, horizon in valid:
            value = get_num(row, name)
            if math.isfinite(value) and math.isfinite(ep):
                out.append({'cell_id': cell, 'epoch': int(ep), 'metric': metric,
                            'method': method, 'steps': steps, 'cfg': cfg,
                            'horizon_suffix': horizon, 'value': value,
                            'source_column': name})
    return pd.DataFrame(out, columns=['cell_id', 'epoch', 'metric', 'method',
                                      'steps', 'cfg', 'horizon_suffix', 'value', 'source_column'])


def select_series(metrics, *, metric='fid', method='rk4', steps=20, cfg=2.5):
    if metrics.empty:
        return metrics.copy()
    z = metrics[(metrics.metric == metric) & (metrics.method == method) &
                (metrics.steps == steps) & np.isclose(metrics.cfg, cfg)].copy()
    # More than one full-T horizon suffix in the same cell/epoch is ambiguous;
    # do not silently choose a value from a different experiment.
    dup = z.groupby(['cell_id', 'epoch']).size()
    if (dup > 1).any():
        raise ValueError(f'Ambiguous Gaussian full-T columns: {dup[dup > 1].to_dict()}')
    return z.sort_values(['cell_id', 'epoch'])


def last_epoch(frame):
    if frame.empty or 'epoch' not in frame.columns:
        return pd.Series(dtype=object)
    a = frame.copy()
    a['epoch'] = pd.to_numeric(a['epoch'], errors='coerce')
    a = a.dropna(subset=['epoch']).sort_values('epoch')
    return a.iloc[-1] if len(a) else pd.Series(dtype=object)


def model_summary(metadata, eval_df, loss_df, metrics):
    cell = metadata['cell_id']
    main = select_series(metrics, **dict(zip(('metric','method','steps','cfg'), PRIMARY)))
    latest_eval = last_epoch(eval_df)
    latest_loss = last_epoch(loss_df)
    series = main[main.cell_id == cell]
    full = {k: v for k, v in metadata.items() if k in
            ('cell_id', 'name', 'result_name', 'status', 'status_returncode',
             'band', 'blocks', 'block_channels', 'score_depths', 'seed', 'epochs',
             'train_last_epoch', 'eval_last_epoch', 'variant', 'block_coordinates')}
    if len(series):
        final = series.iloc[-1]
        best = series.loc[series.value.idxmin()]
        full.update(fid_final=float(final.value), fid_final_epoch=int(final.epoch),
                    fid_best=float(best.value), fid_best_epoch=int(best.epoch),
                    fid_best_to_final=float(final.value - best.value))
    for method, steps, cfg in [('rk4',20,3.0), ('rk4',25,2.5), ('rk4',25,3.0),
                               ('heun',20,2.5), ('heun',20,3.0)]:
        k = f'fid_{method}{steps}_cfg{str(cfg).replace(".", "p")}'
        s = select_series(metrics, method=method, steps=steps, cfg=cfg)
        s = s[s.cell_id == cell] if len(s) else s
        full[k] = float(s.iloc[-1].value) if len(s) else float('nan')
    for m in ('kid', 'feature_sw2', 'sw2'):
        s = select_series(metrics, metric=m, method='rk4', steps=20, cfg=2.5)
        s = s[s.cell_id == cell] if len(s) else s
        full[m+'_rk4_20_cfg2p5'] = float(s.iloc[-1].value) if len(s) else float('nan')
    for key in EXTRA_EVAL:
        full[key] = get_num(latest_eval, key)
    for key in TRAIN_FIELDS:
        full['train_'+key] = get_num(latest_loss, key)
    if math.isfinite(asnum(full.get('fid_final'))) and math.isfinite(asnum(full.get('fid_vae_recon'))):
        full['generated_minus_reconstruction_fid'] = full['fid_final'] - full['fid_vae_recon']
    else:
        full['generated_minus_reconstruction_fid'] = float('nan')
    if math.isfinite(asnum(full.get('fid_encoded_root_generated_children_cfg2_5'))) and math.isfinite(asnum(full.get('fid_final'))):
        full['root_prior_fid_gap'] = full['fid_final'] - full['fid_encoded_root_generated_children_cfg2_5']
    else:
        full['root_prior_fid_gap'] = float('nan')
    return full


def save_plot(fig, name, figures, pdf):
    fig.savefig(figures / f'{name}.png', dpi=140, bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_by_cell(ax, frame, col, *, ylabel, title=None, positive=False, legend=True):
    used = 0
    if not frame.empty and col in frame:
        for cid in EXPECTED_CELLS:
            part = frame[frame.cell_id == cid].sort_values('epoch')
            if not len(part):
                continue
            x = pd.to_numeric(part.epoch, errors='coerce').to_numpy(dtype=float)
            y = pd.to_numeric(part[col], errors='coerce').to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if np.any(ok):
                ax.plot(x[ok], y[ok], label=LABELS[cid], color=COLORS[cid],
                        lw=1.8, linestyle='--' if cid in (1,3) else '-', marker='o' if len(x[ok]) <= 24 else None, ms=2.8)
                used += 1
    if title:
        ax.set_title(title)
    ax.set_xlabel('Cotraining epoch')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.22)
    if positive:
        ymin,ymax = ax.get_ylim()
        if ymin >= 0:
            ax.set_ylim(bottom=0)
    if legend and used:
        ax.legend(fontsize=8)
    if not used:
        ax.text(.5, .5, 'No recorded metric', ha='center', va='center', transform=ax.transAxes)
    return used


def plot_fid(figures, pdf, metrics):
    choices = [('rk4', 20, 2.5), ('rk4', 25, 2.5),
               ('rk4', 20, 3.0), ('heun', 20, 2.5)]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for ax, (solver, n, cfg) in zip(axs.flat, choices):
        z = select_series(metrics, metric='fid', method=solver, steps=n, cfg=cfg)
        if len(z):
            wide = z.pivot(index='epoch', columns='cell_id', values='value').reset_index()
            wide = wide.melt(id_vars='epoch', var_name='cell_id', value_name='fid')
        else:
            wide = pd.DataFrame()
        plot_by_cell(ax, wide, 'fid', ylabel='FID ↓', title=f'{solver.upper()}-{n}, CFG={cfg:g}')
    fig.suptitle('CIFAR-10: generated FID, full-T Gaussian start (no oracle)', weight='bold', fontsize=14)
    save_plot(fig, '01_generated_fid_comparison', figures, pdf)


def plot_fid_bar(figures, pdf, overview):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ids = list(EXPECTED_CELLS)
    values = [get_num(overview.loc[cid], 'fid_final') for cid in ids]
    bars = ax.bar(np.arange(len(ids)), np.nan_to_num(values, nan=0.0),
                  color=[COLORS[i] for i in ids], alpha=.87)
    for b, val in zip(bars, values):
        if math.isfinite(val):
            ax.text(b.get_x()+b.get_width()/2, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            b.set_alpha(.15)
    ax.set_xticks(range(len(ids)), [LABELS[i] for i in ids], rotation=25, ha='right')
    ax.set_ylabel('Final FID ↓')
    ax.set_title('800-epoch final FID: full-T Gaussian, RK4-20, CFG=2.5')
    ax.grid(axis='y', alpha=.22)
    fig.tight_layout()
    save_plot(fig, '02_final_fid_leaderboard', figures, pdf)


def plot_training(figures, pdf, losses):
    options = [('recon', 'Reconstruction loss'), ('score_mse_weighted', 'Weighted CSEM score MSE'),
               ('terminal_kl', 'Terminal KL loss'), ('posterior_var_median', 'Posterior variance median'),
               ('posterior_var', 'Posterior variance mean'), ('latent_rms', 'Latent mean RMS'),
               ('vae_clip_rate', 'VAE gradient clipping rate'),
               ('perc', 'Perceptual loss')]
    fig, axs = plt.subplots(4, 2, figsize=(12.5, 12), constrained_layout=True)
    for ax, (key, label) in zip(axs.flat, options):
        plot_by_cell(ax, losses, key, ylabel=label, title=label, legend=False)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, -.015), fontsize=9)
    fig.suptitle('CIFAR-10: training and representation diagnostics', weight='bold', fontsize=13)
    save_plot(fig, '03_training_diagnostics', figures, pdf)


def plot_block_diagnostics(figures, pdf, losses):
    options = [('score_1', 'Root score loss'), ('score_2', 'Child score loss'),
               ('kl_1', 'Root terminal KL'), ('kl_2', 'Child terminal KL'),
               ('var_1', 'Root posterior variance'), ('var_2', 'Child posterior variance')]
    fig, axs = plt.subplots(3, 2, figsize=(12.0, 10.3), constrained_layout=True)
    subset = losses[losses.cell_id.isin((0, 1, 4, 5, 6, 7))] if len(losses) else losses
    for ax, (key, label) in zip(axs.flat, options):
        plot_by_cell(ax, subset, key, ylabel=label, title=label, legend=False)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, -.005), fontsize=9)
    fig.suptitle('Two-block models: conditional root/child diagnostics', weight='bold', fontsize=13)
    save_plot(fig, '04_root_child_diagnostics', figures, pdf)


def plot_child(figures, pdf, evaluations):
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    choices = [
        ('fid_vae_recon', 'Posterior reconstruction FID'),
        ('fid_encoded_root_generated_children_cfg2_5', 'Encoded root + generated child FID (CFG=2.5)'),
        ('child_use_zero_delta', 'Direct Z2: zero child (decoded MSE change)'),
        ('child_use_shuffle_delta', 'Direct Z2: shuffled child (decoded MSE change)'),
    ]
    for ax, (key, label) in zip(axs.flat, choices):
        plot_by_cell(ax, evaluations, key, ylabel=label, title=label, legend=False)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, -.015), fontsize=9)
    fig.suptitle('Posterior/prior mismatch and cascade child-usage probes', weight='bold', fontsize=13)
    save_plot(fig, '05_hierarchy_generation_diagnostics', figures, pdf)


def markdown_table(frame, cols, headers=None, digits=2):
    headers = headers or cols
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |']
    for _, r in frame.iterrows():
        entries = []
        for key in cols:
            v = r.get(key)
            if key in ('name', 'status'):
                entries.append(str(v).replace('|','/'))
            elif key in ('cell_id', 'fid_final_epoch', 'fid_best_epoch', 'eval_last_epoch'):
                entries.append(str(int(v)) if math.isfinite(asnum(v)) else '—')
            else:
                entries.append(display(v, digits))
        lines.append('| ' + ' | '.join(entries) + ' |')
    return '\n'.join(lines)


def append_diff(lines, table, a, b, title):
    va = get_num(table.loc[a], 'fid_final')
    vb = get_num(table.loc[b], 'fid_final')
    if math.isfinite(va) and math.isfinite(vb):
        lines.append(f'- **{title}:** {vb-va:+.3f} FID ({LABELS[a]} {va:.3f} → {LABELS[b]} {vb:.3f}; negative is better).')
    else:
        lines.append(f'- **{title}:** unavailable (one or both matched FIDs missing).')



def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--strict',action='store_true',help='Require all eight completed 800-epoch runs')
    args=p.parse_args();base=args.base_dir.resolve()
    out=(args.output_dir if args.output_dir else base/'compiled').resolve()
    figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    from paper import manifest,validate_evaluation
    rows=manifest(base)
    losses_all=[];evals_all=[];metrics_all=[];summaries=[];statuses=[]
    for row in rows:
        cid=int(row['cell_id']);path=base/'results'/row['result_name']/'dataframes'
        sp=base/'status'/f'cell_{cid:03d}.json'
        st=json.loads(sp.read_text()) if sp.is_file() else {}
        lp=choose_file(path,'loss_history');ep=choose_file(path,'eval_metrics')
        loss=read_csv(lp);evaluation=read_csv(ep)
        for frame in (loss,evaluation):
            if len(frame) and 'stage' in frame:
                frame.drop(frame[frame.stage.astype(str)!='cotrain'].index,inplace=True)
        if len(evaluation) and 'tag' in evaluation:
            evaluation=evaluation[evaluation.tag.astype(str)=='LSI_Diff'].copy()
        for frame in (loss,evaluation):
            if len(frame):
                frame['epoch']=pd.to_numeric(frame.epoch,errors='raise')
                frame.sort_values('epoch',inplace=True);frame.drop_duplicates('epoch',keep='last',inplace=True)
                frame.reset_index(drop=True,inplace=True)
        for frame,dest in ((loss,losses_all),(evaluation,evals_all)):
            if len(frame):
                frame.insert(0,'cell_id',cid);frame.insert(1,'cell_name',LABELS[cid]);dest.append(frame)
        metrics=extract_gaussian_metrics(evaluation,cid)
        if len(metrics):metrics_all.append(metrics)
        train_ep=int(loss.epoch.max()) if len(loss) else 0
        eval_ep=int(evaluation.epoch.max()) if len(evaluation) else 0
        complete=False;error=st.get('error','')
        if st.get('returncode')==0 and train_ep==800 and eval_ep==800:
            try:validate_evaluation(row,evaluation.to_dict('records'));complete=True
            except ValueError as exc:error=str(exc)
        meta=dict(cell_id=cid,name=LABELS[cid],variant=row['variant'],result_name=row['result_name'],
                  status='complete' if complete else ('error' if st.get('returncode') not in (None,0) else 'incomplete'),
                  status_returncode=st.get('returncode'),blocks=int(row['blocks']),
                  block_channels=row['block_channels'],block_coordinates=row['block_coordinates'],
                  score_depths=row['score_depths'],band='yes',seed=int(row['seed']),epochs=800,
                  train_last_epoch=train_ep,eval_last_epoch=eval_ep)
        item=model_summary(meta,evaluation,loss,metrics)
        for key in ('fid_final','fid_final_epoch','fid_best','fid_best_epoch'):
            item.setdefault(key,float('nan'))
        item['fid_latest']=item.get('fid_final',float('nan'))
        item['fid_latest_epoch']=item.get('fid_final_epoch',float('nan'))
        if item.get('fid_final_epoch')!=800:
            item['fid_final']=float('nan');item['fid_final_epoch']=float('nan')
        if len(loss) and int(row['blocks'])==2:
            latest=last_epoch(loss);w=int(row['block_coordinates'].split(':')[0])/512
            k1=get_num(latest,'kl_1');k2=get_num(latest,'kl_2')
            item['root_terminal_kl_share']=w*k1/(w*k1+(1-w)*k2) if w*k1+(1-w)*k2>0 else float('nan')
        else:item['root_terminal_kl_share']=float('nan')
        final_eval=evaluation[evaluation.epoch==800] if len(evaluation) else pd.DataFrame()
        timing='sampling_seconds_rk4_20_randtok_cfg2_5_temp1_initgaussianT1p35'
        item['sampling_seconds_primary']=get_num(last_epoch(final_eval),timing)
        item['sampling_ms_per_image']=item['sampling_seconds_primary']/10
        summaries.append(item);statuses.append(dict(**meta,error=error,slurm_job_id=st.get('slurm_job_id','')))
        print(f'cell {cid}: {LABELS[cid]} | {meta["status"]} | train {train_ep} | eval {eval_ep}')
    if args.strict and any(s['status']!='complete' for s in summaries):
        raise SystemExit('All eight cells must have completed 800 epochs and all scheduled evaluations')
    overview=pd.DataFrame(summaries).set_index('cell_id',drop=False)
    loss=pd.concat(losses_all,ignore_index=True,sort=False) if losses_all else pd.DataFrame()
    evaluation=pd.concat(evals_all,ignore_index=True,sort=False) if evals_all else pd.DataFrame()
    metrics=pd.concat(metrics_all,ignore_index=True) if metrics_all else pd.DataFrame(
        columns=['cell_id','epoch','metric','method','steps','cfg','horizon_suffix','value','source_column'])
    overview.reset_index(drop=True).to_csv(out/'leaderboard.csv',index=False)
    pd.DataFrame(statuses).to_csv(out/'run_status.csv',index=False)
    loss.to_csv(out/'all_loss_history.csv',index=False);evaluation.to_csv(out/'all_eval_records.csv',index=False)
    metrics.to_csv(out/'gaussian_fullT_metrics_long.csv',index=False)
    curve=select_series(metrics,**dict(zip(('metric','method','steps','cfg'),PRIMARY)))
    curve.to_csv(out/'fid_primary_curve.csv',index=False)
    curve.pivot(index='epoch',columns='cell_id',values='value').to_csv(out/'fid_primary_wide.csv')
    pairs=[]
    for direct,flat,seed in ((0,2,42),(1,3,43)):
        a=curve[curve.cell_id==direct][['epoch','value']]
        b=curve[curve.cell_id==flat][['epoch','value']]
        for _,v in a.merge(b,on='epoch',suffixes=('_cascade','_flat')).iterrows():
            pairs.append(dict(seed=seed,epoch=int(v.epoch),fid_direct=v.value_cascade,fid_flat=v.value_flat,
                              direct_minus_flat=v.value_cascade-v.value_flat))
    paired=pd.DataFrame(pairs,columns=['seed','epoch','fid_direct','fid_flat','direct_minus_flat'])
    paired.to_csv(out/'paired_seed_contrasts.csv',index=False)
    contrasts=[dict(cell_id=i,contrast=LABELS[i]+' vs direct baseline s42',
                    delta_fid=get_num(overview.loc[i],'fid_final')-get_num(overview.loc[0],'fid_final')) for i in range(4,8)]
    pd.DataFrame(contrasts).to_csv(out/'architecture_contrasts.csv',index=False)
    aggregate=overview.reset_index(drop=True).groupby('variant').agg(
        seed_count=('seed','count'),finite_endpoints=('fid_final','count'),
        fid_mean=('fid_final','mean'),fid_std=('fid_final','std')).reset_index()
    aggregate.to_csv(out/'architecture_seed_summary.csv',index=False)
    lines=['# CIFAR direct-child CSEM: eight-cell comparison','',
           f'Compiled {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.','',
           'Primary FID: full-T Gaussian initialization, temperature 1, RK4-20 per block, CFG 2.5, '
           '10000 test images and the original legacy CIFAR Inception backend. No oracle-start or T_K values substitute for it.','',
           '## Fixed endpoint and best observed checkpoints','',
           markdown_table(overview.reset_index(drop=True),['cell_id','name','status','eval_last_epoch','fid_final','fid_best','fid_best_epoch','fid_vae_recon'],
                          ['ID','Variant','Status','Last eval','FID at 800','Best FID','Best epoch','Latest recon FID']), '',
           'Missing epoch-800 results remain blank. Latest partial values are separately recorded as fid_latest. '
           'Best observed FID is a descriptive minimum over checkpoints, not a held-out model-selection guarantee.','',
           '## Paired cascade effect','']
    endpoint=paired[paired.epoch==800]
    for _,r in endpoint.iterrows():
        lines.append(f'- Seed {int(r.seed)}: direct {r.fid_direct:.3f}, flat {r.fid_flat:.3f}, Δ={r.direct_minus_flat:+.3f} FID.')
    if len(endpoint)==2:
        lines.append(f'- Mean paired effect: {endpoint.direct_minus_flat.mean():+.3f} FID. Negative favors direct cascade.')
    elif endpoint.empty:lines.append('No completed paired endpoints yet.')
    lines.extend(['','Two seeds check repeatability; they do not establish precise statistical significance. '
                  'The four allocation/resolution/depth variants have one seed each.','',
                  '## Architecture contrasts against the direct baseline, seed 42',''])
    for item in contrasts:lines.append(f'- {item["contrast"]}: {display(item["delta_fid"],3)} FID at epoch 800 (negative is better).')
    lines.extend(['','## Block and sampling diagnostics','',
         markdown_table(overview.reset_index(drop=True),['cell_id','train_score_1','train_score_2','train_var_1','train_var_2','root_terminal_kl_share','train_band_objective','sampling_ms_per_image'],
                        ['ID','Root CSEM','Child CSEM','Root variance','Child variance','Root fraction of terminal KL','Weighted B','Primary ms/image']), '',
         'Terminal KL shares and training CSEM values are diagnostics, not blockwise mutual-information estimates. '
         'All cascade coordinates are direct stochastic Z2: no decoder-only affine child map. Root context for the child '
         'score/posterior is clean; reconstruction recomputes root features from OU-corrupted root and child.','',
         'The flat model has 12 score trunk layers and the baseline has 6+6. Cross-attention and the multiscale VAE '
         'change parameter count and FLOPs. Two serial RK4-20 solves cost more than a single RK4-20 solve; use timing '
         'alongside FID. CFG 3, RK4-25 and Heun-20 curves are retained as the original sampling controls.','',
         '## Outputs','',
         '- leaderboard.csv, run_status.csv, architecture_seed_summary.csv.',
         '- all_loss_history.csv, all_eval_records.csv, gaussian_fullT_metrics_long.csv.',
         '- fid_primary_curve.csv, fid_primary_wide.csv, paired_seed_contrasts.csv, architecture_contrasts.csv.',
         '- figures/ and eight_cell_review.pdf: six-page illustrated comparison.',''])
    (out/'SUMMARY.md').write_text('\n'.join(lines))
    with PdfPages(out/'eight_cell_review.pdf') as pdf:
        plot_fid(figures,pdf,metrics);plot_fid_bar(figures,pdf,overview)
        plot_training(figures,pdf,loss);plot_block_diagnostics(figures,pdf,loss);plot_child(figures,pdf,evaluation)
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),constrained_layout=True)
        if len(paired):
            for seed,part in paired.groupby('seed'):
                axes[0].plot(part.epoch,part.direct_minus_flat,marker='o',label=f'Seed {seed}')
            axes[0].legend()
        axes[0].axhline(0,color='black',lw=.8)
        axes[0].set(xlabel='Epoch',ylabel='Direct cascade FID − flat FID',title='Paired baseline comparisons')
        for cid,item in overview.iterrows():
            if math.isfinite(get_num(item,'fid_final')) and math.isfinite(get_num(item,'sampling_ms_per_image')):
                axes[1].scatter(item.sampling_ms_per_image,item.fid_final,color=COLORS[cid])
                axes[1].annotate(LABELS[cid],(item.sampling_ms_per_image,item.fid_final),fontsize=8)
        axes[1].set(xlabel='Primary generation + decode ms/image',ylabel='FID at epoch 800',title='Measured cascade quality and latency')
        for ax in axes:ax.grid(alpha=.2)
        save_plot(fig,'06_paired_effect_and_timing',figures,pdf)
    print('\n'+markdown_table(overview.reset_index(drop=True),['cell_id','name','status','fid_final','fid_best']))
    print(f'\nCompiled: {out/"SUMMARY.md"}; visual review: {out/"eight_cell_review.pdf"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
