#!/usr/bin/env python3
"""Compile the five-cell CIFAR-10 hierarchical CSEM experiment.

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


EXPECTED_CELLS = (0, 1, 3, 4, 5)
PRIMARY = ('fid', 'rk4', 20, 2.5)
METRIC_RE = re.compile(
    r'^(fid|kid|feature_sw2|sw2)_(rk4|heun)_(\d+)_randtok_cfg'
    r'(\d+_\d+)_temp1_initgaussianT(?!K)([\w.\-]*)$'
)
LABELS = {
    0: '1-block', 1: '2-block', 3: '1-block + B',
    4: '2-block + B', 5: 'Optimized cascade + B',
}
COLORS = {0: '#4c78a8', 1: '#f58518', 3: '#54a24b',
          4: '#e45756', 5: '#b279a2'}
TRAIN_FIELDS = [
    'recon', 'perc', 'score_lsi', 'score_mse_weighted', 'score_mse_unweighted',
    'terminal_kl', 'kl', 'posterior_var', 'posterior_var_median',
    'posterior_std', 'latent_rms', 'score_1', 'score_2', 'kl_1', 'kl_2',
    'var_1', 'var_2', 'band', 'band_penalty', 'band_weight',
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
             'train_last_epoch', 'eval_last_epoch')}
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
                        lw=1.8, marker='o' if len(x[ok]) <= 24 else None, ms=2.8)
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
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ids = list(EXPECTED_CELLS)
    values = [get_num(overview.loc[cid], 'fid_final') for cid in ids]
    bars = ax.bar(np.arange(len(ids)), np.nan_to_num(values, nan=0.0),
                  color=[COLORS[i] for i in ids], alpha=.87)
    for b, val in zip(bars, values):
        if math.isfinite(val):
            ax.text(b.get_x()+b.get_width()/2, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            b.set_alpha(.15)
    ax.set_xticks(range(len(ids)), [LABELS[i] for i in ids], rotation=12, ha='right')
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
    subset = losses[losses.cell_id.isin((1, 4, 5))] if len(losses) else losses
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
        ('child_use_zero_delta', 'v3: zero child (decoded MSE change)'),
        ('child_use_shuffle_delta', 'v3: shuffled child (decoded MSE change)'),
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--strict', action='store_true', help='Require all five cells to have completed 800 epochs')
    args = p.parse_args()
    base = args.base_dir.resolve()
    out = (args.output_dir if args.output_dir else base/'compiled').resolve()
    figures = out/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    manifest_path = base/'manifest.csv'
    if not manifest_path.is_file():
        p.error(f'Missing {manifest_path}. Place this script beside manifest.csv and paper.py.')
    with manifest_path.open(newline='') as f:
        manifest = list(csv.DictReader(f))
    ids = [int(row['cell_id']) for row in manifest]
    if tuple(ids) != EXPECTED_CELLS:
        p.error(f'Expected manifest cell IDs {EXPECTED_CELLS}, got {ids}. Refusing to mix retired cell 2.')

    all_losses, all_evals, all_metrics, summaries, statuses = [], [], [], [], []
    for row in manifest:
        cid = int(row['cell_id'])
        path = base/'results'/row['result_name']/'dataframes'
        suffix = '_v3' if row.get('hierarchy_variant') == 'optimized_v3' else ''
        status_path = base/'status'/f'cell_{cid:03d}{suffix}.json'
        st = json.loads(status_path.read_text()) if status_path.is_file() else {}
        lp, ep = choose_file(path, 'loss_history'), choose_file(path, 'eval_metrics')
        loss, evaluation = read_csv(lp), read_csv(ep)
        if 'stage' in evaluation.columns:
            evaluation = evaluation[evaluation.stage.astype(str) == 'cotrain'].copy()
        if 'tag' in evaluation.columns:
            evaluation = evaluation[evaluation.tag.astype(str) == 'LSI_Diff'].copy()
        if 'stage' in loss.columns:
            loss = loss[loss.stage.astype(str) == 'cotrain'].copy()
        for fr in (loss, evaluation):
            if len(fr) and 'epoch' in fr:
                fr['epoch'] = pd.to_numeric(fr.epoch, errors='coerce')
                fr.dropna(subset=['epoch'], inplace=True)
                fr.sort_values('epoch', inplace=True)
                fr.drop_duplicates('epoch', keep='last', inplace=True)
                fr.reset_index(drop=True, inplace=True)
        if len(loss):
            loss.insert(0, 'cell_id', cid)
            loss.insert(1, 'cell_name', LABELS[cid])
            all_losses.append(loss)
        if len(evaluation):
            evaluation.insert(0, 'cell_id', cid)
            evaluation.insert(1, 'cell_name', LABELS[cid])
            all_evals.append(evaluation)
        metrics = extract_gaussian_metrics(evaluation, cid)
        if len(metrics):
            all_metrics.append(metrics)
        train_epoch = int(loss.epoch.max()) if len(loss) else 0
        eval_epoch = int(evaluation.epoch.max()) if len(evaluation) else 0
        completed = st.get('returncode') == 0 and train_epoch >= 800 and eval_epoch >= 800
        metadata = dict(cell_id=cid, name=LABELS[cid], result_name=row['result_name'],
                        status='complete' if completed else ('error' if st.get('returncode') not in (None, 0) else 'incomplete'),
                        status_returncode=st.get('returncode'),
                        blocks=int(row['blocks']), block_channels=row['block_channels'],
                        score_depths=row['score_depths'], band='yes' if row.get('tmin_band_weight') else 'no',
                        seed=int(row['seed']), epochs=int(row['epochs_joint']),
                        train_last_epoch=train_epoch, eval_last_epoch=eval_epoch)
        summaries.append(model_summary(metadata, evaluation, loss, metrics))
        statuses.append(dict(**metadata, loss_csv=str(lp) if lp.is_file() else '',
                             eval_csv=str(ep) if ep.is_file() else '',
                             error=st.get('error',''), slurm_job_id=st.get('slurm_job_id','')))
        print(f'cell {cid}: {LABELS[cid]} | {metadata["status"]} | train epoch {train_epoch} | eval epoch {eval_epoch}')

    if args.strict and any(s['status'] != 'complete' for s in summaries):
        raise SystemExit('Not all five cells have final status and training/evaluation at epoch 800. Rerun without --strict for partial compilation.')
    overview = pd.DataFrame(summaries).set_index('cell_id', drop=False)
    losses = pd.concat(all_losses, ignore_index=True, sort=False) if all_losses else pd.DataFrame()
    evaluations = pd.concat(all_evals, ignore_index=True, sort=False) if all_evals else pd.DataFrame()
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame(
        columns=['cell_id','epoch','metric','method','steps','cfg','horizon_suffix','value','source_column'])
    overview.reset_index(drop=True).to_csv(out/'leaderboard.csv', index=False)
    pd.DataFrame(statuses).to_csv(out/'run_status.csv', index=False)
    losses.to_csv(out/'all_loss_history.csv', index=False)
    evaluations.to_csv(out/'all_eval_records.csv', index=False)
    metrics.to_csv(out/'gaussian_fullT_metrics_long.csv', index=False)

    primary_curve = select_series(metrics, **dict(zip(('metric','method','steps','cfg'), PRIMARY)))
    if len(primary_curve):
        primary_curve.to_csv(out/'fid_primary_curve.csv', index=False)
        primary_curve.pivot(index='epoch', columns='cell_id', values='value').to_csv(out/'fid_primary_wide.csv')
    else:
        pd.DataFrame().to_csv(out/'fid_primary_curve.csv', index=False)
        pd.DataFrame().to_csv(out/'fid_primary_wide.csv', index=False)

    lines = [
        '# CIFAR hierarchical CSEM: five-cell comparison', '',
        f'Compiled {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.', '',
        '**Primary FID:** full-T Gaussian initialization, temperature=1, RK4-20, CFG=2.5, '
        'legacy CIFAR Inception backend. All experiments have a single seed (42): '
        'do not interpret differences as statistically significant.', '',
        '## Latest/best comparison', '',
        markdown_table(overview.reset_index(drop=True),
            ['cell_id','name','status','eval_last_epoch','fid_final','fid_best','fid_best_epoch','fid_rk425_cfg2p5','fid_rk420_cfg3p0','fid_vae_recon'],
            ['ID','Variant','Run','Last epoch','Final FID (20/2.5)','Best FID','Best epoch','Final RK4-25/2.5','Final RK4-20/3','Recon FID']),
        '', 'Best FID is the minimum among measured epochs of the same sampling protocol; '
        'it is a descriptive minimum, not a held-out model-selection guarantee.', '',
        '## Matched comparisons (final epoch)', '',
    ]
    append_diff(lines, overview, 0, 3, 'Effect of B in flat model')
    append_diff(lines, overview, 1, 4, 'Effect of B in standard two-block model')
    append_diff(lines, overview, 4, 5, 'Optimized cascade vs standard two-block + B')
    append_diff(lines, overview, 0, 1, 'Two blocks vs one block without B')
    append_diff(lines, overview, 3, 4, 'Two blocks vs one block with B')
    lines.extend(['', '## Representation/prior diagnostics (last epoch)', '',
        markdown_table(overview.reset_index(drop=True),
            ['cell_id','name','train_recon','train_terminal_kl','train_score_mse_weighted',
             'train_posterior_var','train_posterior_var_median',
             'fid_encoded_root_generated_children_cfg2_5', 'root_prior_fid_gap'],
            ['ID','Variant','Recon loss','Terminal KL','CSEM MSE','Posterior var mean',
             'Posterior var median','Encoded root + child FID','Full - fixed-root FID']), '',
        'The encoded-root/generated-child comparison is paired in sampling budget, but it '
        'changes the source of the root. A positive full-minus-fixed-root FID is compatible '
        'with a root-prior contribution; it does not establish a unique causal explanation.', '',
        '## Optimized cascade child-usage checks', '',
    ])
    opt = overview.loc[5]
    lines.extend([
        f'- Fixed-posterior root + zero child, MSE change: **{display(opt.get("child_use_zero_delta"),6)}**.',
        f'- Fixed-posterior root + shuffled child, MSE change: **{display(opt.get("child_use_shuffle_delta"),6)}**.',
        f'- Fixed-posterior root + noise child, MSE change: **{display(opt.get("child_use_noise_delta"),6)}**.',
        f'- Original posterior reconstruction MSE: **{display(opt.get("child_use_posterior_mse"),6)}**.',
        '', 'Zero/shuffle probes quantify whether the decoder uses the child. Large changes '
        'are not by themselves evidence that the learned conditional child prior is accurate.',
        '', '## Outputs', '',
        '- `leaderboard.csv`: final and best FID, solver/CFG controls, reconstruction and block diagnostics.',
        '- `gaussian_fullT_metrics_long.csv`: all recorded matched Gaussian/full-T FID, KID, SW2 measurements.',
        '- `fid_primary_curve.csv` and `fid_primary_wide.csv`: per-epoch matched FID.',
        '- `all_loss_history.csv`, `all_eval_records.csv`, `run_status.csv`: complete cross-cell source tables.',
        '- `figures/`: matched FID trajectories, final leaderboard, training/root/child diagnostics.',
        '- `five_cell_review.pdf`: combined illustrated quick-review document.', '',
    ])
    (out/'SUMMARY.md').write_text('\n'.join(lines))

    with PdfPages(out/'five_cell_review.pdf') as pdf:
        plot_fid(figures, pdf, metrics)
        plot_fid_bar(figures, pdf, overview)
        plot_training(figures, pdf, losses)
        plot_block_diagnostics(figures, pdf, losses)
        plot_child(figures, pdf, evaluations)

    print('\n' + markdown_table(overview.reset_index(drop=True),
                                 ['cell_id','name','status','fid_final','fid_best','fid_rk425_cfg2p5','fid_vae_recon'],
                                 ['ID','Model','Status','Final FID','Best FID','RK4-25 FID','Recon FID']))
    print(f'\nCompiled report: {out/"SUMMARY.md"}\nVisual review: {out/"five_cell_review.pdf"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
