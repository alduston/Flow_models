#!/usr/bin/env python3
"""Compile the 20-cell frozen-checkpoint oracle-score NFE mechanism sweep."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0')
MANIFEST = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_manifest.csv'
RESULTS_ROOT = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_results'
STATUS_ROOT = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_status'
OUT_ROOT = 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_compiled'
META = [
    'cell_id','rep_id','rep_role','csem_w','terminal_kl_w','rk4_steps','rk4_nfe',
    'source_results_root','source_result_name','result_name',
]


def stamp(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for col in reversed(META):
        if col in out.columns:
            out = out.drop(columns=[col])
        out.insert(0, col, row[col])
    return out


def read_csv(path: Path):
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f'[warning] failed reading {path}: {exc}', file=sys.stderr)
        return None


def build_mechanism_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ['fid','kid','sw2','diversity']
    group_cols = META
    for keys, g in curves.groupby(group_cols, dropna=False, sort=True):
        base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        direct = g[g['mode'] == 'direct_q0_train']
        for horizon in ('TK','T'):
            h = g[g['horizon_name'].astype(str) == horizon]
            if h.empty:
                continue
            rec = dict(base)
            rec['horizon_name'] = horizon
            rec['horizon_t'] = float(pd.to_numeric(h['horizon_t'], errors='coerce').dropna().iloc[0])
            mode_map = {
                'oracle_qh': h[(h['score_source']=='oracle') & (h['init_mode']=='q_h')],
                'learned_qh': h[(h['score_source']=='learned') & (h['init_mode']=='q_h')],
                'oracle_gaussian': h[(h['score_source']=='oracle') & (h['init_mode']=='gaussian')],
                'learned_gaussian': h[(h['score_source']=='learned') & (h['init_mode']=='gaussian')],
            }
            for label, frame in mode_map.items():
                if len(frame):
                    q = frame.iloc[0]
                    for metric in metrics:
                        rec[f'{metric}_{label}'] = q.get(metric, np.nan)
            if len(direct):
                q = direct.iloc[0]
                for metric in metrics:
                    rec[f'{metric}_direct_q0'] = q.get(metric, np.nan)

            # Descriptive mechanism gaps.  FID/KID are nonlinear distributional
            # metrics, so these differences are diagnostics, not additive identities.
            for metric in metrics:
                oq = rec.get(f'{metric}_oracle_qh', np.nan)
                lq = rec.get(f'{metric}_learned_qh', np.nan)
                og = rec.get(f'{metric}_oracle_gaussian', np.nan)
                lg = rec.get(f'{metric}_learned_gaussian', np.nan)
                dq = rec.get(f'{metric}_direct_q0', np.nan)
                rec[f'{metric}_score_model_gap_qh'] = lq - oq
                rec[f'{metric}_terminal_init_gap_oracle'] = og - oq
                rec[f'{metric}_terminal_init_gap_learned'] = lg - lq
                rec[f'{metric}_full_vs_direct_q0'] = lg - dq
            rows.append(rec)
    return pd.DataFrame(rows)


def build_profile_summary(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in profile.groupby(META, dropna=False, sort=True):
        rec = dict(zip(META, keys if isinstance(keys, tuple) else (keys,)))
        numeric = {
            'oracle_cond_score_rms_mean': ('oracle_cond_score_rms','mean'),
            'oracle_cond_score_rms_max': ('oracle_cond_score_rms','max'),
            'oracle_cond_drift_rms_mean': ('oracle_cond_ode_drift_rms','mean'),
            'oracle_cond_drift_rms_max': ('oracle_cond_ode_drift_rms','max'),
            'oracle_cond_drift_path_rate_mean': ('oracle_cond_drift_path_rate_rms','mean'),
            'oracle_cond_drift_path_rate_max': ('oracle_cond_drift_path_rate_rms','max'),
            'oracle_cond_drift_direction_cosine_mean': ('oracle_cond_drift_direction_cosine','mean'),
            'oracle_cond_drift_direction_cosine_min': ('oracle_cond_drift_direction_cosine','min'),
            'oracle_cond_minus_uncond_score_rms_mean': ('oracle_cond_minus_uncond_score_rms','mean'),
            'learned_oracle_cond_score_error_mean': ('cond_learned_oracle_score','mean'),
            'learned_oracle_guided_score_error_mean': ('guided_learned_oracle_score','mean'),
        }
        for out_name, (col, agg) in numeric.items():
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors='coerce')
            if agg == 'mean': rec[out_name] = float(vals.mean())
            elif agg == 'max': rec[out_name] = float(vals.max())
            elif agg == 'min': rec[out_name] = float(vals.min())
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE_DIR)
    ap.add_argument('--allow-incomplete', action='store_true')
    args = ap.parse_args()

    base = args.base_dir.resolve()
    manifest = pd.read_csv(base / MANIFEST, dtype={'cell_id': str})
    manifest['cell_id'] = manifest['cell_id'].str.zfill(2)
    out = base / OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    statuses, problems = [], []
    curve_frames, profile_frames, eval_frames = [], [], []
    for _, row in manifest.iterrows():
        root = base / RESULTS_ROOT / row['result_name']
        status_path = base / STATUS_ROOT / f"{row['result_name']}.json"
        curve_path = root / 'run_terminal_kl' / 'dataframes' / 'oracle_sampling_decomposition_ep600.csv'
        profile_path = root / 'run_terminal_kl' / 'dataframes' / 'oracle_score_time_profile_ep600.csv'
        eval_path = root / 'combined_dataframes' / 'combined_eval_metrics.csv'

        status = {
            'cell_id': row['cell_id'], 'rep_id': row['rep_id'],
            'rk4_steps': row['rk4_steps'], 'rk4_nfe': row['rk4_nfe'],
            'result_name': row['result_name'],
            'status_exists': status_path.is_file(),
            'curve_csv_exists': curve_path.is_file(),
            'profile_csv_exists': profile_path.is_file(),
            'eval_csv_exists': eval_path.is_file(),
            'returncode': None, 'elapsed_seconds': None,
        }
        if status_path.is_file():
            try:
                d = json.loads(status_path.read_text())
                status['returncode'] = d.get('returncode')
                status['elapsed_seconds'] = d.get('elapsed_seconds')
            except Exception as exc:
                status['status_parse_error'] = repr(exc)
        statuses.append(status)
        ok = status['status_exists'] and status['returncode'] == 0 and status['curve_csv_exists'] and status['profile_csv_exists']
        if not ok:
            problems.append(status)

        c = read_csv(curve_path)
        if c is not None: curve_frames.append(stamp(c, row))
        p = read_csv(profile_path)
        if p is not None: profile_frames.append(stamp(p, row))
        e = read_csv(eval_path)
        if e is not None: eval_frames.append(stamp(e, row))

    status_df = pd.DataFrame(statuses)
    problem_df = pd.DataFrame(problems)
    status_df.to_csv(out / 'run_status.csv', index=False)
    problem_df.to_csv(out / 'missing_or_failed.csv', index=False)

    curves = pd.concat(curve_frames, ignore_index=True, sort=False) if curve_frames else pd.DataFrame()
    profiles = pd.concat(profile_frames, ignore_index=True, sort=False) if profile_frames else pd.DataFrame()
    evals = pd.concat(eval_frames, ignore_index=True, sort=False) if eval_frames else pd.DataFrame()
    if not curves.empty:
        curves.to_csv(out / 'oracle_nfe_curve_long.csv', index=False)
        mech = build_mechanism_table(curves)
        mech.to_csv(out / 'oracle_nfe_mechanism_wide.csv', index=False)
    else:
        mech = pd.DataFrame()
    if not profiles.empty:
        profiles.to_csv(out / 'oracle_field_profile_all.csv', index=False)
        ps = build_profile_summary(profiles)
        ps.to_csv(out / 'oracle_field_profile_summary.csv', index=False)
    if not evals.empty:
        evals.to_csv(out / 'eval_summary_raw.csv', index=False)

    readme = '''CIFAR oracle-score NFE mechanism compilation\n\nPrimary files:\n  oracle_nfe_curve_long.csv\n      one row per direct/oracle/learned sampling mode, horizon and NFE cell.\n  oracle_nfe_mechanism_wide.csv\n      paired mechanism comparison at each representation x NFE x horizon.\n  oracle_field_profile_all.csv\n      time-resolved learned-vs-oracle error plus exact score/drift geometry.\n  oracle_field_profile_summary.csv\n      compact per-representation/NFE profile statistics.\n\nInterpretation:\n  oracle + q_h       = finite-NFE dynamics of the exact conditional score field.\n  learned + q_h      = exact initialization plus learned-score approximation.\n  oracle + Gaussian  = exact score plus terminal initialization mismatch.\n  learned + Gaussian = both practical errors.\n\nThe same frozen checkpoint is reloaded independently at every NFE point.  No\ntraining occurs in this sweep.  FID/KID differences are descriptive and are not\nassumed additive.  Latent SW2 and the exact-field time profiles are the cleaner\nmechanistic diagnostics.\n'''
    (out / 'README.txt').write_text(readme)

    print(f'Compilation output: {out}')
    print(f'Completed curve rows: {len(curves)}')
    print(f'Problems: {len(problem_df)}')
    if len(problem_df) and not args.allow_incomplete:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
