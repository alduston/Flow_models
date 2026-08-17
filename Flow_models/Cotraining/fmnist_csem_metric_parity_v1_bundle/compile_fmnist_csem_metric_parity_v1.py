#!/usr/bin/env python3
"""Compile fmnist_csem_metric_parity_v1 and rank canonical cells by joint reconstruction/terminal parity."""

from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep")
MANIFEST = "fmnist_csem_metric_parity_v1_manifest.csv"
RESULTS_ROOT = "fmnist_csem_metric_parity_v1_results"
STATUS_ROOT = "fmnist_csem_metric_parity_v1_status"
OUT_ROOT = "fmnist_csem_metric_parity_v1_compiled"

META = [
    "config_id","bundle_id","slot_in_bundle","role",
    "outer_time_weighting","head_time_weighting",
    "csem_w","terminal_kl_w","lr_score_head","score_head_loss_w",
    "T_terminal","epochs","refine_epochs","eval_every","eval_samples",
    "cfg_strength","oracle_profile_query_samples","oracle_profile_time_points",
    "oracle_profile_batch_size","oracle_reference_batch_size",
    "oracle_sampling_samples","oracle_sampling_batch_size",
    "oracle_sampling_steps","result_name",
]

FID_HEUN = "fid_heun_50_randtok_cfg3_0"
FID_RK4 = "fid_rk4_25_randtok_cfg3_0"
KID_HEUN = "kid_heun_50_randtok_cfg3_0"
KID_RK4 = "kid_rk4_25_randtok_cfg3_0"
SW2_HEUN = "sw2_heun_50_randtok_cfg3_0"
SW2_RK4 = "sw2_rk4_25_randtok_cfg3_0"


def read_csv(path):
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warning] failed reading {path}: {exc}", file=sys.stderr)
        return None


def stamp(df, row):
    out = df.copy()
    for col in reversed(META):
        if col in out.columns:
            out = out.drop(columns=[col])
        out.insert(0, col, row[col])
    return out


def status_record(row, path):
    rec = {c: row[c] for c in META}
    rec.update(status_file_exists=path.is_file(), returncode=None,
               elapsed_seconds=None, status_parse_error="")
    if path.is_file():
        try:
            d = json.loads(path.read_text())
            rec["returncode"] = d.get("returncode")
            rec["elapsed_seconds"] = d.get("elapsed_seconds")
        except Exception as exc:
            rec["status_parse_error"] = repr(exc)
    return rec


def safe_logratio(x, ref):
    try:
        x, ref = float(x), float(ref)
        if x <= 0 or ref <= 0 or not np.isfinite(x) or not np.isfinite(ref):
            return np.nan
        return math.log(x/ref)
    except Exception:
        return np.nan


def build_ranking(endpoint):
    ref = endpoint[endpoint["role"] == "unweighted_reference"]
    if len(ref) != 1:
        raise RuntimeError(f"Expected exactly one unweighted reference endpoint row; found {len(ref)}")
    ref = ref.iloc[0]
    cand = endpoint[endpoint["role"] == "canonical_grid"].copy()

    # Reference metrics.
    recon_ref = float(ref["fid_vae_recon"])
    kl_ref = float(ref["terminal_component_kl_fulltrain"])
    tsw2_ref = float(ref["terminal_qT_vs_gaussian_sw2"])

    cand["reference_config_id"] = ref["config_id"]
    cand["reference_recon_fid"] = recon_ref
    cand["reference_terminal_kl"] = kl_ref
    cand["reference_terminal_sw2"] = tsw2_ref

    cand["recon_fid_absdiff"] = (cand["fid_vae_recon"] - recon_ref).abs()
    cand["terminal_kl_ratio"] = cand["terminal_component_kl_fulltrain"] / kl_ref
    cand["terminal_sw2_ratio"] = cand["terminal_qT_vs_gaussian_sw2"] / tsw2_ref
    cand["terminal_kl_abs_logratio"] = cand["terminal_component_kl_fulltrain"].apply(
        lambda x: abs(safe_logratio(x, kl_ref))
    )
    cand["terminal_sw2_abs_logratio"] = cand["terminal_qT_vs_gaussian_sw2"].apply(
        lambda x: abs(safe_logratio(x, tsw2_ref))
    )

    # Default parity tolerances are explicit and easy to change.
    recon_tol = 0.50
    rel_tol = 1.25
    logtol = math.log(rel_tol)
    cand["parity_score"] = np.sqrt(
        (cand["recon_fid_absdiff"]/recon_tol)**2
        + (cand["terminal_kl_abs_logratio"]/logtol)**2
        + (cand["terminal_sw2_abs_logratio"]/logtol)**2
    )
    cand["inside_parity_box"] = (
        (cand["recon_fid_absdiff"] <= recon_tol)
        & (cand["terminal_kl_ratio"].between(1/rel_tol, rel_tol))
        & (cand["terminal_sw2_ratio"].between(1/rel_tol, rel_tol))
    )

    # Distributional generation summaries.
    cand["mean_gen_fid"] = 0.5*(cand[FID_HEUN] + cand[FID_RK4])
    cand["mean_gen_kid"] = 0.5*(cand[KID_HEUN] + cand[KID_RK4])
    cand["mean_gen_sw2"] = 0.5*(cand[SW2_HEUN] + cand[SW2_RK4])
    cand["gen_minus_recon_fid"] = cand["mean_gen_fid"] - cand["fid_vae_recon"]

    ref_mean_fid = 0.5*(float(ref[FID_HEUN]) + float(ref[FID_RK4]))
    ref_mean_kid = 0.5*(float(ref[KID_HEUN]) + float(ref[KID_RK4]))
    ref_mean_sw2 = 0.5*(float(ref[SW2_HEUN]) + float(ref[SW2_RK4]))
    cand["reference_mean_gen_fid"] = ref_mean_fid
    cand["reference_mean_gen_kid"] = ref_mean_kid
    cand["reference_mean_gen_sw2"] = ref_mean_sw2
    cand["mean_gen_fid_CminusU"] = cand["mean_gen_fid"] - ref_mean_fid
    cand["mean_gen_kid_CminusU"] = cand["mean_gen_kid"] - ref_mean_kid
    cand["mean_gen_sw2_CminusU"] = cand["mean_gen_sw2"] - ref_mean_sw2

    # Oracle transport attribution.
    oq = "oracle_transport_fid_oracle_qT_rk4_ode_25"
    og = "oracle_transport_fid_oracle_gaussian_rk4_ode_25"
    lg = "oracle_transport_fid_learned_gaussian_uncond_rk4_ode_25"
    cand["oracle_terminal_init_penalty_fid"] = cand[og] - cand[oq]
    cand["learned_vs_oracle_gaussian_fid_delta"] = cand[lg] - cand[og]
    cand["reference_oracle_terminal_init_penalty_fid"] = float(ref[og] - ref[oq])

    # Score-estimation ratios relative to the unweighted reference.
    score_cols = [
        "oracle_profile_uncond_learned_oracle_score_logtime_mean",
        "oracle_profile_uncond_learned_oracle_score_physical_mean",
        "oracle_profile_uncond_intrinsic_var_score_logtime_mean",
        "oracle_profile_cond_learned_oracle_score_logtime_mean",
        "oracle_profile_guided_learned_oracle_score_logtime_mean",
        "oracle_profile_guided_learned_oracle_score_physical_mean",
    ]
    for col in score_cols:
        if col in cand and col in ref.index:
            rv = float(ref[col])
            cand[f"ratio_to_reference__{col}"] = cand[col] / rv

    priority = [
        "config_id","csem_w","terminal_kl_w","parity_score","inside_parity_box",
        "fid_vae_recon","reference_recon_fid","recon_fid_absdiff",
        "terminal_component_kl_fulltrain","reference_terminal_kl","terminal_kl_ratio",
        "terminal_qT_vs_gaussian_sw2","reference_terminal_sw2","terminal_sw2_ratio",
        "mean_gen_fid","reference_mean_gen_fid","mean_gen_fid_CminusU",
        "mean_gen_kid","mean_gen_kid_CminusU",
        "mean_gen_sw2","mean_gen_sw2_CminusU",
        "gen_minus_recon_fid",
        "oracle_terminal_init_penalty_fid",
        "reference_oracle_terminal_init_penalty_fid",
        "learned_vs_oracle_gaussian_fid_delta",
    ]
    rest = [c for c in cand.columns if c not in priority]
    return cand[priority + rest].sort_values(
        ["inside_parity_box","parity_score"],
        ascending=[False, True],
    ).reset_index(drop=True), ref


def build_time_deltas(profiles, ranking):
    if profiles.empty or ranking.empty:
        return pd.DataFrame()
    # Compare every canonical profile to the single reference profile at the same t_index.
    ref = profiles[profiles["role"] == "unweighted_reference"].copy()
    can = profiles[profiles["role"] == "canonical_grid"].copy()
    if ref.empty or can.empty:
        return pd.DataFrame()
    ref_cols = {
        c: f"{c}_U"
        for c in ref.columns
        if c not in META and c not in ("t_index",)
    }
    ref2 = ref.rename(columns=ref_cols)
    out = can.merge(
        ref2,
        left_on="t_index", right_on="t_index",
        how="left", suffixes=("", "_Umeta")
    )
    metrics = [
        "uncond_learned_oracle_score",
        "uncond_intrinsic_var_score",
        "cond_learned_oracle_score",
        "guided_learned_oracle_score",
    ]
    for m in metrics:
        u = f"{m}_U"
        if m in out and u in out:
            out[f"{m}_CminusU"] = out[m] - out[u]
            out[f"{m}_CoverU"] = out[m] / out[u].replace(0, np.nan)
    if "parity_score" in ranking:
        out = out.merge(
            ranking[["config_id","parity_score","inside_parity_box"]],
            on="config_id", how="left"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    out = base/OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(base/MANIFEST, dtype={"config_id":str,"bundle_id":str})
    manifest["config_id"] = manifest["config_id"].str.zfill(3)
    manifest["bundle_id"] = manifest["bundle_id"].str.zfill(2)
    if len(manifest) != 17:
        raise SystemExit(f"Expected 17 manifest rows, found {len(manifest)}")

    statuses, problems = [], []
    evals, losses, profiles, samplings = [], [], [], []

    for _, row in manifest.iterrows():
        root = base/RESULTS_ROOT/str(row["result_name"])
        sp = base/STATUS_ROOT/f"{row['result_name']}.json"
        ep = root/"combined_dataframes"/"combined_eval_metrics.csv"
        lp = root/"combined_dataframes"/"combined_loss_history.csv"
        pp = sorted(root.glob("**/oracle_score_time_profile_ep*.csv"))
        op = sorted(root.glob("**/oracle_sampling_decomposition_ep*.csv"))

        sr = status_record(row, sp)
        sr.update(
            result_dir_exists=root.is_dir(),
            eval_csv_exists=ep.is_file(),
            loss_csv_exists=lp.is_file(),
            oracle_profile_files=len(pp),
            oracle_sampling_files=len(op),
        )
        statuses.append(sr)
        ok = (
            sr["status_file_exists"] and sr["returncode"] == 0
            and sr["eval_csv_exists"] and sr["loss_csv_exists"]
            and len(pp) >= 1 and len(op) >= 1
            and not sr["status_parse_error"]
        )
        if not ok:
            problems.append(sr)

        d = read_csv(ep)
        if d is not None: evals.append(stamp(d,row))
        d = read_csv(lp)
        if d is not None: losses.append(stamp(d,row))
        for p in pp:
            d = read_csv(p)
            if d is not None: profiles.append(stamp(d,row))
        for p in op:
            d = read_csv(p)
            if d is not None: samplings.append(stamp(d,row))

    status_df = pd.DataFrame(statuses)
    problem_df = pd.DataFrame(problems)
    status_df.to_csv(out/"run_status.csv", index=False)
    problem_df.to_csv(out/"missing_or_failed.csv", index=False)

    all_eval = pd.concat(evals, ignore_index=True, sort=False) if evals else pd.DataFrame()
    all_loss = pd.concat(losses, ignore_index=True, sort=False) if losses else pd.DataFrame()
    all_prof = pd.concat(profiles, ignore_index=True, sort=False) if profiles else pd.DataFrame()
    all_samp = pd.concat(samplings, ignore_index=True, sort=False) if samplings else pd.DataFrame()

    if not all_eval.empty:
        all_eval.to_csv(out/"all_eval_records.csv", index=False)
        endpoint = all_eval[pd.to_numeric(all_eval["epoch"], errors="coerce") == 120].copy()
        endpoint.to_csv(out/"endpoint_eval_epoch120.csv", index=False)
        ranking, ref = build_ranking(endpoint)
        ranking.to_csv(out/"canonical_parity_ranking.csv", index=False)
        ranking.head(5).to_csv(out/"top5_canonical_parity_candidates.csv", index=False)
        pd.DataFrame([ref]).to_csv(out/"unweighted_reference_endpoint.csv", index=False)
    else:
        endpoint = pd.DataFrame()
        ranking = pd.DataFrame()

    if not all_loss.empty:
        all_loss.to_csv(out/"all_loss_history.csv", index=False)
        all_loss[pd.to_numeric(all_loss["epoch"], errors="coerce") == 120].to_csv(
            out/"endpoint_loss_epoch120.csv", index=False
        )
    if not all_prof.empty:
        all_prof.to_csv(out/"all_oracle_score_time_profiles.csv", index=False)
        build_time_deltas(all_prof, ranking).to_csv(
            out/"canonical_time_profiles_vs_reference.csv", index=False
        )
    if not all_samp.empty:
        all_samp.to_csv(out/"all_oracle_sampling_decomposition.csv", index=False)

    # Compact surface table for plotting / inspection.
    if not endpoint.empty:
        surface_cols = [
            c for c in [
                "config_id","role","outer_time_weighting","csem_w","terminal_kl_w",
                "fid_vae_recon","terminal_component_kl_fulltrain",
                "terminal_qT_vs_gaussian_sw2",FID_HEUN,FID_RK4,KID_HEUN,KID_RK4,
                SW2_HEUN,SW2_RK4,
                "oracle_profile_uncond_learned_oracle_score_logtime_mean",
                "oracle_profile_uncond_intrinsic_var_score_logtime_mean",
                "oracle_profile_guided_learned_oracle_score_logtime_mean",
                "oracle_transport_fid_oracle_qT_rk4_ode_25",
                "oracle_transport_fid_oracle_gaussian_rk4_ode_25",
                "oracle_transport_fid_learned_gaussian_uncond_rk4_ode_25",
            ] if c in endpoint.columns
        ]
        endpoint[surface_cols].to_csv(out/"metric_parity_surface.csv", index=False)

    readme = f"""CSEM metric-parity v1 compilation

Rows in manifest: 17
Successful statuses: {int((status_df['returncode']==0).sum()) if 'returncode' in status_df else 0}
Problems: {len(problem_df)}

Primary scientific file:
  canonical_parity_ranking.csv

It ranks the 16 canonical cells against the single unweighted reference using:
  reconstruction FID parity,
  full-training terminal component KL parity,
  empirical qT-vs-N(0,I) latent SW2 parity.

Default parity box:
  |recon FID - reference| <= 0.5
  terminal KL ratio within [0.8, 1.25]
  terminal SW2 ratio within [0.8, 1.25]

parity_score is a continuous normalized distance to the reference using those
same scales. The tolerances are selection aids, not statistical confidence intervals.

Once parity is achieved, compare:
  mean_gen_fid_CminusU
  mean_gen_kid_CminusU
  mean_gen_sw2_CminusU
  oracle_terminal_init_penalty_fid
  learned-vs-oracle score ratios

canonical_time_profiles_vs_reference.csv compares each canonical representation's
score-error profile to the same unweighted reference at every t.
"""
    (out/"README.txt").write_text(readme)

    print("="*80)
    print("METRIC-PARITY V1 COMPILATION")
    print("="*80)
    print("successful:", int((status_df["returncode"]==0).sum()))
    print("problems:", len(problem_df))
    print("output:", out)
    if not ranking.empty:
        cols = [
            "config_id","csem_w","terminal_kl_w","parity_score","inside_parity_box",
            "fid_vae_recon","terminal_kl_ratio","terminal_sw2_ratio","mean_gen_fid",
            "mean_gen_fid_CminusU"
        ]
        print("\nTop canonical parity candidates:")
        print(ranking[cols].head(8).to_string(index=False))

    if len(problem_df) and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
