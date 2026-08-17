#!/usr/bin/env python3
"""
Compile the 8-run FMNIST CSEM "why canonical?" diagnostic sweep.

Outputs:
  fmnist_csem_why_compiled_v2/
    run_status.csv
    missing_or_failed.csv
    all_eval_records.csv
    endpoint_eval_epoch120.csv
    all_loss_history.csv
    endpoint_loss_epoch120.csv
    all_oracle_score_time_profiles.csv
    all_oracle_sampling_decomposition.csv
    pair_endpoint_long.csv
    pair_endpoint_wide.csv
    oracle_profile_pair_deltas.csv
    oracle_score_time_profile_pair_wide.csv
    oracle_score_time_bins.csv
    oracle_score_time_bin_pair_deltas.csv
    oracle_sampling_pair_deltas.csv
    README.txt
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

DEFAULT_BASE_DIR = Path(
    "/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
)
MANIFEST_NAME = "fmnist_csem_why_manifest_v2.csv"
RESULTS_ROOT_NAME = "fmnist_csem_why_results_v2"
STATUS_ROOT_NAME = "fmnist_csem_why_status_v2"
OUT_ROOT_NAME = "fmnist_csem_why_compiled_v2"

META = [
    "config_id","pair_id","slot_in_pair","outer_time_weighting","head_time_weighting",
    "rep_weight","lr_score_head","score_head_loss_w","T_terminal","epochs","refine_epochs",
    "eval_every","eval_samples","cfg_strength","oracle_profile_query_samples",
    "oracle_profile_time_points","oracle_profile_batch_size","oracle_reference_batch_size",
    "oracle_sampling_samples","oracle_sampling_batch_size","oracle_sampling_steps",
    "prior_recon_fid","prior_mean_gen_fid","result_name",
]


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warning] Failed to read {path}: {exc}", file=sys.stderr)
        return None


def stamp(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for col in reversed(META):
        if col in out.columns:
            out = out.drop(columns=[col])
        out.insert(0, col, row[col])
    return out


def status_row(row: pd.Series, path: Path) -> dict:
    rec = {c: row[c] for c in META}
    rec.update({
        "status_file_exists": path.is_file(),
        "returncode": None,
        "elapsed_seconds": None,
        "started_utc": None,
        "finished_utc": None,
        "status_parse_error": "",
    })
    if path.is_file():
        try:
            d = json.loads(path.read_text())
            for k in ("returncode","elapsed_seconds","started_utc","finished_utc"):
                rec[k] = d.get(k)
        except Exception as exc:
            rec["status_parse_error"] = repr(exc)
    return rec


def numeric_pair_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row/pair with _U, _C and _CminusU columns for numeric endpoint fields."""
    rows = []
    for pair_id, g in long_df.groupby("pair_id", sort=True):
        u = g[g["outer_time_weighting"] == "unweighted-eps"]
        c = g[g["outer_time_weighting"] == "canonical"]
        if len(u) != 1 or len(c) != 1:
            continue
        u = u.iloc[0]
        c = c.iloc[0]
        rec = {
            "pair_id": pair_id,
            "lr_score_head": u.get("lr_score_head"),
            "unweighted_config_id": u.get("config_id"),
            "canonical_config_id": c.get("config_id"),
        }
        common_cols = [x for x in long_df.columns if x in u.index and x in c.index]
        for col in common_cols:
            if col in META or col in ("outer_time_weighting","head_time_weighting","stage","tag","scale_anchor"):
                continue
            uv, cv = u[col], c[col]
            try:
                uf, cf = float(uv), float(cv)
            except Exception:
                continue
            if np.isfinite(uf) or np.isfinite(cf):
                rec[f"{col}_U"] = uf
                rec[f"{col}_C"] = cf
                rec[f"{col}_CminusU"] = cf - uf
        rows.append(rec)
    return pd.DataFrame(rows)


def profile_pair_deltas(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    key = ["pair_id","t_index"]
    u = profile[profile.outer_time_weighting == "unweighted-eps"].copy()
    c = profile[profile.outer_time_weighting == "canonical"].copy()
    merged = u.merge(c, on=key, suffixes=("_U","_C"))
    out = merged[key].copy()
    if "lr_score_head_U" in merged:
        out["lr_score_head"] = merged["lr_score_head_U"]
    if "t_U" in merged:
        out["t"] = merged["t_U"]
    for col in profile.select_dtypes(include=[np.number]).columns:
        if col in ("pair_id","t_index"):
            continue
        cu, cc = f"{col}_U", f"{col}_C"
        if cu in merged and cc in merged:
            out[cu] = merged[cu]
            out[cc] = merged[cc]
            out[f"{col}_CminusU"] = merged[cc] - merged[cu]
    return out



TIME_PROFILE_METRICS = [
    "uncond_component_residual_eps",
    "uncond_intrinsic_var_eps",
    "uncond_learned_oracle_eps",
    "uncond_component_residual_score",
    "uncond_intrinsic_var_score",
    "uncond_learned_oracle_score",
    "cond_component_residual_eps",
    "cond_intrinsic_var_eps",
    "cond_learned_oracle_eps",
    "cond_component_residual_score",
    "cond_intrinsic_var_score",
    "cond_learned_oracle_score",
    "guided_learned_oracle_eps",
    "guided_learned_oracle_score",
]


def time_profile_pair_wide(profile: pd.DataFrame) -> pd.DataFrame:
    """One row per matched pair and t, exposing U/C error profiles side-by-side."""
    if profile.empty:
        return pd.DataFrame()

    key = ["pair_id", "t_index"]
    u = profile[profile.outer_time_weighting == "unweighted-eps"].copy()
    c = profile[profile.outer_time_weighting == "canonical"].copy()
    merged = u.merge(c, on=key, suffixes=("_U", "_C"))
    if merged.empty:
        return merged

    out = merged[key].copy()
    for common in (
        "lr_score_head", "t", "log_t", "alpha", "sigma", "sigma_sq",
        "snr", "log_snr", "physical_dt_node_width",
        "canonical_eps_node_mass", "canonical_eps_node_mass_fraction",
    ):
        ucol = f"{common}_U"
        ccol = f"{common}_C"
        if ucol in merged:
            out[common] = merged[ucol]
        elif ccol in merged:
            out[common] = merged[ccol]

    for metric in TIME_PROFILE_METRICS:
        ucol, ccol = f"{metric}_U", f"{metric}_C"
        if ucol not in merged or ccol not in merged:
            continue
        out[ucol] = merged[ucol]
        out[ccol] = merged[ccol]
        out[f"{metric}_CminusU"] = merged[ccol] - merged[ucol]
        out[f"{metric}_CoverU"] = merged[ccol] / merged[ucol].replace(0, np.nan)

        # Also surface the per-node physical-time contributions if present.
        pc = f"{metric}_physical_contribution"
        pcu, pcc = f"{pc}_U", f"{pc}_C"
        if pcu in merged and pcc in merged:
            out[pcu] = merged[pcu]
            out[pcc] = merged[pcc]
            out[f"{pc}_CminusU"] = merged[pcc] - merged[pcu]

    return out.sort_values(["pair_id", "t"]).reset_index(drop=True)


def time_profile_bins(profile: pd.DataFrame, n_bands: int = 6) -> pd.DataFrame:
    """Summarize each score-error profile over fixed equal-width log(t) bands."""
    if profile.empty:
        return pd.DataFrame()

    t_min = float(profile["t"].min())
    t_max = float(profile["t"].max())
    edges = np.geomspace(max(t_min, 1e-30), t_max, n_bands + 1)
    labels = [f"B{i}_{edges[i]:.3g}_to_{edges[i+1]:.3g}" for i in range(n_bands)]

    work = profile.copy()
    # include_lowest ensures the first t point is assigned.
    work["time_band"] = pd.cut(
        work["t"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    group_cols = [
        "config_id", "pair_id", "outer_time_weighting",
        "head_time_weighting", "lr_score_head", "rep_weight", "time_band"
    ]
    for keys, g in work.groupby(group_cols, observed=True, sort=True):
        if g.empty:
            continue
        rec = dict(zip(group_cols, keys))
        band_label = str(rec["time_band"])
        band_idx = labels.index(band_label)
        rec.update({
            "band_index": band_idx,
            "band_t_low": edges[band_idx],
            "band_t_high": edges[band_idx + 1],
            "band_t_geom_mid": float(np.sqrt(edges[band_idx] * edges[band_idx + 1])),
            "profile_points": len(g),
        })

        for metric in TIME_PROFILE_METRICS:
            if metric not in g:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            rec[f"{metric}_mean"] = float(np.nanmean(vals))

            pc = f"{metric}_physical_contribution"
            if pc in g:
                phys = pd.to_numeric(g[pc], errors="coerce").to_numpy(float)
                band_integral = float(np.nansum(phys))
                rec[f"{metric}_physical_integral"] = band_integral

                # Fraction of this run's full physical-time absolute contribution.
                full = work[
                    (work["config_id"] == rec["config_id"])
                ]
                full_pc = pd.to_numeric(full[pc], errors="coerce").to_numpy(float)
                denom = max(float(np.nansum(np.abs(full_pc))), 1e-30)
                rec[f"{metric}_physical_abs_fraction"] = abs(band_integral) / denom
        rows.append(rec)

    return pd.DataFrame(rows).sort_values(
        ["pair_id", "outer_time_weighting", "band_index"]
    ).reset_index(drop=True)


def time_bin_pair_deltas(bins: pd.DataFrame) -> pd.DataFrame:
    """Canonical-minus-unweighted differences within each matched pair and time band."""
    if bins.empty:
        return pd.DataFrame()

    key = ["pair_id", "band_index"]
    u = bins[bins.outer_time_weighting == "unweighted-eps"].copy()
    c = bins[bins.outer_time_weighting == "canonical"].copy()
    merged = u.merge(c, on=key, suffixes=("_U", "_C"))
    if merged.empty:
        return merged

    out = merged[key].copy()
    for common in (
        "lr_score_head", "band_t_low", "band_t_high",
        "band_t_geom_mid", "profile_points"
    ):
        col = f"{common}_U"
        if col in merged:
            out[common] = merged[col]

    for metric in TIME_PROFILE_METRICS:
        for suffix in ("mean", "physical_integral", "physical_abs_fraction"):
            root = f"{metric}_{suffix}"
            ucol, ccol = f"{root}_U", f"{root}_C"
            if ucol not in merged or ccol not in merged:
                continue
            out[ucol] = merged[ucol]
            out[ccol] = merged[ccol]
            out[f"{root}_CminusU"] = merged[ccol] - merged[ucol]
            out[f"{root}_CoverU"] = merged[ccol] / merged[ucol].replace(0, np.nan)

    return out.sort_values(["pair_id", "band_index"]).reset_index(drop=True)


def sampling_pair_deltas(sampling: pd.DataFrame) -> pd.DataFrame:
    if sampling.empty:
        return pd.DataFrame()
    key = ["pair_id","mode"]
    u = sampling[sampling.outer_time_weighting == "unweighted-eps"].copy()
    c = sampling[sampling.outer_time_weighting == "canonical"].copy()
    merged = u.merge(c, on=key, suffixes=("_U","_C"))
    out = merged[key].copy()
    if "lr_score_head_U" in merged:
        out["lr_score_head"] = merged["lr_score_head_U"]
    for metric in ("fid","kid","sw2","diversity"):
        a, b = f"{metric}_U", f"{metric}_C"
        if a in merged and b in merged:
            out[a] = merged[a]
            out[b] = merged[b]
            out[f"{metric}_CminusU"] = merged[b] - merged[a]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    manifest_path = base / MANIFEST_NAME
    results_root = base / RESULTS_ROOT_NAME
    status_root = base / STATUS_ROOT_NAME
    out_root = base / OUT_ROOT_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, dtype={"config_id":str,"pair_id":str})
    manifest["config_id"] = manifest["config_id"].str.zfill(3)
    manifest["pair_id"] = manifest["pair_id"].str.zfill(2)
    if len(manifest) != 8:
        raise SystemExit(f"Expected 8 manifest rows, found {len(manifest)}")

    statuses, problems = [], []
    eval_frames, loss_frames, profile_frames, sampling_frames = [], [], [], []

    for _, row in manifest.iterrows():
        name = str(row["result_name"])
        root = results_root / name
        status_path = status_root / f"{name}.json"
        eval_path = root / "combined_dataframes" / "combined_eval_metrics.csv"
        loss_path = root / "combined_dataframes" / "combined_loss_history.csv"
        profile_paths = sorted(root.glob("**/oracle_score_time_profile_ep*.csv"))
        sampling_paths = sorted(root.glob("**/oracle_sampling_decomposition_ep*.csv"))

        sr = status_row(row, status_path)
        sr.update({
            "result_dir_exists": root.is_dir(),
            "eval_csv_exists": eval_path.is_file(),
            "loss_csv_exists": loss_path.is_file(),
            "oracle_profile_files": len(profile_paths),
            "oracle_sampling_files": len(sampling_paths),
        })
        statuses.append(sr)

        ok = (
            sr["status_file_exists"]
            and sr["returncode"] == 0
            and sr["result_dir_exists"]
            and sr["eval_csv_exists"]
            and sr["loss_csv_exists"]
            and sr["oracle_profile_files"] >= 1
            and sr["oracle_sampling_files"] >= 1
            and not sr["status_parse_error"]
        )
        if not ok:
            problems.append(sr)

        d = read_csv(eval_path)
        if d is not None:
            eval_frames.append(stamp(d, row))
        d = read_csv(loss_path)
        if d is not None:
            loss_frames.append(stamp(d, row))
        for q in profile_paths:
            d = read_csv(q)
            if d is not None:
                eval_epoch = int(q.stem.rsplit("ep",1)[-1])
                d["profile_file_epoch"] = eval_epoch
                profile_frames.append(stamp(d, row))
        for q in sampling_paths:
            d = read_csv(q)
            if d is not None:
                eval_epoch = int(q.stem.rsplit("ep",1)[-1])
                d["sampling_file_epoch"] = eval_epoch
                sampling_frames.append(stamp(d, row))

    status_df = pd.DataFrame(statuses)
    problem_df = pd.DataFrame(problems)
    status_df.to_csv(out_root/"run_status.csv", index=False)
    problem_df.to_csv(out_root/"missing_or_failed.csv", index=False)

    all_eval = pd.concat(eval_frames, ignore_index=True, sort=False) if eval_frames else pd.DataFrame()
    all_loss = pd.concat(loss_frames, ignore_index=True, sort=False) if loss_frames else pd.DataFrame()
    profiles = pd.concat(profile_frames, ignore_index=True, sort=False) if profile_frames else pd.DataFrame()
    sampling = pd.concat(sampling_frames, ignore_index=True, sort=False) if sampling_frames else pd.DataFrame()

    if not all_eval.empty:
        all_eval.to_csv(out_root/"all_eval_records.csv", index=False)
        endpoint_eval = all_eval[pd.to_numeric(all_eval["epoch"], errors="coerce") == 120].copy()
        endpoint_eval.to_csv(out_root/"endpoint_eval_epoch120.csv", index=False)
    else:
        endpoint_eval = pd.DataFrame()

    if not all_loss.empty:
        all_loss.to_csv(out_root/"all_loss_history.csv", index=False)
        endpoint_loss = all_loss[pd.to_numeric(all_loss["epoch"], errors="coerce") == 120].copy()
        endpoint_loss.to_csv(out_root/"endpoint_loss_epoch120.csv", index=False)
    else:
        endpoint_loss = pd.DataFrame()

    if not profiles.empty:
        profiles.to_csv(out_root/"all_oracle_score_time_profiles.csv", index=False)

        time_wide = time_profile_pair_wide(profiles)
        time_wide.to_csv(
            out_root/"oracle_score_time_profile_pair_wide.csv", index=False
        )

        time_bins = time_profile_bins(profiles, n_bands=6)
        time_bins.to_csv(out_root/"oracle_score_time_bins.csv", index=False)

        time_bin_pair_deltas(time_bins).to_csv(
            out_root/"oracle_score_time_bin_pair_deltas.csv", index=False
        )

    if not sampling.empty:
        sampling.to_csv(out_root/"all_oracle_sampling_decomposition.csv", index=False)

    # Merge endpoint evaluation and training-health rows into one long table.
    if not endpoint_eval.empty:
        if not endpoint_loss.empty:
            loss_keep = [
                c for c in endpoint_loss.columns
                if c not in META and c not in ("stage","scale_anchor","epoch")
            ]
            loss_small = endpoint_loss[META + ["epoch"] + loss_keep].copy()
            loss_small = loss_small.rename(
                columns={c:f"train_{c}" for c in loss_keep}
            )
            pair_long = endpoint_eval.merge(
                loss_small,
                on=META + ["epoch"],
                how="left",
            )
        else:
            pair_long = endpoint_eval.copy()
        pair_long.to_csv(out_root/"pair_endpoint_long.csv", index=False)
        numeric_pair_wide(pair_long).to_csv(
            out_root/"pair_endpoint_wide.csv", index=False
        )

    if not profiles.empty:
        profile_pair_deltas(profiles).to_csv(
            out_root/"oracle_profile_pair_deltas.csv", index=False
        )
    if not sampling.empty:
        sampling_pair_deltas(sampling).to_csv(
            out_root/"oracle_sampling_pair_deltas.csv", index=False
        )

    n_ok = int((status_df["returncode"] == 0).sum())
    text = f"""FMNIST CSEM WHY diagnostic compilation

Manifest configurations: {len(manifest)}
Successful status rows:  {n_ok}
Problem/incomplete rows: {len(problem_df)}

Scientific reading order:
1. pair_endpoint_wide.csv
   Canonical-minus-unweighted endpoint differences for each matched pair.
   Includes ordinary recon/generation metrics plus oracle-profile aggregate
   summaries and terminal/oracle-transport summaries.

2. oracle_score_time_profile_pair_wide.csv
   The main raw-by-t comparison table: one row per matched pair and time.
   It contains U, C, C-U, and C/U for actual learned-vs-oracle score error,
   intrinsic CSEM variance, total component residual, and guided score error.
   Time geometry includes t, log(t), SNR/log-SNR, dt node width, and canonical
   dt/sigma^2 node mass.

3. oracle_score_time_bin_pair_deltas.csv
   Six equal-width log(t) bands. This is the easiest table for asking WHERE
   canonical differs: small-t, intermediate-t, or terminal-time. It reports
   band means, physical-time integrals, and fractions of total physical error.

4. oracle_profile_pair_deltas.csv
   Full generic time-resolved comparison. Key columns include:
     uncond_component_residual_eps
     uncond_intrinsic_var_eps
     uncond_learned_oracle_eps
     cond_component_residual_eps
     cond_intrinsic_var_eps
     cond_learned_oracle_eps
     guided_learned_oracle_eps
   and their score-space analogues. CminusU is canonical minus unweighted.

5. oracle_sampling_pair_deltas.csv
   Endpoint transport decomposition for:
     direct_q0_train
     oracle_qT_rk4_ode_25
     oracle_gaussian_rk4_ode_25
     learned_gaussian_uncond_rk4_ode_25
   Difference direct_q0 -> oracle_qT diagnoses finite-NFE oracle transport.
   Difference oracle_qT -> oracle_gaussian diagnoses terminal initialization.
   Difference oracle_gaussian -> learned_gaussian diagnoses score approximation
   with the same Gaussian initialization (metrics are not algebraically additive).

6. all_oracle_score_time_profiles.csv
   Raw time-profile data for plotting.

7. all_oracle_sampling_decomposition.csv
   Raw transport-ablation rows.

Important:
- The exact aggregate oracle is constructed from the full training-set empirical
  posterior mixture, not the 2k test subset.
- Standard endpoint FID/KID/SW2/diversity still use the parent-sweep protocol.
- Score-profile MSE values are means per latent coordinate.
"""
    (out_root/"README.txt").write_text(text)

    print("="*76)
    print("CSEM WHY COMPILATION")
    print("="*76)
    print(f"configs: {len(manifest)}")
    print(f"successful: {n_ok}")
    print(f"problems: {len(problem_df)}")
    print(f"profiles rows: {len(profiles)}")
    print(f"transport rows: {len(sampling)}")
    print(f"output: {out_root}")

    if len(problem_df) and not args.allow_incomplete:
        print("Compilation written, but sweep is incomplete; inspect missing_or_failed.csv.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
