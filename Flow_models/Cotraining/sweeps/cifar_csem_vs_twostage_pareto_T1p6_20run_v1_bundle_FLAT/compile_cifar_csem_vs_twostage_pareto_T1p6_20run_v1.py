#!/usr/bin/env python3
"""Compile the 20-cell Pareto sweep and produce direct dominance tables/plots."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep")
MANIFEST = "cifar_csem_vs_twostage_pareto_T1p6_20run_v1_manifest.csv"
RESULTS_ROOT = "cifar_csem_vs_twostage_pareto_T1p6_20run_v1_results"
STATUS_ROOT = "cifar_csem_vs_twostage_pareto_T1p6_20run_v1_status"
OUT_ROOT = "cifar_csem_vs_twostage_pareto_T1p6_20run_v1_compiled"
EVAL_EPOCH_LABEL = 500
META = [
    "cell_id", "family", "rep_id", "rep_role", "seed", "pareto_lever",
    "pareto_value", "T_K", "T_full", "csem_w", "terminal_kl_w",
]


def read_csv(path: Path):
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warning] failed reading {path}: {exc}", file=sys.stderr)
        return None


def stamp(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for c in reversed(META):
        if c in out.columns:
            out = out.drop(columns=[c])
        out.insert(0, c, row[c])
    return out


def wide_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["fid", "kid", "sw2", "diversity"]
    for keys, g in curves.groupby(META, dropna=False, sort=True):
        base = dict(zip(META, keys if isinstance(keys, tuple) else (keys,)))
        direct = g[g["mode"] == "direct_q0_train"]
        steps_values = sorted(pd.to_numeric(g["steps"], errors="coerce").dropna().astype(int).unique())
        for steps in steps_values:
            if steps <= 0:
                continue
            sg = g[pd.to_numeric(g["steps"], errors="coerce") == steps]
            for h in ("TK", "T"):
                q = sg[sg["horizon_name"].astype(str) == h]
                if q.empty:
                    continue
                rec = dict(base)
                rec["steps"] = steps
                rec["nfe"] = 4 * steps
                rec["horizon_name"] = h
                rec["horizon_t"] = float(q["horizon_t"].iloc[0])
                modes = {
                    "oracle_qh": q[(q["score_source"] == "oracle") & (q["init_mode"] == "q_h")],
                    "learned_qh": q[(q["score_source"] == "learned") & (q["init_mode"] == "q_h")],
                    "oracle_gaussian": q[(q["score_source"] == "oracle") & (q["init_mode"] == "gaussian")],
                    "learned_gaussian": q[(q["score_source"] == "learned") & (q["init_mode"] == "gaussian")],
                }
                for label, fr in modes.items():
                    if len(fr):
                        z = fr.iloc[0]
                        for m in metrics:
                            rec[f"{m}_{label}"] = z.get(m, np.nan)
                        for m in [
                            "endpoint_rms_to_maxnfe", "endpoint_rel_rms_to_maxnfe",
                            "endpoint_rms_learned_vs_oracle", "endpoint_rms_gaussian_vs_qh",
                        ]:
                            rec[f"{m}_{label}"] = z.get(m, np.nan)
                if len(direct):
                    z = direct.iloc[0]
                    for m in metrics:
                        rec[f"{m}_direct_q0"] = z.get(m, np.nan)
                rows.append(rec)
    return pd.DataFrame(rows)


def last_numeric(df: pd.DataFrame, col: str, stage: str | None = None) -> float:
    q = df
    if stage is not None and "stage" in q.columns:
        q = q[q["stage"].astype(str) == stage]
    if q.empty or col not in q.columns:
        return float("nan")
    if "epoch" in q.columns:
        ep = pd.to_numeric(q["epoch"], errors="coerce")
        q = q.loc[ep == ep.max()]
    vals = pd.to_numeric(q[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if len(vals) else float("nan")


def final_eval_metric(df: pd.DataFrame, config_contains: str, metric: str = "fid") -> float:
    if df.empty or "config" not in df.columns or metric not in df.columns:
        return float("nan")
    q = df[df["config"].astype(str).str.contains(config_contains, regex=False, na=False)]
    if q.empty:
        return float("nan")
    if "epoch" in q.columns:
        ep = pd.to_numeric(q["epoch"], errors="coerce")
        q = q.loc[ep == ep.max()]
    vals = pd.to_numeric(q[metric], errors="coerce").dropna()
    return float(vals.iloc[-1]) if len(vals) else float("nan")


def profile_summary(profile: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if profile.empty or "t" not in profile.columns:
        return out
    p = profile.copy().sort_values("t")
    t = pd.to_numeric(p["t"], errors="coerce").to_numpy(dtype=float)
    first = p.iloc[0]
    out["profile_t_min"] = float(t[0])
    for col in [
        "oracle_cond_score_rms", "oracle_cond_ode_drift_rms",
        "oracle_uncond_score_rms", "oracle_uncond_ode_drift_rms",
        "learned_cond_score_rms", "learned_uncond_score_rms",
    ]:
        out[f"nearzero_{col}"] = float(pd.to_numeric(pd.Series([first.get(col, np.nan)]), errors="coerce").iloc[0])

    # Log-time modelability means (same index measure used in the manuscript diagnostics).
    for col in [
        "cond_learned_oracle_score", "uncond_learned_oracle_score",
        "cond_learned_oracle_eps", "uncond_learned_oracle_eps",
        "cond_intrinsic_var_score", "uncond_intrinsic_var_score",
    ]:
        if col in p.columns:
            vals = pd.to_numeric(p[col], errors="coerce").to_numpy(dtype=float)
            out[f"{col}_logtime_mean"] = float(np.nanmean(vals))

    # Use the profile's own trapezoid node widths when available.
    if "physical_dt_node_width" in p.columns:
        widths = pd.to_numeric(p["physical_dt_node_width"], errors="coerce").to_numpy(dtype=float)
    else:
        if len(t) == 1:
            widths = np.ones(1, dtype=float)
        else:
            widths = np.empty(len(t), dtype=float)
            widths[0] = 0.5 * (t[1] - t[0])
            widths[-1] = 0.5 * (t[-1] - t[-2])
            if len(t) > 2:
                widths[1:-1] = 0.5 * (t[2:] - t[:-2])

    # Primary intrinsic regularity statistic proposed for the Pareto test.
    for col in [
        "oracle_cond_drift_path_rate_rms", "oracle_cond_score_path_rate_rms",
        "oracle_uncond_drift_path_rate_rms", "oracle_uncond_score_path_rate_rms",
    ]:
        if col in p.columns:
            vals = pd.to_numeric(p[col], errors="coerce").to_numpy(dtype=float)
            out[f"integrated_{col}"] = float(np.nansum(vals * widths))
            out[f"peak_{col}"] = float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan")
    return out


def extract_cell_summary(row: pd.Series, curves: pd.DataFrame | None, profile: pd.DataFrame | None,
                         train_eval: pd.DataFrame | None, train_loss: pd.DataFrame | None) -> dict:
    rec = {c: row[c] for c in META}
    rec["pareto_value"] = float(row["pareto_value"])
    rec["seed"] = int(row["seed"])
    rec["T_K"] = float(row["T_K"])
    rec["T_full"] = float(row["T_full"])
    rec["csem_w"] = float(row["csem_w"])
    rec["terminal_kl_w"] = float(row["terminal_kl_w"])

    te = train_eval if train_eval is not None else pd.DataFrame()
    tl = train_loss if train_loss is not None else pd.DataFrame()
    pr = profile if profile is not None else pd.DataFrame()
    cv = curves if curves is not None else pd.DataFrame()

    rec["recon_fid"] = final_eval_metric(te, "VAE_Rec_eps", "fid")
    rec["gaussian_T_fid"] = final_eval_metric(te, "initgaussianT1p6", "fid")
    rec["oracle_qT_fid"] = final_eval_metric(te, "initoracleqT1p6", "fid")
    rec["posterior_var"] = last_numeric(tl, "posterior_var", stage="cotrain")
    rec["posterior_var_median"] = last_numeric(tl, "posterior_var_median", stage="cotrain")
    rec["latent_rms_median"] = last_numeric(tl, "latent_rms_median", stage="cotrain")
    rec["boundary_KL"] = last_numeric(tl, "terminal_kl", stage="cotrain")
    rec["train_score_lsi"] = last_numeric(tl, "score_lsi", stage="cotrain")
    rec.update(profile_summary(pr))

    if not cv.empty:
        q = cv[
            (cv["horizon_name"].astype(str) == "T")
            & (pd.to_numeric(cv["steps"], errors="coerce") == 50)
            & (cv["score_source"].astype(str) == "learned")
            & (cv["init_mode"].astype(str) == "q_h")
        ]
        if len(q):
            z = q.iloc[0]
            rec["endpoint_model_rms_T_200nfe"] = float(z.get("endpoint_rms_learned_vs_oracle", np.nan))
            rec["learned_qT_fid_200nfe"] = float(z.get("fid", np.nan))
        q = cv[
            (cv["horizon_name"].astype(str) == "T")
            & (pd.to_numeric(cv["steps"], errors="coerce") == 50)
            & (cv["score_source"].astype(str) == "oracle")
            & (cv["init_mode"].astype(str) == "q_h")
        ]
        if len(q):
            rec["oracle_qT_fid_200nfe_mech"] = float(q.iloc[0].get("fid", np.nan))
    return rec


def aggregate_cells(cell_summary: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = [c for c in cell_summary.columns if c not in META and c not in {"rep_role"}]
    # pareto_value is a grouping key, not an averaged metric.
    numeric_metrics = [c for c in numeric_metrics if c != "pareto_value"]
    rows = []
    group_cols = ["family", "pareto_lever", "pareto_value"]
    for keys, g in cell_summary.groupby(group_cols, sort=True, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_seeds"] = int(len(g))
        for col in numeric_metrics:
            vals = pd.to_numeric(g[col], errors="coerce") if col in g.columns else pd.Series(dtype=float)
            if vals.notna().any():
                rec[f"{col}_mean"] = float(vals.mean())
                rec[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def interpolation_dominance(agg: pd.DataFrame, ymean_col: str, label: str) -> pd.DataFrame:
    a = agg[agg["family"] == "csem"].copy()
    b = agg[agg["family"] == "two_stage"].copy()
    xcol = "recon_fid_mean"
    if xcol not in agg.columns or ymean_col not in agg.columns or a.empty or b.empty:
        return pd.DataFrame()
    a = a[[xcol, ymean_col]].dropna().sort_values(xcol)
    b = b[[xcol, ymean_col]].dropna().sort_values(xcol)
    if len(a) < 2 or len(b) < 2:
        return pd.DataFrame()
    lo = max(a[xcol].min(), b[xcol].min())
    hi = min(a[xcol].max(), b[xcol].max())
    if not lo < hi:
        return pd.DataFrame()
    x_candidates = sorted(set(
        [float(x) for x in a[xcol] if lo <= x <= hi]
        + [float(x) for x in b[xcol] if lo <= x <= hi]
        + [float(lo), float(hi)]
    ))
    ya = np.log10(np.maximum(a[ymean_col].to_numpy(dtype=float), 1e-30))
    yb = np.log10(np.maximum(b[ymean_col].to_numpy(dtype=float), 1e-30))
    xa = a[xcol].to_numpy(dtype=float)
    xb = b[xcol].to_numpy(dtype=float)
    rows = []
    for x in x_candidates:
        ca = float(np.interp(x, xa, ya))
        tb = float(np.interp(x, xb, yb))
        rows.append({
            "metric": label,
            "recon_fid": x,
            "log10_csem": ca,
            "log10_two_stage": tb,
            "delta_log10_csem_minus_two_stage": ca - tb,
            "csem_better": bool(ca < tb),
        })
    return pd.DataFrame(rows)


def save_plots(cell_summary: pd.DataFrame, agg: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] matplotlib unavailable; skipping plots: {exc}", file=sys.stderr)
        return

    specs = [
        ("integrated_oracle_cond_drift_path_rate_rms", "Integrated exact conditional drift path-rate", "pareto_exact_field.png", True),
        ("cond_learned_oracle_score_logtime_mean", "Learned-vs-oracle conditional score MSE (log-time mean)", "pareto_score_modelability.png", True),
        ("endpoint_model_rms_T_200nfe", "Propagated learned-vs-oracle endpoint RMS at T, 200 NFE", "pareto_endpoint_model_error.png", True),
        ("gaussian_T_fid", "Gaussian-start FID at T", "pareto_generation.png", False),
    ]
    for metric, ylabel, filename, logy in specs:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if "recon_fid_mean" not in agg.columns or mean_col not in agg.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for family, g in agg.groupby("family", sort=False):
            gg = g[["recon_fid_mean", "recon_fid_std", mean_col, std_col]].dropna().sort_values("recon_fid_mean")
            if gg.empty:
                continue
            ax.errorbar(
                gg["recon_fid_mean"], gg[mean_col],
                xerr=gg["recon_fid_std"], yerr=gg[std_col],
                marker="o", capsize=3, label=family,
            )
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Reconstruction FID (lower is better)")
        ax.set_ylabel(ylabel + (" (log scale)" if logy else ""))
        ax.set_title("CSEM vs standard two-stage Pareto frontier")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    base = args.base_dir.resolve()
    manifest = pd.read_csv(base / MANIFEST, dtype={"cell_id": str})
    manifest["cell_id"] = manifest["cell_id"].str.zfill(2)
    if len(manifest) != 20:
        raise RuntimeError(f"Expected 20 manifest cells, found {len(manifest)}")

    out = base / OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    statuses, problems = [], []
    curves_all, profiles_all, train_eval_all, train_loss_all = [], [], [], []
    cell_rows = []

    for _, row in manifest.iterrows():
        root = base / RESULTS_ROOT / row["result_name"]
        train = root / "training"
        mech = root / "mechanism_eval"
        statusp = base / STATUS_ROOT / row["result_name"] / "overall.json"
        curvep = mech / "run_terminal_kl" / "dataframes" / f"oracle_sampling_decomposition_ep{EVAL_EPOCH_LABEL}.csv"
        profilep = mech / "run_terminal_kl" / "dataframes" / f"oracle_score_time_profile_ep{EVAL_EPOCH_LABEL}.csv"
        tevalp = train / "combined_dataframes" / "combined_eval_metrics.csv"
        tlossp = train / "combined_dataframes" / "combined_loss_history.csv"

        st = {
            "cell_id": row["cell_id"], "family": row["family"], "rep_id": row["rep_id"],
            "seed": row["seed"], "status_exists": statusp.is_file(),
            "curve_exists": curvep.is_file(), "profile_exists": profilep.is_file(),
            "training_eval_exists": tevalp.is_file(), "training_loss_exists": tlossp.is_file(),
            "returncode": None,
        }
        if statusp.is_file():
            try:
                st["returncode"] = json.loads(statusp.read_text()).get("returncode")
            except Exception as exc:
                st["status_parse_error"] = repr(exc)
        statuses.append(st)
        ok = st["status_exists"] and st["returncode"] == 0 and st["curve_exists"] and st["profile_exists"] and st["training_eval_exists"] and st["training_loss_exists"]
        if not ok:
            problems.append(st)

        curve = read_csv(curvep)
        profile = read_csv(profilep)
        teval = read_csv(tevalp)
        tloss = read_csv(tlossp)
        for d, arr in [(curve, curves_all), (profile, profiles_all), (teval, train_eval_all), (tloss, train_loss_all)]:
            if d is not None:
                arr.append(stamp(d, row))
        cell_rows.append(extract_cell_summary(row, curve, profile, teval, tloss))

    status_df = pd.DataFrame(statuses)
    problem_df = pd.DataFrame(problems)
    status_df.to_csv(out / "run_status.csv", index=False)
    problem_df.to_csv(out / "missing_or_failed.csv", index=False)

    curves_df = pd.concat(curves_all, ignore_index=True, sort=False) if curves_all else pd.DataFrame()
    profiles_df = pd.concat(profiles_all, ignore_index=True, sort=False) if profiles_all else pd.DataFrame()
    teval_df = pd.concat(train_eval_all, ignore_index=True, sort=False) if train_eval_all else pd.DataFrame()
    tloss_df = pd.concat(train_loss_all, ignore_index=True, sort=False) if train_loss_all else pd.DataFrame()
    if not curves_df.empty:
        curves_df.to_csv(out / "oracle_nfe_curve_long.csv", index=False)
        wide_table(curves_df).to_csv(out / "oracle_nfe_mechanism_wide.csv", index=False)
    if not profiles_df.empty:
        profiles_df.to_csv(out / "oracle_field_profile_all.csv", index=False)
    if not teval_df.empty:
        teval_df.to_csv(out / "training_eval_all.csv", index=False)
    if not tloss_df.empty:
        tloss_df.to_csv(out / "training_loss_all.csv", index=False)

    cell_summary = pd.DataFrame(cell_rows)
    cell_summary.to_csv(out / "pareto_cell_summary.csv", index=False)
    agg = aggregate_cells(cell_summary)
    agg.to_csv(out / "pareto_seed_aggregates.csv", index=False)

    dominance_frames = []
    for col, label in [
        ("integrated_oracle_cond_drift_path_rate_rms_mean", "exact_conditional_drift_path_rate"),
        ("cond_learned_oracle_score_logtime_mean_mean", "learned_oracle_score_mse"),
        ("endpoint_model_rms_T_200nfe_mean", "endpoint_model_rms_200nfe"),
    ]:
        d = interpolation_dominance(agg, col, label)
        if not d.empty:
            dominance_frames.append(d)
    dominance = pd.concat(dominance_frames, ignore_index=True) if dominance_frames else pd.DataFrame()
    dominance.to_csv(out / "pareto_overlap_interpolation.csv", index=False)
    save_plots(cell_summary, agg, out)

    (out / "README.txt").write_text(
        "CIFAR CSEM vs standard two-stage Pareto dominance sweep v1\n\n"
        "Headline files:\n"
        "  pareto_cell_summary.csv          one row per independent training\n"
        "  pareto_seed_aggregates.csv       two-seed means/stds per operating point\n"
        "  pareto_overlap_interpolation.csv local overlap comparison; negative delta means CSEM lower/better\n"
        "  pareto_exact_field.png           reconstruction vs intrinsic exact-field regularity\n"
        "  pareto_score_modelability.png    reconstruction vs learned-oracle score error\n"
        "  pareto_endpoint_model_error.png  reconstruction vs propagated model error\n"
        "  pareto_generation.png            secondary reconstruction-vs-generation view\n\n"
        "Primary regularity statistic:\n"
        "  integrated_oracle_cond_drift_path_rate_rms\n"
        "computed from the frozen exact empirical conditional score field over [t_min,T].\n\n"
        "Two-stage cells use exact T_K=0: reconstruction from z0 + beta0*K0, no diffusion-derived VAE gradients.\n"
        "CSEM cells use T_K=1.2, T=1.6, terminal/boundary KL weight .40.\n"
        "All cells train exactly 500 epochs with zero score-only refinement.\n"
    )

    print(f"Compilation output: {out}")
    print(f"cells summarized: {len(cell_summary)}; problems: {len(problems)}")
    if problems and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
