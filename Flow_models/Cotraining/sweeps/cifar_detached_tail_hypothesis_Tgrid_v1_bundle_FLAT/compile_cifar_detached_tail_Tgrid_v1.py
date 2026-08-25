#!/usr/bin/env python3
"""Compile the 18-cell detached-tail sweep and directly test tail-length stability."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep")
MANIFEST = "cifar_detached_tail_Tgrid_v1_manifest.csv"
RESULTS_ROOT = "cifar_detached_tail_Tgrid_v1_results"
STATUS_ROOT = "cifar_detached_tail_Tgrid_v1_status"
OUT_ROOT = "cifar_detached_tail_Tgrid_v1_compiled"
EVAL_EPOCH_LABEL = 500
META = [
    "cell_id", "rep_id", "rep_role", "seed",
    "T_K", "T_full", "delta_T", "tail_fraction", "TK_fraction",
    "csem_w", "terminal_kl_w",
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


def time_tag(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


def latest_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    q = df.copy()
    if "epoch" in q.columns:
        ep = pd.to_numeric(q["epoch"], errors="coerce")
        if ep.notna().any():
            q = q.loc[ep == ep.max()]
    return q.iloc[-1] if len(q) else None


def numeric_value(row: pd.Series | None, col: str) -> float:
    if row is None or col not in row.index:
        return float("nan")
    try:
        return float(pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0])
    except Exception:
        return float("nan")


def final_training_metrics(df: pd.DataFrame | None, T: float, TK: float) -> dict[str, float]:
    z = latest_row(df if df is not None else pd.DataFrame())
    out: dict[str, float] = {}
    if z is None:
        return out
    ttag = time_tag(T)
    ktag = time_tag(TK)
    out["recon_fid"] = numeric_value(z, "fid_vae_recon")
    out["recon_kid"] = numeric_value(z, "kid_vae_recon")
    out["oracle_qT_fid"] = numeric_value(z, f"fid_rk4_25_randtok_cfg3_0_initoracleqT{ttag}")
    out["gaussian_T_fid"] = numeric_value(z, f"fid_rk4_25_randtok_cfg3_0_initgaussianT{ttag}")
    out["oracle_qT_sw2"] = numeric_value(z, f"sw2_rk4_25_randtok_cfg3_0_initoracleqT{ttag}")
    out["gaussian_T_sw2"] = numeric_value(z, f"sw2_rk4_25_randtok_cfg3_0_initgaussianT{ttag}")
    out["score_lsi_gap_T"] = numeric_value(z, f"lsi_gap_rk4_25_randtok_cfg3_0_initoracleqT{ttag}")
    if TK > 0:
        out["oracle_qTK_fid"] = numeric_value(z, f"fid_rk4_25_randtok_cfg3_0_initoracleqTK{ktag}")
        out["gaussian_TK_fid"] = numeric_value(z, f"fid_rk4_25_randtok_cfg3_0_initgaussianTK{ktag}")
        out["oracle_qTK_sw2"] = numeric_value(z, f"sw2_rk4_25_randtok_cfg3_0_initoracleqTK{ktag}")
        out["gaussian_TK_sw2"] = numeric_value(z, f"sw2_rk4_25_randtok_cfg3_0_initgaussianTK{ktag}")
    out["gaussian_minus_oracle_T_fid"] = out.get("gaussian_T_fid", np.nan) - out.get("oracle_qT_fid", np.nan)
    out["abs_gaussian_minus_oracle_T_fid"] = abs(out["gaussian_minus_oracle_T_fid"]) if np.isfinite(out["gaussian_minus_oracle_T_fid"]) else np.nan
    if "gaussian_TK_fid" in out and "oracle_qTK_fid" in out:
        out["gaussian_minus_oracle_TK_fid"] = out["gaussian_TK_fid"] - out["oracle_qTK_fid"]
        out["abs_gaussian_minus_oracle_TK_fid"] = abs(out["gaussian_minus_oracle_TK_fid"])
    return out


def last_loss_metrics(df: pd.DataFrame | None) -> dict[str, float]:
    if df is None or df.empty:
        return {}
    q = df.copy()
    if "stage" in q.columns:
        q2 = q[q["stage"].astype(str) == "cotrain"]
        if not q2.empty:
            q = q2
    z = latest_row(q)
    if z is None:
        return {}
    cols = [
        "loss", "recon", "kl", "terminal_kl", "latent_rms", "latent_rms_median",
        "posterior_var", "posterior_var_median", "posterior_std", "logvar_median",
        "score_lsi", "score_mse_weighted", "score_mse_unweighted",
        "score_mse_head_weighted", "score_head_loss",
    ]
    return {c: numeric_value(z, c) for c in cols}


def profile_summary(profile: pd.DataFrame | None) -> dict[str, float]:
    if profile is None or profile.empty or "t" not in profile.columns:
        return {}
    p = profile.copy().sort_values("t")
    t = pd.to_numeric(p["t"], errors="coerce").to_numpy(dtype=float)
    first = p.iloc[0]
    out: dict[str, float] = {"profile_t_min": float(t[0])}
    for col in [
        "oracle_cond_score_rms", "oracle_cond_ode_drift_rms",
        "learned_cond_score_rms", "cond_learned_oracle_score",
        "cond_learned_oracle_eps", "cond_intrinsic_var_score",
    ]:
        if col in p.columns:
            out[f"nearzero_{col}"] = float(pd.to_numeric(pd.Series([first[col]]), errors="coerce").iloc[0])
            vals = pd.to_numeric(p[col], errors="coerce").to_numpy(dtype=float)
            out[f"{col}_logtime_mean"] = float(np.nanmean(vals))

    if "physical_dt_node_width" in p.columns:
        widths = pd.to_numeric(p["physical_dt_node_width"], errors="coerce").to_numpy(dtype=float)
    else:
        widths = np.empty(len(t), dtype=float)
        if len(t) == 1:
            widths[:] = 1.0
        else:
            widths[0] = 0.5 * (t[1] - t[0]); widths[-1] = 0.5 * (t[-1] - t[-2])
            if len(t) > 2:
                widths[1:-1] = 0.5 * (t[2:] - t[:-2])
    for col in ["oracle_cond_drift_path_rate_rms", "oracle_cond_score_path_rate_rms"]:
        if col in p.columns:
            vals = pd.to_numeric(p[col], errors="coerce").to_numpy(dtype=float)
            out[f"integrated_{col}"] = float(np.nansum(vals * widths))
            out[f"peak_{col}"] = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
    return out


def curve_summary(curve: pd.DataFrame | None) -> dict[str, float]:
    if curve is None or curve.empty:
        return {}
    c = curve.copy()
    out: dict[str, float] = {}
    for h in ("TK", "T"):
        qh = c[c["horizon_name"].astype(str) == h]
        if qh.empty:
            continue
        for steps in (5, 10, 25, 50):
            q = qh[pd.to_numeric(qh["steps"], errors="coerce") == steps]
            oq = q[(q["score_source"].astype(str) == "oracle") & (q["init_mode"].astype(str) == "q_h")]
            lq = q[(q["score_source"].astype(str) == "learned") & (q["init_mode"].astype(str) == "q_h")]
            og = q[(q["score_source"].astype(str) == "oracle") & (q["init_mode"].astype(str) == "gaussian")]
            lg = q[(q["score_source"].astype(str) == "learned") & (q["init_mode"].astype(str) == "gaussian")]
            nfe = 4 * steps
            if len(oq):
                z = oq.iloc[0]
                out[f"oracle_qh_fid_{h}_{nfe}nfe"] = float(z.get("fid", np.nan))
                out[f"exact_dyn_rms_{h}_{nfe}nfe"] = float(z.get("endpoint_rms_to_maxnfe", np.nan))
            if len(lq):
                z = lq.iloc[0]
                out[f"learned_qh_fid_{h}_{nfe}nfe"] = float(z.get("fid", np.nan))
                out[f"model_rms_{h}_{nfe}nfe"] = float(z.get("endpoint_rms_learned_vs_oracle", np.nan))
            if len(og):
                z = og.iloc[0]
                out[f"oracle_gaussian_fid_{h}_{nfe}nfe"] = float(z.get("fid", np.nan))
                out[f"init_rms_oracle_{h}_{nfe}nfe"] = float(z.get("endpoint_rms_gaussian_vs_qh", np.nan))
            if len(lg):
                out[f"learned_gaussian_fid_{h}_{nfe}nfe"] = float(lg.iloc[0].get("fid", np.nan))
    return out


def aggregate_geometry(cells: pd.DataFrame) -> pd.DataFrame:
    group = ["T_full", "delta_T", "T_K", "tail_fraction", "TK_fraction", "csem_w", "terminal_kl_w"]
    excluded = set(META) | {"result_name", "rep_role"}
    metrics = [c for c in cells.columns if c not in excluded and c != "seed"]
    rows = []
    for keys, g in cells.groupby(group, sort=True, dropna=False):
        rec = dict(zip(group, keys)); rec["n_seeds"] = int(len(g))
        for col in metrics:
            vals = pd.to_numeric(g[col], errors="coerce")
            if vals.notna().any():
                rec[f"{col}_mean"] = float(vals.mean())
                rec[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_axis(geom: pd.DataFrame, axis: str) -> pd.DataFrame:
    means = [c for c in geom.columns if c.endswith("_mean")]
    rows = []
    for val, g in geom.groupby(axis, sort=True):
        rec = {axis: val, "n_geometries": len(g)}
        for col in means:
            x = pd.to_numeric(g[col], errors="coerce")
            if x.notna().any():
                rec[f"{col}_across_geometry_mean"] = float(x.mean())
                rec[f"{col}_across_geometry_std"] = float(x.std(ddof=1)) if x.notna().sum() > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def best_delta_table(geom: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("gaussian_T_fid_mean", "min"),
        ("abs_gaussian_minus_oracle_T_fid_mean", "min"),
        ("oracle_qT_fid_mean", "min"),
        ("recon_fid_mean", "min"),
        ("cond_learned_oracle_score_logtime_mean_mean", "min"),
        ("integrated_oracle_cond_drift_path_rate_rms_mean", "min"),
        ("model_rms_T_200nfe_mean", "min"),
        ("exact_dyn_rms_T_20nfe_mean", "min"),
    ]
    rows = []
    for T, g in geom.groupby("T_full", sort=True):
        for metric, direction in metrics:
            if metric not in g.columns:
                continue
            q = g[["delta_T", "T_K", metric]].dropna()
            if q.empty:
                continue
            idx = q[metric].idxmin() if direction == "min" else q[metric].idxmax()
            z = q.loc[idx]
            rows.append({
                "T_full": float(T), "metric": metric, "direction": direction,
                "best_delta_T": float(z["delta_T"]), "best_T_K": float(z["T_K"]),
                "best_value": float(z[metric]),
            })
    return pd.DataFrame(rows)


def stability_summary(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if best.empty:
        return pd.DataFrame()
    for metric, g in best.groupby("metric", sort=True):
        ds = pd.to_numeric(g["best_delta_T"], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(ds):
            continue
        rows.append({
            "metric": metric,
            "n_T_values": len(ds),
            "best_delta_mean": float(ds.mean()),
            "best_delta_min": float(ds.min()),
            "best_delta_max": float(ds.max()),
            "best_delta_range": float(ds.max() - ds.min()),
            "all_best_in_0p4_to_0p5": bool(np.all((ds >= 0.4 - 1e-12) & (ds <= 0.5 + 1e-12))),
            "all_exactly_same_best_delta": bool(np.allclose(ds, ds[0])),
        })
    return pd.DataFrame(rows)


def make_plots(geom: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] matplotlib unavailable: {exc}", file=sys.stderr)
        return

    specs = [
        ("gaussian_T_fid_mean", "Gaussian-start FID at T", "tail_gaussian_fid.png", False),
        ("abs_gaussian_minus_oracle_T_fid_mean", "|Gaussian - oracle qT| FID gap", "tail_init_gap.png", False),
        ("oracle_qT_fid_mean", "Oracle-qT FID", "tail_oracle_fid.png", False),
        ("recon_fid_mean", "Reconstruction FID", "tail_reconstruction.png", False),
        ("cond_learned_oracle_score_logtime_mean_mean", "Learned-vs-oracle score error", "tail_modelability.png", True),
        ("integrated_oracle_cond_drift_path_rate_rms_mean", "Integrated exact drift path-rate", "tail_exact_field.png", True),
    ]
    for metric, ylabel, filename, logy in specs:
        if metric not in geom.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for T, g in geom.groupby("T_full", sort=True):
            gg = g.sort_values("delta_T")
            yerr = gg.get(metric.replace("_mean", "_std"), None)
            ax.errorbar(gg["delta_T"], gg[metric], yerr=yerr, marker="o", capsize=3, label=f"T={T:g}")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"Detached tail length $\Delta T=T-T_K$")
        ax.set_ylabel(ylabel)
        ax.set_title("Detached-tail hypothesis: fixed wC=.05, wK=.60")
        ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(out / filename, dpi=180); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    base = args.base_dir.resolve()
    manifest = pd.read_csv(base / MANIFEST, dtype={"cell_id": str})
    manifest["cell_id"] = manifest["cell_id"].str.zfill(2)
    if len(manifest) != 18:
        raise RuntimeError(f"Expected 18 manifest cells, found {len(manifest)}")

    out = base / OUT_ROOT; out.mkdir(parents=True, exist_ok=True)
    statuses, problems, cell_rows = [], [], []
    curves_all, profiles_all, tevals_all, tloss_all = [], [], [], []

    for _, row in manifest.iterrows():
        root = base / RESULTS_ROOT / row["result_name"]
        train = root / "training"; mech = root / "mechanism_eval"
        statusp = base / STATUS_ROOT / row["result_name"] / "overall.json"
        curvep = mech / "run_terminal_kl" / "dataframes" / f"oracle_sampling_decomposition_ep{EVAL_EPOCH_LABEL}.csv"
        profilep = mech / "run_terminal_kl" / "dataframes" / f"oracle_score_time_profile_ep{EVAL_EPOCH_LABEL}.csv"
        tevalp = train / "combined_dataframes" / "combined_eval_metrics.csv"
        tlossp = train / "combined_dataframes" / "combined_loss_history.csv"

        st = {
            "cell_id": row["cell_id"], "rep_id": row["rep_id"], "seed": row["seed"],
            "T_full": row["T_full"], "delta_T": row["delta_T"], "T_K": row["T_K"],
            "status_exists": statusp.is_file(), "curve_exists": curvep.is_file(),
            "profile_exists": profilep.is_file(), "training_eval_exists": tevalp.is_file(),
            "training_loss_exists": tlossp.is_file(), "returncode": None,
        }
        if statusp.is_file():
            try: st["returncode"] = json.loads(statusp.read_text()).get("returncode")
            except Exception as exc: st["status_parse_error"] = repr(exc)
        statuses.append(st)
        ok = all([st["status_exists"], st["curve_exists"], st["profile_exists"], st["training_eval_exists"], st["training_loss_exists"]]) and st["returncode"] == 0
        if not ok: problems.append(st)

        curve = read_csv(curvep); profile = read_csv(profilep); teval = read_csv(tevalp); tloss = read_csv(tlossp)
        for d, arr in [(curve, curves_all), (profile, profiles_all), (teval, tevals_all), (tloss, tloss_all)]:
            if d is not None: arr.append(stamp(d, row))

        rec = {c: row[c] for c in META}
        for c in ["seed"]: rec[c] = int(row[c])
        for c in ["T_K", "T_full", "delta_T", "tail_fraction", "TK_fraction", "csem_w", "terminal_kl_w"]:
            rec[c] = float(row[c])
        rec.update(final_training_metrics(teval, float(row["T_full"]), float(row["T_K"])))
        rec.update(last_loss_metrics(tloss))
        rec.update(profile_summary(profile))
        rec.update(curve_summary(curve))
        cell_rows.append(rec)

    pd.DataFrame(statuses).to_csv(out / "run_status.csv", index=False)
    pd.DataFrame(problems).to_csv(out / "missing_or_failed.csv", index=False)
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(out / "detached_tail_cell_summary.csv", index=False)
    geom = aggregate_geometry(cells)
    geom.to_csv(out / "detached_tail_geometry_seed_aggregates.csv", index=False)
    aggregate_axis(geom, "delta_T").to_csv(out / "detached_tail_summary_by_delta.csv", index=False)
    aggregate_axis(geom, "T_full").to_csv(out / "detached_tail_summary_by_T.csv", index=False)
    best = best_delta_table(geom); best.to_csv(out / "best_delta_by_T_and_metric.csv", index=False)
    stability_summary(best).to_csv(out / "detached_tail_stability_summary.csv", index=False)

    if curves_all:
        pd.concat(curves_all, ignore_index=True, sort=False).to_csv(out / "oracle_nfe_curve_long.csv", index=False)
    if profiles_all:
        pd.concat(profiles_all, ignore_index=True, sort=False).to_csv(out / "oracle_field_profile_all.csv", index=False)
    if tevals_all:
        pd.concat(tevals_all, ignore_index=True, sort=False).to_csv(out / "training_eval_all.csv", index=False)
    if tloss_all:
        pd.concat(tloss_all, ignore_index=True, sort=False).to_csv(out / "training_loss_all.csv", index=False)

    make_plots(geom, out)
    (out / "README.txt").write_text(
        "CIFAR detached-tail hypothesis sweep v1\n\n"
        "Headline outputs:\n"
        "  detached_tail_geometry_seed_aggregates.csv  two-seed mean/std for each T x DeltaT cell\n"
        "  detached_tail_summary_by_delta.csv          average behavior by tail length across full horizons\n"
        "  best_delta_by_T_and_metric.csv              which DeltaT wins separately at each T\n"
        "  detached_tail_stability_summary.csv         direct test of whether best DeltaT is stable across T\n"
        "  detached_tail_cell_summary.csv              one row per independent training\n\n"
        "Hypothesis support is strongest if deployment/initialization metrics repeatedly select DeltaT=.40 or .50 "
        "across T=1.45,1.60,1.75 while absolute T_K varies. Representation/oracle-qT deterioration with longer "
        "coupling should be interpreted separately from Gaussian-start gap closure.\n"
    )
    print(f"Compilation output: {out}")
    print(f"cells={len(cells)} geometries={len(geom)} problems={len(problems)}")
    if problems and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
