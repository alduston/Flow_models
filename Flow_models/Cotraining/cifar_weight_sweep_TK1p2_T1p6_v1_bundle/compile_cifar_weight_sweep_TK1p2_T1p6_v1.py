#!/usr/bin/env python3
"""Compile the 20-cell CIFAR CSEM/KL weight sweep into endpoint and trajectory tables."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

BASE_DIR = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0')
MANIFEST = 'cifar_weight_sweep_TK1p2_T1p6_v1_manifest.csv'
RESULTS_ROOT = 'cifar_weight_sweep_TK1p2_T1p6_v1_results'
STATUS_ROOT = 'cifar_weight_sweep_TK1p2_T1p6_v1_status'
OUT_ROOT = 'cifar_weight_sweep_TK1p2_T1p6_v1_compiled'

META = ["cell_id", "T_K", "T_full", "csem_w", "terminal_kl_w", "result_name"]


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
    for col in reversed(META):
        if col in out.columns:
            out = out.drop(columns=[col])
        out.insert(0, col, row[col])
    return out


def horizon_tag(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


def find_metric_column(columns, metric: str, marker: str):
    hits = [
        c for c in columns
        if c.startswith(metric + "_rk4_") and marker.lower() in c.lower()
    ]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        return None
    nonclass = [c for c in hits if "_y" not in c]
    if len(nonclass) == 1:
        return nonclass[0]
    return sorted(hits)[0]


def build_summary(final_eval: pd.DataFrame, cotrain_loss: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in final_eval.iterrows():
        tk_tag = horizon_tag(r["T_K"])
        t_tag = horizon_tag(r["T_full"])
        rec = {
            "cell_id": r["cell_id"],
            "csem_w": float(r["csem_w"]),
            "terminal_kl_w": float(r["terminal_kl_w"]),
            "T_K": float(r["T_K"]),
            "T": float(r["T_full"]),
            "final_eval_epoch": int(r["epoch"]),
            "final_stage": r.get("stage", ""),
            "fid_recon": r.get("fid_vae_recon", np.nan),
            "kid_recon": r.get("kid_vae_recon", np.nan),
            "sw2_recon": r.get("sw2_vae_recon", np.nan),
        }

        arm_markers = {
            "oracle_qTK": f"initoracleqTK{tk_tag}",
            "gaussian_TK": f"initgaussianTK{tk_tag}",
            "oracle_qT": f"initoracleqT{t_tag}",
            "gaussian_T": f"initgaussianT{t_tag}",
        }
        for metric in ("fid", "kid", "sw2", "div"):
            for arm, marker in arm_markers.items():
                col = find_metric_column(final_eval.columns, metric, marker)
                if col is not None:
                    rec[f"{metric}_{arm}"] = r.get(col, np.nan)
                    rec[f"{metric}_{arm}_column"] = col

        def diff(name, a, b):
            if a in rec and b in rec:
                rec[name] = rec[a] - rec[b]

        diff("fid_gaussian_minus_oracle_TK", "fid_gaussian_TK", "fid_oracle_qTK")
        diff("fid_gaussian_minus_oracle_T", "fid_gaussian_T", "fid_oracle_qT")
        diff("fid_oracle_T_minus_TK", "fid_oracle_qT", "fid_oracle_qTK")
        diff("fid_gaussian_T_minus_TK", "fid_gaussian_T", "fid_gaussian_TK")
        diff("fid_gaussian_T_minus_recon", "fid_gaussian_T", "fid_recon")

        loss_hit = cotrain_loss[
            cotrain_loss["cell_id"].astype(str).str.zfill(2)
            == str(r["cell_id"]).zfill(2)
        ]
        if len(loss_hit):
            q = loss_hit.iloc[-1]
            for c in [
                "terminal_kl", "latent_rms", "posterior_var",
                "posterior_var_median", "posterior_std", "logvar_median",
                "score_mse_unweighted", "score_mse_weighted",
                "score_mse_head_weighted", "score_head_loss",
                "active_csem_w", "lr_score_head",
            ]:
                if c in q.index:
                    rec[f"cotrain500_{c}"] = q[c]
        rows.append(rec)

    return (
        pd.DataFrame(rows)
        .sort_values(["csem_w", "terminal_kl_w"])
        .reset_index(drop=True)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    manifest = pd.read_csv(base / MANIFEST, dtype={"cell_id": str})
    manifest["cell_id"] = manifest["cell_id"].str.zfill(2)

    out = base / OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    statuses = []
    problems = []
    eval_frames = []
    loss_frames = []

    for _, row in manifest.iterrows():
        root = base / RESULTS_ROOT / row["result_name"]
        status_path = base / STATUS_ROOT / f"{row['result_name']}.json"
        eval_path = root / "combined_dataframes" / "combined_eval_metrics.csv"
        loss_path = root / "combined_dataframes" / "combined_loss_history.csv"

        status = {
            "cell_id": row["cell_id"],
            "csem_w": row["csem_w"],
            "terminal_kl_w": row["terminal_kl_w"],
            "T_K": row["T_K"],
            "T_full": row["T_full"],
            "result_name": row["result_name"],
            "status_exists": status_path.is_file(),
            "result_dir_exists": root.is_dir(),
            "eval_csv_exists": eval_path.is_file(),
            "loss_csv_exists": loss_path.is_file(),
            "returncode": None,
            "elapsed_seconds": None,
        }
        if status_path.is_file():
            try:
                d = json.loads(status_path.read_text())
                status["returncode"] = d.get("returncode")
                status["elapsed_seconds"] = d.get("elapsed_seconds")
            except Exception as exc:
                status["status_parse_error"] = repr(exc)

        statuses.append(status)
        ok = (
            status["status_exists"]
            and status["returncode"] == 0
            and status["eval_csv_exists"]
            and status["loss_csv_exists"]
        )
        if not ok:
            problems.append(status)

        e = read_csv(eval_path)
        if e is not None:
            eval_frames.append(stamp(e, row))
        l = read_csv(loss_path)
        if l is not None:
            loss_frames.append(stamp(l, row))

    status_df = pd.DataFrame(statuses)
    problem_df = pd.DataFrame(problems)
    status_df.to_csv(out / "run_status.csv", index=False)
    problem_df.to_csv(out / "missing_or_failed.csv", index=False)

    all_eval = pd.concat(eval_frames, ignore_index=True, sort=False) if eval_frames else pd.DataFrame()
    all_loss = pd.concat(loss_frames, ignore_index=True, sort=False) if loss_frames else pd.DataFrame()

    if not all_eval.empty:
        all_eval.to_csv(out / "all_eval_records.csv", index=False)
        final_rows = []
        cotrain_rows = []
        for cell_id, g in all_eval.groupby("cell_id", sort=True):
            gg = g.copy()
            gg["epoch_num"] = pd.to_numeric(gg["epoch"], errors="coerce")
            final_rows.append(gg.loc[gg["epoch_num"].idxmax()].drop(labels="epoch_num"))
            cot = gg[gg.get("stage", "") == "cotrain"]
            if len(cot):
                cotrain_rows.append(cot.loc[cot["epoch_num"].idxmax()].drop(labels="epoch_num"))

        final_eval = pd.DataFrame(final_rows)
        cotrain_eval = pd.DataFrame(cotrain_rows)
        final_eval.to_csv(out / "final_eval_epoch600.csv", index=False)
        cotrain_eval.to_csv(out / "cotrain_endpoint_eval_epoch500.csv", index=False)
    else:
        final_eval = pd.DataFrame()
        cotrain_eval = pd.DataFrame()

    if not all_loss.empty:
        all_loss.to_csv(out / "all_loss_history.csv", index=False)
        cotrain_loss = all_loss[
            (pd.to_numeric(all_loss["epoch"], errors="coerce") == 500)
            & (all_loss["stage"] == "cotrain")
        ].copy()
        final_loss = all_loss[pd.to_numeric(all_loss["epoch"], errors="coerce") == 600].copy()
        cotrain_loss.to_csv(out / "cotrain_endpoint_loss_epoch500.csv", index=False)
        final_loss.to_csv(out / "final_loss_epoch600.csv", index=False)
    else:
        cotrain_loss = pd.DataFrame()
        final_loss = pd.DataFrame()

    if not final_eval.empty:
        summary = build_summary(final_eval, cotrain_loss)
        summary.to_csv(out / "weight_sweep_summary.csv", index=False)
        print("\nCSEM/KL weight sweep endpoint summary:")
        display_cols = [
            c for c in [
                "csem_w", "terminal_kl_w", "fid_recon",
                "fid_oracle_qTK", "fid_gaussian_TK",
                "fid_oracle_qT", "fid_gaussian_T",
                "fid_gaussian_minus_oracle_TK", "fid_gaussian_minus_oracle_T",
                "cotrain500_terminal_kl", "cotrain500_posterior_var",
            ] if c in summary.columns
        ]
        print(summary[display_cols].to_string(index=False))

    readme = """CIFAR CSEM / terminal-KL weight sweep compilation

Fixed:
  T_K = 1.2
  T = 1.6
  canonical outer representation weighting
  unweighted-epsilon score-head weighting
  500 cotrain + 100 score-only refinement epochs

Swept:
  csem_w in {0, .02, .04, .06, .08}
  terminal_kl_w in {.10, .20, .30, .40}

Primary file:
  weight_sweep_summary.csv

Four-way learned-score evaluation columns:
  *_oracle_qTK      empirical class-conditional q_TK init, reverse from T_K=1.2
  *_gaussian_TK     Gaussian init, reverse from T_K=1.2
  *_oracle_qT       empirical class-conditional q_T init, reverse from T=1.6
  *_gaussian_T      Gaussian init, reverse from T=1.6

The T-horizon RK4 step count is increased by the driver to match the T_K
integration-grid/NFE density.
"""
    (out / "README.txt").write_text(readme)

    print("\nCompilation output:", out)
    print("Problems:", len(problem_df))
    if len(problem_df) and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
