#!/usr/bin/env python3
"""Compile the 18-cell CIFAR fine T_K x csem_w sweep."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
MANIFEST = "cifar_TKxCSEM_fine_T1p75_v1_manifest.csv"
RESULTS_ROOT = "cifar_TKxCSEM_fine_T1p75_v1_results"
STATUS_ROOT = "cifar_TKxCSEM_fine_T1p75_v1_status"
OUT_ROOT = "cifar_TKxCSEM_fine_T1p75_v1_compiled"

META = ["cell_id", "group_id", "T_K", "T_full", "csem_w", "result_name"]

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

def one_metric(row: pd.Series, metric: str, marker: str):
    hits = [
        c for c in row.index
        if str(c).startswith(metric + "_rk4_")
        and marker.lower() in str(c).lower()
        and pd.notna(row[c])
    ]
    # Avoid class-specific columns if any ever appear.
    nonclass = [c for c in hits if "_y" not in str(c)]
    hits = nonclass if nonclass else hits
    if len(hits) == 1:
        return float(row[hits[0]]), hits[0]
    if len(hits) == 0:
        return np.nan, ""
    # Deterministic fallback, but flag ambiguity in the selected-column field.
    c = sorted(hits)[0]
    return float(row[c]), "AMBIGUOUS:" + "|".join(sorted(hits))

def status_filename(result_name: str) -> str:
    return result_name.replace("/", "__") + ".json"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    base = args.base_dir.resolve()
    manifest_path = base / MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(
        manifest_path,
        dtype={"cell_id": str, "group_id": str},
    )
    manifest["cell_id"] = manifest["cell_id"].str.zfill(2)
    manifest["group_id"] = manifest["group_id"].str.zfill(2)

    out = base / OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    statuses = []
    eval_frames = []
    loss_frames = []
    summary_rows = []

    for _, m in manifest.iterrows():
        root = base / RESULTS_ROOT / str(m["result_name"])
        status_path = base / STATUS_ROOT / status_filename(str(m["result_name"]))
        eval_path = root / "combined_dataframes" / "combined_eval_metrics.csv"
        loss_path = root / "combined_dataframes" / "combined_loss_history.csv"

        st = {
            "cell_id": m["cell_id"],
            "group_id": m["group_id"],
            "T_K": float(m["T_K"]),
            "T_full": float(m["T_full"]),
            "csem_w": float(m["csem_w"]),
            "result_name": m["result_name"],
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
                st["returncode"] = d.get("returncode")
                st["elapsed_seconds"] = d.get("elapsed_seconds")
            except Exception as exc:
                st["status_parse_error"] = repr(exc)
        statuses.append(st)

        ev = read_csv(eval_path)
        lo = read_csv(loss_path)
        if ev is not None:
            eval_frames.append(stamp(ev, m))
        if lo is not None:
            loss_frames.append(stamp(lo, m))

        if ev is None or len(ev) == 0:
            continue

        ev2 = ev.copy()
        ev2["_epoch_num"] = pd.to_numeric(ev2["epoch"], errors="coerce")
        final_ev = ev2.loc[ev2["_epoch_num"].idxmax()].drop(labels="_epoch_num")

        rec = {
            "cell_id": m["cell_id"],
            "group_id": m["group_id"],
            "T_K": float(m["T_K"]),
            "T": float(m["T_full"]),
            "csem_w": float(m["csem_w"]),
            "terminal_kl_w": float(m["terminal_kl_w"]),
            "final_eval_epoch": int(final_ev["epoch"]),
            "stage": final_ev.get("stage", ""),
            "fid_recon": final_ev.get("fid_vae_recon", np.nan),
            "kid_recon": final_ev.get("kid_vae_recon", np.nan),
            "sw2_recon": final_ev.get("sw2_vae_recon", np.nan),
        }

        for metric in ("fid", "kid", "sw2", "div"):
            val, col = one_metric(final_ev, metric, "initoracleqTK")
            rec[f"{metric}_oracle_qTK"] = val
            rec[f"{metric}_oracle_column"] = col
            val, col = one_metric(final_ev, metric, "initgaussianT")
            rec[f"{metric}_gaussian_T"] = val
            rec[f"{metric}_gaussian_column"] = col

        rec["fid_gaussian_minus_oracle"] = (
            rec["fid_gaussian_T"] - rec["fid_oracle_qTK"]
        )
        rec["fid_gaussian_minus_recon"] = (
            rec["fid_gaussian_T"] - rec["fid_recon"]
        )
        rec["fid_oracle_minus_recon"] = (
            rec["fid_oracle_qTK"] - rec["fid_recon"]
        )

        if lo is not None and len(lo):
            lo2 = lo.copy()
            lo2["_epoch_num"] = pd.to_numeric(lo2["epoch"], errors="coerce")
            cot = lo2[lo2["_epoch_num"] == 500]
            if len(cot):
                q = cot.iloc[-1]
            else:
                q = lo2.loc[lo2["_epoch_num"].idxmax()]
            for c in [
                "loss", "recon", "terminal_kl", "latent_rms",
                "latent_rms_median", "posterior_var", "posterior_var_median",
                "posterior_std", "logvar_median", "score_mse_weighted",
                "score_mse_unweighted", "score_mse_head_weighted",
                "score_head_loss", "aux_lam", "active_csem_w",
                "lr_score_head", "vae_grad_preclip", "score_grad_preclip",
                "vae_clip_rate", "score_clip_rate",
                "score_weight_mean", "score_weight_ess_fraction",
            ]:
                if c in q.index:
                    rec[f"epoch500_{c}"] = q[c]

        summary_rows.append(rec)

    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out / "run_status.csv", index=False)
    missing = status_df[
        ~(
            status_df["status_exists"].fillna(False)
            & status_df["result_dir_exists"].fillna(False)
            & status_df["eval_csv_exists"].fillna(False)
            & status_df["loss_csv_exists"].fillna(False)
            & (pd.to_numeric(status_df["returncode"], errors="coerce") == 0)
        )
    ].copy()
    missing.to_csv(out / "missing_or_failed.csv", index=False)

    if eval_frames:
        pd.concat(eval_frames, ignore_index=True, sort=False).to_csv(
            out / "all_eval_records.csv", index=False
        )
    if loss_frames:
        pd.concat(loss_frames, ignore_index=True, sort=False).to_csv(
            out / "all_loss_history.csv", index=False
        )

    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary = summary.sort_values(["csem_w", "T_K"]).reset_index(drop=True)
        summary.to_csv(out / "fine_sweep_summary.csv", index=False)

        complete_for_rank = summary.dropna(subset=["fid_gaussian_T"]).copy()
        if len(complete_for_rank):
            best_by_csem = (
                complete_for_rank.sort_values("fid_gaussian_T")
                .groupby("csem_w", as_index=False)
                .first()
                .sort_values("csem_w")
            )
            best_by_csem.to_csv(out / "best_by_csem.csv", index=False)

            best_overall = (
                complete_for_rank.sort_values("fid_gaussian_T")
                .head(1)
            )
            best_overall.to_csv(out / "best_overall.csv", index=False)

        print("\nEndpoint summary (sorted by csem_w then T_K):")
        cols = [
            "T_K", "csem_w", "fid_recon", "fid_oracle_qTK",
            "fid_gaussian_T", "fid_gaussian_minus_oracle",
            "epoch500_terminal_kl", "epoch500_latent_rms",
            "epoch500_score_mse_head_weighted",
        ]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))

    print(f"\nCompiled output: {out}")
    print(f"Missing/failed cells: {len(missing)} / 18")
    if len(missing) and not args.allow_incomplete:
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
