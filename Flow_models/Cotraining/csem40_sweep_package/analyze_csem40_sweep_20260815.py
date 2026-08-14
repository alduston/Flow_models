#!/usr/bin/env python3
"""Compile CSEM 40-epoch sweep outputs into trajectories and leaderboards."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARM_PATHS = {
    "terminal_kl": "run_terminal_kl",
    "norm": "run_scale_norm",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def safe_stat(series: pd.Series, operation: str) -> float:
    finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return math.nan
    if operation == "mean":
        return float(finite.mean())
    if operation == "min":
        return float(finite.min())
    if operation == "max":
        return float(finite.max())
    raise ValueError(operation)


def first_prefixed(row: pd.Series, prefix: str) -> float:
    for column in row.index:
        if str(column).startswith(prefix):
            value = finite_float(row[column])
            if math.isfinite(value):
                return value
    return math.nan


def read_status(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["status_path"])
    return read_json(path) if path.exists() else {"state": "missing"}


def select_loss_path(result_dir: Path, arm: str) -> Path | None:
    dataframe_dir = result_dir / ARM_PATHS[arm] / "dataframes"
    complete = dataframe_dir / "loss_history.csv"
    in_progress = dataframe_dir / "loss_history_in_progress.csv"
    if complete.exists():
        return complete
    if in_progress.exists():
        return in_progress
    return None


def select_eval_path(result_dir: Path, arm: str) -> Path | None:
    dataframe_dir = result_dir / ARM_PATHS[arm] / "dataframes"
    complete = dataframe_dir / "eval_metrics.csv"
    in_progress = dataframe_dir / "eval_metrics_in_progress.csv"
    if complete.exists():
        return complete
    if in_progress.exists():
        return in_progress
    return None


def summarize_arm(
    record: dict[str, Any],
    arm: str,
    status: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    result_dir = Path(record["result_dir"])
    loss_path = select_loss_path(result_dir, arm)
    eval_path = select_eval_path(result_dir, arm)
    summary: dict[str, Any] = {
        "run_index": record["run_index"],
        "run_id": record["run_id"],
        "design_source": record["design_source"],
        "design_label": record["design_label"],
        "arm": arm,
        "worker_state": status.get("state", "missing"),
        "returncode": status.get("returncode", math.nan),
        "elapsed_seconds": status.get("elapsed_seconds", math.nan),
        "result_dir": str(result_dir),
        "run_log": record["run_log"],
    }
    for name, value in record.get("factor_assignments", {}).items():
        summary[f"factor__{name}"] = value
    for name, value in record.get("args", {}).items():
        summary[f"arg__{name}"] = value
    run_args = record.get("args", {})
    csem_weight = finite_float(run_args.get("csem-w"))
    terminal_weight = finite_float(run_args.get("terminal-kl-w"))
    encoder_lr = finite_float(run_args.get("lr-encoder"))
    score_lr = finite_float(run_args.get("lr-score-head"))
    summary.update(
        {
            "canonical_coefficient_sync": (
                math.isfinite(csem_weight)
                and math.isfinite(terminal_weight)
                and math.isclose(
                    csem_weight, terminal_weight, rel_tol=1e-9, abs_tol=1e-12
                )
            ),
            "csem_to_terminal_coefficient_ratio": (
                csem_weight / terminal_weight
                if math.isfinite(csem_weight)
                and math.isfinite(terminal_weight)
                and terminal_weight > 0.0
                else math.inf if terminal_weight == 0.0 else math.nan
            ),
            # These are scale proxies, not substitutes for measured gradient
            # norms. They make coefficient/LR compensation explicit in tables.
            "encoder_csem_step_scale_proxy": csem_weight * encoder_lr,
            "encoder_terminal_step_scale_proxy": terminal_weight * encoder_lr,
            "score_csem_step_scale_proxy": csem_weight * score_lr,
        }
    )

    if loss_path is None:
        summary.update(
            {
                "last_epoch": 0,
                "trajectory_complete": False,
                "stability_pass": False,
                "stability_violations": "missing_loss_history",
            }
        )
        return summary, None, None

    loss = pd.read_csv(loss_path)
    if loss.empty:
        summary.update(
            {
                "last_epoch": 0,
                "trajectory_complete": False,
                "stability_pass": False,
                "stability_violations": "empty_loss_history",
            }
        )
        return summary, loss, None
    loss["epoch"] = pd.to_numeric(loss["epoch"], errors="coerce")
    loss = loss.sort_values("epoch").reset_index(drop=True)
    last = loss.iloc[-1]
    early = loss[loss["epoch"] <= 10]
    transition = loss[(loss["epoch"] >= 11) & (loss["epoch"] <= 40)]
    if transition.empty:
        transition = loss

    last_epoch = int(finite_float(last.get("epoch"), 0.0))
    recon = numeric_column(loss, "recon")
    posterior = numeric_column(loss, "posterior_var")
    floor_fraction = numeric_column(loss, "logvar_floor_fraction")
    score_clip = numeric_column(loss, "score_clip_rate")
    raw_score = numeric_column(loss, "score_mse_unweighted")
    weighted_score = numeric_column(loss, "score_mse_weighted")

    recon_best = safe_stat(recon, "min")
    recon_final = finite_float(last.get("recon"))
    recon_degradation = (
        recon_final / recon_best
        if math.isfinite(recon_final) and math.isfinite(recon_best) and recon_best > 0
        else math.nan
    )
    posterior_min = safe_stat(posterior, "min")
    posterior_max = safe_stat(posterior, "max")
    posterior_final = finite_float(last.get("posterior_var"))
    latent_rms_final = finite_float(last.get("latent_rms"))
    latent_snr_proxy = (
        latent_rms_final**2 / posterior_final
        if math.isfinite(latent_rms_final)
        and math.isfinite(posterior_final)
        and posterior_final > 0
        else math.nan
    )

    summary.update(
        {
            "loss_path": str(loss_path),
            "last_epoch": last_epoch,
            "trajectory_complete": last_epoch >= 40,
            "recon_final": recon_final,
            "recon_best": recon_best,
            "recon_degradation_ratio": recon_degradation,
            "perc_final": finite_float(last.get("perc")),
            "terminal_kl_final": finite_float(last.get("terminal_kl")),
            "posterior_var_final": posterior_final,
            "posterior_var_min": posterior_min,
            "posterior_var_max": posterior_max,
            "posterior_var_range_ratio": (
                posterior_max / posterior_min
                if math.isfinite(posterior_max)
                and math.isfinite(posterior_min)
                and posterior_min > 0
                else math.nan
            ),
            "latent_rms_final": latent_rms_final,
            "latent_snr_proxy_final": latent_snr_proxy,
            "logvar_median_final": finite_float(last.get("logvar_median")),
            "logvar_floor_fraction_final": finite_float(
                last.get("logvar_floor_fraction")
            ),
            "logvar_floor_fraction_max": safe_stat(floor_fraction, "max"),
            "score_mse_unweighted_final": finite_float(
                last.get("score_mse_unweighted")
            ),
            "score_mse_weighted_final": finite_float(
                last.get("score_mse_weighted")
            ),
            "score_mse_unweighted_early": safe_stat(
                numeric_column(early, "score_mse_unweighted"), "mean"
            ),
            "score_mse_unweighted_transition": safe_stat(
                numeric_column(transition, "score_mse_unweighted"), "mean"
            ),
            "vae_clip_rate_early": safe_stat(
                numeric_column(early, "vae_clip_rate"), "mean"
            ),
            "vae_clip_rate_transition": safe_stat(
                numeric_column(transition, "vae_clip_rate"), "mean"
            ),
            "score_clip_rate_early": safe_stat(
                numeric_column(early, "score_clip_rate"), "mean"
            ),
            "score_clip_rate_transition": safe_stat(
                numeric_column(transition, "score_clip_rate"), "mean"
            ),
            "tracking_clip_rate_transition": safe_stat(
                numeric_column(transition, "tracking_score_clip_rate"), "mean"
            ),
            "grad_recon_csem_encoder_cosine_transition": safe_stat(
                numeric_column(
                    transition, "grad_recon_csem_encoder_cosine"
                ),
                "mean",
            ),
            "grad_csem_to_encoder_transition": safe_stat(
                numeric_column(transition, "grad_csem_to_encoder"), "mean"
            ),
            "grad_recon_to_encoder_transition": safe_stat(
                numeric_column(transition, "grad_recon_to_encoder"), "mean"
            ),
            "grad_kl_to_vae_transition": safe_stat(
                numeric_column(transition, "grad_kl_to_vae"), "mean"
            ),
            "grad_csem_to_vae_transition": safe_stat(
                numeric_column(transition, "grad_csem_to_vae"), "mean"
            ),
            "grad_kl_csem_cosine_transition": safe_stat(
                numeric_column(transition, "grad_kl_csem_cosine"), "mean"
            ),
            "grad_kl_csem_combined_ratio_transition": safe_stat(
                numeric_column(
                    transition, "grad_kl_csem_combined_ratio"
                ),
                "mean",
            ),
            "score_weight_mean_final": finite_float(last.get("score_weight_mean")),
            "score_weight_max_final": finite_float(last.get("score_weight_max")),
            "score_weight_cv_final": finite_float(last.get("score_weight_cv")),
            "score_weight_ess_fraction_final": finite_float(
                last.get("score_weight_ess_fraction")
            ),
            "joint_loss_max_transition": safe_stat(
                numeric_column(transition, "joint_loss_max"), "max"
            ),
            "score_mse_weighted_max_transition": safe_stat(
                numeric_column(transition, "score_mse_weighted_max"), "max"
            ),
            "lr_encoder_final": finite_float(last.get("lr_encoder_current")),
            "lr_decoder_final": finite_float(last.get("lr_decoder_current")),
            "lr_score_final": finite_float(last.get("lr_score_current")),
            "lr_tracking_head_final": finite_float(
                last.get("lr_tracking_head_current")
            ),
        }
    )

    first_warning_epoch = math.nan
    first_warning_reasons = ""
    best_reconstruction_so_far = math.inf
    for _, epoch_row in loss.iterrows():
        epoch_reasons: list[str] = []
        epoch_number = int(finite_float(epoch_row.get("epoch"), 0.0))
        epoch_reconstruction = finite_float(epoch_row.get("recon"))
        epoch_posterior = finite_float(epoch_row.get("posterior_var"))
        epoch_floor = finite_float(epoch_row.get("logvar_floor_fraction"))
        epoch_score_clip = finite_float(epoch_row.get("score_clip_rate"))
        if not math.isfinite(epoch_reconstruction):
            epoch_reasons.append("nonfinite_recon")
        else:
            best_reconstruction_so_far = min(
                best_reconstruction_so_far, epoch_reconstruction
            )
            if (
                best_reconstruction_so_far > 0.0
                and epoch_reconstruction / best_reconstruction_so_far > 1.75
            ):
                epoch_reasons.append("reconstruction_degradation")
        if not math.isfinite(epoch_posterior) or not (0.03 <= epoch_posterior <= 2.0):
            epoch_reasons.append("posterior_outside_[0.03,2]")
        if math.isfinite(epoch_floor) and epoch_floor > 0.25:
            epoch_reasons.append("logvar_floor_saturation")
        if math.isfinite(epoch_score_clip) and epoch_score_clip > 0.75:
            epoch_reasons.append("persistent_score_clipping")
        if epoch_number >= 11 and epoch_reasons:
            first_warning_epoch = epoch_number
            first_warning_reasons = ";".join(epoch_reasons)
            break
    summary["first_transition_warning_epoch"] = first_warning_epoch
    summary["first_transition_warning_reasons"] = first_warning_reasons
    grad_csem_encoder = summary["grad_csem_to_encoder_transition"]
    grad_recon_encoder = summary["grad_recon_to_encoder_transition"]
    summary["grad_csem_to_recon_encoder_ratio_transition"] = (
        grad_csem_encoder / grad_recon_encoder
        if math.isfinite(grad_csem_encoder)
        and math.isfinite(grad_recon_encoder)
        and grad_recon_encoder > 0.0
        else math.nan
    )
    grad_csem_vae = summary["grad_csem_to_vae_transition"]
    grad_kl_vae = summary["grad_kl_to_vae_transition"]
    summary["grad_csem_to_kl_vae_ratio_transition"] = (
        grad_csem_vae / grad_kl_vae
        if math.isfinite(grad_csem_vae)
        and math.isfinite(grad_kl_vae)
        and grad_kl_vae > 0.0
        else math.nan
    )

    violations: list[str] = []
    if last_epoch < 40:
        violations.append("incomplete")
    if not math.isfinite(recon_final):
        violations.append("nonfinite_recon")
    if not math.isfinite(posterior_final) or not (0.03 <= posterior_final <= 2.0):
        violations.append("posterior_final_outside_[0.03,2]")
    floor_max = summary["logvar_floor_fraction_max"]
    if math.isfinite(floor_max) and floor_max > 0.25:
        violations.append("logvar_floor_saturation")
    if math.isfinite(recon_degradation) and recon_degradation > 1.75:
        violations.append("reconstruction_degradation")
    transition_clip = summary["score_clip_rate_transition"]
    if math.isfinite(transition_clip) and transition_clip > 0.75:
        violations.append("persistent_score_clipping")
    summary["stability_pass"] = not violations
    summary["stability_violations"] = ";".join(violations)

    evaluation: pd.DataFrame | None = None
    if eval_path is not None:
        evaluation = pd.read_csv(eval_path)
        if not evaluation.empty:
            if "tag" in evaluation:
                lsi_rows = evaluation[evaluation["tag"].astype(str) == "LSI_Diff"]
                eval_row = lsi_rows.iloc[-1] if not lsi_rows.empty else evaluation.iloc[-1]
            else:
                eval_row = evaluation.iloc[-1]
            summary.update(
                {
                    "eval_path": str(eval_path),
                    "fid_vae_recon": finite_float(eval_row.get("fid_vae_recon")),
                    "kid_vae_recon": finite_float(eval_row.get("kid_vae_recon")),
                    "fid_heun": first_prefixed(eval_row, "fid_heun_"),
                    "kid_heun": first_prefixed(eval_row, "kid_heun_"),
                    "fid_rk4": first_prefixed(eval_row, "fid_rk4_"),
                    "kid_rk4": first_prefixed(eval_row, "kid_rk4_"),
                    "lsi_gap": finite_float(eval_row.get("lsi_gap_unet_uncond")),
                }
            )
    return summary, loss, evaluation


def percentile_cost(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranks = numeric.rank(method="average", pct=True, ascending=True)
    return ranks.fillna(1.0)


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["posterior_penalty"] = (
        (pd.to_numeric(output["posterior_var_final"], errors="coerce") < 0.03)
        | (pd.to_numeric(output["posterior_var_final"], errors="coerce") > 2.0)
    ).astype(float)
    balanced_metrics = {
        "fid_vae_recon": 0.25,
        "fid_heun": 0.20,
        "recon_final": 0.10,
        "recon_degradation_ratio": 0.10,
        "logvar_floor_fraction_max": 0.10,
        "score_clip_rate_transition": 0.075,
        "score_mse_unweighted_final": 0.075,
        "posterior_penalty": 0.10,
    }
    stability_metrics = {
        "recon_degradation_ratio": 0.20,
        "logvar_floor_fraction_max": 0.25,
        "score_clip_rate_transition": 0.15,
        "vae_clip_rate_transition": 0.10,
        "posterior_penalty": 0.20,
        "score_mse_unweighted_final": 0.10,
    }
    quality_metrics = {
        "fid_vae_recon": 0.45,
        "fid_heun": 0.35,
        "fid_rk4": 0.20,
    }
    trajectory_eligible = output["trajectory_complete"].astype(bool)
    fid_recon_series = (
        output["fid_vae_recon"]
        if "fid_vae_recon" in output
        else pd.Series(np.nan, index=output.index)
    )
    quality_eligible = trajectory_eligible & pd.to_numeric(
        fid_recon_series, errors="coerce"
    ).notna()
    for score_name, metrics, eligible in (
        ("balanced_score", balanced_metrics, quality_eligible),
        ("stability_score", stability_metrics, trajectory_eligible),
        ("quality_score", quality_metrics, quality_eligible),
    ):
        score = pd.Series(np.nan, index=output.index, dtype=float)
        score.loc[eligible] = 0.0
        weight_total = 0.0
        for metric, weight in metrics.items():
            if metric not in output:
                continue
            score.loc[eligible] += weight * percentile_cost(
                output.loc[eligible, metric]
            )
            weight_total += weight
        output[score_name] = score / max(weight_total, 1e-12)
    output["balanced_rank"] = output["balanced_score"].rank(method="min")
    output["stability_rank"] = output["stability_score"].rank(method="min")
    output["quality_rank"] = output["quality_score"].rank(method="min")
    return output


def factor_effects(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    factor_columns = [column for column in summary if column.startswith("factor__")]
    metrics = [
        "balanced_score",
        "stability_score",
        "fid_vae_recon",
        "fid_heun",
        "recon_final",
        "posterior_var_final",
        "logvar_floor_fraction_max",
        "score_clip_rate_transition",
        "grad_csem_to_recon_encoder_ratio_transition",
        "grad_csem_to_kl_vae_ratio_transition",
        "grad_kl_csem_cosine_transition",
    ]
    pairwise = summary[
        (summary["design_source"] == "pairwise")
        & summary["trajectory_complete"].astype(bool)
    ]
    for arm in sorted(pairwise["arm"].dropna().unique()):
        arm_frame = pairwise[pairwise["arm"] == arm]
        for column in factor_columns:
            for value, group in arm_frame.groupby(column, dropna=True):
                if value == "":
                    continue
                row: dict[str, Any] = {
                    "arm": arm,
                    "factor": column.removeprefix("factor__"),
                    "value": value,
                    "count": len(group),
                    "stability_pass_rate": float(group["stability_pass"].mean()),
                }
                for metric in metrics:
                    row[f"mean__{metric}"] = safe_stat(group[metric], "mean")
                rows.append(row)
    return pd.DataFrame(rows)


def paired_arm_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Return within-seed terminal-anchor minus normalization contrasts."""
    metrics = [
        "balanced_score",
        "stability_score",
        "quality_score",
        "fid_vae_recon",
        "fid_heun",
        "fid_rk4",
        "recon_final",
        "recon_degradation_ratio",
        "posterior_var_final",
        "posterior_var_min",
        "posterior_var_max",
        "logvar_floor_fraction_max",
        "score_mse_unweighted_final",
        "score_clip_rate_transition",
        "vae_clip_rate_transition",
        "grad_recon_csem_encoder_cosine_transition",
        "grad_csem_to_encoder_transition",
        "grad_csem_to_recon_encoder_ratio_transition",
        "grad_kl_csem_cosine_transition",
        "grad_csem_to_kl_vae_ratio_transition",
        "score_weight_ess_fraction_final",
        "first_transition_warning_epoch",
    ]
    rows: list[dict[str, Any]] = []
    for run_id, group in summary.groupby("run_id", sort=False):
        indexed = group.set_index("arm", drop=False)
        if not {"terminal_kl", "norm"}.issubset(indexed.index):
            continue
        terminal = indexed.loc["terminal_kl"]
        norm = indexed.loc["norm"]
        row: dict[str, Any] = {
            "run_id": run_id,
            "run_index": terminal.get("run_index"),
            "design_source": terminal.get("design_source"),
            "design_label": terminal.get("design_label"),
            "both_trajectories_complete": bool(
                terminal.get("trajectory_complete", False)
                and norm.get("trajectory_complete", False)
            ),
            "both_stability_pass": bool(
                terminal.get("stability_pass", False)
                and norm.get("stability_pass", False)
            ),
            "canonical_coefficient_sync": terminal.get(
                "canonical_coefficient_sync", False
            ),
        }
        for column in summary.columns:
            if column.startswith("arg__") or column.startswith("factor__"):
                row[column] = terminal.get(column)
        for metric in metrics:
            terminal_value = finite_float(terminal.get(metric))
            norm_value = finite_float(norm.get(metric))
            row[f"terminal_kl__{metric}"] = terminal_value
            row[f"norm__{metric}"] = norm_value
            row[f"terminal_minus_norm__{metric}"] = (
                terminal_value - norm_value
                if math.isfinite(terminal_value) and math.isfinite(norm_value)
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(summary: pd.DataFrame, trajectories: pd.DataFrame, output_dir: Path) -> None:
    matplotlib_config = output_dir / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
    except Exception as exception:
        print(f"plotting unavailable: {exception}")
        return

    for arm in sorted(summary["arm"].dropna().unique()):
        arm_summary = summary[summary["arm"] == arm].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        color = pd.to_numeric(arm_summary.get("arg__csem-w"), errors="coerce")
        scatter = axes[0].scatter(
            arm_summary["posterior_var_final"],
            arm_summary["fid_vae_recon"],
            c=color,
            cmap="viridis",
            alpha=0.8,
        )
        axes[0].set_xlabel("posterior variance at epoch 40")
        axes[0].set_ylabel("reconstruction FID")
        axes[0].set_title(f"{arm}: representation quality")
        fig.colorbar(scatter, ax=axes[0], label="CSEM coefficient")

        axes[1].scatter(
            arm_summary["logvar_floor_fraction_max"],
            arm_summary["recon_degradation_ratio"],
            c=color,
            cmap="viridis",
            alpha=0.8,
        )
        axes[1].set_xlabel("maximum logvar-floor fraction")
        axes[1].set_ylabel("final / best reconstruction loss")
        axes[1].set_title(f"{arm}: transient pathology")
        fig.tight_layout()
        fig.savefig(output_dir / f"screening_scatter_{arm}.png", dpi=180)
        plt.close(fig)

        top_ids = (
            arm_summary.sort_values("balanced_score").head(10)["run_id"].tolist()
        )
        arm_trajectories = trajectories[
            (trajectories["arm"] == arm) & trajectories["run_id"].isin(top_ids)
        ]
        if arm_trajectories.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for run_id, group in arm_trajectories.groupby("run_id"):
            group = group.sort_values("epoch")
            short = run_id[:18]
            axes[0].plot(group["epoch"], group["recon"], label=short)
            axes[1].plot(group["epoch"], group["posterior_var"], label=short)
            axes[2].plot(
                group["epoch"], group["score_mse_unweighted"], label=short
            )
        axes[0].set_title("reconstruction loss")
        axes[1].set_title("posterior variance")
        axes[2].set_title("raw score MSE")
        for axis in axes:
            axis.set_xlabel("epoch")
            axis.grid(alpha=0.25)
        axes[2].set_yscale("log")
        axes[2].legend(fontsize=6, loc="best")
        fig.suptitle(f"Top balanced configurations: {arm}")
        fig.tight_layout()
        fig.savefig(output_dir / f"top_trajectories_{arm}.png", dpi=180)
        plt.close(fig)


def write_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        lines_local = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in frame.iterrows():
            rendered: list[str] = []
            for column in frame.columns:
                value = row[column]
                if pd.isna(value):
                    text = ""
                elif isinstance(value, float):
                    text = f"{value:.6g}"
                else:
                    text = str(value)
                rendered.append(text.replace("|", "\\|"))
            lines_local.append("| " + " | ".join(rendered) + " |")
        return "\n".join(lines_local)

    lines = ["# CSEM 40-epoch sweep summary", ""]
    state_counts = summary.groupby(["arm", "worker_state"]).size()
    lines.extend(("## Completion", "", state_counts.to_string(), ""))
    selected_columns = [
        "balanced_rank",
        "run_id",
        "design_label",
        "stability_pass",
        "balanced_score",
        "fid_vae_recon",
        "fid_heun",
        "fid_rk4",
        "recon_final",
        "posterior_var_final",
        "logvar_floor_fraction_max",
        "score_clip_rate_transition",
        "arg__csem-w",
        "arg__terminal-kl-w",
        "arg__T-terminal",
        "canonical_coefficient_sync",
        "arg__lr-encoder",
        "arg__lr-decoder",
        "arg__lr-score-head",
        "arg__lr-tracking-head",
        "arg__adam-beta2",
        "arg__encoder-warmup-mode",
        "arg__encoder-score-warmup-epochs",
        "arg__csem-ramp-epochs",
    ]
    for arm in sorted(summary["arm"].dropna().unique()):
        lines.extend((f"## Top 15: {arm}", ""))
        leaderboard = summary[
            (summary["arm"] == arm) & summary["trajectory_complete"].astype(bool)
        ].sort_values("balanced_score")
        available = [column for column in selected_columns if column in leaderboard]
        lines.append(markdown_table(leaderboard[available].head(15)))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    args = parser.parse_args()
    sweep_root = args.sweep_root.resolve()
    manifest = read_json(sweep_root / "run_manifest.json")
    output_dir = sweep_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    trajectory_frames: list[pd.DataFrame] = []
    evaluation_frames: list[pd.DataFrame] = []
    for record in manifest["runs"]:
        status = read_status(record)
        requested_arms = [
            arm.strip()
            for arm in str(record["args"].get("arms", "terminal_kl,norm")).split(",")
            if arm.strip()
        ]
        for arm in requested_arms:
            summary, loss, evaluation = summarize_arm(record, arm, status)
            summaries.append(summary)
            metadata = {
                "run_id": record["run_id"],
                "design_label": record["design_label"],
                "design_source": record["design_source"],
                "arm": arm,
            }
            if loss is not None and not loss.empty:
                loss = loss.copy()
                for key, value in metadata.items():
                    loss[key] = value
                trajectory_frames.append(loss)
            if evaluation is not None and not evaluation.empty:
                evaluation = evaluation.copy()
                for key, value in metadata.items():
                    evaluation[key] = value
                evaluation_frames.append(evaluation)

    summary_frame = pd.DataFrame(summaries)
    scored_frames: list[pd.DataFrame] = []
    for arm, group in summary_frame.groupby("arm", dropna=False):
        scored_frames.append(add_scores(group))
    summary_frame = pd.concat(scored_frames, ignore_index=True)
    summary_frame = summary_frame.sort_values(["arm", "balanced_score", "run_id"])
    summary_frame.to_csv(output_dir / "all_run_arm_summary.csv", index=False)

    for arm in sorted(summary_frame["arm"].dropna().unique()):
        arm_frame = summary_frame[summary_frame["arm"] == arm].sort_values(
            "balanced_score"
        )
        arm_frame.to_csv(output_dir / f"leaderboard_{arm}.csv", index=False)

    trajectories = (
        pd.concat(trajectory_frames, ignore_index=True)
        if trajectory_frames
        else pd.DataFrame()
    )
    evaluations = (
        pd.concat(evaluation_frames, ignore_index=True)
        if evaluation_frames
        else pd.DataFrame()
    )
    trajectories.to_csv(output_dir / "all_trajectories.csv", index=False)
    evaluations.to_csv(output_dir / "all_evaluations.csv", index=False)
    factor_effects(summary_frame).to_csv(
        output_dir / "factor_effects.csv", index=False
    )
    paired_arm_deltas(summary_frame).to_csv(
        output_dir / "paired_arm_deltas.csv", index=False
    )
    write_markdown(summary_frame, output_dir / "README.md")
    if not trajectories.empty:
        save_plots(summary_frame, trajectories, output_dir)

    completed = int((summary_frame["last_epoch"] >= 40).sum())
    print(f"compiled {len(summary_frame)} run-arm rows; {completed} reached epoch 40")
    print(f"summary: {output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
