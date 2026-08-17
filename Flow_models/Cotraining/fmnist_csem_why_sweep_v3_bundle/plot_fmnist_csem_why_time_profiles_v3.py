#!/usr/bin/env python3
"""Plot the canonical-vs-unweighted exact score-error profiles from WHY-v2."""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_BASE = Path(
    "/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
)


def one_plot(df, pair_id, metric, ylabel, out):
    g = df[df["pair_id"].astype(str).str.zfill(2) == f"{pair_id:02d}"].sort_values("t")
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g["t"], g[f"{metric}_U"], marker="o", ms=3, label="unweighted outer")
    ax.plot(g["t"], g[f"{metric}_C"], marker="o", ms=3, label="canonical outer")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("OU time t")
    ax.set_ylabel(ylabel)
    lr = g["lr_score_head"].iloc[0] if "lr_score_head" in g else float("nan")
    ax.set_title(f"pair {pair_id} | score-head LR={lr:.1e}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = ap.parse_args()

    root = args.base_dir / "fmnist_csem_why_compiled_v3"
    path = root / "oracle_score_time_profile_pair_wide.csv"
    df = pd.read_csv(path)
    outdir = root / "time_profile_plots_v3"
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("uncond_learned_oracle_score", "Unconditional learned-vs-oracle score MSE"),
        ("uncond_intrinsic_var_score", "Unconditional intrinsic CSEM variance"),
        ("cond_learned_oracle_score", "Conditional learned-vs-oracle score MSE"),
        ("guided_learned_oracle_score", "CFG=3 learned-vs-oracle score MSE"),
    ]
    for pair_id in range(4):
        for metric, ylabel in metrics:
            one_plot(
                df, pair_id, metric, ylabel,
                outdir / f"pair{pair_id}_{metric}_v3.png"
            )
    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
