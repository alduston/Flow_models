#!/usr/bin/env python3
"""Generate one-figure-at-a-time diagnostic surfaces for fmnist_csem_metric_parity_v1."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep")
COMPILED = "fmnist_csem_metric_parity_v1_compiled"


def heatmap(df, value_col, title, ylabel, out):
    can = df[df["role"] == "canonical_grid"].copy()
    piv = can.pivot(index="csem_w", columns="terminal_kl_w", values=value_col)
    fig, ax = plt.subplots(figsize=(7.5,5.5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [f"{x:g}" for x in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{x:g}" for x in piv.index])
    ax.set_xlabel("terminal_kl_w")
    ax.set_ylabel("csem_w")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=ylabel)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.iloc[i,j]
            ax.text(j, i, f"{v:.3g}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    args = ap.parse_args()
    root = args.base_dir/COMPILED
    surface = pd.read_csv(root/"metric_parity_surface.csv")
    ranking = pd.read_csv(root/"canonical_parity_ranking.csv")
    out = root/"plots"
    out.mkdir(parents=True, exist_ok=True)

    heatmap(surface, "fid_vae_recon", "Canonical reconstruction FID", "FID", out/"recon_fid.png")
    heatmap(surface, "terminal_component_kl_fulltrain", "Canonical terminal component KL", "KL", out/"terminal_kl.png")
    heatmap(surface, "terminal_qT_vs_gaussian_sw2", "Canonical qT vs Gaussian SW2", "SW2", out/"terminal_sw2.png")

    merged = surface.merge(
        ranking[["config_id","parity_score","mean_gen_fid"]],
        on="config_id", how="left"
    )
    heatmap(merged, "parity_score", "Joint reconstruction + terminal parity score", "score", out/"parity_score.png")
    heatmap(merged, "mean_gen_fid", "Generation FID after decoupling weights", "mean Heun/RK4 FID", out/"mean_generation_fid.png")

    print("Wrote plots to", out)


if __name__ == "__main__":
    main()
