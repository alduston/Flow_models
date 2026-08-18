#!/usr/bin/env python3
"""Plot the compiled CIFAR fine sweep using matplotlib defaults."""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
COMPILED = "cifar_TKxCSEM_fine_T1p75_v1_compiled"

def plot_metric(df, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for cw, g in df.groupby("csem_w"):
        g = g.sort_values("T_K")
        ax.plot(g["T_K"], g[metric], marker="o", label=f"csem_w={cw:g}")
    ax.set_xlabel("T_K")
    ax.set_ylabel(ylabel)
    ax.set_title(f"CIFAR fine sweep, T=1.75: {ylabel}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    args = ap.parse_args()

    root = args.base_dir.resolve() / COMPILED
    path = root / "fine_sweep_summary.csv"
    if not path.is_file():
        raise SystemExit(f"Missing compiled summary: {path}")
    df = pd.read_csv(path)

    plot_metric(
        df, "fid_gaussian_T", "Gaussian-start RK4 FID",
        root / "fid_gaussian_vs_TK.png",
    )
    plot_metric(
        df, "fid_oracle_qTK", "Oracle-q_TK RK4 FID",
        root / "fid_oracle_vs_TK.png",
    )
    plot_metric(
        df, "fid_recon", "Reconstruction FID",
        root / "fid_recon_vs_TK.png",
    )
    plot_metric(
        df, "fid_gaussian_minus_oracle", "Gaussian minus oracle FID",
        root / "fid_gap_vs_TK.png",
    )
    print("Wrote plots to", root)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
