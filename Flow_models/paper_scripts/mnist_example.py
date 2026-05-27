#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST PCA Bayesian logistic-regression LFGI benchmark (standalone).

Purpose
-------
A minimal non-toy benchmark for the LFGI finite-sample story:
  * target is a real-data Bayesian logistic-regression posterior in d=32,
  * negative log-posterior Hessian is PSD everywhere,
  * weak prior + likelihood temperature induce a singular/ill-conditioned score geometry,
  * compare Tweedie, Scalar Blend, moment Matrix Blend, and LFGI using the same OU/SNIS machinery as the main harness.

Default task: MNIST 4-vs-9 classification, anisotropically measured PCA features, d=32.

Outputs
-------
  out_dir/
    metrics.csv
    diagnostic_summary.txt
    mnist_pca_logreg_dashboard.pdf
    heatmaps.png
    metric_bars.png
    score_rmse_vs_t.png
    hessian_spectrum.png
    measurement_operator.png
    energy_hist.png
    predictive_bars.png
    samples_<method>.npy

Repeated-run mode (--n_runs K) writes:
    run_000/, run_001/, ...              # ordinary single-run outputs
    per_run_metrics.csv                  # long-form metrics for every run/method
    aggregate_metrics.csv                # mean/std/SEM/n for every metric
    mnist_table_aggregate.tex            # paper-table fragment with uncertainty
    aggregate_summary.txt

Typical Colab usage
-------------------
  !python mnist_pca_logreg_lfgi_benchmark.py \
      --out_dir Figs/mnist_pca_logreg_d32 \
      --device cuda --dtype float64 \
      --classes 4 9 --feature_operator spectral \
      --operator_stiff_rank 6 --operator_mid_rank 10 --operator_stiff_scale 3.0 \
      --operator_mid_scale 1.0 --operator_sloppy_scale 0.05 \
      --n_ref 512 --n_gate 512 --n_gen 8000 --n_test_ref 12000

Notes
-----
The posterior is
    log p(theta | X,y) = - ||theta||^2/(2 tau^2)
                        - beta * mean_i softplus(-y_i x_i^T theta).
The mean likelihood convention makes beta a dataset-size-independent stiffness knob.
Increasing beta and tau produces an increasingly singular score geometry while preserving
PSD Hessians.  The optional anisotropic measurement operator B further creates
tightly measured, weakly measured, and effectively unmeasured directions:
    P(theta) = tau^{-2} I + beta/n B^T X^T W(theta) X B >= 0.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, List

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------------------------------------------------------
# Publication plotting style
# -----------------------------------------------------------------------------

PUB_DPI = 450

# Direct moment-normal-equation matrix-gate baseline.
# This is the primal/moment estimator for the optimal operator-valued blend,
# used as the finite-sample comparator against the Hessian-resolvent LFGI gate.
MOMENT_MATRIX_RIDGE = float(os.environ.get("LFGI_MOMENT_MATRIX_RIDGE", "1e-8"))
MOMENT_MATRIX_RIDGE_REL = float(os.environ.get("LFGI_MOMENT_MATRIX_RIDGE_REL", "1e-6"))
MOMENT_MATRIX_CHUNK = int(os.environ.get("LFGI_MOMENT_MATRIX_CHUNK", "64"))

# Requested paper convention: TWEEDIE=red, SCALAR BLEND=blue, MATRIX BLEND=purple, LFGI=green.
METHOD_COLORS = {
    "tweedie": "#D62728",
    "blend": "#1F77B4",
    "matrix-blend": "#9467BD",
    "lfgi": "#2CA02C",
    "ce-hlsi": "#2CA02C",
    "ce_hlsi": "#2CA02C",
    "reference": "#4D4D4D",
}
METHOD_MARKERS = {
    "tweedie": "o",
    "blend": "s",
    "matrix-blend": "^",
    "lfgi": "D",
    "ce-hlsi": "D",
    "ce_hlsi": "D",
}


def apply_publication_style() -> None:
    """Readable-after-scaling defaults for arXiv paper figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": PUB_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.25,
        "lines.markersize": 5.6,
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def method_key(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    if key in {"ce-hlsi", "lfgi"}:
        return "lfgi"
    if key.startswith("ce-hlsi") or key.startswith("lfgi"):
        return "lfgi"
    if key in {"moment-matrix-blend", "matrix-blend", "primal-matrix-blend", "moment-blend"}:
        return "matrix-blend"
    if key.startswith("moment-matrix-blend") or key.startswith("matrix-blend") or key.startswith("primal-matrix-blend"):
        return "matrix-blend"
    if key.startswith("tweedie"):
        return "tweedie"
    if key.startswith("blend") or key.startswith("scalar-blend"):
        return "blend"
    if key == "reference":
        return "reference"
    return key


def method_label(name: str) -> str:
    raw = str(name)
    key = method_key(raw)
    base = {
        "tweedie": "TWEEDIE",
        "blend": "SCALAR BLEND",
        "matrix-blend": "MATRIX BLEND",
        "lfgi": "LFGI",
        "reference": "REFERENCE",
    }.get(key, raw.replace("_", "-").upper())
    # Keep optional bank-ablation annotations readable if those diagnostics are enabled.
    low = raw.lower().replace("_", "-")
    if "xg-bank" in low:
        return base + " (FULL BANK)"
    if "xr-gate" in low:
        return base + " (SCORE GATE)"
    if "nr" in low and "ng" in low:
        return base
    return base


def method_color(name: str) -> str:
    return METHOD_COLORS.get(method_key(name), "#7F7F7F")


def method_marker(name: str) -> str:
    return METHOD_MARKERS.get(method_key(name), "o")



def _lfgi_scale_names(names) -> List[str]:
    """Methods whose arrays/metrics should define visual scales.

    Paper-facing figures should not let a failed comparator (usually scalar or
    moment matrix blend) set axis limits or colorbars.  Whenever LFGI is present,
    its finite results define the visualization scale; otherwise we fall back to
    all available names so standalone diagnostic calls still work.
    """
    names = list(names)
    lfgi = [m for m in names if method_key(m) == "lfgi"]
    return lfgi if lfgi else names


def _finite_values_np(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    return vals[np.isfinite(vals)]


def _finite_concat_np(values_list) -> np.ndarray:
    chunks = [_finite_values_np(v) for v in values_list]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks)


def _padded_limits_from_values(
    values,
    *,
    q_low: float = 1.0,
    q_high: float = 99.0,
    pad_frac: float = 0.08,
    positive: bool = False,
    include_zero: bool = False,
    symmetric: bool = False,
    fallback: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    vals = _finite_values_np(values)
    if positive:
        vals = vals[vals > 0]
    if vals.size == 0:
        return fallback
    if symmetric:
        hi = float(np.nanpercentile(np.abs(vals), q_high))
        hi = max(hi, 1e-12)
        return -hi, hi
    lo = float(np.nanpercentile(vals, q_low))
    hi = float(np.nanpercentile(vals, q_high))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if positive:
        lo = max(lo, np.min(vals[vals > 0]) if np.any(vals > 0) else 1e-12)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return fallback
    if hi - lo < 1e-12:
        center = 0.5 * (hi + lo)
        width = max(abs(center) * 0.10, 1e-3)
        lo, hi = center - width, center + width
    pad = pad_frac * (hi - lo)
    lo, hi = lo - pad, hi + pad
    if positive:
        lo = max(lo, 1e-12)
    return float(lo), float(hi)


def _set_ylim_from_anchor(ax: plt.Axes, anchor_values, *, log: bool = False, include_zero: bool = True) -> None:
    vals = _finite_values_np(anchor_values)
    if log:
        vals = vals[vals > 0]
        if vals.size:
            lo, hi = _padded_limits_from_values(vals, q_low=1.0, q_high=99.0, pad_frac=0.12, positive=True, include_zero=False, fallback=(vals.min(), vals.max()))
            ax.set_ylim(lo, hi)
        return
    if vals.size:
        lo, hi = _padded_limits_from_values(vals, q_low=0.0, q_high=100.0, pad_frac=0.12, include_zero=include_zero)
        ax.set_ylim(lo, hi)


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, which="major", alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=4.0, width=0.9)


def save_pub(fig: plt.Figure, out_path: Path) -> None:
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.35)
    fig.savefig(out_path, dpi=PUB_DPI, facecolor="white", transparent=False)
    plt.close(fig)


apply_publication_style()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in {"float64", "fp64", "double"}:
        return torch.float64
    if name in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError(f"unknown dtype {name!r}")


def resolve_device(name: str) -> torch.device:
    name = name.lower().strip()
    if name in {"cuda", "gpu"}:
        name = "cuda:0"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU.")
        name = "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def at(t: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), dtype=torch.get_default_dtype())
    return torch.exp(-t)


def vt(t: torch.Tensor | float) -> torch.Tensor:
    a = at(t)
    return (1.0 - a * a).clamp_min(1e-12)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p) - np.log1p(-p)


# -----------------------------------------------------------------------------
# MNIST PCA feature construction
# -----------------------------------------------------------------------------

def _load_mnist_with_torchvision(root: str, classes: Tuple[int, int], max_train: int, max_test: int):
    from torchvision.datasets import MNIST
    train_ds = MNIST(root=root, train=True, download=True)
    test_ds = MNIST(root=root, train=False, download=True)

    def extract(ds, max_n):
        X = ds.data.float().reshape(len(ds.data), -1) / 255.0
        y_raw = ds.targets.long()
        mask = (y_raw == classes[0]) | (y_raw == classes[1])
        X = X[mask]
        y_raw = y_raw[mask]
        y = torch.where(y_raw == classes[1], torch.ones_like(y_raw), -torch.ones_like(y_raw)).float()
        if max_n and len(X) > max_n:
            g = torch.Generator().manual_seed(12345 + int(max_n))
            idx = torch.randperm(len(X), generator=g)[:max_n]
            X, y = X[idx], y[idx]
        return X, y

    return extract(train_ds, max_train), extract(test_ds, max_test)



def _anisotropic_mnist_observation(X: torch.Tensor, strength: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """A tiny Navier--Stokes-inspired sensor/blur operator for MNIST images.

    This is optional and deliberately simple: retain a few vertical strips and
    two cross rows after a mild anisotropic blur.  It creates a real-data
    observation geometry before PCA, analogous to sparse/anisotropic sensors.
    """
    import torch.nn.functional as Fnn

    imgs = X.reshape(-1, 1, 28, 28)
    # Separable anisotropic smoothing: stronger vertical than horizontal blur.
    ky = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=imgs.dtype, device=imgs.device)
    ky = ky / ky.sum()
    kx = torch.tensor([1.0, 2.0, 1.0], dtype=imgs.dtype, device=imgs.device)
    kx = kx / kx.sum()
    blur_y = ky.reshape(1, 1, 5, 1)
    blur_x = kx.reshape(1, 1, 1, 3)
    imgs_blur = Fnn.conv2d(Fnn.pad(imgs, (0, 0, 2, 2), mode="replicate"), blur_y)
    imgs_blur = Fnn.conv2d(Fnn.pad(imgs_blur, (1, 1, 0, 0), mode="replicate"), blur_x)

    mask = torch.zeros(28, 28, dtype=imgs.dtype, device=imgs.device)
    cols = [10, 14, 18]
    rows = [9, 18]
    for c in cols:
        mask[:, max(0, c - 1):min(28, c + 2)] = 1.0
    for r in rows:
        mask[max(0, r - 1):min(28, r + 2), :] = 1.0
    # Keep a weak global low-pass background so classification is not pure pixels-on-a-mask.
    observed = strength * imgs_blur * mask.reshape(1, 1, 28, 28) + (1.0 - strength) * imgs_blur
    return observed.reshape(len(X), -1), mask.detach().cpu()


def build_feature_operator(
    d: int,
    mode: str,
    stiff_rank: int,
    mid_rank: int,
    stiff_scale: float,
    mid_scale: float,
    sloppy_scale: float,
    random_rotate: bool,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a right-acting feature operator B and its prescribed singular values.

    Features are transformed as X <- X B.  For the logistic posterior this gives
        P(theta)=tau^{-2}I + beta/n B^T X^T W X B,
    so the Hessian remains PSD while B controls anisotropy.
    """
    mode = str(mode).lower().strip()
    if mode in {"none", "identity", "off"}:
        scales = torch.ones(d, dtype=dtype, device=device)
        return torch.eye(d, dtype=dtype, device=device), scales
    if mode != "spectral":
        raise ValueError(f"unknown feature_operator={mode!r}; use none or spectral")

    stiff_rank = max(0, min(int(stiff_rank), d))
    mid_rank = max(0, min(int(mid_rank), d - stiff_rank))
    sloppy_rank = d - stiff_rank - mid_rank
    scales = torch.cat([
        torch.full((stiff_rank,), float(stiff_scale), dtype=dtype, device=device),
        torch.full((mid_rank,), float(mid_scale), dtype=dtype, device=device),
        torch.full((sloppy_rank,), float(sloppy_scale), dtype=dtype, device=device),
    ])
    if random_rotate:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
        A = torch.randn(d, d, generator=gen, dtype=dtype, device="cpu").to(device)
        Q, _ = torch.linalg.qr(A)
        B = Q @ torch.diag(scales) @ Q.T
    else:
        B = torch.diag(scales)
    return B.contiguous(), scales


def feature_covariance_stats(X: torch.Tensor, prefix: str) -> Dict[str, float]:
    Xc = X - X.mean(dim=0, keepdim=True)
    C = (Xc.T @ Xc) / max(int(Xc.shape[0]) - 1, 1)
    evals = torch.linalg.eigvalsh(sym(C)).clamp_min(1e-30)
    p = evals / evals.sum().clamp_min(1e-30)
    effrank = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum())
    return {
        f"{prefix}_cov_lam_min": float(evals.min().detach().cpu()),
        f"{prefix}_cov_lam_max": float(evals.max().detach().cpu()),
        f"{prefix}_cov_cond": float((evals.max() / evals.min().clamp_min(1e-30)).detach().cpu()),
        f"{prefix}_cov_effrank": float(effrank.detach().cpu()),
    }


def plot_measurement_operator(data: Dict[str, torch.Tensor], out_dir: Path) -> None:
    """Plot prescribed operator spectrum and feature covariance spectra."""
    B_scales = data.get("feature_operator_scales")
    X_pre = data.get("X_train_pre_operator")
    X_post = data.get("X_train")
    mask = data.get("image_sensor_mask")

    ncols = 4 if mask is not None else 3
    fig, axs = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.4))
    if ncols == 3:
        ax0, ax1, ax2 = axs
    else:
        axm, ax0, ax1, ax2 = axs
        axm.imshow(mask.numpy(), cmap="gray_r", origin="upper")
        axm.set_title("image sensor mask")
        axm.axis("off")

    if B_scales is not None:
        ax0.plot(np.arange(1, len(B_scales) + 1), B_scales.detach().cpu().numpy(), marker="o", ms=3)
        ax0.set_yscale("log")
        ax0.set_title("feature operator scales")
        ax0.set_xlabel("index")
        ax0.set_ylabel("scale")
        ax0.grid(alpha=0.25)

    def cov_eigs(X):
        Xc = X - X.mean(dim=0, keepdim=True)
        C = (Xc.T @ Xc) / max(int(Xc.shape[0]) - 1, 1)
        return torch.linalg.eigvalsh(sym(C)).clamp_min(1e-30).detach().cpu().numpy()[::-1]

    if X_pre is not None and X_post is not None:
        e_pre = cov_eigs(X_pre)
        e_post = cov_eigs(X_post)
        xs = np.arange(1, len(e_pre) + 1)
        ax1.plot(xs, e_pre, marker="o", ms=3, label="pre")
        ax1.plot(xs, e_post, marker="o", ms=3, label="post")
        ax1.set_yscale("log")
        ax1.set_title("feature covariance eigenvalues")
        ax1.set_xlabel("index")
        ax1.legend()
        ax1.grid(alpha=0.25)
        ax2.plot(xs, e_post / np.maximum(e_pre, 1e-30), marker="o", ms=3)
        ax2.set_yscale("log")
        ax2.set_title("post/pre covariance ratio")
        ax2.set_xlabel("index")
        ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "measurement_operator.png", dpi=180)
    plt.close(fig)

def load_mnist_pca_features(
    d: int = 32,
    classes: Tuple[int, int] = (4, 9),
    root: str = "./data",
    max_train: int = 6000,
    max_test: int = 2000,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    no_whiten: bool = True,
    image_operator: str = "none",
    image_operator_strength: float = 1.0,
    feature_operator: str = "spectral",
    operator_stiff_rank: int = 6,
    operator_mid_rank: int = 10,
    operator_stiff_scale: float = 3.0,
    operator_mid_scale: float = 1.0,
    operator_sloppy_scale: float = 0.05,
    operator_random_rotate: bool = True,
    operator_seed: int = 123,
) -> Dict[str, torch.Tensor]:
    """Load MNIST, keep two classes, and return PCA features in R^d.

    We deliberately do *not* whiten by default.  Keeping the empirical PCA scale
    preserves low-effective-rank feature geometry, which helps create an
    ill-conditioned but PSD posterior Hessian.
    """
    try:
        (Xtr, ytr), (Xte, yte) = _load_mnist_with_torchvision(root, classes, max_train, max_test)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load MNIST with torchvision. Install torchvision or run in an environment "
            "with dataset download access. Original error: " + repr(exc)
        )

    image_sensor_mask = None
    image_operator = str(image_operator).lower().strip()
    if image_operator in {"anisotropic_mask", "sensor", "navier_mask"}:
        Xtr, image_sensor_mask = _anisotropic_mnist_observation(Xtr, strength=float(image_operator_strength))
        Xte, _ = _anisotropic_mnist_observation(Xte, strength=float(image_operator_strength))
    elif image_operator not in {"none", "identity", "off"}:
        raise ValueError(f"unknown image_operator={image_operator!r}; use none or anisotropic_mask")

    # Center using training mean; SVD on CPU is fine for this size.
    X_mean = Xtr.mean(dim=0, keepdim=True)
    Xc = Xtr - X_mean
    # Scale pixel space mildly so dot products are not enormous.
    # This is not PCA whitening; it just normalizes global image energy.
    global_scale = Xc.std().clamp_min(1e-6)
    Xc_scaled = Xc / global_scale
    U, S, Vh = torch.linalg.svd(Xc_scaled, full_matrices=False)
    V = Vh[:d].T.contiguous()

    def project(X):
        Z = ((X - X_mean) / global_scale) @ V
        if not no_whiten:
            # Optional; off by default because it removes the intended feature anisotropy.
            Z = Z / Z.std(dim=0, keepdim=True).clamp_min(1e-6)
        # Center projected train distribution; do not standardize columns by default.
        return Z

    Ztr = project(Xtr)
    z_mean = Ztr.mean(dim=0, keepdim=True)
    Ztr = Ztr - z_mean
    Zte = project(Xte) - z_mean

    Ztr = Ztr.to(device=device, dtype=dtype)
    Zte = Zte.to(device=device, dtype=dtype)
    Ztr_pre = Ztr.clone()
    Zte_pre = Zte.clone()
    B, B_scales = build_feature_operator(
        d=d,
        mode=feature_operator,
        stiff_rank=operator_stiff_rank,
        mid_rank=operator_mid_rank,
        stiff_scale=operator_stiff_scale,
        mid_scale=operator_mid_scale,
        sloppy_scale=operator_sloppy_scale,
        random_rotate=operator_random_rotate,
        seed=operator_seed,
        dtype=dtype,
        device=device,
    )
    Ztr = Ztr @ B
    Zte = Zte @ B

    return {
        "X_train": Ztr,
        "y_train": ytr.to(device=device, dtype=dtype),
        "X_test": Zte,
        "y_test": yte.to(device=device, dtype=dtype),
        "X_train_pre_operator": Ztr_pre,
        "X_test_pre_operator": Zte_pre,
        "feature_operator_matrix": B,
        "feature_operator_scales": B_scales,
        "pca_components": V.to(device=device, dtype=dtype),
        "pixel_global_scale": global_scale.to(device=device, dtype=dtype),
        "image_sensor_mask": image_sensor_mask,
        "pca_singular_values": S[:d].to(dtype=dtype),
        "classes": torch.tensor(classes),
    }


# -----------------------------------------------------------------------------
# Bayesian logistic-regression posterior
# -----------------------------------------------------------------------------

@dataclass
class LogisticPosterior:
    X: torch.Tensor
    y: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    tau: float = 50.0
    beta: float = 50.0
    name: str = "mnist_pca_logreg_d32"

    @property
    def D(self) -> int:
        return int(self.X.shape[1])

    @property
    def prior_prec(self) -> float:
        return 1.0 / (self.tau * self.tau)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        # theta: [B,d]
        z = self.X @ theta.T                              # [n,B]
        yz = self.y[:, None] * z
        nll = F.softplus(-yz).mean(dim=0)                 # [B]
        prior = 0.5 * self.prior_prec * theta.square().sum(dim=1)
        return -(prior + self.beta * nll)

    def energy(self, theta: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(theta)

    def score(self, theta: torch.Tensor) -> torch.Tensor:
        # grad log p = - prior_prec theta + beta/n sum y_i x_i sigmoid(-y_i x_i^T theta)
        z = self.X @ theta.T                              # [n,B]
        yz = self.y[:, None] * z
        r = torch.sigmoid(-yz)                            # [n,B]
        grad_like = (self.X.T @ (self.y[:, None] * r)).T / self.X.shape[0]
        return -self.prior_prec * theta + self.beta * grad_like

    def hessian(self, theta: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        # negative Hessian of log p: prior_prec I + beta/n X^T W X, W=sigmoid(yz)sigmoid(-yz)
        B, d = theta.shape
        X = self.X
        y = self.y
        eye = torch.eye(d, dtype=theta.dtype, device=theta.device)
        out = torch.empty(B, d, d, dtype=theta.dtype, device=theta.device)
        n = X.shape[0]
        for i in range(0, B, chunk):
            th = theta[i:i + chunk]
            z = X @ th.T                                 # [n,b]
            yz = y[:, None] * z
            w = torch.sigmoid(yz) * torch.sigmoid(-yz)    # [n,b]
            # batch X^T diag(w_b) X / n
            H_like = torch.einsum("nb,ni,nj->bij", w, X, X) / n
            out[i:i + chunk] = self.prior_prec * eye + self.beta * H_like
        return sym(out)

    def predictive_probs(self, theta: torch.Tensor, batch: int = 2048) -> torch.Tensor:
        # posterior predictive P(y=+1 | x, data) approximated by sample average sigmoid(x^T theta)
        probs_sum = torch.zeros(self.X_test.shape[0], dtype=theta.dtype, device=theta.device)
        for i in range(0, theta.shape[0], batch):
            logits = self.X_test @ theta[i:i + batch].T
            probs_sum += torch.sigmoid(logits).sum(dim=1)
        return probs_sum / theta.shape[0]

    def predictive_metrics(self, theta: torch.Tensor) -> Dict[str, float]:
        p = self.predictive_probs(theta)
        y01 = (self.y_test > 0).to(p.dtype)
        nll = -(y01 * torch.log(p.clamp_min(1e-12)) + (1 - y01) * torch.log((1 - p).clamp_min(1e-12))).mean()
        pred = torch.where(p >= 0.5, torch.ones_like(self.y_test), -torch.ones_like(self.y_test))
        acc = (pred == self.y_test).to(p.dtype).mean()
        return {"pred_nll": float(nll.detach().cpu()), "pred_acc": float(acc.detach().cpu())}


def find_map_and_metric(target: LogisticPosterior, max_iter: int = 100, verbose: bool = True):
    d = target.D
    theta = torch.zeros(1, d, dtype=target.X.dtype, device=target.X.device, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = target.energy(theta).sum()
        loss.backward()
        return loss

    opt.step(closure)
    theta_map = theta.detach().clone().reshape(d)
    H = target.hessian(theta_map[None, :])[0]
    evals, evecs = torch.linalg.eigh(sym(H))
    evals = evals.clamp_min(1e-10)
    if verbose:
        print(f"[MAP] energy={float(target.energy(theta_map[None,:])[0]):.6g}, "
              f"lambda_min={float(evals.min()):.3e}, lambda_max={float(evals.max()):.3e}, "
              f"cond={float(evals.max()/evals.min()):.3e}")
    return theta_map, evals, evecs


# -----------------------------------------------------------------------------
# Preconditioned MALA reference sampler
# -----------------------------------------------------------------------------

def sample_reference_mala_preconditioned(
    target: LogisticPosterior,
    n: int,
    theta_map: torch.Tensor,
    metric_evals: torch.Tensor,
    metric_evecs: torch.Tensor,
    step_size: float = 0.15,
    n_chains: int = 128,
    burnin: int = 4000,
    thin: int = 10,
    verbose: bool = True,
) -> torch.Tensor:
    """MALA in z coordinates where theta = theta_map + L z, L=H_map^{-1/2}."""
    device, dtype = theta_map.device, theta_map.dtype
    d = theta_map.numel()
    L = metric_evecs @ torch.diag(metric_evals.rsqrt())

    def theta_from_z(z):
        return theta_map[None, :] + z @ L.T

    def lp_score_z(z):
        z_ = z.detach().requires_grad_(True)
        with torch.enable_grad():
            th = theta_from_z(z_)
            lp = target.log_prob(th)
            lp.sum().backward()
        return lp.detach(), z_.grad.detach().clone()

    z = torch.randn(n_chains, d, dtype=dtype, device=device)
    lp, sz = lp_score_z(z)
    samples: List[torch.Tensor] = []
    total = 0
    accepted = 0
    needed_steps = burnin + math.ceil(n / n_chains) * thin + 1
    sqrt_2h = math.sqrt(2.0 * step_size)
    if verbose:
        print(f"[MALA-z] n={n}, chains={n_chains}, burnin={burnin}, thin={thin}, h={step_size}")
    for it in range(needed_steps):
        noise = torch.randn_like(z)
        z_prop = z + step_size * sz + sqrt_2h * noise
        lp_prop, sz_prop = lp_score_z(z_prop)
        log_q_fwd = -((z_prop - z - step_size * sz) ** 2).sum(dim=1) / (4.0 * step_size)
        log_q_bwd = -((z - z_prop - step_size * sz_prop) ** 2).sum(dim=1) / (4.0 * step_size)
        log_alpha = (lp_prop - lp + log_q_bwd - log_q_fwd).clamp(max=0.0)
        acc = torch.rand(n_chains, dtype=dtype, device=device) < log_alpha.exp()
        z = torch.where(acc[:, None], z_prop, z)
        lp = torch.where(acc, lp_prop, lp)
        sz = torch.where(acc[:, None], sz_prop, sz)
        if it >= burnin:
            accepted += int(acc.sum().detach().cpu())
            total += n_chains
            if (it - burnin) % thin == 0:
                samples.append(theta_from_z(z).detach().clone())
                if len(samples) * n_chains >= n:
                    break
    out = torch.cat(samples, dim=0)[:n]
    if verbose and total:
        print(f"[MALA-z] post-burnin acceptance={100.0 * accepted / total:.1f}%")
    return out


# -----------------------------------------------------------------------------
# OU/SNIS estimators: Tweedie, coordinatewise blend, LFGI
# -----------------------------------------------------------------------------

def snis_w(y: torch.Tensor, t: torch.Tensor | float, xr: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    a = at(t).to(device=y.device, dtype=y.dtype)
    v = vt(t).to(device=y.device, dtype=y.dtype)
    # [B,N] log weights proportional to -||y-a*x_i||^2/(2v)
    B = y.shape[0]
    N = xr.shape[0]
    outs = []
    for i in range(0, B, chunk):
        yb = y[i:i + chunk]
        dist2 = torch.cdist(yb, a * xr).square()
        lw = -0.5 * dist2 / v
        lw = lw - lw.max(dim=1, keepdim=True).values
        w = lw.exp()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-30)
        outs.append(w)
    return torch.cat(outs, dim=0)


def est_tweedie(y: torch.Tensor, t, xr: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    a = at(t).to(device=y.device, dtype=y.dtype)
    v = vt(t).to(device=y.device, dtype=y.dtype)
    return (w[:, :, None] * (a * xr[None, :, :] - y[:, None, :])).sum(dim=1) / v


def est_tsi(y: torch.Tensor, t, w: torch.Tensor, s0_ref: torch.Tensor) -> torch.Tensor:
    a = at(t).to(device=y.device, dtype=y.dtype)
    return (w[:, :, None] * s0_ref[None, :, :]).sum(dim=1) / a


def est_blend(y: torch.Tensor, t, xr: torch.Tensor, w: torch.Tensor, s0_ref: torch.Tensor) -> torch.Tensor:
    a = at(t).to(device=y.device, dtype=y.dtype)
    v = vt(t).to(device=y.device, dtype=y.dtype)
    tsi_atom = s0_ref[None, :, :] / a
    twd_atom = (a * xr[None, :, :] - y[:, None, :]) / v
    am = (w[:, :, None] * tsi_atom).sum(dim=1)
    bm = (w[:, :, None] * twd_atom).sum(dim=1)
    ac = tsi_atom - am[:, None, :]
    bc = twd_atom - bm[:, None, :]
    va = (w[:, :, None] * ac.square()).sum(dim=1).clamp_min(1e-30)
    vb = (w[:, :, None] * bc.square()).sum(dim=1).clamp_min(1e-30)
    cab = (w[:, :, None] * ac * bc).sum(dim=1)
    den = (va + vb - 2.0 * cab).clamp_min(1e-20)
    g = ((va - cab) / den).clamp(0.0, 1.0)
    return (1.0 - g) * am + g * bm


def est_moment_matrix_blend(
    y: torch.Tensor,
    t,
    xr: torch.Tensor,
    w: torch.Tensor,
    s0_ref: torch.Tensor,
    *,
    x_gate: torch.Tensor | None = None,
    wg: torch.Tensor | None = None,
    s0_gate: torch.Tensor | None = None,
    ridge: float = MOMENT_MATRIX_RIDGE,
    ridge_rel: float = MOMENT_MATRIX_RIDGE_REL,
    chunk: int = MOMENT_MATRIX_CHUNK,
) -> torch.Tensor:
    """Direct primal/moment estimate of the optimal matrix-valued blend.

    The gate is estimated from the empirical SNIS normal equation

        G_hat = - N_hat (M_hat + ridge I)^{-1},

    where M_hat = E[d d^T | y,t] and N_hat = E[(b - E[b]) d^T | y,t].
    The resulting matrix gate is then applied to the estimator-bank discrepancy
    E[c] - E[b].  If a separate gate bank is provided, moments are estimated
    from that bank while Tweedie/TSI score signals are estimated from xr.
    """
    a = at(t).to(device=y.device, dtype=y.dtype)
    v = vt(t).to(device=y.device, dtype=y.dtype)
    x_gate = xr if x_gate is None else x_gate
    wg = w if wg is None else wg
    s0_gate = s0_ref if s0_gate is None else s0_gate

    B, d = y.shape
    eye = torch.eye(d, dtype=y.dtype, device=y.device)
    out = torch.empty_like(y)
    chunk = max(1, int(chunk))

    tsi_ref_atom = s0_ref[None, :, :] / a
    tsi_gate_atom = s0_gate[None, :, :] / a

    for i in range(0, B, chunk):
        sl = slice(i, min(i + chunk, B))
        yb = y[sl]
        wb = w[sl]
        wgb = wg[sl]

        twd_ref_atom = (a * xr[None, :, :] - yb[:, None, :]) / v
        s_twd = (wb[:, :, None] * twd_ref_atom).sum(dim=1)
        s_tsi = (wb[:, :, None] * tsi_ref_atom).sum(dim=1)
        delta = s_tsi - s_twd

        twd_gate_atom = (a * x_gate[None, :, :] - yb[:, None, :]) / v
        b_gate_mean = (wgb[:, :, None] * twd_gate_atom).sum(dim=1)
        d_gate_atom = tsi_gate_atom - twd_gate_atom
        b_cent = twd_gate_atom - b_gate_mean[:, None, :]

        M = torch.einsum("bn,bni,bnj->bij", wgb, d_gate_atom, d_gate_atom)
        N = torch.einsum("bn,bni,bnj->bij", wgb, b_cent, d_gate_atom)
        ridge_scale = M.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(0.0)
        K = M + (float(ridge) + float(ridge_rel) * ridge_scale)[:, None, None] * eye[None, :, :]

        try:
            G = -torch.linalg.solve(K, N.transpose(-1, -2)).transpose(-1, -2)
        except RuntimeError:
            jitter = (10.0 * float(ridge) + 10.0 * float(ridge_rel) * ridge_scale + 1e-7)
            K = M + jitter[:, None, None] * eye[None, :, :]
            G = -torch.linalg.solve(K.cpu(), N.cpu().transpose(-1, -2)).transpose(-1, -2).to(y.device)

        out[sl] = s_twd + torch.einsum("bij,bj->bi", G, delta)
    return out


def apply_ce_gate(s_twd: torch.Tensor, s_tsi: torch.Tensor, Pbar: torch.Tensor, t, chunk: int = 256) -> torch.Tensor:
    a = at(t).to(device=s_twd.device, dtype=s_twd.dtype)
    a2 = a * a
    v = vt(t).to(device=s_twd.device, dtype=s_twd.dtype)
    B, d = s_twd.shape
    eye = torch.eye(d, dtype=s_twd.dtype, device=s_twd.device)
    delta = s_tsi - s_twd
    out = torch.empty_like(s_twd)
    for i in range(0, B, chunk):
        K = a2 * eye[None, :, :] + v * sym(Pbar[i:i + chunk])
        rhs = a2 * delta[i:i + chunk]
        # solve K z = a^2 delta
        try:
            z = torch.linalg.solve(K, rhs[:, :, None]).squeeze(-1)
        except RuntimeError:
            # small jitter fallback
            K = K + 1e-8 * eye[None, :, :]
            z = torch.linalg.solve(K.cpu(), rhs.cpu()[:, :, None]).squeeze(-1).to(s_twd.device)
        out[i:i + chunk] = s_twd[i:i + chunk] + z
    return out


def est_ce_hlsi(y: torch.Tensor, t, xr: torch.Tensor, w: torch.Tensor, s0_ref: torch.Tensor, P_ref: torch.Tensor) -> torch.Tensor:
    s_twd = est_tweedie(y, t, xr, w)
    s_tsi = est_tsi(y, t, w, s0_ref)
    Pbar = torch.einsum("bn,nij->bij", w, P_ref)
    return apply_ce_gate(s_twd, s_tsi, Pbar, t)


class EstimatorBank:
    def __init__(self, target: LogisticPosterior, xr: torch.Tensor, xg: torch.Tensor | None = None):
        self.target = target
        self.xr = xr.detach()
        self.prefix_gate_bank = False
        self.prefix_gate_err = float("nan")
        if xg is None:
            self.same_gate_bank = True
            self.prefix_gate_bank = True
            self.prefix_gate_err = 0.0
            self.xg = self.xr
        else:
            # ``same_gate_bank`` means exact same finite reference set, hence the
            # two tensors must have the same shape.  In the N_G/N_R sweep with
            # fixed N_G and N_R < N_G, the intended coupled case is instead
            # prefix coupling: xr == xg[:N_R].  Track and print that explicitly.
            self.same_gate_bank = (xg.shape == xr.shape and bool(torch.equal(xg, xr)))
            if xg.shape[0] >= xr.shape[0] and xg.shape[1:] == xr.shape[1:]:
                err = (xg[:xr.shape[0]] - xr).abs().max() if xr.numel() else torch.tensor(0.0, device=xr.device)
                self.prefix_gate_err = float(err.detach().cpu())
                self.prefix_gate_bank = self.prefix_gate_err == 0.0
            self.xg = self.xr if self.same_gate_bank else xg.detach()
        print("[precompute] scores on estimator bank")
        self.s0 = target.score(self.xr).detach()
        print("[precompute] Hessians on gate bank")
        self.Pg = target.hessian(self.xg).detach()
        self.sg = self.s0 if self.same_gate_bank else target.score(self.xg).detach()
        print(
            f"[precompute] estimator bank N={self.xr.shape[0]}, gate bank N={self.xg.shape[0]}, "
            f"same_bank={self.same_gate_bank}, prefix_gate_bank={self.prefix_gate_bank}, "
            f"prefix_err={self.prefix_gate_err:.3e}"
        )

    def score(self, method: str, y: torch.Tensor, t) -> torch.Tensor:
        method = method.lower().strip().replace("_", "-").replace(" ", "-")
        w = snis_w(y, t, self.xr)
        if method == "tweedie":
            return est_tweedie(y, t, self.xr, w)
        if method in {"blend", "scalar-blend"}:
            return est_blend(y, t, self.xr, w, self.s0)
        if method in {"moment-matrix-blend", "matrix-blend", "primal-matrix-blend", "moment-blend"}:
            if self.same_gate_bank:
                wg = w
            else:
                wg = snis_w(y, t, self.xg)
            return est_moment_matrix_blend(
                y,
                t,
                self.xr,
                w,
                self.s0,
                x_gate=self.xg,
                wg=wg,
                s0_gate=self.sg,
            )
        if method in {"ce-hlsi", "lfgi"}:
            if self.same_gate_bank:
                wg = w
            else:
                wg = snis_w(y, t, self.xg)
            s_twd = est_tweedie(y, t, self.xr, w)
            s_tsi = est_tsi(y, t, w, self.s0)
            Pbar = torch.einsum("bn,nij->bij", wg, self.Pg)
            return apply_ce_gate(s_twd, s_tsi, Pbar, t)
        raise ValueError(f"unknown method {method!r}")


class ScoreRouter:
    """Route display labels to (bank, base-method) pairs.

    The main methods use the estimator bank ``xr`` for Tweedie/TSI signals and,
    for LFGI, the gate bank ``xg`` for Hessian aggregation.  When
    ``--add_bank_ablations`` is enabled, extra labels expose two sanity checks:
      * ``*-xg-bank`` gives blend/Tweedie/CE the full gate bank as their estimator bank;
      * ``ce-hlsi-xr-gate`` forces LFGI to use only the estimator-prefix gate.

    These labels make it obvious whether an apparent LFGI win at tiny
    ``n_ref`` is really coming from extra gate-only information.
    """

    def __init__(self):
        self._items: Dict[str, Tuple[EstimatorBank, str]] = {}

    def add(self, label: str, bank: EstimatorBank, base_method: str) -> None:
        if label in self._items:
            raise ValueError(f"duplicate method label {label!r}")
        self._items[label] = (bank, base_method)

    @property
    def methods(self) -> List[str]:
        return list(self._items.keys())

    def score(self, label: str, y: torch.Tensor, t) -> torch.Tensor:
        bank, base_method = self._items[label]
        return bank.score(base_method, y, t)


# -----------------------------------------------------------------------------
# Reverse SDE sampler
# -----------------------------------------------------------------------------

def heun_reverse_sde(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n: int,
    d: int,
    n_steps: int = 300,
    t_max: float = 3.0,
    t_min: float = 0.01,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, float, bool]:
    ts = torch.linspace(t_max, t_min, n_steps + 1, dtype=dtype, device=device)
    y = torch.randn(n, d, dtype=dtype, device=device)
    max_score = 0.0
    fail = False
    for i in range(n_steps):
        tc, tn = ts[i], ts[i + 1]
        h = tc - tn
        s1 = score_fn(y, tc)
        max_score = max(max_score, float(s1.abs().max().detach().cpu()))
        if not torch.isfinite(s1).all():
            fail = True
            break
        d1 = y + 2.0 * s1
        noise = torch.sqrt(2.0 * h) * torch.randn_like(y)
        yh = y + h * d1 + noise
        s2 = score_fn(yh, tn)
        if not torch.isfinite(s2).all():
            fail = True
            break
        d2 = yh + 2.0 * s2
        y = y + 0.5 * h * (d1 + d2) + noise
        if not torch.isfinite(y).all():
            fail = True
            break
    if not fail:
        tf = torch.tensor(t_min, dtype=dtype, device=device)
        sf = score_fn(y, tf)
        max_score = max(max_score, float(sf.abs().max().detach().cpu()))
        if torch.isfinite(sf).all():
            y = (y + vt(tf).to(device=device, dtype=dtype) * sf) / at(tf).to(device=device, dtype=dtype)
        else:
            fail = True
    return y.detach(), max_score, fail


# -----------------------------------------------------------------------------
# Metrics and diagnostics
# -----------------------------------------------------------------------------

def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, n_max: int = 2000, bws=(0.5, 1.0, 2.0, 5.0, 10.0)) -> float:
    n = min(len(X), n_max)
    m = min(len(Y), n_max)
    X = X[torch.randperm(len(X), device=X.device)[:n]]
    Y = Y[torch.randperm(len(Y), device=Y.device)[:m]]
    xx = torch.cdist(X, X).square()
    yy = torch.cdist(Y, Y).square()
    xy = torch.cdist(X, Y).square()
    val = 0.0
    for b in bws:
        g = 0.5 / (b * b)
        val = val + torch.exp(-g * xx).mean() + torch.exp(-g * yy).mean() - 2.0 * torch.exp(-g * xy).mean()
    return float((val / len(bws)).detach().cpu())


def sliced_ks(X: torch.Tensor, Y: torch.Tensor, n_proj: int = 256, n_max: int = 4096) -> float:
    n = min(len(X), n_max)
    m = min(len(Y), n_max)
    X = X[torch.randperm(len(X), device=X.device)[:n]]
    Y = Y[torch.randperm(len(Y), device=Y.device)[:m]]
    d = X.shape[1]
    dirs = torch.randn(n_proj, d, dtype=X.dtype, device=X.device)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp_min(1e-12)
    XP = X @ dirs.T
    YP = Y @ dirs.T
    vals = []
    grid_n = min(512, n, m)
    qs = torch.linspace(0.0, 1.0, grid_n, dtype=X.dtype, device=X.device)
    for j in range(n_proj):
        xq = torch.quantile(XP[:, j], qs)
        yq = torch.quantile(YP[:, j], qs)
        # approximation to 1D KS via CDFs on pooled sorted grid
        grid = torch.sort(torch.cat([xq, yq]))[0]
        Fx = torch.searchsorted(torch.sort(XP[:, j])[0], grid, right=True).to(X.dtype) / n
        Fy = torch.searchsorted(torch.sort(YP[:, j])[0], grid, right=True).to(X.dtype) / m
        vals.append((Fx - Fy).abs().max())
    return float(torch.stack(vals).mean().detach().cpu())


def ksd_rbf(X: torch.Tensor, score_fn: Callable[[torch.Tensor], torch.Tensor], n_max: int = 1000) -> float:
    n = min(len(X), n_max)
    X = X[torch.randperm(len(X), device=X.device)[:n]]
    S = score_fn(X)
    with torch.no_grad():
        D2 = torch.cdist(X, X).square()
        med = torch.median(D2[D2 > 0]).clamp_min(1e-6)
        h = med
        K = torch.exp(-D2 / (2.0 * h))
        diff = X[:, None, :] - X[None, :, :]
        sdot = S @ S.T
        term2 = (S[:, None, :] * (-diff / h)).sum(dim=2)
        term3 = (S[None, :, :] * (diff / h)).sum(dim=2)
        d = X.shape[1]
        term4 = d / h - D2 / (h * h)
        U = K * (sdot + term2 + term3 + term4)
        return float(U.mean().clamp_min(0.0).sqrt().detach().cpu())


def energy_kl_hist(X_ref: torch.Tensor, X: torch.Tensor, target: LogisticPosterior, bins: int = 100) -> float:
    with torch.no_grad():
        Uref = target.energy(X_ref).detach().cpu().numpy()
        U = target.energy(X).detach().cpu().numpy()
    lo, hi = np.percentile(Uref, [0.5, 99.5])
    hist_ref, edges = np.histogram(Uref, bins=bins, range=(lo, hi), density=False)
    hist, _ = np.histogram(U, bins=edges, density=False)
    p = hist_ref.astype(np.float64) + 1e-12
    q = hist.astype(np.float64) + 1e-12
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def score_rmse_vs_t(
    bank: EstimatorBank,
    x_true_bank: torch.Tensor,
    methods: List[str],
    t_grid: torch.Tensor,
    batch: int = 512,
    highN_chunk: int = 4096,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Score RMSE against high-N SNIS Tweedie proxy for true p_t score."""
    device, dtype = x_true_bank.device, x_true_bank.dtype
    out = {m: [] for m in methods}
    for t in t_grid:
        idx = torch.randperm(len(x_true_bank), device=device)[:batch]
        x0 = x_true_bank[idx]
        eps = torch.randn_like(x0)
        y = at(t).to(device=device, dtype=dtype) * x0 + torch.sqrt(vt(t).to(device=device, dtype=dtype)) * eps
        # high-N proxy for true noisy score via Tweedie with large held-out target bank
        # Compute in chunks over reference bank to avoid materializing huge cdist when possible.
        w_hi = snis_w(y, t, x_true_bank, chunk=batch)
        s_true = est_tweedie(y, t, x_true_bank, w_hi)
        for m in methods:
            s_hat = bank.score(m, y, t)
            rmse = torch.sqrt((s_hat - s_true).square().sum(dim=1).mean())
            out[m].append(float(rmse.detach().cpu()))
    avg = {m: float(np.mean(out[m])) for m in methods}
    return out, avg


def hessian_diagnostics(target: LogisticPosterior, X: torch.Tensor, out_dir: Path) -> Dict[str, float]:
    H = target.hessian(X[:min(len(X), 512)])
    evals = torch.linalg.eigvalsh(sym(H)).detach()
    lam_min = evals[:, 0]
    lam_max = evals[:, -1]
    cond = lam_max / lam_min.clamp_min(1e-30)
    trace = evals.sum(dim=1)
    p = evals / trace[:, None].clamp_min(1e-30)
    entropy = -(p * torch.log(p.clamp_min(1e-30))).sum(dim=1)
    effrank = torch.exp(entropy)
    stats = {
        "hess_lam_min_min": float(lam_min.min().cpu()),
        "hess_lam_min_median": float(lam_min.median().cpu()),
        "hess_lam_max_median": float(lam_max.median().cpu()),
        "hess_cond_median": float(cond.median().cpu()),
        "hess_cond_p95": float(torch.quantile(cond, 0.95).cpu()),
        "hess_effrank_median": float(effrank.median().cpu()),
        "hess_frac_non_psd": float((lam_min < -1e-8).to(torch.float64).mean().cpu()),
    }

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    mean_e = evals.mean(dim=0).detach().cpu().numpy()
    p10 = torch.quantile(evals, 0.10, dim=0).detach().cpu().numpy()
    p90 = torch.quantile(evals, 0.90, dim=0).detach().cpu().numpy()
    xs = np.arange(1, len(mean_e) + 1)
    axs[0].fill_between(xs, p10, p90, alpha=0.25)
    axs[0].plot(xs, mean_e, marker="o", ms=3)
    axs[0].set_yscale("log")
    axs[0].set_title("Hessian eigenvalues")
    axs[0].set_xlabel("eigen index")
    axs[0].set_ylabel("precision")
    axs[1].hist(np.log10(cond.detach().cpu().numpy()), bins=40)
    axs[1].set_title("log10 condition number")
    axs[2].hist(effrank.detach().cpu().numpy(), bins=40)
    axs[2].set_title("effective rank")
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "hessian_spectrum.png", dpi=180)
    plt.close(fig)
    return stats


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def fit_projection(X_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X_ref.mean(dim=0, keepdim=True)
    Xc = X_ref - mu
    _, _, Vh = torch.linalg.svd(Xc[:min(len(Xc), 4096)], full_matrices=False)
    return mu, Vh[:2].T.contiguous()


def project2(X: torch.Tensor, mu: torch.Tensor, V2: torch.Tensor) -> np.ndarray:
    return ((X - mu) @ V2).detach().cpu().numpy()


def plot_heatmaps(X_ref: torch.Tensor, samples: Dict[str, torch.Tensor], out_path: Path, title: str):
    """Publication-ready 2D posterior projection histograms.

    This intentionally omits colorbars: the panels are used comparatively, and
    the caption/text carry the interpretation.  Shared axes and fixed histogram
    ranges make the visual comparison stable under minipage scaling.
    """
    mu, V2 = fit_projection(X_ref)
    arrays = {"reference": project2(X_ref, mu, V2)}
    arrays.update({k: project2(v, mu, V2) for k, v in samples.items()})
    scale_names = _lfgi_scale_names(arrays.keys())
    scale_xy = _finite_concat_np([arrays[k] for k in scale_names if k in arrays])
    if scale_xy.size == 0:
        scale_xy = np.concatenate(list(arrays.values()), axis=0)
    scale_xy = np.asarray(scale_xy, dtype=np.float64).reshape(-1, 2)
    xlim = _padded_limits_from_values(scale_xy[:, 0], q_low=0.5, q_high=99.5, pad_frac=0.10)
    ylim = _padded_limits_from_values(scale_xy[:, 1], q_low=0.5, q_high=99.5, pad_frac=0.10)

    bins = 92
    density_scale_vals = []
    for k in scale_names:
        if k in arrays:
            H, _, _ = np.histogram2d(arrays[k][:, 0], arrays[k][:, 1], bins=bins, range=[xlim, ylim], density=True)
            density_scale_vals.append(H[np.isfinite(H) & (H > 0)])
    density_scale = _finite_concat_np(density_scale_vals)
    density_vmax = float(np.nanpercentile(density_scale, 99.5)) if density_scale.size else None

    names = list(arrays.keys())
    fig, axs = plt.subplots(1, len(names), figsize=(3.35 * len(names), 3.25), sharex=True, sharey=True)
    if len(names) == 1:
        axs = [axs]

    for j, (ax, name) in enumerate(zip(axs, names)):
        xy = arrays[name]
        ax.hist2d(xy[:, 0], xy[:, 1], bins=bins, range=[xlim, ylim], density=True, cmap="viridis", vmin=0.0, vmax=density_vmax)
        ax.set_title(method_label(name), pad=5)
        ax.set_xlabel("Posterior PC1")
        if j == 0:
            ax.set_ylabel("Posterior PC2")
        else:
            ax.set_ylabel("")
        style_axis(ax)

    fig.suptitle("MNIST PCA Logistic-Regression Posterior", fontsize=16, y=1.02)
    save_pub(fig, out_path)


def plot_metric_bars(metrics: Dict[str, Dict[str, float]], out_path: Path):
    keys = ["mmd", "sliced_ks", "ksd", "score_rmse", "energy_kl", "pred_nll", "pred_acc", "ess_proxy"]
    titles = ["MMD", "Sliced KS", "KSD", "Score RMSE", "Energy KL", "Pred. NLL", "Pred. Acc.", "ESS Proxy"]
    methods = list(metrics.keys())
    fig, axs = plt.subplots(2, 4, figsize=(14.0, 6.2))
    axs = axs.ravel()
    labels = [method_label(m) for m in methods]
    colors = [method_color(m) for m in methods]
    for ax, key, ttl in zip(axs, keys, titles):
        vals = [metrics[m].get(key, np.nan) for m in methods]
        ax.bar(labels, vals, color=colors, alpha=0.92)
        anchor_vals = [metrics[m].get(key, np.nan) for m in _lfgi_scale_names(methods) if m in metrics]
        _set_ylim_from_anchor(ax, anchor_vals, include_zero=True)
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=25)
        style_axis(ax, grid_axis="y")
    save_pub(fig, out_path)


def plot_energy_hist(target: LogisticPosterior, X_ref: torch.Tensor, samples: Dict[str, torch.Tensor], out_path: Path):
    with torch.no_grad():
        Uref = target.energy(X_ref).detach().cpu().numpy()
        Us = {k: target.energy(v).detach().cpu().numpy() for k, v in samples.items()}
    scale_methods = _lfgi_scale_names(Us.keys())
    scale_E = _finite_concat_np([Us[k] for k in scale_methods if k in Us])
    if scale_E.size == 0:
        scale_E = _finite_values_np(Uref)
    lo, hi = _padded_limits_from_values(scale_E, q_low=0.5, q_high=99.5, pad_frac=0.02)
    bins = np.linspace(lo, hi, 82)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(Uref, bins=bins, density=True, alpha=0.24, color=method_color("reference"), label="REFERENCE")
    for k, u in Us.items():
        ax.hist(
            u,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.35 if method_key(k) == "lfgi" else 2.0,
            color=method_color(k),
            label=method_label(k),
        )
    ax.set_xlabel(r"Energy $U(\theta)$")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Energy Distribution")
    ax.legend(frameon=False, loc="best")
    style_axis(ax)
    save_pub(fig, out_path)


def plot_score_rmse_curves(t_grid: torch.Tensor, curves: Dict[str, List[float]], out_path: Path):
    fig, ax = plt.subplots(figsize=(5.8, 3.85))
    t_np = t_grid.detach().cpu().numpy()
    for m, vals in curves.items():
        ax.plot(
            t_np,
            vals,
            marker=method_marker(m),
            markersize=5.2,
            linewidth=2.55 if method_key(m) == "lfgi" else 2.15,
            color=method_color(m),
            label=method_label(m),
        )
    ax.set_xlabel(r"Diffusion time $t$")
    ax.set_ylabel("Score RMSE")
    ax.set_title("Score RMSE Across Diffusion Times")
    ax.set_yscale("log")
    anchor_vals = _finite_concat_np([curves[m] for m in _lfgi_scale_names(curves.keys()) if m in curves])
    _set_ylim_from_anchor(ax, anchor_vals, log=True, include_zero=False)
    ax.legend(frameon=False, loc="best")
    style_axis(ax)
    save_pub(fig, out_path)



def _rankdata_average_np(x: np.ndarray) -> np.ndarray:
    """Average-rank transform without scipy; ranks are 0-based."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _safe_corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den <= 0:
        return float("nan")
    return float(np.sum(a * b) / den)


def _pairwise_order_accuracy_np(true_delta: np.ndarray, est_delta: np.ndarray, n_pairs: int, seed: int) -> float:
    true_delta = np.asarray(true_delta, dtype=np.float64)
    est_delta = np.asarray(est_delta, dtype=np.float64)
    mask = np.isfinite(true_delta) & np.isfinite(est_delta)
    true_delta = true_delta[mask]
    est_delta = est_delta[mask]
    n = len(true_delta)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=int(n_pairs))
    j = rng.integers(0, n, size=int(n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    td = true_delta[i] - true_delta[j]
    ed = est_delta[i] - est_delta[j]
    keep = np.abs(td) > 1e-12
    if keep.sum() == 0:
        return float("nan")
    return float((np.sign(td[keep]) == np.sign(ed[keep])).mean())


def integrate_score_straight_paths(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xs: torch.Tensor,
    anchor: torch.Tensor,
    t_eval: float,
    n_quad: int = 32,
    batch: int = 128,
) -> torch.Tensor:
    """Estimate log p(x)-log p(anchor) by straight-line score integration.

    For each x, integrate <s_hat(anchor + u(x-anchor), t_eval), x-anchor> du,
    using a trapezoidal quadrature rule.  This is a deliberately operational
    clean-density proxy: if an estimator has learned a usable score field, its
    line integrals should recover posterior log-density contrasts.
    """
    if anchor.ndim == 1:
        anchor = anchor[None, :]
    anchor = anchor.to(device=xs.device, dtype=xs.dtype)
    us = torch.linspace(0.0, 1.0, int(n_quad), dtype=xs.dtype, device=xs.device)
    trap_w = torch.ones_like(us)
    if len(us) > 1:
        trap_w[0] = 0.5
        trap_w[-1] = 0.5
    trap_w = trap_w / max(len(us) - 1, 1)
    t_tensor = torch.tensor(float(t_eval), dtype=xs.dtype, device=xs.device)
    outs: List[torch.Tensor] = []
    for i in range(0, xs.shape[0], int(batch)):
        xb = xs[i:i + int(batch)]
        delta = xb - anchor
        vals = torch.zeros(xb.shape[0], dtype=xs.dtype, device=xs.device)
        for u, wu in zip(us, trap_w):
            z = anchor + u * delta
            s_hat = score_fn(z, t_tensor)
            vals = vals + wu * (s_hat * delta).sum(dim=1)
        outs.append(vals.detach())
    return torch.cat(outs, dim=0)


def path_likelihood_diagnostics(
    *,
    target: LogisticPosterior,
    router: ScoreRouter,
    methods: List[str],
    samples: Dict[str, torch.Tensor],
    anchor: torch.Tensor,
    out_dir: Path,
    t_eval: float,
    n_samples: int = 1024,
    n_quad: int = 32,
    batch: int = 128,
    pairwise_pairs: int = 20000,
    seed: int = 123,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    """Run straight-path score integration likelihood diagnostics for all methods."""
    out: Dict[str, Dict[str, float]] = {}
    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    anchor = anchor.detach()
    logp_anchor = target.log_prob(anchor[None, :])[0].detach()
    for mi, m in enumerate(methods):
        xs_all = samples[m]
        n = min(int(n_samples), int(xs_all.shape[0]))
        g = torch.Generator(device=xs_all.device).manual_seed(int(seed) + 1009 * (mi + 1))
        idx = torch.randperm(xs_all.shape[0], generator=g, device=xs_all.device)[:n]
        xs = xs_all[idx].detach()
        exact_delta = (target.log_prob(xs) - logp_anchor).detach()
        score_fn = lambda y, t, method=m: router.score(method, y, t)
        est_delta = integrate_score_straight_paths(score_fn, xs, anchor, t_eval, n_quad=n_quad, batch=batch)

        e_all = est_delta.detach().cpu().numpy().astype(np.float64)
        y_all = exact_delta.detach().cpu().numpy().astype(np.float64)
        xs_all_np = xs.detach().cpu().numpy().astype(np.float64)
        mask = np.isfinite(e_all) & np.isfinite(y_all)
        e = e_all[mask]; y = y_all[mask]
        xs_np = xs_all_np[mask]
        residual = e - y
        centered = residual - residual.mean() if len(residual) else residual
        rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else float("nan")
        centered_rmse = float(np.sqrt(np.mean(centered ** 2))) if len(residual) else float("nan")
        mae = float(np.mean(np.abs(residual))) if len(residual) else float("nan")
        pearson = _safe_corr_np(y, e)
        spearman = _safe_corr_np(_rankdata_average_np(y), _rankdata_average_np(e)) if len(y) else float("nan")
        # Linear calibration: exact_delta ≈ slope * est_delta + intercept.
        if len(y) >= 3 and np.var(e) > 0:
            slope, intercept = np.polyfit(e, y, 1)
            pred = slope * e + intercept
            calib_rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = float(1.0 - np.sum((pred - y) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
        else:
            slope = intercept = calib_rmse = r2 = float("nan")
        pair_acc = _pairwise_order_accuracy_np(y, e, n_pairs=pairwise_pairs, seed=int(seed) + 17 * mi)
        out[m] = {
            "path_logp_rmse": rmse,
            "path_logp_centered_rmse": centered_rmse,
            "path_logp_mae": mae,
            "path_logp_pearson": pearson,
            "path_logp_spearman": spearman,
            "path_logp_calib_slope": float(slope),
            "path_logp_calib_intercept": float(intercept),
            "path_logp_calib_rmse": calib_rmse,
            "path_logp_calib_r2": r2,
            "path_logp_pairwise_acc": pair_acc,
            "path_logp_n": int(len(y)),
            "path_logp_t_eval": float(t_eval),
            "path_logp_n_quad": int(n_quad),
        }
        arrays[m] = {"xs": xs_np, "exact_delta": y, "est_delta": e, "residual": residual}
        print(f"[path-likelihood] {m}", out[m])
    plot_path_likelihood_diagnostics(arrays, out_dir / "path_likelihood_diagnostics.png")
    return out, arrays


def plot_path_likelihood_diagnostics(arrays: Dict[str, Dict[str, np.ndarray]], out_path: Path) -> None:
    methods = list(arrays.keys())
    scale_methods = _lfgi_scale_names(methods)
    scale_delta = _finite_concat_np(
        [np.concatenate([arrays[m]["exact_delta"], arrays[m]["est_delta"]]) for m in scale_methods if m in arrays]
    )
    delta_lim = _padded_limits_from_values(scale_delta, q_low=1.0, q_high=99.0, pad_frac=0.06, fallback=(-1.0, 1.0))
    scale_resid = _finite_concat_np([arrays[m]["residual"] for m in scale_methods if m in arrays])
    resid_lim = _padded_limits_from_values(scale_resid, q_low=1.0, q_high=99.0, pad_frac=0.06, fallback=(-1.0, 1.0))
    resid_bins = np.linspace(resid_lim[0], resid_lim[1], 50)

    fig, axs = plt.subplots(2, len(methods), figsize=(4.2 * len(methods), 7.0))
    if len(methods) == 1:
        axs = np.asarray(axs).reshape(2, 1)
    for j, m in enumerate(methods):
        y = arrays[m]["exact_delta"]
        e = arrays[m]["est_delta"]
        r = arrays[m]["residual"]
        ax = axs[0, j]
        ax.scatter(y, e, s=8, alpha=0.35)
        lo, hi = delta_lim
        ax.plot([lo, hi], [lo, hi], linewidth=1.0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{method_label(m)}: path log-density")
        ax.set_xlabel("true Δ log p")
        ax.set_ylabel("path-integrated Δ log p")
        ax.grid(alpha=0.25)
        ax = axs[1, j]
        ax.hist(r[np.isfinite(r)], bins=resid_bins)
        ax.set_xlim(resid_lim)
        ax.set_title(f"{method_label(m)}: residual")
        ax.set_xlabel("estimated - true Δ log p")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _percentile_rank01_np(x: np.ndarray) -> np.ndarray:
    """Average percentile rank in [0,1], robust to ties."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float64)
    return _rankdata_average_np(x) / float(len(x) - 1)


def _project_np_with_reference(X_ref: torch.Tensor, arrays: Dict[str, Dict[str, np.ndarray]]):
    """Fit a common reference PCA plane and project path-likelihood samples."""
    mu, V2 = fit_projection(X_ref)
    mu_np = mu.detach().cpu().numpy().astype(np.float64)
    V2_np = V2.detach().cpu().numpy().astype(np.float64)
    xy_by_method = {}
    ref_xy = ((X_ref[:min(len(X_ref), 4096)].detach().cpu().numpy().astype(np.float64) - mu_np) @ V2_np)
    for m, arr in arrays.items():
        xy = (arr["xs"] - mu_np) @ V2_np
        xy_by_method[m] = xy
    scale_methods = _lfgi_scale_names(arrays.keys())
    scale_xy_list = [xy_by_method[m] for m in scale_methods if m in xy_by_method]
    if not scale_xy_list:
        scale_xy_list = [ref_xy] + list(xy_by_method.values())
    scale_xy = np.concatenate(scale_xy_list, axis=0)
    xlim = _padded_limits_from_values(scale_xy[:, 0], q_low=0.5, q_high=99.5, pad_frac=0.08)
    ylim = _padded_limits_from_values(scale_xy[:, 1], q_low=0.5, q_high=99.5, pad_frac=0.08)
    return xy_by_method, xlim, ylim


def plot_path_likelihood_cloud_error(
    X_ref: torch.Tensor,
    arrays: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    title: str = "Path-likelihood error on posterior sample cloud",
) -> None:
    """Posterior PCA cloud colored by centered path log-density error."""
    methods = list(arrays.keys())
    xy_by_method, xlim, ylim = _project_np_with_reference(X_ref, arrays)
    scale_methods = _lfgi_scale_names(methods)
    scale_rc = []
    for sm in scale_methods:
        if sm in arrays:
            rr = arrays[sm]["residual"].astype(np.float64)
            scale_rc.append(rr - np.nanmean(rr))
    vmax = float(np.nanpercentile(np.abs(_finite_concat_np(scale_rc)), 95)) if _finite_concat_np(scale_rc).size else 1.0
    vmax = max(vmax, 1e-12)

    fig, axs = plt.subplots(1, len(methods), figsize=(4.3 * len(methods), 3.8), sharex=True, sharey=True)
    if len(methods) == 1:
        axs = [axs]
    for ax, m in zip(axs, methods):
        xy = xy_by_method[m]
        r = arrays[m]["residual"].astype(np.float64)
        rc = r - np.nanmean(r)
        finite = np.isfinite(rc)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=rc, s=9, alpha=0.72, cmap="coolwarm", vmin=-vmax, vmax=vmax, linewidths=0)
        crmse = np.sqrt(np.nanmean((rc[finite]) ** 2)) if finite.any() else np.nan
        ax.set_title(f"{method_label(m)}\ncentered RMSE={crmse:.3g}")
        ax.set_xlabel("posterior PC1")
        ax.set_ylabel("posterior PC2")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(alpha=0.18)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("centered est−true Δlogp")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def train_linear_svm_direction(target: LogisticPosterior, C: float = 1.0, max_iter: int = 20000) -> Dict[str, np.ndarray | float | bool | str]:
    """Train a linear SVM in the same PCA-feature space as the logistic posterior.

    This is used only as an operational discriminative-geometry reference: it is
    not treated as a likelihood model.  The returned normal vector has the same
    coordinates as posterior samples theta.
    """
    Xtr = target.X.detach().cpu().numpy().astype(np.float64)
    ytr = target.y.detach().cpu().numpy().astype(np.int64)
    Xte = target.X_test.detach().cpu().numpy().astype(np.float64)
    yte = target.y_test.detach().cpu().numpy().astype(np.int64)
    try:
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=float(C), loss="squared_hinge", dual="auto", max_iter=int(max_iter), tol=1e-5)
        clf.fit(Xtr, ytr)
        w = clf.coef_.reshape(-1).astype(np.float64)
        b = float(clf.intercept_[0])
        method = "sklearn.LinearSVC"
        ok = True
    except Exception as exc:
        # Conservative fallback: use the difference of class means as a linear
        # discriminative direction if sklearn is unavailable.
        pos = Xtr[ytr > 0]
        neg = Xtr[ytr < 0]
        w = (pos.mean(axis=0) - neg.mean(axis=0)).astype(np.float64)
        b = -0.5 * float((pos.mean(axis=0) + neg.mean(axis=0)) @ w)
        method = f"mean-difference fallback ({type(exc).__name__})"
        ok = False
    train_acc = float((np.sign(Xtr @ w + b) == ytr).mean())
    test_acc = float((np.sign(Xte @ w + b) == yte).mean())
    norm = float(np.linalg.norm(w))
    if norm <= 0:
        w_unit = w
    else:
        w_unit = w / norm
    return {"w": w, "w_unit": w_unit, "b": b, "train_acc": train_acc, "test_acc": test_acc, "method": method, "ok": ok}


def _cos_to_direction_np(xs: np.ndarray, direction: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64).reshape(-1)
    d_norm = np.linalg.norm(d)
    x_norm = np.linalg.norm(xs, axis=1)
    den = np.maximum(x_norm * max(d_norm, 1e-30), 1e-30)
    return (xs @ d) / den


def _single_classifier_accuracy_np(target: LogisticPosterior, theta: np.ndarray) -> float:
    Xte = target.X_test.detach().cpu().numpy().astype(np.float64)
    yte = target.y_test.detach().cpu().numpy().astype(np.int64)
    pred = np.where(Xte @ theta >= 0.0, 1, -1)
    return float((pred == yte).mean())


def _theta_to_pixel_weight_np(theta: np.ndarray, data: Dict[str, torch.Tensor]) -> np.ndarray | None:
    V = data.get("pca_components")
    B = data.get("feature_operator_matrix")
    scale = data.get("pixel_global_scale")
    if V is None or B is None or scale is None:
        return None
    V_np = V.detach().cpu().numpy().astype(np.float64)  # [784,d]
    B_np = B.detach().cpu().numpy().astype(np.float64)  # [d,d]
    scale_f = float(scale.detach().cpu().numpy()) if hasattr(scale, "detach") else float(scale)
    pix = (np.asarray(theta, dtype=np.float64) @ B_np.T @ V_np.T) / max(scale_f, 1e-12)
    return pix.reshape(28, 28)


def plot_svm_reference_weight(
    svm_info: Dict[str, np.ndarray | float | bool | str],
    data: Dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    img = _theta_to_pixel_weight_np(np.asarray(svm_info["w"], dtype=np.float64), data)
    if img is None:
        return
    vmax = max(float(np.nanpercentile(np.abs(img), 99)), 1e-12)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        "Linear SVM reference direction\n"
        f"test acc={float(svm_info['test_acc']):.3f}, train acc={float(svm_info['train_acc']):.3f}\n"
        f"fit={svm_info['method']}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="pixel-space weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def augment_path_arrays_with_svm(
    arrays: Dict[str, Dict[str, np.ndarray]],
    svm_info: Dict[str, np.ndarray | float | bool | str],
) -> None:
    w = np.asarray(svm_info["w"], dtype=np.float64)
    for arr in arrays.values():
        arr["svm_cos"] = _cos_to_direction_np(arr["xs"], w)


def add_svm_alignment_metrics(
    path_metrics: Dict[str, Dict[str, float]],
    arrays: Dict[str, Dict[str, np.ndarray]],
    svm_info: Dict[str, np.ndarray | float | bool | str],
) -> None:
    for m, arr in arrays.items():
        cosv = arr.get("svm_cos")
        if cosv is None:
            continue
        est_rank = _percentile_rank01_np(arr["est_delta"])
        true_rank = _percentile_rank01_np(arr["exact_delta"])
        finite = np.isfinite(cosv) & np.isfinite(est_rank) & np.isfinite(true_rank)
        if finite.sum() < 3:
            continue
        top = est_rank >= np.nanquantile(est_rank[finite], 0.90)
        bot = est_rank <= np.nanquantile(est_rank[finite], 0.10)
        path_metrics[m]["path_svm_est_rank_corr"] = _safe_corr_np(est_rank[finite], cosv[finite])
        path_metrics[m]["path_svm_true_rank_corr"] = _safe_corr_np(true_rank[finite], cosv[finite])
        path_metrics[m]["path_svm_top10_cos_mean"] = float(np.nanmean(cosv[top])) if np.any(top) else float("nan")
        path_metrics[m]["path_svm_bottom10_cos_mean"] = float(np.nanmean(cosv[bot])) if np.any(bot) else float("nan")
        path_metrics[m]["svm_test_acc"] = float(svm_info["test_acc"])


def plot_path_likelihood_svm_alignment(
    arrays: Dict[str, Dict[str, np.ndarray]],
    svm_info: Dict[str, np.ndarray | float | bool | str],
    out_path: Path,
    title: str = "Path-Likelihood Rank Calibration",
) -> None:
    """Publication-ready top-row-only SVM alignment diagnostic.

    Each panel compares estimated path-likelihood percentile with the true
    posterior log-density percentile.  Points are colored by cosine agreement
    with the linear SVM direction.  The previous bottom-row cosine histograms
    are intentionally removed for the paper-facing figure.
    """
    methods = list(arrays.keys())
    fig, axs = plt.subplots(
        1,
        len(methods),
        figsize=(3.85 * len(methods), 3.75),
        sharex=True,
        sharey=True,
    )
    if len(methods) == 1:
        axs = np.asarray([axs])

    scale_cos = []
    for m in _lfgi_scale_names(arrays.keys()):
        arr = arrays.get(m)
        if arr is None:
            continue
        cosv = arr.get("svm_cos")
        if cosv is not None:
            scale_cos.append(np.asarray(cosv, dtype=np.float64))
    if not scale_cos:
        for arr in arrays.values():
            cosv = arr.get("svm_cos")
            if cosv is not None:
                scale_cos.append(np.asarray(cosv, dtype=np.float64))
    if scale_cos:
        finite_cos = np.concatenate(scale_cos)
        finite_cos = finite_cos[np.isfinite(finite_cos)]
    else:
        finite_cos = np.array([], dtype=np.float64)
    if len(finite_cos):
        cmin = float(np.nanpercentile(finite_cos, 3))
        cmax = float(np.nanpercentile(finite_cos, 99))
        if cmax - cmin < 1e-5:
            cmin, cmax = max(-1.0, cmin - 0.05), min(1.0, cmax + 0.05)
    else:
        cmin, cmax = -1.0, 1.0

    last_sc = None
    for j, m in enumerate(methods):
        arr = arrays[m]
        cosv = arr.get("svm_cos")
        if cosv is None:
            cosv = _cos_to_direction_np(arr["xs"], np.asarray(svm_info["w"], dtype=np.float64))
        est_rank = _percentile_rank01_np(arr["est_delta"])
        true_rank = _percentile_rank01_np(arr["exact_delta"])
        finite = np.isfinite(est_rank) & np.isfinite(true_rank) & np.isfinite(cosv)
        er = est_rank[finite]
        tr = true_rank[finite]
        cv = np.asarray(cosv, dtype=np.float64)[finite]

        rho_et = _safe_corr_np(er, tr)
        rho_ec = _safe_corr_np(er, cv)
        top = er >= 0.90
        top_mean = float(np.nanmean(cv[top])) if np.any(top) else float("nan")

        ax = axs[j]
        last_sc = ax.scatter(
            er,
            tr,
            c=cv,
            s=13,
            alpha=0.78,
            cmap="viridis",
            vmin=cmin,
            vmax=cmax,
            linewidths=0,
            rasterized=True,
        )
        ax.plot([0, 1], [0, 1], color="0.25", linewidth=1.15, alpha=0.65)
        ax.axvline(0.90, color="0.25", linestyle=":", linewidth=0.95, alpha=0.45)
        ax.axhline(0.90, color="0.25", linestyle=":", linewidth=0.95, alpha=0.45)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(method_label(m), pad=5, fontsize=15)
        ann = (
            rf"$\rho_{{\rm rank}}={rho_et:.2f}$" + "\n"
            + rf"$\rho_{{\rm SVM}}={rho_ec:.2f}$" + "\n"
            + rf"top-10 cosine $={top_mean:.2f}$"
        )
        ax.text(
            0.04,
            0.96,
            ann,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.2,
            linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.85", alpha=0.86),
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_axis(ax)

    fig.supxlabel("Estimated likelihood percentile", fontsize=14, y=0.04)
    fig.supylabel("True likelihood percentile", fontsize=14, x=0.018)
    fig.subplots_adjust(right=0.915, top=0.80, bottom=0.20, wspace=0.22)
    if last_sc is not None:
        cax = fig.add_axes([0.93, 0.24, 0.014, 0.44])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label("SVM cosine", fontsize=11)
        cbar.ax.tick_params(labelsize=10)
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.patch.set_facecolor("white")
    fig.savefig(out_path, dpi=PUB_DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white", transparent=False)
    plt.close(fig)

def plot_path_likelihood_cloud_rank(
    X_ref: torch.Tensor,
    arrays: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    title: str = "Estimated likelihood rank on posterior sample cloud",
) -> None:
    """Posterior PCA cloud colored by estimator likelihood percentile rank."""
    methods = list(arrays.keys())
    xy_by_method, xlim, ylim = _project_np_with_reference(X_ref, arrays)
    fig, axs = plt.subplots(1, len(methods), figsize=(4.3 * len(methods), 3.8), sharex=True, sharey=True)
    if len(methods) == 1:
        axs = [axs]
    for ax, m in zip(axs, methods):
        xy = xy_by_method[m]
        rank = _percentile_rank01_np(arrays[m]["est_delta"])
        true_rank = _percentile_rank01_np(arrays[m]["exact_delta"])
        rho = _safe_corr_np(rank, true_rank)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=rank, s=9, alpha=0.72, cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0)
        ax.set_title(f"{method_label(m)}\nrank corr={rho:.3g}")
        ax.set_xlabel("posterior PC1")
        ax.set_ylabel("posterior PC2")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(alpha=0.18)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("estimated likelihood percentile")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_path_likelihood_weight_montage(
    arrays: Dict[str, Dict[str, np.ndarray]],
    data: Dict[str, torch.Tensor],
    target: LogisticPosterior,
    out_path: Path,
    svm_info: Dict[str, np.ndarray | float | bool | str] | None = None,
    title: str = "Likelihood-sorted logistic classifiers",
) -> None:
    """Show classifier weights selected by each estimator's likelihood ranking.

    Rows are score estimators.  Columns are posterior samples selected by that
    estimator's path-likelihood quantiles.  The image shows the pixel-space
    classifier weight, and the small annotation reports

      est: estimator percentile rank; true: exact posterior rank;
      svm: cosine with the linear SVM direction; acc: single-classifier test acc.

    Important plotting guard: failing methods can produce enormous weights.
    They are still displayed, but LFGI-selected samples define the global
    color normalization whenever LFGI is available, so failed comparators cannot
    wash out the methods that actually worked.
    """
    methods = list(arrays.keys())
    qs = [0.02, 0.25, 0.50, 0.75, 0.98]
    qlabs = ["lowest\nest. likelihood", "25%", "median", "75%", "highest\nest. likelihood"]
    n_extra = 1 if svm_info is not None else 0
    nrows = len(methods) + n_extra
    fig, axs = plt.subplots(nrows, len(qs), figsize=(3.15 * len(qs), 3.25 * nrows))
    axs = np.asarray(axs)
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)

    def selected_imgs_for_method(m: str):
        xs = arrays[m]["xs"]
        e = arrays[m]["est_delta"]
        if len(xs) == 0:
            return []
        order = np.argsort(e)
        imgs = []
        for q in qs:
            idx = order[int(np.clip(round(q * (len(order) - 1)), 0, len(order) - 1))]
            img = _theta_to_pixel_weight_np(xs[idx], data)
            if img is not None:
                imgs.append(img)
        return imgs

    # Common color scale from LFGI rows only.  This is the key guard against
    # scalar/moment matrix blow-up destroying the visual scale.
    imgs_for_scale = []
    scale_methods = _lfgi_scale_names(methods)
    for m in scale_methods:
        imgs_for_scale.extend(selected_imgs_for_method(m))
    if imgs_for_scale:
        vals = np.abs(np.concatenate([img.ravel() for img in imgs_for_scale]))
        vals = vals[np.isfinite(vals)]
        vmax_global = float(np.nanpercentile(vals, 98.5)) if len(vals) else 1.0
        vmax_global = max(vmax_global, 1e-12)
    else:
        vmax_global = 1.0

    row_offset = 0
    svm_img = None
    if svm_info is not None:
        svm_img = _theta_to_pixel_weight_np(np.asarray(svm_info["w"], dtype=np.float64), data)
        for j in range(len(qs)):
            ax = axs[0, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j == len(qs)//2 and svm_img is not None:
                ax.imshow(svm_img, cmap="coolwarm", vmin=-vmax_global, vmax=vmax_global)
                ax.set_title(
                    f"SVM reference\ntest acc={float(svm_info['test_acc']):.3f}\ntrain acc={float(svm_info.get('train_acc', np.nan)):.3f}",
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.axis("off")
            if j == 0:
                ax.text(-0.12, 0.5, "linear SVM\nreference", transform=ax.transAxes,
                        ha="right", va="center", fontsize=11, fontweight="bold")
        row_offset = 1

    # For a compact residual cue, use the border style/color: high estimated and
    # high true/SVM-aligned gets thick solid; high estimated but low true/SVM gets
    # dashed.  This is more readable than doubling every cell with a residual image.
    for i, m in enumerate(methods):
        row = i + row_offset
        xs = arrays[m]["xs"]
        e = arrays[m]["est_delta"]
        y = arrays[m]["exact_delta"]
        if len(xs) == 0:
            continue
        est_rank_all = _percentile_rank01_np(e)
        true_rank_all = _percentile_rank01_np(y)
        cos_all = arrays[m].get("svm_cos")
        if cos_all is None and svm_info is not None:
            cos_all = _cos_to_direction_np(xs, np.asarray(svm_info["w"], dtype=np.float64))
        order = np.argsort(e)
        for j, (q, qlab) in enumerate(zip(qs, qlabs)):
            k = int(np.clip(round(q * (len(order) - 1)), 0, len(order) - 1))
            idx = order[k]
            theta = xs[idx]
            img = _theta_to_pixel_weight_np(theta, data)
            ax = axs[row, j]
            if img is not None:
                # Clip values only for display.  The selected sample and metrics are untouched.
                ax.imshow(np.clip(img, -vmax_global, vmax_global), cmap="coolwarm", vmin=-vmax_global, vmax=vmax_global)
            ax.set_xticks([]); ax.set_yticks([])
            if row == row_offset:
                ax.text(0.5, 1.22, qlab, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold")
            if j == 0:
                ax.text(-0.12, 0.5, method_label(m), transform=ax.transAxes, ha="right", va="center", fontsize=11, fontweight="bold")
            acc = _single_classifier_accuracy_np(target, theta)
            cos_val = float(cos_all[idx]) if cos_all is not None and np.isfinite(cos_all[idx]) else float("nan")
            # Residual-like scalar: disagreement between estimator rank and true rank.
            rank_gap = float(est_rank_all[idx] - true_rank_all[idx])
            ax.text(
                0.5, -0.08,
                f"est {est_rank_all[idx]:.2f} | true {true_rank_all[idx]:.2f} | gap {rank_gap:+.2f}\n"
                f"svm {cos_val:.2f} | acc {acc:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.82, linewidth=0.35),
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor("0.35")
                spine.set_linestyle("-")
            high_est = est_rank_all[idx] >= 0.90
            good_true = true_rank_all[idx] >= 0.75
            good_svm = np.isfinite(cos_val) and cos_val >= 0.85
            if high_est and good_true and good_svm:
                for spine in ax.spines.values():
                    spine.set_linewidth(2.5)
                    spine.set_edgecolor("black")
                    spine.set_linestyle("-")
            elif high_est and (not good_true or not good_svm):
                for spine in ax.spines.values():
                    spine.set_linewidth(2.1)
                    spine.set_linestyle((0, (3, 2)))
                    spine.set_edgecolor("black")

    # Colorbar in a small dedicated axis.  Do not attach it to every subplot.
    fig.subplots_adjust(left=0.10, right=0.91, top=0.88, bottom=0.06, hspace=0.92, wspace=0.55)
    cax = fig.add_axes([0.925, 0.28, 0.014, 0.34])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-vmax_global, vmax=vmax_global))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("pixel-space weight\n(LFGI scale)", fontsize=9)
    fig.suptitle(
        title + "\nRows are sorted by each estimator's path likelihood; gap=est rank−true rank; SVM cosine is geometric agreement.",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_path_likelihood_weight_residual_montage(
    arrays: Dict[str, Dict[str, np.ndarray]],
    data: Dict[str, torch.Tensor],
    out_path: Path,
    svm_info: Dict[str, np.ndarray | float | bool | str] | None = None,
    title: str = "Residual from SVM reference for likelihood-sorted classifiers",
) -> None:
    """Optional companion montage: selected classifier image minus scaled SVM image.

    This makes SVM agreement visually explicit without cluttering the main montage.
    Each residual uses the same LFGI-normalized color scale.  We subtract the
    least-squares scalar multiple of the SVM reference image from each selected
    classifier image, because posterior samples may differ in norm while sharing
    the same discriminative direction.
    """
    if svm_info is None:
        return
    svm_img = _theta_to_pixel_weight_np(np.asarray(svm_info["w"], dtype=np.float64), data)
    if svm_img is None:
        return
    methods = list(arrays.keys())
    qs = [0.02, 0.25, 0.50, 0.75, 0.98]
    qlabs = ["lowest", "25%", "median", "75%", "highest"]
    fig, axs = plt.subplots(len(methods), len(qs), figsize=(3.05 * len(qs), 3.0 * len(methods)))
    axs = np.asarray(axs)
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)

    residuals_by_cell = []
    selected = {}
    denom = float(np.sum(svm_img * svm_img)) + 1e-12
    for m in methods:
        xs = arrays[m]["xs"]; e = arrays[m]["est_delta"]
        order = np.argsort(e)
        selected[m] = []
        for q in qs:
            idx = order[int(np.clip(round(q * (len(order) - 1)), 0, len(order) - 1))]
            img = _theta_to_pixel_weight_np(xs[idx], data)
            if img is None:
                resid = None
            else:
                scale = float(np.sum(img * svm_img) / denom)
                resid = img - scale * svm_img
                # Use LFGI methods for normalization if available.
                if method_key(m) == "lfgi":
                    residuals_by_cell.append(resid)
            selected[m].append((idx, resid))
    if not residuals_by_cell:
        residuals_by_cell = [r for cells in selected.values() for _, r in cells if r is not None]
    if residuals_by_cell:
        vals = np.abs(np.concatenate([r.ravel() for r in residuals_by_cell if r is not None]))
        vmax = float(np.nanpercentile(vals[np.isfinite(vals)], 98.5)) if np.any(np.isfinite(vals)) else 1.0
        vmax = max(vmax, 1e-12)
    else:
        vmax = 1.0

    for i, m in enumerate(methods):
        for j, (q, qlab) in enumerate(zip(qs, qlabs)):
            ax = axs[i, j]
            idx, resid = selected[m][j]
            if resid is not None:
                ax.imshow(np.clip(resid, -vmax, vmax), cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(qlab, fontsize=10, fontweight="bold")
            if j == 0:
                ax.text(-0.12, 0.5, method_label(m), transform=ax.transAxes, ha="right", va="center", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.91, top=0.86, bottom=0.06, hspace=0.35, wspace=0.35)
    cax = fig.add_axes([0.925, 0.28, 0.014, 0.38])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("classifier − projected SVM", fontsize=9)
    fig.suptitle(title + "\nResidual after subtracting best scalar multiple of the SVM weight image.", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_path_likelihood_bars(path_metrics: Dict[str, Dict[str, float]], out_path: Path) -> None:
    keys = [
        "path_logp_rmse",
        "path_logp_centered_rmse",
        "path_logp_calib_rmse",
        "path_logp_pearson",
        "path_logp_spearman",
        "path_logp_pairwise_acc",
    ]
    titles = ["RMSE", "Centered RMSE", "Calibrated RMSE", "Pearson", "Spearman", "Pairwise Acc."]
    methods = list(path_metrics.keys())
    labels = [method_label(m) for m in methods]
    colors = [method_color(m) for m in methods]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.1))
    axs = axs.ravel()
    for ax, key, ttl in zip(axs, keys, titles):
        vals = [path_metrics[m].get(key, np.nan) for m in methods]
        ax.bar(labels, vals, color=colors, alpha=0.92)
        anchor_vals = [path_metrics[m].get(key, np.nan) for m in _lfgi_scale_names(methods) if m in path_metrics]
        _set_ylim_from_anchor(ax, anchor_vals, include_zero=True)
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=25)
        style_axis(ax, grid_axis="y")
    fig.suptitle("Path-Integrated Log-Density Diagnostics", fontsize=16, y=1.02)
    save_pub(fig, out_path)

def ess_energy_proxy(target: LogisticPosterior, X: torch.Tensor) -> float:
    # Lightweight diagnostic: ESS of unnormalized exp(-U) values after centering.
    # Not an IS-correctness certificate; useful only to detect collapsed high-energy samples.
    with torch.no_grad():
        logw = target.log_prob(X)
        logw = logw - logw.max()
        w = logw.exp()
        return float((w.sum().square() / w.square().sum().clamp_min(1e-30) / len(w)).detach().cpu())



# -----------------------------------------------------------------------------
# N_G / N_R sweep utilities
# -----------------------------------------------------------------------------

def parse_int_sweep(spec: str, max_value: int | None = None) -> List[int]:
    """Parse sweep specs like '1:100', '1:5:100', or '1,2,5,10'."""
    spec = str(spec).strip()
    vals: List[int] = []
    if not spec:
        raise ValueError("empty sweep spec")
    if "," in spec:
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    elif ":" in spec:
        parts = [int(x.strip()) for x in spec.split(":") if x.strip()]
        if len(parts) == 2:
            lo, hi = parts
            step = 1
        elif len(parts) == 3:
            lo, step, hi = parts
        else:
            raise ValueError(f"bad sweep range {spec!r}; use lo:hi or lo:step:hi")
        if step <= 0:
            raise ValueError("sweep step must be positive")
        vals = list(range(lo, hi + 1, step))
    else:
        vals = [int(spec)]
    vals = sorted(set(vals))
    if any(v <= 0 for v in vals):
        raise ValueError(f"all sweep N_ref values must be positive, got {vals}")
    if max_value is not None and any(v > max_value for v in vals):
        raise ValueError(f"sweep N_ref values exceed max_value={max_value}: {vals}")
    return vals


def run_ng_nr_sweep(
    *,
    args,
    target: LogisticPosterior,
    xr_pool: torch.Tensor,
    xg: torch.Tensor,
    X_test_ref: torch.Tensor,
    pred_ref: Dict[str, float],
    out_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Sweep LFGI over N_R at fixed N_G.

    The gate/Hessian atlas xg is fixed at size N_G.  For each N_R, the base score
    bank is the first N_R samples of xr_pool.  If --bank_coupling=prefix,
    xr_pool is the same tensor as xg and each swept score bank is a prefix of the
    fixed gate bank.  If --bank_coupling=independent, xr_pool is disjoint from xg.
    We also compute fixed full-bank baselines using xg as their estimator bank:
    Tweedie_NG, Blend_NG, and LFGI_NG.
    """
    n_gate = int(xg.shape[0])
    nrefs = parse_int_sweep(args.sweep_n_ref_values, max_value=min(len(xr_pool), n_gate))
    print("=" * 80)
    print(f"[N_G/N_R sweep] fixed N_G={n_gate}; N_R values={nrefs[:8]}{'...' if len(nrefs)>8 else ''} ({len(nrefs)} total)")
    print(f"[N_G/N_R sweep] xR_pool={tuple(xr_pool.shape)}, xG={tuple(xg.shape)}, bank_coupling={args.bank_coupling}, n_gen={args.n_gen}")
    if args.bank_coupling == "prefix":
        prefix_check_n = min(len(xr_pool), len(xg))
        prefix_err = float((xr_pool[:prefix_check_n] - xg[:prefix_check_n]).abs().max().detach().cpu()) if prefix_check_n > 0 else 0.0
        print(f"[N_G/N_R sweep] prefix-coupled pool check: xR_pool[:{prefix_check_n}] == xG[:{prefix_check_n}] max_abs_err={prefix_err:.3e}")
    elif args.bank_coupling == "independent":
        print("[N_G/N_R sweep] independent score/gate pools: swept xR prefixes are disjoint from xG")
    print("=" * 80)

    sweep_dir = out_dir / f"ng_nr_sweep_NG{n_gate}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    def eval_samples(label: str, xs: torch.Tensor, max_score: float, fail: bool, runtime_sec: float) -> Dict[str, float]:
        met = {
            "mmd": mmd_rbf(X_test_ref, xs, n_max=args.metric_max),
            "sliced_ks": sliced_ks(X_test_ref, xs, n_proj=256, n_max=min(4096, args.metric_max * 2)),
            "ksd": ksd_rbf(xs, target.score, n_max=min(1000, args.metric_max)),
            "energy_kl": energy_kl_hist(X_test_ref, xs, target),
            "ess_proxy": ess_energy_proxy(target, xs),
            "max_score": max_score,
            "failed": float(fail),
            "runtime_sec": runtime_sec,
        }
        met.update(target.predictive_metrics(xs[:min(len(xs), 5000)]))
        print(f"[sweep metrics] {label}", met)
        return met

    rows: List[Dict[str, float | str | int]] = []

    # Same-N_G baselines. These are the fairest fixed-bank reference points:
    # all methods use exactly the N_G bank as their primal/reference bank.
    print("-" * 80)
    print("[N_G/N_R sweep] fixed full-N_G baselines")
    bank_ng = EstimatorBank(target, xr=xg, xg=xg)
    router_ng = ScoreRouter()
    router_ng.add("tweedie_NG", bank_ng, "tweedie")
    router_ng.add("blend_NG", bank_ng, "blend")
    router_ng.add("matrix_blend_NG", bank_ng, "matrix-blend")
    router_ng.add("ce_hlsi_NG", bank_ng, "ce-hlsi")
    baseline_metrics: Dict[str, Dict[str, float]] = {}
    for label in router_ng.methods:
        t0 = time.time()
        xs, max_score, fail = heun_reverse_sde(
            lambda y, t, lab=label: router_ng.score(lab, y, t),
            n=args.n_gen,
            d=target.D,
            n_steps=args.n_steps,
            t_max=args.t_max,
            t_min=args.t_min,
            device=device,
            dtype=dtype,
        )
        met = eval_samples(label, xs, max_score, fail, time.time() - t0)
        baseline_metrics[label] = met
        np.save(sweep_dir / f"samples_{label}.npy", xs.detach().cpu().numpy())
        row = {"method": label, "n_ref": n_gate, "n_gate": n_gate, "ratio_ng_over_nr": 1.0}
        row.update(met)
        rows.append(row)

    # LFGI sweep over score-bank size, fixed gate bank.
    for nr in nrefs:
        label = f"ce_hlsi_NR{nr}_NG{n_gate}"
        print("-" * 80)
        print(f"[N_G/N_R sweep] {label}")
        xr = xr_pool[:nr]
        bank = EstimatorBank(target, xr=xr, xg=xg)
        t0 = time.time()
        xs, max_score, fail = heun_reverse_sde(
            lambda y, t, b=bank: b.score("ce-hlsi", y, t),
            n=args.n_gen,
            d=target.D,
            n_steps=args.n_steps,
            t_max=args.t_max,
            t_min=args.t_min,
            device=device,
            dtype=dtype,
        )
        met = eval_samples(label, xs, max_score, fail, time.time() - t0)
        if args.sweep_save_samples:
            np.save(sweep_dir / f"samples_{label}.npy", xs.detach().cpu().numpy())
        row = {"method": "ce-hlsi-sweep", "n_ref": nr, "n_gate": n_gate, "ratio_ng_over_nr": float(n_gate) / float(nr)}
        row.update(met)
        rows.append(row)

        # Incremental CSV checkpoint so long sweeps survive interruption.
        keys = sorted({k for r in rows for k in r.keys()})
        with open(sweep_dir / "ng_nr_sweep_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # Optional score-RMSE curve for a reduced set of representative N_R values.
    rmse_rows: List[Dict[str, float | str | int]] = []
    if args.sweep_score_rmse:
        rep_vals = sorted(set([nrefs[0], nrefs[len(nrefs)//4], nrefs[len(nrefs)//2], nrefs[-1]]))
        t_grid = torch.linspace(args.t_min, args.t_max, args.score_rmse_tgrid, dtype=dtype, device=device)
        for nr in rep_vals:
            bank = EstimatorBank(target, xr=xr_pool[:nr], xg=xg)
            curves, avg = score_rmse_vs_t(
                bank=bank,
                x_true_bank=X_test_ref[:min(len(X_test_ref), 10000)],
                methods=["ce-hlsi"],
                t_grid=t_grid,
                batch=args.score_rmse_batch,
            )
            rmse_rows.append({"method": "ce-hlsi-sweep", "n_ref": nr, "n_gate": n_gate, "score_rmse": avg["ce-hlsi"]})
        with open(sweep_dir / "ng_nr_sweep_score_rmse.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "n_ref", "n_gate", "score_rmse"])
            writer.writeheader(); writer.writerows(rmse_rows)

    # Plots: each metric versus N_R for LFGI, with horizontal same-N_G baselines.
    metric_names = ["mmd", "sliced_ks", "ksd", "energy_kl", "pred_nll", "pred_acc", "ess_proxy"]
    ce_rows = [r for r in rows if r.get("method") == "ce-hlsi-sweep"]
    xvals = np.array([int(r["n_ref"]) for r in ce_rows], dtype=float)
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 10.5))
    axs = axs.ravel()
    for ax, metric in zip(axs, metric_names):
        yvals = np.array([float(r[metric]) for r in ce_rows], dtype=float)
        ax.plot(xvals, yvals, marker="o", ms=3, label="LFGI $(N_R,N_G)$")
        for bname, bmet in baseline_metrics.items():
            if metric in bmet:
                ax.axhline(float(bmet[metric]), linestyle="--", linewidth=1.0, label=bname)
        ax.set_title(metric)
        ax.set_xlabel(r"$N_R$ (score-signal bank size)")
        ax.grid(alpha=0.25)
        if metric in {"mmd", "ksd", "energy_kl", "ess_proxy"}:
            ax.set_yscale("log")
            _set_ylim_from_anchor(ax, yvals, log=True, include_zero=False)
        else:
            _set_ylim_from_anchor(ax, yvals, include_zero=True)
    # Ratio plot in final panel.
    ax = axs[len(metric_names)]
    ax.plot(xvals, n_gate / xvals, marker="o", ms=3)
    ax.set_title(r"$N_G/N_R$")
    ax.set_xlabel(r"$N_R$")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    for j in range(len(metric_names)+1, len(axs)):
        axs[j].axis("off")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle(f"N_G/N_R ablation, fixed N_G={n_gate}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(sweep_dir / "ng_nr_sweep_metrics.png", dpi=180)
    plt.close(fig)

    # Best N_R summary by metric.
    best_lines = []
    for metric in metric_names:
        vals = [(int(r["n_ref"]), float(r[metric])) for r in ce_rows]
        if not vals:
            continue
        if metric in {"pred_acc", "ess_proxy"}:
            best_nr, best_val = max(vals, key=lambda p: p[1])
        else:
            best_nr, best_val = min(vals, key=lambda p: p[1])
        best_lines.append(f"{metric}: best N_R={best_nr}, N_G/N_R={n_gate / best_nr:.3g}, value={best_val:.6g}")
    with open(sweep_dir / "ng_nr_sweep_summary.txt", "w") as f:
        f.write(f"target={target.name}\n")
        f.write(f"fixed_n_gate={n_gate}\n")
        f.write(f"n_ref_values={nrefs}\n")
        f.write(f"reference_pred_nll={pred_ref['pred_nll']}\nreference_pred_acc={pred_ref['pred_acc']}\n")
        f.write("\nSame-N_G baselines:\n")
        for bname, bmet in baseline_metrics.items():
            f.write(bname + ": " + ", ".join(f"{k}={v:.6g}" for k, v in bmet.items() if isinstance(v, (float, int))) + "\n")
        f.write("\nBest LFGI N_R values:\n")
        f.write("\n".join(best_lines) + "\n")

    # Compact dashboard.
    with PdfPages(sweep_dir / "ng_nr_sweep_dashboard.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111); ax.axis("off")
        lines = [
            "MNIST PCA logistic-regression N_G/N_R sweep",
            "",
            f"target: {target.name}",
            f"fixed N_G={n_gate}; N_R values={nrefs[0]}..{nrefs[-1]} ({len(nrefs)} values)",
            f"classes={tuple(args.classes)}; d={args.d}; tau={args.tau}; beta={args.beta}",
            f"bank_coupling={args.bank_coupling}; n_gen={args.n_gen}; steps={args.n_steps}",
            "",
            "Same-N_G baselines:",
        ]
        for bname, bmet in baseline_metrics.items():
            lines.append(f"  {bname}: mmd={bmet['mmd']:.4g}, sliced_ks={bmet['sliced_ks']:.4g}, ksd={bmet['ksd']:.4g}, pred_nll={bmet['pred_nll']:.4g}, pred_acc={bmet['pred_acc']:.4g}")
        lines += ["", "Best LFGI N_R values:"] + ["  " + s for s in best_lines]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig); plt.close(fig)
        for img in ["ng_nr_sweep_metrics.png"]:
            arr = plt.imread(sweep_dir / img)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(arr); ax.axis("off"); ax.set_title(img)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print("=" * 80)
    print(f"[N_G/N_R sweep done] wrote {sweep_dir / 'ng_nr_sweep_metrics.csv'}")
    print(f"dashboard: {sweep_dir / 'ng_nr_sweep_dashboard.pdf'}")
    print("=" * 80)


# -----------------------------------------------------------------------------
# Repeated-run aggregation
# -----------------------------------------------------------------------------

TABLE_METRICS = [
    ("score_rmse", "Score RMSE $\\downarrow$", "down"),
    ("mmd", "MMD $\\downarrow$", "down"),
    ("ksd", "KSD $\\downarrow$", "down"),
    ("energy_kl", "Energy KL $\\downarrow$", "down"),
    ("pred_acc", "Pred. acc. $\\uparrow$", "up"),
    ("path_logp_spearman", "Path Spearman $\\uparrow$", "up"),
    ("path_logp_pairwise_acc", "Pairwise acc. $\\uparrow$", "up"),
    ("path_logp_centered_rmse", "Path cRMSE $\\downarrow$", "down"),
]
BASE_TABLE_METHODS = ["tweedie", "blend", "matrix-blend", "lfgi"]
TABLE_METHOD_LABELS = {
    "tweedie": "Tweedie",
    "blend": "Scalar blend",
    "matrix-blend": "Matrix blend",
    "moment-matrix-blend": "Matrix blend",
    "primal-matrix-blend": "Matrix blend",
    "lfgi": "LFGI",
    "ce-hlsi": "LFGI",
    "reference": "Reference",
}


def _float_or_nan(x) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _strip_repeated_overrides(argv: List[str]) -> List[str]:
    """Remove CLI options that parent repeated mode overrides for each child run."""
    value_opts = {"--n_runs", "--out_dir", "--seed", "--aggregate_uncertainty"}
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in value_opts:
            i += 2
            continue
        if any(tok.startswith(opt + "=") for opt in value_opts):
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _read_metrics_csv(path: Path, run_idx: int, seed: int) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    metric_keys: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            method = str(raw.get("method", "")).strip()
            if not method:
                continue
            rec: Dict[str, object] = {"run": int(run_idx), "seed": int(seed), "method": method}
            for k, v in raw.items():
                if k == "method":
                    continue
                if k not in metric_keys:
                    metric_keys.append(k)
                rec[k] = _float_or_nan(v)
            rows.append(rec)
    return rows, metric_keys


def _method_order(methods: List[str]) -> List[str]:
    preferred = ["tweedie", "blend", "matrix-blend", "moment-matrix-blend", "primal-matrix-blend", "lfgi", "ce-hlsi", "reference"]
    out = [m for m in preferred if m in methods]
    out += sorted([m for m in methods if m not in out])
    return out


def _aggregate_rows(rows: List[Dict[str, object]], metric_keys: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    methods = _method_order(sorted({str(r["method"]) for r in rows}))
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in methods:
        mrows = [r for r in rows if r["method"] == m]
        agg[m] = {}
        for k in metric_keys:
            vals = np.asarray([_float_or_nan(r.get(k, float("nan"))) for r in mrows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                mean = std = sem = float("nan")
                n = 0
            else:
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                sem = float(std / math.sqrt(vals.size)) if vals.size > 0 else float("nan")
                n = int(vals.size)
            agg[m][k] = {"mean": mean, "std": std, "sem": sem, "n": float(n)}
    return agg


def _write_per_run_metrics(rows: List[Dict[str, object]], metric_keys: List[str], out_path: Path) -> None:
    fieldnames = ["run", "seed", "method"] + list(metric_keys)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_aggregate_metrics(agg: Dict[str, Dict[str, Dict[str, float]]], metric_keys: List[str], out_path: Path) -> None:
    fieldnames = ["method"]
    for k in metric_keys:
        fieldnames += [f"{k}_mean", f"{k}_std", f"{k}_sem", f"{k}_n"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, mdict in agg.items():
            row = {"method": method}
            for k in metric_keys:
                stats = mdict.get(k, {})
                row[f"{k}_mean"] = stats.get("mean", float("nan"))
                row[f"{k}_std"] = stats.get("std", float("nan"))
                row[f"{k}_sem"] = stats.get("sem", float("nan"))
                row[f"{k}_n"] = int(stats.get("n", 0)) if np.isfinite(stats.get("n", float("nan"))) else 0
            writer.writerow(row)


def _latex_number(x: float, *, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    ax = abs(float(x))
    if ax == 0:
        return "0"
    if ax >= 1000.0 or ax < 1e-2:
        exp = int(math.floor(math.log10(ax)))
        mant = float(x) / (10.0 ** exp)
        return f"{mant:.{max(digits-1, 1)}f}{{\\times}}10^{{{exp}}}"
    if ax >= 100.0:
        return f"{x:.1f}"
    if ax >= 10.0:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _latex_mean_unc(mean: float, unc: float) -> str:
    if not np.isfinite(mean):
        return "$--$"
    if not np.isfinite(unc) or unc == 0:
        return f"${_latex_number(mean)}$"
    return f"${_latex_number(mean)}\\pm{_latex_number(unc)}$"


def _write_latex_aggregate_table(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
    *,
    n_runs: int,
    uncertainty: str,
    n_ref: int,
    n_gate: int,
    d: int,
    classes: Tuple[int, int],
) -> None:
    uncertainty = uncertainty.lower().strip()
    if uncertainty not in {"std", "sem"}:
        raise ValueError("uncertainty must be 'std' or 'sem'")
    available = [m for m in BASE_TABLE_METHODS if m in agg]
    # Backward compatibility if an older run still wrote ce-hlsi.
    if "lfgi" not in available and "ce-hlsi" in agg:
        available.append("ce-hlsi")

    best: Dict[str, str] = {}
    for key, _, direction in TABLE_METRICS:
        vals = []
        for m in available:
            mean = agg.get(m, {}).get(key, {}).get("mean", float("nan"))
            if np.isfinite(mean):
                vals.append((m, mean))
        if vals:
            best[key] = (max(vals, key=lambda p: p[1]) if direction == "up" else min(vals, key=lambda p: p[1]))[0]

    headers = " & ".join(["Method"] + [label for _, label, _ in TABLE_METRICS])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(TABLE_METRICS) + "}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for m in available:
        label = TABLE_METHOD_LABELS.get(m, method_label(m).title())
        cells = [label]
        for key, _, _ in TABLE_METRICS:
            stats = agg.get(m, {}).get(key, {})
            mean = stats.get("mean", float("nan"))
            unc = stats.get(uncertainty, float("nan"))
            cell = _latex_mean_unc(mean, unc)
            if best.get(key) == m and np.isfinite(mean):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    ref_acc = agg.get("reference", {}).get("pred_acc", {}).get("mean", float("nan"))
    ref_clause = ""
    if np.isfinite(ref_acc):
        ref_clause = f" The reference posterior predictive accuracy is ${_latex_number(ref_acc)}$."
    unc_name = "standard deviation" if uncertainty == "std" else "standard error"
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\caption{{MNIST ${classes[0]}$-vs-${classes[1]}$ PCA logistic-regression posterior, $d={d}$, $N_{{\rm ref}}={n_ref}$, $N_{{\rm gate}}={n_gate}$. Entries report mean $\pm$ {unc_name} over {n_runs} independent runs. Predictive accuracy is already saturated by Tweedie and LFGI, but LFGI is the only estimator that simultaneously improves posterior sample quality and recovers a usable posterior likelihood ordering from score path integration.{ref_clause}}}",
        r"\label{tab:mnist-pca-logreg-results}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines))


def aggregate_repeated_outputs(run_dirs: List[Path], seeds: List[int], out_dir: Path, args) -> None:
    rows: List[Dict[str, object]] = []
    metric_keys_ordered: List[str] = []
    for r, (run_dir, seed) in enumerate(zip(run_dirs, seeds)):
        run_rows, keys = _read_metrics_csv(run_dir / "metrics.csv", run_idx=r, seed=seed)
        rows.extend(run_rows)
        for k in keys:
            if k not in metric_keys_ordered:
                metric_keys_ordered.append(k)
    if not rows:
        raise RuntimeError("No per-run metrics rows were found; cannot aggregate repeated run.")

    _write_per_run_metrics(rows, metric_keys_ordered, out_dir / "per_run_metrics.csv")
    agg = _aggregate_rows(rows, metric_keys_ordered)
    _write_aggregate_metrics(agg, metric_keys_ordered, out_dir / "aggregate_metrics.csv")
    _write_latex_aggregate_table(
        agg,
        out_dir / "mnist_table_aggregate.tex",
        n_runs=int(args.n_runs),
        uncertainty=str(args.aggregate_uncertainty),
        n_ref=int(args.n_ref),
        n_gate=int(args.n_gate),
        d=int(args.d),
        classes=tuple(args.classes),
    )

    with open(out_dir / "aggregate_summary.txt", "w") as f:
        f.write("MNIST PCA logistic-regression repeated-run aggregate\n")
        f.write(f"n_runs={int(args.n_runs)}\n")
        f.write(f"base_seed={int(args.seed)}\n")
        f.write(f"seed_stride={int(args.n_runs_seed_stride)}\n")
        f.write(f"uncertainty={str(args.aggregate_uncertainty)}\n")
        f.write("\nRun directories:\n")
        for r, (run_dir, seed) in enumerate(zip(run_dirs, seeds)):
            f.write(f"  run={r} seed={seed} dir={run_dir}\n")
        f.write("\nFiles:\n")
        f.write("  per_run_metrics.csv\n")
        f.write("  aggregate_metrics.csv\n")
        f.write("  mnist_table_aggregate.tex\n")


def run_repeated_cli(args, argv: List[str]) -> None:
    n_runs = int(args.n_runs)
    if n_runs <= 1:
        return
    if bool(getattr(args, "ng_nr_sweep", False)):
        raise ValueError("--n_runs aggregation currently targets the main MNIST benchmark, not --ng_nr_sweep.")
    if bool(getattr(args, "skip_sampling", False)):
        raise ValueError("--n_runs requires sampling/metrics; remove --skip_sampling.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cli = _strip_repeated_overrides(list(argv))
    script_path = Path(__file__).resolve()
    seeds = [int(args.seed) + int(args.n_runs_seed_stride) * r for r in range(n_runs)]
    run_dirs = [out_dir / f"run_{r:03d}" for r in range(n_runs)]

    print("=" * 80)
    print(f"MNIST repeated-run mode: n_runs={n_runs}, base_seed={args.seed}, stride={args.n_runs_seed_stride}")
    print(f"Output root: {out_dir}")
    print("Each run writes an ordinary single-run output directory; aggregate CSV/LaTeX are written at the root.")
    print("=" * 80)

    for r, (run_dir, seed) in enumerate(zip(run_dirs, seeds)):
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script_path),
            *base_cli,
            "--n_runs", "1",
            "--out_dir", str(run_dir),
            "--seed", str(seed),
        ]
        print("-" * 80)
        print(f"[n_runs] run {r + 1}/{n_runs}: seed={seed}, out_dir={run_dir}")
        print("[n_runs] command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    aggregate_repeated_outputs(run_dirs, seeds, out_dir, args)
    print("=" * 80)
    print(f"[n_runs done] wrote aggregate outputs to {out_dir}")
    print(f"aggregate CSV: {out_dir / 'aggregate_metrics.csv'}")
    print(f"LaTeX table:   {out_dir / 'mnist_table_aggregate.tex'}")
    print("=" * 80)


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs_mnist_pca_logreg_d32")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float64")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_runs", type=int, default=1,
                   help="repeat the full benchmark this many times with seeds seed + r*stride, then aggregate metrics")
    p.add_argument("--n_runs_seed_stride", type=int, default=1,
                   help="seed increment between repeated runs")
    p.add_argument("--aggregate_uncertainty", type=str, default="std", choices=["std", "sem"],
                   help="uncertainty convention used in the generated aggregate LaTeX table")
    p.add_argument("--classes", type=int, nargs=2, default=(4, 9))
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--max_train", type=int, default=6000)
    p.add_argument("--max_test", type=int, default=2000)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--tau", type=float, default=50.0, help="weak Gaussian prior std")
    p.add_argument("--beta", type=float, default=50.0, help="likelihood temperature multiplying mean logistic loss")
    p.add_argument("--image_operator", type=str, default="none", choices=["none", "anisotropic_mask", "sensor", "navier_mask"],
                   help="optional image-domain anisotropic sensor/mask before PCA")
    p.add_argument("--image_operator_strength", type=float, default=1.0,
                   help="1.0 uses only masked/blurred pixels; lower values mix in global blurred image")
    p.add_argument("--feature_operator", type=str, default="spectral", choices=["none", "spectral"],
                   help="right-acting anisotropic feature operator applied after PCA")
    p.add_argument("--operator_stiff_rank", type=int, default=6)
    p.add_argument("--operator_mid_rank", type=int, default=10)
    p.add_argument("--operator_stiff_scale", type=float, default=3.0)
    p.add_argument("--operator_mid_scale", type=float, default=1.0)
    p.add_argument("--operator_sloppy_scale", type=float, default=0.05)
    p.add_argument("--operator_random_rotate", type=int, default=1,
                   help="randomly rotate the stiff/sloppy split so it is not coordinate-aligned")
    p.add_argument("--n_ref", type=int, default=512)
    p.add_argument("--n_gate", type=int, default=512,
                   help=("gate samples. With --bank_coupling=prefix, require n_gate >= n_ref and "
                         "xg[:n_ref] == xr. With --bank_coupling=independent, xr and xg are disjoint posterior draws."))
    p.add_argument("--bank_coupling", type=str, default="prefix", choices=["prefix", "independent"],
                   help=("finite-bank construction: 'prefix' makes the estimator bank the first n_ref samples "
                         "of the gate bank; 'independent' uses disjoint independently drawn posterior samples "
                         "for estimator/base signals and gate/Hessian aggregation."))
    p.add_argument("--n_test_ref", type=int, default=12000)
    p.add_argument("--n_gen", type=int, default=8000)
    p.add_argument("--mala_step", type=float, default=0.15)
    p.add_argument("--mala_burnin", type=int, default=4000)
    p.add_argument("--mala_thin", type=int, default=10)
    p.add_argument("--mala_chains", type=int, default=128)
    p.add_argument("--n_steps", type=int, default=300)
    p.add_argument("--t_min", type=float, default=0.01)
    p.add_argument("--t_max", type=float, default=3.0)
    p.add_argument("--metric_max", type=int, default=2000)
    p.add_argument("--score_rmse_batch", type=int, default=512)
    p.add_argument("--score_rmse_tgrid", type=int, default=16)
    p.add_argument("--path_likelihood", action="store_true", default=True,
                   help="run straight-line score path-integration log-density diagnostics")
    p.add_argument("--no_path_likelihood", dest="path_likelihood", action="store_false",
                   help="disable straight-line score path-integration diagnostics")
    p.add_argument("--path_likelihood_samples", type=int, default=1024)
    p.add_argument("--path_likelihood_quad", type=int, default=32)
    p.add_argument("--path_likelihood_batch", type=int, default=128)
    p.add_argument("--path_likelihood_t", type=float, default=None,
                   help="score time used for clean-density path integration; defaults to --t_min")
    p.add_argument("--path_likelihood_pairs", type=int, default=20000,
                   help="number of random pairs for likelihood-order accuracy")
    p.add_argument("--skip_sampling", action="store_true", help="only build target/reference diagnostics")
    p.add_argument("--add_bank_ablations", action="store_true",
                   help=("also run diagnostic ablations that use the full gate bank as the estimator bank "
                         "and LFGI with only the estimator-prefix gate; useful when n_gate >> n_ref"))
    p.add_argument("--ng_nr_sweep", action="store_true",
                   help=("run an N_G/N_R ablation instead of the single benchmark: fix N_G and sweep the "
                         "LFGI score-signal bank size N_R. Same-N_G blend/Tweedie/CE baselines are also run."))
    p.add_argument("--sweep_n_gate", type=int, default=None,
                   help="fixed N_G for --ng_nr_sweep; defaults to --n_gate")
    p.add_argument("--sweep_n_ref_values", type=str, default="1:100",
                   help="N_R values for --ng_nr_sweep, e.g. '1:100', '1:5:100', or '1,2,5,10,20,50,100'")
    p.add_argument("--sweep_save_samples", action="store_true",
                   help="save every LFGI sweep sample array; off by default to avoid huge outputs")
    p.add_argument("--sweep_score_rmse", action="store_true",
                   help="compute score-RMSE for a few representative N_R values during the sweep")
    args = p.parse_args()

    if int(args.n_runs) > 1:
        run_repeated_cli(args, sys.argv[1:])
        return

    set_seed(args.seed)
    dtype = resolve_dtype(args.dtype)
    torch.set_default_dtype(dtype)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ng_nr_sweep:
        if args.sweep_n_gate is None:
            args.sweep_n_gate = int(args.n_gate)
        args.n_gate = int(args.sweep_n_gate)
        # N_R is varied internally.  Respect --bank_coupling:
        #   prefix      : all swept score banks are prefixes of the fixed N_G gate bank;
        #                 this is the same-pool/subsample ablation.
        #   independent : swept score banks are drawn from a disjoint score pool.
        # In either case, allocate enough score-pool samples for max(N_R).
        sweep_nrefs_preview = parse_int_sweep(args.sweep_n_ref_values, max_value=args.n_gate)
        args.n_ref = max(sweep_nrefs_preview)

    print("=" * 80)
    print("MNIST PCA logistic-regression LFGI benchmark")
    print(f"device={device}, dtype={dtype}, d={args.d}, classes={tuple(args.classes)}")
    print(f"tau={args.tau}, beta={args.beta}, n_ref={args.n_ref}, n_gate={args.n_gate}, bank_coupling={args.bank_coupling}")
    if args.bank_coupling == "prefix" and args.n_gate < args.n_ref:
        raise ValueError(
            f"Prefix-coupled gate banks require n_gate >= n_ref, got n_gate={args.n_gate} < n_ref={args.n_ref}. "
            "Use --bank_coupling independent for disjoint score/gate banks with arbitrary sizes."
        )
    print(f"image_operator={args.image_operator}, feature_operator={args.feature_operator}, "
          f"ranks=({args.operator_stiff_rank},{args.operator_mid_rank}), "
          f"scales=({args.operator_stiff_scale},{args.operator_mid_scale},{args.operator_sloppy_scale}), "
          f"random_rotate={bool(args.operator_random_rotate)}")
    print("=" * 80)

    data = load_mnist_pca_features(
        d=args.d,
        classes=tuple(args.classes),
        root=args.data_root,
        max_train=args.max_train,
        max_test=args.max_test,
        dtype=dtype,
        device=device,
        no_whiten=True,
        image_operator=args.image_operator,
        image_operator_strength=args.image_operator_strength,
        feature_operator=args.feature_operator,
        operator_stiff_rank=args.operator_stiff_rank,
        operator_mid_rank=args.operator_mid_rank,
        operator_stiff_scale=args.operator_stiff_scale,
        operator_mid_scale=args.operator_mid_scale,
        operator_sloppy_scale=args.operator_sloppy_scale,
        operator_random_rotate=bool(args.operator_random_rotate),
        operator_seed=args.seed,
    )
    target = LogisticPosterior(
        X=data["X_train"], y=data["y_train"],
        X_test=data["X_test"], y_test=data["y_test"],
        tau=args.tau, beta=args.beta,
        name=f"mnist_{args.classes[0]}v{args.classes[1]}_illcond_pca_logreg_d{args.d}",
    )
    print(f"[data] train={len(target.X)}, test={len(target.X_test)}, d={target.D}")
    op_stats: Dict[str, float] = {}
    op_stats.update(feature_covariance_stats(data["X_train_pre_operator"], "pre_operator"))
    op_stats.update(feature_covariance_stats(data["X_train"], "post_operator"))
    scales = data["feature_operator_scales"]
    op_stats["operator_scale_min"] = float(scales.min().detach().cpu())
    op_stats["operator_scale_max"] = float(scales.max().detach().cpu())
    op_stats["operator_scale_cond"] = float((scales.max() / scales.min().clamp_min(1e-30)).detach().cpu())
    plot_measurement_operator(data, out_dir)
    print(f"[operator] scale_cond={op_stats['operator_scale_cond']:.3e}, "
          f"pre_cov_cond={op_stats['pre_operator_cov_cond']:.3e}, "
          f"post_cov_cond={op_stats['post_operator_cov_cond']:.3e}")

    theta_map, H_evals, H_evecs = find_map_and_metric(target)

    print("[svm] training linear SVM reference direction in posterior feature space")
    svm_info = train_linear_svm_direction(target)
    print(f"[svm] method={svm_info['method']}; train_acc={float(svm_info['train_acc']):.4f}; test_acc={float(svm_info['test_acc']):.4f}")
    plot_svm_reference_weight(svm_info, data, out_dir / "svm_reference_weight.png")

    n_bank_needed = args.n_gate if args.bank_coupling == "prefix" else (args.n_ref + args.n_gate)
    n_total_ref = args.n_test_ref + n_bank_needed + 1024
    print(f"[reference] drawing {n_total_ref} posterior samples")
    X_all = sample_reference_mala_preconditioned(
        target,
        n=n_total_ref,
        theta_map=theta_map,
        metric_evals=H_evals,
        metric_evecs=H_evecs,
        step_size=args.mala_step,
        n_chains=args.mala_chains,
        burnin=args.mala_burnin,
        thin=args.mala_thin,
        verbose=True,
    )
    perm = torch.randperm(len(X_all), device=device)
    X_all = X_all[perm]
    # Finite bank construction.
    #   prefix      : xr is the first n_ref samples of xg, so xg[:n_ref] == xr.
    #   independent : xr and xg are disjoint posterior draws, intentionally
    #                 removing score/gate sample coupling.
    if args.bank_coupling == "prefix":
        n_bank = args.n_gate
        X_bank_all = X_all[:n_bank]
        xr = X_bank_all[:args.n_ref]
        xg = xr if args.n_gate == args.n_ref else X_bank_all[:args.n_gate]
        X_test_ref = X_all[n_bank:n_bank + args.n_test_ref]
        prefix_err = float((xg[:args.n_ref] - xr).abs().max().detach().cpu())
        coupling_msg = f"xg[:n_ref] == xr max_abs_err={prefix_err:.3e}"
    elif args.bank_coupling == "independent":
        xr = X_all[:args.n_ref]
        xg = X_all[args.n_ref:args.n_ref + args.n_gate]
        X_test_ref = X_all[args.n_ref + args.n_gate:args.n_ref + args.n_gate + args.n_test_ref]
        prefix_err = float("nan")
        # Sanity check: these are slices of a shuffled MALA pool with disjoint indices.
        # Equality should be impossible except by exact duplicate floating-point states.
        overlap_err = float("nan")
        if args.n_ref == args.n_gate and args.n_ref > 0:
            overlap_err = float((xr - xg).abs().max().detach().cpu())
        coupling_msg = (
            "independent/disjoint banks; "
            f"xr range=[0,{args.n_ref}), xg range=[{args.n_ref},{args.n_ref + args.n_gate}); "
            f"same_size_max_abs_diff={overlap_err:.3e}"
        )
    else:
        raise ValueError(f"Unknown bank_coupling={args.bank_coupling!r}")
    print(
        f"[reference] {args.bank_coupling} banks: xr={tuple(xr.shape)}, xg={tuple(xg.shape)}, "
        f"{coupling_msg}"
    )
    if args.n_ref < 16:
        print(
            "[warning] n_ref is extremely small. Blend and Tweedie use only the estimator bank, "
            "whereas Matrix Blend and LFGI can use the larger gate bank for gate aggregation when n_gate > n_ref. "
            "This is a gate-sample-complexity ablation, not a fair same-information sampler comparison."
        )
    if args.n_gate > 2 * max(args.n_ref, 1):
        print(
            "[warning] n_gate is much larger than n_ref. If LFGI improves only in this setting, "
            "the improvement is likely coming from the larger curvature atlas. Use --add_bank_ablations "
            "to compare against blend/tweedie using the same full bank and LFGI using only the score bank as gate. "
            "Use --bank_coupling independent to test whether score/gate sample coupling is responsible."
        )

    print("[diagnostics] Hessian spectrum on reference samples")
    hstats = hessian_diagnostics(target, X_test_ref, out_dir)
    pred_ref = target.predictive_metrics(X_test_ref[:min(len(X_test_ref), 5000)])

    if args.ng_nr_sweep:
        # In sweep mode, xr_pool depends on the requested coupling.
        #   prefix:      sweep over prefixes of the fixed gate bank xg.
        #   independent: sweep over prefixes of the disjoint score bank xr.
        xr_pool_for_sweep = xg if args.bank_coupling == "prefix" else xr
        run_ng_nr_sweep(
            args=args,
            target=target,
            xr_pool=xr_pool_for_sweep,
            xg=xg,
            X_test_ref=X_test_ref,
            pred_ref=pred_ref,
            out_dir=out_dir,
            dtype=dtype,
            device=device,
        )
        return

    # Save a text diagnostic summary early.
    with open(out_dir / "diagnostic_summary.txt", "w") as f:
        f.write(f"target={target.name}\n")
        f.write(f"classes={tuple(args.classes)} d={args.d} train={len(target.X)} test={len(target.X_test)}\n")
        f.write(f"tau={args.tau} beta={args.beta} prior_prec={target.prior_prec}\n")
        f.write(f"image_operator={args.image_operator} image_operator_strength={args.image_operator_strength}\n")
        f.write(f"feature_operator={args.feature_operator} random_rotate={bool(args.operator_random_rotate)}\n")
        f.write(f"operator_ranks=({args.operator_stiff_rank},{args.operator_mid_rank}) operator_scales=({args.operator_stiff_scale},{args.operator_mid_scale},{args.operator_sloppy_scale})\n")
        f.write(f"n_ref={args.n_ref} n_gate={args.n_gate} bank_coupling={args.bank_coupling}\n")
        f.write(f"gate_prefix_max_abs_err={prefix_err:.6e}\n")
        f.write(f"add_bank_ablations={bool(args.add_bank_ablations)}\n")
        for k, v in op_stats.items():
            f.write(f"{k}={v}\n")
        f.write(f"MAP Hessian min={float(H_evals.min().cpu()):.6e} max={float(H_evals.max().cpu()):.6e} cond={float((H_evals.max()/H_evals.min()).cpu()):.6e}\n")
        for k, v in hstats.items():
            f.write(f"{k}={v}\n")
        f.write(f"reference_pred_nll={pred_ref['pred_nll']}\nreference_pred_acc={pred_ref['pred_acc']}\n")

    if args.skip_sampling:
        print("[done] skip_sampling set")
        return

    bank = EstimatorBank(target, xr=xr, xg=xg)
    router = ScoreRouter()
    router.add("tweedie", bank, "tweedie")
    router.add("blend", bank, "blend")
    router.add("matrix-blend", bank, "matrix-blend")
    router.add("lfgi", bank, "lfgi")

    if args.add_bank_ablations and args.n_gate > args.n_ref:
        print("[ablation] adding full-gate-bank and score-bank-gate diagnostic methods")
        # Same-information comparison: give all methods the full gate bank as
        # their estimator bank.  This checks whether LFGI is winning only
        # because it has access to more samples than blend/Tweedie.
        bank_xg = EstimatorBank(target, xr=xg, xg=xg)
        router.add("tweedie-xg-bank", bank_xg, "tweedie")
        router.add("blend-xg-bank", bank_xg, "blend")
        router.add("matrix-blend-xg-bank", bank_xg, "matrix-blend")
        router.add("lfgi-xg-bank", bank_xg, "lfgi")
        # Gate-ablation comparison: force LFGI and Matrix Blend to use only the
        # score/estimator bank for gate construction, even though a larger xg exists.
        bank_xr_gate = EstimatorBank(target, xr=xr, xg=xr)
        router.add("matrix-blend-xr-gate", bank_xr_gate, "matrix-blend")
        router.add("lfgi-xr-gate", bank_xr_gate, "lfgi")

    methods = router.methods

    samples: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for m in methods:
        print("-" * 80)
        print(f"[sample] {m}")
        t0 = time.time()
        score_fn = lambda y, t, method=m: router.score(method, y, t)
        xs, max_score, fail = heun_reverse_sde(
            score_fn,
            n=args.n_gen,
            d=target.D,
            n_steps=args.n_steps,
            t_max=args.t_max,
            t_min=args.t_min,
            device=device,
            dtype=dtype,
        )
        dt = time.time() - t0
        samples[m] = xs
        np.save(out_dir / f"samples_{m.replace('-', '_')}.npy", xs.detach().cpu().numpy())
        print(f"[sample] {m}: time={dt:.1f}s, max|score|={max_score:.3e}, fail={fail}")

        met = {
            "mmd": mmd_rbf(X_test_ref, xs, n_max=args.metric_max),
            "sliced_ks": sliced_ks(X_test_ref, xs, n_proj=256, n_max=min(4096, args.metric_max * 2)),
            "ksd": ksd_rbf(xs, target.score, n_max=min(1000, args.metric_max)),
            "energy_kl": energy_kl_hist(X_test_ref, xs, target),
            "ess_proxy": ess_energy_proxy(target, xs),
            "max_score": max_score,
            "failed": float(fail),
            "runtime_sec": dt,
        }
        met.update(target.predictive_metrics(xs[:min(len(xs), 5000)]))
        metrics[m] = met
        print("[metrics]", m, met)

    print("[score rmse] computing high-N proxy curves")
    t_grid = torch.linspace(args.t_min, args.t_max, args.score_rmse_tgrid, dtype=dtype, device=device)
    curves, avg_rmse = score_rmse_vs_t(
        bank=router,
        x_true_bank=X_test_ref[:min(len(X_test_ref), 10000)],
        methods=methods,
        t_grid=t_grid,
        batch=args.score_rmse_batch,
    )
    for m in methods:
        metrics[m]["score_rmse"] = avg_rmse[m]

    path_metrics: Dict[str, Dict[str, float]] = {}
    if args.path_likelihood:
        print("[path-likelihood] computing straight-line score path-integration diagnostics")
        t_path = args.t_min if args.path_likelihood_t is None else float(args.path_likelihood_t)
        path_metrics, path_arrays = path_likelihood_diagnostics(
            target=target,
            router=router,
            methods=methods,
            samples=samples,
            anchor=theta_map,
            out_dir=out_dir,
            t_eval=t_path,
            n_samples=args.path_likelihood_samples,
            n_quad=args.path_likelihood_quad,
            batch=args.path_likelihood_batch,
            pairwise_pairs=args.path_likelihood_pairs,
            seed=args.seed,
        )
        augment_path_arrays_with_svm(path_arrays, svm_info)
        add_svm_alignment_metrics(path_metrics, path_arrays, svm_info)
        for m in methods:
            metrics[m].update(path_metrics[m])
        plot_path_likelihood_bars(path_metrics, out_dir / "path_likelihood_bars.png")
        plot_path_likelihood_cloud_error(X_test_ref, path_arrays, out_dir / "path_likelihood_cloud_error.png")
        # Keep this for completeness, but the SVM-alignment scatter is now the clearer rank visualization.
        plot_path_likelihood_cloud_rank(X_test_ref, path_arrays, out_dir / "path_likelihood_cloud_rank.png")
        plot_path_likelihood_svm_alignment(path_arrays, svm_info, out_dir / "path_likelihood_svm_alignment.png")
        plot_path_likelihood_weight_montage(path_arrays, data, target, out_dir / "path_likelihood_weight_montage.png", svm_info=svm_info)
        plot_path_likelihood_weight_residual_montage(path_arrays, data, out_dir / "path_likelihood_weight_residual_montage.png", svm_info=svm_info)

    # Add reference row for predictive metrics only.
    metrics_ref = {"pred_nll": pred_ref["pred_nll"], "pred_acc": pred_ref["pred_acc"]}

    # CSV metrics.  Keep runtime_sec near the front because it is a method-level
    # diagnostic rather than a posterior-quality metric.
    metric_keys_all = sorted({k for dct in metrics.values() for k in dct.keys()})
    metric_keys = ["runtime_sec"] + [k for k in metric_keys_all if k != "runtime_sec"]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + metric_keys)
        for m in methods:
            writer.writerow([m] + [metrics[m].get(k, "") for k in metric_keys])
        writer.writerow(["reference"] + [metrics_ref.get(k, "") for k in metric_keys])

    # Plots.
    title = f"MNIST {args.classes[0]} vs {args.classes[1]} ill-conditioned PCA logistic posterior, d={args.d}, tau={args.tau}, beta={args.beta}"
    plot_heatmaps(X_test_ref, samples, out_dir / "heatmaps.png", title=title)
    plot_metric_bars(metrics, out_dir / "metric_bars.png")
    plot_energy_hist(target, X_test_ref, samples, out_dir / "energy_hist.png")
    plot_score_rmse_curves(t_grid, curves, out_dir / "score_rmse_vs_t.png")

    # Predictive bar chart separately so it can include reference.
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.35))
    labels = [method_label(m) for m in methods] + ["REFERENCE"]
    colors = [method_color(m) for m in methods] + [method_color("reference")]
    pred_nlls = [metrics[m]["pred_nll"] for m in methods] + [pred_ref["pred_nll"]]
    pred_accs = [metrics[m]["pred_acc"] for m in methods] + [pred_ref["pred_acc"]]
    axs[0].bar(labels, pred_nlls, color=colors, alpha=0.92); axs[0].set_title("Test Predictive NLL"); axs[0].tick_params(axis="x", rotation=25)
    axs[1].bar(labels, pred_accs, color=colors, alpha=0.92); axs[1].set_title("Test Predictive Accuracy"); axs[1].tick_params(axis="x", rotation=25)
    anchor_methods = _lfgi_scale_names(methods)
    _set_ylim_from_anchor(axs[0], [metrics[m]["pred_nll"] for m in anchor_methods if m in metrics], include_zero=True)
    _set_ylim_from_anchor(axs[1], [metrics[m]["pred_acc"] for m in anchor_methods if m in metrics], include_zero=True)
    for ax in axs: style_axis(ax, grid_axis="y")
    save_pub(fig, out_dir / "predictive_bars.png")

    # Dashboard PDF.
    with PdfPages(out_dir / "mnist_pca_logreg_dashboard.pdf") as pdf:
        # Summary page.
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            "MNIST PCA Bayesian logistic-regression LFGI benchmark",
            "",
            f"target: {target.name}",
            f"classes: {tuple(args.classes)}; d={args.d}; train={len(target.X)}; test={len(target.X_test)}",
            f"tau={args.tau}; beta={args.beta}; prior_prec={target.prior_prec:.3e}",
            f"image_operator={args.image_operator}; feature_operator={args.feature_operator}",
            f"operator ranks=({args.operator_stiff_rank},{args.operator_mid_rank}); scales=({args.operator_stiff_scale},{args.operator_mid_scale},{args.operator_sloppy_scale}); rotate={bool(args.operator_random_rotate)}",
            f"n_ref={args.n_ref}; n_gate={args.n_gate}; bank_coupling={args.bank_coupling}; n_gen={args.n_gen}; n_test_ref={args.n_test_ref}; add_bank_ablations={bool(args.add_bank_ablations)}",
            f"reverse SDE: steps={args.n_steps}; t_min={args.t_min}; t_max={args.t_max}",
            f"linear SVM reference: test_acc={float(svm_info['test_acc']):.4f}; train_acc={float(svm_info['train_acc']):.4f}; fit={svm_info['method']}",
            "",
            "Feature/operator conditioning diagnostics:",
        ]
        for k, v in op_stats.items():
            lines.append(f"  {k}: {v:.6g}")
        lines += ["", "Hessian admissibility / score singularity diagnostics:"]
        for k, v in hstats.items():
            lines.append(f"  {k}: {v:.6g}")
        lines += ["", "Metrics:"]
        for m in methods:
            vals = ", ".join(f"{k}={metrics[m][k]:.4g}" for k in ["mmd", "sliced_ks", "ksd", "score_rmse", "energy_kl", "pred_nll", "pred_acc", "ess_proxy"])
            lines.append(f"  {m}: {vals}")
        if path_metrics:
            lines += ["", "Path-integrated clean log-density diagnostics:"]
            for m in methods:
                vals = ", ".join(
                    f"{k}={metrics[m][k]:.4g}" for k in [
                        "path_logp_rmse", "path_logp_centered_rmse", "path_logp_calib_rmse",
                        "path_logp_spearman", "path_logp_pairwise_acc", "path_svm_est_rank_corr", "path_svm_top10_cos_mean"
                    ] if k in metrics[m]
                )
                lines.append(f"  {m}: {vals}")
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig); plt.close(fig)
        dashboard_imgs = ["measurement_operator.png", "hessian_spectrum.png", "svm_reference_weight.png", "heatmaps.png", "metric_bars.png", "score_rmse_vs_t.png", "energy_hist.png", "predictive_bars.png"]
        if path_metrics:
            dashboard_imgs += [
                "path_likelihood_bars.png",
                "path_likelihood_diagnostics.png",
                "path_likelihood_cloud_error.png",
                "path_likelihood_svm_alignment.png",
                "path_likelihood_weight_montage.png",
                "path_likelihood_weight_residual_montage.png",
                "path_likelihood_cloud_rank.png",
            ]
        for img in dashboard_imgs:
            arr = plt.imread(out_dir / img)
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(arr)
            ax.axis("off")
            ax.set_title(img)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("=" * 80)
    print(f"[done] wrote outputs to {out_dir}")
    print(f"dashboard: {out_dir / 'mnist_pca_logreg_dashboard.pdf'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
