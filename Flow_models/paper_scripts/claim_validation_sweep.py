#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lfgi_claim_validation_sweep_publication_ready.py

Compact publication-ready claim-validation harness for LFGI.

What it produces
----------------
1. Risk/metric-vs-N_ref sweeps on the paper-facing claim-validation target:
     - misaligned_subspace_gmm_d8
   For each target and each N_ref, it compares:
     - tweedie
     - blend        (coordinatewise plug-in scalar variance-minimizing blend)
     - matrix-blend (primal/moment-normal-equation matrix-valued gate blend)
     - lfgi         (operator-valued LFGI Hessian-resolvent gate)
   Metrics saved and plotted:
     - MMD to exact target samples
     - sliced W2 (SW2) to exact target samples
     - NLL under the exact target density
     - KSD under exact target score
     - time-averaged noisy-score RMSE against the exact OU-marginal score

2. Gate-estimation sample-complexity sweep on misaligned_subspace_gmm_d8:
     - primal moment gate: estimates the optimal operator gate directly from
       conditional moments of Tweedie/TSI atoms.
     - LFGI gate: estimates the same gate by posterior averaging clean Hessians
       and applying G = alpha^2 (alpha^2 I + gamma E[P|y,t])^{-1}.
   A large heldout clean pool is used as the oracle proxy. Gate error is reported
   in the risk-weighted norm induced by D = E[(c-b)(c-b)^T | y,t], not merely
   Frobenius norm.

3. Hessian-resolvent learnability diagnostics on misaligned_subspace_gmm_d8:
     - relative conditional-Hessian fluctuation v_A^2,
     - sharpened pole/cancellation factor alpha^4 Lambda_B,
     - oracle gain G_star, optionally normalized by tr(D).
   These populate the theory placeholder panels following the gate-capture
   figures in the manuscript.

Publication plotting defaults
-----------------------------
The generated figures use paper-facing method names, large typography,
print-safe line styles, and the color convention
TWEEDIE=red, SCALAR BLEND=blue, MATRIX BLEND=purple, LFGI=green.

The script intentionally avoids the full dashboard machinery: no histograms, no
PCA panels, no curl diagnostics, no auxiliary metrics. It is meant to populate the
opening experimental-validation section with simple line plots.

Example usage
-------------
python lfgi_claim_validation_sweep_publication_ready.py --out-dir claim_validation_out

Fast smoke test:
python lfgi_claim_validation_sweep_publication_ready.py --out-dir smoke --nref-grid 32 64 \
    --repeats 1 --n-samples 128 --n-metric 128 --n-score 128 --n-steps 12 \
    --gate-n-grid 32 64 --gate-n-query 8 --gate-n-oracle 512 \
    --targets misaligned_subspace_gmm_d8 --sw2-projections 128

GPU / dtype:
python lfgi_claim_validation_sweep_publication_ready.py --device cuda --dtype float32
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Publication plotting style
# -----------------------------------------------------------------------------

PUB_DPI = 450
PUB_FIGSIZE = (5.6, 3.75)
PUB_FIGSIZE_WIDE = (6.0, 3.8)

# Fixed method color convention requested for the paper:
# TWEEDIE = red, SCALAR BLEND = blue, MATRIX BLEND = purple, LFGI = green.
# Markers and line styles are also distinct so the curves remain interpretable
# in grayscale printouts.
METHOD_STYLES = {
    "tweedie": {"label": "TWEEDIE", "color": "#D62728", "marker": "o", "linestyle": "--"},
    "blend": {"label": "SCALAR BLEND", "color": "#1F77B4", "marker": "s", "linestyle": "-."},
    "matrix-blend": {"label": "MATRIX BLEND", "color": "#9467BD", "marker": "^", "linestyle": ":"},
    "lfgi": {"label": "LFGI", "color": "#2CA02C", "marker": "D", "linestyle": "-"},
    # Backward-compatible aliases for old raw logs / method names.
    "ce-hlsi": {"label": "LFGI", "color": "#2CA02C", "marker": "D", "linestyle": "-"},
    "primal-matrix-blend": {"label": "MATRIX BLEND", "color": "#9467BD", "marker": "^", "linestyle": ":"},
    "moment-matrix-blend": {"label": "MATRIX BLEND", "color": "#9467BD", "marker": "^", "linestyle": ":"},
}

GATE_STYLES = {
    "lfgi-hessian": {"label": "LFGI", "color": "#2CA02C", "marker": "D", "linestyle": "-"},
    "primal-moment": {"label": "MATRIX BLEND", "color": "#9467BD", "marker": "^", "linestyle": ":"},
}

TARGET_TITLES = {
    "misaligned_subspace_gmm_d8": r"Misaligned GMM ($d=8$)",
    "misaligned_subspace_gmm_d24": r"Misaligned GMM ($d=24$)",
    "singular_gaussian_d8": r"Singular Gaussian ($d=8$)",
}

METRIC_TITLES = {
    "mmd": "MMD vs. Reference Count",
    "sw2": "Sliced W2 vs. Reference Count",
    "nll": "NLL vs. Reference Count",
    "ksd": "KSD vs. Reference Count",
    "score_rmse": "Score RMSE vs. Reference Count",
}

METRIC_LABELS = {
    "mmd": "MMD (lower is better)",
    "sw2": "SW2 (lower is better)",
    "nll": "Average NLL",
    "ksd": "KSD (lower is better)",
    "score_rmse": "Score RMSE (lower is better)",
}

# For NLL, SCALAR BLEND and MATRIX BLEND can catastrophically blow up at small
# N_ref.  If the y-axis is scaled from all methods, the scientifically relevant
# separation between TWEEDIE and LFGI is visually compressed.  We therefore keep
# all methods plotted, but set the NLL axis limits from the comparison methods
# that remain in the credible regime.
NLL_YLIM_METHODS = ("tweedie", "lfgi", "ce-hlsi")

GATE_METRIC_TITLES = {
    "risk_weighted_rel_to_tweedie_gap": "Missed Oracle Risk Reduction",
    "risk_weighted_rel_to_D_trace": "Risk-Weighted Gate Error",
    "fro_relative_mean": "Relative Frobenius Gate Error",
}

GATE_METRIC_LABELS = {
    "risk_weighted_rel_to_tweedie_gap": "Missed oracle risk reduction / Tweedie gap",
    "risk_weighted_rel_to_D_trace": r"Risk-weighted gate error / tr$(D)$",
    "fro_relative_mean": "Relative Frobenius error",
}


def apply_publication_style() -> None:
    """Set rcParams for figures that remain readable after LaTeX minipage scaling."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": PUB_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.35,
        "lines.markersize": 6.5,
        "mathtext.fontset": "stix",
    })


def paper_target_title(target_name: str) -> str:
    return TARGET_TITLES.get(target_name, target_name.replace("_", " ").title())


def style_axis(
    ax: plt.Axes,
    *,
    log_x: bool = True,
    log_y: Optional[bool] = None,
    y_scale: str = "linear",
    symlog_linthresh: float = 1.0,
) -> None:
    """Apply publication style and choose a safe y-axis transform.

    ``log_y`` is kept for backwards compatibility with older calls in this
    script.  New metric plots should prefer ``y_scale`` with one of
    {"linear", "log", "symlog"}.  The symlog option is essential for NLL,
    because continuous-density NLL can legitimately be negative.
    """
    if log_y is not None:
        y_scale = "log" if log_y else "linear"

    if log_x:
        ax.set_xscale("log", base=2)

    if y_scale == "log":
        ax.set_yscale("log")
    elif y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=float(symlog_linthresh))
        ax.axhline(0.0, color="0.5", linewidth=0.9, alpha=0.70, zorder=1)
    elif y_scale != "linear":
        raise ValueError(f"Unknown y_scale={y_scale!r}; use 'linear', 'log', or 'symlog'.")

    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=4.5, width=0.9)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.7)


def save_publication_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout(pad=0.35)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=PUB_DPI)
    plt.close(fig)


def set_nll_focus_ylim(ax: plt.Axes, agg: List[Dict[str, object]]) -> None:
    """Set NLL y-limits from TWEEDIE/LFGI, not catastrophic blend outliers.

    SCALAR BLEND and MATRIX BLEND are still drawn, but values outside the
    focused range are clipped by the axis.  This makes the LFGI-vs-TWEEDIE
    separation readable while still showing when the moment-based comparators
    re-enter the credible NLL range at large N_ref.
    """
    vals: List[float] = []
    for r in agg:
        if str(r.get("method", "")).lower() not in NLL_YLIM_METHODS:
            continue
        mean = float(r.get("nll_mean", float("nan")))
        std = float(r.get("nll_std", 0.0))
        if not np.isfinite(mean):
            continue
        if not np.isfinite(std):
            std = 0.0
        vals.extend([mean - std, mean, mean + std])

    vals_arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals_arr.size < 2:
        return

    lo = float(np.min(vals_arr))
    hi = float(np.max(vals_arr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return

    span = hi - lo
    if span <= 1e-12:
        span = max(1.0, abs(hi))
    pad = 0.12 * span
    ax.set_ylim(lo - pad, hi + pad)

    # Small, unobtrusive note so the focused axis is not mistaken for missing
    # outlier data.
    ax.text(
        0.02,
        0.03,
        "NLL axis focused on TWEEDIE/LFGI",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="0.35",
    )


apply_publication_style()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    if isinstance(s, str):
        return [float(x) for x in s.replace(",", " ").split() if x]
    return list(s)


def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def at(t: torch.Tensor | float) -> torch.Tensor:
    return torch.exp(-torch.as_tensor(t, dtype=torch.get_default_dtype(), device=current_device()))


def vt(t: torch.Tensor | float) -> torch.Tensor:
    tt = torch.as_tensor(t, dtype=torch.get_default_dtype(), device=current_device())
    return 1.0 - torch.exp(-2.0 * tt)


_DEVICE: Optional[torch.device] = None


def current_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def set_runtime(device: str, dtype: str, seed: int) -> None:
    global _DEVICE
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device in {"cuda", "gpu"}:
        device = "cuda:0"
    _DEVICE = torch.device(device)
    if _DEVICE.type == "cuda":
        torch.cuda.set_device(_DEVICE)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype_l = dtype.lower()
    if dtype_l in {"float64", "fp64", "double"}:
        torch.set_default_dtype(torch.float64)
    elif dtype_l in {"float32", "fp32", "single"}:
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError(f"Unknown dtype {dtype!r}; use float64 or float32")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if _DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def to_device_np(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.get_default_dtype(), device=current_device())


def randperm_prefix(n: int, k: int) -> torch.Tensor:
    return torch.randperm(n, device=current_device())[:k]


def finite_mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# -----------------------------------------------------------------------------
# Analytic targets
# -----------------------------------------------------------------------------


class AnalyticTarget:
    name: str
    d: int

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(x) = -∇² log p0(x)."""
        raise NotImplementedError

    def score_t(self, y: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Exact OU-marginal score at time t."""
        raise NotImplementedError


class SingularGaussianD8Target(AnalyticTarget):
    """Near-singular rotated Gaussian in R^8.

    This matches the controlled claim-validation target in the larger benchmark:
    four moderate directions and four nearly singular directions. Scalar blend
    must compromise across directions; LFGI gates spectrally.
    """

    def __init__(self, seed: int = 17, eps_scale: float = 1.0, rotate: bool = True):
        self.name = "singular_gaussian_d8"
        self.d = 8
        self.seed = int(seed)
        self.eps_scale = float(eps_scale)
        sigmas = np.array([1.60, 0.90, 0.45, 0.20, 0.070, 0.040, 0.025, 0.016], dtype=np.float64)
        sigmas[4:] *= self.eps_scale
        rng = np.random.RandomState(self.seed)
        if rotate:
            A = rng.normal(size=(self.d, self.d))
            Q, R = np.linalg.qr(A)
            Q = Q @ np.diag(np.sign(np.diag(R)) + (np.diag(R) == 0))
        else:
            Q = np.eye(self.d)
        cov = Q @ np.diag(sigmas ** 2) @ Q.T
        prec = Q @ np.diag(1.0 / (sigmas ** 2)) @ Q.T
        chol = Q @ np.diag(sigmas)
        self.mean = torch.zeros(self.d, dtype=torch.get_default_dtype(), device=current_device())
        self.cov = to_device_np(cov)
        self.precision = to_device_np(prec)
        self.chol = to_device_np(chol)
        self.logdet_precision = float(np.linalg.slogdet(prec)[1])
        self.sigmas = sigmas

    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.d, dtype=torch.get_default_dtype(), device=current_device())
        return self.mean.unsqueeze(0) + z @ self.chol.T

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - self.mean.to(x).unsqueeze(0)
        P = self.precision.to(x)
        quad = torch.einsum("bi,ij,bj->b", xc, P, xc)
        const = 0.5 * self.logdet_precision - 0.5 * self.d * math.log(2.0 * math.pi)
        return const - 0.5 * quad

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return -(x - self.mean.to(x).unsqueeze(0)) @ self.precision.to(x).T

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        P = self.precision.to(x)
        return P.unsqueeze(0).expand(x.shape[0], -1, -1).clone()

    def score_t(self, y: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        a = at(t).to(y)
        g = vt(t).to(y)
        cov_t = a.square() * self.cov.to(y) + g * torch.eye(self.d, dtype=y.dtype, device=y.device)
        P_t = torch.linalg.inv(sym(cov_t))
        mean_t = a * self.mean.to(y)
        return -(y - mean_t.unsqueeze(0)) @ P_t.T


class MisalignedSingularSubspaceGMMTarget(AnalyticTarget):
    """Misaligned near-singular subspace Gaussian mixture.

    Each component is a Gaussian tube with broad tangent directions and stiff
    normal directions. Component subspaces are randomly misaligned, so direct
    moment estimation of an operator gate is statistically hard in the singular
    score regime, while LFGI estimates a posterior average of clean curvature.
    """

    def __init__(
        self,
        d: int = 8,
        rank: Optional[int] = None,
        n_components: int = 8,
        seed: int = 29,
        radius: Optional[float] = None,
        sigma_parallel: Optional[Sequence[float]] = None,
        sigma_perp: float = 0.035,
        jitter: float = 0.12,
    ):
        self.d = int(d)
        self.name = f"misaligned_subspace_gmm_d{self.d}"
        self.rank = int(rank if rank is not None else (3 if self.d <= 8 else 6))
        if not (0 < self.rank < self.d):
            raise ValueError("rank must satisfy 0 < rank < d")
        self.n_components = int(n_components)
        self.seed = int(seed)
        self.radius = float(radius if radius is not None else (3.0 if self.d <= 8 else 4.5))
        self.sigma_perp = float(sigma_perp)
        self.jitter = float(jitter)
        rng = np.random.RandomState(self.seed)

        if sigma_parallel is None:
            sigma_parallel = np.geomspace(1.10, 0.30, self.rank)
        sigma_parallel = np.asarray(sigma_parallel, dtype=np.float64)
        if sigma_parallel.shape[0] != self.rank:
            raise ValueError("sigma_parallel length must equal rank")
        base_sigmas = np.concatenate([
            sigma_parallel,
            self.sigma_perp * np.ones(self.d - self.rank, dtype=np.float64),
        ])

        raw = rng.normal(size=(self.n_components, self.d))
        envelope = np.ones(self.d, dtype=np.float64)
        envelope[self.rank:] = 0.35
        raw = raw * envelope[None, :]
        raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        means_np = self.radius * raw

        Ps, covs, chols, logdetPs, logdetCovs, sigmas_all = [], [], [], [], [], []
        for _ in range(self.n_components):
            A = rng.normal(size=(self.d, self.d))
            Q, R = np.linalg.qr(A)
            Q = Q @ np.diag(np.sign(np.diag(R)) + (np.diag(R) == 0))
            sig = base_sigmas.copy()
            perm = rng.permutation(self.rank)
            sig[:self.rank] = sig[:self.rank][perm]
            sig[:self.rank] *= np.exp(self.jitter * rng.normal(size=self.rank))
            sig[self.rank:] *= np.exp(0.5 * self.jitter * rng.normal(size=self.d - self.rank))
            sig = np.clip(sig, 0.5 * self.sigma_perp, None)
            cov = Q @ np.diag(sig ** 2) @ Q.T
            P = Q @ np.diag(1.0 / (sig ** 2)) @ Q.T
            chol = Q @ np.diag(sig)
            covs.append(cov)
            Ps.append(P)
            chols.append(chol)
            logdetPs.append(np.linalg.slogdet(P)[1])
            logdetCovs.append(np.linalg.slogdet(cov)[1])
            sigmas_all.append(sig)

        self.means = to_device_np(means_np)
        self.precisions = to_device_np(np.stack(Ps, axis=0))
        self.covs = to_device_np(np.stack(covs, axis=0))
        self.chols = to_device_np(np.stack(chols, axis=0))
        self.logdet_precisions = to_device_np(np.asarray(logdetPs))
        self.logdet_covs = to_device_np(np.asarray(logdetCovs))
        self.sigmas_all = np.stack(sigmas_all, axis=0)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.n_components, (n,), device=current_device())
        z = torch.randn(n, self.d, dtype=torch.get_default_dtype(), device=current_device())
        means = self.means.to(z)
        chols = self.chols.to(z)
        return means[idx] + torch.einsum("bi,bij->bj", z, chols[idx].transpose(-1, -2))

    def _component_log_scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means = self.means.to(x)
        Ps = self.precisions.to(x)
        logdetP = self.logdet_precisions.to(x)
        diff = x.unsqueeze(1) - means.unsqueeze(0)
        Pdiff = torch.einsum("kij,bkj->bki", Ps, diff)
        mahal = (diff * Pdiff).sum(-1)
        log_comp = 0.5 * logdetP.unsqueeze(0) - 0.5 * mahal
        log_comp = log_comp - 0.5 * self.d * math.log(2.0 * math.pi) - math.log(self.n_components)
        comp_scores = -Pdiff
        return log_comp, comp_scores

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_comp, _ = self._component_log_scores(x)
        return torch.logsumexp(log_comp, dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        log_comp, comp_scores = self._component_log_scores(x)
        w = torch.softmax(log_comp, dim=1)
        return torch.einsum("bk,bki->bi", w, comp_scores)

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        log_comp, comp_scores = self._component_log_scores(x)
        w = torch.softmax(log_comp, dim=1)
        Ps = self.precisions.to(x)
        Pbar = torch.einsum("bk,kij->bij", w, Ps)
        sbar = torch.einsum("bk,bki->bi", w, comp_scores)
        second = torch.einsum("bk,bki,bkj->bij", w, comp_scores, comp_scores)
        cov_scores = second - sbar.unsqueeze(-1) * sbar.unsqueeze(-2)
        return sym(Pbar - cov_scores)

    def score_t(self, y: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        a = at(t).to(y)
        g = vt(t).to(y)
        means_t = a * self.means.to(y)
        covs_t = a.square() * self.covs.to(y) + g * torch.eye(self.d, dtype=y.dtype, device=y.device).unsqueeze(0)
        Ps_t = torch.linalg.inv(sym(covs_t))
        # log N(y; a mu_k, cov_t_k) - log K
        sign, logdet = torch.linalg.slogdet(covs_t)
        diff = y.unsqueeze(1) - means_t.unsqueeze(0)
        Pdiff = torch.einsum("kij,bkj->bki", Ps_t, diff)
        mahal = (diff * Pdiff).sum(-1)
        log_comp = -0.5 * (self.d * math.log(2.0 * math.pi) + logdet.unsqueeze(0) + mahal)
        log_comp = log_comp - math.log(self.n_components)
        w = torch.softmax(log_comp, dim=1)
        comp_scores = -Pdiff
        return torch.einsum("bk,bki->bi", w, comp_scores)


def make_target(name: str) -> AnalyticTarget:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "singular_gaussian": "singular_gaussian_d8",
        "singular_gauss": "singular_gaussian_d8",
        "singular_gaussian_d8": "singular_gaussian_d8",
        "misaligned_gmm": "misaligned_subspace_gmm_d8",
        "singular_gmm": "misaligned_subspace_gmm_d8",
        "misaligned_subspace_gmm": "misaligned_subspace_gmm_d8",
        "misaligned_subspace_gmm_d8": "misaligned_subspace_gmm_d8",
    }
    key = aliases.get(key, key)
    if key == "singular_gaussian_d8":
        return SingularGaussianD8Target()
    if key == "misaligned_subspace_gmm_d8":
        return MisalignedSingularSubspaceGMMTarget(d=8, rank=3)
    if key == "misaligned_subspace_gmm_d24":
        return MisalignedSingularSubspaceGMMTarget(d=24, rank=4)
    raise ValueError(f"Unknown target {name!r}")


# -----------------------------------------------------------------------------
# SNIS score estimators
# -----------------------------------------------------------------------------


@dataclass
class ReferenceBank:
    x: torch.Tensor
    s0: torch.Tensor
    H: torch.Tensor


def make_reference_bank(target: AnalyticTarget, n: int) -> ReferenceBank:
    x = target.sample(n)
    s0 = target.score(x)
    H = sym(target.hessian(x))
    return ReferenceBank(x=x, s0=s0, H=H)


def snis_weights(y: torch.Tensor, t: float | torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
    a = at(t).to(y)
    g = vt(t).to(y).clamp_min(1e-30)
    diff = y.unsqueeze(1) - a * xr.unsqueeze(0)
    lw = -0.5 * diff.square().sum(-1) / g
    lw = lw - lw.max(dim=1, keepdim=True).values
    w = torch.exp(lw)
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-30)


def tweedie_from_weights(y: torch.Tensor, t: float | torch.Tensor, ref: ReferenceBank, w: torch.Tensor) -> torch.Tensor:
    a = at(t).to(y)
    g = vt(t).to(y).clamp_min(1e-30)
    atom = -(y.unsqueeze(1) - a * ref.x.unsqueeze(0)) / g
    return torch.einsum("bn,bnd->bd", w, atom)


def tsi_from_weights(y: torch.Tensor, t: float | torch.Tensor, ref: ReferenceBank, w: torch.Tensor) -> torch.Tensor:
    a = at(t).to(y).clamp_min(1e-30)
    atom = ref.s0.unsqueeze(0) / a
    return torch.einsum("bn,bnd->bd", w, atom.expand(y.shape[0], -1, -1))


def ce_gate_apply(
    s_twd: torch.Tensor,
    s_tsi: torch.Tensor,
    Pbar: torch.Tensor,
    t: float | torch.Tensor,
    gate_clip: float = 0.0,
) -> torch.Tensor:
    Pbar = sym(torch.nan_to_num(Pbar, nan=0.0, posinf=1e12, neginf=-1e12))
    a2 = at(t).to(Pbar).square()
    g = vt(t).to(Pbar).clamp_min(1e-30)
    evals, evecs = torch.linalg.eigh(Pbar)
    denom = a2 + g * evals
    eps = torch.finfo(Pbar.dtype).eps * 100.0
    denom = torch.where(denom.abs() < eps, torch.sign(denom + eps) * eps, denom)
    gate = a2 / denom
    if gate_clip and gate_clip > 0:
        gate = gate.clamp(min=-float(gate_clip), max=float(gate_clip))
    delta = s_tsi - s_twd
    delta_e = torch.einsum("bji,bj->bi", evecs, delta)
    gated = torch.einsum("bij,bj->bi", evecs, gate * delta_e)
    return s_twd + gated


def ce_gate_matrix(Pbar: torch.Tensor, t: float | torch.Tensor, gate_clip: float = 0.0) -> torch.Tensor:
    Pbar = sym(torch.nan_to_num(Pbar, nan=0.0, posinf=1e12, neginf=-1e12))
    a2 = at(t).to(Pbar).square()
    g = vt(t).to(Pbar).clamp_min(1e-30)
    evals, evecs = torch.linalg.eigh(Pbar)
    denom = a2 + g * evals
    eps = torch.finfo(Pbar.dtype).eps * 100.0
    denom = torch.where(denom.abs() < eps, torch.sign(denom + eps) * eps, denom)
    vals = a2 / denom
    if gate_clip and gate_clip > 0:
        vals = vals.clamp(min=-float(gate_clip), max=float(gate_clip))
    return sym(torch.einsum("bij,bj,bkj->bik", evecs, vals, evecs))



def primal_matrix_blend_from_weights(
    y: torch.Tensor,
    t: float | torch.Tensor,
    ref: ReferenceBank,
    w: torch.Tensor,
    *,
    ridge: float = 1e-10,
    gate_clip: float = 0.0,
) -> torch.Tensor:
    """Primal moment-normal-equation matrix blend score estimator.

    This is the direct finite-reference implementation of the local operator
    control-variate optimum.  It estimates
        G = -Cov(b,d) Cov(d,d)^{-1},    d = c - b,
    using the same OU/SNIS posterior weights as the scalar blend, then returns
        E[b|y] + G (E[c|y] - E[b|y]).

    LFGI uses the Hessian-resolvent identity to estimate the same population
    gate more stably; this method is the moment-based comparator.
    """
    a = at(t).to(y).clamp_min(1e-30)
    g = vt(t).to(y).clamp_min(1e-30)
    d_dim = int(ref.x.shape[1])
    eye = torch.eye(d_dim, dtype=y.dtype, device=y.device)

    b_atom = -(y.unsqueeze(1) - a * ref.x.unsqueeze(0)) / g
    c_atom = (ref.s0.unsqueeze(0) / a).expand(y.shape[0], -1, -1)
    d_atom = c_atom - b_atom

    b_bar = torch.einsum("bn,bnd->bd", w, b_atom)
    c_bar = torch.einsum("bn,bnd->bd", w, c_atom)
    d_bar = c_bar - b_bar

    bc = b_atom - b_bar.unsqueeze(1)
    dc = d_atom - d_bar.unsqueeze(1)
    C = torch.einsum("bn,bni,bnj->bij", w, bc, dc)
    D = sym(torch.einsum("bn,bni,bnj->bij", w, dc, dc))

    # Add an absolute ridge to the disagreement covariance.  The primal normal
    # equation is intentionally left as the statistically difficult baseline;
    # the ridge only prevents numerical singularities from terminating a sweep.
    A = D + float(ridge) * eye.unsqueeze(0)
    try:
        G = -torch.linalg.solve(A, C.transpose(-1, -2)).transpose(-1, -2)
    except RuntimeError:
        G = -C @ torch.linalg.pinv(A)
    G = torch.nan_to_num(G, nan=0.0, posinf=1e12, neginf=-1e12)

    if gate_clip and gate_clip > 0:
        # Clip by singular values without assuming the noisy moment estimate is
        # symmetric.  This is off by default, matching the unclipped primal gate.
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        S = S.clamp(max=float(gate_clip))
        G = torch.einsum("bij,bj,bjk->bik", U, S, Vh)

    return b_bar + torch.einsum("bij,bj->bi", G, d_bar)


def estimate_score_chunk(
    y: torch.Tensor,
    t: float | torch.Tensor,
    ref: ReferenceBank,
    method: str,
    chunk: int = 256,
    gate_ref: Optional[ReferenceBank] = None,
    gate_clip: float = 0.0,
    primal_ridge: float = 1e-10,
) -> torch.Tensor:
    method = method.lower().replace("_", "-")
    outs: List[torch.Tensor] = []
    if gate_ref is None:
        gate_ref = ref
    for i in range(0, y.shape[0], chunk):
        yb = y[i : i + chunk]
        w = snis_weights(yb, t, ref.x)
        s_twd = tweedie_from_weights(yb, t, ref, w)
        if method == "tweedie":
            outs.append(s_twd)
            continue
        s_tsi = tsi_from_weights(yb, t, ref, w)
        if method in {"tsi", "target-score", "target-score-identity"}:
            outs.append(s_tsi)
            continue
        if method == "blend":
            a = at(t).to(yb).clamp_min(1e-30)
            g = vt(t).to(yb).clamp_min(1e-30)
            tsi_atom = ref.s0.unsqueeze(0) / a
            twd_atom = -(yb.unsqueeze(1) - a * ref.x.unsqueeze(0)) / g
            am = torch.einsum("bn,bnd->bd", w, tsi_atom.expand(yb.shape[0], -1, -1))
            bm = torch.einsum("bn,bnd->bd", w, twd_atom)
            ac = tsi_atom - am.unsqueeze(1)
            bc = twd_atom - bm.unsqueeze(1)
            va = torch.einsum("bn,bnd->bd", w, ac.square()).clamp_min(1e-30)
            vb = torch.einsum("bn,bnd->bd", w, bc.square()).clamp_min(1e-30)
            cab = torch.einsum("bn,bnd->bd", w, ac * bc)
            den = (va + vb - 2.0 * cab).clamp_min(1e-20)
            lam_twd = ((va - cab) / den).clamp(0.0, 1.0)
            outs.append((1.0 - lam_twd) * am + lam_twd * bm)
            continue
        if method in {"matrix-blend", "primal-matrix-blend", "moment-matrix-blend", "moment-blend", "primal-moment"}:
            outs.append(
                primal_matrix_blend_from_weights(
                    yb,
                    t,
                    ref,
                    w,
                    ridge=primal_ridge,
                    gate_clip=gate_clip,
                )
            )
            continue
        if method in {"lfgi", "ce-hlsi"}:
            if gate_ref is ref:
                wg = w
            else:
                wg = snis_weights(yb, t, gate_ref.x)
            Pbar = torch.einsum("bn,nij->bij", wg, gate_ref.H)
            outs.append(ce_gate_apply(s_twd, s_tsi, Pbar, t, gate_clip=gate_clip))
            continue
        raise ValueError(f"Unknown method {method!r}")
    return torch.cat(outs, dim=0)


# -----------------------------------------------------------------------------
# Reverse sampler and metrics
# -----------------------------------------------------------------------------


@torch.no_grad()
def heun_reverse_sde(
    score_fn,
    n: int,
    d: int,
    n_steps: int,
    t_max: float,
    t_min: float,
) -> Tuple[torch.Tensor, bool, float]:
    ts = torch.linspace(t_max, t_min, n_steps + 1, dtype=torch.get_default_dtype(), device=current_device())
    y = torch.randn(n, d, dtype=torch.get_default_dtype(), device=current_device())
    max_score = 0.0
    failed = False
    for i in range(n_steps):
        tc, tn = ts[i], ts[i + 1]
        h = tc - tn
        s1 = score_fn(y, tc)
        if not torch.isfinite(s1).all():
            failed = True
            break
        max_score = max(max_score, float(s1.abs().max().detach().cpu()))
        d1 = y + 2.0 * s1
        noise = torch.sqrt(2.0 * h) * torch.randn_like(y)
        yh = y + h * d1 + noise
        s2 = score_fn(yh, tn)
        if not torch.isfinite(s2).all():
            failed = True
            break
        d2 = yh + 2.0 * s2
        y = y + 0.5 * h * (d1 + d2) + noise
        if not torch.isfinite(y).all():
            failed = True
            break
    if not failed:
        tf = torch.tensor(t_min, dtype=torch.get_default_dtype(), device=current_device())
        sf = score_fn(y, tf)
        if torch.isfinite(sf).all():
            max_score = max(max_score, float(sf.abs().max().detach().cpu()))
            y = (y + vt(tf).to(y) * sf) / at(tf).to(y).clamp_min(1e-30)
        else:
            failed = True
    return y, failed, max_score


def mmd_rbf_multi(X: torch.Tensor, Y: torch.Tensor, max_n: int = 1000) -> float:
    n = min(int(X.shape[0]), max_n)
    m = min(int(Y.shape[0]), max_n)
    X = X[:n]
    Y = Y[:m]
    if n < 2 or m < 2:
        return float("nan")
    dxy = torch.cdist(X, Y).detach()
    med = torch.median(dxy[dxy > 0]).item() if (dxy > 0).any() else 1.0
    if not np.isfinite(med) or med <= 1e-12:
        med = 1.0
    bws = [0.5 * med, med, 2.0 * med, 4.0 * med]
    xx = torch.cdist(X, X).square()
    yy = torch.cdist(Y, Y).square()
    xy = dxy.square()
    vals = []
    for bw in bws:
        gamma = 0.5 / (bw ** 2)
        vals.append(torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2.0 * torch.exp(-gamma * xy).mean())
    return float(torch.stack(vals).mean().clamp_min(0.0).detach().cpu())




def sliced_w2(
    X: torch.Tensor,
    Y: torch.Tensor,
    max_n: int = 1000,
    n_projections: int = 128,
) -> float:
    """Approximate sliced W2 between samples by random 1D projections.

    We report the square root of the mean projected squared Wasserstein-2
    distance, so the value is on the same scale as the samples.
    """
    n = min(int(X.shape[0]), int(Y.shape[0]), int(max_n))
    if n < 2:
        return float("nan")

    X = X[:n]
    Y = Y[:n]
    if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
        return float("nan")

    d = X.shape[1]
    n_projections = int(max(1, n_projections))
    theta = torch.randn(n_projections, d, dtype=X.dtype, device=X.device)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-30)

    Xp = X @ theta.T
    Yp = Y @ theta.T
    Xs = torch.sort(Xp, dim=0).values
    Ys = torch.sort(Yp, dim=0).values
    sw2_sq = (Xs - Ys).square().mean()
    return float(torch.sqrt(sw2_sq.clamp_min(0.0)).detach().cpu())


def target_nll(samples: torch.Tensor, target: AnalyticTarget, max_n: int = 1000) -> float:
    """Average negative log likelihood of generated samples under exact target."""
    n = min(int(samples.shape[0]), int(max_n))
    if n < 1:
        return float("nan")
    X = samples[:n]
    if not torch.isfinite(X).all():
        return float("nan")
    lp = target.log_prob(X)
    if not torch.isfinite(lp).all():
        return float("nan")
    return float((-lp).mean().detach().cpu())


def ksd_rbf(samples: torch.Tensor, score_fn, max_n: int = 1000, bandwidth: str | float = "median") -> float:
    n = min(int(samples.shape[0]), max_n)
    X = samples[:n]
    if n < 5:
        return float("nan")
    S = score_fn(X)
    if not torch.isfinite(S).all():
        return float("nan")
    d = X.shape[1]
    dmat = torch.cdist(X, X)
    med = torch.median(dmat[dmat > 0]).item() if (dmat > 0).any() else 1.0
    h = med if np.isfinite(med) and med > 1e-12 else 1.0
    if bandwidth != "median":
        h = float(bandwidth)
    d2 = dmat.square()
    K = torch.exp(-d2 / (2.0 * h ** 2))
    diff = X.unsqueeze(1) - X.unsqueeze(0)
    term1 = K * (S @ S.T)
    sdiff = S.unsqueeze(1) - S.unsqueeze(0)
    term2 = -(K.unsqueeze(-1) * sdiff * diff).sum(-1) / (h ** 2)
    term4 = K * (d / (h ** 2) - d2 / (h ** 4))
    ksd2 = (term1 + term2 + term4).mean()
    return float(torch.sqrt(ksd2.clamp_min(0.0)).detach().cpu())


@dataclass
class ScoreBenchmark:
    t_grid: List[float]
    ys: List[torch.Tensor]
    true_scores: List[torch.Tensor]


def make_score_benchmark(target: AnalyticTarget, n_score: int, t_grid: Sequence[float]) -> ScoreBenchmark:
    x0 = target.sample(n_score)
    ys, true_scores = [], []
    for t in t_grid:
        a = at(float(t)).to(x0)
        g = vt(float(t)).to(x0)
        y = a * x0 + torch.sqrt(g) * torch.randn_like(x0)
        ys.append(y)
        true_scores.append(target.score_t(y, float(t)))
    return ScoreBenchmark(t_grid=[float(t) for t in t_grid], ys=ys, true_scores=true_scores)


def time_avg_score_rmse(
    bench: ScoreBenchmark,
    ref: ReferenceBank,
    method: str,
    chunk: int,
    gate_clip: float,
    primal_ridge: float = 1e-10,
) -> float:
    sse = 0.0
    count = 0
    for t, y, s_true in zip(bench.t_grid, bench.ys, bench.true_scores):
        s_hat = estimate_score_chunk(
            y,
            t,
            ref,
            method,
            chunk=chunk,
            gate_clip=gate_clip,
            primal_ridge=primal_ridge,
        )
        err2 = (s_hat - s_true).square()
        if not torch.isfinite(err2).all():
            return float("nan")
        sse += float(err2.sum().detach().cpu())
        count += err2.numel()
    return math.sqrt(max(sse, 0.0) / max(count, 1))


# -----------------------------------------------------------------------------
# Risk / metric sweeps
# -----------------------------------------------------------------------------


METHODS = ["tweedie", "blend", "matrix-blend", "lfgi"]
DISPLAY = {m: METHOD_STYLES[m]["label"] for m in METHODS}


def method_aliases(method: str) -> List[str]:
    """Canonical method key plus backward-compatible raw-log aliases."""
    key = method.lower().replace("_", "-")
    if key == "lfgi":
        return ["lfgi", "ce-hlsi"]
    if key == "matrix-blend":
        return ["matrix-blend", "primal-matrix-blend", "moment-matrix-blend", "moment-blend", "primal-moment"]
    return [key]


def run_metric_sweeps(args) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [make_target(t) for t in args.targets]

    for target in targets:
        print(f"\n[metric sweep] target={target.name}, d={target.d}")
        gt = target.sample(args.n_metric)
        score_bench = make_score_benchmark(target, args.n_score, args.score_t_grid)

        for nref in args.nref_grid:
            for rep in range(args.repeats):
                seed_here = args.seed + 100_000 * rep + 97 * int(nref) + hash(target.name) % 10_000
                torch.manual_seed(seed_here)
                np.random.seed(seed_here % (2**32 - 1))
                ref = make_reference_bank(target, int(nref))

                for method in METHODS:
                    print(f"  N_ref={nref:5d} rep={rep:02d} method={method}")
                    t0 = time.time()
                    def score_fn(y, t, method=method, ref=ref):
                        return estimate_score_chunk(
                            y,
                            t,
                            ref,
                            method,
                            chunk=args.score_chunk,
                            gate_clip=args.gate_clip,
                            primal_ridge=args.primal_ridge,
                        )

                    samples, failed, max_score = heun_reverse_sde(
                        score_fn=score_fn,
                        n=args.n_samples,
                        d=target.d,
                        n_steps=args.n_steps,
                        t_max=args.t_max,
                        t_min=args.t_min,
                    )
                    if failed or not torch.isfinite(samples).all():
                        mmd_val = float("nan")
                        sw2_val = float("nan")
                        nll_val = float("nan")
                        ksd_val = float("nan")
                    else:
                        metric_idx = randperm_prefix(samples.shape[0], min(args.n_metric, samples.shape[0]))
                        gt_idx = randperm_prefix(gt.shape[0], min(args.n_metric, gt.shape[0]))
                        sample_eval = samples[metric_idx]
                        gt_eval = gt[gt_idx]
                        mmd_val = mmd_rbf_multi(sample_eval, gt_eval, max_n=args.n_metric)
                        sw2_val = sliced_w2(
                            sample_eval,
                            gt_eval,
                            max_n=args.n_metric,
                            n_projections=args.sw2_projections,
                        )
                        nll_val = target_nll(sample_eval, target, max_n=args.n_metric)
                        ksd_val = ksd_rbf(sample_eval, target.score, max_n=args.n_metric)
                    score_rmse = time_avg_score_rmse(
                        score_bench,
                        ref,
                        method,
                        chunk=args.score_chunk,
                        gate_clip=args.gate_clip,
                        primal_ridge=args.primal_ridge,
                    )
                    rows.append({
                        "experiment": "metric_sweep",
                        "target": target.name,
                        "d": target.d,
                        "n_ref": int(nref),
                        "repeat": int(rep),
                        "method": method,
                        "mmd": mmd_val,
                        "sw2": sw2_val,
                        "nll": nll_val,
                        "ksd": ksd_val,
                        "score_rmse": score_rmse,
                        "failed": bool(failed),
                        "max_score_abs": max_score,
                        "seconds": time.time() - t0,
                    })
                    flush_csv(out_dir / "metric_sweep_raw.csv", rows)
        plot_metric_sweep(rows, target.name, out_dir)
    return rows


def flush_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def aggregate_metric_rows(rows: List[Dict[str, object]], target_name: str) -> List[Dict[str, object]]:
    out = []
    metrics = ["mmd", "sw2", "nll", "ksd", "score_rmse"]
    for nref in sorted({int(r["n_ref"]) for r in rows if r.get("target") == target_name}):
        for method in METHODS:
            aliases = set(method_aliases(method))
            sub = [r for r in rows if r.get("target") == target_name and int(r["n_ref"]) == nref and str(r.get("method", "")).lower().replace("_", "-") in aliases]
            if not sub:
                continue
            row = {"target": target_name, "n_ref": nref, "method": method}
            for metric in metrics:
                mean, std = finite_mean_std([float(r.get(metric, float("nan"))) for r in sub])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            out.append(row)
    return out


def plot_metric_sweep(all_rows: List[Dict[str, object]], target_name: str, out_dir: Path) -> None:
    agg = aggregate_metric_rows(all_rows, target_name)
    if not agg:
        return
    flush_csv(out_dir / f"metric_sweep_summary_{target_name}.csv", agg)

    metrics = ["mmd", "sw2", "nll", "ksd", "score_rmse"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=PUB_FIGSIZE)
        positive: List[float] = []

        for method in METHODS:
            # Accept backward-compatible raw-log aliases.
            method_keys = set(method_aliases(method))
            sub = [r for r in agg if str(r["method"]).lower().replace("_", "-") in method_keys]
            if not sub:
                continue

            xs = np.asarray([r["n_ref"] for r in sub], dtype=float)
            ys = np.asarray([r[f"{metric}_mean"] for r in sub], dtype=float)
            es = np.asarray([r[f"{metric}_std"] for r in sub], dtype=float)
            es = np.where(np.isfinite(es), es, 0.0)
            mask = np.isfinite(xs) & np.isfinite(ys)
            xs, ys, es = xs[mask], ys[mask], es[mask]
            if xs.size == 0:
                continue
            order = np.argsort(xs)
            xs, ys, es = xs[order], ys[order], es[order]
            positive.extend([float(v) for v in ys if np.isfinite(v) and v > 0])

            st = METHOD_STYLES[method]
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker=st["marker"],
                linestyle=st["linestyle"],
                linewidth=2.35 if method != "lfgi" else 2.8,
                markersize=6.4 if method != "lfgi" else 6.8,
                capsize=3.5,
                elinewidth=1.15,
                capthick=1.15,
                color=st["color"],
                label=st["label"],
                zorder=3 if method == "lfgi" else 2,
            )

        if metric == "nll":
            # Focus NLL on the meaningful TWEEDIE-vs-LFGI comparison.  SCALAR
            # BLEND and MATRIX BLEND can have huge positive NLL at small
            # reference counts, and using those outliers to choose the y-limits
            # makes the best two methods indistinguishable.  Keep a linear
            # scale so negative continuous-NLL values are shown directly.
            y_scale = "linear"
            symlog_linthresh = 1.0
        else:
            finite_y = np.asarray([
                float(r[f"{metric}_mean"])
                for r in agg
                if np.isfinite(float(r.get(f"{metric}_mean", float("nan"))))
            ], dtype=float)

            if finite_y.size == 0:
                y_scale = "linear"
                symlog_linthresh = 1.0
            else:
                has_pos = bool(np.any(finite_y > 0.0))
                has_neg = bool(np.any(finite_y < 0.0))
                positive_arr = finite_y[finite_y > 0.0]

                if has_pos and not has_neg and positive_arr.size >= 2 and positive_arr.max() / max(positive_arr.min(), 1e-300) > 8:
                    y_scale = "log"
                    symlog_linthresh = 1.0
                else:
                    y_scale = "linear"
                    symlog_linthresh = 1.0

        style_axis(ax, log_x=True, y_scale=y_scale, symlog_linthresh=symlog_linthresh)
        if metric == "nll":
            set_nll_focus_ylim(ax, agg)
        ax.set_xlabel(r"Reference count $N_{\rm ref}$")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_TITLES[metric]}\n{paper_target_title(target_name)}", pad=7)
        ax.legend(frameon=False, loc="best", handlelength=2.8, borderaxespad=0.3)

        stem = f"{target_name}_{metric}_vs_nref"
        save_publication_figure(fig, out_dir, stem)


# -----------------------------------------------------------------------------
# Gate sample-complexity comparison
# -----------------------------------------------------------------------------


def weighted_primal_gate(
    y: torch.Tensor,
    t: float,
    ref: ReferenceBank,
    ridge: float,
    chunk: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate G* = -Cov(b,d) Cov(d,d)^{-1}; return G and D=Cov(d,d)."""
    outs_G: List[torch.Tensor] = []
    outs_D: List[torch.Tensor] = []
    a = at(t).to(y).clamp_min(1e-30)
    g = vt(t).to(y).clamp_min(1e-30)
    eye = torch.eye(ref.x.shape[1], dtype=y.dtype, device=y.device)
    for i in range(0, y.shape[0], chunk):
        yb = y[i : i + chunk]
        w = snis_weights(yb, t, ref.x)
        b_atom = -(yb.unsqueeze(1) - a * ref.x.unsqueeze(0)) / g
        c_atom = ref.s0.unsqueeze(0) / a
        c_atom = c_atom.expand(yb.shape[0], -1, -1)
        d_atom = c_atom - b_atom
        b_bar = torch.einsum("bn,bnd->bd", w, b_atom)
        d_bar = torch.einsum("bn,bnd->bd", w, d_atom)
        bc = b_atom - b_bar.unsqueeze(1)
        dc = d_atom - d_bar.unsqueeze(1)
        C = torch.einsum("bn,bni,bnj->bij", w, bc, dc)
        D = sym(torch.einsum("bn,bni,bnj->bij", w, dc, dc))
        A = D + float(ridge) * eye.unsqueeze(0)
        try:
            G = -torch.linalg.solve(A, C.transpose(-1, -2)).transpose(-1, -2)
        except RuntimeError:
            G = -C @ torch.linalg.pinv(A)
        outs_G.append(torch.nan_to_num(G, nan=0.0, posinf=1e12, neginf=-1e12))
        outs_D.append(D)
    return torch.cat(outs_G, dim=0), torch.cat(outs_D, dim=0)


def lfgi_gate_from_ref(
    y: torch.Tensor,
    t: float,
    ref: ReferenceBank,
    gate_clip: float,
    chunk: int = 64,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    for i in range(0, y.shape[0], chunk):
        yb = y[i : i + chunk]
        w = snis_weights(yb, t, ref.x)
        Pbar = torch.einsum("bn,nij->bij", w, ref.H)
        outs.append(ce_gate_matrix(Pbar, t, gate_clip=gate_clip))
    return torch.cat(outs, dim=0)


def risk_weighted_gate_error(G_hat: torch.Tensor, G_star: torch.Tensor, D_star: torch.Tensor) -> torch.Tensor:
    E = G_hat - G_star
    # trace(E D E^T) = sum_{i,j,k} E_ij D_jk E_ik
    return torch.einsum("bij,bjk,bik->b", E, D_star, E).clamp_min(0.0)


def fro_relative_gate_error(G_hat: torch.Tensor, G_star: torch.Tensor) -> torch.Tensor:
    num = (G_hat - G_star).square().sum(dim=(-1, -2)).sqrt()
    den = G_star.square().sum(dim=(-1, -2)).sqrt().clamp_min(1e-30)
    return num / den


def run_gate_sweep(args) -> List[Dict[str, object]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = make_target(args.gate_target)
    print(f"\n[gate sweep] target={target.name}, d={target.d}")
    rows: List[Dict[str, object]] = []

    for t in args.gate_t_grid:
        print(f"  oracle construction at t={t:g}")
        torch.manual_seed(args.seed + int(1_000_000 * t) + 777)
        # Queries are actual noisy states from the OU process.
        xq = target.sample(args.gate_n_query)
        yq = at(float(t)).to(xq) * xq + torch.sqrt(vt(float(t)).to(xq)) * torch.randn_like(xq)

        oracle_ref = make_reference_bank(target, args.gate_n_oracle)
        G_star = lfgi_gate_from_ref(yq, float(t), oracle_ref, gate_clip=args.gate_clip, chunk=args.gate_chunk)
        _, D_star = weighted_primal_gate(yq, float(t), oracle_ref, ridge=args.primal_ridge, chunk=args.gate_chunk)
        tweedie_gap = risk_weighted_gate_error(
            torch.zeros_like(G_star), G_star, D_star
        ).mean().clamp_min(1e-30)
        D_trace = torch.diagonal(D_star, dim1=-2, dim2=-1).sum(-1).mean().clamp_min(1e-30)

        for rep in range(args.gate_repeats):
            seed_here = args.seed + 91_000 * rep + int(10_000 * t)
            torch.manual_seed(seed_here)
            max_n = max(args.gate_n_grid)
            ref_max = make_reference_bank(target, max_n)
            for n_gate in args.gate_n_grid:
                ref = ReferenceBank(x=ref_max.x[:n_gate], s0=ref_max.s0[:n_gate], H=ref_max.H[:n_gate])
                print(f"    t={t:g} N_gate={n_gate:5d} rep={rep:02d}")
                t_start = time.time()
                G_lfgi = lfgi_gate_from_ref(yq, float(t), ref, gate_clip=args.gate_clip, chunk=args.gate_chunk)
                G_primal, _ = weighted_primal_gate(yq, float(t), ref, ridge=args.primal_ridge, chunk=args.gate_chunk)
                for label, G_hat in [("lfgi-hessian", G_lfgi), ("primal-moment", G_primal)]:
                    abs_excess = risk_weighted_gate_error(G_hat, G_star, D_star)
                    fro_rel = fro_relative_gate_error(G_hat, G_star)
                    rows.append({
                        "experiment": "gate_sample_complexity",
                        "target": target.name,
                        "d": target.d,
                        "t": float(t),
                        "n_gate": int(n_gate),
                        "repeat": int(rep),
                        "gate_estimator": label,
                        "risk_weighted_excess_mean": float(abs_excess.mean().detach().cpu()),
                        "risk_weighted_excess_median": float(abs_excess.median().detach().cpu()),
                        "risk_weighted_rel_to_tweedie_gap": float((abs_excess.mean() / tweedie_gap).detach().cpu()),
                        "risk_weighted_rel_to_D_trace": float((abs_excess.mean() / D_trace).detach().cpu()),
                        "fro_relative_mean": float(fro_rel.mean().detach().cpu()),
                        "fro_relative_median": float(fro_rel.median().detach().cpu()),
                        "seconds": time.time() - t_start,
                    })
                flush_csv(out_dir / "gate_sample_complexity_raw.csv", rows)
        plot_gate_sweep(rows, target.name, out_dir)
    return rows


def aggregate_gate_rows(rows: List[Dict[str, object]], target_name: str) -> List[Dict[str, object]]:
    out = []
    metrics = [
        "risk_weighted_excess_mean",
        "risk_weighted_rel_to_tweedie_gap",
        "risk_weighted_rel_to_D_trace",
        "fro_relative_mean",
    ]
    for t in sorted({float(r["t"]) for r in rows if r.get("target") == target_name}):
        for n in sorted({int(r["n_gate"]) for r in rows if r.get("target") == target_name and float(r["t"]) == t}):
            for est in ["lfgi-hessian", "primal-moment"]:
                sub = [r for r in rows if r.get("target") == target_name and float(r["t"]) == t and int(r["n_gate"]) == n and r.get("gate_estimator") == est]
                if not sub:
                    continue
                row = {"target": target_name, "t": t, "n_gate": n, "gate_estimator": est}
                for metric in metrics:
                    mean, std = finite_mean_std([float(r.get(metric, float("nan"))) for r in sub])
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_std"] = std
                out.append(row)
    return out


def plot_gate_sweep(rows: List[Dict[str, object]], target_name: str, out_dir: Path) -> None:
    agg = aggregate_gate_rows(rows, target_name)
    if not agg:
        return
    flush_csv(out_dir / f"gate_sample_complexity_summary_{target_name}.csv", agg)

    metric_specs = [
        "risk_weighted_rel_to_tweedie_gap",
        "risk_weighted_rel_to_D_trace",
        "fro_relative_mean",
    ]
    for t in sorted({r["t"] for r in agg}):
        for metric in metric_specs:
            fig, ax = plt.subplots(figsize=PUB_FIGSIZE_WIDE)
            positive: List[float] = []

            for est in ["lfgi-hessian", "primal-moment"]:
                sub = [r for r in agg if r["t"] == t and r["gate_estimator"] == est]
                if not sub:
                    continue
                xs = np.asarray([r["n_gate"] for r in sub], dtype=float)
                ys = np.asarray([r[f"{metric}_mean"] for r in sub], dtype=float)
                es = np.asarray([r[f"{metric}_std"] for r in sub], dtype=float)
                es = np.where(np.isfinite(es), es, 0.0)
                mask = np.isfinite(xs) & np.isfinite(ys)
                xs, ys, es = xs[mask], ys[mask], es[mask]
                if xs.size == 0:
                    continue
                order = np.argsort(xs)
                xs, ys, es = xs[order], ys[order], es[order]
                positive.extend([float(v) for v in ys if np.isfinite(v) and v > 0])

                st = GATE_STYLES[est]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker=st["marker"],
                    linestyle=st["linestyle"],
                    linewidth=2.8 if est == "lfgi-hessian" else 2.2,
                    markersize=6.8 if est == "lfgi-hessian" else 6.4,
                    capsize=3.5,
                    elinewidth=1.15,
                    capthick=1.15,
                    color=st["color"],
                    label=st["label"],
                    zorder=3 if est == "lfgi-hessian" else 2,
                )

            use_log_y = len(positive) >= 2
            style_axis(ax, log_x=True, log_y=use_log_y)
            ax.set_xlabel(r"Gate-bank size $N_g$")
            ax.set_ylabel(GATE_METRIC_LABELS[metric])
            ax.set_title(
                f"{GATE_METRIC_TITLES[metric]}\n{paper_target_title(target_name)}, $t={float(t):g}$",
                pad=7,
            )
            ax.legend(frameon=False, loc="best", handlelength=2.8, borderaxespad=0.3)

            stem = f"{target_name}_gate_{metric}_t{str(t).replace('.', 'p')}_vs_ngate"
            save_publication_figure(fig, out_dir, stem)


# -----------------------------------------------------------------------------
# Hessian-resolvent learnability diagnostics
# -----------------------------------------------------------------------------


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    """Weighted quantile for one-dimensional finite tensors."""
    values = values.detach().flatten()
    weights = weights.detach().flatten()
    mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if not bool(mask.any()):
        return float("nan")
    v = values[mask]
    w = weights[mask]
    order = torch.argsort(v)
    v = v[order]
    w = w[order]
    cw = torch.cumsum(w, dim=0)
    total = cw[-1].clamp_min(1e-30)
    idx = torch.searchsorted(cw, torch.as_tensor(float(q), dtype=cw.dtype, device=cw.device) * total)
    idx = idx.clamp(max=v.numel() - 1)
    return float(v[idx].detach().cpu())


def finite_summary(xs: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "q10": float("nan"), "q90": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def safe_sym_eigvalsh(A: torch.Tensor, *, name: str = "matrix") -> torch.Tensor:
    """eigvalsh for small symmetric matrices with CUDA CPU fallback.

    Some CUDA/cuSOLVER builds are fragile for large batches of tiny fp64
    symmetric eigensolves.  The learnability diagnostics only need exact
    eigenvalues for 8x8 matrices, so falling back to CPU is cheap and much
    safer than failing a full sweep.
    """
    A = sym(A) if A.ndim >= 2 else A
    if not bool(torch.isfinite(A).all()):
        raise FloatingPointError(f"Non-finite entries encountered before eigvalsh({name}).")
    try:
        return torch.linalg.eigvalsh(A)
    except RuntimeError as err:
        if A.is_cuda:
            try:
                return torch.linalg.eigvalsh(A.detach().cpu()).to(device=A.device, dtype=A.dtype)
            except Exception as cpu_err:  # pragma: no cover - diagnostic message path
                raise RuntimeError(f"eigvalsh({name}) failed on CUDA and CPU fallback also failed") from cpu_err
        raise err


def safe_batched_sym_op_norm(
    A: torch.Tensor,
    *,
    batch_chunk: int = 1024,
    name: str = "matrix_batch",
) -> torch.Tensor:
    """Return max absolute eigenvalue for each small symmetric matrix in a batch.

    This avoids one enormous batched CUDA eigensolve, which can trigger
    CUSOLVER_STATUS_INVALID_VALUE on Colab/A100/T4 builds even when the inputs
    are finite.
    """
    if A.ndim != 3:
        raise ValueError(f"safe_batched_sym_op_norm expects shape (n,d,d), got {tuple(A.shape)}")
    if A.shape[0] == 0:
        return torch.empty(0, dtype=A.dtype, device=A.device)
    A = sym(A)
    out: List[torch.Tensor] = []
    step = max(1, int(batch_chunk))
    for k in range(0, A.shape[0], step):
        Ak = A[k : k + step]
        evals = safe_sym_eigvalsh(Ak, name=f"{name}[{k}:{k+Ak.shape[0]}]")
        out.append(evals.abs().amax(dim=-1))
    return torch.cat(out, dim=0)


def compute_learnability_diagnostics(
    target: AnalyticTarget,
    y: torch.Tensor,
    t: float,
    ref: ReferenceBank,
    *,
    gate_clip: float,
    chunk: int,
    eig_floor: float,
    z_eig_chunk: int = 1024,
) -> List[Dict[str, object]]:
    """Compute the theory-facing LFGI learnability quantities.

    For each noisy query y, this estimates
        v_A^2 = || E[Z^2 | y] ||_op,
        Z = gamma A^{-1/2}(H_0 - Hbar)A^{-1/2},
    the sharpened pole factor alpha^4 Lambda_B using the Fisher identity
        Tr(A^{-1}M) = d/(gamma alpha^2),
    the oracle gain trace(G_* D G_*^T), and the derived sufficient-complexity
    proxy alpha^4 Lambda_B v_A^2 / G_* (up to constants and log factors).

    If A is not numerically positive definite, the resolvent-theory quantities
    depending on A^{-1/2} are recorded as NaN and the invalid query contributes
    only to the positive-definite fraction.
    """
    rows: List[Dict[str, object]] = []
    d = int(target.d)
    a = at(float(t)).to(y).clamp_min(1e-30)
    a2 = a.square()
    alpha4 = a2.square()
    g = vt(float(t)).to(y).clamp_min(1e-30)
    eye = torch.eye(d, dtype=y.dtype, device=y.device)
    eig_floor_t = torch.as_tensor(float(eig_floor), dtype=y.dtype, device=y.device)

    for i in range(0, y.shape[0], chunk):
        yb = y[i : i + chunk]
        w = snis_weights(yb, float(t), ref.x)
        ess = 1.0 / w.square().sum(dim=1).clamp_min(1e-30)

        Hbar = sym(torch.einsum("bn,nij->bij", w, ref.H))
        Abar = sym(a2 * eye.unsqueeze(0) + g * Hbar)
        G_star = ce_gate_matrix(Hbar, float(t), gate_clip=gate_clip)

        # Disagreement covariance D = Cov(d,d) under the same high-reference OU posterior.
        b_atom = -(yb.unsqueeze(1) - a * ref.x.unsqueeze(0)) / g
        c_atom = (ref.s0.unsqueeze(0) / a).expand(yb.shape[0], -1, -1)
        d_atom = c_atom - b_atom
        d_bar = torch.einsum("bn,bnd->bd", w, d_atom)
        dc = d_atom - d_bar.unsqueeze(1)
        D = sym(torch.einsum("bn,bni,bnj->bij", w, dc, dc))

        oracle_gain = risk_weighted_gate_error(torch.zeros_like(G_star), G_star, D)
        D_trace = torch.diagonal(D, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
        hbar_eigs = safe_sym_eigvalsh(Hbar, name="Hbar")
        abar_eigs = safe_sym_eigvalsh(Abar, name="Abar")

        for j in range(yb.shape[0]):
            global_idx = i + j
            min_h = float(hbar_eigs[j].min().detach().cpu())
            max_h = float(hbar_eigs[j].max().detach().cpu())
            min_a = float(abar_eigs[j].min().detach().cpu())
            max_a = float(abar_eigs[j].max().detach().cpu())
            is_pd = bool(min_a > float(eig_floor))

            vA2 = float("nan")
            RA_mean = float("nan")
            RA_q90 = float("nan")
            RA_q99 = float("nan")
            RA_max = float("nan")
            alpha4_lambdaB_identity = float("nan")
            alpha4_lambdaB_empirical_D = float("nan")
            alpha4_lambdaB_over_psd_bound = float("nan")
            sufficient_complexity_proxy = float("nan")
            cond_A = float("nan")

            if is_pd:
                try:
                    evals, evecs = torch.linalg.eigh(Abar[j])
                except RuntimeError:
                    if Abar[j].is_cuda:
                        evals_cpu, evecs_cpu = torch.linalg.eigh(Abar[j].detach().cpu())
                        evals = evals_cpu.to(device=Abar[j].device, dtype=Abar[j].dtype)
                        evecs = evecs_cpu.to(device=Abar[j].device, dtype=Abar[j].dtype)
                    else:
                        raise
                inv_evals = 1.0 / evals.clamp_min(eig_floor_t)
                inv_sqrt_evals = torch.sqrt(inv_evals)
                Ainv = sym(torch.einsum("ik,k,jk->ij", evecs, inv_evals, evecs))
                Ainv_half = sym(torch.einsum("ik,k,jk->ij", evecs, inv_sqrt_evals, evecs))

                Hdiff = ref.H - Hbar[j].unsqueeze(0)
                Z = g * torch.einsum("ij,njk,kl->nil", Ainv_half, Hdiff, Ainv_half)
                Z = sym(Z)
                if not bool(torch.isfinite(Z).all()):
                    # Treat this query as numerically invalid rather than crashing
                    # the sweep.  This should be rare; it is recorded through NaNs
                    # in the row-level diagnostics.
                    is_pd = False
                else:
                    Z_op = safe_batched_sym_op_norm(
                        Z,
                        batch_chunk=int(z_eig_chunk),
                        name=f"Z_t{float(t):.4g}_query{global_idx}",
                    )
                    Vmat = sym(torch.einsum("n,nij,njk->ik", w[j], Z, Z))
                    vA2 = float(safe_sym_eigvalsh(Vmat, name="Vmat").amax().clamp_min(0.0).detach().cpu())
                    RA_mean = float((w[j] * Z_op).sum().detach().cpu())
                    RA_q90 = weighted_quantile(Z_op, w[j], 0.90)
                    RA_q99 = weighted_quantile(Z_op, w[j], 0.99)
                    RA_max = float(Z_op.max().detach().cpu())

                    Ainv_op = float(inv_evals.max().detach().cpu())
                    tr_Ainv_M_identity = float((d / (float(g.detach().cpu()) * float(a2.detach().cpu()))))
                    alpha4_lambdaB_identity = float(float(alpha4.detach().cpu()) * Ainv_op * tr_Ainv_M_identity)
                    tr_Ainv_D = float(torch.einsum("ij,ji->", Ainv, D[j]).detach().cpu())
                    alpha4_lambdaB_empirical_D = float(float(alpha4.detach().cpu()) * Ainv_op * tr_Ainv_D)
                    psd_bound = float(d / float(g.detach().cpu()))
                    alpha4_lambdaB_over_psd_bound = alpha4_lambdaB_identity / max(psd_bound, 1e-300)
                    og = float(oracle_gain[j].detach().cpu())
                    if np.isfinite(vA2) and np.isfinite(alpha4_lambdaB_identity) and og > 1e-300:
                        sufficient_complexity_proxy = float((alpha4_lambdaB_identity * vA2) / og)
                    cond_A = max_a / max(min_a, 1e-300)

            rows.append({
                "experiment": "lfgi_learnability_diagnostics",
                "target": target.name,
                "d": d,
                "t": float(t),
                "query_index": int(global_idx),
                "alpha2": float(a2.detach().cpu()),
                "gamma": float(g.detach().cpu()),
                "posterior_ess": float(ess[j].detach().cpu()),
                "hbar_min_eig": min_h,
                "hbar_max_eig": max_h,
                "abar_min_eig": min_a,
                "abar_max_eig": max_a,
                "abar_condition": cond_A,
                "abar_pd": int(is_pd),
                "vA2": vA2,
                "relative_hessian_R_mean": RA_mean,
                "relative_hessian_R_q90": RA_q90,
                "relative_hessian_R_q99": RA_q99,
                "relative_hessian_R_max": RA_max,
                "alpha4_lambdaB_identity": alpha4_lambdaB_identity,
                "alpha4_lambdaB_empirical_D": alpha4_lambdaB_empirical_D,
                "alpha4_lambdaB_over_psd_bound": alpha4_lambdaB_over_psd_bound,
                "sufficient_complexity_proxy": sufficient_complexity_proxy,
                "psd_cancellation_bound_d_over_gamma": float(d / float(g.detach().cpu())),
                "oracle_gain": float(oracle_gain[j].detach().cpu()),
                "oracle_gain_rel_to_D_trace": float((oracle_gain[j] / D_trace[j]).detach().cpu()),
                "D_trace": float(D_trace[j].detach().cpu()),
            })
    return rows


def aggregate_learnability_rows(rows: List[Dict[str, object]], target_name: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    metrics = [
        "posterior_ess",
        "hbar_min_eig",
        "abar_min_eig",
        "abar_condition",
        "vA2",
        "relative_hessian_R_mean",
        "relative_hessian_R_q90",
        "sufficient_complexity_proxy",
        "alpha4_lambdaB_identity",
        "alpha4_lambdaB_empirical_D",
        "alpha4_lambdaB_over_psd_bound",
        "psd_cancellation_bound_d_over_gamma",
        "oracle_gain",
        "oracle_gain_rel_to_D_trace",
        "D_trace",
    ]
    for t in sorted({float(r["t"]) for r in rows if r.get("target") == target_name}):
        sub = [r for r in rows if r.get("target") == target_name and float(r["t"]) == t]
        if not sub:
            continue
        row: Dict[str, object] = {"target": target_name, "t": t, "n_query": len(sub)}
        row["abar_pd_fraction"] = float(np.mean([float(r.get("abar_pd", 0.0)) for r in sub]))
        for metric in metrics:
            stats = finite_summary([float(r.get(metric, float("nan"))) for r in sub])
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        out.append(row)
    return out


def _plot_summary_series(
    ax: plt.Axes,
    summary: List[Dict[str, object]],
    metric: str,
    *,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
    use_band: bool = True,
) -> None:
    xs = np.asarray([float(r["t"]) for r in summary], dtype=float)
    ys = np.asarray([float(r.get(f"{metric}_median", float("nan"))) for r in summary], dtype=float)
    lo = np.asarray([float(r.get(f"{metric}_q10", float("nan"))) for r in summary], dtype=float)
    hi = np.asarray([float(r.get(f"{metric}_q90", float("nan"))) for r in summary], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not mask.any():
        return
    xs, ys, lo, hi = xs[mask], ys[mask], lo[mask], hi[mask]
    order = np.argsort(xs)
    xs, ys, lo, hi = xs[order], ys[order], lo[order], hi[order]
    ax.plot(xs, ys, marker=marker, linestyle=linestyle, linewidth=2.45, markersize=6.5, label=label)
    if use_band and np.isfinite(lo).any() and np.isfinite(hi).any():
        lo = np.where(np.isfinite(lo), lo, ys)
        hi = np.where(np.isfinite(hi), hi, ys)
        ax.fill_between(xs, lo, hi, alpha=0.16, linewidth=0.0)


def plot_learnability_diagnostics(
    rows: List[Dict[str, object]],
    target_name: str,
    out_dir: Path,
    *,
    plot_t_max: Optional[float] = None,
) -> List[Dict[str, object]]:
    summary = aggregate_learnability_rows(rows, target_name)
    if not summary:
        return []
    flush_csv(out_dir / f"learnability_diagnostics_summary_{target_name}.csv", summary)

    plot_summary = summary
    if plot_t_max is not None:
        plot_summary = [r for r in summary if float(r["t"]) <= float(plot_t_max) + 1e-15]
        if not plot_summary:
            plot_summary = summary

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.85))

    ax = axes[0]
    _plot_summary_series(
        ax,
        plot_summary,
        "sufficient_complexity_proxy",
        label=r"$\alpha_t^4\Lambda_B\,v_A^2/\mathcal G_\star$",
        marker="o",
    )
    style_axis(ax, log_x=False, y_scale="log")
    ax.set_xlabel(r"Diffusion time $t$")
    ax.set_ylabel(r"Required-reference proxy")
    ax.set_title(r"Derived complexity proxy", pad=7)

    ax = axes[1]
    _plot_summary_series(ax, plot_summary, "alpha4_lambdaB_identity", label=r"$\alpha_t^4\Lambda_B$", marker="D")
    xs = np.asarray([float(r["t"]) for r in plot_summary], dtype=float)
    bound = np.asarray([float(r.get("psd_cancellation_bound_d_over_gamma_median", float("nan"))) for r in plot_summary], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(bound)
    if mask.any():
        order = np.argsort(xs[mask])
        ax.plot(xs[mask][order], bound[mask][order], linestyle="--", linewidth=2.0, label=r"PSD bound $d/\gamma_t$")
    style_axis(ax, log_x=False, y_scale="log")
    ax.set_xlabel(r"Diffusion time $t$")
    ax.set_ylabel(r"Sharpened pole factor")
    ax.set_title(r"Resolvent cancellation scale", pad=7)
    ax.legend(frameon=False, loc="best", handlelength=2.4, borderaxespad=0.3)

    ax = axes[2]
    _plot_summary_series(ax, plot_summary, "oracle_gain_rel_to_D_trace", label=r"$\mathcal G_\star/\mathrm{tr}(D)$", marker="s")
    style_axis(ax, log_x=False, y_scale="linear")
    ax.set_xlabel(r"Diffusion time $t$")
    ax.set_ylabel(r"Normalized oracle gain")
    ax.set_title(r"Gain available to capture", pad=7)
    ax.set_ylim(bottom=0.0)

    for ax in axes:
        pd_vals = [float(r.get("abar_pd_fraction", float("nan"))) for r in plot_summary]
        if np.isfinite(pd_vals).all() and min(pd_vals) < 1.0:
            ax.text(0.02, 0.04, "excludes non-PD A queries", transform=ax.transAxes, fontsize=8.5, color="0.35")
        if plot_t_max is not None:
            ax.text(0.98, 0.04, rf"plot window: $t\leq {plot_t_max:g}$", transform=ax.transAxes, ha="right", fontsize=8.5, color="0.35")

    fig.suptitle(f"LFGI learnability diagnostics: {paper_target_title(target_name)}", y=1.02, fontsize=16)
    fig.tight_layout(pad=0.45, w_pad=1.15)
    stem = f"{target_name}_lfgi_learnability_diagnostics"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=PUB_DPI, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_dir / f"{stem}.png", dpi=PUB_DPI, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return summary

def run_learnability_diagnostics(args) -> List[Dict[str, object]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_name = args.learnability_target or args.gate_target
    target = make_target(target_name)
    t_grid = args.learnability_t_grid if args.learnability_t_grid is not None else args.gate_t_grid
    n_query = int(args.learnability_n_query if args.learnability_n_query is not None else args.gate_n_query)
    n_oracle = int(args.learnability_n_oracle if args.learnability_n_oracle is not None else args.gate_n_oracle)
    chunk = int(args.learnability_chunk if args.learnability_chunk is not None else min(args.gate_chunk, 8))

    print(f"\n[learnability diagnostics] target={target.name}, d={target.d}, n_query={n_query}, n_oracle={n_oracle}")
    rows: List[Dict[str, object]] = []
    for t in t_grid:
        print(f"  diagnostics at t={float(t):g}")
        torch.manual_seed(args.seed + int(1_000_000 * float(t)) + 444_111)
        xq = target.sample(n_query)
        yq = at(float(t)).to(xq) * xq + torch.sqrt(vt(float(t)).to(xq)) * torch.randn_like(xq)
        oracle_ref = make_reference_bank(target, n_oracle)
        t_rows = compute_learnability_diagnostics(
            target,
            yq,
            float(t),
            oracle_ref,
            gate_clip=args.gate_clip,
            chunk=chunk,
            eig_floor=args.learnability_eig_floor,
            z_eig_chunk=args.learnability_z_eig_chunk,
        )
        rows.extend(t_rows)
        flush_csv(out_dir / "learnability_diagnostics_raw.csv", rows)
    plot_learnability_diagnostics(rows, target.name, out_dir, plot_t_max=args.learnability_plot_max_t)
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LFGI claim-validation sweeps with publication-ready plotting")
    p.add_argument("--out-dir", type=str, default="lfgi_claim_validation_out")
    p.add_argument("--device", type=str, default=os.environ.get("DW4_DEVICE", "auto"))
    p.add_argument("--dtype", type=str, default=os.environ.get("DW4_DTYPE", "float64"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--targets", nargs="+", default=["misaligned_subspace_gmm_d8"])
    p.add_argument("--methods", nargs="+", default=METHODS, help="Currently informational; script plots TWEEDIE/SCALAR BLEND/MATRIX BLEND/LFGI.")
    p.add_argument("--nref-grid", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=1024, help="Reverse-SDE samples per method/repeat/N_ref")
    p.add_argument("--n-metric", type=int, default=1024, help="Samples used for MMD/SW2/NLL/KSD")
    p.add_argument("--n-score", type=int, default=1024, help="Noisy score RMSE query count per t")
    p.add_argument("--sw2-projections", type=int, default=128, help="Random projections used for sliced W2.")
    p.add_argument("--n-steps", type=int, default=96)
    p.add_argument("--t-max", type=float, default=3.0)
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--score-t-grid", nargs="+", type=float, default=[0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28])
    p.add_argument("--score-chunk", type=int, default=256)
    p.add_argument("--gate-clip", type=float, default=0.0, help="Optional LFGI gate eigenvalue clip. 0 means raw/unclipped.")

    p.add_argument("--skip-metric-sweep", action="store_true")
    p.add_argument("--skip-gate-sweep", action="store_true")

    p.add_argument("--gate-target", type=str, default="misaligned_subspace_gmm_d8")
    p.add_argument("--gate-n-grid", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--gate-repeats", type=int, default=5)
    p.add_argument("--gate-n-query", type=int, default=64)
    p.add_argument("--gate-n-oracle", type=int, default=32768)
    p.add_argument("--gate-t-grid", nargs="+", type=float, default=[0.04, 0.08, 0.16])
    p.add_argument("--gate-chunk", type=int, default=32)
    p.add_argument("--primal-ridge", type=float, default=1e-10)

    p.add_argument("--skip-learnability-diagnostics", action="store_true")
    p.add_argument("--learnability-target", type=str, default=None, help="Target for v_A/Lambda_B/oracle-gain diagnostics. Defaults to --gate-target.")
    p.add_argument("--learnability-t-grid", nargs="+", type=float, default=None, help="Diffusion times for learnability diagnostics. Defaults to --gate-t-grid.")
    p.add_argument("--learnability-n-query", type=int, default=None, help="Noisy query count for diagnostics. Defaults to --gate-n-query.")
    p.add_argument("--learnability-n-oracle", type=int, default=None, help="High-reference oracle bank for diagnostics. Defaults to --gate-n-oracle.")
    p.add_argument("--learnability-chunk", type=int, default=None, help="Query chunk size for diagnostics. Defaults to min(--gate-chunk, 8).")
    p.add_argument("--learnability-eig-floor", type=float, default=1e-10, help="Positive-definite floor for A^{-1/2} diagnostics.")
    p.add_argument("--learnability-z-eig-chunk", type=int, default=1024, help="Chunk size for tiny batched eigvalsh calls used to compute ||Z_i||_op. Lower this if CUDA eigensolvers are fragile.")
    p.add_argument("--learnability-plot-max-t", type=float, default=None, help="Optional maximum diffusion time shown in the learnability figure; CSV outputs still include all computed times.")
    return p


def save_run_config(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["torch_dtype"] = str(torch.get_default_dtype())
    cfg["device"] = str(current_device())
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)


def main() -> None:
    args = build_argparser().parse_args()
    set_runtime(args.device, args.dtype, args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(args)
    print(f"device={current_device()} dtype={torch.get_default_dtype()} out={args.out_dir}")
    print(f"targets={args.targets}")

    all_metric_rows: List[Dict[str, object]] = []
    all_gate_rows: List[Dict[str, object]] = []
    all_learnability_rows: List[Dict[str, object]] = []
    if not args.skip_metric_sweep:
        all_metric_rows = run_metric_sweeps(args)
        flush_csv(Path(args.out_dir) / "metric_sweep_raw.csv", all_metric_rows)
    if not args.skip_gate_sweep:
        all_gate_rows = run_gate_sweep(args)
        flush_csv(Path(args.out_dir) / "gate_sample_complexity_raw.csv", all_gate_rows)
    if not args.skip_learnability_diagnostics:
        all_learnability_rows = run_learnability_diagnostics(args)
        flush_csv(Path(args.out_dir) / "learnability_diagnostics_raw.csv", all_learnability_rows)
    print("\nDone. Key outputs:")
    print(f"  {Path(args.out_dir) / 'metric_sweep_raw.csv'}")
    print(f"  {Path(args.out_dir) / 'gate_sample_complexity_raw.csv'}")
    print(f"  {Path(args.out_dir) / 'learnability_diagnostics_raw.csv'}")
    print(f"  {Path(args.out_dir)}/*.pdf and *.png")


if __name__ == "__main__":
    main()
