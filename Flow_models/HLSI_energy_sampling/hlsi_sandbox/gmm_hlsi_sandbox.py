#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMM HLSI sandbox.

Refactored from the Gamma/CE-HLSI GMM notebook into a configurable sandbox for
testing low-variance score-estimation hypotheses on analytically tractable GMMs.

Default retained methods:
    Tweedie, TSI, HLSI, CE-HLSI, Blended, OP-Blend

Default example after the first rotating-eigenvector run:
    sparse_protective_stress -- designed to test whether CE leaks when
    only a small fraction of the transition cloud carries the stiff
    protective curvature.

The script:
    1. Builds a configurable Gaussian-mixture family.
    2. Draws a reference bank and ground-truth samples.
    3. Runs the retained samplers.
    4. Computes sampler divergences against ground truth.
    5. Computes six CE/HLSI diagnostics.
    6. Saves metrics, plots, and a multipage dashboard.

Run:
    python gmm_hlsi_sandbox.py --output outputs/gmm_sandbox_demo

Optional:
    python gmm_hlsi_sandbox.py --config my_config.json --output outputs/my_run
    python gmm_hlsi_sandbox.py --quick
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# Example configuration
# =============================================================================

EXAMPLE_CONFIG: Dict[str, Any] = {
    "name": "sparse_protective_stress",
    "seed": 42,
    "device": "auto",                       # "auto", "cpu", "cuda"
    "dtype": "float64",

    "gmm": {
        # Families:
        #   shared_covariance
        #   heterogeneous_precision
        #   rotating_eigenvectors
        #   sparse_protective
        #   double_x
        #   rotated_pair
        "family": "sparse_protective",
        "dimension": 2,
        "n_components": 10,

        # If separation is null, overlap is converted to a separation proxy.
        # overlap ~ exp(-sep^2/(8*var_parallel)) for equal-covariance pairs.
        "overlap": 0.15,
        "separation": None,

        # Covariance controls.
        "base_variance": 1.0,
        "stiff_variance": 0.005,
        "precision_ratio": 200.0,
        "angle_spread_deg": 0.0,
        "weight_skew": 0.0,                 # 0 = uniform, >0 = exponential skew
        "sparse_protective_fraction": 0.10,
    },

    "reference": {
        "n_ref": 4000,
        "lmin": 1e-4,
        "lmax": 1e6,
    },

    "sampler": {
        "methods": ["Tweedie", "TSI", "HLSI", "CE-HLSI", "Blended", "OP-Blend"],
        "n_samples": 1500,
        "n_steps": 90,
        "t_max": 3.0,
        "t_min": 0.015,
        "score_batch_size": 256,
        "initial": "normal",                # currently "normal"
    },

    "diagnostics": {
        "n_states": 768,
        "t_grid": [0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0],
        "batch_size": 512,
        # first rotating-eigenvector result showed the crucial action near 0.03--0.12
        "spectrum_t": 0.06,
    },

    "metrics": {
        "n_ground_truth": 5000,
        "mmd_max_points": 1000,
        "ksd_max_points": 700,
        "nll_max_points": 3000,
        "sliced_wasserstein_max_points": 1500,
        "sliced_wasserstein_projections": 128,
        "score_rmse_n_eval": 1500,
        "score_rmse_t_grid": [0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0],
        "hist_bins_2d": 150,
    },

    "plot": {
        "max_scatter_points": 1600,
        "fig_dpi": 180,
    },
}


# =============================================================================
# Utilities
# =============================================================================

def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def make_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_dtype(config: Mapping[str, Any]) -> torch.dtype:
    dtype_name = str(config.get("dtype", "float64")).lower()
    if dtype_name in {"float64", "double", "torch.float64"}:
        return torch.float64
    if dtype_name in {"float32", "single", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def safe_float(x: Any) -> float:
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().item()
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    def convert(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return safe_float(v)
            return v.detach().cpu().tolist()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {str(k): convert(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(val) for val in v]
        return v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2, sort_keys=True)


# =============================================================================
# OU helpers
# =============================================================================

def at(t: torch.Tensor | float) -> torch.Tensor:
    t = torch.as_tensor(t)
    return torch.exp(-t)


def vt(t: torch.Tensor | float) -> torch.Tensor:
    t = torch.as_tensor(t)
    return 1.0 - torch.exp(-2.0 * t)


def as_scalar_time(t: torch.Tensor | float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.to(device=device, dtype=dtype)
    return torch.tensor(float(t), device=device, dtype=dtype)


# =============================================================================
# Linear algebra helpers
# =============================================================================

def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def batch_eye(batch: int, d: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(d, device=device, dtype=dtype).expand(batch, d, d)


def bmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", A, x)


def mat_from_eig(U: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ik,...k,...jk->...ij", U, lam, U)


def psd_clamp(A: torch.Tensor, lmin: float, lmax: float) -> torch.Tensor:
    eig, U = torch.linalg.eigh(sym(A))
    eig = eig.clamp(min=lmin, max=lmax)
    return mat_from_eig(U, eig)


def inv_psd_from_eig(U: torch.Tensor, lam: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return mat_from_eig(U, 1.0 / lam.clamp(min=eps))


def inv_sqrt_psd(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    eig, U = torch.linalg.eigh(sym(A))
    return mat_from_eig(U, torch.rsqrt(eig.clamp(min=eps)))


def spectral_norm_sym(A: torch.Tensor) -> torch.Tensor:
    eig = torch.linalg.eigvalsh(sym(A))
    return eig.abs().max(dim=-1).values


def project_symmetric_gate(G: torch.Tensor, min_eig: float = 0.0, max_eig: float = 1.0) -> torch.Tensor:
    eig, U = torch.linalg.eigh(sym(G))
    eig = eig.clamp(min=min_eig, max=max_eig)
    return mat_from_eig(U, eig)


# =============================================================================
# Gaussian mixture target
# =============================================================================

class GaussianMixture:
    """General d-dimensional Gaussian mixture with exact score and OU marginal score."""

    def __init__(
        self,
        mus: torch.Tensor,
        covs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        *,
        name: str = "GMM",
    ) -> None:
        if mus.ndim != 2:
            raise ValueError("mus must have shape [K,d]")
        if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
            raise ValueError("covs must have shape [K,d,d]")
        if covs.shape[0] != mus.shape[0] or covs.shape[1] != mus.shape[1]:
            raise ValueError("mus/covs shapes are inconsistent")

        self.mus = mus
        self.covs = sym(covs)
        self.K, self.d = mus.shape
        self.device = mus.device
        self.dtype = mus.dtype
        self.name = name

        self.cov_invs = torch.linalg.inv(self.covs)
        self.chols = torch.linalg.cholesky(self.covs)
        self.log_dets = torch.logdet(self.covs)

        if weights is None:
            weights = torch.full((self.K,), 1.0 / self.K, device=self.device, dtype=self.dtype)
        else:
            weights = torch.as_tensor(weights, device=self.device, dtype=self.dtype)
            weights = weights / weights.sum().clamp(min=1e-30)
        self.weights = weights
        self.log_weights = torch.log(weights.clamp(min=1e-30))

    def sample(self, n: int) -> torch.Tensor:
        k = torch.multinomial(self.weights, num_samples=n, replacement=True)
        z = torch.randn(n, self.d, device=self.device, dtype=self.dtype)
        return self.mus[k] + torch.einsum("nij,nj->ni", self.chols[k], z)

    def _log_comp(self, x: torch.Tensor) -> torch.Tensor:
        diffs = x.unsqueeze(1) - self.mus.unsqueeze(0)
        maha = torch.einsum("nki,kij,nkj->nk", diffs, self.cov_invs, diffs)
        const = self.d * math.log(2.0 * math.pi)
        return self.log_weights.unsqueeze(0) - 0.5 * (
            const + self.log_dets.unsqueeze(0) + maha
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self._log_comp(x), dim=1)

    def resp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._log_comp(x), dim=-1)

    def component_scores(self, x: torch.Tensor) -> torch.Tensor:
        diffs = x.unsqueeze(1) - self.mus.unsqueeze(0)
        return -torch.einsum("kij,nkj->nki", self.cov_invs, diffs)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        w = self.resp(x)
        sk = self.component_scores(x)
        return (w.unsqueeze(-1) * sk).sum(1)

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positive precision-like observed Hessian:
            H(x) = -∇ score(x) = E[P_k] - Cov[s_k(x)].
        This can be indefinite for mixtures; downstream HLSI processing chooses
        how to handle negative/untrusted eigendirections.
        """
        w = self.resp(x)
        sk = self.component_scores(x)
        sbar = (w.unsqueeze(-1) * sk).sum(1)
        centered = sk - sbar.unsqueeze(1)
        cov_scores = torch.einsum("nk,nki,nkj->nij", w, centered, centered)
        mean_prec = torch.einsum("nk,kij->nij", w, self.cov_invs)
        return mean_prec - cov_scores

    def forward_sample(self, n: int, t: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = as_scalar_time(t, device=self.device, dtype=self.dtype)
        x0 = self.sample(n)
        y = at(t).to(self.device) * x0 + torch.sqrt(vt(t).to(self.device)) * torch.randn_like(x0)
        return y, x0

    def marginal_params(self, t: torch.Tensor | float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = as_scalar_time(t, device=self.device, dtype=self.dtype)
        a = at(t).to(self.device, self.dtype)
        v = vt(t).to(self.device, self.dtype)
        eye = torch.eye(self.d, device=self.device, dtype=self.dtype)
        mus_t = a * self.mus
        covs_t = a * a * self.covs + v * eye.unsqueeze(0)
        invs_t = torch.linalg.inv(covs_t)
        log_dets_t = torch.logdet(covs_t)
        return mus_t, covs_t, invs_t, log_dets_t

    def _log_comp_t(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        mus_t, covs_t, invs_t, log_dets_t = self.marginal_params(t)
        diffs = y.unsqueeze(1) - mus_t.unsqueeze(0)
        maha = torch.einsum("nki,kij,nkj->nk", diffs, invs_t, diffs)
        const = self.d * math.log(2.0 * math.pi)
        return self.log_weights.unsqueeze(0) - 0.5 * (
            const + log_dets_t.unsqueeze(0) + maha
        )

    def resp_t(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        return torch.softmax(self._log_comp_t(y, t), dim=-1)

    def component_scores_t(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        mus_t, _, invs_t, _ = self.marginal_params(t)
        diffs = y.unsqueeze(1) - mus_t.unsqueeze(0)
        return -torch.einsum("kij,nkj->nki", invs_t, diffs)

    def score_t(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        w = self.resp_t(y, t)
        sk = self.component_scores_t(y, t)
        return (w.unsqueeze(-1) * sk).sum(1)

    def score_cov_t(self, y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        w = self.resp_t(y, t)
        sk = self.component_scores_t(y, t)
        sbar = (w.unsqueeze(-1) * sk).sum(1)
        cen = sk - sbar.unsqueeze(1)
        return torch.einsum("nk,nki,nkj->nij", w, cen, cen)


# =============================================================================
# Configurable GMM families
# =============================================================================

def _orthogonal_matrix(d: int, *, device: torch.device, dtype: torch.dtype, seed_shift: int = 0) -> torch.Tensor:
    # deterministic-ish random orthogonal matrix for d>2 families
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234 + seed_shift)
    A = torch.randn(d, d, generator=gen, dtype=dtype).to(device)
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diagonal(R)).clamp(min=0) * 2 - 1
    return Q * signs.unsqueeze(0)


def _rotation2(theta: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)


def _embed_rotation(d: int, theta: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    R = torch.eye(d, device=device, dtype=dtype)
    if d >= 2:
        R[:2, :2] = _rotation2(theta, device=device, dtype=dtype)
    return R


def _circle_means(K: int, d: int, radius: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mus = torch.zeros(K, d, device=device, dtype=dtype)
    if d == 1:
        vals = torch.linspace(-radius, radius, K, device=device, dtype=dtype)
        mus[:, 0] = vals
        return mus
    angles = torch.linspace(0, 2 * math.pi, K + 1, device=device, dtype=dtype)[:-1]
    mus[:, 0] = radius * torch.cos(angles)
    mus[:, 1] = radius * torch.sin(angles)
    return mus


def _maybe_separation(gcfg: Mapping[str, Any], var_parallel: float) -> float:
    sep = gcfg.get("separation", None)
    if sep is not None:
        return float(sep)
    overlap = float(gcfg.get("overlap", 0.1))
    overlap = min(max(overlap, 1e-8), 0.999)
    return math.sqrt(max(1e-12, -8.0 * var_parallel * math.log(overlap)))


def _weights(K: int, skew: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if abs(skew) < 1e-12:
        return torch.full((K,), 1.0 / K, device=device, dtype=dtype)
    idx = torch.arange(K, device=device, dtype=dtype)
    w = torch.exp(-float(skew) * idx)
    return w / w.sum()


def build_gmm_from_config(config: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype) -> GaussianMixture:
    gcfg = config["gmm"]
    family = str(gcfg.get("family", "rotating_eigenvectors"))
    d = int(gcfg.get("dimension", 2))
    K = int(gcfg.get("n_components", 6))
    base_var = float(gcfg.get("base_variance", 1.0))
    stiff_var = float(gcfg.get("stiff_variance", base_var / float(gcfg.get("precision_ratio", 100.0))))
    precision_ratio = float(gcfg.get("precision_ratio", base_var / max(stiff_var, 1e-12)))
    angle_spread = math.radians(float(gcfg.get("angle_spread_deg", 120.0)))
    skew = float(gcfg.get("weight_skew", 0.0))

    sep = _maybe_separation(gcfg, var_parallel=base_var)
    weights = _weights(K, skew, device=device, dtype=dtype)

    if family == "rotated_pair":
        K = 2
        theta = angle_spread / 2.0
        R = _embed_rotation(d, theta, device=device, dtype=dtype)
        diag = torch.full((d,), base_var, device=device, dtype=dtype)
        if d >= 2:
            diag[1] = stiff_var
        cov = R @ torch.diag(diag) @ R.T
        means = torch.zeros(K, d, device=device, dtype=dtype)
        means[0, 0] = -sep / 2.0
        means[1, 0] = sep / 2.0
        weights = torch.full((2,), 0.5, device=device, dtype=dtype)
        return GaussianMixture(means, cov.unsqueeze(0).repeat(K, 1, 1), weights, name=f"rotated_pair_d{d}")

    if family == "double_x":
        if d < 2:
            raise ValueError("double_x requires dimension >= 2")
        theta = angle_spread / 2.0
        K = 4
        R_plus = _embed_rotation(d, theta, device=device, dtype=dtype)
        R_minus = _embed_rotation(d, -theta, device=device, dtype=dtype)
        diag = torch.full((d,), base_var, device=device, dtype=dtype)
        diag[1] = stiff_var
        cov_plus = R_plus @ torch.diag(diag) @ R_plus.T
        cov_minus = R_minus @ torch.diag(diag) @ R_minus.T

        means = torch.zeros(K, d, device=device, dtype=dtype)
        # Separate along the sloppy axis of R_plus.
        e = R_plus[:, 0]
        means[0] = -0.5 * sep * e
        means[1] = -0.5 * sep * e
        means[2] = 0.5 * sep * e
        means[3] = 0.5 * sep * e
        covs = torch.stack([cov_plus, cov_minus, cov_plus, cov_minus], dim=0)
        weights = torch.full((K,), 1.0 / K, device=device, dtype=dtype)
        return GaussianMixture(means, covs, weights, name=f"double_x_d{d}")

    means = _circle_means(K, d, sep, device=device, dtype=dtype)
    covs: List[torch.Tensor] = []

    if family == "shared_covariance":
        diag = torch.full((d,), base_var, device=device, dtype=dtype)
        if d >= 2:
            diag[1] = stiff_var
        R = _embed_rotation(d, angle_spread / 4.0, device=device, dtype=dtype)
        cov = R @ torch.diag(diag) @ R.T
        covs = [cov for _ in range(K)]

    elif family == "heterogeneous_precision":
        # Same eigenbasis, component-dependent stiffness.
        for k in range(K):
            ratio_k = 1.0 + (precision_ratio - 1.0) * (k / max(1, K - 1))
            diag = torch.full((d,), base_var, device=device, dtype=dtype)
            if d >= 2:
                diag[1] = base_var / ratio_k
            else:
                diag[0] = base_var / ratio_k
            covs.append(torch.diag(diag))

    elif family == "rotating_eigenvectors":
        if K == 1:
            angles = [0.0]
        else:
            angles = np.linspace(-0.5 * angle_spread, 0.5 * angle_spread, K)
        diag = torch.full((d,), base_var, device=device, dtype=dtype)
        if d >= 2:
            diag[1] = stiff_var
        else:
            diag[0] = stiff_var
        for theta in angles:
            R = _embed_rotation(d, float(theta), device=device, dtype=dtype)
            covs.append(R @ torch.diag(diag) @ R.T)

    elif family == "sparse_protective":
        # Most components are soft; a small subset has a stiff protective direction.
        frac = float(gcfg.get("sparse_protective_fraction", 0.20))
        n_stiff = max(1, int(round(K * frac)))
        for k in range(K):
            diag = torch.full((d,), base_var, device=device, dtype=dtype)
            if k < n_stiff:
                diag[0] = stiff_var
            covs.append(torch.diag(diag))
        # Place stiff components near the origin/wall and soft components around them.
        means = _circle_means(K, d, sep, device=device, dtype=dtype)
        means[:n_stiff] *= 0.15
        weights = torch.full((K,), 1.0 / K, device=device, dtype=dtype)

    else:
        raise ValueError(f"Unknown GMM family: {family}")

    return GaussianMixture(means, torch.stack(covs, dim=0), weights, name=f"{family}_d{d}_K{K}")


# =============================================================================
# Reference-bank weights and estimators
# =============================================================================

def snis_weights(y: torch.Tensor, t: torch.Tensor | float, xr: torch.Tensor) -> torch.Tensor:
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    a = at(t).to(y.device, y.dtype)
    v = vt(t).to(y.device, y.dtype)
    diff = y.unsqueeze(1) - a * xr.unsqueeze(0)
    lw = -0.5 * (diff * diff).sum(-1) / v.clamp(min=1e-30)
    lw = lw - lw.max(dim=1, keepdim=True).values
    w = lw.exp()
    return w / w.sum(dim=1, keepdim=True).clamp(min=1e-30)


def weighted_mean(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (w.unsqueeze(-1) * X).sum(1)


def weighted_cov(w: torch.Tensor, X: torch.Tensor, mean: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mean is None:
        mean = weighted_mean(w, X)
    Xc = X - mean.unsqueeze(1)
    return torch.einsum("bm,bmi,bmj->bij", w, Xc, Xc)


def precompute_reference_geometry(
    gmm: GaussianMixture,
    xr: torch.Tensor,
    *,
    lmin: float,
    lmax: float,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        s0 = gmm.score(xr)
        H = gmm.hessian(xr)
        H = sym(H)
        lam, V = torch.linalg.eigh(H)
        ok = (lam > lmin) & (lam <= lmax)
        lam_ce = lam.clamp(min=lmin, max=lmax)
        H_ce = mat_from_eig(V, lam_ce)
        return {
            "s0": s0,
            "H": H,
            "lam": lam,
            "V": V,
            "ok": ok,
            "H_ce": H_ce,
        }


def est_tweedie(y: torch.Tensor, t: torch.Tensor | float, xr: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    a = at(t).to(y.device, y.dtype)
    v = vt(t).to(y.device, y.dtype)
    disp = y.unsqueeze(1) - a * xr.unsqueeze(0)
    return -(w.unsqueeze(-1) * disp).sum(1) / v.clamp(min=1e-30)


def est_tsi(
    y: torch.Tensor,
    t: torch.Tensor | float,
    xr: torch.Tensor,
    w: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
) -> torch.Tensor:
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    return (w.unsqueeze(-1) * precomp["s0"].unsqueeze(0)).sum(1) / at(t).to(y.device, y.dtype)


def est_hlsi(
    y: torch.Tensor,
    t: torch.Tensor | float,
    xr: torch.Tensor,
    w: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
    *,
    lmin: float,
    lmax: float,
) -> torch.Tensor:
    """Reference-bank HLSI with trusted positive Hessian eigendirections."""
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    a = at(t).to(y.device, y.dtype)
    a2 = a * a
    v = vt(t).to(y.device, y.dtype)

    s0 = precomp["s0"]
    lam = precomp["lam"]
    V = precomp["V"]
    ok = (lam > lmin) & (lam <= lmax)
    lam_g = torch.where(ok, lam, torch.zeros_like(lam))

    s0_eig = torch.einsum("mji,mj->mi", V, s0)
    delta_eig = torch.where(ok, s0_eig / lam_g.clamp(min=1e-30), torch.zeros_like(s0_eig))
    delta = torch.einsum("mij,mj->mi", V, delta_eig)
    mu = xr + delta

    sig_inv_eig = torch.where(
        ok,
        lam_g / (a2 + v * lam_g.clamp(min=1e-30)),
        torch.full_like(lam, 1.0 / v.clamp(min=1e-30)),
    )

    disp = y.unsqueeze(1) - a * mu.unsqueeze(0)
    disp_eig = torch.einsum("mji,bmj->bmi", V, disp)
    score_eig = sig_inv_eig.unsqueeze(0) * disp_eig
    comp = -torch.einsum("mij,bmj->bmi", V, score_eig)
    return weighted_mean(w, comp)


def ce_gate_from_Hbar(Hbar: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    t = as_scalar_time(t, device=Hbar.device, dtype=Hbar.dtype)
    a = at(t).to(Hbar.device, Hbar.dtype)
    v = vt(t).to(Hbar.device, Hbar.dtype)
    B, d, _ = Hbar.shape
    I = batch_eye(B, d, device=Hbar.device, dtype=Hbar.dtype)
    M = a * a * I + v * sym(Hbar)
    return (a * a) * torch.linalg.solve(M, I)


def est_ce_hlsi(
    y: torch.Tensor,
    t: torch.Tensor | float,
    xr: torch.Tensor,
    w: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
) -> torch.Tensor:
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    stwd = est_tweedie(y, t, xr, w)
    stsi = est_tsi(y, t, xr, w, precomp)
    Hbar = torch.einsum("bm,mij->bij", w, precomp["H_ce"])
    Hbar = sym(Hbar)
    G = ce_gate_from_Hbar(Hbar, t)
    return stwd + bmv(G, stsi - stwd)


def est_blended(
    y: torch.Tensor,
    t: torch.Tensor | float,
    xr: torch.Tensor,
    w: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Coordinatewise variance-optimal blend between TSI and Tweedie signals."""
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    a = at(t).to(y.device, y.dtype)
    v = vt(t).to(y.device, y.dtype)

    tsi = precomp["s0"].unsqueeze(0) / a
    twd = -(y.unsqueeze(1) - a * xr.unsqueeze(0)) / v.clamp(min=1e-30)

    am = weighted_mean(w, tsi.expand(y.shape[0], -1, -1))
    bm = weighted_mean(w, twd)

    Ac = tsi.expand(y.shape[0], -1, -1) - am.unsqueeze(1)
    Bc = twd - bm.unsqueeze(1)
    va = (w.unsqueeze(-1) * Ac.square()).sum(1).clamp(min=1e-30)
    vb = (w.unsqueeze(-1) * Bc.square()).sum(1).clamp(min=1e-30)
    cab = (w.unsqueeze(-1) * Ac * Bc).sum(1)
    den = (va + vb - 2.0 * cab).clamp(min=1e-20)
    g = ((va - cab) / den).clamp(0.0, 1.0)
    return am + g * (bm - am)


def op_blend_gate(
    y: torch.Tensor,
    t: torch.Tensor | float,
    xr: torch.Tensor,
    w: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
    *,
    reg: float = 1e-8,
    pinv_rtol: float = 1e-6,
    project_gate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full matrix empirical variance-optimal gate between TSI and Tweedie."""
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    a = at(t).to(y.device, y.dtype)
    v = vt(t).to(y.device, y.dtype)

    tsi = precomp["s0"].unsqueeze(0) / a
    tsi = tsi.expand(y.shape[0], -1, -1)
    twd = -(y.unsqueeze(1) - a * xr.unsqueeze(0)) / v.clamp(min=1e-30)

    am = weighted_mean(w, tsi)
    bm = weighted_mean(w, twd)

    D = twd - tsi
    Dm = bm - am
    Ac = tsi - am.unsqueeze(1)
    Dc = D - Dm.unsqueeze(1)

    C_AD = torch.einsum("bm,bmi,bmj->bij", w, Ac, Dc)
    C_DD = torch.einsum("bm,bmi,bmj->bij", w, Dc, Dc)
    C_DD = sym(C_DD)

    B, d = y.shape
    I = batch_eye(B, d, device=y.device, dtype=y.dtype)
    scale = C_DD.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1.0)
    C_DD_solve = C_DD + (reg * scale).view(B, 1, 1) * I
    G = -torch.matmul(C_AD, torch.linalg.pinv(C_DD_solve, rtol=pinv_rtol))

    if project_gate:
        G = project_symmetric_gate(G, 0.0, 1.0)

    return G, am, bm, C_DD


def est_op_blended(
    y: torch.Tensor,
    t: torch.Tensor | float,
    xr: torch.Tensor,
    w: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
    *,
    reg: float = 1e-8,
    pinv_rtol: float = 1e-6,
    project_gate: bool = False,
) -> torch.Tensor:
    G, am, bm, _ = op_blend_gate(
        y, t, xr, w, precomp, reg=reg, pinv_rtol=pinv_rtol, project_gate=project_gate
    )
    return am + bmv(G, bm - am)


EstimatorFn = Callable[[torch.Tensor, torch.Tensor | float, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]


def score_estimator(
    method: str,
    y: torch.Tensor,
    t: torch.Tensor | float,
    xr: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
    *,
    lmin: float,
    lmax: float,
) -> torch.Tensor:
    w = snis_weights(y, t, xr)
    if method == "Tweedie":
        return est_tweedie(y, t, xr, w)
    if method == "TSI":
        return est_tsi(y, t, xr, w, precomp)
    if method == "HLSI":
        return est_hlsi(y, t, xr, w, precomp, lmin=lmin, lmax=lmax)
    if method == "CE-HLSI":
        return est_ce_hlsi(y, t, xr, w, precomp)
    if method == "Blended":
        return est_blended(y, t, xr, w, precomp)
    if method == "OP-Blend":
        return est_op_blended(y, t, xr, w, precomp)
    raise ValueError(f"Unknown method: {method}")


def make_score_fn(
    method: str,
    gmm: GaussianMixture,
    xr: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
    *,
    lmin: float,
    lmax: float,
    batch_size: int = 256,
) -> Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor]:
    def fn(y: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        outs = []
        for start in range(0, y.shape[0], batch_size):
            yb = y[start:start + batch_size]
            outs.append(score_estimator(method, yb, t, xr, precomp, lmin=lmin, lmax=lmax))
        return torch.cat(outs, dim=0)
    return fn


# =============================================================================
# Reverse OU sampler
# =============================================================================

@torch.no_grad()
def reverse_ou_heun_sde(
    score_fn: Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor],
    *,
    n: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    n_steps: int = 100,
    t_max: float = 3.0,
    t_min: float = 0.015,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    ts = torch.linspace(t_max, t_min, n_steps + 1, device=device, dtype=dtype)
    y = torch.randn(n, d, device=device, dtype=dtype)

    max_abs_score = 0.0
    fail = False
    fail_reason = ""

    for i in range(n_steps):
        tc = ts[i]
        tn = ts[i + 1]
        h = tc - tn

        s1 = score_fn(y, tc)
        max_abs_score = max(max_abs_score, safe_float(s1.abs().max()))
        if not torch.isfinite(s1).all():
            fail, fail_reason = True, "nonfinite score at predictor"
            break

        drift1 = y + 2.0 * s1
        noise = torch.sqrt(2.0 * h) * torch.randn_like(y)
        yh = y + h * drift1 + noise

        s2 = score_fn(yh, tn)
        max_abs_score = max(max_abs_score, safe_float(s2.abs().max()))
        if not torch.isfinite(s2).all():
            fail, fail_reason = True, "nonfinite score at corrector"
            break

        drift2 = yh + 2.0 * s2
        y = y + 0.5 * h * (drift1 + drift2) + noise

        if not torch.isfinite(y).all():
            fail, fail_reason = True, "nonfinite state"
            break

    if not fail:
        tf = torch.tensor(t_min, device=device, dtype=dtype)
        sf = score_fn(y, tf)
        max_abs_score = max(max_abs_score, safe_float(sf.abs().max()))
        if not torch.isfinite(sf).all():
            fail, fail_reason = True, "nonfinite final score"
        else:
            # Tweedie final denoising step to approximately x0.
            y = (y + vt(tf).to(device, dtype) * sf) / at(tf).to(device, dtype)

    return y, {
        "failed": bool(fail),
        "fail_reason": fail_reason,
        "max_abs_score": max_abs_score,
    }


# =============================================================================
# Metrics
# =============================================================================

def _subsample(X: torch.Tensor, max_points: int) -> torch.Tensor:
    if X.shape[0] <= max_points:
        return X
    idx = torch.randperm(X.shape[0], device=X.device)[:max_points]
    return X[idx]


@torch.no_grad()
def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, *, max_points: int = 1000) -> float:
    X = _subsample(X, max_points)
    Y = _subsample(Y, max_points)
    Z = torch.cat([X, Y], dim=0)
    d2 = torch.cdist(Z, Z).square()
    med = torch.median(d2[d2 > 0]).clamp(min=1e-8)
    # Multi-bandwidth kernel centered around median.
    scales = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], device=X.device, dtype=X.dtype)
    vals = []
    for s in scales:
        h2 = med * s
        Kxx = torch.exp(-torch.cdist(X, X).square() / (2.0 * h2))
        Kyy = torch.exp(-torch.cdist(Y, Y).square() / (2.0 * h2))
        Kxy = torch.exp(-torch.cdist(X, Y).square() / (2.0 * h2))
        vals.append(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return safe_float(torch.stack(vals).mean().clamp(min=0.0))


@torch.no_grad()
def sliced_wasserstein2(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    max_points: int = 1200,
    n_projections: int = 128,
) -> float:
    X = _subsample(X, max_points)
    Y = _subsample(Y, max_points)
    n = min(X.shape[0], Y.shape[0])
    X = X[:n]
    Y = Y[:n]
    d = X.shape[1]
    theta = torch.randn(n_projections, d, device=X.device, dtype=X.dtype)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp(min=1e-12)
    Xp = X @ theta.T
    Yp = Y @ theta.T
    Xs = torch.sort(Xp, dim=0).values
    Ys = torch.sort(Yp, dim=0).values
    return safe_float(((Xs - Ys) ** 2).mean())


@torch.no_grad()
def target_nll(samples: torch.Tensor, gmm: GaussianMixture, *, max_points: int = 3000) -> float:
    X = _subsample(samples, max_points)
    return safe_float(-gmm.log_prob(X).mean())


@torch.no_grad()
def ksd_rbf(
    samples: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_points: int = 700,
) -> float:
    X = _subsample(samples, max_points)
    S = score_fn(X)
    n, d = X.shape
    diffs = X.unsqueeze(1) - X.unsqueeze(0)
    d2 = (diffs * diffs).sum(-1)
    med = torch.median(d2[d2 > 0]).clamp(min=1e-8)
    h2 = med
    K = torch.exp(-d2 / (2.0 * h2))
    ss = S @ S.T
    sx_grad_y = torch.einsum("id,ijd->ij", S, diffs) / h2 * K
    sy_grad_x = -torch.einsum("jd,ijd->ij", S, diffs) / h2 * K
    trace = (d / h2 - d2 / (h2 * h2)) * K
    kstein = ss * K + sx_grad_y + sy_grad_x + trace
    return safe_float(kstein.mean().clamp(min=0.0).sqrt())


@torch.no_grad()
def hist_kl_2d(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    bins: int = 150,
    smoothing: float = 1e-12,
) -> Tuple[float, float]:
    """Approximate KL(P_X || P_Y), KL(P_Y || P_X) using first two coordinates."""
    Xn = to_cpu_np(X[:, :2])
    Yn = to_cpu_np(Y[:, :2])
    lo = np.minimum(np.percentile(Xn, 0.5, axis=0), np.percentile(Yn, 0.5, axis=0))
    hi = np.maximum(np.percentile(Xn, 99.5, axis=0), np.percentile(Yn, 99.5, axis=0))
    pad = 0.05 * (hi - lo + 1e-8)
    rng = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
    Hx, _, _ = np.histogram2d(Xn[:, 0], Xn[:, 1], bins=bins, range=rng, density=False)
    Hy, _, _ = np.histogram2d(Yn[:, 0], Yn[:, 1], bins=bins, range=rng, density=False)
    px = Hx.astype(np.float64) + smoothing
    py = Hy.astype(np.float64) + smoothing
    px = px / px.sum()
    py = py / py.sum()
    kl_xy = float(np.sum(px * (np.log(px) - np.log(py))))
    kl_yx = float(np.sum(py * (np.log(py) - np.log(px))))
    return kl_xy, kl_yx


@torch.no_grad()
def score_rmse_forward_process(
    score_fn: Callable[[torch.Tensor, torch.Tensor | float], torch.Tensor],
    gmm: GaussianMixture,
    *,
    n_eval: int,
    t_grid: Iterable[float],
    batch_size: int = 256,
) -> float:
    vals = []
    for t in t_grid:
        y, _ = gmm.forward_sample(n_eval, t)
        exact = gmm.score_t(y, t)
        pred_chunks = []
        for start in range(0, n_eval, batch_size):
            pred_chunks.append(score_fn(y[start:start + batch_size], t))
        pred = torch.cat(pred_chunks, dim=0)
        vals.append((pred - exact).square().sum(dim=1).mean())
    return safe_float(torch.stack(vals).mean().sqrt())


@torch.no_grad()
def evaluate_samples(
    samples: torch.Tensor,
    gt: torch.Tensor,
    gmm: GaussianMixture,
    cfg: Mapping[str, Any],
) -> Dict[str, float]:
    mcfg = cfg["metrics"]
    metrics: Dict[str, float] = {}
    if not torch.isfinite(samples).all():
        for key in ["nll", "mmd", "ksd", "sliced_w2", "kl_gt_to_model_2d", "kl_model_to_gt_2d"]:
            metrics[key] = float("nan")
        return metrics

    metrics["nll"] = target_nll(samples, gmm, max_points=int(mcfg["nll_max_points"]))
    metrics["mmd"] = mmd_rbf(samples, gt, max_points=int(mcfg["mmd_max_points"]))
    metrics["ksd"] = ksd_rbf(samples, gmm.score, max_points=int(mcfg["ksd_max_points"]))
    metrics["sliced_w2"] = sliced_wasserstein2(
        samples,
        gt,
        max_points=int(mcfg["sliced_wasserstein_max_points"]),
        n_projections=int(mcfg["sliced_wasserstein_projections"]),
    )
    if gmm.d >= 2:
        kl_gt_model, kl_model_gt = hist_kl_2d(
            gt,
            samples,
            bins=int(mcfg["hist_bins_2d"]),
        )
        metrics["kl_gt_to_model_2d"] = kl_gt_model
        metrics["kl_model_to_gt_2d"] = kl_model_gt
    else:
        metrics["kl_gt_to_model_2d"] = float("nan")
        metrics["kl_model_to_gt_2d"] = float("nan")
    return metrics


# =============================================================================
# Six diagnostics
# =============================================================================

@torch.no_grad()
def local_signal_objects(
    y: torch.Tensor,
    t: torch.Tensor | float,
    gmm: GaussianMixture,
    xr: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    t = as_scalar_time(t, device=y.device, dtype=y.dtype)
    a = at(t).to(y.device, y.dtype)
    v = vt(t).to(y.device, y.dtype)

    w = snis_weights(y, t, xr)
    tsi_i = precomp["s0"].unsqueeze(0).expand(y.shape[0], -1, -1) / a
    twd_i = -(y.unsqueeze(1) - a * xr.unsqueeze(0)) / v.clamp(min=1e-30)

    stsi = weighted_mean(w, tsi_i)
    stwd = weighted_mean(w, twd_i)
    delta_i = tsi_i - twd_i
    delta_bar = stsi - stwd

    H_ref = precomp["H_ce"]
    Hbar = sym(torch.einsum("bm,mij->bij", w, H_ref))
    Gce = ce_gate_from_Hbar(Hbar, t)

    return {
        "w": w,
        "tsi_i": tsi_i,
        "twd_i": twd_i,
        "stsi": stsi,
        "stwd": stwd,
        "delta_i": delta_i,
        "delta_bar": delta_bar,
        "Hbar": Hbar,
        "Gce": Gce,
    }


@torch.no_grad()
def compute_diagnostics(
    gmm: GaussianMixture,
    xr: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
    methods: List[str],
    cfg: Mapping[str, Any],
    *,
    lmin: float,
    lmax: float,
) -> Dict[str, Any]:
    dcfg = cfg["diagnostics"]
    n_states = int(dcfg["n_states"])
    t_grid = [float(t) for t in dcfg["t_grid"]]
    batch_size = int(dcfg.get("batch_size", n_states))

    by_t: List[Dict[str, Any]] = []
    method_risk: Dict[str, List[float]] = {m: [] for m in methods}

    for t in t_grid:
        accum: Dict[str, List[float]] = {
            "epsilon_cov_exact": [],
            "epsilon_cov_bank": [],
            "epsilon_cov_bank_to_exact": [],
            "score_cov_exact_norm": [],
            "score_cov_bank_norm": [],
            # Relative-to-componentwise mismatch is useful but can be misleading
            # when the componentwise gated signal is tiny; keep absolute and
            # pre-disagreement-normalized versions too.
            "ce_gate_signal_rel_mismatch": [],
            "ce_gate_signal_abs_mismatch": [],
            "ce_gate_signal_pre_rel_mismatch": [],
            "ce_attenuation_ratio": [],
            "op_attenuation_ratio": [],
            # Direct leakage diagnostic requested after the first run:
            # how much more post-gated OP disagreement survives than CE.
            "op_ce_leakage_ratio": [],
            "op_minus_ce_attenuation": [],
            "op_minus_ce_local_risk": [],
            "op_to_ce_local_risk_ratio": [],
            "protective_mass": [],
            "oracle_moment_cond": [],
            "oracle_moment_rank_eff": [],
            "ess": [],
            "hlsi_surrogate_target_rmse": [],
            "ce_target_rmse": [],
        }
        risk_accum = {m: [] for m in methods}

        for start in range(0, n_states, batch_size):
            b = min(batch_size, n_states - start)
            y, _ = gmm.forward_sample(b, t)
            true_score = gmm.score_t(y, t)

            obj = local_signal_objects(y, t, gmm, xr, precomp)
            w = obj["w"]
            tsi_i = obj["tsi_i"]
            twd_i = obj["twd_i"]
            delta_i = obj["delta_i"]
            delta_bar = obj["delta_bar"]
            Hbar = obj["Hbar"]
            Gce = obj["Gce"]

            B, M, d = delta_i.shape
            I = batch_eye(B, d, device=y.device, dtype=y.dtype)
            tau = (at(torch.tensor(t, device=y.device, dtype=y.dtype)) ** 2) / vt(torch.tensor(t, device=y.device, dtype=y.dtype)).clamp(min=1e-30)
            Ptau = tau * I + Hbar
            invsqrt = inv_sqrt_psd(Ptau, eps=1e-12)

            # 1. Conditional score covariance, exact GMM components and bank TSI signals.
            Cs_exact = gmm.score_cov_t(y, t)
            Cs_bank = weighted_cov(w, tsi_i)
            norm_exact = spectral_norm_sym(Cs_exact)
            norm_bank = spectral_norm_sym(Cs_bank)
            eps_exact = spectral_norm_sym(invsqrt @ Cs_exact @ invsqrt)
            eps_bank = spectral_norm_sym(invsqrt @ Cs_bank @ invsqrt)

            accum["score_cov_exact_norm"].append(safe_float(norm_exact.mean()))
            accum["score_cov_bank_norm"].append(safe_float(norm_bank.mean()))
            accum["epsilon_cov_exact"].append(safe_float(eps_exact.mean()))
            accum["epsilon_cov_bank"].append(safe_float(eps_bank.mean()))
            accum["epsilon_cov_bank_to_exact"].append(
                safe_float((eps_bank / eps_exact.clamp(min=1e-12)).median())
            )

            # 2. Gate-signal covariance / CE compression mismatch.
            H_i = precomp["H_ce"]
            # Per-reference gate G_i = a^2(a^2 I + v H_i)^{-1}
            a = at(torch.tensor(t, device=y.device, dtype=y.dtype))
            v = vt(torch.tensor(t, device=y.device, dtype=y.dtype))
            Iref = torch.eye(d, device=y.device, dtype=y.dtype).unsqueeze(0).expand(M, d, d)
            Gi = (a * a) * torch.linalg.solve(a * a * Iref + v * H_i, Iref)  # [M,d,d]
            Gi_delta = torch.einsum("mij,bmj->bmi", Gi, delta_i)
            compwise_gate_signal = weighted_mean(w, Gi_delta)
            ce_gate_signal = bmv(Gce, delta_bar)
            mismatch = compwise_gate_signal - ce_gate_signal
            mismatch_norm = mismatch.norm(dim=1)
            rel_mismatch = mismatch_norm / compwise_gate_signal.norm(dim=1).clamp(min=1e-12)
            accum["ce_gate_signal_rel_mismatch"].append(safe_float(rel_mismatch.mean()))
            accum["ce_gate_signal_abs_mismatch"].append(safe_float(mismatch_norm.mean()))

            # 3. Pre/post rectification attenuation.
            pre_norm = delta_bar.norm(dim=1).clamp(min=1e-12)
            pre_rel_mismatch = mismatch_norm / pre_norm
            accum["ce_gate_signal_pre_rel_mismatch"].append(safe_float(pre_rel_mismatch.mean()))

            ce_att = ce_gate_signal.norm(dim=1) / pre_norm
            accum["ce_attenuation_ratio"].append(safe_float(ce_att.mean()))

            Gop, am, bm, C_DD = op_blend_gate(y, t, xr, w, precomp)
            op_delta = bm - am
            op_gate_signal = bmv(Gop, op_delta)
            op_att = op_gate_signal.norm(dim=1) / op_delta.norm(dim=1).clamp(min=1e-12)
            accum["op_attenuation_ratio"].append(safe_float(op_att.mean()))

            op_ce_leak = op_gate_signal.norm(dim=1) / ce_gate_signal.norm(dim=1).clamp(min=1e-12)
            accum["op_ce_leakage_ratio"].append(safe_float(op_ce_leak.median()))
            accum["op_minus_ce_attenuation"].append(safe_float((op_att - ce_att).mean()))

            ce_score_local = obj["stwd"] + ce_gate_signal
            op_score_local = am + op_gate_signal
            ce_local_risk = (ce_score_local - true_score).square().sum(dim=1)
            op_local_risk = (op_score_local - true_score).square().sum(dim=1)
            accum["op_minus_ce_local_risk"].append(safe_float((op_local_risk - ce_local_risk).mean()))
            accum["op_to_ce_local_risk_ratio"].append(
                safe_float((op_local_risk / ce_local_risk.clamp(min=1e-12)).median())
            )

            # 4. Protective mass: Hbar directional mass relative to max component mass.
            eigHbar, Ubar = torch.linalg.eigh(sym(Hbar))
            u = Ubar[:, :, -1]  # top CE precision direction
            Hiu = torch.einsum("mij,bj->bmi", H_i, u)
            masses = torch.einsum("bi,bmi->bm", u, Hiu).clamp(min=0.0)
            hbar_mass = torch.einsum("bi,bij,bj->b", u, Hbar, u).clamp(min=0.0)
            max_mass = masses.max(dim=1).values.clamp(min=1e-12)
            protective_mass = (hbar_mass / max_mass).clamp(min=0.0)
            accum["protective_mass"].append(safe_float(protective_mass.mean()))

            # 5. Oracle moment conditioning for OP-blend moment M = Cov(D,D).
            eigM = torch.linalg.eigvalsh(sym(C_DD)).clamp(min=0.0)
            lam_max = eigM.max(dim=1).values
            lam_min_pos = torch.where(eigM > 1e-10 * lam_max.unsqueeze(1).clamp(min=1e-30), eigM, torch.inf).min(dim=1).values
            cond = lam_max / lam_min_pos.clamp(min=1e-30)
            cond = torch.where(torch.isfinite(cond), cond, torch.full_like(cond, float("nan")))
            rank_eff = (eigM > 1e-8 * lam_max.unsqueeze(1).clamp(min=1e-30)).sum(dim=1).to(y.dtype)
            accum["oracle_moment_cond"].append(safe_float(torch.nanmean(cond)))
            accum["oracle_moment_rank_eff"].append(safe_float(rank_eff.mean()))

            # ESS.
            ess = 1.0 / (w.square().sum(dim=1).clamp(min=1e-30))
            accum["ess"].append(safe_float(ess.mean()))

            # 6. Surrogate-target mismatch: HLSI/CE local score RMSE against exact p_t score.
            hlsi_score = est_hlsi(y, t, xr, w, precomp, lmin=lmin, lmax=lmax)
            ce_score = est_ce_hlsi(y, t, xr, w, precomp)
            accum["hlsi_surrogate_target_rmse"].append(safe_float((hlsi_score - true_score).square().sum(1).mean().sqrt()))
            accum["ce_target_rmse"].append(safe_float((ce_score - true_score).square().sum(1).mean().sqrt()))

            for m in methods:
                pred = score_estimator(m, y, t, xr, precomp, lmin=lmin, lmax=lmax)
                risk = (pred - true_score).square().sum(dim=1)
                risk_accum[m].append(safe_float(risk.mean()))

        row = {"t": t}
        for k, vals in accum.items():
            row[k] = float(np.nanmean(vals)) if len(vals) else float("nan")
        by_t.append(row)
        for m in methods:
            method_risk[m].append(float(np.nanmean(risk_accum[m])) if risk_accum[m] else float("nan"))

    summary: Dict[str, float] = {}
    for key in by_t[0].keys():
        if key == "t":
            continue
        summary[key] = float(np.nanmean([r[key] for r in by_t]))

    return {
        "by_t": by_t,
        "summary": summary,
        "method_score_mse_by_t": method_risk,
    }


@torch.no_grad()
def compute_attenuation_spectrum(
    gmm: GaussianMixture,
    xr: torch.Tensor,
    precomp: Dict[str, torch.Tensor],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    t = float(cfg["diagnostics"].get("spectrum_t", 0.12))
    n = int(min(cfg["diagnostics"]["n_states"], 512))
    y, _ = gmm.forward_sample(n, t)
    obj = local_signal_objects(y, t, gmm, xr, precomp)
    w = obj["w"]
    Hbar = obj["Hbar"]
    Gce = obj["Gce"]
    delta_bar = obj["delta_bar"]

    Gop, am, bm, _ = op_blend_gate(y, t, xr, w, precomp)
    delta_op = bm - am

    eig, U = torch.linalg.eigh(sym(Hbar))
    pre = torch.einsum("bij,bj->bi", U.transpose(-1, -2), delta_bar).abs()
    post_ce = torch.einsum("bij,bj->bi", U.transpose(-1, -2), bmv(Gce, delta_bar)).abs()
    post_op = torch.einsum("bij,bj->bi", U.transpose(-1, -2), bmv(Gop, delta_op)).abs()

    # Sort within each state by eigenvalue, then average by rank.
    idx = torch.argsort(eig, dim=1)
    eig_s = torch.gather(eig, 1, idx)
    pre_s = torch.gather(pre, 1, idx)
    ce_s = torch.gather(post_ce, 1, idx)
    op_s = torch.gather(post_op, 1, idx)

    return {
        "t": t,
        "rank": list(range(gmm.d)),
        "lambda_mean": to_cpu_np(eig_s.mean(0)).tolist(),
        "pre_disagreement_mean": to_cpu_np(pre_s.mean(0)).tolist(),
        "post_ce_mean": to_cpu_np(ce_s.mean(0)).tolist(),
        "post_op_mean": to_cpu_np(op_s.mean(0)).tolist(),
    }


# =============================================================================
# Plotting
# =============================================================================

def _method_color_order(methods: List[str]) -> List[str]:
    return methods


def plot_samples_grid(
    samples: Mapping[str, torch.Tensor],
    gt: torch.Tensor,
    gmm: GaussianMixture,
    cfg: Mapping[str, Any],
    out_path: str,
) -> plt.Figure:
    methods = list(samples.keys())
    max_points = int(cfg["plot"]["max_scatter_points"])
    ncols = min(4, len(methods) + 1)
    nrows = int(math.ceil((len(methods) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()

    all_xy = [gt[:, :2]]
    all_xy += [s[:, :2] for s in samples.values() if torch.isfinite(s).all()]
    cat = torch.cat([_subsample(x, max_points) for x in all_xy], dim=0)
    xy = to_cpu_np(cat)
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    pad = 0.08 * (hi - lo + 1e-8)

    def scatter(ax, X: torch.Tensor, title: str) -> None:
        Xs = _subsample(X, max_points)
        Xn = to_cpu_np(Xs[:, :2])
        ax.scatter(Xn[:, 0], Xn[:, 1], s=4, alpha=0.35)
        if gmm.d >= 2:
            mus = to_cpu_np(gmm.mus[:, :2])
            ax.scatter(mus[:, 0], mus[:, 1], s=50, marker="x")
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    scatter(axes[0], gt, "Ground truth")
    for i, m in enumerate(methods, start=1):
        scatter(axes[i], samples[m], m)

    for j in range(len(methods) + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Samples, first two coordinates — {gmm.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(cfg["plot"]["fig_dpi"]))
    return fig


def plot_method_risk_by_time(
    diagnostics: Mapping[str, Any],
    methods: List[str],
    out_path: str,
    cfg: Mapping[str, Any],
) -> plt.Figure:
    by_t = diagnostics["by_t"]
    t = np.array([r["t"] for r in by_t])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in methods:
        y = np.array(diagnostics["method_score_mse_by_t"][m], dtype=float)
        ax.plot(t, y, marker="o", label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel(r"local score MSE  $\mathbb{E}\|\hat s_t-s_t\|^2$")
    ax.set_title("Local score risk by diffusion time")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(cfg["plot"]["fig_dpi"]))
    return fig


def plot_performance_vs_diagnostics(
    diagnostics: Mapping[str, Any],
    out_path: str,
    cfg: Mapping[str, Any],
) -> plt.Figure:
    by_t = diagnostics["by_t"]
    t = np.array([r["t"] for r in by_t])
    ce_risk = np.array(diagnostics["method_score_mse_by_t"].get("CE-HLSI", [np.nan] * len(t)), dtype=float)
    op_risk = np.array(diagnostics["method_score_mse_by_t"].get("OP-Blend", [np.nan] * len(t)), dtype=float)
    yvals = ce_risk - np.nanmin(np.stack([ce_risk, op_risk], axis=0), axis=0)

    diag_keys = [
        ("epsilon_cov_exact", r"$\epsilon_{\rm cov}$ exact"),
        ("ce_gate_signal_abs_mismatch", r"$\|\Delta_{\rm CE}\|$"),
        ("ce_gate_signal_pre_rel_mismatch", r"$\|\Delta_{\rm CE}\|/\|d\|$"),
        ("protective_mass", "protective mass"),
        ("op_ce_leakage_ratio", r"OP/CE post-gate leakage"),
        ("oracle_moment_cond", r"oracle moment cond."),
        ("ess", "ESS"),
        ("epsilon_cov_bank_to_exact", r"bank/exact $\epsilon_{\rm cov}$"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.0), squeeze=False)
    axes = axes.ravel()
    sc = None
    for ax, (key, label) in zip(axes, diag_keys):
        x = np.array([r.get(key, np.nan) for r in by_t], dtype=float)
        sc = ax.scatter(x, yvals, c=np.log10(t), s=58)
        for xi, yi, ti in zip(x, yvals, t):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(f"{ti:.2g}", (xi, yi), fontsize=7, alpha=0.75)
        ax.set_xlabel(label)
        ax.set_ylabel("CE excess local risk vs best(CE, OP)")
        finite_pos = x[np.isfinite(x) & (x > 0)]
        if finite_pos.size and np.nanmax(finite_pos) / max(np.nanmin(finite_pos), 1e-12) > 100:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8)
        cbar.set_label(r"$\log_{10} t$")
    fig.suptitle("CE performance-vs-diagnostics phase plot", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(cfg["plot"]["fig_dpi"]))
    return fig



def plot_op_leakage_vs_risk(
    diagnostics: Mapping[str, Any],
    out_path: str,
    cfg: Mapping[str, Any],
) -> plt.Figure:
    """Dedicated plot for the post-run hypothesis: OP leaks too much exactly
    when it loses local score risk to CE."""
    by_t = diagnostics["by_t"]
    t = np.array([r["t"] for r in by_t], dtype=float)
    ce_risk = np.array(diagnostics["method_score_mse_by_t"].get("CE-HLSI", [np.nan] * len(t)), dtype=float)
    op_risk = np.array(diagnostics["method_score_mse_by_t"].get("OP-Blend", [np.nan] * len(t)), dtype=float)
    op_minus_ce = op_risk - ce_risk

    leak = np.array([r.get("op_ce_leakage_ratio", np.nan) for r in by_t], dtype=float)
    op_minus_ce_diag = np.array([r.get("op_minus_ce_local_risk", np.nan) for r in by_t], dtype=float)
    prot = np.array([r.get("protective_mass", np.nan) for r in by_t], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), squeeze=False)
    ax = axes[0, 0]
    sc = ax.scatter(leak, op_minus_ce, c=np.log10(t), s=68)
    for xi, yi, ti in zip(leak, op_minus_ce, t):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(f"{ti:.2g}", (xi, yi), fontsize=8, alpha=0.75)
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axvline(1.0, linewidth=1.0, alpha=0.5)
    ax.set_xlabel(r"median $\|G_{\rm OP}d_{\rm OP}\| / \|G_{\rm CE}d_{\rm CE}\|$")
    ax.set_ylabel(r"OP local score MSE $-$ CE local score MSE")
    ax.set_title("OP leakage vs OP-minus-CE risk")
    finite_pos = leak[np.isfinite(leak) & (leak > 0)]
    if finite_pos.size and np.nanmax(finite_pos) / max(np.nanmin(finite_pos), 1e-12) > 50:
        ax.set_xscale("log")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    sc2 = ax.scatter(prot, op_minus_ce_diag, c=np.log10(t), s=68)
    for xi, yi, ti in zip(prot, op_minus_ce_diag, t):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(f"{ti:.2g}", (xi, yi), fontsize=8, alpha=0.75)
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.set_xlabel("protective mass")
    ax.set_ylabel(r"diagnostic OP local risk $-$ CE local risk")
    ax.set_title("Protective mass vs diagnostic risk gap")
    ax.grid(alpha=0.25)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label(r"$\log_{10} t$")
    fig.suptitle("OP leakage and sparse-protection diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(cfg["plot"]["fig_dpi"]))
    return fig


def plot_attenuation_spectrum(
    spectrum: Mapping[str, Any],
    out_path: str,
    cfg: Mapping[str, Any],
) -> plt.Figure:
    rank = np.asarray(spectrum["rank"])
    lam = np.asarray(spectrum["lambda_mean"], dtype=float)
    pre = np.asarray(spectrum["pre_disagreement_mean"], dtype=float)
    ce = np.asarray(spectrum["post_ce_mean"], dtype=float)
    op = np.asarray(spectrum["post_op_mean"], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.plot(rank, pre, marker="o", label=r"pre: $|d_k|$")
    ax1.plot(rank, ce, marker="o", label=r"CE: $|[G_{\rm CE}d]_k|$")
    ax1.plot(rank, op, marker="o", label=r"OP: $|[G_{\rm OP}d]_k|$")
    ax1.set_yscale("log")
    ax1.set_xlabel("eigenvalue rank of CE precision")
    ax1.set_ylabel("mean projected disagreement magnitude")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(rank, lam, marker="x", linestyle="--", label=r"$\lambda_k(\bar P)$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"mean $\lambda_k(\bar P)$")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Pre/post rectification attenuation spectrum at t={spectrum['t']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(cfg["plot"]["fig_dpi"]))
    return fig


def plot_metrics_table(
    metrics: Mapping[str, Mapping[str, float]],
    out_path: str,
    cfg: Mapping[str, Any],
) -> plt.Figure:
    methods = list(metrics.keys())
    cols = ["nll", "mmd", "ksd", "sliced_w2", "score_rmse", "max_abs_score"]
    cell_text = []
    for m in methods:
        row = []
        for c in cols:
            v = metrics[m].get(c, float("nan"))
            row.append("nan" if not np.isfinite(v) else f"{v:.4g}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(12.0, 0.7 + 0.45 * len(methods)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=methods,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    ax.set_title("Sampler summary metrics", pad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(cfg["plot"]["fig_dpi"]), bbox_inches="tight")
    return fig


# =============================================================================
# Experiment runner
# =============================================================================

def write_metrics_csv(path: str, metrics: Mapping[str, Mapping[str, float]]) -> None:
    all_keys = sorted({k for md in metrics.values() for k in md.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + all_keys)
        writer.writeheader()
        for method, md in metrics.items():
            row = {"method": method}
            row.update(md)
            writer.writerow(row)


def write_diagnostics_csv(path: str, diagnostics: Mapping[str, Any]) -> None:
    by_t = diagnostics["by_t"]
    if not by_t:
        return
    keys = list(by_t[0].keys())
    method_risk = diagnostics["method_score_mse_by_t"]
    methods = list(method_risk.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys + [f"risk_{m}" for m in methods])
        writer.writeheader()
        for i, row in enumerate(by_t):
            out = dict(row)
            for m in methods:
                out[f"risk_{m}"] = method_risk[m][i]
            writer.writerow(out)


@torch.no_grad()
def run_experiment(config: Mapping[str, Any], output_dir: str) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    cfg = copy.deepcopy(config)

    device = make_device(cfg)
    dtype = make_dtype(cfg)
    torch.set_default_dtype(dtype)
    seed = int(cfg.get("seed", 42))
    set_seed(seed, device)

    print(f"[sandbox] device={device}, dtype={dtype}, seed={seed}")
    print(f"[sandbox] output={output_dir}")

    gmm = build_gmm_from_config(cfg, device=device, dtype=dtype)
    print(f"[sandbox] target={gmm.name}, K={gmm.K}, d={gmm.d}")

    rcfg = cfg["reference"]
    lmin = float(rcfg.get("lmin", 1e-4))
    lmax = float(rcfg.get("lmax", 1e6))
    n_ref = int(rcfg.get("n_ref", 3000))

    xr = gmm.sample(n_ref)
    precomp = precompute_reference_geometry(gmm, xr, lmin=lmin, lmax=lmax)

    scfg = cfg["sampler"]
    methods = list(scfg.get("methods", EXAMPLE_CONFIG["sampler"]["methods"]))
    n_samples = int(scfg.get("n_samples", 1200))
    n_steps = int(scfg.get("n_steps", 90))
    t_max = float(scfg.get("t_max", 3.0))
    t_min = float(scfg.get("t_min", 0.015))
    score_batch_size = int(scfg.get("score_batch_size", 256))

    gt = gmm.sample(int(cfg["metrics"]["n_ground_truth"]))

    samples: OrderedDict[str, torch.Tensor] = OrderedDict()
    metrics: OrderedDict[str, Dict[str, float]] = OrderedDict()

    for method in methods:
        print(f"[sandbox] sampling {method}...")
        t0 = time.time()
        score_fn = make_score_fn(
            method,
            gmm,
            xr,
            precomp,
            lmin=lmin,
            lmax=lmax,
            batch_size=score_batch_size,
        )
        X, info = reverse_ou_heun_sde(
            score_fn,
            n=n_samples,
            d=gmm.d,
            device=device,
            dtype=dtype,
            n_steps=n_steps,
            t_max=t_max,
            t_min=t_min,
        )
        elapsed = time.time() - t0
        samples[method] = X
        md = evaluate_samples(X, gt, gmm, cfg)
        md["score_rmse"] = score_rmse_forward_process(
            score_fn,
            gmm,
            n_eval=int(cfg["metrics"]["score_rmse_n_eval"]),
            t_grid=cfg["metrics"]["score_rmse_t_grid"],
            batch_size=score_batch_size,
        )
        md["runtime_sec"] = elapsed
        md["failed"] = float(info["failed"])
        md["max_abs_score"] = float(info["max_abs_score"])
        metrics[method] = md
        print(
            f"  {method:10s} nll={md['nll']:.4g} mmd={md['mmd']:.4g} "
            f"ksd={md['ksd']:.4g} sw2={md['sliced_w2']:.4g} "
            f"score_rmse={md['score_rmse']:.4g} time={elapsed:.1f}s"
        )

    print("[sandbox] computing diagnostics...")
    diagnostics = compute_diagnostics(
        gmm,
        xr,
        precomp,
        methods,
        cfg,
        lmin=lmin,
        lmax=lmax,
    )
    spectrum = compute_attenuation_spectrum(gmm, xr, precomp, cfg)

    # Save tensors useful for postmortem, but keep file modest.
    torch.save(
        {
            "gt": gt.detach().cpu(),
            "samples": {m: X.detach().cpu() for m, X in samples.items()},
            "mus": gmm.mus.detach().cpu(),
            "covs": gmm.covs.detach().cpu(),
            "weights": gmm.weights.detach().cpu(),
        },
        os.path.join(output_dir, "samples.pt"),
    )

    # Logs.
    write_json(os.path.join(output_dir, "config_used.json"), cfg)
    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    write_json(os.path.join(output_dir, "diagnostics.json"), diagnostics)
    write_json(os.path.join(output_dir, "attenuation_spectrum.json"), spectrum)
    write_metrics_csv(os.path.join(output_dir, "metrics.csv"), metrics)
    write_diagnostics_csv(os.path.join(output_dir, "diagnostics_by_t.csv"), diagnostics)

    # Plots.
    figs: List[plt.Figure] = []
    plot_paths = {
        "samples": os.path.join(output_dir, "samples_grid.png"),
        "risk": os.path.join(output_dir, "local_score_risk_by_time.png"),
        "diagnostics": os.path.join(output_dir, "performance_vs_diagnostics.png"),
        "op_leakage": os.path.join(output_dir, "op_leakage_vs_risk.png"),
        "spectrum": os.path.join(output_dir, "attenuation_spectrum.png"),
        "metrics_table": os.path.join(output_dir, "metrics_table.png"),
    }
    if gmm.d >= 2:
        figs.append(plot_samples_grid(samples, gt, gmm, cfg, plot_paths["samples"]))
    figs.append(plot_method_risk_by_time(diagnostics, methods, plot_paths["risk"], cfg))
    figs.append(plot_performance_vs_diagnostics(diagnostics, plot_paths["diagnostics"], cfg))
    figs.append(plot_op_leakage_vs_risk(diagnostics, plot_paths["op_leakage"], cfg))
    figs.append(plot_attenuation_spectrum(spectrum, plot_paths["spectrum"], cfg))
    figs.append(plot_metrics_table(metrics, plot_paths["metrics_table"], cfg))

    dashboard_path = os.path.join(output_dir, "summary_dashboard.pdf")
    with PdfPages(dashboard_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    for fig in figs:
        plt.close(fig)

    print(f"[sandbox] saved dashboard: {dashboard_path}")

    return {
        "config": cfg,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "spectrum": spectrum,
        "output_dir": output_dir,
    }


# =============================================================================
# CLI
# =============================================================================

def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(EXAMPLE_CONFIG)
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    return deep_update(cfg, user_cfg)


def make_quick_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    q = copy.deepcopy(cfg)
    # Small enough to be a genuine smoke test on CPU while still exercising every method.
    q["reference"]["n_ref"] = min(int(q["reference"]["n_ref"]), 300)
    q["sampler"]["n_samples"] = min(int(q["sampler"]["n_samples"]), 120)
    q["sampler"]["n_steps"] = min(int(q["sampler"]["n_steps"]), 18)
    q["sampler"]["score_batch_size"] = min(int(q["sampler"]["score_batch_size"]), 64)
    q["diagnostics"]["n_states"] = min(int(q["diagnostics"]["n_states"]), 80)
    q["diagnostics"]["batch_size"] = min(int(q["diagnostics"].get("batch_size", 80)), 80)
    q["diagnostics"]["t_grid"] = [0.03, 0.12, 0.5, 1.5]
    q["metrics"]["n_ground_truth"] = min(int(q["metrics"]["n_ground_truth"]), 500)
    q["metrics"]["mmd_max_points"] = min(int(q["metrics"]["mmd_max_points"]), 250)
    q["metrics"]["ksd_max_points"] = min(int(q["metrics"]["ksd_max_points"]), 180)
    q["metrics"]["sliced_wasserstein_max_points"] = min(int(q["metrics"]["sliced_wasserstein_max_points"]), 250)
    q["metrics"]["score_rmse_n_eval"] = min(int(q["metrics"]["score_rmse_n_eval"]), 160)
    q["metrics"]["score_rmse_t_grid"] = [0.03, 0.12, 0.5]
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Configurable GMM sandbox for CE-HLSI diagnostics.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override.")
    parser.add_argument("--output", type=str, default=None, help="Output directory.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke-test version.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = make_quick_config(cfg)

    if args.output is None:
        name = cfg.get("name", "gmm_sandbox")
        out = os.path.join("outputs", str(name))
    else:
        out = args.output

    run_experiment(cfg, out)


if __name__ == "__main__":
    main()
