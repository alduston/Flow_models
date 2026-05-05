#!/usr/bin/env python3
"""
Robustness of SNIS score estimators to misspecified reference samples on a 2D GMM.

Implements SNIS versions of the main estimators:
  - Tweedie
  - TSI
  - Blended / Blend (coordinatewise variance-optimal blend)
  - OP-Blend (full operator-valued empirical variance-optimal blend)
  - HLSI (componentwise resolvent gate)
  - CE-HLSI (empirical conditional-expectation gate)
  - Hybrid-CE-HLSI (perturbed-reference score aggregate with clean-reference CE gate)
  - DRC-CE-HLSI (density-ratio-corrected CE-HLSI with probability-flow logq updates)
  - optional ORACLE sampler

Experiment:
  1. Draw clean references X_i ~ p_0 from a known 2D GMM.
  2. Perturb anchors with --corruption_mode:
       heat: X_i^eps = X_i + sigma * N(0,I)
       ou:   X_i^eps = exp(-sigma) X_i + sqrt(1-exp(-2 sigma)) N(0,I)
  3. Query exact score and exact observed information H=-∇²log p at those anchors.
  4. Build SNIS estimators using the perturbed anchors.
     Hybrid-CE-HLSI additionally uses the paired clean anchors to compute only
     the CE-HLSI gate, while keeping perturbed-anchor score signals.
     DRC-CE-HLSI uses global log reference weights rho_i updated by analytic
     CE-HLSI probability-flow likelihoods. Method names may be bootstrapped
     chains: Final_Previous means run Previous first, then use its samples as
     references for Final.
  5. Sample with a stochastic predictor-corrector reverse OU sampler.
  6. Measure generated samples using NLL / KSD / MMD / SW2 and the time-integrated
     score RMSE (Fisher-type score divergence) against the exact OU-diffused GMM score.
  7. Save per-sigma histogram panels, a meta metric array, and a meta histogram
     array over all perturbation levels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    outdir: str = "results/ce_hlsi_ref_misspec"
    seed: int = 123
    device: str = "cpu"
    dtype: str = "float64"

    # Sizes
    n_ref: int = 2500
    n_samples: int = 1500
    n_truth: int = 4000
    metrics_max_n: int = 1500
    metrics: Tuple[str, ...] = ("nll", "ksd", "mmd", "sw2", "kl", "kl_rev", "fisher_rmse")
    n_trials: int = 1
    # If true, all sample-vs-target metrics use one fixed unperturbed target
    # evaluation pool and fixed SW2/Fisher probes for the entire perturbation sweep.
    fixed_metric_pool: bool = True

    # Sweep
    perturb_sigmas: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
    # heat/additive: x + sigma*z.  ou: exp(-sigma)*x + sqrt(1-exp(-2*sigma))*z.
    corruption_mode: str = "heat"

    # Reverse OU PC sampler
    t_start: float = 2.0
    t_end: float = 0.005
    n_steps: int = 100
    n_corrector: int = 1
    corrector_snr: float = 0.08
    corrector_step_max: float = 2.0e-3
    init_mode: str = "normal"  # normal or exact_pt; normal matches gmm_hlsi_sandbox.py
    final_denoise: bool = True
    sample_clip: float = 15.0
    score_clip: float = 200.0

    # Time-integrated Fisher / score divergence metric
    fisher_n_t: int = 24
    fisher_n_per_t: int = 1024
    fisher_t_min: Optional[float] = None  # defaults to t_end
    fisher_t_max: Optional[float] = None  # defaults to t_start
    fisher_time_grid: str = "log"  # log or linear

    # Estimators
    #methods: Tuple[str, ...] = ("Tweedie", "HLSI", "CE-HLSI", "Blended", "OP-Blend")
    methods: Tuple[str, ...] = ("CE-HLSI", "CE-HLSI_CE-HLSI", "DRC-CE-HLSI", "DRC-CE-HLSI_DRC-CE-HLSI")
    include_oracle: bool = True
    curvature_mode: str = "raw"  # raw, psd, abs
    curvature_floor: float = -1.0e6
    curvature_cap: float = 1.0e6
    resolvent_eps: float = 1.0e-8
    gate_clip: Optional[float] = 50.0
    weight_temp: float = 1.0
    eval_chunk: int = 512
    op_blend_reg: float = 1.0e-8
    op_blend_pinv_rtol: float = 1.0e-6
    op_blend_project_gate: bool = False

    # Density-ratio-corrected CE-HLSI / probability-flow likelihoods
    pf_steps: int = 64
    pf_t_start: Optional[float] = None  # default: t_end
    pf_t_end: Optional[float] = None    # default: t_start
    rho_beta: float = 1.0
    rho_clip: Optional[float] = 20.0
    rho_ess_floor: float = 0.02
    rho_batch: int = 512
    drc_disable_final_denoise: bool = True
    pf_div_clip: Optional[float] = 1.0e4

    # Computational hypothesis tests for DRC-CE-HLSI. These are off by
    # default because they intentionally add extra bootstrap trajectories,
    # probability-flow likelihoods, KDE comparisons, and oracle-gate probes.
    run_hypothesis_tests: bool = False
    hypothesis_max_depth: int = 0  # 0 means infer from CE/DRC methods in --methods
    hypothesis_n_eval: int = 512
    hypothesis_n_path_t: int = 8
    hypothesis_path_metric: str = "mmd"  # mmd, sw2, hist_l1, hist_l2
    hypothesis_kde_bandwidth: Optional[float] = None
    hypothesis_oracle_n_ref: int = 5000
    hypothesis_gate_n_t: int = 8
    hypothesis_gate_n_per_t: int = 256

    # Metrics
    mmd_bandwidth: Optional[float] = None
    ksd_bandwidth: Optional[float] = None
    sw2_projections: int = 256

    # Plotting
    grid_lim: float = 5.0
    grid_n: int = 180
    hist_bins: int = 90
    hist_cmap: str = "bright_lava"
    hist_gamma: float = 0.35
    hist_vmax_quantile: float = 0.995
    # Residual panels use histogram_probability(method) - histogram_probability(ORACLE).
    # The vmax quantile avoids a single outlier bin washing out the plot, and
    # residual_intensity > 1 lowers the displayed vmax to brighten residual pixels.
    residual_vmax_quantile: float = 0.995
    residual_intensity: float = 1.8
    hist_contour_alpha: float = 0.16
    hist_contour_lw: float = 0.35
    plot_every_sigma: bool = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_dtype(name: str) -> torch.dtype:
    if name in ("float64", "double"):
        return torch.float64
    if name in ("float32", "single"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def alpha_gamma(t: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(t):
        t = torch.tensor(float(t))
    alpha = torch.exp(-t)
    gamma = 1.0 - torch.exp(-2.0 * t)
    return alpha, gamma


def clamp_norm(x: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
    if max_norm is None or max_norm <= 0:
        return x
    n = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (n + 1.0e-12), max=1.0)
    return x * scale


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)


def median_bandwidth(x: torch.Tensor, y: Optional[torch.Tensor] = None, max_n: int = 1200) -> float:
    z = x if y is None else torch.cat([x, y], dim=0)
    if z.shape[0] > max_n:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_n]
        z = z[idx]
    d2 = pairwise_sq_dists(z, z)
    vals = d2[d2 > 0]
    if vals.numel() == 0:
        return 1.0
    med = torch.median(vals).item()
    return float(math.sqrt(max(med, 1.0e-12)))


def safe_float(x) -> float:
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().item()
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def batch_eye(batch: int, d: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(d, device=device, dtype=dtype).expand(batch, d, d)


def bmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", A, x)


def project_symmetric_gate(G: torch.Tensor, min_eig: float = 0.0, max_eig: float = 1.0) -> torch.Tensor:
    Gs = sym(G)
    eig, U = torch.linalg.eigh(Gs)
    eig = eig.clamp(min=min_eig, max=max_eig)
    return U @ torch.diag_embed(eig) @ U.transpose(-1, -2)


METRIC_ALIASES: Dict[str, str] = {
    "sliced_w2": "sw2",
    "score_rmse": "fisher_rmse",
    "fisher_mse": "fisher_rmse",
    "kl_gt_to_model_2d": "kl",
    "kl_model_to_gt_2d": "kl_rev",
}

AVAILABLE_METRICS: Tuple[str, ...] = ("nll", "ksd", "mmd", "sw2", "kl", "kl_rev", "fisher_rmse")
SAMPLE_METRICS: Tuple[str, ...] = ("nll", "ksd", "mmd", "sw2", "kl", "kl_rev")
FISHER_METRICS: Tuple[str, ...] = ("fisher_rmse",)


def canonicalize_metrics(metrics: Tuple[str, ...] | List[str]) -> Tuple[str, ...]:
    if metrics is None or len(metrics) == 0:
        metrics = list(AVAILABLE_METRICS)
    out: List[str] = []
    for raw in metrics:
        if raw is None:
            continue
        name = str(raw).strip().lower()
        if not name:
            continue
        if name == "all":
            for m in AVAILABLE_METRICS:
                if m not in out:
                    out.append(m)
            continue
        name = METRIC_ALIASES.get(name, name)
        if name not in AVAILABLE_METRICS:
            raise ValueError(f"Unknown metric '{raw}'. Available metrics: {', '.join(AVAILABLE_METRICS)}")
        if name not in out:
            out.append(name)
    return tuple(out)


def metric_enabled(cfg: ExperimentConfig, name: str) -> bool:
    return name in cfg.metrics


def any_metric_enabled(cfg: ExperimentConfig, names: Tuple[str, ...] | List[str]) -> bool:
    return any(name in cfg.metrics for name in names)


# -----------------------------------------------------------------------------
# Target 2D GMM
# -----------------------------------------------------------------------------


class GMM2D:
    def __init__(self, weights: torch.Tensor, means: torch.Tensor, covs: torch.Tensor, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.weights = weights.to(device=device, dtype=dtype)
        self.weights = self.weights / self.weights.sum()
        self.log_weights = torch.log(self.weights)
        self.means = means.to(device=device, dtype=dtype)
        self.covs = covs.to(device=device, dtype=dtype)
        self.K, self.d = self.means.shape
        assert self.d == 2
        self.precs = torch.linalg.inv(self.covs)
        self.logdets = torch.linalg.slogdet(self.covs).logabsdet
        self.chols = torch.linalg.cholesky(self.covs)

    @staticmethod
    def default(device: torch.device, dtype: torch.dtype) -> "GMM2D":
        weights = torch.tensor([0.18, 0.22, 0.20, 0.17, 0.23], dtype=dtype)
        means = torch.tensor(
            [
                [-2.30, -1.70],
                [2.10, -1.55],
                [-1.90, 1.65],
                [2.05, 1.85],
                [0.00, 0.15],
            ],
            dtype=dtype,
        )
        eigs = [
            (0.08, 0.55, 0.65),
            (0.11, 0.42, -0.55),
            (0.07, 0.36, -0.95),
            (0.10, 0.60, 0.85),
            (0.23, 0.32, 0.20),
        ]
        covs = []
        for l1, l2, theta in eigs:
            c, s = math.cos(theta), math.sin(theta)
            R = torch.tensor([[c, -s], [s, c]], dtype=dtype)
            D = torch.diag(torch.tensor([l1, l2], dtype=dtype))
            covs.append(R @ D @ R.T)
        covs = torch.stack(covs, dim=0)
        return GMM2D(weights, means, covs, device=device, dtype=dtype)

    def marginal_params(self, t: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        alpha, gamma = alpha_gamma(t)
        means_t = alpha * self.means
        eye = torch.eye(self.d, device=self.device, dtype=self.dtype)[None]
        covs_t = alpha * alpha * self.covs + gamma * eye
        precs_t = torch.linalg.inv(covs_t)
        logdets_t = torch.linalg.slogdet(covs_t).logabsdet
        return means_t, covs_t, precs_t, logdets_t

    def component_log_probs(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        means_t, _, precs_t, logdets_t = self.marginal_params(t)
        diff = x[:, None, :] - means_t[None, :, :]
        maha = torch.einsum("bki,kij,bkj->bk", diff, precs_t, diff)
        return self.log_weights[None, :] - 0.5 * (self.d * math.log(2.0 * math.pi) + logdets_t[None, :] + maha)

    def responsibilities(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        return torch.softmax(self.component_log_probs(x, t=t), dim=1)

    def log_prob(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        return torch.logsumexp(self.component_log_probs(x, t=t), dim=1)

    def score(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        means_t, _, precs_t, _ = self.marginal_params(t)
        r = self.responsibilities(x, t=t)
        diff = x[:, None, :] - means_t[None, :, :]
        comp_scores = -torch.einsum("kij,bkj->bki", precs_t, diff)
        return torch.sum(r[:, :, None] * comp_scores, dim=1)

    def observed_information(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        means_t, _, precs_t, _ = self.marginal_params(t)
        r = self.responsibilities(x, t=t)
        diff = x[:, None, :] - means_t[None, :, :]
        comp_scores = -torch.einsum("kij,bkj->bki", precs_t, diff)
        score = torch.sum(r[:, :, None] * comp_scores, dim=1)
        term = precs_t[None, :, :, :] - torch.einsum("bki,bkj->bkij", comp_scores, comp_scores)
        H = torch.sum(r[:, :, None, None] * term, dim=1) + torch.einsum("bi,bj->bij", score, score)
        return 0.5 * (H + H.transpose(-1, -2))

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        eps = torch.randn((n, self.d), device=self.device, dtype=self.dtype, generator=generator)
        return self.means[idx] + torch.einsum("bij,bj->bi", self.chols[idx], eps)

    def sample_pt(self, n: int, t: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x0 = self.sample(n, generator=generator)
        alpha, gamma = alpha_gamma(torch.tensor(t, device=self.device, dtype=self.dtype))
        eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        return alpha * x0 + torch.sqrt(gamma) * eps


# -----------------------------------------------------------------------------
# SNIS estimators
# -----------------------------------------------------------------------------


def process_curvature(H: torch.Tensor, mode: str, floor: float, cap: float) -> torch.Tensor:
    H = 0.5 * (H + H.transpose(-1, -2))
    if mode == "raw":
        return H
    evals, evecs = torch.linalg.eigh(H)
    if mode == "psd":
        evals = torch.clamp(evals, min=floor, max=cap)
    elif mode == "abs":
        evals = torch.clamp(torch.abs(evals), min=floor, max=cap)
    else:
        raise ValueError(f"Unknown curvature_mode={mode}")
    return evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)


def resolvent_gate(P: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, eps: float, gate_clip: Optional[float]) -> torch.Tensor:
    d = P.shape[-1]
    I = torch.eye(d, device=P.device, dtype=P.dtype)
    A = alpha * alpha * I + gamma * P
    A = 0.5 * (A + A.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(A)
    signs = torch.where(evals >= 0, torch.ones_like(evals), -torch.ones_like(evals))
    evals_safe = torch.where(torch.abs(evals) < eps, signs * eps, evals)
    gvals = (alpha * alpha) / evals_safe
    if gate_clip is not None and gate_clip > 0:
        gvals = torch.clamp(gvals, min=-gate_clip, max=gate_clip)
    return evecs @ torch.diag_embed(gvals) @ evecs.transpose(-1, -2)


class SNISScoreBank:
    def __init__(
        self,
        target: GMM2D,
        anchors: torch.Tensor,
        curvature_mode: str,
        curvature_floor: float,
        curvature_cap: float,
        resolvent_eps: float,
        gate_clip: Optional[float],
        weight_temp: float,
        eval_chunk: int,
        op_blend_reg: float = 1.0e-8,
        op_blend_pinv_rtol: float = 1.0e-6,
        op_blend_project_gate: bool = False,
        gate_anchors: Optional[torch.Tensor] = None,
        log_ref_weights: Optional[torch.Tensor] = None,
    ):
        self.target = target

        # Main bank: perturbed anchors in the misspecification experiment.
        # These anchors define the SNIS weights/signals used for Tweedie, TSI,
        # Blend, OP-Blend, HLSI, CE-HLSI, and the score aggregate part of
        # Hybrid-CE-HLSI.
        self.x = anchors.detach()
        self.N, self.d = self.x.shape
        self.device = self.x.device
        self.dtype = self.x.dtype
        self.score0 = target.score(self.x, t=0.0).detach()
        self.H_raw = target.observed_information(self.x, t=0.0).detach()
        self.P = process_curvature(self.H_raw, curvature_mode, curvature_floor, curvature_cap).detach()
        if log_ref_weights is None:
            self.log_ref_weights = torch.zeros((self.N,), device=self.device, dtype=self.dtype)
        else:
            self.log_ref_weights = log_ref_weights.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
            if self.log_ref_weights.shape[0] != self.N:
                raise ValueError(f"log_ref_weights must have length {self.N}, got {self.log_ref_weights.shape[0]}")
            self.log_ref_weights = torch.nan_to_num(self.log_ref_weights, nan=0.0, posinf=0.0, neginf=0.0)

        # Gate bank: optional clean anchors paired with the perturbed anchors.
        # Hybrid-CE-HLSI uses this bank only for the CE gate: it computes
        # clean-reference OU weights and averages clean-reference curvature, while
        # still applying that gate to the perturbed-reference score aggregate.
        self.gate_x = self.x if gate_anchors is None else gate_anchors.detach().to(device=self.device, dtype=self.dtype)
        if self.gate_x.shape != self.x.shape:
            raise ValueError(f"gate_anchors must have shape {tuple(self.x.shape)}, got {tuple(self.gate_x.shape)}")
        if gate_anchors is None:
            self.H_gate_raw = self.H_raw
            self.P_gate = self.P
        else:
            self.H_gate_raw = target.observed_information(self.gate_x, t=0.0).detach()
            self.P_gate = process_curvature(self.H_gate_raw, curvature_mode, curvature_floor, curvature_cap).detach()

        self.resolvent_eps = resolvent_eps
        self.gate_clip = gate_clip
        self.weight_temp = weight_temp
        self.eval_chunk = eval_chunk
        self.op_blend_reg = op_blend_reg
        self.op_blend_pinv_rtol = op_blend_pinv_rtol
        self.op_blend_project_gate = op_blend_project_gate

    def _weights_for_anchors(
        self,
        y: torch.Tensor,
        t: float,
        anchors: torch.Tensor,
        log_ref_weights: Optional[torch.Tensor] = None,
    ):
        t_tensor = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        alpha, gamma = alpha_gamma(t_tensor)
        gamma = torch.clamp(gamma, min=torch.as_tensor(1.0e-12, device=self.device, dtype=self.dtype))
        diff = y[:, None, :] - alpha * anchors[None, :, :]
        logw = -0.5 * torch.sum(diff * diff, dim=-1) / gamma
        if self.weight_temp != 1.0:
            logw = logw / self.weight_temp
        if log_ref_weights is not None:
            logw = logw + log_ref_weights[None, :]
        logw = logw - torch.max(logw, dim=1, keepdim=True).values
        w = torch.exp(logw)
        w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1.0e-300)
        return w, alpha, gamma

    def _weights_and_signals(self, y: torch.Tensor, t: float):
        w, alpha, gamma = self._weights_for_anchors(y, t, self.x, self.log_ref_weights)
        b = (alpha * self.x[None, :, :] - y[:, None, :]) / gamma
        c = self.score0[None, :, :] / alpha
        return w, b, c, alpha, gamma

    def estimate_chunk(self, y: torch.Tensor, t: float, method: str) -> torch.Tensor:
        method_key = str(method).strip().replace("_", "-").lower()
        w, b, c, alpha, gamma = self._weights_and_signals(y, t)
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)

        if method_key == "tweedie":
            return bbar
        if method_key == "tsi":
            return cbar
        if method_key in {"blend", "blended"}:
            # Match gmm_hlsi_sandbox.py: coordinatewise variance-optimal blend
            # between the per-anchor TSI signal c and Tweedie signal b.
            Ac = c - cbar[:, None, :]
            Bc = b - bbar[:, None, :]
            va = torch.sum(w[:, :, None] * Ac.square(), dim=1).clamp(min=1.0e-30)
            vb = torch.sum(w[:, :, None] * Bc.square(), dim=1).clamp(min=1.0e-30)
            cab = torch.sum(w[:, :, None] * Ac * Bc, dim=1)
            den = (va + vb - 2.0 * cab).clamp(min=1.0e-20)
            g = ((va - cab) / den).clamp(0.0, 1.0)
            return cbar + g * (bbar - cbar)
        if method_key == "op-blend":
            # Full matrix empirical variance-optimal gate, exactly mirroring
            # op_blend_gate / est_op_blended in gmm_hlsi_sandbox.py.
            tsi = c
            twd = b
            am = cbar
            bm = bbar
            D = twd - tsi
            Dm = bm - am
            Ac = tsi - am[:, None, :]
            Dc = D - Dm[:, None, :]
            C_AD = torch.einsum("bm,bmi,bmj->bij", w, Ac, Dc)
            C_DD = torch.einsum("bm,bmi,bmj->bij", w, Dc, Dc)
            C_DD = sym(C_DD)
            Bsz, d = y.shape
            I = batch_eye(Bsz, d, device=y.device, dtype=y.dtype)
            scale = C_DD.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1.0)
            C_DD_solve = C_DD + (self.op_blend_reg * scale).view(Bsz, 1, 1) * I
            G = -torch.matmul(C_AD, torch.linalg.pinv(C_DD_solve, rtol=self.op_blend_pinv_rtol))
            if self.op_blend_project_gate:
                G = project_symmetric_gate(G, 0.0, 1.0)
            return am + bmv(G, bm - am)
        if method_key == "hlsi":
            G_i = resolvent_gate(self.P, alpha, gamma, self.resolvent_eps, self.gate_clip)
            corr = torch.einsum("nij,bnj->bni", G_i, (c - b))
            h_i = b + corr
            return torch.sum(w[:, :, None] * h_i, dim=1)
        if method_key in {"ce-hlsi", "drc-ce-hlsi"}:
            Pbar = torch.sum(w[:, :, None, None] * self.P[None, :, :, :], dim=1)
            Gbar = resolvent_gate(Pbar, alpha, gamma, self.resolvent_eps, self.gate_clip)
            return bbar + torch.einsum("bij,bj->bi", Gbar, (cbar - bbar))
        if method_key in {"hybrid-ce-hlsi", "ce-hlsi-hybrid", "hybrid-ce"}:
            # Score aggregate uses perturbed-reference signals (bbar, cbar).
            # The CE gate itself is computed from the paired clean gate bank.
            w_gate, _, _ = self._weights_for_anchors(y, t, self.gate_x, None)
            Pbar_gate = torch.sum(w_gate[:, :, None, None] * self.P_gate[None, :, :, :], dim=1)
            Gbar_gate = resolvent_gate(Pbar_gate, alpha, gamma, self.resolvent_eps, self.gate_clip)
            return bbar + torch.einsum("bij,bj->bi", Gbar_gate, (cbar - bbar))
        raise ValueError(f"Unknown method={method}")

    def ce_hlsi_score_and_divergence_chunk(self, y: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return corrected CE-HLSI score and analytic divergence for one chunk.

        This implements
            div s = tr(J_b) + tr(G J_r)
                    - (gamma / alpha^2) sum_a e_a^T G (partial_a Pbar) G r,
        where all averages use the bank's global log_ref_weights. The same
        corrected weights are therefore used for bbar, cbar, and Pbar.
        """
        w, b, c, alpha, gamma = self._weights_and_signals(y, t)
        B, N, d = b.shape
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)
        Pbar = torch.sum(w[:, :, None, None] * self.P[None, :, :, :], dim=1)
        G = resolvent_gate(Pbar, alpha, gamma, self.resolvent_eps, self.gate_clip)

        r = cbar - bbar
        score = bbar + torch.einsum("bij,bj->bi", G, r)

        db = b - bbar[:, None, :]
        dc = c - cbar[:, None, :]
        dP = self.P[None, :, :, :] - Pbar[:, None, :, :]
        Cbb = torch.einsum("bn,bni,bnj->bij", w, db, db)
        Ccb = torch.einsum("bn,bni,bnj->bij", w, dc, db)
        I = torch.eye(d, device=y.device, dtype=y.dtype).expand(B, d, d)
        Jb = Cbb - I / gamma
        Jc = Ccb
        Jr = Jc - Jb

        dPdy = torch.einsum("bn,bna,bnuv->bauv", w, db, dP)
        Gr = torch.einsum("bij,bj->bi", G, r)
        gate_trace_term = torch.einsum("bau,bauv,bv->b", G, dPdy, Gr)
        tr_Jb = torch.diagonal(Jb, dim1=-2, dim2=-1).sum(dim=-1)
        tr_GJr = torch.einsum("bij,bji->b", G, Jr)
        div = tr_Jb + tr_GJr - (gamma / torch.clamp(alpha * alpha, min=1.0e-12)) * gate_trace_term
        div = torch.nan_to_num(div, nan=0.0, posinf=1.0e12, neginf=-1.0e12)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score, div

    @torch.no_grad()
    def ce_hlsi_score_and_divergence(self, y: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        scores: List[torch.Tensor] = []
        divs: List[torch.Tensor] = []
        for start in range(0, y.shape[0], self.eval_chunk):
            s, div = self.ce_hlsi_score_and_divergence_chunk(y[start:start + self.eval_chunk], t)
            scores.append(s)
            divs.append(div)
        return torch.cat(scores, dim=0), torch.cat(divs, dim=0)

    def ce_hlsi_gate_chunk(self, y: torch.Tensor, t: float) -> torch.Tensor:
        """Return the corrected CE-HLSI resolvent gate G(y,t) for one chunk."""
        w, _, _, alpha, gamma = self._weights_and_signals(y, t)
        Pbar = torch.sum(w[:, :, None, None] * self.P[None, :, :, :], dim=1)
        G = resolvent_gate(Pbar, alpha, gamma, self.resolvent_eps, self.gate_clip)
        return torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def ce_hlsi_gate(self, y: torch.Tensor, t: float) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for start in range(0, y.shape[0], self.eval_chunk):
            outs.append(self.ce_hlsi_gate_chunk(y[start:start + self.eval_chunk], t))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def estimate(self, y: torch.Tensor, t: float, method: str) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for start in range(0, y.shape[0], self.eval_chunk):
            outs.append(self.estimate_chunk(y[start:start + self.eval_chunk], t, method))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def ess(self, y: torch.Tensor, t: float) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        for start in range(0, y.shape[0], self.eval_chunk):
            w, _, _, _, _ = self._weights_and_signals(y[start:start + self.eval_chunk], t)
            vals.append(1.0 / torch.clamp(torch.sum(w * w, dim=1), min=1.0e-30))
        return torch.cat(vals, dim=0)


# -----------------------------------------------------------------------------
# Reverse OU predictor-corrector sampler
# -----------------------------------------------------------------------------


@torch.no_grad()
def reverse_ou_heun_sde(
    target: GMM2D,
    score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    cfg: ExperimentConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float | str | bool]]:
    """Reverse OU Heun SDE sampler matched to gmm_hlsi_sandbox.py.

    Forward OU is dx=-x dt+sqrt(2)dW.  In decreasing diffusion time,
    reverse drift is y + 2 score_t(y).  We use the same predictor/corrector
    noise as the sandbox Heun implementation and finish with a Tweedie
    denoising map at t_min.
    """
    device = target.device
    dtype = target.dtype
    d = target.d

    if cfg.init_mode == "normal":
        y = torch.randn((cfg.n_samples, d), device=device, dtype=dtype, generator=generator)
    elif cfg.init_mode == "exact_pt":
        y = target.sample_pt(cfg.n_samples, cfg.t_start, generator=generator)
    else:
        raise ValueError("init_mode must be normal or exact_pt")

    ts = torch.linspace(cfg.t_start, cfg.t_end, cfg.n_steps + 1, device=device, dtype=dtype)
    max_abs_score = 0.0
    fail = False
    fail_reason = ""

    for i in range(cfg.n_steps):
        tc = ts[i]
        tn = ts[i + 1]
        h = tc - tn

        s1 = clamp_norm(score_fn(y, float(tc.item())), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(s1.abs().max()))
        if not torch.isfinite(s1).all():
            fail, fail_reason = True, "nonfinite score at predictor"
            break

        drift1 = y + 2.0 * s1
        noise = torch.sqrt(2.0 * h) * torch.randn(y.shape, device=device, dtype=dtype, generator=generator)
        yh = y + h * drift1 + noise

        s2 = clamp_norm(score_fn(yh, float(tn.item())), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(s2.abs().max()))
        if not torch.isfinite(s2).all():
            fail, fail_reason = True, "nonfinite score at corrector"
            break

        drift2 = yh + 2.0 * s2
        y = y + 0.5 * h * (drift1 + drift2) + noise

        if cfg.sample_clip and cfg.sample_clip > 0:
            y = torch.clamp(y, min=-cfg.sample_clip, max=cfg.sample_clip)
        if not torch.isfinite(y).all():
            fail, fail_reason = True, "nonfinite state"
            break

    if not fail and cfg.final_denoise:
        tf = torch.tensor(cfg.t_end, device=device, dtype=dtype)
        sf = clamp_norm(score_fn(y, cfg.t_end), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(sf.abs().max()))
        if not torch.isfinite(sf).all():
            fail, fail_reason = True, "nonfinite final score"
        else:
            y = (y + (1.0 - torch.exp(-2.0 * tf)) * sf) / torch.exp(-tf)
            if cfg.sample_clip and cfg.sample_clip > 0:
                y = torch.clamp(y, min=-cfg.sample_clip, max=cfg.sample_clip)

    info = {"failed": bool(fail), "fail_reason": fail_reason, "max_abs_score": float(max_abs_score)}
    return y.detach(), info



@torch.no_grad()
def reverse_ou_heun_sde_with_snapshots(
    target: GMM2D,
    score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    cfg: ExperimentConfig,
    generator: torch.Generator,
    snapshot_times: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[float, torch.Tensor], Dict[str, float | str | bool]]:
    """Reverse OU Heun sampler that also records q_rev,t snapshots.

    The hypothesis-test path diagnostic compares these reverse-time marginals
    against forward OU noising of the endpoint samples.  We normally call this
    with final_denoise=False so the endpoint is the t_end state used by the
    path-level continuity check.
    """
    device = target.device
    dtype = target.dtype
    d = target.d
    snap = snapshot_times.detach().to(device=device, dtype=dtype).flatten()
    snap = torch.clamp(snap, min=float(cfg.t_end), max=float(cfg.t_start))
    snap_vals = sorted({float(v) for v in snap.detach().cpu().tolist()}, reverse=True)
    snapshots: Dict[float, torch.Tensor] = {}

    if cfg.init_mode == "normal":
        y = torch.randn((cfg.n_samples, d), device=device, dtype=dtype, generator=generator)
    elif cfg.init_mode == "exact_pt":
        y = target.sample_pt(cfg.n_samples, cfg.t_start, generator=generator)
    else:
        raise ValueError("init_mode must be normal or exact_pt")

    ts = torch.linspace(cfg.t_start, cfg.t_end, cfg.n_steps + 1, device=device, dtype=dtype)
    max_abs_score = 0.0
    fail = False
    fail_reason = ""

    # Snapshot at exactly t_start before any reverse update.
    for t_snap in list(snap_vals):
        if abs(t_snap - float(cfg.t_start)) <= 1.0e-10:
            snapshots[t_snap] = y.detach().clone()

    for i in range(cfg.n_steps):
        tc = ts[i]
        tn = ts[i + 1]
        tc_f = float(tc.item())
        tn_f = float(tn.item())
        h = tc - tn
        y_old = y.detach().clone()

        s1 = clamp_norm(score_fn(y, tc_f), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(s1.abs().max()))
        if not torch.isfinite(s1).all():
            fail, fail_reason = True, "nonfinite score at predictor"
            break

        drift1 = y + 2.0 * s1
        noise = torch.sqrt(2.0 * h) * torch.randn(y.shape, device=device, dtype=dtype, generator=generator)
        yh = y + h * drift1 + noise

        s2 = clamp_norm(score_fn(yh, tn_f), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(s2.abs().max()))
        if not torch.isfinite(s2).all():
            fail, fail_reason = True, "nonfinite score at corrector"
            break

        drift2 = yh + 2.0 * s2
        y = y + 0.5 * h * (drift1 + drift2) + noise

        if cfg.sample_clip and cfg.sample_clip > 0:
            y = torch.clamp(y, min=-cfg.sample_clip, max=cfg.sample_clip)
        if not torch.isfinite(y).all():
            fail, fail_reason = True, "nonfinite state"
            break

        # Record any requested time crossed by this interval.  Linear state
        # interpolation is only a diagnostic convenience; for exact snapshots,
        # choose --hypothesis_n_path_t so times align with the sampler grid.
        for t_snap in snap_vals:
            if t_snap in snapshots:
                continue
            if tc_f >= t_snap >= tn_f:
                denom = max(tc_f - tn_f, 1.0e-12)
                frac = (tc_f - t_snap) / denom
                snapshots[t_snap] = ((1.0 - frac) * y_old + frac * y).detach().clone()

    if not fail and cfg.final_denoise:
        tf = torch.tensor(cfg.t_end, device=device, dtype=dtype)
        sf = clamp_norm(score_fn(y, cfg.t_end), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(sf.abs().max()))
        if not torch.isfinite(sf).all():
            fail, fail_reason = True, "nonfinite final score"
        else:
            y = (y + (1.0 - torch.exp(-2.0 * tf)) * sf) / torch.exp(-tf)
            if cfg.sample_clip and cfg.sample_clip > 0:
                y = torch.clamp(y, min=-cfg.sample_clip, max=cfg.sample_clip)

    # Fill any missing low-time snapshots with the final state after a failure or
    # floating-point crossing miss, rather than dropping rows from the diagnostic.
    for t_snap in snap_vals:
        if t_snap not in snapshots:
            snapshots[t_snap] = y.detach().clone()

    info = {"failed": bool(fail), "fail_reason": fail_reason, "max_abs_score": float(max_abs_score)}
    return y.detach(), snapshots, info

# Backward-compatible name; now uses the sandbox-matched Heun sampler.
def pc_reverse_sampler(target: GMM2D, score_fn: Callable[[torch.Tensor, float], torch.Tensor], cfg: ExperimentConfig, generator: torch.Generator) -> torch.Tensor:
    y, _ = reverse_ou_heun_sde(target, score_fn, cfg, generator)
    return y


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


@torch.no_grad()
def nll_metric(target: GMM2D, x: torch.Tensor) -> float:
    return safe_float((-target.log_prob(x, t=0.0)).mean())


@torch.no_grad()
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: Optional[float] = None) -> float:
    """Sandbox-matched biased multiscale RBF MMD^2 diagnostic.

    This intentionally matches gmm_hlsi_sandbox.py more closely than the older
    square-root MMD used by early versions of this robustness script.
    """
    if bandwidth is None:
        bandwidth = median_bandwidth(x, y)
    base_h2 = float(bandwidth) ** 2
    scales = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], device=x.device, dtype=x.dtype)
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)
    vals = []
    for scale in scales:
        h2 = torch.clamp(base_h2 * scale, min=torch.as_tensor(1.0e-12, device=x.device, dtype=x.dtype))
        Kxx = torch.exp(-dxx / (2.0 * h2))
        Kyy = torch.exp(-dyy / (2.0 * h2))
        Kxy = torch.exp(-dxy / (2.0 * h2))
        vals.append(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return safe_float(torch.stack(vals).mean().clamp(min=0.0))


@torch.no_grad()
def sliced_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    n_proj: int = 256,
    generator: Optional[torch.Generator] = None,
    dirs: Optional[torch.Tensor] = None,
) -> float:
    """Sandbox-matched sliced Wasserstein-2: mean squared 1D W2, not sqrt."""
    d = x.shape[1]
    if dirs is None:
        dirs = torch.randn((n_proj, d), device=x.device, dtype=x.dtype, generator=generator)
        dirs = dirs / torch.clamp(torch.linalg.norm(dirs, dim=1, keepdim=True), min=1.0e-12)
    else:
        dirs = dirs.to(device=x.device, dtype=x.dtype)
    px = x @ dirs.T
    py = y @ dirs.T
    n = min(px.shape[0], py.shape[0])
    px = px[:n]
    py = py[:n]
    sx = torch.sort(px, dim=0).values
    sy = torch.sort(py, dim=0).values
    return safe_float(((sx - sy) ** 2).mean())


@torch.no_grad()
def ksd_rbf(target: GMM2D, x: torch.Tensor, bandwidth: Optional[float] = None) -> float:
    """Sandbox-matched biased RBF KSD with diagonal included."""
    n, d = x.shape
    if bandwidth is None:
        bandwidth = median_bandwidth(x)
    h2 = float(bandwidth) ** 2
    s = target.score(x, t=0.0)
    diffs = x[:, None, :] - x[None, :, :]
    d2 = torch.sum(diffs * diffs, dim=-1)
    K = torch.exp(-d2 / (2.0 * h2))
    ss = s @ s.T
    sx_grad_y = torch.einsum("id,ijd->ij", s, diffs) / h2 * K
    sy_grad_x = -torch.einsum("jd,ijd->ij", s, diffs) / h2 * K
    trace = (d / h2 - d2 / (h2 * h2)) * K
    kstein = ss * K + sx_grad_y + sy_grad_x + trace
    return safe_float(kstein.mean().clamp(min=0.0).sqrt())


def hist_kl_2d(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    bins: int = 150,
    smoothing: float = 1.0e-12,
) -> Tuple[float, float]:
    """Sandbox-matched approximate KL(P_X||P_Y), KL(P_Y||P_X)."""
    Xn = as_numpy(X[:, :2])
    Yn = as_numpy(Y[:, :2])
    lo = np.minimum(np.percentile(Xn, 0.5, axis=0), np.percentile(Yn, 0.5, axis=0))
    hi = np.maximum(np.percentile(Xn, 99.5, axis=0), np.percentile(Yn, 99.5, axis=0))
    pad = 0.05 * (hi - lo + 1.0e-8)
    rng = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
    Hx, _, _ = np.histogram2d(Xn[:, 0], Xn[:, 1], bins=bins, range=rng, density=False)
    Hy, _, _ = np.histogram2d(Yn[:, 0], Yn[:, 1], bins=bins, range=rng, density=False)
    px = Hx.astype(np.float64) + smoothing
    py = Hy.astype(np.float64) + smoothing
    px = px / px.sum()
    py = py / py.sum()
    return float(np.sum(px * (np.log(px) - np.log(py)))), float(np.sum(py * (np.log(py) - np.log(px))))


@dataclass
class MetricContext:
    """Fixed metric state shared by every perturbation and method.

    This prevents accidental comparisons against perturbed references and removes
    metric noise from changing target pools, changing SW2 projections, or changing
    Fisher probe samples across methods. Expensive metric-specific state is only
    built when that metric is requested through --metrics.
    """
    truth_eval: torch.Tensor
    sw2_dirs: Optional[torch.Tensor]
    ksd_bandwidth: Optional[float]
    mmd_bandwidth: Optional[float]
    fisher_times: torch.Tensor
    fisher_y: List[torch.Tensor]
    fisher_score: List[torch.Tensor]


@torch.no_grad()
def build_metric_context(target: GMM2D, truth_pool: torch.Tensor, cfg: ExperimentConfig, generator: torch.Generator) -> MetricContext:
    n_eval = min(cfg.metrics_max_n, truth_pool.shape[0])
    # Fixed unperturbed target pool, held constant across every perturbation sigma.
    truth_eval = truth_pool[:n_eval].detach()

    dirs: Optional[torch.Tensor] = None
    if metric_enabled(cfg, "sw2"):
        dirs = torch.randn((cfg.sw2_projections, truth_eval.shape[1]), device=truth_eval.device, dtype=truth_eval.dtype, generator=generator)
        dirs = dirs / torch.clamp(torch.linalg.norm(dirs, dim=1, keepdim=True), min=1.0e-12)

    ksd_bw: Optional[float] = None
    if metric_enabled(cfg, "ksd"):
        ksd_bw = float(cfg.ksd_bandwidth) if cfg.ksd_bandwidth is not None else median_bandwidth(truth_eval)

    mmd_bw: Optional[float] = None
    if metric_enabled(cfg, "mmd"):
        mmd_bw = float(cfg.mmd_bandwidth) if cfg.mmd_bandwidth is not None else median_bandwidth(truth_eval)

    fisher_times = torch.empty((0,), device=target.device, dtype=target.dtype)
    fisher_y: List[torch.Tensor] = []
    fisher_score: List[torch.Tensor] = []
    if metric_enabled(cfg, "fisher_rmse"):
        fisher_times = make_fisher_time_grid(cfg, target.device, target.dtype)
        for t_tensor in fisher_times:
            t = float(t_tensor.item())
            y = target.sample_pt(cfg.fisher_n_per_t, t, generator=generator).detach()
            fisher_y.append(y)
            fisher_score.append(target.score(y, t=t).detach())

    return MetricContext(
        truth_eval=truth_eval,
        sw2_dirs=dirs,
        ksd_bandwidth=ksd_bw,
        mmd_bandwidth=mmd_bw,
        fisher_times=fisher_times,
        fisher_y=fisher_y,
        fisher_score=fisher_score,
    )


@torch.no_grad()
def compute_all_metrics(target: GMM2D, samples: torch.Tensor, metric_ctx: MetricContext, cfg: ExperimentConfig) -> Dict[str, float]:
    n = min(cfg.metrics_max_n, samples.shape[0], metric_ctx.truth_eval.shape[0])
    x = samples[:n]
    y = metric_ctx.truth_eval[:n]
    out: Dict[str, float] = {"metric_target_n": float(n)}

    if not torch.isfinite(x).all():
        for metric in SAMPLE_METRICS:
            if metric_enabled(cfg, metric):
                out[metric] = float("nan")
        return out

    if metric_enabled(cfg, "nll"):
        out["nll"] = nll_metric(target, x)
    if metric_enabled(cfg, "ksd"):
        out["ksd"] = ksd_rbf(target, x, bandwidth=metric_ctx.ksd_bandwidth)
    if metric_enabled(cfg, "mmd"):
        out["mmd"] = mmd_rbf(x, y, bandwidth=metric_ctx.mmd_bandwidth)
    if metric_enabled(cfg, "sw2"):
        if metric_ctx.sw2_dirs is None:
            raise RuntimeError("SW2 was requested but fixed projection directions were not initialized.")
        out["sw2"] = sliced_w2(x, y, n_proj=cfg.sw2_projections, dirs=metric_ctx.sw2_dirs)
    if metric_enabled(cfg, "kl") or metric_enabled(cfg, "kl_rev"):
        kl_gt_model, kl_model_gt = hist_kl_2d(y, x, bins=cfg.hist_bins)
        if metric_enabled(cfg, "kl"):
            out["kl"] = kl_gt_model
        if metric_enabled(cfg, "kl_rev"):
            out["kl_rev"] = kl_model_gt
    return out


@torch.no_grad()
def make_fisher_time_grid(cfg: ExperimentConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t_min = cfg.t_end if cfg.fisher_t_min is None else cfg.fisher_t_min
    t_max = cfg.t_start if cfg.fisher_t_max is None else cfg.fisher_t_max
    t_min = max(float(t_min), 1.0e-8)
    t_max = max(float(t_max), t_min)
    if cfg.fisher_n_t <= 1:
        return torch.tensor([t_max], device=device, dtype=dtype)
    if cfg.fisher_time_grid == "log":
        return torch.exp(torch.linspace(math.log(t_min), math.log(t_max), cfg.fisher_n_t, device=device, dtype=dtype))
    if cfg.fisher_time_grid == "linear":
        return torch.linspace(t_min, t_max, cfg.fisher_n_t, device=device, dtype=dtype)
    raise ValueError("fisher_time_grid must be 'log' or 'linear'")


@torch.no_grad()
def integrated_score_fisher_metric(score_fn: Callable[[torch.Tensor, float], torch.Tensor], metric_ctx: MetricContext, cfg: ExperimentConfig) -> Dict[str, float]:
    if not metric_enabled(cfg, "fisher_rmse"):
        return {}
    if cfg.fisher_n_t <= 0 or cfg.fisher_n_per_t <= 0 or len(metric_ctx.fisher_y) == 0:
        return {"fisher_rmse": float("nan")}
    mse_vals = []
    for t_tensor, y, s_true in zip(metric_ctx.fisher_times, metric_ctx.fisher_y, metric_ctx.fisher_score):
        t = float(t_tensor.item())
        s_hat = score_fn(y, t)
        err2 = torch.sum((s_hat - s_true) ** 2, dim=1)
        mse_vals.append(torch.mean(err2))
    mse = float(torch.mean(torch.stack(mse_vals)).item())
    rmse = float(math.sqrt(max(mse, 0.0)))
    return {"fisher_rmse": rmse}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def density_grid(target: GMM2D, lim: float, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-lim, lim, n, device=target.device, dtype=target.dtype)
    ys = torch.linspace(-lim, lim, n, device=target.device, dtype=target.dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    with torch.no_grad():
        logp = target.log_prob(pts, t=0.0).reshape(n, n)
        p = torch.exp(logp - logp.max())
    return as_numpy(X), as_numpy(Y), as_numpy(p)


def get_hist_cmap(name: str):
    if name == "bright_lava":
        # Brighter low-count response than standard inferno/magma.  Combined with
        # PowerNorm below, singleton / low-count pixels remain visible while the
        # shared count-to-brightness normalization is preserved across panels.
        return LinearSegmentedColormap.from_list(
            "bright_lava",
            ["#000000", "#3b0000", "#9a0000", "#ff1f00", "#ff9500", "#ffe066", "#ffffff"],
        )
    return plt.get_cmap(name)


def hist_count_image(arr: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    H, _, _ = np.histogram2d(
        arr[:, 0],
        arr[:, 1],
        bins=cfg.hist_bins,
        range=[[-cfg.grid_lim, cfg.grid_lim], [-cfg.grid_lim, cfg.grid_lim]],
        density=False,
    )
    return H


def hist_global_vmax(arrays: List[np.ndarray], cfg: ExperimentConfig) -> float:
    counts: List[np.ndarray] = []
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        H = hist_count_image(arr, cfg).reshape(-1)
        H = H[H > 0]
        if H.size:
            counts.append(H.astype(np.float64))
    if not counts:
        return 1.0
    vals = np.concatenate(counts)
    q = float(np.clip(cfg.hist_vmax_quantile, 0.50, 1.0))
    # Use a high quantile rather than a single global max so one hot bin does not
    # make all ordinary mass nearly invisible.  This is still a shared, fixed
    # count-to-brightness map for every panel in the figure.
    return max(float(np.quantile(vals, q)), 1.0)


def _draw_hist_panel(
    ax,
    arr: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    Pg: np.ndarray,
    cfg: ExperimentConfig,
    hist_vmax: Optional[float] = None,
) -> None:
    H = hist_count_image(arr, cfg)
    cmap = get_hist_cmap(cfg.hist_cmap)
    vmax = max(float(hist_vmax if hist_vmax is not None else H.max()), 1.0)
    ax.imshow(
        H.T,
        origin="lower",
        extent=[-cfg.grid_lim, cfg.grid_lim, -cfg.grid_lim, cfg.grid_lim],
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        norm=PowerNorm(gamma=float(cfg.hist_gamma), vmin=0.0, vmax=vmax),
        alpha=1.0,
    )
    ax.contour(
        Xg,
        Yg,
        Pg,
        levels=8,
        linewidths=float(cfg.hist_contour_lw),
        alpha=float(cfg.hist_contour_alpha),
        colors="white",
    )
    ax.set_xlim(-cfg.grid_lim, cfg.grid_lim)
    ax.set_ylim(-cfg.grid_lim, cfg.grid_lim)
    ax.set_aspect("equal")




def cfg_n_trials_from_df(df: pd.DataFrame) -> int:
    if "n_trials_observed" not in df.columns or df.empty:
        return 1
    vals = pd.to_numeric(df["n_trials_observed"], errors="coerce")
    if vals.notna().sum() == 0:
        return 1
    return int(max(1, vals.max()))


def plot_metric_array(metrics_df: pd.DataFrame, metric_names: List[str], outpath: str) -> None:
    metric_names = [m for m in metric_names if m in metrics_df.columns]
    if not metric_names:
        return
    ncols = min(3, len(metric_names))
    nrows = int(math.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.1 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    methods = list(metrics_df["method"].unique())
    show_err = cfg_n_trials_from_df(metrics_df) > 1
    for ax, metric in zip(axes, metric_names):
        for method in methods:
            sub = metrics_df[metrics_df["method"] == method].sort_values("perturb_sigma")
            if sub.empty or metric not in sub.columns:
                continue
            y = pd.to_numeric(sub[metric], errors="coerce")
            if y.notna().sum() == 0:
                continue
            yerr_col = f"{metric}_sem"
            if show_err and yerr_col in sub.columns:
                yerr = pd.to_numeric(sub[yerr_col], errors="coerce")
                ax.errorbar(sub["perturb_sigma"], y, yerr=yerr, marker="o", linewidth=1.8, capsize=2, label=method)
            else:
                ax.plot(sub["perturb_sigma"], y, marker="o", linewidth=1.8, label=method)
        ax.set_xlabel("reference perturbation sigma")
        label = "time-integrated score RMSE" if metric == "fisher_rmse" else metric.upper()
        ax.set_ylabel(label)
        ax.set_title(label + " vs reference perturbation")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(metric_names):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(outpath, dpi=190)
    plt.close(fig)


def plot_contrast_sweeps(metrics_df: pd.DataFrame, outpath: str) -> None:
    metric_names = [m for m in SAMPLE_METRICS if f"delta_vs_ref_{m}" in metrics_df.columns]
    if not metric_names:
        return
    ncols = min(3, len(metric_names))
    nrows = int(math.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.1 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    sample_df = metrics_df[metrics_df["method"] != "PERTURBED_REF"].copy()
    show_err = cfg_n_trials_from_df(metrics_df) > 1
    for ax, metric in zip(axes, metric_names):
        col = f"delta_vs_ref_{metric}"
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        for method in sample_df["method"].unique():
            sub = sample_df[sample_df["method"] == method].sort_values("perturb_sigma")
            y = pd.to_numeric(sub[col], errors="coerce")
            if y.notna().sum() == 0:
                continue
            yerr_col = f"{col}_sem"
            if show_err and yerr_col in sub.columns:
                yerr = pd.to_numeric(sub[yerr_col], errors="coerce")
                ax.errorbar(sub["perturb_sigma"], y, yerr=yerr, marker="o", linewidth=1.8, capsize=2, label=method)
            else:
                ax.plot(sub["perturb_sigma"], y, marker="o", linewidth=1.8, label=method)
        ax.set_xlabel("reference perturbation sigma")
        ax.set_ylabel(f"{metric.upper()} - reference {metric.upper()}")
        ax.set_title("contrastive improvement (<0 is better)")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(metric_names):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def _hist_prob_image(arr: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Histogram as a probability image on the plotting grid."""
    H = hist_count_image(arr, cfg).astype(np.float64)
    total = float(H.sum())
    if total > 0.0:
        H = H / total
    return H


def _draw_hist_prob_panel(
    ax,
    arr: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    Pg: np.ndarray,
    cfg: ExperimentConfig,
    hist_vmax: Optional[float] = None,
) -> None:
    """Draw the usual lava count histogram panel with target contours."""
    _draw_hist_panel(ax, arr, Xg, Yg, Pg, cfg, hist_vmax=hist_vmax)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])


def _draw_residual_panel(
    ax,
    residual: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    Pg: np.ndarray,
    cfg: ExperimentConfig,
    resid_vmax: float,
) -> None:
    """Draw empirical histogram probability residuals against the oracle histogram."""
    vmax = max(float(resid_vmax), 1.0e-12)
    ax.imshow(
        residual.T,
        origin="lower",
        extent=[-cfg.grid_lim, cfg.grid_lim, -cfg.grid_lim, cfg.grid_lim],
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        alpha=1.0,
    )
    ax.contour(
        Xg,
        Yg,
        Pg,
        levels=8,
        linewidths=float(cfg.hist_contour_lw),
        alpha=float(cfg.hist_contour_alpha),
        colors="black",
    )
    ax.set_xlim(-cfg.grid_lim, cfg.grid_lim)
    ax.set_ylim(-cfg.grid_lim, cfg.grid_lim)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])


def plot_samples_for_sigma(*args, **kwargs) -> None:
    """Deprecated: per-sigma heatmaps are intentionally disabled.

    The experiment now emits one histogram diagnostic only:
    histogram_array_over_sigmas.png.  That plot has two rows per perturbation
    level: a histogram row followed by a residual-to-ORACLE row.  This no-op is
    kept only so stale calls from older branches do not create extra figures.
    """
    return None


def plot_histogram_array_over_sigmas(
    target: GMM2D,
    refs_by_sigma: Dict[float, torch.Tensor],
    samples_by_sigma: Dict[float, Dict[str, torch.Tensor]],
    method_order: List[str],
    cfg: ExperimentConfig,
    outpath: str,
) -> None:
    """One horizontal sampler array per sigma, with residuals to ORACLE below.

    Layout is exactly:
        row 2*k     : histograms for perturb_sigma[k]
        row 2*k + 1 : histogram_probability(method) - histogram_probability(ORACLE)

    Columns are sampler methods only, including ORACLE when present.  The
    perturbed reference cloud is intentionally not plotted here because it is not
    a sampler and the requested diagnostic is sampler-vs-oracle residuals.
    """
    sigmas = sorted(refs_by_sigma.keys())
    if not sigmas:
        return

    # Use sampler columns only. Preserve caller order and force ORACLE at the end
    # if present in the data but absent from method_order.
    sampler_names: List[str] = []
    for name in method_order:
        if name in samples_by_sigma.get(sigmas[0], {}) and name not in sampler_names:
            sampler_names.append(name)
    if "ORACLE" not in sampler_names and "ORACLE" in samples_by_sigma.get(sigmas[0], {}):
        sampler_names.append("ORACLE")
    if not sampler_names:
        raise RuntimeError("No sampler samples available for histogram array plot.")
    if any("ORACLE" not in samples_by_sigma.get(sigma, {}) for sigma in sigmas):
        raise RuntimeError(
            "histogram_array_over_sigmas requires ORACLE samples so residuals can be computed "
            "as method histogram minus ORACLE histogram. Re-run with --include_oracle."
        )

    Xg, Yg, Pg = density_grid(target, cfg.grid_lim, cfg.grid_n)

    # Shared histogram count brightness across all sampler panels.
    hist_arrays: List[np.ndarray] = []
    for sigma in sigmas:
        for name in sampler_names:
            hist_arrays.append(as_numpy(samples_by_sigma[sigma][name]))
    hist_vmax = hist_global_vmax(hist_arrays, cfg)

    # Precompute probability histograms and residuals to ORACLE.  Residuals use
    # normalized histogram images, not raw counts, so unequal sample counts do not
    # show up as spurious global offsets.
    residual_by_sigma: Dict[float, Dict[str, np.ndarray]] = {}
    abs_residual_values: List[np.ndarray] = []
    for sigma in sigmas:
        residual_by_sigma[sigma] = {}
        oracle_prob = _hist_prob_image(as_numpy(samples_by_sigma[sigma]["ORACLE"]), cfg)
        for name in sampler_names:
            prob = _hist_prob_image(as_numpy(samples_by_sigma[sigma][name]), cfg)
            resid = prob - oracle_prob
            residual_by_sigma[sigma][name] = resid
            vals = np.abs(resid).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                abs_residual_values.append(vals)

    if abs_residual_values:
        all_abs = np.concatenate(abs_residual_values)
        positive_abs = all_abs[all_abs > 0.0]
        scale_vals = positive_abs if positive_abs.size else all_abs
        q = float(np.clip(getattr(cfg, "residual_vmax_quantile", 0.995), 0.50, 1.0))
        resid_ref = float(np.quantile(scale_vals, q)) if scale_vals.size else 1.0e-12
    else:
        resid_ref = 1.0e-12
    intensity = max(float(getattr(cfg, "residual_intensity", 1.8)), 1.0e-6)
    resid_vmax = max(resid_ref / intensity, 1.0e-12)

    nrows = 2 * len(sigmas)
    ncols = len(sampler_names)
    fig_w = max(3.2 * ncols, 6.0)
    fig_h = max(3.1 * nrows, 4.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(nrows, ncols)

    for s_idx, sigma in enumerate(sigmas):
        hist_row = 2 * s_idx
        resid_row = hist_row + 1
        for col_idx, name in enumerate(sampler_names):
            arr = as_numpy(samples_by_sigma[sigma][name])
            ax_hist = axes[hist_row, col_idx]
            _draw_hist_prob_panel(ax_hist, arr, Xg, Yg, Pg, cfg, hist_vmax=hist_vmax)
            if s_idx == 0:
                ax_hist.set_title(name)
            if col_idx == 0:
                ax_hist.set_ylabel(f"sigma={sigma:g}\nhist")

            ax_resid = axes[resid_row, col_idx]
            _draw_residual_panel(
                ax_resid,
                residual_by_sigma[sigma][name],
                Xg,
                Yg,
                Pg,
                cfg,
                resid_vmax=resid_vmax,
            )
            if col_idx == 0:
                ax_resid.set_ylabel(f"sigma={sigma:g}\nresid")

    fig.suptitle(
        "Histogram rows and residual-to-ORACLE rows over reference perturbation levels | "
        f"shared histogram vmax count = {hist_vmax:.0f} | residual scale = ±{resid_vmax:.3g}",
        fontsize=14,
    )
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def aggregate_trials(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Average numeric trial-level diagnostics over repeated random trials."""
    if df.empty:
        return df.copy()
    numeric_cols = [
        c for c in df.columns
        if c not in set(group_cols + ["trial", "trial_seed"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = df.groupby(group_cols, dropna=False)
    mean_df = grouped[numeric_cols].mean().reset_index() if numeric_cols else grouped.size().reset_index(name="n_rows")
    if numeric_cols:
        std_df = grouped[numeric_cols].std(ddof=1).add_suffix("_std").reset_index()
        sem_df = grouped[numeric_cols].sem(ddof=1).add_suffix("_sem").reset_index()
        mean_df = mean_df.merge(std_df, on=group_cols, how="left").merge(sem_df, on=group_cols, how="left")
    mean_df["n_trials_observed"] = grouped.size().to_numpy()
    return mean_df


def add_reference_contrasts(metrics_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Add per-trial sample-vs-perturbed-reference contrasts before averaging."""
    metrics_df = metrics_df.copy()
    contrast_metrics = [m for m in SAMPLE_METRICS if metric_enabled(cfg, m) and m in metrics_df.columns]
    for metric in contrast_metrics:
        ref_col = f"_ref_{metric}"
        ref_df = (
            metrics_df[metrics_df["method"] == "PERTURBED_REF"][["perturb_sigma", "trial", metric]]
            .rename(columns={metric: ref_col})
        )
        metrics_df = metrics_df.merge(ref_df, on=["perturb_sigma", "trial"], how="left")
        metrics_df[f"delta_vs_ref_{metric}"] = metrics_df[metric] - metrics_df[ref_col]
        denom = metrics_df[ref_col].abs().clip(lower=1.0e-12)
        metrics_df[f"ratio_vs_ref_{metric}"] = metrics_df[metric] / denom
        metrics_df = metrics_df.drop(columns=[ref_col])
    return metrics_df


def format_mean_sem(row: pd.Series, metric: str, n_trials: int) -> str:
    val = row.get(metric, np.nan)
    try:
        val_f = float(val)
    except Exception:
        return "nan"
    if not math.isfinite(val_f):
        return "nan"
    sem_col = f"{metric}_sem"
    sem = row.get(sem_col, np.nan)
    try:
        sem_f = float(sem)
    except Exception:
        sem_f = float("nan")
    if n_trials > 1 and math.isfinite(sem_f):
        return f"{val_f:.5g}±{sem_f:.2g}"
    return f"{val_f:.5g}"




def normalize_corruption_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "add": "heat",
        "additive": "heat",
        "additive-noise": "heat",
        "noise": "heat",
        "gaussian": "heat",
        "heat": "heat",
        "ou": "ou",
        "ornstein-uhlenbeck": "ou",
        "ornstein": "ou",
    }
    if mode_key not in aliases:
        raise ValueError("Unknown corruption_mode={!r}. Use 'heat' or 'ou'.".format(mode))
    return aliases[mode_key]


@torch.no_grad()
def perturb_reference_bank(
    clean_refs: torch.Tensor,
    sigma: float,
    corruption_mode: str,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float | str]]:
    """Perturb a clean reference bank in either heat or OU mode.

    heat interprets perturb_sigma as an additive Gaussian standard deviation:
        x_eps = x + sigma z.

    ou interprets perturb_sigma as an OU diffusion time:
        x_eps = exp(-sigma) x + sqrt(1 - exp(-2 sigma)) z.

    This lets values such as sigma=1.2 remain meaningful in OU mode: they
    represent stronger mean reversion rather than an invalid noise standard
    deviation larger than one.
    """
    mode = normalize_corruption_mode(corruption_mode)
    sigma_f = float(sigma)
    noise = torch.randn(clean_refs.shape, device=clean_refs.device, dtype=clean_refs.dtype, generator=generator)
    if mode == "heat":
        signal_scale = 1.0
        noise_scale = sigma_f
    elif mode == "ou":
        if sigma_f < 0.0:
            raise ValueError("OU corruption requires perturb_sigma >= 0, got {}".format(sigma_f))
        signal_scale = math.exp(-sigma_f)
        noise_scale = math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * sigma_f)))
    else:  # normalize_corruption_mode makes this unreachable.
        raise ValueError("Unknown corruption mode: {}".format(corruption_mode))
    refs = (signal_scale * clean_refs + noise_scale * noise).detach()
    return refs, {
        "corruption_mode": mode,
        "corruption_signal_scale": float(signal_scale),
        "corruption_noise_scale": float(noise_scale),
    }


def expand_bootstrap_token(token: str) -> List[str]:
    """Expand shorthand like HLSI2 into [HLSI, HLSI].

    Method names in this script do not end in digits, so trailing digits are
    unambiguously interpreted as a repetition count.  Examples:
        HLSI2              -> HLSI_HLSI
        CE-HLSI3           -> CE-HLSI_CE-HLSI_CE-HLSI
        DRC-CE-HLSI2_HLSI  -> DRC-CE-HLSI_DRC-CE-HLSI_HLSI
    """
    token = str(token).strip()
    if not token:
        raise ValueError("Empty method token in --methods")
    split = len(token)
    while split > 0 and token[split - 1].isdigit():
        split -= 1
    if split == len(token):
        return [token]
    base = token[:split]
    count_str = token[split:]
    if not base:
        raise ValueError(f"Invalid bootstrap shorthand '{token}': missing method name before count.")
    count = int(count_str)
    if count <= 0:
        raise ValueError(f"Invalid bootstrap shorthand '{token}': repetition count must be positive.")
    return [base] * count


def split_bootstrap_method(method: str) -> List[str]:
    """Return the actual execution order for a bootstrap chain.

    Naming convention follows the inverse-problem sampler convention: A_B means
    run B first, then use B's samples as the reference bank for A. Therefore the
    execution order is the underscore-separated tokens read right-to-left.

    Numeric shorthand is also accepted: HLSI2 is interpreted as HLSI_HLSI,
    CE-HLSI3 as CE-HLSI_CE-HLSI_CE-HLSI, etc.  Shorthand expansion is applied
    before the right-to-left execution-order reversal.
    """
    raw_parts = [part.strip() for part in str(method).split("_") if part.strip()]
    if not raw_parts:
        raise ValueError("Empty method name in --methods")
    parts: List[str] = []
    for part in raw_parts:
        parts.extend(expand_bootstrap_token(part))
    return list(reversed(parts)) if len(parts) > 1 else parts


def method_bootstrap_depth(method: str) -> int:
    if str(method).upper() == "ORACLE":
        return 1
    return len(split_bootstrap_method(method))


def has_bootstrap_chains(methods: Tuple[str, ...] | List[str]) -> bool:
    return any(method_bootstrap_depth(method) > 1 for method in methods)


def build_snis_bank(
    target: GMM2D,
    anchors: torch.Tensor,
    gate_anchors: Optional[torch.Tensor],
    cfg: ExperimentConfig,
    log_ref_weights: Optional[torch.Tensor] = None,
) -> SNISScoreBank:
    return SNISScoreBank(
        target=target,
        anchors=anchors,
        curvature_mode=cfg.curvature_mode,
        curvature_floor=cfg.curvature_floor,
        curvature_cap=cfg.curvature_cap,
        resolvent_eps=cfg.resolvent_eps,
        gate_clip=cfg.gate_clip,
        weight_temp=cfg.weight_temp,
        eval_chunk=cfg.eval_chunk,
        op_blend_reg=cfg.op_blend_reg,
        op_blend_pinv_rtol=cfg.op_blend_pinv_rtol,
        op_blend_project_gate=cfg.op_blend_project_gate,
        gate_anchors=gate_anchors,
        log_ref_weights=log_ref_weights,
    )



def normalize_method_key(method: str) -> str:
    return str(method).strip().replace("_", "-").lower()


def is_drc_method_token(method: str) -> bool:
    key = normalize_method_key(method)
    return key in {"drc-ce-hlsi", "drc-ce", "density-ratio-ce-hlsi", "density-ratio-corrected-ce-hlsi"}


def score_method_for_stage(method: str) -> str:
    return "CE-HLSI" if is_drc_method_token(method) else method


def standard_normal_logprob(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return -0.5 * (d * math.log(2.0 * math.pi) + torch.sum(x * x, dim=-1))


def centered_log_weights(logw: torch.Tensor) -> torch.Tensor:
    logw = torch.nan_to_num(logw, nan=0.0, posinf=0.0, neginf=0.0)
    return logw - (torch.logsumexp(logw, dim=0) - math.log(max(int(logw.numel()), 1)))


def log_weight_ess(logw: torch.Tensor) -> Tuple[float, float]:
    if logw.numel() == 0:
        return float("nan"), float("nan")
    lw = logw - torch.max(logw)
    w = torch.exp(lw)
    ess = (w.sum() ** 2) / torch.clamp(torch.sum(w * w), min=1.0e-30)
    ess_f = safe_float(ess)
    return ess_f, ess_f / float(logw.numel())


def finalize_density_ratio_weights(raw_rho: torch.Tensor, cfg: ExperimentConfig) -> Tuple[torch.Tensor, Dict[str, float | bool]]:
    """Center, temper, clip, and optionally shrink beta to satisfy ESS floor."""
    raw_centered = centered_log_weights(raw_rho)
    beta_target = max(float(cfg.rho_beta), 0.0)
    clip = cfg.rho_clip

    def apply(beta: float) -> torch.Tensor:
        out = beta * raw_centered
        if clip is not None and clip > 0:
            out = torch.clamp(out, min=-float(clip), max=float(clip))
        return centered_log_weights(out)

    rho = apply(beta_target)
    ess, ess_frac = log_weight_ess(rho)
    beta_eff = beta_target
    floor = max(float(cfg.rho_ess_floor), 0.0)
    adapted = False
    if floor > 0.0 and math.isfinite(ess_frac) and ess_frac < floor and beta_target > 0.0:
        lo, hi = 0.0, beta_target
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            cand = apply(mid)
            _, cand_frac = log_weight_ess(cand)
            if math.isfinite(cand_frac) and cand_frac >= floor:
                lo = mid
            else:
                hi = mid
        beta_eff = lo
        rho = apply(beta_eff)
        ess, ess_frac = log_weight_ess(rho)
        adapted = True

    return rho.detach(), {
        "rho_beta_target": float(beta_target),
        "rho_beta_eff": float(beta_eff),
        "rho_adapted_for_ess": bool(adapted),
        "rho_ess": float(ess),
        "rho_ess_frac": float(ess_frac),
        "rho_mean": safe_float(rho.mean()),
        "rho_std": safe_float(rho.std(unbiased=False)),
        "rho_min": safe_float(rho.min()),
        "rho_max": safe_float(rho.max()),
        "rho_raw_mean": safe_float(raw_centered.mean()),
        "rho_raw_std": safe_float(raw_centered.std(unbiased=False)),
        "rho_raw_min": safe_float(raw_centered.min()),
        "rho_raw_max": safe_float(raw_centered.max()),
    }


@torch.no_grad()
def pf_logprob_ce_hlsi(bank: SNISScoreBank, x0: torch.Tensor, cfg: ExperimentConfig) -> Tuple[torch.Tensor, Dict[str, float | bool]]:
    """Estimate log q_k(x0) by the forward OU probability-flow ODE.

    The frozen bank's corrected CE-HLSI field supplies both score and analytic
    divergence. This intentionally integrates from max(t_end, pf_t_start) to
    pf_t_end, so DRC methods default to using non-denoised sampler endpoints.
    """
    if x0.numel() == 0:
        return torch.empty((0,), device=bank.device, dtype=bank.dtype), {"pf_failed_frac": 0.0}
    t0 = float(cfg.t_end if cfg.pf_t_start is None else cfg.pf_t_start)
    t1 = float(cfg.t_start if cfg.pf_t_end is None else cfg.pf_t_end)
    t0 = max(t0, 1.0e-6)
    t1 = max(t1, t0 + 1.0e-6)
    n_steps = max(int(cfg.pf_steps), 1)
    batch = max(int(cfg.rho_batch), 1)
    out: List[torch.Tensor] = []
    failed_total = 0
    max_abs_div = 0.0
    max_abs_state = 0.0
    ts = torch.linspace(t0, t1, n_steps + 1, device=bank.device, dtype=bank.dtype)
    d = x0.shape[1]

    for start in range(0, x0.shape[0], batch):
        x = x0[start:start + batch].detach().clone()
        A = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        alive = torch.ones((x.shape[0],), device=x.device, dtype=torch.bool)
        for j in range(n_steps):
            t = float(ts[j].item())
            tn = float(ts[j + 1].item())
            h = tn - t
            s, div = bank.ce_hlsi_score_and_divergence(x, t)
            s = clamp_norm(s, cfg.score_clip)
            if cfg.pf_div_clip is not None and cfg.pf_div_clip > 0:
                div = torch.clamp(div, min=-float(cfg.pf_div_clip), max=float(cfg.pf_div_clip))
            v = -x - s
            a = float(d) + div

            x_e = x + h * v
            s_e, div_e = bank.ce_hlsi_score_and_divergence(x_e, tn)
            s_e = clamp_norm(s_e, cfg.score_clip)
            if cfg.pf_div_clip is not None and cfg.pf_div_clip > 0:
                div_e = torch.clamp(div_e, min=-float(cfg.pf_div_clip), max=float(cfg.pf_div_clip))
            v_e = -x_e - s_e
            a_e = float(d) + div_e

            finite = torch.isfinite(x_e).all(dim=1) & torch.isfinite(v_e).all(dim=1) & torch.isfinite(a) & torch.isfinite(a_e)
            alive = alive & finite
            x = x + 0.5 * h * (v + v_e)
            A = A + 0.5 * h * (a + a_e)
            if cfg.sample_clip and cfg.sample_clip > 0:
                x = torch.clamp(x, min=-cfg.sample_clip, max=cfg.sample_clip)
            max_abs_div = max(max_abs_div, safe_float(torch.max(torch.abs(torch.cat([div.reshape(-1), div_e.reshape(-1)])))))
            max_abs_state = max(max_abs_state, safe_float(x.abs().max()))
        logq = standard_normal_logprob(x) - A
        good = alive & torch.isfinite(logq)
        failed_total += int((~good).sum().item())
        if (~good).any():
            replacement = torch.nanmedian(logq[good]) if good.any() else torch.tensor(0.0, device=x.device, dtype=x.dtype)
            logq = torch.where(good, logq, replacement)
        out.append(logq.detach())

    logq_all = torch.cat(out, dim=0)
    return logq_all, {
        "pf_failed_frac": float(failed_total) / float(max(1, x0.shape[0])),
        "pf_t_start": float(t0),
        "pf_t_end": float(t1),
        "pf_steps": int(n_steps),
        "pf_max_abs_div": float(max_abs_div),
        "pf_max_abs_state": float(max_abs_state),
        "pf_logq_mean": safe_float(logq_all.mean()),
        "pf_logq_std": safe_float(logq_all.std(unbiased=False)),
        "pf_logq_min": safe_float(logq_all.min()),
        "pf_logq_max": safe_float(logq_all.max()),
    }


@torch.no_grad()
def compute_drc_next_weights(
    target: GMM2D,
    frozen_bank: SNISScoreBank,
    next_refs: torch.Tensor,
    cfg: ExperimentConfig,
) -> Tuple[torch.Tensor, Dict[str, float | bool]]:
    logq, pf_info = pf_logprob_ce_hlsi(frozen_bank, next_refs, cfg)
    logpi = target.log_prob(next_refs, t=0.0)
    raw_rho = logpi - logq
    rho, rho_info = finalize_density_ratio_weights(raw_rho, cfg)

    # Mode-mass diagnostic under exact GMM responsibilities. This is not used by
    # the algorithm; it is useful for checking whether DRC is correcting mode mass.
    resp = target.responsibilities(next_refs, t=0.0)
    assign = torch.argmax(resp, dim=1)
    target_w = target.weights.detach()
    mode_fracs = torch.stack([(assign == k).to(target.dtype).mean() for k in range(target.K)])
    mode_l1 = torch.sum(torch.abs(mode_fracs - target_w))

    info: Dict[str, float | bool] = {
        **pf_info,
        **rho_info,
        "rho_logpi_mean": safe_float(logpi.mean()),
        "rho_logpi_std": safe_float(logpi.std(unbiased=False)),
        "mode_mass_l1": safe_float(mode_l1),
    }
    for k in range(target.K):
        info[f"mode_mass_{k}"] = safe_float(mode_fracs[k])
        info[f"target_mode_weight_{k}"] = safe_float(target_w[k])
    return rho.detach(), info


# -----------------------------------------------------------------------------
# Computational hypothesis tests for the DRC-CE-HLSI claim
# -----------------------------------------------------------------------------


def infer_hypothesis_max_depth(cfg: ExperimentConfig) -> int:
    if int(cfg.hypothesis_max_depth) > 0:
        return int(cfg.hypothesis_max_depth)
    depths = []
    for method in cfg.methods:
        if str(method).upper() == "ORACLE":
            continue
        try:
            tokens = split_bootstrap_method(method)
        except Exception:
            continue
        if all(normalize_method_key(score_method_for_stage(tok)) == "ce-hlsi" for tok in tokens):
            depths.append(len(tokens))
    return max([1, *depths])


def make_hypothesis_time_grid(cfg: ExperimentConfig, device: torch.device, dtype: torch.dtype, n_t: int) -> torch.Tensor:
    n_t = max(int(n_t), 2)
    t_min = max(float(cfg.t_end), 1.0e-6)
    t_max = max(float(cfg.t_start), t_min + 1.0e-6)
    # Use log spacing because most score/path pathologies concentrate near small t.
    return torch.exp(torch.linspace(math.log(t_min), math.log(t_max), n_t, device=device, dtype=dtype))


@torch.no_grad()
def ou_forward_noise(x0: torch.Tensor, t: float, generator: torch.Generator) -> torch.Tensor:
    t_tensor = torch.as_tensor(t, device=x0.device, dtype=x0.dtype)
    alpha, gamma = alpha_gamma(t_tensor)
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    return alpha * x0 + torch.sqrt(torch.clamp(gamma, min=0.0)) * eps


def hist_prob_distance(x: torch.Tensor, y: torch.Tensor, cfg: ExperimentConfig, kind: str) -> float:
    Hx = _hist_prob_image(as_numpy(x[:, :2]), cfg)
    Hy = _hist_prob_image(as_numpy(y[:, :2]), cfg)
    diff = Hx - Hy
    if kind == "hist_l1":
        return float(np.sum(np.abs(diff)))
    if kind == "hist_l2":
        return float(np.sqrt(np.sum(diff * diff)))
    raise ValueError(f"Unknown histogram path metric: {kind}")


@torch.no_grad()
def distribution_discrepancy(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: ExperimentConfig,
    metric: str,
    generator: torch.Generator,
) -> float:
    metric_key = str(metric).strip().lower().replace("-", "_")
    n = min(int(cfg.hypothesis_n_eval), x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    if n <= 1 or (not torch.isfinite(x).all()) or (not torch.isfinite(y).all()):
        return float("nan")
    if metric_key == "mmd":
        return mmd_rbf(x, y, bandwidth=None)
    if metric_key in {"sw2", "sliced_w2", "sliced-w2"}:
        return sliced_w2(x, y, n_proj=cfg.sw2_projections, generator=generator)
    if metric_key in {"hist_l1", "hist_l2"}:
        return hist_prob_distance(x, y, cfg, metric_key)
    raise ValueError("Unknown hypothesis_path_metric={!r}. Use mmd, sw2, hist_l1, or hist_l2.".format(metric))


def pearson_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return float("nan")
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if safe_float(denom) <= 1.0e-30:
        return float("nan")
    return safe_float(torch.sum(x * y) / denom)


@torch.no_grad()
def gaussian_kde_log_density(
    eval_x: torch.Tensor,
    ref_x: torch.Tensor,
    cfg: ExperimentConfig,
    leave_one_out_prefix: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Gaussian KDE log density for the empirical q_k law.

    If eval_x is the prefix of ref_x, leave_one_out_prefix removes the diagonal
    self-kernel.  The diagnostic only uses corrections up to an additive
    constant, but including the Gaussian normalizer makes the scale interpretable.
    """
    ref_x = ref_x.detach()
    eval_x = eval_x.detach()
    n_ref, d = ref_x.shape
    if n_ref <= 1:
        return torch.zeros((eval_x.shape[0],), device=eval_x.device, dtype=eval_x.dtype), 1.0
    if cfg.hypothesis_kde_bandwidth is not None and cfg.hypothesis_kde_bandwidth > 0:
        h = float(cfg.hypothesis_kde_bandwidth)
    else:
        # Median distance times Scott's factor gives a stable 2D KDE bandwidth
        # without overfitting every sample with a near-delta kernel.
        h = median_bandwidth(ref_x, max_n=min(1200, n_ref)) * (float(n_ref) ** (-1.0 / (d + 4.0)))
        h = max(float(h), 1.0e-3)
    batch = max(int(cfg.rho_batch), 1)
    norm_const = d * math.log(h) + 0.5 * d * math.log(2.0 * math.pi)
    outs: List[torch.Tensor] = []
    for start in range(0, eval_x.shape[0], batch):
        end = min(start + batch, eval_x.shape[0])
        xb = eval_x[start:end]
        d2 = pairwise_sq_dists(xb, ref_x) / max(h * h, 1.0e-12)
        logk = -0.5 * d2
        denom_n = n_ref
        if leave_one_out_prefix:
            rows = torch.arange(end - start, device=eval_x.device)
            cols = torch.arange(start, end, device=eval_x.device)
            valid = cols < n_ref
            if bool(valid.any().item()):
                logk[rows[valid], cols[valid]] = -torch.inf
                denom_n = max(n_ref - 1, 1)
        logden = torch.logsumexp(logk, dim=1) - math.log(float(denom_n)) - norm_const
        outs.append(logden)
    out = torch.cat(outs, dim=0)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, float(h)


@torch.no_grad()
def compute_pf_kde_correction_diagnostics(
    target: GMM2D,
    bank: SNISScoreBank,
    samples: torch.Tensor,
    cfg: ExperimentConfig,
) -> Tuple[Dict[str, float | bool], torch.Tensor]:
    n_eval = min(int(cfg.hypothesis_n_eval), samples.shape[0])
    eval_x = samples[:n_eval].detach()
    logq_pf, pf_info = pf_logprob_ce_hlsi(bank, eval_x, cfg)
    logpi = target.log_prob(eval_x, t=0.0)
    rho_pf_raw = logpi - logq_pf
    rho_pf = rho_pf_raw - rho_pf_raw.mean()

    logq_kde, kde_bw = gaussian_kde_log_density(eval_x, samples.detach(), cfg, leave_one_out_prefix=True)
    rho_kde_raw = logpi - logq_kde
    rho_kde = rho_kde_raw - rho_kde_raw.mean()

    diff = rho_pf - rho_kde
    rho_lme = centered_log_weights(rho_pf_raw)
    ess, ess_frac = log_weight_ess(rho_lme)
    q05 = torch.quantile(rho_pf, 0.05)
    q95 = torch.quantile(rho_pf, 0.95)
    info: Dict[str, float | bool] = {
        **pf_info,
        "rho_pf_kde_corr": pearson_corr_torch(rho_pf, rho_kde),
        "rho_pf_kde_rmse": safe_float(torch.sqrt(torch.mean(diff * diff))),
        "rho_pf_kde_mae": safe_float(torch.mean(torch.abs(diff))),
        "rho_pf_std": safe_float(rho_pf.std(unbiased=False)),
        "rho_pf_q05": safe_float(q05),
        "rho_pf_q95": safe_float(q95),
        "rho_pf_q95_q05": safe_float(q95 - q05),
        "rho_pf_ess": float(ess),
        "rho_pf_ess_frac": float(ess_frac),
        "rho_kde_std": safe_float(rho_kde.std(unbiased=False)),
        "rho_kde_bandwidth": float(kde_bw),
        "rho_eval_n": int(n_eval),
    }
    return info, rho_pf_raw.detach()


@torch.no_grad()
def compute_oracle_gate_error(
    bank: SNISScoreBank,
    oracle_bank: SNISScoreBank,
    gate_eval: List[Tuple[float, torch.Tensor]],
) -> Dict[str, float]:
    frob_vals: List[torch.Tensor] = []
    eig_vals: List[torch.Tensor] = []
    align_vals: List[torch.Tensor] = []
    for t, y in gate_eval:
        G = bank.ce_hlsi_gate(y, t)
        Go = oracle_bank.ce_hlsi_gate(y, t)
        diff = G - Go
        frob_vals.append(torch.sum(diff * diff, dim=(-1, -2)))
        evals, evecs = torch.linalg.eigh(sym(G))
        evals_o, evecs_o = torch.linalg.eigh(sym(Go))
        eig_vals.append(torch.mean((evals - evals_o) ** 2, dim=-1))
        # Dominant-eigenvector alignment error; sign-invariant and diagnostic only.
        v = evecs[:, :, -1]
        vo = evecs_o[:, :, -1]
        align = torch.abs(torch.sum(v * vo, dim=-1)).clamp(0.0, 1.0)
        align_vals.append(1.0 - align)
    frob = torch.cat(frob_vals, dim=0)
    eig = torch.cat(eig_vals, dim=0)
    align_err = torch.cat(align_vals, dim=0)
    return {
        "gate_rmse_frob": safe_float(torch.sqrt(torch.mean(frob))),
        "gate_eig_rmse": safe_float(torch.sqrt(torch.mean(eig))),
        "gate_dominant_alignment_error": safe_float(torch.mean(align_err)),
        "gate_eval_n": int(sum(y.shape[0] for _, y in gate_eval)),
    }


@torch.no_grad()
def run_drc_hypothesis_tests_for_trial(
    target: GMM2D,
    initial_refs: torch.Tensor,
    initial_gate_refs: torch.Tensor,
    cfg: ExperimentConfig,
    trial_seed: int,
    sigma_idx: int,
) -> Tuple[List[Dict[str, float | str | int | bool]], List[Dict[str, float | str | int | bool]], List[Dict[str, float | str | int | bool]]]:
    """Run Tests 1--4 from the DRC report for one sigma/trial.

    Tests are evaluated on paired CE-HLSI and DRC-CE-HLSI bootstrap trajectories
    of the same depth.  CE-HLSI receives counterfactual PF corrections for Tests
    2--3, but those corrections are not fed back into its trajectory.
    """
    depth = infer_hypothesis_max_depth(cfg)
    path_times = make_hypothesis_time_grid(cfg, target.device, target.dtype, cfg.hypothesis_n_path_t)
    gate_times = make_hypothesis_time_grid(cfg, target.device, target.dtype, cfg.hypothesis_gate_n_t)

    oracle_gen = make_generator(int(trial_seed + 870_001 + 13 * sigma_idx), target.device)
    oracle_n = max(int(cfg.hypothesis_oracle_n_ref), int(cfg.n_ref))
    oracle_refs = target.sample(oracle_n, generator=oracle_gen).detach()
    oracle_bank = build_snis_bank(target, oracle_refs, oracle_refs, cfg, log_ref_weights=None)

    gate_gen = make_generator(int(trial_seed + 870_101 + 17 * sigma_idx), target.device)
    gate_eval: List[Tuple[float, torch.Tensor]] = []
    for t_tensor in gate_times:
        t = float(t_tensor.item())
        gate_eval.append((t, target.sample_pt(max(int(cfg.hypothesis_gate_n_per_t), 1), t, generator=gate_gen).detach()))

    path_rows: List[Dict[str, float | str | int | bool]] = []
    scalar_rows: List[Dict[str, float | str | int | bool]] = []
    gate_rows: List[Dict[str, float | str | int | bool]] = []

    for family, use_drc in [("CE-HLSI", False), ("DRC-CE-HLSI", True)]:
        current_refs = initial_refs.detach()
        current_gate_refs = initial_gate_refs.detach()
        current_log_ref_weights = torch.zeros((current_refs.shape[0],), device=current_refs.device, dtype=current_refs.dtype)
        for k in range(1, depth + 1):
            bank = build_snis_bank(target, current_refs, current_gate_refs, cfg, log_ref_weights=current_log_ref_weights)
            score_fn = lambda x, t, bank=bank: bank.estimate(x, t=t, method="CE-HLSI")
            stage_cfg = replace(cfg, final_denoise=False)
            stage_seed = int(trial_seed + 910_000 + 10_000 * sigma_idx + (1_000_003 if use_drc else 0) + k)
            stage_gen = make_generator(stage_seed, target.device)
            samples, snapshots, sampler_info = reverse_ou_heun_sde_with_snapshots(
                target=target,
                score_fn=score_fn,
                cfg=stage_cfg,
                generator=stage_gen,
                snapshot_times=path_times,
            )

            # Test 1: compare reverse-path marginal at t with forward OU noising of q_k samples.
            path_gen = make_generator(int(stage_seed + 123_457), target.device)
            n_eval = min(int(cfg.hypothesis_n_eval), samples.shape[0])
            endpoint_eval = samples[:n_eval].detach()
            for t_tensor in path_times:
                t = float(t_tensor.item())
                # Dict keys are Python floats from path_times; use nearest key for safety.
                snap_key = min(snapshots.keys(), key=lambda z: abs(float(z) - t))
                rev = snapshots[snap_key][:n_eval].detach()
                fwd = ou_forward_noise(endpoint_eval, t, generator=path_gen)
                disc = distribution_discrepancy(rev, fwd, cfg, cfg.hypothesis_path_metric, path_gen)
                path_rows.append({
                    "method_family": family,
                    "iteration": int(k),
                    "t": float(t),
                    "path_metric": str(cfg.hypothesis_path_metric),
                    "path_discrepancy": float(disc),
                    "path_eval_n": int(n_eval),
                    "sampler_failed": bool(sampler_info.get("failed", False)),
                    "sampler_fail_reason": str(sampler_info.get("fail_reason", "")),
                    "sampler_max_abs_score": safe_float(sampler_info.get("max_abs_score", float("nan"))),
                })

            # Tests 2--3: PF correction coherence and correction decay.
            scalar_info, _rho_pf_raw = compute_pf_kde_correction_diagnostics(target, bank, samples.detach(), cfg)
            scalar_rows.append({
                "method_family": family,
                "iteration": int(k),
                "correction_is_applied_to_next_round": bool(use_drc),
                **scalar_info,
            })

            # Test 4: gate convergence to oracle target-reference CE-HLSI gate.
            gate_info = compute_oracle_gate_error(bank, oracle_bank, gate_eval)
            gate_rows.append({
                "method_family": family,
                "iteration": int(k),
                **gate_info,
            })

            # Advance the trajectory.  DRC uses the actual PF correction; CE-HLSI
            # deliberately discards its counterfactual correction.
            next_ref_n = min(int(cfg.n_ref), int(samples.shape[0]))
            next_refs = samples[:next_ref_n].detach()
            if use_drc and next_ref_n > 0:
                next_log_ref_weights, _ratio_info = compute_drc_next_weights(target, bank, next_refs, cfg)
            else:
                next_log_ref_weights = torch.zeros((next_ref_n,), device=next_refs.device, dtype=next_refs.dtype)
            current_refs = next_refs
            current_gate_refs = current_refs
            current_log_ref_weights = next_log_ref_weights.detach()

    return path_rows, scalar_rows, gate_rows


def _line_style_for_family(family: str) -> str:
    return "-" if str(family).startswith("DRC") else "--"


def plot_hypothesis_path_self_consistency(path_df: pd.DataFrame, outpath: str) -> None:
    if path_df.empty or "path_discrepancy" not in path_df.columns:
        return
    sigmas = sorted(path_df["perturb_sigma"].unique()) if "perturb_sigma" in path_df.columns else [None]
    ncols = min(3, len(sigmas))
    nrows = int(math.ceil(len(sigmas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.0 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    cmap = plt.get_cmap("tab10")
    for ax, sigma in zip(axes, sigmas):
        sub_sigma = path_df if sigma is None else path_df[path_df["perturb_sigma"] == sigma]
        for family in ["DRC-CE-HLSI", "CE-HLSI"]:
            sub_family = sub_sigma[sub_sigma["method_family"] == family]
            for idx, k in enumerate(sorted(sub_family["iteration"].unique())):
                sub = sub_family[sub_family["iteration"] == k].sort_values("t")
                y = pd.to_numeric(sub["path_discrepancy"], errors="coerce")
                if y.notna().sum() == 0:
                    continue
                ax.plot(sub["t"], y, linestyle=_line_style_for_family(family), color=cmap((int(k) - 1) % 10), linewidth=1.8, marker="o", label=f"{family} k={int(k)}")
        ax.set_xscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("D(q_rev,t, q_fwd,t)")
        title = "path self-consistency" if sigma is None else f"sigma={float(sigma):g}"
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(sigmas):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle("Test 1: path-marginal self-consistency | solid=DRC, dashed=CE, color=iteration", fontsize=13)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_hypothesis_density_ratio_scalars(scalar_df: pd.DataFrame, outpath: str) -> None:
    if scalar_df.empty:
        return
    metrics = [
        ("rho_pf_kde_corr", "PF/KDE correction corr", False),
        ("rho_pf_kde_rmse", "PF/KDE correction RMSE", False),
        ("rho_pf_std", "std(centered PF correction)", False),
        ("rho_pf_q95_q05", "q95-q05(centered PF correction)", False),
        ("rho_pf_ess_frac", "ESS(PF correction)/N", False),
    ]
    metrics = [m for m in metrics if m[0] in scalar_df.columns]
    if not metrics:
        return
    sigmas = sorted(scalar_df["perturb_sigma"].unique()) if "perturb_sigma" in scalar_df.columns else [None]
    nrows = len(metrics)
    ncols = len(sigmas)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows), squeeze=False, constrained_layout=True)
    for c, sigma in enumerate(sigmas):
        sub_sigma = scalar_df if sigma is None else scalar_df[scalar_df["perturb_sigma"] == sigma]
        for r, (metric, label, _logy) in enumerate(metrics):
            ax = axes[r, c]
            for family in ["DRC-CE-HLSI", "CE-HLSI"]:
                sub = sub_sigma[sub_sigma["method_family"] == family].sort_values("iteration")
                y = pd.to_numeric(sub[metric], errors="coerce")
                if y.notna().sum() == 0:
                    continue
                ax.plot(sub["iteration"], y, marker="o", linewidth=1.8, linestyle=_line_style_for_family(family), label=family)
            ax.set_xlabel("bootstrap iteration k")
            ax.set_ylabel(label)
            if c == 0:
                ax.set_title(label)
            else:
                ax.set_title(f"sigma={float(sigma):g}")
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Tests 2--3: density-ratio coherence and correction decay | solid=DRC, dashed=CE", fontsize=13)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_hypothesis_gate_convergence(gate_df: pd.DataFrame, outpath: str) -> None:
    if gate_df.empty or "gate_rmse_frob" not in gate_df.columns:
        return
    metrics = [
        ("gate_rmse_frob", "gate Frobenius RMSE"),
        ("gate_eig_rmse", "gate eigenvalue RMSE"),
        ("gate_dominant_alignment_error", "dominant eigenspace error"),
    ]
    metrics = [m for m in metrics if m[0] in gate_df.columns]
    sigmas = sorted(gate_df["perturb_sigma"].unique()) if "perturb_sigma" in gate_df.columns else [None]
    nrows = len(metrics)
    ncols = len(sigmas)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows), squeeze=False, constrained_layout=True)
    for c, sigma in enumerate(sigmas):
        sub_sigma = gate_df if sigma is None else gate_df[gate_df["perturb_sigma"] == sigma]
        for r, (metric, label) in enumerate(metrics):
            ax = axes[r, c]
            for family in ["DRC-CE-HLSI", "CE-HLSI"]:
                sub = sub_sigma[sub_sigma["method_family"] == family].sort_values("iteration")
                y = pd.to_numeric(sub[metric], errors="coerce")
                if y.notna().sum() == 0:
                    continue
                ax.plot(sub["iteration"], y, marker="o", linewidth=1.8, linestyle=_line_style_for_family(family), label=family)
            ax.set_xlabel("bootstrap iteration k")
            ax.set_ylabel(label)
            ax.set_title(label if c == 0 else f"sigma={float(sigma):g}")
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Test 4: convergence to oracle CE-HLSI gate | solid=DRC, dashed=CE", fontsize=13)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

@torch.no_grad()
def run_method_or_bootstrap_chain(
    target: GMM2D,
    method: str,
    initial_refs: torch.Tensor,
    initial_gate_refs: torch.Tensor,
    cfg: ExperimentConfig,
    trial_seed: int,
    sigma_idx: int,
    method_idx: int,
) -> Tuple[torch.Tensor, Callable[[torch.Tensor, float], torch.Tensor], Dict[str, float | str | bool | int], List[Dict[str, float | str | bool | int]]]:
    """Run either one legacy method or an underscore-defined bootstrap chain.

    For method='Final_Previous', this executes Previous first on the corrupted
    reference bank, then rebuilds a fresh SNIS bank using the first cfg.n_ref
    generated samples as references and runs Final. Longer chains repeat this
    rule. This keeps cfg.n_ref decoupled from cfg.n_samples: samplers may
    generate more samples than are used as the next reference bank.
    """
    method_name = str(method)
    if method_name.upper() == "ORACLE":
        sample_seed = int(trial_seed + 10000 + 101 * sigma_idx + method_idx)
        sample_gen = make_generator(sample_seed, target.device)
        score_fn = lambda x, t: target.score(x, t=t)
        t0 = time.time()
        samples, sampler_info = reverse_ou_heun_sde(target, score_fn, cfg, generator=sample_gen)
        elapsed = time.time() - t0
        summary = {
            "bootstrap_depth": 1,
            "execution_order": "ORACLE",
            "final_stage_method": "ORACLE",
            "final_ref_n": 0,
            "elapsed_sec": elapsed,
            "sampler_failed": bool(sampler_info.get("failed", False)),
            "sampler_fail_reason": str(sampler_info.get("fail_reason", "")),
            "sampler_max_abs_score": safe_float(sampler_info.get("max_abs_score", float("nan"))),
        }
        stage_rows = [{
            "method": method_name,
            "bootstrap_depth": 1,
            "stage_index": 0,
            "stage_method": "ORACLE",
            "stage_ref_source": "none",
            "stage_ref_n": 0,
            "stage_seed": sample_seed,
            "stage_elapsed_sec": elapsed,
            "stage_sampler_failed": bool(sampler_info.get("failed", False)),
            "stage_sampler_fail_reason": str(sampler_info.get("fail_reason", "")),
            "stage_sampler_max_abs_score": safe_float(sampler_info.get("max_abs_score", float("nan"))),
        }]
        return samples, score_fn, summary, stage_rows

    execution_order = split_bootstrap_method(method_name)
    current_refs = initial_refs.detach()
    current_gate_refs = initial_gate_refs.detach()
    current_log_ref_weights = torch.zeros((current_refs.shape[0],), device=current_refs.device, dtype=current_refs.dtype)
    final_score_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None
    final_samples: Optional[torch.Tensor] = None
    final_sampler_info: Dict[str, float | str | bool] = {}
    total_elapsed = 0.0
    stage_rows: List[Dict[str, float | str | bool | int]] = []
    last_stage_ref_n = int(current_refs.shape[0])

    seed_base = int(trial_seed + 10000 + 101 * sigma_idx + method_idx)
    for stage_idx, stage_method in enumerate(execution_order):
        last_stage_ref_n = int(current_refs.shape[0])
        stage_is_drc = is_drc_method_token(stage_method)
        stage_bank = build_snis_bank(
            target,
            current_refs,
            current_gate_refs,
            cfg,
            log_ref_weights=current_log_ref_weights,
        )
        stage_score_method = score_method_for_stage(stage_method)
        score_fn = lambda x, t, bank=stage_bank, m=stage_score_method: bank.estimate(x, t=t, method=m)
        stage_seed = seed_base if stage_idx == 0 else int(seed_base + 1_000_003 * stage_idx)
        stage_gen = make_generator(stage_seed, target.device)
        stage_cfg = cfg
        if stage_is_drc and cfg.drc_disable_final_denoise and cfg.final_denoise:
            stage_cfg = replace(cfg, final_denoise=False)
        t0 = time.time()
        samples, sampler_info = reverse_ou_heun_sde(target, score_fn, stage_cfg, generator=stage_gen)
        elapsed = time.time() - t0
        total_elapsed += elapsed

        next_ref_n = min(int(cfg.n_ref), int(samples.shape[0]))
        drc_info: Dict[str, float | bool] = {
            "stage_is_drc": bool(stage_is_drc),
            "stage_score_method": str(stage_score_method),
            "stage_final_denoise_used": bool(stage_cfg.final_denoise),
            "stage_input_rho_ess": log_weight_ess(current_log_ref_weights)[0],
            "stage_input_rho_ess_frac": log_weight_ess(current_log_ref_weights)[1],
        }
        next_log_ref_weights = torch.zeros((next_ref_n,), device=current_refs.device, dtype=current_refs.dtype)
        if stage_is_drc and next_ref_n > 0:
            ratio_t0 = time.time()
            next_log_ref_weights, ratio_info = compute_drc_next_weights(target, stage_bank, samples[:next_ref_n].detach(), cfg)
            drc_info.update(ratio_info)
            drc_info["rho_update_elapsed_sec"] = float(time.time() - ratio_t0)

        stage_rows.append({
            "method": method_name,
            "bootstrap_depth": len(execution_order),
            "execution_order": " -> ".join(execution_order),
            "stage_index": int(stage_idx),
            "stage_method": str(stage_method),
            "stage_ref_source": "initial_corrupted_refs" if stage_idx == 0 else "previous_stage_samples_first_n_ref",
            "stage_ref_n": int(current_refs.shape[0]),
            "stage_generated_n": int(samples.shape[0]),
            "stage_next_ref_n": int(next_ref_n) if stage_idx < len(execution_order) - 1 else 0,
            "stage_seed": int(stage_seed),
            "stage_elapsed_sec": float(elapsed),
            "stage_sampler_failed": bool(sampler_info.get("failed", False)),
            "stage_sampler_fail_reason": str(sampler_info.get("fail_reason", "")),
            "stage_sampler_max_abs_score": safe_float(sampler_info.get("max_abs_score", float("nan"))),
            **drc_info,
        })

        final_samples = samples
        final_score_fn = score_fn
        final_sampler_info = sampler_info

        # Bootstrap update: only the first n_ref generated samples become the
        # next reference bank. This keeps n_samples and n_ref decoupled, e.g.
        # n_ref=500, n_samples=2000 generates 2000 final samples but passes
        # only samples[:500] into the next bootstrap layer.
        #
        # There is no separate paired clean bank after the first layer, so the
        # gate bank is set to the same generated references. Thus a later
        # Hybrid-CE-HLSI stage is well-defined but no longer uses the original
        # clean/perturbed pairing.
        if stage_idx < len(execution_order) - 1:
            current_refs = samples[:next_ref_n].detach()
            current_gate_refs = current_refs
            current_log_ref_weights = next_log_ref_weights.detach() if stage_is_drc else torch.zeros((next_ref_n,), device=current_refs.device, dtype=current_refs.dtype)

    if final_samples is None or final_score_fn is None:
        raise RuntimeError("Bootstrap chain produced no samples for method {}".format(method_name))

    summary = {
        "bootstrap_depth": int(len(execution_order)),
        "execution_order": " -> ".join(execution_order),
        "final_stage_method": str(execution_order[-1]),
        "final_ref_n": int(last_stage_ref_n),
        "elapsed_sec": float(total_elapsed),
        "sampler_failed": bool(final_sampler_info.get("failed", False)),
        "sampler_fail_reason": str(final_sampler_info.get("fail_reason", "")),
        "sampler_max_abs_score": safe_float(final_sampler_info.get("max_abs_score", float("nan"))),
    }
    return final_samples, final_score_fn, summary, stage_rows

def print_sigma_metric_summary(sigma: float, sigma_metrics_df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    """Print one compact trial-averaged metric summary for a perturbation level."""
    if sigma_metrics_df.empty:
        return
    metrics_to_print = [m for m in cfg.metrics if m in sigma_metrics_df.columns]
    if not metrics_to_print:
        return
    if "n_trials_observed" in sigma_metrics_df.columns:
        observed = pd.to_numeric(sigma_metrics_df["n_trials_observed"], errors="coerce")
    else:
        observed = pd.Series(1, index=sigma_metrics_df.index, dtype=float)
    n_trials = int(max(1, observed.max()))
    print(f"\n=== perturb_sigma={sigma:g} averaged over {n_trials} trial(s) ===", flush=True)
    preferred_order = ["PERTURBED_REF", *list(cfg.methods), "ORACLE"]
    order_rank = {name: i for i, name in enumerate(preferred_order)}
    printable_df = sigma_metrics_df.copy()
    printable_df["_order"] = printable_df["method"].map(lambda name: order_rank.get(str(name), len(order_rank)))
    printable_df = printable_df.sort_values(["_order", "method"])
    for _, row in printable_df.iterrows():
        method = str(row.get("method", ""))
        parts = []
        for m in metrics_to_print:
            formatted = format_mean_sem(row, m, n_trials)
            # PERTURBED_REF has no score estimator, so Fisher score error is not applicable.
            if method == "PERTURBED_REF" and formatted == "nan":
                continue
            parts.append(f"{m}={formatted}")
        print(f"  {method}: " + ", ".join(parts), flush=True)


def run_experiment(cfg: ExperimentConfig) -> None:
    cfg.metrics = canonicalize_metrics(cfg.metrics)
    cfg.n_trials = max(int(cfg.n_trials), 1)
    cfg.corruption_mode = normalize_corruption_mode(cfg.corruption_mode)
    cfg.n_ref = int(cfg.n_ref)
    cfg.n_samples = int(cfg.n_samples)
    needs_chained_refs = has_bootstrap_chains(list(cfg.methods)) or (
        bool(cfg.run_hypothesis_tests) and infer_hypothesis_max_depth(cfg) > 1
    )
    if needs_chained_refs and cfg.n_samples < cfg.n_ref:
        warnings.warn(
            "Bootstrap chains and multi-round hypothesis tests require n_samples >= n_ref "
            "so each stage can pass a full reference bank to the next stage. Resetting "
            f"n_samples from {cfg.n_samples} to n_ref={cfg.n_ref}.",
            RuntimeWarning,
        )
        cfg.n_samples = cfg.n_ref
    ensure_dir(cfg.outdir)
    # This script now emits exactly one histogram diagnostic.  Remove stale
    # per-sigma heatmaps from older runs in the same output directory so they
    # cannot be mistaken for current outputs.
    for _fname in list(os.listdir(cfg.outdir)):
        if _fname.startswith("sample_heatmaps_sigma_") and _fname.endswith(".png"):
            try:
                os.remove(os.path.join(cfg.outdir, _fname))
            except OSError:
                pass
    dtype = get_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    torch.set_default_dtype(dtype)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    target = GMM2D.default(device=device, dtype=dtype)

    methods = list(cfg.methods)
    if cfg.include_oracle and "ORACLE" not in methods:
        methods.append("ORACLE")

    with open(os.path.join(cfg.outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_rows: List[Dict[str, float | str | int | bool]] = []
    ess_rows: List[Dict[str, float | str | int]] = []
    bootstrap_rows: List[Dict[str, float | str | int | bool]] = []
    hypothesis_path_rows: List[Dict[str, float | str | int | bool]] = []
    hypothesis_scalar_rows: List[Dict[str, float | str | int | bool]] = []
    hypothesis_gate_rows: List[Dict[str, float | str | int | bool]] = []
    refs_by_sigma: Dict[float, torch.Tensor] = {}
    samples_by_sigma: Dict[float, Dict[str, torch.Tensor]] = {}

    for sigma_idx, sigma in enumerate(cfg.perturb_sigmas):
        print(
            f"\nRunning perturb_sigma={sigma:g} with corruption_mode={cfg.corruption_mode} "
            f"across {cfg.n_trials} trial(s)...",
            flush=True,
        )
        ref_plot_parts: List[torch.Tensor] = []
        sample_plot_parts: Dict[str, List[torch.Tensor]] = {method: [] for method in methods}
        sigma_rows: List[Dict[str, float | str | int | bool]] = []

        for trial_idx in range(cfg.n_trials):
            trial_seed = int(cfg.seed + 1_000_003 * trial_idx)

            # Fresh target/reference/metric randomness per trial.  This averages out
            # stochasticity from finite reference banks, sampling, histogram KL bins,
            # SW2 directions, and Fisher probe draws.
            trial_gen = make_generator(trial_seed + 17, device)
            truth = target.sample(cfg.n_truth, generator=trial_gen).detach()
            metric_gen = make_generator(trial_seed + 424242, device)
            metric_ctx = build_metric_context(target, truth, cfg, generator=metric_gen)
            clean_refs = target.sample(cfg.n_ref, generator=trial_gen).detach()

            pert_gen = make_generator(trial_seed + 1000 + sigma_idx, device)
            refs, corruption_info = perturb_reference_bank(
                clean_refs=clean_refs,
                sigma=float(sigma),
                corruption_mode=cfg.corruption_mode,
                generator=pert_gen,
            )
            ref_plot_parts.append(refs.detach().cpu())

            initial_bank = build_snis_bank(target, refs, clean_refs, cfg)

            if cfg.run_hypothesis_tests:
                print(
                    f"  Running DRC hypothesis tests for sigma={sigma:g}, trial={trial_idx}...",
                    flush=True,
                )
                h_path, h_scalar, h_gate = run_drc_hypothesis_tests_for_trial(
                    target=target,
                    initial_refs=refs,
                    initial_gate_refs=clean_refs,
                    cfg=cfg,
                    trial_seed=trial_seed,
                    sigma_idx=sigma_idx,
                )
                for hrow in h_path:
                    hypothesis_path_rows.append({
                        "perturb_sigma": float(sigma),
                        "trial": int(trial_idx),
                        "trial_seed": int(trial_seed),
                        "corruption_mode": cfg.corruption_mode,
                        "corruption_signal_scale": float(corruption_info["corruption_signal_scale"]),
                        "corruption_noise_scale": float(corruption_info["corruption_noise_scale"]),
                        **hrow,
                    })
                for hrow in h_scalar:
                    hypothesis_scalar_rows.append({
                        "perturb_sigma": float(sigma),
                        "trial": int(trial_idx),
                        "trial_seed": int(trial_seed),
                        "corruption_mode": cfg.corruption_mode,
                        "corruption_signal_scale": float(corruption_info["corruption_signal_scale"]),
                        "corruption_noise_scale": float(corruption_info["corruption_noise_scale"]),
                        **hrow,
                    })
                for hrow in h_gate:
                    hypothesis_gate_rows.append({
                        "perturb_sigma": float(sigma),
                        "trial": int(trial_idx),
                        "trial_seed": int(trial_seed),
                        "corruption_mode": cfg.corruption_mode,
                        "corruption_signal_scale": float(corruption_info["corruption_signal_scale"]),
                        "corruption_noise_scale": float(corruption_info["corruption_noise_scale"]),
                        **hrow,
                    })

            ref_metrics = compute_all_metrics(target, refs, metric_ctx, cfg)
            ref_rms_shift = float(torch.sqrt(torch.mean(torch.sum((refs - clean_refs) ** 2, dim=1))).item())
            ref_mean_shift = float(torch.mean(torch.linalg.norm(refs - clean_refs, dim=1)).item())
            ref_row = {
                "perturb_sigma": float(sigma),
                "trial": int(trial_idx),
                "trial_seed": int(trial_seed),
                "method": "PERTURBED_REF",
                "n_ref": int(cfg.n_ref),
                "bootstrap_depth": 0,
                "execution_order": "reference_only",
                "final_stage_method": "reference_only",
                "final_ref_n": int(cfg.n_ref),
                "rms_anchor_shift": ref_rms_shift,
                "mean_anchor_shift": ref_mean_shift,
                "metric_target": "fresh_unperturbed_target_pool_per_trial",
                **corruption_info,
                **({"fisher_rmse": float("nan")} if metric_enabled(cfg, "fisher_rmse") else {}),
                **ref_metrics,
            }
            all_rows.append(ref_row)
            sigma_rows.append(ref_row)

            sample_by_method: Dict[str, torch.Tensor] = {}
            for method_idx, method in enumerate(methods):
                samples, score_fn, run_info, stage_rows = run_method_or_bootstrap_chain(
                    target=target,
                    method=method,
                    initial_refs=refs,
                    initial_gate_refs=clean_refs,
                    cfg=cfg,
                    trial_seed=trial_seed,
                    sigma_idx=sigma_idx,
                    method_idx=method_idx,
                )
                sample_by_method[method] = samples
                sample_plot_parts[method].append(samples.detach().cpu())
                metrics = compute_all_metrics(target, samples, metric_ctx, cfg)
                fisher_metrics = integrated_score_fisher_metric(score_fn, metric_ctx, cfg)
                row = {
                    "perturb_sigma": float(sigma),
                    "trial": int(trial_idx),
                    "trial_seed": int(trial_seed),
                    "method": method,
                    "n_ref": int(cfg.n_ref),
                    "rms_anchor_shift": ref_rms_shift,
                    "mean_anchor_shift": ref_mean_shift,
                    "metric_target": "fresh_unperturbed_target_pool_per_trial",
                    **corruption_info,
                    **run_info,
                    **metrics,
                    **fisher_metrics,
                }
                all_rows.append(row)
                sigma_rows.append(row)

                for stage_row in stage_rows:
                    bootstrap_rows.append({
                        "perturb_sigma": float(sigma),
                        "trial": int(trial_idx),
                        "trial_seed": int(trial_seed),
                        "corruption_mode": cfg.corruption_mode,
                        "corruption_signal_scale": float(corruption_info["corruption_signal_scale"]),
                        "corruption_noise_scale": float(corruption_info["corruption_noise_scale"]),
                        **stage_row,
                    })

            for t_probe in [cfg.t_start, 1.5, 0.7, 0.25, 0.08, cfg.t_end]:
                y_probe = target.sample_pt(min(1024, cfg.n_samples), t_probe, generator=pert_gen)
                ess = initial_bank.ess(y_probe, t_probe)
                ess_rows.append(
                    {
                        "perturb_sigma": float(sigma),
                        "trial": int(trial_idx),
                        "trial_seed": int(trial_seed),
                        "corruption_mode": cfg.corruption_mode,
                        "bank": "initial_corrupted_reference_bank",
                        "t": float(t_probe),
                        "ess_mean": float(ess.mean().item()),
                        "ess_median": float(ess.median().item()),
                        "ess_min": float(ess.min().item()),
                        "ess_max": float(ess.max().item()),
                    }
                )

            npz_payload = {
                "truth": as_numpy(truth),
                "truth_eval_fixed": as_numpy(metric_ctx.truth_eval),
                "clean_refs": as_numpy(clean_refs),
                "perturbed_refs": as_numpy(refs),
            }
            for method, samples in sample_by_method.items():
                npz_payload[method.replace("-", "_")] = as_numpy(samples)
            if cfg.n_trials == 1:
                npz_name = f"samples_sigma_{sigma:g}.npz"
            else:
                npz_name = f"samples_sigma_{sigma:g}_trial_{trial_idx:03d}.npz"
            np.savez_compressed(os.path.join(cfg.outdir, npz_name), **npz_payload)

        refs_by_sigma[float(sigma)] = torch.cat(ref_plot_parts, dim=0)
        samples_by_sigma[float(sigma)] = {
            method: torch.cat(parts, dim=0) for method, parts in sample_plot_parts.items() if parts
        }

        # Per-sigma sample_heatmaps plots are intentionally disabled; the single
        # histogram diagnostic is emitted after all sigmas are complete.

        sigma_trials_df = pd.DataFrame(sigma_rows)
        sigma_trials_df = add_reference_contrasts(sigma_trials_df, cfg)
        sigma_metrics_df = aggregate_trials(sigma_trials_df, ["perturb_sigma", "method"])
        print_sigma_metric_summary(float(sigma), sigma_metrics_df, cfg)

    metrics_trials_df = pd.DataFrame(all_rows)
    metrics_trials_df = add_reference_contrasts(metrics_trials_df, cfg)
    metrics_df = aggregate_trials(metrics_trials_df, ["perturb_sigma", "method"])

    ess_trials_df = pd.DataFrame(ess_rows)
    ess_df = aggregate_trials(ess_trials_df, ["perturb_sigma", "t"]) if not ess_trials_df.empty else ess_trials_df
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    hypothesis_path_trials_df = pd.DataFrame(hypothesis_path_rows)
    hypothesis_scalar_trials_df = pd.DataFrame(hypothesis_scalar_rows)
    hypothesis_gate_trials_df = pd.DataFrame(hypothesis_gate_rows)
    hypothesis_path_df = (
        aggregate_trials(hypothesis_path_trials_df, ["perturb_sigma", "method_family", "iteration", "t", "path_metric"])
        if not hypothesis_path_trials_df.empty else hypothesis_path_trials_df
    )
    hypothesis_scalar_df = (
        aggregate_trials(hypothesis_scalar_trials_df, ["perturb_sigma", "method_family", "iteration"])
        if not hypothesis_scalar_trials_df.empty else hypothesis_scalar_trials_df
    )
    hypothesis_gate_df = (
        aggregate_trials(hypothesis_gate_trials_df, ["perturb_sigma", "method_family", "iteration"])
        if not hypothesis_gate_trials_df.empty else hypothesis_gate_trials_df
    )

    metrics_trials_csv = os.path.join(cfg.outdir, "metrics_trials.csv")
    metrics_csv = os.path.join(cfg.outdir, "metrics.csv")
    ess_trials_csv = os.path.join(cfg.outdir, "ess_diagnostics_trials.csv")
    ess_csv = os.path.join(cfg.outdir, "ess_diagnostics.csv")
    bootstrap_csv = os.path.join(cfg.outdir, "bootstrap_stages.csv")
    hypothesis_path_trials_csv = os.path.join(cfg.outdir, "hypothesis_path_self_consistency_trials.csv")
    hypothesis_path_csv = os.path.join(cfg.outdir, "hypothesis_path_self_consistency.csv")
    hypothesis_scalar_trials_csv = os.path.join(cfg.outdir, "hypothesis_density_ratio_tests_trials.csv")
    hypothesis_scalar_csv = os.path.join(cfg.outdir, "hypothesis_density_ratio_tests.csv")
    hypothesis_gate_trials_csv = os.path.join(cfg.outdir, "hypothesis_gate_convergence_trials.csv")
    hypothesis_gate_csv = os.path.join(cfg.outdir, "hypothesis_gate_convergence.csv")
    metrics_trials_df.to_csv(metrics_trials_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    ess_trials_df.to_csv(ess_trials_csv, index=False)
    ess_df.to_csv(ess_csv, index=False)
    bootstrap_df.to_csv(bootstrap_csv, index=False)
    if cfg.run_hypothesis_tests:
        hypothesis_path_trials_df.to_csv(hypothesis_path_trials_csv, index=False)
        hypothesis_path_df.to_csv(hypothesis_path_csv, index=False)
        hypothesis_scalar_trials_df.to_csv(hypothesis_scalar_trials_csv, index=False)
        hypothesis_scalar_df.to_csv(hypothesis_scalar_csv, index=False)
        hypothesis_gate_trials_df.to_csv(hypothesis_gate_trials_csv, index=False)
        hypothesis_gate_df.to_csv(hypothesis_gate_csv, index=False)

    plot_metric_array(
        metrics_df,
        [m for m in cfg.metrics if m in metrics_df.columns],
        os.path.join(cfg.outdir, "meta_metric_array.png"),
    )
    plot_contrast_sweeps(metrics_df, os.path.join(cfg.outdir, "contrast_vs_reference.png"))
    if any("ORACLE" in by_method for by_method in samples_by_sigma.values()):
        plot_histogram_array_over_sigmas(
            target=target,
            refs_by_sigma=refs_by_sigma,
            samples_by_sigma=samples_by_sigma,
            method_order=methods,
            cfg=cfg,
            outpath=os.path.join(cfg.outdir, "histogram_array_over_sigmas.png"),
        )
    else:
        warnings.warn(
            "Skipping histogram_array_over_sigmas.png because ORACLE samples are unavailable.",
            RuntimeWarning,
        )

    if cfg.run_hypothesis_tests:
        plot_hypothesis_path_self_consistency(
            hypothesis_path_df,
            os.path.join(cfg.outdir, "hypothesis_path_self_consistency.png"),
        )
        plot_hypothesis_density_ratio_scalars(
            hypothesis_scalar_df,
            os.path.join(cfg.outdir, "hypothesis_density_ratio_tests.png"),
        )
        plot_hypothesis_gate_convergence(
            hypothesis_gate_df,
            os.path.join(cfg.outdir, "hypothesis_gate_convergence.png"),
        )

    if not ess_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for sigma in sorted(ess_df["perturb_sigma"].unique()):
            sub = ess_df[ess_df["perturb_sigma"] == sigma].sort_values("t")
            y = pd.to_numeric(sub["ess_median"], errors="coerce")
            yerr = pd.to_numeric(sub.get("ess_median_sem", pd.Series(index=sub.index, dtype=float)), errors="coerce")
            if yerr.notna().sum() > 0 and cfg.n_trials > 1:
                ax.errorbar(sub["t"], y, yerr=yerr, marker="o", linewidth=1.8, capsize=2, label=f"sigma={sigma:g}")
            else:
                ax.plot(sub["t"], y, marker="o", label=f"sigma={sigma:g}")
        ax.set_xscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("median ESS")
        ax.set_title(f"OU SNIS ESS under {cfg.corruption_mode} reference corruption")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(os.path.join(cfg.outdir, "ess_vs_t.png"), dpi=180)
        plt.close(fig)

    print(
        f"\nDone. Wrote:\n"
        f"  {metrics_csv}              # trial-averaged metrics\n"
        f"  {metrics_trials_csv}       # raw per-trial metrics\n"
        f"  {ess_csv}                  # trial-averaged ESS diagnostics\n"
        f"  {ess_trials_csv}           # raw per-trial ESS diagnostics\n"
        f"  {bootstrap_csv}            # per-stage bootstrap diagnostics\n"
        + (
            f"  {hypothesis_path_csv}  # Test 1 path self-consistency\n"
            f"  {hypothesis_scalar_csv} # Tests 2--3 density-ratio coherence/decay\n"
            f"  {hypothesis_gate_csv}   # Test 4 oracle-gate convergence\n"
            if cfg.run_hypothesis_tests else ""
        )
        + f"  {cfg.outdir}"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", type=str, default=ExperimentConfig.outdir)
    p.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    p.add_argument("--device", type=str, default=ExperimentConfig.device)
    p.add_argument("--dtype", type=str, default=ExperimentConfig.dtype)
    p.add_argument("--n_ref", type=int, default=ExperimentConfig.n_ref)
    p.add_argument("--n_samples", type=int, default=ExperimentConfig.n_samples, help="Number of generated samples per sampler stage. For bootstrap chains, if n_samples > n_ref only the first n_ref generated samples feed the next stage; if n_samples < n_ref, n_samples is reset to n_ref with a warning.")
    p.add_argument("--n_truth", type=int, default=ExperimentConfig.n_truth)
    p.add_argument("--metrics_max_n", type=int, default=ExperimentConfig.metrics_max_n)
    p.add_argument("--n_trials", type=int, default=ExperimentConfig.n_trials, help="Number of independent trials per perturbation sigma. Each trial redraws reference banks, metric probes, and sampler seeds; metrics.csv reports trial means and SEM columns.")
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=list(ExperimentConfig.metrics),
        help=f"Metrics to compute. Use any of: {', '.join(AVAILABLE_METRICS)}. Aliases accepted: sliced_w2->sw2, score_rmse->fisher_rmse, old KL names->kl/kl_rev.",
    )
    p.add_argument("--perturb_sigmas", type=float, nargs="+", default=list(ExperimentConfig.perturb_sigmas))
    p.add_argument("--corruption_mode", type=str, choices=["heat", "ou"], default=ExperimentConfig.corruption_mode, help="Reference corruption mode. heat: x+sigma*z. ou: exp(-sigma)*x+sqrt(1-exp(-2*sigma))*z, interpreting perturb_sigma as OU time.")
    p.add_argument("--t_start", type=float, default=ExperimentConfig.t_start)
    p.add_argument("--t_end", type=float, default=ExperimentConfig.t_end)
    p.add_argument("--n_steps", type=int, default=ExperimentConfig.n_steps)
    p.add_argument("--n_corrector", type=int, default=ExperimentConfig.n_corrector)
    p.add_argument("--corrector_snr", type=float, default=ExperimentConfig.corrector_snr)
    p.add_argument("--corrector_step_max", type=float, default=ExperimentConfig.corrector_step_max)
    p.add_argument("--init_mode", type=str, choices=["exact_pt", "normal"], default=ExperimentConfig.init_mode)
    p.add_argument("--no_final_denoise", action="store_true")
    p.add_argument("--sample_clip", type=float, default=ExperimentConfig.sample_clip)
    p.add_argument("--score_clip", type=float, default=ExperimentConfig.score_clip)
    p.add_argument("--fisher_n_t", type=int, default=ExperimentConfig.fisher_n_t)
    p.add_argument("--fisher_n_per_t", type=int, default=ExperimentConfig.fisher_n_per_t)
    p.add_argument("--fisher_t_min", type=float, default=None)
    p.add_argument("--fisher_t_max", type=float, default=None)
    p.add_argument("--fisher_time_grid", type=str, choices=["log", "linear"], default=ExperimentConfig.fisher_time_grid)
    p.add_argument("--methods", type=str, nargs="+", default=list(ExperimentConfig.methods))
    p.add_argument("--include_oracle", action="store_true")
    p.add_argument("--no_oracle", action="store_true")
    p.add_argument("--curvature_mode", type=str, choices=["raw", "psd", "abs"], default=ExperimentConfig.curvature_mode)
    p.add_argument("--curvature_floor", type=float, default=ExperimentConfig.curvature_floor)
    p.add_argument("--curvature_cap", type=float, default=ExperimentConfig.curvature_cap)
    p.add_argument("--resolvent_eps", type=float, default=ExperimentConfig.resolvent_eps)
    p.add_argument("--gate_clip", type=float, default=ExperimentConfig.gate_clip if ExperimentConfig.gate_clip is not None else 0.0)
    p.add_argument("--no_gate_clip", action="store_true")
    p.add_argument("--weight_temp", type=float, default=ExperimentConfig.weight_temp)
    p.add_argument("--eval_chunk", type=int, default=ExperimentConfig.eval_chunk)
    p.add_argument("--op_blend_reg", type=float, default=ExperimentConfig.op_blend_reg)
    p.add_argument("--op_blend_pinv_rtol", type=float, default=ExperimentConfig.op_blend_pinv_rtol)
    p.add_argument("--op_blend_project_gate", action="store_true")
    p.add_argument("--pf_steps", type=int, default=ExperimentConfig.pf_steps, help="Probability-flow Heun steps for DRC log q estimates.")
    p.add_argument("--pf_t_start", type=float, default=None, help="Start time for DRC probability-flow logq; defaults to t_end.")
    p.add_argument("--pf_t_end", type=float, default=None, help="Terminal time for DRC probability-flow logq; defaults to t_start.")
    p.add_argument("--rho_beta", type=float, default=ExperimentConfig.rho_beta, help="Tempering exponent for DRC log density-ratio weights.")
    p.add_argument("--rho_clip", type=float, default=ExperimentConfig.rho_clip if ExperimentConfig.rho_clip is not None else 0.0, help="Absolute clip for centered DRC log density-ratio weights; use --no_rho_clip to disable.")
    p.add_argument("--no_rho_clip", action="store_true")
    p.add_argument("--rho_ess_floor", type=float, default=ExperimentConfig.rho_ess_floor, help="Minimum ESS/N for DRC weights; beta is shrunk if needed.")
    p.add_argument("--rho_batch", type=int, default=ExperimentConfig.rho_batch, help="Batch size for DRC probability-flow logq estimates.")
    p.add_argument("--drc_final_denoise", action="store_true", help="Use final Tweedie denoising inside DRC sampler stages. Default disables it so path-integral ratios match sampler endpoints.")
    p.add_argument("--pf_div_clip", type=float, default=ExperimentConfig.pf_div_clip if ExperimentConfig.pf_div_clip is not None else 0.0, help="Absolute clip for analytic divergence during DRC probability-flow integration; use --no_pf_div_clip to disable.")
    p.add_argument("--no_pf_div_clip", action="store_true")
    p.add_argument("--run_hypothesis_tests", action="store_true", help="Run the four computational hypothesis tests for the DRC-CE-HLSI claim: path self-consistency, PF/KDE correction coherence, correction decay, and oracle-gate convergence.")
    p.add_argument("--hypothesis_max_depth", type=int, default=ExperimentConfig.hypothesis_max_depth, help="Maximum CE/DRC bootstrap iteration k for hypothesis tests. Use 0 to infer from --methods.")
    p.add_argument("--hypothesis_n_eval", type=int, default=ExperimentConfig.hypothesis_n_eval, help="Number of samples used for path, KDE, and PF correction diagnostics.")
    p.add_argument("--hypothesis_n_path_t", type=int, default=ExperimentConfig.hypothesis_n_path_t, help="Number of t values for Test 1 path-marginal self-consistency.")
    p.add_argument("--hypothesis_path_metric", type=str, choices=["mmd", "sw2", "hist_l1", "hist_l2"], default=ExperimentConfig.hypothesis_path_metric, help="Distributional discrepancy for Test 1.")
    p.add_argument("--hypothesis_kde_bandwidth", type=float, default=None, help="Gaussian KDE bandwidth for Test 2. Defaults to median distance times Scott factor.")
    p.add_argument("--hypothesis_oracle_n_ref", type=int, default=ExperimentConfig.hypothesis_oracle_n_ref, help="Target reference-bank size for oracle CE-HLSI gate in Test 4.")
    p.add_argument("--hypothesis_gate_n_t", type=int, default=ExperimentConfig.hypothesis_gate_n_t, help="Number of t values for Test 4 gate convergence.")
    p.add_argument("--hypothesis_gate_n_per_t", type=int, default=ExperimentConfig.hypothesis_gate_n_per_t, help="Number of y~pi_t probes per t for Test 4 gate convergence.")
    p.add_argument("--mmd_bandwidth", type=float, default=None)
    p.add_argument("--ksd_bandwidth", type=float, default=None)
    p.add_argument("--sw2_projections", type=int, default=ExperimentConfig.sw2_projections)
    p.add_argument("--grid_lim", type=float, default=ExperimentConfig.grid_lim)
    p.add_argument("--grid_n", type=int, default=ExperimentConfig.grid_n)
    p.add_argument("--hist_bins", type=int, default=ExperimentConfig.hist_bins)
    p.add_argument("--hist_cmap", type=str, default=ExperimentConfig.hist_cmap)
    p.add_argument("--hist_gamma", type=float, default=ExperimentConfig.hist_gamma, help="PowerNorm gamma for histogram brightness; smaller values brighten low-count pixels.")
    p.add_argument("--hist_vmax_quantile", type=float, default=ExperimentConfig.hist_vmax_quantile, help="Shared high-count quantile used as histogram vmax; avoids one hot bin making all panels faint.")
    p.add_argument("--residual_vmax_quantile", type=float, default=ExperimentConfig.residual_vmax_quantile, help="Quantile of absolute residual values used for the shared red/blue residual scale.")
    p.add_argument("--residual_intensity", type=float, default=ExperimentConfig.residual_intensity, help="Multiplier that brightens residual panels by dividing the residual vmax. Values >1 increase residual pixel intensity.")
    p.add_argument("--hist_contour_alpha", type=float, default=ExperimentConfig.hist_contour_alpha)
    p.add_argument("--hist_contour_lw", type=float, default=ExperimentConfig.hist_contour_lw)
    p.add_argument("--no_plots_per_sigma", action="store_true")
    args = p.parse_args()

    include_oracle = ExperimentConfig.include_oracle
    if args.include_oracle:
        include_oracle = True
    if args.no_oracle:
        include_oracle = False

    gate_clip = None if args.no_gate_clip else args.gate_clip
    rho_clip = None if args.no_rho_clip else args.rho_clip
    pf_div_clip = None if args.no_pf_div_clip else args.pf_div_clip

    return ExperimentConfig(
        outdir=args.outdir,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        n_ref=args.n_ref,
        n_samples=args.n_samples,
        n_truth=args.n_truth,
        metrics_max_n=args.metrics_max_n,
        metrics=canonicalize_metrics(args.metrics),
        n_trials=max(int(args.n_trials), 1),
        perturb_sigmas=tuple(args.perturb_sigmas),
        corruption_mode=args.corruption_mode,
        t_start=args.t_start,
        t_end=args.t_end,
        n_steps=args.n_steps,
        n_corrector=args.n_corrector,
        corrector_snr=args.corrector_snr,
        corrector_step_max=args.corrector_step_max,
        init_mode=args.init_mode,
        final_denoise=not args.no_final_denoise,
        sample_clip=args.sample_clip,
        score_clip=args.score_clip,
        fisher_n_t=args.fisher_n_t,
        fisher_n_per_t=args.fisher_n_per_t,
        fisher_t_min=args.fisher_t_min,
        fisher_t_max=args.fisher_t_max,
        fisher_time_grid=args.fisher_time_grid,
        methods=tuple(args.methods),
        include_oracle=include_oracle,
        curvature_mode=args.curvature_mode,
        curvature_floor=args.curvature_floor,
        curvature_cap=args.curvature_cap,
        resolvent_eps=args.resolvent_eps,
        gate_clip=gate_clip,
        weight_temp=args.weight_temp,
        eval_chunk=args.eval_chunk,
        op_blend_reg=args.op_blend_reg,
        op_blend_pinv_rtol=args.op_blend_pinv_rtol,
        op_blend_project_gate=args.op_blend_project_gate,
        pf_steps=args.pf_steps,
        pf_t_start=args.pf_t_start,
        pf_t_end=args.pf_t_end,
        rho_beta=args.rho_beta,
        rho_clip=rho_clip,
        rho_ess_floor=args.rho_ess_floor,
        rho_batch=args.rho_batch,
        drc_disable_final_denoise=not args.drc_final_denoise,
        pf_div_clip=pf_div_clip,
        run_hypothesis_tests=bool(args.run_hypothesis_tests),
        hypothesis_max_depth=max(int(args.hypothesis_max_depth), 0),
        hypothesis_n_eval=max(int(args.hypothesis_n_eval), 1),
        hypothesis_n_path_t=max(int(args.hypothesis_n_path_t), 2),
        hypothesis_path_metric=args.hypothesis_path_metric,
        hypothesis_kde_bandwidth=args.hypothesis_kde_bandwidth,
        hypothesis_oracle_n_ref=max(int(args.hypothesis_oracle_n_ref), 1),
        hypothesis_gate_n_t=max(int(args.hypothesis_gate_n_t), 1),
        hypothesis_gate_n_per_t=max(int(args.hypothesis_gate_n_per_t), 1),
        mmd_bandwidth=args.mmd_bandwidth,
        ksd_bandwidth=args.ksd_bandwidth,
        sw2_projections=args.sw2_projections,
        grid_lim=args.grid_lim,
        grid_n=args.grid_n,
        hist_bins=args.hist_bins,
        hist_cmap=args.hist_cmap,
        hist_gamma=args.hist_gamma,
        hist_vmax_quantile=args.hist_vmax_quantile,
        residual_vmax_quantile=args.residual_vmax_quantile,
        residual_intensity=args.residual_intensity,
        hist_contour_alpha=args.hist_contour_alpha,
        hist_contour_lw=args.hist_contour_lw,
        plot_every_sigma=not args.no_plots_per_sigma,
    )


if __name__ == "__main__":
    run_experiment(parse_args())