#!/usr/bin/env python3
"""
Robustness of SNIS score estimators to misspecified reference samples on a 2D GMM.

Implements SNIS versions of the main estimators:
  - Tweedie
  - TSI
  - Blend / Blended (scalar variance-optimal blend)
  - OP-Blend (full operator-valued empirical variance-optimal blend)
  - HLSI (componentwise resolvent gate)
  - CE-HLSI (empirical conditional-expectation gate)
  - Hybrid-CE-HLSI (perturbed score signals with clean-reference CE gate)
  - optional ORACLE sampler

Experiment:
  1. Draw clean references X_i ~ p_0 from a known 2D GMM.
  2. Perturb anchors: X_i^eps = X_i + sigma * N(0,I).
  3. Query exact score and exact observed information H=-∇²log p at those anchors.
  4. Build SNIS estimators using the perturbed anchors. Hybrid-CE-HLSI
     uses perturbed anchors for Tweedie/TSI score signals but clean anchors for
     the CE attenuation gate.
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
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    # If true, all sample-vs-target metrics use one fixed unperturbed target
    # evaluation pool and fixed SW2/Fisher probes for the entire perturbation sweep.
    fixed_metric_pool: bool = True

    # Sweep
    perturb_sigmas: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)

    # Reverse OU PC sampler
    t_start: float = 3.0
    t_end: float = 1.0e-3
    n_steps: int = 160
    n_corrector: int = 1
    corrector_snr: float = 0.08
    corrector_step_max: float = 2.0e-3
    init_mode: str = "exact_pt"  # exact_pt or normal
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
    methods: Tuple[str, ...] = ("Tweedie", "TSI", "Blend", "OP-Blend", "HLSI", "CE-HLSI", "Hybrid-CE-HLSI")
    include_oracle: bool = True
    curvature_mode: str = "raw"  # raw, psd, abs
    curvature_floor: float = 1.0e-5
    curvature_cap: float = 1.0e6
    resolvent_eps: float = 1.0e-8
    gate_clip: Optional[float] = 50.0
    weight_temp: float = 1.0
    eval_chunk: int = 512
    op_blend_reg: float = 1.0e-8
    op_blend_pinv_rtol: float = 1.0e-6
    op_blend_project_gate: bool = False

    # Metrics
    mmd_bandwidth: Optional[float] = None
    ksd_bandwidth: Optional[float] = None
    sw2_projections: int = 256

    # Plotting
    grid_lim: float = 5.0
    grid_n: int = 180
    hist_bins: int = 90
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
        gate_anchors: Optional[torch.Tensor] = None,
        op_blend_reg: float = 1.0e-8,
        op_blend_pinv_rtol: float = 1.0e-6,
        op_blend_project_gate: bool = False,
    ):
        self.target = target

        # Score bank: always the possibly perturbed anchors. These determine
        # SNIS weights/signals b,c for Tweedie, TSI, Blend, HLSI, CE-HLSI, and
        # the signal side of Hybrid-CE-HLSI.
        self.x = anchors.detach()
        self.N, self.d = self.x.shape
        self.device = self.x.device
        self.dtype = self.x.dtype
        self.score0 = target.score(self.x, t=0.0).detach()
        self.H_raw = target.observed_information(self.x, t=0.0).detach()
        self.P = process_curvature(self.H_raw, curvature_mode, curvature_floor, curvature_cap).detach()

        # Gate bank: optional clean/unperturbed anchors used only by
        # Hybrid-CE-HLSI.  Its OU weights and curvature average define the
        # attenuation matrix while the score signal remains the perturbed-bank
        # bbar/cbar above.
        self.x_gate = None if gate_anchors is None else gate_anchors.detach()
        if self.x_gate is not None:
            self.H_gate_raw = target.observed_information(self.x_gate, t=0.0).detach()
            self.P_gate = process_curvature(self.H_gate_raw, curvature_mode, curvature_floor, curvature_cap).detach()
        else:
            self.H_gate_raw = None
            self.P_gate = None

        self.resolvent_eps = resolvent_eps
        self.gate_clip = gate_clip
        self.weight_temp = weight_temp
        self.eval_chunk = eval_chunk
        self.op_blend_reg = op_blend_reg
        self.op_blend_pinv_rtol = op_blend_pinv_rtol
        self.op_blend_project_gate = op_blend_project_gate

    def _weights_from_anchors(self, y: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        diff = y[:, None, :] - alpha * anchors[None, :, :]
        logw = -0.5 * torch.sum(diff * diff, dim=-1) / gamma
        if self.weight_temp != 1.0:
            logw = logw / self.weight_temp
        logw = logw - torch.max(logw, dim=1, keepdim=True).values
        w = torch.exp(logw)
        return w / torch.clamp(w.sum(dim=1, keepdim=True), min=1.0e-300)

    def _weights_and_signals(self, y: torch.Tensor, t: float):
        t_tensor = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        alpha, gamma = alpha_gamma(t_tensor)
        w = self._weights_from_anchors(y, alpha, gamma, self.x)
        b = (alpha * self.x[None, :, :] - y[:, None, :]) / gamma
        c = self.score0[None, :, :] / alpha
        return w, b, c, alpha, gamma

    def estimate_chunk(self, y: torch.Tensor, t: float, method: str) -> torch.Tensor:
        w, b, c, alpha, gamma = self._weights_and_signals(y, t)
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)

        if method == "Tweedie":
            return bbar
        if method == "TSI":
            return cbar
        if method in ("Blend", "Blended"):
            wb = w * w
            db = b - bbar[:, None, :]
            dc = c - cbar[:, None, :]
            sig_b = torch.sum(wb * torch.sum(db * db, dim=-1), dim=1)
            sig_c = torch.sum(wb * torch.sum(dc * dc, dim=-1), dim=1)
            rho = torch.sum(wb * torch.sum(db * dc, dim=-1), dim=1)
            denom = torch.clamp(sig_b + sig_c - 2.0 * rho, min=1.0e-30)
            g_tsi = torch.clamp((sig_b - rho) / denom, min=0.0, max=1.0)
            return bbar + g_tsi[:, None] * (cbar - bbar)
        if method == "OP-Blend":
            # Sandbox-style full operator blend: cbar + G(bbar-cbar), where
            # G minimizes empirical conditional variance of TSI + G(Tweedie-TSI).
            D = b - c
            Dbar = bbar - cbar
            Ac = c - cbar[:, None, :]
            Dc = D - Dbar[:, None, :]
            C_AD = torch.einsum("bn,bni,bnj->bij", w, Ac, Dc)
            C_DD = torch.einsum("bn,bni,bnj->bij", w, Dc, Dc)
            C_DD = 0.5 * (C_DD + C_DD.transpose(-1, -2))
            B, d = y.shape
            I = torch.eye(d, device=y.device, dtype=y.dtype).expand(B, d, d)
            scale = C_DD.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1.0)
            C_DD_solve = C_DD + (self.op_blend_reg * scale).view(B, 1, 1) * I
            G = -torch.matmul(C_AD, torch.linalg.pinv(C_DD_solve, rtol=self.op_blend_pinv_rtol))
            if self.op_blend_project_gate:
                eig, U = torch.linalg.eigh(0.5 * (G + G.transpose(-1, -2)))
                eig = eig.clamp(min=0.0, max=1.0)
                G = U @ torch.diag_embed(eig) @ U.transpose(-1, -2)
            return cbar + torch.einsum("bij,bj->bi", G, Dbar)
        if method == "HLSI":
            G_i = resolvent_gate(self.P, alpha, gamma, self.resolvent_eps, self.gate_clip)
            corr = torch.einsum("nij,bnj->bni", G_i, (c - b))
            h_i = b + corr
            return torch.sum(w[:, :, None] * h_i, dim=1)
        if method == "CE-HLSI":
            Pbar = torch.sum(w[:, :, None, None] * self.P[None, :, :, :], dim=1)
            Gbar = resolvent_gate(Pbar, alpha, gamma, self.resolvent_eps, self.gate_clip)
            return bbar + torch.einsum("bij,bj->bi", Gbar, (cbar - bbar))
        if method == "Hybrid-CE-HLSI":
            if self.x_gate is None or self.P_gate is None:
                raise ValueError("Hybrid-CE-HLSI requires gate_anchors=clean_refs when constructing SNISScoreBank.")
            w_gate = self._weights_from_anchors(y, alpha, gamma, self.x_gate)
            Pbar_gate = torch.sum(w_gate[:, :, None, None] * self.P_gate[None, :, :, :], dim=1)
            Gbar_gate = resolvent_gate(Pbar_gate, alpha, gamma, self.resolvent_eps, self.gate_clip)
            return bbar + torch.einsum("bij,bj->bi", Gbar_gate, (cbar - bbar))
        raise ValueError(f"Unknown method={method}")

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
def pc_reverse_sampler(target: GMM2D, score_fn: Callable[[torch.Tensor, float], torch.Tensor], cfg: ExperimentConfig, generator: torch.Generator) -> torch.Tensor:
    device = target.device
    dtype = target.dtype
    if cfg.init_mode == "exact_pt":
        x = target.sample_pt(cfg.n_samples, cfg.t_start, generator=generator)
    elif cfg.init_mode == "normal":
        x = torch.randn((cfg.n_samples, 2), device=device, dtype=dtype, generator=generator)
    else:
        raise ValueError("init_mode must be exact_pt or normal")

    times = torch.linspace(cfg.t_start, cfg.t_end, cfg.n_steps + 1, device=device, dtype=dtype)
    for k in range(cfg.n_steps):
        t_cur = float(times[k].item())
        h = float((times[k] - times[k + 1]).item())

        # Corrector steps: ULA targeting p_t.
        for _ in range(cfg.n_corrector):
            score = clamp_norm(score_fn(x, t_cur), cfg.score_clip)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            grad_norm = torch.mean(torch.linalg.norm(score, dim=1))
            noise_norm = torch.mean(torch.linalg.norm(noise, dim=1))
            step = 2.0 * (cfg.corrector_snr * noise_norm / (grad_norm + 1.0e-12)) ** 2
            step = torch.clamp(step, min=1.0e-7, max=cfg.corrector_step_max)
            x = x + step * score + torch.sqrt(2.0 * step) * noise
            if cfg.sample_clip and cfg.sample_clip > 0:
                x = torch.clamp(x, min=-cfg.sample_clip, max=cfg.sample_clip)

        # Predictor for reverse OU SDE.
        score = clamp_norm(score_fn(x, t_cur), cfg.score_clip)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = x + h * (x + 2.0 * score) + math.sqrt(2.0 * h) * noise
        if cfg.sample_clip and cfg.sample_clip > 0:
            x = torch.clamp(x, min=-cfg.sample_clip, max=cfg.sample_clip)

    if cfg.final_denoise:
        alpha, gamma = alpha_gamma(torch.tensor(cfg.t_end, device=device, dtype=dtype))
        score = clamp_norm(score_fn(x, cfg.t_end), cfg.score_clip)
        x = (x + gamma * score) / alpha
        if cfg.sample_clip and cfg.sample_clip > 0:
            x = torch.clamp(x, min=-cfg.sample_clip, max=cfg.sample_clip)

    return x.detach()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


@torch.no_grad()
def nll_metric(target: GMM2D, x: torch.Tensor) -> float:
    return float((-target.log_prob(x, t=0.0)).mean().item())


@torch.no_grad()
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: Optional[float] = None) -> float:
    """Biased multiscale RBF MMD diagnostic.

    The earlier unbiased single-band estimator can become negative at finite n and
    then clamp to zero, which is visually misleading in robustness sweeps.  This
    biased multiscale version is more stable for comparing sampler degradation.
    """
    if bandwidth is None:
        bandwidth = median_bandwidth(x, y)
    base = float(bandwidth)
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)
    mmd2 = torch.zeros((), device=x.device, dtype=x.dtype)
    for scale in scales:
        h2 = (base * scale) ** 2
        Kxx = torch.exp(-dxx / (2.0 * h2))
        Kyy = torch.exp(-dyy / (2.0 * h2))
        Kxy = torch.exp(-dxy / (2.0 * h2))
        mmd2 = mmd2 + Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    mmd2 = mmd2 / len(scales)
    return float(torch.sqrt(torch.clamp(mmd2, min=0.0)).item())


@torch.no_grad()
def sliced_w2(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256, generator: Optional[torch.Generator] = None, dirs: Optional[torch.Tensor] = None) -> float:
    d = x.shape[1]
    if dirs is None:
        dirs = torch.randn((n_proj, d), device=x.device, dtype=x.dtype, generator=generator)
        dirs = dirs / torch.clamp(torch.linalg.norm(dirs, dim=1, keepdim=True), min=1.0e-12)
    else:
        dirs = dirs.to(device=x.device, dtype=x.dtype)
    px = x @ dirs.T
    py = y @ dirs.T
    n = min(px.shape[0], py.shape[0])
    # Deterministic truncation. The truth pool itself is randomly drawn once upstream
    # and then held fixed across the entire perturbation sweep.
    px = px[:n]
    py = py[:n]
    sx = torch.sort(px, dim=0).values
    sy = torch.sort(py, dim=0).values
    return float(torch.sqrt(torch.clamp(torch.mean((sx - sy) ** 2), min=0.0)).item())


@torch.no_grad()
def ksd_rbf(target: GMM2D, x: torch.Tensor, bandwidth: Optional[float] = None) -> float:
    n, d = x.shape
    if bandwidth is None:
        bandwidth = median_bandwidth(x)
    h2 = bandwidth * bandwidth
    h4 = h2 * h2
    s = target.score(x, t=0.0)
    dxy = x[:, None, :] - x[None, :, :]
    r2 = torch.sum(dxy * dxy, dim=-1)
    K = torch.exp(-r2 / (2.0 * h2))
    ss = s @ s.T
    sd = torch.einsum("id,ijd->ij", s, dxy)
    ds = torch.einsum("jd,ijd->ij", s, dxy)
    stein = K * (ss + (sd - ds) / h2 + d / h2 - r2 / h4)
    if n > 1:
        val = (stein.sum() - stein.diag().sum()) / (n * (n - 1))
    else:
        val = stein.mean()
    return float(torch.sqrt(torch.clamp(val, min=0.0)).item())


@dataclass
class MetricContext:
    """Fixed metric state shared by every perturbation and method.

    This prevents accidental comparisons against perturbed references and removes
    metric noise from changing target pools, changing SW2 projections, or changing
    Fisher probe samples across methods.
    """
    truth_eval: torch.Tensor
    sw2_dirs: torch.Tensor
    ksd_bandwidth: float
    mmd_bandwidth: float
    fisher_times: torch.Tensor
    fisher_y: List[torch.Tensor]
    fisher_score: List[torch.Tensor]


@torch.no_grad()
def build_metric_context(target: GMM2D, truth_pool: torch.Tensor, cfg: ExperimentConfig, generator: torch.Generator) -> MetricContext:
    n_eval = min(cfg.metrics_max_n, truth_pool.shape[0])
    # Fixed unperturbed target pool, held constant across every perturbation sigma.
    truth_eval = truth_pool[:n_eval].detach()

    dirs = torch.randn((cfg.sw2_projections, truth_eval.shape[1]), device=truth_eval.device, dtype=truth_eval.dtype, generator=generator)
    dirs = dirs / torch.clamp(torch.linalg.norm(dirs, dim=1, keepdim=True), min=1.0e-12)

    ksd_bw = float(cfg.ksd_bandwidth) if cfg.ksd_bandwidth is not None else median_bandwidth(truth_eval)
    mmd_bw = float(cfg.mmd_bandwidth) if cfg.mmd_bandwidth is not None else median_bandwidth(truth_eval)

    fisher_times = make_fisher_time_grid(cfg, target.device, target.dtype)
    fisher_y: List[torch.Tensor] = []
    fisher_score: List[torch.Tensor] = []
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
    return {
        "nll": nll_metric(target, x),
        "ksd": ksd_rbf(target, x, bandwidth=metric_ctx.ksd_bandwidth),
        "mmd": mmd_rbf(x, y, bandwidth=metric_ctx.mmd_bandwidth),
        "sw2": sliced_w2(x, y, n_proj=cfg.sw2_projections, dirs=metric_ctx.sw2_dirs),
        "metric_target_n": float(n),
    }


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
    if cfg.fisher_n_t <= 0 or cfg.fisher_n_per_t <= 0:
        return {"fisher_mse": float("nan"), "fisher_rmse": float("nan")}
    mse_vals = []
    for t_tensor, y, s_true in zip(metric_ctx.fisher_times, metric_ctx.fisher_y, metric_ctx.fisher_score):
        t = float(t_tensor.item())
        s_hat = score_fn(y, t)
        err2 = torch.sum((s_hat - s_true) ** 2, dim=1)
        mse_vals.append(torch.mean(err2))
    mse = float(torch.mean(torch.stack(mse_vals)).item())
    return {"fisher_mse": mse, "fisher_rmse": float(math.sqrt(max(mse, 0.0)))}


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


def _draw_hist_panel(ax, arr: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, Pg: np.ndarray, cfg: ExperimentConfig) -> None:
    ax.contour(Xg, Yg, Pg, levels=8, linewidths=0.6, alpha=0.55)
    ax.hist2d(
        arr[:, 0],
        arr[:, 1],
        bins=cfg.hist_bins,
        range=[[-cfg.grid_lim, cfg.grid_lim], [-cfg.grid_lim, cfg.grid_lim]],
        density=True,
    )
    ax.set_xlim(-cfg.grid_lim, cfg.grid_lim)
    ax.set_ylim(-cfg.grid_lim, cfg.grid_lim)
    ax.set_aspect("equal")



def plot_metric_array(metrics_df: pd.DataFrame, metric_names: List[str], outpath: str) -> None:
    metric_names = [m for m in metric_names if m in metrics_df.columns]
    if not metric_names:
        return
    ncols = min(3, len(metric_names))
    nrows = int(math.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.1 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    methods = list(metrics_df["method"].unique())
    for ax, metric in zip(axes, metric_names):
        for method in methods:
            sub = metrics_df[metrics_df["method"] == method].sort_values("perturb_sigma")
            if sub.empty or metric not in sub.columns:
                continue
            y = pd.to_numeric(sub[metric], errors="coerce")
            if y.notna().sum() == 0:
                continue
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
    metric_names = ["ksd", "nll", "mmd", "sw2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    sample_df = metrics_df[metrics_df["method"] != "PERTURBED_REF"].copy()
    for ax, metric in zip(axes, metric_names):
        col = f"delta_vs_ref_{metric}"
        if col not in sample_df.columns:
            continue
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        for method in sample_df["method"].unique():
            sub = sample_df[sample_df["method"] == method].sort_values("perturb_sigma")
            ax.plot(sub["perturb_sigma"], sub[col], marker="o", linewidth=1.8, label=method)
        ax.set_xlabel("reference perturbation sigma")
        ax.set_ylabel(f"{metric.upper()} - reference {metric.upper()}")
        ax.set_title("contrastive improvement (<0 is better)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def plot_samples_for_sigma(target: GMM2D, ref: torch.Tensor, sample_by_method: Dict[str, torch.Tensor], sigma: float, cfg: ExperimentConfig, outpath: str) -> None:
    Xg, Yg, Pg = density_grid(target, cfg.grid_lim, cfg.grid_n)
    panels: List[Tuple[str, Optional[torch.Tensor], str]] = [("target density", None, "density"), ("perturbed refs", ref, "hist")]
    for method, samples in sample_by_method.items():
        panels.append((method, samples, "hist"))
    ncols = 4
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for ax, (title, pts, kind) in zip(axes, panels):
        ax.contour(Xg, Yg, Pg, levels=8, linewidths=0.7, alpha=0.65)
        if kind == "density":
            ax.imshow(Pg, origin="lower", extent=[-cfg.grid_lim, cfg.grid_lim, -cfg.grid_lim, cfg.grid_lim], aspect="equal", alpha=0.85)
        else:
            arr = as_numpy(pts)
            _draw_hist_panel(ax, arr, Xg, Yg, Pg, cfg)
        ax.set_xlim(-cfg.grid_lim, cfg.grid_lim)
        ax.set_ylim(-cfg.grid_lim, cfg.grid_lim)
        ax.set_aspect("equal")
        ax.set_title(title)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Reference perturbation sigma = {sigma:g}", fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def plot_histogram_array_over_sigmas(
    target: GMM2D,
    refs_by_sigma: Dict[float, torch.Tensor],
    samples_by_sigma: Dict[float, Dict[str, torch.Tensor]],
    method_order: List[str],
    cfg: ExperimentConfig,
    outpath: str,
) -> None:
    """Grid with rows = methods (plus perturbed refs), cols = perturbation sigmas."""
    sigmas = sorted(refs_by_sigma.keys())
    row_names = ["PERTURBED_REF"] + method_order
    Xg, Yg, Pg = density_grid(target, cfg.grid_lim, cfg.grid_n)

    nrows = len(row_names)
    ncols = len(sigmas)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for j, sigma in enumerate(sigmas):
        for i, row_name in enumerate(row_names):
            ax = axes[i, j]
            if row_name == "PERTURBED_REF":
                arr = as_numpy(refs_by_sigma[sigma])
            else:
                arr = as_numpy(samples_by_sigma[sigma][row_name])
            _draw_hist_panel(ax, arr, Xg, Yg, Pg, cfg)
            if i == 0:
                ax.set_title(f"sigma={sigma:g}")
            if j == 0:
                ax.set_ylabel(row_name)
    fig.suptitle("Histogram array over reference perturbation levels", fontsize=14)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def run_experiment(cfg: ExperimentConfig) -> None:
    ensure_dir(cfg.outdir)
    dtype = get_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    torch.set_default_dtype(dtype)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    target = GMM2D.default(device=device, dtype=dtype)
    gen = make_generator(cfg.seed, device)
    # This is the large, separate, unperturbed target pool. It is generated once
    # and then held fixed for MMD/SW2 and all sample-vs-target metrics.
    truth = target.sample(cfg.n_truth, generator=gen).detach()
    metric_gen = make_generator(cfg.seed + 424242, device)
    metric_ctx = build_metric_context(target, truth, cfg, generator=metric_gen)
    clean_refs = target.sample(cfg.n_ref, generator=gen).detach()

    methods = list(cfg.methods)
    if cfg.include_oracle and "ORACLE" not in methods:
        methods.append("ORACLE")

    with open(os.path.join(cfg.outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_rows: List[Dict[str, float | str]] = []
    ess_rows: List[Dict[str, float | str]] = []
    refs_by_sigma: Dict[float, torch.Tensor] = {}
    samples_by_sigma: Dict[float, Dict[str, torch.Tensor]] = {}

    for sigma_idx, sigma in enumerate(cfg.perturb_sigmas):
        print(f"\n=== perturb_sigma={sigma:g} ===", flush=True)
        pert_gen = make_generator(cfg.seed + 1000 + sigma_idx, device)
        noise = torch.randn(clean_refs.shape, device=device, dtype=dtype, generator=pert_gen)
        refs = (clean_refs + float(sigma) * noise).detach()
        refs_by_sigma[float(sigma)] = refs

        bank = SNISScoreBank(
            target=target,
            anchors=refs,
            curvature_mode=cfg.curvature_mode,
            curvature_floor=cfg.curvature_floor,
            curvature_cap=cfg.curvature_cap,
            resolvent_eps=cfg.resolvent_eps,
            gate_clip=cfg.gate_clip,
            weight_temp=cfg.weight_temp,
            eval_chunk=cfg.eval_chunk,
            gate_anchors=clean_refs,
            op_blend_reg=cfg.op_blend_reg,
            op_blend_pinv_rtol=cfg.op_blend_pinv_rtol,
            op_blend_project_gate=cfg.op_blend_project_gate,
        )

        ref_metrics = compute_all_metrics(target, refs, metric_ctx, cfg)
        ref_rms_shift = float(torch.sqrt(torch.mean(torch.sum((refs - clean_refs) ** 2, dim=1))).item())
        ref_mean_shift = float(torch.mean(torch.linalg.norm(refs - clean_refs, dim=1)).item())
        ref_row = {
            "perturb_sigma": float(sigma),
            "method": "PERTURBED_REF",
            "rms_anchor_shift": ref_rms_shift,
            "mean_anchor_shift": ref_mean_shift,
            "metric_target": "fixed_unperturbed_target_pool",
            "fisher_mse": float("nan"),
            "fisher_rmse": float("nan"),
            **ref_metrics,
        }
        all_rows.append(ref_row)
        print("reference metrics:", {k: round(v, 5) for k, v in ref_metrics.items()}, flush=True)

        sample_by_method: Dict[str, torch.Tensor] = {}
        for method_idx, method in enumerate(methods):
            print(f"  sampling {method}...", flush=True)
            sample_gen = make_generator(cfg.seed + 10000 + 101 * sigma_idx + method_idx, device)
            if method == "ORACLE":
                score_fn = lambda x, t: target.score(x, t=t)
            else:
                score_fn = lambda x, t, m=method: bank.estimate(x, t=t, method=m)
            t0 = time.time()
            samples = pc_reverse_sampler(target, score_fn, cfg, generator=sample_gen)
            elapsed = time.time() - t0
            sample_by_method[method] = samples
            metrics = compute_all_metrics(target, samples, metric_ctx, cfg)
            fisher_metrics = integrated_score_fisher_metric(score_fn, metric_ctx, cfg)
            row = {
                "perturb_sigma": float(sigma),
                "method": method,
                "elapsed_sec": elapsed,
                "rms_anchor_shift": ref_rms_shift,
                "mean_anchor_shift": ref_mean_shift,
                "metric_target": "fixed_unperturbed_target_pool",
                **metrics,
                **fisher_metrics,
            }
            all_rows.append(row)
            printable = {**metrics, **fisher_metrics}
            print(f"    {method} metrics:", {k: round(v, 5) for k, v in printable.items()}, f"elapsed={elapsed:.1f}s", flush=True)

        samples_by_sigma[float(sigma)] = sample_by_method

        for t_probe in [cfg.t_start, 1.5, 0.7, 0.25, 0.08, cfg.t_end]:
            y_probe = target.sample_pt(min(1024, cfg.n_samples), t_probe, generator=pert_gen)
            ess = bank.ess(y_probe, t_probe)
            ess_rows.append(
                {
                    "perturb_sigma": float(sigma),
                    "t": float(t_probe),
                    "ess_mean": float(ess.mean().item()),
                    "ess_median": float(ess.median().item()),
                    "ess_min": float(ess.min().item()),
                    "ess_max": float(ess.max().item()),
                }
            )

        if cfg.plot_every_sigma:
            plot_samples_for_sigma(
                target=target,
                ref=refs,
                sample_by_method=sample_by_method,
                sigma=float(sigma),
                cfg=cfg,
                outpath=os.path.join(cfg.outdir, f"sample_heatmaps_sigma_{sigma:g}.png"),
            )

        npz_payload = {"truth": as_numpy(truth), "truth_eval_fixed": as_numpy(metric_ctx.truth_eval), "perturbed_refs": as_numpy(refs)}
        for method, samples in sample_by_method.items():
            npz_payload[method.replace("-", "_")] = as_numpy(samples)
        np.savez_compressed(os.path.join(cfg.outdir, f"samples_sigma_{sigma:g}.npz"), **npz_payload)

    metrics_df = pd.DataFrame(all_rows)
    for metric in ["ksd", "nll", "mmd", "sw2"]:
        ref_map = metrics_df[metrics_df["method"] == "PERTURBED_REF"].set_index("perturb_sigma")[metric].to_dict()
        metrics_df[f"delta_vs_ref_{metric}"] = metrics_df.apply(
            lambda r, m=metric: float(r[m] - ref_map.get(r["perturb_sigma"], np.nan)), axis=1
        )
        metrics_df[f"ratio_vs_ref_{metric}"] = metrics_df.apply(
            lambda r, m=metric: float(r[m] / max(ref_map.get(r["perturb_sigma"], np.nan), 1.0e-12)), axis=1
        )

    ess_df = pd.DataFrame(ess_rows)
    metrics_csv = os.path.join(cfg.outdir, "metrics.csv")
    ess_csv = os.path.join(cfg.outdir, "ess_diagnostics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    ess_df.to_csv(ess_csv, index=False)

    plot_metric_array(metrics_df, ["ksd", "nll", "mmd", "sw2", "fisher_rmse"], os.path.join(cfg.outdir, "meta_metric_array.png"))
    plot_contrast_sweeps(metrics_df, os.path.join(cfg.outdir, "contrast_vs_reference.png"))
    plot_histogram_array_over_sigmas(
        target=target,
        refs_by_sigma=refs_by_sigma,
        samples_by_sigma=samples_by_sigma,
        method_order=methods,
        cfg=cfg,
        outpath=os.path.join(cfg.outdir, "histogram_array_over_sigmas.png"),
    )

    if not ess_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for sigma in sorted(ess_df["perturb_sigma"].unique()):
            sub = ess_df[ess_df["perturb_sigma"] == sigma].sort_values("t")
            ax.plot(sub["t"], sub["ess_median"], marker="o", label=f"sigma={sigma:g}")
        ax.set_xscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("median ESS")
        ax.set_title("OU SNIS effective sample size under perturbed reference banks")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(os.path.join(cfg.outdir, "ess_vs_t.png"), dpi=180)
        plt.close(fig)

    print(f"\nDone. Wrote:\n  {metrics_csv}\n  {ess_csv}\n  {cfg.outdir}")


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
    p.add_argument("--n_samples", type=int, default=ExperimentConfig.n_samples)
    p.add_argument("--n_truth", type=int, default=ExperimentConfig.n_truth)
    p.add_argument("--metrics_max_n", type=int, default=ExperimentConfig.metrics_max_n)
    p.add_argument("--perturb_sigmas", type=float, nargs="+", default=list(ExperimentConfig.perturb_sigmas))
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
    p.add_argument("--mmd_bandwidth", type=float, default=None)
    p.add_argument("--ksd_bandwidth", type=float, default=None)
    p.add_argument("--sw2_projections", type=int, default=ExperimentConfig.sw2_projections)
    p.add_argument("--grid_lim", type=float, default=ExperimentConfig.grid_lim)
    p.add_argument("--grid_n", type=int, default=ExperimentConfig.grid_n)
    p.add_argument("--hist_bins", type=int, default=ExperimentConfig.hist_bins)
    p.add_argument("--no_plots_per_sigma", action="store_true")
    args = p.parse_args()

    include_oracle = ExperimentConfig.include_oracle
    if args.include_oracle:
        include_oracle = True
    if args.no_oracle:
        include_oracle = False

    gate_clip = None if args.no_gate_clip else args.gate_clip

    return ExperimentConfig(
        outdir=args.outdir,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        n_ref=args.n_ref,
        n_samples=args.n_samples,
        n_truth=args.n_truth,
        metrics_max_n=args.metrics_max_n,
        perturb_sigmas=tuple(args.perturb_sigmas),
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
        mmd_bandwidth=args.mmd_bandwidth,
        ksd_bandwidth=args.ksd_bandwidth,
        sw2_projections=args.sw2_projections,
        grid_lim=args.grid_lim,
        grid_n=args.grid_n,
        hist_bins=args.hist_bins,
        plot_every_sigma=not args.no_plots_per_sigma,
    )


if __name__ == "__main__":
    run_experiment(parse_args())
