#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alternating_drc_lfgi_vs_blend_misaligned8d.py

Compare alternating density-ratio-corrected bootstrapping from isotropic
Gaussian-prior or oracle-target reference samples on a selectable normalized target.

Targets now include the original d=8 misaligned singular-subspace GMM,
the d=10 Neal funnel stress test from the benchmark sweep, and intermediate-dimensional
molecular LJ/DW-style particle potentials with exact t=0 score and Hessian evaluations.

Purpose
-------
This is the exact test described in the chat:

  * Targets: choose with --target.  The default is the original d=8, K=8,
    rank=3 misaligned near-singular subspace GMM.  The --target funnel_d10 option
    adds the benchmark-sweep Neal funnel example.  The molecular option
    --target lj13_2d builds a 26d LJ13-like bonded cluster with manifold
    concentration, heterogeneous anisotropy, and exact autograd Hessians.
  * Normalization: targets are affinely whitened so E[X]≈0 and Cov[X]≈I.
    Therefore the initial N(0,I) prior bank has the same global location and
    volume as the target, but it is not a target sample bank unless --initial_reference_mode target is used.
  * Inputs available to the estimators: prior/reference coordinates, target
    energy/log-density, target gradient/score, and target Hessian at those
    coordinates.  Target samples are used for evaluation metrics/plots, and optionally for oracle initial references with --initial_reference_mode target.
  * Methods: selected with --methods.  Atomic estimators include Blend,
    CE-HLSI/LFGI, MP-Leaf-LFGI, Tweedie, and None.  Hybrid tokens use
    transport_correction order, e.g. blend_lfgi, lfgi_blend, tweedie_lfgi,
    lfgi_none, none_lfgi.
  * Algorithm: alternating projected-IPF-style DRC:

        (R_j, rho_j) --S(score estimator)--> q_j samples R_{j+1}
        q_j --R(probability-flow logq)--> rho_{j+1} = log pi - log q_j

    The same number of S/R rounds is run for Blend and CE-HLSI.  The R-step for
    CE-HLSI uses the analytic CE-HLSI divergence; the R-step for Blend uses a
    Hutchinson finite-difference divergence of the same frozen Blend field.
    Use --pf_divergence hutchinson to force both methods through the generic
    divergence estimator.

Outputs
-------
  outdir/config.json
  outdir/metrics_by_round.csv
  outdir/stage_diagnostics.csv  # includes PF-vs-KDE likelihood correction calibration
  outdir/heatmaps_final.png
  outdir/heatmaps_by_round.png
  outdir/metric_curves.png
  outdir/projection_basis.npz

Example
-------
python alternating_drc_lfgi_vs_blend_targets.py \
  --target misaligned_gmm \
  --outdir results/alt_drc_misaligned8d \
  --device cuda --dtype float64 \
  --n_ref 3000 --n_samples 3000 --n_truth 12000 \
  --n_rounds 4 --n_steps 150 --pf_steps 64 \
  --t_min 0.005 --t_max 3.0 --time_schedule linear \
  --methods blend_blend,blend_lfgi,lfgi_blend,lfgi_lfgi,tweedie_lfgi,lfgi_none,none_lfgi

python alternating_drc_lfgi_vs_blend_targets.py \
  --target lj13_2d \
  --outdir results/alt_drc_lj13_2d \
  --device cuda --dtype float64 \
  --n_ref 3000 --n_samples 3000 --n_truth 12000 \
  --n_rounds 4 --n_steps 150 --pf_steps 64 \
  --mol_sample_steps 900 --mol_norm_samples 4096 --mol_score_bank 8192 \
  --methods blend_blend,blend_lfgi,lfgi_blend,lfgi_lfgi

Quick smoke test
----------------
python alternating_drc_lfgi_vs_blend_misaligned8d.py \
  --outdir /tmp/alt_drc_smoke --device cpu --dtype float32 \
  --n_ref 128 --n_samples 128 --n_truth 512 \
  --n_rounds 1 --n_steps 8 --pf_steps 4 --metrics_max_n 128 \
  --fisher_n_t 2 --fisher_n_per_t 64 --eval_chunk 64 --rho_batch 64
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
from matplotlib.colors import PowerNorm
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Config:
    outdir: str = "results/alternating_drc_misaligned8d"
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"

    # Target selector.  Use misaligned_gmm/gmm8 for the original benchmark;
    # lj13_2d/molecular for a 26d LJ13-like bonded cluster; dw4_16d for a
    # smaller 16d molecular stress test.
    target: str = "misaligned_gmm"

    # Original misaligned singular-subspace GMM parameters
    d: int = 8
    rank: int = 3
    n_components: int = 8
    target_seed: int = 29
    radius: float = 3.0
    sigma_perp: float = 0.035
    jitter: float = 0.12
    normalize_target: bool = True

    # Neal funnel target parameters.  The default eta^2=6 matches the benchmark
    # sweep's d=10 funnel stress test; normalize_target controls diagonal
    # whitening of the exact funnel moments for fairer N(0,I) initialization.
    funnel_d: int = 10
    funnel_eta2: float = 6.0
    funnel_score_bank: int = 8192
    funnel_score_chunk: int = 512

    # Molecular LJ/DW-style target parameters.  The defaults define a 2D LJ13-like
    # cluster: 13 particles x 2 coordinates = 26 dimensions.  The target is a
    # Boltzmann density exp(-beta U) with soft-core LJ pair interactions, stiff
    # heterogeneous nearest-neighbor bonds around a hexagonal reference cluster,
    # weak rotationally invariant confinement, and a COM pinning penalty.  This
    # produces the intended non-Gaussian, anisotropic, nearly-manifold regime while
    # keeping exact scores and Hessians available by autodiff.
    mol_n_particles: int = 13
    mol_particle_dim: int = 2
    mol_beta: float = 1.0
    mol_lj_eps: float = 0.18
    mol_lj_sigma: float = 1.0
    mol_lj_soft_core: float = 0.08
    mol_bond_k: float = 80.0
    mol_confinement_k: float = 0.015
    mol_com_k: float = 6.0
    mol_init_noise: float = 0.18
    mol_sample_steps: int = 800
    mol_sample_step_size: float = 2.0e-4
    mol_sample_batch: int = 512
    mol_norm_samples: int = 2048
    mol_norm_eig_floor: float = 1.0e-4
    mol_score_bank: int = 4096
    mol_score_chunk: int = 256
    mol_hessian_chunk: int = 16

    # Reference/evaluation sizes
    n_ref: int = 3000
    # Number of proposal-bank samples used to estimate gate objects.
    # gate_n <= 0 defaults to n_ref.
    gate_n: int = 0
    # Score/gate bank coupling:
    #   shared      : gate bank is exactly the score bank; gate_n ignored.
    #   prefix      : gate bank is current_pool[:gate_n], score bank is current_pool[:n_ref].
    #                 Thus gate_n<n_ref gives a prefix subset, and gate_n>n_ref adds gate-only samples.
    #   independent : gate bank is current_pool[:gate_n], score bank is current_pool[gate_n:gate_n+n_ref].
    bank_coupling: str = "shared"
    n_samples: int = 3000
    n_truth: int = 12000
    metrics_max_n: int = 2000

    # Alternating rounds
    n_rounds: int = 4
    # Initial proposal/reference law:
    #   prior/gaussian : draw the initial split-compatible pool from N(0,I).
    #   target/oracle  : draw the initial split-compatible pool from the target.
    #                    This is an oracle-reference stability test; initial
    #                    density-ratio weights are forced to zero because q0=p0.
    initial_reference_mode: str = "prior"
    initial_weight_mode: str = "prior_ratio"  # prior_ratio or zero; ignored for target initial references
    # Comma-separated estimator pairs to run. Atomic aliases such as blend,
    # lfgi/ce-hlsi, or leaf-lfgi/mp-leaf-lfgi mean diagonal transport/correction.
    # Hybrid aliases use transport_correction order, e.g. blend_lfgi or lfgi_blend.
    # Special values:
    #   all/default = diagonal blend, lfgi, leaf-lfgi, tweedie
    #   hybrids     = four blend/lfgi pairs
    #   grid/full   = full transport/correction grid over blend, lfgi, leaf-lfgi, tweedie, none
    methods: str = "blend_blend,blend_lfgi,lfgi_blend,lfgi_lfgi"

    # Reverse OU sampler / probability-flow time interval.
    # The old names t_start/t_end are kept as aliases for backward compatibility;
    # t_max/t_min are the canonical knobs used below.
    t_min: float = 0.005
    t_max: float = 3.0
    time_schedule: str = "linear"  # linear or log_linear
    t_start: float = 3.0  # legacy alias for t_max
    t_end: float = 0.005  # legacy alias for t_min
    n_steps: int = 150
    final_denoise: bool = False  # keep PF-compatible endpoint law by default
    eval_final_denoise: bool = False
    sample_clip: float = 25.0
    score_clip: float = 250.0

    # SNIS / gates
    curvature_mode: str = "raw"  # raw, psd, abs
    curvature_floor: float = -1.0e6
    curvature_cap: float = 1.0e6
    resolvent_eps: float = 1.0e-8
    gate_clip: float = 50.0
    # Minimal MP-leaf precision completion for Leaf-LFGI.  The completed
    # gate precision is Q = V diag(max(lambda, mp_leaf_floor)) V^T.
    # This is the fixed-floor/moment-preserving correction: score signals are
    # unchanged, but the CE/LFGI gate sees the PSD-completed precision.
    mp_leaf_floor: float = 0.0
    mp_leaf_tol: float = 1.0e-12
    weight_temp: float = 1.0
    eval_chunk: int = 512

    # Density-ratio / probability-flow
    pf_steps: int = 64
    rho_batch: int = 512
    rho_beta: float = 1.0
    rho_clip: float = 20.0
    rho_ess_floor: float = 0.02
    pf_div_clip: float = 1.0e4
    pf_divergence: str = "auto"  # auto, analytic_ce, hutchinson
    hutchinson_probes: int = 1
    hutchinson_eps: float = 1.0e-3

    # Likelihood-correction calibration against particle KDE in full d=8.
    # The diagnostic compares PF log q and rho=log pi-log q against a
    # leave-one-out Gaussian KDE fit to the same generated proposal bank.
    likelihood_calibration: bool = True
    kde_n_eval: int = 1000
    kde_n_fit: int = 3000
    kde_bandwidth: float = 0.0  # <=0: median distance * Scott factor
    kde_min_bandwidth: float = 0.05
    kde_chunk: int = 256

    # Score RMSE metric
    fisher_n_t: int = 12
    fisher_n_per_t: int = 512
    fisher_time_grid: str = "log"  # log or linear

    # MMD / KSD / SW2 / plots
    sw2_projections: int = 256
    hist_bins: int = 90
    hist_gamma: float = 0.45
    hist_vmax_quantile: float = 0.995


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in {"float64", "double", "fp64"}:
        return torch.float64
    if key in {"float32", "single", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def canonical_bank_coupling(value: str) -> str:
    """Normalize user-facing score/gate bank-coupling aliases."""
    key = str(value or "shared").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "shared": "shared",
        "same": "shared",
        "same-bank": "shared",
        "score": "shared",
        "prefix": "prefix",
        "prefix-coupled": "prefix",
        "prefix-coupling": "prefix",
        "coupled": "prefix",
        "subset": "prefix",
        "independent": "independent",
        "indep": "independent",
        "indpendent": "independent",
        "disjoint": "independent",
        "separate": "independent",
        "decoupled": "independent",
    }
    if key not in aliases:
        raise ValueError(f"Unknown bank_coupling={value!r}; use shared, prefix, or independent")
    return aliases[key]


def canonical_initial_reference_mode(value: str) -> str:
    """Normalize user-facing aliases for the initial proposal/reference law."""
    key = str(value or "prior").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "prior": "prior",
        "gaussian": "prior",
        "normal": "prior",
        "naive": "prior",
        "naive-gaussian": "prior",
        "n0i": "prior",
        "n-0-i": "prior",
        "target": "target",
        "oracle": "target",
        "oracle-target": "target",
        "truth": "target",
        "true": "target",
        "p0": "target",
    }
    if key not in aliases:
        raise ValueError(f"Unknown initial_reference_mode={value!r}; use prior or target")
    return aliases[key]


def make_initial_reference_pool(target, cfg: Config, n_pool: int, generator: torch.Generator) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Draw the initial split-compatible proposal/reference pool.

    ``prior`` is the original naive Gaussian start.  ``target`` is an oracle
    start: the proposal/reference coordinates are target samples, so the
    correct initial density ratio is zero before the first alternating update.
    """
    mode = canonical_initial_reference_mode(cfg.initial_reference_mode)
    n_pool = int(n_pool)
    if mode == "prior":
        x = torch.randn((n_pool, int(target.d)), device=target.device, dtype=target.dtype, generator=generator)
    elif mode == "target":
        x = target.sample(n_pool, generator=generator).detach()
    else:
        raise RuntimeError(f"Unhandled initial_reference_mode={mode!r}")
    return x.detach(), {
        "initial_reference_mode": mode,
        "initial_reference_n": int(x.shape[0]),
    }


def effective_gate_n(cfg: Config) -> int:
    """Return the actual number of gate-bank samples requested.

    In shared mode the gate bank is exactly the score bank, so a user-supplied
    gate_n is intentionally ignored and the effective value is n_ref.
    """
    n_ref = int(cfg.n_ref)
    if canonical_bank_coupling(cfg.bank_coupling) == "shared":
        return n_ref
    g = int(cfg.gate_n) if int(cfg.gate_n) > 0 else n_ref
    if g <= 0:
        raise ValueError(f"gate_n must be positive after defaulting; got gate_n={cfg.gate_n}, n_ref={cfg.n_ref}")
    return g


def proposal_pool_size(cfg: Config) -> int:
    """Number of proposal samples needed to materialize score and gate banks."""
    n_ref = int(cfg.n_ref)
    gate_n = effective_gate_n(cfg)
    mode = canonical_bank_coupling(cfg.bank_coupling)
    if mode == "shared":
        return n_ref
    if mode == "prefix":
        return max(n_ref, gate_n)
    if mode == "independent":
        return n_ref + gate_n
    raise RuntimeError(f"Unhandled bank_coupling={mode!r}")


def split_score_gate_banks(pool_x: torch.Tensor, pool_logw: torch.Tensor, cfg: Config):
    """Split a proposal pool into score-signal and gate-estimation banks.

    Semantics are intentionally simple and match the requested alternating-DRC
    flags:
      * shared:      score = gate = X[0:n_ref].
      * prefix:      gate = X[0:gate_n], score = X[0:n_ref].
      * independent: gate = X[0:gate_n], score = X[gate_n:gate_n+n_ref].
    """
    n_ref = int(cfg.n_ref)
    gate_n = effective_gate_n(cfg)
    mode = canonical_bank_coupling(cfg.bank_coupling)
    need = proposal_pool_size(cfg)
    if int(pool_x.shape[0]) < need:
        raise ValueError(f"Proposal pool has {pool_x.shape[0]} samples but bank_coupling={mode} requires {need}")
    lw = pool_logw.detach().reshape(-1)
    if int(lw.shape[0]) != int(pool_x.shape[0]):
        raise ValueError(f"pool_logw has length {lw.shape[0]} but pool_x has length {pool_x.shape[0]}")
    if mode == "shared":
        score_x = pool_x[:n_ref].contiguous()
        score_w = lw[:n_ref].contiguous()
        gate_x = score_x
        gate_w = score_w
        overlap_n = n_ref
        score_slice = f"[0:{n_ref}]"
        gate_slice = f"[0:{n_ref}]"
    elif mode == "prefix":
        score_x = pool_x[:n_ref].contiguous()
        score_w = lw[:n_ref].contiguous()
        gate_x = pool_x[:gate_n].contiguous()
        gate_w = lw[:gate_n].contiguous()
        overlap_n = min(n_ref, gate_n)
        score_slice = f"[0:{n_ref}]"
        gate_slice = f"[0:{gate_n}]"
    elif mode == "independent":
        gate_x = pool_x[:gate_n].contiguous()
        gate_w = lw[:gate_n].contiguous()
        score_x = pool_x[gate_n:gate_n + n_ref].contiguous()
        score_w = lw[gate_n:gate_n + n_ref].contiguous()
        overlap_n = 0
        score_slice = f"[{gate_n}:{gate_n + n_ref}]"
        gate_slice = f"[0:{gate_n}]"
    else:
        raise RuntimeError(f"Unhandled bank_coupling={mode!r}")
    info = {
        "bank_coupling": mode,
        "score_n": int(score_x.shape[0]),
        "gate_n": int(gate_x.shape[0]),
        "pool_n": int(need),
        "bank_overlap_n": int(overlap_n),
        "score_slice": score_slice,
        "gate_slice": gate_slice,
    }
    return score_x, score_w, gate_x, gate_w, info


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def safe_float(x) -> float:
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().item()
        x = float(x)
        if not math.isfinite(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def alpha_gamma(t: float | torch.Tensor, *, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=device, dtype=dtype)
    else:
        if device is not None or dtype is not None:
            t = t.to(device=device if device is not None else t.device, dtype=dtype if dtype is not None else t.dtype)
    alpha = torch.exp(-t)
    gamma = 1.0 - torch.exp(-2.0 * t)
    return alpha, gamma


def canonical_time_schedule(value: str) -> str:
    """Normalize sampler/PF time-grid aliases."""
    key = str(value or "linear").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "linear": "linear",
        "lin": "linear",
        "uniform": "linear",
        "uniform_t": "linear",
        "log_linear": "log_linear",
        "loglinear": "log_linear",
        "log": "log_linear",
        "log_t": "log_linear",
        "geometric": "log_linear",
        "geom": "log_linear",
    }
    if key not in aliases:
        raise ValueError(f"Unknown time_schedule={value!r}; use linear or log_linear")
    return aliases[key]


def effective_time_bounds(cfg: Config) -> Tuple[float, float]:
    """Return canonical (t_min, t_max), accepting legacy t_end/t_start aliases."""
    default = Config()
    # Prefer explicit new names.  If only a legacy alias differs from the default,
    # promote it so old command lines keep their previous behavior.
    t_min = float(cfg.t_min)
    t_max = float(cfg.t_max)
    if float(cfg.t_min) == float(default.t_min) and float(cfg.t_end) != float(default.t_end):
        t_min = float(cfg.t_end)
    if float(cfg.t_max) == float(default.t_max) and float(cfg.t_start) != float(default.t_start):
        t_max = float(cfg.t_start)
    if not math.isfinite(t_min) or not math.isfinite(t_max):
        raise ValueError(f"Nonfinite time bounds: t_min={t_min}, t_max={t_max}")
    if t_min <= 0.0:
        raise ValueError(f"t_min must be positive for OU weights and log schedules; got {t_min}")
    if t_max <= t_min:
        raise ValueError(f"Require t_max > t_min; got t_min={t_min}, t_max={t_max}")
    return t_min, t_max


def make_time_grid(cfg: Config, steps: int, *, direction: str, device, dtype) -> torch.Tensor:
    """Build the sampler/PF time grid.

    direction='reverse' gives t_max -> t_min for reverse SDE sampling.
    direction='forward' gives t_min -> t_max for PF density evaluation.
    """
    n_steps = int(steps)
    if n_steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")
    t_min, t_max = effective_time_bounds(cfg)
    schedule = canonical_time_schedule(cfg.time_schedule)
    if schedule == "linear":
        lo = torch.tensor(t_min, device=device, dtype=dtype)
        hi = torch.tensor(t_max, device=device, dtype=dtype)
        if direction == "forward":
            return torch.linspace(lo, hi, n_steps + 1, device=device, dtype=dtype)
        if direction == "reverse":
            return torch.linspace(hi, lo, n_steps + 1, device=device, dtype=dtype)
    elif schedule == "log_linear":
        log_lo = math.log(max(t_min, 1.0e-12))
        log_hi = math.log(max(t_max, t_min + 1.0e-12))
        if direction == "forward":
            logs = torch.linspace(log_lo, log_hi, n_steps + 1, device=device, dtype=dtype)
        elif direction == "reverse":
            logs = torch.linspace(log_hi, log_lo, n_steps + 1, device=device, dtype=dtype)
        else:
            raise ValueError(f"direction must be forward or reverse; got {direction!r}")
        return torch.exp(logs)
    raise RuntimeError(f"Unhandled time_schedule={schedule!r}")


def standard_normal_logprob(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return -0.5 * (d * math.log(2.0 * math.pi) + torch.sum(x * x, dim=-1))


def clamp_norm(x: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
    if max_norm is None or max_norm <= 0:
        return x
    n = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / (n + 1.0e-12), max=1.0)
    return x * scale


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return torch.clamp(x2 + y2 - 2.0 * (x @ y.T), min=0.0)


def median_bandwidth(x: torch.Tensor, y: Optional[torch.Tensor] = None, max_n: int = 1200) -> float:
    z = x if y is None else torch.cat([x, y], dim=0)
    if z.shape[0] > max_n:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_n]]
    d2 = pairwise_sq_dists(z, z)
    vals = d2[d2 > 0]
    if vals.numel() == 0:
        return 1.0
    med = torch.median(vals).item()
    return float(math.sqrt(max(med, 1.0e-12)))


# -----------------------------------------------------------------------------
# Normalized d=8 misaligned singular-subspace GMM
# -----------------------------------------------------------------------------


class MisalignedSubspaceGMM:
    """Analytic d-dimensional GMM with misaligned near-singular subspaces.

    The construction mirrors the benchmark harness target but then optionally
    whitens the full mixture so mean=0 and covariance=I.  This makes the
    isotropic Gaussian prior a fair global-volume initial bank while preserving
    local singular component geometry.
    """

    def __init__(
        self,
        d: int = 8,
        rank: int = 3,
        n_components: int = 8,
        seed: int = 29,
        radius: float = 3.0,
        sigma_perp: float = 0.035,
        jitter: float = 0.12,
        normalize: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.d = int(d)
        self.K = int(n_components)
        self.rank = int(rank)
        self.device = device
        self.dtype = dtype
        self.seed = int(seed)
        self.radius = float(radius)
        self.sigma_perp = float(sigma_perp)
        self.jitter = float(jitter)
        if not (0 < self.rank < self.d):
            raise ValueError(f"rank must satisfy 0 < rank < d; got rank={rank}, d={d}")

        rng = np.random.RandomState(self.seed)
        sigma_parallel = np.geomspace(1.10, 0.30, self.rank)
        base_sigmas = np.concatenate([sigma_parallel, self.sigma_perp * np.ones(self.d - self.rank)])

        raw = rng.normal(size=(self.K, self.d))
        envelope = np.ones(self.d)
        envelope[self.rank:] = 0.35
        raw *= envelope[None, :]
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        means = self.radius * raw

        covs = []
        sigmas_all = []
        for _k in range(self.K):
            A = rng.normal(size=(self.d, self.d))
            Q, R = np.linalg.qr(A)
            Q = Q @ np.diag(np.sign(np.diag(R)) + (np.diag(R) == 0))
            sig = base_sigmas.copy()
            perm = rng.permutation(self.rank)
            sig[:self.rank] = sig[:self.rank][perm]
            sig[:self.rank] *= np.exp(self.jitter * rng.normal(size=self.rank))
            sig[self.rank:] *= np.exp(0.5 * self.jitter * rng.normal(size=self.d - self.rank))
            sig = np.clip(sig, 0.5 * self.sigma_perp, None)
            covs.append(Q @ np.diag(sig ** 2) @ Q.T)
            sigmas_all.append(sig)
        covs = np.stack(covs, axis=0)
        weights = np.ones(self.K, dtype=np.float64) / float(self.K)

        self.original_mean = np.sum(weights[:, None] * means, axis=0)
        centered = means - self.original_mean[None, :]
        mixture_cov = np.sum(weights[:, None, None] * (covs + centered[:, :, None] * centered[:, None, :]), axis=0)
        self.original_cov_eigs = np.linalg.eigvalsh(0.5 * (mixture_cov + mixture_cov.T))

        if normalize:
            evals, evecs = np.linalg.eigh(0.5 * (mixture_cov + mixture_cov.T))
            evals = np.clip(evals, 1.0e-12, None)
            W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            means = (means - self.original_mean[None, :]) @ W.T
            covs = np.stack([W @ C @ W.T for C in covs], axis=0)
            covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))

        self.weights = torch.tensor(weights, device=device, dtype=dtype)
        self.log_weights = torch.log(self.weights)
        self.means = torch.tensor(means, device=device, dtype=dtype)
        self.covs = torch.tensor(covs, device=device, dtype=dtype)
        # Add tiny jitter only for numerical Cholesky/inverse; the model remains the same to displayed precision.
        eye = torch.eye(self.d, device=device, dtype=dtype)
        self.covs = sym(self.covs) + 1.0e-12 * eye.unsqueeze(0)
        self.precisions = torch.linalg.inv(self.covs)
        self.logdets = torch.linalg.slogdet(self.covs).logabsdet
        self.chols = torch.linalg.cholesky(self.covs)
        self.sigmas_all = np.asarray(sigmas_all)
        self.normalized = bool(normalize)

        with torch.no_grad():
            m = torch.sum(self.weights[:, None] * self.means, dim=0)
            c = torch.sum(self.weights[:, None, None] * (self.covs + (self.means - m)[..., None] * (self.means - m)[:, None, :]), dim=0)
            self.moment_mean_norm = safe_float(torch.linalg.norm(m))
            self.moment_cov_frob_err = safe_float(torch.linalg.matrix_norm(c - eye, ord="fro"))
            self.global_cov = c.detach()

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
        means_t, _covs_t, precs_t, logdets_t = self.marginal_params(t)
        diff = x[:, None, :] - means_t[None, :, :]
        mahal = torch.einsum("bki,kij,bkj->bk", diff, precs_t, diff)
        return self.log_weights[None, :] - 0.5 * (self.d * math.log(2.0 * math.pi) + logdets_t[None, :] + mahal)

    def responsibilities(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        return torch.softmax(self.component_log_probs(x, t=t), dim=1)

    def log_prob(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        return torch.logsumexp(self.component_log_probs(x, t=t), dim=1)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, t=0.0)

    def score(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        means_t, _covs_t, precs_t, _logdets_t = self.marginal_params(t)
        r = self.responsibilities(x, t=t)
        diff = x[:, None, :] - means_t[None, :, :]
        comp_scores = -torch.einsum("kij,bkj->bki", precs_t, diff)
        return torch.sum(r[:, :, None] * comp_scores, dim=1)

    def observed_information(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        means_t, _covs_t, precs_t, _logdets_t = self.marginal_params(t)
        r = self.responsibilities(x, t=t)
        diff = x[:, None, :] - means_t[None, :, :]
        comp_scores = -torch.einsum("kij,bkj->bki", precs_t, diff)
        score = torch.sum(r[:, :, None] * comp_scores, dim=1)
        Pbar = torch.einsum("bk,kij->bij", r, precs_t)
        second = torch.einsum("bk,bki,bkj->bij", r, comp_scores, comp_scores)
        cov_scores = second - score[:, :, None] * score[:, None, :]
        return sym(Pbar - cov_scores)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        idx = torch.multinomial(self.weights, int(n), replacement=True, generator=generator)
        eps = torch.randn((int(n), self.d), device=self.device, dtype=self.dtype, generator=generator)
        return self.means[idx] + torch.einsum("bij,bj->bi", self.chols[idx], eps)

    def sample_pt(self, n: int, t: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x0 = self.sample(n, generator=generator)
        alpha, gamma = alpha_gamma(torch.tensor(float(t), device=self.device, dtype=self.dtype))
        eps = torch.randn(x0.shape, device=self.device, dtype=self.dtype, generator=generator)
        return alpha * x0 + torch.sqrt(torch.clamp(gamma, min=0.0)) * eps


# -----------------------------------------------------------------------------
# Neal funnel target
# -----------------------------------------------------------------------------


class NealFunnelTarget:
    """Neal's d-dimensional funnel with exact t=0 score and Hessian.

    Native coordinates are
        z_1 ~ N(0, eta^2),   z_{2:d} | z_1 ~ N(0, exp(z_1) I).
    When ``normalize=True`` we return x = z / std(z), using the exact diagonal
    standard deviations.  This preserves the funnel geometry while making the
    initial N(0,I) bank globally volume-matched, consistent with the other
    alternating-DRC targets in this script.  Set ``--no_normalize_target`` to
    run the raw benchmark-sweep coordinates.
    """

    def __init__(
        self,
        d: int = 10,
        eta2: float = 6.0,
        normalize: bool = True,
        score_bank_size: int = 8192,
        score_chunk: int = 512,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.d = int(d)
        self.D = int(d)
        self.eta2 = float(eta2)
        self.normalized = bool(normalize)
        self.score_bank_size = int(score_bank_size)
        self.score_chunk = int(score_chunk)
        self.device = device
        self.dtype = dtype
        self.name = f"funnel_d{self.d}"
        if self.d < 2:
            raise ValueError("NealFunnelTarget requires d >= 2")
        scale = torch.ones((self.d,), device=device, dtype=dtype)
        if self.normalized:
            scale[0] = math.sqrt(self.eta2)
            scale[1:] = math.exp(self.eta2 / 4.0)
        self.scale = scale
        self.logabsdet_scale = torch.log(scale).sum()
        self._ou_score_bank: Optional[torch.Tensor] = None
        self.original_cov_eigs = [float(self.eta2)] + [float(math.exp(self.eta2 / 2.0))] * (self.d - 1)
        with torch.no_grad():
            cov_diag = torch.ones((self.d,), device=device, dtype=dtype) if self.normalized else torch.tensor(self.original_cov_eigs, device=device, dtype=dtype)
            self.moment_mean_norm = 0.0
            self.moment_cov_frob_err = safe_float(torch.linalg.norm(torch.diag(cov_diag) - torch.eye(self.d, device=device, dtype=dtype)))
            self.global_cov = torch.diag(cov_diag).detach()

    def _to_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.to(device=x.device, dtype=x.dtype)

    def _from_native(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.scale.to(device=z.device, dtype=z.dtype)

    def _native_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z1 = z[:, 0]
        rest = z[:, 1:]
        inv_var = torch.exp(-z1).clamp(max=1.0e30)
        return (
            -0.5 * z1.square() / self.eta2
            -0.5 * float(self.d - 1) * z1
            -0.5 * rest.square().sum(dim=-1) * inv_var
        )

    def log_prob(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            return self._empirical_ou_log_prob(x, t_float)
        z = self._to_native(x)
        return self._native_log_prob(z) + self.logabsdet_scale.to(device=x.device, dtype=x.dtype)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, t=0.0)

    def score(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            return self._empirical_ou_score(x, t_float)
        z = self._to_native(x)
        z1 = z[:, 0]
        rest = z[:, 1:]
        inv_var = torch.exp(-z1).clamp(max=1.0e30)
        score_z = torch.empty_like(z)
        score_z[:, 0] = -z1 / self.eta2 - 0.5 * float(self.d - 1) + 0.5 * rest.square().sum(dim=-1) * inv_var
        score_z[:, 1:] = -rest * inv_var[:, None]
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        return torch.nan_to_num(score_z * scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)

    def observed_information(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            raise NotImplementedError("NealFunnelTarget only provides exact observed information at t=0.")
        z = self._to_native(x)
        z1 = z[:, 0]
        rest = z[:, 1:]
        inv_var = torch.exp(-z1).clamp(max=1.0e30)
        B = int(x.shape[0])
        H_z = torch.zeros((B, self.d, self.d), device=x.device, dtype=x.dtype)
        rest_sq = rest.square().sum(dim=-1)
        H_z[:, 0, 0] = 1.0 / self.eta2 + 0.5 * rest_sq * inv_var
        cross = -rest * inv_var[:, None]
        H_z[:, 0, 1:] = cross
        H_z[:, 1:, 0] = cross
        idx = torch.arange(1, self.d, device=x.device)
        H_z[:, idx, idx] = inv_var[:, None]
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        H_x = H_z * scale[None, :, None] * scale[None, None, :]
        return torch.nan_to_num(sym(H_x), nan=0.0, posinf=1.0e12, neginf=-1.0e12)

    @torch.no_grad()
    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        n = int(n)
        z1 = math.sqrt(self.eta2) * torch.randn((n, 1), device=self.device, dtype=self.dtype, generator=generator)
        rest = torch.exp(0.5 * z1) * torch.randn((n, self.d - 1), device=self.device, dtype=self.dtype, generator=generator)
        return self._from_native(torch.cat([z1, rest], dim=1)).detach()

    @torch.no_grad()
    def sample_pt(self, n: int, t: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x0 = self.sample(n, generator=generator)
        alpha, gamma = alpha_gamma(torch.tensor(float(t), device=self.device, dtype=self.dtype))
        eps = torch.randn(x0.shape, device=self.device, dtype=self.dtype, generator=generator)
        return alpha * x0 + torch.sqrt(torch.clamp(gamma, min=0.0)) * eps

    def _ensure_ou_score_bank(self) -> torch.Tensor:
        if self._ou_score_bank is None or int(self._ou_score_bank.shape[0]) < max(8, int(self.score_bank_size)):
            gen = make_generator(71_337, self.device)
            self._ou_score_bank = self.sample(max(8, int(self.score_bank_size)), generator=gen).detach()
        return self._ou_score_bank

    @torch.no_grad()
    def _empirical_ou_score(self, y: torch.Tensor, t: float) -> torch.Tensor:
        bank = self._ensure_ou_score_bank()
        alpha, gamma = alpha_gamma(float(t), device=self.device, dtype=self.dtype)
        gamma = torch.clamp(gamma, min=torch.as_tensor(1.0e-8, device=self.device, dtype=self.dtype))
        outs: List[torch.Tensor] = []
        chunk = max(1, int(self.score_chunk))
        for start in range(0, y.shape[0], chunk):
            yy = y[start:start + chunk]
            diff = yy[:, None, :] - alpha * bank[None, :, :]
            logw = -0.5 * torch.sum(diff * diff, dim=-1) / gamma
            logw = logw - torch.max(logw, dim=1, keepdim=True).values
            w = torch.exp(logw)
            w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1.0e-300)
            b = (alpha * bank[None, :, :] - yy[:, None, :]) / gamma
            outs.append(torch.sum(w[:, :, None] * b, dim=1))
        return torch.nan_to_num(torch.cat(outs, dim=0), nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _empirical_ou_log_prob(self, y: torch.Tensor, t: float) -> torch.Tensor:
        bank = self._ensure_ou_score_bank()
        alpha, gamma = alpha_gamma(float(t), device=self.device, dtype=self.dtype)
        gamma = torch.clamp(gamma, min=torch.as_tensor(1.0e-8, device=self.device, dtype=self.dtype))
        d = int(y.shape[1])
        outs: List[torch.Tensor] = []
        chunk = max(1, int(self.score_chunk))
        const = -0.5 * d * math.log(2.0 * math.pi) - 0.5 * d * torch.log(gamma)
        for start in range(0, y.shape[0], chunk):
            yy = y[start:start + chunk]
            diff = yy[:, None, :] - alpha * bank[None, :, :]
            logk = const - 0.5 * torch.sum(diff * diff, dim=-1) / gamma
            outs.append(torch.logsumexp(logk, dim=1) - math.log(int(bank.shape[0])))
        return torch.nan_to_num(torch.cat(outs, dim=0), nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6)

    def target_info(self) -> Dict[str, object]:
        return {
            "target_name": self.name,
            "target_type": "neal_funnel",
            "target_dim": int(self.d),
            "funnel_eta2": float(self.eta2),
            "funnel_normalized": bool(self.normalized),
            "funnel_score_bank": int(self.score_bank_size),
        }


# -----------------------------------------------------------------------------
# Intermediate-dimensional molecular LJ/DW-style targets
# -----------------------------------------------------------------------------


def _hexagonal_cluster_2d(n: int, r0: float = 1.12) -> np.ndarray:
    """Deterministic 2D seed geometry: center + hexagonal shells."""
    pts: List[Tuple[float, float]] = [(0.0, 0.0)]
    if n <= 1:
        return np.asarray(pts[:n], dtype=np.float64)
    # Inner ring: six neighbors.
    for k in range(6):
        th = 2.0 * math.pi * k / 6.0
        pts.append((r0 * math.cos(th), r0 * math.sin(th)))
        if len(pts) >= n:
            return np.asarray(pts[:n], dtype=np.float64)
    # Outer ring/interstitials.  For n=13 this gives a compact LJ13-like island.
    r_outer = math.sqrt(3.0) * r0
    for k in range(6):
        th = 2.0 * math.pi * (k + 0.5) / 6.0
        pts.append((r_outer * math.cos(th), r_outer * math.sin(th)))
        if len(pts) >= n:
            return np.asarray(pts[:n], dtype=np.float64)
    # Fallback additional spiral shells.
    shell = 2
    while len(pts) < n:
        count = 6 * shell
        rad = shell * r0
        for k in range(count):
            th = 2.0 * math.pi * k / count
            pts.append((rad * math.cos(th), rad * math.sin(th)))
            if len(pts) >= n:
                break
        shell += 1
    return np.asarray(pts[:n], dtype=np.float64)


def _generic_cluster(n: int, particle_dim: int, r0: float = 1.12, seed: int = 0) -> np.ndarray:
    if int(particle_dim) == 2:
        return _hexagonal_cluster_2d(n, r0=r0)
    rng = np.random.RandomState(seed)
    pts = rng.normal(size=(n, particle_dim))
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(pts * pts, axis=1)))
    return (r0 * pts / max(scale, 1.0e-12)).astype(np.float64)


class MolecularLJTarget:
    """Whitened molecular potential target with exact score/Hessian at t=0.

    This target is intentionally not a Gaussian mixture.  It is a small bonded
    particle cluster with a soft-core Lennard-Jones nonbonded field plus stiff,
    heterogeneous nearest-neighbor bond constraints around a reference LJ13-like
    hexagonal cluster.  In normalized coordinates it is approximately globally
    mean-zero/covariance-one, but locally it has the features that should stress
    score estimators: thin bond-length manifolds, rotation/reflection modes,
    strongly heterogeneous curvatures, and non-Gaussian multimodal tails.

    The score and observed information at t=0 are exact autograd derivatives of
    the Boltzmann log density.  For evaluation-only Fisher RMSE at t>0, where no
    closed form OU-marginal score is available, the class uses a cached empirical
    Tweedie score from target samples.  That approximation affects diagnostics
    only; the benchmark estimators still receive exact t=0 scores and Hessians.
    """

    def __init__(
        self,
        n_particles: int = 13,
        particle_dim: int = 2,
        seed: int = 29,
        beta: float = 1.0,
        lj_eps: float = 0.18,
        lj_sigma: float = 1.0,
        lj_soft_core: float = 0.08,
        bond_k: float = 80.0,
        confinement_k: float = 0.015,
        com_k: float = 6.0,
        init_noise: float = 0.18,
        sample_steps: int = 800,
        sample_step_size: float = 2.0e-4,
        sample_batch: int = 512,
        normalize: bool = True,
        norm_samples: int = 2048,
        norm_eig_floor: float = 1.0e-4,
        score_bank_size: int = 4096,
        score_chunk: int = 256,
        hessian_chunk: int = 16,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        name: str = "lj13_2d",
    ):
        self.n_particles = int(n_particles)
        self.particle_dim = int(particle_dim)
        self.d = self.n_particles * self.particle_dim
        self.K = 0
        self.weights = torch.empty((0,), device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.seed = int(seed)
        self.beta = float(beta)
        self.lj_eps = float(lj_eps)
        self.lj_sigma = float(lj_sigma)
        self.lj_soft_core = float(lj_soft_core)
        self.bond_k = float(bond_k)
        self.confinement_k = float(confinement_k)
        self.com_k = float(com_k)
        self.init_noise = float(init_noise)
        self.sample_steps = int(sample_steps)
        self.sample_step_size = float(sample_step_size)
        self.sample_batch = int(sample_batch)
        self.normalized = bool(normalize)
        self.norm_samples = int(norm_samples)
        self.norm_eig_floor = float(norm_eig_floor)
        self.score_bank_size = int(score_bank_size)
        self.score_chunk = int(score_chunk)
        self.hessian_chunk = int(hessian_chunk)
        self.name = str(name)
        self._ou_score_bank: Optional[torch.Tensor] = None

        r0 = (2.0 ** (1.0 / 6.0)) * self.lj_sigma
        base_np = _generic_cluster(self.n_particles, self.particle_dim, r0=r0, seed=self.seed)
        base_np = base_np - base_np.mean(axis=0, keepdims=True)
        self.base_pos = torch.tensor(base_np, device=device, dtype=dtype)

        pair_i, pair_j = np.triu_indices(self.n_particles, k=1)
        self.pair_i = torch.tensor(pair_i, device=device, dtype=torch.long)
        self.pair_j = torch.tensor(pair_j, device=device, dtype=torch.long)
        rng = np.random.RandomState(self.seed + 17)
        eps = self.lj_eps * np.exp(0.25 * rng.normal(size=len(pair_i)))
        sig = self.lj_sigma * np.exp(0.07 * rng.normal(size=len(pair_i)))
        self.pair_eps = torch.tensor(eps, device=device, dtype=dtype)
        self.pair_sigma = torch.tensor(sig, device=device, dtype=dtype)

        # Stiff bond graph from nearby edges of the reference cluster.  These
        # constraints create the nearly singular score geometry LFGI is meant to
        # exploit, while the pair field keeps the example molecular rather than a
        # hand-built Gaussian tube.
        diff0 = base_np[pair_i] - base_np[pair_j]
        dist0 = np.sqrt(np.sum(diff0 * diff0, axis=1))
        threshold = 1.32 * r0
        edge_mask = dist0 <= threshold
        if not np.any(edge_mask):
            edge_mask[: min(len(edge_mask), self.n_particles - 1)] = True
        edge_i = pair_i[edge_mask]
        edge_j = pair_j[edge_mask]
        edge_r0 = dist0[edge_mask]
        edge_k = self.bond_k * np.exp(0.45 * rng.normal(size=len(edge_i)))
        self.edge_i = torch.tensor(edge_i, device=device, dtype=torch.long)
        self.edge_j = torch.tensor(edge_j, device=device, dtype=torch.long)
        self.edge_r0 = torch.tensor(edge_r0, device=device, dtype=dtype)
        self.edge_k = torch.tensor(edge_k, device=device, dtype=dtype)

        # The affine transform is z = norm_mean + norm_L @ x, with x the public
        # normalized coordinate used by the rest of the benchmark.
        self.norm_mean = torch.zeros((self.d,), device=device, dtype=dtype)
        self.norm_L = torch.eye(self.d, device=device, dtype=dtype)
        self.norm_W = torch.eye(self.d, device=device, dtype=dtype)
        self.norm_logabsdet_L = torch.tensor(0.0, device=device, dtype=dtype)
        self.original_cov_eigs = np.ones(self.d, dtype=np.float64)

        if self.normalized:
            gen = make_generator(self.seed + 50_000, device)
            pilot_n = max(64, int(self.norm_samples))
            pilot_z = self._sample_physical(pilot_n, gen, steps=max(50, self.sample_steps // 2))
            mean = pilot_z.mean(dim=0)
            X = pilot_z - mean
            cov = (X.T @ X) / max(pilot_n - 1, 1)
            cov = sym(cov)
            evals, evecs = torch.linalg.eigh(cov)
            evals_clamped = torch.clamp(evals, min=self.norm_eig_floor)
            sqrt_e = torch.sqrt(evals_clamped)
            inv_sqrt_e = 1.0 / sqrt_e
            self.norm_mean = mean.detach()
            self.norm_L = (evecs @ torch.diag(sqrt_e) @ evecs.T).detach()
            self.norm_W = (evecs @ torch.diag(inv_sqrt_e) @ evecs.T).detach()
            self.norm_logabsdet_L = torch.sum(torch.log(sqrt_e)).detach()
            self.original_cov_eigs = as_numpy(evals)
            pilot_x = self._from_physical(pilot_z)
        else:
            pilot_x = self._from_physical(self._initial_physical(512, make_generator(self.seed + 7, device)))
            self.original_cov_eigs = np.ones(self.d, dtype=np.float64)

        with torch.no_grad():
            m = pilot_x.mean(dim=0)
            X = pilot_x - m
            C = (X.T @ X) / max(int(pilot_x.shape[0]) - 1, 1)
            eye = torch.eye(self.d, device=device, dtype=dtype)
            self.moment_mean_norm = safe_float(torch.linalg.norm(m))
            self.moment_cov_frob_err = safe_float(torch.linalg.matrix_norm(C - eye, ord="fro"))
            self.global_cov = C.detach()

    def _positions(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(z.shape[0], self.n_particles, self.particle_dim)

    def _to_physical(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_mean[None, :] + x @ self.norm_L.T

    def _from_physical(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.norm_mean[None, :]) @ self.norm_W.T

    def _initial_physical(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        n = int(n)
        pos = self.base_pos[None, :, :].expand(n, -1, -1).clone()
        if self.particle_dim == 2:
            theta = 2.0 * math.pi * torch.rand((n,), device=self.device, dtype=self.dtype, generator=generator)
            c, s = torch.cos(theta), torch.sin(theta)
            R = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
            pos = torch.einsum("bij,bnj->bni", R, pos)
            refl = torch.where(torch.rand((n, 1, 1), device=self.device, dtype=self.dtype, generator=generator) < 0.5, -1.0, 1.0)
            pos[:, :, 0:1] = refl * pos[:, :, 0:1]
        noise = self.init_noise * torch.randn(pos.shape, device=self.device, dtype=self.dtype, generator=generator)
        pos = pos + noise
        pos = pos - pos.mean(dim=1, keepdim=True)
        return pos.reshape(n, self.d)

    def _physical_energy(self, z: torch.Tensor) -> torch.Tensor:
        pos = self._positions(z)
        centered = pos - pos.mean(dim=1, keepdim=True)
        pi = pos[:, self.pair_i, :]
        pj = pos[:, self.pair_j, :]
        diff = pi - pj
        r2 = torch.sum(diff * diff, dim=-1) + self.lj_soft_core ** 2
        sig2_over_r2 = (self.pair_sigma[None, :] ** 2) / torch.clamp(r2, min=1.0e-12)
        sr6 = sig2_over_r2 ** 3
        lj = 4.0 * self.pair_eps[None, :] * (sr6 * sr6 - sr6)

        ei = pos[:, self.edge_i, :]
        ej = pos[:, self.edge_j, :]
        ed = ei - ej
        er = torch.sqrt(torch.sum(ed * ed, dim=-1) + 1.0e-12)
        bond = 0.5 * self.edge_k[None, :] * (er - self.edge_r0[None, :]) ** 2

        conf = 0.5 * self.confinement_k * torch.sum(centered * centered, dim=(1, 2))
        com = pos.mean(dim=1)
        com_pen = 0.5 * self.com_k * float(self.n_particles) * torch.sum(com * com, dim=1)
        return self.beta * (torch.sum(lj, dim=1) + torch.sum(bond, dim=1) + conf + com_pen)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        z = self._to_physical(x)
        # Add the affine Jacobian constant so log_prob is a proper normalized-coordinate
        # density up to the unknown physical partition constant.  The constant cancels
        # in all DRC ratios but keeps reported NLLs comparable across normalizations.
        return self._physical_energy(z) - self.norm_logabsdet_L

    def log_prob(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        if float(torch.as_tensor(t).detach().cpu().item()) != 0.0:
            # Evaluation-only empirical OU log density.  It is not used by the
            # benchmark's DRC weights, which always call t=0.
            return self._empirical_ou_log_prob(x, float(torch.as_tensor(t).detach().cpu().item()))
        return -self.energy(x)

    def score(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            return self._empirical_ou_score(x, t_float)
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            e = self.energy(x_req).sum()
            grad = torch.autograd.grad(e, x_req, create_graph=False, retain_graph=False)[0]
        return torch.nan_to_num(-grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    def observed_information(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            raise NotImplementedError("MolecularLJTarget only provides exact observed information at t=0.")
        outs: List[torch.Tensor] = []
        chunk = max(1, int(self.hessian_chunk))
        with torch.enable_grad():
            try:
                from torch.func import hessian, vmap
                def e_single(x1: torch.Tensor) -> torch.Tensor:
                    return self.energy(x1.unsqueeze(0))[0]
                hess_fn = vmap(hessian(e_single))
                for start in range(0, x.shape[0], chunk):
                    xb = x[start:start + chunk].detach()
                    outs.append(hess_fn(xb).detach())
            except Exception:
                for start in range(0, x.shape[0], chunk):
                    xb = x[start:start + chunk].detach()
                    local: List[torch.Tensor] = []
                    for i in range(xb.shape[0]):
                        xi = xb[i].detach().clone().requires_grad_(True)
                        H = torch.autograd.functional.hessian(lambda zz: self.energy(zz.unsqueeze(0))[0], xi)
                        local.append(H.detach())
                    outs.append(torch.stack(local, dim=0))
        H = torch.cat(outs, dim=0)
        return torch.nan_to_num(sym(H), nan=0.0, posinf=0.0, neginf=0.0)

    def _physical_score(self, z: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z_req = z.detach().clone().requires_grad_(True)
            e = self._physical_energy(z_req).sum()
            grad = torch.autograd.grad(e, z_req, create_graph=False, retain_graph=False)[0]
        return torch.nan_to_num(-grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _sample_physical(self, n: int, generator: Optional[torch.Generator], steps: Optional[int] = None) -> torch.Tensor:
        n = int(n)
        steps = int(self.sample_steps if steps is None else steps)
        batch = max(1, int(self.sample_batch))
        outs: List[torch.Tensor] = []
        dt = float(self.sample_step_size)
        noise_scale = math.sqrt(2.0 * dt)
        for start in range(0, n, batch):
            b = min(batch, n - start)
            z = self._initial_physical(b, generator)
            for _ in range(max(0, steps)):
                score = clamp_norm(self._physical_score(z), 1.0e4)
                z = z + dt * score + noise_scale * torch.randn(z.shape, device=self.device, dtype=self.dtype, generator=generator)
                z = torch.nan_to_num(z, nan=0.0, posinf=25.0, neginf=-25.0)
                z = torch.clamp(z, min=-25.0, max=25.0)
            outs.append(z.detach())
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        z = self._sample_physical(int(n), generator)
        return self._from_physical(z).detach()

    @torch.no_grad()
    def sample_pt(self, n: int, t: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x0 = self.sample(n, generator=generator)
        alpha, gamma = alpha_gamma(torch.tensor(float(t), device=self.device, dtype=self.dtype))
        eps = torch.randn(x0.shape, device=self.device, dtype=self.dtype, generator=generator)
        return alpha * x0 + torch.sqrt(torch.clamp(gamma, min=0.0)) * eps

    def _ensure_ou_score_bank(self) -> torch.Tensor:
        if self._ou_score_bank is None or int(self._ou_score_bank.shape[0]) < max(8, int(self.score_bank_size)):
            gen = make_generator(self.seed + 70_000, self.device)
            self._ou_score_bank = self.sample(max(8, int(self.score_bank_size)), generator=gen).detach()
        return self._ou_score_bank

    @torch.no_grad()
    def _empirical_ou_score(self, y: torch.Tensor, t: float) -> torch.Tensor:
        bank = self._ensure_ou_score_bank()
        alpha, gamma = alpha_gamma(float(t), device=self.device, dtype=self.dtype)
        gamma = torch.clamp(gamma, min=torch.as_tensor(1.0e-8, device=self.device, dtype=self.dtype))
        outs: List[torch.Tensor] = []
        chunk = max(1, int(self.score_chunk))
        for start in range(0, y.shape[0], chunk):
            yy = y[start:start + chunk]
            diff = yy[:, None, :] - alpha * bank[None, :, :]
            logw = -0.5 * torch.sum(diff * diff, dim=-1) / gamma
            logw = logw - torch.max(logw, dim=1, keepdim=True).values
            w = torch.exp(logw)
            w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1.0e-300)
            b = (alpha * bank[None, :, :] - yy[:, None, :]) / gamma
            outs.append(torch.sum(w[:, :, None] * b, dim=1))
        return torch.nan_to_num(torch.cat(outs, dim=0), nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _empirical_ou_log_prob(self, y: torch.Tensor, t: float) -> torch.Tensor:
        bank = self._ensure_ou_score_bank()
        alpha, gamma = alpha_gamma(float(t), device=self.device, dtype=self.dtype)
        gamma = torch.clamp(gamma, min=torch.as_tensor(1.0e-8, device=self.device, dtype=self.dtype))
        d = int(y.shape[1])
        outs: List[torch.Tensor] = []
        chunk = max(1, int(self.score_chunk))
        const = -0.5 * d * math.log(2.0 * math.pi) - 0.5 * d * torch.log(gamma)
        for start in range(0, y.shape[0], chunk):
            yy = y[start:start + chunk]
            diff = yy[:, None, :] - alpha * bank[None, :, :]
            logk = const - 0.5 * torch.sum(diff * diff, dim=-1) / gamma
            outs.append(torch.logsumexp(logk, dim=1) - math.log(int(bank.shape[0])))
        return torch.nan_to_num(torch.cat(outs, dim=0), nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6)

    def target_info(self) -> Dict[str, object]:
        return {
            "target_name": self.name,
            "target_type": "molecular_lj",
            "target_dim": int(self.d),
            "mol_n_particles": int(self.n_particles),
            "mol_particle_dim": int(self.particle_dim),
            "mol_n_pairs": int(self.pair_i.numel()),
            "mol_n_bonds": int(self.edge_i.numel()),
            "mol_beta": float(self.beta),
            "mol_lj_eps": float(self.lj_eps),
            "mol_bond_k_mean": safe_float(self.edge_k.mean()) if self.edge_k.numel() else float("nan"),
            "mol_bond_k_max": safe_float(self.edge_k.max()) if self.edge_k.numel() else float("nan"),
            "mol_score_t_mode": "empirical_ou_for_t_gt_0",
        }


# -----------------------------------------------------------------------------
# Weighted SNIS score bank: Blend and CE-HLSI/LFGI
# -----------------------------------------------------------------------------


def process_curvature(H: torch.Tensor, mode: str, floor: float, cap: float) -> torch.Tensor:
    H = sym(H)
    key = str(mode).lower()
    if key == "raw":
        return torch.clamp(H, min=-float(cap), max=float(cap)) if False else H
    evals, evecs = torch.linalg.eigh(H)
    if key == "psd":
        evals = torch.clamp(evals, min=max(float(floor), 0.0), max=float(cap))
    elif key == "abs":
        evals = torch.clamp(torch.abs(evals), min=max(float(floor), 0.0), max=float(cap))
    else:
        raise ValueError("curvature_mode must be raw, psd, or abs")
    return sym(evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2))




def mp_leaf_precision_completion(H: torch.Tensor, floor: float = 0.0, tol: float = 1.0e-12) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Minimal moment-preserving MP-leaf PSD precision completion.

    This is the fixed-floor version of the MP-leaf construction used in the
    larger sandbox.  If H = V diag(lambda) V^T, define
        q_j = max(lambda_j, floor, 0),  c_j = q_j - lambda_j.
    The implicit +/- sigma leaves have mean score s0 and precision certificate
        E[Q_leaf] - Cov(s_leaf) = H,
    while the CE/LFGI gate uses Q = V diag(q) V^T.  We do not materialize the
    leaves here because the alternating DRC experiment only needs the completed
    gate precision.
    """
    H = sym(H)
    lam, V = torch.linalg.eigh(H)
    floor_t = torch.as_tensor(max(float(floor), 0.0), device=H.device, dtype=H.dtype)
    q = torch.maximum(lam, floor_t.expand_as(lam))
    q = torch.maximum(q, torch.zeros_like(q))
    c = (q - lam).clamp_min(0.0)
    Q = sym(V @ torch.diag_embed(q) @ V.transpose(-1, -2))
    active = c > float(tol)
    active_rank = active.sum(dim=1) if active.ndim == 2 else torch.zeros((H.shape[0],), device=H.device)
    P_cert = sym(Q - (V @ torch.diag_embed(c) @ V.transpose(-1, -2)))
    rel = torch.linalg.matrix_norm(P_cert - H, ord="fro", dim=(1, 2)) / (1.0 + torch.linalg.matrix_norm(H, ord="fro", dim=(1, 2))).clamp_min(1.0e-30)
    info = {
        "mp_leaf_floor": float(floor_t.detach().cpu().item()),
        "mp_leaf_active_frac": safe_float(active.to(H.dtype).mean()),
        "mp_leaf_active_parent_frac": safe_float((active_rank > 0).to(H.dtype).mean()),
        "mp_leaf_active_rank_mean": safe_float(active_rank.to(H.dtype).mean()),
        "mp_leaf_completion_trace_mean": safe_float(c.sum(dim=1).mean()),
        "mp_leaf_q_eig_p95": safe_float(torch.quantile(q.reshape(-1), 0.95)) if q.numel() else float("nan"),
        "mp_leaf_precision_cert_rel_max": safe_float(rel.max()) if rel.numel() else 0.0,
    }
    return Q.detach(), info

def resolvent_gate(P: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, eps: float, gate_clip: Optional[float]) -> torch.Tensor:
    d = P.shape[-1]
    I = torch.eye(d, device=P.device, dtype=P.dtype)
    A = sym(alpha * alpha * I + gamma * P)
    evals, evecs = torch.linalg.eigh(A)
    signs = torch.where(evals >= 0, torch.ones_like(evals), -torch.ones_like(evals))
    evals_safe = torch.where(torch.abs(evals) < float(eps), signs * float(eps), evals)
    gvals = (alpha * alpha) / evals_safe
    if gate_clip is not None and gate_clip > 0:
        gvals = torch.clamp(gvals, min=-float(gate_clip), max=float(gate_clip))
    return sym(evecs @ torch.diag_embed(gvals) @ evecs.transpose(-1, -2))


class SNISScoreBank:
    def __init__(
        self,
        target,
        anchors: torch.Tensor,
        cfg: Config,
        log_ref_weights: Optional[torch.Tensor] = None,
        gate_anchors: Optional[torch.Tensor] = None,
        gate_log_ref_weights: Optional[torch.Tensor] = None,
    ):
        self.target = target
        self.x = anchors.detach().to(device=target.device, dtype=target.dtype)
        self.N, self.d = self.x.shape
        self.device = target.device
        self.dtype = target.dtype
        self.cfg = cfg

        self.score0 = target.score(self.x, t=0.0).detach()
        H = target.observed_information(self.x, t=0.0).detach()
        self.H_raw = sym(H).detach()
        self.P = process_curvature(self.H_raw, cfg.curvature_mode, cfg.curvature_floor, cfg.curvature_cap).detach()
        self.P_mp, self.mp_leaf_info = mp_leaf_precision_completion(self.H_raw, cfg.mp_leaf_floor, cfg.mp_leaf_tol)
        if log_ref_weights is None:
            self.log_ref_weights = torch.zeros((self.N,), device=self.device, dtype=self.dtype)
        else:
            lw = log_ref_weights.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
            if lw.shape[0] != self.N:
                raise ValueError(f"log_ref_weights has length {lw.shape[0]} but anchors have length {self.N}")
            self.log_ref_weights = torch.nan_to_num(lw, nan=0.0, posinf=0.0, neginf=0.0)

        # Gate bank.  LFGI/leaf-LFGI use this bank for the Hessian/precision
        # average Pbar/Qbar.  Scalar Blend also estimates its scalar gate from
        # this bank, then applies that gate to the score-bank Tweedie/TSI means.
        if gate_anchors is None:
            self.x_gate = self.x
            self.log_gate_weights = self.log_ref_weights
            self.score0_gate = self.score0
            self.H_gate_raw = self.H_raw
            self.P_gate = self.P
            self.P_gate_mp = self.P_mp
            self.gate_is_score_bank = True
        else:
            self.x_gate = gate_anchors.detach().to(device=target.device, dtype=target.dtype)
            self.gate_is_score_bank = (
                self.x_gate.shape == self.x.shape
                and self.x_gate.data_ptr() == self.x.data_ptr()
            )
            if gate_log_ref_weights is None:
                self.log_gate_weights = torch.zeros((self.x_gate.shape[0],), device=self.device, dtype=self.dtype)
            else:
                glw = gate_log_ref_weights.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
                if glw.shape[0] != self.x_gate.shape[0]:
                    raise ValueError(f"gate_log_ref_weights has length {glw.shape[0]} but gate_anchors have length {self.x_gate.shape[0]}")
                self.log_gate_weights = torch.nan_to_num(glw, nan=0.0, posinf=0.0, neginf=0.0)
            if self.gate_is_score_bank:
                self.score0_gate = self.score0
                self.H_gate_raw = self.H_raw
                self.P_gate = self.P
                self.P_gate_mp = self.P_mp
            else:
                self.score0_gate = target.score(self.x_gate, t=0.0).detach()
                Hg = target.observed_information(self.x_gate, t=0.0).detach()
                self.H_gate_raw = sym(Hg).detach()
                self.P_gate = process_curvature(self.H_gate_raw, cfg.curvature_mode, cfg.curvature_floor, cfg.curvature_cap).detach()
                self.P_gate_mp, gate_mp_info = mp_leaf_precision_completion(self.H_gate_raw, cfg.mp_leaf_floor, cfg.mp_leaf_tol)
                # Keep both score-bank and gate-bank MP diagnostics if available.
                for k, v in gate_mp_info.items():
                    self.mp_leaf_info[f"gate_{k}"] = v
        self.N_gate = int(self.x_gate.shape[0])
        self.mp_leaf_info.update({
            "score_bank_n": int(self.N),
            "gate_bank_n": int(self.N_gate),
            "gate_bank_separate": bool(not self.gate_is_score_bank),
        })

    def _weights_and_signals_for(self, y: torch.Tensor, t: float, x: torch.Tensor, score0: torch.Tensor, log_weights: torch.Tensor):
        alpha, gamma = alpha_gamma(float(t), device=self.device, dtype=self.dtype)
        gamma = torch.clamp(gamma, min=torch.as_tensor(1.0e-12, device=self.device, dtype=self.dtype))
        diff = y[:, None, :] - alpha * x[None, :, :]
        logw = -0.5 * torch.sum(diff * diff, dim=-1) / gamma
        if float(self.cfg.weight_temp) != 1.0:
            logw = logw / float(self.cfg.weight_temp)
        logw = logw + log_weights[None, :]
        logw = logw - torch.max(logw, dim=1, keepdim=True).values
        w = torch.exp(logw)
        w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1.0e-300)
        b = (alpha * x[None, :, :] - y[:, None, :]) / gamma
        c = score0[None, :, :] / alpha
        return w, b, c, alpha, gamma

    def _weights_and_signals(self, y: torch.Tensor, t: float):
        return self._weights_and_signals_for(y, t, self.x, self.score0, self.log_ref_weights)

    def _gate_weights_and_signals(self, y: torch.Tensor, t: float):
        return self._weights_and_signals_for(y, t, self.x_gate, self.score0_gate, self.log_gate_weights)

    def _gate_precision_for_method(self, method: str) -> torch.Tensor:
        key = str(method).strip().lower().replace("_", "-")
        if key in {"leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"}:
            return self.P_gate_mp
        return self.P_gate

    def estimate_chunk(self, y: torch.Tensor, t: float, method: str) -> torch.Tensor:
        key = str(method).strip().lower().replace("_", "-")
        w, b, c, alpha, gamma = self._weights_and_signals(y, t)
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)
        if key in {"blend", "blended", "scalar-blend", "scalar"}:
            wg, bg, cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t)
            bgbar = torch.sum(wg[:, :, None] * bg, dim=1)
            cgbar = torch.sum(wg[:, :, None] * cg, dim=1)
            Ac = cg - cgbar[:, None, :]
            Bc = bg - bgbar[:, None, :]
            va = torch.sum(wg[:, :, None] * Ac.square(), dim=1).clamp(min=1.0e-30)
            vb = torch.sum(wg[:, :, None] * Bc.square(), dim=1).clamp(min=1.0e-30)
            cab = torch.sum(wg[:, :, None] * Ac * Bc, dim=1)
            den = (va + vb - 2.0 * cab).clamp(min=1.0e-20)
            g = ((va - cab) / den).clamp(0.0, 1.0)
            return cbar + g * (bbar - cbar)
        if key in {"ce-hlsi", "lfgi", "ce-lfgi", "leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"}:
            wg, _bg, _cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t)
            Pgate = self._gate_precision_for_method(method)
            Pbar = torch.sum(wg[:, :, None, None] * Pgate[None, :, :, :], dim=1)
            G = resolvent_gate(Pbar, alpha, gamma, self.cfg.resolvent_eps, self.cfg.gate_clip)
            return bbar + torch.einsum("bij,bj->bi", G, cbar - bbar)
        if key in {"tweedie", "twd"}:
            return bbar
        if key in {"tsi"}:
            return cbar
        raise ValueError(f"Unknown score method {method!r}")

    @torch.no_grad()
    def estimate(self, y: torch.Tensor, t: float, method: str) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for start in range(0, y.shape[0], int(self.cfg.eval_chunk)):
            outs.append(self.estimate_chunk(y[start:start + int(self.cfg.eval_chunk)], t, method))
        out = torch.cat(outs, dim=0)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def ce_score_and_divergence_chunk(self, y: torch.Tensor, t: float, method: str = "ce-hlsi") -> Tuple[torch.Tensor, torch.Tensor]:
        """Analytic CE-HLSI divergence for probability flow.

        The score-bank Tweedie/TSI means and the gate-bank Hessian average may
        come from different proposal samples.  When the banks are shared this
        reduces to the original single-bank formula.
        """
        w, b, c, alpha, gamma = self._weights_and_signals(y, t)
        B, _N, d = b.shape
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)

        wg, bg, _cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t)
        bbar_gate = torch.sum(wg[:, :, None] * bg, dim=1)
        Pgate = self._gate_precision_for_method(method)
        Pbar = torch.sum(wg[:, :, None, None] * Pgate[None, :, :, :], dim=1)
        G = resolvent_gate(Pbar, alpha, gamma, self.cfg.resolvent_eps, self.cfg.gate_clip)
        r = cbar - bbar
        score = bbar + torch.einsum("bij,bj->bi", G, r)

        db = b - bbar[:, None, :]
        dc = c - cbar[:, None, :]
        Cbb = torch.einsum("bn,bni,bnj->bij", w, db, db)
        Ccb = torch.einsum("bn,bni,bnj->bij", w, dc, db)
        I = torch.eye(d, device=y.device, dtype=y.dtype).expand(B, d, d)
        Jb = Cbb - I / gamma
        Jc = Ccb
        Jr = Jc - Jb

        dbg = bg - bbar_gate[:, None, :]
        dP = Pgate[None, :, :, :] - Pbar[:, None, :, :]
        dPdy = torch.einsum("bn,bna,bnuv->bauv", wg, dbg, dP)
        Gr = torch.einsum("bij,bj->bi", G, r)
        gate_trace_term = torch.einsum("bau,bauv,bv->b", G, dPdy, Gr)
        tr_Jb = torch.diagonal(Jb, dim1=-2, dim2=-1).sum(dim=-1)
        tr_GJr = torch.einsum("bij,bji->b", G, Jr)
        div = tr_Jb + tr_GJr - (gamma / torch.clamp(alpha * alpha, min=1.0e-12)) * gate_trace_term
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def ce_score_and_divergence(self, y: torch.Tensor, t: float, method: str = "ce-hlsi") -> Tuple[torch.Tensor, torch.Tensor]:
        scores: List[torch.Tensor] = []
        divs: List[torch.Tensor] = []
        for start in range(0, y.shape[0], int(self.cfg.eval_chunk)):
            s, div = self.ce_score_and_divergence_chunk(y[start:start + int(self.cfg.eval_chunk)], t, method=method)
            scores.append(s)
            divs.append(div)
        return torch.cat(scores, dim=0), torch.cat(divs, dim=0)


# -----------------------------------------------------------------------------
# Alternating DRC primitives
# -----------------------------------------------------------------------------


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


def finalize_density_ratio_weights(raw_rho: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, Dict[str, float | bool]]:
    raw_centered = centered_log_weights(raw_rho)
    beta_target = max(float(cfg.rho_beta), 0.0)
    clip = None if cfg.rho_clip is None or cfg.rho_clip <= 0 else float(cfg.rho_clip)

    def apply(beta: float) -> torch.Tensor:
        out = beta * raw_centered
        if clip is not None:
            out = torch.clamp(out, min=-clip, max=clip)
        return centered_log_weights(out)

    rho = apply(beta_target)
    ess, ess_frac = log_weight_ess(rho)
    beta_eff = beta_target
    adapted = False
    floor = max(float(cfg.rho_ess_floor), 0.0)
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
def reverse_ou_heun_sde(
    target,
    score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    cfg: Config,
    generator: torch.Generator,
    n_samples: Optional[int] = None,
    final_denoise: Optional[bool] = None,
) -> Tuple[torch.Tensor, Dict[str, float | str | bool]]:
    device, dtype, d = target.device, target.dtype, target.d
    n = int(cfg.n_samples if n_samples is None else n_samples)
    use_final_denoise = bool(cfg.final_denoise if final_denoise is None else final_denoise)
    y = torch.randn((n, d), device=device, dtype=dtype, generator=generator)
    ts = make_time_grid(cfg, int(cfg.n_steps), direction="reverse", device=device, dtype=dtype)
    max_abs_score = 0.0
    fail = False
    fail_reason = ""
    for i in range(int(cfg.n_steps)):
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
            y = torch.clamp(y, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))
        if not torch.isfinite(y).all():
            fail, fail_reason = True, "nonfinite state"
            break
    if not fail and use_final_denoise:
        t_min, _t_max = effective_time_bounds(cfg)
        tf = torch.tensor(t_min, device=device, dtype=dtype)
        sf = clamp_norm(score_fn(y, t_min), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(sf.abs().max()))
        if torch.isfinite(sf).all():
            alpha, gamma = alpha_gamma(tf)
            y = (y + gamma * sf) / alpha
            if cfg.sample_clip and cfg.sample_clip > 0:
                y = torch.clamp(y, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))
        else:
            fail, fail_reason = True, "nonfinite final score"
    t_min, t_max = effective_time_bounds(cfg)
    return y.detach(), {
        "failed": bool(fail),
        "fail_reason": fail_reason,
        "max_abs_score": float(max_abs_score),
        "sampler_t_min": float(t_min),
        "sampler_t_max": float(t_max),
        "sampler_time_schedule": canonical_time_schedule(cfg.time_schedule),
    }


@torch.no_grad()
def score_and_hutchinson_divergence(bank: SNISScoreBank, x: torch.Tensor, t: float, method: str, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    s = bank.estimate(x, t, method)
    probes = max(int(cfg.hutchinson_probes), 1)
    eps = float(cfg.hutchinson_eps)
    div_acc = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
    for _ in range(probes):
        # Rademacher probe gives E[v_i v_j]=delta_ij, so v^T J v estimates tr J.
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        sp = bank.estimate(x + eps * v, t, method)
        sm = bank.estimate(x - eps * v, t, method)
        div_acc = div_acc + torch.sum((sp - sm) * v, dim=1) / (2.0 * eps)
    div = div_acc / float(probes)
    return s, torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def bank_score_and_divergence(bank: SNISScoreBank, x: torch.Tensor, t: float, method: str, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    div_mode = str(cfg.pf_divergence).lower()
    key = str(method).lower()
    if div_mode in {"auto", "analytic_ce", "analytic"} and key in {"ce-hlsi", "lfgi", "ce-lfgi", "leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"}:
        return bank.ce_score_and_divergence(x, t, method=method)
    return score_and_hutchinson_divergence(bank, x, t, method, cfg)


@torch.no_grad()
def pf_logprob_bank(bank: SNISScoreBank, x0: torch.Tensor, method: str, cfg: Config) -> Tuple[torch.Tensor, Dict[str, float | bool | str]]:
    """Estimate log q(x0) for the frozen reverse sampler endpoint law.

    Integrates the OU probability-flow ODE forward from t_end to t_start:
        dx/dt = -x - s_t(x),
        log q_0(x0) = log N(x_T) - integral (d + div s_t)(x_t) dt.
    """
    if x0.numel() == 0:
        return torch.empty((0,), device=bank.device, dtype=bank.dtype), {"pf_failed_frac": 0.0}
    # Use the same user-selected time interval and schedule as the reverse sampler,
    # but in the forward direction for endpoint density evaluation.
    ts = make_time_grid(cfg, int(cfg.pf_steps), direction="forward", device=bank.device, dtype=bank.dtype)
    batch = max(int(cfg.rho_batch), 1)
    d = int(x0.shape[1])
    outs: List[torch.Tensor] = []
    failed_total = 0
    max_abs_div = 0.0
    max_abs_state = 0.0
    for start in range(0, x0.shape[0], batch):
        x = x0[start:start + batch].detach().clone()
        A = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        alive = torch.ones((x.shape[0],), device=x.device, dtype=torch.bool)
        for j in range(int(cfg.pf_steps)):
            t = float(ts[j].item())
            tn = float(ts[j + 1].item())
            h = tn - t
            s, div = bank_score_and_divergence(bank, x, t, method, cfg)
            s = clamp_norm(s, cfg.score_clip)
            if cfg.pf_div_clip and cfg.pf_div_clip > 0:
                div = torch.clamp(div, min=-float(cfg.pf_div_clip), max=float(cfg.pf_div_clip))
            v = -x - s
            a = float(d) + div

            x_e = x + h * v
            s_e, div_e = bank_score_and_divergence(bank, x_e, tn, method, cfg)
            s_e = clamp_norm(s_e, cfg.score_clip)
            if cfg.pf_div_clip and cfg.pf_div_clip > 0:
                div_e = torch.clamp(div_e, min=-float(cfg.pf_div_clip), max=float(cfg.pf_div_clip))
            v_e = -x_e - s_e
            a_e = float(d) + div_e

            finite = torch.isfinite(x_e).all(dim=1) & torch.isfinite(v_e).all(dim=1) & torch.isfinite(a) & torch.isfinite(a_e)
            alive = alive & finite
            x = x + 0.5 * h * (v + v_e)
            A = A + 0.5 * h * (a + a_e)
            if cfg.sample_clip and cfg.sample_clip > 0:
                x = torch.clamp(x, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))
            max_abs_div = max(max_abs_div, safe_float(torch.max(torch.abs(torch.cat([div.reshape(-1), div_e.reshape(-1)])))))
            max_abs_state = max(max_abs_state, safe_float(x.abs().max()))
        logq = standard_normal_logprob(x) - A
        good = alive & torch.isfinite(logq)
        failed_total += int((~good).sum().item())
        if (~good).any():
            replacement = torch.nanmedian(logq[good]) if good.any() else torch.tensor(0.0, device=x.device, dtype=x.dtype)
            logq = torch.where(good, logq, replacement)
        outs.append(logq.detach())
    logq_all = torch.cat(outs, dim=0)
    return logq_all, {
        "pf_method": str(method),
        "pf_divergence_mode": str(cfg.pf_divergence),
        "pf_failed_frac": float(failed_total) / float(max(1, x0.shape[0])),
        "pf_steps": int(cfg.pf_steps),
        "pf_t_min": float(effective_time_bounds(cfg)[0]),
        "pf_t_max": float(effective_time_bounds(cfg)[1]),
        "pf_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "pf_max_abs_div": float(max_abs_div),
        "pf_max_abs_state": float(max_abs_state),
        "pf_logq_mean": safe_float(logq_all.mean()),
        "pf_logq_std": safe_float(logq_all.std(unbiased=False)),
        "pf_logq_min": safe_float(logq_all.min()),
        "pf_logq_max": safe_float(logq_all.max()),
    }



# -----------------------------------------------------------------------------
# Likelihood-correction calibration against particle KDE
# -----------------------------------------------------------------------------


def _center_finite(x: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(x)
    if int(mask.sum().item()) == 0:
        return x * float("nan")
    xc = x.clone()
    xc[mask] = xc[mask] - xc[mask].mean()
    return xc


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return float("nan")
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).clamp_min(1.0e-30)
    return safe_float(torch.sum(x * y) / denom)


def centered_rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 1:
        return float("nan")
    xc = x[mask] - x[mask].mean()
    yc = y[mask] - y[mask].mean()
    return safe_float(torch.sqrt(torch.mean((xc - yc) ** 2)))


def calibration_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    """Least-squares slope y ~= a + slope*x on finite entries."""
    x = x.reshape(-1)
    y = y.reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return float("nan")
    xc = x[mask] - x[mask].mean()
    yc = y[mask] - y[mask].mean()
    varx = torch.sum(xc * xc)
    if safe_float(varx) <= 1.0e-30:
        return float("nan")
    return safe_float(torch.sum(xc * yc) / varx)


def pairwise_order_accuracy(x: torch.Tensor, y: torch.Tensor, max_pairs: int = 20000, seed: int = 0) -> float:
    """Agreement of pairwise orderings for centered scalar corrections."""
    x = x.reshape(-1)
    y = y.reshape(-1)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.numel())
    if n < 3:
        return float("nan")
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    m = min(int(max_pairs), n * (n - 1) // 2)
    i = torch.randint(0, n, (m,), device=x.device, generator=gen)
    j = torch.randint(0, n, (m,), device=x.device, generator=gen)
    good = i != j
    i, j = i[good], j[good]
    if i.numel() == 0:
        return float("nan")
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    nz = (dx.abs() > 1.0e-12) & (dy.abs() > 1.0e-12)
    if int(nz.sum().item()) == 0:
        return float("nan")
    return safe_float((torch.sign(dx[nz]) == torch.sign(dy[nz])).to(x.dtype).mean())


@torch.no_grad()
def gaussian_kde_log_density(
    eval_x: torch.Tensor,
    ref_x: torch.Tensor,
    cfg: Config,
    eval_indices: Optional[torch.Tensor] = None,
    ref_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Full-dimensional Gaussian KDE log density with optional leave-one-out.

    This is used only as a particle-density diagnostic for q_j.  For the common
    case eval_x == ref_x[prefix], pass matching eval_indices/ref_indices so the
    self-kernel can be removed from the logsumexp and the denominator corrected.
    """
    eval_x = eval_x.detach()
    ref_x = ref_x.detach()
    n_ref, d = int(ref_x.shape[0]), int(ref_x.shape[1])
    if n_ref <= 1 or eval_x.numel() == 0:
        return torch.zeros((eval_x.shape[0],), device=eval_x.device, dtype=eval_x.dtype), 1.0
    if float(cfg.kde_bandwidth) > 0:
        h = float(cfg.kde_bandwidth)
    else:
        h0 = median_bandwidth(ref_x, max_n=min(1200, n_ref))
        h = h0 * (float(n_ref) ** (-1.0 / float(d + 4)))
        h = max(float(h), float(cfg.kde_min_bandwidth))
    h2 = max(h * h, 1.0e-12)
    log_norm = -0.5 * d * math.log(2.0 * math.pi) - d * math.log(h)
    outs: List[torch.Tensor] = []
    chunk = max(int(cfg.kde_chunk), 1)
    has_indices = eval_indices is not None and ref_indices is not None
    if has_indices:
        eval_indices = eval_indices.to(device=eval_x.device)
        ref_indices = ref_indices.to(device=eval_x.device)
    for start in range(0, eval_x.shape[0], chunk):
        xx = eval_x[start:start + chunk]
        d2 = pairwise_sq_dists(xx, ref_x)
        logits = -0.5 * d2 / h2
        denom = torch.full((xx.shape[0],), float(n_ref), device=xx.device, dtype=xx.dtype)
        if has_indices:
            ei = eval_indices[start:start + chunk]
            eq = ei[:, None] == ref_indices[None, :]
            if bool(eq.any().detach().cpu().item()):
                logits = logits.masked_fill(eq, -float("inf"))
                denom = denom - eq.any(dim=1).to(xx.dtype)
        denom = denom.clamp_min(1.0)
        lse = torch.logsumexp(logits, dim=1)
        # If all kernels were removed (pathological n_ref=1), use a broad fallback.
        lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
        outs.append(log_norm + lse - torch.log(denom))
    return torch.cat(outs, dim=0), float(h)


@torch.no_grad()
def likelihood_correction_calibration(
    target,
    refs: torch.Tensor,
    logq_pf: torch.Tensor,
    raw_rho_pf: torch.Tensor,
    cfg: Config,
) -> Dict[str, float]:
    """Compare PF likelihood correction to a full-dimensional KDE correction.

    The diagnostic is intentionally centered: both PF and KDE densities are only
    meaningful up to practical smoothing/normalization error for this purpose.
    Low centered RMSE and high correlation/order accuracy indicate that the
    estimator's probability-flow correction is coherent with an independent
    particle-density view of the proposal bank.
    """
    if not bool(cfg.likelihood_calibration):
        return {}
    n_total = int(refs.shape[0])
    if n_total < 5:
        return {"calib_n": float(n_total)}
    n_eval = min(int(cfg.kde_n_eval), n_total)
    n_fit = min(int(cfg.kde_n_fit), n_total)
    eval_idx = torch.arange(n_eval, device=refs.device)
    ref_idx = torch.arange(n_fit, device=refs.device)
    eval_x = refs[:n_eval]
    ref_x = refs[:n_fit]
    logq_kde, h = gaussian_kde_log_density(eval_x, ref_x, cfg, eval_indices=eval_idx, ref_indices=ref_idx)
    logq_pf_eval = logq_pf[:n_eval]
    logpi_eval = target.log_prob(eval_x, t=0.0)
    rho_pf_eval = raw_rho_pf[:n_eval]
    rho_kde = logpi_eval - logq_kde
    return {
        "calib_n": float(n_eval),
        "calib_kde_n_fit": float(n_fit),
        "calib_kde_bandwidth": float(h),
        "calib_logq_pf_vs_kde_corr": pearson_corr(logq_pf_eval, logq_kde),
        "calib_logq_pf_vs_kde_centered_rmse": centered_rmse(logq_pf_eval, logq_kde),
        "calib_logq_pf_to_kde_slope": calibration_slope(logq_pf_eval, logq_kde),
        "calib_rho_pf_vs_kde_corr": pearson_corr(rho_pf_eval, rho_kde),
        "calib_rho_pf_vs_kde_centered_rmse": centered_rmse(rho_pf_eval, rho_kde),
        "calib_rho_pf_to_kde_slope": calibration_slope(rho_pf_eval, rho_kde),
        "calib_rho_pair_order_acc": pairwise_order_accuracy(rho_pf_eval, rho_kde, seed=int(cfg.seed + n_total)),
        "calib_logq_pf_std": safe_float(logq_pf_eval.std(unbiased=False)),
        "calib_logq_kde_std": safe_float(logq_kde.std(unbiased=False)),
        "calib_rho_pf_std": safe_float(rho_pf_eval.std(unbiased=False)),
        "calib_rho_kde_std": safe_float(rho_kde.std(unbiased=False)),
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


@torch.no_grad()
def nll_metric(target, samples: torch.Tensor) -> float:
    return safe_float((-target.log_prob(samples, t=0.0)).mean())


@torch.no_grad()
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: Optional[float] = None, max_n: int = 2000) -> float:
    n = min(max_n, x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    if bandwidth is None:
        bandwidth = median_bandwidth(x, y, max_n=min(1200, 2 * n))
    h2 = max(float(bandwidth) ** 2, 1.0e-12)
    xx = pairwise_sq_dists(x, x)
    yy = pairwise_sq_dists(y, y)
    xy = pairwise_sq_dists(x, y)
    val = torch.exp(-xx / (2.0 * h2)).mean() + torch.exp(-yy / (2.0 * h2)).mean() - 2.0 * torch.exp(-xy / (2.0 * h2)).mean()
    return safe_float(torch.sqrt(torch.clamp(val, min=0.0)))


@torch.no_grad()
def ksd_rbf(target, samples: torch.Tensor, bandwidth: Optional[float] = None, max_n: int = 1000) -> float:
    n = min(max_n, samples.shape[0])
    X = samples[:n]
    if n < 5 or not torch.isfinite(X).all():
        return float("nan")
    S = target.score(X, t=0.0)
    if bandwidth is None:
        bandwidth = median_bandwidth(X, max_n=min(1200, n))
    h2 = max(float(bandwidth) ** 2, 1.0e-12)
    d = X.shape[1]
    d2 = pairwise_sq_dists(X, X)
    K = torch.exp(-d2 / (2.0 * h2))
    diff = X[:, None, :] - X[None, :, :]
    term1 = K * (S @ S.T)
    term2 = K * torch.sum((S[:, None, :] - S[None, :, :]) * diff, dim=-1) / h2
    term3 = K * (d / h2 - d2 / (h2 * h2))
    ksd2 = (term1 + term2 + term3).mean()
    return safe_float(torch.sqrt(torch.clamp(ksd2, min=0.0)))


@torch.no_grad()
def sliced_w2(x: torch.Tensor, y: torch.Tensor, n_proj: int, generator: torch.Generator, max_n: int = 2000) -> float:
    n = min(max_n, x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    dirs = torch.randn((int(n_proj), x.shape[1]), device=x.device, dtype=x.dtype, generator=generator)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp(min=1.0e-30)
    xp = torch.sort(x @ dirs.T, dim=0).values
    yp = torch.sort(y @ dirs.T, dim=0).values
    return safe_float(torch.sqrt(torch.mean((xp - yp) ** 2)))


@torch.no_grad()
def sliced_ks(x: torch.Tensor, y: torch.Tensor, n_proj: int, generator: torch.Generator, max_n: int = 2000) -> float:
    n = min(max_n, x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    dirs = torch.randn((int(n_proj), x.shape[1]), device=x.device, dtype=x.dtype, generator=generator)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp(min=1.0e-30)
    xp = (x @ dirs.T).detach().cpu().numpy()
    yp = (y @ dirs.T).detach().cpu().numpy()
    vals = []
    for j in range(xp.shape[1]):
        xs = np.sort(xp[:, j])
        ys = np.sort(yp[:, j])
        grid = np.concatenate([xs, ys])
        cdfx = np.searchsorted(xs, grid, side="right") / max(len(xs), 1)
        cdfy = np.searchsorted(ys, grid, side="right") / max(len(ys), 1)
        vals.append(np.max(np.abs(cdfx - cdfy)))
    return float(np.mean(vals))


@torch.no_grad()
def mode_mass_l1(target, samples: torch.Tensor) -> float:
    if not hasattr(target, "responsibilities") or int(getattr(target, "K", 0)) <= 0:
        return float("nan")
    resp = target.responsibilities(samples, t=0.0)
    assign = torch.argmax(resp, dim=1)
    fracs = torch.stack([(assign == k).to(target.dtype).mean() for k in range(target.K)])
    return safe_float(torch.sum(torch.abs(fracs - target.weights)))


@torch.no_grad()
def moment_errors(samples: torch.Tensor) -> Tuple[float, float]:
    m = samples.mean(dim=0)
    X = samples - m
    C = (X.T @ X) / max(int(samples.shape[0]) - 1, 1)
    I = torch.eye(samples.shape[1], device=samples.device, dtype=samples.dtype)
    return safe_float(torch.linalg.norm(m)), safe_float(torch.linalg.matrix_norm(C - I, ord="fro"))


@torch.no_grad()
def fisher_rmse(target, score_fn: Callable[[torch.Tensor, float], torch.Tensor], cfg: Config, generator: torch.Generator) -> float:
    if int(cfg.fisher_n_t) <= 0 or int(cfg.fisher_n_per_t) <= 0:
        return float("nan")
    t_min, t_max = effective_time_bounds(cfg)
    t_min = max(float(t_min), 1.0e-6)
    t_max = max(float(t_max), t_min)
    if cfg.fisher_time_grid == "linear":
        t_grid = torch.linspace(t_min, t_max, int(cfg.fisher_n_t), device=target.device, dtype=target.dtype)
    else:
        t_grid = torch.exp(torch.linspace(math.log(t_min), math.log(t_max), int(cfg.fisher_n_t), device=target.device, dtype=target.dtype))
    vals = []
    for tt in t_grid:
        t = float(tt.item())
        y = target.sample_pt(int(cfg.fisher_n_per_t), t, generator=generator)
        s_true = target.score(y, t=t)
        s_hat = score_fn(y, t)
        vals.append(torch.mean(torch.sum((s_hat - s_true) ** 2, dim=1)))
    return float(math.sqrt(max(safe_float(torch.mean(torch.stack(vals))), 0.0)))


@torch.no_grad()
def compute_metrics(target, samples: torch.Tensor, truth: torch.Tensor, score_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]], cfg: Config, generator: torch.Generator) -> Dict[str, float]:
    n = min(int(cfg.metrics_max_n), samples.shape[0], truth.shape[0])
    x = samples[:n]
    y = truth[:n]
    if not torch.isfinite(x).all():
        return {"metric_n": n, "nll": float("nan"), "mmd": float("nan"), "ksd": float("nan"), "sw2": float("nan"), "sliced_ks": float("nan"), "mode_l1": float("nan"), "mean_norm": float("nan"), "cov_frob_err": float("nan"), "fisher_rmse": float("nan")}
    mean_norm, cov_err = moment_errors(x)
    out = {
        "metric_n": float(n),
        "nll": nll_metric(target, x),
        "mmd": mmd_rbf(x, y, max_n=n),
        "ksd": ksd_rbf(target, x, max_n=min(n, 1000)),
        "sw2": sliced_w2(x, y, cfg.sw2_projections, generator=generator, max_n=n),
        "sliced_ks": sliced_ks(x, y, cfg.sw2_projections, generator=generator, max_n=n),
        "mode_l1": mode_mass_l1(target, x),
        "mean_norm": mean_norm,
        "cov_frob_err": cov_err,
    }
    out["fisher_rmse"] = fisher_rmse(target, score_fn, cfg, generator) if score_fn is not None else float("nan")
    return out


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def fit_pca_projection(truth: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    X = truth.detach().cpu().double().numpy()
    mean = X.mean(axis=0)
    Xc = X - mean[None, :]
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    basis = vt[:2].T
    return mean, basis


def project_np(x: torch.Tensor, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    X = x.detach().cpu().double().numpy()
    return (X - mean[None, :]) @ basis


def plot_heatmap_panel(ax, pts2: np.ndarray, title: str, lims: Tuple[float, float, float, float], cfg: Config, vmax: Optional[float] = None):
    H, xe, ye = np.histogram2d(pts2[:, 0], pts2[:, 1], bins=int(cfg.hist_bins), range=[[lims[0], lims[1]], [lims[2], lims[3]]], density=False)
    H = H.T.astype(np.float64)
    if H.sum() > 0:
        H = H / H.sum()
    if vmax is None:
        vals = H[H > 0]
        vmax = float(np.quantile(vals, float(cfg.hist_vmax_quantile))) if vals.size else 1.0
    ax.imshow(H, origin="lower", extent=lims, aspect="auto", norm=PowerNorm(gamma=float(cfg.hist_gamma), vmin=0.0, vmax=max(vmax, 1.0e-12)))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return vmax


def make_projection_limits(arrays: List[np.ndarray], pad: float = 0.08) -> Tuple[float, float, float, float]:
    Z = np.concatenate(arrays, axis=0)
    lo = np.quantile(Z, 0.005, axis=0)
    hi = np.quantile(Z, 0.995, axis=0)
    span = np.maximum(hi - lo, 1.0)
    lo = lo - pad * span
    hi = hi + pad * span
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def save_heatmaps(outdir: str, target, truth: torch.Tensor, init_refs: torch.Tensor, samples_by_family_round: Dict[str, List[torch.Tensor]], cfg: Config):
    mean, basis = fit_pca_projection(truth)
    np.savez(os.path.join(outdir, "projection_basis.npz"), mean=mean, basis=basis)
    truth2 = project_np(truth, mean, basis)
    init2 = project_np(init_refs, mean, basis)
    final_arrays = [truth2, init2]
    for fam, arrs in samples_by_family_round.items():
        if arrs:
            final_arrays.append(project_np(arrs[-1], mean, basis))
    lims = make_projection_limits(final_arrays)
    vmax = None
    # Use truth's nonzero quantile as shared scale so weak/spread methods remain visible.
    H_truth, _, _ = np.histogram2d(truth2[:, 0], truth2[:, 1], bins=int(cfg.hist_bins), range=[[lims[0], lims[1]], [lims[2], lims[3]]])
    H_truth = H_truth.T
    if H_truth.sum() > 0:
        H_truth = H_truth / H_truth.sum()
        vals = H_truth[H_truth > 0]
        if vals.size:
            vmax = float(np.quantile(vals, float(cfg.hist_vmax_quantile)))

    init_label = "Initial target refs" if canonical_initial_reference_mode(cfg.initial_reference_mode) == "target" else "Initial N(0,I) prior refs"
    panels = [("Target truth (eval only)", truth2), (init_label, init2)]
    family_order = list(samples_by_family_round.keys())
    for fam in family_order:
        if samples_by_family_round.get(fam):
            panels.append((f"{fam} alternating DRC round {len(samples_by_family_round[fam])}", project_np(samples_by_family_round[fam][-1], mean, basis)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, pts) in zip(axes, panels):
        plot_heatmap_panel(ax, pts, title, lims, cfg, vmax=vmax)
    fig.suptitle(f"PCA heatmaps: {getattr(target, 'name', getattr(target, '__class__', type(target)).__name__)}", fontsize=12)
    fig.savefig(os.path.join(outdir, "heatmaps_final.png"), dpi=220)
    plt.close(fig)

    ncols = int(cfg.n_rounds) + 2
    nrows = max(1, len(family_order))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(nrows, ncols)
    for row, fam in enumerate(family_order):
        plot_heatmap_panel(axes[row, 0], truth2, "Target", lims, cfg, vmax=vmax)
        plot_heatmap_panel(axes[row, 1], init2, f"{fam}: {init_label}", lims, cfg, vmax=vmax)
        arrs = samples_by_family_round.get(fam, [])
        for j in range(int(cfg.n_rounds)):
            ax = axes[row, j + 2]
            if j < len(arrs):
                plot_heatmap_panel(ax, project_np(arrs[j], mean, basis), f"{fam}: round {j+1}", lims, cfg, vmax=vmax)
            else:
                ax.axis("off")
    fig.suptitle("Alternating DRC progression by estimator family", fontsize=12)
    fig.savefig(os.path.join(outdir, "heatmaps_by_round.png"), dpi=220)
    plt.close(fig)


def save_metric_curves(outdir: str, rows: List[Dict[str, object]]):
    try:
        import pandas as pd
    except Exception:
        return
    df = pd.DataFrame(rows)
    df = df[df["kind"] == "sample"].copy()
    if df.empty:
        return
    metrics = ["mmd", "ksd", "sw2", "sliced_ks", "mode_l1", "fisher_rmse"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            ax.axis("off")
            continue
        family_order = list(dict.fromkeys(df["family"].tolist()))
        for fam in family_order:
            sub = df[df["family"] == fam].sort_values("round")
            ax.plot(sub["round"], sub[metric], marker="o", label=fam)
        ax.set_title(metric + " ↓")
        ax.set_xlabel("alternating round")
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Alternating DRC metric curves", fontsize=12)
    fig.savefig(os.path.join(outdir, "metric_curves.png"), dpi=220)
    plt.close(fig)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    # Union of keys, stable-ish ordering.
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------


@torch.no_grad()
def initial_log_weights(target, init_refs: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, Dict[str, float | bool | str]]:
    ref_mode = canonical_initial_reference_mode(cfg.initial_reference_mode)
    if ref_mode == "target":
        # Oracle target references represent q0=p0, so log pi - log q0 is zero.
        # Using prior_ratio here would answer a different question and would
        # artificially reweight exact target samples by p0/N(0,I).
        rho = torch.zeros((init_refs.shape[0],), device=init_refs.device, dtype=init_refs.dtype)
        ess, ess_frac = log_weight_ess(rho)
        return rho, {
            "initial_reference_mode": ref_mode,
            "initial_weight_mode": "zero_oracle_target",
            "initial_rho_ess": ess,
            "initial_rho_ess_frac": ess_frac,
            "initial_weight_mode_requested": str(cfg.initial_weight_mode),
        }

    mode = str(cfg.initial_weight_mode).lower()
    if mode == "zero":
        rho = torch.zeros((init_refs.shape[0],), device=init_refs.device, dtype=init_refs.dtype)
        ess, ess_frac = log_weight_ess(rho)
        return rho, {
            "initial_reference_mode": ref_mode,
            "initial_weight_mode": "zero",
            "initial_rho_ess": ess,
            "initial_rho_ess_frac": ess_frac,
        }
    if mode != "prior_ratio":
        raise ValueError("initial_weight_mode must be prior_ratio or zero")
    raw = target.log_prob(init_refs, t=0.0) - standard_normal_logprob(init_refs)
    rho, info = finalize_density_ratio_weights(raw, cfg)
    info = {f"initial_{k}": v for k, v in info.items()}
    info["initial_reference_mode"] = ref_mode
    info["initial_weight_mode"] = "prior_ratio"
    return rho, info



def family_seed_offset(family: str) -> int:
    table = {"Blend": 17, "LFGI": 53, "Leaf-LFGI": 89, "Tweedie": 131, "None": 173}
    if family in table:
        return table[family]
    # Stable across Python processes, unlike hash().
    return 100 + (sum((i + 1) * ord(ch) for i, ch in enumerate(str(family))) % 10_000)


@torch.no_grad()
def weighted_resample(x: torch.Tensor, logw: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    """Return an unweighted particle cloud representing a weighted empirical law."""
    if x.numel() == 0:
        return x
    lw = torch.nan_to_num(logw.detach().to(device=x.device, dtype=x.dtype).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if lw.shape[0] != x.shape[0]:
        lw = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
    probs = torch.softmax(lw, dim=0)
    idx = torch.multinomial(probs, int(n), replacement=True, generator=generator)
    return x[idx].detach()


def blank_pf_info(method: str, reason: str = "skipped") -> Dict[str, float | bool | str]:
    return {
        "pf_method": str(method),
        "pf_divergence_mode": "none",
        "pf_skipped": True,
        "pf_skip_reason": reason,
        "pf_failed_frac": 0.0,
        "pf_steps": 0,
        "pf_max_abs_div": float("nan"),
        "pf_max_abs_state": float("nan"),
        "pf_logq_mean": float("nan"),
        "pf_logq_std": float("nan"),
        "pf_logq_min": float("nan"),
        "pf_logq_max": float("nan"),
    }


def blank_calibration_info(reason: str = "skipped") -> Dict[str, float | str]:
    return {
        "calib_skipped": True,
        "calib_skip_reason": reason,
        "calib_logq_pf_vs_kde_corr": float("nan"),
        "calib_logq_pf_vs_kde_centered_rmse": float("nan"),
        "calib_logq_pf_to_kde_slope": float("nan"),
        "calib_rho_pf_vs_kde_corr": float("nan"),
        "calib_rho_pf_vs_kde_centered_rmse": float("nan"),
        "calib_rho_pf_to_kde_slope": float("nan"),
        "calib_rho_pair_order_acc": float("nan"),
        "calib_kde_bandwidth": float("nan"),
    }


@torch.no_grad()
def run_family(
    family: str,
    transport_method: str,
    correction_method: str,
    target,
    init_refs: torch.Tensor,
    init_rho: torch.Tensor,
    truth: torch.Tensor,
    cfg: Config,
) -> Tuple[List[torch.Tensor], List[Dict[str, object]], List[Dict[str, object]]]:
    current_pool = init_refs.detach().clone()
    current_rho = init_rho.detach().clone()
    samples_by_round: List[torch.Tensor] = []
    metric_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []

    for r in range(1, int(cfg.n_rounds) + 1):
        round_t0 = time.time()
        score_refs, score_rho, gate_refs, gate_rho, split_info = split_score_gate_banks(current_pool, current_rho, cfg)
        bank = SNISScoreBank(
            target,
            score_refs,
            cfg,
            log_ref_weights=score_rho,
            gate_anchors=gate_refs,
            gate_log_ref_weights=gate_rho,
        )
        gen = make_generator(int(cfg.seed + 10_000 * r + family_seed_offset(family)), target.device)
        next_pool_n = proposal_pool_size(cfg)
        generate_n = max(int(cfg.n_samples), int(next_pool_n))

        # S-step / transport field.  The special transport method ``none`` is a
        # true no-op on coordinates: it never instantiates a reverse score field.
        # For plots/metrics, we resample from the current weighted empirical law
        # represented by the score bank, while the next proposal pool preserves
        # the current split-compatible pool coordinates.
        if str(transport_method).lower() == "none":
            score_fn = None
            samples_all = current_pool[:next_pool_n].detach()
            if int(cfg.n_samples) <= int(samples_all.shape[0]):
                samples_eval = samples_all[:int(cfg.n_samples)].detach()
            else:
                samples_eval = weighted_resample(score_refs, score_rho, int(cfg.n_samples), gen)
            sampler_info = {
                "failed": False,
                "fail_reason": "",
                "max_abs_score": 0.0,
                "transport_none": True,
                "generated_n": int(samples_all.shape[0]),
                "sampler_t_min": float(effective_time_bounds(cfg)[0]),
                "sampler_t_max": float(effective_time_bounds(cfg)[1]),
                "sampler_time_schedule": canonical_time_schedule(cfg.time_schedule),
            }
        else:
            # This is the field used to generate R_{j+1}.  We generate enough
            # samples to populate both the next score bank and the next gate bank.
            score_fn = lambda y, t, bank=bank, method=transport_method: bank.estimate(y, t, method)
            samples_all, sampler_info = reverse_ou_heun_sde(
                target,
                score_fn,
                cfg,
                generator=gen,
                n_samples=generate_n,
                final_denoise=cfg.final_denoise,
            )
            sampler_info["generated_n"] = int(samples_all.shape[0])
            if cfg.eval_final_denoise:
                samples_eval, _ = reverse_ou_heun_sde(
                    target,
                    score_fn,
                    cfg,
                    generator=make_generator(int(cfg.seed + 999_000 + 10_000 * r + family_seed_offset(family)), target.device),
                    n_samples=int(cfg.n_samples),
                    final_denoise=True,
                )
            else:
                samples_eval = samples_all[:int(cfg.n_samples)].detach()
        samples_by_round.append(samples_eval.detach())

        metric_gen = make_generator(int(cfg.seed + 220_000 + 10_000 * r + family_seed_offset(family)), target.device)
        metrics = compute_metrics(target, samples_eval, truth, score_fn, cfg, metric_gen)
        in_ess, in_ess_frac = log_weight_ess(score_rho)
        row = {
            "kind": "sample",
            "family": family,
            "method": f"{transport_method}_{correction_method}",
            "transport_method": transport_method,
            "correction_method": correction_method,
            "round": int(r),
            "input_ref_n": int(score_refs.shape[0]),
            "input_gate_n": int(gate_refs.shape[0]),
            "input_pool_n": int(current_pool.shape[0]),
            "bank_coupling": split_info["bank_coupling"],
            "score_slice": split_info["score_slice"],
            "gate_slice": split_info["gate_slice"],
            "bank_overlap_n": int(split_info["bank_overlap_n"]),
            "input_rho_ess": in_ess,
            "input_rho_ess_frac": in_ess_frac,
            "sampler_failed": bool(sampler_info.get("failed", False)),
            "sampler_fail_reason": str(sampler_info.get("fail_reason", "")),
            "sampler_max_abs_score": safe_float(sampler_info.get("max_abs_score", float("nan"))),
            "sampler_t_min": safe_float(sampler_info.get("sampler_t_min", effective_time_bounds(cfg)[0])),
            "sampler_t_max": safe_float(sampler_info.get("sampler_t_max", effective_time_bounds(cfg)[1])),
            "sampler_time_schedule": str(sampler_info.get("sampler_time_schedule", canonical_time_schedule(cfg.time_schedule))),
            "generated_n": int(sampler_info.get("generated_n", samples_all.shape[0])),
            "elapsed_sec_so_far": float(time.time() - round_t0),
            **metrics,
        }
        metric_rows.append(row)

        # R-step: compute weights for the next S-step on the entire split-compatible
        # proposal pool, not only on the score slice.  This is necessary when
        # bank_coupling=independent because the gate-only samples need their own
        # density-ratio weights in the following round.
        next_pool = samples_all[:next_pool_n].detach()
        next_score_refs, _next_score_rho_placeholder, next_gate_refs, _next_gate_rho_placeholder, next_split_info = split_score_gate_banks(
            next_pool,
            torch.zeros((next_pool.shape[0],), device=next_pool.device, dtype=next_pool.dtype),
            cfg,
        )
        pf_t0 = time.time()
        logpi = target.log_prob(next_pool, t=0.0)
        if str(correction_method).lower() == "none":
            logq = torch.full_like(logpi, float("nan"))
            raw_rho = torch.zeros_like(logpi)
            pf_info = blank_pf_info(correction_method, reason="correction_method_none")
            calib_info = blank_calibration_info(reason="correction_method_none")
        else:
            # This estimates the endpoint density used in rho = log pi - log q.
            # Hybrid runs intentionally allow this field to differ from the
            # transport field above.  Tweedie corrections are supported and use
            # the generic Hutchinson divergence path unless pf_divergence forces
            # another mode.
            logq, pf_info = pf_logprob_bank(bank, next_pool, correction_method, cfg)
            raw_rho = logpi - logq
            # Keep the expensive PF-vs-KDE diagnostic on the score slice so the
            # reported calibration remains comparable across coupling modes.
            next_score_start = int(effective_gate_n(cfg)) if next_split_info["bank_coupling"] == "independent" else 0
            score_raw_rho = raw_rho[next_score_start:next_score_start + int(cfg.n_ref)]
            score_logq = logq[next_score_start:next_score_start + int(cfg.n_ref)]
            calib_info = likelihood_correction_calibration(target, next_score_refs, score_logq, score_raw_rho, cfg)
        next_rho, rho_info = finalize_density_ratio_weights(raw_rho, cfg)
        pf_elapsed = time.time() - pf_t0
        mode_before = mode_mass_l1(target, next_score_refs)
        next_score_start = int(effective_gate_n(cfg)) if next_split_info["bank_coupling"] == "independent" else 0
        next_score_rho = next_rho[next_score_start:next_score_start + int(cfg.n_ref)]
        w = torch.exp(next_score_rho - torch.max(next_score_rho))
        w = w / torch.clamp(w.sum(), min=1.0e-30)
        if hasattr(target, "responsibilities") and int(getattr(target, "K", 0)) > 0:
            resp = target.responsibilities(next_score_refs, t=0.0)
            weighted_mode = torch.einsum("n,nk->k", w, resp)
            weighted_mode_l1 = safe_float(torch.sum(torch.abs(weighted_mode - target.weights)))
        else:
            weighted_mode_l1 = float("nan")
        stage_rows.append({
            "family": family,
            "method": f"{transport_method}_{correction_method}",
            "transport_method": transport_method,
            "correction_method": correction_method,
            "round": int(r),
            "r_step_ref_n": int(next_score_refs.shape[0]),
            "r_step_gate_n": int(next_gate_refs.shape[0]),
            "r_step_pool_n": int(next_pool.shape[0]),
            "bank_coupling": next_split_info["bank_coupling"],
            "score_slice": next_split_info["score_slice"],
            "gate_slice": next_split_info["gate_slice"],
            "bank_overlap_n": int(next_split_info["bank_overlap_n"]),
            "r_step_elapsed_sec": float(pf_elapsed),
            "raw_rho_mean": safe_float(raw_rho.mean()),
            "raw_rho_std": safe_float(raw_rho.std(unbiased=False)),
            "logpi_mean": safe_float(logpi.mean()),
            "logpi_std": safe_float(logpi.std(unbiased=False)),
            "mode_l1_unweighted_next_refs": mode_before,
            "mode_l1_weighted_next_refs": weighted_mode_l1,
            **pf_info,
            **calib_info,
            **(bank.mp_leaf_info if any(str(m).lower().replace("_", "-") in {"leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"} for m in (transport_method, correction_method)) else {}),
            **rho_info,
        })

        # Option B alternating update: the next S-step uses the same coordinates
        # produced by this S-step, plus the endpoint ratio weights from the R-step.
        current_pool = next_pool
        current_rho = next_rho.detach()
        print(
            f"[{family} | S={transport_method}, R={correction_method}] round {r}/{cfg.n_rounds}: "
            f"MMD={metrics['mmd']:.4g}, KSD={metrics['ksd']:.4g}, SW2={metrics['sw2']:.4g}, "
            f"FisherRMSE={metrics['fisher_rmse']:.4g}, rhoESS={rho_info['rho_ess_frac']:.3f}, "
            f"bank={next_split_info['bank_coupling']} score_n={next_split_info['score_n']} gate_n={next_split_info['gate_n']}, "
            f"pf_fail={pf_info['pf_failed_frac']:.3f}, "
            f"rhoPF-KDEcorr={calib_info.get('calib_rho_pf_vs_kde_corr', float('nan')):.3f}",
            flush=True,
        )
    return samples_by_round, metric_rows, stage_rows




def _estimator_alias_table() -> Dict[str, Tuple[str, str]]:
    """Map user-facing estimator aliases to (short display, internal method key)."""
    return {
        "blend": ("Blend", "blend"),
        "blended": ("Blend", "blend"),
        "scalar-blend": ("Blend", "blend"),
        "scalar": ("Blend", "blend"),
        "lfgi": ("LFGI", "ce-hlsi"),
        "ce-hlsi": ("LFGI", "ce-hlsi"),
        "ce_hlsi": ("LFGI", "ce-hlsi"),
        "ce-lfgi": ("LFGI", "ce-hlsi"),
        "leaf-lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "leaf_lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp-leaf-lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp_leaf_lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp-leaf-ce-hlsi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp_leaf_ce_hlsi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "tweedie": ("Tweedie", "tweedie"),
        "twd": ("Tweedie", "tweedie"),
        "none": ("None", "none"),
        "no-op": ("None", "none"),
        "noop": ("None", "none"),
        "no_correction": ("None", "none"),
        "no-correction": ("None", "none"),
        "zero": ("None", "none"),
    }


def _normalize_estimator_alias(key: str) -> Optional[Tuple[str, str]]:
    aliases = _estimator_alias_table()
    raw = str(key).strip().lower()
    candidates = [raw, raw.replace("_", "-"), raw.replace("-", "_")]
    for cand in candidates:
        if cand in aliases:
            return aliases[cand]
    return None


def _parse_method_token(token: str) -> Tuple[str, str, str]:
    """Resolve one --methods token to (display family, transport method, correction method).

    Atomic tokens such as ``blend`` or ``lfgi`` mean diagonal pairs.  Hybrid tokens
    use transport_correction order, for example ``blend_lfgi`` means Blend for the
    S-step transport and LFGI for the R-step likelihood correction.  We also accept
    ``transport:correction``, ``transport/correction``, and ``transport->correction``.
    """
    raw = str(token).strip().lower()
    if not raw:
        raise ValueError("empty method token")

    # Explicit separators first; underscores are handled below because aliases
    # such as ce_hlsi and leaf_lfgi also contain underscores.
    for sep in ("->", ":", "/"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            lspec = _normalize_estimator_alias(left)
            rspec = _normalize_estimator_alias(right)
            if lspec is None or rspec is None:
                break
            ldisp, lmethod = lspec
            rdisp, rmethod = rspec
            return f"{ldisp}->{rdisp}", lmethod, rmethod

    # Atomic diagonal alias.
    spec = _normalize_estimator_alias(raw)
    if spec is not None:
        disp, method = spec
        return f"{disp}->{disp}", method, method

    # Hybrid underscore syntax.  Try every split position so ce_hlsi_blend and
    # blend_leaf_lfgi can still be parsed unambiguously.
    parts = raw.split("_")
    for i in range(1, len(parts)):
        left = "_".join(parts[:i])
        right = "_".join(parts[i:])
        lspec = _normalize_estimator_alias(left)
        rspec = _normalize_estimator_alias(right)
        if lspec is not None and rspec is not None:
            ldisp, lmethod = lspec
            rdisp, rmethod = rspec
            return f"{ldisp}->{rdisp}", lmethod, rmethod

    valid = ", ".join(sorted(_estimator_alias_table().keys()) + ["all", "hybrids"])
    raise ValueError(
        f"Unknown method/hybrid token {token!r}. Use atomic aliases or transport_correction "
        f"tokens like blend_lfgi, lfgi_none, none_lfgi, or tweedie_lfgi. Valid estimator aliases: {valid}"
    )


def selected_method_specs(methods: str) -> List[Tuple[str, str, str]]:
    """Resolve comma-separated method/hybrid aliases.

    Returns tuples ``(display_family, transport_method, correction_method)``.
    The token order for hybrids is always transport/correction.
    """
    raw = str(methods or "hybrids").strip().lower()
    if raw in {"all", "default", "*"}:
        keys = ["blend", "lfgi", "leaf-lfgi", "tweedie"]
    elif raw in {"hybrid", "hybrids", "blend-lfgi-hybrids", "lfgi-blend-hybrids"}:
        keys = ["blend_blend", "blend_lfgi", "lfgi_blend", "lfgi_lfgi"]
    elif raw in {"grid", "full-grid", "fullgrid", "full", "allpairs", "all-pairs"}:
        atoms = ["blend", "lfgi", "leaf-lfgi", "tweedie", "none"]
        keys = [f"{a}_{b}" for a in atoms for b in atoms]
    else:
        keys = [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for key in keys:
        fam, transport, correction = _parse_method_token(key)
        unique = (transport, correction)
        if unique not in seen:
            out.append((fam, transport, correction))
            seen.add(unique)
    if not out:
        raise ValueError("No methods selected. Example: --methods blend_lfgi,lfgi_blend,lfgi_none,none_lfgi,tweedie_lfgi")
    return out


def make_target(cfg: Config, device: torch.device, dtype: torch.dtype):
    key = str(cfg.target).strip().lower().replace("-", "_")
    if key in {"misaligned_gmm", "gmm", "gmm8", "misaligned8d", "current", "current8d"}:
        target = MisalignedSubspaceGMM(
            d=cfg.d,
            rank=cfg.rank,
            n_components=cfg.n_components,
            seed=cfg.target_seed,
            radius=cfg.radius,
            sigma_perp=cfg.sigma_perp,
            jitter=cfg.jitter,
            normalize=cfg.normalize_target,
            device=device,
            dtype=dtype,
        )
        target.name = "misaligned_gmm"
        target.target_info = lambda target=target: {
            "target_name": "misaligned_gmm",
            "target_type": "gmm",
            "target_dim": int(target.d),
            "gmm_rank": int(target.rank),
            "gmm_n_components": int(target.K),
        }
        return target

    if key in {"funnel", "funnel_d10", "neal_funnel", "neal-funnel", "dpsmc_funnel"}:
        return NealFunnelTarget(
            d=int(cfg.funnel_d),
            eta2=float(cfg.funnel_eta2),
            normalize=bool(cfg.normalize_target),
            score_bank_size=int(cfg.funnel_score_bank),
            score_chunk=int(cfg.funnel_score_chunk),
            device=device,
            dtype=dtype,
        )

    if key in {"lj13", "lj13_2d", "molecular", "molecular_lj", "mol_lj13"}:
        return MolecularLJTarget(
            n_particles=int(cfg.mol_n_particles),
            particle_dim=int(cfg.mol_particle_dim),
            seed=int(cfg.target_seed),
            beta=float(cfg.mol_beta),
            lj_eps=float(cfg.mol_lj_eps),
            lj_sigma=float(cfg.mol_lj_sigma),
            lj_soft_core=float(cfg.mol_lj_soft_core),
            bond_k=float(cfg.mol_bond_k),
            confinement_k=float(cfg.mol_confinement_k),
            com_k=float(cfg.mol_com_k),
            init_noise=float(cfg.mol_init_noise),
            sample_steps=int(cfg.mol_sample_steps),
            sample_step_size=float(cfg.mol_sample_step_size),
            sample_batch=int(cfg.mol_sample_batch),
            normalize=bool(cfg.normalize_target),
            norm_samples=int(cfg.mol_norm_samples),
            norm_eig_floor=float(cfg.mol_norm_eig_floor),
            score_bank_size=int(cfg.mol_score_bank),
            score_chunk=int(cfg.mol_score_chunk),
            hessian_chunk=int(cfg.mol_hessian_chunk),
            device=device,
            dtype=dtype,
            name="lj13_2d",
        )

    if key in {"dw4", "dw4_16d", "molecular_dw4", "mol_dw4"}:
        # A smaller intermediate-dimensional variant: 8 particles x 2 coordinates.
        # This is not the original low-dimensional DW4 toy; it keeps the DW/molecular
        # flavor while staying in the requested 16--32d regime.
        n_particles = int(cfg.mol_n_particles)
        if n_particles == int(Config().mol_n_particles):
            n_particles = 8
        return MolecularLJTarget(
            n_particles=n_particles,
            particle_dim=int(cfg.mol_particle_dim),
            seed=int(cfg.target_seed),
            beta=float(cfg.mol_beta),
            lj_eps=float(cfg.mol_lj_eps),
            lj_sigma=float(cfg.mol_lj_sigma),
            lj_soft_core=float(cfg.mol_lj_soft_core),
            bond_k=1.35 * float(cfg.mol_bond_k),
            confinement_k=float(cfg.mol_confinement_k),
            com_k=float(cfg.mol_com_k),
            init_noise=float(cfg.mol_init_noise),
            sample_steps=int(cfg.mol_sample_steps),
            sample_step_size=float(cfg.mol_sample_step_size),
            sample_batch=int(cfg.mol_sample_batch),
            normalize=bool(cfg.normalize_target),
            norm_samples=int(cfg.mol_norm_samples),
            norm_eig_floor=float(cfg.mol_norm_eig_floor),
            score_bank_size=int(cfg.mol_score_bank),
            score_chunk=int(cfg.mol_score_chunk),
            hessian_chunk=int(cfg.mol_hessian_chunk),
            device=device,
            dtype=dtype,
            name="dw4_16d",
        )

    raise ValueError("Unknown --target {!r}. Use misaligned_gmm, funnel_d10, lj13_2d, or dw4_16d.".format(cfg.target))

def run(cfg: Config) -> None:
    ensure_dir(cfg.outdir)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    dtype = get_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)
    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    target = make_target(cfg, device, dtype)
    cfg.d = int(target.d)
    config_dict = asdict(cfg)
    target_info = target.target_info() if hasattr(target, "target_info") else {"target_name": type(target).__name__, "target_dim": int(target.d)}
    config_dict.update({
        "actual_device": str(device),
        "effective_gate_n": int(effective_gate_n(cfg)),
        "effective_bank_coupling": canonical_bank_coupling(cfg.bank_coupling),
        "effective_initial_reference_mode": canonical_initial_reference_mode(cfg.initial_reference_mode),
        "effective_t_min": float(effective_time_bounds(cfg)[0]),
        "effective_t_max": float(effective_time_bounds(cfg)[1]),
        "effective_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "proposal_pool_n": int(proposal_pool_size(cfg)),
        "target_moment_mean_norm": target.moment_mean_norm,
        "target_moment_cov_frob_err": target.moment_cov_frob_err,
        "target_original_cov_eigs": [float(v) for v in getattr(target, "original_cov_eigs", [])],
        **target_info,
    })
    with open(os.path.join(cfg.outdir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Target {getattr(target, 'name', type(target).__name__)} normalized moment check:", flush=True)
    print(f"  ||E[X]||={target.moment_mean_norm:.3e}, ||Cov[X]-I||_F={target.moment_cov_frob_err:.3e}", flush=True)
    cfg.bank_coupling = canonical_bank_coupling(cfg.bank_coupling)
    init_pool_n = proposal_pool_size(cfg)
    print(
        f"  device={device}, dtype={dtype}, n_ref={cfg.n_ref}, gate_n={effective_gate_n(cfg)}, "
        f"bank_coupling={cfg.bank_coupling}, pool_n={init_pool_n}, n_samples={cfg.n_samples}, n_rounds={cfg.n_rounds}, "
        f"t_min={effective_time_bounds(cfg)[0]:.6g}, t_max={effective_time_bounds(cfg)[1]:.6g}, schedule={canonical_time_schedule(cfg.time_schedule)}",
        flush=True,
    )

    init_gen = make_generator(int(cfg.seed + 101), device)
    init_refs, init_ref_info = make_initial_reference_pool(target, cfg, init_pool_n, init_gen)
    init_score_refs, init_score_rho0, init_gate_refs, init_gate_rho0, init_split_info = split_score_gate_banks(
        init_refs,
        torch.zeros((init_refs.shape[0],), device=device, dtype=dtype),
        cfg,
    )
    truth_gen = make_generator(int(cfg.seed + 202), device)
    truth = target.sample(int(cfg.n_truth), generator=truth_gen).detach()
    init_rho, init_info = initial_log_weights(target, init_refs, cfg)
    init_info.update(init_ref_info)
    print(
        f"Initial references: mode={init_info['initial_reference_mode']}; "
        f"weights={init_info['initial_weight_mode']}; ESS/N={init_info.get('initial_rho_ess_frac', float('nan')):.3f}",
        flush=True,
    )

    # Baseline rows: truth floor against another truth draw, and prior bank metrics.
    metric_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []
    baseline_gen = make_generator(int(cfg.seed + 303), device)
    truth2 = target.sample(min(int(cfg.n_truth), int(cfg.metrics_max_n)), generator=baseline_gen).detach()
    metric_rows.append({"kind": "baseline", "family": "TARGET_FLOOR", "method": "TARGET_FLOOR", "round": 0, **compute_metrics(target, truth2, truth, None, cfg, baseline_gen)})
    metric_rows.append({"kind": "baseline", "family": "INIT_REFS", "method": f"INIT_REFS_{init_info['initial_reference_mode']}", "round": 0, **compute_metrics(target, init_score_refs, truth, None, cfg, baseline_gen), **init_info, **init_split_info})

    all_samples: Dict[str, List[torch.Tensor]] = {}
    method_specs = selected_method_specs(cfg.methods)
    print("Selected method pairs: " + ", ".join([f"{fam} (S={tm}, R={cm})" for fam, tm, cm in method_specs]), flush=True)
    for family, transport_method, correction_method in method_specs:
        print(f"\n=== Running alternating DRC pair: {family} (S={transport_method}, R={correction_method}) ===", flush=True)
        samples_by_round, rows, stages = run_family(family, transport_method, correction_method, target, init_refs, init_rho, truth, cfg)
        all_samples[family] = samples_by_round
        metric_rows.extend(rows)
        stage_rows.extend(stages)

    write_csv(os.path.join(cfg.outdir, "metrics_by_round.csv"), metric_rows)
    write_csv(os.path.join(cfg.outdir, "stage_diagnostics.csv"), stage_rows)
    save_heatmaps(cfg.outdir, target, truth, init_score_refs, all_samples, cfg)
    save_metric_curves(cfg.outdir, metric_rows)

    print("\nDone. Wrote:", flush=True)
    for name in ["config.json", "metrics_by_round.csv", "stage_diagnostics.csv", "heatmaps_final.png", "heatmaps_by_round.png", "metric_curves.png", "projection_basis.npz"]:
        print("  " + os.path.join(cfg.outdir, name), flush=True)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Alternating DRC hybrid test: choose target, initial references, and transport/correction score estimators.")
    defaults = Config()
    for field_name, default_value in asdict(defaults).items():
        arg = "--" + field_name
        if isinstance(default_value, bool):
            group = p.add_mutually_exclusive_group(required=False)
            group.add_argument(arg, dest=field_name, action="store_true")
            group.add_argument("--no_" + field_name, dest=field_name, action="store_false")
            p.set_defaults(**{field_name: default_value})
        elif isinstance(default_value, int):
            p.add_argument(arg, type=int, default=default_value)
        elif isinstance(default_value, float):
            p.add_argument(arg, type=float, default=default_value)
        else:
            p.add_argument(arg, type=str, default=default_value)
    ns = p.parse_args()

    # Backward compatibility: old command lines used --t_start/--t_end.
    # The canonical flags are now --t_max/--t_min; promote changed legacy
    # aliases only when the new flag is still at its default.
    if float(ns.t_max) == float(defaults.t_max) and float(ns.t_start) != float(defaults.t_start):
        ns.t_max = float(ns.t_start)
    if float(ns.t_min) == float(defaults.t_min) and float(ns.t_end) != float(defaults.t_end):
        ns.t_min = float(ns.t_end)
    ns.t_start = float(ns.t_max)
    ns.t_end = float(ns.t_min)
    ns.time_schedule = canonical_time_schedule(ns.time_schedule)

    cfg = Config(**vars(ns))
    # Fail fast for invalid interval/schedule.
    effective_time_bounds(cfg)
    return cfg


if __name__ == "__main__":
    run(parse_args())
