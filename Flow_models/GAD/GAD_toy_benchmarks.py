#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iterative_transport_bflow_ratio_fisher_v7.py

Refreshed alternating transport and ratio-flow test bench with reverse-SDE transport.

The transport block uses the same reverse OU Heun SDE as iterative_lfgi.py.
Likelihood factors at the transport endpoint are reconstructed in the same way
as iterative_lfgi.py: build the selected ratio-score estimator on the endpoint
(or legacy generator bank) and integrate its frozen probability-flow density.
The newer ratio-node, diagnostics, plotting, and refreshed-bank machinery are
otherwise retained.

Supported targets include the normalized stiff 8D misaligned GMM, a dedicated
rank-3-in-16D singular GMM importance-weight stress test, the fixed stiff 3D
GMM, Neal's funnel, stiff analytic 2D targets, and molecular LJ/DW examples.
Supported gate families include local scalar/matrix Blend, uniform
scalar/matrix Blend, LFGI variants, Tweedie (G=0), and TSI (G=I).

Method syntax
-------------

    <transport>-<repeats>_<ratio-carrier>_<ratio-node>-<ratio-rounds>

The two moved-particle ratio nodes are retained side by side:

* ``gated-pflow`` (legacy):
  ``s_R=s_A+lambda(I-G_A)(b_t^pi-s_A)``.
* ``gated-bflow`` (shared-statistic complement):
  ``s_R=s_A+lambda(I-G_A)(b_t^pi-b_t^q)``.
* ``raw-bflow`` (ungated shared-statistic b-residual control):
  ``s_R=s_A+lambda(b_t^pi-b_t^q)``.
* ``completed-bflow`` (finite-bank shared-LFGI completion):
  ``s_R=s_A+lambda[(I-G_A)(b_t^pi-b_t^q)+G_A(c_t^pi-c_t^q)]``.
  For a gate-consistent carrier and ``lambda=1`` this is exactly
  ``b_t^pi+G_A(c_t^pi-b_t^pi)`` at the particle-estimator level.

For example, strict refreshed LFGI b-flow alternation is
``lfgi_lfgi_gated-bflow-1``.  The raw-flow control is
``lfgi_lfgi_raw-bflow-1``.  The completed shared-LFGI control is
``lfgi_lfgi_completed-bflow-1``.  With Tweedie as the ratio estimator, all
gated flow definitions reduce to the full tilted Tweedie/Doob score because
``s_A=b_t^q`` and ``G_A=0``.

Algorithmic invariants
----------------------

* Transport substep m builds its frozen field from the actual output particles
  of substep m-1.
* Ratio substep r rebuilds its carrier, gate, density tilt, and conditional
  particle bank from the actual output particles of ratio substep r-1.
* The density denominator at q_k is the incoming certificate produced by the
  field built from q_{k-1}; rebuilding an outgoing q_k field is not substituted
  away from TSC.
* Certified generation and density evaluation use the same deterministic
  probability-flow path and divergence convention.
* The ratio flow returns moved unweighted particles and its own incoming
  certificate for the next half-step.
* FisherRMSE on each full-cycle sample row is evaluated from a score bank
  rebuilt on the actual post-ratio law that is fed into the next transport
  half-step; it is not inherited from the pre-ratio transport bank.

Outputs
-------
``config.json``, ``metrics_by_round.csv``, ``stage_diagnostics.csv``,
``convergence_by_round.csv``, and plotting dashboards in ``outdir``.

Strict alternating example
--------------------------

python iterative_transport_bflow_ratio_fisher_v7.py \
  --target misaligned_gmm \
  --outdir results/certified_lfgi_bflow \
  --device cuda --dtype float64 \
  --n_ref 3000 --n_samples 3000 --n_truth 12000 \
  --n_rounds 6 --n_steps 150 --pf_steps 64 \
  --t_min 0.005 --t_max 3.0 \
  --methods lfgi_lfgi_gated-bflow-1 \
  --ratio_reference_mode endpoint \
  --ratio_tilt_policy exact --lambda_guard 1

Transport-only theorem diagnostics remain available with
``--force_no_likelihood_correction`` and a ``*_none`` method token.
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


# Funnel plotting defaults copied from the benchmark sweep convention.  These
# robust quantile limits keep the long right tail from collapsing the visible
# neck/body geometry into a tiny corner of the panel.
FUNNEL_HEATMAP_X_Q_LOW = float(os.environ.get("LFGI_BENCH_FUNNEL_HEATMAP_X_Q_LOW", "0.5"))
FUNNEL_HEATMAP_X_Q_HIGH = float(os.environ.get("LFGI_BENCH_FUNNEL_HEATMAP_X_Q_HIGH", "99.0"))
FUNNEL_HEATMAP_Y_Q_ABS = float(os.environ.get("LFGI_BENCH_FUNNEL_HEATMAP_Y_Q_ABS", "99.0"))


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Config:
    outdir: str = "results/refreshed_certified_alternation_v4"
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float64"

    # Target selector.  gmm_16 is the dedicated rank-3-in-16D singular-GMM
    # ratio-degeneracy stress test.  In addition to the original GMM/funnel/
    # molecular targets, the script includes stiff two-dimensional toys:
    # banana, sine, ring, rings, spiral, and double_well.
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

    # Stiff analytic 2D target controls.  These targets are affinely normalized
    # to mean zero and covariance identity when normalize_target=True, while
    # retaining strong local anisotropy through thin curved/radial directions.
    toy_norm_samples: int = 32768
    toy_norm_eig_floor: float = 1.0e-8
    toy_score_bank: int = 8192
    toy_score_chunk: int = 512
    toy_hessian_chunk: int = 1024
    banana_bend: float = 0.50
    banana_normal_std: float = 0.12
    sine_amplitude: float = 0.70
    sine_frequency: float = 1.35
    sine_normal_std: float = 0.10
    ring_radius: float = 3.0
    ring_radial_std: float = 0.12
    rings_inner_radius: float = 2.0
    rings_outer_radius: float = 4.0
    rings_radial_std: float = 0.11
    double_well_barrier: float = 4.0
    double_well_bend: float = 0.20
    double_well_normal_std: float = 0.12
    spiral_turns: float = 2.25
    spiral_r_min: float = 1.0
    spiral_r_max: float = 4.5
    spiral_u_std: float = 0.85
    spiral_logradial_std: float = 0.055

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
    # Benchmark-sweep-compatible sample-quality metrics.  The sliced KS default
    # matches benchmark_sweep.py's LFGI_BENCH_KS_PROJ fallback, while nll_kde_n_fit
    # matches benchmark_sweep.py's nll_kde(samples, truth, n_fit=min(5000, n)).
    ks_projections: int = int(os.environ.get("LFGI_BENCH_KS_PROJ", "1000"))
    nll_kde_n_fit: int = 5000
    nll_kde_min_bandwidth: float = 0.05

    # Number of strict alternating outer cycles.  The production default is
    # one refreshed transport step followed by one refreshed ratio-flow step.
    # No-correction transport-only diagnostics remain available by flag.
    n_rounds: int = 6
    # Initial proposal/reference law:
    #   prior/gaussian : draw the initial split-compatible pool from N(0,I).
    #   target/oracle  : draw the initial split-compatible pool from the target.
    #                    This is an oracle-reference stability test; initial
    #                    density-ratio weights are forced to zero because q0=p0.
    initial_reference_mode: str = "prior"
    initial_weight_mode: str = "zero"  # prior_ratio or zero; ignored for target initial references
    # Comma-separated method specifications.  The canonical syntax is
    #
    #   <transport score>-<transport repeats>_<ratio/PF score>_<ratio mode>-<ratio rounds>
    #
    # e.g. ``lfgi-2_lfgi_gated-bflow-1``.  ``raw-w`` keeps the legacy
    # fixed-coordinate importance-weight correction and ignores the optional
    # ratio-round count.  ``gated-pflow`` retains the previous residual
    # s_A+(I-G_A)(b^pi-s_A), while ``gated-bflow`` implements the shared-statistic
    # complement s_A+(I-G_A)(b^pi-b^q).  ``raw-bflow`` uses the identical b-flow
    # carrier, tilt, particles, and integration but replaces I-G_A by I.
    # ``completed-bflow`` restores the finite-bank cancellation term on the same
    # carrier: s_A+(I-G_A)(b^pi-b^q)+G_A(c^pi-c^q).  The
    # first score method generates the
    # transport endpoint by reverse SDE.  The second score method is used to
    # reconstruct the endpoint likelihood factors and supplies the refreshed ratio carrier and native
    # gate.  The b^q and b^pi vectors are the canonical untilted and density-tilted
    # Tweedie conditional statistics on that same current score bank; they do not
    # introduce a separate estimator choice.  Each inner ratio round refreshes all
    # current-law objects from the moved particles and carries forward the incoming
    # certificate emitted by the preceding ratio field.  Legacy two-field labels
    # remain accepted and default to raw-w.
    # Special values:
    #   all/default = diagonal blend, matrix_blend, unif_blend, unif_matrix_blend, lfgi, os-lfgi, pi-lfgi, lfgi-N, leaf-lfgi, tweedie
    #   hybrids     = four blend/lfgi pairs
    #   grid/full   = full transport/correction grid over blend, matrix_blend, unif_blend, unif_matrix_blend, lfgi, os-lfgi, pi-lfgi, lfgi-N, leaf-lfgi, tweedie, none
    methods: str = "lfgi_lfgi_gated-pflow-1"
    # Frozen bank used for the post-transport log-q reconstruction, matching
    # iterative_lfgi.py.  endpoint (default) rebuilds the ratio estimator on the
    # final endpoint cloud; generator uses the bank that produced that endpoint.
    ratio_reference_mode: str = "endpoint"
    # Retained for CLI compatibility.  This variant always uses reverse SDE for
    # transport; probability-flow transport is rejected.
    transport_dynamics: str = "reverse_sde"  # retained for CLI compatibility; transport is always reverse SDE
    # Native uses the literal estimator gate and literal I-G.  Project first
    # projects to 0<=G<=I and rebuilds a gate-consistent carrier.
    ratio_gate_policy: str = "native"  # native or project
    # Ratio endpoint labels: exact uses log pi-log q without clipping, tempering,
    # or ESS adaptation. stabilized preserves the legacy guarded-weight policy.
    ratio_tilt_policy: str = "exact"  # exact or stabilized
    # The outgoing endpoint-field density is a legacy TSC-only diagnostic and
    # is rejected during correction unless this escape hatch is explicit.
    allow_legacy_endpoint_certificate: bool = False
    # A certified law must not be silently altered by final denoising or clipping.
    certified_flow_strict: bool = True
    # Explicit abort thresholds for the certified generating flow.  These are
    # intentionally separate from the legacy score/sample/divergence clipping
    # knobs: certified PF never clips the state, score, or divergence because
    # doing so would invalidate the incoming Liouville density certificate.
    # A nonpositive value disables the corresponding optional audit threshold;
    # nonfinite values are always fatal.
    certified_score_abort: float = 0.0
    certified_state_abort: float = 0.0
    certified_div_abort: float = 0.0
    # Raw indefinite LFGI followed by a hard post-resolvent spectral gate floor
    # can be discontinuous when the resolvent denominator crosses zero.  Such a
    # field does not have an ordinary pointwise divergence suitable for the
    # certified PF density path.  The default rejects that configuration before
    # a long run.  This escape hatch exists only for reproducing legacy behavior.
    allow_raw_lfgi_hard_projection_certified: bool = False
    # Force the R/projection step to be the identity: all next-round references
    # are unweighted endpoint particles.  This is the convergence check for the
    # score-to-transport map, not a test of likelihood-ratio correction.
    force_no_likelihood_correction: bool = False

    # Adjacent-field convergence diagnostics for Theorem 5.3.  delta_pf_* uses
    # deterministic probability-flow paths and estimates
    #   int ||s_k(y,t)-s_{k+1}(y,t)||^2 d( q_{s_k,t}^{PF}+q_{s_{k+1},t}^{PF})/2 dt.
    convergence_check: bool = True
    delta_pf_n: int = 256
    delta_pf_steps: int = 24
    # Additional fixed-law diagnostic requested for theorem debugging:
    #   delta_pf_target^2 = int E_{x_t~pi_t} ||s_k(x_t,t)-s_{k+1}(x_t,t)||^2 dt.
    # Use <=0 to default to delta_pf_n.
    delta_pf_target_n: int = 0
    adjacent_metrics_max_n: int = 2000

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
    # Optional eigenvalue floor for symmetric gate matrices.  The default -inf
    # leaves gates unchanged.  Setting 0 projects to PSD; setting 1e-2, for
    # example, projects to eigenvalues >= 0.01.
    gate_min_eval: float = -float("inf")

    # One-step residual-corrected q-LFGI.  tau=1 applies the full matrix-free
    # first-order normal-equation correction from the paired particle-wise
    # TSI/Tweedie signals; tau=0 reduces exactly to ordinary q-LFGI.
    os_lfgi_tau: float = 1.0

    # Matrix/uniform blend controls, matching the sampler-mode defaults used in
    # sampling.py.  matrix_blend is the centered local matrix regression gate;
    # unif_blend and unif_matrix_blend use spatially homogeneous scalar/matrix
    # gates estimated from the global target-score second moment.
    matrix_blend_center: bool = True
    matrix_blend_ridge: float = 1.0e-8
    matrix_blend_ridge_rel: float = 1.0e-6
    matrix_blend_sym_gate: bool = False
    matrix_blend_gate_clip: float = 1.0e6
    uniform_blend_clamp: bool = True

    # Minimal MP-leaf precision completion for Leaf-LFGI.  The completed
    # gate precision is Q = V diag(max(lambda, mp_leaf_floor)) V^T.
    # This is the fixed-floor/moment-preserving correction: score signals are
    # unchanged, but the CE/LFGI gate sees the PSD-completed precision.
    mp_leaf_floor: float = 0.0
    mp_leaf_tol: float = 1.0e-12
    weight_temp: float = 1.0
    # Correction strength for every moved-particle ratio node.
    # gated-pflow: s_R = s_A + lambda_guard (I-G_A)(b_pi-s_A).
    # gated-bflow: s_R = s_A + lambda_guard (I-G_A)(b_pi-b_q).
    # raw-bflow:       s_R = s_A + lambda_guard (b_pi-b_q).
    # completed-bflow: s_R = s_A + lambda_guard * [
    #                       (I-G_A)(b_pi-b_q) + G_A(c_pi-c_q)].
    # The name is retained to match the requested command-line interface.
    lambda_guard: float = 1.0
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
    likelihood_calibration: bool = False
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

def time_grid_step_stats(ts: torch.Tensor) -> Dict[str, float]:
    """Diagnostics for auditing nonuniform time integration.

    The integrators below step in the physical OU time variable t.  A log-linear
    schedule therefore only changes the locations of the t-grid points; each
    quadrature/drift update must still be multiplied by the actual interval
    h = t_{j+1} - t_j (or its positive reverse-time counterpart).  If we ever
    reparameterize the ODE itself by u = log t, the vector field/integrand would
    need an additional dt/du = t factor.  This code intentionally does not do
    that; it performs ordinary trapezoidal integration on a nonuniform t-grid.
    """
    if ts.numel() < 2:
        return {"dt_min": 0.0, "dt_max": 0.0, "dt_sum": 0.0}
    dt = torch.abs(ts[1:] - ts[:-1])
    return {
        "dt_min": safe_float(dt.min()),
        "dt_max": safe_float(dt.max()),
        "dt_sum": safe_float(dt.sum()),
    }


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
# Exact normalized stiff, heterogeneous, misaligned 3D GMM used by the
# focused gate-comparison experiment.
# -----------------------------------------------------------------------------


class StiffMisalignedGMM3D:
    """The exact six-component 3D GMM from lfgi_d3_gate_comparison_normalized.py.

    The raw mixture parameters and analytic global whitening are intentionally
    fixed so this target can be used to cross-reference the focused 3D harness
    without changing any other behavior in this script.
    """

    def __init__(
        self,
        normalize: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.device = device
        self.dtype = dtype
        self.d = 3
        self.K = 6
        self.normalized = bool(normalize)
        self.name = "stiff_misaligned_gmm3d"

        w = torch.tensor([0.07, 0.13, 0.18, 0.21, 0.17, 0.24], device=device, dtype=dtype)
        self.weights = w / w.sum()
        self.log_weights = torch.log(self.weights)

        radius = 2.7
        raw_means = torch.tensor([
            [-1.10, -0.55,  0.25],
            [-0.62,  0.92, -0.52],
            [ 0.10, -1.02,  0.78],
            [ 0.78,  0.62,  0.55],
            [ 1.08, -0.28, -0.62],
            [-0.18,  0.18, -1.08],
        ], device=device, dtype=dtype)
        raw_means = radius * raw_means / torch.linalg.norm(raw_means, dim=1, keepdim=True)

        stiff_std = 0.075
        mid_std = 0.24
        soft_std = 0.72
        base_stds = torch.tensor([
            [stiff_std,        1.05 * mid_std, 0.95 * soft_std],
            [1.15 * stiff_std, 0.88 * mid_std, 1.10 * soft_std],
            [0.90 * stiff_std, 1.18 * mid_std, 0.84 * soft_std],
            [1.05 * stiff_std, 0.95 * mid_std, 1.20 * soft_std],
            [0.82 * stiff_std, 1.10 * mid_std, 0.92 * soft_std],
            [1.22 * stiff_std, 0.82 * mid_std, 1.05 * soft_std],
        ], device=device, dtype=dtype)
        angles = torch.tensor([
            [ 0.15,  0.52, -0.35],
            [ 0.72, -0.28,  0.61],
            [-0.44,  0.83,  0.24],
            [ 0.91,  0.38, -0.76],
            [-0.68, -0.57,  0.88],
            [ 0.37, -0.92, -0.21],
        ], device=device, dtype=dtype)
        Q = self._rotation_matrix_xyz(angles)
        raw_covs = Q @ torch.diag_embed(base_stds.square()) @ Q.transpose(-1, -2)
        raw_covs = raw_covs + 0.015 ** 2 * torch.eye(3, device=device, dtype=dtype)

        raw_global_mean = torch.einsum("k,kd->d", self.weights, raw_means)
        raw_centered_means = raw_means - raw_global_mean
        raw_global_cov = torch.einsum("k,kij->ij", self.weights, raw_covs)
        raw_global_cov = raw_global_cov + torch.einsum(
            "k,ki,kj->ij", self.weights, raw_centered_means, raw_centered_means
        )
        raw_evals, raw_evecs = torch.linalg.eigh(sym(raw_global_cov))
        whitening = raw_evecs @ torch.diag(torch.clamp(raw_evals, min=1.0e-12).rsqrt()) @ raw_evecs.T

        if self.normalized:
            self.means = raw_centered_means @ whitening.T
            self.covs = sym(whitening[None, :, :] @ raw_covs @ whitening.T[None, :, :])
        else:
            self.means = raw_means
            self.covs = raw_covs

        self.precisions = torch.linalg.inv(self.covs)
        self.logdets = torch.linalg.slogdet(self.covs).logabsdet
        self.chols = torch.linalg.cholesky(self.covs)

        global_mean = torch.einsum("k,kd->d", self.weights, self.means)
        centered = self.means - global_mean
        global_cov = torch.einsum("k,kij->ij", self.weights, self.covs)
        global_cov = global_cov + torch.einsum("k,ki,kj->ij", self.weights, centered, centered)
        eye = torch.eye(3, device=device, dtype=dtype)
        self.global_cov = global_cov.detach()
        self.moment_mean_norm = safe_float(torch.linalg.norm(global_mean))
        self.moment_cov_frob_err = safe_float(torch.linalg.matrix_norm(global_cov - eye, ord="fro"))
        self.original_cov_eigs = raw_evals.detach().cpu().numpy()
        self._raw_global_mean = raw_global_mean.detach()
        self._raw_global_cov = raw_global_cov.detach()
        self._whitening = whitening.detach()

        if self.normalized:
            tol = max(1.0e-9, 100.0 * torch.finfo(dtype).eps)
            if self.moment_mean_norm > tol or self.moment_cov_frob_err > tol:
                raise RuntimeError(
                    "Analytic target normalization failed: "
                    f"||E[X]||={self.moment_mean_norm:.3e}, "
                    f"||Cov[X]-I||_F={self.moment_cov_frob_err:.3e}, tol={tol:.3e}"
                )

    @staticmethod
    def _rotation_matrix_xyz(angles: torch.Tensor) -> torch.Tensor:
        ax, ay, az = angles.unbind(dim=-1)
        one = torch.ones_like(ax)
        zero = torch.zeros_like(ax)
        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)
        Rx = torch.stack([
            one, zero, zero,
            zero, cx, -sx,
            zero, sx, cx,
        ], dim=-1).reshape(*angles.shape[:-1], 3, 3)
        Ry = torch.stack([
            cy, zero, sy,
            zero, one, zero,
            -sy, zero, cy,
        ], dim=-1).reshape(*angles.shape[:-1], 3, 3)
        Rz = torch.stack([
            cz, -sz, zero,
            sz, cz, zero,
            zero, zero, one,
        ], dim=-1).reshape(*angles.shape[:-1], 3, 3)
        return Rz @ Ry @ Rx

    def marginal_params(self, t: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tt = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        alpha, gamma = alpha_gamma(tt)
        means_t = alpha * self.means
        eye = torch.eye(3, device=self.device, dtype=self.dtype)[None, :, :]
        covs_t = alpha * alpha * self.covs + gamma * eye
        precs_t = torch.linalg.inv(covs_t)
        logdets_t = torch.linalg.slogdet(covs_t).logabsdet
        return means_t, covs_t, precs_t, logdets_t

    def component_log_probs(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        means_t, _covs_t, precs_t, logdets_t = self.marginal_params(t)
        diff = x[:, None, :] - means_t[None, :, :]
        quad = torch.einsum("nki,kij,nkj->nk", diff, precs_t, diff)
        return self.log_weights[None, :] - 0.5 * (
            3.0 * math.log(2.0 * math.pi) + logdets_t[None, :] + quad
        )

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
        comp_scores = -torch.einsum("kij,nkj->nki", precs_t, diff)
        return torch.sum(r[:, :, None] * comp_scores, dim=1)

    def observed_information(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        means_t, _covs_t, precs_t, _logdets_t = self.marginal_params(t)
        r = self.responsibilities(x, t=t)
        diff = x[:, None, :] - means_t[None, :, :]
        comp_scores = -torch.einsum("kij,nkj->nki", precs_t, diff)
        score = torch.sum(r[:, :, None] * comp_scores, dim=1)
        mean_precision = torch.einsum("nk,kij->nij", r, precs_t)
        second = torch.einsum("nk,nki,nkj->nij", r, comp_scores, comp_scores)
        cov_scores = second - score[:, :, None] * score[:, None, :]
        return sym(mean_precision - cov_scores)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        ids = torch.multinomial(self.weights, int(n), replacement=True, generator=generator)
        z = torch.randn((int(n), 3), device=self.device, dtype=self.dtype, generator=generator)
        return self.means[ids] + torch.einsum("nij,nj->ni", self.chols[ids], z)

    def sample_pt(self, n: int, t: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x0 = self.sample(int(n), generator=generator)
        alpha, gamma = alpha_gamma(torch.tensor(float(t), device=self.device, dtype=self.dtype))
        eps = torch.randn(x0.shape, device=self.device, dtype=self.dtype, generator=generator)
        return alpha * x0 + torch.sqrt(torch.clamp(gamma, min=0.0)) * eps

    def target_info(self) -> Dict[str, object]:
        local_eigs = torch.linalg.eigvalsh(self.covs)
        return {
            "target_name": self.name,
            "target_type": "gmm",
            "target_dim": 3,
            "gmm_n_components": 6,
            "gmm_weights": [float(v) for v in self.weights.detach().cpu()],
            "target_normalized": bool(self.normalized),
            "normalization_definition": "E_pi[X]=0 and Cov_pi[X]=I",
            "raw_global_mean": self._raw_global_mean.cpu().tolist(),
            "raw_global_covariance": self._raw_global_cov.cpu().tolist(),
            "whitening_matrix": self._whitening.cpu().tolist(),
            "normalized_means": self.means.detach().cpu().tolist(),
            "normalized_covariance_eigenvalues": local_eigs.detach().cpu().tolist(),
        }

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

    def plot_projection(self, x: torch.Tensor, fit_ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return benchmark-sweep funnel display coordinates.

        The benchmark plots Neal funnels in native (x_1, x_2) coordinates, not
        in a PCA projection.  This matters when normalize_target=True: the
        sampler operates in whitened coordinates for fair N(0,I) initialization,
        but the plot should be mapped back to the native funnel variables so the
        visual convention matches benchmark_sweep.py.
        """
        return self._to_native(x)[:, :2]

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
# Stiff analytic two-dimensional non-Gaussian targets
# -----------------------------------------------------------------------------


class AnalyticToy2DTarget:
    """Base class for exact-sampling 2D targets with autodiff score/Hessian.

    Subclasses define a normalized native-coordinate density and an exact native
    sampler.  The public coordinates optionally apply an affine whitening
    transform.  This removes trivial global scale mismatch while retaining the
    local ill-conditioning generated by thin curved tubes, shells, and wells.
    """

    def __init__(
        self,
        *,
        name: str,
        seed: int,
        normalize: bool,
        norm_samples: int,
        norm_eig_floor: float,
        score_bank_size: int,
        score_chunk: int,
        hessian_chunk: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.name = str(name)
        self.seed = int(seed)
        self.d = 2
        self.D = 2
        self.K = 0
        self.weights = torch.empty((0,), device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.normalized = bool(normalize)
        self.norm_samples = int(norm_samples)
        self.norm_eig_floor = float(norm_eig_floor)
        self.score_bank_size = int(score_bank_size)
        self.score_chunk = int(score_chunk)
        self.hessian_chunk = int(hessian_chunk)
        self._ou_score_bank: Optional[torch.Tensor] = None

        self.norm_mean = torch.zeros((2,), device=device, dtype=dtype)
        self.norm_L = torch.eye(2, device=device, dtype=dtype)
        self.norm_W = torch.eye(2, device=device, dtype=dtype)
        self.norm_logabsdet_L = torch.tensor(0.0, device=device, dtype=dtype)

        pilot_gen = make_generator(self.seed + 41_003, self.device)
        pilot_n = max(4096, int(self.norm_samples))
        pilot_native = self._sample_native(pilot_n, pilot_gen).detach()
        native_mean = pilot_native.mean(dim=0)
        Xn = pilot_native - native_mean
        native_cov = sym((Xn.T @ Xn) / max(pilot_n - 1, 1))
        native_evals, native_evecs = torch.linalg.eigh(native_cov)
        self.original_cov_eigs = as_numpy(native_evals)

        if self.normalized:
            evals = torch.clamp(native_evals, min=self.norm_eig_floor)
            sqrt_e = torch.sqrt(evals)
            inv_sqrt_e = 1.0 / sqrt_e
            self.norm_mean = native_mean.detach()
            self.norm_L = (native_evecs @ torch.diag(sqrt_e) @ native_evecs.T).detach()
            self.norm_W = (native_evecs @ torch.diag(inv_sqrt_e) @ native_evecs.T).detach()
            self.norm_logabsdet_L = torch.sum(torch.log(sqrt_e)).detach()
            pilot = self._from_native(pilot_native)
        else:
            pilot = pilot_native

        with torch.no_grad():
            m = pilot.mean(dim=0)
            X = pilot - m
            C = sym((X.T @ X) / max(int(pilot.shape[0]) - 1, 1))
            eye = torch.eye(2, device=device, dtype=dtype)
            self.moment_mean_norm = safe_float(torch.linalg.norm(m))
            self.moment_cov_frob_err = safe_float(torch.linalg.matrix_norm(C - eye, ord="fro"))
            self.global_cov = C.detach()

    def _sample_native(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        raise NotImplementedError

    def _native_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _extra_target_info(self) -> Dict[str, object]:
        return {}

    def _to_native(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_mean[None, :] + x @ self.norm_L.T

    def _from_native(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.norm_mean[None, :]) @ self.norm_W.T

    def log_prob(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            return self._empirical_ou_log_prob(x, t_float)
        z = self._to_native(x)
        return self._native_log_prob(z) + self.norm_logabsdet_L.to(device=x.device, dtype=x.dtype)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x, t=0.0)

    def score(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            return self._empirical_ou_score(x, t_float)
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            lp = self.log_prob(x_req, t=0.0).sum()
            grad = torch.autograd.grad(lp, x_req, create_graph=False, retain_graph=False)[0]
        return torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    def observed_information(self, x: torch.Tensor, t: float | torch.Tensor = 0.0) -> torch.Tensor:
        t_float = float(torch.as_tensor(t).detach().cpu().item())
        if abs(t_float) > 0.0:
            raise NotImplementedError(f"{self.name} only provides exact observed information at t=0")
        outs: List[torch.Tensor] = []
        chunk = max(1, int(self.hessian_chunk))
        with torch.enable_grad():
            try:
                from torch.func import hessian, vmap

                def energy_single(x1: torch.Tensor) -> torch.Tensor:
                    return self.energy(x1.unsqueeze(0))[0]

                hess_fn = vmap(hessian(energy_single))
                for start in range(0, x.shape[0], chunk):
                    outs.append(hess_fn(x[start:start + chunk].detach()).detach())
            except Exception:
                for start in range(0, x.shape[0], chunk):
                    local: List[torch.Tensor] = []
                    for xi0 in x[start:start + chunk]:
                        xi = xi0.detach().clone().requires_grad_(True)
                        H = torch.autograd.functional.hessian(lambda zz: self.energy(zz.unsqueeze(0))[0], xi)
                        local.append(H.detach())
                    outs.append(torch.stack(local, dim=0))
        return torch.nan_to_num(sym(torch.cat(outs, dim=0)), nan=0.0, posinf=1.0e12, neginf=-1.0e12)

    @torch.no_grad()
    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self._from_native(self._sample_native(int(n), generator)).detach()

    @torch.no_grad()
    def sample_pt(self, n: int, t: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x0 = self.sample(int(n), generator=generator)
        alpha, gamma = alpha_gamma(float(t), device=self.device, dtype=self.dtype)
        eps = torch.randn(x0.shape, device=self.device, dtype=self.dtype, generator=generator)
        return alpha * x0 + torch.sqrt(torch.clamp(gamma, min=0.0)) * eps

    def _ensure_ou_score_bank(self) -> torch.Tensor:
        need = max(8, int(self.score_bank_size))
        if self._ou_score_bank is None or int(self._ou_score_bank.shape[0]) < need:
            self._ou_score_bank = self.sample(need, generator=make_generator(self.seed + 71_003, self.device)).detach()
        return self._ou_score_bank

    @torch.no_grad()
    def _empirical_ou_score(self, y: torch.Tensor, t: float) -> torch.Tensor:
        bank = self._ensure_ou_score_bank()
        alpha, gamma = alpha_gamma(float(t), device=self.device, dtype=self.dtype)
        gamma = torch.clamp(gamma, min=torch.as_tensor(1.0e-8, device=self.device, dtype=self.dtype))
        outs: List[torch.Tensor] = []
        for start in range(0, y.shape[0], max(1, int(self.score_chunk))):
            yy = y[start:start + max(1, int(self.score_chunk))]
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
        outs: List[torch.Tensor] = []
        const = -math.log(2.0 * math.pi) - torch.log(gamma)
        for start in range(0, y.shape[0], max(1, int(self.score_chunk))):
            yy = y[start:start + max(1, int(self.score_chunk))]
            diff = yy[:, None, :] - alpha * bank[None, :, :]
            logk = const - 0.5 * torch.sum(diff * diff, dim=-1) / gamma
            outs.append(torch.logsumexp(logk, dim=1) - math.log(int(bank.shape[0])))
        return torch.nan_to_num(torch.cat(outs, dim=0), nan=-1.0e6, posinf=1.0e6, neginf=-1.0e6)

    def target_info(self) -> Dict[str, object]:
        return {
            "target_name": self.name,
            "target_type": "analytic_toy_2d",
            "target_dim": 2,
            "toy_normalized": bool(self.normalized),
            "toy_score_t_mode": "empirical_ou_for_t_gt_0",
            **self._extra_target_info(),
        }


class BananaTarget2D(AnalyticToy2DTarget):
    def __init__(self, *, bend: float = 0.50, normal_std: float = 0.12, **kwargs):
        self.bend = float(bend)
        self.normal_std = float(normal_std)
        if self.normal_std <= 0.0:
            raise ValueError("banana_normal_std must be positive")
        super().__init__(name="banana", **kwargs)

    def _sample_native(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        u = torch.randn((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        v = self.normal_std * torch.randn((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        return torch.stack([u, self.bend * (u.square() - 1.0) + v], dim=1)

    def _native_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        u = z[:, 0]
        v = z[:, 1] - self.bend * (u.square() - 1.0)
        return -0.5 * u.square() - 0.5 * (v / self.normal_std).square() - math.log(2.0 * math.pi * self.normal_std)

    def _extra_target_info(self) -> Dict[str, object]:
        return {"banana_bend": self.bend, "banana_normal_std": self.normal_std}


class SineTarget2D(AnalyticToy2DTarget):
    def __init__(self, *, amplitude: float = 0.70, frequency: float = 1.35, normal_std: float = 0.10, **kwargs):
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.normal_std = float(normal_std)
        if self.normal_std <= 0.0:
            raise ValueError("sine_normal_std must be positive")
        super().__init__(name="sine", **kwargs)

    def _sample_native(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        u = torch.randn((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        v = self.normal_std * torch.randn((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        return torch.stack([u, self.amplitude * torch.sin(self.frequency * u) + v], dim=1)

    def _native_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        u = z[:, 0]
        v = z[:, 1] - self.amplitude * torch.sin(self.frequency * u)
        return -0.5 * u.square() - 0.5 * (v / self.normal_std).square() - math.log(2.0 * math.pi * self.normal_std)

    def _extra_target_info(self) -> Dict[str, object]:
        return {
            "sine_amplitude": self.amplitude,
            "sine_frequency": self.frequency,
            "sine_normal_std": self.normal_std,
        }


class RadialShellTarget2D(AnalyticToy2DTarget):
    def __init__(self, *, name: str, radii: Tuple[float, ...], radial_stds: Tuple[float, ...], **kwargs):
        if len(radii) < 1 or len(radii) != len(radial_stds):
            raise ValueError("radii and radial_stds must be nonempty and have equal length")
        self.radii_tuple = tuple(float(v) for v in radii)
        self.radial_stds_tuple = tuple(float(v) for v in radial_stds)
        if min(self.radii_tuple) <= 0.0 or min(self.radial_stds_tuple) <= 0.0:
            raise ValueError("ring radii and radial standard deviations must be positive")
        self.shell_radii = torch.tensor(self.radii_tuple, device=kwargs["device"], dtype=kwargs["dtype"])
        self.shell_stds = torch.tensor(self.radial_stds_tuple, device=kwargs["device"], dtype=kwargs["dtype"])
        # Approximately equalize integrated mass across shells: Cartesian shell
        # mass scales as radius times its radial width.
        amp = 1.0 / (self.shell_radii * self.shell_stds)
        self.shell_log_amplitudes = torch.log(amp / amp.sum())
        self._prepare_radial_quadrature()
        super().__init__(name=name, **kwargs)

    def _radial_log_unnormalized(self, r: torch.Tensor) -> torch.Tensor:
        q = (r[..., None] - self.shell_radii) / self.shell_stds
        return torch.logsumexp(self.shell_log_amplitudes - 0.5 * q.square(), dim=-1)

    def _prepare_radial_quadrature(self) -> None:
        r_max = max(self.radii_tuple) + 10.0 * max(self.radial_stds_tuple)
        grid = torch.linspace(0.0, r_max, 65536, device=self.shell_radii.device, dtype=self.shell_radii.dtype)
        logf = self._radial_log_unnormalized(grid)
        density = grid * torch.exp(logf)
        dr = grid[1] - grid[0]
        cdf = torch.zeros_like(grid)
        cdf[1:] = torch.cumsum(0.5 * (density[1:] + density[:-1]) * dr, dim=0)
        radial_integral = cdf[-1]
        cdf = cdf / torch.clamp(radial_integral, min=1.0e-300)
        self.radial_grid = grid
        self.radial_cdf = cdf
        self.native_log_normalizer = torch.log(2.0 * math.pi * torch.clamp(radial_integral, min=1.0e-300))

    def _sample_radius(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        u = torch.rand((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        idx = torch.searchsorted(self.radial_cdf, u, right=False).clamp(1, self.radial_grid.numel() - 1)
        c0, c1 = self.radial_cdf[idx - 1], self.radial_cdf[idx]
        r0, r1 = self.radial_grid[idx - 1], self.radial_grid[idx]
        frac = (u - c0) / torch.clamp(c1 - c0, min=1.0e-30)
        return r0 + frac * (r1 - r0)

    def _sample_native(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        r = self._sample_radius(int(n), generator)
        theta = 2.0 * math.pi * torch.rand((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

    def _native_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.sum(z.square(), dim=1) + 1.0e-18)
        return self._radial_log_unnormalized(r) - self.native_log_normalizer

    def _extra_target_info(self) -> Dict[str, object]:
        return {
            "ring_radii": list(self.radii_tuple),
            "ring_radial_stds": list(self.radial_stds_tuple),
            "ring_n_shells": len(self.radii_tuple),
        }


class DoubleWellTarget2D(AnalyticToy2DTarget):
    def __init__(self, *, barrier: float = 4.0, bend: float = 0.20, normal_std: float = 0.12, **kwargs):
        self.barrier = float(barrier)
        self.bend = float(bend)
        self.normal_std = float(normal_std)
        if self.barrier <= 0.0 or self.normal_std <= 0.0:
            raise ValueError("double-well barrier and normal_std must be positive")
        self._prepare_x_quadrature(kwargs["device"], kwargs["dtype"])
        super().__init__(name="double_well", **kwargs)

    def _x_log_unnormalized(self, x: torch.Tensor) -> torch.Tensor:
        return -self.barrier * (x.square() - 1.0).square()

    def _prepare_x_quadrature(self, device: torch.device, dtype: torch.dtype) -> None:
        grid = torch.linspace(-3.0, 3.0, 65536, device=device, dtype=dtype)
        density = torch.exp(self._x_log_unnormalized(grid))
        dx = grid[1] - grid[0]
        cdf = torch.zeros_like(grid)
        cdf[1:] = torch.cumsum(0.5 * (density[1:] + density[:-1]) * dx, dim=0)
        integral = cdf[-1]
        self.x_grid = grid
        self.x_cdf = cdf / torch.clamp(integral, min=1.0e-300)
        self.x_log_normalizer = torch.log(torch.clamp(integral, min=1.0e-300))

    def _sample_x(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        u = torch.rand((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        idx = torch.searchsorted(self.x_cdf, u, right=False).clamp(1, self.x_grid.numel() - 1)
        c0, c1 = self.x_cdf[idx - 1], self.x_cdf[idx]
        x0, x1 = self.x_grid[idx - 1], self.x_grid[idx]
        frac = (u - c0) / torch.clamp(c1 - c0, min=1.0e-30)
        return x0 + frac * (x1 - x0)

    def _sample_native(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        x = self._sample_x(int(n), generator)
        mean_y = self.bend * (x.square() - 1.0)
        y = mean_y + self.normal_std * torch.randn((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        return torch.stack([x, y], dim=1)

    def _native_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        x, y = z[:, 0], z[:, 1]
        resid = y - self.bend * (x.square() - 1.0)
        logpy = -0.5 * (resid / self.normal_std).square() - 0.5 * math.log(2.0 * math.pi) - math.log(self.normal_std)
        return self._x_log_unnormalized(x) - self.x_log_normalizer + logpy

    def _extra_target_info(self) -> Dict[str, object]:
        return {
            "double_well_barrier": self.barrier,
            "double_well_bend": self.bend,
            "double_well_normal_std": self.normal_std,
        }


class SpiralTarget2D(AnalyticToy2DTarget):
    """Finite multi-turn Archimedean spiral with exact latent change of variables.

    A bounded logistic-normal angular coordinate u controls the spiral radius,
    while a narrow Gaussian log-radial perturbation supplies the normal tube.
    The Cartesian density sums exactly over all wrapped angular preimages.
    """

    def __init__(
        self,
        *,
        turns: float = 2.25,
        r_min: float = 1.0,
        r_max: float = 4.5,
        u_std: float = 0.85,
        logradial_std: float = 0.055,
        **kwargs,
    ):
        self.turns = float(turns)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.u_std = float(u_std)
        self.logradial_std = float(logradial_std)
        if self.turns <= 0.5 or self.r_min <= 0.0 or self.r_max <= self.r_min:
            raise ValueError("spiral requires turns>0.5 and 0<r_min<r_max")
        if self.u_std <= 0.0 or self.logradial_std <= 0.0:
            raise ValueError("spiral latent standard deviations must be positive")
        self.u_lo = -math.pi * self.turns
        self.u_hi = math.pi * self.turns
        self.u_mid = 0.5 * (self.u_lo + self.u_hi)
        self.u_half = 0.5 * (self.u_hi - self.u_lo)
        self.pitch = (self.r_max - self.r_min) / (self.u_hi - self.u_lo)
        kmax = int(math.ceil(self.turns / 2.0)) + 3
        self.branch_ks = tuple(range(-kmax, kmax + 1))
        super().__init__(name="spiral", **kwargs)

    def _radius_curve(self, u: torch.Tensor) -> torch.Tensor:
        return self.r_min + self.pitch * (u - self.u_lo)

    def _sample_native(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        z_u = self.u_std * torch.randn((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        u = self.u_mid + self.u_half * torch.tanh(z_u)
        v = self.logradial_std * torch.randn((int(n),), device=self.device, dtype=self.dtype, generator=generator)
        rho = self._radius_curve(u) * torch.exp(v)
        return torch.stack([rho * torch.cos(u), rho * torch.sin(u)], dim=1)

    def _log_pu(self, u: torch.Tensor) -> torch.Tensor:
        q_raw = (u - self.u_mid) / self.u_half
        valid = torch.abs(q_raw) < 1.0
        q = torch.clamp(q_raw, min=-1.0 + 1.0e-12, max=1.0 - 1.0e-12)
        z_u = torch.atanh(q)
        log_jac_inv = -math.log(self.u_half) - torch.log(torch.clamp(1.0 - q.square(), min=1.0e-30))
        lp = -0.5 * (z_u / self.u_std).square() - 0.5 * math.log(2.0 * math.pi) - math.log(self.u_std) + log_jac_inv
        return torch.where(valid, lp, torch.full_like(lp, -float("inf")))

    def _native_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        rho = torch.sqrt(torch.sum(z.square(), dim=1) + 1.0e-24)
        phi = torch.atan2(z[:, 1], z[:, 0])
        components: List[torch.Tensor] = []
        for k in self.branch_ks:
            u = phi + 2.0 * math.pi * float(k)
            r_curve = self._radius_curve(u)
            valid_r = r_curve > 0.0
            v = torch.log(rho / torch.clamp(r_curve, min=1.0e-30))
            lpv = -0.5 * (v / self.logradial_std).square() - 0.5 * math.log(2.0 * math.pi) - math.log(self.logradial_std)
            comp = self._log_pu(u) + lpv - 2.0 * torch.log(rho)
            components.append(torch.where(valid_r, comp, torch.full_like(comp, -float("inf"))))
        return torch.logsumexp(torch.stack(components, dim=1), dim=1)

    def _extra_target_info(self) -> Dict[str, object]:
        return {
            "spiral_turns": self.turns,
            "spiral_r_min": self.r_min,
            "spiral_r_max": self.r_max,
            "spiral_u_std": self.u_std,
            "spiral_logradial_std": self.logradial_std,
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

def project_symmetric_gate_min_eval(G: torch.Tensor, gate_min_eval: float) -> torch.Tensor:
    """Project a symmetric gate to eigenvalues >= gate_min_eval when finite."""
    floor = float(gate_min_eval)
    G = sym(G)
    if not math.isfinite(floor):
        return G
    evals, evecs = torch.linalg.eigh(G)
    evals = torch.clamp(evals, min=floor)
    return sym(evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2))


def resolvent_gate(
    P: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    gate_clip: Optional[float],
    gate_min_eval: float = -float("inf"),
) -> torch.Tensor:
    d = P.shape[-1]
    I = torch.eye(d, device=P.device, dtype=P.dtype)
    A = sym(alpha * alpha * I + gamma * P)
    evals, evecs = torch.linalg.eigh(A)
    signs = torch.where(evals >= 0, torch.ones_like(evals), -torch.ones_like(evals))
    evals_safe = torch.where(torch.abs(evals) < float(eps), signs * float(eps), evals)
    gvals = (alpha * alpha) / evals_safe
    if gate_clip is not None and gate_clip > 0:
        gvals = torch.clamp(gvals, min=-float(gate_clip), max=float(gate_clip))
    if math.isfinite(float(gate_min_eval)):
        gvals = torch.clamp(gvals, min=float(gate_min_eval))
    return sym(evecs @ torch.diag_embed(gvals) @ evecs.transpose(-1, -2))


def project_symmetric_gate_interval(
    G: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project a symmetric matrix field to ``lower I <= G <= upper I``.

    The q-LFGI resolvent is not automatically a contraction when the localized
    observed-information estimate is indefinite.  The complement ``I-G`` used
    by either complement-gated ratio node is meaningful only for a contraction gate.
    This helper therefore performs the explicit spectral safeguard required by
    that node and returns both the projected gate and the pre-projection
    eigenvalues for diagnostics.
    """
    if not (math.isfinite(float(lower)) and math.isfinite(float(upper)) and float(lower) <= float(upper)):
        raise ValueError(f"Invalid gate interval [{lower}, {upper}]")
    Gs = sym(G)
    evals, evecs = torch.linalg.eigh(Gs)
    evals_proj = torch.clamp(evals, min=float(lower), max=float(upper))
    Gp = sym(evecs @ torch.diag_embed(evals_proj) @ evecs.transpose(-1, -2))
    return Gp, evals


PI_LFGI_METHOD_KEYS = {
    "pi-lfgi",
    "pi-ce-hlsi",
    "oracle-lfgi",
    "target-lfgi",
}

N_LFGI_METHOD_KEYS = {
    "lfgi-n",
    "normal-lfgi",
    "gaussian-lfgi",
    "standard-normal-lfgi",
    "n-lfgi",
}

OS_LFGI_METHOD_KEYS = {
    "os-lfgi",
    "one-step-lfgi",
    "residual-corrected-lfgi",
}


def canonical_score_method_key(method: str) -> str:
    return str(method).strip().lower().replace("_", "-")


def is_pi_lfgi_method(method: str) -> bool:
    return canonical_score_method_key(method) in PI_LFGI_METHOD_KEYS


def is_n_lfgi_method(method: str) -> bool:
    return canonical_score_method_key(method) in N_LFGI_METHOD_KEYS


def is_os_lfgi_method(method: str) -> bool:
    return canonical_score_method_key(method) in OS_LFGI_METHOD_KEYS


RAW_LFGI_RESOLVENT_METHOD_KEYS = {
    "ce-hlsi", "lfgi", "ce-lfgi",
    *PI_LFGI_METHOD_KEYS,
    *N_LFGI_METHOD_KEYS,
    *OS_LFGI_METHOD_KEYS,
}


def is_raw_lfgi_resolvent_method(method: str) -> bool:
    """True for LFGI variants that use the configured raw/PSD curvature bank.

    MP-leaf LFGI is deliberately excluded because it uses an explicitly PSD
    completed precision bank and therefore has no raw resolvent pole.
    """
    return canonical_score_method_key(method) in RAW_LFGI_RESOLVENT_METHOD_KEYS


class SNISScoreBank:
    def __init__(
        self,
        target,
        anchors: torch.Tensor,
        cfg: Config,
        log_ref_weights: Optional[torch.Tensor] = None,
        gate_anchors: Optional[torch.Tensor] = None,
        gate_log_ref_weights: Optional[torch.Tensor] = None,
        pi_gate_anchors: Optional[torch.Tensor] = None,
        pi_gate_log_ref_weights: Optional[torch.Tensor] = None,
        n_gate_anchors: Optional[torch.Tensor] = None,
        n_gate_log_ref_weights: Optional[torch.Tensor] = None,
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

        # Optional oracle target gate bank for pi-LFGI. The evolving q bank
        # still supplies Tweedie and TSI/cross-score signals; only the LFGI
        # curvature resolvent is localized with independent target samples.
        self.has_pi_gate_bank = pi_gate_anchors is not None
        if self.has_pi_gate_bank:
            self.x_pi_gate = pi_gate_anchors.detach().to(device=target.device, dtype=target.dtype)
            self.N_pi_gate = int(self.x_pi_gate.shape[0])
            if self.N_pi_gate <= 0:
                raise ValueError("pi_gate_anchors must contain at least one target sample")
            if pi_gate_log_ref_weights is None:
                self.log_pi_gate_weights = torch.zeros((self.N_pi_gate,), device=self.device, dtype=self.dtype)
            else:
                plw = pi_gate_log_ref_weights.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
                if plw.shape[0] != self.N_pi_gate:
                    raise ValueError(
                        f"pi_gate_log_ref_weights has length {plw.shape[0]} but pi_gate_anchors have length {self.N_pi_gate}"
                    )
                self.log_pi_gate_weights = torch.nan_to_num(plw, nan=0.0, posinf=0.0, neginf=0.0)
            self.score0_pi_gate = target.score(self.x_pi_gate, t=0.0).detach()
            Hpi_gate = target.observed_information(self.x_pi_gate, t=0.0).detach()
            self.H_pi_gate_raw = sym(Hpi_gate).detach()
            self.P_pi_gate = process_curvature(
                self.H_pi_gate_raw, cfg.curvature_mode, cfg.curvature_floor, cfg.curvature_cap
            ).detach()
        else:
            self.x_pi_gate = None
            self.N_pi_gate = 0
            self.log_pi_gate_weights = None
            self.score0_pi_gate = None
            self.H_pi_gate_raw = None
            self.P_pi_gate = None

        self.mp_leaf_info.update({
            "pi_gate_bank_available": bool(self.has_pi_gate_bank),
            "pi_gate_bank_n": int(self.N_pi_gate),
            "pi_gate_bank_source": "target" if self.has_pi_gate_bank else "none",
        })

        # Optional fixed standard-normal gate bank for LFGI-N.  As with pi-LFGI,
        # the evolving q bank still supplies the Tweedie and TSI/cross-score
        # signals.  Only the curvature-resolvent localization measure changes,
        # here to an independently drawn N(0,I_d) bank.
        self.has_n_gate_bank = n_gate_anchors is not None
        if self.has_n_gate_bank:
            self.x_n_gate = n_gate_anchors.detach().to(device=target.device, dtype=target.dtype)
            self.N_n_gate = int(self.x_n_gate.shape[0])
            if self.N_n_gate <= 0:
                raise ValueError("n_gate_anchors must contain at least one standard-normal sample")
            if n_gate_log_ref_weights is None:
                self.log_n_gate_weights = torch.zeros((self.N_n_gate,), device=self.device, dtype=self.dtype)
            else:
                nlw = n_gate_log_ref_weights.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
                if nlw.shape[0] != self.N_n_gate:
                    raise ValueError(
                        f"n_gate_log_ref_weights has length {nlw.shape[0]} but n_gate_anchors have length {self.N_n_gate}"
                    )
                self.log_n_gate_weights = torch.nan_to_num(nlw, nan=0.0, posinf=0.0, neginf=0.0)
            self.score0_n_gate = target.score(self.x_n_gate, t=0.0).detach()
            Hn_gate = target.observed_information(self.x_n_gate, t=0.0).detach()
            self.H_n_gate_raw = sym(Hn_gate).detach()
            self.P_n_gate = process_curvature(
                self.H_n_gate_raw, cfg.curvature_mode, cfg.curvature_floor, cfg.curvature_cap
            ).detach()
        else:
            self.x_n_gate = None
            self.N_n_gate = 0
            self.log_n_gate_weights = None
            self.score0_n_gate = None
            self.H_n_gate_raw = None
            self.P_n_gate = None

        self.mp_leaf_info.update({
            "n_gate_bank_available": bool(self.has_n_gate_bank),
            "n_gate_bank_n": int(self.N_n_gate),
            "n_gate_bank_source": "standard_normal" if self.has_n_gate_bank else "none",
        })
        self._uniform_score_moment_cache: Optional[torch.Tensor] = None
        self._uniform_matrix_gate_cache: Dict[Tuple[float, float, float, bool], torch.Tensor] = {}
        self._uniform_scalar_gate_cache: Dict[Tuple[float, float, bool], torch.Tensor] = {}

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

    def _gate_weights_and_signals(self, y: torch.Tensor, t: float, method: Optional[str] = None):
        if method is not None and is_pi_lfgi_method(method):
            if not self.has_pi_gate_bank:
                raise RuntimeError(
                    "pi-LFGI requested, but this SNISScoreBank has no independently drawn target gate bank"
                )
            return self._weights_and_signals_for(
                y, t, self.x_pi_gate, self.score0_pi_gate, self.log_pi_gate_weights
            )
        if method is not None and is_n_lfgi_method(method):
            if not self.has_n_gate_bank:
                raise RuntimeError(
                    "LFGI-N requested, but this SNISScoreBank has no independently drawn standard-normal gate bank"
                )
            return self._weights_and_signals_for(
                y, t, self.x_n_gate, self.score0_n_gate, self.log_n_gate_weights
            )
        return self._weights_and_signals_for(y, t, self.x_gate, self.score0_gate, self.log_gate_weights)

    def _gate_precision_for_method(self, method: str) -> torch.Tensor:
        key = canonical_score_method_key(method)
        if is_pi_lfgi_method(key):
            if self.P_pi_gate is None:
                raise RuntimeError(
                    "pi-LFGI requested, but this SNISScoreBank has no independently drawn target gate bank"
                )
            return self.P_pi_gate
        if is_n_lfgi_method(key):
            if self.P_n_gate is None:
                raise RuntimeError(
                    "LFGI-N requested, but this SNISScoreBank has no independently drawn standard-normal gate bank"
                )
            return self.P_n_gate
        if key in {"leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"}:
            return self.P_gate_mp
        return self.P_gate

    def _global_gate_score_second_moment(self) -> torch.Tensor:
        """Weighted global E[s_0(X)s_0(X)^T] for spatially uniform blends."""
        if self._uniform_score_moment_cache is not None:
            return self._uniform_score_moment_cache
        s0 = self.score0_gate
        n, d = s0.shape
        if n <= 0:
            raise ValueError("uniform blend moment bank is empty")
        logw = torch.nan_to_num(self.log_gate_weights.reshape(-1), nan=-float("inf"), posinf=0.0, neginf=-float("inf"))
        if logw.numel() != n:
            raise ValueError(f"log_gate_weights has length {logw.numel()} but gate score bank has length {n}")
        m = torch.max(logw)
        if not bool(torch.isfinite(m).detach().cpu().item()):
            w = torch.full((n,), 1.0 / float(n), device=self.device, dtype=self.dtype)
        else:
            w = torch.exp(logw - m)
            w = w / torch.clamp(w.sum(), min=1.0e-300)
        Ipi = s0.transpose(0, 1) @ (w[:, None] * s0)
        Ipi = sym(torch.nan_to_num(Ipi, nan=0.0, posinf=0.0, neginf=0.0))
        self._uniform_score_moment_cache = Ipi
        return Ipi

    def _uniform_scalar_tweedie_weight(self, t: float, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        ridge = float(getattr(self.cfg, "matrix_blend_ridge", 1.0e-8))
        clamp = bool(getattr(self.cfg, "uniform_blend_clamp", True))
        cache_key = (round(float(t), 15), ridge, clamp)
        cached = self._uniform_scalar_gate_cache.get(cache_key)
        if cached is not None:
            return cached
        Ipi = self._global_gate_score_second_moment()
        if ridge > 0.0:
            Ipi = Ipi + ridge * torch.eye(self.d, device=self.device, dtype=self.dtype)
        tr_ipi = torch.diagonal(Ipi, dim1=-2, dim2=-1).sum().clamp(min=0.0)
        alpha2 = alpha * alpha
        denom = alpha2 * float(max(self.d, 1)) + gamma * tr_ipi
        a = (gamma * tr_ipi) / torch.clamp(denom, min=1.0e-300)
        if clamp:
            a = a.clamp(0.0, 1.0)
        a = torch.nan_to_num(a, nan=0.0, posinf=1.0 if clamp else 0.0, neginf=0.0)
        self._uniform_scalar_gate_cache[cache_key] = a
        return a

    def _uniform_matrix_tweedie_weight(self, t: float, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        ridge = float(getattr(self.cfg, "matrix_blend_ridge", 1.0e-8))
        ridge_rel = float(getattr(self.cfg, "matrix_blend_ridge_rel", 1.0e-6))
        clamp = bool(getattr(self.cfg, "uniform_blend_clamp", True))
        cache_key = (round(float(t), 15), ridge, ridge_rel, clamp)
        cached = self._uniform_matrix_gate_cache.get(cache_key)
        if cached is not None:
            return cached
        Ipi = self._global_gate_score_second_moment()
        evals, evecs = torch.linalg.eigh(sym(Ipi))
        evals = torch.nan_to_num(evals, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        tr_scale = torch.mean(evals).clamp(min=0.0) if evals.numel() else torch.tensor(0.0, device=self.device, dtype=self.dtype)
        ridge_eff = ridge + ridge_rel * float(tr_scale.detach().cpu().item())
        alpha2 = alpha * alpha
        a_eig = (gamma * evals) / torch.clamp(alpha2 + gamma * evals + ridge_eff, min=1.0e-300)
        if clamp:
            a_eig = a_eig.clamp(0.0, 1.0)
        A = torch.einsum("ik,k,jk->ij", evecs, a_eig, evecs)
        A = sym(torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0))
        if math.isfinite(float(getattr(self.cfg, "gate_min_eval", -float("inf")))):
            A = project_symmetric_gate_min_eval(A, float(self.cfg.gate_min_eval))
        self._uniform_matrix_gate_cache[cache_key] = A
        return A

    def _q_lfgi_score_and_os_correction_chunk(
        self,
        y: torch.Tensor,
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ordinary q-LFGI and its one-step residual correction.

        Let B be the particle-wise Tweedie signal, U the particle-wise TSI
        signal, R=U-B, r=E[R|y], and G0 the usual q-LFGI gate.  With

            D  = R-r,
            E0 = (B-E[B|y]) + G0 D,

        the one-step score correction is

            -tau Cov(E0,D|y) M0^{-1} r.

        Because G0=(1/gamma) M0^{-1} for the unprojected resolvent, the action
        M0^{-1}r is evaluated as gamma*G0*r.  This keeps the implementation
        matrix-free and inherits the same spectral clipping/projection used by
        the actual LFGI gate.
        """
        w, b, c, alpha, gamma = self._weights_and_signals(y, t)
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)

        wg, _bg, _cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t, method="ce-hlsi")
        Pbar = torch.sum(wg[:, :, None, None] * self.P_gate[None, :, :, :], dim=1)
        G0 = resolvent_gate(
            Pbar,
            alpha,
            gamma,
            self.cfg.resolvent_eps,
            self.cfg.gate_clip,
            getattr(self.cfg, "gate_min_eval", -float("inf")),
        )

        rbar = cbar - bbar
        G0r = torch.einsum("bij,bj->bi", G0, rbar)
        lfgi_score = bbar + G0r

        # Center the paired disagreement and the LFGI residual under the same
        # score-bank conditional weights.  Only the covariance action needed by
        # the score is formed; no dense dxd moment matrix is materialized.
        D = (c - b) - rbar[:, None, :]
        G0D = torch.einsum("bij,bnj->bni", G0, D)
        E0 = (b - bbar[:, None, :]) + G0D
        z = gamma * G0r
        Dz = torch.einsum("bni,bi->bn", D, z)
        correction = -torch.sum(w[:, :, None] * E0 * Dz[:, :, None], dim=1)

        tau = float(getattr(self.cfg, "os_lfgi_tau", 1.0))
        if not math.isfinite(tau):
            raise ValueError(f"os_lfgi_tau must be finite; got {tau}")
        correction = tau * correction
        return (
            torch.nan_to_num(lfgi_score, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0),
        )

    @torch.no_grad()
    def os_lfgi_score_and_correction(
        self,
        y: torch.Tensor,
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores: List[torch.Tensor] = []
        corrections: List[torch.Tensor] = []
        for start in range(0, y.shape[0], int(self.cfg.eval_chunk)):
            score, correction = self._q_lfgi_score_and_os_correction_chunk(
                y[start:start + int(self.cfg.eval_chunk)],
                t,
            )
            scores.append(score)
            corrections.append(correction)
        return torch.cat(scores, dim=0), torch.cat(corrections, dim=0)

    @torch.no_grad()
    def os_lfgi_correction(self, y: torch.Tensor, t: float) -> torch.Tensor:
        _score, correction = self.os_lfgi_score_and_correction(y, t)
        return correction

    def estimator_gate_components_chunk(
        self,
        y: torch.Tensor,
        t: float,
        method: str,
    ) -> Dict[str, object]:
        """Return an estimator carrier together with its native gate.

        All supported score estimators are expressed, where possible, in the
        common form

            s_A = bbar + G_A (cbar - bbar).

        Here ``bbar`` is the current-reference Tweedie score and ``cbar`` is the
        target-on-reference conditional score signal.  The ratio flow uses the
        literal complement of the same estimator-native gate.
        ``ratio_gate_policy=native`` preserves that gate exactly.
        ``ratio_gate_policy=project`` instead projects to ``0 <= G <= I`` and
        rebuilds a carrier consistent with the projected gate; diagnostics make
        that change explicit.

        OS-LFGI contains an additive one-step residual correction that is not
        itself represented by a unique matrix gate.  Its carrier retains that
        correction, while its complement uses the underlying q-LFGI resolvent.
        This is the canonical gate associated with the estimator, but the
        returned consistency residual makes the distinction auditable.
        """
        key = canonical_score_method_key(method)
        if key == "none":
            raise ValueError("The complement ratio flow requires a score estimator with a defined carrier/gate; 'none' is not admissible")

        w, b, c, alpha, gamma = self._weights_and_signals(y, t)
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)
        disagreement = cbar - bbar
        B, d = bbar.shape
        eye = torch.eye(d, device=y.device, dtype=y.dtype)
        eye_b = eye.expand(B, d, d)
        gate_kind = "unknown"

        if key in {"blend", "blended", "scalar-blend", "scalar"}:
            # Existing Blend uses a coordinate-wise Tweedie weight g.  In the
            # target-correction convention s=bbar+G(cbar-bbar), G=diag(1-g).
            wg, bg, cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t)
            bgbar = torch.sum(wg[:, :, None] * bg, dim=1)
            cgbar = torch.sum(wg[:, :, None] * cg, dim=1)
            Ac = cg - cgbar[:, None, :]
            Bc = bg - bgbar[:, None, :]
            va = torch.sum(wg[:, :, None] * Ac.square(), dim=1).clamp(min=1.0e-30)
            vb = torch.sum(wg[:, :, None] * Bc.square(), dim=1).clamp(min=1.0e-30)
            cab = torch.sum(wg[:, :, None] * Ac * Bc, dim=1)
            den = (va + vb - 2.0 * cab).clamp(min=1.0e-20)
            tweedie_weight = ((va - cab) / den).clamp(0.0, 1.0)
            G_raw = torch.diag_embed(1.0 - tweedie_weight)
            carrier = cbar + tweedie_weight * (bbar - cbar)
            gate_kind = "local-diagonal-blend"

        elif key in {"unif-blend", "unif-scalar-blend", "uniform-blend", "uniform-scalar-blend", "global-scalar-blend"}:
            tweedie_weight = self._uniform_scalar_tweedie_weight(float(t), alpha, gamma)
            G0 = (1.0 - tweedie_weight) * eye
            G_raw = G0.expand(B, d, d)
            carrier = cbar + tweedie_weight * (bbar - cbar)
            gate_kind = "uniform-scalar-blend"

        elif key in {"unif-matrix-blend", "uniform-matrix-blend", "global-matrix-blend"}:
            A = self._uniform_matrix_tweedie_weight(float(t), alpha, gamma)
            G0 = eye - A
            G_raw = G0.expand(B, d, d)
            carrier = cbar + torch.einsum("ij,bj->bi", A, bbar - cbar)
            gate_kind = "uniform-matrix-blend"

        elif key in {"matrix-blend", "centered-matrix-blend", "centered-blend", "local-matrix-blend"}:
            wg, bg, cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t)
            dg = cg - bg
            if bool(getattr(self.cfg, "matrix_blend_center", True)):
                bgbar = torch.sum(wg[:, :, None] * bg, dim=1)
                dgbar = torch.sum(wg[:, :, None] * dg, dim=1)
                b_mom = bg - bgbar[:, None, :]
                d_mom = dg - dgbar[:, None, :]
            else:
                b_mom = bg
                d_mom = dg
            M = torch.einsum("bn,bni,bnj->bij", wg, d_mom, d_mom)
            N = torch.einsum("bn,bni,bnj->bij", wg, b_mom, d_mom)
            M = sym(torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0))
            N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
            tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1) / float(max(d, 1))
            ridge_vec = float(getattr(self.cfg, "matrix_blend_ridge", 1.0e-8)) + float(getattr(self.cfg, "matrix_blend_ridge_rel", 1.0e-6)) * tr.clamp(min=0.0)
            M_reg = M + ridge_vec[:, None, None] * eye_b
            try:
                G_raw = torch.linalg.solve(M_reg.transpose(-1, -2), (-N).transpose(-1, -2)).transpose(-1, -2)
            except RuntimeError:
                G_raw = -torch.matmul(N, torch.linalg.pinv(M_reg))
            if not bool(torch.isfinite(G_raw).all().detach().cpu().item()):
                G_pinv = -torch.matmul(N, torch.linalg.pinv(M_reg))
                G_raw = torch.where(torch.isfinite(G_raw), G_raw, G_pinv)
            G_raw = torch.nan_to_num(G_raw, nan=0.0, posinf=0.0, neginf=0.0)
            projection_active = math.isfinite(float(getattr(self.cfg, "gate_min_eval", -float("inf"))))
            if bool(getattr(self.cfg, "matrix_blend_sym_gate", False)) or projection_active:
                G_raw = sym(G_raw)
            clip = float(getattr(self.cfg, "matrix_blend_gate_clip", 1.0e6))
            if math.isfinite(clip) and clip > 0.0:
                G_raw = G_raw.clamp(min=-clip, max=clip)
            if projection_active:
                G_raw = project_symmetric_gate_min_eval(G_raw, float(self.cfg.gate_min_eval))
            carrier = bbar + torch.einsum("bij,bj->bi", G_raw, disagreement)
            gate_kind = "local-matrix-blend"

        elif key in {
            "ce-hlsi", "lfgi", "ce-lfgi",
            "pi-lfgi", "pi-ce-hlsi", "oracle-lfgi", "target-lfgi",
            "lfgi-n", "normal-lfgi", "gaussian-lfgi", "standard-normal-lfgi", "n-lfgi",
            "leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi",
        } or is_os_lfgi_method(key):
            gate_method = "ce-hlsi" if is_os_lfgi_method(key) else method
            wg, _bg, _cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t, method=gate_method)
            Pgate = self._gate_precision_for_method(gate_method)
            Pbar = torch.sum(wg[:, :, None, None] * Pgate[None, :, :, :], dim=1)
            G_raw = resolvent_gate(
                Pbar,
                alpha,
                gamma,
                self.cfg.resolvent_eps,
                self.cfg.gate_clip,
                getattr(self.cfg, "gate_min_eval", -float("inf")),
            )
            if is_os_lfgi_method(key):
                lfgi_score, correction = self._q_lfgi_score_and_os_correction_chunk(y, t)
                carrier = lfgi_score + correction
                gate_kind = "os-lfgi-with-base-lfgi-gate"
            else:
                carrier = bbar + torch.einsum("bij,bj->bi", G_raw, disagreement)
                gate_kind = "lfgi-resolvent"

        elif key in {"tweedie", "twd"}:
            G_raw = torch.zeros((B, d, d), device=y.device, dtype=y.dtype)
            carrier = bbar
            gate_kind = "zero-gate-tweedie"

        elif key in {"tsi", "target-score", "target-side-information", "target-side-score"}:
            G_raw = eye_b.clone()
            carrier = cbar
            gate_kind = "identity-gate-tsi"

        else:
            raise ValueError(
                f"Score method {method!r} does not expose a mathematically defined gate for a complement-gated ratio flow. "
                "Supported families are Blend, Matrix Blend, uniform scalar/matrix Blend, LFGI variants, Tweedie, and TSI."
            )

        G_raw = torch.nan_to_num(G_raw, nan=0.0, posinf=0.0, neginf=0.0)
        G_admissible, raw_gate_eigs = project_symmetric_gate_interval(G_raw, lower=0.0, upper=1.0)
        projected_carrier = bbar + torch.einsum("bij,bj->bi", G_admissible, disagreement)
        gate_policy = canonical_ratio_gate_policy(getattr(self.cfg, "ratio_gate_policy", "native"))
        if gate_policy == "native":
            G_used = G_raw
            carrier_used = carrier
        elif gate_policy == "project":
            G_used = G_admissible
            carrier_used = projected_carrier
        else:
            raise RuntimeError(f"Unhandled ratio_gate_policy={gate_policy!r}")
        gate_consistent_carrier = bbar + torch.einsum("bij,bj->bi", G_used, disagreement)
        consistency_residual = carrier_used - gate_consistent_carrier
        antisym = G_raw - G_raw.transpose(-1, -2)
        antisym_norm = torch.linalg.matrix_norm(antisym, ord="fro", dim=(-2, -1))
        interval_violation = (raw_gate_eigs < 0.0) | (raw_gate_eigs > 1.0)
        return {
            "carrier_score": torch.nan_to_num(carrier_used, nan=0.0, posinf=0.0, neginf=0.0),
            "base_score": torch.nan_to_num(bbar, nan=0.0, posinf=0.0, neginf=0.0),
            "target_signal_score": torch.nan_to_num(cbar, nan=0.0, posinf=0.0, neginf=0.0),
            "disagreement": torch.nan_to_num(disagreement, nan=0.0, posinf=0.0, neginf=0.0),
            "gate_raw": G_raw,
            "gate_projected": G_admissible,
            "gate": G_used,
            "gate_policy": gate_policy,
            "raw_gate_eigs": raw_gate_eigs,
            "raw_gate_interval_violation": interval_violation,
            "gate_kind": gate_kind,
            "gate_antisym_frob": antisym_norm,
            "gate_consistency_residual": torch.nan_to_num(consistency_residual, nan=0.0, posinf=0.0, neginf=0.0),
            "conditional_weights": w,
            "particle_tweedie_signals": b,
            "particle_target_signals": c,
        }

    def gated_pflow_components_chunk(
        self,
        y: torch.Tensor,
        t: float,
        method: str,
        endpoint_log_tilt: torch.Tensor,
        complement_strength: float,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate an estimator-native complement-gated ratio score.

        The selected score estimator supplies both the carrier and its native
        gate modality.  The density-derived target score is rebuilt from the
        current particle bank and endpoint density-ratio labels.  The flow is

            s_R = s_A + lambda (I-G_A) (s_pi,dens - s_A).

        ``G_A`` is the estimator-native gate under the configured gate policy.
        This generalizes the old LFGI-only implementation without changing the
        information used by the tilt.
        """
        log_tilt = endpoint_log_tilt.to(device=y.device, dtype=y.dtype).reshape(-1)
        if int(log_tilt.numel()) != int(self.N):
            raise ValueError(
                f"endpoint_log_tilt has length {log_tilt.numel()} but score bank has {self.N} anchors"
            )
        strength = float(complement_strength)
        if not math.isfinite(strength):
            raise ValueError(f"lambda_guard/complement strength must be finite; got {strength}")

        native = self.estimator_gate_components_chunk(y, t, method)
        wq = native["conditional_weights"]
        b = native["particle_tweedie_signals"]
        s_q = native["carrier_score"]
        G_comp = native["gate"]

        # Multiplying the current q-fiber by the current endpoint tilt gives the
        # current particle approximation of the diffused target density score.
        target_logits = torch.log(torch.clamp(wq, min=1.0e-300)) + log_tilt[None, :]
        wpi = torch.softmax(target_logits, dim=1)
        s_pi_dens = torch.sum(wpi[:, :, None] * b, dim=1)

        residual = s_pi_dens - s_q
        complement = torch.eye(self.d, device=y.device, dtype=y.dtype)[None, :, :] - G_comp
        correction = torch.einsum("bij,bj->bi", complement, residual)
        score = s_q + strength * correction
        cond_ess = 1.0 / torch.clamp(torch.sum(wpi * wpi, dim=1), min=1.0e-30)
        return {
            "score": torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
            "s_q": torch.nan_to_num(s_q, nan=0.0, posinf=0.0, neginf=0.0),
            "s_pi_dens": torch.nan_to_num(s_pi_dens, nan=0.0, posinf=0.0, neginf=0.0),
            "gate": G_comp,
            "raw_gate_eigs": native["raw_gate_eigs"],
            "gate_antisym_frob": native["gate_antisym_frob"],
            "gate_consistency_residual": native["gate_consistency_residual"],
            "residual": torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0),
            "correction": torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0),
            "target_cond_ess": torch.nan_to_num(cond_ess, nan=0.0, posinf=0.0, neginf=0.0),
        }

    @torch.no_grad()
    def gated_pflow_score(
        self,
        y: torch.Tensor,
        t: float,
        method: str,
        endpoint_log_tilt: torch.Tensor,
        complement_strength: float,
    ) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        chunk = max(int(self.cfg.eval_chunk), 1)
        for start in range(0, y.shape[0], chunk):
            parts = self.gated_pflow_components_chunk(
                y[start:start + chunk], t, method, endpoint_log_tilt, complement_strength
            )
            outs.append(parts["score"])
        return torch.cat(outs, dim=0)

    def gated_bflow_components_chunk(
        self,
        y: torch.Tensor,
        t: float,
        method: str,
        endpoint_log_tilt: torch.Tensor,
        complement_strength: float,
        apply_gate_filter: bool = True,
        restore_cancellation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate gated, raw, or completed shared-statistic b-flow scores.

        The selected estimator supplies the current-law carrier ``s_A`` and its
        native gate ``G_A``.  The OU Tweedie statistic is evaluated twice on the
        same current score bank and with the same OU kernel:

            b_q  = E_q[beta_t(X;y) | Y_t=y],
            b_pi = E_q[exp(r(X)) beta_t(X;y) | Y_t=y]
                   / E_q[exp(r(X)) | Y_t=y].

        The gated field is

            s_R = s_A + lambda (I-G_A) (b_pi-b_q).

        With ``apply_gate_filter=False`` the same carrier, conditional kernels,
        endpoint labels, and particle bank instead give the raw-flow control

            s_R = s_A + lambda (b_pi-b_q).

        With ``restore_cancellation=True`` the same coupled particle statistics
        restore the finite-bank cancellation term:

            s_R = s_A + lambda[(I-G_A)(b_pi-b_q) + G_A(c_pi-c_q)].

        For a gate-consistent carrier and lambda=1 this is algebraically identical,
        query by query and sample by sample, to

            b_pi + G_A(c_pi-b_pi).

        There is no independent estimator modality for ``b_q`` or ``b_pi``:
        they are the canonical untilted/tilted Tweedie conditionals.  ``method``
        controls the carrier, gate, score bank convention, and any external gate
        bank used by the selected estimator.
        """
        log_tilt = endpoint_log_tilt.to(device=y.device, dtype=y.dtype).reshape(-1)
        if int(log_tilt.numel()) != int(self.N):
            raise ValueError(
                f"endpoint_log_tilt has length {log_tilt.numel()} but score bank has {self.N} anchors"
            )
        strength = float(complement_strength)
        if not math.isfinite(strength):
            raise ValueError(f"lambda_guard/complement strength must be finite; got {strength}")

        native = self.estimator_gate_components_chunk(y, t, method)
        wq = native["conditional_weights"]
        b_particles = native["particle_tweedie_signals"]
        c_particles = native["particle_target_signals"]
        s_q_method = native["carrier_score"]
        b_q = native["base_score"]
        c_q = native["target_signal_score"]
        G_comp = native["gate"]

        # The same endpoint labels and OU kernel tilt the current q fiber to the
        # target fiber.  At population level b_pi=s_pi, while finite-bank b_pi
        # remains coupled to b_q through the common particles and kernel weights.
        target_logits = torch.log(torch.clamp(wq, min=1.0e-300)) + log_tilt[None, :]
        wpi = torch.softmax(target_logits, dim=1)
        b_pi = torch.sum(wpi[:, :, None] * b_particles, dim=1)
        c_pi = torch.sum(wpi[:, :, None] * c_particles, dim=1)

        b_residual = b_pi - b_q
        c_residual = c_pi - c_q
        eye = torch.eye(self.d, device=y.device, dtype=y.dtype)[None, :, :]
        ratio_filter = eye - G_comp if bool(apply_gate_filter) else eye.expand(y.shape[0], -1, -1)
        b_correction = torch.einsum("bij,bj->bi", ratio_filter, b_residual)
        cancellation_correction = (
            torch.einsum("bij,bj->bi", G_comp, c_residual)
            if bool(restore_cancellation)
            else torch.zeros_like(b_correction)
        )
        correction = b_correction + cancellation_correction
        score = s_q_method + strength * correction

        # Direct finite-bank tilted-LFGI representation.  At strength=1 this
        # agrees with the completed shared representation whenever the carrier
        # is gate-consistent (ordinary LFGI/Blend/Tweedie/TSI cases).
        direct_full_score = b_pi + torch.einsum("bij,bj->bi", G_comp, c_pi - b_pi)
        completed_identity_residual = score - direct_full_score
        cond_ess = 1.0 / torch.clamp(torch.sum(wpi * wpi, dim=1), min=1.0e-30)
        return {
            "score": torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
            "s_q": torch.nan_to_num(s_q_method, nan=0.0, posinf=0.0, neginf=0.0),
            "b_q": torch.nan_to_num(b_q, nan=0.0, posinf=0.0, neginf=0.0),
            "b_pi": torch.nan_to_num(b_pi, nan=0.0, posinf=0.0, neginf=0.0),
            "c_q": torch.nan_to_num(c_q, nan=0.0, posinf=0.0, neginf=0.0),
            "c_pi": torch.nan_to_num(c_pi, nan=0.0, posinf=0.0, neginf=0.0),
            # Keep the common diagnostic key used by the ratio-field probes.
            "s_pi_dens": torch.nan_to_num(b_pi, nan=0.0, posinf=0.0, neginf=0.0),
            "gate": G_comp,
            "ratio_filter": ratio_filter,
            "raw_gate_eigs": native["raw_gate_eigs"],
            "gate_antisym_frob": native["gate_antisym_frob"],
            "gate_consistency_residual": native["gate_consistency_residual"],
            "residual": torch.nan_to_num(b_residual, nan=0.0, posinf=0.0, neginf=0.0),
            "b_residual": torch.nan_to_num(b_residual, nan=0.0, posinf=0.0, neginf=0.0),
            "c_residual": torch.nan_to_num(c_residual, nan=0.0, posinf=0.0, neginf=0.0),
            "b_correction": torch.nan_to_num(b_correction, nan=0.0, posinf=0.0, neginf=0.0),
            "cancellation_correction": torch.nan_to_num(cancellation_correction, nan=0.0, posinf=0.0, neginf=0.0),
            "correction": torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0),
            "direct_full_score": torch.nan_to_num(direct_full_score, nan=0.0, posinf=0.0, neginf=0.0),
            "completed_identity_residual": torch.nan_to_num(completed_identity_residual, nan=0.0, posinf=0.0, neginf=0.0),
            "restore_cancellation": bool(restore_cancellation),
            "target_cond_ess": torch.nan_to_num(cond_ess, nan=0.0, posinf=0.0, neginf=0.0),
        }

    @torch.no_grad()
    def gated_bflow_score(
        self,
        y: torch.Tensor,
        t: float,
        method: str,
        endpoint_log_tilt: torch.Tensor,
        complement_strength: float,
        apply_gate_filter: bool = True,
        restore_cancellation: bool = False,
    ) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        chunk = max(int(self.cfg.eval_chunk), 1)
        for start in range(0, y.shape[0], chunk):
            parts = self.gated_bflow_components_chunk(
                y[start:start + chunk], t, method, endpoint_log_tilt,
                complement_strength, apply_gate_filter=apply_gate_filter,
                restore_cancellation=restore_cancellation,
            )
            outs.append(parts["score"])
        return torch.cat(outs, dim=0)

    def estimate_chunk(self, y: torch.Tensor, t: float, method: str) -> torch.Tensor:
        key = str(method).strip().lower().replace("_", "-")
        if is_os_lfgi_method(key):
            lfgi_score, correction = self._q_lfgi_score_and_os_correction_chunk(y, t)
            return lfgi_score + correction
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
        if key in {"unif-blend", "unif-scalar-blend", "uniform-blend", "uniform-scalar-blend", "global-scalar-blend"}:
            a = self._uniform_scalar_tweedie_weight(float(t), alpha, gamma)
            return cbar + a * (bbar - cbar)
        if key in {"unif-matrix-blend", "uniform-matrix-blend", "global-matrix-blend"}:
            A = self._uniform_matrix_tweedie_weight(float(t), alpha, gamma)
            return cbar + torch.einsum("ij,bj->bi", A, bbar - cbar)
        if key in {"matrix-blend", "centered-matrix-blend", "centered-blend", "local-matrix-blend"}:
            wg, bg, cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t)
            dg = cg - bg
            if bool(getattr(self.cfg, "matrix_blend_center", True)):
                bgbar = torch.sum(wg[:, :, None] * bg, dim=1)
                dgbar = torch.sum(wg[:, :, None] * dg, dim=1)
                b_mom = bg - bgbar[:, None, :]
                d_mom = dg - dgbar[:, None, :]
            else:
                b_mom = bg
                d_mom = dg
            M = torch.einsum("bn,bni,bnj->bij", wg, d_mom, d_mom)
            N = torch.einsum("bn,bni,bnj->bij", wg, b_mom, d_mom)
            M = sym(torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0))
            N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
            Bsz, d = M.shape[0], M.shape[-1]
            eye = torch.eye(d, device=y.device, dtype=y.dtype).expand(Bsz, d, d)
            tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1) / float(max(d, 1))
            ridge_vec = float(getattr(self.cfg, "matrix_blend_ridge", 1.0e-8)) + float(getattr(self.cfg, "matrix_blend_ridge_rel", 1.0e-6)) * tr.clamp(min=0.0)
            M_reg = M + ridge_vec[:, None, None] * eye
            try:
                G = torch.linalg.solve(M_reg.transpose(-1, -2), (-N).transpose(-1, -2)).transpose(-1, -2)
            except RuntimeError:
                G = -torch.matmul(N, torch.linalg.pinv(M_reg))
            if not bool(torch.isfinite(G).all().detach().cpu().item()):
                G_pinv = -torch.matmul(N, torch.linalg.pinv(M_reg))
                G = torch.where(torch.isfinite(G), G, G_pinv)
            G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
            projection_active = math.isfinite(float(getattr(self.cfg, "gate_min_eval", -float("inf"))))
            if bool(getattr(self.cfg, "matrix_blend_sym_gate", False)) or projection_active:
                G = sym(G)
            clip = float(getattr(self.cfg, "matrix_blend_gate_clip", 1.0e6))
            if math.isfinite(clip) and clip > 0.0:
                G = G.clamp(min=-clip, max=clip)
            if projection_active:
                G = project_symmetric_gate_min_eval(G, float(self.cfg.gate_min_eval))
            return bbar + torch.einsum("bij,bj->bi", G, cbar - bbar)
        if key in {"ce-hlsi", "lfgi", "ce-lfgi", "pi-lfgi", "pi-ce-hlsi", "oracle-lfgi", "target-lfgi", "lfgi-n", "normal-lfgi", "gaussian-lfgi", "standard-normal-lfgi", "n-lfgi", "leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"}:
            wg, _bg, _cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t, method=method)
            Pgate = self._gate_precision_for_method(method)
            Pbar = torch.sum(wg[:, :, None, None] * Pgate[None, :, :, :], dim=1)
            G = resolvent_gate(
                Pbar, alpha, gamma, self.cfg.resolvent_eps, self.cfg.gate_clip,
                getattr(self.cfg, "gate_min_eval", -float("inf")),
            )
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

        wg, bg, _cg, _alpha_g, _gamma_g = self._gate_weights_and_signals(y, t, method=method)
        bbar_gate = torch.sum(wg[:, :, None] * bg, dim=1)
        Pgate = self._gate_precision_for_method(method)
        Pbar = torch.sum(wg[:, :, None, None] * Pgate[None, :, :, :], dim=1)
        G = resolvent_gate(
            Pbar, alpha, gamma, self.cfg.resolvent_eps, self.cfg.gate_clip,
            getattr(self.cfg, "gate_min_eval", -float("inf")),
        )
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
    ts_stats = time_grid_step_stats(ts)
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
        "sampler_dt_min": float(ts_stats["dt_min"]),
        "sampler_dt_max": float(ts_stats["dt_max"]),
        "sampler_dt_sum": float(ts_stats["dt_sum"]),
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
def os_lfgi_score_and_hybrid_divergence(
    bank: SNISScoreBank,
    x: torch.Tensor,
    t: float,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic q-LFGI divergence plus Hutchinson OS-correction divergence.

    The ordinary LFGI field retains the closed-form CE-HLSI divergence.  Only
    the additional one-step residual correction is finite-differenced, avoiding
    Hutchinson noise on the dominant closed-form part of the score field.
    """
    lfgi_score, lfgi_div = bank.ce_score_and_divergence(x, t, method="ce-hlsi")
    correction = bank.os_lfgi_correction(x, t)

    probes = max(int(cfg.hutchinson_probes), 1)
    eps = float(cfg.hutchinson_eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"hutchinson_eps must be positive and finite; got {eps}")
    div_corr = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        cp = bank.os_lfgi_correction(x + eps * v, t)
        cm = bank.os_lfgi_correction(x - eps * v, t)
        div_corr = div_corr + torch.sum((cp - cm) * v, dim=1) / (2.0 * eps)
    div_corr = div_corr / float(probes)
    score = lfgi_score + correction
    div = lfgi_div + div_corr
    return (
        torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0),
    )


@torch.no_grad()
def bank_score_and_divergence(bank: SNISScoreBank, x: torch.Tensor, t: float, method: str, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    div_mode = str(cfg.pf_divergence).lower()
    key = canonical_score_method_key(method)
    projection_active = math.isfinite(float(getattr(cfg, "gate_min_eval", -float("inf"))))
    if (
        not projection_active
        and div_mode in {"auto", "analytic_ce", "analytic"}
        and is_os_lfgi_method(key)
    ):
        return os_lfgi_score_and_hybrid_divergence(bank, x, t, cfg)
    if (
        not projection_active
        and div_mode in {"auto", "analytic_ce", "analytic"}
        and key in {"ce-hlsi", "lfgi", "ce-lfgi", "pi-lfgi", "pi-ce-hlsi", "oracle-lfgi", "target-lfgi", "lfgi-n", "normal-lfgi", "gaussian-lfgi", "standard-normal-lfgi", "n-lfgi", "leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"}
    ):
        return bank.ce_score_and_divergence(x, t, method=method)
    return score_and_hutchinson_divergence(bank, x, t, method, cfg)


def effective_pf_divergence_mode(method: str, cfg: Config) -> str:
    """Describe the divergence path actually used for a frozen score field."""
    div_mode = str(cfg.pf_divergence).lower()
    key = canonical_score_method_key(method)
    projection_active = math.isfinite(float(getattr(cfg, "gate_min_eval", -float("inf"))))
    if projection_active or div_mode not in {"auto", "analytic_ce", "analytic"}:
        return "hutchinson_full"
    if is_os_lfgi_method(key):
        return "analytic_lfgi_plus_hutchinson_os"
    if key in {"ce-hlsi", "lfgi", "ce-lfgi", "pi-lfgi", "pi-ce-hlsi", "oracle-lfgi", "target-lfgi", "lfgi-n", "normal-lfgi", "gaussian-lfgi", "standard-normal-lfgi", "n-lfgi", "leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"}:
        return "analytic_ce"
    return "hutchinson_full"


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
    ts_stats = time_grid_step_stats(ts)
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
            # Physical-time interval.  For log_linear this is nonuniform Δt,
            # not a unit log-time step, so no extra dt/dlog(t) factor appears.
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
        "pf_divergence_effective": effective_pf_divergence_mode(method, cfg),
        "pf_failed_frac": float(failed_total) / float(max(1, x0.shape[0])),
        "pf_steps": int(cfg.pf_steps),
        "pf_t_min": float(effective_time_bounds(cfg)[0]),
        "pf_t_max": float(effective_time_bounds(cfg)[1]),
        "pf_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "pf_dt_min": float(ts_stats["dt_min"]),
        "pf_dt_max": float(ts_stats["dt_max"]),
        "pf_dt_sum": float(ts_stats["dt_sum"]),
        "pf_max_abs_div": float(max_abs_div),
        "pf_max_abs_state": float(max_abs_state),
        "pf_logq_mean": safe_float(logq_all.mean()),
        "pf_logq_std": safe_float(logq_all.std(unbiased=False)),
        "pf_logq_min": safe_float(logq_all.min()),
        "pf_logq_max": safe_float(logq_all.max()),
    }


class GatedPFlowRatioField:
    """One-inner-round field using the selected estimator's native complement.

    The field is frozen only during its ODE integration.  The enclosing ratio
    iteration reconstructs this object from the current moved particle bank.
    """

    def __init__(
        self,
        bank: SNISScoreBank,
        method: str,
        endpoint_log_tilt: torch.Tensor,
        complement_strength: float,
    ):
        self.bank = bank
        self.method = str(method)
        self.endpoint_log_tilt = endpoint_log_tilt.detach().to(
            device=bank.device, dtype=bank.dtype
        ).reshape(-1)
        self.complement_strength = float(complement_strength)
        if int(self.endpoint_log_tilt.numel()) != int(bank.N):
            raise ValueError(
                f"GatedPFlowRatioField tilt has {self.endpoint_log_tilt.numel()} labels "
                f"for {bank.N} score anchors"
            )

    @torch.no_grad()
    def estimate(self, y: torch.Tensor, t: float) -> torch.Tensor:
        return self.bank.gated_pflow_score(
            y,
            float(t),
            self.method,
            self.endpoint_log_tilt,
            self.complement_strength,
        )

    @torch.no_grad()
    def estimate_and_divergence(
        self,
        y: torch.Tensor,
        t: float,
        cfg: Config,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ratio field and a divergence estimate of that same field."""
        return score_fn_hutchinson_divergence(self.estimate, y, float(t), cfg)

    @torch.no_grad()
    def probe_diagnostics(self, cfg: Config, n: int = 128, n_t: int = 5) -> Dict[str, float]:
        n = min(max(int(n), 1), int(self.bank.N))
        ids = torch.linspace(0, self.bank.N - 1, n, device=self.bank.device).round().long()
        x0 = self.bank.x[ids]
        t_min, t_max = effective_time_bounds(cfg)
        if canonical_time_schedule(cfg.time_schedule) == "log_linear":
            ts = torch.exp(torch.linspace(
                math.log(t_min), math.log(t_max), max(int(n_t), 2),
                device=self.bank.device, dtype=self.bank.dtype,
            ))
        else:
            ts = torch.linspace(
                t_min, t_max, max(int(n_t), 2),
                device=self.bank.device, dtype=self.bank.dtype,
            )
        cond_ess_vals: List[torch.Tensor] = []
        raw_eig_vals: List[torch.Tensor] = []
        corr_norm_vals: List[torch.Tensor] = []
        residual_norm_vals: List[torch.Tensor] = []
        gate_consistency_vals: List[torch.Tensor] = []
        gate_antisym_vals: List[torch.Tensor] = []
        for tt in ts:
            alpha, gamma = alpha_gamma(tt, device=self.bank.device, dtype=self.bank.dtype)
            # Deterministic probe locations avoid adding a second source of
            # Monte Carlo variation to the method comparison.
            y = alpha * x0
            parts = self.bank.gated_pflow_components_chunk(
                y,
                float(tt.item()),
                self.method,
                self.endpoint_log_tilt,
                self.complement_strength,
            )
            cond_ess_vals.append(parts["target_cond_ess"].reshape(-1))
            raw_eig_vals.append(parts["raw_gate_eigs"].reshape(-1))
            corr_norm_vals.append(torch.linalg.norm(parts["correction"], dim=1))
            residual_norm_vals.append(torch.linalg.norm(parts["residual"], dim=1))
            gate_consistency_vals.append(torch.linalg.norm(parts["gate_consistency_residual"], dim=1))
            gate_antisym_vals.append(parts["gate_antisym_frob"].reshape(-1))
        cond = torch.cat(cond_ess_vals)
        eig = torch.cat(raw_eig_vals)
        corr = torch.cat(corr_norm_vals)
        resid = torch.cat(residual_norm_vals)
        gate_consistency = torch.cat(gate_consistency_vals)
        gate_antisym = torch.cat(gate_antisym_vals)
        clipped = (eig < 0.0) | (eig > 1.0)
        gate_policy = canonical_ratio_gate_policy(getattr(cfg, "ratio_gate_policy", "native"))
        return {
            "ratio_gate_policy": gate_policy,
            "ratio_target_cond_ess_min": safe_float(cond.min()),
            "ratio_target_cond_ess_median": safe_float(torch.median(cond)),
            "ratio_target_cond_ess_mean": safe_float(cond.mean()),
            "ratio_gate_raw_eig_min": safe_float(eig.min()),
            "ratio_gate_raw_eig_max": safe_float(eig.max()),
            "ratio_gate_interval_violation_frac": safe_float(clipped.to(eig.dtype).mean()),
            "ratio_gate_projection_applied": bool(gate_policy == "project"),
            "ratio_gate_interval_clipped_frac": (
                safe_float(clipped.to(eig.dtype).mean()) if gate_policy == "project" else 0.0
            ),
            "ratio_complement_correction_norm_mean": safe_float(corr.mean()),
            "ratio_density_residual_norm_mean": safe_float(resid.mean()),
            "ratio_gate_carrier_consistency_residual_mean": safe_float(gate_consistency.mean()),
            "ratio_gate_raw_antisym_frob_mean": safe_float(gate_antisym.mean()),
        }


class GatedBFlowRatioField:
    """One-inner-round gated, raw, or completed shared-statistic b-flow field.

    The selected estimator supplies ``s_A`` and ``G_A``.  The correction is
    ``(I-G_A)(b_pi-b_q)``, the raw ``b_pi-b_q`` control, or the completed
    ``(I-G_A)(b_pi-b_q)+G_A(c_pi-c_q)`` shared-LFGI correction.  The field is
    frozen only during one ODE integration; the enclosing ratio iteration
    rebuilds it from every moved particle bank.
    """

    def __init__(
        self,
        bank: SNISScoreBank,
        method: str,
        endpoint_log_tilt: torch.Tensor,
        complement_strength: float,
        apply_gate_filter: bool = True,
        restore_cancellation: bool = False,
    ):
        self.bank = bank
        self.method = str(method)
        self.endpoint_log_tilt = endpoint_log_tilt.detach().to(
            device=bank.device, dtype=bank.dtype
        ).reshape(-1)
        self.complement_strength = float(complement_strength)
        self.apply_gate_filter = bool(apply_gate_filter)
        self.restore_cancellation = bool(restore_cancellation)
        if int(self.endpoint_log_tilt.numel()) != int(bank.N):
            raise ValueError(
                f"GatedBFlowRatioField tilt has {self.endpoint_log_tilt.numel()} labels "
                f"for {bank.N} score anchors"
            )

    @torch.no_grad()
    def estimate(self, y: torch.Tensor, t: float) -> torch.Tensor:
        return self.bank.gated_bflow_score(
            y,
            float(t),
            self.method,
            self.endpoint_log_tilt,
            self.complement_strength,
            apply_gate_filter=self.apply_gate_filter,
            restore_cancellation=self.restore_cancellation,
        )

    @torch.no_grad()
    def estimate_and_divergence(
        self,
        y: torch.Tensor,
        t: float,
        cfg: Config,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ratio field and a divergence estimate of that same field."""
        return score_fn_hutchinson_divergence(self.estimate, y, float(t), cfg)

    @torch.no_grad()
    def probe_diagnostics(self, cfg: Config, n: int = 128, n_t: int = 5) -> Dict[str, float]:
        n = min(max(int(n), 1), int(self.bank.N))
        ids = torch.linspace(0, self.bank.N - 1, n, device=self.bank.device).round().long()
        x0 = self.bank.x[ids]
        t_min, t_max = effective_time_bounds(cfg)
        if canonical_time_schedule(cfg.time_schedule) == "log_linear":
            ts = torch.exp(torch.linspace(
                math.log(t_min), math.log(t_max), max(int(n_t), 2),
                device=self.bank.device, dtype=self.bank.dtype,
            ))
        else:
            ts = torch.linspace(
                t_min, t_max, max(int(n_t), 2),
                device=self.bank.device, dtype=self.bank.dtype,
            )
        cond_ess_vals: List[torch.Tensor] = []
        raw_eig_vals: List[torch.Tensor] = []
        corr_norm_vals: List[torch.Tensor] = []
        residual_norm_vals: List[torch.Tensor] = []
        gate_consistency_vals: List[torch.Tensor] = []
        gate_antisym_vals: List[torch.Tensor] = []
        cancellation_norm_vals: List[torch.Tensor] = []
        identity_residual_vals: List[torch.Tensor] = []
        for tt in ts:
            alpha, gamma = alpha_gamma(tt, device=self.bank.device, dtype=self.bank.dtype)
            # Deterministic probe locations avoid adding a second source of
            # Monte Carlo variation to the method comparison.
            y = alpha * x0
            parts = self.bank.gated_bflow_components_chunk(
                y,
                float(tt.item()),
                self.method,
                self.endpoint_log_tilt,
                self.complement_strength,
                apply_gate_filter=self.apply_gate_filter,
                restore_cancellation=self.restore_cancellation,
            )
            cond_ess_vals.append(parts["target_cond_ess"].reshape(-1))
            raw_eig_vals.append(parts["raw_gate_eigs"].reshape(-1))
            corr_norm_vals.append(torch.linalg.norm(parts["correction"], dim=1))
            residual_norm_vals.append(torch.linalg.norm(parts["residual"], dim=1))
            gate_consistency_vals.append(torch.linalg.norm(parts["gate_consistency_residual"], dim=1))
            gate_antisym_vals.append(parts["gate_antisym_frob"].reshape(-1))
            cancellation_norm_vals.append(torch.linalg.norm(parts["cancellation_correction"], dim=1))
            identity_residual_vals.append(torch.linalg.norm(parts["completed_identity_residual"], dim=1))
        cond = torch.cat(cond_ess_vals)
        eig = torch.cat(raw_eig_vals)
        corr = torch.cat(corr_norm_vals)
        resid = torch.cat(residual_norm_vals)
        gate_consistency = torch.cat(gate_consistency_vals)
        gate_antisym = torch.cat(gate_antisym_vals)
        cancellation_norm = torch.cat(cancellation_norm_vals)
        identity_residual = torch.cat(identity_residual_vals)
        clipped = (eig < 0.0) | (eig > 1.0)
        gate_policy = canonical_ratio_gate_policy(getattr(cfg, "ratio_gate_policy", "native"))
        return {
            "ratio_bflow_filter": "I-G+Gdc" if self.restore_cancellation else ("I-G" if self.apply_gate_filter else "I"),
            "ratio_bflow_filter_applied": bool(self.apply_gate_filter),
            "ratio_cancellation_restored": bool(self.restore_cancellation),
            "ratio_gate_policy": gate_policy,
            "ratio_target_cond_ess_min": safe_float(cond.min()),
            "ratio_target_cond_ess_median": safe_float(torch.median(cond)),
            "ratio_target_cond_ess_mean": safe_float(cond.mean()),
            "ratio_gate_raw_eig_min": safe_float(eig.min()),
            "ratio_gate_raw_eig_max": safe_float(eig.max()),
            "ratio_gate_interval_violation_frac": safe_float(clipped.to(eig.dtype).mean()),
            "ratio_gate_projection_applied": bool(gate_policy == "project"),
            "ratio_gate_interval_clipped_frac": (
                safe_float(clipped.to(eig.dtype).mean()) if gate_policy == "project" else 0.0
            ),
            "ratio_complement_correction_norm_mean": safe_float(corr.mean()),
            "ratio_density_residual_norm_mean": safe_float(resid.mean()),
            "ratio_bflow_residual_norm_mean": safe_float(resid.mean()),
            "ratio_cancellation_correction_norm_mean": safe_float(cancellation_norm.mean()),
            "ratio_completed_identity_residual_mean": safe_float(identity_residual.mean()),
            "ratio_completed_identity_residual_max": safe_float(identity_residual.max()),
            "ratio_gate_carrier_consistency_residual_mean": safe_float(gate_consistency.mean()),
            "ratio_gate_raw_antisym_frob_mean": safe_float(gate_antisym.mean()),
        }


@torch.no_grad()
def reverse_ou_heun_probability_flow(
    target,
    score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    cfg: Config,
    generator: torch.Generator,
    n_samples: int,
    steps: Optional[int] = None,
    final_denoise: Optional[bool] = None,
) -> Tuple[torch.Tensor, Dict[str, float | str | bool]]:
    """Sample a frozen score field with the deterministic reverse OU PF ODE."""
    device, dtype, d = target.device, target.dtype, int(target.d)
    n_steps = int(cfg.pf_steps if steps is None else steps)
    if n_steps < 1:
        raise ValueError(f"ratio probability-flow steps must be >=1; got {n_steps}")
    use_final_denoise = bool(cfg.final_denoise if final_denoise is None else final_denoise)
    y = torch.randn((int(n_samples), d), device=device, dtype=dtype, generator=generator)
    ts = make_time_grid(cfg, n_steps, direction="reverse", device=device, dtype=dtype)
    ts_stats = time_grid_step_stats(ts)
    max_abs_score = 0.0
    max_abs_state = safe_float(y.abs().max())
    fail = False
    fail_reason = ""
    for i in range(n_steps):
        tc = float(ts[i].item())
        tn = float(ts[i + 1].item())
        dt = tn - tc  # negative on the reverse grid
        s1 = clamp_norm(score_fn(y, tc), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(s1.abs().max()))
        if not bool(torch.isfinite(s1).all().item()):
            fail, fail_reason = True, "nonfinite score at PF predictor"
            break
        v1 = -y - s1
        y_pred = y + dt * v1
        s2 = clamp_norm(score_fn(y_pred, tn), cfg.score_clip)
        max_abs_score = max(max_abs_score, safe_float(s2.abs().max()))
        if not bool(torch.isfinite(s2).all().item()):
            fail, fail_reason = True, "nonfinite score at PF corrector"
            break
        v2 = -y_pred - s2
        y = y + 0.5 * dt * (v1 + v2)
        if cfg.sample_clip and cfg.sample_clip > 0:
            y = torch.clamp(y, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))
        max_abs_state = max(max_abs_state, safe_float(y.abs().max()))
        if not bool(torch.isfinite(y).all().item()):
            fail, fail_reason = True, "nonfinite PF state"
            break
    if not fail and use_final_denoise:
        t_min, _ = effective_time_bounds(cfg)
        sf = clamp_norm(score_fn(y, t_min), cfg.score_clip)
        if bool(torch.isfinite(sf).all().item()):
            alpha, gamma = alpha_gamma(t_min, device=device, dtype=dtype)
            y = (y + gamma * sf) / alpha
            if cfg.sample_clip and cfg.sample_clip > 0:
                y = torch.clamp(y, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))
        else:
            fail, fail_reason = True, "nonfinite final PF score"
    t_min, t_max = effective_time_bounds(cfg)
    return y.detach(), {
        "ratio_flow_failed": bool(fail),
        "ratio_flow_fail_reason": str(fail_reason),
        "ratio_flow_max_abs_score": float(max_abs_score),
        "ratio_flow_max_abs_state": float(max_abs_state),
        "ratio_flow_steps": int(n_steps),
        "ratio_flow_t_min": float(t_min),
        "ratio_flow_t_max": float(t_max),
        "ratio_flow_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "ratio_flow_dt_min": float(ts_stats["dt_min"]),
        "ratio_flow_dt_max": float(ts_stats["dt_max"]),
        "ratio_flow_dt_sum": float(ts_stats["dt_sum"]),
        "ratio_flow_terminal_policy": "standard_normal",
    }


@torch.no_grad()
def score_fn_hutchinson_divergence(
    score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x: torch.Tensor,
    t: float,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-field Hutchinson divergence for a generic score callback."""
    s = score_fn(x, float(t))
    probes = max(int(cfg.hutchinson_probes), 1)
    eps = float(cfg.hutchinson_eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"hutchinson_eps must be positive and finite; got {eps}")
    div = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
    for _ in range(probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        sp = score_fn(x + eps * v, float(t))
        sm = score_fn(x - eps * v, float(t))
        div = div + torch.sum((sp - sm) * v, dim=1) / (2.0 * eps)
    div = div / float(probes)
    return (
        torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0),
    )


@torch.no_grad()
def reverse_ou_heun_probability_flow_certified(
    target,
    score_div_fn: Callable[[torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]],
    cfg: Config,
    generator: torch.Generator,
    n_samples: int,
    steps: int,
    phase_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float | str | bool]]:
    """Generate a PF endpoint and its incoming Liouville density together.

    The same frozen field, time grid, predictor/corrector evaluations, and
    divergence rule drive both particle motion and the log-density certificate.
    This is the discrete implementation of the incoming-certificate principle:
    the field built from q_{k-1} generates and certifies q_k.  No endpoint-field
    rebuild or TSC assumption is used.

    In strict mode, final denoising and clipping are rejected because they mutate
    the generated law without a Jacobian correction.  score_clip, sample_clip,
    and pf_div_clip are treated as audit thresholds rather than transformations.
    """
    device, dtype, d = target.device, target.dtype, int(target.d)
    n_steps = int(steps)
    if n_steps < 1:
        raise ValueError(f"certified PF steps must be >=1; got {steps}")
    if bool(getattr(cfg, "final_denoise", False)):
        raise ValueError(
            "final_denoise changes the endpoint after the Liouville flow and invalidates its density certificate. "
            "Disable --final_denoise for certified alternating runs."
        )

    y = torch.randn((int(n_samples), d), device=device, dtype=dtype, generator=generator)
    logq = standard_normal_logprob(y)
    ts = make_time_grid(cfg, n_steps, direction="reverse", device=device, dtype=dtype)
    ts_stats = time_grid_step_stats(ts)
    max_abs_score = 0.0
    max_abs_div = 0.0
    max_abs_state = safe_float(y.abs().max())
    fail = False
    fail_reason = ""
    strict = bool(getattr(cfg, "certified_flow_strict", True))
    score_threshold = float(getattr(cfg, "certified_score_abort", 0.0) or 0.0)
    state_threshold = float(getattr(cfg, "certified_state_abort", 0.0) or 0.0)
    div_threshold = float(getattr(cfg, "certified_div_abort", 0.0) or 0.0)

    for i in range(n_steps):
        tc = float(ts[i].item())
        tn = float(ts[i + 1].item())
        dt = tn - tc
        s1, div1 = score_div_fn(y, tc)
        s1 = torch.nan_to_num(s1, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
        div1 = torch.nan_to_num(div1, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
        max_abs_score = max(max_abs_score, safe_float(torch.nan_to_num(s1.abs(), nan=0.0).max()))
        max_abs_div = max(max_abs_div, safe_float(torch.nan_to_num(div1.abs(), nan=0.0).max()))
        if not bool(torch.isfinite(s1).all().item()) or not bool(torch.isfinite(div1).all().item()):
            fail, fail_reason = True, "nonfinite score/divergence at certified PF predictor"
            break
        if strict and score_threshold > 0.0 and safe_float(torch.linalg.norm(s1, dim=1).max()) > score_threshold:
            fail, fail_reason = True, f"score norm exceeded certified_score_abort={score_threshold:.6g} at predictor t={tc:.6g}"
            break
        if strict and div_threshold > 0.0 and safe_float(div1.abs().max()) > div_threshold:
            fail, fail_reason = True, f"|divergence| exceeded certified_div_abort={div_threshold:.6g} at predictor t={tc:.6g}"
            break

        v1 = -y - s1
        a1 = float(d) + div1
        y_pred = y + dt * v1
        logq_pred = logq + dt * a1
        s2, div2 = score_div_fn(y_pred, tn)
        s2 = torch.nan_to_num(s2, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
        div2 = torch.nan_to_num(div2, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
        max_abs_score = max(max_abs_score, safe_float(torch.nan_to_num(s2.abs(), nan=0.0).max()))
        max_abs_div = max(max_abs_div, safe_float(torch.nan_to_num(div2.abs(), nan=0.0).max()))
        if not bool(torch.isfinite(s2).all().item()) or not bool(torch.isfinite(div2).all().item()):
            fail, fail_reason = True, "nonfinite score/divergence at certified PF corrector"
            break
        if strict and score_threshold > 0.0 and safe_float(torch.linalg.norm(s2, dim=1).max()) > score_threshold:
            fail, fail_reason = True, f"score norm exceeded certified_score_abort={score_threshold:.6g} at corrector t={tn:.6g}"
            break
        if strict and div_threshold > 0.0 and safe_float(div2.abs().max()) > div_threshold:
            fail, fail_reason = True, f"|divergence| exceeded certified_div_abort={div_threshold:.6g} at corrector t={tn:.6g}"
            break

        v2 = -y_pred - s2
        a2 = float(d) + div2
        y = y + 0.5 * dt * (v1 + v2)
        logq = logq + 0.5 * dt * (a1 + a2)
        max_abs_state = max(max_abs_state, safe_float(y.abs().max()))
        finite = torch.isfinite(y).all() and torch.isfinite(logq).all()
        if not bool(finite):
            fail, fail_reason = True, "nonfinite state/log-density in certified PF flow"
            break
        if strict and state_threshold > 0.0 and safe_float(y.abs().max()) > state_threshold:
            fail, fail_reason = True, f"state magnitude exceeded certified_state_abort={state_threshold:.6g} after step {i + 1}"
            break

    if fail and strict:
        raise RuntimeError(f"{phase_name}: {fail_reason}")
    t_min, t_max = effective_time_bounds(cfg)
    info: Dict[str, float | str | bool] = {
        "failed": bool(fail),
        "fail_reason": str(fail_reason),
        "max_abs_score": float(max_abs_score),
        "sampler_t_min": float(t_min),
        "sampler_t_max": float(t_max),
        "sampler_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "pf_method": str(phase_name),
        "pf_divergence_mode": "same-field-on-generation-path",
        "pf_divergence_effective": "same-field-on-generation-path",
        "pf_skipped": False,
        "pf_failed_frac": 1.0 if fail else 0.0,
        "pf_fail_reason": str(fail_reason),
        "pf_steps": int(n_steps),
        "pf_t_min": float(t_min),
        "pf_t_max": float(t_max),
        "pf_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "pf_dt_min": float(ts_stats["dt_min"]),
        "pf_dt_max": float(ts_stats["dt_max"]),
        "pf_dt_sum": float(ts_stats["dt_sum"]),
        "pf_max_abs_div": float(max_abs_div),
        "pf_max_abs_score": float(max_abs_score),
        "pf_max_abs_state": float(max_abs_state),
        "pf_logq_mean": safe_float(logq.mean()),
        "pf_logq_std": safe_float(logq.std(unbiased=False)),
        "pf_logq_min": safe_float(logq.min()),
        "pf_logq_max": safe_float(logq.max()),
        "certificate_provenance": "incoming-generating-field",
        "certificate_same_particles_and_field": True,
        "certificate_postflow_mutation": False,
    }
    return y.detach(), logq.detach(), info


@torch.no_grad()
def pf_logprob_score_fn(
    score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x0: torch.Tensor,
    cfg: Config,
    method_name: str,
) -> Tuple[torch.Tensor, Dict[str, float | bool | str]]:
    """Reconstruct endpoint density for a generic frozen PF score field."""
    if x0.numel() == 0:
        return torch.empty((0,), device=x0.device, dtype=x0.dtype), {"pf_failed_frac": 0.0}
    ts = make_time_grid(cfg, int(cfg.pf_steps), direction="forward", device=x0.device, dtype=x0.dtype)
    ts_stats = time_grid_step_stats(ts)
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
            dt = tn - t
            s, div = score_fn_hutchinson_divergence(score_fn, x, t, cfg)
            s = clamp_norm(s, cfg.score_clip)
            if cfg.pf_div_clip and cfg.pf_div_clip > 0:
                div = torch.clamp(div, min=-float(cfg.pf_div_clip), max=float(cfg.pf_div_clip))
            v = -x - s
            a = float(d) + div
            x_e = x + dt * v
            s_e, div_e = score_fn_hutchinson_divergence(score_fn, x_e, tn, cfg)
            s_e = clamp_norm(s_e, cfg.score_clip)
            if cfg.pf_div_clip and cfg.pf_div_clip > 0:
                div_e = torch.clamp(div_e, min=-float(cfg.pf_div_clip), max=float(cfg.pf_div_clip))
            v_e = -x_e - s_e
            a_e = float(d) + div_e
            finite = torch.isfinite(x_e).all(dim=1) & torch.isfinite(v_e).all(dim=1) & torch.isfinite(a) & torch.isfinite(a_e)
            alive = alive & finite
            x = x + 0.5 * dt * (v + v_e)
            A = A + 0.5 * dt * (a + a_e)
            if cfg.sample_clip and cfg.sample_clip > 0:
                x = torch.clamp(x, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))
            max_abs_div = max(max_abs_div, safe_float(torch.max(torch.abs(torch.cat([div, div_e])))))
            max_abs_state = max(max_abs_state, safe_float(x.abs().max()))
        logq = standard_normal_logprob(x) - A
        good = alive & torch.isfinite(logq)
        failed_total += int((~good).sum().item())
        if (~good).any():
            replacement = torch.nanmedian(logq[good]) if good.any() else torch.tensor(0.0, device=x.device, dtype=x.dtype)
            logq = torch.where(good, logq, replacement)
        outs.append(logq.detach())
    logq_all = torch.cat(outs)
    return logq_all, {
        "pf_method": str(method_name),
        "pf_divergence_mode": "hutchinson_full_custom",
        "pf_divergence_effective": "hutchinson_full_custom",
        "pf_skipped": False,
        "pf_failed_frac": float(failed_total) / float(max(int(x0.shape[0]), 1)),
        "pf_steps": int(cfg.pf_steps),
        "pf_t_min": float(effective_time_bounds(cfg)[0]),
        "pf_t_max": float(effective_time_bounds(cfg)[1]),
        "pf_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "pf_dt_min": float(ts_stats["dt_min"]),
        "pf_dt_max": float(ts_stats["dt_max"]),
        "pf_dt_sum": float(ts_stats["dt_sum"]),
        "pf_max_abs_div": float(max_abs_div),
        "pf_max_abs_state": float(max_abs_state),
        "pf_logq_mean": safe_float(logq_all.mean()),
        "pf_logq_std": safe_float(logq_all.std(unbiased=False)),
        "pf_logq_min": safe_float(logq_all.min()),
        "pf_logq_max": safe_float(logq_all.max()),
    }



# -----------------------------------------------------------------------------
# Theorem-5.3 no-correction convergence diagnostics
# -----------------------------------------------------------------------------


def blank_convergence_info(reason: str = "skipped") -> Dict[str, float | bool | str]:
    return {
        "delta_pf_skipped": True,
        "delta_pf_skip_reason": str(reason),
        "delta_pf": float("nan"),
        "delta_pf_sq": float("nan"),
        "delta_pf_n": 0,
        "delta_pf_steps": 0,
        "delta_pf_t_min": float("nan"),
        "delta_pf_t_max": float("nan"),
        "delta_pf_time_schedule": "none",
        "delta_pf_dt_min": float("nan"),
        "delta_pf_dt_max": float("nan"),
        "delta_pf_dt_sum": float("nan"),
        "delta_pf_integral_path_a": float("nan"),
        "delta_pf_integral_path_b": float("nan"),
        "delta_pf_endpoint_mmd": float("nan"),
        "delta_pf_endpoint_sw2": float("nan"),
        "delta_pf_endpoint_sliced_ks": float("nan"),
        "delta_pf_endpoint_mean_l2": float("nan"),
        "delta_pf_max_score_diff": float("nan"),
        "delta_pf_max_abs_state": float("nan"),
        "delta_pf_failed": False,
        "delta_pf_fail_reason": "",
        "delta_pf_target": float("nan"),
        "delta_pf_target_sq": float("nan"),
        "delta_pf_target_n": 0,
        "delta_pf_target_steps": 0,
        "delta_pf_target_t_min": float("nan"),
        "delta_pf_target_t_max": float("nan"),
        "delta_pf_target_time_schedule": "none",
        "delta_pf_target_dt_min": float("nan"),
        "delta_pf_target_dt_max": float("nan"),
        "delta_pf_target_dt_sum": float("nan"),
        "delta_pf_target_max_score_diff": float("nan"),
        "delta_pf_target_max_abs_state": float("nan"),
        "delta_pf_target_failed": False,
        "delta_pf_target_fail_reason": "",
    }


@torch.no_grad()
def _pf_path_integral_for_delta(
    path_bank: SNISScoreBank,
    path_method: str,
    other_bank: SNISScoreBank,
    other_method: str,
    z_base: torch.Tensor,
    cfg: Config,
    ts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float | bool | str]]:
    """Integrate one half of Delta_PF along q_{path_bank,t}^{PF} paths.

    The probability-flow ODE is dz_t/dt=-z_t-s(z_t,t).  We integrate it from
    t_max down to t_min on the same nonuniform physical-time grid used elsewhere.
    Scores are norm-clipped exactly as in the transport integrator, so the
    diagnostic measures the numerical frozen field actually queried by this
    script.
    """
    x = z_base.detach().clone()
    integral = torch.zeros((), device=x.device, dtype=x.dtype)
    max_score_diff = 0.0
    max_abs_state = safe_float(x.abs().max())
    failed = False
    fail_reason = ""

    for i in range(int(ts.numel()) - 1):
        tc = float(ts[i].item())
        tn = float(ts[i + 1].item())
        h = tn - tc
        dt_abs = abs(h)

        s_path = clamp_norm(path_bank.estimate(x, tc, path_method), cfg.score_clip)
        s_other = clamp_norm(other_bank.estimate(x, tc, other_method), cfg.score_clip)
        diff0 = torch.sum((s_path - s_other) ** 2, dim=1)
        v0 = -x - s_path
        x_e = x + h * v0
        if cfg.sample_clip and cfg.sample_clip > 0:
            x_e = torch.clamp(x_e, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))

        s_path_e = clamp_norm(path_bank.estimate(x_e, tn, path_method), cfg.score_clip)
        s_other_e = clamp_norm(other_bank.estimate(x_e, tn, other_method), cfg.score_clip)
        diff1 = torch.sum((s_path_e - s_other_e) ** 2, dim=1)
        v1 = -x_e - s_path_e

        finite = (
            torch.isfinite(x).all()
            and torch.isfinite(x_e).all()
            and torch.isfinite(s_path).all()
            and torch.isfinite(s_other).all()
            and torch.isfinite(s_path_e).all()
            and torch.isfinite(s_other_e).all()
        )
        if not bool(finite):
            failed = True
            fail_reason = f"nonfinite path state or score at step {i}"
            break

        integral = integral + 0.5 * dt_abs * (diff0.mean() + diff1.mean())
        max_score_diff = max(max_score_diff, safe_float(torch.sqrt(torch.clamp(torch.cat([diff0, diff1]).max(), min=0.0))))
        x = x + 0.5 * h * (v0 + v1)
        if cfg.sample_clip and cfg.sample_clip > 0:
            x = torch.clamp(x, min=-float(cfg.sample_clip), max=float(cfg.sample_clip))
        max_abs_state = max(max_abs_state, safe_float(x.abs().max()))

    return integral.detach(), x.detach(), {
        "failed": bool(failed),
        "fail_reason": fail_reason,
        "max_score_diff": float(max_score_diff),
        "max_abs_state": float(max_abs_state),
    }


@torch.no_grad()
def probability_flow_score_discrepancy(
    bank_a: SNISScoreBank,
    method_a: str,
    bank_b: SNISScoreBank,
    method_b: str,
    cfg: Config,
    generator: torch.Generator,
) -> Dict[str, float | bool | str]:
    """Monte Carlo estimate of Delta_PF(s_a,s_b) and endpoint movement.

    This is the finite-reference diagnostic corresponding to the theorem's
    adjacent-field condition.  It samples shared Gaussian base points, transports
    them deterministically under each frozen field, and averages the score-field
    discrepancy on both induced PF path laws.
    """
    if not bool(cfg.convergence_check):
        return blank_convergence_info("convergence_check_false")
    n = int(getattr(cfg, "delta_pf_n", 0))
    steps = int(getattr(cfg, "delta_pf_steps", 0))
    if n <= 0 or steps <= 0:
        return blank_convergence_info("delta_pf_n_or_steps_nonpositive")
    if str(method_a).lower() == "none" or str(method_b).lower() == "none":
        return blank_convergence_info("none_transport_method")

    ts = make_time_grid(cfg, steps, direction="reverse", device=bank_a.device, dtype=bank_a.dtype)
    ts_stats = time_grid_step_stats(ts)
    z = torch.randn((n, int(bank_a.d)), device=bank_a.device, dtype=bank_a.dtype, generator=generator)

    int_a, end_a, info_a = _pf_path_integral_for_delta(bank_a, method_a, bank_b, method_b, z, cfg, ts)
    int_b, end_b, info_b = _pf_path_integral_for_delta(bank_b, method_b, bank_a, method_a, z, cfg, ts)
    delta_sq = 0.5 * (int_a + int_b)
    delta = torch.sqrt(torch.clamp(delta_sq, min=0.0))
    endpoint_gen = make_generator(int(cfg.seed + 811_000 + steps + n), bank_a.device)
    max_n = min(int(getattr(cfg, "adjacent_metrics_max_n", cfg.metrics_max_n)), int(end_a.shape[0]), int(end_b.shape[0]))
    endpoint_mmd = mmd_rbf(end_a, end_b, max_n=max_n)
    endpoint_sw2 = sliced_w2(end_a, end_b, cfg.sw2_projections, endpoint_gen, max_n=max_n)
    endpoint_sks = sliced_ks(end_a, end_b, cfg.sw2_projections, endpoint_gen, max_n=max_n)
    mean_l2 = safe_float(torch.linalg.norm(end_a[:max_n].mean(dim=0) - end_b[:max_n].mean(dim=0))) if max_n > 0 else float("nan")
    failed = bool(info_a.get("failed", False)) or bool(info_b.get("failed", False))
    fail_reason = str(info_a.get("fail_reason", "") or info_b.get("fail_reason", ""))
    return {
        "delta_pf_skipped": False,
        "delta_pf_skip_reason": "",
        "delta_pf": safe_float(delta),
        "delta_pf_sq": safe_float(delta_sq),
        "delta_pf_n": int(n),
        "delta_pf_steps": int(steps),
        "delta_pf_t_min": float(effective_time_bounds(cfg)[0]),
        "delta_pf_t_max": float(effective_time_bounds(cfg)[1]),
        "delta_pf_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "delta_pf_dt_min": float(ts_stats["dt_min"]),
        "delta_pf_dt_max": float(ts_stats["dt_max"]),
        "delta_pf_dt_sum": float(ts_stats["dt_sum"]),
        "delta_pf_integral_path_a": safe_float(int_a),
        "delta_pf_integral_path_b": safe_float(int_b),
        "delta_pf_endpoint_mmd": endpoint_mmd,
        "delta_pf_endpoint_sw2": endpoint_sw2,
        "delta_pf_endpoint_sliced_ks": endpoint_sks,
        "delta_pf_endpoint_mean_l2": mean_l2,
        "delta_pf_max_score_diff": max(float(info_a.get("max_score_diff", float("nan"))), float(info_b.get("max_score_diff", float("nan")))),
        "delta_pf_max_abs_state": max(float(info_a.get("max_abs_state", float("nan"))), float(info_b.get("max_abs_state", float("nan")))),
        "delta_pf_failed": failed,
        "delta_pf_fail_reason": fail_reason,
    }


@torch.no_grad()
def target_marginal_score_discrepancy(
    target,
    bank_a: SNISScoreBank,
    method_a: str,
    bank_b: SNISScoreBank,
    method_b: str,
    cfg: Config,
    generator: torch.Generator,
) -> Dict[str, float | bool | str]:
    """Estimate Delta_PF on the true target OU marginals pi_t.

    This diagnostic keeps the adjacent frozen score estimators constructed from
    their respective reference clouds q_k and q_{k+1}, but changes only the
    evaluation law.  Instead of querying along the round-dependent PF path laws,
    it estimates

        int_{t_min}^{t_max} E_{x_t~pi_t} ||s_a(x_t,t)-s_b(x_t,t)||^2 dt,

    using a coupled OU draw x_t = alpha_t x_0 + sqrt(gamma_t) eps with
    x_0~pi and eps~N(0,I).  The square-root column delta_pf_target is reported
    alongside the squared integral delta_pf_target_sq.
    """
    if not bool(cfg.convergence_check):
        return {
            "delta_pf_target": float("nan"),
            "delta_pf_target_sq": float("nan"),
            "delta_pf_target_n": 0,
            "delta_pf_target_steps": 0,
            "delta_pf_target_t_min": float("nan"),
            "delta_pf_target_t_max": float("nan"),
            "delta_pf_target_time_schedule": "none",
            "delta_pf_target_dt_min": float("nan"),
            "delta_pf_target_dt_max": float("nan"),
            "delta_pf_target_dt_sum": float("nan"),
            "delta_pf_target_max_score_diff": float("nan"),
            "delta_pf_target_max_abs_state": float("nan"),
            "delta_pf_target_failed": False,
            "delta_pf_target_fail_reason": "convergence_check_false",
        }
    n = int(getattr(cfg, "delta_pf_target_n", 0))
    if n <= 0:
        n = int(getattr(cfg, "delta_pf_n", 0))
    steps = int(getattr(cfg, "delta_pf_steps", 0))
    if n <= 0 or steps <= 0:
        return {
            "delta_pf_target": float("nan"),
            "delta_pf_target_sq": float("nan"),
            "delta_pf_target_n": 0,
            "delta_pf_target_steps": 0,
            "delta_pf_target_t_min": float("nan"),
            "delta_pf_target_t_max": float("nan"),
            "delta_pf_target_time_schedule": "none",
            "delta_pf_target_dt_min": float("nan"),
            "delta_pf_target_dt_max": float("nan"),
            "delta_pf_target_dt_sum": float("nan"),
            "delta_pf_target_max_score_diff": float("nan"),
            "delta_pf_target_max_abs_state": float("nan"),
            "delta_pf_target_failed": False,
            "delta_pf_target_fail_reason": "delta_pf_target_n_or_steps_nonpositive",
        }
    if str(method_a).lower() == "none" or str(method_b).lower() == "none":
        return {
            "delta_pf_target": float("nan"),
            "delta_pf_target_sq": float("nan"),
            "delta_pf_target_n": 0,
            "delta_pf_target_steps": 0,
            "delta_pf_target_t_min": float("nan"),
            "delta_pf_target_t_max": float("nan"),
            "delta_pf_target_time_schedule": "none",
            "delta_pf_target_dt_min": float("nan"),
            "delta_pf_target_dt_max": float("nan"),
            "delta_pf_target_dt_sum": float("nan"),
            "delta_pf_target_max_score_diff": float("nan"),
            "delta_pf_target_max_abs_state": float("nan"),
            "delta_pf_target_failed": False,
            "delta_pf_target_fail_reason": "none_transport_method",
        }

    ts = make_time_grid(cfg, steps, direction="forward", device=bank_a.device, dtype=bank_a.dtype)
    ts_stats = time_grid_step_stats(ts)
    x0 = target.sample(n, generator=generator).detach()
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)

    vals: List[torch.Tensor] = []
    max_score_diff = 0.0
    max_abs_state = 0.0
    failed = False
    fail_reason = ""
    for i in range(int(ts.numel())):
        t_val = float(ts[i].item())
        alpha, gamma = alpha_gamma(t_val, device=x0.device, dtype=x0.dtype)
        xt = alpha * x0 + torch.sqrt(torch.clamp(gamma, min=0.0)) * eps
        s_a = clamp_norm(bank_a.estimate(xt, t_val, method_a), cfg.score_clip)
        s_b = clamp_norm(bank_b.estimate(xt, t_val, method_b), cfg.score_clip)
        diff_sq = torch.sum((s_a - s_b) ** 2, dim=1)
        finite = torch.isfinite(xt).all() and torch.isfinite(s_a).all() and torch.isfinite(s_b).all() and torch.isfinite(diff_sq).all()
        if not bool(finite):
            failed = True
            fail_reason = f"nonfinite target-marginal state or score at step {i}"
            break
        vals.append(diff_sq.mean())
        max_score_diff = max(max_score_diff, safe_float(torch.sqrt(torch.clamp(diff_sq.max(), min=0.0))))
        max_abs_state = max(max_abs_state, safe_float(xt.abs().max()))

    if failed or len(vals) < 2:
        delta_sq = torch.tensor(float("nan"), device=bank_a.device, dtype=bank_a.dtype)
    else:
        delta_sq = torch.zeros((), device=bank_a.device, dtype=bank_a.dtype)
        for i in range(len(vals) - 1):
            dt_abs = abs(float(ts[i + 1].item() - ts[i].item()))
            delta_sq = delta_sq + 0.5 * dt_abs * (vals[i] + vals[i + 1])
    delta = torch.sqrt(torch.clamp(delta_sq, min=0.0)) if torch.isfinite(delta_sq) else delta_sq
    return {
        "delta_pf_target": safe_float(delta),
        "delta_pf_target_sq": safe_float(delta_sq),
        "delta_pf_target_n": int(n),
        "delta_pf_target_steps": int(steps),
        "delta_pf_target_t_min": float(effective_time_bounds(cfg)[0]),
        "delta_pf_target_t_max": float(effective_time_bounds(cfg)[1]),
        "delta_pf_target_time_schedule": canonical_time_schedule(cfg.time_schedule),
        "delta_pf_target_dt_min": float(ts_stats["dt_min"]),
        "delta_pf_target_dt_max": float(ts_stats["dt_max"]),
        "delta_pf_target_dt_sum": float(ts_stats["dt_sum"]),
        "delta_pf_target_max_score_diff": float(max_score_diff),
        "delta_pf_target_max_abs_state": float(max_abs_state),
        "delta_pf_target_failed": bool(failed),
        "delta_pf_target_fail_reason": fail_reason,
    }


@torch.no_grad()
def adjacent_sample_discrepancy(
    samples: torch.Tensor,
    previous: Optional[torch.Tensor],
    cfg: Config,
    generator: torch.Generator,
) -> Dict[str, float]:
    """Distributional movement between adjacent unweighted proposal clouds."""
    if previous is None or samples.numel() == 0 or previous.numel() == 0:
        return {
            "adjacent_sample_n": 0.0,
            "adjacent_sample_mmd": float("nan"),
            "adjacent_sample_sw2": float("nan"),
            "adjacent_sample_sliced_ks": float("nan"),
            "adjacent_sample_mean_l2": float("nan"),
            "adjacent_sample_cov_frob": float("nan"),
        }
    n = min(int(getattr(cfg, "adjacent_metrics_max_n", cfg.metrics_max_n)), int(samples.shape[0]), int(previous.shape[0]))
    x = samples[:n]
    y = previous[:n]
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    X = x - mx
    Y = y - my
    Cx = (X.T @ X) / max(n - 1, 1)
    Cy = (Y.T @ Y) / max(n - 1, 1)
    return {
        "adjacent_sample_n": float(n),
        "adjacent_sample_mmd": mmd_rbf(x, y, max_n=n),
        "adjacent_sample_sw2": sliced_w2(x, y, cfg.sw2_projections, generator, max_n=n),
        "adjacent_sample_sliced_ks": sliced_ks(x, y, cfg.sw2_projections, generator, max_n=n),
        "adjacent_sample_mean_l2": safe_float(torch.linalg.norm(mx - my)),
        "adjacent_sample_cov_frob": safe_float(torch.linalg.matrix_norm(Cx - Cy, ord="fro")),
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
def target_nll_metric(target, samples: torch.Tensor) -> float:
    """Average target negative log density at generated samples.

    This is the old iterative_lfgi.py NLL diagnostic.  The benchmark_sweep.py
    NLL is KDE-based and is added below as nll_kde_metric.
    """
    return safe_float((-target.log_prob(samples, t=0.0)).mean())


def nll_kde_metric(samples: torch.Tensor, test_points: torch.Tensor, n_fit: int = 5000, min_bandwidth: float = 0.05, seed: Optional[int] = None) -> float:
    """Benchmark-sweep-compatible KDE NLL.

    Fits a Gaussian KDE to generated samples and evaluates -E_{test}[log q_hat].
    This mirrors benchmark_sweep.py::nll_kde, with optional deterministic
    subsampling and configurable bandwidth floor.
    """
    try:
        from sklearn.neighbors import KernelDensity
    except Exception:
        return float("nan")
    if samples.dim() > 2:
        samples = samples.reshape(samples.shape[0], -1)
    if test_points.dim() > 2:
        test_points = test_points.reshape(test_points.shape[0], -1)
    if int(samples.shape[0]) <= 1 or int(test_points.shape[0]) <= 0:
        return float("nan")
    sc_np = samples.detach().cpu().double().numpy()
    te_np = test_points.detach().cpu().double().numpy()
    n, d = len(sc_np), sc_np.shape[1]
    n_fit = int(n_fit) if n_fit is not None else n
    if n_fit > 0 and n_fit < n:
        if seed is None:
            idx = np.random.choice(n, n_fit, replace=False)
        else:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(n, n_fit, replace=False)
        sc_np = sc_np[idx]
        n = len(sc_np)
    bw = max(n ** (-1.0 / (d + 4)), float(min_bandwidth))
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(sc_np)
    return float(-kde.score_samples(te_np).mean())


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
def _ks_1d_sorted(x, y) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n, m = len(x), len(y)
    vals = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, vals, side="right") / max(n, 1)
    cdf_y = np.searchsorted(y, vals, side="right") / max(m, 1)
    return float(np.max(np.abs(cdf_x - cdf_y))) if vals.size else float("nan")


@torch.no_grad()
def sliced_ks_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_projections: int = 1000,
    max_points: Optional[int] = None,
    seed: int = 0,
    reduce: str = "mean",
    generator: Optional[torch.Generator] = None,
) -> float:
    """Sliced Kolmogorov--Smirnov distance from benchmark_sweep.py.

    The benchmark version uses random one-dimensional projections and averages
    the 1D KS distance.  This port keeps the same direction seeding and adds an
    optional torch generator for deterministic subsampling inside this script.
    """
    if X.dim() > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2:
        Y = Y.reshape(Y.shape[0], -1)
    if max_points is None:
        max_points = int(os.environ.get("LFGI_BENCH_METRIC_MAX", "4096"))
    n = min(int(max_points), int(X.shape[0]), int(Y.shape[0]))
    if n <= 1:
        return float("inf")
    if generator is None:
        X = X[torch.randperm(X.shape[0], device=X.device)[:n]]
        Y = Y[torch.randperm(Y.shape[0], device=Y.device)[:n]]
    else:
        X = X[torch.randperm(X.shape[0], device=X.device, generator=generator)[:n]]
        Y = Y[torch.randperm(Y.shape[0], device=Y.device, generator=generator)[:n]]
    d = X.shape[1]
    dir_gen = torch.Generator(device=X.device)
    dir_gen.manual_seed(int(seed))
    dirs = torch.randn(int(n_projections), d, generator=dir_gen, dtype=X.dtype, device=X.device)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp_min(1.0e-30)
    Xp = (X @ dirs.T).detach().cpu().numpy()
    Yp = (Y @ dirs.T).detach().cpu().numpy()
    vals = np.array([_ks_1d_sorted(Xp[:, j], Yp[:, j]) for j in range(dirs.shape[0])], dtype=np.float64)
    if reduce == "max":
        return float(np.max(vals))
    if reduce == "median":
        return float(np.median(vals))
    return float(np.mean(vals))


@torch.no_grad()
def sliced_ks(x: torch.Tensor, y: torch.Tensor, n_proj: int, generator: torch.Generator, max_n: int = 2000) -> float:
    """Backward-compatible wrapper used by adjacent/convergence diagnostics."""
    return sliced_ks_distance(x, y, n_projections=int(n_proj), max_points=int(max_n), seed=0, generator=generator)


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
        return {
            "metric_n": n,
            "nll": float("nan"),
            "kde_nll": float("nan"),
            "target_nll": float("nan"),
            "mmd": float("nan"),
            "ksd": float("nan"),
            "sw2": float("nan"),
            "sliced_ks": float("nan"),
            "mode_l1": float("nan"),
            "mean_norm": float("nan"),
            "cov_frob_err": float("nan"),
            "fisher_rmse": float("nan"),
        }
    mean_norm, cov_err = moment_errors(x)
    kde_nll = nll_kde_metric(
        x,
        y,
        n_fit=min(int(cfg.nll_kde_n_fit), int(n)),
        min_bandwidth=float(cfg.nll_kde_min_bandwidth),
        seed=int(cfg.seed + 9_001),
    )
    out = {
        "metric_n": float(n),
        # Match benchmark_sweep.py: nll is KDE-based sample-quality NLL.
        "nll": kde_nll,
        "kde_nll": kde_nll,
        # Preserve the previous iterative_lfgi.py target-energy diagnostic.
        "target_nll": target_nll_metric(target, x),
        "mmd": mmd_rbf(x, y, max_n=n),
        "ksd": ksd_rbf(target, x, max_n=min(n, 1000)),
        "sw2": sliced_w2(x, y, cfg.sw2_projections, generator=generator, max_n=n),
        "sliced_ks": sliced_ks_distance(
            x,
            y,
            n_projections=int(getattr(target, "metric_n_projections", cfg.ks_projections)),
            max_points=n,
            seed=0,
            generator=generator,
        ),
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


def _is_funnel_target(target) -> bool:
    return isinstance(target, NealFunnelTarget) or str(getattr(target, "name", "")).startswith("funnel_d")


def _heatmap_from_points(
    pts2: np.ndarray,
    lims: Tuple[float, float, float, float],
    cfg: Config,
    *,
    bins: Optional[int] = None,
    density: bool = False,
) -> np.ndarray:
    bins = int(cfg.hist_bins if bins is None else bins)
    H, _xe, _ye = np.histogram2d(
        pts2[:, 0],
        pts2[:, 1],
        bins=bins,
        range=[[lims[0], lims[1]], [lims[2], lims[3]]],
        density=bool(density),
    )
    H = H.T.astype(np.float64)
    if not density and H.sum() > 0:
        H = H / H.sum()
    H[~np.isfinite(H)] = 0.0
    return H


def plot_heatmap_panel(
    ax,
    pts2: np.ndarray,
    title: str,
    lims: Tuple[float, float, float, float],
    cfg: Config,
    vmax: Optional[float] = None,
    *,
    bins: Optional[int] = None,
    density: bool = False,
    gamma: Optional[float] = None,
    aspect: str = "auto",
    interpolation: str = "nearest",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_ticks: bool = False,
):
    H = _heatmap_from_points(pts2, lims, cfg, bins=bins, density=density)
    if vmax is None:
        vals = H[H > 0]
        vmax = float(np.quantile(vals, float(cfg.hist_vmax_quantile))) if vals.size else 1.0
    gamma = float(cfg.hist_gamma if gamma is None else gamma)
    ax.imshow(
        H,
        origin="lower",
        extent=lims,
        aspect=aspect,
        interpolation=interpolation,
        norm=PowerNorm(gamma=gamma, vmin=0.0, vmax=max(float(vmax), 1.0e-12)),
    )
    ax.set_title(title, fontsize=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(axis="both", labelsize=9, length=2.5, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    return vmax


def make_projection_limits(arrays: List[np.ndarray], pad: float = 0.08) -> Tuple[float, float, float, float]:
    Z = np.concatenate(arrays, axis=0)
    lo = np.quantile(Z, 0.005, axis=0)
    hi = np.quantile(Z, 0.995, axis=0)
    span = np.maximum(hi - lo, 1.0)
    lo = lo - pad * span
    hi = hi + pad * span
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def make_funnel_projection_limits(arrays: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """Benchmark-sweep robust plotting limits for Neal funnel panels."""
    Z = np.concatenate(arrays, axis=0)
    x_lo, x_hi = np.percentile(Z[:, 0], [FUNNEL_HEATMAP_X_Q_LOW, FUNNEL_HEATMAP_X_Q_HIGH])
    y_abs = float(np.percentile(np.abs(Z[:, 1]), FUNNEL_HEATMAP_Y_Q_ABS))
    x_pad = 0.06 * max(float(x_hi - x_lo), 1.0e-9)
    y_pad = 0.06 * max(2.0 * y_abs, 1.0e-9)
    return float(x_lo - x_pad), float(x_hi + x_pad), float(-y_abs - y_pad), float(y_abs + y_pad)


def shared_heatmap_vmax(
    arrays: List[np.ndarray],
    lims: Tuple[float, float, float, float],
    cfg: Config,
    *,
    bins: Optional[int] = None,
    density: bool = False,
    q_percent: Optional[float] = None,
) -> float:
    positives: List[np.ndarray] = []
    for arr in arrays:
        if arr is None or len(arr) < 1:
            continue
        H = _heatmap_from_points(arr, lims, cfg, bins=bins, density=density)
        z = H[np.isfinite(H) & (H > 0.0)]
        if z.size:
            positives.append(z)
    if not positives:
        return 1.0
    vals = np.concatenate(positives)
    if q_percent is None:
        # Generic path uses cfg.hist_vmax_quantile as a [0,1] quantile.
        vmax = float(np.quantile(vals, float(cfg.hist_vmax_quantile)))
    else:
        # Benchmark-sweep funnel path uses a percentile, default 98.5.
        vmax = float(np.percentile(vals, float(q_percent)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(vals))
    return max(float(vmax), 1.0e-12)


def save_heatmaps(outdir: str, target, truth: torch.Tensor, init_refs: torch.Tensor, samples_by_family_round: Dict[str, List[torch.Tensor]], cfg: Config):
    is_funnel = _is_funnel_target(target)

    if is_funnel and hasattr(target, "plot_projection"):
        def to_plot_np(x: torch.Tensor) -> np.ndarray:
            return target.plot_projection(x, fit_ref=truth).detach().cpu().double().numpy()
        # The saved file keeps the same name as before but now records that this
        # is the native benchmark-sweep funnel projection rather than PCA.
        np.savez(
            os.path.join(outdir, "projection_basis.npz"),
            projection="funnel_native_x1_x2",
            normalized_target=bool(getattr(target, "normalized", False)),
            scale=as_numpy(getattr(target, "scale", torch.ones(int(target.d), device=target.device, dtype=target.dtype))),
        )
        projection_title = f"Funnel coordinates — Neal Funnel ($d={int(getattr(target, 'd', truth.shape[1]))}$)"
        axis_labels = (r"$x_1$", r"$x_2$")
        panel_aspect = "auto"
        heatmap_bins = 100
        heatmap_density = True
        heatmap_gamma = min(max(float(os.environ.get("LFGI_BENCH_HEATMAP_GAMMA", "0.42")), 0.05), 1.0)
        heatmap_vmax_q = float(os.environ.get("LFGI_BENCH_HEATMAP_VMAX_Q", "98.5"))
        heatmap_interp = os.environ.get("LFGI_BENCH_HEATMAP_INTERP", "nearest").strip() or "nearest"
    else:
        mean, basis = fit_pca_projection(truth)
        np.savez(os.path.join(outdir, "projection_basis.npz"), mean=mean, basis=basis, projection="pca")
        def to_plot_np(x: torch.Tensor) -> np.ndarray:
            return project_np(x, mean, basis)
        projection_title = f"PCA heatmaps: {getattr(target, 'name', getattr(target, '__class__', type(target)).__name__)}"
        axis_labels = (None, None)
        panel_aspect = "auto"
        heatmap_bins = None
        heatmap_density = False
        heatmap_gamma = float(cfg.hist_gamma)
        heatmap_vmax_q = None
        heatmap_interp = "nearest"

    truth2 = to_plot_np(truth)
    init2 = to_plot_np(init_refs)
    family_order = list(samples_by_family_round.keys())

    all_arrays = [truth2, init2]
    final_arrays = [truth2, init2]
    for fam in family_order:
        arrs = samples_by_family_round.get(fam, [])
        for x in arrs:
            all_arrays.append(to_plot_np(x))
        if arrs:
            final_arrays.append(to_plot_np(arrs[-1]))

    # Use target truth only to determine the plotted window and color normalization.
    # This prevents a single diverged variant from expanding the visible extent and
    # shrinking the target into a tiny star in the corner.
    lims = make_funnel_projection_limits([truth2]) if is_funnel else make_projection_limits([truth2])
    vmax = shared_heatmap_vmax(
        [truth2],
        lims,
        cfg,
        bins=heatmap_bins,
        density=heatmap_density,
        q_percent=heatmap_vmax_q,
    )

    init_label = "Initial target refs" if canonical_initial_reference_mode(cfg.initial_reference_mode) == "target" else "Initial N(0,I) prior refs"
    panels = [("Target truth (eval only)", truth2), (init_label, init2)]
    for fam in family_order:
        if samples_by_family_round.get(fam):
            panels.append((f"{fam} alternating DRC round {len(samples_by_family_round[fam])}", to_plot_np(samples_by_family_round[fam][-1])))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ci, (ax, (title, pts)) in enumerate(zip(axes, panels)):
        plot_heatmap_panel(
            ax,
            pts,
            title,
            lims,
            cfg,
            vmax=vmax,
            bins=heatmap_bins,
            density=heatmap_density,
            gamma=heatmap_gamma,
            aspect=panel_aspect,
            interpolation=heatmap_interp,
            xlabel=axis_labels[0] if is_funnel else None,
            ylabel=axis_labels[1] if (is_funnel and ci == 0) else None,
            show_ticks=bool(is_funnel),
        )
        if is_funnel and ci > 0:
            ax.set_yticklabels([])
    fig.suptitle(projection_title, fontsize=12 if not is_funnel else 15.5)
    fig.savefig(os.path.join(outdir, "heatmaps_final.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    ncols = int(cfg.n_rounds) + 2
    nrows = max(1, len(family_order))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(nrows, ncols)
    for row, fam in enumerate(family_order):
        plot_heatmap_panel(
            axes[row, 0], truth2, "Target", lims, cfg, vmax=vmax,
            bins=heatmap_bins, density=heatmap_density, gamma=heatmap_gamma,
            aspect=panel_aspect, interpolation=heatmap_interp,
            xlabel=axis_labels[0] if is_funnel else None,
            ylabel=axis_labels[1] if is_funnel else None,
            show_ticks=bool(is_funnel),
        )
        plot_heatmap_panel(
            axes[row, 1], init2, f"{fam}: {init_label}", lims, cfg, vmax=vmax,
            bins=heatmap_bins, density=heatmap_density, gamma=heatmap_gamma,
            aspect=panel_aspect, interpolation=heatmap_interp,
            xlabel=axis_labels[0] if is_funnel else None,
            ylabel=None,
            show_ticks=bool(is_funnel),
        )
        if is_funnel:
            axes[row, 1].set_yticklabels([])
        arrs = samples_by_family_round.get(fam, [])
        for j in range(int(cfg.n_rounds)):
            ax = axes[row, j + 2]
            if j < len(arrs):
                plot_heatmap_panel(
                    ax, to_plot_np(arrs[j]), f"{fam}: round {j+1}", lims, cfg, vmax=vmax,
                    bins=heatmap_bins, density=heatmap_density, gamma=heatmap_gamma,
                    aspect=panel_aspect, interpolation=heatmap_interp,
                    xlabel=axis_labels[0] if is_funnel else None,
                    ylabel=None,
                    show_ticks=bool(is_funnel),
                )
                if is_funnel:
                    ax.set_yticklabels([])
            else:
                ax.axis("off")
    fig.suptitle("Alternating DRC progression by estimator family", fontsize=12)
    fig.savefig(os.path.join(outdir, "heatmaps_by_round.png"), dpi=220, bbox_inches="tight")
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
    metrics = ["mmd", "ksd", "sw2", "sliced_ks", "nll", "target_nll", "mode_l1", "fisher_rmse"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
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



def combine_convergence_rows(metric_rows: List[Dict[str, object]], stage_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Merge sample-level and PF-field convergence diagnostics by family/round."""
    stage_lookup = {}
    for row in stage_rows:
        key = (row.get("family"), row.get("method"), row.get("round"))
        stage_lookup[key] = row
    keys_from_stage = [
        "delta_pf",
        "delta_pf_sq",
        "delta_pf_n",
        "delta_pf_steps",
        "delta_pf_endpoint_mmd",
        "delta_pf_endpoint_sw2",
        "delta_pf_endpoint_sliced_ks",
        "delta_pf_endpoint_mean_l2",
        "delta_pf_max_score_diff",
        "delta_pf_max_abs_state",
        "delta_pf_failed",
        "delta_pf_target",
        "delta_pf_target_sq",
        "delta_pf_target_n",
        "delta_pf_target_steps",
        "delta_pf_target_t_min",
        "delta_pf_target_t_max",
        "delta_pf_target_time_schedule",
        "delta_pf_target_dt_min",
        "delta_pf_target_dt_max",
        "delta_pf_target_dt_sum",
        "delta_pf_target_max_score_diff",
        "delta_pf_target_max_abs_state",
        "delta_pf_target_failed",
        "delta_pf_target_fail_reason",
        "mode_l1_unweighted_next_refs",
        "mode_l1_weighted_next_refs",
        "rho_ess_frac",
        "pf_skipped",
        "pf_skip_reason",
    ]
    out: List[Dict[str, object]] = []
    for row in metric_rows:
        if row.get("kind") != "sample":
            continue
        key = (row.get("family"), row.get("method"), row.get("round"))
        st = stage_lookup.get(key, {})
        merged = dict(row)
        for k in keys_from_stage:
            if k in st:
                merged[k] = st[k]
        out.append(merged)
    return out


def save_convergence_curves(outdir: str, convergence_rows: List[Dict[str, object]]):
    try:
        import pandas as pd
    except Exception:
        return
    if not convergence_rows:
        return
    df = pd.DataFrame(convergence_rows)
    if df.empty:
        return
    metrics = [
        ("delta_pf", r"$\Delta_{PF}$ on induced PF paths ↓"),
        ("delta_pf_target", r"$\Delta_{PF}$ on true $\pi_t$ ↓"),
        ("delta_pf_endpoint_mmd", r"PF endpoint MMD ↓"),
        ("delta_pf_endpoint_sw2", r"PF endpoint SW2 ↓"),
        ("adjacent_sample_mmd", "sample q_k MMD ↓"),
        ("adjacent_sample_sw2", r"sample SW2 ↓"),
        ("fisher_rmse", r"score RMSE to target ↓"),
        ("mmd", r"MMD to target ↓"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    family_order = list(dict.fromkeys(df["family"].tolist()))
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df.columns:
            ax.axis("off")
            continue
        for fam in family_order:
            sub = df[df["family"] == fam].sort_values("round")
            ax.plot(sub["round"], sub[metric], marker="o", label=fam)
        ax.set_title(title)
        ax.set_xlabel("iteration round")
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("No-correction convergence diagnostics", fontsize=12)
    fig.savefig(os.path.join(outdir, "convergence_curves.png"), dpi=220)
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
        "pf_divergence_effective": "none",
        "pf_skipped": True,
        "pf_skip_reason": reason,
        "pf_failed_frac": 0.0,
        "pf_steps": 0,
        "pf_t_min": float("nan"),
        "pf_t_max": float("nan"),
        "pf_time_schedule": "none",
        "pf_dt_min": float("nan"),
        "pf_dt_max": float("nan"),
        "pf_dt_sum": float("nan"),
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


def canonical_ratio_reference_mode(value: str) -> str:
    """Normalize the frozen-bank convention used to reconstruct endpoint log q.

    ``endpoint`` builds the ratio-score estimator on the final transport
    endpoint bank.  ``generator`` uses the bank that generated that endpoint.
    This matches iterative_lfgi.py.  ``incoming`` is accepted as a backward-
    compatible alias for ``generator`` but no Liouville certificate is carried.
    """
    key = str(value or "endpoint").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "endpoint": "endpoint",
        "settled": "endpoint",
        "final": "endpoint",
        "fixed-point": "endpoint",
        "fixedpoint": "endpoint",
        "tsc": "endpoint",
        "old": "endpoint",
        "legacy": "endpoint",
        "generator": "generator",
        "generating": "generator",
        "source": "generator",
        "pre-endpoint": "generator",
        "exact-induced": "generator",
        "incoming": "generator",
        "certificate": "generator",
        "incoming-certificate": "generator",
    }
    if key not in aliases:
        raise ValueError(f"Unknown ratio_reference_mode={value!r}; use endpoint or generator")
    return aliases[key]


def canonical_transport_dynamics(value: str, *, correction_enabled: bool) -> str:
    """Normalize the retained CLI flag; transport is always reverse SDE.

    Probability-flow transport and its generation-time density certificate were
    removed to restore the runtime profile of iterative_lfgi.py.  Endpoint log q
    is reconstructed afterward by ``pf_logprob_bank``.
    """
    key = str(value or "reverse_sde").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "auto": "reverse-sde",
        "reverse-sde": "reverse-sde",
        "sde": "reverse-sde",
        "stochastic": "reverse-sde",
    }
    if key in {"pf", "pflow", "probability-flow", "probabilityflow", "ode"}:
        raise ValueError(
            "Probability-flow transport has been removed from this variant. "
            "Use reverse-SDE transport; endpoint likelihood factors are reconstructed afterward."
        )
    if key not in aliases:
        raise ValueError(f"Unknown transport_dynamics={value!r}; use auto or reverse_sde")
    return aliases[key]


def canonical_ratio_gate_policy(value: str) -> str:
    key = str(value or "native").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "native": "native",
        "literal": "native",
        "raw": "native",
        "project": "project",
        "projected": "project",
        "admissible": "project",
        "soft-channel": "project",
    }
    if key not in aliases:
        raise ValueError(f"Unknown ratio_gate_policy={value!r}; use native or project")
    return aliases[key]


def canonical_ratio_tilt_policy(value: str) -> str:
    """Normalize finite-bank endpoint-label handling for the ratio flow."""
    key = str(value or "exact").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "exact": "exact",
        "raw": "exact",
        "untempered": "exact",
        "certificate": "exact",
        "stabilized": "stabilized",
        "stabilised": "stabilized",
        "guarded": "stabilized",
        "tempered": "stabilized",
        "legacy": "stabilized",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown ratio_tilt_policy={value!r}; use exact or stabilized"
        )
    return aliases[key]


def prepare_ratio_flow_tilt(
    raw_rho: torch.Tensor, cfg: Config
) -> Tuple[torch.Tensor, Dict[str, float | bool | str]]:
    """Prepare ratio-flow endpoint labels without silently changing the algorithm.

    ``exact`` keeps the certificate-consistent labels log pi-log q up to the
    irrelevant common additive constant.  ``stabilized`` applies the legacy
    beta/clipping/ESS guard and is therefore a deliberately tempered algorithm.
    """
    policy = canonical_ratio_tilt_policy(getattr(cfg, "ratio_tilt_policy", "exact"))
    if policy == "stabilized":
        rho, info = finalize_density_ratio_weights(raw_rho, cfg)
        return rho, {"ratio_tilt_policy": policy, **info}

    raw = raw_rho.detach()
    if not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError(
            "ratio_tilt_policy=exact requires finite log pi-log q labels; "
            "use the stabilized policy only if deliberate regularization is intended"
        )
    rho = raw - (torch.logsumexp(raw, dim=0) - math.log(max(int(raw.numel()), 1)))
    ess, ess_frac = log_weight_ess(rho)
    return rho, {
        "ratio_tilt_policy": policy,
        "rho_beta_target": 1.0,
        "rho_beta_eff": 1.0,
        "rho_adapted_for_ess": False,
        "rho_clip_effective": False,
        "rho_ess": float(ess),
        "rho_ess_frac": float(ess_frac),
        "rho_mean": safe_float(rho.mean()),
        "rho_std": safe_float(rho.std(unbiased=False)),
        "rho_min": safe_float(rho.min()),
        "rho_max": safe_float(rho.max()),
        "rho_raw_mean": safe_float(raw.mean()),
        "rho_raw_std": safe_float(raw.std(unbiased=False)),
        "rho_raw_min": safe_float(raw.min()),
        "rho_raw_max": safe_float(raw.max()),
    }


def canonical_ratio_method(value: str) -> str:
    key = str(value or "raw-w").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "raw-w": "raw-w",
        "raw": "raw-w",
        "weights": "raw-w",
        "reweight": "raw-w",
        "reweighting": "raw-w",
        "gated-pflow": "gated-pflow",
        "gated-pf": "gated-pflow",
        "iglfgi": "gated-pflow",
        "complement-pflow": "gated-pflow",
        "gated-bflow": "gated-bflow",
        "gated-bf": "gated-bflow",
        "bflow": "gated-bflow",
        "complement-bflow": "gated-bflow",
        "shared-bflow": "gated-bflow",
        "raw-bflow": "raw-bflow",
        "raw-bf": "raw-bflow",
        "ungated-bflow": "raw-bflow",
        "ungated-bf": "raw-bflow",
        "full-bflow": "raw-bflow",
        "doob-bflow": "raw-bflow",
        "completed-bflow": "completed-bflow",
        "completed-bf": "completed-bflow",
        "complete-bflow": "completed-bflow",
        "full-lfgi-bflow": "completed-bflow",
        "tilted-lfgi-bflow": "completed-bflow",
        "shared-full-bflow": "completed-bflow",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown ratio method {value!r}; use raw-w, gated-pflow, "
            "gated-bflow, raw-bflow, or completed-bflow"
        )
    return aliases[key]


def is_moved_ratio_flow(value: str) -> bool:
    return canonical_ratio_method(value) in {"gated-pflow", "gated-bflow", "raw-bflow", "completed-bflow"}


def method_label(
    transport_method: str,
    transport_repeats: int,
    pf_method: str,
    ratio_method: str,
    ratio_rounds: int,
) -> str:
    """Stable method label for CSV joins and plot titles."""
    n = int(transport_repeats)
    prefix = str(transport_method) if n == 1 else f"{transport_method}-{n}"
    ratio = canonical_ratio_method(ratio_method)
    if is_moved_ratio_flow(ratio):
        return f"{prefix}_{pf_method}_{ratio}-{int(ratio_rounds)}"
    return f"{prefix}_{pf_method}_{ratio}"


def score_bank_values_from_pool(values: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Select the entries corresponding to the score-anchor slice."""
    mode = canonical_bank_coupling(cfg.bank_coupling)
    start = effective_gate_n(cfg) if mode == "independent" else 0
    stop = start + int(cfg.n_ref)
    if int(values.shape[0]) < stop:
        raise ValueError(
            f"Pool value vector has length {values.shape[0]} but score slice [{start}:{stop}] is required"
        )
    return values[start:stop].contiguous()


@torch.no_grad()
def run_gated_pflow_ratio_node(
    *,
    starting_pool: torch.Tensor,
    starting_logq: torch.Tensor,
    pf_method: str,
    ratio_rounds: int,
    target,
    truth: torch.Tensor,
    cfg: Config,
    outer_round: int,
    pi_gate_refs: Optional[torch.Tensor],
    pi_gate_rho: Optional[torch.Tensor],
    n_gate_refs: Optional[torch.Tensor],
    n_gate_rho: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, object],
    Dict[str, object],
    Dict[str, object],
    Dict[str, object],
]:
    """Apply one or more refreshed legacy complement p-flow ratio transports.

    Each inner round rebuilds the selected Method-2 carrier and native gate on
    the current unweighted particle bank, forms the density-tilted target score
    ``b_pi``, and integrates
    ``s_method + lambda(I-G_method)(b_pi-s_method)``.  The moved endpoint and
    its incoming certificate define the next inner round, so repeated p-flow
    rounds are unfrozen at the iteration level.
    """
    n_rounds = int(ratio_rounds)
    if n_rounds < 1:
        raise ValueError(f"gated-pflow ratio rounds must be >=1; got {ratio_rounds}")
    if canonical_score_method_key(pf_method) == "none":
        raise ValueError("gated-pflow requires a non-none ratio/PF score estimator with a defined gate")

    pool_n = proposal_pool_size(cfg)
    current_pool = starting_pool[:pool_n].detach().clone()
    current_logq = starting_logq[:pool_n].detach().clone()
    final_eval = current_pool
    last_logpi = target.log_prob(current_pool, t=0.0).detach()
    last_raw_rho = last_logpi - current_logq
    last_rho, last_rho_info = prepare_ratio_flow_tilt(last_raw_rho, cfg)
    last_pf_info: Dict[str, object] = {}
    inner_infos: List[Dict[str, object]] = []
    total_start = time.time()

    for inner in range(1, n_rounds + 1):
        logpi = target.log_prob(current_pool, t=0.0).detach()
        raw_rho = logpi - current_logq
        effective_tilt, rho_info = prepare_ratio_flow_tilt(raw_rho, cfg)

        zero = torch.zeros((current_pool.shape[0],), device=current_pool.device, dtype=current_pool.dtype)
        score_refs, score_rho0, gate_refs, gate_rho0, split_info = split_score_gate_banks(
            current_pool, zero, cfg
        )
        carrier_bank = SNISScoreBank(
            target,
            score_refs,
            cfg,
            log_ref_weights=score_rho0,
            gate_anchors=gate_refs,
            gate_log_ref_weights=gate_rho0,
            pi_gate_anchors=pi_gate_refs,
            pi_gate_log_ref_weights=pi_gate_rho,
            n_gate_anchors=n_gate_refs,
            n_gate_log_ref_weights=n_gate_rho,
        )
        score_tilt = score_bank_values_from_pool(effective_tilt, cfg)
        field = GatedPFlowRatioField(
            carrier_bank,
            pf_method,
            score_tilt,
            complement_strength=float(cfg.lambda_guard),
        )

        # Only the final inner round needs the full evaluation cloud.  Earlier
        # rounds generate exactly the bank required to define the next frozen
        # carrier, avoiding a large amount of discarded work.
        generate_n = max(int(cfg.n_samples), int(pool_n)) if inner == n_rounds else int(pool_n)
        flow_gen = make_generator(
            int(cfg.seed + 880_000 + 10_000 * int(outer_round) + 101 * inner),
            target.device,
        )
        moved_all, moved_logq_all, flow_info = reverse_ou_heun_probability_flow_certified(
            target,
            lambda x, t, field=field: field.estimate_and_divergence(x, t, cfg),
            cfg,
            generator=flow_gen,
            n_samples=generate_n,
            steps=int(cfg.pf_steps),
            phase_name=f"ratio:{pf_method}:inner{inner}",
        )
        moved_pool = moved_all[:pool_n].detach()
        moved_logq = moved_logq_all[:pool_n].detach()
        final_eval = moved_all[:int(cfg.n_samples)].detach()
        moved_pf_info = dict(flow_info)
        density_elapsed = 0.0
        probe_info = field.probe_diagnostics(cfg)
        movement_gen = make_generator(
            int(cfg.seed + 881_000 + 10_000 * int(outer_round) + 101 * inner),
            target.device,
        )
        movement = adjacent_sample_discrepancy(moved_pool, current_pool, cfg, movement_gen)

        inner_info = {
            "ratio_inner_round": int(inner),
            "ratio_inner_rounds_total": int(n_rounds),
            "ratio_input_pool_n": int(current_pool.shape[0]),
            "ratio_generated_n": int(moved_all.shape[0]),
            "ratio_complement_strength": float(cfg.lambda_guard),
            "ratio_input_logq_mean": safe_float(current_logq.mean()),
            "ratio_input_logq_std": safe_float(current_logq.std(unbiased=False)),
            "ratio_raw_log_ratio_mean": safe_float(raw_rho.mean()),
            "ratio_raw_log_ratio_std": safe_float(raw_rho.std(unbiased=False)),
            "ratio_density_reconstruction_sec": float(density_elapsed),
            # The current q_k coordinates rebuild the carrier/gate/tilt, while
            # q_k's log density is the incoming certificate emitted by the
            # realized field built from q_{k-1}.  The raw previous bank need not
            # remain in memory once that realized certificate has been carried.
            "ratio_input_particles_refreshed": True,
            "ratio_carrier_gate_refreshed": True,
            "ratio_density_tilt_refreshed": True,
            "ratio_density_certificate_uses_previous_realized_field": True,
            "ratio_density_certificate_reference_round": int(inner - 1),
            **rho_info,
            **flow_info,
            **{f"ratio_output_{k}": v for k, v in moved_pf_info.items()},
            **probe_info,
            **{f"ratio_map_{k}": v for k, v in movement.items()},
        }
        inner_infos.append(inner_info)

        current_pool = moved_pool
        current_logq = moved_logq.detach()
        last_logpi = logpi
        last_raw_rho = raw_rho
        last_rho = effective_tilt
        last_rho_info = rho_info
        last_pf_info = moved_pf_info

    output_zero_rho = torch.zeros((current_pool.shape[0],), device=current_pool.device, dtype=current_pool.dtype)
    aggregate = {
        "ratio_method": "gated-pflow",
        "ratio_score_method": str(pf_method),
        "ratio_gate_source": "estimator-native",
        "ratio_refresh_mode": "current-particles-every-inner-round",
        "ratio_rounds": int(n_rounds),
        "ratio_rounds_requested": int(n_rounds),
        "ratio_complement_strength": float(cfg.lambda_guard),
        "ratio_tilt_policy": canonical_ratio_tilt_policy(getattr(cfg, "ratio_tilt_policy", "exact")),
        "ratio_inner_summary_json": json.dumps(inner_infos),
        "ratio_total_elapsed_sec": float(time.time() - total_start),
        "ratio_returns_unweighted_particles": True,
        **(inner_infos[-1] if inner_infos else {}),
    }
    metric_gen = make_generator(
        int(cfg.seed + 882_000 + 10_000 * int(outer_round)), target.device
    )
    ratio_metrics = compute_metrics(target, final_eval, truth, None, cfg, metric_gen)
    return (
        current_pool,
        output_zero_rho,
        final_eval,
        current_logq,
        aggregate,
        last_pf_info,
        last_rho_info,
        {
            "logpi": last_logpi,
            "raw_rho": last_raw_rho,
            "effective_tilt": last_rho,
            "metrics": ratio_metrics,
        },
    )


@torch.no_grad()
def run_gated_bflow_ratio_node(
    *,
    starting_pool: torch.Tensor,
    starting_logq: torch.Tensor,
    pf_method: str,
    ratio_method: str,
    ratio_rounds: int,
    target,
    truth: torch.Tensor,
    cfg: Config,
    outer_round: int,
    pi_gate_refs: Optional[torch.Tensor],
    pi_gate_rho: Optional[torch.Tensor],
    n_gate_refs: Optional[torch.Tensor],
    n_gate_rho: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, object],
    Dict[str, object],
    Dict[str, object],
    Dict[str, object],
]:
    """Apply refreshed gated, raw, or completed b-flow ratio transports.

    Each inner round rebuilds the selected Method-2 carrier and native gate on
    the current unweighted particle bank, together with coupled untilted/tilted
    Tweedie and target-score statistics ``b_q,b_pi,c_q,c_pi``.  The gated mode
    uses ``s_method + lambda(I-G_method)(b_pi-b_q)``; the raw control uses
    ``s_method + lambda(b_pi-b_q)``; completed-bflow additionally restores
    ``lambda G_method(c_pi-c_q)``.  The moved endpoint and its incoming
    certificate define the next inner round, so repeated b-flow rounds are
    unfrozen at the iteration level.
    """
    ratio_method = canonical_ratio_method(ratio_method)
    if ratio_method not in {"gated-bflow", "raw-bflow", "completed-bflow"}:
        raise ValueError(
            f"b-flow runner requires gated-bflow, raw-bflow, or completed-bflow; got {ratio_method!r}"
        )
    apply_gate_filter = ratio_method in {"gated-bflow", "completed-bflow"}
    restore_cancellation = ratio_method == "completed-bflow"
    n_rounds = int(ratio_rounds)
    if n_rounds < 1:
        raise ValueError(f"{ratio_method} ratio rounds must be >=1; got {ratio_rounds}")
    if canonical_score_method_key(pf_method) == "none":
        raise ValueError(
            f"{ratio_method} requires a non-none ratio/PF score estimator with a defined carrier"
        )

    pool_n = proposal_pool_size(cfg)
    current_pool = starting_pool[:pool_n].detach().clone()
    current_logq = starting_logq[:pool_n].detach().clone()
    final_eval = current_pool
    last_logpi = target.log_prob(current_pool, t=0.0).detach()
    last_raw_rho = last_logpi - current_logq
    last_rho, last_rho_info = prepare_ratio_flow_tilt(last_raw_rho, cfg)
    last_pf_info: Dict[str, object] = {}
    inner_infos: List[Dict[str, object]] = []
    total_start = time.time()

    for inner in range(1, n_rounds + 1):
        logpi = target.log_prob(current_pool, t=0.0).detach()
        raw_rho = logpi - current_logq
        effective_tilt, rho_info = prepare_ratio_flow_tilt(raw_rho, cfg)

        zero = torch.zeros((current_pool.shape[0],), device=current_pool.device, dtype=current_pool.dtype)
        score_refs, score_rho0, gate_refs, gate_rho0, split_info = split_score_gate_banks(
            current_pool, zero, cfg
        )
        carrier_bank = SNISScoreBank(
            target,
            score_refs,
            cfg,
            log_ref_weights=score_rho0,
            gate_anchors=gate_refs,
            gate_log_ref_weights=gate_rho0,
            pi_gate_anchors=pi_gate_refs,
            pi_gate_log_ref_weights=pi_gate_rho,
            n_gate_anchors=n_gate_refs,
            n_gate_log_ref_weights=n_gate_rho,
        )
        score_tilt = score_bank_values_from_pool(effective_tilt, cfg)
        field = GatedBFlowRatioField(
            carrier_bank,
            pf_method,
            score_tilt,
            complement_strength=float(cfg.lambda_guard),
            apply_gate_filter=apply_gate_filter,
            restore_cancellation=restore_cancellation,
        )

        # Only the final inner round needs the full evaluation cloud.  Earlier
        # rounds generate exactly the bank required to define the next frozen
        # carrier, avoiding a large amount of discarded work.
        generate_n = max(int(cfg.n_samples), int(pool_n)) if inner == n_rounds else int(pool_n)
        flow_gen = make_generator(
            int(cfg.seed + 880_000 + 10_000 * int(outer_round) + 101 * inner),
            target.device,
        )
        moved_all, moved_logq_all, flow_info = reverse_ou_heun_probability_flow_certified(
            target,
            lambda x, t, field=field: field.estimate_and_divergence(x, t, cfg),
            cfg,
            generator=flow_gen,
            n_samples=generate_n,
            steps=int(cfg.pf_steps),
            phase_name=f"ratio:{ratio_method}:{pf_method}:inner{inner}",
        )
        moved_pool = moved_all[:pool_n].detach()
        moved_logq = moved_logq_all[:pool_n].detach()
        final_eval = moved_all[:int(cfg.n_samples)].detach()
        moved_pf_info = dict(flow_info)
        density_elapsed = 0.0
        probe_info = field.probe_diagnostics(cfg)
        movement_gen = make_generator(
            int(cfg.seed + 881_000 + 10_000 * int(outer_round) + 101 * inner),
            target.device,
        )
        movement = adjacent_sample_discrepancy(moved_pool, current_pool, cfg, movement_gen)

        inner_info = {
            "ratio_inner_round": int(inner),
            "ratio_inner_rounds_total": int(n_rounds),
            "ratio_input_pool_n": int(current_pool.shape[0]),
            "ratio_generated_n": int(moved_all.shape[0]),
            "ratio_complement_strength": float(cfg.lambda_guard),
            "ratio_bflow_filter": "I-G+Gdc" if restore_cancellation else ("I-G" if apply_gate_filter else "I"),
            "ratio_bflow_filter_applied": bool(apply_gate_filter),
            "ratio_cancellation_restored": bool(restore_cancellation),
            "ratio_input_logq_mean": safe_float(current_logq.mean()),
            "ratio_input_logq_std": safe_float(current_logq.std(unbiased=False)),
            "ratio_raw_log_ratio_mean": safe_float(raw_rho.mean()),
            "ratio_raw_log_ratio_std": safe_float(raw_rho.std(unbiased=False)),
            "ratio_density_reconstruction_sec": float(density_elapsed),
            # The current q_k coordinates rebuild the carrier/gate/tilt, while
            # q_k's log density is the incoming certificate emitted by the
            # realized field built from q_{k-1}.  The raw previous bank need not
            # remain in memory once that realized certificate has been carried.
            "ratio_input_particles_refreshed": True,
            "ratio_carrier_gate_refreshed": True,
            "ratio_density_tilt_refreshed": True,
            "ratio_density_certificate_uses_previous_realized_field": True,
            "ratio_density_certificate_reference_round": int(inner - 1),
            **rho_info,
            **flow_info,
            **{f"ratio_output_{k}": v for k, v in moved_pf_info.items()},
            **probe_info,
            **{f"ratio_map_{k}": v for k, v in movement.items()},
        }
        inner_infos.append(inner_info)

        current_pool = moved_pool
        current_logq = moved_logq.detach()
        last_logpi = logpi
        last_raw_rho = raw_rho
        last_rho = effective_tilt
        last_rho_info = rho_info
        last_pf_info = moved_pf_info

    output_zero_rho = torch.zeros((current_pool.shape[0],), device=current_pool.device, dtype=current_pool.dtype)
    aggregate = {
        "ratio_method": ratio_method,
        "ratio_score_method": str(pf_method),
        "ratio_gate_source": (
            "estimator-native-complement-plus-cancellation"
            if restore_cancellation else (
                "estimator-native-complement" if apply_gate_filter
                else "identity-filter; estimator-native-gate-used-only-by-carrier"
            )
        ),
        "ratio_bflow_filter": "I-G+Gdc" if restore_cancellation else ("I-G" if apply_gate_filter else "I"),
        "ratio_bflow_filter_applied": bool(apply_gate_filter),
        "ratio_cancellation_restored": bool(restore_cancellation),
        "ratio_field_definition": (
            "s_method + lambda*((I-G_method)*(b_pi-b_q) + G_method*(c_pi-c_q))"
            if restore_cancellation else (
                "s_method + lambda*(I-G_method)*(b_pi-b_q)" if apply_gate_filter
                else "s_method + lambda*(b_pi-b_q)"
            )
        ),
        "ratio_b_vectors": "canonical-current-bank-untilted-and-density-tilted-tweedie",
        "ratio_c_vectors": "canonical-current-bank-untilted-and-density-tilted-target-score",
        "ratio_refresh_mode": "current-particles-every-inner-round",
        "ratio_rounds": int(n_rounds),
        "ratio_rounds_requested": int(n_rounds),
        "ratio_complement_strength": float(cfg.lambda_guard),
        "ratio_tilt_policy": canonical_ratio_tilt_policy(getattr(cfg, "ratio_tilt_policy", "exact")),
        "ratio_inner_summary_json": json.dumps(inner_infos),
        "ratio_total_elapsed_sec": float(time.time() - total_start),
        "ratio_returns_unweighted_particles": True,
        **(inner_infos[-1] if inner_infos else {}),
    }
    metric_gen = make_generator(
        int(cfg.seed + 882_000 + 10_000 * int(outer_round)), target.device
    )
    ratio_metrics = compute_metrics(target, final_eval, truth, None, cfg, metric_gen)
    return (
        current_pool,
        output_zero_rho,
        final_eval,
        current_logq,
        aggregate,
        last_pf_info,
        last_rho_info,
        {
            "logpi": last_logpi,
            "raw_rho": last_raw_rho,
            "effective_tilt": last_rho,
            "metrics": ratio_metrics,
        },
    )


@torch.no_grad()
def run_family(
    family: str,
    transport_method: str,
    pf_method: str,
    transport_repeats: int,
    ratio_method: str,
    ratio_rounds: int,
    target,
    init_refs: torch.Tensor,
    init_rho: torch.Tensor,
    truth: torch.Tensor,
    cfg: Config,
) -> Tuple[List[torch.Tensor], List[Dict[str, object]], List[Dict[str, object]]]:
    """Run one estimator family.

    Each outer round consists of ``transport_repeats`` current-particle steps,
    followed by either the legacy raw importance-weight node or one or more
    refreshed gated-p-flow, gated-b-flow, or raw-b-flow ratio rounds.  The transport
    field that actually generates the endpoint also emits its incoming density
    certificate.  The second score method supplies the refreshed ratio carrier/gate modality; b-flow additionally uses the canonical b_q/b_pi pair on that same current score bank.  Transport substeps consume the particles produced by
    the preceding substep; ratio rounds do the same with their moved particles.
    """
    transport_repeats = int(transport_repeats)
    if transport_repeats < 1:
        raise ValueError(f"transport_repeats must be >=1; got {transport_repeats}")
    ratio_reference_mode = canonical_ratio_reference_mode(getattr(cfg, "ratio_reference_mode", "endpoint"))
    ratio_method = canonical_ratio_method(ratio_method)
    ratio_rounds = int(ratio_rounds)
    if is_moved_ratio_flow(ratio_method) and ratio_rounds < 1:
        raise ValueError(f"ratio_rounds must be >=1 for {ratio_method}; got {ratio_rounds}")
    correction_enabled = (
        not bool(getattr(cfg, "force_no_likelihood_correction", False))
        and canonical_score_method_key(pf_method) != "none"
    )
    # This variant deliberately restores iterative_lfgi.py transport semantics:
    # reverse SDE for particle generation, followed by a separate frozen-field
    # PF density reconstruction for the likelihood factors.
    transport_dynamics = canonical_transport_dynamics(
        getattr(cfg, "transport_dynamics", "reverse_sde"),
        correction_enabled=correction_enabled,
    )
    method_name = method_label(transport_method, transport_repeats, pf_method, ratio_method, ratio_rounds)

    # pi-LFGI and LFGI-N keep q-derived score signals but estimate the
    # curvature gate from an external localization bank.  Reuse one fixed bank
    # across rounds/substeps so Monte Carlo refresh cannot masquerade as
    # adaptation of the evolving reference.
    needs_pi_gate = is_pi_lfgi_method(transport_method) or is_pi_lfgi_method(pf_method)
    if needs_pi_gate:
        pi_gate_gen = make_generator(int(cfg.seed + 606_000), target.device)
        pi_gate_refs = target.sample(int(effective_gate_n(cfg)), generator=pi_gate_gen).detach()
        pi_gate_rho = torch.zeros((pi_gate_refs.shape[0],), device=target.device, dtype=target.dtype)
    else:
        pi_gate_refs = None
        pi_gate_rho = None

    needs_n_gate = is_n_lfgi_method(transport_method) or is_n_lfgi_method(pf_method)
    if needs_n_gate:
        n_gate_gen = make_generator(int(cfg.seed + 707_000), target.device)
        n_gate_refs = torch.randn(
            (int(effective_gate_n(cfg)), int(target.d)),
            device=target.device,
            dtype=target.dtype,
            generator=n_gate_gen,
        ).detach()
        n_gate_rho = torch.zeros((n_gate_refs.shape[0],), device=target.device, dtype=target.dtype)
    else:
        n_gate_refs = None
        n_gate_rho = None

    current_pool = init_refs.detach().clone()
    current_rho = init_rho.detach().clone()
    # Ratio-mode comparisons should begin from the same settled transport
    # realization whenever their transport specification is the same.  Do not
    # let the display family (which contains the ratio mode/round count) alter
    # transport or diagnostic random seeds.
    transport_seed_offset = family_seed_offset(
        f"transport:{transport_method}:{int(transport_repeats)}"
    )
    samples_by_round: List[torch.Tensor] = []
    metric_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []
    previous_samples_for_adj: Optional[torch.Tensor] = None

    for r in range(1, int(cfg.n_rounds) + 1):
        round_t0 = time.time()
        next_pool_n = proposal_pool_size(cfg)
        generate_n = max(int(cfg.n_samples), int(next_pool_n))

        # Keep the incoming bank diagnostics for the round-level CSV row.
        input_score_refs, input_score_rho, input_gate_refs, input_gate_rho, input_split_info = split_score_gate_banks(
            current_pool, current_rho, cfg
        )
        if previous_samples_for_adj is None:
            previous_samples_for_adj = input_score_refs.detach()
        in_ess, in_ess_frac = log_weight_ess(input_score_rho)

        # The transport block.  Substep 1 consumes the current possibly weighted
        # empirical reference.  Subsequent substeps are pure uncorrected transport
        # iterations, so they receive zero ratio weights.
        transport_pool = current_pool.detach()
        transport_rho = current_rho.detach()
        samples_all: Optional[torch.Tensor] = None
        samples_eval: Optional[torch.Tensor] = None
        sampler_info: Dict[str, object] = {}
        generator_bank: Optional[SNISScoreBank] = None
        endpoint_bank: Optional[SNISScoreBank] = None
        endpoint_score_refs: Optional[torch.Tensor] = None
        endpoint_score_rho0: Optional[torch.Tensor] = None
        endpoint_gate_refs: Optional[torch.Tensor] = None
        endpoint_gate_rho0: Optional[torch.Tensor] = None
        endpoint_split_info: Optional[Dict[str, object]] = None
        last_convergence_info: Dict[str, object] = blank_convergence_info("not_computed")
        transport_substep_infos: List[Dict[str, object]] = []

        for m in range(1, transport_repeats + 1):
            score_refs, score_rho, gate_refs, gate_rho, split_info = split_score_gate_banks(transport_pool, transport_rho, cfg)
            bank = SNISScoreBank(
                target,
                score_refs,
                cfg,
                log_ref_weights=score_rho,
                gate_anchors=gate_refs,
                gate_log_ref_weights=gate_rho,
                pi_gate_anchors=pi_gate_refs,
                pi_gate_log_ref_weights=pi_gate_rho,
                n_gate_anchors=n_gate_refs,
                n_gate_log_ref_weights=n_gate_rho,
            )
            gen = make_generator(int(cfg.seed + 10_000 * r + 701 * m + transport_seed_offset), target.device)

            if str(transport_method).lower() == "none":
                step_samples_all = transport_pool[:next_pool_n].detach()
                if int(cfg.n_samples) <= int(step_samples_all.shape[0]):
                    step_samples_eval = step_samples_all[:int(cfg.n_samples)].detach()
                else:
                    step_samples_eval = weighted_resample(score_refs, score_rho, int(cfg.n_samples), gen)
                step_sampler_info = {
                    "failed": False,
                    "fail_reason": "",
                    "max_abs_score": 0.0,
                    "transport_none": True,
                    "generated_n": int(step_samples_all.shape[0]),
                    "sampler_t_min": float(effective_time_bounds(cfg)[0]),
                    "sampler_t_max": float(effective_time_bounds(cfg)[1]),
                    "sampler_time_schedule": canonical_time_schedule(cfg.time_schedule),
                    "transport_dynamics": "identity",
                    "transport_density_certificate_source": "none; endpoint density reconstructed separately",
                }
            else:
                score_fn_step = lambda y, t, bank=bank, method=transport_method: bank.estimate(y, t, method)
                step_samples_all, step_sampler_info = reverse_ou_heun_sde(
                    target,
                    score_fn_step,
                    cfg,
                    generator=gen,
                    n_samples=generate_n,
                    final_denoise=cfg.final_denoise,
                )
                step_sampler_info["transport_dynamics"] = "reverse-sde"
                step_sampler_info["transport_density_certificate_source"] = "none; endpoint density reconstructed separately"
                step_sampler_info["generated_n"] = int(step_samples_all.shape[0])
                if m == transport_repeats and cfg.eval_final_denoise:
                    step_samples_eval, _ = reverse_ou_heun_sde(
                        target,
                        score_fn_step,
                        cfg,
                        generator=make_generator(int(cfg.seed + 999_000 + 10_000 * r + 701 * m + transport_seed_offset), target.device),
                        n_samples=int(cfg.n_samples),
                        final_denoise=True,
                    )
                else:
                    step_samples_eval = step_samples_all[:int(cfg.n_samples)].detach()

            next_pool = step_samples_all[:next_pool_n].detach()
            zero_next_rho = torch.zeros((next_pool.shape[0],), device=next_pool.device, dtype=next_pool.dtype)
            next_score_refs0, next_score_rho0, next_gate_refs0, next_gate_rho0, next_split_info0 = split_score_gate_banks(
                next_pool, zero_next_rho, cfg
            )
            next_bank0 = SNISScoreBank(
                target,
                next_score_refs0,
                cfg,
                log_ref_weights=next_score_rho0,
                gate_anchors=next_gate_refs0,
                gate_log_ref_weights=next_gate_rho0,
                pi_gate_anchors=pi_gate_refs,
                pi_gate_log_ref_weights=pi_gate_rho,
                n_gate_anchors=n_gate_refs,
                n_gate_log_ref_weights=n_gate_rho,
            )

            # The full Delta_PF diagnostics are the expensive theorem-facing
            # quantities.  For multi-transport blocks we report them for the final
            # adjacent pair only: s_{k+n-1} versus s_{k+n}, immediately before
            # the optional ratio correction.
            if (
                m == transport_repeats
                and bool(getattr(cfg, "convergence_check", False))
                and str(transport_method).lower() != "none"
            ):
                delta_gen = make_generator(int(cfg.seed + 440_000 + 10_000 * r + 701 * m + transport_seed_offset), target.device)
                step_conv = probability_flow_score_discrepancy(
                    bank, transport_method, next_bank0, transport_method, cfg, delta_gen
                )
                target_delta_gen = make_generator(int(cfg.seed + 441_000 + 10_000 * r + 701 * m + transport_seed_offset), target.device)
                step_conv.update(target_marginal_score_discrepancy(
                    target, bank, transport_method, next_bank0, transport_method, cfg, target_delta_gen
                ))
            else:
                reason = "nonfinal_transport_substep" if m != transport_repeats else "convergence_check_false_or_none_transport"
                step_conv = blank_convergence_info(reason)

            transport_substep_infos.append({
                "transport_substep": int(m),
                "transport_substeps_total": int(transport_repeats),
                "transport_substep_input_rho_ess_frac": log_weight_ess(score_rho)[1],
                "transport_substep_generated_n": int(step_sampler_info.get("generated_n", step_samples_all.shape[0])),
                "transport_substep_sampler_failed": bool(step_sampler_info.get("failed", False)),
                "transport_substep_sampler_fail_reason": str(step_sampler_info.get("fail_reason", "")),
                "transport_substep_sampler_max_abs_score": safe_float(step_sampler_info.get("max_abs_score", float("nan"))),
                "transport_substep_delta_pf": safe_float(step_conv.get("delta_pf", float("nan"))),
                "transport_substep_delta_pf_target": safe_float(step_conv.get("delta_pf_target", float("nan"))),
                "transport_substep_updates_current_particles": True,
                "transport_substep_density_certificate_incoming": bool(
                    str(step_sampler_info.get("transport_density_certificate_source", "")) == "incoming-generating-field"
                ),
                "transport_substep_input_pool_n": int(transport_pool.shape[0]),
                "transport_substep_output_pool_n": int(next_pool.shape[0]),
            })

            # Final-substep objects are used for metrics, convergence reporting,
            # and the optional likelihood-ratio correction.
            generator_bank = bank
            endpoint_bank = next_bank0
            endpoint_score_refs = next_score_refs0
            endpoint_score_rho0 = next_score_rho0
            endpoint_gate_refs = next_gate_refs0
            endpoint_gate_rho0 = next_gate_rho0
            endpoint_split_info = next_split_info0
            samples_all = step_samples_all
            samples_eval = step_samples_eval
            sampler_info = step_sampler_info
            last_convergence_info = step_conv

            # Intermediate transport rounds are deliberately unweighted and
            # genuinely refreshed: the next estimator is built from the actual
            # particles produced by this transport substep.
            transport_pool = next_pool.detach().clone()
            transport_rho = zero_next_rho.detach().clone()

        assert samples_all is not None and samples_eval is not None
        assert endpoint_bank is not None and generator_bank is not None
        assert endpoint_score_refs is not None and endpoint_score_rho0 is not None
        assert endpoint_gate_refs is not None and endpoint_gate_rho0 is not None
        assert endpoint_split_info is not None
        # Report score quality for the settled endpoint reference rather than the
        # pre-final generator bank.  This is the object the new TSC workflow uses
        # for ratio evaluation by default.
        if str(transport_method).lower() == "none":
            metric_score_fn = None
        else:
            metric_score_fn = lambda y, t, bank=endpoint_bank, method=transport_method: bank.estimate(y, t, method)
        metric_gen = make_generator(int(cfg.seed + 220_000 + 10_000 * r + transport_seed_offset), target.device)
        metrics = compute_metrics(target, samples_eval, truth, metric_score_fn, cfg, metric_gen)
        adjacent_gen = make_generator(int(cfg.seed + 330_000 + 10_000 * r + transport_seed_offset), target.device)
        adjacent_metrics = adjacent_sample_discrepancy(samples_eval, previous_samples_for_adj, cfg, adjacent_gen)
        row = {
            "kind": "transport_output",
            "family": family,
            "method": method_name,
            "transport_method": transport_method,
            "transport_repeats": int(transport_repeats),
            "pf_method": pf_method,
            "correction_method": pf_method,
            "ratio_method": ratio_method,
            "ratio_rounds": int(ratio_rounds),
            "ratio_reference_mode": ratio_reference_mode,
            "round": int(r),
            "input_ref_n": int(input_score_refs.shape[0]),
            "input_gate_n": int(input_gate_refs.shape[0]),
            "input_pool_n": int(current_pool.shape[0]),
            "bank_coupling": input_split_info["bank_coupling"],
            "score_slice": input_split_info["score_slice"],
            "gate_slice": input_split_info["gate_slice"],
            "bank_overlap_n": int(input_split_info["bank_overlap_n"]),
            "pi_gate_source": "target" if needs_pi_gate else "none",
            "pi_gate_n": int(pi_gate_refs.shape[0]) if pi_gate_refs is not None else 0,
            "pi_gate_fixed_across_rounds": bool(needs_pi_gate),
            "n_gate_source": "standard_normal" if needs_n_gate else "none",
            "n_gate_n": int(n_gate_refs.shape[0]) if n_gate_refs is not None else 0,
            "n_gate_fixed_across_rounds": bool(needs_n_gate),
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
            "fisher_rmse_method": str(transport_method),
            "fisher_rmse_state": "post-transport-current-law",
            **metrics,
            **adjacent_metrics,
        }
        metric_rows.append(row)

        # Ratio step on the reverse-SDE transport endpoint.  Match
        # iterative_lfgi.py exactly for the likelihood factors: build the
        # selected PF/ratio estimator on either the endpoint bank (default) or
        # the bank that generated the endpoint, then reconstruct log q with the
        # frozen-field probability-flow density integral.
        final_pool = samples_all[:next_pool_n].detach()
        pf_t0 = time.time()
        logpi = target.log_prob(final_pool, t=0.0)
        skip_likelihood_correction = bool(getattr(cfg, "force_no_likelihood_correction", False)) or str(pf_method).lower() == "none"
        ratio_info: Dict[str, object] = {
            "ratio_method": ratio_method,
            "ratio_rounds": int(ratio_rounds if is_moved_ratio_flow(ratio_method) else 0),
            "ratio_rounds_requested": int(ratio_rounds),
            "ratio_returns_unweighted_particles": False,
        }
        ratio_eval_samples: Optional[torch.Tensor] = None
        gated_payload: Optional[Dict[str, object]] = None
        if skip_likelihood_correction:
            logq = torch.full_like(logpi, float("nan"))
            raw_rho = torch.zeros_like(logpi)
            skip_reason = "force_no_likelihood_correction" if bool(getattr(cfg, "force_no_likelihood_correction", False)) else "pf_method_none"
            pf_info = blank_pf_info(pf_method, reason=skip_reason)
            calib_info = blank_calibration_info(reason=skip_reason)
            ratio_bank_source = "none"
            next_pool_out = final_pool
            next_rho = torch.zeros_like(raw_rho)
            rho_info = finalize_density_ratio_weights(raw_rho, cfg)[1]
            ratio_info.update({
                "ratio_skipped": True,
                "ratio_skip_reason": skip_reason,
            })
        else:
            correction_bank = endpoint_bank if ratio_reference_mode == "endpoint" else generator_bank
            ratio_bank_source = ratio_reference_mode
            logq, pf_info = pf_logprob_bank(correction_bank, final_pool, pf_method, cfg)
            pf_info["certificate_provenance"] = f"legacy-frozen-{ratio_reference_mode}-field-reconstruction"
            raw_rho = logpi - logq
            next_score_start = int(effective_gate_n(cfg)) if endpoint_split_info["bank_coupling"] == "independent" else 0
            score_raw_rho = raw_rho[next_score_start:next_score_start + int(cfg.n_ref)]
            score_logq = logq[next_score_start:next_score_start + int(cfg.n_ref)]
            calib_info = likelihood_correction_calibration(target, endpoint_score_refs, score_logq, score_raw_rho, cfg)
            if ratio_method == "raw-w":
                next_pool_out = final_pool
                # Keep the raw-weight arm on the same endpoint-label policy as
                # both flow arms.  Under ratio_tilt_policy=exact this is the
                # literal centered log(pi/q), with no clipping, tempering, or
                # ESS floor; stabilized retains the legacy guarded behavior.
                next_rho, rho_info = prepare_ratio_flow_tilt(raw_rho, cfg)
                ratio_info.update({
                    "ratio_skipped": False,
                    "ratio_returns_unweighted_particles": False,
                    "ratio_rounds": 0,
                    "ratio_rounds_ignored_for_raw_w": int(ratio_rounds),
                    "ratio_tilt_policy": canonical_ratio_tilt_policy(
                        getattr(cfg, "ratio_tilt_policy", "exact")
                    ),
                })
            elif ratio_method == "gated-pflow":
                (
                    next_pool_out,
                    next_rho,
                    ratio_eval_samples,
                    _ratio_output_logq,
                    gated_info,
                    gated_output_pf_info,
                    rho_info,
                    gated_payload,
                ) = run_gated_pflow_ratio_node(
                    starting_pool=final_pool,
                    starting_logq=logq,
                    pf_method=pf_method,
                    ratio_rounds=ratio_rounds,
                    target=target,
                    truth=truth,
                    cfg=cfg,
                    outer_round=r,
                    pi_gate_refs=pi_gate_refs,
                    pi_gate_rho=pi_gate_rho,
                    n_gate_refs=n_gate_refs,
                    n_gate_rho=n_gate_rho,
                )
                ratio_info.update(gated_info)
                ratio_info.update({f"ratio_final_{k}": v for k, v in gated_output_pf_info.items()})
            elif ratio_method in {"gated-bflow", "raw-bflow", "completed-bflow"}:
                (
                    next_pool_out,
                    next_rho,
                    ratio_eval_samples,
                    _ratio_output_logq,
                    gated_info,
                    gated_output_pf_info,
                    rho_info,
                    gated_payload,
                ) = run_gated_bflow_ratio_node(
                    starting_pool=final_pool,
                    starting_logq=logq,
                    pf_method=pf_method,
                    ratio_method=ratio_method,
                    ratio_rounds=ratio_rounds,
                    target=target,
                    truth=truth,
                    cfg=cfg,
                    outer_round=r,
                    pi_gate_refs=pi_gate_refs,
                    pi_gate_rho=pi_gate_rho,
                    n_gate_refs=n_gate_refs,
                    n_gate_rho=n_gate_rho,
                )
                ratio_info.update(gated_info)
                ratio_info.update({f"ratio_final_{k}": v for k, v in gated_output_pf_info.items()})
            else:
                raise RuntimeError(f"Unhandled ratio_method={ratio_method!r}")
        pf_elapsed = time.time() - pf_t0

        # Re-split the actual next-round particle law before computing full-cycle
        # diagnostics.  This law -- not the pre-ratio transport endpoint -- is
        # the state consumed by the next outer transport half-step.
        next_score_refs, next_score_rho, next_gate_refs, next_gate_rho, next_split_info = split_score_gate_banks(
            next_pool_out, next_rho, cfg
        )

        # The primary round sample is the full-cycle output, not the pre-ratio
        # transport checkpoint.  Moved ratio flows already provide an unweighted
        # evaluation cloud.  For raw endpoint weighting, draw an evaluation cloud
        # from the carried weighted law so sample metrics and plots represent the
        # actual state passed to the next round.
        if ratio_eval_samples is not None:
            cycle_samples = ratio_eval_samples
            cycle_sample_source = "moved-ratio-output"
        elif ratio_method == "raw-w" and not skip_likelihood_correction:
            cycle_sample_gen = make_generator(
                int(cfg.seed + 332_000 + 10_000 * r + transport_seed_offset), target.device
            )
            cycle_samples = weighted_resample(
                next_score_refs, next_score_rho, int(cfg.n_samples), cycle_sample_gen
            )
            cycle_sample_source = "weighted-current-law-resample"
        else:
            cycle_samples = samples_eval
            cycle_sample_source = "transport-output"

        if gated_payload is not None:
            cycle_metrics = dict(gated_payload["metrics"])
        elif ratio_method == "raw-w" and not skip_likelihood_correction:
            # The old implementation inherited pre-ratio sample metrics here,
            # even though the state carried into the next round was weighted.
            # Recompute sample-quality metrics on the weighted-law resample.
            cycle_metric_gen = make_generator(
                int(cfg.seed + 333_000 + 10_000 * r + transport_seed_offset), target.device
            )
            cycle_metrics = compute_metrics(target, cycle_samples, truth, None, cfg, cycle_metric_gen)
        else:
            cycle_metrics = dict(metrics)

        # FisherRMSE is a score-of-the-current-law diagnostic.  The moved ratio
        # runners intentionally computed their sample metrics with score_fn=None,
        # which made FisherRMSE identically NaN for every gated-pflow/bflow row.
        # Rebuild the estimator on the actual post-ratio bank and evaluate the
        # transport-side score family, since that is the score rule consumed by
        # the next T half-step.  This keeps the metric comparable when Method 2
        # differs (e.g. LFGI transport with a Tweedie ratio carrier).
        if canonical_score_method_key(transport_method) == "none":
            cycle_fisher = float("nan")
            cycle_fisher_status = "transport-method-none"
        else:
            cycle_bank = SNISScoreBank(
                target,
                next_score_refs,
                cfg,
                log_ref_weights=next_score_rho,
                gate_anchors=next_gate_refs,
                gate_log_ref_weights=next_gate_rho,
                pi_gate_anchors=pi_gate_refs,
                pi_gate_log_ref_weights=pi_gate_rho,
                n_gate_anchors=n_gate_refs,
                n_gate_log_ref_weights=n_gate_rho,
            )
            cycle_fisher_fn = lambda y, t, bank=cycle_bank, method=transport_method: bank.estimate(y, t, method)
            cycle_fisher_gen = make_generator(
                int(cfg.seed + 334_000 + 10_000 * r + transport_seed_offset), target.device
            )
            cycle_fisher = fisher_rmse(target, cycle_fisher_fn, cfg, cycle_fisher_gen)
            cycle_fisher_status = "post-ratio-current-law"
        cycle_metrics["fisher_rmse"] = float(cycle_fisher)
        cycle_metrics["fisher_rmse_method"] = str(transport_method)
        cycle_metrics["fisher_rmse_state"] = cycle_fisher_status
        cycle_metrics["cycle_sample_source"] = cycle_sample_source

        cycle_adj_gen = make_generator(
            int(cfg.seed + 331_000 + 10_000 * r + transport_seed_offset), target.device
        )
        cycle_adjacent = adjacent_sample_discrepancy(
            cycle_samples, previous_samples_for_adj, cfg, cycle_adj_gen
        )
        cycle_row = dict(row)
        cycle_row["kind"] = "sample"
        cycle_row["cycle_includes_ratio"] = bool(not skip_likelihood_correction)
        cycle_row["cycle_ratio_returns_unweighted_particles"] = bool(
            ratio_info.get("ratio_returns_unweighted_particles", False)
        )
        cycle_row.update(cycle_metrics)
        cycle_row.update(cycle_adjacent)
        metric_rows.append(cycle_row)
        samples_by_round.append(cycle_samples.detach())
        previous_samples_for_adj = cycle_samples.detach()
        mode_before = mode_mass_l1(target, next_score_refs)
        w = torch.exp(next_score_rho - torch.max(next_score_rho))
        w = w / torch.clamp(w.sum(), min=1.0e-30)
        if hasattr(target, "responsibilities") and int(getattr(target, "K", 0)) > 0:
            resp = target.responsibilities(next_score_refs, t=0.0)
            weighted_mode = torch.einsum("n,nk->k", w, resp)
            weighted_mode_l1 = safe_float(torch.sum(torch.abs(weighted_mode - target.weights)))
        else:
            weighted_mode_l1 = float("nan")

        # Flatten final substep diagnostics into the stage row.  The full list is
        # stored as JSON for auditability without exploding the main CSV width.
        convergence_info = dict(last_convergence_info)
        stage_rows.append({
            "family": family,
            "method": method_name,
            "transport_method": transport_method,
            "transport_repeats": int(transport_repeats),
            "pf_method": pf_method,
            "correction_method": pf_method,
            "ratio_method": ratio_method,
            "ratio_rounds": int(ratio_rounds if is_moved_ratio_flow(ratio_method) else 0),
            "ratio_reference_mode": ratio_reference_mode,
            "ratio_bank_source": ratio_bank_source,
            "transport_dynamics_effective": transport_dynamics,
            "density_certificate_provenance": str(pf_info.get("certificate_provenance", "legacy-frozen-field-reconstruction")),
            "density_certificate_uses_transport_method": False,
            "density_certificate_transport_method": "none",
            "likelihood_reconstruction_method": str(pf_method),
            "ratio_carrier_method": str(pf_method),
            "round": int(r),
            "r_step_ref_n": int(next_score_refs.shape[0]),
            "r_step_gate_n": int(next_gate_refs.shape[0]),
            "r_step_input_pool_n": int(final_pool.shape[0]),
            "r_step_pool_n": int(next_pool_out.shape[0]),
            "bank_coupling": next_split_info["bank_coupling"],
            "score_slice": next_split_info["score_slice"],
            "gate_slice": next_split_info["gate_slice"],
            "bank_overlap_n": int(next_split_info["bank_overlap_n"]),
            "pi_gate_source": "target" if needs_pi_gate else "none",
            "pi_gate_n": int(pi_gate_refs.shape[0]) if pi_gate_refs is not None else 0,
            "pi_gate_fixed_across_rounds": bool(needs_pi_gate),
            "transport_substep_summary_json": json.dumps(transport_substep_infos),
            "final_transport_substep": int(transport_repeats),
            "r_step_elapsed_sec": float(pf_elapsed),
            "raw_rho_mean": safe_float(raw_rho.mean()),
            "raw_rho_std": safe_float(raw_rho.std(unbiased=False)),
            "logpi_mean": safe_float(logpi.mean()),
            "logpi_std": safe_float(logpi.std(unbiased=False)),
            "mode_l1_unweighted_next_refs": mode_before,
            "mode_l1_weighted_next_refs": weighted_mode_l1,
            **convergence_info,
            **pf_info,
            **calib_info,
            **ratio_info,
            **(endpoint_bank.mp_leaf_info if any(str(m).lower().replace("_", "-") in {"leaf-lfgi", "mp-leaf-lfgi", "leaf-ce-hlsi", "mp-leaf-ce-hlsi", "leaf-ce-lfgi"} for m in (transport_method, pf_method)) else {}),
            **rho_info,
            # Reassert stable method columns after nested PF/ratio diagnostics,
            # whose internal phase labels also use a ``pf_method`` key.
            "pf_method": str(pf_method),
            "correction_method": str(pf_method),
            "ratio_carrier_method": str(pf_method),
            "density_flow_phase": str(pf_info.get("pf_method", "unknown")),
            "ratio_final_flow_phase": str(
                ratio_info.get("pf_method", ratio_info.get("ratio_output_pf_method", "none"))
            ),
        })

        current_pool = next_pool_out.detach()
        current_rho = next_rho.detach()
        print(
            f"[{family} | S={transport_method}x{transport_repeats}, PF={pf_method}, ratio={ratio_method}x{ratio_rounds if is_moved_ratio_flow(ratio_method) else 1}, ratio_ref={ratio_reference_mode}] "
            f"round {r}/{cfg.n_rounds}: "
            f"MMD={cycle_metrics['mmd']:.4g}, KSD={cycle_metrics['ksd']:.4g}, SW2={cycle_metrics['sw2']:.4g}, "
            f"SKS={cycle_metrics['sliced_ks']:.4g}, NLL={cycle_metrics['nll']:.4g}, "
            f"FisherRMSE={cycle_metrics['fisher_rmse']:.4g}, adjMMD={cycle_adjacent['adjacent_sample_mmd']:.4g}, "
            f"DeltaPF={convergence_info.get('delta_pf', float('nan')):.4g}, DeltaPF_pi={convergence_info.get('delta_pf_target', float('nan')):.4g}, "
            f"PFEpMMD={convergence_info.get('delta_pf_endpoint_mmd', float('nan')):.4g}, "
            f"rhoESS={rho_info['rho_ess_frac']:.3f}, bank={next_split_info['bank_coupling']} "
            f"score_n={next_split_info['score_n']} gate_n={next_split_info['gate_n']}, "
            f"pf_skip={pf_info.get('pf_skipped', False)}, ratio_unweighted={ratio_info.get('ratio_returns_unweighted_particles', False)}",
            flush=True,
        )
    return samples_by_round, metric_rows, stage_rows





def _estimator_alias_table() -> Dict[str, Tuple[str, str]]:
    """Map user-facing estimator aliases to (short display, internal method key)."""
    return {
        "blend": ("Blend", "blend"),
        "blended": ("Blend", "blend"),
        "scalar-blend": ("Blend", "blend"),
        "scalar_blend": ("Blend", "blend"),
        "local-scalar-blend": ("Blend", "blend"),
        "local_scalar_blend": ("Blend", "blend"),
        "scalar": ("Blend", "blend"),
        "matrix-blend": ("MatrixBlend", "matrix-blend"),
        "matrix_blend": ("MatrixBlend", "matrix-blend"),
        "centered-blend": ("MatrixBlend", "matrix-blend"),
        "centered_blend": ("MatrixBlend", "matrix-blend"),
        "centered-matrix-blend": ("MatrixBlend", "matrix-blend"),
        "centered_matrix_blend": ("MatrixBlend", "matrix-blend"),
        "local-matrix-blend": ("MatrixBlend", "matrix-blend"),
        "local_matrix_blend": ("MatrixBlend", "matrix-blend"),
        "unif-blend": ("UnifBlend", "unif-blend"),
        "unif_blend": ("UnifBlend", "unif-blend"),
        "unif-scalar-blend": ("UnifBlend", "unif-blend"),
        "unif_scalar_blend": ("UnifBlend", "unif-blend"),
        "uniform-blend": ("UnifBlend", "unif-blend"),
        "uniform_blend": ("UnifBlend", "unif-blend"),
        "uniform-scalar-blend": ("UnifBlend", "unif-blend"),
        "uniform_scalar_blend": ("UnifBlend", "unif-blend"),
        "global-scalar-blend": ("UnifBlend", "unif-blend"),
        "global_scalar_blend": ("UnifBlend", "unif-blend"),
        "unif-matrix-blend": ("UnifMatrixBlend", "unif-matrix-blend"),
        "unif_matrix_blend": ("UnifMatrixBlend", "unif-matrix-blend"),
        "uniform-matrix-blend": ("UnifMatrixBlend", "unif-matrix-blend"),
        "uniform_matrix_blend": ("UnifMatrixBlend", "unif-matrix-blend"),
        "global-matrix-blend": ("UnifMatrixBlend", "unif-matrix-blend"),
        "global_matrix_blend": ("UnifMatrixBlend", "unif-matrix-blend"),
        "lfgi": ("LFGI", "ce-hlsi"),
        "ce-hlsi": ("LFGI", "ce-hlsi"),
        "ce_hlsi": ("LFGI", "ce-hlsi"),
        "ce-lfgi": ("LFGI", "ce-hlsi"),
        "os-lfgi": ("OS-LFGI", "os-lfgi"),
        "os_lfgi": ("OS-LFGI", "os-lfgi"),
        "one-step-lfgi": ("OS-LFGI", "os-lfgi"),
        "one_step_lfgi": ("OS-LFGI", "os-lfgi"),
        "residual-corrected-lfgi": ("OS-LFGI", "os-lfgi"),
        "residual_corrected_lfgi": ("OS-LFGI", "os-lfgi"),
        "pi-lfgi": ("pi-LFGI", "pi-lfgi"),
        # Backward-compatible legacy spelling; canonical user-facing token is pi-lfgi.
        "pi_lfgi": ("pi-LFGI", "pi-lfgi"),
        "pi-ce-hlsi": ("pi-LFGI", "pi-lfgi"),
        "pi_ce_hlsi": ("pi-LFGI", "pi-lfgi"),
        "oracle-lfgi": ("pi-LFGI", "pi-lfgi"),
        "oracle_lfgi": ("pi-LFGI", "pi-lfgi"),
        "target-lfgi": ("pi-LFGI", "pi-lfgi"),
        "target_lfgi": ("pi-LFGI", "pi-lfgi"),
        "lfgi-n": ("LFGI-N", "lfgi-N"),
        # Backward-compatible/verbose aliases; canonical user-facing token is lfgi-N.
        "lfgi_n": ("LFGI-N", "lfgi-N"),
        "normal-lfgi": ("LFGI-N", "lfgi-N"),
        "normal_lfgi": ("LFGI-N", "lfgi-N"),
        "gaussian-lfgi": ("LFGI-N", "lfgi-N"),
        "gaussian_lfgi": ("LFGI-N", "lfgi-N"),
        "standard-normal-lfgi": ("LFGI-N", "lfgi-N"),
        "standard_normal_lfgi": ("LFGI-N", "lfgi-N"),
        "n-lfgi": ("LFGI-N", "lfgi-N"),
        "n_lfgi": ("LFGI-N", "lfgi-N"),
        "leaf-lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "leaf_lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp-leaf-lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp_leaf_lfgi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp-leaf-ce-hlsi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "mp_leaf_ce_hlsi": ("Leaf-LFGI", "mp-leaf-lfgi"),
        "tweedie": ("Tweedie", "tweedie"),
        "twd": ("Tweedie", "tweedie"),
        "tsi": ("TSI", "tsi"),
        "target-score": ("TSI", "tsi"),
        "target_score": ("TSI", "tsi"),
        "target-side-information": ("TSI", "tsi"),
        "target_side_information": ("TSI", "tsi"),
        "target-side-score": ("TSI", "tsi"),
        "target_side_score": ("TSI", "tsi"),
        "identity-gate": ("TSI", "tsi"),
        "identity_gate": ("TSI", "tsi"),
        "none": ("None", "none"),
        "nome": ("None", "none"),
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


def _parse_transport_spec(text: str) -> Optional[Tuple[str, str, int]]:
    """Parse a transport estimator alias with optional trailing -<n>."""
    raw = str(text).strip().lower()
    spec = _normalize_estimator_alias(raw)
    if spec is not None:
        disp, method = spec
        return disp, method, 1
    import re
    m = re.match(r"^(.+)-([0-9]+)$", raw)
    if m is None:
        return None
    base, n_raw = m.group(1), m.group(2)
    n = int(n_raw)
    if n < 1:
        raise ValueError(f"transport repeat count must be >=1 in {text!r}")
    spec = _normalize_estimator_alias(base)
    if spec is None:
        return None
    disp, method = spec
    return disp, method, n


def _parse_score_pair_token(token: str) -> Tuple[str, str, str, int]:
    """Resolve the transport/PF-score prefix of one method token.

    Atomic tokens such as ``blend`` or ``lfgi`` mean diagonal pairs with one
    transport step.  Hybrid tokens use transport_correction order, for example
    ``blend_lfgi`` means Blend for transport and LFGI for likelihood correction.

    Multi-transport tokens attach a count to the transport alias:
        ``lfgi-2_lfgi`` means LFGI transport twice, then LFGI ratio correction.
        ``blend-3_lfgi`` means three Blend transports, then LFGI correction.

    The current names are exactly the ``n=1`` case: ``lfgi_lfgi`` is equivalent
    to ``lfgi-1_lfgi``.
    """
    raw = str(token).strip().lower()
    if not raw:
        raise ValueError("empty method token")

    # Explicit separators first; underscores are handled below because aliases
    # such as ce_hlsi and leaf_lfgi also contain underscores.
    for sep in ("->", ":", "/"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            lspec = _parse_transport_spec(left)
            rspec = _normalize_estimator_alias(right)
            if lspec is None or rspec is None:
                break
            ldisp, lmethod, repeats = lspec
            rdisp, rmethod = rspec
            if repeats == 1:
                return f"{ldisp}->{rdisp}", lmethod, rmethod, repeats
            return f"{ldisp}x{repeats}->{rdisp}", lmethod, rmethod, repeats

    # Atomic diagonal alias, with optional repeat count: lfgi or lfgi-3.
    lspec = _parse_transport_spec(raw)
    if lspec is not None:
        disp, method, repeats = lspec
        if repeats == 1:
            return f"{disp}->{disp}", method, method, repeats
        return f"{disp}x{repeats}->{disp}", method, method, repeats

    # Hybrid underscore syntax.  Try every split position so ce_hlsi-2_blend and
    # blend_leaf_lfgi can still be parsed unambiguously.
    parts = raw.split("_")
    for i in range(1, len(parts)):
        left = "_".join(parts[:i])
        right = "_".join(parts[i:])
        lspec = _parse_transport_spec(left)
        rspec = _normalize_estimator_alias(right)
        if lspec is not None and rspec is not None:
            ldisp, lmethod, repeats = lspec
            rdisp, rmethod = rspec
            if repeats == 1:
                return f"{ldisp}->{rdisp}", lmethod, rmethod, repeats
            return f"{ldisp}x{repeats}->{rdisp}", lmethod, rmethod, repeats

    valid = ", ".join(sorted(_estimator_alias_table().keys()) + ["all", "hybrids"])
    raise ValueError(
        f"Unknown transport/PF-score token {token!r}. Use atomic aliases or transport_pf-score "
        f"tokens like blend_lfgi, lfgi_none, os-lfgi_none, lfgi_os-lfgi, pi-lfgi_none, lfgi-N_none, none_lfgi, tweedie_lfgi, tsi_tsi, or the "
        f"multi-transport form lfgi-2_lfgi. Valid estimator aliases: {valid}"
    )


def _parse_method_token(token: str) -> Tuple[str, str, str, int, str, int]:
    """Parse the full ratio-node method grammar.

    Canonical form::

        <transport estimator>-<transport repeats>_<ratio estimator>_<ratio method>-<ratio rounds>

    ``raw-w`` accepts and ignores an optional final integer; all moved-flow nodes use it.  For backward
    compatibility, a token with no explicit ratio suffix is interpreted as the
    old transport/PF pair followed by ``raw-w``.
    """
    import re

    raw = str(token).strip().lower()
    if not raw:
        raise ValueError("empty method token")
    # Match the supported ratio modes at the right edge.  The prefix
    # is then delegated to the existing alias-aware pair parser, so estimator
    # names containing underscores remain unambiguous.
    m = re.match(
        r"^(.*)_((?:gated|raw|completed|full[-_]lfgi|tilted[-_]lfgi)[-_](?:pflow|bflow)|raw[-_]w)(?:-([0-9]+)|_([0-9]+))?$",
        raw,
    )
    if m is None:
        fam, transport, pf_method, repeats = _parse_score_pair_token(raw)
        ratio_method = "raw-w"
        ratio_rounds = 1
    else:
        prefix, ratio_raw = m.group(1), m.group(2)
        # Group 3 is the canonical hyphenated count; group 4 accepts the
        # earlier underscore spelling only for backward compatibility.
        rounds_raw = m.group(3) if m.group(3) is not None else m.group(4)
        fam, transport, pf_method, repeats = _parse_score_pair_token(prefix)
        ratio_method = canonical_ratio_method(ratio_raw)
        ratio_rounds = int(rounds_raw) if rounds_raw is not None else 1
        if ratio_method in {"gated-bflow", "raw-bflow", "completed-bflow"} and canonical_score_method_key(pf_method) == "none":
            raise ValueError(
                f"{ratio_method} requires a non-none method-2 carrier/gate in {token!r}"
            )
        if ratio_rounds < 1:
            raise ValueError(f"ratio round count must be >=1 in {token!r}")
        if is_moved_ratio_flow(ratio_method) and rounds_raw is None:
            # Defaulting to one is convenient and unambiguous, while the
            # displayed label still records the effective count.
            ratio_rounds = 1
    ratio_disp = {
        "gated-pflow": "GatedPFlow",
        "gated-bflow": "GatedBFlow",
        "raw-bflow": "RawBFlow",
        "completed-bflow": "CompletedBFlow",
        "raw-w": "Raw-W",
    }[ratio_method]
    family = f"{fam}->{ratio_disp}"
    if is_moved_ratio_flow(ratio_method):
        family = f"{family}x{ratio_rounds}"
    return family, transport, pf_method, repeats, ratio_method, ratio_rounds


def selected_method_specs(methods: str) -> List[Tuple[str, str, str, int, str, int]]:
    """Resolve comma-separated method specifications.

    Returns ``(family, transport_method, pf_method, transport_repeats,
    ratio_method, ratio_rounds)``.
    """
    raw = str(methods or "hybrids").strip().lower()
    if raw in {"all", "default", "*"}:
        keys = ["blend", "matrix_blend", "unif_blend", "unif_matrix_blend", "lfgi", "os-lfgi", "pi-lfgi", "lfgi-N", "leaf-lfgi", "tweedie", "tsi"]
    elif raw in {"hybrid", "hybrids", "blend-lfgi-hybrids", "lfgi-blend-hybrids"}:
        keys = ["blend_blend", "blend_lfgi", "lfgi_blend", "lfgi_lfgi"]
    elif raw in {"alternating", "strict-alternation", "strict_alternation", "gated-grid", "ratio-grid", "gate-grid", "gate-comparison"}:
        atoms = ["blend", "matrix_blend", "unif_blend", "unif_matrix_blend", "lfgi", "os-lfgi", "pi-lfgi", "lfgi-N", "leaf-lfgi", "tweedie", "tsi"]
        keys = [f"{a}_{a}_gated-pflow-1" for a in atoms]
    elif raw in {"bflow-alternating", "bflow-grid", "shared-bflow-grid", "gated-bflow-grid"}:
        atoms = ["blend", "matrix_blend", "unif_blend", "unif_matrix_blend", "lfgi", "os-lfgi", "pi-lfgi", "lfgi-N", "leaf-lfgi", "tweedie", "tsi"]
        keys = [f"{a}_{a}_gated-bflow-1" for a in atoms]
    elif raw in {"grid", "full-grid", "fullgrid", "full", "allpairs", "all-pairs"}:
        atoms = ["blend", "matrix_blend", "unif_blend", "unif_matrix_blend", "lfgi", "os-lfgi", "pi-lfgi", "lfgi-N", "leaf-lfgi", "tweedie", "tsi", "none"]
        keys = [f"{a}_{b}" for a in atoms for b in atoms]
    else:
        keys = [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]
    out: List[Tuple[str, str, str, int, str, int]] = []
    seen = set()
    for key in keys:
        fam, transport, pf_method, repeats, ratio_method, ratio_rounds = _parse_method_token(key)
        unique = (transport, int(repeats), pf_method, ratio_method, int(ratio_rounds))
        if unique not in seen:
            out.append((fam, transport, pf_method, int(repeats), ratio_method, int(ratio_rounds)))
            seen.add(unique)
    if not out:
        raise ValueError(
            "No methods selected. Example: --methods "
            "lfgi_lfgi_gated-bflow-1,lfgi_lfgi_completed-bflow-1,"
            "lfgi_lfgi_raw-bflow-1,lfgi_lfgi_raw-w "
            "or --methods bflow-alternating"
        )
    return out


def make_target(cfg: Config, device: torch.device, dtype: torch.dtype):
    key = str(cfg.target).strip().lower().replace("-", "_")
    if key in {"stiff_misaligned_gmm3d", "stiff_gmm3d", "gate_comparison_gmm3d", "d3_gate_gmm", "gmm3d"}:
        return StiffMisalignedGMM3D(
            normalize=bool(cfg.normalize_target),
            device=device,
            dtype=dtype,
        )
    if key in {"gmm_16", "gmm16", "singular_gmm16", "singular_gmm_16", "rank3_gmm16"}:
        # Dedicated finite-bank stress target: every component has only three
        # appreciable local covariance directions in ambient dimension 16.
        # The component subspaces are independently rotated, then the full
        # mixture is whitened.  Consequently N(0,I) has the correct global
        # moments but extremely poor local overlap with the thin component
        # tubes—the intended importance-weight/conditional-ESS failure mode.
        target = MisalignedSubspaceGMM(
            d=16,
            rank=3,
            n_components=8,
            seed=cfg.target_seed,
            radius=cfg.radius,
            sigma_perp=cfg.sigma_perp,
            jitter=cfg.jitter,
            normalize=cfg.normalize_target,
            device=device,
            dtype=dtype,
        )
        target.name = "gmm_16"
        target.target_info = lambda target=target: {
            "target_name": "gmm_16",
            "target_type": "singular_subspace_gmm_stress",
            "target_dim": int(target.d),
            "gmm_rank": int(target.rank),
            "gmm_n_components": int(target.K),
            "gmm_sigma_perp": float(target.sigma_perp),
            "gmm_radius": float(target.radius),
            "gmm_jitter": float(target.jitter),
            "gmm_normalized": bool(target.normalized),
            "stress_comparison": "gated-bflow_vs_completed-bflow_vs_raw-bflow_vs_raw-w",
        }
        return target
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

    toy_common = dict(
        seed=int(cfg.target_seed),
        normalize=bool(cfg.normalize_target),
        norm_samples=int(cfg.toy_norm_samples),
        norm_eig_floor=float(cfg.toy_norm_eig_floor),
        score_bank_size=int(cfg.toy_score_bank),
        score_chunk=int(cfg.toy_score_chunk),
        hessian_chunk=int(cfg.toy_hessian_chunk),
        device=device,
        dtype=dtype,
    )

    if key in {"banana", "banana_2d", "curved_banana"}:
        return BananaTarget2D(
            bend=float(cfg.banana_bend),
            normal_std=float(cfg.banana_normal_std),
            **toy_common,
        )

    if key in {"sine", "sine_2d", "sinusoid", "wave", "wavy"}:
        return SineTarget2D(
            amplitude=float(cfg.sine_amplitude),
            frequency=float(cfg.sine_frequency),
            normal_std=float(cfg.sine_normal_std),
            **toy_common,
        )

    if key in {"ring", "ring_2d", "single_ring"}:
        return RadialShellTarget2D(
            name="ring",
            radii=(float(cfg.ring_radius),),
            radial_stds=(float(cfg.ring_radial_std),),
            **toy_common,
        )

    if key in {"rings", "rings_2d", "double_ring", "two_rings", "concentric_rings"}:
        return RadialShellTarget2D(
            name="rings",
            radii=(float(cfg.rings_inner_radius), float(cfg.rings_outer_radius)),
            radial_stds=(float(cfg.rings_radial_std), float(cfg.rings_radial_std)),
            **toy_common,
        )

    if key in {"double_well", "doublewell", "double_well_2d", "dw2"}:
        return DoubleWellTarget2D(
            barrier=float(cfg.double_well_barrier),
            bend=float(cfg.double_well_bend),
            normal_std=float(cfg.double_well_normal_std),
            **toy_common,
        )

    if key in {"spiral", "spiral_2d", "archimedean_spiral"}:
        return SpiralTarget2D(
            turns=float(cfg.spiral_turns),
            r_min=float(cfg.spiral_r_min),
            r_max=float(cfg.spiral_r_max),
            u_std=float(cfg.spiral_u_std),
            logradial_std=float(cfg.spiral_logradial_std),
            **toy_common,
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

    raise ValueError(
        "Unknown --target {!r}. Use stiff_misaligned_gmm3d, misaligned_gmm, gmm_16, funnel_d10, banana, sine, ring, rings, "
        "spiral, double_well, lj13_2d, or dw4_16d.".format(cfg.target)
    )

def validate_certified_positive_resolvent_configuration(
    method_specs: List[Tuple[str, str, str, int, str, int]],
    cfg: Config,
) -> None:
    """Reject a known-invalid certified PF gate regularization.

    With raw observed information, the LFGI denominator
        alpha^2 I + gamma Hbar(y,t)
    can cross zero on nonconvex targets.  Inverting first and then hard-flooring
    the resulting gate eigenvalues makes G(y,t) jump across that pole.  A
    Hutchinson finite difference then correctly reports an enormous/divergent
    derivative; clipping that derivative would no longer certify the generated
    law.  Use a PSD curvature surrogate or MP-leaf precision completion instead.
    """
    correction_enabled = any(
        (str(ratio_mode).lower() != "raw-w") or (str(pf_method).lower() != "none")
        for _fam, _tm, pf_method, _rep, ratio_mode, _rr in method_specs
    )
    dynamics = canonical_transport_dynamics(
        getattr(cfg, "transport_dynamics", "auto"),
        correction_enabled=correction_enabled,
    )
    if dynamics != "probability-flow":
        return
    if str(getattr(cfg, "curvature_mode", "raw")).strip().lower() != "raw":
        return
    if not math.isfinite(float(getattr(cfg, "gate_min_eval", -float("inf")))):
        return
    risky = []
    for family, transport_method, pf_method, _rep, ratio_mode, _rr in method_specs:
        if is_raw_lfgi_resolvent_method(transport_method) or (
            is_moved_ratio_flow(ratio_mode) and is_raw_lfgi_resolvent_method(pf_method)
        ):
            risky.append(family)
    if not risky:
        return
    if bool(getattr(cfg, "allow_raw_lfgi_hard_projection_certified", False)):
        print(
            "WARNING: allowing raw LFGI with hard post-resolvent gate projection in a certified PF run. "
            "The field may be discontinuous at resolvent poles and its Liouville density may be unreliable.",
            flush=True,
        )
        return
    names = ", ".join(risky)
    raise ValueError(
        "Certified PF rejected raw LFGI plus finite gate_min_eval for: " + names + ". "
        "On a nonconvex target the raw localized observed-information resolvent can cross zero; "
        "inverting it and then hard-flooring the gate creates a discontinuous field, so the PF "
        "divergence is not an ordinary bounded function. This was previously hidden by divergence "
        "clipping. Use the recommended stable configuration `--curvature_mode psd "
        "--gate_min_eval=-inf`, or use the MP-leaf estimator (`mp-leaf-lfgi`) with "
        "`--gate_min_eval=-inf`. The legacy behavior can be forced with "
        "`--allow_raw_lfgi_hard_projection_certified`, but its density certificate should not be trusted."
    )


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
    if math.isfinite(float(getattr(cfg, "gate_min_eval", -float("inf")))) and str(cfg.pf_divergence).lower() in {"auto", "analytic_ce", "analytic"}:
        print(
            "Finite gate_min_eval changes the score field by hard spectral gate projection; "
            "analytic LFGI divergence no longer applies and the fallback is Hutchinson finite differences.",
            flush=True,
        )
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
        "effective_ratio_reference_mode": canonical_ratio_reference_mode(getattr(cfg, "ratio_reference_mode", "endpoint")),
        "effective_ratio_gate_policy": canonical_ratio_gate_policy(getattr(cfg, "ratio_gate_policy", "native")),
        "effective_ratio_tilt_policy": canonical_ratio_tilt_policy(getattr(cfg, "ratio_tilt_policy", "exact")),
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
        f"t_min={effective_time_bounds(cfg)[0]:.6g}, t_max={effective_time_bounds(cfg)[1]:.6g}, schedule={canonical_time_schedule(cfg.time_schedule)}, "
        f"gate_min_eval={cfg.gate_min_eval}",
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
    if bool(getattr(cfg, "force_no_likelihood_correction", False)):
        init_rho = torch.zeros_like(init_rho)
        ess, ess_frac = log_weight_ess(init_rho)
        init_info.update({
            "initial_weight_mode": "zero_forced_no_likelihood_correction",
            "initial_rho_ess": ess,
            "initial_rho_ess_frac": ess_frac,
        })
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
    if bool(getattr(cfg, "force_no_likelihood_correction", False)):
        # The theorem-5.3 numerical harness iterates only the score-to-transport
        # map q_{k+1}=F_A(q_k).  Keep each selected S-step estimator and its
        # transport repeat count, but force the projection/ratio node to identity.
        method_specs = [
            (f"{fam.split('->')[0]}->None", tm, "none", rep, "raw-w", 1)
            for fam, tm, _pm, rep, _rm, _rr in method_specs
        ]
    validate_certified_positive_resolvent_configuration(method_specs, cfg)
    print(
        "Selected methods: " + ", ".join([
            f"{fam} (S={tm}x{rep}, PF={pm}, ratio={rm}x{rr if is_moved_ratio_flow(rm) else 1})"
            for fam, tm, pm, rep, rm, rr in method_specs
        ]),
        flush=True,
    )
    for family, transport_method, pf_method, transport_repeats, ratio_method, ratio_rounds in method_specs:
        print(
            f"\n=== Running method: {family} "
            f"(S={transport_method}x{transport_repeats}, PF={pf_method}, "
            f"ratio={ratio_method}x{ratio_rounds if is_moved_ratio_flow(ratio_method) else 1}) ===",
            flush=True,
        )
        samples_by_round, rows, stages = run_family(
            family,
            transport_method,
            pf_method,
            transport_repeats,
            ratio_method,
            ratio_rounds,
            target,
            init_refs,
            init_rho,
            truth,
            cfg,
        )
        all_samples[family] = samples_by_round
        metric_rows.extend(rows)
        stage_rows.extend(stages)

    convergence_rows = combine_convergence_rows(metric_rows, stage_rows)
    write_csv(os.path.join(cfg.outdir, "metrics_by_round.csv"), metric_rows)
    write_csv(os.path.join(cfg.outdir, "stage_diagnostics.csv"), stage_rows)
    write_csv(os.path.join(cfg.outdir, "convergence_by_round.csv"), convergence_rows)
    save_heatmaps(cfg.outdir, target, truth, init_score_refs, all_samples, cfg)
    save_metric_curves(cfg.outdir, metric_rows)
    save_convergence_curves(cfg.outdir, convergence_rows)

    print("\nDone. Wrote:", flush=True)
    for name in ["config.json", "metrics_by_round.csv", "stage_diagnostics.csv", "convergence_by_round.csv", "heatmaps_final.png", "heatmaps_by_round.png", "metric_curves.png", "convergence_curves.png", "projection_basis.npz"]:
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
    # Backward-compatible readable aliases.  The certificate-consistent script
    # enables ratio flow by default; --force_no_likelihood_correction disables it.
    p.add_argument("--enable_likelihood_correction", dest="force_no_likelihood_correction", action="store_false")
    p.add_argument("--enable_ratio_flow", dest="force_no_likelihood_correction", action="store_false")
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
    ns.ratio_reference_mode = canonical_ratio_reference_mode(getattr(ns, "ratio_reference_mode", "endpoint"))
    ns.ratio_gate_policy = canonical_ratio_gate_policy(getattr(ns, "ratio_gate_policy", "native"))
    ns.ratio_tilt_policy = canonical_ratio_tilt_policy(getattr(ns, "ratio_tilt_policy", "exact"))
    # Validate spelling here; the correction-dependent auto resolution occurs
    # inside each family after the method token is known.
    _ = canonical_transport_dynamics(getattr(ns, "transport_dynamics", "auto"), correction_enabled=False)

    cfg = Config(**vars(ns))
    # Fail fast for invalid interval/schedule.
    effective_time_bounds(cfg)
    return cfg


if __name__ == "__main__":
    run(parse_args())
