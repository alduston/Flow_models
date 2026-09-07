# -*- coding: utf-8 -*-
"""Compute-matched MALA -> LFGI / Dirichlet test for the Darcy posterior.

The experiment constructs a sequence of N-particle MALA laws

    pi_MALA(C_1), ..., pi_MALA(C_n)

by snapshotting N parallel, prior-initialized MALA chains at increasing
per-chain transition budgets.  At every checkpoint the *same unweighted N
particles* are used to build the finite-bank LFGI and raw Dirichlet score
estimators.  Each estimator then transports a fresh N-particle cloud with the
GAD reverse-OU sampler.

For each posterior functional the available compute curves are:

  1. MALA,
  2. MALA -> LFGI,
  3. MALA -> Dirichlet,
  4. prior -> iterative GAD-LFGI, and
  5. prior -> iterative GAD-Dirichlet.

``--gate-methods`` accepts comma-separated combinations such as
``lfgi,GAD-lfgi``.  The first GAD-LFGI/Dirichlet output is exactly the matching
MALA-gate output at the zero-compute prior checkpoint: its bank and transported
cloud are computed once and shared.  Later GAD points recursively rebuild the
unweighted bank on the preceding transported cloud.  As many complete GAD
rounds as fit inside the existing C_n + max(K) compute horizon are retained.

Synchronized wall time is the default compute axis, so the gates may have
different overheads.  The existing portable PDE-work proxy and a
dimension-aware analytical FLOP count are retained in the output CSV; select
them with ``--compute-axis pde_proxy`` or ``--compute-axis flops``.  The
MALA-only trajectory is continued far enough to cover C_n + max(K), making
every refinement point comparable to MALA at matched total compute.

The default scientific configuration requested for the test is n=10 and
N=1000.  Its first checkpoint is an unevaluated Gaussian cloud at exactly zero
compute.  The ten checkpoints are spread across the full 0--70 transition
trajectory, including the adaptive transient, so the shifted gate curve has no
artificial early-compute hole.  MALA adapts for 60 transitions and is then held
fixed for the final base interval and the matched-compute extension.  The
reverse-time Heun solver uses 100 steps by default
(pass ``--transport-steps 200`` for the usual discretization-sensitivity
rerun).  A small smoke test can be run, for example, with

    python darcy_mala_lfgi_dirichlet_flop_frontier_20260821_v4.py \
        --grid-size 8 --latent-dim 4 --n-observations 20 --n-holdout 8 \
        --n-particles 8 --n-budgets 2 --mala-budget-min 1 \
        --mala-budget-max 2 --mala-adapt-steps 1 --transport-steps 2 \
        --reference-rounds 2 --extension-points 1 --eval-chunk 4

The script expects the current ``gad_sampling.py`` beside it (or importable on
PYTHONPATH).  No change to that helper is required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # notebook / pasted-cell fallback
    THIS_DIR = os.getcwd()

REPO_ROOT = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from gad_sampling import (
    GaussianPrior,
    ScoreField,
    configure_sampling,
    device,
    make_physics_likelihood,
    precompute_reference_bank,
    run_reverse_ou_heun,
)


jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)


METHOD_LABELS = {
    "mcmc": "MALA",
    "lfgi": "MALA → LFGI",
    "dirichlet": "MALA → Dirichlet",
    "gad_lfgi": "GAD-LFGI",
    "gad_dirichlet": "GAD-Dirichlet",
}
METHOD_COLORS = {
    "mcmc": "#202020",
    "lfgi": "#2f6fdd",
    "dirichlet": "#d97919",
    "gad_lfgi": "#159d82",
    "gad_dirichlet": "#8b5bb5",
}
METHOD_MARKERS = {
    "mcmc": "o",
    "lfgi": "s",
    "dirichlet": "^",
    "gad_lfgi": "D",
    "gad_dirichlet": "P",
}
METHOD_LINESTYLES = {
    "mcmc": "-",
    "lfgi": "-",
    "dirichlet": "-",
    "gad_lfgi": "--",
    "gad_dirichlet": "--",
}
METHOD_ORDER = ("mcmc", "lfgi", "gad_lfgi", "dirichlet", "gad_dirichlet")
SELECTABLE_METHODS = ("lfgi", "dirichlet", "gad_lfgi", "gad_dirichlet")
GATE_FAMILY = {
    "lfgi": "lfgi",
    "dirichlet": "dirichlet",
    "gad_lfgi": "lfgi",
    "gad_dirichlet": "dirichlet",
}


def parse_gate_methods(value: str) -> Tuple[str, ...]:
    """Parse case-insensitive comma-separated refinement arms."""
    aliases = {
        "lfgi": ("lfgi",),
        "mala-lfgi": ("lfgi",),
        "dirichlet": ("dirichlet",),
        "mala-dirichlet": ("dirichlet",),
        "gad-lfgi": ("gad_lfgi",),
        "gadlfgi": ("gad_lfgi",),
        "gad-dirichlet": ("gad_dirichlet",),
        "gaddirichlet": ("gad_dirichlet",),
        "both": ("lfgi", "dirichlet"),
        "gad-both": ("gad_lfgi", "gad_dirichlet"),
        "all": SELECTABLE_METHODS,
    }
    requested = []
    for raw_token in str(value).split(","):
        token = raw_token.strip().lower().replace("_", "-").replace(" ", "")
        if not token:
            continue
        if token not in aliases:
            expected = "lfgi, dirichlet, GAD-lfgi, GAD-dirichlet, both, gad-both, or all"
            raise argparse.ArgumentTypeError(
                f"Unknown --gate-methods token {raw_token!r}; expected {expected}."
            )
        requested.extend(aliases[token])
    selected = tuple(method for method in SELECTABLE_METHODS if method in requested)
    if not selected:
        raise argparse.ArgumentTypeError("--gate-methods must select at least one arm.")
    return selected


FUNCTIONAL_SPECS = [
    (
        "MeanWhitenedRMSE",
        "Posterior mean\n(whitened RMSE)",
    ),
    (
        "CovarianceRelFro",
        "Posterior covariance\n(relative Frobenius error)",
    ),
    (
        "LogKMeanRelL2",
        "Mean log-permeability field\n(relative $L^2$ error)",
    ),
    (
        "LogKStdRelL2",
        "Log-permeability uncertainty\n(relative std-field error)",
    ),
    (
        "KMeanRelL2",
        "Mean permeability field\n(relative $L^2$ error)",
    ),
    (
        "HoldoutPressureAtMeanRelL2",
        "Held-out pressure at posterior mean\n(relative $L^2$ error)",
    ),
]


@dataclass(frozen=True)
class FlopModel:
    """Dimension-aware analytical FLOP proxy for the three compute paths.

    These are algorithmic operation counts, not hardware throughput estimates.
    The Darcy implementation uses a dense solve on the interior grid.  We count
    one such solve as 2/3 m^3 + 2 m^2 FLOPs (LU plus triangular solves), a
    reverse-mode value/gradient as two forward equivalents, and the JAX
    ``jacfwd`` Gauss--Newton construction as a configurable number of forward
    equivalents.  The default is d+1, reflecting a primal plus d directional
    sensitivity passes.  Gate-transport counts include every finite-bank
    contraction and all 2*S+1 score evaluations made by Heun.
    """

    grid_size: int
    interior_dim: int
    latent_dim: int
    n_observations: int
    n_particles: int
    n_reference: int
    transport_steps: int
    gn_forward_equivalents: float
    dense_solve_flops: float
    forward_flops: float
    posterior_value_grad_particle_flops: float
    gn_hessian_particle_flops: float
    bank_particle_flops: float
    bank_total_flops: float
    mala_transition_particle_flops: float
    lfgi_score_evaluation_flops: float
    dirichlet_score_evaluation_flops: float
    lfgi_transport_flops: float
    dirichlet_transport_flops: float

    def transport_flops(self, method: str) -> float:
        if method == "lfgi":
            return self.lfgi_transport_flops
        if method == "dirichlet":
            return self.dirichlet_transport_flops
        raise ValueError(f"Unknown gate method for FLOP accounting: {method!r}")

    def gate_flops(self, method: str) -> float:
        return self.bank_total_flops + self.transport_flops(method)


def build_flop_model(args: argparse.Namespace) -> FlopModel:
    m = int((args.grid_size - 2) ** 2)
    g2 = int(args.grid_size ** 2)
    d = int(args.latent_dim)
    o = int(args.n_observations)
    n = int(args.n_particles)
    r = int(args.n_particles)  # shared, unweighted signal/gate bank
    steps = int(args.transport_steps)

    dense_solve = (2.0 / 3.0) * float(m) ** 3 + 2.0 * float(m) ** 2
    # Basis projection, permeability/stencil assembly, and residual formation.
    forward = dense_solve + 2.0 * g2 * d + 25.0 * m + 3.0 * o
    posterior_value_grad = 2.0 * forward + 20.0 * d + 5.0 * o
    gn_equiv = (
        float(args.gn_forward_equivalents)
        if args.gn_forward_equivalents > 0.0
        else float(d + 1)
    )
    gn_hessian = gn_equiv * forward + 2.0 * o * d * d
    # Bank construction separately evaluates log likelihood, its gradient, the
    # GN Hessian, and a dxd symmetric eigendecomposition per reference point.
    bank_particle = (
        forward
        + posterior_value_grad
        + gn_hessian
        + 12.0 * d ** 3
        + 30.0 * d * d
    )
    bank_total = r * bank_particle

    # Proposal/drift, two Gaussian log kernels, MH test, and masked state update.
    mala_transition_particle = 30.0 * d + 20.0

    # Conditional weights/signals and the two conditional score means.
    conditional = 12.0 * n * r * d + 8.0 * n * r
    lfgi_score_eval = (
        conditional
        + 2.0 * n * r * d * d
        + 12.0 * n * d ** 3
        + 2.0 * n * d * d
    )
    dirichlet_score_eval = (
        conditional
        + 4.0 * n * r * d * d
        + 3.0 * n * d ** 3
        + 2.0 * n * d * d
    )
    score_evaluations = 2 * steps + 1
    heun_state_updates = steps * 20.0 * n * d
    lfgi_transport = score_evaluations * lfgi_score_eval + heun_state_updates
    # H_i^2 is cached once per Dirichlet field.
    dirichlet_transport = (
        score_evaluations * dirichlet_score_eval
        + 2.0 * r * d ** 3
        + heun_state_updates
    )
    return FlopModel(
        grid_size=args.grid_size,
        interior_dim=m,
        latent_dim=d,
        n_observations=o,
        n_particles=n,
        n_reference=r,
        transport_steps=steps,
        gn_forward_equivalents=gn_equiv,
        dense_solve_flops=dense_solve,
        forward_flops=forward,
        posterior_value_grad_particle_flops=posterior_value_grad,
        gn_hessian_particle_flops=gn_hessian,
        bank_particle_flops=bank_particle,
        bank_total_flops=bank_total,
        mala_transition_particle_flops=mala_transition_particle,
        lfgi_score_evaluation_flops=lfgi_score_eval,
        dirichlet_score_evaluation_flops=dirichlet_score_eval,
        lfgi_transport_flops=lfgi_transport,
        dirichlet_transport_flops=dirichlet_transport,
    )


def format_flops(value: float) -> str:
    value = float(value)
    if value >= 1e15:
        return f"{value / 1e15:.3f} PFLOP"
    if value >= 1e12:
        return f"{value / 1e12:.3f} TFLOP"
    if value >= 1e9:
        return f"{value / 1e9:.3f} GFLOP"
    if value >= 1e6:
        return f"{value / 1e6:.3f} MFLOP"
    return f"{value:.0f} FLOP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MALA, MALA->gate, and prior-initialized iterative GAD "
            "posterior-functional accuracy at matched total compute on the "
            "Darcy inverse problem."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-observations", type=int, default=120)
    parser.add_argument("--n-holdout", type=int, default=30)
    parser.add_argument("--noise-std", type=float, default=1e-3)

    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument("--n-budgets", type=int, default=10)
    # A zero lower endpoint distributes all n checkpoints across the complete
    # finite-compute trajectory.  Positive values retain the old convention of
    # adding zero and distributing the remaining n-1 checkpoints over [min,max].
    parser.add_argument(
        "--mala-budget-min",
        type=int,
        default=0,
        help=(
            "Smallest transition checkpoint. With the default 0 and linear "
            "spacing, all n checkpoints are spread uniformly over [0,max]."
        ),
    )
    parser.add_argument("--mala-budget-max", type=int, default=70)
    parser.add_argument(
        "--mala-budget-steps",
        type=str,
        default="",
        help="Optional comma-separated transition checkpoints; overrides min/max/count.",
    )
    parser.add_argument(
        "--budget-spacing", choices=("linear", "log"), default="linear"
    )
    parser.add_argument("--mala-dt", type=float, default=1e-4)
    parser.add_argument(
        "--mala-adapt-steps",
        type=int,
        default=60,
        help=(
            "Dual-averaging transitions. The longer default lets tuning track "
            "the increasing posterior stiffness as chains leave the prior."
        ),
    )
    parser.add_argument("--mala-target-accept", type=float, default=0.574)
    parser.add_argument("--mala-min-dt", type=float, default=1e-12)
    parser.add_argument("--mala-max-dt", type=float, default=1e-1)
    parser.add_argument(
        "--min-post-adapt-acceptance",
        type=float,
        default=0.40,
        help=(
            "Fail before gate construction if a wholly post-adaptation checkpoint "
            "window falls below this acceptance rate; set to 0 to disable."
        ),
    )

    # One hundred log-time Heun steps is the suite's conservative lower end;
    # use 200 explicitly for a reversal-discretization sensitivity run.
    parser.add_argument("--transport-steps", type=int, default=100)
    parser.add_argument("--t-min", type=float, default=10 ** (-2.5))
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--eval-chunk", type=int, default=64)
    parser.add_argument("--hess-min", type=float, default=1e-6)
    parser.add_argument("--hess-max", type=float, default=1e6)
    parser.add_argument("--curvature-ridge", type=float, default=1e-6)
    parser.add_argument("--likelihood-batch-size", type=int, default=50)
    parser.add_argument("--gradient-batch-size", type=int, default=25)
    parser.add_argument("--hessian-batch-size", type=int, default=2)
    parser.add_argument(
        "--gn-forward-equivalents",
        type=float,
        default=0.0,
        help=(
            "Forward-solve equivalents charged to one jacfwd GN Hessian. "
            "The default 0 selects latent_dim+1."
        ),
    )
    parser.add_argument(
        "--gate-methods",
        type=parse_gate_methods,
        default=parse_gate_methods("both"),
        metavar="METHODS",
        help=(
            "Comma-separated refinement arms: lfgi, dirichlet, GAD-lfgi, "
            "GAD-dirichlet. Shorthands: both, gad-both, all."
        ),
    )
    parser.add_argument(
        "--max-gad-rounds",
        type=int,
        default=50,
        help=(
            "Safety cap on retained transport-only GAD outputs, including round "
            "zero. The compute horizon normally stops the loop first."
        ),
    )

    parser.add_argument(
        "--reference-rounds",
        type=int,
        default=3,
        help=(
            "Total GAD rounds used for the functional reference, including the "
            "already-computed seed refinement. With both gate families this is "
            "applied to two balanced alternating branches."
        ),
    )
    parser.add_argument(
        "--reference-samples",
        type=str,
        default="",
        help="Optional .npy/.npz/.pt/.pth/.csv reference cloud; skips GAD refinement.",
    )
    parser.add_argument(
        "--extension-points",
        type=int,
        default=10,
        help="Number of extra MALA-only snapshots across the shifted-compute tail.",
    )
    parser.add_argument(
        "--max-extension-transitions",
        type=int,
        default=2000,
        help="Safety cap on MALA transitions added after the largest base budget.",
    )

    parser.add_argument(
        "--compute-axis",
        choices=("wall_seconds", "pde_proxy", "flops"),
        default="wall_seconds",
        help=(
            "Common compute unit used for MALA extension, matched-compute gains, "
            "and the plot x-axis. 'flops' uses the analytical model recorded in "
            "run_config.json."
        ),
    )
    parser.add_argument("--xscale", choices=("linear", "log"), default="linear")
    parser.add_argument("--output-root", type=str, default="run_results")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.grid_size < 5:
        raise ValueError("--grid-size must be at least 5.")
    if args.latent_dim <= 0 or args.latent_dim > args.grid_size ** 2:
        raise ValueError("--latent-dim must lie in [1, grid_size**2].")
    n_interior = (args.grid_size - 2) ** 2
    if args.n_observations <= 0 or args.n_holdout <= 0:
        raise ValueError("Training and held-out observation counts must be positive.")
    if args.n_observations + args.n_holdout > n_interior:
        raise ValueError(
            "Training plus held-out observations exceed the number of interior grid points."
        )
    if args.noise_std <= 0.0:
        raise ValueError("--noise-std must be positive.")
    if args.n_particles < 2:
        raise ValueError("--n-particles must be at least 2 for covariance estimation.")
    if args.n_budgets <= 0:
        raise ValueError("--n-budgets must be positive.")
    if args.mala_budget_min < 0 or args.mala_budget_max < args.mala_budget_min:
        raise ValueError("Invalid MALA budget interval.")
    if args.mala_adapt_steps < 0:
        raise ValueError("--mala-adapt-steps must be nonnegative.")
    if not 0.0 < args.mala_target_accept < 1.0:
        raise ValueError("--mala-target-accept must lie in (0,1).")
    if args.mala_dt <= 0.0 or args.mala_min_dt <= 0.0:
        raise ValueError("MALA step sizes must be positive.")
    if args.mala_max_dt < args.mala_min_dt:
        raise ValueError("--mala-max-dt must be at least --mala-min-dt.")
    if not 0.0 <= args.min_post_adapt_acceptance < 1.0:
        raise ValueError("--min-post-adapt-acceptance must lie in [0,1).")
    if args.transport_steps <= 0:
        raise ValueError("--transport-steps must be positive.")
    if args.gn_forward_equivalents < 0.0:
        raise ValueError("--gn-forward-equivalents must be nonnegative.")
    if args.max_gad_rounds < 1:
        raise ValueError("--max-gad-rounds must be at least 1.")
    if args.t_min <= 0.0 or args.t_max <= args.t_min:
        raise ValueError("Expected 0 < t_min < t_max.")
    if args.reference_rounds < 1:
        raise ValueError("--reference-rounds must be at least 1.")
    if args.extension_points < 1:
        raise ValueError("--extension-points must be at least 1.")
    if args.max_extension_transitions < 1:
        raise ValueError("--max-extension-transitions must be at least 1.")


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sync_torch() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    sync_torch()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    sync_torch()
    return result, time.perf_counter() - start


@dataclass
class DarcyProblem:
    grid_size: int
    latent_dim: int
    basis_np: np.ndarray
    obs_locations_train: np.ndarray
    obs_locations_holdout: np.ndarray
    solve_forward: object
    solve_forward_holdout: object


def build_darcy_problem(
    *,
    grid_size: int,
    latent_dim: int,
    n_observations: int,
    n_holdout: int,
    seed: int,
) -> DarcyProblem:
    """Build the same exponential-covariance KL Darcy problem as darcy_gad.py."""
    n_grid = int(grid_size)
    x = np.linspace(0.0, 1.0, n_grid)
    xx, yy = np.meshgrid(x, x)
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    ell = 0.1
    sigma_prior = 1.0
    dists = cdist(coords, coords)
    covariance = sigma_prior ** 2 * np.exp(-dists / ell)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:latent_dim]
    eigvecs = eigvecs[:, order][:, :latent_dim]
    basis_np = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
    basis = jnp.asarray(basis_np, dtype=jnp.float64)

    interior_mask_np = np.ones((n_grid, n_grid), dtype=bool)
    interior_mask_np[0, :] = False
    interior_mask_np[-1, :] = False
    interior_mask_np[:, 0] = False
    interior_mask_np[:, -1] = False
    interior_flat_np = np.where(interior_mask_np.ravel())[0]

    key_train = jax.random.PRNGKey(seed)
    obs_train = np.asarray(
        jax.random.choice(
            key_train,
            jnp.asarray(interior_flat_np),
            shape=(n_observations,),
            replace=False,
        )
    )
    remaining = np.setdiff1d(interior_flat_np, obs_train)
    key_holdout = jax.random.PRNGKey(seed + 1)
    obs_holdout = np.asarray(
        jax.random.choice(
            key_holdout,
            jnp.asarray(remaining),
            shape=(n_holdout,),
            replace=False,
        )
    )
    obs_train_jax = jnp.asarray(obs_train, dtype=jnp.int32)
    obs_holdout_jax = jnp.asarray(obs_holdout, dtype=jnp.int32)

    h = 1.0 / float(n_grid - 1)
    f_darcy = jnp.ones((n_grid, n_grid), dtype=jnp.float64)
    interior_mask = jnp.zeros((n_grid, n_grid), dtype=bool)
    interior_mask = interior_mask.at[1:-1, 1:-1].set(True)
    int_rows, int_cols = jnp.where(interior_mask)
    n_int = int(int_rows.shape[0])
    int_id = -jnp.ones((n_grid, n_grid), dtype=jnp.int32)
    int_id = int_id.at[int_rows, int_cols].set(jnp.arange(n_int, dtype=jnp.int32))
    int_flat = int_rows * n_grid + int_cols

    def assemble_darcy(k_field):
        h2 = h * h
        k_xp = (
            2.0
            * k_field[:-1, :]
            * k_field[1:, :]
            / (k_field[:-1, :] + k_field[1:, :] + 1e-30)
        )
        k_yp = (
            2.0
            * k_field[:, :-1]
            * k_field[:, 1:]
            / (k_field[:, :-1] + k_field[:, 1:] + 1e-30)
        )
        ir, ic = int_rows, int_cols
        c_e = k_xp[ir, ic] / h2
        c_w = k_xp[ir - 1, ic] / h2
        c_n = k_yp[ir, ic] / h2
        c_s = k_yp[ir, ic - 1] / h2
        diag = c_e + c_w + c_n + c_s
        idx = jnp.arange(n_int)
        nbr_e = int_id[ir + 1, ic]
        nbr_w = int_id[ir - 1, ic]
        nbr_n = int_id[ir, ic + 1]
        nbr_s = int_id[ir, ic - 1]
        matrix = jnp.zeros((n_int, n_int), dtype=jnp.float64)
        matrix = matrix.at[idx, idx].add(diag)
        matrix = matrix.at[idx, nbr_e].add(jnp.where(nbr_e >= 0, -c_e, 0.0))
        matrix = matrix.at[idx, nbr_w].add(jnp.where(nbr_w >= 0, -c_w, 0.0))
        matrix = matrix.at[idx, nbr_n].add(jnp.where(nbr_n >= 0, -c_n, 0.0))
        matrix = matrix.at[idx, nbr_s].add(jnp.where(nbr_s >= 0, -c_s, 0.0))
        rhs = f_darcy[int_rows, int_cols]
        return matrix, rhs

    def solve_at(alpha, locations):
        log_k = jnp.reshape(basis @ alpha, (n_grid, n_grid))
        k_field = jnp.exp(log_k)
        matrix, rhs = assemble_darcy(k_field)
        p_int = jnp.linalg.solve(matrix, rhs)
        p_full = jnp.zeros(n_grid * n_grid, dtype=jnp.float64)
        p_full = p_full.at[int_flat].set(p_int)
        return p_full[locations]

    solve_forward = jax.jit(lambda alpha: solve_at(alpha, obs_train_jax))
    solve_forward_holdout = jax.jit(lambda alpha: solve_at(alpha, obs_holdout_jax))

    return DarcyProblem(
        grid_size=n_grid,
        latent_dim=latent_dim,
        basis_np=np.asarray(basis_np, dtype=np.float64),
        obs_locations_train=obs_train,
        obs_locations_holdout=obs_holdout,
        solve_forward=solve_forward,
        solve_forward_holdout=solve_forward_holdout,
    )


def make_budget_steps(args: argparse.Namespace) -> List[int]:
    if args.mala_budget_steps.strip():
        tokens = [token.strip() for token in args.mala_budget_steps.split(",") if token.strip()]
        values = [int(token) for token in tokens]
        if not values or min(values) < 0:
            raise ValueError("--mala-budget-steps must contain nonnegative integers.")
        values = sorted(set(values))
        if len(values) != len(tokens):
            print("WARNING: duplicate MALA checkpoints were removed.")
        if 0 not in values:
            values.insert(0, 0)
            print("Prepended the required zero-compute Gaussian checkpoint.")
        return values

    if args.n_budgets == 1:
        return [0]

    if args.mala_budget_min == 0 and args.budget_spacing == "linear":
        if args.mala_budget_max + 1 < args.n_budgets:
            raise ValueError(
                "The integer budget interval [0,mala_budget_max] is too short "
                "for n_budgets distinct checkpoints."
            )
        raw = np.linspace(0, args.mala_budget_max, args.n_budgets)
        values = sorted(set(int(v) for v in np.rint(raw).astype(int)))
        if len(values) < args.n_budgets:
            available = [
                step
                for step in range(0, args.mala_budget_max + 1)
                if step not in values
            ]
            targets = np.linspace(
                0,
                max(0, len(available) - 1),
                args.n_budgets - len(values),
            )
            values.extend(available[int(round(idx))] for idx in targets)
            values = sorted(set(values))
        if len(values) != args.n_budgets or values[0] != 0:
            raise RuntimeError("Could not construct the requested full MALA grid.")
        return values

    positive_count = args.n_budgets - 1
    positive_min = max(1, int(args.mala_budget_min))
    if args.mala_budget_max - positive_min + 1 < positive_count:
        raise ValueError(
            "The positive integer budget interval is too short for n_budgets-1 "
            "distinct checkpoints in addition to the required zero checkpoint."
        )
    if args.budget_spacing == "log":
        raw = np.geomspace(positive_min, args.mala_budget_max, positive_count)
    else:
        raw = np.linspace(positive_min, args.mala_budget_max, positive_count)
    rounded = np.rint(raw).astype(int)
    values = sorted(set(int(v) for v in rounded))
    if len(values) < positive_count:
        available = [
            step
            for step in range(positive_min, args.mala_budget_max + 1)
            if step not in values
        ]
        targets = np.linspace(0, max(0, len(available) - 1), positive_count - len(values))
        values.extend(available[int(round(idx))] for idx in targets)
        values = sorted(set(values))
    if len(values) != positive_count:
        raise RuntimeError("Could not construct the requested positive MALA checkpoints.")
    return [0] + values


@dataclass
class MALAState:
    x: torch.Tensor
    log_post: Optional[torch.Tensor]
    score: Optional[torch.Tensor]
    transition: int
    active_seconds: float
    log_eps: float
    log_eps_bar: float
    mu: float
    h_bar: float
    accepted: int
    proposals: int
    initial_score_norm: float
    posterior_particle_evals: int


def posterior_log_density_and_score(x, prior_model, lik_model):
    log_lik, grad_lik = lik_model.log_likelihood_and_grad(x)
    return prior_model.log_prob(x) + log_lik, prior_model.score0(x) + grad_lik


def initialize_mala_state(
    n_particles: int,
    prior_model,
    lik_model,
    *,
    dt: float,
    min_dt: float,
    max_dt: float,
) -> MALAState:
    """Create the zero-compute Gaussian checkpoint without target evaluation."""
    dt_initial = min(max(float(dt), float(min_dt)), float(max_dt))
    eps = math.sqrt(2.0 * dt_initial)
    # Drawing the initial N(0,I) cloud defines C=0 by convention.  Its first
    # target value/score evaluation is charged when the first MALA transition
    # is actually requested.
    x = prior_model.sample(n_particles).detach().to(device=device, dtype=torch.float64)
    return MALAState(
        x=x,
        log_post=None,
        score=None,
        transition=0,
        active_seconds=0.0,
        log_eps=math.log(eps),
        log_eps_bar=math.log(eps),
        mu=math.log(10.0) + math.log(eps),
        h_bar=0.0,
        accepted=0,
        proposals=0,
        initial_score_norm=float("nan"),
        posterior_particle_evals=0,
    )


def mala_transition(
    state: MALAState,
    prior_model,
    lik_model,
    *,
    adapt_steps: int,
    target_accept: float,
    min_dt: float,
    max_dt: float,
) -> Dict[str, float]:
    """Advance every chain once and add only active sampler time to C."""
    eps_min = math.sqrt(2.0 * float(min_dt))
    eps_max = math.sqrt(2.0 * float(max_dt))
    eps = math.exp(state.log_eps)
    proposal_dt = 0.5 * eps * eps

    sync_torch()
    start = time.perf_counter()
    if state.log_post is None or state.score is None:
        with torch.no_grad():
            state.log_post, state.score = posterior_log_density_and_score(
                state.x, prior_model, lik_model
            )
        state.posterior_particle_evals += int(state.x.shape[0])
        state.initial_score_norm = float(
            torch.linalg.vector_norm(state.score, dim=1).mean().item()
        )
    x_old = state.x
    drift = proposal_dt * state.score
    proposal = state.x + drift + eps * torch.randn_like(state.x)
    with torch.no_grad():
        log_post_prop, score_prop = posterior_log_density_and_score(
            proposal, prior_model, lik_model
        )
    state.posterior_particle_evals += int(state.x.shape[0])

    reverse_drift = proposal_dt * score_prop
    log_q_forward = -torch.sum((proposal - state.x - drift) ** 2, dim=1) / (
        4.0 * proposal_dt
    )
    log_q_reverse = -torch.sum(
        (state.x - proposal - reverse_drift) ** 2, dim=1
    ) / (4.0 * proposal_dt)
    log_alpha = log_post_prop - state.log_post + log_q_reverse - log_q_forward
    proposal_finite = (
        torch.isfinite(log_post_prop)
        & torch.isfinite(score_prop).all(dim=1)
        & torch.isfinite(log_alpha)
    )
    current_finite = torch.isfinite(state.log_post) & torch.isfinite(state.score).all(dim=1)
    log_alpha = torch.where(
        proposal_finite, log_alpha, torch.full_like(log_alpha, -torch.inf)
    )
    log_alpha = torch.where(
        (~current_finite) & proposal_finite,
        torch.full_like(log_alpha, torch.inf),
        log_alpha,
    )
    clipped_log_alpha = torch.minimum(log_alpha, torch.zeros_like(log_alpha))
    accept_prob = torch.nan_to_num(
        torch.exp(clipped_log_alpha), nan=0.0, posinf=1.0, neginf=0.0
    )
    accept = torch.log(torch.rand(state.x.shape[0], device=state.x.device)) < clipped_log_alpha
    mask = accept.unsqueeze(1)
    state.x = torch.where(mask, proposal, state.x)
    state.log_post = torch.where(accept, log_post_prop, state.log_post)
    state.score = torch.where(mask, score_prop, state.score)
    sync_torch()
    state.active_seconds += time.perf_counter() - start

    state.transition += 1
    state.accepted += int(accept.sum().item())
    state.proposals += int(accept.numel())
    mean_accept_prob = float(accept_prob.mean().item())

    if state.transition <= int(adapt_steps) and adapt_steps > 0:
        da_gamma = 0.05
        da_t0 = 10.0
        da_kappa = 0.75
        m = state.transition
        eta = 1.0 / (float(m) + da_t0)
        state.h_bar = (1.0 - eta) * state.h_bar + eta * (
            float(target_accept) - mean_accept_prob
        )
        candidate = state.mu - (math.sqrt(float(m)) / da_gamma) * state.h_bar
        candidate = min(max(candidate, math.log(eps_min)), math.log(eps_max))
        avg_weight = float(m) ** (-da_kappa)
        state.log_eps_bar = (
            avg_weight * candidate + (1.0 - avg_weight) * state.log_eps_bar
        )
        state.log_eps = candidate
        if state.transition == int(adapt_steps):
            state.log_eps = min(
                max(state.log_eps_bar, math.log(eps_min)), math.log(eps_max)
            )

    jump_sq = float(torch.sum((state.x - x_old) ** 2, dim=1).mean().item())
    return {
        "proposal_dt": proposal_dt,
        "mean_accept_prob": mean_accept_prob,
        "acceptance_rate_step": float(accept.to(torch.float64).mean().item()),
        "mean_squared_jump": jump_sq,
    }


def mcmc_pde_proxy(state: MALAState, n_particles: int) -> int:
    del n_particles  # the state records the exact number of evaluated particles
    # Matches gad_sampling.py: likelihood + score are two portable units per
    # target-evaluated particle.  The Gaussian checkpoint has no evaluations.
    return int(2 * state.posterior_particle_evals)


def mcmc_flops(state: MALAState, flop_model: FlopModel) -> float:
    posterior = (
        float(state.posterior_particle_evals)
        * flop_model.posterior_value_grad_particle_flops
    )
    proposal_algebra = (
        float(state.transition)
        * flop_model.n_particles
        * flop_model.mala_transition_particle_flops
    )
    return posterior + proposal_algebra


def snapshot_mala(
    state: MALAState,
    n_particles: int,
    flop_model: FlopModel,
    *,
    budget_index: Optional[int],
    phase: str,
) -> Dict[str, object]:
    return {
        "method": "mcmc",
        "phase": phase,
        "budget_index": budget_index,
        "base_transition": int(state.transition),
        "samples": state.x.detach().cpu().clone(),
        "mcmc_wall_seconds": float(state.active_seconds),
        "gate_wall_seconds": 0.0,
        "total_wall_seconds": float(state.active_seconds),
        "mcmc_pde_proxy": mcmc_pde_proxy(state, n_particles),
        "gate_pde_proxy": 0,
        "total_pde_proxy": mcmc_pde_proxy(state, n_particles),
        "mcmc_flops": mcmc_flops(state, flop_model),
        "bank_flops": 0.0,
        "transport_flops": 0.0,
        "gate_flops": 0.0,
        "total_flops": mcmc_flops(state, flop_model),
        "acceptance_rate": float(state.accepted / max(1, state.proposals)),
        "acceptance_rate_cumulative": float(
            state.accepted / max(1, state.proposals)
        ),
        "acceptance_rate_window": float("nan"),
        "mala_dt": float(0.5 * math.exp(2.0 * state.log_eps)),
    }


def warm_up_compiled_kernels(lik_model, prior_model, n_particles: int) -> None:
    """Exclude one-time JAX compilation from the scientific compute curves."""
    print("Warming JAX/Torch kernels (excluded from compute accounting)...")
    n_log = min(max(1, int(getattr(lik_model, "log_batch_size", 50))), n_particles)
    n_grad = min(max(1, int(getattr(lik_model, "grad_batch_size", 25))), n_particles)
    n_hess = min(max(1, int(getattr(lik_model, "hess_batch_size", 2))), n_particles)
    x_log = prior_model.sample(n_log).detach().to(device=device, dtype=torch.float64)
    x_grad = x_log[:n_grad]
    x_hess = x_log[:n_hess]
    with torch.no_grad():
        _ = lik_model.log_likelihood(x_log)
        _ = lik_model.grad_log_likelihood(x_grad)
        _ = lik_model.log_likelihood_and_grad(x_grad)
        _ = lik_model.hess_log_likelihood(x_hess)
    sync_torch()


def save_rng_state() -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_state


def restore_rng_state(state: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def precompute_unweighted_bank(
    particles: torch.Tensor,
    prior_model,
    lik_model,
    *,
    label: str,
):
    particles = particles.detach().cpu().to(dtype=torch.float64).contiguous()
    zeros = torch.zeros(particles.shape[0], dtype=torch.float64)
    return timed_call(
        precompute_reference_bank,
        particles,
        prior_model,
        lik_model,
        label,
        zeros,
    )


def run_gate_transport(
    bank,
    method: str,
    *,
    n_particles: int,
    latent_dim: int,
    transport_steps: int,
    t_min: float,
    t_max: float,
    eval_chunk: int,
    seed: int,
):
    # Reset to common random numbers at a checkpoint: both gates see the same
    # terminal Gaussian cloud and reverse-SDE noise, reducing comparison noise.
    set_all_seeds(seed)

    def _run():
        score_field = ScoreField(bank, method, eval_chunk=eval_chunk)
        samples, _ess, flow_info = run_reverse_ou_heun(
            n_particles,
            score_field,
            steps=transport_steps,
            dim=latent_dim,
            t_min=t_min,
            t_max=t_max,
            log_mean_ess=False,
        )
        return samples, flow_info

    return timed_call(_run)


def make_gate_record(
    *,
    method: str,
    samples: torch.Tensor,
    base_record: Dict[str, object],
    bank_seconds: float,
    transport_seconds: float,
    n_particles: int,
    flop_model: FlopModel,
    flow_info: Dict[str, object],
) -> Dict[str, object]:
    gate_seconds = float(bank_seconds + transport_seconds)
    gate_proxy = int(3 * n_particles)  # likelihood + score + GN Hessian per bank point
    transport_flops = flop_model.transport_flops(method)
    gate_flops = flop_model.bank_total_flops + transport_flops
    return {
        "method": method,
        "phase": "gate",
        "budget_index": base_record["budget_index"],
        "base_transition": base_record["base_transition"],
        "samples": samples.detach().cpu().to(dtype=torch.float64).contiguous(),
        "mcmc_wall_seconds": float(base_record["mcmc_wall_seconds"]),
        "bank_wall_seconds": float(bank_seconds),
        "transport_wall_seconds": float(transport_seconds),
        "gate_wall_seconds": gate_seconds,
        "total_wall_seconds": float(base_record["mcmc_wall_seconds"]) + gate_seconds,
        "mcmc_pde_proxy": int(base_record["mcmc_pde_proxy"]),
        "gate_pde_proxy": gate_proxy,
        "total_pde_proxy": int(base_record["mcmc_pde_proxy"]) + gate_proxy,
        "mcmc_flops": float(base_record["mcmc_flops"]),
        "bank_flops": flop_model.bank_total_flops,
        "transport_flops": transport_flops,
        "gate_flops": gate_flops,
        "total_flops": float(base_record["mcmc_flops"]) + gate_flops,
        "acceptance_rate": float(base_record["acceptance_rate"]),
        "acceptance_rate_cumulative": float(
            base_record.get("acceptance_rate_cumulative", base_record["acceptance_rate"])
        ),
        "acceptance_rate_window": float(
            base_record.get("acceptance_rate_window", float("nan"))
        ),
        "mala_dt": float(base_record["mala_dt"]),
        "flow_score_norm": float(flow_info.get("score_norm", float("nan"))),
    }


def make_gad_record(
    *,
    method: str,
    round_index: int,
    samples: torch.Tensor,
    bank_seconds: float,
    transport_seconds: float,
    n_particles: int,
    flop_model: FlopModel,
    flow_info: Dict[str, object],
    previous_record: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Record one cumulative, transport-only GAD iterate.

    Round zero is the first prior-bank transport.  When the matching MALA gate
    arm is present, the caller passes the very same transported tensor and
    timings to both records, so the two curves coincide exactly at that point.
    """
    if method not in ("gad_lfgi", "gad_dirichlet"):
        raise ValueError(f"Not a transport-only GAD method: {method!r}")
    family = GATE_FAMILY[method]
    round_wall_seconds = float(bank_seconds + transport_seconds)
    round_pde_proxy = int(3 * n_particles)
    round_transport_flops = flop_model.transport_flops(family)
    round_flops = flop_model.bank_total_flops + round_transport_flops

    previous_wall = (
        0.0 if previous_record is None else float(previous_record["total_wall_seconds"])
    )
    previous_proxy = (
        0 if previous_record is None else int(previous_record["total_pde_proxy"])
    )
    previous_flops = (
        0.0 if previous_record is None else float(previous_record["total_flops"])
    )
    previous_bank_wall = (
        0.0
        if previous_record is None
        else float(previous_record.get("cumulative_bank_wall_seconds", 0.0))
    )
    previous_transport_wall = (
        0.0
        if previous_record is None
        else float(previous_record.get("cumulative_transport_wall_seconds", 0.0))
    )
    previous_bank_flops = (
        0.0
        if previous_record is None
        else float(previous_record.get("cumulative_bank_flops", 0.0))
    )
    previous_transport_flops = (
        0.0
        if previous_record is None
        else float(previous_record.get("cumulative_transport_flops", 0.0))
    )

    total_wall = previous_wall + round_wall_seconds
    total_proxy = previous_proxy + round_pde_proxy
    total_flops = previous_flops + round_flops
    return {
        "method": method,
        "phase": "gad",
        "budget_index": None,
        "base_transition": 0,
        "gad_round": int(round_index),
        "samples": samples.detach().cpu().to(dtype=torch.float64).contiguous(),
        "mcmc_wall_seconds": 0.0,
        "bank_wall_seconds": float(bank_seconds),
        "transport_wall_seconds": float(transport_seconds),
        "round_wall_seconds": round_wall_seconds,
        "cumulative_bank_wall_seconds": previous_bank_wall + float(bank_seconds),
        "cumulative_transport_wall_seconds": (
            previous_transport_wall + float(transport_seconds)
        ),
        "gate_wall_seconds": total_wall,
        "total_wall_seconds": total_wall,
        "mcmc_pde_proxy": 0,
        "round_pde_proxy": round_pde_proxy,
        "gate_pde_proxy": total_proxy,
        "total_pde_proxy": total_proxy,
        "mcmc_flops": 0.0,
        "bank_flops": flop_model.bank_total_flops,
        "transport_flops": round_transport_flops,
        "round_flops": round_flops,
        "cumulative_bank_flops": previous_bank_flops + flop_model.bank_total_flops,
        "cumulative_transport_flops": (
            previous_transport_flops + round_transport_flops
        ),
        "gate_flops": total_flops,
        "total_flops": total_flops,
        "acceptance_rate": float("nan"),
        "acceptance_rate_cumulative": float("nan"),
        "acceptance_rate_window": float("nan"),
        "mala_dt": float("nan"),
        "flow_score_norm": float(flow_info.get("score_norm", float("nan"))),
    }


def load_reference_samples(path: str, latent_dim: int) -> torch.Tensor:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = "samples" if "samples" in archive.files else archive.files[0]
        arr = archive[key]
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj.get("samples", next(iter(obj.values())))
        arr = torch.as_tensor(obj).detach().cpu().numpy()
    elif suffix == ".csv":
        arr = pd.read_csv(path, header=None).to_numpy()
    else:
        raise ValueError("Reference samples must be .npy, .npz, .pt, .pth, or .csv.")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != latent_dim:
        raise ValueError(
            f"Reference cloud has shape {arr.shape}; expected [n,{latent_dim}]."
        )
    if arr.shape[0] < 2 or not np.isfinite(arr).all():
        raise ValueError("Reference cloud must contain at least two finite particles.")
    return torch.as_tensor(arr, dtype=torch.float64)


def refine_reference_branch(
    particles: torch.Tensor,
    first_method: str,
    *,
    total_rounds: int,
    branch_index: int,
    alternate_methods: bool,
    prior_model,
    lik_model,
    args: argparse.Namespace,
) -> torch.Tensor:
    current = particles.detach().cpu().to(dtype=torch.float64).contiguous()
    method = first_method
    for round_index in range(2, total_rounds + 1):
        if alternate_methods:
            method = "dirichlet" if method == "lfgi" else "lfgi"
        bank, _bank_seconds = precompute_unweighted_bank(
            current,
            prior_model,
            lik_model,
            label=f"reference-branch-{branch_index}-round-{round_index}",
        )
        (current_and_info, _elapsed) = run_gate_transport(
            bank,
            method,
            n_particles=args.n_particles,
            latent_dim=args.latent_dim,
            transport_steps=args.transport_steps,
            t_min=args.t_min,
            t_max=args.t_max,
            eval_chunk=args.eval_chunk,
            seed=args.seed + 500_000 + 10_000 * branch_index + round_index,
        )
        current, _flow_info = current_and_info
        print(
            f"Reference branch {branch_index}, round {round_index}/{total_rounds}: {method}"
        )
    return current.detach().cpu()


def assert_unweighted_finite(samples: torch.Tensor, label: str, latent_dim: int) -> np.ndarray:
    arr = samples.detach().cpu().numpy() if torch.is_tensor(samples) else np.asarray(samples)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != latent_dim:
        raise ValueError(f"{label}: expected particle array [n,{latent_dim}], got {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError(f"{label}: at least two particles are required.")
    if not np.isfinite(arr).all():
        raise FloatingPointError(
            f"{label}: non-finite particles found; refusing to drop or reweight particles."
        )
    return arr


def stable_covariance(samples_np: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(samples_np, rowvar=False, ddof=1)).astype(np.float64)


def relative_l2(value: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> float:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(value - reference) / (np.linalg.norm(reference) + eps))


def summarize_posterior_functionals(
    samples: torch.Tensor,
    *,
    problem: DarcyProblem,
    label: str,
) -> Dict[str, np.ndarray]:
    x = assert_unweighted_finite(samples, label, problem.latent_dim)
    mean = np.mean(x, axis=0)
    covariance = stable_covariance(x)
    log_k = x @ problem.basis_np.T
    # Clip only before exponentiation to prevent numerical overflow in a failed
    # approximation from poisoning every diagnostic.  The unmodified log-field
    # functionals still reveal such failures.
    permeability = np.exp(np.clip(log_k, -60.0, 60.0))
    holdout_pressure = np.asarray(
        problem.solve_forward_holdout(jnp.asarray(mean, dtype=jnp.float64))
    )
    return {
        "mean": mean,
        "covariance": covariance,
        "log_k_mean": np.mean(log_k, axis=0),
        "log_k_std": np.std(log_k, axis=0, ddof=1),
        "k_mean": np.mean(permeability, axis=0),
        "holdout_pressure_at_mean": holdout_pressure,
    }


def build_reference_summary(
    reference_samples: torch.Tensor,
    *,
    problem: DarcyProblem,
) -> Dict[str, np.ndarray]:
    summary = summarize_posterior_functionals(
        reference_samples, problem=problem, label="reference"
    )
    covariance = summary["covariance"]
    d = covariance.shape[0]
    ridge = 1e-10 + 1e-8 * float(np.trace(covariance)) / float(max(1, d))
    eigvals, eigvecs = np.linalg.eigh(0.5 * (covariance + covariance.T))
    eigvals = np.maximum(eigvals, ridge)
    summary["precision_sqrt"] = (eigvecs / np.sqrt(eigvals)[None, :]) @ eigvecs.T
    return summary


def functional_errors(
    samples: torch.Tensor,
    *,
    reference_summary: Dict[str, np.ndarray],
    problem: DarcyProblem,
    label: str,
) -> Dict[str, float]:
    summary = summarize_posterior_functionals(samples, problem=problem, label=label)
    delta_mean = summary["mean"] - reference_summary["mean"]
    whitened = reference_summary["precision_sqrt"] @ delta_mean
    covariance_den = np.linalg.norm(reference_summary["covariance"], ord="fro") + 1e-12
    return {
        "MeanWhitenedRMSE": float(
            np.linalg.norm(whitened) / math.sqrt(float(problem.latent_dim))
        ),
        "CovarianceRelFro": float(
            np.linalg.norm(
                summary["covariance"] - reference_summary["covariance"], ord="fro"
            )
            / covariance_den
        ),
        "LogKMeanRelL2": relative_l2(
            summary["log_k_mean"], reference_summary["log_k_mean"]
        ),
        "LogKStdRelL2": relative_l2(
            summary["log_k_std"], reference_summary["log_k_std"]
        ),
        "KMeanRelL2": relative_l2(summary["k_mean"], reference_summary["k_mean"]),
        "HoldoutPressureAtMeanRelL2": relative_l2(
            summary["holdout_pressure_at_mean"],
            reference_summary["holdout_pressure_at_mean"],
        ),
    }


def records_to_dataframe(
    records: Sequence[Dict[str, object]],
    *,
    reference_summary: Dict[str, np.ndarray],
    problem: DarcyProblem,
) -> pd.DataFrame:
    rows = []
    for record_index, record in enumerate(records):
        method = str(record["method"])
        errors = functional_errors(
            record["samples"],
            reference_summary=reference_summary,
            problem=problem,
            label=f"{method}-{record_index}",
        )
        row = {key: value for key, value in record.items() if key != "samples"}
        row["display_method"] = METHOD_LABELS[method]
        row.update(errors)
        rows.append(row)
        if method.startswith("gad_"):
            location = f"GAD round {int(record['gad_round'])}"
        else:
            location = f"MALA transition {record['base_transition']}"
        print(
            f"Functional evaluation {record_index + 1}/{len(records)}: "
            f"{METHOD_LABELS[method]} at {location}"
        )
    return pd.DataFrame(rows)


def add_matched_compute_gains(
    frontier: pd.DataFrame,
    *,
    x_column: str,
) -> pd.DataFrame:
    baseline = frontier[frontier["method"] == "mcmc"].sort_values(x_column)
    x_base = baseline[x_column].to_numpy(dtype=float)
    gain_rows = []
    available = set(frontier["method"].astype(str))
    for method in (
        name for name in METHOD_ORDER if name != "mcmc" and name in available
    ):
        method_rows = frontier[frontier["method"] == method].sort_values(x_column)
        for _, row in method_rows.iterrows():
            for metric, _title in FUNCTIONAL_SPECS:
                y_base = baseline[metric].to_numpy(dtype=float)
                matched = float(np.interp(float(row[x_column]), x_base, y_base))
                gate_error = float(row[metric])
                gain_rows.append(
                    {
                        "method": method,
                        "display_method": METHOD_LABELS[method],
                        "budget_index": row["budget_index"],
                        "base_transition": row["base_transition"],
                        "compute_axis": x_column,
                        "total_compute": float(row[x_column]),
                        "functional": metric,
                        "gate_error": gate_error,
                        "matched_mala_error": matched,
                        "improvement_ratio_mala_over_gate": matched / max(gate_error, 1e-300),
                        "gate_better_at_matched_compute": bool(gate_error < matched),
                    }
                )
    return pd.DataFrame(gain_rows)


def plot_frontier(
    frontier: pd.DataFrame,
    *,
    x_column: str,
    xlim: Tuple[float, float],
    xscale: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=False)
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.075, top=0.84, wspace=0.20, hspace=0.32
    )
    axes = axes.ravel()
    available = set(frontier["method"].astype(str))
    methods = [name for name in METHOD_ORDER if name in available]
    x_divisor = 1e12 if x_column == "total_flops" else 1.0
    display_xlim = (xlim[0] / x_divisor, xlim[1] / x_divisor)
    for ax, (metric, title) in zip(axes, FUNCTIONAL_SPECS):
        for method in methods:
            subset = frontier[frontier["method"] == method].sort_values(x_column)
            x = subset[x_column].to_numpy(dtype=float) / x_divisor
            y = subset[metric].to_numpy(dtype=float)
            y_plot = np.maximum(y, 1e-14)
            ax.plot(
                x,
                y_plot,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=5.5,
                linewidth=2.0,
                linestyle=METHOD_LINESTYLES[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(title, fontsize=12)
        ax.set_yscale("log")
        ax.set_xscale(xscale)
        ax.set_xlim(*display_xlim)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylabel("Error (lower is better)")

    if x_column == "total_wall_seconds":
        x_label = "Total measured compute (seconds)"
    elif x_column == "total_flops":
        x_label = "Total estimated compute (TFLOPs)"
    else:
        x_label = "Total compute (portable PDE-evaluation proxy)"
    for ax in axes[3:]:
        ax.set_xlabel(x_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(methods),
        frameon=False,
        fontsize=10.5 if len(methods) >= 4 else 12,
    )
    fig.suptitle(
        "Darcy posterior functional error at matched compute",
        fontsize=16,
        y=0.985,
    )
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def jsonable_args(args: argparse.Namespace, budget_steps: Sequence[int]) -> Dict[str, object]:
    result = vars(args).copy()
    result["resolved_budget_steps"] = list(map(int, budget_steps))
    result["torch_device"] = str(device)
    result["torch_version"] = torch.__version__
    result["jax_version"] = jax.__version__
    result["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    budget_steps = make_budget_steps(args)
    args.n_budgets = len(budget_steps)
    selected_methods = tuple(args.gate_methods)
    selected_mala_gate_methods = tuple(
        method for method in ("lfgi", "dirichlet") if method in selected_methods
    )
    selected_gad_methods = tuple(
        method
        for method in ("gad_lfgi", "gad_dirichlet")
        if method in selected_methods
    )
    selected_gate_families = tuple(
        family
        for family in ("lfgi", "dirichlet")
        if any(GATE_FAMILY[method] == family for method in selected_methods)
    )
    flop_model = build_flop_model(args)
    set_all_seeds(args.seed)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"darcy_mala_gad_compute_frontier_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results directory: {output_dir}")
    print(f"MALA checkpoints: {budget_steps}")
    print(
        "Refinement comparison: "
        + ", ".join(METHOD_LABELS[method] for method in selected_methods)
    )
    pre_adapt_positive = [
        step for step in budget_steps if 0 < step <= args.mala_adapt_steps
    ]
    if pre_adapt_positive:
        print(
            "INFO: finite-compute MALA checkpoints intentionally include the "
            f"dual-averaging transient: {pre_adapt_positive}."
        )
    print("\n=== Analytical FLOP model (rough, hardware independent) ===")
    print(
        f"Dense Darcy forward solve:       {format_flops(flop_model.forward_flops)}"
    )
    print(
        "Posterior value+score / particle: "
        f"{format_flops(flop_model.posterior_value_grad_particle_flops)}"
    )
    one_cloud_posterior_flops = (
        flop_model.n_particles * flop_model.posterior_value_grad_particle_flops
    )
    mala_algebra_cloud_flops = (
        flop_model.n_particles * flop_model.mala_transition_particle_flops
    )
    ordinary_mala_step_flops = (
        one_cloud_posterior_flops + mala_algebra_cloud_flops
    )
    print(
        "One N-particle MALA transition:    "
        f"{format_flops(ordinary_mala_step_flops)} after initialization; "
        f"first transition {format_flops(ordinary_mala_step_flops + one_cloud_posterior_flops)}"
    )
    print(
        "GN Hessian / bank particle:       "
        f"{format_flops(flop_model.gn_hessian_particle_flops)} "
        f"({flop_model.gn_forward_equivalents:.1f} forward equivalents)"
    )
    print(f"Full unweighted bank:           {format_flops(flop_model.bank_total_flops)}")
    if "lfgi" in selected_gate_families:
        print(
            "LFGI reverse transport:         "
            f"{format_flops(flop_model.lfgi_transport_flops)}"
        )
        print(
            "Full LFGI refinement round:     "
            f"{format_flops(flop_model.gate_flops('lfgi'))}"
        )
    if "dirichlet" in selected_gate_families:
        print(
            "Dirichlet reverse transport:    "
            f"{format_flops(flop_model.dirichlet_transport_flops)}"
        )
        print(
            "Full Dirichlet refinement round: "
            f"{format_flops(flop_model.gate_flops('dirichlet'))}"
        )
    if args.compute_axis == "flops":
        scheduled_mala_flops = [
            0.0
            if transition == 0
            else (
                (transition + 1) * one_cloud_posterior_flops
                + transition * mala_algebra_cloud_flops
            )
            for transition in budget_steps
        ]
        print(
            "Scheduled MALA x (TFLOPs):       "
            + str([round(value / 1e12, 2) for value in scheduled_mala_flops])
        )
        for method in selected_mala_gate_methods:
            shifted = [
                (value + flop_model.gate_flops(method)) / 1e12
                for value in scheduled_mala_flops
            ]
            print(
                f"Scheduled {METHOD_LABELS[method]} x (TFLOPs): "
                + str([round(value, 2) for value in shifted])
            )
        planned_horizon_flops = scheduled_mala_flops[-1] + max(
            flop_model.gate_flops(family) for family in selected_gate_families
        )
        for method in selected_gad_methods:
            round_flops = flop_model.gate_flops(GATE_FAMILY[method])
            n_complete = min(
                args.max_gad_rounds,
                int(math.floor((planned_horizon_flops + 1e-6) / round_flops)),
            )
            planned = [
                round_number * round_flops / 1e12
                for round_number in range(1, n_complete + 1)
            ]
            print(
                f"Planned {METHOD_LABELS[method]} x (TFLOPs):   "
                + str([round(value, 2) for value in planned])
            )

    problem = build_darcy_problem(
        grid_size=args.grid_size,
        latent_dim=args.latent_dim,
        n_observations=args.n_observations,
        n_holdout=args.n_holdout,
        seed=args.seed,
    )
    data_rng = np.random.default_rng(args.seed)
    alpha_true = data_rng.normal(0.0, 0.5, size=args.latent_dim)
    y_clean = np.asarray(problem.solve_forward(jnp.asarray(alpha_true, dtype=jnp.float64)))
    y_obs = y_clean + data_rng.normal(0.0, args.noise_std, size=y_clean.shape)

    configure_sampling(
        active_dim=args.latent_dim,
        default_n_gen=args.n_particles,
        hess_min=args.hess_min,
        hess_max=args.hess_max,
        curvature_ridge=args.curvature_ridge,
    )
    prior_model = GaussianPrior(dim=args.latent_dim)
    lik_model, _lik_aux = make_physics_likelihood(
        problem.solve_forward,
        y_obs,
        args.noise_std,
        use_gauss_newton_hessian=True,
        log_batch_size=args.likelihood_batch_size,
        grad_batch_size=args.gradient_batch_size,
        hess_batch_size=args.hessian_batch_size,
    )
    warm_up_compiled_kernels(lik_model, prior_model, args.n_particles)

    # Re-seed after compilation so warmup consumes no scientific randomness.
    set_all_seeds(args.seed + 10_000)
    mala_state = initialize_mala_state(
        args.n_particles,
        prior_model,
        lik_model,
        dt=args.mala_dt,
        min_dt=args.mala_min_dt,
        max_dt=args.mala_max_dt,
    )

    base_mcmc_records: List[Dict[str, object]] = []
    checkpoint_set = set(budget_steps)
    max_base_step = max(budget_steps)
    checkpoint_accepted = 0
    checkpoint_proposals = 0
    previous_checkpoint_transition = 0
    if 0 in checkpoint_set:
        zero_record = snapshot_mala(
            mala_state,
            args.n_particles,
            flop_model,
            budget_index=budget_steps.index(0),
            phase="base",
        )
        base_mcmc_records.append(zero_record)
        if any(
            float(zero_record[key]) != 0.0
            for key in (
                "mcmc_wall_seconds",
                "total_wall_seconds",
                "mcmc_pde_proxy",
                "total_pde_proxy",
                "mcmc_flops",
                "total_flops",
            )
        ):
            raise AssertionError("The Gaussian checkpoint must have exactly zero compute.")
        print(
            f"MALA checkpoint {budget_steps.index(0) + 1}/{len(budget_steps)}: "
            "transition=0, C=0.000s, proxy=0, flops=0; unevaluated Gaussian cloud"
        )
    last_diag = None
    for _ in range(max_base_step):
        last_diag = mala_transition(
            mala_state,
            prior_model,
            lik_model,
            adapt_steps=args.mala_adapt_steps,
            target_accept=args.mala_target_accept,
            min_dt=args.mala_min_dt,
            max_dt=args.mala_max_dt,
        )
        if mala_state.transition in checkpoint_set:
            budget_index = budget_steps.index(mala_state.transition)
            window_start_transition = previous_checkpoint_transition
            new_accepted = mala_state.accepted - checkpoint_accepted
            new_proposals = mala_state.proposals - checkpoint_proposals
            window_acceptance = float(new_accepted / max(1, new_proposals))
            kernel_phase = (
                "adapting"
                if mala_state.transition <= args.mala_adapt_steps
                else "fixed"
            )
            record = snapshot_mala(
                mala_state,
                args.n_particles,
                flop_model,
                budget_index=budget_index,
                phase="base",
            )
            record["acceptance_rate_window"] = window_acceptance
            base_mcmc_records.append(record)
            checkpoint_accepted = mala_state.accepted
            checkpoint_proposals = mala_state.proposals
            previous_checkpoint_transition = mala_state.transition
            print(
                f"MALA checkpoint {budget_index + 1}/{len(budget_steps)}: "
                f"transition={mala_state.transition}, "
                f"C={mala_state.active_seconds:.3f}s, "
                f"proxy={mcmc_pde_proxy(mala_state, args.n_particles)}, "
                f"flops={format_flops(mcmc_flops(mala_state, flop_model))}, "
                f"accept_window={window_acceptance:.3f}, "
                f"accept_cumulative={mala_state.accepted / max(1, mala_state.proposals):.3f}, "
                f"dt={0.5 * math.exp(2.0 * mala_state.log_eps):.3e}, "
                f"kernel={kernel_phase}"
            )
            if (
                window_start_transition >= args.mala_adapt_steps
                and window_acceptance < args.min_post_adapt_acceptance
            ):
                raise RuntimeError(
                    "Post-adaptation MALA acceptance fell below the configured "
                    f"floor ({window_acceptance:.3f} < "
                    f"{args.min_post_adapt_acceptance:.3f}) in the checkpoint "
                    "window ending here. Gate construction has not begun. Increase "
                    "--mala-adapt-steps, reduce --mala-dt, or set "
                    "--min-post-adapt-acceptance 0 only if this low-acceptance "
                    "kernel is intentional."
                )
    base_mcmc_records.sort(key=lambda item: int(item["base_transition"]))
    if len(base_mcmc_records) != len(budget_steps):
        raise RuntimeError("Failed to capture every requested MALA checkpoint.")

    # Preserve the uninterrupted MALA random stream while all refinement arms run.
    post_base_rng_state = save_rng_state()
    gate_records: List[Dict[str, object]] = []
    gad_records: List[Dict[str, object]] = []
    gad_records_by_method: Dict[str, List[Dict[str, object]]] = {
        method: [] for method in selected_gad_methods
    }
    final_gate_samples: Dict[str, torch.Tensor] = {}

    for base_record in base_mcmc_records:
        budget_index = int(base_record["budget_index"])
        required_families = set(selected_mala_gate_methods)
        if budget_index == 0:
            required_families.update(
                GATE_FAMILY[method] for method in selected_gad_methods
            )
        if not required_families:
            continue

        bank, bank_seconds = precompute_unweighted_bank(
            base_record["samples"],
            prior_model,
            lik_model,
            label=f"mala-checkpoint-{budget_index:02d}",
        )
        # Alternate family execution order to reduce systematic cache-order bias.
        family_order = (
            ("lfgi", "dirichlet")
            if budget_index % 2 == 0
            else ("dirichlet", "lfgi")
        )
        family_order = tuple(
            family for family in family_order if family in required_families
        )
        for family in family_order:
            (samples_and_info, transport_seconds) = run_gate_transport(
                bank,
                family,
                n_particles=args.n_particles,
                latent_dim=args.latent_dim,
                transport_steps=args.transport_steps,
                t_min=args.t_min,
                t_max=args.t_max,
                eval_chunk=args.eval_chunk,
                seed=args.seed + 100_000 + budget_index,
            )
            samples, flow_info = samples_and_info

            if family in selected_mala_gate_methods:
                record = make_gate_record(
                    method=family,
                    samples=samples,
                    base_record=base_record,
                    bank_seconds=bank_seconds,
                    transport_seconds=transport_seconds,
                    n_particles=args.n_particles,
                    flop_model=flop_model,
                    flow_info=flow_info,
                )
                gate_records.append(record)
                if budget_index == len(base_mcmc_records) - 1:
                    final_gate_samples[family] = samples.detach().cpu()
                print(
                    f"Gate checkpoint {budget_index + 1}/{len(base_mcmc_records)}: "
                    f"{METHOD_LABELS[family]}, K={record['gate_wall_seconds']:.3f}s "
                    f"/ {format_flops(record['gate_flops'])} "
                    f"(bank={bank_seconds:.3f}s, transport={transport_seconds:.3f}s)"
                )

            gad_method = f"gad_{family}"
            if budget_index == 0 and gad_method in selected_gad_methods:
                gad_record = make_gad_record(
                    method=gad_method,
                    round_index=0,
                    samples=samples,
                    bank_seconds=bank_seconds,
                    transport_seconds=transport_seconds,
                    n_particles=args.n_particles,
                    flop_model=flop_model,
                    flow_info=flow_info,
                )
                gad_records.append(gad_record)
                gad_records_by_method[gad_method].append(gad_record)
                shared_note = (
                    f"; shared exactly with {METHOD_LABELS[family]} checkpoint 0"
                    if family in selected_mala_gate_methods
                    else ""
                )
                print(
                    f"GAD round 0: {METHOD_LABELS[gad_method]}, "
                    f"C={format_flops(gad_record['total_flops'])}{shared_note}"
                )

    if set(final_gate_samples) != set(selected_mala_gate_methods):
        raise RuntimeError("Did not retain every selected final-budget MALA-gate cloud.")
    if any(len(records) != 1 for records in gad_records_by_method.values()):
        raise RuntimeError("Did not construct exactly one prior-bank GAD round-zero cloud.")

    initial_pipeline_records = gate_records + [
        records[0] for records in gad_records_by_method.values()
    ]
    if not initial_pipeline_records:
        raise RuntimeError("No selected refinement arm produced a first-round record.")
    k_wall_max = max(
        float(
            record.get("round_wall_seconds", record["gate_wall_seconds"])
        )
        for record in initial_pipeline_records
    )
    k_proxy_max = max(
        int(record.get("round_pde_proxy", record["gate_pde_proxy"]))
        for record in initial_pipeline_records
    )
    k_flops_max = max(
        float(record.get("round_flops", record["gate_flops"]))
        for record in initial_pipeline_records
    )
    c_wall_n = float(base_mcmc_records[-1]["mcmc_wall_seconds"])
    c_proxy_n = int(base_mcmc_records[-1]["mcmc_pde_proxy"])
    c_flops_n = float(base_mcmc_records[-1]["mcmc_flops"])
    target_wall = c_wall_n + k_wall_max
    target_proxy = c_proxy_n + k_proxy_max
    target_flops = c_flops_n + k_flops_max
    c_wall_1 = float(base_mcmc_records[0]["mcmc_wall_seconds"])
    c_proxy_1 = int(base_mcmc_records[0]["mcmc_pde_proxy"])
    c_flops_1 = float(base_mcmc_records[0]["mcmc_flops"])

    selected_x_column = {
        "wall_seconds": "total_wall_seconds",
        "pde_proxy": "total_pde_proxy",
        "flops": "total_flops",
    }[args.compute_axis]
    selected_horizon = {
        "wall_seconds": target_wall,
        "pde_proxy": float(target_proxy),
        "flops": target_flops,
    }[args.compute_axis]

    # Continue each transport-only GAD arm only while another complete round
    # fits inside the original MALA C_n + max(K) plotting horizon.
    for method_index, method in enumerate(selected_gad_methods):
        family = GATE_FAMILY[method]
        records = gad_records_by_method[method]
        current_record = records[-1]
        while len(records) < args.max_gad_rounds:
            if args.compute_axis == "flops":
                predicted_increment = flop_model.gate_flops(family)
            elif args.compute_axis == "pde_proxy":
                predicted_increment = float(3 * args.n_particles)
            else:
                predicted_increment = float(current_record["round_wall_seconds"])
            predicted_total = (
                float(current_record[selected_x_column]) + predicted_increment
            )
            if predicted_total > selected_horizon * (1.0 + 1e-12):
                break

            round_index = len(records)
            bank, bank_seconds = precompute_unweighted_bank(
                current_record["samples"],
                prior_model,
                lik_model,
                label=f"{method.replace('_', '-')}-round-{round_index:02d}-input",
            )
            (samples_and_info, transport_seconds) = run_gate_transport(
                bank,
                family,
                n_particles=args.n_particles,
                latent_dim=args.latent_dim,
                transport_steps=args.transport_steps,
                t_min=args.t_min,
                t_max=args.t_max,
                eval_chunk=args.eval_chunk,
                seed=args.seed + 300_000 + 10_000 * method_index + round_index,
            )
            samples, flow_info = samples_and_info
            next_record = make_gad_record(
                method=method,
                round_index=round_index,
                samples=samples,
                bank_seconds=bank_seconds,
                transport_seconds=transport_seconds,
                n_particles=args.n_particles,
                flop_model=flop_model,
                flow_info=flow_info,
                previous_record=current_record,
            )
            if float(next_record[selected_x_column]) > selected_horizon * (1.0 + 1e-12):
                print(
                    f"Excluded {METHOD_LABELS[method]} round {round_index}: measured "
                    f"wall time crossed the fixed compute horizon."
                )
                break
            records.append(next_record)
            gad_records.append(next_record)
            current_record = next_record
            print(
                f"GAD round {round_index}: {METHOD_LABELS[method]}, "
                f"cumulative wall={next_record['total_wall_seconds']:.3f}s, "
                f"proxy={next_record['total_pde_proxy']}, "
                f"flops={format_flops(next_record['total_flops'])}"
            )
        if len(records) == args.max_gad_rounds:
            print(
                f"WARNING: {METHOD_LABELS[method]} reached --max-gad-rounds="
                f"{args.max_gad_rounds}; increase it if the selected horizon permits more."
            )

    gad_terminal_samples = {
        method: records[-1]["samples"]
        for method, records in gad_records_by_method.items()
    }

    def _domain_overlap(base_domain, method_domain):
        low = max(float(base_domain[0]), float(method_domain[0]))
        high = min(float(base_domain[1]), float(method_domain[1]))
        width = max(0.0, high - low)
        method_width = float(method_domain[1]) - float(method_domain[0])
        fraction = width / method_width if method_width > 0.0 else float("nan")
        return width, fraction

    refinement_records = gate_records + gad_records
    method_compute_domains: Dict[str, Dict[str, object]] = {}
    for method in selected_methods:
        records = [record for record in refinement_records if record["method"] == method]
        if not records:
            raise RuntimeError(f"No records were produced for selected method {method!r}.")
        wall_domain = (
            min(float(record["total_wall_seconds"]) for record in records),
            max(float(record["total_wall_seconds"]) for record in records),
        )
        proxy_domain = (
            min(float(record["total_pde_proxy"]) for record in records),
            max(float(record["total_pde_proxy"]) for record in records),
        )
        flop_domain = (
            min(float(record["total_flops"]) for record in records),
            max(float(record["total_flops"]) for record in records),
        )
        selected_domain = {
            "wall_seconds": wall_domain,
            "pde_proxy": proxy_domain,
            "flops": flop_domain,
        }[args.compute_axis]
        selected_base_domain = {
            "wall_seconds": (c_wall_1, c_wall_n),
            "pde_proxy": (c_proxy_1, c_proxy_n),
            "flops": (c_flops_1, c_flops_n),
        }[args.compute_axis]
        selected_overlap, selected_overlap_fraction = _domain_overlap(
            selected_base_domain, selected_domain
        )
        method_compute_domains[method] = {
            "wall_seconds": list(wall_domain),
            "pde_proxy": list(proxy_domain),
            "flops": list(flop_domain),
            "n_points": len(records),
            "selected_axis_overlap": selected_overlap,
            "selected_axis_overlap_fraction": selected_overlap_fraction,
        }

    print("\n=== Compute-domain audit ===")
    print(f"Base MALA wall domain:       [{c_wall_1:.3f}, {c_wall_n:.3f}] s")
    print(f"Base MALA PDE-proxy domain:  [{c_proxy_1}, {c_proxy_n}]")
    print(
        "Base MALA FLOP domain:       "
        f"[{format_flops(c_flops_1)}, {format_flops(c_flops_n)}]"
    )
    print(
        "Common horizons C_n+max(K): "
        f"{target_wall:.3f}s; {target_proxy} proxy; {format_flops(target_flops)}"
    )
    for method in selected_methods:
        domain = method_compute_domains[method]
        selected_domain = domain[args.compute_axis]
        fraction = float(domain["selected_axis_overlap_fraction"])
        fraction_text = "n/a" if not math.isfinite(fraction) else f"{100.0 * fraction:.1f}%"
        print(
            f"{METHOD_LABELS[method]} selected-axis domain: "
            f"[{selected_domain[0]:.6g}, {selected_domain[1]:.6g}], "
            f"points={domain['n_points']}, overlap={fraction_text}"
        )

    # Continue the same MALA random stream to the fixed common horizon.
    restore_rng_state(post_base_rng_state)

    if args.compute_axis == "wall_seconds":
        extension_start = c_wall_n
        extension_target = target_wall
        current_compute = lambda: float(mala_state.active_seconds)
        extension_unit = "seconds"
    elif args.compute_axis == "pde_proxy":
        extension_start = float(c_proxy_n)
        extension_target = float(target_proxy)
        current_compute = lambda: float(mcmc_pde_proxy(mala_state, args.n_particles))
        extension_unit = "PDE-proxy units"
    else:
        extension_start = c_flops_n
        extension_target = target_flops
        current_compute = lambda: float(mcmc_flops(mala_state, flop_model))
        extension_unit = "FLOPs"
    print(
        f"Selected {args.compute_axis} MALA extension target: "
        f"{extension_target:.6g} {extension_unit}"
    )
    extension_targets = np.linspace(
        extension_start, extension_target, args.extension_points + 1
    )[1:]
    next_extension_target = 0
    extension_records: List[Dict[str, object]] = []
    extension_start_transition = mala_state.transition
    while current_compute() < extension_target:
        if mala_state.transition - extension_start_transition >= args.max_extension_transitions:
            raise RuntimeError(
                "MALA extension hit --max-extension-transitions before covering "
                "C_n + max(K). Increase the cap to retain matched-compute coverage."
            )
        last_diag = mala_transition(
            mala_state,
            prior_model,
            lik_model,
            adapt_steps=args.mala_adapt_steps,
            target_accept=args.mala_target_accept,
            min_dt=args.mala_min_dt,
            max_dt=args.mala_max_dt,
        )
        crossed_target = (
            next_extension_target < len(extension_targets)
            and current_compute() >= extension_targets[next_extension_target]
        )
        if crossed_target:
            extension_records.append(
                snapshot_mala(
                    mala_state,
                    args.n_particles,
                    flop_model,
                    budget_index=None,
                    phase="extension",
                )
            )
            while (
                next_extension_target < len(extension_targets)
                and current_compute() >= extension_targets[next_extension_target]
            ):
                next_extension_target += 1
    print(
        f"MALA extension complete at transition {mala_state.transition}: "
        f"wall={mala_state.active_seconds:.3f}s, "
        f"proxy={mcmc_pde_proxy(mala_state, args.n_particles)}, "
        f"flops={format_flops(mcmc_flops(mala_state, flop_model))}"
    )

    # Construct or load a high-quality functional reference.  Prefer the final
    # MALA-gated cloud for a selected family; in GAD-only mode use that family's
    # last in-horizon iterate.  Two-family comparisons refine and pool balanced
    # alternating branches so the reference does not privilege one gate family.
    if args.reference_samples:
        reference_samples = load_reference_samples(args.reference_samples, args.latent_dim)
        reference_source = f"loaded:{args.reference_samples}"
    else:
        reference_branches = []
        seed_descriptions = []
        alternate_reference_methods = len(selected_gate_families) == 2
        for branch_index, family in enumerate(selected_gate_families):
            if family in final_gate_samples:
                seed_samples = final_gate_samples[family]
                seed_descriptions.append(f"final MALA->{family}")
            else:
                gad_method = f"gad_{family}"
                seed_samples = gad_terminal_samples[gad_method]
                seed_descriptions.append(
                    f"{METHOD_LABELS[gad_method]} round "
                    f"{len(gad_records_by_method[gad_method]) - 1}"
                )
            reference_branches.append(
                refine_reference_branch(
                    seed_samples,
                    family,
                    total_rounds=args.reference_rounds,
                    branch_index=branch_index,
                    alternate_methods=alternate_reference_methods,
                    prior_model=prior_model,
                    lik_model=lik_model,
                    args=args,
                )
            )
        if len(reference_branches) == 1:
            reference_samples = reference_branches[0]
        else:
            reference_samples = torch.cat(reference_branches, dim=0)
        reference_source = (
            "pooled " if len(reference_branches) > 1 else ""
        ) + (
            "; ".join(seed_descriptions)
            + f" plus {max(0, args.reference_rounds - 1)} off-frontier refinement rounds"
        )
    np.savez_compressed(
        output_dir / "reference_samples.npz",
        samples=reference_samples.detach().cpu().numpy(),
        source=np.asarray(reference_source),
    )
    reference_summary = build_reference_summary(reference_samples, problem=problem)

    all_records = base_mcmc_records + extension_records + gate_records + gad_records
    frontier = records_to_dataframe(
        all_records,
        reference_summary=reference_summary,
        problem=problem,
    )
    frontier = frontier.sort_values(["method", selected_x_column]).reset_index(drop=True)
    frontier_path = output_dir / "functional_compute_frontier.csv"
    frontier.to_csv(frontier_path, index=False)

    x_column = selected_x_column
    gains = add_matched_compute_gains(frontier, x_column=x_column)
    gains_path = output_dir / "matched_compute_gains.csv"
    gains.to_csv(gains_path, index=False)

    if x_column == "total_wall_seconds":
        x_min = float(base_mcmc_records[0]["mcmc_wall_seconds"])
        x_max = target_wall
    elif x_column == "total_pde_proxy":
        x_min = float(base_mcmc_records[0]["mcmc_pde_proxy"])
        x_max = float(target_proxy)
    else:
        x_min = float(base_mcmc_records[0]["mcmc_flops"])
        x_max = float(target_flops)
    if args.xscale == "log" and x_min <= 0.0:
        positive = frontier.loc[frontier[x_column] > 0.0, x_column]
        x_min = float(positive.min())
    output_png = output_dir / "darcy_functional_compute_frontier.png"
    output_pdf = output_dir / "darcy_functional_compute_frontier.pdf"
    plot_frontier(
        frontier,
        x_column=x_column,
        xlim=(x_min, x_max),
        xscale=args.xscale,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    config = jsonable_args(args, budget_steps)
    config.update(
        {
            "reference_source": reference_source,
            "reference_particle_count": int(reference_samples.shape[0]),
            "largest_gate_wall_seconds": k_wall_max,
            "largest_gate_pde_proxy": k_proxy_max,
            "largest_gate_flops": k_flops_max,
            "selected_methods": list(selected_methods),
            "selected_mala_gate_methods": list(selected_mala_gate_methods),
            "selected_gad_methods": list(selected_gad_methods),
            "selected_gate_families": list(selected_gate_families),
            "gad_rounds_retained": {
                method: len(records)
                for method, records in gad_records_by_method.items()
            },
            "method_compute_domains": method_compute_domains,
            "base_mcmc_wall_domain": [c_wall_1, c_wall_n],
            "base_mcmc_pde_proxy_domain": [c_proxy_1, c_proxy_n],
            "base_mcmc_flop_domain": [c_flops_1, c_flops_n],
            "common_compute_horizons": {
                "wall_seconds": target_wall,
                "pde_proxy": target_proxy,
                "flops": target_flops,
            },
            "flop_model": asdict(flop_model),
            "flop_model_notes": (
                "Analytical operation-count proxy: dense Darcy LU+triangular solve; "
                "reverse-mode value/gradient charged as two forward equivalents; "
                "jacfwd GN Hessian charged at gn_forward_equivalents; gate transport "
                "counts finite-bank contractions, matrix decompositions/solves, and "
                "all 2*transport_steps+1 score evaluations."
            ),
            "zero_compute_checkpoint": (
                "The transition-0 MALA cloud is an unevaluated N(0,I) draw and is "
                "assigned zero seconds, zero PDE-proxy units, and zero FLOPs."
            ),
            "checkpoint_policy": (
                "With mala_budget_min=0 and linear spacing, checkpoints cover the "
                "entire 0..mala_budget_max trajectory, including dual-averaging "
                "warmup. This removes the nonstructural gap formerly caused by "
                "waiting until adaptation ended before the first positive checkpoint."
            ),
            "mala_tuning_policy": (
                "Dual averaging targets mala_target_accept through "
                "mala_adapt_steps. The acceptance floor is applied only to a "
                "checkpoint window lying wholly after adaptation."
            ),
            "gad_round_zero_policy": (
                "GAD round zero is the prior-bank transport and is exactly the "
                "same sample cloud as the matching MALA-gate checkpoint zero when "
                "both arms are selected; bank and transport are executed once."
            ),
            "plot_x_min": x_min,
            "plot_x_max": x_max,
            "final_mala_transition": int(mala_state.transition),
            "final_mala_acceptance_rate": float(
                mala_state.accepted / max(1, mala_state.proposals)
            ),
            "final_mala_dt": float(0.5 * math.exp(2.0 * mala_state.log_eps)),
            "last_mala_diagnostic": last_diag,
            "functional_definitions": {key: title for key, title in FUNCTIONAL_SPECS},
            "unweighted_particle_policy": (
                "All MALA and recursive GAD bank particles and all transported "
                "output particles carry equal weight; non-finite particles cause "
                "failure rather than filtering or reweighting."
            ),
        }
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    print("\n=== Compute-frontier experiment complete ===")
    print(f"Plot: {output_png}")
    print(f"Plot (PDF): {output_pdf}")
    print(f"Functional data: {frontier_path}")
    print(f"Matched-compute gains: {gains_path}")
    print(f"Reference cloud: {output_dir / 'reference_samples.npz'}")


if __name__ == "__main__":
    main()
