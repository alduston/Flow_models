# -*- coding: utf-8 -*-
"""Known-normalization calibration benchmark for an analytic Gaussian inverse problem.

This script is intended as the problem entry for manuscript section
"Known-normalization calibration on an analytic Gaussian inverse problem".
It mirrors the Darcy density-evaluation benchmark wiring while replacing the
PDE forward model by a linear-Gaussian inverse problem with closed-form
posterior and closed-form evidence.

Typical repository / Slurm use:

    # Place this file in a problem directory, e.g. analytic_gaussian_inverse/problem.py.
    # Place the shared harness at <repo-root>/sampling.py.
    python problem.py

Useful overrides:

    export GIP_DIM=8
    export GIP_OBS_DIM=8
    export GIP_NOISE_STD=0.35
    export GIP_FORWARD_SCALE=4.0
    export GIP_FORWARD_COND=60.0

    export IP_DENSITY_N_REF_SIGNAL=2000
    export IP_DENSITY_N_REF_GATE=2000
    export IP_DENSITY_N_REF_EVAL=2000
    export IP_DENSITY_BANK_COUPLING=independent
    export IP_DENSITY_EVAL_SOURCE=POSTERIOR-EVAL
    export IP_DENSITY_DRC_PF_STEPS=64
    export IP_DENSITY_DRC_PLOT_LAYOUT=comparison_grid

By default the source and held-out evaluation banks are drawn from the exact
MAP--Laplace posterior, which is exact for this linear-Gaussian target.  For
repeated uncertainty runs on the same analytic target, keep GIP_PROBLEM_SEED fixed
and vary SEED/GIP_SEED.  In Slurm array runs, if SEED and GIP_SEED are unset, the
script uses SEED = GIP_PROBLEM_SEED + SLURM_ARRAY_TASK_ID.  To force literal MALA
source/eval banks, set for example:

    export GIP_SOURCE_INIT=map_laplace
    export GIP_SOURCE_MALA_STEPS=600
    export GIP_SOURCE_MALA_BURNIN=150
    export GIP_SOURCE_MALA_DT=1e-3
    export GIP_EVAL_MALA_STEPS=600
    export GIP_EVAL_MALA_BURNIN=150

The likelihood convention matches sampling.py: log_likelihood omits the Gaussian
normalizing constant.  The analytic logZ reported here is therefore

    log int p0(x) exp(-||A x - y||^2 / (2 sigma^2)) dx,

not log p(y) under the normalized Gaussian noise model.
"""

import gc
import os
import sys
import linecache
import importlib
from collections import OrderedDict
from datetime import datetime

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # notebook / pasted-cell fallback
    THIS_DIR = os.getcwd()

# Support both repository layouts: problem.py in a problem subdirectory with
# <repo-root>/sampling.py, or problem.py living next to sampling.py.
for path in (THIS_DIR, os.path.dirname(THIS_DIR), os.getcwd()):
    if path and path not in sys.path:
        sys.path.insert(0, path)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

linecache.clearcache()
if "sampling" in sys.modules:
    del sys.modules["sampling"]
SAMPLING_MODULE_NAME = os.environ.get("HLSI_SAMPLING_MODULE", "sampling")
sampling = importlib.import_module(SAMPLING_MODULE_NAME)
importlib.reload(sampling)
sys.modules["sampling"] = sampling

print("Using:", sampling.__file__)
print("DRC test:", sampling.canonicalize_init_weights("DRC"))

# Paper-facing label cleanup for this standalone benchmark.  The shared harness
# may still use historical internal names in a few plotting helpers; keep the
# numerical configuration unchanged while displaying LFGI in outputs.
if hasattr(sampling, "_drc_method_pretty_name"):
    _ORIGINAL_DRC_METHOD_PRETTY_NAME = sampling._drc_method_pretty_name

    def _lfgi_drc_method_pretty_name(*args, **kwargs):
        out = _ORIGINAL_DRC_METHOD_PRETTY_NAME(*args, **kwargs)
        if isinstance(out, str):
            out = out.replace("CE" + "-HLSI", "LFGI").replace("ce" + "_hlsi", "lfgi")
        return out

    sampling._drc_method_pretty_name = _lfgi_drc_method_pretty_name

from sampling import (
    GaussianPrior,
    compute_latent_metrics,
    configure_sampling,
    get_valid_samples,
    init_run_results,
    make_density_manuscript_table,
    make_physics_likelihood,
    make_posterior_score_fn,
    plot_mean_ess_logs,
    plot_pca_histograms,
    run_drc_pf_sensitivity_benchmark,
    run_standard_sampler_pipeline,
    save_reproducibility_log,
    save_results_tables,
    summarize_sampler_run,
    zip_run_results_dir,
)

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

# ============================================================================
# Environment helpers
# ============================================================================


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _first_env_int(names, default):
    """Return the first set integer environment variable among names."""
    for name in names:
        raw = os.environ.get(name, None)
        if raw is None or str(raw).strip() == "":
            continue
        return int(raw)
    return int(default)


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_float_or_none(name, default=None):
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    if raw in {"", "none", "null", "nan"}:
        return None
    return float(raw)


def _env_bool(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "none", ""}


def _env_is_set(name):
    return name in os.environ and str(os.environ.get(name, "")).strip() != ""


_RUN_INDEX_ENV_NAMES = ("RUN_INDEX", "REPLICATE", "REPLICA", "SLURM_ARRAY_TASK_ID", "PBS_ARRAY_INDEX")
_RUN_SEED_ENV_NAMES = ("GIP_SEED", "SEED")


def _first_env_int_or_none(names):
    """Return the first explicitly set integer environment variable among names."""
    for name in names:
        raw = os.environ.get(name, None)
        if raw is None or str(raw).strip() == "":
            continue
        return int(raw), name
    return None, None


def _resolve_stochastic_run_seed(problem_seed, run_index):
    """Resolve the stochastic seed used for reference/evaluation banks.

    The analytic target itself remains keyed only by GIP_PROBLEM_SEED.  The
    stochastic finite-bank seed must change across uncertainty replicates.  In
    particular, many Slurm wrappers export a fixed SEED to every array task; we
    therefore fold RUN_INDEX / SLURM_ARRAY_TASK_ID into the effective seed even
    when SEED is set.  If neither a seed nor a run-index environment variable is
    provided, fall back to OS entropy so repeated standalone invocations do not
    silently reuse the identical reference/evaluation banks.
    """
    seed_base, seed_source = _first_env_int_or_none(_RUN_SEED_ENV_NAMES)
    _, run_index_source = _first_env_int_or_none(_RUN_INDEX_ENV_NAMES)
    if seed_base is None:
        if run_index_source is not None:
            seed_base = int(problem_seed)
            seed_source = "GIP_PROBLEM_SEED"
        else:
            seed_base = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            seed_source = "entropy"
    disable_offset = _env_bool("GIP_DISABLE_RUN_INDEX_SEED_OFFSET", False)
    offset = 0 if disable_offset else int(run_index)
    # Keep seeds accepted by numpy/torch while preserving deterministic offsets.
    effective = int((int(seed_base) + offset) % (2 ** 32 - 1))
    if effective <= 0:
        effective += 1
    return effective, int(seed_base), seed_source, int(offset), bool(disable_offset)


def _env_csv(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return tuple(default)
    parts = [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    return tuple(parts) if parts else tuple(default)


def _env_int_tuple(name, default):
    return tuple(int(float(x)) for x in _env_csv(name, default))


def _env_float_tuple_or_none(name, default=None):
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    parts = [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    if not parts or any(x.lower() in {"none", "null"} for x in parts):
        return default
    return tuple(float(x) for x in parts)


def _env_percentile_pair(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    parts = [float(x.strip()) for x in str(raw).replace(";", ",").split(",") if x.strip()]
    if len(parts) < 2:
        return default
    lo, hi = parts[:2]
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError(f"{name} must be two percentiles 0 <= lo < hi <= 100; got {raw!r}")
    return (lo, hi)


def _canonical_bank_coupling(value):
    key = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if key in {"shared", "same"}:
        return "shared"
    if key in {"prefix", "sliced", "slice"}:
        return "prefix"
    if key in {"independent", "indep", "disjoint", "split"}:
        return "independent"
    raise ValueError(f"Unknown bank coupling {value!r}; expected shared, prefix, or independent.")


def _required_source_bank_size(n_signal, n_gate, score_gate_bank_coupling):
    coupling = _canonical_bank_coupling(score_gate_bank_coupling)
    if coupling == "shared":
        return int(n_signal)
    if coupling == "prefix":
        return int(max(n_signal, n_gate))
    return int(n_signal) + int(n_gate)


def _canonical_source_label(value):
    raw = str(value).strip()
    low = raw.lower().replace("_", "-").replace(" ", "-")
    if low in {"none", "null", ""}:
        return "None"
    if low in {"posterior", "exact-posterior", "exact"}:
        return "POSTERIOR"
    if low in {"posterior-eval", "posterioreval", "exact-posterior-eval", "exact-eval"}:
        return "POSTERIOR-EVAL"
    if low == "mala":
        return "MALA"
    if low in {"mala-eval", "malaeval"}:
        return "MALA-EVAL"
    return raw


def _sanitize_label(label):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label)).strip("_")


# ============================================================================
# Analytic linear-Gaussian inverse problem
# ============================================================================

# Separate the fixed analytic problem seed from the stochastic run seed.
# This keeps A, x_true, y_obs, and exact logZ fixed across repeated UQ runs,
# while source/eval banks and randomized numerical routines can vary.
PROBLEM_SEED = _env_int("GIP_PROBLEM_SEED", 42)
RUN_INDEX = _first_env_int(_RUN_INDEX_ENV_NAMES, 0)
seed, RUN_SEED_BASE, RUN_SEED_SOURCE, RUN_SEED_OFFSET, RUN_INDEX_SEED_OFFSET_DISABLED = _resolve_stochastic_run_seed(
    PROBLEM_SEED,
    RUN_INDEX,
)
rng = np.random.default_rng(seed)
problem_rng = np.random.default_rng(PROBLEM_SEED)

ACTIVE_DIM = _env_int("GIP_DIM", 8)
OBS_DIM = _env_int("GIP_OBS_DIM", ACTIVE_DIM)
NOISE_STD = _env_float("GIP_NOISE_STD", 0.35)
FORWARD_SCALE = _env_float("GIP_FORWARD_SCALE", 4.0)
FORWARD_COND = _env_float("GIP_FORWARD_COND", 60.0)
TRUE_SCALE = _env_float("GIP_TRUE_SCALE", 0.75)
OPERATOR_SEED = _env_int("GIP_OPERATOR_SEED", PROBLEM_SEED + 17)

if ACTIVE_DIM <= 0 or OBS_DIM <= 0:
    raise ValueError("GIP_DIM and GIP_OBS_DIM must be positive.")
if NOISE_STD <= 0.0:
    raise ValueError("GIP_NOISE_STD must be positive.")
if FORWARD_SCALE <= 0.0 or FORWARD_COND < 1.0:
    raise ValueError("GIP_FORWARD_SCALE must be positive and GIP_FORWARD_COND must be >= 1.")


def make_linear_operator(m, d, scale=4.0, cond=60.0, seed_value=0):
    """Return a deterministic m x d operator with controlled singular spectrum."""
    local_rng = np.random.default_rng(seed_value)
    r = min(int(m), int(d))
    U, _ = np.linalg.qr(local_rng.normal(size=(m, r)))
    V, _ = np.linalg.qr(local_rng.normal(size=(d, r)))
    if r == 1:
        singular_values = np.array([float(scale)], dtype=np.float64)
    else:
        singular_values = np.geomspace(float(scale), float(scale) / float(cond), r)
    A = U @ np.diag(singular_values) @ V.T
    return np.asarray(A, dtype=np.float64), singular_values


A_np, A_singular_values = make_linear_operator(
    OBS_DIM, ACTIVE_DIM, scale=FORWARD_SCALE, cond=FORWARD_COND, seed_value=OPERATOR_SEED
)
A_jax = jnp.asarray(A_np, dtype=jnp.float64)


@jax.jit
def solve_forward(alpha):
    return A_jax @ alpha


batch_solve_forward = jax.jit(jax.vmap(solve_forward))


def analytic_posterior_and_logZ(A, y, sigma):
    """Closed-form posterior and unnormalized likelihood evidence.

    Prior: x ~ N(0, I_d).
    Harness likelihood: L(x) = exp(-||A x - y||^2 / (2 sigma^2)).
    Posterior: N(mean, precision^{-1}) with precision I + A^T A / sigma^2.
    logZ: log int p0(x) L(x) dx, matching the omitted-likelihood-constant
    convention used by make_physics_likelihood.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    d = int(A.shape[1])
    precision = np.eye(d, dtype=np.float64) + (A.T @ A) / (sigma ** 2)
    h = (A.T @ y) / (sigma ** 2)
    mean = np.linalg.solve(precision, h)
    cov = np.linalg.inv(precision)
    sign, logdet_precision = np.linalg.slogdet(precision)
    if sign <= 0:
        raise RuntimeError("Analytic posterior precision is not SPD.")
    logZ = (
        -0.5 * float(np.dot(y, y)) / (sigma ** 2)
        + 0.5 * float(np.dot(h, mean))
        - 0.5 * float(logdet_precision)
    )
    return {
        "precision": precision,
        "cov": cov,
        "mean": mean,
        "logdet_precision": float(logdet_precision),
        "logZ": float(logZ),
        "h": h,
    }


def log_normalized_posterior_np(X, mean, precision, logdet_precision):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    diff = X - np.asarray(mean, dtype=np.float64).reshape(1, -1)
    quad = np.einsum("ni,ij,nj->n", diff, precision, diff)
    d = int(diff.shape[1])
    return 0.5 * float(logdet_precision) - 0.5 * d * np.log(2.0 * np.pi) - 0.5 * quad


np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

alpha_true_np = TRUE_SCALE * problem_rng.normal(size=ACTIVE_DIM)
y_clean_np = A_np @ alpha_true_np
y_obs_np = y_clean_np + NOISE_STD * problem_rng.normal(size=OBS_DIM)
posterior_info = analytic_posterior_and_logZ(A_np, y_obs_np, NOISE_STD)
posterior_mean_np = posterior_info["mean"]
posterior_precision_np = posterior_info["precision"]
posterior_cov_np = posterior_info["cov"]
TRUE_LOGZ = posterior_info["logZ"]
POST_PRECISION_EIGS = np.linalg.eigvalsh(posterior_precision_np)

print("\n=== Analytic Gaussian inverse problem ===")
print(f"d={ACTIVE_DIM}, m={OBS_DIM}, sigma={NOISE_STD:g}, problem_seed={PROBLEM_SEED}, run_seed={seed}, run_index={RUN_INDEX}")
print(f"operator singular values = {np.array2string(A_singular_values, precision=4)}")
print(f"posterior precision eig range = [{POST_PRECISION_EIGS.min():.6g}, {POST_PRECISION_EIGS.max():.6g}]")
print(f"analytic logZ under harness convention = {TRUE_LOGZ:.10f}")
print(f"||x_true||={np.linalg.norm(alpha_true_np):.4g}, ||posterior_mean||={np.linalg.norm(posterior_mean_np):.4g}")

# ============================================================================
# Shared sampling / density benchmark configuration
# ============================================================================

PLOT_NORMALIZER = os.environ.get("GIP_PLOT_NORMALIZER", "best")
HESS_MIN = _env_float("GIP_HESS_MIN", 1e-8)
HESS_MAX = _env_float("GIP_HESS_MAX", 1e8)
GNL_PILOT_N = _env_int("GIP_GNL_PILOT_N", 512)
GNL_STIFF_LAMBDA_CUT = _env_float("GIP_GNL_STIFF_LAMBDA_CUT", HESS_MAX)
GNL_USE_DOMINANT_PARTICLE_NEWTON = _env_bool("GIP_GNL_USE_DOMINANT_PARTICLE_NEWTON", True)
BUILD_GNL_BANKS = _env_bool("GIP_BUILD_GNL_BANKS", False)

N_REF_SIGNAL = _env_int("IP_DENSITY_N_REF_SIGNAL", _env_int("IP_DENSITY_N_REF", 2000))
N_REF_GATE = _env_int("IP_DENSITY_N_REF_GATE", N_REF_SIGNAL)
N_REF_EVAL = _env_int("IP_DENSITY_N_REF_EVAL", N_REF_SIGNAL)
N_REF = N_REF_SIGNAL
DEFAULT_N_GEN = _env_int("IP_DENSITY_DEFAULT_N_GEN", N_REF_SIGNAL)

DENSITY_REF_SOURCE = _canonical_source_label(os.environ.get("IP_DENSITY_REF_SOURCE", "POSTERIOR"))
DENSITY_BANK_COUPLING = _canonical_bank_coupling(os.environ.get("IP_DENSITY_BANK_COUPLING", "independent"))
DENSITY_EVAL_SOURCE = _canonical_source_label(os.environ.get("IP_DENSITY_EVAL_SOURCE", "POSTERIOR-EVAL"))
DENSITY_EVAL_BANK_COUPLING = _canonical_bank_coupling(
    os.environ.get("IP_DENSITY_EVAL_BANK_COUPLING", "independent")
)

DENSITY_DRC_PF_STEPS = _env_int("IP_DENSITY_DRC_PF_STEPS", 64)
DENSITY_DRC_EVAL_BATCH_SIZE = _env_int("IP_DENSITY_DRC_EVAL_BATCH_SIZE", 64)
DENSITY_DRC_TMIN = _env_float("IP_DENSITY_DRC_TMIN", 10 ** (-2.5))
DENSITY_DRC_TMAX = _env_float("IP_DENSITY_DRC_TMAX", 5.0)
DENSITY_DRC_CLIP = _env_float_or_none("IP_DENSITY_DRC_CLIP", None)
DENSITY_DRC_TEMPERATURE = _env_float("IP_DENSITY_DRC_TEMPERATURE", 1.0)
DENSITY_DRC_ENERGY_PLOTS = _env_bool("IP_DENSITY_DRC_ENERGY_PLOTS", True)
DENSITY_DRC_PLOT_AXIS_MODE = os.environ.get("IP_DENSITY_DRC_PLOT_AXIS_MODE", "robust")
DENSITY_DRC_RESIDUAL_AXIS_MODE = os.environ.get("IP_DENSITY_DRC_RESIDUAL_AXIS_MODE", "robust")
DENSITY_DRC_RESIDUAL_KIND = os.environ.get("IP_DENSITY_DRC_RESIDUAL_KIND", "affine_normalized")
DENSITY_DRC_AFFINE_FIT_SCOPE = os.environ.get("IP_DENSITY_DRC_AFFINE_FIT_SCOPE", "central")
DENSITY_DRC_ROBUST_PERCENTILES = _env_percentile_pair("IP_DENSITY_DRC_ROBUST_PERCENTILES", (2.0, 98.0))
DENSITY_DRC_SAVE_RAW_PLOTS = _env_bool("IP_DENSITY_DRC_SAVE_RAW_PLOTS", False)
DENSITY_DRC_SAVE_LOGLOG_PLOTS = _env_bool("IP_DENSITY_DRC_SAVE_LOGLOG_PLOTS", False)
DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS = _env_bool("IP_DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS", False)
DENSITY_DRC_SAVE_LEGACY_ALIAS = _env_bool("IP_DENSITY_DRC_SAVE_LEGACY_ALIAS", True)
DENSITY_DRC_PLOT_LAYOUT = os.environ.get("IP_DENSITY_DRC_PLOT_LAYOUT", "comparison_grid")
DENSITY_DRC_GRID_MAX_POINTS = _env_int("IP_DENSITY_DRC_GRID_MAX_POINTS", 5000)
DENSITY_DRC_GRID_SAVE_PDF = _env_bool("IP_DENSITY_DRC_GRID_SAVE_PDF", True)

DENSITY_TWEEDIE_DIVERGENCE = os.environ.get("IP_DENSITY_TWEEDIE_DIVERGENCE", "auto")
DENSITY_BLEND_DIVERGENCE = os.environ.get("IP_DENSITY_BLEND_DIVERGENCE", "auto")
DENSITY_MATRIX_BLEND_DIVERGENCE = os.environ.get("IP_DENSITY_MATRIX_BLEND_DIVERGENCE", "auto")
DENSITY_LFGI_DIVERGENCE = os.environ.get("IP_DENSITY_LFGI_DIVERGENCE", "auto")
DENSITY_DIV_PROBES = _env_int("IP_DENSITY_DRC_DIV_PROBES", 1)

DENSITY_BASELINES = _env_csv("IP_DENSITY_BASELINES", ("map_laplace",))
DENSITY_RUN_PF_SENSITIVITY = _env_bool("IP_DENSITY_RUN_PF_SENSITIVITY", False)
DENSITY_PF_SENSITIVITY_LABELS = _env_csv("IP_DENSITY_PF_SENSITIVITY_LABELS", ("DENS-LFGI", "DENS-MatrixBlend", "DENS-Tweedie"))
DENSITY_PF_SENSITIVITY_STEPS = _env_int_tuple("IP_DENSITY_PF_SENSITIVITY_STEPS", (32, 64, 128))
DENSITY_PF_SENSITIVITY_TMINS = _env_float_tuple_or_none("IP_DENSITY_PF_SENSITIVITY_TMINS", None)

# Source/eval bank generation.  The default MAP--Laplace initial law is exact for
# this target, so default MALA steps are zero.  Nonzero MALA steps are supported
# for literal MCMC-bank calibration runs.
DENSITY_SOURCE_REQUIRED_N = _required_source_bank_size(N_REF_SIGNAL, N_REF_GATE, DENSITY_BANK_COUPLING)
SOURCE_LABEL = "POSTERIOR"
EVAL_LABEL = "POSTERIOR-EVAL"
SOURCE_INIT = os.environ.get("GIP_SOURCE_INIT", "map_laplace")
SOURCE_N_SAMPLES = _env_int("GIP_SOURCE_N_SAMPLES", DENSITY_SOURCE_REQUIRED_N)
SOURCE_MALA_STEPS = _env_int("GIP_SOURCE_MALA_STEPS", 0)
SOURCE_MALA_BURNIN = _env_int("GIP_SOURCE_MALA_BURNIN", 0)
SOURCE_MALA_DT = _env_float("GIP_SOURCE_MALA_DT", 1e-3)
SOURCE_MALA_PRECOND = _env_bool("GIP_SOURCE_MALA_PRECOND", False)

EVAL_INIT = os.environ.get("GIP_EVAL_INIT", SOURCE_INIT)
EVAL_N_SAMPLES = _env_int("GIP_EVAL_N_SAMPLES", N_REF_EVAL)
EVAL_MALA_STEPS = _env_int("GIP_EVAL_MALA_STEPS", SOURCE_MALA_STEPS)
EVAL_MALA_BURNIN = _env_int("GIP_EVAL_MALA_BURNIN", SOURCE_MALA_BURNIN)
EVAL_MALA_DT = _env_float("GIP_EVAL_MALA_DT", SOURCE_MALA_DT)
EVAL_MALA_PRECOND = _env_bool("GIP_EVAL_MALA_PRECOND", SOURCE_MALA_PRECOND)

MAP_LAPLACE_STARTS = _env_int("IP_DENSITY_MAP_LAPLACE_STARTS", 32)
MAP_LAPLACE_MAX_ITER = _env_int("IP_DENSITY_MAP_LAPLACE_MAX_ITER", 25)
MAP_LAPLACE_TOL = _env_float("IP_DENSITY_MAP_LAPLACE_TOL", 1e-10)
MAP_LAPLACE_RIDGE = _env_float("IP_DENSITY_MAP_LAPLACE_RIDGE", 1e-10)
MAP_LAPLACE_MAX_STEP_NORM = _env_float("IP_DENSITY_MAP_LAPLACE_MAX_STEP_NORM", 10.0)
MAP_LAPLACE_BACKTRACK_STEPS = _env_int("IP_DENSITY_MAP_LAPLACE_BACKTRACK_STEPS", 8)

if SOURCE_N_SAMPLES < DENSITY_SOURCE_REQUIRED_N:
    raise ValueError(
        f"SOURCE_N_SAMPLES={SOURCE_N_SAMPLES} must be >= {DENSITY_SOURCE_REQUIRED_N} for "
        f"N_REF_SIGNAL={N_REF_SIGNAL}, N_REF_GATE={N_REF_GATE}, "
        f"DENSITY_BANK_COUPLING={DENSITY_BANK_COUPLING!r}."
    )
if DENSITY_EVAL_SOURCE == EVAL_LABEL and EVAL_N_SAMPLES < N_REF_EVAL:
    raise ValueError(f"EVAL_N_SAMPLES={EVAL_N_SAMPLES} must be >= N_REF_EVAL={N_REF_EVAL}.")

configure_sampling(
    active_dim=ACTIVE_DIM,
    default_n_gen=DEFAULT_N_GEN,
    hess_min=HESS_MIN,
    hess_max=HESS_MAX,
    leaf_min_prec=HESS_MIN,
    leaf_max_prec=HESS_MAX,
    leaf_abs_scale=1.0,
    gnl_pilot_n=GNL_PILOT_N,
    gnl_stiff_lambda_cut=GNL_STIFF_LAMBDA_CUT,
    gnl_use_dominant_particle_newton=GNL_USE_DOMINANT_PARTICLE_NEWTON,
)

run_ctx = init_run_results("known_z_gaussian_inverse_bench")
RUN_COMMAND_HINT = (
    "GIP_PROBLEM_SEED={problem_seed} SEED={run_seed} "
    "GIP_DIM={d} GIP_OBS_DIM={m} GIP_NOISE_STD={sigma:g} "
    "IP_DENSITY_N_REF_SIGNAL={n_signal} IP_DENSITY_N_REF_GATE={n_gate} "
    "IP_DENSITY_N_REF_EVAL={n_eval} IP_DENSITY_BANK_COUPLING={bank_coupling} "
    "IP_DENSITY_EVAL_SOURCE={eval_source} GIP_SOURCE_N_SAMPLES={src_n} "
    "GIP_EVAL_N_SAMPLES={eval_n} GIP_SOURCE_INIT={src_init} GIP_SOURCE_MALA_STEPS={src_mala} "
    "IP_DENSITY_DRC_PF_STEPS={pf_steps} IP_DENSITY_DRC_PLOT_LAYOUT={layout} python problem.py"
).format(
    problem_seed=PROBLEM_SEED,
    run_seed=seed,
    d=ACTIVE_DIM,
    m=OBS_DIM,
    sigma=NOISE_STD,
    n_signal=N_REF_SIGNAL,
    n_gate=N_REF_GATE,
    n_eval=N_REF_EVAL,
    bank_coupling=DENSITY_BANK_COUPLING,
    eval_source=DENSITY_EVAL_SOURCE,
    src_n=SOURCE_N_SAMPLES,
    eval_n=EVAL_N_SAMPLES,
    src_init=SOURCE_INIT,
    src_mala=SOURCE_MALA_STEPS,
    pf_steps=DENSITY_DRC_PF_STEPS,
    layout=DENSITY_DRC_PLOT_LAYOUT,
)

# ============================================================================
# Model objects and sampler configuration
# ============================================================================

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, lik_aux = make_physics_likelihood(
    solve_forward,
    y_obs_np,
    NOISE_STD,
    use_gauss_newton_hessian=True,
    log_batch_size=256,
    grad_batch_size=256,
    hess_batch_size=256,
)
posterior_score_fn = make_posterior_score_fn(lik_model)


def _posterior_source_config(display_name, n_samples, init, mala_steps, mala_burnin, mala_dt, precond, include_results):
    return {
        "display_name": display_name,
        "init": init,
        "init_weights": "None",
        "transition_w": "ou",
        "n_ref": int(n_samples),
        "n_samples": int(n_samples),
        "init_steps": 0,
        "mala_steps": int(mala_steps),
        "mala_burnin": int(mala_burnin),
        "mala_dt": float(mala_dt),
        "precond_mala": bool(precond),
        "map_laplace_starts": MAP_LAPLACE_STARTS,
        "map_laplace_max_iter": MAP_LAPLACE_MAX_ITER,
        "map_laplace_tol": MAP_LAPLACE_TOL,
        "map_laplace_ridge": MAP_LAPLACE_RIDGE,
        "map_laplace_max_step_norm": MAP_LAPLACE_MAX_STEP_NORM,
        "map_laplace_backtrack_steps": MAP_LAPLACE_BACKTRACK_STEPS,
        "log_mean_ess": False,
        "include_results": bool(include_results),
        "is_reference": bool(include_results),
    }


def _density_eval_config(ref_source, score_init, divergence, display_name):
    return {
        "display_name": display_name,
        "ref_source": ref_source,
        "init": "DRC-R",
        "init_weights": "None",
        "drc_score_init": score_init,
        # The source bank is already a posterior/reference bank.  Do not apply
        # likelihood weights a second time when fitting the frozen score field.
        "drc_score_init_weights": "None",
        "transition_w": "ou",
        "n_ref": N_REF_SIGNAL,
        "n_ref_signal": N_REF_SIGNAL,
        "n_ref_gate": N_REF_GATE,
        "score_gate_bank_coupling": DENSITY_BANK_COUPLING,
        "drc_eval_source": DENSITY_EVAL_SOURCE,
        "drc_eval_n_ref": N_REF_EVAL,
        "drc_eval_bank_coupling": DENSITY_EVAL_BANK_COUPLING,
        "include_results": False,
        "drc_pf_steps": DENSITY_DRC_PF_STEPS,
        "drc_divergence": divergence,
        "drc_div_probes": DENSITY_DIV_PROBES,
        "drc_eval_batch_size": DENSITY_DRC_EVAL_BATCH_SIZE,
        "drc_clip": DENSITY_DRC_CLIP,
        "drc_temperature": DENSITY_DRC_TEMPERATURE,
        "drc_fd_eps": 1e-3,
        "drc_tmin": DENSITY_DRC_TMIN,
        "drc_tmax": DENSITY_DRC_TMAX,
        "drc_store_details": True,
        "drc_energy_benchmark": True,
        "drc_energy_plots": DENSITY_DRC_ENERGY_PLOTS,
        "drc_energy_plot_axis_mode": DENSITY_DRC_PLOT_AXIS_MODE,
        "drc_energy_residual_axis_mode": DENSITY_DRC_RESIDUAL_AXIS_MODE,
        "drc_energy_residual_kind": DENSITY_DRC_RESIDUAL_KIND,
        "drc_energy_affine_fit_scope": DENSITY_DRC_AFFINE_FIT_SCOPE,
        "drc_energy_robust_percentiles": DENSITY_DRC_ROBUST_PERCENTILES,
        "drc_energy_save_raw_plots": DENSITY_DRC_SAVE_RAW_PLOTS,
        "drc_energy_save_loglog_plots": DENSITY_DRC_SAVE_LOGLOG_PLOTS,
        "drc_energy_save_logratio_residual_plots": DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS,
        "drc_energy_save_legacy_alias": DENSITY_DRC_SAVE_LEGACY_ALIAS,
        "drc_energy_plot_layout": DENSITY_DRC_PLOT_LAYOUT,
        "drc_energy_grid_method_order": ("DENS-LFGI", "DENS-MatrixBlend", "DENS-ScalarBlend", "DENS-Tweedie", "DENS-MAP-Laplace"),
        "drc_energy_grid_axis_reference": "DENS-LFGI",
        "drc_energy_grid_max_points": DENSITY_DRC_GRID_MAX_POINTS,
        "drc_energy_grid_save_pdf": DENSITY_DRC_GRID_SAVE_PDF,
    }


SAMPLER_CONFIGS = OrderedDict()
SAMPLER_CONFIGS[SOURCE_LABEL] = _posterior_source_config(
    "Exact Gaussian posterior source bank" if SOURCE_MALA_STEPS == 0 else "Posterior MALA source bank",
    SOURCE_N_SAMPLES,
    SOURCE_INIT,
    SOURCE_MALA_STEPS,
    SOURCE_MALA_BURNIN,
    SOURCE_MALA_DT,
    SOURCE_MALA_PRECOND,
    include_results=True,
)

if DENSITY_EVAL_SOURCE == EVAL_LABEL:
    SAMPLER_CONFIGS[EVAL_LABEL] = _posterior_source_config(
        "Held-out exact Gaussian posterior eval bank" if EVAL_MALA_STEPS == 0 else "Held-out posterior MALA eval bank",
        EVAL_N_SAMPLES,
        EVAL_INIT,
        EVAL_MALA_STEPS,
        EVAL_MALA_BURNIN,
        EVAL_MALA_DT,
        EVAL_MALA_PRECOND,
        include_results=False,
    )

SAMPLER_CONFIGS.update(OrderedDict([
    ("DENS-Tweedie", _density_eval_config(
        DENSITY_REF_SOURCE, "tweedie", DENSITY_TWEEDIE_DIVERGENCE, "Tweedie PF",
    )),
    ("DENS-ScalarBlend", _density_eval_config(
        DENSITY_REF_SOURCE, "scalar_blend", DENSITY_BLEND_DIVERGENCE, "Scalar Blend PF",
    )),
    ("DENS-MatrixBlend", _density_eval_config(
        DENSITY_REF_SOURCE, "matrix_blend", DENSITY_MATRIX_BLEND_DIVERGENCE, "MATRIX BLEND",
    )),
    ("DENS-LFGI", _density_eval_config(
        DENSITY_REF_SOURCE, "lfgi", DENSITY_LFGI_DIVERGENCE, "LFGI--GN PF",
    )),
]))

# ============================================================================
# Experiment execution
# ============================================================================

pipeline = run_standard_sampler_pipeline(
    prior_model,
    lik_model,
    SAMPLER_CONFIGS,
    n_ref=N_REF_SIGNAL,
    build_gnl_banks=BUILD_GNL_BANKS,
    compute_pou=True,
)
precomp = pipeline["precomp"]
samples = pipeline["samples"]
ess_logs = pipeline["ess_logs"]
sampler_run_info = pipeline["sampler_run_info"]
display_names = pipeline["display_names"]
reference_key = pipeline["reference_key"]
reference_title = pipeline["reference_title"]

display_names.update({
    "DENS-Tweedie": "Tweedie",
    "DENS-ScalarBlend": "Scalar Blend",
    "DENS-MatrixBlend": "MATRIX BLEND",
    "DENS-LFGI": "LFGI--GN",
    "DENS-MAP-Laplace": "MAP--Laplace Gaussian",
})

summarize_sampler_run(sampler_run_info)
plot_mean_ess_logs(ess_logs, display_names=display_names)

# Add known-Z to the PF density metric rows.  The shared harness computes these
# metrics when known_logZ is supplied; we recompute here so sampling.py is not
# modified.
metric_kwargs = dict(
    plot_axis_mode=DENSITY_DRC_PLOT_AXIS_MODE,
    residual_axis_mode=DENSITY_DRC_RESIDUAL_AXIS_MODE,
    robust_percentiles=DENSITY_DRC_ROBUST_PERCENTILES,
    residual_kind=DENSITY_DRC_RESIDUAL_KIND,
    affine_fit_scope=DENSITY_DRC_AFFINE_FIT_SCOPE,
    also_save_raw_plots=DENSITY_DRC_SAVE_RAW_PLOTS,
    also_save_loglog_plots=DENSITY_DRC_SAVE_LOGLOG_PLOTS,
    also_save_logratio_residual_plots=DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS,
    save_legacy_alias=DENSITY_DRC_SAVE_LEGACY_ALIAS,
)

for label, details in list(precomp.get("drc_details", {}).items()):
    if not str(label).startswith("DENS-"):
        continue
    details["known_logZ"] = TRUE_LOGZ
    details.setdefault("diagnostics", {})["analytic_logZ"] = TRUE_LOGZ
    details["diagnostics"]["target_kind"] = "linear_gaussian_known_z"
    precomp.setdefault("drc_energy_benchmarks", {})[label] = sampling.compute_drc_energy_benchmark_from_details(
        details,
        label=label,
        save_dir=run_ctx["run_results_dir"],
        run_stem=run_ctx["run_results_stem"],
        make_plots=False,
        known_logZ=TRUE_LOGZ,
        method_family=str(details.get("source_mode", "probability_flow")),
        extra_metrics={"analytic_logZ": TRUE_LOGZ},
        **metric_kwargs,
    )


def _first_density_eval_bank(precomp_dict):
    preferred = ("DENS-LFGI", "DENS-MatrixBlend", "DENS-ScalarBlend", "DENS-Tweedie")
    for lab in preferred:
        bank = precomp_dict.get("eval_banks", {}).get(lab)
        if bank is not None:
            return bank
    for lab in preferred:
        det = precomp_dict.get("drc_details", {}).get(lab)
        if det is not None:
            return {
                "X_ref": det["X_ref"],
                "log_lik_ref": det["log_lik"],
                "bank_name": det.get("eval_bank_name", "details_eval"),
            }
    return None


baseline_eval_bank = _first_density_eval_bank(precomp)
if baseline_eval_bank is not None and any(str(b).lower().replace("-", "_") in {"map_laplace", "laplace_map", "map"} for b in DENSITY_BASELINES):
    try:
        mean_t = torch.tensor(posterior_mean_np, device=sampling.device, dtype=torch.float64)
        precision_t = torch.tensor(posterior_precision_np, device=sampling.device, dtype=torch.float64)
        precision_eigs_t = torch.tensor(POST_PRECISION_EIGS, device=sampling.device, dtype=torch.float64)
        map_df, map_details = sampling.evaluate_gaussian_precision_density_baseline(
            baseline_eval_bank,
            prior_model,
            lik_model,
            mean=mean_t,
            precision=precision_t,
            precision_eigs=precision_eigs_t,
            label="DENS-MAP-Laplace",
            batch_size=max(1, DENSITY_DRC_EVAL_BATCH_SIZE),
            save_dir=run_ctx["run_results_dir"],
            run_stem=run_ctx["run_results_stem"],
            make_plots=False,
            known_logZ=TRUE_LOGZ,
            method_family="analytic_map_laplace_gaussian",
            extra_metrics={
                "analytic_logZ": TRUE_LOGZ,
                "analytic_posterior_mean_norm": float(np.linalg.norm(posterior_mean_np)),
                "analytic_posterior_cov_trace": float(np.trace(posterior_cov_np)),
                "analytic_precision_min_eig": float(POST_PRECISION_EIGS.min()),
                "analytic_precision_max_eig": float(POST_PRECISION_EIGS.max()),
            },
            **metric_kwargs,
        )
        precomp.setdefault("drc_energy_benchmarks", {})["DENS-MAP-Laplace"] = map_df
        precomp.setdefault("drc_details", {})["DENS-MAP-Laplace"] = map_details
        print("\n=== Added exact MAP--Laplace Gaussian density baseline ===")
        print(map_df.to_string(index=False))
    except Exception as exc:
        print(f"WARNING: exact MAP--Laplace density baseline failed and will be skipped: {exc}")
elif baseline_eval_bank is None:
    print("WARNING: no density eval bank found; MAP--Laplace density baseline skipped.")

# Regenerate the comparison grid after adding the exact Gaussian baseline and
# after known-Z diagnostics have been attached to the details.
density_grid_method_order = ("DENS-LFGI", "DENS-MatrixBlend", "DENS-ScalarBlend", "DENS-Tweedie", "DENS-MAP-Laplace")
if DENSITY_DRC_ENERGY_PLOTS and precomp.get("drc_details"):
    try:
        all_drc_details = precomp.get("drc_details", {})
        grid_labels = [lab for lab in density_grid_method_order if lab in all_drc_details]
        grid_labels += [lab for lab in all_drc_details.keys() if lab not in grid_labels and str(lab).startswith("DENS-")]
        details_for_grid = OrderedDict((lab, all_drc_details[lab]) for lab in grid_labels)
        cfg_for_grid = {lab: dict(SAMPLER_CONFIGS.get(lab, {})) for lab in grid_labels}
        cfg_for_grid.setdefault("DENS-MAP-Laplace", {})["display_name"] = "MAP--Laplace Gaussian"
        refreshed_grid = sampling.save_drc_energy_comparison_grid(
            details_for_grid,
            cfg_by_label=cfg_for_grid,
            save_dir=run_ctx["run_results_dir"],
            run_stem=run_ctx["run_results_stem"],
            method_order=density_grid_method_order,
            axis_reference_label="DENS-LFGI" if "DENS-LFGI" in details_for_grid else None,
            plot_axis_mode=DENSITY_DRC_PLOT_AXIS_MODE,
            residual_axis_mode=DENSITY_DRC_RESIDUAL_AXIS_MODE,
            robust_percentiles=DENSITY_DRC_ROBUST_PERCENTILES,
            affine_fit_scope=DENSITY_DRC_AFFINE_FIT_SCOPE,
            residual_kind=DENSITY_DRC_RESIDUAL_KIND,
            max_points=DENSITY_DRC_GRID_MAX_POINTS,
            save_pdf=DENSITY_DRC_GRID_SAVE_PDF,
        )
        precomp["drc_energy_comparison_grid"] = refreshed_grid
        print("Refreshed density-energy comparison grid with known-Z rows and MAP--Laplace baseline.")
    except Exception as exc:
        print(f"WARNING: failed to refresh density-energy comparison grid: {exc}")

# Optional PF-discretization sensitivity diagnostic.  This reuses frozen score
# fields and eval banks; it does not rebuild the source/gate banks.
pf_sensitivity_df = pd.DataFrame()
pf_sensitivity_fig_path = None
if DENSITY_RUN_PF_SENSITIVITY:
    try:
        pf_sensitivity_df, pf_sensitivity_fig_path = run_drc_pf_sensitivity_benchmark(
            precomp,
            SAMPLER_CONFIGS,
            prior_model,
            lik_model,
            labels=DENSITY_PF_SENSITIVITY_LABELS,
            pf_steps_list=DENSITY_PF_SENSITIVITY_STEPS,
            tmin_list=DENSITY_PF_SENSITIVITY_TMINS,
            save_dir=run_ctx["run_results_dir"],
            run_stem=run_ctx["run_results_stem"],
            batch_size=max(1, DENSITY_DRC_EVAL_BATCH_SIZE),
            robust_percentiles=DENSITY_DRC_ROBUST_PERCENTILES,
            affine_fit_scope=DENSITY_DRC_AFFINE_FIT_SCOPE,
            known_logZ=TRUE_LOGZ,
            make_plot=True,
        )
        if not pf_sensitivity_df.empty:
            pf_path = os.path.join(run_ctx["run_results_dir"], f"{run_ctx['run_results_stem']}_pf_sensitivity.csv")
            pf_sensitivity_df.to_csv(pf_path, index=False)
            print(f"Saved PF sensitivity table to {pf_path}")
    except Exception as exc:
        print(f"WARNING: PF sensitivity benchmark failed and will be skipped: {exc}")

# ============================================================================
# Tables and plots
# ============================================================================

# Latent sample metrics are still useful as a sanity check for the source bank.
metrics = compute_latent_metrics(
    samples,
    reference_key,
    alpha_true_np,
    prior_model,
    lik_model,
    posterior_score_fn,
    display_names=display_names,
)

# Add closed-form posterior calibration for the included posterior-source row.
for label, samps in samples.items():
    clean = get_valid_samples(samps)
    if clean.shape[0] == 0:
        continue
    X_np = clean.detach().cpu().numpy() if torch.is_tensor(clean) else np.asarray(clean, dtype=np.float64)
    mean_err = np.linalg.norm(np.mean(X_np, axis=0) - posterior_mean_np)
    cov_err = np.linalg.norm(np.cov(X_np.T) - posterior_cov_np, ord="fro") / (np.linalg.norm(posterior_cov_np, ord="fro") + 1e-12)
    metrics.setdefault(label, {})["RMSE_to_posterior_mean"] = float(mean_err / np.sqrt(ACTIVE_DIM))
    metrics[label]["RelFrob_to_posterior_cov"] = float(cov_err)
    metrics[label]["analytic_logZ"] = float(TRUE_LOGZ)

try:
    plot_pca_histograms(samples, alpha_true_np, display_names=display_names, normalizer=PLOT_NORMALIZER, metrics_dict=metrics)
except Exception as exc:
    print(f"WARNING: PCA histogram plot failed and will be skipped: {exc}")

results_df, results_runinfo_df, results_df_path, results_runinfo_df_path = save_results_tables(
    metrics,
    sampler_run_info,
    n_ref=N_REF_SIGNAL,
    target_name="analytic Gaussian inverse problem",
    display_names=display_names,
    reference_name=reference_title,
)

# Full density/energy rows from PF and the exact MAP--Laplace Gaussian baseline.
drc_energy_tables = precomp.get("drc_energy_benchmarks", {})
if drc_energy_tables:
    drc_energy_df = pd.concat(
        [df for df in drc_energy_tables.values() if isinstance(df, pd.DataFrame) and not df.empty],
        ignore_index=True,
    )
else:
    drc_energy_df = pd.DataFrame()

if not drc_energy_df.empty:
    drc_energy_path = os.path.join(run_ctx["run_results_dir"], f"{run_ctx['run_results_stem']}_known_z_density_energy_full.csv")
    drc_energy_df.to_csv(drc_energy_path, index=False)
    print(f"Saved full known-Z density/energy table to {drc_energy_path}")

    manuscript_density_df = make_density_manuscript_table(
        drc_energy_tables,
        display_names=display_names,
        method_order=density_grid_method_order,
        include_known_z=True,
    )
    manuscript_density_path = os.path.join(run_ctx["run_results_dir"], f"{run_ctx['run_results_stem']}_known_z_density_manuscript_table.csv")
    manuscript_density_tex_path = os.path.join(run_ctx["run_results_dir"], f"{run_ctx['run_results_stem']}_known_z_density_manuscript_table.tex")
    manuscript_density_df.to_csv(manuscript_density_path, index=False)
    with open(manuscript_density_tex_path, "w", encoding="utf-8") as f:
        f.write(manuscript_density_df.to_latex(index=False, escape=False, float_format="%.6g"))
    print(f"Saved manuscript density table to {manuscript_density_path}")
    print(f"Saved manuscript density LaTeX table to {manuscript_density_tex_path}")
else:
    manuscript_density_df = pd.DataFrame()
    print("WARNING: no density/energy benchmark rows were found.")


def make_known_z_calibration_table(df, display_names=None, method_order=None):
    """Return the compact table needed by Sec. 12.7."""
    display_names = display_names or {}
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    order = list(method_order or [])
    if "label" in df.columns:
        labels = list(df["label"].astype(str).unique())
    else:
        labels = [str(i) for i in range(len(df))]
    labels_ordered = [lab for lab in order if lab in labels] + [lab for lab in labels if lab not in order]
    for lab in labels_ordered:
        row_df = df[df["label"].astype(str) == lab] if "label" in df.columns else df.iloc[[int(lab)]]
        if row_df.empty:
            continue
        rec = row_df.iloc[0]
        n_eval = float(rec.get("n_eval", np.nan))
        ess = float(rec.get("raw_logw_ess", np.nan))
        rows.append(OrderedDict([
            ("Method", display_names.get(lab, lab)),
            ("$\\log q$ bias", float(rec.get("pointwise_logq_bias", np.nan))),
            ("$\\log q$ RMSE", float(rec.get("pointwise_logq_rmse", np.nan))),
            ("$|\\widehat{\\log Z}-\\log Z|$", float(rec.get("known_logZ_abs_error", np.nan))),
            ("Correction ESS", ess),
            ("Correction ESS / $n$", ess / n_eval if np.isfinite(ess) and np.isfinite(n_eval) and n_eval > 0 else np.nan),
        ]))
    return pd.DataFrame(rows)


known_z_calibration_df = make_known_z_calibration_table(
    drc_energy_df,
    display_names=display_names,
    method_order=density_grid_method_order,
)
if not known_z_calibration_df.empty:
    known_z_path = os.path.join(run_ctx["run_results_dir"], f"{run_ctx['run_results_stem']}_known_z_calibration_table.csv")
    known_z_tex_path = os.path.join(run_ctx["run_results_dir"], f"{run_ctx['run_results_stem']}_known_z_calibration_table.tex")
    known_z_calibration_df.to_csv(known_z_path, index=False)
    with open(known_z_tex_path, "w", encoding="utf-8") as f:
        # The first five columns match the manuscript placeholder exactly; the
        # ESS fraction is kept in the CSV for auditability but omitted here.
        latex_df = known_z_calibration_df.iloc[:, :5].copy()
        f.write(latex_df.to_latex(index=False, escape=False, float_format="%.6g"))
    print("\n=== Known-Z calibration table for Sec. 12.7 ===")
    print(known_z_calibration_df.to_string(index=False))
    print(f"Saved known-Z calibration table to {known_z_path}")
    print(f"Saved known-Z calibration LaTeX table to {known_z_tex_path}")
else:
    print("WARNING: known-Z calibration table is empty.")

# A compact analytic-problem summary for reproducibility and for updating the
# appendix protocol text.
problem_summary = OrderedDict([
    ("problem_seed", PROBLEM_SEED),
    ("run_seed", seed),
    ("run_seed_base", RUN_SEED_BASE),
    ("run_seed_source", RUN_SEED_SOURCE),
    ("run_seed_offset", RUN_SEED_OFFSET),
    ("run_index_seed_offset_disabled", RUN_INDEX_SEED_OFFSET_DISABLED),
    ("run_index", RUN_INDEX),
    ("operator_seed", OPERATOR_SEED),
    ("seed", seed),
    ("ACTIVE_DIM", ACTIVE_DIM),
    ("OBS_DIM", OBS_DIM),
    ("NOISE_STD", NOISE_STD),
    ("FORWARD_SCALE", FORWARD_SCALE),
    ("FORWARD_COND", FORWARD_COND),
    ("operator_singular_values", A_singular_values),
    ("alpha_true_np", alpha_true_np),
    ("y_obs_np", y_obs_np),
    ("posterior_mean_np", posterior_mean_np),
    ("posterior_cov_diag", np.diag(posterior_cov_np)),
    ("posterior_precision_eigs", POST_PRECISION_EIGS),
    ("analytic_logZ", TRUE_LOGZ),
    ("likelihood_convention", "unnormalized Gaussian likelihood exp(-||Ax-y||^2/(2 sigma^2))"),
])

save_reproducibility_log(
    title="Known-Z analytic Gaussian inverse-problem calibration reproducibility log",
    config=OrderedDict([
        ("run_command_hint", RUN_COMMAND_HINT),
        ("run_results_dir", run_ctx["run_results_dir"]),
        ("GIP_PROBLEM_SEED", PROBLEM_SEED),
        ("SEED", seed),
        ("RUN_SEED_BASE", RUN_SEED_BASE),
        ("RUN_SEED_SOURCE", RUN_SEED_SOURCE),
        ("RUN_SEED_OFFSET", RUN_SEED_OFFSET),
        ("RUN_INDEX_SEED_OFFSET_DISABLED", RUN_INDEX_SEED_OFFSET_DISABLED),
        ("RUN_INDEX", RUN_INDEX),
        ("GIP_OPERATOR_SEED", OPERATOR_SEED),
        ("DENSITY_REF_SOURCE", DENSITY_REF_SOURCE),
        ("DENSITY_EVAL_SOURCE", DENSITY_EVAL_SOURCE),
        ("N_REF_SIGNAL", N_REF_SIGNAL),
        ("N_REF_GATE", N_REF_GATE),
        ("N_REF_EVAL", N_REF_EVAL),
        ("DENSITY_BANK_COUPLING", DENSITY_BANK_COUPLING),
        ("DENSITY_EVAL_BANK_COUPLING", DENSITY_EVAL_BANK_COUPLING),
        ("SOURCE_INIT", SOURCE_INIT),
        ("SOURCE_N_SAMPLES", SOURCE_N_SAMPLES),
        ("SOURCE_MALA_STEPS", SOURCE_MALA_STEPS),
        ("SOURCE_MALA_BURNIN", SOURCE_MALA_BURNIN),
        ("SOURCE_MALA_DT", SOURCE_MALA_DT),
        ("EVAL_INIT", EVAL_INIT),
        ("EVAL_N_SAMPLES", EVAL_N_SAMPLES),
        ("EVAL_MALA_STEPS", EVAL_MALA_STEPS),
        ("DENSITY_DRC_PF_STEPS", DENSITY_DRC_PF_STEPS),
        ("DENSITY_DRC_TMIN", DENSITY_DRC_TMIN),
        ("DENSITY_DRC_TMAX", DENSITY_DRC_TMAX),
        ("DENSITY_DRC_EVAL_BATCH_SIZE", DENSITY_DRC_EVAL_BATCH_SIZE),
        ("DENSITY_TWEEDIE_DIVERGENCE", DENSITY_TWEEDIE_DIVERGENCE),
        ("DENSITY_BLEND_DIVERGENCE", DENSITY_BLEND_DIVERGENCE),
        ("DENSITY_MATRIX_BLEND_DIVERGENCE", DENSITY_MATRIX_BLEND_DIVERGENCE),
        ("DENSITY_LFGI_DIVERGENCE", DENSITY_LFGI_DIVERGENCE),
        ("DENSITY_DRC_ROBUST_PERCENTILES", DENSITY_DRC_ROBUST_PERCENTILES),
        ("DENSITY_BASELINES", DENSITY_BASELINES),
        ("TRUE_LOGZ", TRUE_LOGZ),
        ("HESS_MIN", HESS_MIN),
        ("HESS_MAX", HESS_MAX),
        ("BUILD_GNL_BANKS", BUILD_GNL_BANKS),
    ]),
    extra_sections={
        "Analytic Gaussian inverse problem": problem_summary,
        "Sampler configurations": SAMPLER_CONFIGS,
        "Known-Z calibration table": known_z_calibration_df.to_dict("records") if not known_z_calibration_df.empty else [],
        "PF sensitivity": pf_sensitivity_df.to_dict("records") if not pf_sensitivity_df.empty else [],
    },
)

# Save any figures left open through sampling.py's run-results hook, then zip the
# whole run directory.  The zip will include the density grid PNG/PDF, CSVs, TeX
# tables, and reproducibility log.
try:
    plt.show()
except Exception:
    pass

zip_path = zip_run_results_dir(extra_paths=[pf_sensitivity_fig_path] if pf_sensitivity_fig_path else None)
print(f"\nDone. Run artifacts are in: {run_ctx['run_results_dir']}")
print(f"Zip archive: {zip_path}")
