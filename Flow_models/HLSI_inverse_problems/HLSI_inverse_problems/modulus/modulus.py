# -*- coding: utf-8 -*-
"""Inverse hyperelastic shear-modulus imaging on a regular grid.

This script is a lightweight, compute-conscious modification of the current
Darcy-flow experiment structure. It keeps the shared sampling harness in
``sampling.py`` unchanged and swaps in a simplified generalized neo-Hookean
membrane model with multiple loading conditions and sparse interior
measurement of displacement.

Model summary
-------------
Let u=(u_x,u_y) be a two-component displacement field on a 2D square. We use
an energy density of the form

    W(q; mu) = mu * (0.5 q + 0.25 beta q^2),
    q = |grad u_x|^2 + |grad u_y|^2,

which yields a quasi-linear hyperelastic membrane equilibrium. The spatially
varying shear modulus is mu(x)=mu0 * exp(m(x)), with log-modulus m(x)
parameterized in a 32-dimensional KL basis exactly as in the other inverse
problems.

To keep runtime commensurate with the rest of the suite, the nonlinear PDE is
solved with a small number of Picard iterations over variable-coefficient
linear elliptic solves on a regular grid. Training data consist of sparse
interior displacement observations under two loading conditions; held-out
predictive metrics use a third mixed loading condition on disjoint sensors.
"""

import gc
import os
import sys
from collections import OrderedDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

try:
    from sampling import (
        GaussianPrior,
        compute_field_summary_metrics,
        compute_heldout_predictive_metrics,
        compute_latent_metrics,
        configure_sampling,
        get_valid_samples,
        init_run_results,
        make_physics_likelihood,
        make_posterior_score_fn,
        plot_mean_ess_logs,
        plot_pca_histograms,
        rmse_array,
        run_standard_sampler_pipeline,
        save_reproducibility_log,
        save_results_tables,
        summarize_sampler_run,
        zip_run_results_dir,
    )
except Exception:
    _required_sampling_symbols = [
        'GaussianPrior', 'compute_field_summary_metrics', 'compute_heldout_predictive_metrics',
        'compute_latent_metrics', 'configure_sampling', 'get_valid_samples', 'init_run_results',
        'make_physics_likelihood', 'make_posterior_score_fn', 'plot_mean_ess_logs',
        'plot_pca_histograms', 'rmse_array', 'run_standard_sampler_pipeline',
        'save_reproducibility_log', 'save_results_tables', 'summarize_sampler_run',
        'zip_run_results_dir',
    ]
    _missing_sampling_symbols = [name for name in _required_sampling_symbols if name not in globals()]
    if _missing_sampling_symbols:
        raise ImportError(
            'sampling imports failed and these notebook-scope symbols are missing: ' +
            ', '.join(_missing_sampling_symbols)
        )

# ==========================================
# 0. KL basis generation
# ==========================================
os.makedirs('data', exist_ok=True)

jax.config.update("jax_enable_x64", True)

plt.rcParams.setdefault('figure.dpi', 160)
plt.rcParams.setdefault('savefig.dpi', 300)

# Match the Allen–Cahn ambient plot resolution when possible.
N = 32
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ell = 0.12
sigma_prior = 1.0
q_max = 100

dists = cdist(coords, coords)
C = sigma_prior ** 2 * np.exp(-dists / ell)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/Hyperelastic_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files
# ==========================================
num_observation = 48
num_holdout_observation = 24
num_truncated_series = 32
seed = 42

dimension_of_PoI = N * N

interior_mask = np.ones((N, N), dtype=bool)
interior_mask[0, :] = False
interior_mask[-1, :] = False
interior_mask[:, 0] = False
interior_mask[:, -1] = False
interior_indices = jnp.array(np.where(interior_mask.ravel())[0])

key = jax.random.PRNGKey(seed)
obs_indices_train = np.array(
    jax.random.choice(key, interior_indices, shape=(num_observation,), replace=False)
)
remaining_interior_indices = np.setdiff1d(np.asarray(interior_indices), obs_indices_train)
key_holdout = jax.random.PRNGKey(seed + 1)
obs_indices_holdout = np.array(
    jax.random.choice(
        key_holdout,
        jnp.array(remaining_interior_indices),
        shape=(num_holdout_observation,),
        replace=False,
    )
)
obs_indices = obs_indices_train

# Match the I/O pattern of the other modular experiments.
df_modes = pd.read_csv('data/Hyperelastic_Basis_Modes.csv', header=None)
if isinstance(df_modes.iloc[0, 0], str):
    df_modes = pd.read_csv('data/Hyperelastic_Basis_Modes.csv')

modes_raw = df_modes.to_numpy().flatten()
num_modes_available = modes_raw.size // dimension_of_PoI
full_basis = modes_raw.reshape((dimension_of_PoI, num_modes_available))
basis_truncated = full_basis[:, :num_truncated_series]

pd.DataFrame(basis_truncated).to_csv('data/Hyperelastic_Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices_train).to_csv('data/hyperelastic_obs_locations.csv', index=False, header=False)

df_Basis = pd.read_csv('data/Hyperelastic_Basis.csv', header=None)
df_obs = pd.read_csv('data/hyperelastic_obs_locations.csv', header=None)

basis_raw = df_Basis.to_numpy().flatten()
if basis_raw.size % dimension_of_PoI == 1:
    basis_raw = basis_raw[1:]
basis_raw = basis_raw.astype(np.float64)

if basis_raw.size % dimension_of_PoI != 0:
    raise ValueError(
        f"Basis file size {basis_raw.size} is not divisible by grid size {dimension_of_PoI}."
    )

num_modes_in_file = basis_raw.size // dimension_of_PoI
full_basis = np.reshape(basis_raw, (dimension_of_PoI, num_modes_in_file))
Basis = jnp.array(full_basis[:, :num_truncated_series], dtype=jnp.float64)
Basis_np = np.array(Basis)

obs_raw = df_obs.to_numpy().flatten()
if obs_raw.size == num_observation + 1:
    obs_raw = obs_raw[1:]
obs_raw = obs_raw.astype(int)
if obs_raw.size > num_observation:
    obs_raw = obs_raw[:num_observation]
elif obs_raw.size < num_observation:
    raise ValueError(f"Obs file only has {obs_raw.size} locations, need {num_observation}.")
obs_locations_train = jnp.array(obs_raw, dtype=int)
obs_locations_holdout = jnp.array(obs_indices_holdout, dtype=int)
obs_locations = obs_locations_train

obs_locs_np = np.array(obs_locations_train)
obs_row = obs_locs_np // N
obs_col = obs_locs_np % N
holdout_locs_np = np.array(obs_locations_holdout)
holdout_row = holdout_locs_np // N
holdout_col = holdout_locs_np % N

# Plot-resolution controls: show native 32x32 fields when available; otherwise
# artificially upscale smaller grids so notebook renders match the Allen–Cahn look.
PLOT_NATIVE_MIN_RES = 32
PLOT_UPSCALE_TARGET = 96


def _prepare_plot_panel(field, rows=None, cols=None):
    arr = np.asarray(field)
    if arr.ndim != 2:
        return arr, rows, cols, 'nearest'
    min_side = int(min(arr.shape))
    if min_side >= PLOT_NATIVE_MIN_RES:
        return arr, rows, cols, 'nearest'

    factor = int(np.ceil(PLOT_UPSCALE_TARGET / float(min_side)))
    factor = max(1, factor)
    up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    row_up = None if rows is None else np.asarray(rows, dtype=np.float64) * factor + 0.5 * (factor - 1)
    col_up = None if cols is None else np.asarray(cols, dtype=np.float64) * factor + 0.5 * (factor - 1)
    return up, row_up, col_up, 'nearest'

# ==========================================
# 2. Physics: generalized neo-Hookean membrane surrogate
# ==========================================
NOISE_STD = 0.001
MU0 = 1.0
BETA_NL = 6.0
PICARD_ITERS = 3
PICARD_DAMP = 0.75
COEFF_MIN = 1e-3
COEFF_MAX = 1e3

h = 1.0 / (N - 1)
x_1d = jnp.linspace(0.0, 1.0, N)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d, indexing='xy')

_int_mask = jnp.zeros((N, N), dtype=bool)
_int_mask = _int_mask.at[1:-1, 1:-1].set(True)
_int_rows, _int_cols = jnp.where(_int_mask)
n_int = _int_rows.shape[0]

_int_id = -jnp.ones((N, N), dtype=jnp.int32)
_int_id = _int_id.at[_int_rows, _int_cols].set(jnp.arange(n_int, dtype=jnp.int32))
int_flat = _int_rows * N + _int_cols

# Precompute neighbor bookkeeping for interior assembly.
ir = _int_rows
ic = _int_cols
idx_int = jnp.arange(n_int)

nbr_E = _int_id[ir + 1, ic]
nbr_W = _int_id[ir - 1, ic]
nbr_N = _int_id[ir, ic + 1]
nbr_S = _int_id[ir, ic - 1]

mask_E_int = nbr_E >= 0
mask_W_int = nbr_W >= 0
mask_N_int = nbr_N >= 0
mask_S_int = nbr_S >= 0

# Boundary values for RHS when a neighbor lies on the boundary.
bdry_E_r = ir + 1
bdry_E_c = ic
bdry_W_r = ir - 1
bdry_W_c = ic
bdry_N_r = ir
bdry_N_c = ic + 1
bdry_S_r = ir
bdry_S_c = ic - 1


def _make_bc(load_id: int) -> np.ndarray:
    """Construct displacement-controlled boundary actuation for one load case."""
    xx = x_1d
    yy = x_1d
    bc = np.zeros((N, N, 2), dtype=np.float64)

    if load_id == 0:
        # Horizontal pull on the right edge.
        bc[-1, :, 0] = 0.08 * np.sin(np.pi * np.asarray(yy))
    elif load_id == 1:
        # Vertical lift on the top edge plus mild lateral bias on the right.
        bc[:, -1, 1] = 0.08 * np.sin(np.pi * np.asarray(xx))
        bc[-1, :, 0] = -0.02 * np.sin(np.pi * np.asarray(yy))
    elif load_id == 2:
        # Held-out mixed load with both right-edge and top-edge actuation.
        bc[-1, :, 0] = 0.05 * np.sin(np.pi * np.asarray(yy))
        bc[-1, :, 1] = 0.03 * np.sin(2.0 * np.pi * np.asarray(yy))
        bc[:, -1, 0] += 0.02 * np.sin(np.pi * np.asarray(xx))
        bc[:, -1, 1] += 0.06 * np.sin(np.pi * np.asarray(xx))
    else:
        raise ValueError(f"Unknown load_id={load_id}")

    # Keep the left edge fixed to prevent rigid translations.
    bc[0, :, :] = 0.0
    # Bottom edge fixed as a stable support, except where another load already vanishes.
    bc[:, 0, :] = 0.0
    return bc


BC_CASES_NP = np.stack([_make_bc(0), _make_bc(1), _make_bc(2)], axis=0)
BC_CASES = jnp.array(BC_CASES_NP, dtype=jnp.float64)
TRAIN_LOAD_IDS = (0, 1)
HELDOUT_LOAD_ID = 2


@jax.jit
def _assemble_scalar_system(a_field, bc_component):
    """Assemble the interior linear system for -div(a grad u)=0 with Dirichlet BC."""
    h2 = h * h

    a_xp = 2.0 * a_field[:-1, :] * a_field[1:, :] / (a_field[:-1, :] + a_field[1:, :] + 1e-30)
    a_yp = 2.0 * a_field[:, :-1] * a_field[:, 1:] / (a_field[:, :-1] + a_field[:, 1:] + 1e-30)

    c_E = a_xp[ir, ic] / h2
    c_W = a_xp[ir - 1, ic] / h2
    c_N = a_yp[ir, ic] / h2
    c_S = a_yp[ir, ic - 1] / h2

    diag = c_E + c_W + c_N + c_S

    A = jnp.zeros((n_int, n_int), dtype=jnp.float64)
    A = A.at[idx_int, idx_int].add(diag)
    A = A.at[idx_int, nbr_E].add(jnp.where(mask_E_int, -c_E, 0.0))
    A = A.at[idx_int, nbr_W].add(jnp.where(mask_W_int, -c_W, 0.0))
    A = A.at[idx_int, nbr_N].add(jnp.where(mask_N_int, -c_N, 0.0))
    A = A.at[idx_int, nbr_S].add(jnp.where(mask_S_int, -c_S, 0.0))

    rhs = jnp.where(mask_E_int, 0.0, c_E * bc_component[bdry_E_r, bdry_E_c])
    rhs = rhs + jnp.where(mask_W_int, 0.0, c_W * bc_component[bdry_W_r, bdry_W_c])
    rhs = rhs + jnp.where(mask_N_int, 0.0, c_N * bc_component[bdry_N_r, bdry_N_c])
    rhs = rhs + jnp.where(mask_S_int, 0.0, c_S * bc_component[bdry_S_r, bdry_S_c])
    return A, rhs


@jax.jit
def _solve_scalar_dirichlet(a_field, bc_component):
    A, rhs = _assemble_scalar_system(a_field, bc_component)
    u_int = jnp.linalg.solve(A, rhs)
    u_full = jnp.array(bc_component)
    u_full = u_full.at[_int_rows, _int_cols].set(u_int)
    return u_full


@jax.jit
def _strain_measure(U):
    q = jnp.zeros((N, N), dtype=jnp.float64)
    ux = U[..., 0]
    uy = U[..., 1]

    dux_dx = (ux[2:, 1:-1] - ux[:-2, 1:-1]) / (2.0 * h)
    dux_dy = (ux[1:-1, 2:] - ux[1:-1, :-2]) / (2.0 * h)
    duy_dx = (uy[2:, 1:-1] - uy[:-2, 1:-1]) / (2.0 * h)
    duy_dy = (uy[1:-1, 2:] - uy[1:-1, :-2]) / (2.0 * h)

    q_int = dux_dx ** 2 + dux_dy ** 2 + duy_dx ** 2 + duy_dy ** 2
    q = q.at[1:-1, 1:-1].set(q_int)
    return q


@jax.jit
def _effective_coeff(mu_field, U):
    q = _strain_measure(U)
    a = mu_field * (1.0 + BETA_NL * q)
    return jnp.clip(a, COEFF_MIN, COEFF_MAX)


@jax.jit
def _solve_one_load(alpha, bc):
    log_mu = jnp.reshape(Basis @ alpha, (N, N))
    mu_field = MU0 * jnp.exp(log_mu)

    # Start from the linear membrane response.
    u0 = _solve_scalar_dirichlet(mu_field, bc[..., 0])
    v0 = _solve_scalar_dirichlet(mu_field, bc[..., 1])
    U0 = jnp.stack([u0, v0], axis=-1)

    def body_fun(_, U_prev):
        a_field = _effective_coeff(mu_field, U_prev)
        u_new = _solve_scalar_dirichlet(a_field, bc[..., 0])
        v_new = _solve_scalar_dirichlet(a_field, bc[..., 1])
        U_new = jnp.stack([u_new, v_new], axis=-1)
        return PICARD_DAMP * U_new + (1.0 - PICARD_DAMP) * U_prev

    return jax.lax.fori_loop(0, PICARD_ITERS, body_fun, U0)


@jax.jit
def solve_forward(alpha):
    obs_chunks = []
    for load_id in TRAIN_LOAD_IDS:
        U = _solve_one_load(alpha, BC_CASES[load_id])
        obs_chunks.append(U.reshape(-1, 2)[obs_locations_train].reshape(-1))
    return jnp.concatenate(obs_chunks, axis=0)


@jax.jit
def solve_forward_holdout(alpha):
    U = _solve_one_load(alpha, BC_CASES[HELDOUT_LOAD_ID])
    return U.reshape(-1, 2)[obs_locations_holdout].reshape(-1)


@jax.jit
def solve_full_displacement(alpha, load_id):
    return _solve_one_load(alpha, BC_CASES[load_id])


# ==========================================
# Shared sampling configuration
# ==========================================
ACTIVE_DIM = num_truncated_series
HESS_MIN = 1e-4
HESS_MAX = 1e6
GNL_PILOT_N = 768
GNL_STIFF_LAMBDA_CUT = HESS_MAX
GNL_USE_DOMINANT_PARTICLE_NEWTON = True
DEFAULT_N_GEN = 500
N_REF = 10000
BUILD_GNL_BANKS = False

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
run_ctx = init_run_results('hyperelastic_shear_hlsi')

# ==========================================
# 3. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
alpha_true_np = np.random.randn(ACTIVE_DIM) * 0.45

y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)
y_holdout_clean_np = np.array(solve_forward_holdout(jnp.array(alpha_true_np)))
y_holdout_obs_np = y_holdout_clean_np + np.random.normal(0.0, NOISE_STD, size=y_holdout_clean_np.shape)

batch_solve_forward_holdout = jax.jit(jax.vmap(solve_forward_holdout))
HELDOUT_BATCH_SIZE = 8

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, lik_aux = make_physics_likelihood(
    solve_forward,
    y_obs_np,
    NOISE_STD,
    use_gauss_newton_hessian=True,
    log_batch_size=50,
    grad_batch_size=25,
    hess_batch_size=2,
)
posterior_score_fn = make_posterior_score_fn(lik_model)


SAMPLER_CONFIGS = OrderedDict([
    ('MALA', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'precond_mala': False, 'is_reference': True}),
    ('Precond MALA', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 3e-3, 'precond_mala': True, 'is_reference': True}),
    ('Ref_Laplace', {'init': 'Ref_Laplace', 'init_weights': 'WC', 'init_steps': 0, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),
    ('Tweedie', {'init': 'tweedie', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('Blend', {'init': 'blend', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),

    ('HLSI', {'init': 'HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI', {'init': 'HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI', {'init': 'HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),
    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-WC-HLSI', {'init': 'CE-HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-PoU-HLSI', {'init': 'CE-HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),

    ('KAPPA05_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.5}),
    ('KAPPA10_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),

    ('KAPPA05_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.5, 'log_mean_ess': True}),
    ('KAPPA10_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),
])


pipeline = run_standard_sampler_pipeline(
    prior_model,
    lik_model,
    SAMPLER_CONFIGS,
    n_ref=N_REF,
    build_gnl_banks=BUILD_GNL_BANKS,
    compute_pou=True,
)
samples = pipeline['samples']
ess_logs = pipeline['ess_logs']
sampler_run_info = pipeline['sampler_run_info']
display_names = pipeline['display_names']

summarize_sampler_run(sampler_run_info)
plot_mean_ess_logs(ess_logs, display_names=display_names)

metrics = compute_latent_metrics(
    samples,
    'MALA',
    alpha_true_np,
    prior_model,
    lik_model,
    posterior_score_fn,
    display_names=display_names,
)


def reconstruct_log_shear(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))



def latent_to_log_shear(alpha):
    return reconstruct_log_shear(np.asarray(alpha)[None, :])[0]



def solve_displacement_field(alpha_vec, load_id):
    return np.array(solve_full_displacement(jnp.array(alpha_vec), load_id))


true_field = latent_to_log_shear(alpha_true_np)
true_mu = MU0 * np.exp(true_field)
true_disp_holdout = solve_displacement_field(alpha_true_np, HELDOUT_LOAD_ID)
true_dispmag_holdout = np.linalg.norm(true_disp_holdout, axis=-1)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_log_shear,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_clean_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

try:
    metrics = compute_heldout_predictive_metrics(
        samples,
        metrics,
        heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
        batched_forward_eval_fn=lambda a_batch: np.asarray(
            batch_solve_forward_holdout(jnp.asarray(a_batch, dtype=jnp.float64))
        ),
        batched_forward_eval_batch_size=HELDOUT_BATCH_SIZE,
        y_holdout_obs_np=y_holdout_obs_np,
        noise_std=NOISE_STD,
        display_names=display_names,
        min_valid=10,
    )
except Exception as exc:
    print(f"WARNING: held-out predictive metrics failed and will be skipped: {exc}")

mean_displacements = {}
mean_disp_magnitudes = {}
mean_shear_moduli = {}
norm_true_disp = np.linalg.norm(true_disp_holdout) + 1e-12

print('\n=== Hyperelastic physical-space metrics ===')
print(
    f"{'Method':<24} | {'LogMu RelL2(%)':<16} | {'Pearson':<10} | {'RMSE_a':<12} | "
    f"{'DispRel':<12} | {'TrainSensorRel':<14} | {'HeldoutNLL':<12} | {'HeldoutStdSq':<12}"
)
print('-' * 132)
for label in [lab for lab in samples.keys() if lab in mean_fields]:
    mean_latent = np.asarray(metrics[label]['mean_latent'])
    mean_disp = solve_displacement_field(mean_latent, HELDOUT_LOAD_ID)
    mean_displacements[label] = mean_disp
    mean_dispmag = np.linalg.norm(mean_disp, axis=-1)
    mean_disp_magnitudes[label] = mean_dispmag
    mean_mu = MU0 * np.exp(mean_fields[label])
    mean_shear_moduli[label] = mean_mu

    disp_rel = float(np.linalg.norm(mean_disp - true_disp_holdout) / norm_true_disp)
    metrics[label]['RMSE_disp'] = rmse_array(mean_disp, true_disp_holdout)
    metrics[label]['RelL2_disp'] = disp_rel
    logmu_rel_pct = 100.0 * float(metrics[label]['RelL2_field'])
    print(
        f"{display_names.get(label, label):<24} | {logmu_rel_pct:<16.4f} | "
        f"{metrics[label]['Pearson_field']:<10.4f} | {metrics[label]['RMSE_alpha']:<12.4e} | "
        f"{disp_rel:<12.4e} | {metrics[label]['FwdRelErr']:<14.4e} | "
        f"{metrics[label].get('HeldoutPredNLL', np.nan):<12.4e} | "
        f"{metrics[label].get('HeldoutStdResSq', np.nan):<12.4e}"
    )

plot_pca_histograms(
    samples,
    alpha_true_np,
    display_names=display_names,
)

results_df, results_runinfo_df, results_df_path, results_runinfo_df_path = save_results_tables(
    metrics,
    sampler_run_info,
    n_ref=N_REF,
    target_name='Hyperelastic shear-modulus imaging',
    display_names=display_names,
    reference_name=display_names.get('MALA', 'MALA'),
)

save_reproducibility_log(
    title='Hyperelastic shear-modulus HLSI run reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'BETA_NL': BETA_NL,
        'DEFAULT_N_GEN': DEFAULT_N_GEN,
        'GNL_PILOT_N': GNL_PILOT_N,
        'GNL_STIFF_LAMBDA_CUT': GNL_STIFF_LAMBDA_CUT,
        'GNL_USE_DOMINANT_PARTICLE_NEWTON': GNL_USE_DOMINANT_PARTICLE_NEWTON,
        'HESS_MAX': HESS_MAX,
        'HESS_MIN': HESS_MIN,
        'MU0': MU0,
        'N': N,
        'NOISE_STD': NOISE_STD,
        'N_REF': N_REF,
        'PICARD_DAMP': PICARD_DAMP,
        'PICARD_ITERS': PICARD_ITERS,
        'PLOT_NATIVE_MIN_RES': PLOT_NATIVE_MIN_RES,
        'PLOT_UPSCALE_TARGET': PLOT_UPSCALE_TARGET,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'TRAIN_LOAD_IDS': TRAIN_LOAD_IDS,
        'HELDOUT_LOAD_ID': HELDOUT_LOAD_ID,
        'USE_GAUSS_NEWTON_HESSIAN': True,
        'C': C,
        'X': X,
        'Y': Y,
        'dimension_of_PoI': dimension_of_PoI,
        'display_names': display_names,
        'num_modes_available': num_modes_available,
        'num_observation': num_observation,
        'num_holdout_observation': num_holdout_observation,
        'num_truncated_series': num_truncated_series,
        'obs_indices_train': obs_indices_train,
        'obs_indices_holdout': obs_indices_holdout,
        'sampler_run_info': sampler_run_info,
        'ell': ell,
        'sigma_prior': sigma_prior,
    },
    extra_sections={
        'saved_results_files': {'metrics_csv': results_df_path, 'runinfo_csv': results_runinfo_df_path},
    },
)

# ==========================================
# 4. Problem-specific visualization
# ==========================================
print('\nVisualizing hyperelastic shear-modulus reconstructions...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1

fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))

vis_anchor_key = 'MALA' if 'MALA' in mean_fields else next(iter(mean_fields.keys()))

vmin = float(np.min(true_field))
vmax = float(np.max(true_field))

max_err = 1e-12
max_std = 1e-12
for label in methods_to_plot:
    mean_f = mean_fields[label]
    max_err = max(max_err, float(np.abs(mean_f - true_field).max()))
    samps = get_valid_samples(samples[label])[:500]
    if samps.shape[0] > 0:
        fields = reconstruct_log_shear(samps[:, :ACTIVE_DIM])
        max_std = max(max_std, float(np.std(fields, axis=0).max()))

true_panel, obs_row_plot, obs_col_plot, field_interp = _prepare_plot_panel(true_field, obs_row, obs_col)
_, holdout_row_plot, holdout_col_plot, _ = _prepare_plot_panel(true_field, holdout_row, holdout_col)
im0 = axes[0, 0].imshow(true_panel, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation=field_interp)
axes[0, 0].scatter(obs_col_plot, obs_row_plot, c='lime', s=8, marker='.', alpha=0.7, label='Train sensors')
axes[0, 0].scatter(holdout_col_plot, holdout_row_plot, c='cyan', s=8, marker='x', alpha=0.6, label='Held-out sensors')
axes[0, 0].set_title('Ground Truth\nLog-shear $m(x)$', fontsize=18)
axes[0, 0].axis('off')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
axes[3, 0].imshow(true_panel, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation=field_interp)
axes[3, 0].set_title('Ground Truth', fontsize=14)
axes[3, 0].axis('off')
axes[1, 0].axis('off')
axes[2, 0].axis('off')

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_f = mean_fields[label]

    mean_panel, obs_row_panel, obs_col_panel, panel_interp = _prepare_plot_panel(mean_f, obs_row, obs_col)
    axes[0, col].imshow(mean_panel, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation=panel_interp)
    axes[0, col].scatter(obs_col_panel, obs_row_panel, c='lime', s=8, marker='.', alpha=0.5)
    axes[0, col].set_title(f"{display_names.get(label, label)}\nMean Posterior", fontsize=18)
    axes[0, col].axis('off')

    err_f = np.abs(mean_f - true_field)
    err_panel, _, _, _ = _prepare_plot_panel(err_f)
    axes[1, col].imshow(err_panel, cmap='inferno', origin='lower', vmin=0, vmax=max_err, interpolation=panel_interp)
    axes[1, col].set_title(f"Error Map\n(Max: {err_f.max():.2f})", fontsize=16)
    axes[1, col].axis('off')

    samps = get_valid_samples(samples[label])[:1000]
    if samps.shape[0] > 0:
        fields = reconstruct_log_shear(samps[:, :ACTIVE_DIM])
        std_f = np.std(fields, axis=0)
    else:
        std_f = np.zeros_like(true_field)
    std_panel, _, _, _ = _prepare_plot_panel(std_f)
    axes[2, col].imshow(std_panel, cmap='viridis', origin='lower', vmin=0, vmax=max_std, interpolation=panel_interp)
    axes[2, col].set_title(f"Uncertainty\n(Max std: {std_f.max():.2f})", fontsize=16)
    axes[2, col].axis('off')

    if samps.shape[0] > 0:
        sample_field = reconstruct_log_shear(samps[:1, :ACTIVE_DIM])[0]
        sample_panel, _, _, _ = _prepare_plot_panel(sample_field)
        axes[3, col].imshow(sample_panel, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation=panel_interp)
        axes[3, col].set_title('Posterior Sample', fontsize=14)
    else:
        axes[3, col].text(0.5, 0.5, 'No valid\nsamples', ha='center', va='center', transform=axes[3, col].transAxes)
    axes[3, col].axis('off')

plt.suptitle(f'Inverse hyperelastic shear imaging (d={ACTIVE_DIM})', fontsize=22, y=1.01)
plt.tight_layout()
plt.show()

print('\nVisualizing held-out displacement magnitude under mixed loading...')
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

true_dmin = float(np.min(true_dispmag_holdout))
true_dmax = float(np.max(true_dispmag_holdout))
true_disp_panel, holdout_row_disp, holdout_col_disp, disp_interp = _prepare_plot_panel(true_dispmag_holdout, holdout_row, holdout_col)
im_true_disp = axes2[0].imshow(true_disp_panel, cmap='viridis', origin='lower', vmin=true_dmin, vmax=true_dmax, interpolation=disp_interp)
axes2[0].scatter(holdout_col_disp, holdout_row_disp, c='red', s=12, marker='x', alpha=0.7, label='Held-out sensors')
axes2[0].set_title('Ground Truth\nHeld-out |u(x)|', fontsize=14)
axes2[0].axis('off')
axes2[0].legend(fontsize=8, loc='upper right')
plt.colorbar(im_true_disp, ax=axes2[0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_dispmag = mean_disp_magnitudes.get(label)
    if mean_dispmag is None:
        axes2[col].axis('off')
        continue
    disp_panel, holdout_row_panel, holdout_col_panel, disp_panel_interp = _prepare_plot_panel(mean_dispmag, holdout_row, holdout_col)
    axes2[col].imshow(disp_panel, cmap='viridis', origin='lower', vmin=true_dmin, vmax=true_dmax, interpolation=disp_panel_interp)
    axes2[col].scatter(holdout_col_panel, holdout_row_panel, c='red', s=12, marker='x', alpha=0.5)
    axes2[col].set_title(f"{display_names.get(label, label)}\nHeld-out |u(x)|", fontsize=14)
    axes2[col].axis('off')

plt.suptitle('Held-out mixed loading: displacement magnitude', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

print('\nVisualizing shear modulus fields $\\mu(x)=\\mu_0 e^{m(x)}$...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

mu_vmin = float(np.min(true_mu))
mu_vmax = float(np.max(true_mu))
true_mu_panel, _, _, mu_interp = _prepare_plot_panel(true_mu)
im_true_mu = axes3[0].imshow(true_mu_panel, cmap='magma', origin='lower', vmin=mu_vmin, vmax=mu_vmax, interpolation=mu_interp)
axes3[0].set_title('Ground Truth\n$\\mu(x)=\\mu_0 e^{m(x)}$', fontsize=14)
axes3[0].axis('off')
plt.colorbar(im_true_mu, ax=axes3[0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_mu = mean_shear_moduli.get(label)
    if mean_mu is None:
        axes3[col].axis('off')
        continue
    mean_mu_panel, _, _, mu_panel_interp = _prepare_plot_panel(mean_mu)
    axes3[col].imshow(mean_mu_panel, cmap='magma', origin='lower', vmin=mu_vmin, vmax=mu_vmax, interpolation=mu_panel_interp)
    axes3[col].set_title(f"{display_names.get(label, label)}\n$\\mu(x)$", fontsize=14)
    axes3[col].axis('off')

plt.suptitle(f'Inverse hyperelastic imaging (d={ACTIVE_DIM}): shear modulus field', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Hyperelastic shear-modulus HLSI pipeline complete ===')
