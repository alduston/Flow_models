# -*- coding: utf-8 -*-
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

# ==========================================
# 0. KL basis generation (match old script)
# ==========================================
os.makedirs('data', exist_ok=True)

N = 32
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ell = 0.1
sigma_prior = 1.0
q_max = 100

dists = cdist(coords, coords)
C = sigma_prior ** 2 * np.exp(-dists / ell)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/Darcy_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files (follow old I/O path exactly)
# ==========================================
num_observation = 120
num_holdout_observation = 30
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
    jax.random.choice(key_holdout, jnp.array(remaining_interior_indices), shape=(num_holdout_observation,), replace=False)
)
obs_indices = obs_indices_train

# Load / truncate / resave exactly like the old script rather than using the
# in-memory eigendecomposition directly. This keeps the modular version aligned
# with the old data-generation path.
df_modes = pd.read_csv('data/Darcy_Basis_Modes.csv', header=None)
if isinstance(df_modes.iloc[0, 0], str):
    df_modes = pd.read_csv('data/Darcy_Basis_Modes.csv')

modes_raw = df_modes.to_numpy().flatten()
num_modes_available = modes_raw.size // dimension_of_PoI
full_basis = modes_raw.reshape((dimension_of_PoI, num_modes_available))
basis_truncated = full_basis[:, :num_truncated_series]

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices_train).to_csv('data/obs_locations.csv', index=False, header=False)

# Match old script's reload-from-disk path too.
df_Basis = pd.read_csv('data/Basis.csv', header=None)
df_obs = pd.read_csv('data/obs_locations.csv', header=None)

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

# ==========================================
# 2. Physics: Darcy flow
# ==========================================
jax.config.update("jax_enable_x64", True)

NOISE_STD = 0.001

h = 1.0 / (N - 1)
x_1d = jnp.linspace(0.0, 1.0, N)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d)
f_darcy = jnp.ones((N, N), dtype=jnp.float64)

_int_mask = jnp.zeros((N, N), dtype=bool)
_int_mask = _int_mask.at[1:-1, 1:-1].set(True)
_int_rows, _int_cols = jnp.where(_int_mask)
n_int = _int_rows.shape[0]

_int_id = -jnp.ones((N, N), dtype=jnp.int32)
_int_id = _int_id.at[_int_rows, _int_cols].set(jnp.arange(n_int, dtype=jnp.int32))
int_flat = _int_rows * N + _int_cols


def _assemble_darcy_vectorized(k_field):
    """
    Vectorized assembly of the interior Darcy stiffness matrix using the
    5-point finite-difference stencil with harmonic face averages.
    """
    h2 = h * h

    k_xp = 2.0 * k_field[:-1, :] * k_field[1:, :] / (k_field[:-1, :] + k_field[1:, :] + 1e-30)
    k_yp = 2.0 * k_field[:, :-1] * k_field[:, 1:] / (k_field[:, :-1] + k_field[:, 1:] + 1e-30)

    ir = _int_rows
    ic = _int_cols

    c_E = k_xp[ir, ic] / h2
    c_W = k_xp[ir - 1, ic] / h2
    c_N = k_yp[ir, ic] / h2
    c_S = k_yp[ir, ic - 1] / h2

    diag = c_E + c_W + c_N + c_S
    idx = jnp.arange(n_int)

    nbr_E = _int_id[ir + 1, ic]
    nbr_W = _int_id[ir - 1, ic]
    nbr_N = _int_id[ir, ic + 1]
    nbr_S = _int_id[ir, ic - 1]

    A = jnp.zeros((n_int, n_int), dtype=jnp.float64)
    A = A.at[idx, idx].add(diag)
    A = A.at[idx, nbr_E].add(jnp.where(nbr_E >= 0, -c_E, 0.0))
    A = A.at[idx, nbr_W].add(jnp.where(nbr_W >= 0, -c_W, 0.0))
    A = A.at[idx, nbr_N].add(jnp.where(nbr_N >= 0, -c_N, 0.0))
    A = A.at[idx, nbr_S].add(jnp.where(nbr_S >= 0, -c_S, 0.0))

    rhs = f_darcy[_int_rows, _int_cols]
    return A, rhs


@jax.jit
def solve_forward(alpha):
    log_k = jnp.reshape(Basis @ alpha, (N, N))
    k_field = jnp.exp(log_k)
    A, rhs = _assemble_darcy_vectorized(k_field)
    p_int = jnp.linalg.solve(A, rhs)
    p_full = jnp.zeros(N * N, dtype=jnp.float64)
    p_full = p_full.at[int_flat].set(p_int)
    return p_full[obs_locations_train]


@jax.jit
def solve_forward_holdout(alpha):
    log_k = jnp.reshape(Basis @ alpha, (N, N))
    k_field = jnp.exp(log_k)
    A, rhs = _assemble_darcy_vectorized(k_field)
    p_int = jnp.linalg.solve(A, rhs)
    p_full = jnp.zeros(N * N, dtype=jnp.float64)
    p_full = p_full.at[int_flat].set(p_int)
    return p_full[obs_locations_holdout]


@jax.jit
def solve_full_pressure(alpha):
    log_k = jnp.reshape(Basis @ alpha, (N, N))
    k_field = jnp.exp(log_k)
    A, rhs = _assemble_darcy_vectorized(k_field)
    p_int = jnp.linalg.solve(A, rhs)
    p_full = jnp.zeros(N * N, dtype=jnp.float64)
    p_full = p_full.at[int_flat].set(p_int)
    return p_full.reshape(N, N)


# ==========================================
# Shared sampling configuration
# ==========================================
ACTIVE_DIM = num_truncated_series
HESS_MIN = 1e-4
HESS_MAX = 1e6
GNL_PILOT_N = 1024
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
run_ctx = init_run_results('darcy_wc_ce_hlsi')

# ==========================================
# 3. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
alpha_true_np = np.random.randn(ACTIVE_DIM) * 0.5

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

'''
SAMPLER_CONFIGS = OrderedDict([
    ('MALA', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'precond_mala': False, 'is_reference': True}),
    ('Precond MALA', { 'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 4e-3, 'precond_mala': True, 'is_reference': True,}),
    ('Ref_Laplace', {'init': 'Ref_Laplace', 'init_weights': 'WC', 'init_steps': 0, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),
    ('Tweedie', {'init': 'tweedie', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('Blend', {'init': 'blend', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),

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
'''

SAMPLER_CONFIGS = OrderedDict([
    ('MALA', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'is_reference': True}),
    ('HLSI-OU', {'init': 'HLSI', 'init_weights': 'L', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('HLSI-Surr', {'init': 'HLSI', 'init_weights': 'L', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI-OU', {'init': 'CE-HLSI', 'init_weights': 'L', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI-Surr', {'init': 'CE-HLSI', 'init_weights': 'L', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI-OU', {'init': 'HLSI', 'init_weights': 'WC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI-Surr', {'init': 'HLSI', 'init_weights': 'WC', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI-OU', {'init': 'HLSI', 'init_weights': 'PoU', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI-Surr', {'init': 'HLSI', 'init_weights': 'PoU', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
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

Basis_np = np.array(Basis)
obs_locs_np = np.array(obs_locations)
obs_row = obs_locs_np // N
obs_col = obs_locs_np % N


def reconstruct_log_permeability(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))



def latent_to_log_permeability(alpha):
    return reconstruct_log_permeability(np.asarray(alpha)[None, :])[0]



def solve_pressure_field(alpha_vec):
    return np.array(solve_full_pressure(jnp.array(alpha_vec)))


true_field = latent_to_log_permeability(alpha_true_np)
true_pressure = solve_pressure_field(alpha_true_np)
true_perm = np.exp(true_field)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_log_permeability,
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

mean_pressures = {}
mean_permeabilities = {}
norm_true_pressure = np.linalg.norm(true_pressure) + 1e-12

print('\n=== Darcy physical-space metrics ===')
print(f"{'Method':<24} | {'LogPerm RelL2(%)':<18} | {'Pearson':<10} | {'RMSE_a':<12} | {'PressureRel':<12} | {'SensorRel':<12}")
print('-' * 113)
for label in [lab for lab in samples.keys() if lab in mean_fields]:
    mean_latent = np.asarray(metrics[label]['mean_latent'])
    mean_pressure = solve_pressure_field(mean_latent)
    mean_pressures[label] = mean_pressure
    mean_perm = np.exp(mean_fields[label])
    mean_permeabilities[label] = mean_perm
    pressure_rel = float(np.linalg.norm(mean_pressure - true_pressure) / norm_true_pressure)
    metrics[label]['RMSE_pressure'] = rmse_array(mean_pressure, true_pressure)
    metrics[label]['RelL2_pressure'] = pressure_rel
    logperm_rel_pct = 100.0 * float(metrics[label]['RelL2_field'])
    print(
        f"{display_names.get(label, label):<24} | {logperm_rel_pct:<18.4f} | "
        f"{metrics[label]['Pearson_field']:<10.4f} | {metrics[label]['RMSE_alpha']:<12.4e} | {pressure_rel:<12.4e} | {metrics[label]['FwdRelErr']:<12.4e} | "
        f"{metrics[label].get('HeldoutPredNLL', np.nan):<12.4e} | {metrics[label].get('HeldoutStdResSq', np.nan):<12.4e}"
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
    target_name='Darcy flow log-permeability',
    display_names=display_names,
    reference_name=display_names.get('MALA (prior)', 'Prior'),
)

save_reproducibility_log(
    title='Darcy flow HLSI run reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'C': C,
        'DEFAULT_N_GEN': DEFAULT_N_GEN,
        'GNL_PILOT_N': GNL_PILOT_N,
        'GNL_STIFF_LAMBDA_CUT': GNL_STIFF_LAMBDA_CUT,
        'GNL_USE_DOMINANT_PARTICLE_NEWTON': GNL_USE_DOMINANT_PARTICLE_NEWTON,
        'HESS_MAX': HESS_MAX,
        'HESS_MIN': HESS_MIN,
        'N': N,
        'NOISE_STD': NOISE_STD,
        'N_REF': N_REF,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'USE_GAUSS_NEWTON_HESSIAN': True,
        'X': X,
        'Y': Y,
        'd_lat': ACTIVE_DIM,
        'dimension_of_PoI': dimension_of_PoI,
        'display_names': display_names,
        'interior_indices': interior_indices,
        'interior_mask': interior_mask,
        'n_int': n_int,
        'num_modes_available': num_modes_available,
        'num_observation': num_observation,
        'num_holdout_observation': num_holdout_observation,
        'num_truncated_series': num_truncated_series,
        'obs_col': obs_col,
        'obs_indices': obs_indices,
        'obs_indices_train': obs_indices_train,
        'obs_indices_holdout': obs_indices_holdout,
        'obs_locations': obs_locations,
        'obs_locations_train': obs_locations_train,
        'obs_locations_holdout': obs_locations_holdout,
        'obs_row': obs_row,
        'sampler_run_info': sampler_run_info,
        'sigma_prior': sigma_prior,
        'ell': ell,
    },
    extra_sections={
        'saved_results_files': {'metrics_csv': results_df_path, 'runinfo_csv': results_runinfo_df_path},
    },
)
 
# ==========================================
# 4. Problem-specific visualization (restore old layout / scaling)
# ==========================================
print('\nVisualizing Darcy field reconstructions...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1

# --- Figure 1: Log-permeability reconstruction ---
fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))

vis_anchor_key = 'MALA (prior)' if 'MALA (prior)' in mean_fields else next(iter(mean_fields.keys()))

vmin = float(np.min(true_field))
vmax = float(np.max(true_field))

if vis_anchor_key in samples and vis_anchor_key in mean_fields:
    anchor_vis_samps = get_valid_samples(samples[vis_anchor_key])[:1000]
    if anchor_vis_samps.shape[0] > 0:
        anchor_vis_fields = reconstruct_log_permeability(anchor_vis_samps[:, :ACTIVE_DIM])
        max_err = max(1e-12, float(np.abs(mean_fields[vis_anchor_key] - true_field).max()))
        max_std = max(1e-12, float(np.std(anchor_vis_fields, axis=0).max()))
    else:
        max_err = 1e-12
        max_std = 1e-12
else:
    max_err = 1e-12
    max_std = 1e-12

im0 = axes[0, 0].imshow(true_field, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
axes[0, 0].scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.7, label='Sensors')
axes[0, 0].set_title('Ground Truth\nLog-Permeability $m(x)$', fontsize=18)
axes[0, 0].axis('off')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

axes[3, 0].imshow(true_field, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
axes[3, 0].set_title('Ground Truth', fontsize=14)
axes[3, 0].axis('off')
axes[1, 0].axis('off')
axes[2, 0].axis('off')

if vis_anchor_key not in mean_fields:
    max_err = 1e-12
    max_std = 1e-12
    for label in methods_to_plot:
        mean_f = mean_fields[label]
        max_err = max(max_err, np.abs(mean_f - true_field).max())
        samps = get_valid_samples(samples[label])[:500]
        if samps.shape[0] > 0:
            fields = reconstruct_log_permeability(samps[:, :ACTIVE_DIM])
            max_std = max(max_std, np.std(fields, axis=0).max())

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_f = mean_fields[label]

    axes[0, col].imshow(mean_f, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, col].scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.5)
    axes[0, col].set_title(f"{display_names.get(label, label)}\nMean Posterior", fontsize=18)
    axes[0, col].axis('off')

    err_f = np.abs(mean_f - true_field)
    axes[1, col].imshow(err_f, cmap='inferno', origin='lower', vmin=0, vmax=max_err)
    axes[1, col].set_title(f"Error Map\n(Max: {err_f.max():.2f})", fontsize=16)
    axes[1, col].axis('off')

    samps = get_valid_samples(samples[label])[:1000]
    if samps.shape[0] > 0:
        fields = reconstruct_log_permeability(samps[:, :ACTIVE_DIM])
        std_f = np.std(fields, axis=0)
    else:
        std_f = np.zeros_like(true_field)
    axes[2, col].imshow(std_f, cmap='viridis', origin='lower', vmin=0, vmax=max_std)
    axes[2, col].set_title(f"Uncertainty\n(Max std: {std_f.max():.2f})", fontsize=16)
    axes[2, col].axis('off')

    if samps.shape[0] > 0:
        sample_field = reconstruct_log_permeability(samps[:1, :ACTIVE_DIM])[0]
        axes[3, col].imshow(sample_field, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[3, col].set_title('Posterior Sample', fontsize=14)
    else:
        axes[3, col].text(0.5, 0.5, 'No valid\nsamples', ha='center', va='center', transform=axes[3, col].transAxes)
    axes[3, col].axis('off')

plt.suptitle(f'Inverse Darcy flow (d={ACTIVE_DIM})', fontsize=22, y=1.01)
plt.tight_layout()
plt.show()

print('\nVisualizing pressure fields...')
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

true_pmin = float(np.min(true_pressure))
true_pmax = float(np.max(true_pressure))
im_true_pressure = axes2[0].imshow(true_pressure, cmap='viridis', origin='lower', vmin=true_pmin, vmax=true_pmax)
axes2[0].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.7, label='Sensors')
axes2[0].set_title('Ground Truth\nPressure $p(x)$', fontsize=14)
axes2[0].axis('off')
axes2[0].legend(fontsize=8, loc='upper right')
plt.colorbar(im_true_pressure, ax=axes2[0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_pressure = mean_pressures.get(label)
    if mean_pressure is None:
        axes2[col].axis('off')
        continue
    axes2[col].imshow(mean_pressure, cmap='viridis', origin='lower', vmin=true_pmin, vmax=true_pmax)
    axes2[col].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.5)
    axes2[col].set_title(f"{display_names.get(label, label)}\nPressure", fontsize=14)
    axes2[col].axis('off')

plt.suptitle(f'Inverse Darcy flow (d={ACTIVE_DIM}): pressure field', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

print('\nVisualizing permeability fields $k(x)=e^{m(x)}$...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

perm_vmin = float(np.min(true_perm))
perm_vmax = float(np.max(true_perm))
im_true_perm = axes3[0].imshow(true_perm, cmap='magma', origin='lower', vmin=perm_vmin, vmax=perm_vmax)
axes3[0].set_title('Ground Truth\n$k(x)=e^{m(x)}$', fontsize=14)
axes3[0].axis('off')
plt.colorbar(im_true_perm, ax=axes3[0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_perm = mean_permeabilities.get(label)
    if mean_perm is None:
        axes3[col].axis('off')
        continue
    axes3[col].imshow(mean_perm, cmap='magma', origin='lower', vmin=perm_vmin, vmax=perm_vmax)
    axes3[col].set_title(f"{display_names.get(label, label)}\n$k(x)=e^{{m(x)}}$", fontsize=14)
    axes3[col].axis('off')

plt.suptitle(f'Inverse Darcy flow (d={ACTIVE_DIM}): permeability field', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Darcy flow HLSI pipeline complete ===')
