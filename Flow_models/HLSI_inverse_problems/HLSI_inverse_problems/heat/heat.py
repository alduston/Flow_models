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

from sampling import (
    GaussianPrior,
    compute_field_summary_metrics,
    compute_latent_metrics,
    configure_sampling,
    get_valid_samples,
    init_run_results,
    make_physics_likelihood,
    make_posterior_score_fn,
    plot_field_reconstruction_grid,
    plot_mean_ess_logs,
    plot_pca_histograms,
    resolve_plot_normalizer,
    rmse_array,
    run_standard_sampler_pipeline,
    save_reproducibility_log,
    save_results_tables,
    summarize_sampler_run,
    zip_run_results_dir,
)

# ==========================================
# 0. Self-contained basis / configuration generation
# ==========================================
os.makedirs('data', exist_ok=True)

GRID_SIZE = 16
N = 15  # legacy latent dimension symbol used in prior files / old logs
num_observation = 10
num_truncated_series = 15
num_modes_available = 15
seed = 42
prior_length_scale = 0.18
ROOT_EDGE = 'top'
SIGMA_PRIOR = 1.0

obs_indices = np.array([7, 32, 52, 110, 121, 150, 156, 177, 210, 236], dtype=int)


def build_heat_kl_basis(grid_size, q_max=15, length_scale=0.18, sigma_prior=1.0):
    """
    Build a self-contained KL-style basis on the node grid using an exponential
    covariance kernel, replacing the old external eigen/sigma files.
    """
    x = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel()])
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    C = sigma_prior ** 2 * np.exp(-dists / length_scale)
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[idx], 0.0, None)
    eigvecs = eigvecs[:, idx]
    basis = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])[None, :]
    return basis.astype(np.float64), eigvals[:q_max].astype(np.float64)



def build_free_mask(grid_size, root_edge='top'):
    free_mask = np.zeros((grid_size, grid_size), dtype=bool)
    free_mask[1:-1, 1:-1] = True
    if root_edge == 'top':
        free_mask[0, 1:-1] = True
    elif root_edge == 'bottom':
        free_mask[-1, 1:-1] = True
    elif root_edge == 'left':
        free_mask[1:-1, 0] = True
    elif root_edge == 'right':
        free_mask[1:-1, -1] = True
    else:
        raise ValueError(f"Unknown ROOT_EDGE={root_edge}")
    return free_mask


full_basis, basis_eigs = build_heat_kl_basis(
    grid_size=GRID_SIZE,
    q_max=num_modes_available,
    length_scale=prior_length_scale,
    sigma_prior=SIGMA_PRIOR,
)
basis_truncated = full_basis[:, :num_truncated_series]
basis_modes_path = 'data/Heat_Basis_Modes_generated.csv'

pd.DataFrame(full_basis).to_csv(basis_modes_path, index=False, header=False)
pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices).to_csv('data/obs_locations.csv', index=False, header=False)

free_mask_np = build_free_mask(GRID_SIZE, ROOT_EDGE)
free_index_np = np.where(free_mask_np.ravel())[0].astype(np.int64)
pd.DataFrame(free_index_np).to_csv('data/free_index.csv', index=False, header=False)

# ==========================================
# 1. Physics: bespoke mixed-BC heat equation
# ==========================================
jax.config.update("jax_enable_x64", True)

DIMENSION_OF_POI = GRID_SIZE ** 2
Basis = jnp.array(basis_truncated, dtype=jnp.float64)
obs_locations = jnp.array(obs_indices, dtype=int)
free_index = jnp.array(free_index_np, dtype=int)

G = GRID_SIZE
h = 1.0 / (G - 1)
x_1d = jnp.linspace(0.0, 1.0, G)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d, indexing='ij')
source_field = 20.0 * jnp.ones((G, G), dtype=jnp.float64)
NOISE_STD = 0.05

free_mask = jnp.array(free_mask_np)
free_rows_np, free_cols_np = np.where(free_mask_np)
free_rows = jnp.array(free_rows_np, dtype=jnp.int32)
free_cols = jnp.array(free_cols_np, dtype=jnp.int32)
n_free = free_rows.shape[0]
free_id = -jnp.ones((G, G), dtype=jnp.int32)
free_id = free_id.at[free_rows, free_cols].set(jnp.arange(n_free, dtype=jnp.int32))

obs_operator_np = np.zeros((num_observation, n_free), dtype=np.float64)
free_lookup = {int(idx): ii for ii, idx in enumerate(np.array(free_index_np))}
for row, idx in enumerate(obs_indices):
    if int(idx) in free_lookup:
        obs_operator_np[row, free_lookup[int(idx)]] = 1.0
obs_operator = jnp.array(obs_operator_np, dtype=jnp.float64)



def _assemble_heat_system(k_field):
    """
    Assemble the dense linear system for
        -div(k(x) grad w(x)) = 20,
    with homogeneous Dirichlet on the exterior boundary except for one root edge
    carrying homogeneous Neumann data.
    """
    h2 = h * h
    k_ip = 2.0 * k_field[:-1, :] * k_field[1:, :] / (k_field[:-1, :] + k_field[1:, :] + 1e-30)
    k_jp = 2.0 * k_field[:, :-1] * k_field[:, 1:] / (k_field[:, :-1] + k_field[:, 1:] + 1e-30)

    fr = free_rows
    fc = free_cols
    idx = jnp.arange(n_free)

    c_down = jnp.where(fr < G - 1, k_ip[fr, fc] / h2, 0.0)
    c_up = jnp.where(fr > 0, k_ip[fr - 1, fc] / h2, 0.0)
    c_right = jnp.where(fc < G - 1, k_jp[fr, fc] / h2, 0.0)
    c_left = jnp.where(fc > 0, k_jp[fr, fc - 1] / h2, 0.0)

    nbr_down = jnp.where(fr < G - 1, free_id[fr + 1, fc], -1)
    nbr_up = jnp.where(fr > 0, free_id[fr - 1, fc], -1)
    nbr_right = jnp.where(fc < G - 1, free_id[fr, fc + 1], -1)
    nbr_left = jnp.where(fc > 0, free_id[fr, fc - 1], -1)

    diag = c_down + c_up + c_right + c_left
    A = jnp.zeros((n_free, n_free), dtype=jnp.float64)
    A = A.at[idx, idx].add(diag)
    A = A.at[idx, nbr_down].add(jnp.where(nbr_down >= 0, -c_down, 0.0))
    A = A.at[idx, nbr_up].add(jnp.where(nbr_up >= 0, -c_up, 0.0))
    A = A.at[idx, nbr_right].add(jnp.where(nbr_right >= 0, -c_right, 0.0))
    A = A.at[idx, nbr_left].add(jnp.where(nbr_left >= 0, -c_left, 0.0))

    rhs = source_field[fr, fc]
    return A, rhs


@jax.jit
def latent_to_log_conductivity(alpha):
    return jnp.reshape(Basis @ alpha, (G, G))


@jax.jit
def solve_forward_full(alpha):
    log_k = latent_to_log_conductivity(alpha)
    k_field = jnp.exp(log_k)
    A, rhs = _assemble_heat_system(k_field)
    w_free = jnp.linalg.solve(A, rhs)
    w_full = jnp.zeros(G * G, dtype=jnp.float64)
    w_full = w_full.at[free_index].set(w_free)
    return jnp.reshape(w_full, (G, G))


@jax.jit
def solve_forward(alpha):
    w_full = solve_forward_full(alpha)
    return w_full.reshape(-1)[obs_locations]


# ==========================================
# Shared sampling configuration
# ==========================================
ACTIVE_DIM = num_truncated_series
PLOT_NORMALIZER = 'best'
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
run_ctx = init_run_results('heat_hlsi')

# ==========================================
# 2. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
alpha_true_np = np.random.randn(ACTIVE_DIM) * 0.5
y_clean_np = np.array(solve_forward(jnp.array(alpha_true_np)))
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, lik_aux = make_physics_likelihood(
    solve_forward,
    y_obs_np,
    NOISE_STD,
    use_gauss_newton_hessian=True,
    log_batch_size=100,
    grad_batch_size=100,
    hess_batch_size=25,
)
posterior_score_fn = make_posterior_score_fn(lik_model)

'''
SAMPLER_CONFIGS = OrderedDict([
    ('MALA (prior)', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'is_reference': True}),
    ('Tweedie', {'init': 'tweedie', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('Blend', {'init': 'blend', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),
    ('HLSI', {'init': 'HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI', {'init': 'HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI', {'init': 'HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-WC-HLSI', {'init': 'CE-HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-PoU-HLSI', {'init': 'CE-HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
])
'''

SAMPLER_CONFIGS = OrderedDict([
    ('MALA (prior)', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'is_reference': True}),
    #('Tweedie', {'init': 'tweedie', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('Blend', {'init': 'blend', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),

    ('HLSI', {'init': 'HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI', {'init': 'HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('PoU-HLSI', {'init': 'HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-WC-HLSI', {'init': 'CE-HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('CE-PoU-HLSI', {'init': 'CE-HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    # Principled HLSI <-> CE-HLSI interpolation, L weights
    ('Gate-25-HLSI', {
        'init': 'GATE-HLSI',
        'init_weights': 'L',
        'gate_rho': 0.25,
        'gate_beta': 1.0,
        'gate_kappa': 0.0,
        'gate_topk': 64,
        'init_steps': 200,
        'mala_steps': 0,
        'mala_burnin': 0,
        'log_mean_ess': True,
    }),
    ('Gate-50-HLSI', {
        'init': 'GATE-HLSI',
        'init_weights': 'L',
        'gate_rho': 0.50,
        'gate_beta': 1.0,
        'gate_kappa': 0.0,
        'gate_topk': 64,
        'init_steps': 200,
        'mala_steps': 0,
        'mala_burnin': 0,
        'log_mean_ess': True,
    }),
    ('Gate-75-HLSI', {
        'init': 'GATE-HLSI',
        'init_weights': 'L',
        'gate_rho': 0.75,
        'gate_beta': 1.0,
        'gate_kappa': 0.0,
        'gate_topk': 64,
        'init_steps': 200,
        'mala_steps': 0,
        'mala_burnin': 0,
        'log_mean_ess': True,
    }),

    # Same interpolation, WC weights
    ('Gate-25-WC-HLSI', {
        'init': 'GATE-HLSI',
        'init_weights': 'WC',
        'gate_rho': 0.25,
        'gate_beta': 1.0,
        'gate_kappa': 0.0,
        'gate_topk': 64,
        'init_steps': 200,
        'mala_steps': 0,
        'mala_burnin': 0,
        'log_mean_ess': True,
    }),
    ('Gate-50-WC-HLSI', {
        'init': 'GATE-HLSI',
        'init_weights': 'WC',
        'gate_rho': 0.50,
        'gate_beta': 1.0,
        'gate_kappa': 0.0,
        'gate_topk': 64,
        'init_steps': 200,
        'mala_steps': 0,
        'mala_burnin': 0,
        'log_mean_ess': True,
    }),
    ('Gate-75-WC-HLSI', {
        'init': 'GATE-HLSI',
        'init_weights': 'WC',
        'gate_rho': 0.75,
        'gate_beta': 1.0,
        'gate_kappa': 0.0,
        'gate_topk': 64,
        'init_steps': 200,
        'mala_steps': 0,
        'mala_burnin': 0,
        'log_mean_ess': True,
    }),
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
reference_key = pipeline['reference_key']
reference_title = pipeline['reference_title']

summarize_sampler_run(sampler_run_info)
plot_mean_ess_logs(ess_logs, display_names=display_names)

metrics = compute_latent_metrics(
    samples,
    reference_key,
    alpha_true_np,
    prior_model,
    lik_model,
    posterior_score_fn,
    display_names=display_names,
)

Basis_np = np.array(Basis)
obs_locs_np = np.array(obs_locations)
obs_row = obs_locs_np // G
obs_col = obs_locs_np % G



def reconstruct_log_conductivity(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, G, G))



def latent_to_log_conductivity_np(alpha):
    return reconstruct_log_conductivity(np.asarray(alpha)[None, :])[0]



def solve_temperature_field(alpha_vec):
    return np.array(solve_forward_full(jnp.array(alpha_vec)))


true_field = latent_to_log_conductivity_np(alpha_true_np)
true_temperature_field = solve_temperature_field(alpha_true_np)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_log_conductivity_np,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_clean_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

mean_temperature_fields = {}
sensor_residuals = {}
norm_true_temperature = np.linalg.norm(true_temperature_field) + 1e-12

print('\n=== Heat-equation physical-space metrics ===')
print(f"{'Method':<24} | {'LogCond RelL2(%)':<18} | {'Pearson':<10} | {'RMSE_a':<12} | {'TempRel':<12} | {'SensorRel':<12}")
print('-' * 114)
for label in [lab for lab in samples.keys() if lab in mean_fields]:
    mean_latent = np.asarray(metrics[label]['mean_latent'])
    mean_temperature = solve_temperature_field(mean_latent)
    mean_temperature_fields[label] = mean_temperature

    sensor_pred = np.asarray(solve_forward(jnp.array(mean_latent)))
    sensor_resid = np.abs(sensor_pred - y_clean_np)
    sensor_residuals[label] = sensor_resid

    temperature_rel = float(np.linalg.norm(mean_temperature - true_temperature_field) / norm_true_temperature)
    metrics[label]['RMSE_temperature'] = rmse_array(mean_temperature, true_temperature_field)
    metrics[label]['RelL2_temperature'] = temperature_rel
    logcond_rel_pct = 100.0 * float(metrics[label]['RelL2_field'])
    print(
        f"{display_names.get(label, label):<24} | {logcond_rel_pct:<18.4f} | {metrics[label].get('Pearson_field', float('nan')):<10.4f} | "
        f"{metrics[label]['RMSE_alpha']:<12.4e} | {temperature_rel:<12.4e} | {metrics[label]['FwdRelErr']:<12.4e}"
    )

plot_normalizer_key = resolve_plot_normalizer(
    PLOT_NORMALIZER,
    list(mean_fields.keys()),
    display_names=display_names,
    metrics_dict=metrics,
    fallback=reference_key,
    best_metric_keys=('RelL2_field',),
)
plot_normalizer_title = display_names.get(plot_normalizer_key, plot_normalizer_key)
plot_pca_histograms(
    samples,
    alpha_true_np,
    display_names=display_names,
    normalizer=plot_normalizer_key,
    metrics_dict=metrics,
    fallback_key=reference_key,
)

results_df, results_runinfo_df, results_df_path, results_runinfo_df_path = save_results_tables(
    metrics,
    sampler_run_info,
    n_ref=N_REF,
    target_name='Heat equation log-conductivity',
    display_names=display_names,
    reference_name=reference_title,
)

save_reproducibility_log(
    title='Heat equation inversion HLSI run reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'N_REF': N_REF,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'PLOT_NORMALIZER': PLOT_NORMALIZER,
        'HESS_MIN': HESS_MIN,
        'HESS_MAX': HESS_MAX,
        'NOISE_STD': NOISE_STD,
        'GRID_SIZE': GRID_SIZE,
        'ROOT_EDGE': ROOT_EDGE,
        'num_observation': num_observation,
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
        'prior_length_scale': prior_length_scale,
        'SIGMA_PRIOR': SIGMA_PRIOR,
        'n_free': int(n_free),
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
    },
    extra_sections={
        'saved_results_files': {'metrics_csv': results_df_path, 'runinfo_csv': results_runinfo_df_path},
        'summary_stats': {
            'reference_key': reference_key,
            'reference_title': reference_title,
            'plot_normalizer_key': plot_normalizer_key,
            'plot_normalizer_title': plot_normalizer_title,
            'num_methods_evaluated': len(results_df.columns),
            'num_methods_with_samples': len(samples),
            'num_methods_with_mean_fields': len(mean_fields),
            'num_methods_with_mean_temperature_fields': len(mean_temperature_fields),
            'num_methods_with_ess_logs': len(ess_logs),
        },
        'basis_generation': {
            'basis_source': 'internally generated exponential-covariance KL basis',
            'basis_modes_csv': basis_modes_path,
            'prior_length_scale': prior_length_scale,
            'num_modes_available': num_modes_available,
            'active_dim': ACTIVE_DIM,
            'ambient_dim': DIMENSION_OF_POI,
        },
        'boundary_conditions': {
            'equation': '-div(exp(u) grad w) = 20',
            'dirichlet_zero': 'all non-root boundary nodes',
            'root_edge': ROOT_EDGE,
            'root_bc': 'homogeneous Neumann',
            'n_free': int(n_free),
            'free_index_file': 'data/free_index.csv',
        },
        'input_file_replacement': {
            'replaced_external_files': [
                'prior_mean_n15_AC_1_1_pt5.csv',
                'prior_covariance_n15_AC_1_1_pt5.csv',
                'prior_covariance_cholesky_n15_AC_1_1_pt5.csv',
                'prestiffness_n15.npz',
                'load_vector_n15.npz',
            ],
            'replacement': 'self-contained finite-difference assembly with generated KL basis',
        },
    },
)

# ==========================================
# 3. Problem-specific visualization
# ==========================================

def _overlay_sensors(ax):
    # Match the lighter advection-diffusion overlay styling rather than the
    # older large cyan markers that dominate the 16x16 heat fields.
    ax.scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.7)


def plot_smoothed_field_reconstruction_grid(samples_dict, mean_fields, reconstruct_field_fn,
                                            display_names=None,
                                            true_field=None,
                                            plot_normalizer_key=None,
                                            reference_bottom_panel=None,
                                            reference_bottom_title='Reference',
                                            methods_to_plot=None,
                                            field_cmap='viridis',
                                            sample_cmap=None,
                                            bottom_cmap=None,
                                            overlay_reference_fn=None,
                                            overlay_method_fn=None,
                                            suptitle=None,
                                            field_name='field',
                                            n_sample_max=1000,
                                            interpolation='bicubic'):
    """Heat-specific reconstruction grid with smooth interpolation.

    The shared helper in sampling.py renders the native 16x16 arrays directly,
    which makes the heat reconstructions look blocky after the refactor. Keep
    the shared normalization / layout logic, but smooth the displayed images so
    this plot visually matches the advect_diff presentation more closely.
    """
    if len(mean_fields) == 0:
        raise ValueError('mean_fields is empty.')
    if display_names is None:
        display_names = {label: label for label in samples_dict.keys()}
    if methods_to_plot is None:
        methods_to_plot = [label for label in samples_dict.keys() if label in mean_fields]
    if sample_cmap is None:
        sample_cmap = field_cmap
    if bottom_cmap is None:
        bottom_cmap = field_cmap

    n_cols = len(methods_to_plot) + 1
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))
    has_true_field = true_field is not None
    vis_anchor_key = plot_normalizer_key if plot_normalizer_key in mean_fields else next(iter(mean_fields.keys()))
    vis_anchor_title = display_names.get(vis_anchor_key, vis_anchor_key)
    vis_reference_field = np.asarray(true_field if has_true_field else mean_fields[vis_anchor_key])
    vis_reference_bottom = np.asarray(reference_bottom_panel if reference_bottom_panel is not None else vis_reference_field)

    vmin = float(np.min(vis_reference_field))
    vmax = float(np.max(vis_reference_field))

    max_std = 1e-12
    if vis_anchor_key in samples_dict and vis_anchor_key in mean_fields:
        anchor_vis_samps = get_valid_samples(samples_dict[vis_anchor_key])[:n_sample_max]
        if anchor_vis_samps.shape[0] > 0:
            anchor_vis_fields = np.asarray(reconstruct_field_fn(anchor_vis_samps))
            max_std = max(1e-12, float(np.std(anchor_vis_fields, axis=0).max()))
    if has_true_field:
        max_err = max(1e-12, float(np.abs(mean_fields[vis_anchor_key] - vis_reference_field).max()))
    else:
        max_err = max(
            1e-12,
            max(float(np.abs(mean_fields[label] - vis_reference_field).max()) for label in methods_to_plot),
        )

    im0 = axes[0, 0].imshow(
        vis_reference_field,
        cmap=field_cmap,
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    if overlay_reference_fn is not None:
        overlay_reference_fn(axes[0, 0])
    axes[0, 0].set_title(
        f"Ground Truth\n{field_name}" if has_true_field else f"Normalizer\n{vis_anchor_title} {field_name}",
        fontsize=18,
    )
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    axes[3, 0].imshow(
        vis_reference_bottom,
        cmap=bottom_cmap,
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    if overlay_reference_fn is not None:
        overlay_reference_fn(axes[3, 0])
    axes[3, 0].set_title(reference_bottom_title, fontsize=14)
    axes[3, 0].axis('off')

    for i, label in enumerate(methods_to_plot):
        col = i + 1
        mean_f = np.asarray(mean_fields[label])
        axes[0, col].imshow(
            mean_f,
            cmap=field_cmap,
            origin='lower',
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
        if overlay_method_fn is not None:
            overlay_method_fn(axes[0, col])
        axes[0, col].set_title(f"{display_names.get(label, label)}\nMean Posterior", fontsize=18)
        axes[0, col].axis('off')

        err_f = np.abs(mean_f - vis_reference_field)
        axes[1, col].imshow(
            err_f,
            cmap='inferno',
            origin='lower',
            vmin=0,
            vmax=max_err,
            interpolation=interpolation,
        )
        if overlay_method_fn is not None:
            overlay_method_fn(axes[1, col])
        err_title = (
            f"Error Map\n(Max: {err_f.max():.2f})"
            if has_true_field else f"Deviation from {vis_anchor_title}\n(Max: {err_f.max():.2f})"
        )
        axes[1, col].set_title(err_title, fontsize=16)
        axes[1, col].axis('off')

        samps = get_valid_samples(samples_dict[label])[:n_sample_max]
        if samps.shape[0] > 0:
            fields = np.asarray(reconstruct_field_fn(samps))
            std_f = np.std(fields, axis=0)
        else:
            fields = None
            std_f = np.zeros_like(vis_reference_field)
        im_std = axes[2, col].imshow(
            std_f,
            cmap='viridis',
            origin='lower',
            vmin=0,
            vmax=max_std,
            interpolation=interpolation,
        )
        if overlay_method_fn is not None:
            overlay_method_fn(axes[2, col])
        axes[2, col].set_title('Posterior std', fontsize=16)
        axes[2, col].axis('off')
        plt.colorbar(im_std, ax=axes[2, col], fraction=0.046, pad=0.04)

        if fields is not None and samps.shape[0] > 0:
            samp_f = fields[-1]
            im_samp = axes[3, col].imshow(
                samp_f,
                cmap=sample_cmap,
                origin='lower',
                vmin=vmin,
                vmax=vmax,
                interpolation=interpolation,
            )
            if overlay_method_fn is not None:
                overlay_method_fn(axes[3, col])
            axes[3, col].set_title('Random posterior sample', fontsize=14)
            axes[3, col].axis('off')
            plt.colorbar(im_samp, ax=axes[3, col], fraction=0.046, pad=0.04)
        else:
            axes[3, col].axis('off')

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()
    return fig, axes


fig_field, axes_field = plot_smoothed_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_log_conductivity,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_field,
    reference_bottom_title='Ground Truth\nLog-conductivity $u(x)$',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_sensors,
    overlay_method_fn=_overlay_sensors,
    suptitle=f'Inverse heat equation (d={ACTIVE_DIM}): log-conductivity reconstruction',
    field_name='Log-conductivity $u(x)$',
)
plt.show()

print('\nVisualizing reconstructed temperature fields...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

sol_vmin = float(np.min(true_temperature_field))
sol_vmax = float(np.max(true_temperature_field))
im_true_temp = axes2[0].imshow(true_temperature_field, cmap='inferno', origin='lower', vmin=sol_vmin, vmax=sol_vmax)
axes2[0].scatter(obs_col, obs_row, c='cyan', s=18, marker='.', alpha=0.8, label='Sensors')
axes2[0].set_title('Ground Truth\nTemperature $w(x)$', fontsize=14)
axes2[0].axis('off')
axes2[0].legend(fontsize=8, loc='upper right')
plt.colorbar(im_true_temp, ax=axes2[0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    field_sol = mean_temperature_fields.get(label)
    if field_sol is None:
        axes2[col].axis('off')
        continue
    axes2[col].imshow(field_sol, cmap='inferno', origin='lower', vmin=sol_vmin, vmax=sol_vmax)
    axes2[col].scatter(obs_col, obs_row, c='cyan', s=18, marker='.', alpha=0.5)
    axes2[col].set_title(f"{display_names.get(label, label)}\nTemperature", fontsize=14)
    axes2[col].axis('off')

plt.suptitle(
    f'Inverse heat equation (d={ACTIVE_DIM}): temperature field',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

print('\nVisualizing sensor residual maps...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
resid0 = np.zeros_like(obs_row, dtype=float)
sc0 = axes3[0].scatter(obs_col, obs_row, c=resid0, cmap='inferno', s=60, vmin=0.0, vmax=1.0)
axes3[0].set_title('Ground Truth\n|Residual| = 0', fontsize=14)
axes3[0].set_xlim(-0.5, G - 0.5)
axes3[0].set_ylim(-0.5, G - 0.5)
axes3[0].set_aspect('equal')
axes3[0].invert_yaxis()
axes3[0].grid(alpha=0.15)

max_resid = max(1e-12, max(float(sensor_residuals[label].max()) for label in sensor_residuals))
for i, label in enumerate(methods_to_plot):
    col = i + 1
    resid = sensor_residuals.get(label)
    if resid is None:
        axes3[col].axis('off')
        continue
    sc = axes3[col].scatter(obs_col, obs_row, c=resid, cmap='inferno', s=60, vmin=0.0, vmax=max_resid)
    axes3[col].set_title(f"{display_names.get(label, label)}\nSensor |Residual|", fontsize=14)
    axes3[col].set_xlim(-0.5, G - 0.5)
    axes3[col].set_ylim(-0.5, G - 0.5)
    axes3[col].set_aspect('equal')
    axes3[col].invert_yaxis()
    axes3[col].grid(alpha=0.15)
    plt.colorbar(sc, ax=axes3[col], fraction=0.046, pad=0.04)

plt.suptitle(
    f'Inverse heat equation (d={ACTIVE_DIM}): sensor residuals',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Heat equation HLSI pipeline complete ===')
