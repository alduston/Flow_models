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
    compute_latent_metrics,
    configure_sampling,
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
# 0. KL basis generation
# ==========================================
os.makedirs('data', exist_ok=True)

N = 32
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ELL = 0.09
SIGMA_PRIOR = 1.0
Q_MAX = 100

dists = cdist(coords, coords)
C = SIGMA_PRIOR ** 2 * np.exp(-dists / ELL)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :Q_MAX] * np.sqrt(eigvals[:Q_MAX])
np.savetxt('data/PoissonCoeff_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files
# ==========================================
num_observation_generated = 100
num_observation = 40
num_truncated_series = 48
seed = 42

dimension_of_PoI = N * N
num_modes_available = Basis_Modes.shape[1]
basis_truncated = Basis_Modes[:, :num_truncated_series]

interior_mask = np.ones((N, N), dtype=bool)
interior_mask[0, :] = False
interior_mask[-1, :] = False
interior_mask[:, 0] = False
interior_mask[:, -1] = False
interior_indices = jnp.array(np.where(interior_mask.ravel())[0])

key = jax.random.PRNGKey(seed)
obs_indices_all = np.array(
    jax.random.choice(key, interior_indices, shape=(num_observation_generated,), replace=False)
)
obs_indices = obs_indices_all[:num_observation]

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices).to_csv('data/obs_locations.csv', index=False, header=False)

# ==========================================
# 2. Physics: Poisson coefficient inversion
# ==========================================
jax.config.update("jax_enable_x64", True)

Basis = jnp.array(basis_truncated, dtype=jnp.float64)
obs_locations = jnp.array(obs_indices, dtype=int)

NOISE_STD = 2.0e-5

x_1d = jnp.linspace(0.0, 1.0, N)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d, indexing='ij')
h = 1.0 / (N - 1)
Q_FLOOR = 0.15
SOURCE_FIELD = (
    0.18
    + 1.15 * jnp.exp(-(((X_grid - 0.18) ** 2) / (2.0 * 0.050 ** 2)
                       + ((Y_grid - 0.22) ** 2) / (2.0 * 0.070 ** 2)))
    + 0.95 * jnp.exp(-(((X_grid - 0.72) ** 2) / (2.0 * 0.085 ** 2)
                       + ((Y_grid - 0.32) ** 2) / (2.0 * 0.055 ** 2)))
    + 1.05 * jnp.exp(-(((X_grid - 0.56) ** 2) / (2.0 * 0.060 ** 2)
                       + ((Y_grid - 0.78) ** 2) / (2.0 * 0.065 ** 2)))
)

_int_mask = jnp.zeros((N, N), dtype=bool)
_int_mask = _int_mask.at[1:-1, 1:-1].set(True)
_int_rows, _int_cols = jnp.where(_int_mask)
n_int = _int_rows.shape[0]

_int_id = -jnp.ones((N, N), dtype=jnp.int32)
_int_id = _int_id.at[_int_rows, _int_cols].set(jnp.arange(n_int, dtype=jnp.int32))
int_flat = _int_rows * N + _int_cols


def _build_dirichlet_laplacian_matrix():
    h2 = h * h
    idx = jnp.arange(n_int)
    ir = _int_rows
    ic = _int_cols

    A = jnp.zeros((n_int, n_int), dtype=jnp.float64)
    A = A.at[idx, idx].add(4.0 / h2)

    nbr_E = _int_id[ir + 1, ic]
    nbr_W = _int_id[ir - 1, ic]
    nbr_N = _int_id[ir, ic + 1]
    nbr_S = _int_id[ir, ic - 1]

    A = A.at[idx, nbr_E].add(jnp.where(nbr_E >= 0, -1.0 / h2, 0.0))
    A = A.at[idx, nbr_W].add(jnp.where(nbr_W >= 0, -1.0 / h2, 0.0))
    A = A.at[idx, nbr_N].add(jnp.where(nbr_N >= 0, -1.0 / h2, 0.0))
    A = A.at[idx, nbr_S].add(jnp.where(nbr_S >= 0, -1.0 / h2, 0.0))

    rhs = SOURCE_FIELD[ir, ic]
    return A, rhs


LAPLACIAN_INT, RHS_INT = _build_dirichlet_laplacian_matrix()


@jax.jit
def raw_field_from_alpha(alpha):
    return jnp.reshape(Basis @ alpha, (N, N))


@jax.jit
def coefficient_from_raw(raw_field):
    return Q_FLOOR + jax.nn.softplus(raw_field)


@jax.jit
def solve_forward(alpha):
    raw_q = raw_field_from_alpha(alpha)
    q_field = coefficient_from_raw(raw_q)
    q_int = q_field[_int_rows, _int_cols]
    A = LAPLACIAN_INT + jnp.diag(q_int)
    u_int = jnp.linalg.solve(A, RHS_INT)
    u_full = jnp.zeros((N * N,), dtype=jnp.float64)
    u_full = u_full.at[int_flat].set(u_int)
    return u_full[obs_locations]


@jax.jit
def solve_full_state(alpha):
    raw_q = raw_field_from_alpha(alpha)
    q_field = coefficient_from_raw(raw_q)
    q_int = q_field[_int_rows, _int_cols]
    A = LAPLACIAN_INT + jnp.diag(q_int)
    u_int = jnp.linalg.solve(A, RHS_INT)
    u_full = jnp.zeros((N * N,), dtype=jnp.float64)
    u_full = u_full.at[int_flat].set(u_int)
    return u_full.reshape(N, N)



def make_structured_truth_coefficients(latent_dim=num_truncated_series):
    """Build a moderately structured synthetic raw coefficient field and project it into the KL basis."""
    X_np = np.array(X_grid)
    Y_np = np.array(Y_grid)

    # Dial back the old "8 blobs + 2 thin ridges + 2 high-frequency waves" setup.
    # Keep clear nontrivial structure, but make the field smoother and less crowded.
    blob_specs = [
        (0.95, 0.18, 0.20, 0.060, 0.075),
        (-0.80, 0.34, 0.72, 0.085, 0.070),
        (0.72, 0.54, 0.38, 0.070, 0.060),
        (-0.78, 0.69, 0.62, 0.075, 0.085),
        (0.62, 0.80, 0.24, 0.060, 0.055),
    ]

    raw_truth = np.zeros_like(X_np, dtype=np.float64)
    for amp, cx, cy, sx, sy in blob_specs:
        raw_truth += amp * np.exp(
            -(((X_np - cx) ** 2) / (2.0 * sx ** 2) + ((Y_np - cy) ** 2) / (2.0 * sy ** 2))
        )

    # Keep one elongated channel-like feature, but make it broader and less sharp.
    ridge_center = 0.82 - 0.60 * X_np + 0.03 * np.sin(2.0 * np.pi * X_np)
    raw_truth += -0.70 * np.exp(-((Y_np - ridge_center) ** 2) / (2.0 * 0.032 ** 2))

    # Retain multiscale texture, but reduce both frequency and amplitude.
    raw_truth += 0.16 * np.sin(4.5 * np.pi * X_np + 1.5 * np.pi * Y_np)
    raw_truth += 0.11 * np.cos(3.5 * np.pi * (X_np - 0.30 * Y_np))

    # Gentle low-frequency background variation so the field is not just isolated blobs.
    raw_truth += 0.10 * np.cos(2.0 * np.pi * Y_np)

    B = np.array(Basis)[:, :latent_dim]
    coeffs, *_ = np.linalg.lstsq(B, raw_truth.reshape(-1), rcond=None)
    return coeffs.astype(np.float64), raw_truth.astype(np.float64)


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
run_ctx = init_run_results('poisson_inversion')

# ==========================================
# 3. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

alpha_true_np, raw_truth = make_structured_truth_coefficients(ACTIVE_DIM)
true_field = np.array(coefficient_from_raw(jnp.array(raw_truth)))

# Keep the same current behavior as the uploaded monolith: use only the first
# 25 sensors from a 100-sensor random draw, and use a tiny observation noise.
y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, lik_aux = make_physics_likelihood(
    solve_forward,
    y_obs_np,
    NOISE_STD,
    use_gauss_newton_hessian=True,
    log_batch_size=50,
    grad_batch_size=25,
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
    ('Tweedie', {'init': 'tweedie', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('Blend', {'init': 'blend', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),

    ('HLSI', {'init': 'HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI', {'init': 'HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI', {'init': 'HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-WC-HLSI', {'init': 'CE-HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-PoU-HLSI', {'init': 'CE-HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    ('KAPPA00', {'init': 'HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.0}),
    ('KAPPA04', {'init': 'HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.4}),
    ('KAPPA08', {'init': 'HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.8}),
    ('KAPPA10', {'init': 'HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),
    ('KAPPA12', {'init': 'HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.2}),

    ('KAPPA00', {'init': 'HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.0}),
    ('KAPPA04', {'init': 'HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.4}),
    ('KAPPA08', {'init': 'HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.8}),
    ('KAPPA10', {'init': 'HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),
    ('KAPPA12', {'init': 'HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.2}),
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
obs_row = obs_locs_np // N
obs_col = obs_locs_np % N



def reconstruct_raw_fields(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))



def reconstruct_coefficient_fields(latents):
    raw_fields = reconstruct_raw_fields(latents)
    return Q_FLOOR + np.log1p(np.exp(raw_fields))



def latent_to_coefficient(alpha):
    return reconstruct_coefficient_fields(np.asarray(alpha)[None, :])[0]



def solve_solution_field(alpha_vec):
    return np.array(solve_full_state(jnp.array(alpha_vec)))



def sensor_vector_to_grid(values, fill_value=0.0):
    grid = np.full((N, N), fill_value, dtype=np.float64)
    flat = grid.reshape(-1)
    flat[obs_locs_np] = np.asarray(values, dtype=np.float64)
    return grid


true_solution = solve_solution_field(alpha_true_np)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_coefficient,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_clean_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

mean_solution_states = {}
norm_true_solution = np.linalg.norm(true_solution) + 1e-12

print('\n=== Poisson physical-space metrics ===')
print(f"{'Method':<24} | {'Coeff RelL2(%)':<16} | {'Pearson':<10} | {'RMSE_a':<12} | {'SolutionRel':<12} | {'SensorRel':<12}")
print('-' * 109)
for label in [lab for lab in samples.keys() if lab in mean_fields]:
    mean_latent = np.asarray(metrics[label]['mean_latent'])
    mean_solution = solve_solution_field(mean_latent)
    mean_solution_states[label] = mean_solution
    solution_rel = float(np.linalg.norm(mean_solution - true_solution) / norm_true_solution)
    metrics[label]['RMSE_solution'] = rmse_array(mean_solution, true_solution)
    metrics[label]['RelL2_solution'] = solution_rel
    coeff_rel_pct = 100.0 * float(metrics[label]['RelL2_field'])
    print(
        f"{display_names.get(label, label):<24} | {coeff_rel_pct:<16.4f} | "
        f"{metrics[label]['Pearson_field']:<10.4f} | {metrics[label]['RMSE_alpha']:<12.4e} | {solution_rel:<12.4e} | {metrics[label]['FwdRelErr']:<12.4e}"
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
    target_name='Poisson coefficient inversion',
    display_names=display_names,
    reference_name=reference_title,
)

save_reproducibility_log(
    title='Poisson coefficient inversion HLSI run reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'N_REF': N_REF,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'PLOT_NORMALIZER': PLOT_NORMALIZER,
        'HESS_MIN': HESS_MIN,
        'HESS_MAX': HESS_MAX,
        'NOISE_STD': NOISE_STD,
        'num_observation_generated': num_observation_generated,
        'num_observation': num_observation,
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
        'ELL': ELL,
        'SIGMA_PRIOR': SIGMA_PRIOR,
        'Q_FLOOR': Q_FLOOR,
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
            'num_methods_with_mean_solution_states': len(mean_solution_states),
            'num_methods_with_ess_logs': len(ess_logs),
        },
        'sensor_geometry': {
            'interior_only_observations': True,
            'generated_sensor_count': num_observation_generated,
            'effective_sensor_count': num_observation,
            'effective_sensor_indices': obs_locs_np.tolist(),
        },
        'source_field': {
            'type': 'three localized sources',
            'description': 'sum of three anisotropic Gaussians plus baseline 0.18',
        },
        'truth_field': {
            'type': 'multi-scale raw field',
            'description': '5 blobs + 1 broad diagonal channel + mild oscillatory background, then softplus-shifted to coefficient field',
        },
    },
)

# ==========================================
# 4. Problem-specific visualization
# ==========================================

def _overlay_sensors(ax):
    ax.scatter(obs_col, obs_row, c='lime', s=12, marker='.', alpha=0.7)


fig_field, axes_field = plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_coefficient_fields,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_field,
    reference_bottom_title='Ground Truth\nCoefficient $q(x)$',
    field_cmap='viridis',
    sample_cmap='viridis',
    bottom_cmap='viridis',
    overlay_reference_fn=_overlay_sensors,
    overlay_method_fn=_overlay_sensors,
    suptitle=f'Poisson coefficient inversion (d={ACTIVE_DIM}): coefficient reconstruction',
    field_name='Coefficient $q(x)$',
)
plt.show()

print('\nVisualizing solution fields...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

u_vmin = float(np.min(true_solution))
u_vmax = float(np.max(true_solution))
im_true_u = axes2[0].imshow(true_solution, cmap='RdBu_r', origin='lower', vmin=u_vmin, vmax=u_vmax)
axes2[0].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.7)
axes2[0].set_title('Ground Truth\nSolution field', fontsize=14)
axes2[0].axis('off')
plt.colorbar(im_true_u, ax=axes2[0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_u = mean_solution_states.get(label)
    if mean_u is None:
        axes2[col].axis('off')
        continue
    axes2[col].imshow(mean_u, cmap='RdBu_r', origin='lower', vmin=u_vmin, vmax=u_vmax)
    axes2[col].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.5)
    axes2[col].set_title(f"{display_names.get(label, label)}\nSolution field", fontsize=14)
    axes2[col].axis('off')

plt.suptitle(
    f'Poisson coefficient inversion (d={ACTIVE_DIM}): solution field comparison',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

print('\nVisualizing sensor residual maps...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
obs_grid = sensor_vector_to_grid(y_obs_np)
im_obs = axes3[0].imshow(obs_grid, cmap='coolwarm', origin='lower')
axes3[0].set_title('Noisy observations\nat sensors', fontsize=14)
axes3[0].axis('off')
plt.colorbar(im_obs, ax=axes3[0], fraction=0.046, pad=0.04)

residual_scale = 1e-12
residual_grids = {}
for label in methods_to_plot:
    mean_lat = np.asarray(metrics[label]['mean_latent'])
    y_pred = np.array(solve_forward(jnp.array(mean_lat)))
    resid_grid = sensor_vector_to_grid(y_pred - y_obs_np, fill_value=0.0)
    residual_grids[label] = resid_grid
    residual_scale = max(residual_scale, float(np.max(np.abs(resid_grid))))

for i, label in enumerate(methods_to_plot):
    col = i + 1
    im_res = axes3[col].imshow(
        residual_grids[label],
        cmap='RdBu_r',
        origin='lower',
        vmin=-residual_scale,
        vmax=residual_scale,
    )
    axes3[col].set_title(f"{display_names.get(label, label)}\nSensor residuals", fontsize=14)
    axes3[col].axis('off')
    plt.colorbar(im_res, ax=axes3[col], fraction=0.046, pad=0.04)

plt.suptitle(
    f'Poisson coefficient inversion (d={ACTIVE_DIM}): sensor-space residuals',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Poisson coefficient inversion HLSI pipeline complete ===')
