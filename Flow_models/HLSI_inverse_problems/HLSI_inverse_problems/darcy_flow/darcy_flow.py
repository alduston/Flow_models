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
# 0. KL basis generation
# ==========================================
os.makedirs('data', exist_ok=True)

N = 32
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ELL = 0.2
SIGMA_PRIOR = 1.0
q_max = 100

dists = cdist(coords, coords)
C = SIGMA_PRIOR ** 2 * np.exp(-dists / ELL)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/Darcy_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files
# ==========================================
num_observation = 100
num_truncated_series = 32
seed = 42

basis_truncated = Basis_Modes[:, :num_truncated_series]
dimension_of_PoI = N * N
num_modes_available = Basis_Modes.shape[1]

interior_mask = np.ones((N, N), dtype=bool)
interior_mask[0, :] = False
interior_mask[-1, :] = False
interior_mask[:, 0] = False
interior_mask[:, -1] = False
interior_indices = jnp.array(np.where(interior_mask.ravel())[0])

key = jax.random.PRNGKey(seed)
obs_indices = np.array(
    jax.random.choice(key, interior_indices, shape=(num_observation,), replace=False)
)

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices).to_csv('data/obs_locations.csv', index=False, header=False)

# ==========================================
# 2. Physics: Darcy flow
# ==========================================
jax.config.update("jax_enable_x64", True)

Basis = jnp.array(basis_truncated, dtype=jnp.float64)
obs_locations = jnp.array(obs_indices, dtype=int)

NOISE_STD = 0.005

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

    rhs = f_darcy[ir, ic]
    return A, rhs


@jax.jit
def solve_forward(alpha):
    log_k = jnp.reshape(Basis @ alpha, (N, N))
    k_field = jnp.exp(log_k)
    A, rhs = _assemble_darcy_vectorized(k_field)
    p_int = jnp.linalg.solve(A, rhs)
    p_full = jnp.zeros(N * N, dtype=jnp.float64)
    p_full = p_full.at[int_flat].set(p_int)
    return p_full[obs_locations]


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
run_ctx = init_run_results('darcy_flow_hlsi')

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
    ('MALA (prior)', {'init': 'prior', 'init_steps': 0, 'mala_steps': 200, 'mala_burnin': 50, 'mala_dt': 1e-4, 'is_reference': True}),
    ('Tweedie', {'init': 'tweedie', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('Blend', {'init': 'blend', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),
    ('HLSI', {'init': 'HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI', {'init': 'HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI', {'init': 'HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-WC-HLSI', {'init': 'CE-HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-PoU-HLSI', {'init': 'CE-HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
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

mean_pressures = {}
mean_permeabilities = {}
norm_true_pressure = np.linalg.norm(true_pressure) + 1e-12

print('\n=== Darcy physical-space metrics ===')
print(f"{'Method':<24} | {'LogPerm RelL2(%)':<18} | {'RMSE_a':<12} | {'PressureRel':<12} | {'SensorRel':<12}")
print('-' * 100)
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
        f"{metrics[label]['RMSE_alpha']:<12.4e} | {pressure_rel:<12.4e} | {metrics[label]['FwdRelErr']:<12.4e}"
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
    target_name='Darcy flow log-permeability',
    display_names=display_names,
    reference_name=reference_title,
)

save_reproducibility_log(
    title='Darcy flow HLSI run reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'N_REF': N_REF,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'PLOT_NORMALIZER': PLOT_NORMALIZER,
        'HESS_MIN': HESS_MIN,
        'HESS_MAX': HESS_MAX,
        'NOISE_STD': NOISE_STD,
        'num_observation': num_observation,
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
        'ELL': ELL,
        'SIGMA_PRIOR': SIGMA_PRIOR,
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
            'num_methods_with_mean_pressures': len(mean_pressures),
            'num_methods_with_ess_logs': len(ess_logs),
        },
    },
)

# ==========================================
# 4. Problem-specific visualization
# ==========================================

def _overlay_sensors(ax):
    ax.scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.7)


fig_field, axes_field = plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_log_permeability,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_field,
    reference_bottom_title='Ground Truth\nLog-permeability $m(x)$',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_sensors,
    overlay_method_fn=_overlay_sensors,
    suptitle=f'Inverse Darcy flow (d={ACTIVE_DIM}): log-permeability reconstruction',
    field_name='Log-permeability $m(x)$',
)
plt.show()

print('\nVisualizing pressure fields...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
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

plt.suptitle(
    f'Inverse Darcy flow (d={ACTIVE_DIM}): pressure field',
    fontsize=16,
    y=1.05,
)
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

plt.suptitle(
    f'Inverse Darcy flow (d={ACTIVE_DIM}): permeability field',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Darcy flow HLSI pipeline complete ===')
