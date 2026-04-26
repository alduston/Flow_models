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
x = np.linspace(0.0, 1.0, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing='ij')
coords = np.column_stack([X.ravel(), Y.ravel()])

ELL = 0.10
SIGMA_PRIOR = 1.0
q_max = 100

dists = cdist(coords, coords)
C = SIGMA_PRIOR ** 2 * np.exp(-dists / ELL)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/AdvecDiff_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files
# ==========================================
num_observation = 25
num_holdout_observation = 25
num_truncated_series = 32
seed = 42

dimension_of_PoI = N * N
num_modes_available = Basis_Modes.shape[1]
basis_truncated = Basis_Modes[:, :num_truncated_series]

key = jax.random.PRNGKey(seed)
obs_indices_train = np.array(
    jax.random.choice(key, jnp.arange(dimension_of_PoI), shape=(num_observation,), replace=False)
)
remaining_indices = np.setdiff1d(np.arange(dimension_of_PoI), obs_indices_train)
key_holdout = jax.random.PRNGKey(seed + 1)
obs_indices_holdout = np.array(
    jax.random.choice(key_holdout, jnp.array(remaining_indices), shape=(num_holdout_observation,), replace=False)
)
obs_indices = obs_indices_train

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices_train).to_csv('data/obs_locations.csv', index=False, header=False)

# ==========================================
# 2. Physics: periodic advection-diffusion
# ==========================================
jax.config.update("jax_enable_x64", True)

Basis = jnp.array(basis_truncated, dtype=jnp.float64)
obs_locations_train = jnp.array(obs_indices_train, dtype=int)
obs_locations_holdout = jnp.array(obs_indices_holdout, dtype=int)
obs_locations = obs_locations_train

x_1d = jnp.linspace(0.0, 1.0, N, endpoint=False)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d, indexing='ij')

OBS_TIME = 0.5
DIFFUSIVITY = 2.5e-3
ADV_VX = 0.90
ADV_VY = -0.55
NOISE_STD = 0.01

kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=1.0 / N)
ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=1.0 / N)
KX, KY = jnp.meshgrid(kx_1d, ky_1d, indexing='ij')
K2 = KX ** 2 + KY ** 2
ADV_PHASE = ADV_VX * KX + ADV_VY * KY


def _propagate_advection_diffusion(u0_field, t=OBS_TIME):
    """Exact periodic advection-diffusion propagator in Fourier space."""
    u0_hat = jnp.fft.fftn(u0_field)
    propagator = jnp.exp((-DIFFUSIVITY * K2 - 1j * ADV_PHASE) * t)
    ut_hat = u0_hat * propagator
    ut_field = jnp.real(jnp.fft.ifftn(ut_hat))
    return ut_field


@jax.jit
def solve_forward(alpha):
    u0 = jnp.reshape(Basis @ alpha, (N, N))
    uT = _propagate_advection_diffusion(u0, OBS_TIME)
    return uT.reshape(-1)[obs_locations_train]


@jax.jit
def solve_forward_holdout(alpha):
    u0 = jnp.reshape(Basis @ alpha, (N, N))
    uT = _propagate_advection_diffusion(u0, OBS_TIME)
    return uT.reshape(-1)[obs_locations_holdout]


@jax.jit
def solve_full_state(alpha, t=OBS_TIME):
    u0 = jnp.reshape(Basis @ alpha, (N, N))
    return _propagate_advection_diffusion(u0, t)


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
run_ctx = init_run_results('advect_diff_hlsi')

# ==========================================
# 3. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)


def make_structured_truth_coefficients(active_dim=ACTIVE_DIM):
    """Build a structured synthetic initial condition and project it into the KL basis."""
    X_np = np.array(X_grid)
    Y_np = np.array(Y_grid)
    truth_field = (
        1.40 * np.exp(-((X_np - 0.23) ** 2 + (Y_np - 0.31) ** 2) / (2.0 * 0.075 ** 2))
        - 1.15 * np.exp(-((X_np - 0.70) ** 2 + (Y_np - 0.66) ** 2) / (2.0 * 0.11 ** 2))
        + 0.45 * np.sin(2.0 * np.pi * (X_np + 0.35 * Y_np))
        + 0.30 * np.cos(2.0 * np.pi * (2.0 * Y_np - 0.5 * X_np))
    )
    truth_field = truth_field - truth_field.mean()
    B = basis_truncated[:, :active_dim]
    coeffs, *_ = np.linalg.lstsq(B, truth_field.reshape(-1), rcond=None)
    return coeffs.astype(np.float64), truth_field.astype(np.float64)


alpha_true_np, true_u0_target = make_structured_truth_coefficients(ACTIVE_DIM)
y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)
y_holdout_clean_np = np.array(solve_forward_holdout(jnp.array(alpha_true_np)))
y_holdout_obs_np = y_holdout_clean_np + np.random.normal(0.0, NOISE_STD, size=y_holdout_clean_np.shape)

batch_solve_forward_holdout = jax.jit(jax.vmap(solve_forward_holdout))

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
    ('MALA (prior)', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'is_reference': True}),
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


def reconstruct_initial_condition(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))



def latent_to_initial_condition(alpha):
    return reconstruct_initial_condition(np.asarray(alpha)[None, :])[0]



def solve_state_field(alpha_vec, t=OBS_TIME):
    return np.array(solve_full_state(jnp.array(alpha_vec), t=t))



def sensor_vector_to_grid(vec, fill_value=np.nan):
    grid = np.full((N, N), fill_value, dtype=np.float64)
    grid.flat[obs_locs_np] = np.asarray(vec)
    return grid


true_u0 = latent_to_initial_condition(alpha_true_np)
true_uT = solve_state_field(alpha_true_np, t=OBS_TIME)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_u0,
    field_from_latent_fn=latent_to_initial_condition,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_clean_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

metrics = compute_heldout_predictive_metrics(
    samples,
    metrics,
    heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
    batched_forward_eval_fn=lambda a_batch: np.asarray(
        batch_solve_forward_holdout(jnp.asarray(a_batch, dtype=jnp.float64))
    ),
    y_holdout_obs_np=y_holdout_obs_np,
    noise_std=NOISE_STD,
    display_names=display_names,
    min_valid=10,
)

mean_final_states = {}
norm_true_uT = np.linalg.norm(true_uT) + 1e-12

print('\n=== Advection-diffusion field/state metrics ===')
print(f"{'Method':<24} | {'IC RelL2(%)':<12} | {'Pearson':<10} | {'RMSE_a':<12} | {'FinalRel':<12} | {'SensorRel':<12} | {'HeldoutNLL':<12} | {'HeldoutZ2':<12}")
print('-' * 135)
for label in [lab for lab in samples.keys() if lab in mean_fields]:
    mean_latent = np.asarray(metrics[label]['mean_latent'])
    mean_uT = solve_state_field(mean_latent, t=OBS_TIME)
    mean_final_states[label] = mean_uT
    final_rel = float(np.linalg.norm(mean_uT - true_uT) / norm_true_uT)
    metrics[label]['RMSE_final'] = rmse_array(mean_uT, true_uT)
    metrics[label]['RelL2_final'] = final_rel
    ic_rel_l2_pct = 100.0 * float(metrics[label]['RelL2_field'])
    print(
        f"{display_names.get(label, label):<24} | {ic_rel_l2_pct:<12.4f} | "
        f"{metrics[label]['Pearson_field']:<10.4f} | {metrics[label]['RMSE_alpha']:<12.4e} | "
        f"{final_rel:<12.4e} | {metrics[label]['FwdRelErr']:<12.4e} | "
        f"{metrics[label].get('HeldoutPredNLL', np.nan):<12.4e} | {metrics[label].get('HeldoutStdResSq', np.nan):<12.4e}"
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
    target_name='Advection-diffusion initial condition',
    display_names=display_names,
    reference_name=reference_title,
)

save_reproducibility_log(
    title='Advection-diffusion IC HLSI run reproducibility log',
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
        'num_holdout_observation': num_holdout_observation,
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
        'OBS_TIME': OBS_TIME,
        'DIFFUSIVITY': DIFFUSIVITY,
        'ADV_VX': ADV_VX,
        'ADV_VY': ADV_VY,
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
            'num_methods_with_mean_final_states': len(mean_final_states),
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
    reconstruct_initial_condition,
    display_names=display_names,
    true_field=true_u0,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_u0,
    reference_bottom_title='Ground Truth\nInitial condition $u_0(x)$',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_sensors,
    overlay_method_fn=_overlay_sensors,
    suptitle=f'Advection-diffusion IC inversion (d={ACTIVE_DIM}): initial condition reconstruction',
    field_name='Initial condition $u_0(x)$',
)
plt.show()

print('\nVisualizing propagated states at observation time...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

im_true_final = axes2[0].imshow(true_uT, cmap='viridis', origin='lower')
axes2[0].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.7, label='Sensors')
axes2[0].set_title(f'Ground Truth\nState $u(x,T)$, $T={OBS_TIME:.2f}$', fontsize=14)
axes2[0].axis('off')
axes2[0].legend(fontsize=8, loc='upper right')
plt.colorbar(im_true_final, ax=axes2[0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_uT = mean_final_states.get(label)
    if mean_uT is None:
        axes2[col].axis('off')
        continue
    axes2[col].imshow(mean_uT, cmap='viridis', origin='lower', vmin=float(np.min(true_uT)), vmax=float(np.max(true_uT)))
    axes2[col].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.5)
    axes2[col].set_title(f"{display_names.get(label, label)}\nPropagated state", fontsize=14)
    axes2[col].axis('off')

plt.suptitle(
    f'Advection-diffusion IC inversion (d={ACTIVE_DIM}): state at observation time',
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
    f'Advection-diffusion IC inversion (d={ACTIVE_DIM}): sensor-space residuals',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Advection-diffusion IC HLSI pipeline complete ===')
