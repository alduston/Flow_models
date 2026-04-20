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
    compute_latent_metrics,
    configure_sampling,
    get_valid_samples,
    init_run_results,
    make_physics_likelihood,
    make_posterior_score_fn,
    plot_field_reconstruction_grid,
    plot_mean_ess_logs,
    plot_pca_histograms,
    run_standard_sampler_pipeline,
    save_reproducibility_log,
    save_results_tables,
    summarize_sampler_run,
    zip_run_results_dir,
)

# ==========================================
# Config generator
# ==========================================
N = 32
num_observation = 100
num_truncated_series = 32
num_modes_available = 100
seed = 42
prior_length_scale = 0.09

os.makedirs('data', exist_ok=True)
rng = np.random.default_rng(seed)


def build_periodic_fourier_basis(N, q_max=100, length_scale=0.18):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    freq_pairs = []
    kmax = int(np.ceil(np.sqrt(q_max))) + 8
    for kx in range(0, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if kx == 0 and ky <= 0:
                continue
            freq_pairs.append((kx, ky))
    freq_pairs.sort(key=lambda kk: (kk[0] ** 2 + kk[1] ** 2, abs(kk[0]) + abs(kk[1]), kk[0], kk[1]))
    cols, meta = [], []
    for kx, ky in freq_pairs:
        phase = 2.0 * np.pi * (kx * X + ky * Y)
        k2 = float(kx * kx + ky * ky)
        amp = np.exp(-0.5 * (2.0 * np.pi * length_scale) ** 2 * k2)
        cos_mode = amp * np.cos(phase)
        cos_mode = cos_mode - cos_mode.mean()
        cos_norm = np.linalg.norm(cos_mode.ravel())
        if cos_norm > 1e-12:
            cols.append((amp * cos_mode / cos_norm).ravel())
            meta.append((kx, ky, 'cos', amp))
            if len(cols) >= q_max:
                break
        sin_mode = amp * np.sin(phase)
        sin_mode = sin_mode - sin_mode.mean()
        sin_norm = np.linalg.norm(sin_mode.ravel())
        if sin_norm > 1e-12:
            cols.append((amp * sin_mode / sin_norm).ravel())
            meta.append((kx, ky, 'sin', amp))
            if len(cols) >= q_max:
                break
    return np.column_stack(cols[:q_max]).astype(np.float64), meta[:q_max]


full_basis, basis_meta = build_periodic_fourier_basis(N=N, q_max=num_modes_available, length_scale=prior_length_scale)
basis_truncated = full_basis[:, :num_truncated_series]
basis_modes_path = 'data/NavierStokes_Basis_Modes_generated.csv'
pd.DataFrame(full_basis).to_csv(basis_modes_path, index=False, header=False)
pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
obs_indices = rng.choice(N * N, size=(num_observation,), replace=False)
pd.DataFrame(obs_indices).to_csv('data/obs_locations.csv', index=False, header=False)

# ==========================================
# Physics
# ==========================================
jax.config.update("jax_enable_x64", True)
dimension_of_PoI = N ** 2
nu = 1e-3
delta_t = 0.05
T_end = 10.0
num_time_steps = int(round(T_end / delta_t))

Basis = jnp.array(basis_truncated)
obs_locations = jnp.array(obs_indices, dtype=int)

x_1d = jnp.linspace(0.0, 1.0, N, endpoint=False)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d, indexing='ij')
freq_1d = jnp.fft.fftfreq(N, d=1.0 / N)
KX, KY = jnp.meshgrid(2.0 * jnp.pi * freq_1d, 2.0 * jnp.pi * freq_1d, indexing='ij')
K2 = KX ** 2 + KY ** 2
K2_safe = jnp.where(K2 == 0.0, 1.0, K2)
freq_abs_x = jnp.abs(jnp.fft.fftfreq(N, d=1.0 / N))
freq_abs_y = jnp.abs(jnp.fft.fftfreq(N, d=1.0 / N))
DEALIAS = ((freq_abs_x[:, None] <= (N / 3.0)) & (freq_abs_y[None, :] <= (N / 3.0))).astype(jnp.float64)
forcing_field = 0.1 * (jnp.sin(2.0 * jnp.pi * (X_grid + Y_grid)) + jnp.cos(2.0 * jnp.pi * (X_grid + Y_grid)))
forcing_hat = jnp.fft.fftn(forcing_field)
CN_NUM = 1.0 - 0.5 * delta_t * nu * K2
CN_DEN = 1.0 + 0.5 * delta_t * nu * K2


def _latent_to_initial_vorticity(alpha):
    omega0 = jnp.reshape(Basis @ alpha, (N, N))
    return omega0 - jnp.mean(omega0)



def _ns_step(_, omega_hat):
    omega_hat = omega_hat * DEALIAS
    psi_hat = -omega_hat / K2_safe
    psi_hat = jnp.where(K2 == 0.0, 0.0, psi_hat)
    vel_x = jnp.fft.ifftn(1j * KY * psi_hat).real
    vel_y = jnp.fft.ifftn(-1j * KX * psi_hat).real
    omega_x = jnp.fft.ifftn(1j * KX * omega_hat).real
    omega_y = jnp.fft.ifftn(1j * KY * omega_hat).real
    adv_hat = jnp.fft.fftn(vel_x * omega_x + vel_y * omega_y) * DEALIAS
    rhs_hat = CN_NUM * omega_hat - delta_t * adv_hat + delta_t * forcing_hat
    omega_hat_next = (rhs_hat / CN_DEN) * DEALIAS
    return jnp.where(K2 == 0.0, 0.0, omega_hat_next)


@jax.jit
def solve_forward_full(alpha):
    omega0 = _latent_to_initial_vorticity(alpha)
    omega_hat0 = jnp.fft.fftn(omega0)
    omega_hatT = jax.lax.fori_loop(0, num_time_steps, _ns_step, omega_hat0)
    return jnp.fft.ifftn(omega_hatT).real


@jax.jit
def solve_forward(alpha):
    omega_T = solve_forward_full(alpha)
    return omega_T.reshape(-1)[obs_locations]

# ==========================================
# Shared sampling config
# ==========================================
ACTIVE_DIM = num_truncated_series
NOISE_STD = 0.01
HESS_MIN = 1e-4
HESS_MAX = 1e6
GNL_PILOT_N = 1024
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
    gnl_stiff_lambda_cut=HESS_MAX,
    gnl_use_dominant_particle_newton=True,
)
init_run_results('navier_stokes_bespoke')

# ==========================================
# Execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
alpha_true_np = np.random.randn(ACTIVE_DIM) * 0.5
y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
y_obs_np = y_clean_np + np.random.normal(0, NOISE_STD, size=y_clean_np.shape)

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, _ = make_physics_likelihood(solve_forward, y_obs_np, NOISE_STD, use_gauss_newton_hessian=True)
posterior_score_fn = make_posterior_score_fn(lik_model)

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

pipeline = run_standard_sampler_pipeline(prior_model, lik_model, SAMPLER_CONFIGS, n_ref=N_REF, build_gnl_banks=BUILD_GNL_BANKS, compute_pou=True)
samples = pipeline['samples']
ess_logs = pipeline['ess_logs']
sampler_run_info = pipeline['sampler_run_info']
display_names = pipeline['display_names']
reference_key = pipeline['reference_key']
reference_title = pipeline['reference_title']

summarize_sampler_run(sampler_run_info)
plot_mean_ess_logs(ess_logs, display_names=display_names)
metrics = compute_latent_metrics(samples, reference_key, alpha_true_np, prior_model, lik_model, posterior_score_fn, display_names=display_names)
plot_pca_histograms(samples, alpha_true_np, display_names=display_names)

Basis_np = np.array(Basis)
obs_locs_np = np.array(obs_locations)
obs_row = obs_locs_np // N
obs_col = obs_locs_np % N


def reconstruct_initial_vorticity(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_loc = latents.shape[1]
    B = Basis_np[:, :d_loc]
    fields = (latents @ B.T).reshape(latents.shape[0], N, N)
    return fields - fields.mean(axis=(1, 2), keepdims=True)



def solve_final_vorticity_field(alpha_vec):
    return np.array(solve_forward_full(jnp.array(alpha_vec)))


true_field = reconstruct_initial_vorticity(alpha_true_np)[0]
true_final_field = solve_final_vorticity_field(alpha_true_np)
norm_true = np.linalg.norm(true_field) + 1e-12
norm_y_clean = np.linalg.norm(y_clean_np) + 1e-12

print('\n=== Physical Parameter Space Metrics (Initial Vorticity + Forward) ===')
print(f"{'Method':<24} | {'Inv RelL2(%)':<12} | {'RMSE_alpha':<12} | {'FwdRelErr':<12}")
print('-' * 76)
mean_fields = {}
mean_final_fields = {}
sensor_residuals = {}
for label, samps in samples.items():
    samps_clean = get_valid_samples(samps)
    if samps_clean.shape[0] < 10:
        continue
    mean_latent = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    mean_field = reconstruct_initial_vorticity(mean_latent)[0]
    mean_fields[label] = mean_field
    inv_rel_l2_pct = 100.0 * (np.linalg.norm(mean_field - true_field) / norm_true)
    y_pred = np.array(solve_forward(jnp.array(mean_latent)))
    fwd_rel = float(np.linalg.norm(y_pred - y_clean_np) / norm_y_clean)
    sensor_residuals[label] = np.abs(y_pred - y_clean_np)
    mean_final_fields[label] = solve_final_vorticity_field(mean_latent)
    metrics.setdefault(label, {})
    metrics[label].update(dict(
        mean_latent=mean_latent,
        RMSE_alpha=float(np.sqrt(np.mean((mean_latent - alpha_true_np) ** 2))),
        RMSE_field=float(np.sqrt(np.mean((mean_field - true_field) ** 2))),
        RelL2_field=float(np.linalg.norm(mean_field - true_field) / norm_true),
        FwdRelErr=fwd_rel,
    ))
    print(f"{display_names.get(label, label):<24} | {inv_rel_l2_pct:<12.4f} | {metrics[label]['RMSE_alpha']:<12.4e} | {fwd_rel:<12.4e}")

results_df, results_runinfo_df, results_df_path, results_runinfo_df_path = save_results_tables(metrics, sampler_run_info, n_ref=N_REF, target_name='Navier-Stokes inversion', display_names=display_names, reference_name=reference_title)

save_reproducibility_log(
    title='Navier-Stokes inversion HLSI run reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'N_REF': N_REF,
        'NOISE_STD': NOISE_STD,
        'HESS_MIN': HESS_MIN,
        'HESS_MAX': HESS_MAX,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'basis_modes_path': basis_modes_path,
        'prior_length_scale': prior_length_scale,
    },
    extra_sections={
        'saved_results_files': {'metrics_csv': results_df_path, 'runinfo_csv': results_runinfo_df_path},
        'summary_stats': {
            'reference_key': reference_key,
            'reference_title': reference_title,
            'num_methods_evaluated': len(results_df.columns),
            'num_methods_with_samples': len(samples),
            'num_methods_with_mean_fields': len(mean_fields),
            'num_methods_with_mean_final_fields': len(mean_final_fields),
        },
    },
)


def _overlay(ax):
    ax.scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.6)


plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_initial_vorticity,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=reference_key,
    reference_bottom_panel=true_field,
    reference_bottom_title='Ground Truth',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay,
    overlay_method_fn=_overlay,
    suptitle=f'Inverse Navier-Stokes (Latent Dim={ACTIVE_DIM}): Initial Vorticity Reconstruction',
    field_name='Initial Vorticity $\\omega_0$',
)

print('\nVisualizing final-time vorticity fields...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
axes2[0].imshow(true_final_field, cmap='RdBu_r', origin='lower')
axes2[0].scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.6, label='Sensors')
axes2[0].set_title(r'Ground Truth\nFinal Vorticity $\omega(T)$', fontsize=14)
axes2[0].axis('off')
axes2[0].legend(fontsize=8, loc='upper right')
final_vmin = float(np.min(true_final_field))
final_vmax = float(np.max(true_final_field))
for i, label in enumerate(methods_to_plot):
    col = i + 1
    field_T = mean_final_fields.get(label)
    if field_T is None:
        axes2[col].axis('off')
        continue
    axes2[col].imshow(field_T, cmap='RdBu_r', origin='lower', vmin=final_vmin, vmax=final_vmax)
    axes2[col].scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.4)
    axes2[col].set_title(f"{display_names.get(label, label)}\nFinal Vorticity", fontsize=14)
    axes2[col].axis('off')
plt.tight_layout()
plt.show()

print('\nVisualizing sensor residual maps...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
resid0 = np.zeros_like(obs_row, dtype=float)
axes3[0].scatter(obs_col, obs_row, c=resid0, cmap='inferno', s=40, vmin=0.0, vmax=1.0)
axes3[0].set_title('Ground Truth\n|Residual| = 0', fontsize=14)
axes3[0].set_xlim(-0.5, N - 0.5)
axes3[0].set_ylim(-0.5, N - 0.5)
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
    sc = axes3[col].scatter(obs_col, obs_row, c=resid, cmap='inferno', s=40, vmin=0.0, vmax=max_resid)
    axes3[col].set_title(f"{display_names.get(label, label)}\nSensor |Residual|", fontsize=14)
    axes3[col].set_xlim(-0.5, N - 0.5)
    axes3[col].set_ylim(-0.5, N - 0.5)
    axes3[col].set_aspect('equal')
    axes3[col].invert_yaxis()
    axes3[col].grid(alpha=0.15)
    plt.colorbar(sc, ax=axes3[col], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f'Run-results zip: {run_results_zip_path}')
