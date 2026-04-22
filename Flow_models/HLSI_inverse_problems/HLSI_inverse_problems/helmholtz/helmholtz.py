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
    run_standard_sampler_pipeline,
    save_reproducibility_log,
    save_results_tables,
    summarize_sampler_run,
    zip_run_results_dir,
)

# ==========================================
# KL basis generation
# ==========================================
os.makedirs('data', exist_ok=True)
N = 32
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ell = 0.10
sigma_prior = 1.0
dists = cdist(coords, coords)
C = sigma_prior ** 2 * np.exp(-dists / ell)

eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
q_max = 100
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/Helmholtz_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# Config
# ==========================================
num_truncated_series = 32
seed = 42
HELMHOLTZ_K = 24.0
HELMHOLTZ_DAMPING = 6.0
BACKGROUND_N2 = 1.0
SCATTERER_AMPLITUDE = 0.45
SCATTERER_RADIUS = 0.30
SCATTERER_SOFTNESS = 0.035
N_SOURCES = 4
N_RECEIVERS = 64
SOURCE_WIDTH = 0.055


def _ordered_boundary_indices(n):
    idx = []
    for j in range(n):
        idx.append(0 * n + j)
    for i in range(1, n):
        idx.append(i * n + (n - 1))
    for j in range(n - 2, -1, -1):
        idx.append((n - 1) * n + j)
    for i in range(n - 2, 0, -1):
        idx.append(i * n + 0)
    return np.array(idx, dtype=int)


boundary_indices_ordered = _ordered_boundary_indices(N)
n_boundary = len(boundary_indices_ordered)
receiver_spacing = n_boundary / N_RECEIVERS
receiver_boundary_pos = np.round(np.arange(N_RECEIVERS) * receiver_spacing).astype(int)
receiver_boundary_pos = np.clip(receiver_boundary_pos, 0, n_boundary - 1)
receiver_flat_indices = boundary_indices_ordered[receiver_boundary_pos]
rr = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
support_mask = 1.0 / (1.0 + np.exp((rr - SCATTERER_RADIUS) / SCATTERER_SOFTNESS))
source_centers = np.array([[0.18, 0.22], [0.82, 0.26], [0.74, 0.82], [0.24, 0.76]], dtype=np.float64)

source_terms = []
for sx, sy in source_centers:
    direction = np.array([0.5 - sx, 0.5 - sy], dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-12
    dist2 = (X - sx) ** 2 + (Y - sy) ** 2
    envelope = np.exp(-0.5 * dist2 / (SOURCE_WIDTH ** 2))
    phase = np.exp(1j * HELMHOLTZ_K * (direction[0] * (X - sx) + direction[1] * (Y - sy)))
    src = envelope * phase
    src = src / np.sqrt(np.sum(np.abs(src) ** 2) + 1e-12)
    source_terms.append(src.reshape(-1))
source_terms = np.stack(source_terms, axis=1)

num_observation = N_SOURCES * 2 * N_RECEIVERS
dimension_of_PoI = N * N

df_modes = pd.read_csv('data/Helmholtz_Basis_Modes.csv', header=None)
modes_raw = df_modes.to_numpy().flatten().astype(np.float64)
num_modes_available = modes_raw.size // dimension_of_PoI
full_basis = modes_raw.reshape((dimension_of_PoI, num_modes_available))
basis_truncated = full_basis[:, :num_truncated_series]
pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(receiver_flat_indices).to_csv('data/obs_locations.csv', index=False, header=False)

# ==========================================
# Physics
# ==========================================
jax.config.update("jax_enable_x64", True)
Basis = jnp.array(basis_truncated)
support_mask_jax = jnp.array(support_mask, dtype=jnp.float64)
receiver_indices_jax = jnp.array(receiver_flat_indices, dtype=int)
boundary_indices_jax = jnp.array(boundary_indices_ordered, dtype=int)
source_terms_jax = jnp.array(source_terms, dtype=jnp.complex128)

h = 1.0 / (N - 1)
n_total = N * N
_xface_left = (jnp.arange(N - 1)[:, None] * N + jnp.arange(N)[None, :]).ravel()
_xface_right = _xface_left + N
_yface_bot = (jnp.arange(N)[:, None] * N + jnp.arange(N - 1)[None, :]).ravel()
_yface_top = _yface_bot + 1


def _assemble_negative_laplacian():
    weight = 1.0 / (h * h)
    A = jnp.zeros((n_total, n_total), dtype=jnp.complex128)
    A = A.at[_xface_left, _xface_left].add(weight)
    A = A.at[_xface_right, _xface_right].add(weight)
    A = A.at[_xface_left, _xface_right].add(-weight)
    A = A.at[_xface_right, _xface_left].add(-weight)
    A = A.at[_yface_bot, _yface_bot].add(weight)
    A = A.at[_yface_top, _yface_top].add(weight)
    A = A.at[_yface_bot, _yface_top].add(-weight)
    A = A.at[_yface_top, _yface_bot].add(-weight)
    return A


NEG_LAPLACIAN = _assemble_negative_laplacian()
IDENTITY_WAVE = jnp.eye(n_total, dtype=jnp.complex128)


def _flatten_measurements_by_source(meas_complex):
    parts = []
    for s in range(N_SOURCES):
        parts.append(jnp.real(meas_complex[:, s]))
        parts.append(jnp.imag(meas_complex[:, s]))
    return jnp.concatenate(parts, axis=0)



def _alpha_to_raw_and_contrast(alpha):
    raw_field = jnp.reshape(Basis @ alpha, (N, N))
    contrast = SCATTERER_AMPLITUDE * jnp.tanh(raw_field) * support_mask_jax
    n2_field = BACKGROUND_N2 + contrast
    return raw_field, contrast, n2_field



def _assemble_helmholtz_operator(n2_field):
    diag = -(HELMHOLTZ_K ** 2) * n2_field.reshape(-1)
    return NEG_LAPLACIAN + jnp.diag(diag.astype(jnp.complex128)) + 1j * HELMHOLTZ_DAMPING * IDENTITY_WAVE


BACKGROUND_OPERATOR = _assemble_helmholtz_operator(jnp.ones((N, N), dtype=jnp.float64) * BACKGROUND_N2)
BACKGROUND_FIELDS = jnp.linalg.solve(BACKGROUND_OPERATOR, source_terms_jax)


@jax.jit
def solve_forward(alpha):
    _, _, n2_field = _alpha_to_raw_and_contrast(alpha)
    A = _assemble_helmholtz_operator(n2_field)
    U_total = jnp.linalg.solve(A, source_terms_jax)
    U_scat = U_total - BACKGROUND_FIELDS
    meas = U_scat[receiver_indices_jax, :]
    return _flatten_measurements_by_source(meas)


@jax.jit
def solve_single_pattern(alpha, pattern_idx):
    _, _, n2_field = _alpha_to_raw_and_contrast(alpha)
    A = _assemble_helmholtz_operator(n2_field)
    u_total = jnp.linalg.solve(A, source_terms_jax[:, pattern_idx])
    u_scat = u_total - BACKGROUND_FIELDS[:, pattern_idx]
    return u_scat.reshape(N, N)


@jax.jit
def solve_single_total_field(alpha, pattern_idx):
    _, _, n2_field = _alpha_to_raw_and_contrast(alpha)
    A = _assemble_helmholtz_operator(n2_field)
    u_total = jnp.linalg.solve(A, source_terms_jax[:, pattern_idx])
    return u_total.reshape(N, N)

# ==========================================
# Shared sampling config
# ==========================================
ACTIVE_DIM = num_truncated_series
PLOT_NORMALIZER = 'best'
HESS_MIN = 1e-6
HESS_MAX = 1e8
GNL_PILOT_N = 512
GNL_STIFF_LAMBDA_CUT = HESS_MAX
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
    gnl_use_dominant_particle_newton=True,
)
init_run_results('helmholtz_scattering_hlsi')

# ==========================================
# Execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
mode_decay = np.linspace(1.0, 0.45, ACTIVE_DIM)
alpha_true_np = 0.95 * np.random.randn(ACTIVE_DIM) * mode_decay
y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
NOISE_STD = 1e-4
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

    ('KAPPA00_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.0}),
    ('KAPPA04_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.4}),
    ('KAPPA08_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.8}),
    ('KAPPA10_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),
    ('KAPPA12_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.2}),

    ('KAPPA00_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.0}),
    ('KAPPA04_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.4}),
    ('KAPPA08_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.8}),
    ('KAPPA10_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),
    ('KAPPA12_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.2}),
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

Basis_np = np.array(Basis)
support_mask_np = np.array(support_mask_jax)
receiver_row = receiver_flat_indices // N
receiver_col = receiver_flat_indices % N


def _nearest_grid_index(xy):
    rr_loc = (X - xy[0]) ** 2 + (Y - xy[1]) ** 2
    idx_loc = int(np.argmin(rr_loc))
    return idx_loc // N, idx_loc % N


source_rowcol = [_nearest_grid_index(xy) for xy in source_centers]
source_rows = np.array([rc[0] for rc in source_rowcol])
source_cols = np.array([rc[1] for rc in source_rowcol])


def reconstruct_raw_field(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    return (latents @ B.T).reshape((-1, N, N))



def raw_to_contrast(raw_fields):
    raw_fields = np.asarray(raw_fields)
    return SCATTERER_AMPLITUDE * np.tanh(raw_fields) * support_mask_np[None, :, :]



def reconstruct_contrast_field(latents):
    return raw_to_contrast(reconstruct_raw_field(latents))



def latent_to_contrast(alpha):
    raw = reconstruct_raw_field(np.asarray(alpha)[None, :])[0]
    return SCATTERER_AMPLITUDE * np.tanh(raw) * support_mask_np



def unpack_measurement_vector(y_vec):
    y_vec = np.asarray(y_vec)
    out = np.zeros((N_SOURCES, N_RECEIVERS), dtype=np.complex128)
    idx_loc = 0
    for s in range(N_SOURCES):
        re = y_vec[idx_loc:idx_loc + N_RECEIVERS]
        idx_loc += N_RECEIVERS
        im = y_vec[idx_loc:idx_loc + N_RECEIVERS]
        idx_loc += N_RECEIVERS
        out[s] = re + 1j * im
    return out



def solve_complex_fields(alpha_latent, source_idx=0):
    alpha_jax = jnp.array(np.asarray(alpha_latent), dtype=jnp.float64)
    u_scat = np.array(solve_single_pattern(alpha_jax, source_idx))
    u_total = np.array(solve_single_total_field(alpha_jax, source_idx))
    return u_total, u_scat



def trace_style_for_label(label):
    label_l = label.lower()
    base = dict(linestyle='--', linewidth=1.55, alpha=0.92, zorder=6, marker='o', markersize=3.2, markerfacecolor='white', markeredgewidth=0.8, markevery=4)
    if 'prior' in label_l or 'mala' in label_l:
        return dict(base, color='tab:blue', linewidth=1.25, alpha=0.55, markersize=2.8, zorder=5)
    if 'tweedie' in label_l:
        return dict(base, color='tab:orange')
    if 'blend' in label_l:
        return dict(base, color='tab:green')
    if 'wc-hlsi' in label_l:
        return dict(base, color='tab:brown')
    if 'ce' in label_l and 'hlsi' in label_l:
        return dict(base, color='tab:pink')
    if 'hlsi' in label_l:
        return dict(base, color='tab:purple', linewidth=1.75, alpha=0.96, zorder=7, marker='D', markersize=3.4)
    return dict(base)



def legend_priority(label):
    label_l = label.lower()
    if label == 'Clean':
        return 0
    if label == 'Noisy obs':
        return 1
    if 'hlsi' in label_l and 'wc' not in label_l and 'ce' not in label_l:
        return 2
    if 'blend' in label_l:
        return 3
    if 'tweedie' in label_l:
        return 4
    if 'prior' in label_l or 'mala' in label_l:
        return 5
    return 6


true_raw = reconstruct_raw_field(alpha_true_np)[0]
true_field = latent_to_contrast(alpha_true_np)
true_meas = unpack_measurement_vector(y_clean_np)
obs_meas = unpack_measurement_vector(y_obs_np)
theta_receivers = 2.0 * np.pi * receiver_boundary_pos / n_boundary

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_contrast,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_obs_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

print('\n=== Helmholtz field/data metrics ===')
print(f"{'Method':<24} | {'RelL2_q (%)':<12} | {'Pearson':<10} | {'RMSE_a':<12} | {'FwdRel':<12}")
print('-' * 84)
for label in mean_fields:
    data = metrics[label]
    print(f"{display_names.get(label, label):<24} | {100.0 * data['RelL2_field']:<12.4f} | {data.get('Pearson_field', float('nan')):<10.4f} | {data['RMSE_alpha']:<12.4e} | {data['FwdRelErr']:<12.4e}")

plot_normalizer_key = resolve_plot_normalizer(PLOT_NORMALIZER, list(mean_fields.keys()), display_names=display_names, metrics_dict=metrics, fallback=reference_key, best_metric_keys=('RelL2_field', 'IC RelL2(%)', 'RelL2_q(%)'))
plot_normalizer_title = display_names.get(plot_normalizer_key, plot_normalizer_key)
plot_pca_histograms(samples, alpha_true_np, display_names=display_names, normalizer=plot_normalizer_key, metrics_dict=metrics, fallback_key=reference_key)

results_df, results_runinfo_df, results_df_path, results_runinfo_df_path = save_results_tables(metrics, sampler_run_info, n_ref=N_REF, target_name='Helmholtz scattering', display_names=display_names, reference_name=reference_title)

save_reproducibility_log(
    title='Helmholtz scattering HLSI run reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'N_REF': N_REF,
        'PLOT_NORMALIZER': PLOT_NORMALIZER,
        'HESS_MIN': HESS_MIN,
        'HESS_MAX': HESS_MAX,
        'NOISE_STD': NOISE_STD,
        'num_observation': num_observation,
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
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
            'num_methods_with_ess_logs': len(ess_logs),
        },
    },
)


def _overlay_field(ax):
    ax.scatter(receiver_col, receiver_row, c='lime', s=12, marker='s', alpha=0.8)
    ax.scatter(source_cols, source_rows, c='cyan', s=50, marker='*', alpha=0.9)
    ax.contour(support_mask_np, levels=[0.5], colors='white', linewidths=1.0)


plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_contrast_field,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_raw,
    reference_bottom_title='Ground Truth\nRaw latent field $m(x)$',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_field,
    overlay_method_fn=_overlay_field,
    suptitle=f'Nonlinear Helmholtz inverse scattering (d={ACTIVE_DIM}, k={HELMHOLTZ_K:g})',
    field_name='Contrast $q(x)$',
)

print('\nVisualizing complex wavefields for source 0...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
wave_reference_latent = alpha_true_np
wave_reference_total_s0, wave_reference_scat_s0 = solve_complex_fields(wave_reference_latent, source_idx=0)
amp_reference = np.log10(np.abs(wave_reference_scat_s0) + 1e-6)
phase_reference = np.angle(wave_reference_total_s0)
fig2, axes2 = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

im_amp0 = axes2[0, 0].imshow(amp_reference, cmap='magma', origin='lower')
axes2[0, 0].scatter(source_cols[0], source_rows[0], c='cyan', s=60, marker='*')
axes2[0, 0].set_title('Ground Truth\nlog10 |u_scat| (src 0)', fontsize=14)
axes2[0, 0].axis('off')
plt.colorbar(im_amp0, ax=axes2[0, 0], fraction=0.046, pad=0.04)

im_phase0 = axes2[1, 0].imshow(phase_reference, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
axes2[1, 0].scatter(source_cols[0], source_rows[0], c='cyan', s=60, marker='*')
axes2[1, 0].set_title('Ground Truth\narg(u_total) (src 0)', fontsize=14)
axes2[1, 0].axis('off')
plt.colorbar(im_phase0, ax=axes2[1, 0], fraction=0.046, pad=0.04)

amp_vmin = float(np.min(amp_reference))
amp_vmax = float(np.max(amp_reference))
for i, label in enumerate(methods_to_plot):
    col = i + 1
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        axes2[0, col].axis('off')
        axes2[1, col].axis('off')
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    u_total, u_scat = solve_complex_fields(mean_lat, source_idx=0)
    axes2[0, col].imshow(np.log10(np.abs(u_scat) + 1e-6), cmap='magma', origin='lower', vmin=amp_vmin, vmax=amp_vmax)
    axes2[0, col].scatter(source_cols[0], source_rows[0], c='cyan', s=30, marker='*')
    axes2[0, col].set_title(f"{display_names.get(label, label)}\nlog10 |u_scat|", fontsize=14)
    axes2[0, col].axis('off')
    axes2[1, col].imshow(np.angle(u_total), cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
    axes2[1, col].scatter(source_cols[0], source_rows[0], c='cyan', s=30, marker='*')
    axes2[1, col].set_title(f"{display_names.get(label, label)}\narg(u_total)", fontsize=14)
    axes2[1, col].axis('off')
plt.tight_layout()
plt.show()

print('\nVisualizing boundary receiver traces for source 0...')
fig3, axes3 = plt.subplots(2, 2, figsize=(32, 7.8), sharex='col', gridspec_kw={'height_ratios': [1.0, 1.0], 'wspace': 0.14, 'hspace': 0.16})
(ax3a, ax3b), (ax3c, ax3d) = axes3
y_true_s0 = true_meas[0]
y_obs_s0 = obs_meas[0]
real_true = np.real(y_true_s0)
imag_true = np.imag(y_true_s0)
real_obs = np.real(y_obs_s0)
imag_obs = np.imag(y_obs_s0)
model_trace_data = OrderedDict()
for label in methods_to_plot[:4]:
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    y_pred = unpack_measurement_vector(np.array(solve_forward(jnp.array(mean_lat))))[0]
    pretty_label = display_names.get(label, label)
    model_trace_data[pretty_label] = {'real': np.real(y_pred).copy(), 'imag': np.imag(y_pred).copy(), 'style': trace_style_for_label(pretty_label)}
obs_scatter_style = dict(color='tab:red', s=10, alpha=0.42, linewidths=0.0, zorder=1)
clean_main_style = dict(color='k', linewidth=2.4, alpha=0.92, zorder=4)
resid_zero_style = dict(color='0.25', linewidth=1.0, linestyle='--', alpha=0.75, zorder=0)
ax3a.plot(theta_receivers, real_true, label='Clean', **clean_main_style)
ax3b.plot(theta_receivers, imag_true, label='Clean', **clean_main_style)
ax3a.scatter(theta_receivers, real_obs, label='Noisy obs', **obs_scatter_style)
ax3b.scatter(theta_receivers, imag_obs, label='Noisy obs', **obs_scatter_style)
real_resid_max = 0.0
imag_resid_max = 0.0
hlsi_main_real = None
hlsi_main_imag = None
for pretty_label, trace_info in model_trace_data.items():
    main_style = trace_info['style']
    resid_style = dict(main_style)
    resid_style['linewidth'] = max(1.1, 0.92 * main_style.get('linewidth', 1.4))
    resid_style['alpha'] = min(0.98, main_style.get('alpha', 0.9))
    resid_style['zorder'] = main_style.get('zorder', 6)
    real_pred = trace_info['real']
    imag_pred = trace_info['imag']
    real_resid = np.abs(real_pred - real_true)
    imag_resid = np.abs(imag_pred - imag_true)
    ax3a.plot(theta_receivers, real_pred, label=pretty_label, **main_style)
    ax3b.plot(theta_receivers, imag_pred, label=pretty_label, **main_style)
    ax3c.plot(theta_receivers, real_resid, label=pretty_label, **resid_style)
    ax3d.plot(theta_receivers, imag_resid, label=pretty_label, **resid_style)
    real_resid_max = max(real_resid_max, float(np.max(np.abs(real_resid))))
    imag_resid_max = max(imag_resid_max, float(np.max(np.abs(imag_resid))))
    if pretty_label.lower() == 'hlsi':
        hlsi_main_real = real_pred
        hlsi_main_imag = imag_pred
ax3c.axhline(0.0, **resid_zero_style)
ax3d.axhline(0.0, **resid_zero_style)
for ax in [ax3a, ax3b, ax3c, ax3d]:
    ax.grid(True, alpha=0.28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
if hlsi_main_real is not None:
    real_lo = min(np.min(real_true), np.min(hlsi_main_real))
    real_hi = max(np.max(real_true), np.max(hlsi_main_real))
    real_pad = max(1e-8, 0.12 * (real_hi - real_lo))
    ax3a.set_ylim(real_lo - real_pad, real_hi + real_pad)
if hlsi_main_imag is not None:
    imag_lo = min(np.min(imag_true), np.min(hlsi_main_imag))
    imag_hi = max(np.max(imag_true), np.max(hlsi_main_imag))
    imag_pad = max(1e-8, 0.12 * (imag_hi - imag_lo))
    ax3b.set_ylim(imag_lo - imag_pad, imag_hi + imag_pad)
if real_resid_max > 0:
    ax3c.set_ylim(0.0, 1.15 * real_resid_max)
if imag_resid_max > 0:
    ax3d.set_ylim(0.0, 1.15 * imag_resid_max)
ax3a.set_title('Source-0 boundary traces: real part', fontsize=15)
ax3b.set_title('Source-0 boundary traces: imaginary part', fontsize=15)
ax3a.set_ylabel('Signal', fontsize=13)
ax3c.set_ylabel('|Residual|', fontsize=13)
ax3c.set_xlabel('Boundary angle (rad)', fontsize=13)
ax3d.set_xlabel('Boundary angle (rad)', fontsize=13)
handles, labels = [], []
for ax in (ax3a, ax3b):
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
legend_map = OrderedDict()
for h, l in sorted(zip(handles, labels), key=lambda pair: (legend_priority(pair[1]), pair[1])):
    if l not in legend_map:
        legend_map[l] = h
fig3.legend(legend_map.values(), legend_map.keys(), loc='upper center', ncol=min(6, len(legend_map)), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
fig3.suptitle('Source-0 scattered-field boundary traces', fontsize=16, y=1.08)
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
plt.show()

print('\nVisualizing Gauss-Newton curvature spectrum...')
fig4, ax4 = plt.subplots(1, 1, figsize=(9, 5))
true_curv = np.linalg.eigvalsh(-np.array(lik_aux['hess_lik_gn_jax'](jnp.array(alpha_true_np), jnp.array(y_obs_np), NOISE_STD)))
true_curv = np.clip(np.sort(true_curv)[::-1], 1e-16, None)
ax4.semilogy(np.arange(1, true_curv.size + 1), true_curv, marker='o', linewidth=2, label='Truth GN spectrum')
if reference_key in mean_fields:
    ref_alpha = np.asarray(metrics[reference_key]['mean_latent'])
    ref_curv = np.linalg.eigvalsh(-np.array(lik_aux['hess_lik_gn_jax'](jnp.array(ref_alpha), jnp.array(y_obs_np), NOISE_STD)))
    ref_curv = np.clip(np.sort(ref_curv)[::-1], 1e-16, None)
    ax4.semilogy(np.arange(1, ref_curv.size + 1), ref_curv, marker='s', linewidth=2, label=f'{display_names.get(reference_key, reference_key)} GN spectrum')
ax4.axhline(HESS_MIN, linestyle='--', linewidth=1.5, label='HESS_MIN')
ax4.axhline(HESS_MAX, linestyle='--', linewidth=1.5, label='HESS_MAX')
ax4.set_xlabel('Eigenvalue rank', fontsize=13)
ax4.set_ylabel('Curvature magnitude', fontsize=13)
ax4.set_title('Gauss-Newton curvature spectrum', fontsize=15)
ax4.grid(True, which='both', alpha=0.25)
ax4.legend(fontsize=9)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f'Run-results zip: {run_results_zip_path}')
