# -*- coding: utf-8 -*-
import gc
import json
import os
import sys
import time
from collections import OrderedDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

THIS_DIR = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
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

'''
from sampling import (
    GaussianPrior,
    build_results_dataframes,
    compute_field_summary_metrics,
    compute_latent_metrics,
    configure_sampling,
    device,
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
'''

# ==========================================
# 0. KL BASIS GENERATION
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
np.savetxt('data/Marmousi_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# CONFIGURATION GENERATOR (Marmousi acoustic FWI)
# ==========================================
num_truncated_series = 64
seed = 42

# Keep the same 32x32 latent/simulation grid as the old AFWI setup so the
# forward solve remains in the same compute regime, but drive the truth from
# a cropped/downsampled Marmousi velocity model.
BACKGROUND_VELOCITY = 2.50  # km/s
VELOCITY_LOG_PERTURB_SCALE = 0.65
RAW_TANH_CLIP = 0.98

MARMOUSI_NPY_PATH = os.path.join('data', 'marmousi_vp_20m_851x176.npy')
MARMOUSI_BIN_PATH = os.path.join('data', 'marmousi_vp_20m_851x176.bin')
MARMOUSI_META_PATH = os.path.join('data', 'marmousi_vp_20m_851x176_metadata.json')

# Use a broad central crop containing the faulted/high-contrast Marmousi
# structure, then resize to the same 32x32 grid as the old AFWI example.
MARM_X_START = 180
MARM_X_END = 620
MARM_Z_START = 0
MARM_Z_END = 120
MARM_SMOOTH_PASSES = 2

N_SOURCES = 12
SOURCE_DEPTH = 0.12
RECEIVER_DEPTH = 0.09
SOURCE_WIDTH = 0.020
SOURCE_COL_INDICES = np.round(np.linspace(4, N - 5, N_SOURCES)).astype(int)
RECEIVER_COL_INDICES = np.arange(1, N - 1)
N_RECEIVERS = RECEIVER_COL_INDICES.size

N_TIME_STEPS = 240
DT = 3.5e-3
RECORD_STRIDE = 4
RICKER_FREQ = 5.0
SPONGE_WIDTH_CELLS = 5
SPONGE_MAX_DAMP = 18.0


def _smooth_box_mask(xx, yy, x_lo, x_hi, y_lo, y_hi, softness):
    sig = lambda z: 1.0 / (1.0 + np.exp(-z / softness))
    return sig(xx - x_lo) * sig(x_hi - xx) * sig(yy - y_lo) * sig(y_hi - yy)


# Unlike the old hand-built AFWI example, the Marmousi-derived model already
# supplies the desired spatial structure. Use a full-domain support mask so the
# latent-to-velocity map can represent the crop directly.
model_support_mask = np.ones_like(X, dtype=np.float64)

source_xs = x[SOURCE_COL_INDICES]
source_centers = np.column_stack([source_xs, np.full(N_SOURCES, SOURCE_DEPTH)])
receiver_xs = x[RECEIVER_COL_INDICES]
receiver_coords = np.column_stack([receiver_xs, np.full(N_RECEIVERS, RECEIVER_DEPTH)])

receiver_rows = np.full(N_RECEIVERS, np.argmin(np.abs(x - RECEIVER_DEPTH)), dtype=int)
receiver_cols = RECEIVER_COL_INDICES.astype(int)
receiver_flat_indices = receiver_rows * N + receiver_cols

source_patterns = []
for sx, sy in source_centers:
    dist2 = (X - sx) ** 2 + (Y - sy) ** 2
    src = np.exp(-0.5 * dist2 / (SOURCE_WIDTH ** 2))
    src = src / np.sqrt(np.sum(src ** 2) + 1e-12)
    source_patterns.append(src)
source_patterns = np.stack(source_patterns, axis=0)


def _ricker_wavelet(times, f0):
    t0 = 4.0 / f0
    arg = np.pi * f0 * (times - t0)
    return (1.0 - 2.0 * arg ** 2) * np.exp(-arg ** 2)


times_np = DT * np.arange(N_TIME_STEPS)
source_wavelet = _ricker_wavelet(times_np, RICKER_FREQ)
source_time_series = np.tile(source_wavelet[None, :], (N_SOURCES, 1))
record_step_indices_np = np.arange(0, N_TIME_STEPS, RECORD_STRIDE, dtype=int)
record_times_np = times_np[record_step_indices_np]
N_RECORD_STEPS = record_step_indices_np.size

ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
dist_to_edge = np.minimum.reduce([ii, jj, (N - 1) - ii, (N - 1) - jj]).astype(np.float64)
sponge_profile = np.clip((SPONGE_WIDTH_CELLS - dist_to_edge) / max(SPONGE_WIDTH_CELLS, 1), 0.0, None)
damping_field = SPONGE_MAX_DAMP * sponge_profile ** 2

num_observation = N_SOURCES * N_RECEIVERS * N_RECORD_STEPS
dimension_of_PoI = N * N

df_modes = pd.read_csv('data/Marmousi_Basis_Modes.csv', header=None)
modes_raw = df_modes.to_numpy().flatten().astype(np.float64)
num_modes_available = modes_raw.size // dimension_of_PoI
full_basis = modes_raw.reshape((dimension_of_PoI, num_modes_available))
basis_truncated = full_basis[:, :num_truncated_series]

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(receiver_flat_indices).to_csv('data/obs_locations.csv', index=False, header=False)
pd.DataFrame(source_centers, columns=['x', 'y']).to_csv('data/source_locations.csv', index=False)



def _load_marmousi_velocity_grid_kms():
    if os.path.exists(MARMOUSI_NPY_PATH):
        marm_raw = np.load(MARMOUSI_NPY_PATH).astype(np.float64)
    elif os.path.exists(MARMOUSI_BIN_PATH) and os.path.exists(MARMOUSI_META_PATH):
        with open(MARMOUSI_META_PATH, 'r') as f:
            metadata = json.load(f)
        shape = tuple(metadata['downsampled_shape'])
        marm_raw = np.fromfile(MARMOUSI_BIN_PATH, dtype=np.float32).reshape(shape).astype(np.float64)
    else:
        raise FileNotFoundError(
            'Could not find a Marmousi model file. Expected one of: '
            f'{MARMOUSI_NPY_PATH} or ({MARMOUSI_BIN_PATH} with {MARMOUSI_META_PATH}).'
        )

    # The prep script writes the array in (n_traces, n_samples). Transpose to
    # (depth, lateral) for the PDE grid used below.
    if marm_raw.shape[0] > marm_raw.shape[1]:
        marm_depth_x = marm_raw.T
    else:
        marm_depth_x = marm_raw

    return marm_depth_x / 1000.0  # m/s -> km/s


def _resize_array_bilinear(arr, out_shape):
    in_h, in_w = arr.shape
    out_h, out_w = out_shape
    if (in_h, in_w) == (out_h, out_w):
        return arr.copy()

    x_old = np.linspace(0.0, 1.0, in_w)
    x_new = np.linspace(0.0, 1.0, out_w)
    tmp = np.empty((in_h, out_w), dtype=np.float64)
    for i in range(in_h):
        tmp[i] = np.interp(x_new, x_old, arr[i])

    y_old = np.linspace(0.0, 1.0, in_h)
    y_new = np.linspace(0.0, 1.0, out_h)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for j in range(out_w):
        out[:, j] = np.interp(y_new, y_old, tmp[:, j])
    return out


def _smooth_array_local_average(arr, passes=MARM_SMOOTH_PASSES):
    out = np.asarray(arr, dtype=np.float64).copy()
    if passes <= 0:
        return out
    for _ in range(int(passes)):
        pad = np.pad(out, ((1, 1), (1, 1)), mode='edge')
        out = (
            4.0 * pad[1:-1, 1:-1]
            + pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
            + 0.5 * (pad[:-2, :-2] + pad[:-2, 2:] + pad[2:, :-2] + pad[2:, 2:])
        ) / 10.0
    return out


def _prepare_marmousi_velocity_truth():
    marm = _load_marmousi_velocity_grid_kms()
    crop = marm[MARM_Z_START:MARM_Z_END, MARM_X_START:MARM_X_END]
    if crop.size == 0:
        raise ValueError(
            'Chosen Marmousi crop is empty. Check MARM_{X,Z}_{START,END} against the model shape.'
        )
    resized = _resize_array_bilinear(crop, (N, N)).astype(np.float64)
    smoothed = _smooth_array_local_average(resized, passes=MARM_SMOOTH_PASSES)
    return smoothed


def _velocity_field_to_raw(velocity_field):
    log_ratio = np.log(np.clip(velocity_field, 1e-8, None) / BACKGROUND_VELOCITY)
    tanh_arg = np.clip(log_ratio / VELOCITY_LOG_PERTURB_SCALE, -RAW_TANH_CLIP, RAW_TANH_CLIP)
    return np.arctanh(tanh_arg).astype(np.float64)

# ==========================================
# Physics engine
# ==========================================
jax.config.update("jax_enable_x64", True)
Basis = jnp.array(basis_truncated)
model_support_mask_jax = jnp.array(model_support_mask, dtype=jnp.float64)
receiver_indices_jax = jnp.array(receiver_flat_indices, dtype=int)
source_patterns_jax = jnp.array(source_patterns, dtype=jnp.float64)
source_time_series_jax = jnp.array(source_time_series, dtype=jnp.float64)
damping_field_jax = jnp.array(damping_field, dtype=jnp.float64)
record_step_indices_jax = jnp.array(record_step_indices_np, dtype=int)

h = 1.0 / (N - 1)
ZERO_FIELD = jnp.zeros((N, N), dtype=jnp.float64)


def _flatten_measurements_by_source(gathers):
    return gathers.reshape(-1)


def _alpha_to_raw_and_velocity(alpha):
    raw_field = jnp.reshape(Basis @ alpha, (N, N))
    velocity = BACKGROUND_VELOCITY * jnp.exp(
        VELOCITY_LOG_PERTURB_SCALE * jnp.tanh(raw_field) * model_support_mask_jax
    )
    return raw_field, velocity


def _enforce_zero_boundary(u):
    u = u.at[0, :].set(0.0)
    u = u.at[-1, :].set(0.0)
    u = u.at[:, 0].set(0.0)
    u = u.at[:, -1].set(0.0)
    return u


def _laplacian_dirichlet(u):
    u_pad = jnp.pad(u, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    return (
        u_pad[2:, 1:-1] + u_pad[:-2, 1:-1] + u_pad[1:-1, 2:] + u_pad[1:-1, :-2] - 4.0 * u
    ) / (h * h)


@jax.jit
def _simulate_single_shot_gather(velocity_field, source_idx):
    c2 = velocity_field ** 2
    src_spatial = source_patterns_jax[source_idx]
    src_time = source_time_series_jax[source_idx]

    def step(carry, src_amp):
        u_prev, u_curr = carry
        lap = _laplacian_dirichlet(u_curr)
        forcing = src_amp * src_spatial
        numer = (
            2.0 * u_curr
            - (1.0 - 0.5 * DT * damping_field_jax) * u_prev
            + (DT ** 2) * (c2 * lap + forcing)
        )
        u_next = numer / (1.0 + 0.5 * DT * damping_field_jax)
        u_next = _enforce_zero_boundary(u_next)
        rec = u_next.reshape(-1)[receiver_indices_jax]
        return (u_curr, u_next), rec

    (_, _), rec_all = jax.lax.scan(step, (ZERO_FIELD, ZERO_FIELD), src_time)
    rec_sub = rec_all[record_step_indices_jax, :]
    return rec_sub.T


@jax.jit
def solve_forward(alpha):
    _, velocity_field = _alpha_to_raw_and_velocity(alpha)
    shot_ids = jnp.arange(N_SOURCES)
    gathers = jax.vmap(lambda s_idx: _simulate_single_shot_gather(velocity_field, s_idx))(shot_ids)
    return _flatten_measurements_by_source(gathers)


@jax.jit
def solve_single_pattern(alpha, pattern_idx):
    _, velocity_field = _alpha_to_raw_and_velocity(alpha)
    return _simulate_single_shot_gather(velocity_field, pattern_idx)


# ==========================================
# Shared sampling configuration
# ==========================================
ACTIVE_DIM = num_truncated_series
PLOT_NORMALIZER = 'best'
HESS_MIN = 1e-8
HESS_MAX = 1e8
GNL_PILOT_N = 512
GNL_STIFF_LAMBDA_CUT = HESS_MAX
GNL_USE_DOMINANT_PARTICLE_NEWTON = True
DEFAULT_N_GEN = 10000
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
init_run_results('marmousi_hlsi')

# ==========================================
# Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)


def _project_raw_field_to_latent(raw_field, active_dim=ACTIVE_DIM):
    B = basis_truncated[:, :active_dim]
    alpha, *_ = np.linalg.lstsq(B, raw_field.reshape(-1), rcond=None)
    return alpha.astype(np.float64)



def _make_marmousi_target_velocity():
    return _prepare_marmousi_velocity_truth()


marmousi_target_velocity_np = _make_marmousi_target_velocity()
raw_truth_np = _velocity_field_to_raw(marmousi_target_velocity_np)
alpha_true_np = _project_raw_field_to_latent(raw_truth_np)

y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
NOISE_STD = 0.05 * np.std(y_clean_np)
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
    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI_CE-HLSI', {'ref_source': 'CE-HLSI', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI_CE_HLSI_CE_HLSI', {'ref_source': 'CE-HLSI_CE_HLSI', 'init': ' CE_HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
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
model_support_mask_np = np.array(model_support_mask_jax)
receiver_row = receiver_rows
receiver_col = receiver_cols
receiver_x_positions = receiver_xs.copy()
source_rows = np.array([np.argmin(np.abs(x - xy[1])) for xy in source_centers], dtype=int)
source_cols = np.array([np.argmin(np.abs(x - xy[0])) for xy in source_centers], dtype=int)


def reconstruct_raw_field(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))



def raw_to_velocity(raw_fields):
    raw_fields = np.asarray(raw_fields)
    return BACKGROUND_VELOCITY * np.exp(
        VELOCITY_LOG_PERTURB_SCALE * np.tanh(raw_fields) * model_support_mask_np[None, :, :]
    )



def reconstruct_velocity_field(latents):
    return raw_to_velocity(reconstruct_raw_field(latents))



def latent_to_velocity(alpha):
    raw = reconstruct_raw_field(np.asarray(alpha)[None, :])[0]
    return raw_to_velocity(raw[None, :, :])[0]



def unpack_measurement_vector(y_vec):
    return np.asarray(y_vec).reshape(N_SOURCES, N_RECEIVERS, N_RECORD_STEPS)



def _laplacian_dirichlet_np(u):
    u_pad = np.pad(u, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    return (
        u_pad[2:, 1:-1] + u_pad[:-2, 1:-1] + u_pad[1:-1, 2:] + u_pad[1:-1, :-2] - 4.0 * u
    ) / (h * h)



def simulate_single_shot_numpy(alpha_latent, source_idx=0):
    raw = reconstruct_raw_field(np.asarray(alpha_latent)[None, :])[0]
    vel = raw_to_velocity(raw[None, :, :])[0]
    c2 = vel ** 2
    src_spatial = source_patterns[source_idx]
    src_time = source_time_series[source_idx]

    u_prev = np.zeros((N, N), dtype=np.float64)
    u_curr = np.zeros((N, N), dtype=np.float64)
    rec_all = np.zeros((N_TIME_STEPS, N_RECEIVERS), dtype=np.float64)

    for n in range(N_TIME_STEPS):
        lap = _laplacian_dirichlet_np(u_curr)
        forcing = src_time[n] * src_spatial
        numer = 2.0 * u_curr - (1.0 - 0.5 * DT * damping_field) * u_prev + (DT ** 2) * (c2 * lap + forcing)
        u_next = numer / (1.0 + 0.5 * DT * damping_field)
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        rec_all[n] = u_next.reshape(-1)[receiver_flat_indices]
        u_prev, u_curr = u_curr, u_next
    return rec_all[record_step_indices_np].T



def moving_average_1d(y, window=5):
    y = np.asarray(y)
    if window is None or int(window) <= 1 or y.size < 3:
        return y.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    y_pad = np.pad(y, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(y_pad, kernel, mode='valid')



def trace_style_for_label(label):
    label_l = label.lower()
    base = dict(linestyle='--', linewidth=1.55, alpha=0.92, zorder=6, marker='o', markersize=3.2, markerfacecolor='white', markeredgewidth=0.8, markevery=3)
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



def gather_to_spectral_slice(gather, freq_idx):
    spec = np.fft.rfft(gather, axis=1)
    amp = np.abs(spec[:, freq_idx])
    phase = np.unwrap(np.angle(spec[:, freq_idx]))
    return amp, phase


def wrapped_phase_residual(phi_pred, phi_true):
    return np.abs(np.angle(np.exp(1j * (phi_pred - phi_true))))


true_raw = reconstruct_raw_field(alpha_true_np)[0]
true_field = latent_to_velocity(alpha_true_np)
projection_rel_l2_truth = np.linalg.norm(true_field - marmousi_target_velocity_np) / (np.linalg.norm(marmousi_target_velocity_np) + 1e-12)
print(f"Projected Marmousi truth rel-L2 mismatch on the 32x32 latent grid: {projection_rel_l2_truth:.4%}")
true_meas = unpack_measurement_vector(y_clean_np)
obs_meas = unpack_measurement_vector(y_obs_np)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_velocity,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_obs_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

print('\n=== Marmousi acoustic FWI field/data metrics ===')
print(f"{'Method':<24} | {'RelL2_c (%)':<12} | {'Pearson':<10} | {'RMSE_a':<12} | {'FwdRel':<12}")
print('-' * 84)
norm_true = np.linalg.norm(true_field) + 1e-12
for label in mean_fields:
    data = metrics[label]
    inv_rel_l2_pct = 100.0 * data['RelL2_field']
    print(f"{display_names.get(label, label):<24} | {inv_rel_l2_pct:<12.4f} | {data['Pearson_field']:<10.4f} | {data['RMSE_alpha']:<12.4e} | {data['FwdRelErr']:<12.4e}")

plot_normalizer_key = resolve_plot_normalizer(
    PLOT_NORMALIZER,
    list(mean_fields.keys()),
    display_names=display_names,
    metrics_dict=metrics,
    fallback=reference_key,
    best_metric_keys=('RelL2_field', 'IC RelL2(%)', 'RelL2_c(%)'),
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
    target_name='Marmousi Acoustic FWI (easy)',
    display_names=display_names,
    reference_name=reference_title,
)

config_dict = {
    'seed': seed,
    'ACTIVE_DIM': ACTIVE_DIM,
    'N_REF': N_REF,
    'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
    'PLOT_NORMALIZER': PLOT_NORMALIZER,
    'HESS_MIN': HESS_MIN,
    'HESS_MAX': HESS_MAX,
    'NOISE_STD': NOISE_STD,
    'marmousi_npy_path': MARMOUSI_NPY_PATH,
    'marmousi_bin_path': MARMOUSI_BIN_PATH,
    'marmousi_meta_path': MARMOUSI_META_PATH,
    'marmousi_crop': {'x_start': MARM_X_START, 'x_end': MARM_X_END, 'z_start': MARM_Z_START, 'z_end': MARM_Z_END},
    'marmousi_smooth_passes': MARM_SMOOTH_PASSES,
    'num_observation': num_observation,
    'num_truncated_series': num_truncated_series,
    'num_modes_available': num_modes_available,
    'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
}

save_reproducibility_log(
    title='Marmousi Acoustic FWI (easy) HLSI run reproducibility log',
    config=config_dict,
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
            'projected_truth_rel_l2_mismatch': float(projection_rel_l2_truth),
        },
    },
)


def _overlay_field(ax):
    ax.scatter(receiver_col, receiver_row, c='lime', s=10, marker='s', alpha=0.8)
    ax.scatter(source_cols, source_rows, c='cyan', s=40, marker='*', alpha=0.9)
    if not np.allclose(model_support_mask_np, 1.0):
        ax.contour(model_support_mask_np, levels=[0.5], colors='white', linewidths=0.9)


fig_field, axes_field = plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_velocity_field,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=marmousi_target_velocity_np,
    reference_bottom_title='Target Marmousi crop\n(resized to 32x32)',
    field_cmap='viridis',
    sample_cmap='viridis',
    bottom_cmap='viridis',
    overlay_reference_fn=_overlay_field,
    overlay_method_fn=_overlay_field,
    suptitle=f'Marmousi acoustic FWI (d={ACTIVE_DIM}): velocity reconstruction',
    field_name='Velocity $c(x)$ [km/s]',
)
axes_field[0, 0].legend(['Receivers', 'Sources'], fontsize=8, loc='upper right')
plt.show()

# ==========================================
# Figure 2: all-source shot gathers + aggregate RMS residuals
# ==========================================
print('\nVisualizing shot gathers for all sources plus all-shot RMS summaries...')

true_meas = np.asarray(true_meas)
obs_meas = np.asarray(obs_meas)
extent = [receiver_x_positions[0], receiver_x_positions[-1], record_times_np[0], record_times_np[-1]]

# Precompute mean-latent predicted gathers once per method so the all-source
# panel and the spectral diagnostics remain consistent.
predicted_meas_by_label = OrderedDict()
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
for label in methods_to_plot:
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        predicted_meas_by_label[label] = None
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    gather_pred_all = unpack_measurement_vector(np.array(solve_forward(jnp.array(mean_lat))))
    predicted_meas_by_label[label] = gather_pred_all

# Use a single symmetric scale for all per-source signed gather panels and a
# separate symmetric scale for signed residual panels.
all_signed_gathers = [true_meas, obs_meas - true_meas]
for pred_all in predicted_meas_by_label.values():
    if pred_all is not None:
        all_signed_gathers.append(pred_all)
        all_signed_gathers.append(pred_all - true_meas)

gather_vlim = max(
    1e-12,
    float(np.percentile(
        np.abs(np.concatenate([x.ravel() for x in all_signed_gathers if x is not None])),
        99.2,
    ))
)
resid_vlim = max(
    1e-12,
    float(np.percentile(
        np.abs(np.concatenate(
            [(obs_meas - true_meas).ravel()] +
            [(pred_all - true_meas).ravel() for pred_all in predicted_meas_by_label.values() if pred_all is not None]
        )),
        99.2,
    ))
)

# Aggregate all-shot RMS summaries collapse the source dimension using RMS, so
# they are nonnegative and highlight where residual energy persists over the
# acquisition rather than allowing signed cancellations across shots.
clean_rms = np.sqrt(np.mean(true_meas ** 2, axis=0))
obs_clean_rms_resid = np.sqrt(np.mean((obs_meas - true_meas) ** 2, axis=0))
pred_rms_by_label = OrderedDict()
rms_resid_by_label = OrderedDict()
for label, pred_all in predicted_meas_by_label.items():
    if pred_all is None:
        pred_rms_by_label[label] = None
        rms_resid_by_label[label] = None
    else:
        pred_rms_by_label[label] = np.sqrt(np.mean(pred_all ** 2, axis=0))
        rms_resid_by_label[label] = np.sqrt(np.mean((pred_all - true_meas) ** 2, axis=0))

agg_gather_vmax = max(
    1e-12,
    float(np.percentile(
        np.abs(np.concatenate([clean_rms.ravel()] + [x.ravel() for x in pred_rms_by_label.values() if x is not None])),
        99.2,
    ))
)
agg_resid_vmax = max(
    1e-12,
    float(np.percentile(
        np.abs(np.concatenate([obs_clean_rms_resid.ravel()] + [x.ravel() for x in rms_resid_by_label.values() if x is not None])),
        99.2,
    ))
)

n_cols = len(methods_to_plot) + 1
n_panel_rows = 2 * (N_SOURCES + 1)
fig2, axes2 = plt.subplots(
    n_panel_rows,
    n_cols,
    figsize=(4.15 * n_cols, 2.18 * n_panel_rows),
    sharex='col',
    sharey='row',
)

panel_row_titles = []
for src_idx in range(N_SOURCES):
    panel_row_titles.extend([
        f'Source-{src_idx} shot gather',
        f'Source-{src_idx} residual',
    ])
panel_row_titles.extend([
    'All-shot RMS gather',
    'All-shot RMS residual',
])

for src_idx in range(N_SOURCES):
    row_g = 2 * src_idx
    row_r = row_g + 1
    clean_gather = true_meas[src_idx]
    obs_gather = obs_meas[src_idx]

    im_g0 = axes2[row_g, 0].imshow(
        clean_gather.T,
        cmap='seismic',
        origin='lower',
        aspect='auto',
        vmin=-gather_vlim,
        vmax=gather_vlim,
        extent=extent,
    )
    axes2[row_g, 0].set_title(f'Ground Truth\nShot gather (src {src_idx})', fontsize=12)
    axes2[row_g, 0].set_ylabel('Time', fontsize=11)
    if src_idx == N_SOURCES - 1:
        axes2[row_g, 0].set_xlabel('Receiver x', fontsize=11)
    plt.colorbar(im_g0, ax=axes2[row_g, 0], fraction=0.046, pad=0.04)

    im_r0 = axes2[row_r, 0].imshow(
        (obs_gather - clean_gather).T,
        cmap='seismic',
        origin='lower',
        aspect='auto',
        vmin=-resid_vlim,
        vmax=resid_vlim,
        extent=extent,
    )
    axes2[row_r, 0].set_title('Noisy - clean\n(data residual)', fontsize=12)
    axes2[row_r, 0].set_ylabel('Time', fontsize=11)
    if src_idx == N_SOURCES - 1:
        axes2[row_r, 0].set_xlabel('Receiver x', fontsize=11)
    plt.colorbar(im_r0, ax=axes2[row_r, 0], fraction=0.046, pad=0.04)

    for i, label in enumerate(methods_to_plot):
        col = i + 1
        pred_all = predicted_meas_by_label[label]
        if pred_all is None:
            axes2[row_g, col].axis('off')
            axes2[row_r, col].axis('off')
            continue

        gather_pred = pred_all[src_idx]
        resid = gather_pred - clean_gather

        axes2[row_g, col].imshow(
            gather_pred.T,
            cmap='seismic',
            origin='lower',
            aspect='auto',
            vmin=-gather_vlim,
            vmax=gather_vlim,
            extent=extent,
        )
        axes2[row_g, col].set_title(f"{display_names.get(label, label)}\nShot gather", fontsize=12)
        if src_idx == N_SOURCES - 1:
            axes2[row_g, col].set_xlabel('Receiver x', fontsize=11)

        axes2[row_r, col].imshow(
            resid.T,
            cmap='seismic',
            origin='lower',
            aspect='auto',
            vmin=-resid_vlim,
            vmax=resid_vlim,
            extent=extent,
        )
        axes2[row_r, col].set_title('Pred - clean residual', fontsize=12)
        if src_idx == N_SOURCES - 1:
            axes2[row_r, col].set_xlabel('Receiver x', fontsize=11)

# Aggregate all-shot RMS block at the bottom.
agg_row_g = 2 * N_SOURCES
agg_row_r = agg_row_g + 1

im_ag0 = axes2[agg_row_g, 0].imshow(
    clean_rms.T,
    cmap='magma',
    origin='lower',
    aspect='auto',
    vmin=0.0,
    vmax=agg_gather_vmax,
    extent=extent,
)
axes2[agg_row_g, 0].set_title('Ground Truth\nAll-shot RMS gather', fontsize=12)
axes2[agg_row_g, 0].set_ylabel('Time', fontsize=11)
axes2[agg_row_g, 0].set_xlabel('Receiver x', fontsize=11)
plt.colorbar(im_ag0, ax=axes2[agg_row_g, 0], fraction=0.046, pad=0.04)

im_ar0 = axes2[agg_row_r, 0].imshow(
    obs_clean_rms_resid.T,
    cmap='magma',
    origin='lower',
    aspect='auto',
    vmin=0.0,
    vmax=agg_resid_vmax,
    extent=extent,
)
axes2[agg_row_r, 0].set_title('Noisy - clean\nAll-shot RMS residual', fontsize=12)
axes2[agg_row_r, 0].set_ylabel('Time', fontsize=11)
axes2[agg_row_r, 0].set_xlabel('Receiver x', fontsize=11)
plt.colorbar(im_ar0, ax=axes2[agg_row_r, 0], fraction=0.046, pad=0.04)

for i, label in enumerate(methods_to_plot):
    col = i + 1
    pred_rms = pred_rms_by_label[label]
    resid_rms = rms_resid_by_label[label]
    if pred_rms is None:
        axes2[agg_row_g, col].axis('off')
        axes2[agg_row_r, col].axis('off')
        continue

    axes2[agg_row_g, col].imshow(
        pred_rms.T,
        cmap='magma',
        origin='lower',
        aspect='auto',
        vmin=0.0,
        vmax=agg_gather_vmax,
        extent=extent,
    )
    axes2[agg_row_g, col].set_title(f"{display_names.get(label, label)}\nAll-shot RMS gather", fontsize=12)
    axes2[agg_row_g, col].set_xlabel('Receiver x', fontsize=11)

    axes2[agg_row_r, col].imshow(
        resid_rms.T,
        cmap='magma',
        origin='lower',
        aspect='auto',
        vmin=0.0,
        vmax=agg_resid_vmax,
        extent=extent,
    )
    axes2[agg_row_r, col].set_title('Pred - clean\nAll-shot RMS residual', fontsize=12)
    axes2[agg_row_r, col].set_xlabel('Receiver x', fontsize=11)

# Put row labels on the left margin for readability in the large grid.
for r, row_name in enumerate(panel_row_titles):
    axes2[r, 0].annotate(
        row_name,
        xy=(-0.16, 0.5),
        xycoords='axes fraction',
        va='center',
        ha='right',
        rotation=90,
        fontsize=12,
    )

for ax in axes2.ravel():
    ax.set_aspect('auto')

plt.suptitle(
    f'Marmousi all-source acoustic shot gathers and residual images ({N_SOURCES} sources + all-shot RMS aggregate)',
    fontsize=16,
    y=1.002,
)
plt.tight_layout()
plt.show()

# ==========================================
# Figure 3: amplitude / phase receiver diagnostics
# ==========================================
print('\nVisualizing source-0 amplitude/phase diagnostics...')

# Reuse the precomputed all-shot gather tensors from Figure 2 and extract source 0.
clean_gather_s0 = np.asarray(true_meas[0])
obs_gather_s0 = np.asarray(obs_meas[0])

freqs = np.fft.rfftfreq(N_RECORD_STEPS, d=DT * RECORD_STRIDE)
clean_spec = np.fft.rfft(clean_gather_s0, axis=1)
mean_amp_spec = np.mean(np.abs(clean_spec), axis=0)
valid_band = (freqs >= 0.5 * RICKER_FREQ) & (freqs <= 2.5 * RICKER_FREQ)
if np.any(valid_band[1:]):
    band_idx = np.where(valid_band)[0]
    dominant_freq_idx = int(band_idx[np.argmax(mean_amp_spec[band_idx])])
else:
    dominant_freq_idx = int(np.argmax(mean_amp_spec[1:]) + 1)

dominant_freq_hz = float(freqs[dominant_freq_idx])

amp_true, phase_true = gather_to_spectral_slice(clean_gather_s0, dominant_freq_idx)
amp_obs, phase_obs = gather_to_spectral_slice(obs_gather_s0, dominant_freq_idx)

fig3, axes3 = plt.subplots(
    2,
    2,
    figsize=(30, 7.8),
    sharex='col',
    gridspec_kw={'height_ratios': [1.0, 1.0], 'wspace': 0.16, 'hspace': 0.16},
)
(ax3a, ax3b), (ax3c, ax3d) = axes3

phase_methods = methods_to_plot[:4]
model_trace_data = OrderedDict()
for label in phase_methods:
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    gather_pred = unpack_measurement_vector(np.array(solve_forward(jnp.array(mean_lat))))[0]
    amp_pred, phase_pred = gather_to_spectral_slice(gather_pred, dominant_freq_idx)
    pretty_label = display_names.get(label, label)
    model_trace_data[pretty_label] = {
        'amp': amp_pred.copy(),
        'phase': phase_pred.copy(),
        'style': trace_style_for_label(pretty_label),
    }

obs_scatter_style = dict(color='tab:red', s=11, alpha=0.42, linewidths=0.0, zorder=1)
clean_main_style = dict(color='k', linewidth=2.4, alpha=0.92, zorder=4)
resid_zero_style = dict(color='0.25', linewidth=1.0, linestyle='--', alpha=0.75, zorder=0)

ax3a.plot(receiver_x_positions, amp_true, label='Clean', **clean_main_style)
ax3b.plot(receiver_x_positions, phase_true, label='Clean', **clean_main_style)
ax3a.scatter(receiver_x_positions, amp_obs, label='Noisy obs', **obs_scatter_style)
ax3b.scatter(receiver_x_positions, phase_obs, label='Noisy obs', **obs_scatter_style)

amp_resid_max = 0.0
phase_resid_max = 0.0
hlsi_main_amp = None
hlsi_main_phase = None
for pretty_label, trace_info in model_trace_data.items():
    main_style = trace_info['style']
    resid_style = dict(main_style)
    resid_style['linewidth'] = max(1.1, 0.92 * main_style.get('linewidth', 1.4))
    resid_style['alpha'] = min(0.98, main_style.get('alpha', 0.9))
    resid_style['zorder'] = main_style.get('zorder', 6)

    amp_pred = moving_average_1d(trace_info['amp'], window=3)
    phase_pred = moving_average_1d(trace_info['phase'], window=3)
    amp_resid = np.abs(amp_pred - amp_true)
    phase_resid = wrapped_phase_residual(phase_pred, phase_true)

    ax3a.plot(receiver_x_positions, amp_pred, label=pretty_label, **main_style)
    ax3b.plot(receiver_x_positions, phase_pred, label=pretty_label, **main_style)
    ax3c.plot(receiver_x_positions, amp_resid, label=pretty_label, **resid_style)
    ax3d.plot(receiver_x_positions, phase_resid, label=pretty_label, **resid_style)

    amp_resid_max = max(amp_resid_max, float(np.max(np.abs(amp_resid))))
    phase_resid_max = max(phase_resid_max, float(np.max(np.abs(phase_resid))))

    if pretty_label.lower() == 'hlsi':
        hlsi_main_amp = amp_pred
        hlsi_main_phase = phase_pred

ax3c.axhline(0.0, **resid_zero_style)
ax3d.axhline(0.0, **resid_zero_style)

for ax in [ax3a, ax3b, ax3c, ax3d]:
    ax.grid(True, alpha=0.28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

if hlsi_main_amp is not None:
    amp_lo = min(np.min(amp_true), np.min(hlsi_main_amp))
    amp_hi = max(np.max(amp_true), np.max(hlsi_main_amp))
    amp_pad = max(1e-8, 0.12 * (amp_hi - amp_lo))
    ax3a.set_ylim(amp_lo - amp_pad, amp_hi + amp_pad)
if hlsi_main_phase is not None:
    phase_lo = min(np.min(phase_true), np.min(hlsi_main_phase))
    phase_hi = max(np.max(phase_true), np.max(hlsi_main_phase))
    phase_pad = max(1e-8, 0.12 * (phase_hi - phase_lo))
    ax3b.set_ylim(phase_lo - phase_pad, phase_hi + phase_pad)

if amp_resid_max > 0:
    ax3c.set_ylim(0.0, 1.15 * amp_resid_max)
if phase_resid_max > 0:
    ax3d.set_ylim(0.0, 1.15 * phase_resid_max)

ax3a.set_title(f'Source-0 dominant-frequency amplitude (f ≈ {dominant_freq_hz:.2f} Hz)', fontsize=15)
ax3b.set_title(f'Source-0 dominant-frequency phase (f ≈ {dominant_freq_hz:.2f} Hz)', fontsize=15)
ax3a.set_ylabel('Amplitude', fontsize=13)
ax3c.set_ylabel('Absolute residual', fontsize=13)
ax3c.set_xlabel('Receiver x', fontsize=13)
ax3d.set_xlabel('Receiver x', fontsize=13)

handles, labels = [], []
for ax in (ax3a, ax3b):
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
legend_map = OrderedDict()
for h, l in sorted(zip(handles, labels), key=lambda pair: (legend_priority(pair[1]), pair[1])):
    if l not in legend_map:
        legend_map[l] = h
fig3.legend(
    legend_map.values(),
    legend_map.keys(),
    loc='upper center',
    ncol=min(6, len(legend_map)),
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.03),
)
fig3.suptitle(
    'Marmousi source-0 frequency-slice amplitude / phase receiver traces and residuals',
    fontsize=16,
    y=1.10,
)
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
plt.show()

# ==========================================
# Figure 4: GN curvature spectrum
# ==========================================
print('\nVisualizing Marmousi Gauss-Newton curvature spectrum...')
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
ax4.set_title('Marmousi acoustic FWI Gauss-Newton curvature spectrum', fontsize=15)
ax4.grid(True, which='both', alpha=0.25)
ax4.legend(fontsize=9)
plt.tight_layout()
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f'Run-results zip: {run_results_zip_path}')
