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
    compute_heldout_predictive_metrics,
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
# 0. KL basis generation
# ==========================================
os.makedirs('data', exist_ok=True)

N = 32
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ELL = 0.06
SIGMA_PRIOR = 1.0
q_max = 100

dists = cdist(coords, coords)
C = SIGMA_PRIOR ** 2 * np.exp(-dists / ELL)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/EIT_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files
# ==========================================
num_truncated_series = 32
seed = 42
N_CURRENT_PATTERNS = 24

# Clustered sensor layout: dense coverage on a few boundary arcs and sparse/no
# coverage elsewhere. This is intended to create locally stiff likelihoods near
# instrumented regions while preserving genuine posterior ambiguity away from
# the sensor clusters.
SENSOR_LAYOUT_NAME = 'clustered_3arc_backbone'
SENSOR_CLUSTER_SPECS = (
    {'center_frac': 0.10, 'half_width_frac': 0.08, 'count': 14},
    {'center_frac': 0.40, 'half_width_frac': 0.08, 'count': 14},
    {'center_frac': 0.72, 'half_width_frac': 0.08, 'count': 14},
)
SENSOR_BACKBONE_COUNT = 6
N_ELECTRODES = int(sum(spec['count'] for spec in SENSOR_CLUSTER_SPECS) + SENSOR_BACKBONE_COUNT)
dimension_of_PoI = N * N
num_modes_available = Basis_Modes.shape[1]


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


def _select_evenly_spaced_positions(positions, count):
    positions = np.array(sorted(set(int(p) for p in np.asarray(positions).tolist())), dtype=int)
    if count <= 0 or positions.size == 0:
        return np.array([], dtype=int)
    if count >= positions.size:
        return positions.copy()

    raw_idx = np.linspace(0, positions.size - 1, count)
    selected = []
    used = set()
    for idx in np.round(raw_idx).astype(int):
        idx = int(np.clip(idx, 0, positions.size - 1))
        pos = int(positions[idx])
        if pos not in used:
            selected.append(pos)
            used.add(pos)

    if len(selected) < count:
        remaining = np.array([p for p in positions.tolist() if int(p) not in used], dtype=int)
        if remaining.size > 0:
            extra = _select_evenly_spaced_positions(remaining, count - len(selected))
            for pos in extra.tolist():
                pos = int(pos)
                if pos not in used:
                    selected.append(pos)
                    used.add(pos)

    return np.array(sorted(selected), dtype=int)


def _circular_arc_positions(n_boundary, center_frac, half_width_frac):
    center = float(center_frac % 1.0) * n_boundary
    positions = np.arange(n_boundary, dtype=float)
    clockwise = np.mod(positions - center, n_boundary)
    counterclockwise = np.mod(center - positions, n_boundary)
    circular_distance = np.minimum(clockwise, counterclockwise)
    mask = circular_distance <= (float(half_width_frac) * n_boundary)
    return np.where(mask)[0].astype(int)


def _build_clustered_electrode_positions(n_boundary, cluster_specs, backbone_count=0):
    selected = []
    used = set()
    target_count = int(sum(int(spec['count']) for spec in cluster_specs) + int(backbone_count))

    for spec in cluster_specs:
        arc_positions = _circular_arc_positions(
            n_boundary,
            center_frac=spec['center_frac'],
            half_width_frac=spec['half_width_frac'],
        )
        arc_positions = np.array([p for p in arc_positions.tolist() if int(p) not in used], dtype=int)
        chosen = _select_evenly_spaced_positions(arc_positions, int(spec['count']))
        for pos in chosen.tolist():
            pos = int(pos)
            if pos not in used:
                selected.append(pos)
                used.add(pos)

    if backbone_count > 0:
        remaining = np.array([p for p in range(n_boundary) if p not in used], dtype=int)
        backbone = _select_evenly_spaced_positions(remaining, int(backbone_count))
        for pos in backbone.tolist():
            pos = int(pos)
            if pos not in used:
                selected.append(pos)
                used.add(pos)

    if len(selected) < target_count:
        remaining = np.array([p for p in range(n_boundary) if p not in used], dtype=int)
        filler = _select_evenly_spaced_positions(remaining, target_count - len(selected))
        for pos in filler.tolist():
            pos = int(pos)
            if pos not in used:
                selected.append(pos)
                used.add(pos)

    return np.array(sorted(selected), dtype=int)


boundary_indices_ordered = _ordered_boundary_indices(N)
n_boundary = len(boundary_indices_ordered)
electrode_boundary_pos = _build_clustered_electrode_positions(
    n_boundary,
    cluster_specs=SENSOR_CLUSTER_SPECS,
    backbone_count=SENSOR_BACKBONE_COUNT,
)
if electrode_boundary_pos.size != N_ELECTRODES:
    raise ValueError(f'Expected {N_ELECTRODES} electrodes, got {electrode_boundary_pos.size}')

electrode_flat_indices = boundary_indices_ordered[electrode_boundary_pos]
_train_electrode_set = set(int(i) for i in electrode_flat_indices.tolist())
_holdout_electrode_candidates = np.array(
    [int(i) for i in boundary_indices_ordered.tolist() if int(i) not in _train_electrode_set],
    dtype=int,
)
heldout_electrode_flat_indices = np.array(sorted(_holdout_electrode_candidates.tolist()), dtype=int)
N_HOLDOUT_ELECTRODES = int(heldout_electrode_flat_indices.size)

print(
    f"EIT sensor layout '{SENSOR_LAYOUT_NAME}': {N_ELECTRODES} training sensors across "
    f"{len(SENSOR_CLUSTER_SPECS)} dense clusters + {SENSOR_BACKBONE_COUNT} backbone sensors; "
    f"{N_HOLDOUT_ELECTRODES} boundary nodes held out."
)

boundary_theta = 2.0 * np.pi * np.arange(n_boundary) / n_boundary
current_patterns = np.zeros((N_CURRENT_PATTERNS, n_boundary), dtype=np.float64)
for l in range(N_CURRENT_PATTERNS):
    current_patterns[l] = np.cos((l + 1) * boundary_theta)

num_observation = N_CURRENT_PATTERNS * N_ELECTRODES
basis_truncated = Basis_Modes[:, :num_truncated_series]

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(electrode_flat_indices).to_csv('data/obs_locations.csv', index=False, header=False)
pd.DataFrame(heldout_electrode_flat_indices).to_csv('data/heldout_obs_locations.csv', index=False, header=False)

# ==========================================
# 2. Physics: EIT forward model
# ==========================================
jax.config.update("jax_enable_x64", True)

Basis = jnp.array(basis_truncated, dtype=jnp.float64)
obs_locations = jnp.array(electrode_flat_indices, dtype=int)
heldout_obs_locations = jnp.array(heldout_electrode_flat_indices, dtype=int)
current_patterns_jax = jnp.array(current_patterns, dtype=jnp.float64)
boundary_indices_jax = jnp.array(boundary_indices_ordered, dtype=int)

h = 1.0 / (N - 1)
NOISE_STD = 5e-4

_xface_left = (jnp.arange(N - 1)[:, None] * N + jnp.arange(N)[None, :]).ravel()
_xface_right = _xface_left + N
_yface_bot = (jnp.arange(N)[:, None] * N + jnp.arange(N - 1)[None, :]).ravel()
_yface_top = _yface_bot + 1
_boundary_ds = h * jnp.ones(n_boundary, dtype=jnp.float64)
_GAUGE_NODE = 0


def _assemble_eit_neumann(sigma_field):
    """
    Assemble the full Neumann stiffness matrix with a gauge pin at node 0.
    """
    n_total = N * N
    h2 = h * h

    sigma_xp = 2.0 * sigma_field[:-1, :] * sigma_field[1:, :] / (sigma_field[:-1, :] + sigma_field[1:, :] + 1e-30)
    sigma_yp = 2.0 * sigma_field[:, :-1] * sigma_field[:, 1:] / (sigma_field[:, :-1] + sigma_field[:, 1:] + 1e-30)

    kx = sigma_xp.ravel() / h2
    ky = sigma_yp.ravel() / h2

    A = jnp.zeros((n_total, n_total), dtype=jnp.float64)
    A = A.at[_xface_left, _xface_left].add(kx)
    A = A.at[_xface_right, _xface_right].add(kx)
    A = A.at[_xface_left, _xface_right].add(-kx)
    A = A.at[_xface_right, _xface_left].add(-kx)

    A = A.at[_yface_bot, _yface_bot].add(ky)
    A = A.at[_yface_top, _yface_top].add(ky)
    A = A.at[_yface_bot, _yface_top].add(-ky)
    A = A.at[_yface_top, _yface_bot].add(-ky)

    A = A.at[_GAUGE_NODE, :].set(0.0)
    A = A.at[:, _GAUGE_NODE].set(0.0)
    A = A.at[_GAUGE_NODE, _GAUGE_NODE].set(1.0)
    return A


@jax.jit
def _build_eit_rhs(current_pattern_boundary):
    rhs = jnp.zeros(N * N, dtype=jnp.float64)
    flux = current_pattern_boundary * _boundary_ds
    rhs = rhs.at[boundary_indices_jax].add(flux)
    rhs = rhs.at[_GAUGE_NODE].set(0.0)
    return rhs


_RHS_ALL_PATTERNS = jnp.stack(
    [_build_eit_rhs(current_patterns_jax[l]) for l in range(N_CURRENT_PATTERNS)],
    axis=1,
)


@jax.jit
def solve_forward(alpha):
    log_sigma = jnp.reshape(Basis @ alpha, (N, N))
    sigma_field = jnp.exp(log_sigma)
    A = _assemble_eit_neumann(sigma_field)
    U = jnp.linalg.solve(A, _RHS_ALL_PATTERNS)
    V = U[obs_locations, :]
    V = V - jnp.mean(V, axis=0, keepdims=True)
    return V.T.ravel()


@jax.jit
def solve_forward_holdout(alpha):
    log_sigma = jnp.reshape(Basis @ alpha, (N, N))
    sigma_field = jnp.exp(log_sigma)
    A = _assemble_eit_neumann(sigma_field)
    U = jnp.linalg.solve(A, _RHS_ALL_PATTERNS)
    V = U[heldout_obs_locations, :]
    V = V - jnp.mean(V, axis=0, keepdims=True)
    return V.T.ravel()


@jax.jit
def solve_single_pattern(alpha, pattern_idx):
    log_sigma = jnp.reshape(Basis @ alpha, (N, N))
    sigma_field = jnp.exp(log_sigma)
    A = _assemble_eit_neumann(sigma_field)
    rhs = _build_eit_rhs(current_patterns_jax[pattern_idx])
    u = jnp.linalg.solve(A, rhs)
    return u.reshape(N, N)


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
DEFAULT_N_GEN = 400
N_REF = 5000
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
run_ctx = init_run_results('eit_hlsi')

# ==========================================
# 3. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
alpha_true_np = np.random.randn(ACTIVE_DIM) * 0.5
y_clean_np = np.array(solve_forward(jnp.array(alpha_true_np)))
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)
y_holdout_clean_np = np.array(solve_forward_holdout(jnp.array(alpha_true_np)))
y_holdout_obs_np = y_holdout_clean_np + np.random.normal(0.0, NOISE_STD, size=y_holdout_clean_np.shape)

_batched_solve_forward_holdout = jax.jit(jax.vmap(solve_forward_holdout))
HELDOUT_BATCH_SIZE = 2

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, lik_aux = make_physics_likelihood(
    solve_forward,
    y_obs_np,
    NOISE_STD,
    use_gauss_newton_hessian=True,
    log_batch_size=25,
    grad_batch_size=10,
    hess_batch_size=1,
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
electrode_row = electrode_flat_indices // N
electrode_col = electrode_flat_indices % N
d_lat = ACTIVE_DIM


def reconstruct_log_conductivity(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    B = Basis_np[:, :latents.shape[1]]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))


def latent_to_log_conductivity(alpha):
    return reconstruct_log_conductivity(np.asarray(alpha)[None, :])[0]


def solve_potential_field(alpha_vec, pattern_idx=0):
    return np.array(solve_single_pattern(jnp.array(alpha_vec), pattern_idx))


true_field = latent_to_log_conductivity(alpha_true_np)
true_potential = solve_potential_field(alpha_true_np, pattern_idx=0)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_log_conductivity,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_clean_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    metrics = compute_heldout_predictive_metrics(
        samples,
        metrics,
        heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
        batched_forward_eval_fn=lambda alpha_batch: np.asarray(
            _batched_solve_forward_holdout(jnp.asarray(alpha_batch, dtype=jnp.float64))
        ),
        batched_forward_eval_batch_size=HELDOUT_BATCH_SIZE,
        y_holdout_obs_np=y_holdout_obs_np,
        noise_std=NOISE_STD,
        display_names=display_names,
        min_valid=10,
    )
except Exception as exc:
    print(f"WARNING: held-out predictive metrics failed and will be skipped: {exc}")

print('\n=== EIT field/data metrics ===')
print(f"{'Method':<24} | {'RelL2_m (%)':<12} | {'Pearson':<10} | {'RMSE_a':<12} | {'FwdRel':<12}")
print('-' * 84)
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
    target_name='EIT log-conductivity',
    display_names=display_names,
    reference_name=reference_title,
)

save_reproducibility_log(
    title='EIT HLSI run reproducibility log',
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
        'N_CURRENT_PATTERNS': N_CURRENT_PATTERNS,
        'SENSOR_LAYOUT_NAME': SENSOR_LAYOUT_NAME,
        'SENSOR_CLUSTER_SPECS': SENSOR_CLUSTER_SPECS,
        'SENSOR_BACKBONE_COUNT': SENSOR_BACKBONE_COUNT,
        'N_ELECTRODES': N_ELECTRODES,
        'N_HOLDOUT_ELECTRODES': N_HOLDOUT_ELECTRODES,
        'electrode_boundary_pos': electrode_boundary_pos,
        'heldout_electrode_flat_indices': heldout_electrode_flat_indices,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'HELDOUT_BATCH_SIZE': HELDOUT_BATCH_SIZE,
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

# ==========================================
# 4. Problem-specific visualization
# ==========================================

def _overlay_electrodes(ax):
    ax.scatter(electrode_col, electrode_row, c='lime', s=10, marker='s', alpha=0.8)


fig_field, axes_field = plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_log_conductivity,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_field,
    reference_bottom_title='Ground Truth\nLog-conductivity $m(x)$',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_electrodes,
    overlay_method_fn=_overlay_electrodes,
    suptitle=f'Inverse EIT (d={ACTIVE_DIM}): log-conductivity reconstruction',
    field_name='Log-conductivity $m(x)$',
)
plt.show()

print('\nVisualizing electric potential fields (pattern 1)...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

pot_vmin = float(np.min(true_potential))
pot_vmax = float(np.max(true_potential))
axes2[0].imshow(true_potential, cmap='RdBu_r', origin='lower', vmin=pot_vmin, vmax=pot_vmax)
axes2[0].scatter(electrode_col, electrode_row, c='yellow', s=18, marker='s', alpha=0.8)
axes2[0].set_title('Ground Truth\nPotential $u(x)$', fontsize=14)
axes2[0].axis('off')

for i, label in enumerate(methods_to_plot):
    col = i + 1
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        axes2[col].axis('off')
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    try:
        u_field = solve_potential_field(mean_lat, pattern_idx=0)
        axes2[col].imshow(u_field, cmap='RdBu_r', origin='lower', vmin=pot_vmin, vmax=pot_vmax)
        axes2[col].scatter(electrode_col, electrode_row, c='yellow', s=10, marker='s', alpha=0.6)
        axes2[col].set_title(f"{display_names.get(label, label)}\nPotential", fontsize=14)
    except Exception:
        axes2[col].set_title(f"{display_names.get(label, label)}\n(solve failed)", fontsize=14)
    axes2[col].axis('off')

plt.suptitle(f'Inverse EIT (d={ACTIVE_DIM}): electric potential (pattern 1)', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

print('\nVisualizing conductivity fields $\\sigma(x)=e^{m(x)}$...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
true_conductivity = np.exp(true_field)
cond_vmin = float(np.min(true_conductivity))
cond_vmax = float(np.max(true_conductivity))
axes3[0].imshow(true_conductivity, cmap='magma', origin='lower', vmin=cond_vmin, vmax=cond_vmax)
axes3[0].set_title('Ground Truth\n$\\sigma(x)=e^{m(x)}$', fontsize=14)
axes3[0].axis('off')

for i, label in enumerate(methods_to_plot):
    col = i + 1
    cond_f = np.exp(mean_fields[label])
    axes3[col].imshow(cond_f, cmap='magma', origin='lower', vmin=cond_vmin, vmax=cond_vmax)
    axes3[col].set_title(f"{display_names.get(label, label)}\n$\\sigma(x)=e^{{m(x)}}$", fontsize=14)
    axes3[col].axis('off')

plt.suptitle(f'Inverse EIT (d={ACTIVE_DIM}): conductivity field', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

print('\nVisualizing boundary voltage profiles...')
theta_electrodes = 2.0 * np.pi * electrode_boundary_pos / n_boundary
method_voltage_preds = {}
for label in methods_to_plot:
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    try:
        method_voltage_preds[label] = np.array(solve_forward(jnp.array(mean_lat)))
    except Exception:
        pass

preferred_order = [
    'HLSI', 'WC-HLSI', 'PoU-HLSI', 'CE-HLSI', 'CE-WC-HLSI', 'CE-PoU-HLSI', 'MALA (prior)'
]
labels_for_voltage_plot = [lab for lab in preferred_order if lab in method_voltage_preds]
if len(labels_for_voltage_plot) == 0:
    labels_for_voltage_plot = list(method_voltage_preds.keys())

n_patterns_show = min(4, N_CURRENT_PATTERNS)
pattern_indices = list(range(n_patterns_show))
n_cols_plot = min(2, n_patterns_show)
n_rows_plot = int(np.ceil(n_patterns_show / n_cols_plot))
fig4, axes4 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(7 * n_cols_plot, 4.5 * n_rows_plot), squeeze=False)

for ax, pattern_idx in zip(axes4.ravel(), pattern_indices):
    sl = slice(pattern_idx * N_ELECTRODES, (pattern_idx + 1) * N_ELECTRODES)
    y_obs_pat = y_obs_np[sl]
    y_clean_pat = y_clean_np[sl]
    ax.plot(theta_electrodes, y_clean_pat, 'k-o', label='Clean', linewidth=2, markersize=4)
    ax.plot(theta_electrodes, y_obs_pat, 'r.', label='Noisy obs', markersize=5, alpha=0.7)
    for label in labels_for_voltage_plot:
        y_pred_pat = method_voltage_preds[label][sl]
        ax.plot(theta_electrodes, y_pred_pat, '--', label=display_names.get(label, label), linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Boundary angle (rad)', fontsize=12)
    ax.set_ylabel('Voltage', fontsize=12)
    ax.set_title(f'Boundary voltage profile (pattern {pattern_idx + 1})', fontsize=14)
    ax.grid(True, alpha=0.3)

for ax in axes4.ravel()[len(pattern_indices):]:
    ax.axis('off')

handles, labels = axes4.ravel()[0].get_legend_handles_labels()
fig4.legend(handles, labels, loc='upper center', ncol=min(len(labels), 4), fontsize=10)
fig4.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== EIT HLSI pipeline complete ===')
