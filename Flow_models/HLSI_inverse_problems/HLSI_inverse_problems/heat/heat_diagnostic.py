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
# CE/WC compatibility diagnostics
# ==========================================
DIAG_QUERY_SOURCE_LABELS = ('HLSI', 'WC-HLSI', 'CE-HLSI', 'CE-WC-HLSI')
DIAG_T_GRID = np.array([0.01, 0.05, 0.15, 0.40, 1.00], dtype=np.float64)
DIAG_MAX_SAMPLES_PER_METHOD = 25
DIAG_TOPK_REF = 256
DIAG_QUERY_SEED = 123


def _diag_require_bank_keys(bank, required_keys):
    missing = [key for key in required_keys if key not in bank]
    if missing:
        raise KeyError(
            f"Diagnostic bank is missing keys {missing}. "
            f"Available keys: {sorted(bank.keys())}"
        )


def _diag_softmax_from_log(logw):
    logw = logw - torch.max(logw)
    w = torch.exp(logw)
    return w / torch.clamp(torch.sum(w), min=1e-300)


def _diag_make_query_bank(samples_dict, source_labels, t_grid, max_samples_per_method, seed=123):
    rng = np.random.default_rng(int(seed))
    queries = []
    query_id = 0
    for label in source_labels:
        if label not in samples_dict:
            continue
        samps = get_valid_samples(samples_dict[label])
        if samps.shape[0] == 0:
            continue
        samps = np.asarray(samps)
        n_take = min(int(max_samples_per_method), samps.shape[0])
        choose = rng.choice(samps.shape[0], size=n_take, replace=False)
        x_sel = samps[choose]
        for sample_idx, x in enumerate(x_sel):
            x = np.asarray(x, dtype=np.float64)
            for t in np.asarray(t_grid, dtype=np.float64):
                var_t = max(1e-12, 1.0 - np.exp(-2.0 * float(t)))
                et = np.exp(-float(t))
                noise = rng.standard_normal(size=x.shape[0])
                y = et * x + np.sqrt(var_t) * noise
                queries.append({
                    'query_id': int(query_id),
                    'source_label': str(label),
                    'sample_local_idx': int(sample_idx),
                    't': float(t),
                    'x_source': x.copy(),
                    'y': y.astype(np.float64, copy=False),
                })
                query_id += 1
    if not queries:
        raise RuntimeError('Diagnostic query bank is empty.')
    return queries


def _diag_compute_log_weights_batch(y_batch, t, X_ref, log_ref):
    et = math.exp(-float(t))
    var_t = max(1e-12, 1.0 - math.exp(-2.0 * float(t)))
    y2 = torch.sum(y_batch * y_batch, dim=1, keepdim=True)
    x2 = torch.sum(X_ref * X_ref, dim=1).unsqueeze(0)
    cross = y_batch @ X_ref.T
    log_ou = -0.5 * (y2 + (et * et) * x2 - 2.0 * et * cross) / var_t
    return log_ou + log_ref.unsqueeze(0)


def _diag_component_gate_apply(d_comp, eigvecs, eigvals, trusted, t):
    et = math.exp(-float(t))
    et2 = et * et
    var_t = max(1e-12, 1.0 - math.exp(-2.0 * float(t)))
    gate_eig = torch.where(
        trusted,
        et2 / (et2 + var_t * eigvals + 1e-30),
        torch.zeros_like(eigvals),
    )
    proj = torch.einsum('kij,ki->kj', eigvecs, d_comp)
    return torch.einsum('kij,kj->ki', eigvecs, gate_eig * proj)


def _diag_ce_gate_apply(delta_bar, P_bar, t):
    et = math.exp(-float(t))
    et2 = et * et
    var_t = max(1e-12, 1.0 - math.exp(-2.0 * float(t)))
    P_bar = 0.5 * (P_bar + P_bar.T)
    eigvals_bar, eigvecs_bar = torch.linalg.eigh(P_bar)
    gate_bar = et2 / (et2 + var_t * eigvals_bar + 1e-30)
    proj_bar = eigvecs_bar.T @ delta_bar
    gated = eigvecs_bar @ (gate_bar * proj_bar)
    return gated, eigvals_bar, eigvecs_bar


def run_ce_wc_diagnostics(samples_dict, precomp_bank, display_names, run_results_dir,
                          problem_tag='problem',
                          weight_modes=('L', 'WC'),
                          t_grid=DIAG_T_GRID,
                          source_labels=DIAG_QUERY_SOURCE_LABELS,
                          max_samples_per_method=DIAG_MAX_SAMPLES_PER_METHOD,
                          topk_ref=DIAG_TOPK_REF,
                          seed=DIAG_QUERY_SEED):
    _diag_require_bank_keys(
        precomp_bank,
        ['X_ref', 's0_post_ref', 'P_ref', 'gated_info', 'log_lik_ref', 'log_mass_ref'],
    )
    gated_info = precomp_bank['gated_info']
    _diag_require_bank_keys(gated_info, ['eigvecs', 'eigvals', 'trusted'])

    print(f"\n=== {problem_tag}: CE/WC compatibility diagnostics ===")
    query_bank = _diag_make_query_bank(
        samples_dict,
        source_labels=source_labels,
        t_grid=t_grid,
        max_samples_per_method=max_samples_per_method,
        seed=seed,
    )
    print(f"Built diagnostic query bank with {len(query_bank)} queries.")

    X_ref = precomp_bank['X_ref']
    s0_post_ref = precomp_bank['s0_post_ref']
    P_ref = precomp_bank['P_ref']
    eigvecs_ref = gated_info['eigvecs']
    eigvals_ref = gated_info['eigvals']
    trusted_ref = gated_info['trusted']

    bank_device = X_ref.device
    bank_dtype = X_ref.dtype
    if X_ref.ndim != 2:
        raise ValueError(f"Expected X_ref to have shape [N, d], got {tuple(X_ref.shape)}")

    log_bank_map = {
        'L': precomp_bank['log_lik_ref'],
        'WC': precomp_bank['log_mass_ref'],
    }
    weight_modes = tuple(str(mode) for mode in weight_modes if mode in log_bank_map)
    if not weight_modes:
        raise ValueError('No valid weight modes requested for diagnostics.')

    N_ref = int(X_ref.shape[0])
    d_lat = int(X_ref.shape[1])
    topk_ref = min(int(topk_ref), N_ref)

    query_rows = []
    direction_rows = []

    queries_by_t = {}
    for item in query_bank:
        queries_by_t.setdefault(float(item['t']), []).append(item)

    with torch.no_grad():
        for t in sorted(queries_by_t.keys()):
            group = queries_by_t[float(t)]
            y_batch = torch.tensor(
                np.stack([item['y'] for item in group], axis=0),
                device=bank_device,
                dtype=bank_dtype,
            )
            et = math.exp(-float(t))
            inv_v = 1.0 / max(1e-12, 1.0 - math.exp(-2.0 * float(t)))

            for weight_mode in weight_modes:
                log_ref = log_bank_map[weight_mode].to(device=bank_device, dtype=bank_dtype)
                logw_batch = _diag_compute_log_weights_batch(y_batch, t, X_ref, log_ref)
                lse_full = torch.logsumexp(logw_batch, dim=1, keepdim=True)
                w_full = torch.exp(logw_batch - lse_full)
                ess_full = 1.0 / torch.clamp(torch.sum(w_full ** 2, dim=1), min=1e-300)
                entropy_full = -torch.sum(
                    w_full * (logw_batch - lse_full),
                    dim=1,
                )
                norm_entropy = entropy_full / math.log(float(N_ref) + 1e-30)

                topv, topi = torch.topk(logw_batch, k=topk_ref, dim=1)
                lse_topk = torch.logsumexp(topv, dim=1, keepdim=True)
                wk_top = torch.exp(topv - lse_topk)
                topk_mass = torch.exp(lse_topk.squeeze(1) - lse_full.squeeze(1))

                for m, item in enumerate(group):
                    idx = topi[m]
                    wk = wk_top[m]
                    xk = X_ref[idx]
                    s0k = s0_post_ref[idx]
                    Pk = P_ref[idx]
                    eigvecs_k = eigvecs_ref[idx]
                    eigvals_k = eigvals_ref[idx]
                    trusted_k = trusted_ref[idx]

                    y = y_batch[m]
                    s_twd_comp = -inv_v * (y.unsqueeze(0) - et * xk)
                    s_tsi_comp = (1.0 / et) * s0k
                    d_comp = s_tsi_comp - s_twd_comp

                    coupled_terms = _diag_component_gate_apply(
                        d_comp, eigvecs_k, eigvals_k, trusted_k, t
                    )
                    coupled = torch.einsum('k,ki->i', wk, coupled_terms)
                    delta_bar = torch.einsum('k,ki->i', wk, d_comp)
                    P_bar = torch.einsum('k,kij->ij', wk, Pk)
                    decoupled, eigvals_bar, eigvecs_bar = _diag_ce_gate_apply(delta_bar, P_bar, t)
                    defect = decoupled - coupled

                    dir_proj = d_comp @ eigvecs_bar
                    dir_mean = torch.einsum('k,kj->j', wk, dir_proj)
                    dir_var = torch.einsum('k,kj->j', wk, (dir_proj - dir_mean.unsqueeze(0)) ** 2)

                    row = {
                        'query_id': int(item['query_id']),
                        'source_label': item['source_label'],
                        'source_display': display_names.get(item['source_label'], item['source_label']),
                        'sample_local_idx': int(item['sample_local_idx']),
                        'weight_mode': weight_mode,
                        't': float(t),
                        'ess_full': float(ess_full[m].item()),
                        'entropy_full': float(entropy_full[m].item()),
                        'norm_entropy': float(norm_entropy[m].item()),
                        'topk_mass': float(topk_mass[m].item()),
                        'delta_bar_norm': float(torch.linalg.norm(delta_bar).item()),
                        'coupled_norm': float(torch.linalg.norm(coupled).item()),
                        'decoupled_norm': float(torch.linalg.norm(decoupled).item()),
                        'gate_defect_norm': float(torch.linalg.norm(defect).item()),
                        'gate_defect_rel': float(
                            torch.linalg.norm(defect).item() /
                            max(1e-12, torch.linalg.norm(coupled).item())
                        ),
                        'dir_var_total': float(torch.sum(dir_var).item()),
                        'dir_var_max': float(torch.max(dir_var).item()),
                        'lambda_bar_min': float(torch.min(eigvals_bar).item()),
                        'lambda_bar_max': float(torch.max(eigvals_bar).item()),
                    }
                    query_rows.append(row)

                    for k in range(d_lat):
                        direction_rows.append({
                            'query_id': int(item['query_id']),
                            'source_label': item['source_label'],
                            'source_display': display_names.get(item['source_label'], item['source_label']),
                            'weight_mode': weight_mode,
                            't': float(t),
                            'eig_rank': int(k + 1),
                            'lambda_bar': float(eigvals_bar[k].item()),
                            'dir_var': float(dir_var[k].item()),
                        })

    diag_df = pd.DataFrame(query_rows)
    dir_df = pd.DataFrame(direction_rows)

    if diag_df.empty:
        raise RuntimeError('Diagnostic dataframe is empty.')

    summary_df = (
        diag_df
        .groupby(['weight_mode', 't'], as_index=False)
        .agg(
            n_queries=('query_id', 'count'),
            ess_full_mean=('ess_full', 'mean'),
            ess_full_std=('ess_full', 'std'),
            norm_entropy_mean=('norm_entropy', 'mean'),
            norm_entropy_std=('norm_entropy', 'std'),
            topk_mass_mean=('topk_mass', 'mean'),
            gate_defect_norm_mean=('gate_defect_norm', 'mean'),
            gate_defect_norm_std=('gate_defect_norm', 'std'),
            gate_defect_rel_mean=('gate_defect_rel', 'mean'),
            dir_var_total_mean=('dir_var_total', 'mean'),
            dir_var_total_std=('dir_var_total', 'std'),
            dir_var_max_mean=('dir_var_max', 'mean'),
            lambda_bar_max_mean=('lambda_bar_max', 'mean'),
        )
    )

    dir_summary_df = (
        dir_df
        .groupby(['weight_mode', 't', 'eig_rank'], as_index=False)
        .agg(
            dir_var_mean=('dir_var', 'mean'),
            dir_var_std=('dir_var', 'std'),
            lambda_bar_mean=('lambda_bar', 'mean'),
        )
    )

    tag = str(problem_tag).lower().replace(' ', '_')
    query_csv = os.path.join(run_results_dir, f'{tag}_ce_wc_diag_queries.csv')
    summary_csv = os.path.join(run_results_dir, f'{tag}_ce_wc_diag_summary.csv')
    dir_csv = os.path.join(run_results_dir, f'{tag}_ce_wc_diag_directional.csv')
    dir_summary_csv = os.path.join(run_results_dir, f'{tag}_ce_wc_diag_directional_summary.csv')

    diag_df.to_csv(query_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    dir_df.to_csv(dir_csv, index=False)
    dir_summary_df.to_csv(dir_summary_csv, index=False)

    print(f"Saved diagnostic query table to {query_csv}")
    print(f"Saved diagnostic summary table to {summary_csv}")
    print(f"Saved diagnostic directional table to {dir_csv}")
    print(f"Saved diagnostic directional summary table to {dir_summary_csv}")

    print('\nDiagnostic summary by weight mode and time:')
    print(summary_df.to_string(index=False))

    # Figure 1: mean ESS
    plt.figure(figsize=(7.2, 4.6))
    for weight_mode in weight_modes:
        sub = summary_df[summary_df['weight_mode'] == weight_mode].sort_values('t')
        plt.semilogx(sub['t'], sub['ess_full_mean'], marker='o', linewidth=2, label=weight_mode)
    plt.xlabel('Diffusion time t')
    plt.ylabel('Mean ESS')
    plt.title(f'{problem_tag}: SNIS ESS diagnostic')
    plt.grid(True, which='both', alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 2: normalized entropy
    plt.figure(figsize=(7.2, 4.6))
    for weight_mode in weight_modes:
        sub = summary_df[summary_df['weight_mode'] == weight_mode].sort_values('t')
        plt.semilogx(sub['t'], sub['norm_entropy_mean'], marker='o', linewidth=2, label=weight_mode)
    plt.xlabel('Diffusion time t')
    plt.ylabel('Mean normalized entropy')
    plt.title(f'{problem_tag}: weight-entropy diagnostic')
    plt.grid(True, which='both', alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 3: gate defect norm
    plt.figure(figsize=(7.2, 4.6))
    for weight_mode in weight_modes:
        sub = summary_df[summary_df['weight_mode'] == weight_mode].sort_values('t')
        plt.semilogx(sub['t'], sub['gate_defect_norm_mean'], marker='o', linewidth=2, label=weight_mode)
    plt.xlabel('Diffusion time t')
    plt.ylabel(r'Mean $\|\Delta_{\mathrm{gate}}\|_2$')
    plt.title(f'{problem_tag}: coupled-vs-decoupled gate defect')
    plt.grid(True, which='both', alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 4+: directional disagreement spectra
    for weight_mode in weight_modes:
        plt.figure(figsize=(7.2, 4.8))
        sub = dir_summary_df[dir_summary_df['weight_mode'] == weight_mode]
        for t in sorted(sub['t'].unique()):
            cur = sub[sub['t'] == t].sort_values('eig_rank')
            plt.semilogy(cur['eig_rank'], cur['dir_var_mean'] + 1e-16, marker='o', linewidth=1.5, label=f't={t:g}')
        plt.xlabel('Local CE eigendirection rank')
        plt.ylabel('Mean directional disagreement variance')
        plt.title(f'{problem_tag}: directional disagreement spectrum ({weight_mode})')
        plt.grid(True, which='both', alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return {
        'query_df': diag_df,
        'summary_df': summary_df,
        'direction_df': dir_df,
        'direction_summary_df': dir_summary_df,
        'query_csv': query_csv,
        'summary_csv': summary_csv,
        'direction_csv': dir_csv,
        'direction_summary_csv': dir_summary_csv,
    }


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

diag_bank = pipeline['precomp']['base']
diag_outputs = run_ce_wc_diagnostics(
    samples,
    diag_bank,
    display_names=display_names,
    run_results_dir=run_ctx['run_results_dir'],
    problem_tag='heat',
)


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
