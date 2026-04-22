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
run_ctx = init_run_results('helmholtz_scattering_hlsi')

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
NOISE_STD = 3e-4
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

diag_bank = pipeline['precomp']['base']
diag_outputs = run_ce_wc_diagnostics(
    samples,
    diag_bank,
    display_names=display_names,
    run_results_dir=run_ctx['run_results_dir'],
    problem_tag='helmholtz',
)


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
