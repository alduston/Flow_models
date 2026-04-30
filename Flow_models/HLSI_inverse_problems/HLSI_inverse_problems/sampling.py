# -*- coding: utf-8 -*-
"""Shared HLSI sampling, precomputation, metrics, and plotting utilities.

This module centralizes the common logic duplicated across the inverse-problem
experiment scripts so sampler changes can be made once and reused everywhere.
"""

import os 
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import gc
import math
import platform
import random
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pprint import pformat
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTIVE_DIM = None
DEFAULT_N_GEN = 500
HESS_MIN = 1e-6
HESS_MAX = 1e8
LEAF_MIN_PREC = HESS_MIN
LEAF_MAX_PREC = HESS_MAX
LEAF_ABS_SCALE = 1.0
GNL_PILOT_N = 512
GNL_STIFF_LAMBDA_CUT = HESS_MAX
GNL_USE_DOMINANT_PARTICLE_NEWTON = True

RUN_TIMESTAMP = None
RUN_RESULTS_ROOT = 'run_results'
RUN_RESULTS_DIR = None
RUN_RESULTS_STEM = None
_PLOT_SAVE_COUNTER = 0
_ORIGINAL_PLT_SHOW = getattr(plt.show, '_run_results_original_show', plt.show)
_RUN_RESULTS_SHOW_IS_ACTIVE = False


def configure_sampling(active_dim=None, default_n_gen=500,
                       hess_min=1e-6, hess_max=1e8,
                       leaf_min_prec=None, leaf_max_prec=None, leaf_abs_scale=1.0,
                       gnl_pilot_n=512, gnl_stiff_lambda_cut=None,
                       gnl_use_dominant_particle_newton=True):
    global ACTIVE_DIM, DEFAULT_N_GEN, HESS_MIN, HESS_MAX
    global LEAF_MIN_PREC, LEAF_MAX_PREC, LEAF_ABS_SCALE
    global GNL_PILOT_N, GNL_STIFF_LAMBDA_CUT, GNL_USE_DOMINANT_PARTICLE_NEWTON

    ACTIVE_DIM = active_dim
    DEFAULT_N_GEN = int(default_n_gen)
    HESS_MIN = float(hess_min)
    HESS_MAX = float(hess_max)
    LEAF_MIN_PREC = float(HESS_MIN if leaf_min_prec is None else leaf_min_prec)
    LEAF_MAX_PREC = float(HESS_MAX if leaf_max_prec is None else leaf_max_prec)
    LEAF_ABS_SCALE = float(leaf_abs_scale)
    GNL_PILOT_N = int(gnl_pilot_n)
    GNL_STIFF_LAMBDA_CUT = float(HESS_MAX if gnl_stiff_lambda_cut is None else gnl_stiff_lambda_cut)
    GNL_USE_DOMINANT_PARTICLE_NEWTON = bool(gnl_use_dominant_particle_newton)


def _sanitize_run_results_name(text, max_len=96):
    text = str(text).strip().replace('\n', ' ')
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)
    while '__' in safe:
        safe = safe.replace('__', '_')
    safe = safe.strip('_')
    if not safe:
        safe = 'figure'
    return safe[:max_len]


def _infer_figure_basename(fig, fallback):
    title_candidates = []
    suptitle = getattr(fig, '_suptitle', None)
    if suptitle is not None:
        try:
            txt = suptitle.get_text().strip()
            if txt:
                title_candidates.append(txt)
        except Exception:
            pass
    for ax in fig.axes:
        try:
            txt = ax.get_title().strip()
            if txt:
                title_candidates.append(txt)
                break
        except Exception:
            pass
    if title_candidates:
        return _sanitize_run_results_name(title_candidates[0])
    return _sanitize_run_results_name(fallback)


def _save_all_open_figures_to_run_results():
    global _PLOT_SAVE_COUNTER
    if RUN_RESULTS_DIR is None or RUN_RESULTS_STEM is None:
        return
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        if getattr(fig, '_run_results_saved', False):
            continue
        _PLOT_SAVE_COUNTER += 1
        basename = _infer_figure_basename(fig, f'figure_{_PLOT_SAVE_COUNTER:02d}')
        png_path = os.path.join(
            RUN_RESULTS_DIR,
            f'{RUN_RESULTS_STEM}_figure_{_PLOT_SAVE_COUNTER:02d}_{basename}.png',
        )
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig._run_results_saved = True
        print(f'Saved figure to {png_path}')


def _patched_show(*args, **kwargs):
    global _RUN_RESULTS_SHOW_IS_ACTIVE
    if _RUN_RESULTS_SHOW_IS_ACTIVE:
        return _ORIGINAL_PLT_SHOW(*args, **kwargs)
    _RUN_RESULTS_SHOW_IS_ACTIVE = True
    try:
        _save_all_open_figures_to_run_results()
        return _ORIGINAL_PLT_SHOW(*args, **kwargs)
    finally:
        _RUN_RESULTS_SHOW_IS_ACTIVE = False


_patched_show._run_results_original_show = _ORIGINAL_PLT_SHOW
_patched_show._run_results_is_patched = True
plt.show = _patched_show


def init_run_results(run_prefix: str, root: str = 'run_results'):
    global RUN_TIMESTAMP, RUN_RESULTS_ROOT, RUN_RESULTS_DIR, RUN_RESULTS_STEM, _PLOT_SAVE_COUNTER
    RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_RESULTS_ROOT = root

    os.makedirs(RUN_RESULTS_ROOT, exist_ok=True)

    # Append a random 3-digit suffix so batched jobs started at nearly the same
    # time do not overwrite one another. If a collision still occurs, resample.
    for _ in range(1000):
        run_suffix = f'{random.randint(0, 999):03d}'
        run_name = f'{run_prefix}_{RUN_TIMESTAMP}_{run_suffix}'
        run_dir = os.path.join(RUN_RESULTS_ROOT, run_name)
        if not os.path.exists(run_dir):
            RUN_RESULTS_DIR = run_dir
            RUN_RESULTS_STEM = run_name
            break
    else:
        raise RuntimeError(
            f'Could not find a unique run-results directory for prefix={run_prefix!r} '
            f'under {RUN_RESULTS_ROOT!r} after 1000 attempts.'
        )

    _PLOT_SAVE_COUNTER = 0
    os.makedirs(RUN_RESULTS_DIR, exist_ok=False)
    return {
        'run_timestamp': RUN_TIMESTAMP,
        'run_results_root': RUN_RESULTS_ROOT,
        'run_results_dir': RUN_RESULTS_DIR,
        'run_results_stem': RUN_RESULTS_STEM,
    }


def _summarize_for_repro(value):
    if isinstance(value, (bool, int, float, str, type(None))):
        return repr(value)
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return repr(value.tolist())
        return (
            f'np.ndarray(shape={value.shape}, dtype={value.dtype}, '
            f'min={float(np.min(value)):.6g}, max={float(np.max(value)):.6g}, '
            f'mean={float(np.mean(value)):.6g}, std={float(np.std(value)):.6g})'
        )
    if torch.is_tensor(value):
        value_cpu = value.detach().cpu()
        if value_cpu.numel() <= 32:
            return repr(value_cpu.tolist())
        return (
            f'torch.Tensor(shape={tuple(value_cpu.shape)}, dtype={value_cpu.dtype}, '
            f'min={float(value_cpu.min().item()):.6g}, max={float(value_cpu.max().item()):.6g}, '
            f'mean={float(value_cpu.double().mean().item()):.6g}, '
            f'std={float(value_cpu.double().std(unbiased=False).item()):.6g})'
        )
    if isinstance(value, (list, tuple, dict, set)):
        try:
            formatted = pformat(value, width=100, compact=False, sort_dicts=False)
        except TypeError:
            formatted = pformat(value, width=100, compact=False)
        if len(formatted) > 5000:
            formatted = formatted[:5000] + '\n... [truncated]'
        return formatted
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return f'{type(value).__name__}(shape={getattr(value, "shape", None)}, dtype={getattr(value, "dtype", None)})'
    return repr(value)


def save_reproducibility_log(title='HLSI run reproducibility log', config=None, extra_sections=None):
    if RUN_RESULTS_DIR is None or RUN_RESULTS_STEM is None:
        raise RuntimeError('init_run_results must be called before save_reproducibility_log.')
    log_path = os.path.join(RUN_RESULTS_DIR, f'{RUN_RESULTS_STEM}_parameters.txt')
    lines = [title, '=' * 72]
    lines.append(f'run_timestamp = {RUN_TIMESTAMP}')
    lines.append(f'python_version = {platform.python_version()}')
    lines.append(f'platform = {platform.platform()}')
    lines.append(f'numpy_version = {np.__version__}')
    lines.append(f'pandas_version = {pd.__version__}')
    lines.append(f'torch_version = {torch.__version__}')
    lines.append(f'jax_version = {jax.__version__}')
    lines.append(f'device = {device}')
    lines.append(f'cuda_available = {torch.cuda.is_available()}')
    lines.append(f'run_results_dir = {RUN_RESULTS_DIR}')
    lines.append('')
    if config:
        lines.append('Key configuration values')
        lines.append('-' * 72)
        for name, value in config.items():
            lines.append(f'{name} = {_summarize_for_repro(value)}')
        lines.append('')
    if extra_sections:
        for section_name, section_value in extra_sections.items():
            lines.append(section_name)
            lines.append('-' * 72)
            lines.append(_summarize_for_repro(section_value))
            lines.append('')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Saved reproducibility log to {log_path}')
    return log_path


def zip_run_results_dir():
    if RUN_RESULTS_DIR is None:
        raise RuntimeError('init_run_results must be called before zip_run_results_dir.')
    _save_all_open_figures_to_run_results()
    zip_path = shutil.make_archive(
        RUN_RESULTS_DIR,
        'zip',
        root_dir=RUN_RESULTS_ROOT,
        base_dir=os.path.basename(RUN_RESULTS_DIR),
    )
    print(f'Compressed run-results directory to {zip_path}')
    return zip_path


class PhysicsLikelihood:
    def __init__(self, y_obs_np, sigma, batch_log_lik, batch_grad_lik,
                 batch_loglik_and_grad, batch_hess_lik_python,
                 solve_forward_jac_jax,
                 log_batch_size=50, grad_batch_size=25, hess_batch_size=2):
        self.y_obs_jax = jnp.array(y_obs_np)
        self.sigma = float(sigma)
        self.batch_log_lik = batch_log_lik
        self.batch_grad_lik = batch_grad_lik
        self.batch_loglik_and_grad = batch_loglik_and_grad
        self.batch_hess_lik_python = batch_hess_lik_python
        self.solve_forward_jac_jax = solve_forward_jac_jax
        self.log_batch_size = int(log_batch_size)
        self.grad_batch_size = int(grad_batch_size)
        self.hess_batch_size = int(hess_batch_size)

    def _to_numpy_batch(self, x_torch):
        x_np = np.asarray(x_torch.detach().cpu().numpy(), dtype=np.float64)
        if x_np.ndim == 1:
            x_np = x_np[None, :]
        return x_np

    def _chunked_eval(self, x_torch, fn, batch_size):
        x_np = self._to_numpy_batch(x_torch)
        outs = []
        for i in range(0, x_np.shape[0], batch_size):
            outs.append(np.asarray(fn(x_np[i:i + batch_size], self.y_obs_jax, self.sigma)))
        out_np = np.concatenate(outs, axis=0)
        return torch.tensor(out_np, device=x_torch.device, dtype=torch.float64)

    def log_likelihood(self, x_torch, batch_size=None):
        if batch_size is None:
            batch_size = self.log_batch_size
        return self._chunked_eval(x_torch, self.batch_log_lik, batch_size)

    def grad_log_likelihood(self, x_torch, batch_size=None):
        if batch_size is None:
            batch_size = self.grad_batch_size
        return self._chunked_eval(x_torch, self.batch_grad_lik, batch_size)

    def log_likelihood_and_grad(self, x_torch, batch_size=None):
        if batch_size is None:
            batch_size = min(self.log_batch_size, self.grad_batch_size)
        x_np = self._to_numpy_batch(x_torch)
        ll_list, grad_list = [], []
        for i in range(0, x_np.shape[0], batch_size):
            ll_chunk, grad_chunk = self.batch_loglik_and_grad(x_np[i:i + batch_size], self.y_obs_jax, self.sigma)
            ll_list.append(np.asarray(ll_chunk))
            grad_list.append(np.asarray(grad_chunk))
        ll_np = np.concatenate(ll_list, axis=0)
        grad_np = np.concatenate(grad_list, axis=0)
        return (
            torch.tensor(ll_np, device=x_torch.device, dtype=torch.float64),
            torch.tensor(grad_np, device=x_torch.device, dtype=torch.float64),
        )

    def hess_log_likelihood(self, x_torch, batch_size=None):
        if batch_size is None:
            batch_size = self.hess_batch_size
        x_np = self._to_numpy_batch(x_torch)
        hess_list = []
        for i in range(0, x_np.shape[0], batch_size):
            hess_list.append(self.batch_hess_lik_python(x_np[i:i + batch_size], self.y_obs_jax, self.sigma))
        hess_np = np.concatenate(hess_list, axis=0)
        return torch.tensor(hess_np, device=x_torch.device, dtype=torch.float64)


def make_physics_likelihood(solve_forward: Callable, y_obs_np, sigma,
                            use_gauss_newton_hessian=True,
                            log_batch_size=50, grad_batch_size=25, hess_batch_size=2):
    @jax.jit
    def log_likelihood_jax(alpha_k, y_obs_jax, sigma_inner):
        y_pred = solve_forward(alpha_k)
        resid = y_pred - y_obs_jax
        return -jnp.sum(resid ** 2) / (2.0 * sigma_inner ** 2)

    grad_lik_jax = jax.jit(jax.grad(log_likelihood_jax, argnums=0))
    loglik_and_grad_jax = jax.jit(jax.value_and_grad(log_likelihood_jax, argnums=0))
    solve_forward_jac_jax = jax.jit(jax.jacfwd(solve_forward))

    @jax.jit
    def hess_lik_gn_jax(alpha_k, y_obs_jax, sigma_inner):
        J = solve_forward_jac_jax(alpha_k)
        return -(J.T @ J) / (sigma_inner ** 2)

    hess_lik_exact_jax = jax.jit(jax.hessian(log_likelihood_jax, argnums=0))

    def hess_lik_jax(alpha_k, y_obs_jax, sigma_inner):
        if use_gauss_newton_hessian:
            return hess_lik_gn_jax(alpha_k, y_obs_jax, sigma_inner)
        return hess_lik_exact_jax(alpha_k, y_obs_jax, sigma_inner)

    batch_log_lik = jax.vmap(log_likelihood_jax, in_axes=(0, None, None))
    batch_grad_lik = jax.vmap(grad_lik_jax, in_axes=(0, None, None))
    batch_loglik_and_grad = jax.vmap(loglik_and_grad_jax, in_axes=(0, None, None))
    batch_hess_lik = jax.vmap(hess_lik_jax, in_axes=(0, None, None))

    def batch_hess_lik_python(x_np, y_obs_jax, sigma_inner):
        return np.stack(
            [np.asarray(hess_lik_jax(x_np[i], y_obs_jax, sigma_inner)) for i in range(x_np.shape[0])],
            axis=0,
        )

    lik_model = PhysicsLikelihood(
        y_obs_np=y_obs_np,
        sigma=sigma,
        batch_log_lik=batch_log_lik,
        batch_grad_lik=batch_grad_lik,
        batch_loglik_and_grad=batch_loglik_and_grad,
        batch_hess_lik_python=batch_hess_lik_python,
        solve_forward_jac_jax=solve_forward_jac_jax,
        log_batch_size=log_batch_size,
        grad_batch_size=grad_batch_size,
        hess_batch_size=hess_batch_size,
    )
    aux = {
        'log_likelihood_jax': log_likelihood_jax,
        'grad_lik_jax': grad_lik_jax,
        'loglik_and_grad_jax': loglik_and_grad_jax,
        'solve_forward_jac_jax': solve_forward_jac_jax,
        'hess_lik_gn_jax': hess_lik_gn_jax,
        'hess_lik_exact_jax': hess_lik_exact_jax,
        'hess_lik_jax': hess_lik_jax,
        'batch_log_lik': batch_log_lik,
        'batch_grad_lik': batch_grad_lik,
        'batch_loglik_and_grad': batch_loglik_and_grad,
        'batch_hess_lik': batch_hess_lik,
    }
    return lik_model, aux

class GaussianPrior(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def sample(self, n):
        return torch.randn(n, self.dim, device=device)

    def log_prob(self, x):
        return -0.5 * torch.sum(x ** 2, dim=1) - (self.dim / 2.0) * math.log(2 * math.pi)

    def score0(self, x):
        return -x


def _batched_log_likelihood_only(lik_model, x, batch_size=256):
    chunks = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            chunks.append(lik_model.log_likelihood(x[i:i + batch_size]))
    return torch.cat(chunks, dim=0)


def sample_gaussian_from_precision(mean, precision, n_samples, eps=1e-10):
    """Sample x ~ N(mean, precision^{-1}) for dense SPD precision."""
    evals, evecs = torch.linalg.eigh(precision)
    evals = torch.clamp(evals, min=eps)
    transform = evecs * torch.rsqrt(evals).unsqueeze(0)
    z = torch.randn(n_samples, mean.numel(), device=mean.device, dtype=mean.dtype)
    return mean.unsqueeze(0) + z @ transform.T


def build_gnl_factorization(prior_model, lik_model, pilot_n=GNL_PILOT_N,
                            stiff_lambda_cut=GNL_STIFF_LAMBDA_CUT,
                            use_dominant_particle_newton=GNL_USE_DOMINANT_PARTICLE_NEWTON):
    """
    Build the partially informed base law ṡp0 ∝ p0 * L_S from the GN factorization
    around a fixed expansion point x_star. The expansion point is chosen once using
    a pilot prior cloud, then reused for all downstream queries.
    """
    d = prior_model.dim
    I = torch.eye(d, device=device, dtype=torch.float64)

    print(f"Building GNL factorization with pilot cloud of {pilot_n} prior samples...")
    x_pilot = prior_model.sample(pilot_n)
    log_lik_pilot = _batched_log_likelihood_only(lik_model, x_pilot, batch_size=256)
    best_idx = int(torch.argmax(log_lik_pilot).item())
    x_anchor = x_pilot[best_idx].clone()

    if use_dominant_particle_newton:
        grad_anchor = lik_model.grad_log_likelihood(x_anchor.unsqueeze(0))[0]
        hess_anchor = lik_model.hess_log_likelihood(x_anchor.unsqueeze(0))[0]
        P_anchor = I - 0.5 * (hess_anchor + hess_anchor.T)
        s_post_anchor = prior_model.score0(x_anchor.unsqueeze(0))[0] + grad_anchor
        delta_anchor = torch.linalg.solve(P_anchor + 1e-6 * I, s_post_anchor)
        x_star = x_anchor + delta_anchor
    else:
        x_star = x_anchor

    J_star_np = np.array(lik_model.solve_forward_jac_jax(np.array(x_star.detach().cpu().numpy())))
    J_star = torch.tensor(J_star_np, device=device, dtype=torch.float64)

    # Thin SVD J = U diag(s) V^T. The GN likelihood precision along v_k is s_k^2 / sigma^2.
    _, singvals, Vh = torch.linalg.svd(J_star, full_matrices=False)
    V = Vh.mT
    lik_prec = (singvals ** 2) / (lik_model.sigma ** 2)
    post_lam = 1.0 + lik_prec

    stiff_mask = post_lam > stiff_lambda_cut
    residual_mask = ~stiff_mask

    VS = V[:, stiff_mask]
    VG = V[:, residual_mask]
    stiff_lik_prec = lik_prec[stiff_mask]
    residual_lik_prec = lik_prec[residual_mask]

    if stiff_lik_prec.numel() > 0:
        stiff_update = VS @ torch.diag(stiff_lik_prec) @ VS.T
    else:
        stiff_update = torch.zeros((d, d), device=device, dtype=torch.float64)

    P_tilde0 = I + stiff_update
    mu_tilde0 = torch.linalg.solve(P_tilde0, stiff_update @ x_star)

    print(f"  expansion index={best_idx}, pilot max logL={log_lik_pilot[best_idx].item():.4f}")
    print(f"  ||x_anchor||={torch.norm(x_anchor).item():.4f}, ||x_star||={torch.norm(x_star).item():.4f}")
    print(f"  stiff absorbed directions: {int(stiff_mask.sum().item())}/{post_lam.numel()}")

    return {
        'x_anchor': x_anchor.detach().clone(),
        'x_star': x_star.detach().clone(),
        'J_star': J_star.detach().clone(),
        'singvals': singvals.detach().clone(),
        'V': V.detach().clone(),
        'lik_prec': lik_prec.detach().clone(),
        'post_lam': post_lam.detach().clone(),
        'stiff_mask': stiff_mask.detach().clone(),
        'residual_mask': residual_mask.detach().clone(),
        'VS': VS.detach().clone(),
        'VG': VG.detach().clone(),
        'stiff_lik_prec': stiff_lik_prec.detach().clone(),
        'residual_lik_prec': residual_lik_prec.detach().clone(),
        'P_tilde0': P_tilde0.detach().clone(),
        'mu_tilde0': mu_tilde0.detach().clone(),
        'stiff_lambda_cut': float(stiff_lambda_cut),
    }


def eval_gnl_residual_loglik(x, gnl_info):
    """Residual GN log-likelihood ℓ~(x) = log L_G(x) + log L_F(x), up to an additive constant."""
    VG = gnl_info['VG']
    residual_lik_prec = gnl_info['residual_lik_prec']
    x_star = gnl_info['x_star']
    if residual_lik_prec.numel() == 0:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    centered = x - x_star.unsqueeze(0)
    proj = centered @ VG
    return -0.5 * torch.sum((proj ** 2) * residual_lik_prec.unsqueeze(0), dim=1)


POU_QUERY_BLOCK = 128
POU_REF_BLOCK = 512


def _compute_log_pou_overlap_penalty(X_ref, mu_ref, eigvecs, prec_eig, trusted,
                                     label='base', query_block=POU_QUERY_BLOCK,
                                     ref_block=POU_REF_BLOCK):
    """
    Practical PoU correction for the WC masses:

        log m_i^{PoU} pprox log 	ilde m_i - log \sum_j H(mu_i; j),

    where H(\cdot; j) is the local Gaussian window centered at x_j with the same
    band-gated precision used by the HLSI geometry. We only change the global
    mixing weights here; the particle-wise score signal remains whatever the
    sampler config selects (HLSI, CE-HLSI, etc.).
    """
    n_ref = X_ref.shape[0]
    work_device = X_ref.device
    work_dtype = X_ref.dtype

    eig_for_logdet = torch.where(trusted, torch.clamp(prec_eig, min=1e-30), torch.ones_like(prec_eig))
    log_window_norm = 0.5 * torch.sum(torch.log(eig_for_logdet), dim=1)

    print(f"  [{label}] Computing PoU overlap penalties with query_block={query_block}, ref_block={ref_block}...")
    t0_pou = time.time()

    log_overlap = torch.empty((n_ref,), device=work_device, dtype=work_dtype)
    n_query_blocks = (n_ref + query_block - 1) // query_block

    for qb, q0 in enumerate(range(0, n_ref, query_block)):
        q1 = min(q0 + query_block, n_ref)
        mu_block = mu_ref[q0:q1]
        block_max = torch.full((q1 - q0,), -float('inf'), device=work_device, dtype=work_dtype)
        block_sumexp = torch.zeros((q1 - q0,), device=work_device, dtype=work_dtype)

        for r0 in range(0, n_ref, ref_block):
            r1 = min(r0 + ref_block, n_ref)
            X_block = X_ref[r0:r1]
            V_block = eigvecs[r0:r1]
            lam_block = prec_eig[r0:r1]
            norm_block = log_window_norm[r0:r1]

            diff = mu_block.unsqueeze(1) - X_block.unsqueeze(0)
            proj = torch.einsum('qrd,rdk->qrk', diff, V_block)
            mahal = torch.sum((proj ** 2) * lam_block.unsqueeze(0), dim=2)
            log_H = norm_block.unsqueeze(0) - 0.5 * mahal

            local_max = torch.max(log_H, dim=1).values
            new_max = torch.maximum(block_max, local_max)
            block_sumexp = (
                block_sumexp * torch.exp(block_max - new_max)
                + torch.sum(torch.exp(log_H - new_max.unsqueeze(1)), dim=1)
            )
            block_max = new_max

            del X_block, V_block, lam_block, norm_block, diff, proj, mahal, log_H, local_max, new_max

        log_overlap[q0:q1] = block_max + torch.log(torch.clamp(block_sumexp, min=1e-300))

        if (qb + 1) % max(1, n_query_blocks // 10) == 0 or q1 == n_ref:
            print(f"    [{label}] PoU overlap block {qb + 1}/{n_query_blocks} complete")

        del mu_block, block_max, block_sumexp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  [{label}] PoU overlap time: {time.time() - t0_pou:.2f}s")
    return log_overlap

def _compute_log_pou_denom_grad_hess_at_points(points, X_ref, eigvecs, prec_eig, trusted,
                                              label='base', query_block=POU_QUERY_BLOCK,
                                              ref_block=POU_REF_BLOCK):
    """
    Evaluate the undiffused PoU denominator

        H̄(x) = sum_j H_j(x)

    together with its gradient and Hessian at a batch of query points.
    This is the Stage-A PoU object needed to build the PoU-adjusted local bank
        s_{i,0}^{PoU}, P_i^{PoU}, mu_i^{PoU}, m_i^{PoU}.
    """
    n_query, d = points.shape
    work_device = points.device
    work_dtype = points.dtype

    eig_for_logdet = torch.where(
        trusted, torch.clamp(prec_eig, min=1e-30), torch.ones_like(prec_eig)
    )
    log_window_norm = 0.5 * torch.sum(torch.log(eig_for_logdet), dim=1)

    print(f"  [{label}] Computing PoU denominator grad/Hess at query_block={query_block}, ref_block={ref_block}...")
    t0_pou = time.time()

    log_denom = torch.empty((n_query,), device=work_device, dtype=work_dtype)
    grad_log_denom = torch.empty((n_query, d), device=work_device, dtype=work_dtype)
    hess_log_denom = torch.empty((n_query, d, d), device=work_device, dtype=work_dtype)
    n_query_blocks = (n_query + query_block - 1) // query_block

    for qb, q0 in enumerate(range(0, n_query, query_block)):
        q1 = min(q0 + query_block, n_query)
        x_block = points[q0:q1]
        qsz = q1 - q0
        block_max = torch.full((qsz,), -float('inf'), device=work_device, dtype=work_dtype)
        block_sumexp = torch.zeros((qsz,), device=work_device, dtype=work_dtype)
        block_grad_num = torch.zeros((qsz, d), device=work_device, dtype=work_dtype)
        block_second_num = torch.zeros((qsz, d, d), device=work_device, dtype=work_dtype)

        for r0 in range(0, X_ref.shape[0], ref_block):
            r1 = min(r0 + ref_block, X_ref.shape[0])
            X_block = X_ref[r0:r1]
            V_block = eigvecs[r0:r1]
            lam_block = prec_eig[r0:r1]
            norm_block = log_window_norm[r0:r1]

            diff = x_block.unsqueeze(1) - X_block.unsqueeze(0)                       # [q, r, d]
            proj = torch.einsum('qrd,rdk->qrk', diff, V_block)                       # [q, r, d]
            mahal = torch.sum((proj ** 2) * lam_block.unsqueeze(0), dim=2)           # [q, r]
            log_H = norm_block.unsqueeze(0) - 0.5 * mahal                            # [q, r]

            scaled_proj = proj * lam_block.unsqueeze(0)
            grad_log_H = -torch.einsum('qrk,rdk->qrd', scaled_proj, V_block)         # [q, r, d]

            P_block = torch.einsum('rij,rj,rkj->rik', V_block, lam_block, V_block)   # [r, d, d]
            hess_log_H = -P_block                                                    # [r, d, d]
            second_term = hess_log_H.unsqueeze(0) + torch.einsum('qri,qrj->qrij', grad_log_H, grad_log_H)

            local_max = torch.max(log_H, dim=1).values
            new_max = torch.maximum(block_max, local_max)
            rescale_old = torch.exp(block_max - new_max)
            weights = torch.exp(log_H - new_max.unsqueeze(1))

            block_sumexp = rescale_old * block_sumexp + torch.sum(weights, dim=1)
            block_grad_num = rescale_old.unsqueeze(1) * block_grad_num + torch.einsum(
                'qr,qrd->qd', weights, grad_log_H
            )
            block_second_num = rescale_old.unsqueeze(-1).unsqueeze(-1) * block_second_num + torch.einsum(
                'qr,qrij->qij', weights, second_term
            )
            block_max = new_max

            del X_block, V_block, lam_block, norm_block
            del diff, proj, mahal, log_H, scaled_proj, grad_log_H, P_block, hess_log_H, second_term
            del local_max, new_max, rescale_old, weights

        block_sumexp = torch.clamp(block_sumexp, min=1e-300)
        grad = block_grad_num / block_sumexp.unsqueeze(1)
        second = block_second_num / block_sumexp.unsqueeze(-1).unsqueeze(-1)

        log_denom[q0:q1] = block_max + torch.log(block_sumexp)
        grad_log_denom[q0:q1] = grad
        hess_log_denom[q0:q1] = second - torch.einsum('qi,qj->qij', grad, grad)

        if (qb + 1) % max(1, n_query_blocks // 10) == 0 or q1 == n_query:
            print(f"    [{label}] PoU denominator grad/Hess block {qb + 1}/{n_query_blocks} complete")

        del x_block, block_max, block_sumexp, block_grad_num, block_second_num
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  [{label}] PoU denominator grad/Hess time: {time.time() - t0_pou:.2f}s")
    return log_denom, grad_log_denom, hess_log_denom


def _compute_log_pou_denom_and_grad_at_points(points, X_ref, eigvecs, prec_eig, trusted,
                                              label='base', query_block=POU_QUERY_BLOCK,
                                              ref_block=POU_REF_BLOCK):
    log_denom, grad_log_denom, _ = _compute_log_pou_denom_grad_hess_at_points(
        points, X_ref, eigvecs, prec_eig, trusted,
        label=label, query_block=query_block, ref_block=ref_block,
    )
    return log_denom, grad_log_denom


def _build_leaf_repaired_local_bank(X_ref, s_local, P_raw_local, log_base_mass, I,
                                    label='base', bank_name='base'):
    """
    Build a PSD leaf-repaired local bank from a potentially indefinite local
    precision field. Negative eigendirections are assigned adaptive positive
    leaf precisions based on their curvature magnitude,

        p_{i,k} = clip(|lambda_{i,k}|, LEAF_MIN_PREC, LEAF_MAX_PREC),

    which is the simplest curvature-adaptive two-leaf-inspired repair. The
    resulting PSD precision is then used in the same downstream HLSI / CE-HLSI
    machinery so we can test whether leaf rescue changes PoU behavior.
    """
    eigvals_leaf_raw, eigvecs_leaf = torch.linalg.eigh(P_raw_local)
    neg_mask = eigvals_leaf_raw < -HESS_MIN
    trusted_pos = (eigvals_leaf_raw >= HESS_MIN) & (eigvals_leaf_raw <= HESS_MAX)
    trusted_leaf = trusted_pos | neg_mask

    leaf_neg_prec = torch.clamp(
        LEAF_ABS_SCALE * torch.abs(eigvals_leaf_raw),
        min=LEAF_MIN_PREC,
        max=LEAF_MAX_PREC,
    )
    prec_eig_leaf = torch.where(
        neg_mask,
        leaf_neg_prec,
        torch.where(trusted_pos, eigvals_leaf_raw, torch.zeros_like(eigvals_leaf_raw)),
    )

    P_leaf_ref = torch.einsum('nij,nj,nkj->nik', eigvecs_leaf, prec_eig_leaf, eigvecs_leaf)
    P_leaf_ref = P_leaf_ref + 1e-6 * I

    s_local_eig = torch.einsum('nij,nj->ni', eigvecs_leaf.transpose(-1, -2), s_local)
    delta_leaf_eig = torch.where(
        trusted_leaf,
        s_local_eig / torch.clamp(prec_eig_leaf, min=LEAF_MIN_PREC),
        torch.zeros_like(s_local_eig),
    )
    delta_leaf = torch.einsum('nij,nj->ni', eigvecs_leaf, delta_leaf_eig)
    mu_leaf_ref = X_ref + delta_leaf

    # Two-leaf moment-matching metadata for future diagnostics / extensions.
    leaf_offset_sq = torch.where(
        neg_mask,
        torch.clamp((leaf_neg_prec - eigvals_leaf_raw) / torch.clamp(leaf_neg_prec ** 2, min=1e-30), min=0.0),
        torch.zeros_like(eigvals_leaf_raw),
    )
    leaf_offset = torch.sqrt(leaf_offset_sq)

    quad_gain_leaf = 0.5 * torch.sum(s_local * delta_leaf, dim=1)
    eig_for_logdet_leaf = torch.where(trusted_leaf, prec_eig_leaf, torch.ones_like(prec_eig_leaf))
    logdet_P_leaf = torch.sum(torch.log(torch.clamp(eig_for_logdet_leaf, min=1e-30)), dim=1)
    log_mass_leaf = (log_base_mass + quad_gain_leaf) - 0.5 * logdet_P_leaf

    mean_neg = neg_mask.float().sum(dim=1).mean().item()
    mean_leaf = trusted_leaf.float().sum(dim=1).mean().item()
    print(f"    [{label}:{bank_name}] leaf rescue mean neg dirs={mean_neg:.2f}, active dirs={mean_leaf:.2f}")

    return {
        'P_ref': P_leaf_ref,
        'mu_ref': mu_leaf_ref,
        'gated_info': {
            'eigvecs': eigvecs_leaf,
            'eigvals': prec_eig_leaf,
            'trusted': trusted_leaf,
        },
        'log_mass_ref': log_mass_leaf,
        'leaf_neg_mask': neg_mask,
        'leaf_prec_eig': prec_eig_leaf,
        'leaf_offset_eig': leaf_offset,
        'leaf_raw_eigvals': eigvals_leaf_raw,
    }

def precompute_reference_bank(X_ref, prior_model, lik_model, label='base',
                              residual_log_weights=None, compute_pou=True,
                              pou_query_block=POU_QUERY_BLOCK,
                              pou_ref_block=POU_REF_BLOCK):
    """
    Precompute the local posterior HLSI objects on a fixed reference bank.
    In Stage-A PoU mode we also build a PoU-adjusted local bank with
        s_{i,0}^{PoU}, P_i^{PoU}, mu_i^{PoU}, m_i^{PoU},
    so PoU-HLSI / CE-PoU-HLSI use PoU-adjusted local geometry rather than
    the old base bank plus a query-time denominator subtraction.
    """
    print(f"Precomputing {label} reference bank with {X_ref.shape[0]} particles...")
    t0_bank = time.time()

    with torch.no_grad():
        s0_prior = prior_model.score0(X_ref)
        log_prior = prior_model.log_prob(X_ref)

        BATCH_LIK = 50
        BATCH_HESS = 2

        print(f"  [{label}] Calculating Likelihoods / Grads (batched)...")
        log_lik_list = []
        grad_lik_list = []
        for i in range(0, X_ref.shape[0], BATCH_LIK):
            batch = X_ref[i:i + BATCH_LIK]
            log_lik_list.append(lik_model.log_likelihood(batch))
            grad_lik_list.append(lik_model.grad_log_likelihood(batch))
        log_lik = torch.cat(log_lik_list, dim=0)
        grad_lik = torch.cat(grad_lik_list, dim=0)

        print(f"  [{label}] Calculating Hessians (batched)...")
        hess_lik_list = []
        for i in range(0, X_ref.shape[0], BATCH_HESS):
            if i % (BATCH_HESS * 20) == 0:
                print(f"    [{label}] Hessian batch {i}/{X_ref.shape[0]}...")
            batch = X_ref[i:i + BATCH_HESS]
            hess_lik_list.append(lik_model.hess_log_likelihood(batch))
        hess_lik = torch.cat(hess_lik_list, dim=0)

        s0_post = s0_prior + grad_lik

        d = X_ref.shape[1]
        I = torch.eye(d, device=device, dtype=torch.float64).unsqueeze(0)
        P_raw = I - hess_lik
        P_raw = 0.5 * (P_raw + P_raw.transpose(-1, -2))

        print(f"  [{label}] Spectral band gating: [{HESS_MIN:.1e}, {HESS_MAX:.1e}]")
        eigvals, eigvecs = torch.linalg.eigh(P_raw)
        trusted = (eigvals >= HESS_MIN) & (eigvals <= HESS_MAX)

        n_below = (eigvals < HESS_MIN).float().sum(dim=1).mean().item()
        n_above = (eigvals > HESS_MAX).float().sum(dim=1).mean().item()
        n_neg = (eigvals < -HESS_MIN).float().sum(dim=1).mean().item()
        n_band = trusted.float().sum(dim=1).mean().item()
        print(f"    [{label}] mean {n_band:.1f} in-band, {n_below:.1f} too-sloppy, {n_above:.1f} too-stiff, {n_neg:.1f} negative (of {d})")

        prec_eig = torch.where(trusted, eigvals, torch.zeros_like(eigvals))
        P_ref = torch.einsum('nij,nj,nkj->nik', eigvecs, prec_eig, eigvecs)
        P_ref = P_ref + 1e-6 * I

        s_eig = (eigvecs.mT @ s0_post.unsqueeze(-1)).squeeze(-1)
        delta_eig = torch.where(
            trusted,
            s_eig / torch.clamp(prec_eig, min=HESS_MIN),
            torch.zeros_like(s_eig),
        )
        delta = (eigvecs @ delta_eig.unsqueeze(-1)).squeeze(-1)
        mu_ref = X_ref + delta

        gated_info = {
            'eigvecs': eigvecs,
            'eigvals': prec_eig,
            'trusted': trusted,
        }

        log_post_x = log_prior + log_lik
        quad_gain = 0.5 * torch.sum(s0_post * delta, dim=1)
        eig_for_logdet = torch.where(trusted, eigvals, torch.ones_like(eigvals))
        logdet_P = torch.sum(torch.log(torch.clamp(eig_for_logdet, min=1e-30)), dim=1)
        log_mass = (log_post_x + quad_gain) - 0.5 * logdet_P

        leaf_base = _build_leaf_repaired_local_bank(
            X_ref, s0_post, P_raw, log_post_x, I, label=label, bank_name='base'
        )

        log_pou = None
        log_window_overlap = None
        log_pou_denom_ref = None
        grad_log_pou_denom_ref = None
        hess_log_pou_denom_ref = None
        s0_pou = None
        P_pou_ref = None
        mu_pou_ref = None
        gated_info_pou = None
        if compute_pou:
            log_window_norm = 0.5 * torch.sum(torch.log(torch.clamp(eig_for_logdet, min=1e-30)), dim=1)
            P_window_ref = torch.einsum('nij,nj,nkj->nik', eigvecs, prec_eig, eigvecs)

            log_pou_denom_ref, grad_log_pou_denom_ref, hess_log_pou_denom_ref = _compute_log_pou_denom_grad_hess_at_points(
                X_ref, X_ref, eigvecs, prec_eig, trusted,
                label=label, query_block=pou_query_block, ref_block=pou_ref_block,
            )

            s0_pou = s0_post - grad_log_pou_denom_ref
            # For q_i^*(x) = pi(x) H_i(x) / Hbar(x),
            #   -∇² log q_i^* = P_post + P_window + ∇² log Hbar.
            # The denominator Hessian therefore enters with a plus sign.
            P_pou_raw = P_raw + P_window_ref + hess_log_pou_denom_ref
            P_pou_raw = 0.5 * (P_pou_raw + P_pou_raw.transpose(-1, -2))

            eigvals_pou, eigvecs_pou = torch.linalg.eigh(P_pou_raw)
            trusted_pou = (eigvals_pou >= HESS_MIN) & (eigvals_pou <= HESS_MAX)
            prec_eig_pou = torch.where(trusted_pou, eigvals_pou, torch.zeros_like(eigvals_pou))
            P_pou_ref = torch.einsum('nij,nj,nkj->nik', eigvecs_pou, prec_eig_pou, eigvecs_pou) + 1e-6 * I

            s0_pou_eig = torch.einsum('nij,nj->ni', eigvecs_pou.transpose(-1, -2), s0_pou)
            delta_pou_eig = torch.where(
                trusted_pou,
                s0_pou_eig / torch.clamp(prec_eig_pou, min=HESS_MIN),
                torch.zeros_like(s0_pou_eig),
            )
            delta_pou = torch.einsum('nij,nj->ni', eigvecs_pou, delta_pou_eig)
            mu_pou_ref = X_ref + delta_pou

            gated_info_pou = {
                'eigvecs': eigvecs_pou,
                'eigvals': prec_eig_pou,
                'trusted': trusted_pou,
            }

            quad_gain_pou = 0.5 * torch.sum(s0_pou * delta_pou, dim=1)
            eig_for_logdet_pou = torch.where(trusted_pou, eigvals_pou, torch.ones_like(eigvals_pou))
            logdet_P_pou = torch.sum(torch.log(torch.clamp(eig_for_logdet_pou, min=1e-30)), dim=1)
            log_pou = (log_post_x + log_window_norm - log_pou_denom_ref + quad_gain_pou) - 0.5 * logdet_P_pou
            log_window_overlap = log_pou_denom_ref

            leaf_pou_base_mass = log_post_x + log_window_norm - log_pou_denom_ref
            leaf_pou = _build_leaf_repaired_local_bank(
                X_ref, s0_pou, P_pou_raw, leaf_pou_base_mass, I, label=label, bank_name='pou'
            )

        norm_prior = torch.norm(s0_prior, dim=1).mean().item()
        norm_lik = torch.norm(grad_lik, dim=1).mean().item()
        avg_ll = log_lik.mean().item()
        print(f"  [{label}] Avg Prior Score Norm:     {norm_prior:.4f}")
        print(f"  [{label}] Avg Likelihood Grad Norm: {norm_lik:.4f}")
        print(f"  [{label}] Avg Log-Likelihood:       {avg_ll:.4f}")

        bank = {
            'X_ref': X_ref.detach().cpu(),
            's0_post_ref': s0_post.detach().cpu(),
            'log_lik_ref': log_lik.detach().cpu(),
            'log_none_ref': torch.zeros_like(log_lik).detach().cpu(),
            'log_mass_ref': log_mass.detach().cpu(),
            'P_ref': P_ref.detach().cpu(),
            'mu_ref': mu_ref.detach().cpu(),
            'gated_info': {k: v.detach().cpu() for k, v in gated_info.items()},
            'P_leaf_ref': leaf_base['P_ref'].detach().cpu(),
            'mu_leaf_ref': leaf_base['mu_ref'].detach().cpu(),
            'log_mass_leaf_ref': leaf_base['log_mass_ref'].detach().cpu(),
            'gated_info_leaf': {k: v.detach().cpu() for k, v in leaf_base['gated_info'].items()},
            'leaf_neg_mask_ref': leaf_base['leaf_neg_mask'].detach().cpu(),
            'leaf_prec_eig_ref': leaf_base['leaf_prec_eig'].detach().cpu(),
            'leaf_offset_eig_ref': leaf_base['leaf_offset_eig'].detach().cpu(),
            'leaf_raw_eigvals_ref': leaf_base['leaf_raw_eigvals'].detach().cpu(),
        }
        if log_pou is not None:
            bank['s0_pou_ref'] = s0_pou.detach().cpu()
            bank['log_pou_ref'] = log_pou.detach().cpu()
            bank['log_window_overlap_ref'] = log_window_overlap.detach().cpu()
            bank['log_pou_denom_ref'] = log_pou_denom_ref.detach().cpu()
            bank['grad_log_pou_denom_ref'] = grad_log_pou_denom_ref.detach().cpu()
            bank['hess_log_pou_denom_ref'] = hess_log_pou_denom_ref.detach().cpu()
            bank['P_pou_ref'] = P_pou_ref.detach().cpu()
            bank['mu_pou_ref'] = mu_pou_ref.detach().cpu()
            bank['gated_info_pou'] = {k: v.detach().cpu() for k, v in gated_info_pou.items()}
            bank['P_pou_leaf_ref'] = leaf_pou['P_ref'].detach().cpu()
            bank['mu_pou_leaf_ref'] = leaf_pou['mu_ref'].detach().cpu()
            bank['log_pou_leaf_ref'] = leaf_pou['log_mass_ref'].detach().cpu()
            bank['gated_info_pou_leaf'] = {k: v.detach().cpu() for k, v in leaf_pou['gated_info'].items()}
            bank['leaf_pou_neg_mask_ref'] = leaf_pou['leaf_neg_mask'].detach().cpu()
            bank['leaf_pou_prec_eig_ref'] = leaf_pou['leaf_prec_eig'].detach().cpu()
            bank['leaf_pou_offset_eig_ref'] = leaf_pou['leaf_offset_eig'].detach().cpu()
            bank['leaf_pou_raw_eigvals_ref'] = leaf_pou['leaf_raw_eigvals'].detach().cpu()
        if residual_log_weights is not None:
            bank['log_lik_res_ref'] = residual_log_weights.detach().cpu()

    print(f"  [{label}] Precomputation time: {time.time() - t0_bank:.2f}s")

    del s0_prior, log_prior, log_lik, grad_lik, hess_lik, s0_post
    del P_raw, eigvals, eigvecs, trusted, prec_eig, delta, P_ref, mu_ref, gated_info
    del log_post_x, quad_gain, eig_for_logdet, logdet_P, I, log_mass
    if log_pou is not None:
        del log_pou, log_window_overlap, log_pou_denom_ref, grad_log_pou_denom_ref, hess_log_pou_denom_ref
        del s0_pou, P_pou_ref, mu_pou_ref, gated_info_pou
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return bank


# ==========================================
# 5. SAMPLERS (physics-agnostic — updated to stream reference chunks)
# ==========================================

REF_STREAM_BATCH = 512


def _canonicalize_time(t):
    t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
    return max(t_val, 1e-4)


def _streaming_max_logw(y, t, X_ref_cpu, log_lik_ref_cpu, batch_size=REF_STREAM_BATCH):
    return _streaming_max_logw_transition(
        y, t, X_ref_cpu, log_lik_ref_cpu,
        batch_size=batch_size, transition_w='ou')


def _surrogate_chunk_log_transition(y, t_val, P_chunk, mu_chunk, gated_chunk=None):
    """
    Log transition density, up to constants common across components, for

        X_0 | i ~ N(mu_i, P_i^{-1}),
        Y_t = exp(-t) X_0 + sqrt(1-exp(-2t)) Z.

    This gives Y_t | i ~ N(exp(-t) mu_i, exp(-2t) P_i^{-1} + v_t I).
    For gated/spectral HLSI banks, we reuse the same trusted-band convention
    as the diffused PoU-window code.
    """
    SigmaInv_chunk, b_chunk, log_norm_chunk = _get_pou_window_chunk_time_mats(
        t_val, P_chunk, mu_chunk, gated_chunk=gated_chunk)
    sigma_y = torch.einsum('bij,mj->mbi', SigmaInv_chunk, y)
    quad_sigma = torch.sum(sigma_y * y.unsqueeze(1), dim=2)
    cross = torch.einsum('md,bd->mb', y, b_chunk)
    et = math.exp(t_val * -1.0)
    mu_t = et * mu_chunk
    quad_mu = torch.sum(mu_t * b_chunk, dim=1).unsqueeze(0)
    return log_norm_chunk.unsqueeze(0) - 0.5 * (quad_sigma - 2.0 * cross + quad_mu)


def _streaming_max_logw_transition(y, t, X_ref_cpu, log_lik_ref_cpu,
                                   batch_size=REF_STREAM_BATCH,
                                   transition_w='ou', P_ref_cpu=None,
                                   mu_ref_cpu=None, gated_info=None):
    transition_w = canonicalize_transition_w(transition_w)
    t_val = _canonicalize_time(t)
    et = math.exp(-t_val)
    var_t = 1.0 - math.exp(-2.0 * t_val)

    m_query = y.shape[0]
    n_ref = X_ref_cpu.shape[0]
    max_log_w = torch.full((m_query,), -float('inf'), device=y.device, dtype=y.dtype)
    if transition_w == 'surrogate' and mu_ref_cpu is None:
        raise ValueError("transition_w='surrogate' requires mu_ref_cpu and P_ref_cpu or gated_info.")

    for i in range(0, n_ref, batch_size):
        sl = slice(i, i + batch_size)
        ll_batch = log_lik_ref_cpu[sl].to(y.device, non_blocking=True)
        if transition_w == 'ou':
            X_batch = X_ref_cpu[sl].to(y.device, non_blocking=True)
            mus = et * X_batch
            diff = y.unsqueeze(1) - mus.unsqueeze(0)
            dists_sq = torch.sum(diff * diff, dim=2)
            log_trans = -dists_sq / (2.0 * var_t)
            del X_batch, mus, diff, dists_sq
        else:
            mu_chunk = mu_ref_cpu[sl].to(y.device, non_blocking=True)
            if gated_info is not None:
                gated_chunk = {
                    'eigvecs': gated_info['eigvecs'][sl].to(y.device, non_blocking=True),
                    'eigvals': gated_info['eigvals'][sl].to(y.device, non_blocking=True),
                    'trusted': gated_info['trusted'][sl].to(y.device, non_blocking=True),
                }
                P_chunk = None
            else:
                if P_ref_cpu is None:
                    raise ValueError("transition_w='surrogate' requires P_ref_cpu when gated_info is not supplied.")
                gated_chunk = None
                P_chunk = P_ref_cpu[sl].to(y.device, non_blocking=True)
            log_trans = _surrogate_chunk_log_transition(
                y, t_val, P_chunk, mu_chunk, gated_chunk=gated_chunk)
            del mu_chunk
            if gated_chunk is not None:
                del gated_chunk
            if P_chunk is not None:
                del P_chunk
        log_w_batch = log_trans + ll_batch.unsqueeze(0)
        max_log_w = torch.maximum(max_log_w, torch.max(log_w_batch, dim=1).values)
        del ll_batch, log_trans, log_w_batch
    return t_val, et, var_t, max_log_w

def get_posterior_snis_weights(y, t, X_ref_cpu, log_lik_ref_cpu,
                               batch_size=REF_STREAM_BATCH,
                               transition_w='ou', P_ref_cpu=None,
                               mu_ref_cpu=None, gated_info=None):
    """
    Memory-safe two-pass SNIS weights. ``transition_w`` controls only the
    transition factor in the component responsibilities:

      - ``ou``:        K_i(y,t) = p_OU(y | x_i)
      - ``surrogate``: K_i(y,t) = p(y | X_0 ~ N(mu_i, P_i^{-1})) under OU

    The static weights in ``log_lik_ref_cpu`` remain orthogonal to this choice;
    they can be likelihood, WC mass, PoU mass, leaf-WC mass, etc.
    """
    transition_w = canonicalize_transition_w(transition_w)
    t_val, et, var_t, max_log_w = _streaming_max_logw_transition(
        y, t, X_ref_cpu, log_lik_ref_cpu, batch_size=batch_size,
        transition_w=transition_w, P_ref_cpu=P_ref_cpu,
        mu_ref_cpu=mu_ref_cpu, gated_info=gated_info)

    chunks = []
    denom_Z = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)

    for i in range(0, X_ref_cpu.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        ll_batch = log_lik_ref_cpu[sl].to(y.device, non_blocking=True)
        if transition_w == 'ou':
            X_batch = X_ref_cpu[sl].to(y.device, non_blocking=True)
            mus = et * X_batch
            diff = y.unsqueeze(1) - mus.unsqueeze(0)
            dists_sq = torch.sum(diff * diff, dim=2)
            log_trans = -dists_sq / (2.0 * var_t)
            del X_batch, mus, diff, dists_sq
        else:
            mu_chunk = mu_ref_cpu[sl].to(y.device, non_blocking=True)
            if gated_info is not None:
                gated_chunk = {
                    'eigvecs': gated_info['eigvecs'][sl].to(y.device, non_blocking=True),
                    'eigvals': gated_info['eigvals'][sl].to(y.device, non_blocking=True),
                    'trusted': gated_info['trusted'][sl].to(y.device, non_blocking=True),
                }
                P_chunk = None
            else:
                if P_ref_cpu is None:
                    raise ValueError("transition_w='surrogate' requires P_ref_cpu when gated_info is not supplied.")
                gated_chunk = None
                P_chunk = P_ref_cpu[sl].to(y.device, non_blocking=True)
            log_trans = _surrogate_chunk_log_transition(
                y, t_val, P_chunk, mu_chunk, gated_chunk=gated_chunk)
            del mu_chunk
            if gated_chunk is not None:
                del gated_chunk
            if P_chunk is not None:
                del P_chunk
        log_w_batch = log_trans + ll_batch.unsqueeze(0)
        w_batch = torch.exp(log_w_batch - max_log_w.unsqueeze(1))
        denom_Z += torch.sum(w_batch, dim=1, keepdim=True)
        chunks.append(w_batch)
        del ll_batch, log_trans, log_w_batch

    w = torch.cat(chunks, dim=1)
    return w / torch.clamp(denom_Z, min=1e-30)

def eval_blend_posterior_score(y, t, X_ref_cpu, s0_post_ref_cpu, log_lik_ref_cpu,
                               batch_size=REF_STREAM_BATCH):
    return eval_score_batched(
        y, t, X_ref_cpu, s0_post_ref_cpu, log_lik_ref_cpu,
        batch_size=batch_size, mode='blend_posterior')


def eval_score_batched(y, t, X_ref_cpu, s0_ref_cpu, log_lik_ref_cpu,
                       batch_size=REF_STREAM_BATCH, mode='blend_posterior'):
    """
    Streaming batched Tweedie / Blend Posterior score to avoid OOM.
    X_ref_cpu / s0_ref_cpu / log_lik_ref_cpu are expected to live on CPU.
    """
    t_val = _canonicalize_time(t)

    et = math.exp(-t_val)
    var_t = 1.0 - math.exp(-2 * t_val)
    inv_v = 1.0 / var_t
    scale_factor = 1.0 / et

    m_query = y.shape[0]
    n_ref = X_ref_cpu.shape[0]

    # --- PASS 1: Find Global Max Log-Weight ---
    max_log_w = torch.full((m_query,), -float('inf'), device=y.device, dtype=y.dtype)

    for i in range(0, n_ref, batch_size):
        X_batch = X_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
        ll_batch = log_lik_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
        mus = et * X_batch
        dists_sq = torch.sum((y.unsqueeze(1) - mus.unsqueeze(0)) ** 2, dim=2)
        log_w_batch = -dists_sq / (2 * var_t) + ll_batch.unsqueeze(0)
        current_max = torch.max(log_w_batch, dim=1).values
        max_log_w = torch.maximum(max_log_w, current_max)
        del X_batch, ll_batch, mus, dists_sq, log_w_batch, current_max

    # --- PASS 2: Accumulate Moments ---
    denom_Z = torch.zeros((m_query, 1), device=y.device, dtype=y.dtype)
    numer_mu_x = torch.zeros_like(y)

    if mode == 'blend_posterior':
        numer_s0 = torch.zeros_like(y)
        acc_w2 = torch.zeros((m_query, 1), device=y.device, dtype=y.dtype)
        acc_w2_s0_norm = torch.zeros((m_query,), device=y.device, dtype=y.dtype)
        acc_w2_s0 = torch.zeros_like(y)
        acc_w2_x = torch.zeros_like(y)
        acc_w2_x_norm = torch.zeros((m_query,), device=y.device, dtype=y.dtype)
        acc_w2_dot = torch.zeros((m_query,), device=y.device, dtype=y.dtype)

    for i in range(0, n_ref, batch_size):
        X_batch = X_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
        ll_batch = log_lik_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
        mus = et * X_batch
        dists_sq = torch.sum((y.unsqueeze(1) - mus.unsqueeze(0)) ** 2, dim=2)
        log_w = -dists_sq / (2 * var_t) + ll_batch.unsqueeze(0)
        w_batch = torch.exp(log_w - max_log_w.unsqueeze(1))

        denom_Z += torch.sum(w_batch, dim=1, keepdim=True)
        numer_mu_x += torch.einsum('mb,bd->md', w_batch, X_batch)

        if mode == 'blend_posterior':
            s0_batch = s0_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
            numer_s0 += torch.einsum('mb,bd->md', w_batch, s0_batch)

            w2_batch = w_batch ** 2
            acc_w2 += torch.sum(w2_batch, dim=1, keepdim=True)
            s0_sq_batch = torch.sum(s0_batch ** 2, dim=1)
            x_sq_batch = torch.sum(X_batch ** 2, dim=1)
            dot_batch = torch.sum(s0_batch * X_batch, dim=1)

            acc_w2_s0_norm += torch.mv(w2_batch, s0_sq_batch)
            acc_w2_s0 += torch.mm(w2_batch, s0_batch)
            acc_w2_x += torch.mm(w2_batch, X_batch)
            acc_w2_x_norm += torch.mv(w2_batch, x_sq_batch)
            acc_w2_dot += torch.mv(w2_batch, dot_batch)
            del s0_batch, w2_batch, s0_sq_batch, x_sq_batch, dot_batch

        del X_batch, ll_batch, mus, dists_sq, log_w, w_batch

    # --- FINALIZE ---
    eps = 1e-12
    mu_x = numer_mu_x / denom_Z
    score_twd = -inv_v * (y - et * mu_x)

    if mode == 'tweedie':
        return score_twd

    Z_sq = denom_Z ** 2
    mu_a = scale_factor * (numer_s0 / denom_Z)
    mu_b = score_twd
    S0 = acc_w2 / Z_sq

    S1a = (scale_factor ** 2) * (acc_w2_s0_norm.unsqueeze(1) / Z_sq)
    S2a = scale_factor * (acc_w2_s0 / Z_sq)

    den_sn = torch.clamp(1.0 - S0, min=eps)
    mu_a_norm2 = torch.sum(mu_a ** 2, dim=1, keepdim=True)
    num_Vk = S1a - 2.0 * torch.sum(mu_a * S2a, dim=1, keepdim=True) + mu_a_norm2 * S0
    Vk = num_Vk / den_sn

    term_w2_x = acc_w2_x / Z_sq
    S2b = -inv_v * (y * S0 - et * term_w2_x)
    y_norm2 = torch.sum(y ** 2, dim=1, keepdim=True)
    y_dot_w2x = torch.sum(y * term_w2_x, dim=1, keepdim=True)
    term_w2_x_norm = acc_w2_x_norm.unsqueeze(1) / Z_sq
    S1b = (inv_v ** 2) * (y_norm2 * S0 - 2.0 * et * y_dot_w2x + (et ** 2) * term_w2_x_norm)
    mu_b_norm2 = torch.sum(mu_b ** 2, dim=1, keepdim=True)
    num_Vt = S1b - 2.0 * torch.sum(mu_b * S2b, dim=1, keepdim=True) + mu_b_norm2 * S0
    Vt = num_Vt / den_sn

    term_w2_dot = acc_w2_dot.unsqueeze(1) / Z_sq
    term_c2 = scale_factor * term_w2_dot
    term_c1 = torch.sum(S2a * y, dim=1, keepdim=True)
    Sab = -inv_v * (term_c1 - et * term_c2)
    num_C = (Sab
             - torch.sum(mu_a * S2b, dim=1, keepdim=True)
             - torch.sum(mu_b * S2a, dim=1, keepdim=True)
             + torch.sum(mu_a * mu_b, dim=1, keepdim=True) * S0)
    C = num_C / den_sn

    denom = torch.clamp(Vk + Vt - 2.0 * C, min=eps)
    lam = (Vk - C) / denom
    lam = torch.clamp(lam, 0.0, 0.95)

    return lam * score_twd + (1.0 - lam) * mu_a


def _get_ce_hlsi_gate_eigenbasis(P_bar, et2, var_t):
    """
    Given the SNIS-averaged posterior Hessian P_bar [M, d, d],
    returns the gate eigenvalues and eigenvectors for the
    certainty-equivalent CE-HLSI gate

        A_bar = e^{-2t} (e^{-2t} I + v_t P_bar)^{-1}.

    In the eigenbasis of P_bar with eigenvalues lam_k, the gate
    eigenvalues are

        A_k = e^{-2t} / (e^{-2t} + v_t * lam_k).
    """
    lam, V = torch.linalg.eigh(P_bar)
    lam = lam.clamp(min=1e-6)
    gate_eig = et2 / (et2 + var_t * lam)
    return gate_eig, V


@dataclass
class HLSIComponentState:
    y: torch.Tensor
    t_val: float
    et: float
    et2: float
    var_t: float
    alpha: torch.Tensor
    alpha_top: torch.Tensor
    alpha_top_norm: torch.Tensor
    idx_top: torch.Tensor
    s_twd: torch.Tensor
    d_top: torch.Tensor
    metric_points_top: torch.Tensor
    metric_prec_top: torch.Tensor
    gate_top: torch.Tensor


class BaseHLSIGateLaw:
    family = 'base'

    def fast_path_kind(self):
        return None

    def apply(self, component_state: HLSIComponentState):
        raise NotImplementedError


class DiracGateLaw(BaseHLSIGateLaw):
    family = 'dirac'

    def fast_path_kind(self):
        return 'dirac'


class PosteriorAverageGateLaw(BaseHLSIGateLaw):
    family = 'posterior_average'

    def fast_path_kind(self):
        return 'posterior_average'


class TemperedLikelihoodGateLaw(BaseHLSIGateLaw):
    family = 'tempered_likelihood'

    def __init__(self, rho=0.5, beta=1.0, kappa=1.0, topk=64,
                 metric_source='mu', eps=1e-12):
        self.rho = float(rho)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.topk = int(max(1, topk))
        self.metric_source = str(metric_source).strip().lower()
        if self.metric_source not in {'mu', 'x'}:
            raise ValueError(f"Unsupported gate metric_source: {metric_source}")
        self.eps = float(eps)

    def fast_path_kind(self):
        if self.rho <= self.eps:
            return 'dirac'
        if abs(self.rho - 1.0) <= self.eps and abs(self.beta - 1.0) <= self.eps and abs(self.kappa) <= self.eps:
            return 'posterior_average'
        return None

    def _anchor_donor_log_compatibility(self, component_state: HLSIComponentState):
        diff = component_state.metric_points_top.unsqueeze(1) - component_state.metric_points_top.unsqueeze(2)
        mahal = torch.einsum('mijd,mide,mije->mij', diff, component_state.metric_prec_top, diff)
        return -0.5 * mahal

    def apply(self, component_state: HLSIComponentState):
        fast_path = self.fast_path_kind()
        if fast_path is not None:
            raise RuntimeError('TemperedLikelihoodGateLaw.apply was called even though an exact fast path is available.')

        donor_logits = self.beta * torch.log(torch.clamp(component_state.alpha_top_norm, min=self.eps)).unsqueeze(1)
        if abs(self.kappa) > self.eps:
            donor_logits = donor_logits + self.kappa * self._anchor_donor_log_compatibility(component_state)

        q_soft = torch.softmax(donor_logits, dim=2)
        if self.rho < 1.0 - self.eps:
            eye = torch.eye(q_soft.shape[1], device=q_soft.device, dtype=q_soft.dtype).unsqueeze(0)
            q = (1.0 - self.rho) * eye + self.rho * q_soft
        else:
            q = q_soft

        A_tilde = torch.einsum('mij,mjab->miab', q, component_state.gate_top)
        corr_top = torch.einsum('miab,mib->mia', A_tilde, component_state.d_top)
        return component_state.s_twd + torch.einsum('mi,mia->ma', component_state.alpha_top, corr_top)


def _materialize_precision_from_gated(gated_chunk):
    eigvecs = gated_chunk['eigvecs']
    eigvals = torch.clamp(gated_chunk['eigvals'], min=0.0)
    trusted = gated_chunk['trusted']
    active_eig = torch.where(trusted, eigvals, torch.zeros_like(eigvals))
    return torch.einsum('...ij,...j,...kj->...ik', eigvecs, active_eig, eigvecs)


def _materialize_hlsi_gate_from_gated(gated_chunk, et2, var_t):
    eigvecs = gated_chunk['eigvecs']
    eigvals = torch.clamp(gated_chunk['eigvals'], min=0.0)
    trusted = gated_chunk['trusted']
    gate_eig = torch.where(
        trusted,
        et2 / torch.clamp(et2 + var_t * eigvals, min=1e-30),
        torch.zeros_like(eigvals),
    )
    return torch.einsum('...ij,...j,...kj->...ik', eigvecs, gate_eig, eigvecs)


def _materialize_hlsi_gate_from_precision(P_chunk, et2, var_t):
    lam, V = torch.linalg.eigh(0.5 * (P_chunk + P_chunk.transpose(-1, -2)))
    lam = torch.clamp(lam, min=0.0)
    gate_eig = et2 / torch.clamp(et2 + var_t * lam, min=1e-30)
    return torch.einsum('...ij,...j,...kj->...ik', V, gate_eig, V)


def _gather_rows_cpu_to_device(source_cpu, row_idx, device_target):
    row_idx = row_idx.reshape(-1).detach().cpu()
    gathered = source_cpu.index_select(0, row_idx)
    new_shape = tuple(row_idx.shape[:-1])
    return gathered


def _gather_topk_cpu_to_device(source_cpu, idx_top, device_target):
    idx_flat = idx_top.reshape(-1).detach().cpu()
    gathered = source_cpu.index_select(0, idx_flat)
    new_shape = tuple(idx_top.shape) + tuple(source_cpu.shape[1:])
    return gathered.view(*new_shape).to(device_target, non_blocking=True)


def _build_hlsi_component_state(y, t, X_ref_cpu, s0_post_ref_cpu, log_lik_ref_cpu,
                                P_ref_cpu=None, mu_ref_cpu=None, gated_info=None,
                                gate_law=None, batch_size=REF_STREAM_BATCH,
                                transition_w='ou'):
    if gate_law is None:
        raise ValueError('_build_hlsi_component_state requires a gate_law.')
    if mu_ref_cpu is None:
        raise ValueError('_build_hlsi_component_state requires mu_ref_cpu.')

    t_val = _canonicalize_time(t)
    et = math.exp(-t_val)
    et2 = et * et
    var_t = 1.0 - math.exp(-2.0 * t_val)

    alpha = get_posterior_snis_weights(
        y, t_val, X_ref_cpu, log_lik_ref_cpu, batch_size=batch_size,
        transition_w=transition_w, P_ref_cpu=P_ref_cpu,
        mu_ref_cpu=mu_ref_cpu, gated_info=gated_info)
    m_query, n_ref = alpha.shape
    topk = min(max(1, int(getattr(gate_law, 'topk', 64))), n_ref)
    alpha_top, idx_top = torch.topk(alpha, k=topk, dim=1, largest=True, sorted=False)
    alpha_top_norm = alpha_top / torch.clamp(alpha_top.sum(dim=1, keepdim=True), min=1e-30)

    d = y.shape[1]
    mu_x = torch.zeros((m_query, d), device=y.device, dtype=y.dtype)
    s_tsi_num = torch.zeros((m_query, d), device=y.device, dtype=y.dtype)
    for i in range(0, n_ref, batch_size):
        sl = slice(i, i + batch_size)
        w_batch = alpha[:, sl]
        X_batch = X_ref_cpu[sl].to(y.device, non_blocking=True)
        s0_batch = s0_post_ref_cpu[sl].to(y.device, non_blocking=True)
        mu_x += torch.einsum('mb,bd->md', w_batch, X_batch)
        s_tsi_num += torch.einsum('mb,bd->md', w_batch, s0_batch)
        del w_batch, X_batch, s0_batch

    s_twd = -(1.0 / var_t) * (y - et * mu_x)
    s0_top = _gather_topk_cpu_to_device(s0_post_ref_cpu, idx_top, y.device)
    d_top = s0_top / et - s_twd.unsqueeze(1)

    metric_source = getattr(gate_law, 'metric_source', 'mu')
    if metric_source == 'x':
        metric_points_top = _gather_topk_cpu_to_device(X_ref_cpu, idx_top, y.device)
    else:
        metric_points_top = _gather_topk_cpu_to_device(mu_ref_cpu, idx_top, y.device)

    if gated_info is not None:
        gated_top = {
            'eigvecs': _gather_topk_cpu_to_device(gated_info['eigvecs'], idx_top, y.device),
            'eigvals': _gather_topk_cpu_to_device(gated_info['eigvals'], idx_top, y.device),
            'trusted': _gather_topk_cpu_to_device(gated_info['trusted'], idx_top, y.device),
        }
        metric_prec_top = _materialize_precision_from_gated(gated_top)
        gate_top = _materialize_hlsi_gate_from_gated(gated_top, et2, var_t)
    else:
        if P_ref_cpu is None:
            raise ValueError('_build_hlsi_component_state requires either gated_info or P_ref_cpu.')
        P_top = _gather_topk_cpu_to_device(P_ref_cpu, idx_top, y.device)
        metric_prec_top = P_top
        gate_top = _materialize_hlsi_gate_from_precision(P_top, et2, var_t)

    return HLSIComponentState(
        y=y,
        t_val=t_val,
        et=et,
        et2=et2,
        var_t=var_t,
        alpha=alpha,
        alpha_top=alpha_top,
        alpha_top_norm=alpha_top_norm,
        idx_top=idx_top,
        s_twd=s_twd,
        d_top=d_top,
        metric_points_top=metric_points_top,
        metric_prec_top=metric_prec_top,
        gate_top=gate_top,
    )


def resolve_hlsi_gate_law(mode, gate_rho=None, gate_beta=None, gate_kappa=None,
                          gate_topk=64, gate_metric_source='mu'):
    mode = canonicalize_init_name(mode)

    if mode in {'hlsi_posterior', 'leaf_hlsi', 'gnl_hlsi'}:
        return DiracGateLaw()
    if mode in {'ce_hlsi', 'leaf_ce_hlsi', 'gnl_ce_hlsi'}:
        return PosteriorAverageGateLaw()
    if mode in {'tl_hlsi', 'leaf_tl_hlsi'}:
        return TemperedLikelihoodGateLaw(
            rho=0.5 if gate_rho is None else gate_rho,
            beta=1.0 if gate_beta is None else gate_beta,
            kappa=1.0 if gate_kappa is None else gate_kappa,
            topk=64 if gate_topk is None else gate_topk,
            metric_source=gate_metric_source,
        )

    raise ValueError(f'No HLSI gate law is defined for mode={mode!r}.')


def eval_modular_hlsi_posterior_score(y, t, mode, X_ref_cpu, log_lik_ref_cpu,
                                      P_ref_cpu, mu_ref_cpu, s0_post_ref_cpu,
                                      gated_info=None, batch_size=REF_STREAM_BATCH,
                                      gate_rho=None, gate_beta=None, gate_kappa=None,
                                      gate_topk=64, gate_metric_source='mu',
                                      transition_w='ou'):
    gate_law = resolve_hlsi_gate_law(
        mode,
        gate_rho=gate_rho, gate_beta=gate_beta, gate_kappa=gate_kappa,
        gate_topk=gate_topk, gate_metric_source=gate_metric_source,
    )

    fast_path = gate_law.fast_path_kind()
    if fast_path == 'dirac':
        return eval_hlsi_posterior_score(
            y, t, X_ref_cpu, log_lik_ref_cpu, P_ref_cpu, mu_ref_cpu,
            gated_info=gated_info, batch_size=batch_size,
            transition_w=transition_w,
        )
    if fast_path == 'posterior_average':
        return eval_ce_hlsi_posterior_score(
            y, t, X_ref_cpu, log_lik_ref_cpu, P_ref_cpu, s0_post_ref_cpu,
            mu_ref_cpu=mu_ref_cpu, gated_info=gated_info, batch_size=batch_size,
            transition_w=transition_w,
        )

    component_state = _build_hlsi_component_state(
        y, t, X_ref_cpu, s0_post_ref_cpu, log_lik_ref_cpu,
        P_ref_cpu=P_ref_cpu, mu_ref_cpu=mu_ref_cpu, gated_info=gated_info,
        gate_law=gate_law, batch_size=batch_size, transition_w=transition_w,
    )
    return gate_law.apply(component_state)


def eval_ce_hlsi_posterior_score(y, t, X_ref_cpu, log_lik_ref_cpu,
                                 P_ref_cpu, s0_post_ref_cpu, mu_ref_cpu=None,
                                 gated_info=None, batch_size=REF_STREAM_BATCH,
                                 apply_pou_correction=False, transition_w='ou'):
    """
    Certainty-Equivalent HLSI with externally supplied SNIS weights.

    The caller provides the log-weights used to form

        w_i(y,t) ∝ p_{t|0}(y | x_i) * weight_i,

    then forms the measurable gate from the correspondingly weighted
    SNIS-averaged posterior Hessian

        P_bar(y,t) = sum_i w_i P_i,
        A_bar(y,t) = e^{-2t} (e^{-2t} I + v_t P_bar)^{-1}.

    The CE-HLSI score is

        s_CE = s_Tweedie + A_bar (s_TSI - s_Tweedie),

    where both s_Tweedie and s_TSI use the same standard SNIS weights.
    """
    t_val = _canonicalize_time(t)
    et = math.exp(-t_val)
    et2 = et * et
    var_t = 1.0 - math.exp(-2.0 * t_val)

    w = get_posterior_snis_weights(
        y, t_val, X_ref_cpu, log_lik_ref_cpu, batch_size=batch_size,
        transition_w=transition_w, P_ref_cpu=P_ref_cpu,
        mu_ref_cpu=mu_ref_cpu, gated_info=gated_info)

    m_query, d = y.shape
    mu_x = torch.zeros((m_query, d), device=y.device, dtype=y.dtype)
    s_tsi_num = torch.zeros((m_query, d), device=y.device, dtype=y.dtype)
    P_bar = torch.zeros((m_query, d, d), device=y.device, dtype=y.dtype)

    for i in range(0, X_ref_cpu.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        w_batch = w[:, sl]
        X_batch = X_ref_cpu[sl].to(y.device, non_blocking=True)
        s0_batch = s0_post_ref_cpu[sl].to(y.device, non_blocking=True)
        P_batch = P_ref_cpu[sl].to(y.device, non_blocking=True)

        mu_x += torch.einsum('mb,bd->md', w_batch, X_batch)
        s_tsi_num += torch.einsum('mb,bd->md', w_batch, s0_batch)
        P_bar += torch.einsum('mb,bij->mij', w_batch, P_batch)

        del w_batch, X_batch, s0_batch, P_batch

    s_twd = -(1.0 / var_t) * (y - et * mu_x)
    s_tsi = (1.0 / et) * s_tsi_num

    gate_eig, V = _get_ce_hlsi_gate_eigenbasis(P_bar, et2, var_t)

    diff = s_tsi - s_twd
    diff_eig = torch.einsum('mji,mj->mi', V, diff)
    A_diff = torch.einsum('mij,mj->mi', V, gate_eig * diff_eig)
    score = s_twd + A_diff
    return score



def compute_pou_weighted_denominator_score(y, t, X_ref, log_lik_ref,
                                           grad_log_pou_denom_ref,
                                           batch_size=REF_STREAM_BATCH):
    """
    Coherent PoU denominator correction available from the existing SNIS bank.

    The final PoU mixture score should subtract
        E[∇_x log H̄(X) | Y=y]
    under the same PoU-weighted reference law already used for the SNIS weights.
    """
    if grad_log_pou_denom_ref is None:
        raise ValueError("PoU correction requires grad_log_pou_denom_ref")
    w = get_posterior_snis_weights(
        y, t, X_ref, log_lik_ref, batch_size=batch_size)
    return torch.einsum('mb,bd->md', w, grad_log_pou_denom_ref.to(y.device, non_blocking=True))


def get_score_wrapper(y, t, mode, X_ref, s0_post_ref, log_lik_ref,
                      P_ref=None, mu_ref=None, gated_info=None, init_weights='L',
                      transition_w='ou', grad_log_pou_denom_ref=None,
                      gate_rho=None, gate_beta=None, gate_kappa=None,
                      gate_topk=64, gate_metric_source='mu'):
    """
    No-CFG wrapper: returns only the conditional/posterior score estimate.

    The HLSI-family branch is modularized into:
      1. component weights (selected upstream through log_lik_ref / init_weights),
      2. a reusable per-query component state, and
      3. a gate law.

    Legacy modes such as HLSI and CE-HLSI remain exact fast paths so existing
    experiment scripts stay backwards compatible, while TL-HLSI/GATE-HLSI uses
    the modular gate-law machinery directly.
    """
    t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
    if t_val < 1e-4:
        t_val = 1e-4

    if mode == 'tweedie':
        score = eval_score_batched(
            y, t_val, X_ref, s0_post_ref, log_lik_ref,
            batch_size=REF_STREAM_BATCH, mode='tweedie')

    elif mode == 'blend_posterior':
        score = eval_blend_posterior_score(
            y, t_val, X_ref, s0_post_ref, log_lik_ref,
            batch_size=REF_STREAM_BATCH)

    elif mode in {
        'hlsi_posterior', 'ce_hlsi', 'leaf_hlsi', 'leaf_ce_hlsi',
        'gnl_hlsi', 'gnl_ce_hlsi', 'tl_hlsi', 'leaf_tl_hlsi',
    }:
        if P_ref is None or mu_ref is None:
            raise ValueError(f"mode={mode!r} requires both P_ref and mu_ref.")
        score = eval_modular_hlsi_posterior_score(
            y, t_val, mode,
            X_ref, log_lik_ref,
            P_ref, mu_ref, s0_post_ref,
            gated_info=gated_info, batch_size=REF_STREAM_BATCH,
            gate_rho=gate_rho, gate_beta=gate_beta, gate_kappa=gate_kappa,
            gate_topk=gate_topk, gate_metric_source=gate_metric_source,
            transition_w=transition_w,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return score


def compute_mean_ess(y, t, X_ref_cpu, log_lik_ref_cpu, eps=1e-30,
                     batch_size=REF_STREAM_BATCH, transition_w='ou',
                     P_ref_cpu=None, mu_ref_cpu=None, gated_info=None):
    transition_w = canonicalize_transition_w(transition_w)
    t_val, et, var_t, max_log_w = _streaming_max_logw_transition(
        y, t, X_ref_cpu, log_lik_ref_cpu, batch_size=batch_size,
        transition_w=transition_w, P_ref_cpu=P_ref_cpu,
        mu_ref_cpu=mu_ref_cpu, gated_info=gated_info)

    del et, var_t  # not needed below, but returned by helper for symmetry.

    z1 = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
    z2 = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)

    for i in range(0, X_ref_cpu.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        ll_batch = log_lik_ref_cpu[sl].to(y.device, non_blocking=True)
        if transition_w == 'ou':
            X_batch = X_ref_cpu[sl].to(y.device, non_blocking=True)
            mus = math.exp(-t_val) * X_batch
            diff = y.unsqueeze(1) - mus.unsqueeze(0)
            dists_sq = torch.sum(diff * diff, dim=2)
            log_trans = -dists_sq / (2.0 * (1.0 - math.exp(-2.0 * t_val)))
            del X_batch, mus, diff, dists_sq
        else:
            mu_chunk = mu_ref_cpu[sl].to(y.device, non_blocking=True)
            if gated_info is not None:
                gated_chunk = {
                    'eigvecs': gated_info['eigvecs'][sl].to(y.device, non_blocking=True),
                    'eigvals': gated_info['eigvals'][sl].to(y.device, non_blocking=True),
                    'trusted': gated_info['trusted'][sl].to(y.device, non_blocking=True),
                }
                P_chunk = None
            else:
                if P_ref_cpu is None:
                    raise ValueError("transition_w='surrogate' requires P_ref_cpu when gated_info is not supplied.")
                gated_chunk = None
                P_chunk = P_ref_cpu[sl].to(y.device, non_blocking=True)
            log_trans = _surrogate_chunk_log_transition(
                y, t_val, P_chunk, mu_chunk, gated_chunk=gated_chunk)
            del mu_chunk
            if gated_chunk is not None:
                del gated_chunk
            if P_chunk is not None:
                del P_chunk
        log_w_batch = log_trans + ll_batch.unsqueeze(0)
        w_batch = torch.exp(log_w_batch - max_log_w.unsqueeze(1))
        z1 += torch.sum(w_batch, dim=1, keepdim=True)
        z2 += torch.sum(w_batch ** 2, dim=1, keepdim=True)
        del ll_batch, log_trans, log_w_batch, w_batch

    ess_per_particle = (z1 ** 2) / torch.clamp(z2, min=eps)
    return ess_per_particle.mean().item()



def _mean_vector_norm(x):
    if x is None:
        return float('nan')
    if x.ndim == 1:
        return float(torch.linalg.vector_norm(x).item())
    return float(torch.linalg.vector_norm(x, dim=1).mean().item())


def _estimate_sampler_pde_eval_counts(cfg, n_ref=0, n_samples=None):
    """
    Lightweight, physics-agnostic accounting proxy for how much PDE-derived
    information each sampler consumes.

    Conventions:
      - Tweedie bank: likelihoods only.
      - Blend bank: likelihoods + scores.
      - HLSI-family banks: likelihoods + scores + GN Hessian proxies.
      - MALA: one joint posterior evaluation per chain at initialization and one
        per proposal step; we count that as one likelihood eval and one score
        eval each time.

    The aggregate ``pde_solve_count`` is the simple sum of these three counters.
    This is intentionally a portable solve-accounting proxy rather than a
    problem-specific wall-clock or adjoint-weighted cost model.
    """
    init_mode = canonicalize_init_name(cfg.get('init', 'prior'))
    n_ref = int(max(0, n_ref or 0))
    n_samples = int(cfg.get('n_samples', 0) if n_samples is None else n_samples)

    counts = {
        'pde_likelihood_evals': 0,
        'pde_score_evals': 0,
        'pde_gn_hessian_evals': 0,
    }

    if init_mode == 'tweedie':
        counts['pde_likelihood_evals'] += n_ref
    elif init_mode == 'blend_posterior':
        counts['pde_likelihood_evals'] += n_ref
        counts['pde_score_evals'] += n_ref
    elif init_mode in {'hlsi_posterior', 'ce_hlsi', 'leaf_hlsi', 'leaf_ce_hlsi', 'tl_hlsi', 'leaf_tl_hlsi', 'ref_laplace'}:
        counts['pde_likelihood_evals'] += n_ref
        counts['pde_score_evals'] += n_ref
        counts['pde_gn_hessian_evals'] += n_ref

    mala_steps = int(max(0, cfg.get('mala_steps', 0)))
    if mala_steps > 0 and n_samples > 0:
        n_posterior_evals = n_samples * (mala_steps + 1)
        counts['pde_likelihood_evals'] += n_posterior_evals
        counts['pde_score_evals'] += n_posterior_evals
        if bool(cfg.get('precond_mala', False)):
            counts['pde_gn_hessian_evals'] += 1

    counts['pde_solve_count'] = (
        counts['pde_likelihood_evals']
        + counts['pde_score_evals']
        + counts['pde_gn_hessian_evals']
    )
    return counts


def run_sampler_heun(n_samples, mode, X_ref, s0_post_ref, log_lik_ref,
                     P_ref=None, mu_ref=None, gated_info=None, init_weights='L',
                     steps=40, dim=15, log_mean_ess=False, x_init=None,
                     t_max=2.0, t_min=10 ** (-2.0),
                     grad_log_pou_denom_ref=None, transition_w='ou',
                     gate_rho=None, gate_beta=None, gate_kappa=None,
                     gate_topk=64, gate_metric_source='mu',
                     return_info=False):
    if x_init is None:
        y = torch.randn(n_samples, dim, device=device, dtype=torch.float64)
    else:
        y = x_init.detach().clone().to(device=device, dtype=torch.float64)
        if y.ndim != 2:
            raise ValueError('x_init must have shape [n_samples, dim].')
        n_samples = y.shape[0]
        dim = y.shape[1]

    info = {
        'score_norm_initial': float('nan'),
        'score_norm_mean': float('nan'),
        'score_norm_final': float('nan'),
        'score_norm_max': float('nan'),
        'score_norm_num_steps': int(max(0, steps)),
    }

    if steps <= 0:
        if log_mean_ess:
            ess_trace = {'t': np.array([]), 'mean_ess': np.array([])}
            if return_info:
                return y, ess_trace, info
            return y, ess_trace
        if return_info:
            return y, info
        return y

    ts = torch.logspace(math.log10(t_max), math.log10(t_min), steps + 1,
                        device=device, dtype=torch.float64)

    ess_trace = None
    if log_mean_ess:
        ess_trace = {
            't': [ts[0].item()],
            'mean_ess': [compute_mean_ess(
                y, ts[0].item(), X_ref, log_lik_ref,
                P_ref_cpu=P_ref, mu_ref_cpu=mu_ref, gated_info=gated_info,
                transition_w=transition_w)],
        }

    score_norm_sum = 0.0
    score_norm_max = 0.0
    final_score = None

    for i in range(steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = t_cur - t_next

        s_cur = get_score_wrapper(y, t_cur, mode, X_ref, s0_post_ref, log_lik_ref,
                                  P_ref=P_ref, mu_ref=mu_ref, gated_info=gated_info,
                                  init_weights=init_weights,
                                  grad_log_pou_denom_ref=grad_log_pou_denom_ref,
                                  gate_rho=gate_rho, gate_beta=gate_beta, gate_kappa=gate_kappa,
                                  gate_topk=gate_topk, gate_metric_source=gate_metric_source,
                                  transition_w=transition_w)
        cur_norm = _mean_vector_norm(s_cur)
        if i == 0:
            info['score_norm_initial'] = cur_norm
        score_norm_sum += cur_norm
        score_norm_max = max(score_norm_max, cur_norm)
        d_cur = y + 2.0 * s_cur

        z = torch.randn_like(y)
        y_hat = y + d_cur * dt + math.sqrt(2.0 * dt.item()) * z

        s_next = get_score_wrapper(y_hat, t_next, mode, X_ref, s0_post_ref, log_lik_ref,
                                   P_ref=P_ref, mu_ref=mu_ref, gated_info=gated_info,
                                   init_weights=init_weights,
                                   grad_log_pou_denom_ref=grad_log_pou_denom_ref,
                                   gate_rho=gate_rho, gate_beta=gate_beta, gate_kappa=gate_kappa,
                                   gate_topk=gate_topk, gate_metric_source=gate_metric_source,
                                   transition_w=transition_w)
        d_next = y_hat + 2.0 * s_next

        y = y + 0.5 * (d_cur + d_next) * dt + math.sqrt(2.0 * dt.item()) * z

        if log_mean_ess:
            ess_trace['t'].append(t_next.item())
            ess_trace['mean_ess'].append(compute_mean_ess(
                y, t_next.item(), X_ref, log_lik_ref,
                P_ref_cpu=P_ref, mu_ref_cpu=mu_ref, gated_info=gated_info,
                transition_w=transition_w))

    final_score = get_score_wrapper(y, ts[-1], mode, X_ref, s0_post_ref, log_lik_ref,
                                    P_ref=P_ref, mu_ref=mu_ref, gated_info=gated_info,
                                    init_weights=init_weights,
                                    grad_log_pou_denom_ref=grad_log_pou_denom_ref,
                                    gate_rho=gate_rho, gate_beta=gate_beta, gate_kappa=gate_kappa,
                                    gate_topk=gate_topk, gate_metric_source=gate_metric_source,
                                    transition_w=transition_w)
    info['score_norm_mean'] = score_norm_sum / float(max(1, steps))
    info['score_norm_final'] = _mean_vector_norm(final_score)
    info['score_norm_max'] = max(score_norm_max, info['score_norm_final'])

    if log_mean_ess:
        ess_trace = {k: np.array(v) for k, v in ess_trace.items()}
        if return_info:
            return y, ess_trace, info
        return y, ess_trace

    if return_info:
        return y, info
    return y


def _build_ref_laplace_component(bank, log_weight_key='log_mass_ref'):
    """Select the single dominant Ref_Laplace component from a reference bank."""
    if log_weight_key not in bank:
        raise KeyError(
            f"Reference bank is missing '{log_weight_key}'. Available keys: {sorted(bank.keys())}"
        )
    log_weights = bank[log_weight_key]
    if log_weights.ndim != 1:
        raise ValueError(f"Expected 1D log weights for Ref_Laplace, got shape={tuple(log_weights.shape)}")

    best_idx = int(torch.argmax(log_weights).item())
    component = {
        'selected_ref_index': best_idx,
        'selected_log_mass': float(log_weights[best_idx].item()),
        'x_map': bank['X_ref'][best_idx].to(device=device, dtype=torch.float64),
        'mean': bank['mu_ref'][best_idx].to(device=device, dtype=torch.float64),
        'precision': bank['P_ref'][best_idx].to(device=device, dtype=torch.float64),
    }
    return component


def sample_ref_laplace(n_samples, bank, log_weight_key='log_mass_ref'):
    """
    Draw samples from the single local Laplace component with the largest
    approximate posterior mass in the precomputed reference bank.
    """
    component = _build_ref_laplace_component(bank, log_weight_key=log_weight_key)
    mean = component['mean']
    precision = component['precision']
    samples = sample_gaussian_from_precision(mean, precision, n_samples)
    cov_eigs = torch.clamp(torch.linalg.eigvalsh(precision), min=1e-12)
    info = {
        'selected_ref_index': component['selected_ref_index'],
        'selected_log_mass': component['selected_log_mass'],
        'selected_component_mean_norm': float(torch.linalg.vector_norm(mean).item()),
        'selected_x_map_norm': float(torch.linalg.vector_norm(component['x_map']).item()),
        'selected_component_cov_trace': float(torch.sum(torch.reciprocal(cov_eigs)).item()),
        'score_norm_initial': float('nan'),
        'score_norm_mean': float('nan'),
        'score_norm_final': float('nan'),
        'score_norm_max': float('nan'),
        'score_norm_num_steps': 0,
    }
    return samples, info


def _make_frozen_mala_preconditioner(component, ridge=1e-10):
    """
    Build a fixed global MALA preconditioner from the same dominant reference
    component selected by Ref_Laplace.
    """
    precision = 0.5 * (component['precision'] + component['precision'].T)
    evals, evecs = torch.linalg.eigh(precision)
    evals = torch.clamp(evals, min=max(float(HESS_MIN), float(ridge)))
    inv_evals = torch.reciprocal(evals)
    cov = (evecs * inv_evals.unsqueeze(0)) @ evecs.T
    sqrt_cov = evecs * torch.sqrt(inv_evals).unsqueeze(0)
    precision_stable = (evecs * evals.unsqueeze(0)) @ evecs.T
    return {
        'selected_ref_index': int(component['selected_ref_index']),
        'selected_log_mass': float(component['selected_log_mass']),
        'x_map': component['x_map'],
        'precision': precision_stable,
        'cov': cov,
        'sqrt_cov': sqrt_cov,
    }


def _rowwise_quadratic_form(v, precision):
    return torch.sum(v * (v @ precision), dim=1)


def run_mala_sampler(n_samples, prior_model, lik_model, steps=1000, dt=5e-4,
                     burn_in=200, x_init=None, verbose=True, return_info=False,
                     preconditioner=None):
    if x_init is None:
        x = prior_model.sample(n_samples)
    else:
        x = x_init.detach().clone().to(device=device, dtype=torch.float64)
        if x.ndim != 2:
            raise ValueError('x_init must have shape [n_samples, dim].')
        n_samples = x.shape[0]

    use_precond = preconditioner is not None
    if use_precond:
        precision = preconditioner['precision'].to(device=device, dtype=torch.float64)
        cov = preconditioner['cov'].to(device=device, dtype=torch.float64)
        sqrt_cov = preconditioner['sqrt_cov'].to(device=device, dtype=torch.float64)

    log_prior = prior_model.log_prob(x)
    log_lik, grad_lik = lik_model.log_likelihood_and_grad(x)
    score_prior = prior_model.score0(x)

    log_post = log_prior + log_lik
    grad_log_post = score_prior + grad_lik

    accept_count = 0.0
    denom_accept = max(1, steps - burn_in)
    score_norm_initial = _mean_vector_norm(grad_log_post)
    score_norm_sum = 0.0
    score_norm_max = score_norm_initial

    for i in range(steps):
        noise = torch.randn_like(x)
        if use_precond:
            drift = dt * (grad_log_post @ cov)
            noise_term = math.sqrt(2.0 * dt) * (noise @ sqrt_cov.T)
            x_prop = x + drift + noise_term
        else:
            drift = dt * grad_log_post
            noise_term = math.sqrt(2.0 * dt) * noise
            x_prop = x + drift + noise_term

        log_prior_prop = prior_model.log_prob(x_prop)
        log_lik_prop, grad_lik_prop = lik_model.log_likelihood_and_grad(x_prop)
        score_prior_prop = prior_model.score0(x_prop)

        log_post_prop = log_prior_prop + log_lik_prop
        grad_log_post_prop = score_prior_prop + grad_lik_prop

        if use_precond:
            drift_prop = dt * (grad_log_post_prop @ cov)
            log_q_fwd = -_rowwise_quadratic_form(x_prop - x - drift, precision) / (4.0 * dt)
            log_q_bwd = -_rowwise_quadratic_form(x - x_prop - drift_prop, precision) / (4.0 * dt)
        else:
            log_q_fwd = -torch.sum((x_prop - x - drift) ** 2, dim=1) / (4.0 * dt)
            log_q_bwd = -torch.sum((x - x_prop - dt * grad_log_post_prop) ** 2, dim=1) / (4.0 * dt)

        log_alpha = log_post_prop - log_post + log_q_bwd - log_q_fwd
        accept = torch.log(torch.rand(n_samples, device=device)) < log_alpha

        x[accept] = x_prop[accept]
        log_post[accept] = log_post_prop[accept]
        grad_log_post[accept] = grad_log_post_prop[accept]

        step_score_norm = _mean_vector_norm(grad_log_post)
        score_norm_sum += step_score_norm
        score_norm_max = max(score_norm_max, step_score_norm)

        if i >= burn_in:
            accept_count += accept.float().mean().item()

        if verbose and (i % 100 == 0):
            mode_name = 'Precond-MALA' if use_precond else 'MALA'
            print(f"{mode_name} Iteration {i}/{steps}")

    accept_rate = accept_count / denom_accept
    if verbose:
        mode_name = 'Precond-MALA' if use_precond else 'MALA'
        print(f"{mode_name} Acceptance: {accept_rate:.2f}")

    info = {
        'acceptance_post_burnin': accept_rate,
        'steps': steps,
        'burn_in': burn_in,
        'dt': dt,
        'precond_mala': bool(use_precond),
        'score_norm_initial': score_norm_initial,
        'score_norm_mean': score_norm_sum / float(max(1, steps)),
        'score_norm_final': _mean_vector_norm(grad_log_post),
        'score_norm_max': score_norm_max,
        'score_norm_num_steps': int(max(0, steps)),
    }
    if use_precond:
        info.update({
            'precond_selected_ref_index': int(preconditioner['selected_ref_index']),
            'precond_selected_log_mass': float(preconditioner['selected_log_mass']),
            'precond_x_map_norm': float(torch.linalg.vector_norm(preconditioner['x_map']).item()),
            'precond_trace_cov': float(torch.trace(cov).item()),
        })
    if return_info:
        return x, info
    return x


# ==========================================
# 6. EVALUATION UTILS (physics-agnostic — unchanged)
# ==========================================

def robust_clean_samples(samps):
    samps_np = samps.cpu().numpy() if isinstance(samps, torch.Tensor) else samps
    valid_mask = np.isfinite(samps_np).all(axis=1)
    if valid_mask.sum() < 10:
        return torch.tensor(samps_np[valid_mask], device=device)
    q25 = np.percentile(samps_np[valid_mask], 25, axis=0)
    q75 = np.percentile(samps_np[valid_mask], 75, axis=0)
    iqr = q75 - q25
    lower = q25 - 5.0 * iqr
    upper = q75 + 5.0 * iqr
    in_bounds = (samps_np >= lower) & (samps_np <= upper)
    valid_mask = valid_mask & in_bounds.all(axis=1)
    return torch.tensor(samps_np[valid_mask], device=device)

def get_valid_samples(samps):
    """Return a NumPy array of finite, non-extreme samples for plotting/metrics."""
    clean = robust_clean_samples(samps)
    if isinstance(clean, torch.Tensor):
        return clean.detach().cpu().numpy()
    return np.asarray(clean)


def rmse_array(x_hat, x_true):
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)))


def pearson_corr_array(x_hat, x_true, eps=1e-12):
    x_hat = np.asarray(x_hat, dtype=np.float64).reshape(-1)
    x_true = np.asarray(x_true, dtype=np.float64).reshape(-1)
    x_hat_centered = x_hat - np.mean(x_hat)
    x_true_centered = x_true - np.mean(x_true)
    denom = np.linalg.norm(x_hat_centered) * np.linalg.norm(x_true_centered)
    if denom <= eps:
        return float(1.0 if np.linalg.norm(x_hat - x_true) <= eps else np.nan)
    return float(np.dot(x_hat_centered, x_true_centered) / denom)


def sliced_wasserstein_distance(X_a, X_b, num_projections=500, p=2):
    n_a = X_a.shape[0]
    n_b = X_b.shape[0]
    if n_a > n_b:
        idx = torch.randperm(n_a)[:n_b]
        X_a = X_a[idx]
    elif n_b > n_a:
        idx = torch.randperm(n_b)[:n_a]
        X_b = X_b[idx]
    dim = X_a.shape[1]
    projections = torch.randn((num_projections, dim), device=X_a.device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    proj_a = torch.matmul(X_a, projections.t())
    proj_b = torch.matmul(X_b, projections.t())
    proj_a_sorted, _ = torch.sort(proj_a, dim=0)
    proj_b_sorted, _ = torch.sort(proj_b, dim=0)
    wd = torch.pow(torch.abs(proj_a_sorted - proj_b_sorted), p).mean()
    return torch.pow(wd, 1.0 / p).item()


def compute_moment_errors(samples_approx, samples_ref):
    mean_approx = torch.mean(samples_approx, dim=0)
    mean_ref = torch.mean(samples_ref, dim=0)
    mean_err = torch.norm(mean_approx - mean_ref).item()
    centered_approx = samples_approx - mean_approx
    centered_ref = samples_ref - mean_ref
    cov_approx = torch.matmul(centered_approx.t(), centered_approx) / (samples_approx.shape[0] - 1)
    cov_ref = torch.matmul(centered_ref.t(), centered_ref) / (samples_ref.shape[0] - 1)
    cov_err = torch.norm(cov_approx - cov_ref).item()
    return mean_err, cov_err


def compute_mmd_rbf(X, Y, sigma=None):
    n_max = 2000
    if X.shape[0] > n_max:
        X = X[:n_max]
    if Y.shape[0] > n_max:
        Y = Y[:n_max]
    dist_xx = torch.cdist(X, X, p=2) ** 2
    dist_yy = torch.cdist(Y, Y, p=2) ** 2
    dist_xy = torch.cdist(X, Y, p=2) ** 2
    if sigma is None:
        combined = torch.cat([dist_xx.view(-1), dist_yy.view(-1), dist_xy.view(-1)])
        sigma = torch.median(combined[combined > 0])
        sigma = torch.sqrt(sigma) if sigma > 0 else 1.0
    gamma = 1.0 / (2 * sigma ** 2)
    K_xx = torch.exp(-gamma * dist_xx)
    K_yy = torch.exp(-gamma * dist_yy)
    K_xy = torch.exp(-gamma * dist_xy)
    mmd_sq = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return torch.sqrt(torch.clamp(mmd_sq, min=0.0)).item()


def rmse_vec(x_hat, x_true, eps=1e-12):
    return torch.sqrt(torch.mean((x_hat - x_true) ** 2)).item()


def rel_l2_vec(x_hat, x_true, eps=1e-12):
    num = torch.norm(x_hat - x_true).item()
    den = torch.norm(x_true).item() + eps
    return num / den


def compute_knn_entropy(samples, k=5):
    n, d = samples.shape
    if n <= k:
        return 0.0
    dists = torch.cdist(samples, samples)
    k_dists, _ = torch.topk(dists, k + 1, largest=False, dim=1)
    r_k = k_dists[:, k]
    log_vd = (d / 2.0) * math.log(math.pi) - torch.lgamma(
        torch.tensor(d / 2.0 + 1.0, device=samples.device))
    avg_log_dist = torch.log(r_k + 1e-10).mean()
    digamma_k = torch.digamma(torch.tensor(float(k), device=samples.device))
    entropy = d * avg_log_dist + math.log(n) - digamma_k + log_vd
    return entropy.item()


def compute_kl_divergence(samples, prior_model, lik_model):
    clean_x = robust_clean_samples(samples)
    if len(clean_x) < 20:
        return float('inf')
    entropy = compute_knn_entropy(clean_x, k=5)
    with torch.no_grad():
        log_prior = prior_model.log_prob(clean_x)
        log_lik = lik_model.log_likelihood(clean_x)
        unnorm_log_post = log_prior + log_lik
        expected_log_p = torch.mean(unnorm_log_post).item()
    return -entropy - expected_log_p


def compute_multiscale_ksd(samples, score_func, sigmas=(0.1, 0.2, 0.4, 0.8)):
    N = samples.shape[0]
    if N > 1000:
        idx = torch.randperm(N)[:1000]
        samples = samples[idx]
        N = 1000

    X = samples
    D = X.shape[1]
    s = score_func(X)

    diff = X.unsqueeze(1) - X.unsqueeze(0)
    r2 = torch.sum(diff ** 2, dim=-1)

    ksd2 = 0.0
    for sigma in sigmas:
        K = torch.exp(-r2 / (2 * sigma ** 2))
        sdot = torch.matmul(s, s.t())
        term1 = sdot * K

        r_dot_sx = torch.einsum('ijd,id->ij', diff, s)
        r_dot_sy = torch.einsum('ijd,jd->ij', diff, s)
        term2 = (r_dot_sx - r_dot_sy) / (sigma ** 2) * K

        term3 = (D / (sigma ** 2) - r2 / (sigma ** 4)) * K

        U = term1 + term2 + term3
        ksd2 += torch.sum(U) / (N * N)

    return ksd2.item() / len(sigmas)


# ==========================================
# 7. PCA VISUALIZATION
# ==========================================

def resolve_plot_normalizer(normalizer, available_labels, display_names=None,
                            metrics_dict=None, fallback=None,
                            best_metric_keys=('RelL2_field', 'IC RelL2(%)', 'RelL2_q(%)')):
    if len(available_labels) == 0:
        raise ValueError('No available sampler labels to resolve a plot normalizer.')
    if display_names is None:
        display_names = {label: label for label in available_labels}

    def _norm_text(x):
        return str(x).strip().lower().replace('_', ' ').replace('-', ' ')

    available_labels = list(available_labels)
    fallback = fallback if fallback in available_labels else available_labels[0]

    if normalizer is None:
        return fallback

    normalizer_key = _norm_text(normalizer)
    if normalizer_key in {'reference', 'default', 'fallback'}:
        return fallback
    if normalizer_key == 'best':
        if metrics_dict is not None:
            for metric_key in best_metric_keys:
                best_label = None
                best_value = float('inf')
                for label in available_labels:
                    value = metrics_dict.get(label, {}).get(metric_key, np.nan)
                    if np.isfinite(value) and value < best_value:
                        best_value = float(value)
                        best_label = label
                if best_label is not None:
                    return best_label
        print(
            f"[resolve_plot_normalizer] Could not resolve 'best' via metrics {best_metric_keys}. "
            f"Falling back to {fallback}."
        )
        return fallback

    for label in available_labels:
        if normalizer == label:
            return label
    for label in available_labels:
        if _norm_text(label) == normalizer_key:
            return label
    for label in available_labels:
        disp = display_names.get(label, label)
        if normalizer == disp or _norm_text(disp) == normalizer_key:
            return label

    available_display = [display_names.get(label, label) for label in available_labels]
    raise ValueError(
        f"Unknown plot normalizer '{normalizer}'. Available labels: {available_labels}. "
        f"Available display names: {available_display}."
    )


def plot_pca_histograms(samples_dict, alpha_true=None, display_names=None,
                        normalizer='best', metrics_dict=None, fallback_key=None):
    if len(samples_dict) == 0:
        raise ValueError('samples_dict is empty.')

    if display_names is None:
        display_names = {k: k for k in samples_dict.keys()}

    if "ACTIVE_DIM" in globals():
        d_lat = int(ACTIVE_DIM)
    else:
        any_key = next(iter(samples_dict.keys()))
        d_lat = int(robust_clean_samples(samples_dict[any_key]).shape[1])

    has_alpha_true = alpha_true is not None
    if has_alpha_true:
        alpha_true = np.asarray(alpha_true).reshape(-1)[:d_lat]

    anchor = resolve_plot_normalizer(
        normalizer,
        list(samples_dict.keys()),
        display_names=display_names,
        metrics_dict=metrics_dict,
        fallback=fallback_key,
        best_metric_keys=('RelL2_field', 'IC RelL2(%)', 'RelL2_q(%)'),
    )
    anchor_data = robust_clean_samples(samples_dict[anchor])
    if anchor_data.shape[0] < 10:
        raise ValueError(f"Not enough valid samples in anchor method '{anchor}' for PCA.")

    mean_anchor = torch.mean(anchor_data[:, :d_lat], dim=0)
    centered_anchor = anchor_data[:, :d_lat] - mean_anchor
    U, S, Vh = torch.linalg.svd(centered_anchor, full_matrices=False)
    V = Vh.T

    pairs = [(0, 1)]
    if V.shape[1] >= 4:
        pairs.append((2, 3))

    methods = list(samples_dict.keys())
    fig, axes = plt.subplots(len(pairs), len(methods), figsize=(5 * len(methods), 5 * len(pairs)))
    if len(pairs) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(methods) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (d1, d2) in enumerate(pairs):
        v1 = V[:, d1]
        v2 = V[:, d2]

        if has_alpha_true:
            true_cent = torch.tensor(alpha_true, device=mean_anchor.device,
                                     dtype=mean_anchor.dtype) - mean_anchor
            t1 = torch.dot(true_cent, v1).item()
            t2 = torch.dot(true_cent, v2).item()

        proj_anchor_1 = torch.matmul(centered_anchor, v1).detach().cpu().numpy()
        proj_anchor_2 = torch.matmul(centered_anchor, v2).detach().cpu().numpy()
        q01_x, q99_x = np.percentile(proj_anchor_1, [1, 99])
        q01_y, q99_y = np.percentile(proj_anchor_2, [1, 99])
        span_x = max(q99_x - q01_x, 1e-12)
        span_y = max(q99_y - q01_y, 1e-12)
        pad = 0.5
        xlims = [q01_x - pad * span_x, q99_x + pad * span_x]
        ylims = [q01_y - pad * span_y, q99_y + pad * span_y]

        ref_hist, _, _ = np.histogram2d(
            proj_anchor_1, proj_anchor_2,
            bins=60, range=[xlims, ylims], density=True,
        )
        hist_vmax = max(float(np.nanmax(ref_hist)), 1e-12)

        for col_idx, label in enumerate(methods):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            samps = robust_clean_samples(samples_dict[label])
            if samps.shape[0] < 10:
                ax.set_title(f"{display_names.get(label, label)} (unstable)", fontsize=16)
                ax.axis('off')
                continue

            centered = samps[:, :d_lat] - mean_anchor
            p1 = torch.matmul(centered, v1).detach().cpu().numpy()
            p2 = torch.matmul(centered, v2).detach().cpu().numpy()

            ax.hist2d(
                p1, p2, bins=60, range=[xlims, ylims],
                cmap='inferno', density=True, vmax=hist_vmax,
            )
            if has_alpha_true:
                ax.scatter(t1, t2, c='cyan', marker='x', s=200, linewidth=4,
                           label='True $alpha$')

            if row_idx == 0:
                ax.set_title(display_names.get(label, label), fontsize=18)
            if col_idx == 0:
                ax.set_ylabel(f"PC {d1 + 1} vs PC {d2 + 1}", fontsize=18)
            if has_alpha_true and row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=14)

    plt.suptitle(
        f"PCA of posterior samples (normalizer={display_names.get(anchor, anchor)}, dim={d_lat})",
        fontsize=18, y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_mean_ess_logs(ess_logs_dict, display_names=None):
    if len(ess_logs_dict) == 0:
        print('\n=== Mean ESS vs t ===')
        print('No ESS traces were requested.')
        return

    plt.figure(figsize=(8, 5))
    for label, trace in ess_logs_dict.items():
        if trace is None or len(trace.get('t', [])) == 0:
            continue
        title = display_names.get(label, label) if display_names is not None else label
        t_vec = trace['t']
        ess_vec = trace['mean_ess']
        order = np.argsort(t_vec)
        plt.plot(t_vec[order], ess_vec[order], marker='o', linewidth=2, label=title)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Diffusion time t')
    plt.ylabel('Mean ESS across particles')
    plt.title('Mean ESS vs diffusion time t')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ==========================================
# 8. HLSI CORE (streaming, no full time-cache)
# ==========================================

def _compute_sigmainv_gated(gated_info, et2, var_t):
    eigvecs = gated_info['eigvecs']
    eigvals = gated_info['eigvals']
    trusted = gated_info['trusted']
    hlsi_eig = eigvals / (et2 + var_t * eigvals + 1e-30)
    twd_eig = torch.full_like(eigvals, 1.0 / var_t)
    sigmainv_eig = torch.where(trusted, hlsi_eig, twd_eig)
    SigmaInv = torch.einsum('nij,nj,nkj->nik', eigvecs, sigmainv_eig, eigvecs)
    return SigmaInv
def _get_hlsi_chunk_time_mats(t, P_chunk, mu_chunk, gated_chunk=None):
    t_val = _canonicalize_time(t)
    et = math.exp(-t_val)
    et2 = et * et
    var_t = 1.0 - math.exp(-2.0 * t_val)
    if gated_chunk is not None:
        SigmaInv = _compute_sigmainv_gated(gated_chunk, et2, var_t)
    else:
        _, d = mu_chunk.shape
        I = torch.eye(d, device=P_chunk.device, dtype=P_chunk.dtype).unsqueeze(0)
        A = et2 * I + var_t * P_chunk
        SigmaInv = torch.linalg.solve(A, P_chunk)
    mu_t = et * mu_chunk
    b = torch.einsum('nij,nj->ni', SigmaInv, mu_t)
    return SigmaInv, b


def _get_pou_window_chunk_time_mats(t, P_chunk, X_chunk, gated_chunk=None):
    """
    Diffused PoU window parameters for H_t(y; i), where the base window is

        H_i(x) \propto exp[-0.5 (x - x_i)^T P_i (x - x_i)].

    After the OU forward process,

        Y_t | i  ~  N(e^{-t} x_i,\; e^{-2t} P_i^{-1} + v_t I),

    with score

        ∇_y log H_t(y; i) = -SigmaInv_i(t) y + b_i(t).

    The returned log_norm is the component-dependent normalization term
    (up to an additive constant common across components), needed so the
    PoU denominator uses the properly normalized diffused windows.
    """
    t_val = _canonicalize_time(t)
    et = math.exp(-t_val)
    et2 = et * et
    var_t = 1.0 - math.exp(-2.0 * t_val)

    if gated_chunk is not None:
        eigvecs = gated_chunk['eigvecs']
        eigvals = gated_chunk['eigvals']
        trusted = gated_chunk['trusted']

        denom_eig = et2 + var_t * eigvals
        sigmainv_eig = torch.where(
            trusted,
            eigvals / torch.clamp(denom_eig, min=1e-30),
            torch.zeros_like(eigvals),
        )
        SigmaInv = torch.einsum('nij,nj,nkj->nik', eigvecs, sigmainv_eig, eigvecs)

        log_norm_eig = torch.where(
            trusted,
            torch.log(torch.clamp(eigvals, min=1e-30))
            - torch.log(torch.clamp(denom_eig, min=1e-30)),
            torch.zeros_like(eigvals),
        )
        log_norm = 0.5 * torch.sum(log_norm_eig, dim=1)
    else:
        _, d = X_chunk.shape
        I = torch.eye(d, device=P_chunk.device, dtype=P_chunk.dtype).unsqueeze(0)
        A = et2 * I + var_t * P_chunk
        SigmaInv = torch.linalg.solve(A, P_chunk)
        sign_P, logabsdet_P = torch.linalg.slogdet(P_chunk)
        sign_A, logabsdet_A = torch.linalg.slogdet(A)
        if not torch.all(sign_P > 0):
            raise RuntimeError('PoU window precision must be positive definite.')
        if not torch.all(sign_A > 0):
            raise RuntimeError('PoU diffused window covariance term must be positive definite.')
        log_norm = 0.5 * (logabsdet_P - logabsdet_A)

    mu_t = et * X_chunk
    b = torch.einsum('nij,nj->ni', SigmaInv, mu_t)
    return SigmaInv, b, log_norm


def eval_pou_coverage_score(y, t, X_ref_cpu, P_ref_cpu, gated_info=None,
                            batch_size=REF_STREAM_BATCH):
    """
    Compute the PoU denominator correction

        ∇_y log sum_j H_t(y; j),

    where H_t(y; j) is the OU-diffused Gaussian window associated with
    reference j. This is the score of the diffused reference coverage density
    and is subtracted from PoU-HLSI / CE-PoU-HLSI.
    """
    t_val = _canonicalize_time(t)
    quad_y = torch.sum(y * y, dim=1, keepdim=True)
    max_log_h = torch.full((y.shape[0], 1), -float('inf'), device=y.device, dtype=y.dtype)

    for i in range(0, X_ref_cpu.shape[0], batch_size):
        X_chunk = X_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
        if gated_info is not None:
            gated_chunk = {
                'eigvecs': gated_info['eigvecs'][i:i + batch_size].to(y.device, non_blocking=True),
                'eigvals': gated_info['eigvals'][i:i + batch_size].to(y.device, non_blocking=True),
                'trusted': gated_info['trusted'][i:i + batch_size].to(y.device, non_blocking=True),
            }
            P_chunk = None
        else:
            gated_chunk = None
            P_chunk = P_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)

        SigmaInv_chunk, b_chunk, log_norm_chunk = _get_pou_window_chunk_time_mats(
            t_val, P_chunk, X_chunk, gated_chunk=gated_chunk)
        sigma_y = torch.einsum('bij,mj->mbi', SigmaInv_chunk, y)
        quad_sigma = torch.sum(sigma_y * y.unsqueeze(1), dim=2)
        cross = torch.einsum('md,bd->mb', y, b_chunk)
        quad_mu = torch.sum((math.exp(-t_val) * X_chunk) * b_chunk, dim=1).unsqueeze(0)
        log_h = log_norm_chunk.unsqueeze(0) - 0.5 * (quad_sigma - 2.0 * cross + quad_mu)
        max_log_h = torch.maximum(max_log_h, torch.max(log_h, dim=1, keepdim=True).values)

        del X_chunk, SigmaInv_chunk, b_chunk, log_norm_chunk, sigma_y, quad_sigma, cross, quad_mu, log_h
        if gated_chunk is not None:
            del gated_chunk
        if P_chunk is not None:
            del P_chunk

    denom_Z = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
    term1_num = torch.zeros_like(y)
    term2_num = torch.zeros_like(y)

    for i in range(0, X_ref_cpu.shape[0], batch_size):
        X_chunk = X_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
        if gated_info is not None:
            gated_chunk = {
                'eigvecs': gated_info['eigvecs'][i:i + batch_size].to(y.device, non_blocking=True),
                'eigvals': gated_info['eigvals'][i:i + batch_size].to(y.device, non_blocking=True),
                'trusted': gated_info['trusted'][i:i + batch_size].to(y.device, non_blocking=True),
            }
            P_chunk = None
        else:
            gated_chunk = None
            P_chunk = P_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)

        SigmaInv_chunk, b_chunk, log_norm_chunk = _get_pou_window_chunk_time_mats(
            t_val, P_chunk, X_chunk, gated_chunk=gated_chunk)
        sigma_y = torch.einsum('bij,mj->mbi', SigmaInv_chunk, y)
        quad_sigma = torch.sum(sigma_y * y.unsqueeze(1), dim=2)
        cross = torch.einsum('md,bd->mb', y, b_chunk)
        quad_mu = torch.sum((math.exp(-t_val) * X_chunk) * b_chunk, dim=1).unsqueeze(0)
        log_h = log_norm_chunk.unsqueeze(0) - 0.5 * (quad_sigma - 2.0 * cross + quad_mu)
        h_batch = torch.exp(log_h - max_log_h)

        denom_Z += torch.sum(h_batch, dim=1, keepdim=True)
        term2_num += torch.einsum('mb,bd->md', h_batch, b_chunk)
        term1_num += torch.einsum('mb,mbd->md', h_batch, sigma_y)

        del X_chunk, SigmaInv_chunk, b_chunk, log_norm_chunk, sigma_y, quad_sigma, cross, quad_mu, log_h, h_batch
        if gated_chunk is not None:
            del gated_chunk
        if P_chunk is not None:
            del P_chunk

    denom_Z = torch.clamp(denom_Z, min=1e-30)
    return -term1_num / denom_Z + term2_num / denom_Z


def eval_hlsi_posterior_score(y, t, X_ref_cpu, log_lik_ref_cpu, P_ref_cpu, mu_ref_cpu,
                              gated_info=None, batch_size=REF_STREAM_BATCH,
                              apply_pou_correction=False, transition_w='ou'):
    transition_w = canonicalize_transition_w(transition_w)
    t_val, et, var_t, max_log_w = _streaming_max_logw_transition(
        y, t, X_ref_cpu, log_lik_ref_cpu, batch_size=batch_size,
        transition_w=transition_w, P_ref_cpu=P_ref_cpu,
        mu_ref_cpu=mu_ref_cpu, gated_info=gated_info)
    denom_Z = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
    term1_num = torch.zeros_like(y)
    term2_num = torch.zeros_like(y)
    for i in range(0, X_ref_cpu.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        X_batch = X_ref_cpu[sl].to(y.device, non_blocking=True)
        ll_batch = log_lik_ref_cpu[sl].to(y.device, non_blocking=True)
        mu_chunk = mu_ref_cpu[sl].to(y.device, non_blocking=True)
        if gated_info is not None:
            gated_chunk = {
                'eigvecs': gated_info['eigvecs'][i:i + batch_size].to(y.device, non_blocking=True),
                'eigvals': gated_info['eigvals'][i:i + batch_size].to(y.device, non_blocking=True),
                'trusted': gated_info['trusted'][i:i + batch_size].to(y.device, non_blocking=True),
            }
            P_chunk = None
        else:
            gated_chunk = None
            P_chunk = P_ref_cpu[i:i + batch_size].to(y.device, non_blocking=True)
        if transition_w == 'ou':
            mus = et * X_batch
            diff = y.unsqueeze(1) - mus.unsqueeze(0)
            dists_sq = torch.sum(diff * diff, dim=2)
            log_trans = -dists_sq / (2.0 * var_t)
            del mus, diff, dists_sq
        else:
            log_trans = _surrogate_chunk_log_transition(
                y, t_val, P_chunk, mu_chunk, gated_chunk=gated_chunk)
        log_w = log_trans + ll_batch.unsqueeze(0)
        w_batch = torch.exp(log_w - max_log_w.unsqueeze(1))

        SigmaInv_chunk, b_chunk = _get_hlsi_chunk_time_mats(
            t_val, P_chunk, mu_chunk, gated_chunk=gated_chunk)
        denom_Z += torch.sum(w_batch, dim=1, keepdim=True)
        term2_num += torch.einsum('mb,bd->md', w_batch, b_chunk)
        sigma_y = torch.einsum('bij,mj->mbi', SigmaInv_chunk, y)
        term1_num += torch.einsum('mb,mbd->md', w_batch, sigma_y)
        del X_batch, ll_batch, log_trans, log_w, w_batch
        del mu_chunk, SigmaInv_chunk, b_chunk, sigma_y
        if gated_chunk is not None:
            del gated_chunk
        if P_chunk is not None:
            del P_chunk
    denom_Z = torch.clamp(denom_Z, min=1e-30)
    score = -term1_num / denom_Z + term2_num / denom_Z
    return score


# ==========================================
# 9. SAMPLER CONFIG MACHINERY
# ==========================================
from collections import OrderedDict


def _normalize_sampler_key(name):
    if name is None:
        raise ValueError('Sampler/init name cannot be None.')
    key = str(name).strip().lower().replace('_', ' ').replace('-', ' ')
    return ' '.join(key.split())


INIT_ALIASES = {
    _normalize_sampler_key('prior'): 'prior',
    _normalize_sampler_key('tweedie'): 'tweedie',
    _normalize_sampler_key('blend'): 'blend_posterior',
    _normalize_sampler_key('blend_posterior'): 'blend_posterior',
    _normalize_sampler_key('ref-laplace'): 'ref_laplace',
    _normalize_sampler_key('ref_laplace'): 'ref_laplace',
    _normalize_sampler_key('ref laplace'): 'ref_laplace',
    _normalize_sampler_key('hlsi'): 'hlsi_posterior',
    _normalize_sampler_key('hlsi_posterior'): 'hlsi_posterior',
    _normalize_sampler_key('ce-hlsi'): 'ce_hlsi',
    _normalize_sampler_key('ce_hlsi'): 'ce_hlsi',
    _normalize_sampler_key('ce hlsi'): 'ce_hlsi',
    _normalize_sampler_key('leaf-hlsi'): 'leaf_hlsi',
    _normalize_sampler_key('leaf_hlsi'): 'leaf_hlsi',
    _normalize_sampler_key('leaf hlsi'): 'leaf_hlsi',
    _normalize_sampler_key('leaf-ce-hlsi'): 'leaf_ce_hlsi',
    _normalize_sampler_key('leaf_ce_hlsi'): 'leaf_ce_hlsi',
    _normalize_sampler_key('leaf ce hlsi'): 'leaf_ce_hlsi',
    _normalize_sampler_key('leaf-ce'): 'leaf_ce_hlsi',
    _normalize_sampler_key('tl-hlsi'): 'tl_hlsi',
    _normalize_sampler_key('tl_hlsi'): 'tl_hlsi',
    _normalize_sampler_key('tl hlsi'): 'tl_hlsi',
    _normalize_sampler_key('gate-hlsi'): 'tl_hlsi',
    _normalize_sampler_key('gate_hlsi'): 'tl_hlsi',
    _normalize_sampler_key('gate hlsi'): 'tl_hlsi',
    _normalize_sampler_key('mod-hlsi'): 'tl_hlsi',
    _normalize_sampler_key('mod_hlsi'): 'tl_hlsi',
    _normalize_sampler_key('mod hlsi'): 'tl_hlsi',
    _normalize_sampler_key('tempered-likelihood-hlsi'): 'tl_hlsi',
    _normalize_sampler_key('leaf-tl-hlsi'): 'leaf_tl_hlsi',
    _normalize_sampler_key('leaf_tl_hlsi'): 'leaf_tl_hlsi',
    _normalize_sampler_key('leaf tl hlsi'): 'leaf_tl_hlsi',
    _normalize_sampler_key('leaf-gate-hlsi'): 'leaf_tl_hlsi',
    _normalize_sampler_key('leaf_gate_hlsi'): 'leaf_tl_hlsi',
    _normalize_sampler_key('leaf gate hlsi'): 'leaf_tl_hlsi',
}

INIT_DISPLAY_NAMES = {
    'prior': 'Prior',
    'tweedie': 'Tweedie',
    'blend_posterior': 'Blend',
    'ref_laplace': 'Ref_Laplace',
    'hlsi_posterior': 'HLSI',
    'ce_hlsi': 'CE-HLSI',
    'leaf_hlsi': 'Leaf-HLSI',
    'leaf_ce_hlsi': 'Leaf-CE-HLSI',
    'tl_hlsi': 'TL-HLSI',
    'leaf_tl_hlsi': 'Leaf-TL-HLSI',
    'mala': 'MALA',
}

INIT_WEIGHT_ALIASES = {
    _normalize_sampler_key('none'): 'None',
    _normalize_sampler_key('no weights'): 'None',
    _normalize_sampler_key('unweighted'): 'None',
    _normalize_sampler_key('uniform'): 'None',
    _normalize_sampler_key('l'): 'L',
    _normalize_sampler_key('likelihood'): 'L',
    _normalize_sampler_key('default'): 'L',
    _normalize_sampler_key('posterior'): 'L',
    _normalize_sampler_key('wc'): 'WC',
    _normalize_sampler_key('weight corrected'): 'WC',
    _normalize_sampler_key('pou'): 'PoU',
    _normalize_sampler_key('po u'): 'PoU',
    _normalize_sampler_key('p ou'): 'PoU',
    _normalize_sampler_key('partition of unity'): 'PoU',
}

INIT_WEIGHT_BANK_KEYS = {
    'None': 'log_none_ref',
    'L': 'log_lik_ref',
    'WC': 'log_mass_ref',
    'PoU': 'log_pou_ref',
}


TRANSITION_WEIGHT_ALIASES = {
    _normalize_sampler_key('ou'): 'ou',
    _normalize_sampler_key('target'): 'ou',
    _normalize_sampler_key('target ou'): 'ou',
    _normalize_sampler_key('surrogate'): 'surrogate',
    _normalize_sampler_key('gaussian surrogate'): 'surrogate',
    _normalize_sampler_key('surrogate gaussian'): 'surrogate',
    _normalize_sampler_key('mixture'): 'surrogate',
    _normalize_sampler_key('surrogate mixture'): 'surrogate',
}


def canonicalize_transition_w(name):
    if name is None:
        return 'ou'
    key = _normalize_sampler_key(name)
    if key in TRANSITION_WEIGHT_ALIASES:
        return TRANSITION_WEIGHT_ALIASES[key]
    raise ValueError(f"Unknown transition_w mode: {name!r}. Expected one of: 'ou', 'surrogate'.")


LEGACY_INIT_SPECS = {
    _normalize_sampler_key('wc-hlsi'): ('hlsi_posterior', 'WC'),
    _normalize_sampler_key('wc_hlsi'): ('hlsi_posterior', 'WC'),
    _normalize_sampler_key('wc hlsi'): ('hlsi_posterior', 'WC'),
    _normalize_sampler_key('pou-hlsi'): ('hlsi_posterior', 'PoU'),
    _normalize_sampler_key('pou_hlsi'): ('hlsi_posterior', 'PoU'),
    _normalize_sampler_key('pou hlsi'): ('hlsi_posterior', 'PoU'),
    _normalize_sampler_key('ce-wc-hlsi'): ('ce_hlsi', 'WC'),
    _normalize_sampler_key('ce_wc_hlsi'): ('ce_hlsi', 'WC'),
    _normalize_sampler_key('ce wc hlsi'): ('ce_hlsi', 'WC'),
    _normalize_sampler_key('ce-pou-hlsi'): ('ce_hlsi', 'PoU'),
    _normalize_sampler_key('ce_pou_hlsi'): ('ce_hlsi', 'PoU'),
    _normalize_sampler_key('ce pou hlsi'): ('ce_hlsi', 'PoU'),
    _normalize_sampler_key('leaf-wc-hlsi'): ('leaf_hlsi', 'WC'),
    _normalize_sampler_key('leaf_wc_hlsi'): ('leaf_hlsi', 'WC'),
    _normalize_sampler_key('leaf wc hlsi'): ('leaf_hlsi', 'WC'),
    _normalize_sampler_key('leaf-pou-hlsi'): ('leaf_hlsi', 'PoU'),
    _normalize_sampler_key('leaf_pou_hlsi'): ('leaf_hlsi', 'PoU'),
    _normalize_sampler_key('leaf pou hlsi'): ('leaf_hlsi', 'PoU'),
    _normalize_sampler_key('leaf-ce-wc-hlsi'): ('leaf_ce_hlsi', 'WC'),
    _normalize_sampler_key('leaf_ce_wc_hlsi'): ('leaf_ce_hlsi', 'WC'),
    _normalize_sampler_key('leaf ce wc hlsi'): ('leaf_ce_hlsi', 'WC'),
    _normalize_sampler_key('leaf-ce-pou-hlsi'): ('leaf_ce_hlsi', 'PoU'),
    _normalize_sampler_key('leaf_ce_pou_hlsi'): ('leaf_ce_hlsi', 'PoU'),
    _normalize_sampler_key('leaf ce pou hlsi'): ('leaf_ce_hlsi', 'PoU'),
    _normalize_sampler_key('tl-wc-hlsi'): ('tl_hlsi', 'WC'),
    _normalize_sampler_key('tl_wc_hlsi'): ('tl_hlsi', 'WC'),
    _normalize_sampler_key('tl wc hlsi'): ('tl_hlsi', 'WC'),
    _normalize_sampler_key('tl-pou-hlsi'): ('tl_hlsi', 'PoU'),
    _normalize_sampler_key('tl_pou_hlsi'): ('tl_hlsi', 'PoU'),
    _normalize_sampler_key('tl pou hlsi'): ('tl_hlsi', 'PoU'),
    _normalize_sampler_key('gate-wc-hlsi'): ('tl_hlsi', 'WC'),
    _normalize_sampler_key('gate_wc_hlsi'): ('tl_hlsi', 'WC'),
    _normalize_sampler_key('gate wc hlsi'): ('tl_hlsi', 'WC'),
    _normalize_sampler_key('gate-pou-hlsi'): ('tl_hlsi', 'PoU'),
    _normalize_sampler_key('gate_pou_hlsi'): ('tl_hlsi', 'PoU'),
    _normalize_sampler_key('gate pou hlsi'): ('tl_hlsi', 'PoU'),
    _normalize_sampler_key('leaf-tl-wc-hlsi'): ('leaf_tl_hlsi', 'WC'),
    _normalize_sampler_key('leaf_tl_wc_hlsi'): ('leaf_tl_hlsi', 'WC'),
    _normalize_sampler_key('leaf tl wc hlsi'): ('leaf_tl_hlsi', 'WC'),
    _normalize_sampler_key('leaf-tl-pou-hlsi'): ('leaf_tl_hlsi', 'PoU'),
    _normalize_sampler_key('leaf_tl_pou_hlsi'): ('leaf_tl_hlsi', 'PoU'),
    _normalize_sampler_key('leaf tl pou hlsi'): ('leaf_tl_hlsi', 'PoU'),
}



def canonicalize_init_name(name):
    key = _normalize_sampler_key(name)
    if key not in INIT_ALIASES:
        raise ValueError(f"Unknown sampler/init family: {name}")
    return INIT_ALIASES[key]



def canonicalize_init_weights(name):
    if name is None:
        return 'None'
    key = _normalize_sampler_key(name)
    if key not in INIT_WEIGHT_ALIASES:
        raise ValueError(f"Unknown init_weights mode: {name}")
    return INIT_WEIGHT_ALIASES[key]



def canonicalize_gate_metric_source(name):
    if name is None:
        return 'mu'
    key = _normalize_sampler_key(name)
    metric_aliases = {
        'mu': 'mu',
        'mean': 'mu',
        'posterior mean': 'mu',
        'local mean': 'mu',
        'x': 'x',
        'xref': 'x',
        'reference': 'x',
        'reference point': 'x',
    }
    if key not in metric_aliases:
        raise ValueError(f"Unknown gate metric source: {name}")
    return metric_aliases[key]



def parse_init_spec(name):
    key = _normalize_sampler_key(name)
    if key in LEGACY_INIT_SPECS:
        return LEGACY_INIT_SPECS[key]
    return canonicalize_init_name(name), None



def format_sampler_display_name(init_mode, init_weights='L'):
    if init_mode == 'prior':
        return 'Prior'
    if init_mode == 'ref_laplace':
        return 'Ref_Laplace'
    base = INIT_DISPLAY_NAMES.get(init_mode, str(init_mode))
    if init_weights == 'None':
        return f'{base} [None]'
    if init_weights == 'L':
        return base
    if init_mode == 'hlsi_posterior' and init_weights == 'WC':
        return 'WC-HLSI'
    if init_mode == 'hlsi_posterior' and init_weights == 'PoU':
        return 'PoU-HLSI'
    if init_mode == 'ce_hlsi' and init_weights == 'WC':
        return 'CE-WC-HLSI'
    if init_mode == 'ce_hlsi' and init_weights == 'PoU':
        return 'CE-PoU-HLSI'
    if init_mode == 'leaf_hlsi' and init_weights == 'WC':
        return 'Leaf-WC-HLSI'
    if init_mode == 'leaf_hlsi' and init_weights == 'PoU':
        return 'Leaf-PoU-HLSI'
    if init_mode == 'leaf_ce_hlsi' and init_weights == 'WC':
        return 'Leaf-CE-WC-HLSI'
    if init_mode == 'leaf_ce_hlsi' and init_weights == 'PoU':
        return 'Leaf-CE-PoU-HLSI'
    if init_mode == 'tl_hlsi' and init_weights == 'L':
        return 'TL-HLSI'
    if init_mode == 'tl_hlsi' and init_weights == 'WC':
        return 'TL-WC-HLSI'
    if init_mode == 'tl_hlsi' and init_weights == 'PoU':
        return 'TL-PoU-HLSI'
    if init_mode == 'leaf_tl_hlsi' and init_weights == 'L':
        return 'Leaf-TL-HLSI'
    if init_mode == 'leaf_tl_hlsi' and init_weights == 'WC':
        return 'Leaf-TL-WC-HLSI'
    if init_mode == 'leaf_tl_hlsi' and init_weights == 'PoU':
        return 'Leaf-TL-PoU-HLSI'
    return f"{base} [{init_weights}]"



def canonicalize_ref_source(value):
    """Normalize sampler-tree reference sources.

    ``None``/``'None'``/``'base'``/``'prior'`` mean the ordinary prior reference
    bank. Any other string is interpreted as the name of another sampler whose
    already-generated samples should be used as this sampler's reference bank.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == '':
        return None
    key = _normalize_sampler_key(text)
    if key in {'none', 'null', 'nil', 'base', 'prior', 'default'}:
        return None
    return text


def get_sampler_bank_key(init_mode=None, mala_refs=False, ref_source=None, n_ref=None):
    ref_source = canonicalize_ref_source(ref_source)
    if ref_source is not None:
        n_part = 'all' if n_ref is None else str(int(n_ref))
        return f'source::{ref_source}::{n_part}'
    if bool(mala_refs):
        return 'mala_refs'
    if n_ref is not None:
        return f'base::{int(n_ref)}'
    return 'base'



def get_sampler_precomp_bank(precomp, init_mode=None, mala_refs=False, ref_source=None, n_ref=None):
    """Fetch a precomputed bank while preserving the legacy flat precomp layout."""
    bank_key = get_sampler_bank_key(
        init_mode, mala_refs=mala_refs, ref_source=ref_source, n_ref=n_ref,
    )
    if bank_key in precomp:
        return precomp[bank_key]
    if bank_key.startswith('base::') and 'base' in precomp:
        return precomp['base']
    if 'banks' in precomp and bank_key in precomp['banks']:
        return precomp['banks'][bank_key]
    if 'source_banks' in precomp and bank_key in precomp['source_banks']:
        return precomp['source_banks'][bank_key]
    raise KeyError(
        f"Reference bank '{bank_key}' was requested by init='{init_mode}' "
        f"with ref_source={ref_source!r}, mala_refs={bool(mala_refs)}, n_ref={n_ref!r}, "
        f"but was not precomputed. Available flat keys: {sorted(k for k in precomp.keys() if k not in {'banks', 'source_banks'})}."
    )



def get_sampler_log_weights(init_mode, init_weights, bank):
    init_mode = canonicalize_init_name(init_mode)
    init_weights = canonicalize_init_weights(init_weights)
    use_leaf = init_mode in {'leaf_hlsi', 'leaf_ce_hlsi', 'leaf_tl_hlsi'}
    if use_leaf and init_weights == 'WC':
        bank_key = 'log_mass_leaf_ref'
    elif use_leaf and init_weights == 'PoU':
        bank_key = 'log_pou_leaf_ref'
    else:
        bank_key = INIT_WEIGHT_BANK_KEYS[init_weights]
    if bank_key not in bank:
        raise KeyError(
            f"Reference bank is missing '{bank_key}'. Available keys: {sorted(bank.keys())}"
        )
    return bank[bank_key]



def get_sampler_log_weight_name(init_mode, init_weights):
    init_mode = canonicalize_init_name(init_mode)
    init_weights = canonicalize_init_weights(init_weights)
    use_leaf = init_mode in {'leaf_hlsi', 'leaf_ce_hlsi', 'leaf_tl_hlsi'}
    if use_leaf and init_weights == 'WC':
        return 'log_mass_leaf_ref'
    if use_leaf and init_weights == 'PoU':
        return 'log_pou_leaf_ref'
    return INIT_WEIGHT_BANK_KEYS[init_weights]



def select_local_bank(bank, init_mode, init_weights):
    init_mode = canonicalize_init_name(init_mode)
    init_weights = canonicalize_init_weights(init_weights)
    use_leaf = init_mode in {'leaf_hlsi', 'leaf_ce_hlsi', 'leaf_tl_hlsi'}

    if init_weights == 'PoU':
        if use_leaf:
            required = ['s0_pou_ref', 'P_pou_leaf_ref', 'mu_pou_leaf_ref', 'gated_info_pou_leaf']
            missing = [k for k in required if k not in bank]
            if missing:
                raise KeyError(f"PoU leaf local bank is missing {missing}. Available keys: {sorted(bank.keys())}")
            return {
                'X_ref': bank['X_ref'],
                's0_post_ref': bank['s0_pou_ref'],
                'P_ref': bank['P_pou_leaf_ref'],
                'mu_ref': bank['mu_pou_leaf_ref'],
                'gated_info': bank['gated_info_pou_leaf'],
                'bank_name': 'pou_leaf',
            }
        required = ['s0_pou_ref', 'P_pou_ref', 'mu_pou_ref', 'gated_info_pou']
        missing = [k for k in required if k not in bank]
        if missing:
            raise KeyError(f"PoU local bank is missing {missing}. Available keys: {sorted(bank.keys())}")
        return {
            'X_ref': bank['X_ref'],
            's0_post_ref': bank['s0_pou_ref'],
            'P_ref': bank['P_pou_ref'],
            'mu_ref': bank['mu_pou_ref'],
            'gated_info': bank['gated_info_pou'],
            'bank_name': 'pou',
        }

    if use_leaf:
        required = ['s0_post_ref', 'P_leaf_ref', 'mu_leaf_ref', 'gated_info_leaf']
        missing = [k for k in required if k not in bank]
        if missing:
            raise KeyError(f"Leaf local bank is missing {missing}. Available keys: {sorted(bank.keys())}")
        return {
            'X_ref': bank['X_ref'],
            's0_post_ref': bank['s0_post_ref'],
            'P_ref': bank['P_leaf_ref'],
            'mu_ref': bank['mu_leaf_ref'],
            'gated_info': bank['gated_info_leaf'],
            'bank_name': 'leaf',
        }

    return {
        'X_ref': bank['X_ref'],
        's0_post_ref': bank['s0_post_ref'],
        'P_ref': bank['P_ref'],
        'mu_ref': bank['mu_ref'],
        'gated_info': bank['gated_info'],
        'bank_name': 'base',
    }


def choose_reference_key(samples_dict, sampler_run_info=None, preferred=None):
    if preferred is not None and preferred in samples_dict:
        return preferred
    if sampler_run_info is not None:
        for label, info in sampler_run_info.items():
            if info.get('is_reference', False) and label in samples_dict:
                return label
        for label, info in sampler_run_info.items():
            if info.get('mala_steps', 0) > 0 and label in samples_dict:
                return label
    for label in samples_dict:
        return label
    raise ValueError('No sampler outputs available to choose a reference key.')



def normalize_sampler_config(label, config, default_n_samples, default_dim):
    if isinstance(config, dict) and config.get('_normalized', False):
        return dict(config)

    cfg = dict(config)
    cfg.setdefault('init', 'prior')

    init_raw = cfg['init']
    if str(init_raw).lower() == 'prior':
        cfg['init'] = 'prior'
        implied_weights = None
    else:
        cfg['init'], implied_weights = parse_init_spec(init_raw)

    if cfg['init'] == 'prior':
        cfg['init_weights'] = 'prior'
    elif cfg['init'] == 'ref_laplace':
        cfg['init_weights'] = canonicalize_init_weights(
            cfg.get('init_weights', implied_weights if implied_weights is not None else 'WC')
        )
    else:
        cfg['init_weights'] = canonicalize_init_weights(
            cfg.get('init_weights', implied_weights if implied_weights is not None else 'L')
        )

    # By default, preserve the exact sampler-config key as the user-facing
    # method name throughout summaries, metrics tables, and plots. This keeps
    # distinct gate variants with the same internal canonical init mode from
    # collapsing onto the same displayed label. Users can still override this by
    # passing an explicit `display_name` field in the sampler config.
    cfg.setdefault('display_name', str(label))
    cfg.setdefault('canonical_display_name', format_sampler_display_name(
        cfg['init'], 'L' if cfg['init'] == 'prior' else cfg['init_weights']))
    cfg.setdefault('init_steps', 0 if cfg['init'] in {'prior', 'ref_laplace'} else 200)
    cfg.setdefault('init_tmax', 10.0)
    cfg.setdefault('init_tmin', 10 ** (-2.5))
    cfg.setdefault('log_mean_ess', cfg['init'] not in {'prior', 'ref_laplace'})
    cfg.setdefault('n_samples', default_n_samples)
    cfg['n_samples'] = int(cfg['n_samples'])
    cfg.setdefault('n_ref', None)
    if cfg['n_ref'] is not None:
        cfg['n_ref'] = int(cfg['n_ref'])
    cfg.setdefault('dim', default_dim)
    cfg.setdefault('mala_steps', 0)
    cfg.setdefault('mala_burnin', 0)
    cfg.setdefault('mala_dt', 5e-4)
    cfg.setdefault('precond_mala', False)
    cfg['precond_mala'] = bool(cfg['precond_mala'])
    cfg.setdefault('is_reference', False)

    # Legacy flag retained for old configs. New configs should prefer
    # ref_source='<sampler-name>' or ref_source='None'.
    cfg.setdefault('mala_refs', False)
    cfg['mala_refs'] = bool(cfg['mala_refs'])
    cfg['ref_source'] = canonicalize_ref_source(cfg.get('ref_source', None))

    cfg['transition_w'] = canonicalize_transition_w(cfg.get('transition_w', 'ou'))

    gate_defaults = {
        'hlsi_posterior': dict(gate_rho=0.0, gate_beta=1.0, gate_kappa=0.0),
        'ce_hlsi': dict(gate_rho=1.0, gate_beta=1.0, gate_kappa=0.0),
        'leaf_hlsi': dict(gate_rho=0.0, gate_beta=1.0, gate_kappa=0.0),
        'leaf_ce_hlsi': dict(gate_rho=1.0, gate_beta=1.0, gate_kappa=0.0),
        'tl_hlsi': dict(gate_rho=0.5, gate_beta=1.0, gate_kappa=1.0),
        'leaf_tl_hlsi': dict(gate_rho=0.5, gate_beta=1.0, gate_kappa=1.0),
    }
    for gate_key, gate_val in gate_defaults.get(cfg['init'], {}).items():
        cfg.setdefault(gate_key, gate_val)
    cfg.setdefault('gate_topk', 64)
    cfg['gate_topk'] = int(max(1, cfg['gate_topk']))
    cfg.setdefault('gate_metric_source', 'mu')
    cfg['gate_metric_source'] = canonicalize_gate_metric_source(cfg['gate_metric_source'])

    if cfg['init'] in {'prior', 'ref_laplace'} and cfg['init_steps'] != 0:
        print(f"[normalize_sampler_config] '{label}': init='{cfg['init']}' ignores "
              f"init_steps={cfg['init_steps']}; setting to 0.")
        cfg['init_steps'] = 0

    if cfg['mala_steps'] <= 0:
        cfg['mala_steps'] = 0
        cfg['mala_burnin'] = 0

    cfg['_normalized'] = True
    return cfg



def _normalize_sampler_configs(sampler_configs, default_n_samples=DEFAULT_N_GEN, default_dim=ACTIVE_DIM):
    return OrderedDict(
        (label, normalize_sampler_config(label, cfg, default_n_samples, default_dim))
        for label, cfg in sampler_configs.items()
    )



def _config_uses_reference_bank(cfg):
    return cfg['init'] == 'ref_laplace' or (cfg['init'] != 'prior' and cfg['init_steps'] > 0)



def _config_requires_pou_bank(cfg):
    return _config_uses_reference_bank(cfg) and canonicalize_init_weights(cfg.get('init_weights', 'L')) == 'PoU'



def _resolve_sampler_execution_order(normalized_configs):
    labels = list(normalized_configs.keys())
    label_set = set(labels)
    visiting = set()
    visited = set()
    order = []

    def visit(label):
        if label in visited:
            return
        if label in visiting:
            cycle = ' -> '.join(list(visiting) + [label])
            raise ValueError(f"Cycle detected in sampler ref_source graph: {cycle}")
        visiting.add(label)
        src = normalized_configs[label].get('ref_source')
        if src is not None:
            if src not in label_set:
                raise KeyError(
                    f"Sampler '{label}' declares ref_source={src!r}, but no sampler with that name exists. "
                    f"Available sampler labels: {labels}"
                )
            visit(src)
        visiting.remove(label)
        visited.add(label)
        order.append(label)

    for label in labels:
        visit(label)
    return order



def _validate_sampler_ref_counts(normalized_configs):
    for label, cfg in normalized_configs.items():
        src = cfg.get('ref_source')
        if src is None:
            continue
        requested = cfg.get('n_ref')
        available = normalized_configs[src].get('n_samples')
        if requested is not None and available is not None and int(requested) > int(available):
            raise ValueError(
                f"Sampler '{label}' requests n_ref={requested} from ref_source='{src}', "
                f"but '{src}' is configured to generate only n_samples={available}. "
                f"Increase '{src}' n_samples or lower '{label}' n_ref."
            )



def _finite_reference_samples(samples_cpu, source_label):
    if not torch.is_tensor(samples_cpu):
        samples_cpu = torch.tensor(np.asarray(samples_cpu), dtype=torch.float64)
    samples_cpu = samples_cpu.detach().cpu().to(dtype=torch.float64)
    finite = torch.isfinite(samples_cpu).all(dim=1)
    n_bad = int((~finite).sum().item())
    if n_bad > 0:
        print(f"  [ref_source={source_label}] dropping {n_bad} non-finite samples before bank construction")
    samples_cpu = samples_cpu[finite]
    if samples_cpu.shape[0] == 0:
        raise ValueError(f"ref_source='{source_label}' produced no finite samples for a reference bank.")
    return samples_cpu



def _take_reference_subset(samples_cpu, n_ref, source_label, consumer_label):
    samples_cpu = _finite_reference_samples(samples_cpu, source_label)
    available = int(samples_cpu.shape[0])
    if n_ref is None:
        n_take = available
    else:
        n_take = int(n_ref)
        if n_take > available:
            raise ValueError(
                f"Sampler '{consumer_label}' requested n_ref={n_take} from ref_source='{source_label}', "
                f"but only {available} finite samples are available."
            )
    return samples_cpu[:n_take].contiguous(), n_take



def _precompute_bank_from_reference_samples(x_ref_cpu, prior_model, lik_model, label, compute_pou):
    x_ref = x_ref_cpu.to(device=device, dtype=torch.float64)
    bank = precompute_reference_bank(
        x_ref, prior_model, lik_model, label=label, compute_pou=bool(compute_pou),
    )
    del x_ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bank



def _build_prior_reference_bank(prior_model, lik_model, n_ref, compute_pou, label='base'):
    print(f"Generating {n_ref} reference particles for the {label} prior bank...")
    x_ref = prior_model.sample(int(n_ref))
    bank = precompute_reference_bank(
        x_ref, prior_model, lik_model, label=label, compute_pou=bool(compute_pou),
    )
    del x_ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bank



def _get_or_build_prior_bank(precomp, prior_model, lik_model, n_ref, compute_pou):
    n_ref = int(n_ref)
    key = ('prior', n_ref, bool(compute_pou))
    superset_key = ('prior', n_ref, True)
    precomp.setdefault('bank_cache', {})
    if not compute_pou and superset_key in precomp['bank_cache']:
        return precomp['bank_cache'][superset_key]
    if key not in precomp['bank_cache']:
        label = 'base' if n_ref == precomp.get('default_n_ref') else f'base_n{n_ref}'
        precomp['bank_cache'][key] = _build_prior_reference_bank(
            prior_model, lik_model, n_ref, compute_pou=compute_pou, label=label,
        )
        # Preserve the historical flat key for external callers when this is the default bank.
        if n_ref == precomp.get('default_n_ref') and 'base' not in precomp:
            precomp['base'] = precomp['bank_cache'][key]
    return precomp['bank_cache'][key]



def _get_or_build_source_bank(precomp, samples, source_label, consumer_label, prior_model, lik_model, n_ref, compute_pou):
    source_samples = samples[source_label]
    x_ref_cpu, n_ref_actual = _take_reference_subset(
        source_samples, n_ref, source_label=source_label, consumer_label=consumer_label,
    )
    key = ('source', source_label, n_ref_actual, bool(compute_pou))
    superset_key = ('source', source_label, n_ref_actual, True)
    precomp.setdefault('bank_cache', {})
    if not compute_pou and superset_key in precomp['bank_cache']:
        return precomp['bank_cache'][superset_key], n_ref_actual
    if key not in precomp['bank_cache']:
        bank_label = f'{consumer_label}_ref_from_{source_label}_n{n_ref_actual}'
        precomp['bank_cache'][key] = _precompute_bank_from_reference_samples(
            x_ref_cpu, prior_model, lik_model, label=bank_label, compute_pou=compute_pou,
        )
    return precomp['bank_cache'][key], n_ref_actual



def _select_reference_bank_for_config(label, cfg, samples, precomp, prior_model, lik_model, default_n_ref, compute_pou_default):
    if not _config_uses_reference_bank(cfg):
        return None, None, 0

    ref_source = cfg.get('ref_source')
    needs_pou = _config_requires_pou_bank(cfg)
    # For ordinary prior banks, preserve the historical compute_pou flag. For
    # named source banks, only build PoU objects when the consumer actually needs
    # them; descendants build their own banks from the resulting samples later.
    if ref_source is None:
        n_ref_used = int(default_n_ref if cfg.get('n_ref') is None else cfg['n_ref'])
        bank = _get_or_build_prior_bank(
            precomp, prior_model, lik_model, n_ref=n_ref_used,
            compute_pou=bool(compute_pou_default or needs_pou),
        )
        return bank, 'None', n_ref_used

    bank, n_ref_used = _get_or_build_source_bank(
        precomp, samples, ref_source, label, prior_model, lik_model,
        n_ref=cfg.get('n_ref'), compute_pou=needs_pou,
    )
    return bank, ref_source, n_ref_used



def run_single_sampler_config(label, config, prior_model, lik_model, precomp=None,
                              ref_bank=None, ref_bank_source='None', n_ref_used=0):
    if precomp is None:
        precomp = {}
    cfg = normalize_sampler_config(label, config, DEFAULT_N_GEN, ACTIVE_DIM)
    display_name = cfg['display_name']

    print(f"\n=== Running {display_name} ===")
    print(
        f"  init={cfg['init']} | init_weights={cfg['init_weights']} | transition_w={cfg['transition_w']} | "
        f"ref_source={cfg.get('ref_source')!r} | n_ref={n_ref_used if n_ref_used else cfg.get('n_ref')} | "
        f"init_steps={cfg['init_steps']} | mala_steps={cfg['mala_steps']} | "
        f"mala_burnin={cfg['mala_burnin']} | mala_dt={cfg['mala_dt']}"
    )

    init_samples = None
    ess_trace = None
    bank = None
    local_bank = None
    init_stage_info = None

    if cfg['init'] == 'ref_laplace':
        bank = ref_bank if ref_bank is not None else get_sampler_precomp_bank(
            precomp, cfg['init'], mala_refs=cfg.get('mala_refs', False),
            ref_source=cfg.get('ref_source'), n_ref=cfg.get('n_ref'),
        )
        local_bank = select_local_bank(bank, 'hlsi_posterior', 'L')
        init_samples, init_stage_info = sample_ref_laplace(
            cfg['n_samples'],
            bank,
            log_weight_key='log_mass_ref',
        )
    elif cfg['init'] != 'prior' and cfg['init_steps'] > 0:
        bank = ref_bank if ref_bank is not None else get_sampler_precomp_bank(
            precomp, cfg['init'], mala_refs=cfg.get('mala_refs', False),
            ref_source=cfg.get('ref_source'), n_ref=cfg.get('n_ref'),
        )
        local_bank = select_local_bank(bank, cfg['init'], cfg['init_weights'])
        init_log_weights = get_sampler_log_weights(cfg['init'], cfg['init_weights'], bank)
        init_out = run_sampler_heun(
            cfg['n_samples'], cfg['init'],
            local_bank['X_ref'], local_bank['s0_post_ref'], init_log_weights,
            P_ref=local_bank['P_ref'], mu_ref=local_bank['mu_ref'],
            gated_info=local_bank['gated_info'], init_weights=cfg['init_weights'],
            steps=cfg['init_steps'], dim=cfg['dim'],
            log_mean_ess=cfg['log_mean_ess'],
            t_max=cfg['init_tmax'], t_min=cfg['init_tmin'],
            grad_log_pou_denom_ref=bank.get('grad_log_pou_denom_ref'),
            gate_rho=cfg.get('gate_rho'), gate_beta=cfg.get('gate_beta'),
            gate_kappa=cfg.get('gate_kappa'), gate_topk=cfg.get('gate_topk', 64),
            gate_metric_source=cfg.get('gate_metric_source', 'mu'),
            transition_w=cfg.get('transition_w', 'ou'),
            return_info=True,
        )
        if cfg['log_mean_ess']:
            init_samples, ess_trace, init_stage_info = init_out
        else:
            init_samples, init_stage_info = init_out
    else:
        init_samples = prior_model.sample(cfg['n_samples'])

    final_samples = init_samples
    mala_info = None
    if cfg['mala_steps'] > 0:
        mala_preconditioner = None
        if cfg.get('precond_mala', False):
            ref_laplace_bank = get_sampler_precomp_bank(precomp, 'ref_laplace')
            ref_laplace_component = _build_ref_laplace_component(
                ref_laplace_bank,
                log_weight_key='log_mass_ref',
            )
            mala_preconditioner = _make_frozen_mala_preconditioner(ref_laplace_component)

        final_samples, mala_info = run_mala_sampler(
            cfg['n_samples'], prior_model, lik_model,
            steps=cfg['mala_steps'], dt=cfg['mala_dt'],
            burn_in=cfg['mala_burnin'],
            x_init=init_samples, verbose=True, return_info=True,
            preconditioner=mala_preconditioner,
        )

    run_info = dict(cfg)
    run_info['ref_source'] = ref_bank_source
    run_info['init_reference_bank'] = ref_bank_source
    run_info['n_ref'] = int(n_ref_used) if n_ref_used else 0
    run_info['init_bank'] = local_bank['bank_name'] if cfg['init'] != 'prior' and local_bank is not None else 'base'
    if cfg['init'] == 'ref_laplace':
        run_info['init_log_weights'] = 'log_mass_ref'
    else:
        run_info['init_log_weights'] = get_sampler_log_weight_name(cfg['init'], cfg['init_weights']) if cfg['init'] != 'prior' else 'prior'
    run_info['transition_w'] = cfg.get('transition_w', 'ou')
    if cfg['init'] != 'prior':
        run_info['gate_family'] = resolve_hlsi_gate_law(
            cfg['init'],
            gate_rho=cfg.get('gate_rho'), gate_beta=cfg.get('gate_beta'), gate_kappa=cfg.get('gate_kappa'),
            gate_topk=cfg.get('gate_topk', 64), gate_metric_source=cfg.get('gate_metric_source', 'mu'),
        ).family if cfg['init'] in {'hlsi_posterior', 'ce_hlsi', 'leaf_hlsi', 'leaf_ce_hlsi', 'tl_hlsi', 'leaf_tl_hlsi'} else cfg['init']

    if init_stage_info is not None:
        for key, value in init_stage_info.items():
            run_info[f'init_{key}'] = value
    if mala_info is not None:
        for key, value in mala_info.items():
            if key in {'score_norm_initial', 'score_norm_mean', 'score_norm_final', 'score_norm_max', 'score_norm_num_steps'}:
                run_info[f'mala_{key}'] = value
            else:
                run_info[key] = value

    chosen_score_info = mala_info if mala_info is not None else init_stage_info
    if chosen_score_info is not None:
        run_info['score_norm'] = float(chosen_score_info.get('score_norm_final', np.nan))
        run_info['score_norm_initial'] = float(chosen_score_info.get('score_norm_initial', np.nan))
        run_info['score_norm_mean'] = float(chosen_score_info.get('score_norm_mean', np.nan))
        run_info['score_norm_final'] = float(chosen_score_info.get('score_norm_final', np.nan))
        run_info['score_norm_max'] = float(chosen_score_info.get('score_norm_max', np.nan))
    else:
        run_info['score_norm'] = float('nan')
        run_info['score_norm_initial'] = float('nan')
        run_info['score_norm_mean'] = float('nan')
        run_info['score_norm_final'] = float('nan')
        run_info['score_norm_max'] = float('nan')

    n_ref_local = int(local_bank['X_ref'].shape[0]) if local_bank is not None else 0
    run_info.update(_estimate_sampler_pde_eval_counts(cfg, n_ref=n_ref_local, n_samples=cfg['n_samples']))

    return final_samples.detach().cpu(), ess_trace, run_info



def run_sampler_suite(sampler_configs, prior_model, lik_model, precomp):
    """Legacy sequential runner.

    For tree-structured ``ref_source`` configs, prefer
    ``run_tree_sampler_suite`` or ``run_standard_sampler_pipeline``.
    """
    samples = OrderedDict()
    ess_logs = OrderedDict()
    run_info = OrderedDict()

    for label, cfg in sampler_configs.items():
        t_start = time.time()
        samps, ess_trace, info = run_single_sampler_config(label, cfg, prior_model, lik_model, precomp)
        elapsed = time.time() - t_start
        samples[label] = samps
        if ess_trace is not None and len(ess_trace.get('t', [])) > 0:
            ess_logs[label] = ess_trace
        info = dict(info)
        info['runtime_seconds'] = elapsed
        run_info[label] = info
        print(f"{label}: {elapsed:.2f}s")

    return samples, ess_logs, run_info


DEFAULT_MALA_REF_CONFIG = {
    'init': 'prior',
    'init_steps': 0,
    'mala_steps': 1000,
    'mala_burnin': 200,
    'mala_dt': 1e-4,
    'is_reference': True,
}



def build_precomp(prior_model, lik_model, n_ref=10000, build_gnl_banks=False, compute_pou=True,
                  build_mala_refs=False, mala_ref_config=None, compute_pou_mala_refs=False):
    """Build legacy precomp banks.

    New sampler-tree execution builds source-dependent banks lazily, but this
    function is retained for scripts that still call ``build_precomp`` and
    ``run_sampler_suite`` directly.
    """
    print(f"Generating {n_ref} reference particles for the base bank...")
    t0 = time.time()
    x_ref_base = prior_model.sample(n_ref)
    base_bank = precompute_reference_bank(
        x_ref_base, prior_model, lik_model, label='base', compute_pou=compute_pou,
    )
    precomp = {
        'base': base_bank,
        'default_n_ref': int(n_ref),
        'bank_cache': {('prior', int(n_ref), bool(compute_pou)): base_bank},
    }

    if build_mala_refs:
        mala_cfg = dict(DEFAULT_MALA_REF_CONFIG)
        if mala_ref_config is not None:
            mala_cfg.update(mala_ref_config)
        print(
            f"Generating {n_ref} dedicated MALA reference particles for mala_refs bank "
            f"(steps={mala_cfg['mala_steps']}, burn_in={mala_cfg['mala_burnin']}, dt={mala_cfg['mala_dt']})."
        )
        x_ref_mala, mala_ref_info = run_mala_sampler(
            n_ref, prior_model, lik_model,
            steps=int(mala_cfg['mala_steps']),
            dt=float(mala_cfg['mala_dt']),
            burn_in=int(mala_cfg['mala_burnin']),
            x_init=None, verbose=True, return_info=True,
        )
        mala_bank = precompute_reference_bank(
            x_ref_mala, prior_model, lik_model, label='mala_refs',
            compute_pou=bool(compute_pou_mala_refs),
        )
        precomp['mala_refs'] = mala_bank
        precomp['mala_refs_info'] = dict(mala_ref_info)
        del x_ref_mala, mala_ref_info, mala_bank
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if build_gnl_banks:
        gnl_info = build_gnl_factorization(
            prior_model, lik_model,
            pilot_n=GNL_PILOT_N,
            stiff_lambda_cut=GNL_STIFF_LAMBDA_CUT,
            use_dominant_particle_newton=GNL_USE_DOMINANT_PARTICLE_NEWTON,
        )
        x_ref_gnl = sample_gaussian_from_precision(gnl_info['mu_tilde0'], gnl_info['P_tilde0'], n_ref)
        log_lik_res_gnl = eval_gnl_residual_loglik(x_ref_gnl, gnl_info)
        gnl_bank = precompute_reference_bank(
            x_ref_gnl, prior_model, lik_model, label='gnl',
            residual_log_weights=log_lik_res_gnl, compute_pou=compute_pou,
        )
        precomp['gnl'] = gnl_bank
        precomp['gnl_info'] = {
            key: (val.cpu() if torch.is_tensor(val) else val)
            for key, val in gnl_info.items()
            if key in {
                'x_anchor', 'x_star', 'singvals', 'lik_prec', 'post_lam',
                'stiff_mask', 'residual_mask', 'P_tilde0', 'mu_tilde0', 'stiff_lambda_cut'
            }
        }
        del x_ref_gnl, log_lik_res_gnl, gnl_info, gnl_bank
    print(f"Total bank construction time: {time.time() - t0:.2f}s")
    del x_ref_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return precomp



def run_tree_sampler_suite(sampler_configs, prior_model, lik_model, n_ref=10000,
                           build_gnl_banks=False, compute_pou=True):
    """Run a sampler config DAG defined by ``ref_source`` fields.

    Each sampler is run exactly once. If sampler B has ``ref_source='A'``, then
    B's reference bank is precomputed from the already-generated samples of A.
    ``n_ref`` may be specified per child; if omitted for a named source, all
    finite samples from the source are used. For ``ref_source='None'``/missing,
    a prior bank of size ``n_ref`` (global or per-config) is used.
    """
    normalized = _normalize_sampler_configs(sampler_configs, DEFAULT_N_GEN, ACTIVE_DIM)

    # Backward compatibility for the previous one-off mala_refs=True flag. The
    # tree-native spelling is now an explicit MALA sampler plus
    # ref_source='<that sampler>'. For old configs, synthesize one hidden MALA
    # source and point all such consumers to it.
    hidden_labels = set()
    legacy_mala_consumers = [
        (label, cfg) for label, cfg in normalized.items()
        if cfg.get('mala_refs', False) and cfg.get('ref_source') is None
    ]
    if legacy_mala_consumers:
        hidden_label = '__mala_refs__'
        suffix = 0
        while hidden_label in normalized:
            suffix += 1
            hidden_label = f'__mala_refs_{suffix}__'
        hidden_cfg_raw = dict(DEFAULT_MALA_REF_CONFIG)
        hidden_cfg_raw['n_samples'] = max(
            int(cfg.get('n_ref') or n_ref) for _, cfg in legacy_mala_consumers
        )
        hidden_cfg_raw['display_name'] = hidden_label
        hidden_cfg = normalize_sampler_config(hidden_label, hidden_cfg_raw, DEFAULT_N_GEN, ACTIVE_DIM)
        hidden_cfg['_hidden'] = True
        hidden_labels.add(hidden_label)
        new_normalized = OrderedDict([(hidden_label, hidden_cfg)])
        for label, cfg in normalized.items():
            if cfg.get('mala_refs', False) and cfg.get('ref_source') is None:
                cfg = dict(cfg)
                cfg['ref_source'] = hidden_label
            new_normalized[label] = cfg
        normalized = new_normalized

    _validate_sampler_ref_counts(normalized)
    execution_order = _resolve_sampler_execution_order(normalized)
    print("\n=== Sampler execution order ===")
    print(" -> ".join(execution_order))

    precomp = {'default_n_ref': int(n_ref), 'bank_cache': {}}
    if build_gnl_banks:
        # Preserve the previous optional GNL side bank for legacy configs that
        # still fetch it directly from precomp.
        gnl_precomp = build_precomp(
            prior_model, lik_model, n_ref=n_ref,
            build_gnl_banks=True, compute_pou=compute_pou,
            build_mala_refs=False,
        )
        precomp.update(gnl_precomp)
        precomp.setdefault('bank_cache', {}).update(gnl_precomp.get('bank_cache', {}))

    all_samples = OrderedDict()
    samples = OrderedDict()
    ess_logs = OrderedDict()
    run_info = OrderedDict()

    for label in execution_order:
        cfg = normalized[label]
        ref_bank, ref_bank_source, n_ref_used = _select_reference_bank_for_config(
            label, cfg, all_samples, precomp, prior_model, lik_model,
            default_n_ref=n_ref, compute_pou_default=compute_pou,
        )
        t_start = time.time()
        samps, ess_trace, info = run_single_sampler_config(
            label, cfg, prior_model, lik_model, precomp,
            ref_bank=ref_bank, ref_bank_source=ref_bank_source, n_ref_used=n_ref_used,
        )
        elapsed = time.time() - t_start
        all_samples[label] = samps
        if label not in hidden_labels:
            samples[label] = samps
            if ess_trace is not None and len(ess_trace.get('t', [])) > 0:
                ess_logs[label] = ess_trace
            info = dict(info)
            info['runtime_seconds'] = elapsed
            run_info[label] = info
        print(f"{label}: {elapsed:.2f}s")

    return samples, ess_logs, run_info, precomp



def run_standard_sampler_pipeline(prior_model, lik_model, sampler_configs, n_ref=10000,
                                  build_gnl_banks=False, compute_pou=True, mala_ref_config=None):
    # ``mala_ref_config`` is accepted for backward API compatibility; tree-style
    # configs should express that experiment by adding an explicit MALA node and
    # setting ref_source to that node's label.
    samples, ess_logs, sampler_run_info, precomp = run_tree_sampler_suite(
        sampler_configs, prior_model, lik_model,
        n_ref=n_ref, build_gnl_banks=build_gnl_banks, compute_pou=compute_pou,
    )
    display_names = {label: info.get('display_name', label) for label, info in sampler_run_info.items()}
    reference_key = choose_reference_key(samples, sampler_run_info)
    reference_title = display_names.get(reference_key, reference_key)
    return {
        'precomp': precomp,
        'samples': samples,
        'ess_logs': ess_logs,
        'sampler_run_info': sampler_run_info,
        'display_names': display_names,
        'reference_key': reference_key,
        'reference_title': reference_title,
        'n_ref': n_ref,
        'n_ref_by_sampler': {label: int(info.get('n_ref', 0)) for label, info in sampler_run_info.items()},
    }

def summarize_sampler_run(sampler_run_info):
    print('\n=== Config summary ===')
    for label, info in sampler_run_info.items():
        init_mode = info.get('init', 'prior')
        init_weights = info.get('init_weights', 'prior')
        gate_family = info.get('gate_family', '-')
        gate_bits = ''
        if init_mode not in {'prior', 'tweedie', 'blend_posterior', 'ref_laplace'}:
            gate_bits = (
                f" | gate={gate_family}"
                f" | rho={info.get('gate_rho', '-') }"
                f" | beta={info.get('gate_beta', '-') }"
                f" | kappa={info.get('gate_kappa', '-') }"
                f" | topk={info.get('gate_topk', '-') }"
            )
        print(
            f"{label:<16} -> init={init_mode:<14} | init_weights={init_weights:<5} | "
            f"ref_source={info.get('ref_source', 'None')} | n_ref={info.get('n_ref', 0)} | "
            f"weight_tensor={info.get('init_log_weights', 'prior')}{gate_bits}"
        )


def make_posterior_score_fn(lik_model):
    def posterior_score_fn(x):
        with torch.no_grad():
            s_prior = -x
            s_lik = lik_model.grad_log_likelihood(x)
            return s_prior + s_lik
    return posterior_score_fn


def compute_latent_metrics(samples_dict, reference_key, alpha_true_np,
                           prior_model, lik_model, posterior_score_fn,
                           display_names=None, min_valid=50):
    if display_names is None:
        display_names = {label: label for label in samples_dict.keys()}
    print('\n=== Evaluation (latent/coordinate metrics) ===')
    print(
        f"{'Method':<24} | {'RMSE_alpha':<10} | {'RelL2_alpha':<11} | "
        f"{('MMD->' + str(reference_key))[:14]:<14} | {'KSD':<10} | {'KLdiag':<10}"
    )
    print('-' * 95)

    ref_clean = robust_clean_samples(samples_dict[reference_key])
    alpha_true_t = torch.tensor(alpha_true_np, device=device, dtype=torch.float64)
    metrics = {}

    for label, raw in samples_dict.items():
        samps = robust_clean_samples(raw)
        if samps.shape[0] < min_valid:
            continue
        mean_latent = torch.mean(samps, dim=0)
        rmse_alpha = rmse_vec(mean_latent, alpha_true_t)
        rel_alpha = rel_l2_vec(mean_latent, alpha_true_t)
        mmd = compute_mmd_rbf(samps, ref_clean)
        ksd = compute_multiscale_ksd(samps, posterior_score_fn)
        kl = compute_kl_divergence(samps, prior_model, lik_model)
        metrics[label] = dict(
            mean_latent=mean_latent.detach().cpu().numpy(),
            RMSE_alpha=rmse_alpha,
            RelL2_alpha=rel_alpha,
            MMD_to_reference=mmd,
            KSD=ksd,
            KLdiag=kl,
        )
        print(
            f"{display_names.get(label, label):<24} | {rmse_alpha:<10.4f} | "
            f"{rel_alpha:<11.4f} | {mmd:<14.4f} | {ksd:<10.4f} | {kl:<10.4f}"
        )
    return metrics


def compute_field_summary_metrics(samples_dict, metrics, alpha_true_np, true_field,
                                  field_from_latent_fn,
                                  forward_eval_fn=None, y_ref_np=None,
                                  display_names=None, min_valid=10, d_lat=None):
    if display_names is None:
        display_names = {label: label for label in samples_dict.keys()}
    alpha_true_np = np.asarray(alpha_true_np).reshape(-1)
    if d_lat is None:
        d_lat = alpha_true_np.shape[0]
    norm_true = np.linalg.norm(true_field) + 1e-12
    mean_fields = {}
    for label, samps in samples_dict.items():
        samps_clean = get_valid_samples(samps)
        if samps_clean.shape[0] < min_valid:
            continue
        mean_latent = np.mean(samps_clean, axis=0)[:d_lat]
        mean_field = np.asarray(field_from_latent_fn(mean_latent))
        mean_fields[label] = mean_field
        rmse_alpha = rmse_array(mean_latent, alpha_true_np[:d_lat])
        rel_l2_field = float(np.linalg.norm(mean_field - true_field) / norm_true)
        fwd_rel = float('nan')
        if forward_eval_fn is not None and y_ref_np is not None:
            y_pred = np.asarray(forward_eval_fn(mean_latent))
            fwd_rel = float(np.linalg.norm(y_pred - y_ref_np) / (np.linalg.norm(y_ref_np) + 1e-12))
        metrics.setdefault(label, {})
        metrics[label].update(dict(
            mean_latent=mean_latent,
            RMSE_alpha=rmse_alpha,
            RMSE_field=rmse_array(mean_field, true_field),
            Pearson_field=pearson_corr_array(mean_field, true_field),
            RelL2_field=rel_l2_field,
            FwdRelErr=fwd_rel,
        ))
    return mean_fields, metrics


def compute_heldout_predictive_metrics(samples_dict, metrics,
                                       heldout_forward_eval_fn,
                                       y_holdout_obs_np,
                                       noise_std,
                                       display_names=None,
                                       min_valid=10,
                                       cov_regularization=1e-8,
                                       batched_forward_eval_fn=None,
                                       batched_forward_eval_batch_size=None,
                                       print_summary=True):
    """
    Add held-out posterior predictive calibration metrics to an existing metrics dict.

    Metrics added per method:
      - HeldoutPredNLL: average Gaussian posterior-predictive NLL per held-out sensor
      - HeldoutStdResSq: mean squared standardized held-out residual

    The predictive distribution is approximated by a Gaussian whose mean/covariance
    are estimated from posterior predictive samples, with observation noise variance
    noise_std**2 added on top.

    Robustness notes:
      - non-finite predictive samples are dropped row-wise
      - covariance is symmetrized before factorization
      - Cholesky with escalating jitter is used instead of raw eigh/slogdet
      - if the full covariance remains numerically unstable, we fall back to a
        diagonal predictive covariance rather than failing the whole script
    """
    if display_names is None:
        display_names = {label: label for label in samples_dict.keys()}

    y_holdout_obs_np = np.asarray(y_holdout_obs_np, dtype=np.float64).reshape(-1)
    n_holdout = int(y_holdout_obs_np.size)
    if n_holdout == 0:
        if print_summary:
            print('=== Held-out predictive metrics ===')
            print('No held-out observations were provided; skipping held-out predictive metrics.')
        return metrics

    if print_summary:
        print('=== Held-out predictive metrics ===')
        print(
            f"{'Method':<24} | {'HeldoutPredNLL':<16} | {'HeldoutStdResSq':<16} | {'HeldoutStdResRMS':<17}"
        )
        print('-' * 83)

    obs_noise_var = float(noise_std) ** 2
    base_eye = np.eye(n_holdout, dtype=np.float64)

    def _evaluate_pred_samples(alpha_samples):
        alpha_samples = np.asarray(alpha_samples, dtype=np.float64)
        if alpha_samples.ndim != 2:
            raise ValueError(f'Expected alpha_samples to have shape (n_samples, d); got {alpha_samples.shape}.')

        if batched_forward_eval_fn is None:
            return np.stack(
                [np.asarray(heldout_forward_eval_fn(alpha), dtype=np.float64).reshape(-1)
                 for alpha in alpha_samples],
                axis=0,
            )

        batch_size = batched_forward_eval_batch_size
        if batch_size is None or int(batch_size) <= 0:
            return np.asarray(batched_forward_eval_fn(alpha_samples), dtype=np.float64)

        batch_size = int(batch_size)
        pred_chunks = []
        for start in range(0, alpha_samples.shape[0], batch_size):
            stop = min(start + batch_size, alpha_samples.shape[0])
            pred_chunk = np.asarray(batched_forward_eval_fn(alpha_samples[start:stop]), dtype=np.float64)
            pred_chunks.append(pred_chunk)
        return np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros((0, n_holdout), dtype=np.float64)

    def _stable_gaussian_nll(resid, pred_cov, pred_var):
        pred_cov = np.asarray(pred_cov, dtype=np.float64)
        pred_cov = 0.5 * (pred_cov + pred_cov.T)
        pred_cov = np.where(np.isfinite(pred_cov), pred_cov, 0.0)
        diag_floor = np.maximum(np.asarray(pred_var, dtype=np.float64), 1e-18)
        pred_cov = pred_cov.copy()
        pred_cov[np.diag_indices_from(pred_cov)] = np.maximum(
            pred_cov[np.diag_indices_from(pred_cov)], diag_floor
        )

        scale = max(1.0, float(np.mean(diag_floor)))
        jitter = max(float(cov_regularization) * scale, 1e-12)
        max_tries = 8

        for _ in range(max_tries):
            cov_try = pred_cov + jitter * base_eye
            cov_try = 0.5 * (cov_try + cov_try.T)
            try:
                chol = np.linalg.cholesky(cov_try)
                y = np.linalg.solve(chol, resid)
                precision_apply = np.linalg.solve(chol.T, y)
                logdet = float(2.0 * np.sum(np.log(np.clip(np.diag(chol), 1e-300, None))))
                quad = float(resid @ precision_apply)
                return 0.5 * (n_holdout * np.log(2.0 * np.pi) + logdet + quad) / n_holdout, 'full'
            except np.linalg.LinAlgError:
                jitter *= 10.0
            except FloatingPointError:
                jitter *= 10.0

        diag_cov = np.maximum(diag_floor + jitter, 1e-18)
        quad = float(np.sum((resid ** 2) / diag_cov))
        logdet = float(np.sum(np.log(diag_cov)))
        nll = 0.5 * (n_holdout * np.log(2.0 * np.pi) + logdet + quad) / n_holdout
        return nll, 'diag_fallback'

    for label, samps in samples_dict.items():
        samps_clean = np.asarray(get_valid_samples(samps), dtype=np.float64)
        if samps_clean.shape[0] < min_valid:
            continue

        heldout_warning = None
        try:
            pred_samples = _evaluate_pred_samples(samps_clean)
        except Exception as exc:
            if batched_forward_eval_fn is not None and heldout_forward_eval_fn is not None:
                try:
                    pred_samples = np.stack(
                        [np.asarray(heldout_forward_eval_fn(alpha), dtype=np.float64).reshape(-1)
                         for alpha in samps_clean],
                        axis=0,
                    )
                    heldout_warning = f'batched heldout forward eval failed; fell back to per-sample eval: {exc}'
                except Exception as exc_fallback:
                    metrics.setdefault(label, {})
                    metrics[label].update(dict(
                        HeldoutPredNLL=np.nan,
                        HeldoutStdResSq=np.nan,
                        HeldoutStdResRMS=np.nan,
                        HeldoutPredMean=np.full((n_holdout,), np.nan, dtype=np.float64),
                        HeldoutPredVar=np.full((n_holdout,), np.nan, dtype=np.float64),
                        HeldoutPredCovMode='forward_eval_failed',
                        HeldoutPredNumValid=0,
                        HeldoutPredWarning=(
                            f'heldout forward eval failed; batched error: {exc}; fallback error: {exc_fallback}'
                        ),
                    ))
                    if print_summary:
                        print(f"{display_names.get(label, label):<24} | {'nan':<16} | {'nan':<16} | {'nan':<17}")
                    continue
            else:
                metrics.setdefault(label, {})
                metrics[label].update(dict(
                    HeldoutPredNLL=np.nan,
                    HeldoutStdResSq=np.nan,
                    HeldoutStdResRMS=np.nan,
                    HeldoutPredMean=np.full((n_holdout,), np.nan, dtype=np.float64),
                    HeldoutPredVar=np.full((n_holdout,), np.nan, dtype=np.float64),
                    HeldoutPredCovMode='forward_eval_failed',
                    HeldoutPredNumValid=0,
                    HeldoutPredWarning=f'heldout forward eval failed: {exc}',
                ))
                if print_summary:
                    print(f"{display_names.get(label, label):<24} | {'nan':<16} | {'nan':<16} | {'nan':<17}")
                continue

        if pred_samples.ndim != 2 or pred_samples.shape[1] != n_holdout:
            raise ValueError(
                f'Expected predictive samples of shape (n_samples, {n_holdout}), '
                f'got {pred_samples.shape} for label={label!r}.'
            )

        finite_rows = np.all(np.isfinite(pred_samples), axis=1)
        pred_samples = pred_samples[finite_rows]
        n_valid_pred = int(pred_samples.shape[0])
        if n_valid_pred < min_valid:
            metrics.setdefault(label, {})
            metrics[label].update(dict(
                HeldoutPredNLL=np.nan,
                HeldoutStdResSq=np.nan,
                HeldoutStdResRMS=np.nan,
                HeldoutPredMean=np.full((n_holdout,), np.nan, dtype=np.float64),
                HeldoutPredVar=np.full((n_holdout,), np.nan, dtype=np.float64),
                HeldoutPredCovMode='insufficient_valid_predictions',
                HeldoutPredNumValid=n_valid_pred,
                HeldoutPredWarning='too few finite held-out predictions after filtering',
            ))
            if print_summary:
                print(f"{display_names.get(label, label):<24} | {'nan':<16} | {'nan':<16} | {'nan':<17}")
            continue

        pred_mean = np.mean(pred_samples, axis=0)
        resid = y_holdout_obs_np - pred_mean

        ddof = 1 if n_valid_pred > 1 else 0
        pred_var = np.var(pred_samples, axis=0, ddof=ddof) + obs_noise_var
        pred_var = np.maximum(pred_var, 1e-18)
        heldout_std_res_sq = float(np.mean((resid ** 2) / pred_var))
        heldout_std_res_rms = float(np.sqrt(heldout_std_res_sq))

        if n_valid_pred > 1:
            centered = pred_samples - pred_mean[None, :]
            pred_cov = (centered.T @ centered) / float(max(n_valid_pred - 1, 1))
        else:
            pred_cov = np.zeros((n_holdout, n_holdout), dtype=np.float64)
        if pred_cov.ndim == 0:
            pred_cov = np.array([[float(pred_cov)]], dtype=np.float64)
        pred_cov = np.asarray(pred_cov, dtype=np.float64) + obs_noise_var * base_eye

        try:
            heldout_pred_nll, cov_mode = _stable_gaussian_nll(resid, pred_cov, pred_var)
        except Exception as exc:
            heldout_pred_nll = float(np.nan)
            cov_mode = 'nll_failed'
            extra_warning = f'heldout predictive covariance failed: {exc}'
            if heldout_warning is None:
                heldout_warning = extra_warning
            else:
                heldout_warning = f'{heldout_warning}; {extra_warning}'

        metrics.setdefault(label, {})
        metrics[label].update(dict(
            HeldoutPredNLL=heldout_pred_nll,
            HeldoutStdResSq=heldout_std_res_sq,
            HeldoutStdResRMS=heldout_std_res_rms,
            HeldoutPredMean=pred_mean,
            HeldoutPredVar=np.asarray(pred_var, dtype=np.float64),
            HeldoutPredCovMode=cov_mode,
            HeldoutPredNumValid=n_valid_pred,
        ))
        if heldout_warning is not None:
            metrics[label]['HeldoutPredWarning'] = heldout_warning

        if print_summary:
            nll_print = heldout_pred_nll if np.isfinite(heldout_pred_nll) else float('nan')
            print(
                f"{display_names.get(label, label):<24} | {nll_print:<16.6f} | "
                f"{heldout_std_res_sq:<16.6f} | {heldout_std_res_rms:<17.6f}"
            )

    return metrics

def results_method_family(label, info):
    raw_init_mode = info.get('init', label)
    label_l = str(label).lower()
    try:
        init_mode = canonicalize_init_name(raw_init_mode) if str(raw_init_mode).lower() != 'prior' else 'prior'
    except Exception:
        init_mode = str(raw_init_mode)
    if 'mala' in label_l or init_mode == 'prior':
        return 'Prior'
    family_map = {
        'tweedie': 'Tweedie',
        'blend_posterior': 'Blend',
        'ref_laplace': 'Ref_Laplace',
        'hlsi_posterior': 'HLSI',
        'ce_hlsi': 'CE-HLSI',
        'gnl_hlsi': 'HLSI',
        'gnl_ce_hlsi': 'CE-HLSI',
        'tl_hlsi': 'TL-HLSI',
        'leaf_tl_hlsi': 'Leaf-TL-HLSI',
    }
    return family_map.get(init_mode, str(raw_init_mode))


def results_weight_mode(label, info):
    raw_init_mode = info.get('init', label)
    if 'mala' in str(label).lower() or str(raw_init_mode).lower() == 'prior':
        return 'prior'
    try:
        return canonicalize_init_weights(info.get('init_weights', 'L'))
    except Exception:
        return str(info.get('init_weights', 'L'))


def build_results_dataframes(metrics_dict, run_info_dict, n_ref, target_name,
                             display_names=None, reference_name=None):
    if display_names is None:
        display_names = {label: label for label in run_info_dict.keys()}
    metric_rows = [
        'RMSE_alpha', 'RelL2_alpha', 'MMD_to_reference', 'KSD', 'KLdiag',
        'RMSE_field', 'Pearson_field', 'RelL2_field', 'FwdRelErr',
    ]
    for label in run_info_dict.keys():
        metric_dict = metrics_dict.get(label, {})
        for metric_name, metric_value in metric_dict.items():
            if metric_name in metric_rows:
                continue
            if isinstance(metric_value, (int, float, np.floating, np.integer)) and not isinstance(metric_value, bool):
                metric_rows.append(metric_name)
    ordered_methods = [label for label in run_info_dict.keys() if label in metrics_dict]
    results_df = pd.DataFrame(index=metric_rows, columns=ordered_methods, dtype=np.float64)
    results_df.index.name = 'metric'
    runinfo_rows = []
    for label in ordered_methods:
        info = dict(run_info_dict[label])
        metric_dict = metrics_dict.get(label, {})
        for metric_name in metric_rows:
            results_df.loc[metric_name, label] = metric_dict.get(metric_name, np.nan)
        runinfo_rows.append({
            'target': target_name,
            'label': label,
            'display_name': display_names.get(label, label),
            'method': results_method_family(label, info),
            'weight_mode': results_weight_mode(label, info),
            'N_ref': int(n_ref),
            'steps': int(info.get('init_steps', 0)),
            'mala_steps': int(info.get('mala_steps', info.get('steps', 0))),
            'mala_burnin': int(info.get('mala_burnin', info.get('burn_in', 0))),
            'mala_step_size': float(info.get('mala_dt', info.get('dt', np.nan))),
            'score_norm': float(info.get('score_norm', np.nan)),
            'score_norm_initial': float(info.get('score_norm_initial', np.nan)),
            'score_norm_mean': float(info.get('score_norm_mean', np.nan)),
            'score_norm_final': float(info.get('score_norm_final', np.nan)),
            'score_norm_max': float(info.get('score_norm_max', np.nan)),
            'pde_likelihood_evals': int(info.get('pde_likelihood_evals', 0)),
            'pde_score_evals': int(info.get('pde_score_evals', 0)),
            'pde_gn_hessian_evals': int(info.get('pde_gn_hessian_evals', 0)),
            'pde_solve_count': int(info.get('pde_solve_count', 0)),
            'runtime_seconds': float(info.get('runtime_seconds', np.nan)),
            'reference_method': reference_name,
        })
    results_runinfo_df = pd.DataFrame(runinfo_rows)
    return results_df, results_runinfo_df


def save_results_tables(metrics_dict, run_info_dict, n_ref, target_name,
                        display_names=None, reference_name=None):
    results_df, results_runinfo_df = build_results_dataframes(
        metrics_dict, run_info_dict, n_ref=n_ref, target_name=target_name,
        display_names=display_names, reference_name=reference_name,
    )
    results_df_path = os.path.join(RUN_RESULTS_DIR, f'{RUN_RESULTS_STEM}_metrics.csv')
    results_runinfo_df_path = os.path.join(RUN_RESULTS_DIR, f'{RUN_RESULTS_STEM}_runinfo.csv')
    results_df.to_csv(results_df_path)
    results_runinfo_df.to_csv(results_runinfo_df_path, index=False)
    print(f"\nSaved results dataframe to {results_df_path}")
    print(f"Saved run-info dataframe to {results_runinfo_df_path}")
    return results_df, results_runinfo_df, results_df_path, results_runinfo_df_path


def plot_field_reconstruction_grid(samples_dict, mean_fields, reconstruct_field_fn,
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
                                   n_sample_max=1000):
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
    vis_reference_bottom = reference_bottom_panel if reference_bottom_panel is not None else vis_reference_field

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

    im0 = axes[0, 0].imshow(vis_reference_field, cmap=field_cmap, origin='lower', vmin=vmin, vmax=vmax)
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
    axes[3, 0].imshow(vis_reference_bottom, cmap=bottom_cmap, origin='lower')
    if overlay_reference_fn is not None:
        overlay_reference_fn(axes[3, 0])
    axes[3, 0].set_title(reference_bottom_title, fontsize=14)
    axes[3, 0].axis('off')

    for i, label in enumerate(methods_to_plot):
        col = i + 1
        mean_f = np.asarray(mean_fields[label])
        axes[0, col].imshow(mean_f, cmap=field_cmap, origin='lower', vmin=vmin, vmax=vmax)
        if overlay_method_fn is not None:
            overlay_method_fn(axes[0, col])
        axes[0, col].set_title(f"{display_names.get(label, label)}\nMean Posterior", fontsize=18)
        axes[0, col].axis('off')

        err_f = np.abs(mean_f - vis_reference_field)
        axes[1, col].imshow(err_f, cmap='inferno', origin='lower', vmin=0, vmax=max_err)
        if overlay_method_fn is not None:
            overlay_method_fn(axes[1, col])
        err_title = f"Error Map\n(Max: {err_f.max():.2f})" if has_true_field else f"Deviation from {vis_anchor_title}\n(Max: {err_f.max():.2f})"
        axes[1, col].set_title(err_title, fontsize=16)
        axes[1, col].axis('off')

        samps = get_valid_samples(samples_dict[label])[:n_sample_max]
        if samps.shape[0] > 0:
            fields = np.asarray(reconstruct_field_fn(samps))
            std_f = np.std(fields, axis=0)
        else:
            fields = None
            std_f = np.zeros_like(vis_reference_field)
        im_std = axes[2, col].imshow(std_f, cmap='viridis', origin='lower', vmin=0, vmax=max_std)
        if overlay_method_fn is not None:
            overlay_method_fn(axes[2, col])
        axes[2, col].set_title('Posterior std', fontsize=16)
        axes[2, col].axis('off')
        plt.colorbar(im_std, ax=axes[2, col], fraction=0.046, pad=0.04)

        if fields is not None and samps.shape[0] > 0:
            samp_f = fields[-1]
            im_samp = axes[3, col].imshow(samp_f, cmap=sample_cmap, origin='lower', vmin=vmin, vmax=vmax)
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
