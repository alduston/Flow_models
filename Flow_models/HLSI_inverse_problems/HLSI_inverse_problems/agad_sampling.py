# -*- coding: utf-8 -*-
"""Iterative LFGI transport, ratio-flow correction, metrics, and plotting.

Sampler configurations form a recursive dictionary DAG.  A transport node
builds one of the paper score estimators on its input particle law and samples
with the existing stochastic reverse-OU Heun scheme.  A ratio node reconstructs
the endpoint log density and either carries static log weights or moves a fresh,
unweighted cloud with the completed shared-statistic ratio field.

Only the score families used in the current paper are exposed: LFGI, local and
global scalar/matrix blends, Tweedie, and TSI.  Ratio nodes expose exactly two
modes: ``pflow`` and ``static``.  ``pflow`` denotes the moved-particle ratio
node; particle generation deliberately retains this module's stochastic
reverse-SDE integrator, while the deterministic probability flow is retained
for endpoint-density reconstruction.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import math
import platform
import random
import shutil
import time
import warnings
import glob
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pprint import pformat
from typing import Callable, Dict

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
CURVATURE_RIDGE = 1e-6

RUN_TIMESTAMP = None
RUN_RESULTS_ROOT = 'run_results'
RUN_RESULTS_DIR = None
RUN_RESULTS_STEM = None
_PLOT_SAVE_COUNTER = 0
_ORIGINAL_PLT_SHOW = getattr(plt.show, '_run_results_original_show', plt.show)
_RUN_RESULTS_SHOW_IS_ACTIVE = False


def configure_sampling(active_dim=None, default_n_gen=500,
                       hess_min=1e-6, hess_max=1e8,
                       curvature_ridge=1e-6):
    """Set the small collection of numerical defaults shared by all nodes."""
    global ACTIVE_DIM, DEFAULT_N_GEN, HESS_MIN, HESS_MAX
    global CURVATURE_RIDGE

    ACTIVE_DIM = active_dim
    DEFAULT_N_GEN = int(default_n_gen)
    HESS_MIN = float(hess_min)
    HESS_MAX = float(hess_max)
    CURVATURE_RIDGE = float(curvature_ridge)
    if HESS_MIN <= 0.0 or HESS_MAX < HESS_MIN:
        raise ValueError(f'Expected 0 < hess_min <= hess_max; got {HESS_MIN}, {HESS_MAX}.')
    if CURVATURE_RIDGE <= 0.0:
        raise ValueError('curvature_ridge must be positive.')


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


def save_reproducibility_log(title='Iterative LFGI run reproducibility log', config=None, extra_sections=None):
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


def _copy_orphan_run_artifact(path, run_dir_abs, copied):
    """Copy a run artifact saved outside RUN_RESULTS_DIR back into the run dir."""
    try:
        if not path or not os.path.isfile(path):
            return
        src = os.path.abspath(path)
        if src.startswith(run_dir_abs + os.sep):
            return
        if src.lower().endswith(('.zip', '.tmp')):
            return
        dst = os.path.join(run_dir_abs, os.path.basename(src))
        if os.path.abspath(dst) == src:
            return
        # Preserve a newer destination if it already exists; otherwise copy.
        if os.path.exists(dst):
            try:
                if os.path.getmtime(dst) >= os.path.getmtime(src) and os.path.getsize(dst) == os.path.getsize(src):
                    return
            except OSError:
                pass
        shutil.copy2(src, dst)
        copied.append((src, dst))
    except Exception as exc:
        warnings.warn(f"Could not copy run artifact {path!r} into run-results directory: {exc}", RuntimeWarning)


def _stage_orphan_run_artifacts(run_dir_abs, extra_paths=None, include_cwd_stem_artifacts=True):
    """Best-effort catch for figures/tables accidentally saved outside run dir.

    Most artifacts should already be written directly under RUN_RESULTS_DIR.  Some
    ad-hoc benchmark code historically wrote prefix-matched PNG/PDF/CSV/TXT files
    into the current working directory before the final zip step.  This helper
    copies those files into RUN_RESULTS_DIR so the zip is a complete run snapshot.
    """
    copied = []
    for path in (extra_paths or []):
        _copy_orphan_run_artifact(path, run_dir_abs, copied)
    if include_cwd_stem_artifacts and RUN_RESULTS_STEM:
        artifact_exts = {
            '.png', '.jpg', '.jpeg', '.pdf', '.csv', '.txt', '.json', '.tex',
            '.npz', '.npy', '.pkl', '.pt', '.pth', '.html', '.md', '.log',
        }
        search_dirs = []
        for d in (os.getcwd(), os.path.abspath(RUN_RESULTS_ROOT or '.'), os.path.dirname(run_dir_abs)):
            if d and os.path.isdir(d) and d not in search_dirs:
                search_dirs.append(d)
        for d in search_dirs:
            for path in glob.glob(os.path.join(d, f'{RUN_RESULTS_STEM}*')):
                if os.path.isdir(path):
                    continue
                if os.path.splitext(path)[1].lower() not in artifact_exts:
                    continue
                _copy_orphan_run_artifact(path, run_dir_abs, copied)
    if copied:
        print(f'Staged {len(copied)} orphan run artifact(s) into {run_dir_abs}')
    return copied


def _write_run_results_manifest(run_dir_abs, copied_orphans=None):
    """Write a manifest of every file that will be included in the run zip."""
    if RUN_RESULTS_STEM:
        manifest_path = os.path.join(run_dir_abs, f'{RUN_RESULTS_STEM}_artifact_manifest.txt')
    else:
        manifest_path = os.path.join(run_dir_abs, 'artifact_manifest.txt')
    lines = [
        'Run artifact manifest',
        '=' * 72,
        f'run_results_dir = {run_dir_abs}',
        f'run_results_stem = {RUN_RESULTS_STEM}',
        '',
    ]
    copied_orphans = copied_orphans or []
    if copied_orphans:
        lines.append('Orphan artifacts copied into run-results directory before zipping')
        lines.append('-' * 72)
        for src, dst in copied_orphans:
            lines.append(f'{src} -> {dst}')
        lines.append('')
    rows = []
    for root, _, files in os.walk(run_dir_abs):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, run_dir_abs)
            try:
                stat = os.stat(path)
                rows.append((rel, stat.st_size, stat.st_mtime))
            except OSError:
                rows.append((rel, -1, 0.0))
    rows.sort(key=lambda r: r[0])
    lines.append(f'Files included: {len(rows)}')
    lines.append('-' * 72)
    for rel, size, mtime in rows:
        lines.append(f'{rel}\t{size} bytes\tmtime={mtime:.6f}')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return manifest_path


def zip_run_results_dir(extra_paths=None, include_cwd_stem_artifacts=True, write_manifest=True):
    """Create a complete zip snapshot of the current run-results directory.

    This intentionally does not rely on `shutil.make_archive`, because a few
    benchmark scripts save final diagnostics (for example density-energy grids,
    per-method CSV/NPZ files, dashboard PDFs, or ad-hoc figures) at different
    points in the shutdown sequence.  Before zipping we:
      1. force-save all still-open matplotlib figures into RUN_RESULTS_DIR;
      2. copy any prefix-matched artifacts accidentally written outside the run
         directory back into RUN_RESULTS_DIR;
      3. write an artifact manifest; and
      4. zip every file currently under RUN_RESULTS_DIR recursively.
    """
    if RUN_RESULTS_DIR is None:
        raise RuntimeError('init_run_results must be called before zip_run_results_dir.')
    _save_all_open_figures_to_run_results()
    run_dir_abs = os.path.abspath(RUN_RESULTS_DIR)
    os.makedirs(run_dir_abs, exist_ok=True)
    copied = _stage_orphan_run_artifacts(
        run_dir_abs,
        extra_paths=extra_paths,
        include_cwd_stem_artifacts=include_cwd_stem_artifacts,
    )
    if write_manifest:
        manifest_path = _write_run_results_manifest(run_dir_abs, copied_orphans=copied)
        print(f'Wrote run artifact manifest to {manifest_path}')

    zip_path = run_dir_abs + '.zip'
    tmp_zip_path = zip_path + '.tmp'
    if os.path.exists(tmp_zip_path):
        os.remove(tmp_zip_path)
    base_arcdir = os.path.basename(run_dir_abs)
    n_files = 0
    with zipfile.ZipFile(tmp_zip_path, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for root, _, files in os.walk(run_dir_abs):
            files = sorted(files)
            for name in files:
                path = os.path.join(root, name)
                if not os.path.isfile(path):
                    continue
                # The zip is outside run_dir_abs by construction; this is an extra guard
                # for manual calls where paths may be unusual.
                if os.path.abspath(path) in {os.path.abspath(zip_path), os.path.abspath(tmp_zip_path)}:
                    continue
                rel = os.path.relpath(path, run_dir_abs)
                arcname = os.path.join(base_arcdir, rel)
                zf.write(path, arcname=arcname)
                n_files += 1
    os.replace(tmp_zip_path, zip_path)
    print(f'Compressed run-results directory to {zip_path} ({n_files} files)')
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


# ==========================================
# Paper sampler core
# ==========================================

SCORE_METHODS = (
    'lfgi',
    'local_scalar_blend',
    'local_matrix_blend',
    'global_scalar_blend',
    'global_matrix_blend',
    'tweedie',
    'tsi',
)
RATIO_MODES = ('pflow', 'static')
BANK_COUPLINGS = ('shared', 'prefix', 'independent')
DIVERGENCE_MODES = ('auto', 'analytic', 'hutchinson', 'coordinate_fd')


def _canonical_token(value):
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def canonicalize_score_method(value):
    method = _canonical_token(value)
    if method not in SCORE_METHODS:
        raise ValueError(
            f"Unknown score method {value!r}. Expected one of {', '.join(SCORE_METHODS)}."
        )
    return method


def canonicalize_ratio_mode(value):
    mode = _canonical_token(value)
    if mode not in RATIO_MODES:
        raise ValueError(
            f"Unknown ratio mode {value!r}. Expected one of {', '.join(RATIO_MODES)}."
        )
    return mode


def canonicalize_bank_coupling(value):
    coupling = _canonical_token(value)
    if coupling not in BANK_COUPLINGS:
        raise ValueError(
            f"Unknown bank coupling {value!r}. Expected one of {', '.join(BANK_COUPLINGS)}."
        )
    return coupling


def canonicalize_divergence_mode(value):
    mode = _canonical_token(value)
    aliases = {'coordinate': 'coordinate_fd', 'finite_difference': 'coordinate_fd'}
    mode = aliases.get(mode, mode)
    if mode not in DIVERGENCE_MODES:
        raise ValueError(
            f"Unknown divergence mode {value!r}. Expected one of {', '.join(DIVERGENCE_MODES)}."
        )
    return mode


def _coerce_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if key in {'0', 'false', 'no', 'n', 'off'}:
            return False
    return bool(value)


def _canonical_ref_source(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {'', 'none', 'null', 'prior'}:
        return None
    return text


def _sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


def _safe_symmetric_eigh(A, label='matrix'):
    A = _sym(torch.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12))
    try:
        return torch.linalg.eigh(A)
    except RuntimeError as exc:
        d = int(A.shape[-1])
        eye = torch.eye(d, device=A.device, dtype=A.dtype)
        scale = torch.diagonal(A, dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)
        jitter = (1e-10 + 1e-8 * scale).unsqueeze(-1)
        try:
            return torch.linalg.eigh(A + jitter * eye)
        except RuntimeError as retry_exc:
            raise RuntimeError(f'Symmetric eigendecomposition failed for {label}.') from retry_exc


def _canonicalize_time(t):
    if torch.is_tensor(t):
        t = float(t.detach().cpu().item())
    return max(float(t), 1e-4)


def _alpha_gamma(t, target_device=None, dtype=torch.float64):
    t_val = _canonicalize_time(t)
    dev = device if target_device is None else target_device
    alpha = torch.as_tensor(math.exp(-t_val), device=dev, dtype=dtype)
    gamma = torch.as_tensor(1.0 - math.exp(-2.0 * t_val), device=dev, dtype=dtype)
    return alpha, torch.clamp(gamma, min=1e-12)


def centered_log_weights(logw):
    logw = torch.as_tensor(logw, dtype=torch.float64).reshape(-1)
    if logw.numel() == 0:
        return logw
    if not bool(torch.isfinite(logw).all().item()):
        raise ValueError('Log weights must be finite.')
    return logw - torch.mean(logw)


def global_log_weight_ess(logw):
    logw = torch.as_tensor(logw, dtype=torch.float64).reshape(-1)
    if logw.numel() == 0:
        return float('nan')
    shifted = logw - torch.max(logw)
    w = torch.exp(shifted)
    return float(((w.sum() ** 2) / torch.clamp(torch.sum(w * w), min=1e-30)).item())


@dataclass
class ReferenceBank:
    """Fixed finite bank carrying the current law and target-side information."""

    X_ref: torch.Tensor
    s0_post_ref: torch.Tensor
    P_ref: torch.Tensor
    log_weights: torch.Tensor
    log_target_ref: torch.Tensor
    name: str
    curvature_stats: Dict[str, float]

    @property
    def n(self):
        return int(self.X_ref.shape[0])

    @property
    def d(self):
        return int(self.X_ref.shape[1])


def _finite_cpu_samples(X_ref, label):
    X = torch.as_tensor(X_ref).detach().cpu().to(dtype=torch.float64)
    if X.ndim != 2 or X.shape[0] <= 0:
        raise ValueError(f'{label} must have shape [n,d] with n > 0; got {tuple(X.shape)}.')
    if not bool(torch.isfinite(X).all().item()):
        raise ValueError(f'{label} contains non-finite samples.')
    return X.contiguous()


def precompute_reference_bank(X_ref, prior_model, lik_model, label='reference',
                              log_weights=None):
    """Evaluate the target score and the existing band-regularized GN curvature.

    This intentionally keeps the curvature convention from the previous
    ``sampling.py``: ``P_raw = I - Hessian(log likelihood)``; eigen-directions
    outside ``[HESS_MIN, HESS_MAX]`` are discarded and a ``CURVATURE_RIDGE``
    identity term is added.
    """
    X_cpu = _finite_cpu_samples(X_ref, label)
    n, d = X_cpu.shape
    if log_weights is None:
        logw_cpu = torch.zeros(n, dtype=torch.float64)
    else:
        logw_cpu = centered_log_weights(log_weights).cpu()
        if logw_cpu.numel() != n:
            raise ValueError(
                f'{label}: log_weights has length {logw_cpu.numel()}, expected {n}.'
            )

    print(f'Precomputing {label} reference bank with {n} particles...')
    t0 = time.time()
    score_chunks = []
    target_chunks = []
    curvature_chunks = []
    batch_lik = max(1, int(getattr(lik_model, 'grad_batch_size', 25)))
    batch_hess = max(1, int(getattr(lik_model, 'hess_batch_size', 2)))

    with torch.no_grad():
        for start in range(0, n, batch_lik):
            x = X_cpu[start:start + batch_lik].to(device=device, dtype=torch.float64)
            log_prior = prior_model.log_prob(x)
            score_prior = prior_model.score0(x)
            log_lik = lik_model.log_likelihood(x)
            grad_lik = lik_model.grad_log_likelihood(x)
            score_chunks.append((score_prior + grad_lik).detach().cpu())
            target_chunks.append((log_prior + log_lik).detach().cpu())

        eye = torch.eye(d, device=device, dtype=torch.float64).unsqueeze(0)
        below_total = above_total = negative_total = trusted_total = 0.0
        for start in range(0, n, batch_hess):
            x = X_cpu[start:start + batch_hess].to(device=device, dtype=torch.float64)
            hess_lik = lik_model.hess_log_likelihood(x)
            P_raw = _sym(eye - hess_lik)
            eigvals, eigvecs = _safe_symmetric_eigh(P_raw, label=f'{label} curvature')
            trusted = (eigvals >= HESS_MIN) & (eigvals <= HESS_MAX)
            prec_eig = torch.where(trusted, eigvals, torch.zeros_like(eigvals))
            P = torch.einsum('nij,nj,nkj->nik', eigvecs, prec_eig, eigvecs)
            P = P + CURVATURE_RIDGE * eye
            curvature_chunks.append(P.detach().cpu())
            below_total += float((eigvals < HESS_MIN).sum().item())
            above_total += float((eigvals > HESS_MAX).sum().item())
            negative_total += float((eigvals < -HESS_MIN).sum().item())
            trusted_total += float(trusted.sum().item())

    bank = ReferenceBank(
        X_ref=X_cpu,
        s0_post_ref=torch.cat(score_chunks, dim=0).contiguous(),
        P_ref=torch.cat(curvature_chunks, dim=0).contiguous(),
        log_weights=logw_cpu.contiguous(),
        log_target_ref=torch.cat(target_chunks, dim=0).contiguous(),
        name=str(label),
        curvature_stats={
            'mean_in_band': trusted_total / float(n),
            'mean_below_band': below_total / float(n),
            'mean_above_band': above_total / float(n),
            'mean_negative': negative_total / float(n),
        },
    )
    print(
        f"  [{label}] curvature: {bank.curvature_stats['mean_in_band']:.1f}/{d} in band; "
        f"bank time {time.time() - t0:.2f}s"
    )
    return bank


@dataclass
class ScoreComponents:
    score: torch.Tensor
    gate: torch.Tensor
    b_q: torch.Tensor
    c_q: torch.Tensor
    conditional_weights: torch.Tensor
    particle_b: torch.Tensor
    particle_c: torch.Tensor


class ScoreField:
    """One of the seven paper score estimators on fixed score/gate banks."""

    def __init__(self, signal_bank, method, gate_bank=None, *, eval_chunk=64,
                 matrix_blend_center=True, matrix_blend_ridge=1e-8,
                 matrix_blend_ridge_rel=1e-6, matrix_blend_sym_gate=False,
                 matrix_blend_gate_clip=1e6, global_blend_clamp=True):
        if not isinstance(signal_bank, ReferenceBank):
            raise TypeError('signal_bank must be a ReferenceBank.')
        if gate_bank is None:
            gate_bank = signal_bank
        if signal_bank.d != gate_bank.d:
            raise ValueError('Signal and gate banks must have the same dimension.')
        self.signal_bank = signal_bank
        self.gate_bank = gate_bank
        self.method = canonicalize_score_method(method)
        self.d = signal_bank.d
        self.eval_chunk = max(1, int(eval_chunk))
        self.matrix_blend_center = bool(matrix_blend_center)
        self.matrix_blend_ridge = float(matrix_blend_ridge)
        self.matrix_blend_ridge_rel = float(matrix_blend_ridge_rel)
        self.matrix_blend_sym_gate = bool(matrix_blend_sym_gate)
        self.matrix_blend_gate_clip = matrix_blend_gate_clip
        self.global_blend_clamp = bool(global_blend_clamp)
        self._global_moment = None
        self._device_bank_cache = {}
        self._device_precision_cache = {}

    def _bank_tensors(self, bank, target_device, dtype):
        key = (id(bank), str(target_device), dtype)
        cached = self._device_bank_cache.get(key)
        if cached is None:
            cached = {
                'x': bank.X_ref.to(target_device, non_blocking=True, dtype=dtype),
                's0': bank.s0_post_ref.to(target_device, non_blocking=True, dtype=dtype),
                'logw': bank.log_weights.to(target_device, non_blocking=True, dtype=dtype),
            }
            self._device_bank_cache[key] = cached
        return cached

    def _bank_precision(self, bank, target_device, dtype):
        key = (id(bank), str(target_device), dtype)
        cached = self._device_precision_cache.get(key)
        if cached is None:
            cached = bank.P_ref.to(target_device, non_blocking=True, dtype=dtype)
            self._device_precision_cache[key] = cached
        return cached

    def _weights_and_signals(self, y, t, bank):
        alpha, gamma = _alpha_gamma(t, target_device=y.device, dtype=y.dtype)
        tensors = self._bank_tensors(bank, y.device, y.dtype)
        x = tensors['x']
        s0 = tensors['s0']
        logw0 = tensors['logw']
        diff = y[:, None, :] - alpha * x[None, :, :]
        logits = -0.5 * torch.sum(diff * diff, dim=-1) / gamma + logw0[None, :]
        logits = logits - torch.max(logits, dim=1, keepdim=True).values
        weights = torch.exp(logits)
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-300)
        b = (alpha * x[None, :, :] - y[:, None, :]) / gamma
        c = s0[None, :, :] / alpha
        return weights, b, c, alpha, gamma

    @staticmethod
    def _means(w, b, c):
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        cbar = torch.sum(w[:, :, None] * c, dim=1)
        return bbar, cbar

    def _lfgi_gate(self, y, t, alpha, gamma, gate_weights=None):
        if gate_weights is None:
            gate_weights, _bg, _cg, _ag, _gg = self._weights_and_signals(
                y, t, self.gate_bank
            )
        P = self._bank_precision(self.gate_bank, y.device, y.dtype)
        Pbar = torch.einsum('bn,nij->bij', gate_weights, P)
        eigvals, eigvecs = _safe_symmetric_eigh(Pbar, label='LFGI conditional curvature')
        eigvals = eigvals.clamp(min=1e-6)
        gate_eig = (alpha * alpha) / torch.clamp(alpha * alpha + gamma * eigvals, min=1e-30)
        return torch.einsum('bik,bk,bjk->bij', eigvecs, gate_eig, eigvecs)

    def _local_scalar_gate(self, y, t, gate_values=None):
        """Existing self-normalized finite-bank scalar blend calculation."""
        if gate_values is None:
            w, b, c, _alpha, _gamma = self._weights_and_signals(y, t, self.gate_bank)
        else:
            w, b, c = gate_values
        mu_a = torch.sum(w[:, :, None] * c, dim=1)
        mu_b = torch.sum(w[:, :, None] * b, dim=1)
        w2 = w.square()
        S0 = torch.sum(w2, dim=1, keepdim=True)
        S1a = torch.sum(w2 * torch.sum(c.square(), dim=-1), dim=1, keepdim=True)
        S2a = torch.sum(w2[:, :, None] * c, dim=1)
        S1b = torch.sum(w2 * torch.sum(b.square(), dim=-1), dim=1, keepdim=True)
        S2b = torch.sum(w2[:, :, None] * b, dim=1)
        Sab = torch.sum(w2 * torch.sum(c * b, dim=-1), dim=1, keepdim=True)
        den_sn = torch.clamp(1.0 - S0, min=1e-12)
        Vk = (
            S1a - 2.0 * torch.sum(mu_a * S2a, dim=1, keepdim=True)
            + torch.sum(mu_a.square(), dim=1, keepdim=True) * S0
        ) / den_sn
        Vt = (
            S1b - 2.0 * torch.sum(mu_b * S2b, dim=1, keepdim=True)
            + torch.sum(mu_b.square(), dim=1, keepdim=True) * S0
        ) / den_sn
        C = (
            Sab - torch.sum(mu_a * S2b, dim=1, keepdim=True)
            - torch.sum(mu_b * S2a, dim=1, keepdim=True)
            + torch.sum(mu_a * mu_b, dim=1, keepdim=True) * S0
        ) / den_sn
        tweedie_weight = (Vk - C) / torch.clamp(Vk + Vt - 2.0 * C, min=1e-12)
        return tweedie_weight.clamp(0.0, 0.95)

    def _local_matrix_gate(self, y, t, gate_values=None):
        if gate_values is None:
            w, b, c, _alpha, _gamma = self._weights_and_signals(y, t, self.gate_bank)
        else:
            w, b, c = gate_values
        delta = c - b
        if self.matrix_blend_center:
            bbar = torch.sum(w[:, :, None] * b, dim=1)
            dbar = torch.sum(w[:, :, None] * delta, dim=1)
            b_mom = b - bbar[:, None, :]
            d_mom = delta - dbar[:, None, :]
        else:
            b_mom = b
            d_mom = delta
        M = torch.einsum('bn,bni,bnj->bij', w, d_mom, d_mom)
        N = torch.einsum('bn,bni,bnj->bij', w, b_mom, d_mom)
        M = _sym(torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0))
        N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
        batch, d = M.shape[0], M.shape[-1]
        eye = torch.eye(d, device=y.device, dtype=y.dtype).expand(batch, d, d)
        scale = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1) / float(max(d, 1))
        ridge = self.matrix_blend_ridge + self.matrix_blend_ridge_rel * scale.clamp(min=0.0)
        M_reg = M + ridge[:, None, None] * eye
        try:
            G = torch.linalg.solve(M_reg.transpose(-1, -2), (-N).transpose(-1, -2)).transpose(-1, -2)
        except RuntimeError:
            G = -torch.matmul(N, torch.linalg.pinv(M_reg))
        if not bool(torch.isfinite(G).all().item()):
            fallback = -torch.matmul(N, torch.linalg.pinv(M_reg))
            G = torch.where(torch.isfinite(G), G, fallback)
        G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        if self.matrix_blend_sym_gate:
            G = _sym(G)
        if self.matrix_blend_gate_clip is not None:
            clip = float(self.matrix_blend_gate_clip)
            if math.isfinite(clip) and clip > 0.0:
                G = G.clamp(min=-clip, max=clip)
        return G

    def _global_score_moment(self, target_device, dtype):
        if self._global_moment is None:
            tensors = self._bank_tensors(self.gate_bank, device, torch.float64)
            s0 = tensors['s0']
            logw = tensors['logw']
            w = torch.softmax(logw, dim=0)
            moment = s0.transpose(0, 1) @ (w[:, None] * s0)
            self._global_moment = _sym(
                torch.nan_to_num(moment, nan=0.0, posinf=0.0, neginf=0.0)
            ).detach().cpu()
        return self._global_moment.to(target_device, dtype=dtype, non_blocking=True)

    def _global_scalar_gate(self, y, alpha, gamma):
        Ipi = self._global_score_moment(y.device, y.dtype)
        Ipi = Ipi + 1e-12 * torch.eye(self.d, device=y.device, dtype=y.dtype)
        tr_ipi = torch.diagonal(Ipi).sum().clamp(min=0.0)
        a = gamma * tr_ipi / torch.clamp(alpha * alpha * float(self.d) + gamma * tr_ipi, min=1e-300)
        if self.global_blend_clamp:
            a = a.clamp(0.0, 1.0)
        eye = torch.eye(self.d, device=y.device, dtype=y.dtype)
        return ((1.0 - a) * eye).expand(y.shape[0], self.d, self.d)

    def _global_matrix_gate(self, y, alpha, gamma):
        Ipi = self._global_score_moment(y.device, y.dtype)
        eigvals, eigvecs = _safe_symmetric_eigh(Ipi, label='global matrix blend moment')
        eigvals = torch.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        scale = float(torch.mean(eigvals).item()) if eigvals.numel() else 0.0
        ridge = self.matrix_blend_ridge + self.matrix_blend_ridge_rel * max(scale, 0.0)
        a_eig = gamma * eigvals / torch.clamp(alpha * alpha + gamma * eigvals + ridge, min=1e-300)
        if self.global_blend_clamp:
            a_eig = a_eig.clamp(0.0, 1.0)
        A = torch.einsum('ik,k,jk->ij', eigvecs, a_eig, eigvecs)
        G = torch.eye(self.d, device=y.device, dtype=y.dtype) - _sym(A)
        return G.expand(y.shape[0], self.d, self.d)

    def components_chunk(self, y, t):
        w, b, c, alpha, gamma = self._weights_and_signals(y, t, self.signal_bank)
        bbar, cbar = self._means(w, b, c)
        batch, d = y.shape
        eye = torch.eye(d, device=y.device, dtype=y.dtype)

        if self.method == 'lfgi':
            G = self._lfgi_gate(
                y, t, alpha, gamma,
                gate_weights=w if self.gate_bank is self.signal_bank else None,
            )
        elif self.method == 'local_scalar_blend':
            a = self._local_scalar_gate(
                y, t, gate_values=(w, b, c) if self.gate_bank is self.signal_bank else None
            )
            G = torch.diag_embed((1.0 - a).expand(batch, d))
        elif self.method == 'local_matrix_blend':
            G = self._local_matrix_gate(
                y, t, gate_values=(w, b, c) if self.gate_bank is self.signal_bank else None
            )
        elif self.method == 'global_scalar_blend':
            G = self._global_scalar_gate(y, alpha, gamma)
        elif self.method == 'global_matrix_blend':
            G = self._global_matrix_gate(y, alpha, gamma)
        elif self.method == 'tweedie':
            G = torch.zeros((batch, d, d), device=y.device, dtype=y.dtype)
        elif self.method == 'tsi':
            G = eye.expand(batch, d, d)
        else:  # guarded by canonicalize_score_method
            raise RuntimeError(f'Unhandled score method {self.method!r}.')

        score = bbar + torch.einsum('bij,bj->bi', G, cbar - bbar)
        return ScoreComponents(
            score=torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
            gate=torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0),
            b_q=bbar,
            c_q=cbar,
            conditional_weights=w,
            particle_b=b,
            particle_c=c,
        )

    @torch.no_grad()
    def estimate(self, y, t):
        y = y.to(device=device, dtype=torch.float64)
        parts = []
        for start in range(0, y.shape[0], self.eval_chunk):
            parts.append(self.components_chunk(y[start:start + self.eval_chunk], t).score)
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def mean_ess(self, y, t):
        values = []
        for start in range(0, y.shape[0], self.eval_chunk):
            w, _b, _c, _a, _g = self._weights_and_signals(
                y[start:start + self.eval_chunk], t, self.signal_bank
            )
            values.append(1.0 / torch.clamp(torch.sum(w.square(), dim=1), min=1e-30))
        return float(torch.cat(values).mean().item())

    def completed_ratio_components_chunk(self, y, t, endpoint_log_tilt):
        native = self.components_chunk(y, t)
        tilt = endpoint_log_tilt.to(y.device, dtype=y.dtype).reshape(-1)
        if tilt.numel() != self.signal_bank.n:
            raise ValueError(
                f'Ratio tilt has length {tilt.numel()}, expected {self.signal_bank.n}.'
            )
        target_logits = torch.log(torch.clamp(native.conditional_weights, min=1e-300)) + tilt[None, :]
        w_pi = torch.softmax(target_logits, dim=1)
        b_pi = torch.sum(w_pi[:, :, None] * native.particle_b, dim=1)
        c_pi = torch.sum(w_pi[:, :, None] * native.particle_c, dim=1)
        b_residual = b_pi - native.b_q
        c_residual = c_pi - native.c_q
        eye = torch.eye(self.d, device=y.device, dtype=y.dtype).expand(y.shape[0], self.d, self.d)
        correction = (
            torch.einsum('bij,bj->bi', eye - native.gate, b_residual)
            + torch.einsum('bij,bj->bi', native.gate, c_residual)
        )
        score = native.score + correction
        direct = b_pi + torch.einsum('bij,bj->bi', native.gate, c_pi - b_pi)
        return {
            'score': torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
            'direct_score': torch.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0),
            'identity_residual': torch.nan_to_num(score - direct, nan=0.0, posinf=0.0, neginf=0.0),
            'target_conditional_ess': 1.0 / torch.clamp(torch.sum(w_pi.square(), dim=1), min=1e-30),
            'gate': native.gate,
            'b_q': native.b_q,
            'b_pi': b_pi,
            'c_q': native.c_q,
            'c_pi': c_pi,
        }

    def _tweedie_score_and_divergence_chunk(self, y, t):
        w, b, c, _alpha, gamma = self._weights_and_signals(y, t, self.signal_bank)
        del c
        bbar = torch.sum(w[:, :, None] * b, dim=1)
        db = b - bbar[:, None, :]
        Cbb = torch.einsum('bn,bni,bnj->bij', w, db, db)
        div = torch.diagonal(Cbb, dim1=-2, dim2=-1).sum(dim=-1) - float(self.d) / gamma
        return (
            torch.nan_to_num(bbar, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _lfgi_score_and_divergence_chunk(self, y, t):
        w, b, c, alpha, gamma = self._weights_and_signals(y, t, self.signal_bank)
        bbar, cbar = self._means(w, b, c)
        if self.gate_bank is self.signal_bank:
            wg, bg = w, b
        else:
            wg, bg, _cg, _ag, _gg = self._weights_and_signals(y, t, self.gate_bank)
        bbar_gate = torch.sum(wg[:, :, None] * bg, dim=1)
        P = self._bank_precision(self.gate_bank, y.device, y.dtype)
        Pbar = torch.einsum('bn,nij->bij', wg, P)
        eigvals, eigvecs = _safe_symmetric_eigh(Pbar, label='LFGI density curvature')
        eigvals = eigvals.clamp(min=1e-6)
        gate_eig = (alpha * alpha) / torch.clamp(alpha * alpha + gamma * eigvals, min=1e-30)
        G = torch.einsum('bik,bk,bjk->bij', eigvecs, gate_eig, eigvecs)
        residual = cbar - bbar
        score = bbar + torch.einsum('bij,bj->bi', G, residual)

        db = b - bbar[:, None, :]
        dc = c - cbar[:, None, :]
        Cbb = torch.einsum('bn,bni,bnj->bij', w, db, db)
        Ccb = torch.einsum('bn,bni,bnj->bij', w, dc, db)
        eye = torch.eye(self.d, device=y.device, dtype=y.dtype).expand(y.shape[0], self.d, self.d)
        Jb = Cbb - eye / gamma
        Jr = Ccb - Jb

        dbg = bg - bbar_gate[:, None, :]
        # sum_n w_n (b_n-bbar) = 0, so the centered-P term can be evaluated
        # without materializing the prohibitive [batch,n,d,d] tensor.
        dPdy = torch.einsum('bn,bna,nuv->bauv', wg, dbg, P)
        Gr = torch.einsum('bij,bj->bi', G, residual)
        gate_trace = torch.einsum('bau,bauv,bv->b', G, dPdy, Gr)
        div = (
            torch.diagonal(Jb, dim1=-2, dim2=-1).sum(dim=-1)
            + torch.einsum('bij,bji->b', G, Jr)
            - gamma / torch.clamp(alpha * alpha, min=1e-12) * gate_trace
        )
        return (
            torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0),
        )

    @torch.no_grad()
    def score_and_divergence(self, y, t, mode='auto', n_probes=1, fd_eps=1e-3):
        mode = canonicalize_divergence_mode(mode)
        y = y.to(device=device, dtype=torch.float64)
        if mode in {'auto', 'analytic'} and self.method in {'lfgi', 'tweedie'}:
            scores, divs = [], []
            for start in range(0, y.shape[0], self.eval_chunk):
                chunk = y[start:start + self.eval_chunk]
                if self.method == 'lfgi':
                    score, div = self._lfgi_score_and_divergence_chunk(chunk, t)
                else:
                    score, div = self._tweedie_score_and_divergence_chunk(chunk, t)
                scores.append(score)
                divs.append(div)
            return torch.cat(scores), torch.cat(divs), 'analytic'
        if mode == 'analytic':
            raise ValueError(f'Analytic divergence is not implemented for {self.method}.')
        effective = 'hutchinson' if mode == 'auto' else mode
        score = self.estimate(y, t)
        div = _finite_difference_divergence(
            self.estimate, y, t, mode=effective, n_probes=n_probes, fd_eps=fd_eps
        )
        return score, div, effective


class CompletedRatioField:
    """Completed shared-statistic ratio correction built on a native estimator."""

    def __init__(self, base_field, endpoint_log_tilt):
        self.base_field = base_field
        self.endpoint_log_tilt = centered_log_weights(endpoint_log_tilt).cpu()
        self.method = base_field.method
        self.d = base_field.d
        self.eval_chunk = base_field.eval_chunk

    @torch.no_grad()
    def estimate(self, y, t):
        y = y.to(device=device, dtype=torch.float64)
        outputs = []
        for start in range(0, y.shape[0], self.eval_chunk):
            outputs.append(self.base_field.completed_ratio_components_chunk(
                y[start:start + self.eval_chunk], t, self.endpoint_log_tilt
            )['score'])
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def mean_ess(self, y, t):
        y = y.to(device=device, dtype=torch.float64)
        values = []
        for start in range(0, y.shape[0], self.eval_chunk):
            values.append(self.base_field.completed_ratio_components_chunk(
                y[start:start + self.eval_chunk], t, self.endpoint_log_tilt
            )['target_conditional_ess'])
        return float(torch.cat(values).mean().item())

    @torch.no_grad()
    def score_and_divergence(self, y, t, mode='auto', n_probes=1, fd_eps=1e-3):
        mode = canonicalize_divergence_mode(mode)
        if mode == 'analytic':
            raise ValueError('Completed ratio fields use finite-difference divergence.')
        effective = 'hutchinson' if mode == 'auto' else mode
        y = y.to(device=device, dtype=torch.float64)
        score = self.estimate(y, t)
        div = _finite_difference_divergence(
            self.estimate, y, t, mode=effective, n_probes=n_probes, fd_eps=fd_eps
        )
        return score, div, effective


def _finite_difference_divergence(score_fn, y, t, mode='hutchinson', n_probes=1, fd_eps=1e-3):
    eps = float(fd_eps)
    if eps <= 0.0:
        raise ValueError('fd_eps must be positive.')
    if mode == 'coordinate_fd':
        div = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        for j in range(y.shape[1]):
            direction = torch.zeros_like(y)
            direction[:, j] = eps
            s_plus = score_fn(y + direction, t)
            s_minus = score_fn(y - direction, t)
            div += (s_plus[:, j] - s_minus[:, j]) / (2.0 * eps)
        return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)
    if mode != 'hutchinson':
        raise ValueError(f'Finite-difference divergence does not support mode={mode!r}.')
    probes = max(1, int(n_probes))
    div = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
    for _ in range(probes):
        z = torch.empty_like(y).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        s_plus = score_fn(y + eps * z, t)
        s_minus = score_fn(y - eps * z, t)
        div += torch.sum((s_plus - s_minus) * z, dim=1) / (2.0 * eps)
    return torch.nan_to_num(div / float(probes), nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def run_reverse_ou_heun(n_samples, score_field, *, steps=200, dim=None,
                        t_max=10.0, t_min=10 ** (-2.5), log_mean_ess=False):
    """Existing stochastic reverse-OU predictor/corrector used by every flow."""
    n_samples = int(n_samples)
    dim = int(score_field.d if dim is None else dim)
    steps = int(steps)
    if n_samples <= 0 or dim <= 0:
        raise ValueError('n_samples and dim must be positive.')
    if steps <= 0:
        raise ValueError('Flow nodes require steps > 0.')
    t_max = float(t_max)
    t_min = max(float(t_min), 1e-4)
    if t_max <= t_min:
        raise ValueError(f'Expected t_max > t_min; got {t_max} <= {t_min}.')

    y = torch.randn(n_samples, dim, device=device, dtype=torch.float64)
    ts = torch.logspace(
        math.log10(t_max), math.log10(t_min), steps + 1,
        device=device, dtype=torch.float64,
    )
    ess_trace = {'t': [], 'mean_ess': []} if log_mean_ess else None
    if ess_trace is not None:
        ess_trace['t'].append(float(ts[0].item()))
        ess_trace['mean_ess'].append(score_field.mean_ess(y, ts[0]))

    score_norm_sum = 0.0
    score_norm_max = 0.0
    score_norm_initial = float('nan')
    for i in range(steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = float((t_cur - t_next).item())
        score_cur = score_field.estimate(y, t_cur)
        cur_norm = float(torch.linalg.vector_norm(score_cur, dim=1).mean().item())
        if i == 0:
            score_norm_initial = cur_norm
        score_norm_sum += cur_norm
        score_norm_max = max(score_norm_max, cur_norm)
        drift_cur = y + 2.0 * score_cur

        noise = torch.randn_like(y)
        y_hat = y + dt * drift_cur + math.sqrt(2.0 * dt) * noise
        score_next = score_field.estimate(y_hat, t_next)
        drift_next = y_hat + 2.0 * score_next
        y = y + 0.5 * dt * (drift_cur + drift_next) + math.sqrt(2.0 * dt) * noise
        if not bool(torch.isfinite(y).all().item()):
            raise FloatingPointError(
                f'Non-finite reverse-flow state at step {i + 1}/{steps} (t={float(t_next):.4g}).'
            )
        if ess_trace is not None:
            ess_trace['t'].append(float(t_next.item()))
            ess_trace['mean_ess'].append(score_field.mean_ess(y, t_next))

    final_score = score_field.estimate(y, ts[-1])
    score_norm_final = float(torch.linalg.vector_norm(final_score, dim=1).mean().item())
    info = {
        'score_norm': score_norm_final,
        'score_norm_initial': score_norm_initial,
        'score_norm_mean': score_norm_sum / float(steps),
        'score_norm_final': score_norm_final,
        'score_norm_max': max(score_norm_max, score_norm_final),
        'score_norm_num_steps': steps,
        'flow_dynamics': 'stochastic_reverse_ou_heun',
        't_min': t_min,
        't_max': t_max,
    }
    if ess_trace is not None:
        ess_trace = {key: np.asarray(value) for key, value in ess_trace.items()}
    return y.detach().cpu(), ess_trace, info


@torch.no_grad()
def estimate_logq_probability_flow(X0, source_score_field, *, t_min=10 ** (-2.5),
                                   t_max=10.0, n_steps=32, n_div_probes=1,
                                   fd_eps=1e-3, divergence_mode='auto',
                                   batch_size=32, return_diagnostics=False):
    """Reconstruct endpoint log density with the existing frozen-score PF ODE."""
    X_cpu = _finite_cpu_samples(X0, 'density evaluation points')
    t_min = max(float(t_min), 1e-4)
    t_max = float(t_max)
    n_steps = max(1, int(n_steps))
    batch_size = max(1, int(batch_size))
    divergence_mode = canonicalize_divergence_mode(divergence_mode)
    if t_max <= t_min:
        raise ValueError(f'Expected density t_max > t_min; got {t_max} <= {t_min}.')
    ts = torch.logspace(
        math.log10(t_min), math.log10(t_max), n_steps + 1,
        device=device, dtype=torch.float64,
    )
    outputs = []
    effective_modes = set()
    for batch_start in range(0, X_cpu.shape[0], batch_size):
        y = X_cpu[batch_start:batch_start + batch_size].to(device=device, dtype=torch.float64)
        div_accum = torch.zeros(y.shape[0], device=device, dtype=torch.float64)
        for i in range(n_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_cur
            score_cur, div_score_cur, kind_cur = source_score_field.score_and_divergence(
                y, t_cur, mode=divergence_mode, n_probes=n_div_probes, fd_eps=fd_eps
            )
            effective_modes.add(kind_cur)
            drift_cur = -(y + score_cur)
            div_drift_cur = -float(y.shape[1]) - div_score_cur
            y_hat = y + dt * drift_cur
            score_next, div_score_next, kind_next = source_score_field.score_and_divergence(
                y_hat, t_next, mode=divergence_mode, n_probes=n_div_probes, fd_eps=fd_eps
            )
            effective_modes.add(kind_next)
            drift_next = -(y_hat + score_next)
            div_drift_next = -float(y.shape[1]) - div_score_next
            y = y + 0.5 * dt * (drift_cur + drift_next)
            div_accum = div_accum + 0.5 * dt * (div_drift_cur + div_drift_next)
        log_terminal = (
            -0.5 * torch.sum(y.square(), dim=1)
            - 0.5 * float(y.shape[1]) * math.log(2.0 * math.pi)
        )
        outputs.append((log_terminal + div_accum).detach().cpu())
    logq = torch.cat(outputs, dim=0)
    if not bool(torch.isfinite(logq).all().item()):
        raise FloatingPointError('Probability-flow density reconstruction produced non-finite log q.')
    diagnostics = {
        'density_divergence_requested': divergence_mode,
        'density_divergence_effective': '+'.join(sorted(effective_modes)),
        'density_steps': n_steps,
        'density_batch_size': batch_size,
        'density_t_min': t_min,
        'density_t_max': t_max,
    }
    if return_diagnostics:
        return logq, diagnostics
    return logq


def _finalize_ratio_log_tilt(raw_log_ratio, *, temperature=1.0, clip=20.0):
    """Keep the previous ratio-weight centering, temperature, and clipping."""
    raw = torch.as_tensor(raw_log_ratio, dtype=torch.float64).detach().cpu().reshape(-1)
    if not bool(torch.isfinite(raw).all().item()):
        raise ValueError('Raw endpoint log ratios must be finite.')
    logw = raw - torch.mean(raw)
    logw = float(temperature) * logw
    if clip is not None:
        clip = float(clip)
        if clip > 0.0 and math.isfinite(clip):
            median = torch.median(logw)
            logw = torch.clamp(logw, median - clip, median + clip)
    return (logw - torch.mean(logw)).contiguous()


_CONFIG_KEYS = {
    'node', 'score', 'ratio_mode', 'ref_source', 'display_name',
    'include_results', 'is_reference', 'n_samples', 'n_ref', 'n_gate',
    'bank_coupling', 'steps', 't_min', 't_max', 'log_mean_ess', 'dim',
    'eval_chunk', 'matrix_blend_center', 'matrix_blend_ridge',
    'matrix_blend_ridge_rel', 'matrix_blend_sym_gate',
    'matrix_blend_gate_clip', 'global_blend_clamp',
    'density_steps', 'density_divergence', 'density_div_probes',
    'density_fd_eps', 'density_batch_size', 'density_t_min', 'density_t_max',
    'ratio_temperature', 'ratio_log_weight_clip',
}


def normalize_sampler_config(label, config, default_n_samples=None, default_dim=None):
    """Normalize one paper-era node and reject removed legacy controls."""
    if not isinstance(config, dict):
        raise TypeError(f'Sampler config {label!r} must be a dictionary.')
    if config.get('_normalized', False):
        return dict(config)
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"Sampler config {label!r} contains removed/unknown keys: {unknown}. "
            f"Supported keys are: {sorted(_CONFIG_KEYS)}."
        )
    cfg = dict(config)
    node = _canonical_token(cfg.get('node', 'transport'))
    if node not in {'transport', 'ratio'}:
        raise ValueError(f"Sampler {label!r}: node must be 'transport' or 'ratio'.")
    cfg['node'] = node
    cfg['score'] = canonicalize_score_method(cfg.get('score', 'lfgi'))
    cfg['ratio_mode'] = canonicalize_ratio_mode(cfg.get('ratio_mode', 'pflow')) if node == 'ratio' else None
    cfg['ref_source'] = _canonical_ref_source(cfg.get('ref_source'))
    cfg['display_name'] = str(cfg.get('display_name', label))
    cfg['include_results'] = _coerce_bool(cfg.get('include_results', True), True)
    cfg['is_reference'] = _coerce_bool(cfg.get('is_reference', False), False)
    cfg['n_samples'] = int(DEFAULT_N_GEN if default_n_samples is None else default_n_samples) \
        if cfg.get('n_samples') is None else int(cfg['n_samples'])
    if cfg['n_samples'] <= 0:
        raise ValueError(f'Sampler {label!r}: n_samples must be positive.')
    cfg['n_ref'] = None if cfg.get('n_ref') is None else int(cfg['n_ref'])
    cfg['n_gate'] = None if cfg.get('n_gate') is None else int(cfg['n_gate'])
    cfg['bank_coupling'] = canonicalize_bank_coupling(cfg.get('bank_coupling', 'shared'))
    cfg['steps'] = int(cfg.get('steps', 200))
    cfg['t_min'] = max(float(cfg.get('t_min', 10 ** (-2.5))), 1e-4)
    cfg['t_max'] = float(cfg.get('t_max', 10.0))
    cfg['log_mean_ess'] = _coerce_bool(cfg.get('log_mean_ess', node == 'transport'), node == 'transport')
    cfg['dim'] = ACTIVE_DIM if cfg.get('dim') is None else int(cfg['dim'])
    if cfg['dim'] is None and default_dim is not None:
        cfg['dim'] = int(default_dim)
    if cfg['dim'] is None:
        raise ValueError(f'Sampler {label!r}: dim is unset; call configure_sampling or set dim.')
    if cfg['t_max'] <= cfg['t_min']:
        raise ValueError(f'Sampler {label!r}: t_max must exceed t_min.')
    if cfg['steps'] <= 0 and not (node == 'ratio' and cfg['ratio_mode'] == 'static'):
        raise ValueError(f'Sampler {label!r}: flow nodes require steps > 0.')

    cfg['eval_chunk'] = max(1, int(cfg.get('eval_chunk', 64)))
    cfg['matrix_blend_center'] = _coerce_bool(cfg.get('matrix_blend_center', True), True)
    cfg['matrix_blend_ridge'] = float(cfg.get('matrix_blend_ridge', 1e-8))
    cfg['matrix_blend_ridge_rel'] = float(cfg.get('matrix_blend_ridge_rel', 1e-6))
    if cfg['matrix_blend_ridge'] < 0.0 or cfg['matrix_blend_ridge_rel'] < 0.0:
        raise ValueError(f'Sampler {label!r}: matrix blend ridges must be nonnegative.')
    cfg['matrix_blend_sym_gate'] = _coerce_bool(cfg.get('matrix_blend_sym_gate', False), False)
    cfg['matrix_blend_gate_clip'] = cfg.get('matrix_blend_gate_clip', 1e6)
    if cfg['matrix_blend_gate_clip'] is not None:
        cfg['matrix_blend_gate_clip'] = float(cfg['matrix_blend_gate_clip'])
    cfg['global_blend_clamp'] = _coerce_bool(cfg.get('global_blend_clamp', True), True)

    cfg['density_steps'] = max(1, int(cfg.get('density_steps', 32)))
    cfg['density_divergence'] = canonicalize_divergence_mode(cfg.get('density_divergence', 'auto'))
    cfg['density_div_probes'] = max(1, int(cfg.get('density_div_probes', 1)))
    cfg['density_fd_eps'] = float(cfg.get('density_fd_eps', 1e-3))
    cfg['density_batch_size'] = max(1, int(cfg.get('density_batch_size', 32)))
    cfg['density_t_min'] = max(float(cfg.get('density_t_min', cfg['t_min'])), 1e-4)
    cfg['density_t_max'] = float(cfg.get('density_t_max', cfg['t_max']))
    cfg['ratio_temperature'] = float(cfg.get('ratio_temperature', 1.0))
    if cfg['density_fd_eps'] <= 0.0 or cfg['density_t_max'] <= cfg['density_t_min']:
        raise ValueError(f'Sampler {label!r}: invalid density interval or finite-difference epsilon.')
    if not math.isfinite(cfg['ratio_temperature']):
        raise ValueError(f'Sampler {label!r}: ratio_temperature must be finite.')
    cfg['ratio_log_weight_clip'] = cfg.get('ratio_log_weight_clip', 20.0)
    if cfg['ratio_log_weight_clip'] is not None:
        cfg['ratio_log_weight_clip'] = float(cfg['ratio_log_weight_clip'])
    cfg['_normalized'] = True
    return cfg


def _normalize_sampler_configs(sampler_configs, default_n_samples=None, default_dim=None):
    if not isinstance(sampler_configs, dict) or len(sampler_configs) == 0:
        raise ValueError('sampler_configs must be a non-empty dictionary.')
    return OrderedDict(
        (label, normalize_sampler_config(label, cfg, default_n_samples, default_dim))
        for label, cfg in sampler_configs.items()
    )


def sampler_config_includes_results(config):
    return _coerce_bool(config.get('include_results', True), True)


def _resolve_sampler_execution_order(normalized_configs):
    order = []
    states = {}

    def visit(label):
        state = states.get(label, 0)
        if state == 1:
            raise ValueError(f'Cycle detected in sampler ref_source graph at {label!r}.')
        if state == 2:
            return
        if label not in normalized_configs:
            raise KeyError(f'Unknown sampler label {label!r}.')
        states[label] = 1
        source = normalized_configs[label]['ref_source']
        if source is not None:
            if source not in normalized_configs:
                raise KeyError(f"Sampler {label!r} references missing ref_source={source!r}.")
            visit(source)
        states[label] = 2
        order.append(label)

    for label in normalized_configs:
        visit(label)
    return order


@dataclass
class NodeState:
    samples: torch.Tensor
    log_weights: torch.Tensor
    generated_by_flow: bool


@dataclass
class ReferenceSelection:
    source_label: str
    pool: torch.Tensor
    pool_log_weights: torch.Tensor
    signal_slice: slice
    gate_slice: slice
    signal_bank: ReferenceBank
    gate_bank: ReferenceBank


def _resolved_bank_counts(cfg, default_n_ref):
    n_ref = int(default_n_ref if cfg['n_ref'] is None else cfg['n_ref'])
    n_gate = int(n_ref if cfg['n_gate'] is None else cfg['n_gate'])
    if n_ref <= 0 or n_gate <= 0:
        raise ValueError('n_ref and n_gate must be positive.')
    if cfg['bank_coupling'] == 'shared' and n_ref != n_gate:
        raise ValueError('bank_coupling=shared requires n_ref == n_gate; use prefix otherwise.')
    return n_ref, n_gate


def _required_pool_size(n_ref, n_gate, coupling):
    return n_ref + n_gate if coupling == 'independent' else max(n_ref, n_gate)


def _slice_key(sl):
    return int(sl.start or 0), int(sl.stop)


def _build_cached_bank(precomp, source_label, pool, pool_logw, sl,
                       prior_model, lik_model):
    start, stop = _slice_key(sl)
    cache_key = (str(source_label), int(pool.shape[0]), start, stop)
    cache = precomp.setdefault('reference_banks', {})
    if cache_key not in cache:
        cache[cache_key] = precompute_reference_bank(
            pool[sl], prior_model, lik_model,
            label=f'{source_label}:{start}:{stop}', log_weights=pool_logw[sl],
        )
    return cache[cache_key]


def _select_reference_banks(label, cfg, states, precomp, prior_model, lik_model,
                            default_n_ref):
    n_ref, n_gate = _resolved_bank_counts(cfg, default_n_ref)
    coupling = cfg['bank_coupling']
    required = _required_pool_size(n_ref, n_gate, coupling)
    source = cfg['ref_source']
    if source is None:
        source_label = 'prior'
        prior_pools = precomp.setdefault('prior_pools', {})
        if required not in prior_pools:
            prior_pools[required] = NodeState(
                samples=prior_model.sample(required).detach().cpu().to(dtype=torch.float64),
                log_weights=torch.zeros(required, dtype=torch.float64),
                generated_by_flow=False,
            )
        state = prior_pools[required]
    else:
        source_label = source
        if source not in states:
            raise RuntimeError(f'Sampler {label!r}: ref_source={source!r} has not run.')
        state = states[source]
    if state.samples.shape[0] < required:
        raise ValueError(
            f"Sampler {label!r} needs {required} particles for {coupling} banks, but "
            f"ref_source={source_label!r} provides {state.samples.shape[0]}."
        )
    pool = state.samples[:required].detach().cpu().to(dtype=torch.float64).contiguous()
    pool_logw = state.log_weights[:required].detach().cpu().to(dtype=torch.float64).contiguous()
    if coupling == 'independent':
        gate_slice = slice(0, n_gate)
        signal_slice = slice(n_gate, n_gate + n_ref)
    else:
        signal_slice = slice(0, n_ref)
        gate_slice = slice(0, n_gate)
    signal_bank = _build_cached_bank(
        precomp, source_label, pool, pool_logw, signal_slice, prior_model, lik_model
    )
    if _slice_key(signal_slice) == _slice_key(gate_slice):
        gate_bank = signal_bank
    else:
        gate_bank = _build_cached_bank(
            precomp, source_label, pool, pool_logw, gate_slice, prior_model, lik_model
        )
    return ReferenceSelection(
        source_label=source_label,
        pool=pool,
        pool_log_weights=pool_logw,
        signal_slice=signal_slice,
        gate_slice=gate_slice,
        signal_bank=signal_bank,
        gate_bank=gate_bank,
    )


def _make_score_field(selection, cfg):
    return ScoreField(
        selection.signal_bank,
        cfg['score'],
        gate_bank=selection.gate_bank,
        eval_chunk=cfg['eval_chunk'],
        matrix_blend_center=cfg['matrix_blend_center'],
        matrix_blend_ridge=cfg['matrix_blend_ridge'],
        matrix_blend_ridge_rel=cfg['matrix_blend_ridge_rel'],
        matrix_blend_sym_gate=cfg['matrix_blend_sym_gate'],
        matrix_blend_gate_clip=cfg['matrix_blend_gate_clip'],
        global_blend_clamp=cfg['global_blend_clamp'],
    )


def _selection_target_log_density(selection):
    """Assemble target log densities already evaluated with the selected banks."""
    out = torch.empty(selection.pool.shape[0], dtype=torch.float64)
    filled = torch.zeros(selection.pool.shape[0], dtype=torch.bool)
    out[selection.signal_slice] = selection.signal_bank.log_target_ref
    filled[selection.signal_slice] = True
    out[selection.gate_slice] = selection.gate_bank.log_target_ref
    filled[selection.gate_slice] = True
    if not bool(filled.all().item()):
        raise RuntimeError('Signal/gate bank slices do not cover the selected ratio pool.')
    return out


def _node_pde_counts(selection):
    n_unique = selection.signal_bank.n
    if selection.gate_bank is not selection.signal_bank:
        n_unique += selection.gate_bank.n
    counts = {
        'pde_likelihood_evals': n_unique,
        'pde_score_evals': n_unique,
        'pde_gn_hessian_evals': n_unique,
    }
    counts['pde_solve_count'] = sum(counts.values())
    return counts


def run_single_sampler_config(label, config, prior_model, lik_model, *, states,
                              precomp, default_n_ref):
    cfg = config if config.get('_normalized', False) else normalize_sampler_config(
        label, config, DEFAULT_N_GEN, ACTIVE_DIM
    )
    selection = _select_reference_banks(
        label, cfg, states, precomp, prior_model, lik_model, default_n_ref
    )
    field = _make_score_field(selection, cfg)
    n_ref, n_gate = _resolved_bank_counts(cfg, default_n_ref)
    run_info = dict(cfg)
    run_info.update({
        'ref_source': selection.source_label,
        'n_ref': n_ref,
        'n_gate': n_gate,
        'init_steps': cfg['steps'],
        'method': cfg['score'],
        'weight_mode': 'unweighted',
    })

    if cfg['node'] == 'transport':
        samples, ess_trace, flow_info = run_reverse_ou_heun(
            cfg['n_samples'], field, steps=cfg['steps'], dim=cfg['dim'],
            t_max=cfg['t_max'], t_min=cfg['t_min'], log_mean_ess=cfg['log_mean_ess'],
        )
        state = NodeState(
            samples=samples,
            log_weights=torch.zeros(samples.shape[0], dtype=torch.float64),
            generated_by_flow=True,
        )
        precomp.setdefault('score_fields', {})[label] = field
        run_info.update(flow_info)
        run_info.update(_node_pde_counts(selection))
        return state, ess_trace, run_info

    source_label = cfg['ref_source']
    if source_label is None:
        raise ValueError(f'Ratio node {label!r} requires a named ref_source transport node.')
    source_density_field = precomp.get('score_fields', {}).get(source_label)
    if source_density_field is None:
        raise ValueError(
            f"Ratio node {label!r} cannot reconstruct q: ref_source={source_label!r} "
            'does not expose a generating flow field.'
        )
    eval_pool = selection.pool
    print(
        f"  [{label}] reconstructing log q for {eval_pool.shape[0]} endpoints "
        f"from frozen source field {source_label!r}"
    )
    logq, density_info = estimate_logq_probability_flow(
        eval_pool,
        source_density_field,
        t_min=cfg['density_t_min'],
        t_max=cfg['density_t_max'],
        n_steps=cfg['density_steps'],
        n_div_probes=cfg['density_div_probes'],
        fd_eps=cfg['density_fd_eps'],
        divergence_mode=cfg['density_divergence'],
        batch_size=cfg['density_batch_size'],
        return_diagnostics=True,
    )
    logpi = _selection_target_log_density(selection)
    raw_log_ratio = logpi - logq
    log_tilt = _finalize_ratio_log_tilt(
        raw_log_ratio,
        temperature=cfg['ratio_temperature'],
        clip=cfg['ratio_log_weight_clip'],
    )
    ess = global_log_weight_ess(log_tilt)
    run_info.update(density_info)
    run_info.update({
        'density_certificate_source': 'frozen_generating_field',
        'ratio_carrier_source': 'settled_endpoint_bank',
        'ratio_logq_mean': float(logq.mean().item()),
        'ratio_logq_std': float(logq.std(unbiased=False).item()),
        'ratio_raw_log_weight_std': float(raw_log_ratio.std(unbiased=False).item()),
        'ratio_log_weight_mean': float(log_tilt.mean().item()),
        'ratio_log_weight_std': float(log_tilt.std(unbiased=False).item()),
        'ratio_log_weight_min': float(log_tilt.min().item()),
        'ratio_log_weight_max': float(log_tilt.max().item()),
        'ratio_global_ess': ess,
        'ratio_global_ess_fraction': ess / float(log_tilt.numel()),
    })
    run_info.update(_node_pde_counts(selection))

    if cfg['ratio_mode'] == 'static':
        state = NodeState(
            samples=eval_pool.detach().cpu(),
            log_weights=log_tilt.detach().cpu(),
            generated_by_flow=False,
        )
        run_info.update({
            'weight_mode': 'static',
            'ratio_returns_unweighted_particles': False,
            'score_norm': float('nan'),
            'score_norm_initial': float('nan'),
            'score_norm_mean': float('nan'),
            'score_norm_final': float('nan'),
            'score_norm_max': float('nan'),
        })
        return state, None, run_info

    signal_tilt = log_tilt[selection.signal_slice].contiguous()
    ratio_field = CompletedRatioField(field, signal_tilt)
    probe_n = min(32, eval_pool.shape[0])
    probe = eval_pool[:probe_n].to(device=device, dtype=torch.float64)
    identity = field.completed_ratio_components_chunk(probe, 0.1, signal_tilt)['identity_residual']
    samples, ess_trace, flow_info = run_reverse_ou_heun(
        cfg['n_samples'], ratio_field, steps=cfg['steps'], dim=cfg['dim'],
        t_max=cfg['t_max'], t_min=cfg['t_min'], log_mean_ess=cfg['log_mean_ess'],
    )
    state = NodeState(
        samples=samples,
        log_weights=torch.zeros(samples.shape[0], dtype=torch.float64),
        generated_by_flow=True,
    )
    precomp.setdefault('score_fields', {})[label] = ratio_field
    run_info.update(flow_info)
    run_info.update({
        'weight_mode': 'pflow',
        'ratio_returns_unweighted_particles': True,
        'ratio_completed_identity_rmse': float(torch.sqrt(torch.mean(identity.square())).item()),
        'ratio_completed_identity_max_abs': float(torch.max(torch.abs(identity)).item()),
    })
    return state, ess_trace, run_info


def choose_reference_key(samples_dict, sampler_run_info=None, preferred=None):
    if preferred is not None and preferred in samples_dict:
        return preferred
    if sampler_run_info is not None:
        for label, info in sampler_run_info.items():
            if info.get('is_reference', False) and label in samples_dict:
                return label
    for label in samples_dict:
        return label
    raise ValueError('No sampler outputs are available.')


def run_tree_sampler_suite(sampler_configs, prior_model, lik_model, n_ref=10000):
    """Execute a recursive dictionary sampler DAG exactly once per node."""
    normalized = _normalize_sampler_configs(sampler_configs, DEFAULT_N_GEN, ACTIVE_DIM)
    execution_order = _resolve_sampler_execution_order(normalized)
    print('\n=== Sampler execution order ===')
    print(' -> '.join(execution_order))
    states = OrderedDict()
    samples = OrderedDict()
    ess_logs = OrderedDict()
    run_info = OrderedDict()
    precomp = {
        'default_n_ref': int(n_ref),
        'reference_banks': {},
        'prior_pools': {},
        'score_fields': {},
        'excluded_run_info': OrderedDict(),
    }
    included = []
    excluded = []
    for label in execution_order:
        cfg = normalized[label]
        t0 = time.time()
        state, ess_trace, info = run_single_sampler_config(
            label, cfg, prior_model, lik_model,
            states=states, precomp=precomp, default_n_ref=n_ref,
        )
        states[label] = state
        info = dict(info)
        info['runtime_seconds'] = time.time() - t0
        if sampler_config_includes_results(cfg):
            samples[label] = state.samples
            run_info[label] = info
            included.append(label)
            if ess_trace is not None and len(ess_trace.get('t', [])) > 0:
                ess_logs[label] = ess_trace
        else:
            excluded.append(label)
            precomp['excluded_run_info'][label] = info
        print(f"{label}: {info['runtime_seconds']:.2f}s")
    precomp['states'] = states
    precomp['included_result_labels'] = included
    precomp['excluded_result_labels'] = excluded
    return samples, ess_logs, run_info, precomp


def run_standard_sampler_pipeline(prior_model, lik_model, sampler_configs, n_ref=10000):
    samples, ess_logs, sampler_run_info, precomp = run_tree_sampler_suite(
        sampler_configs, prior_model, lik_model, n_ref=n_ref
    )
    if len(samples) == 0:
        raise ValueError('No sampler nodes have include_results=True.')
    display_names = {
        label: info.get('display_name', label) for label, info in sampler_run_info.items()
    }
    reference_key = choose_reference_key(samples, sampler_run_info)
    return {
        'precomp': precomp,
        'samples': samples,
        'ess_logs': ess_logs,
        'sampler_run_info': sampler_run_info,
        'display_names': display_names,
        'reference_key': reference_key,
        'reference_title': display_names.get(reference_key, reference_key),
        'n_ref': int(n_ref),
        'n_ref_by_sampler': {
            label: int(info.get('n_ref', 0)) for label, info in sampler_run_info.items()
        },
        'included_result_labels': list(precomp['included_result_labels']),
        'excluded_result_labels': list(precomp['excluded_result_labels']),
    }


def summarize_sampler_run(sampler_run_info):
    print('\n=== Config summary ===')
    for label, info in sampler_run_info.items():
        ratio = f"/{info.get('ratio_mode')}" if info.get('node') == 'ratio' else ''
        print(
            f"{label:<24} -> {info.get('node')}{ratio} | score={info.get('score')} | "
            f"ref_source={info.get('ref_source')} | n_ref={info.get('n_ref')} | "
            f"n_gate={info.get('n_gate')} | coupling={info.get('bank_coupling')}"
        )


# ==========================================
# 6. EVALUATION UTILS (physics-agnostic â€” unchanged)
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
    family_map = {
        'lfgi': 'LFGI',
        'local_scalar_blend': 'LOCAL SCALAR BLEND',
        'local_matrix_blend': 'MATRIX BLEND',
        'global_scalar_blend': 'GLOBAL SCALAR BLEND',
        'global_matrix_blend': 'GLOBAL MATRIX BLEND',
        'tweedie': 'Tweedie',
        'tsi': 'TSI',
    }
    method = info.get('score', info.get('method', label))
    return family_map.get(str(method), str(method))


def results_weight_mode(label, info):
    return str(info.get('weight_mode', 'unweighted'))


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
            'steps': int(info.get('steps', info.get('init_steps', 0))),
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
