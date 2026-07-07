# -*- coding: utf-8 -*- 
"""Darcy flow inverse-problem iterative-transport benchmark.

Runs Algorithm 1, alternating likelihood-corrected transport, for GN-LFGI,
local matrix blend, local scalar blend, uniform matrix blend, uniform scalar
blend, and Tweedie.  Each method performs three transport/ratio rounds and
reports transport rounds 1 and 3.

Useful overrides:

    export IP_ITER_N_REF=1000
    export IP_ITER_ROUNDS=3
    export IP_ITER_TRANSPORT_STEPS=200
    export IP_ITER_DRC_PF_STEPS=32
"""
import gc
import os
import sys
from collections import OrderedDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # notebook / pasted-cell fallback
    THIS_DIR = os.getcwd()

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

################################################################################
import sys, importlib, linecache, os

# Make sure the problem directory and repo root are import-visible.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Clear stale source-line cache and stale imported module.
linecache.clearcache()
if "sampling" in sys.modules:
    del sys.modules["sampling"]

# Prefer the suffixed patched helper when this file is run without overwriting
# repository sampling.py.  Set HLSI_SAMPLING_MODULE=sampling to force the
# standard repository module, or copy sampling_uniform_blends.py to sampling.py.
SAMPLING_MODULE_NAME = os.environ.get("HLSI_SAMPLING_MODULE", None)
if SAMPLING_MODULE_NAME is None:
    try:
        sampling = importlib.import_module("sampling_uniform_blends")
        SAMPLING_MODULE_NAME = "sampling_uniform_blends"
    except ModuleNotFoundError:
        SAMPLING_MODULE_NAME = "sampling"
        sampling = importlib.import_module(SAMPLING_MODULE_NAME)
else:
    sampling = importlib.import_module(SAMPLING_MODULE_NAME)
importlib.reload(sampling)
# Preserve the historical module name for downstream `from sampling import ...`.
sys.modules["sampling"] = sampling

print("Using:", sampling.__file__)
print("DRC test:", sampling.canonicalize_init_weights("DRC"))

# The shared helper still uses the historical implementation name `ce_hlsi`
# internally.  For this Darcy manuscript script, every plot/table-facing
# occurrence of that method should be displayed as LFGI.  The score initializer
# remains `ce_hlsi` for compatibility with sampling.py; only labels are changed.
_orig_drc_method_pretty_name = getattr(sampling, "_drc_method_pretty_name", None)
if _orig_drc_method_pretty_name is not None:
    def _darcy_lfgi_drc_method_pretty_name(label, cfg=None):
        cfg = cfg or {}
        label_s = str(label)
        display = str(cfg.get("display_name", "")) if isinstance(cfg, dict) else ""
        init = str(cfg.get("drc_score_init", cfg.get("init", ""))) if isinstance(cfg, dict) else ""
        tokens = " ".join([label_s, display, init]).lower().replace("_", "-")
        if ("ce-hlsi" in tokens) or ("hlsi" in tokens) or ("lfgi" in tokens):
            return "LFGI"
        if ("matrix" in tokens) or ("centered" in tokens):
            return "MATRIX BLEND"
        pretty = _orig_drc_method_pretty_name(label, cfg)
        if ("CE-HLSI" in str(pretty)) or ("HLSI" in str(pretty)):
            return "LFGI"
        if "MATRIX" in str(pretty).upper() or "CENTERED" in str(pretty).upper():
            return "MATRIX BLEND"
        return pretty

    sampling._drc_method_pretty_name = _darcy_lfgi_drc_method_pretty_name
    # Be explicit about the exact global lookup used inside the comparison-grid
    # function that writes the text boxes in Figure 5.  This is the line that
    # produces the old ``CE-HLSI`` label in the density-energy scatterplot grid.
    for _fn_name in ("save_drc_energy_comparison_grid", "finalize_drc_energy_benchmark_plots"):
        _fn = getattr(sampling, _fn_name, None)
        if _fn is not None and hasattr(_fn, "__globals__"):
            _fn.__globals__["_drc_method_pretty_name"] = _darcy_lfgi_drc_method_pretty_name


# Compatibility shim for current sampling.py versions.  In the direct transport
# path, run_single_sampler_config calls resolve_hlsi_gate_law(init, init_weights,
# gate_rho=...), while the current helper signature expects the second positional
# argument to be gate_rho.  DRC/density-only scripts avoid this path; iterative
# transport with a direct LFGI T-step hits it.  Treat a positional init-weight
# token (None/L/WC/PoU/DRC) as metadata and forward the actual gate kwargs.
_orig_resolve_hlsi_gate_law = sampling.resolve_hlsi_gate_law


def _resolve_hlsi_gate_law_compat(mode, maybe_gate_rho=None, *args,
                                  gate_rho=None, gate_beta=None, gate_kappa=None,
                                  gate_topk=64, gate_metric_source='mu', **kwargs):
    positional_is_weight = False
    if maybe_gate_rho is not None:
        try:
            sampling.canonicalize_init_weights(maybe_gate_rho)
            positional_is_weight = True
        except Exception:
            positional_is_weight = False

    if positional_is_weight:
        # This is the accidental cfg['init_weights'] positional argument.
        pass
    elif maybe_gate_rho is not None and gate_rho is None:
        gate_rho = maybe_gate_rho

    # Preserve compatibility with any older positional calls after gate_rho.
    if len(args) >= 1 and gate_beta is None:
        gate_beta = args[0]
    if len(args) >= 2 and gate_kappa is None:
        gate_kappa = args[1]
    if len(args) >= 3:
        gate_topk = args[2]
    if len(args) >= 4:
        gate_metric_source = args[3]
    if len(args) > 4:
        raise TypeError(f"resolve_hlsi_gate_law expected at most 6 positional arguments, got {2 + len(args)}")

    return _orig_resolve_hlsi_gate_law(
        mode,
        gate_rho=gate_rho,
        gate_beta=gate_beta,
        gate_kappa=gate_kappa,
        gate_topk=gate_topk,
        gate_metric_source=gate_metric_source,
    )


sampling.resolve_hlsi_gate_law = _resolve_hlsi_gate_law_compat
for _fn_name in (
    'eval_modular_hlsi_posterior_score',
    'run_single_sampler_config',
    '_run_drc_ratio_update_config',
):
    _fn = getattr(sampling, _fn_name, None)
    if _fn is not None and hasattr(_fn, '__globals__'):
        _fn.__globals__['resolve_hlsi_gate_law'] = _resolve_hlsi_gate_law_compat


# Non-finite transport hardening for pilot inverse-problem comparisons.
# Some score families can occasionally produce a complete blow-up in a transport
# node.  The shared helper's reference-bank constructor correctly rejects a
# source with no finite samples, but that stops the full comparison.  For this
# experiment we keep the run alive by replacing non-finite output rows at the
# boundary of each sampler node.  Partial blow-ups are repaired from the finite
# rows of the same output; complete blow-ups fall back to the node's finite
# input reference bank, and finally to fresh prior samples if no finite input
# bank exists.  The run-info table records when this happens.
import warnings as _warnings

_orig_run_single_sampler_config = sampling.run_single_sampler_config


def _as_finite_cpu_pool(x, dim=None):
    if x is None:
        return None
    try:
        if torch.is_tensor(x):
            y = x.detach().cpu().to(dtype=torch.float64)
        else:
            y = torch.tensor(np.asarray(x), dtype=torch.float64)
    except Exception:
        return None
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        return None
    if dim is not None and y.shape[1] != int(dim):
        return None
    finite = torch.isfinite(y).all(dim=1)
    if not bool(finite.any().item()):
        return None
    return y[finite].contiguous()


def _repeat_pool_to_n(pool, n, dim):
    pool = _as_finite_cpu_pool(pool, dim=dim)
    if pool is None or pool.shape[0] == 0:
        return None
    n = int(n)
    idx = torch.arange(n, dtype=torch.long) % int(pool.shape[0])
    return pool[idx].contiguous()


def _prior_fallback_samples(prior_model, n, dim):
    try:
        x = prior_model.sample(int(n))
        return _repeat_pool_to_n(x, int(n), int(dim))
    except Exception:
        return torch.zeros((int(n), int(dim)), dtype=torch.float64)


def _repair_nonfinite_sampler_output(label, config, prior_model, ref_bank, samps, info, precomp):
    if torch.is_tensor(samps):
        samples_cpu = samps.detach().cpu().to(dtype=torch.float64)
    else:
        samples_cpu = torch.tensor(np.asarray(samps), dtype=torch.float64)

    cfg_dim = None
    if isinstance(config, dict):
        cfg_dim = config.get('dim')
    if cfg_dim is None:
        cfg_dim = getattr(prior_model, 'dim', None) or getattr(sampling, 'ACTIVE_DIM', None)

    if samples_cpu.ndim == 1:
        samples_cpu = samples_cpu.reshape(1, -1)
    if samples_cpu.ndim != 2:
        dim = int(cfg_dim) if cfg_dim is not None else int(getattr(prior_model, 'dim', 1))
        n_target = int(config.get('n_samples', getattr(sampling, 'DEFAULT_N_GEN', 500))) if isinstance(config, dict) else int(getattr(sampling, 'DEFAULT_N_GEN', 500))
        repaired = None
        source = None
        if isinstance(ref_bank, dict):
            repaired = _repeat_pool_to_n(ref_bank.get('X_ref'), n_target, dim)
            source = 'input_reference_bank'
        if repaired is None:
            repaired = _prior_fallback_samples(prior_model, n_target, dim)
            source = 'fresh_prior'
        n_bad = n_target
        reason = f'non-2D sampler output with shape={tuple(samples_cpu.shape)}'
    else:
        n_target = int(samples_cpu.shape[0])
        dim = int(samples_cpu.shape[1])
        finite = torch.isfinite(samples_cpu).all(dim=1)
        n_bad = int((~finite).sum().item())
        if n_bad == 0:
            return samps, info
        if bool(finite.any().item()):
            repaired = samples_cpu.clone()
            pool = samples_cpu[finite]
            fill_idx = torch.arange(n_bad, dtype=torch.long) % int(pool.shape[0])
            repaired[~finite] = pool[fill_idx]
            source = 'finite_rows_from_same_output'
            reason = f'{n_bad}/{n_target} non-finite output rows'
        else:
            repaired = None
            source = None
            if isinstance(ref_bank, dict):
                repaired = _repeat_pool_to_n(ref_bank.get('X_ref'), n_target, dim)
                source = 'input_reference_bank'
            if repaired is None:
                repaired = _prior_fallback_samples(prior_model, n_target, dim)
                source = 'fresh_prior'
            reason = f'all {n_target} output rows were non-finite'

    msg = (
        f"[{label}] sampler output repair: {reason}; "
        f"using {source} fallback so downstream reference-bank construction can continue."
    )
    print('  ' + msg)
    _warnings.warn(msg, RuntimeWarning)

    info = dict(info or {})
    info['sample_fallback_used'] = True
    info['sample_fallback_source'] = source
    info['sample_fallback_n_bad'] = int(n_bad)
    info['sample_fallback_reason'] = reason
    if precomp is not None:
        precomp.setdefault('sample_fallbacks', OrderedDict())[label] = {
            'source': source,
            'n_bad': int(n_bad),
            'reason': reason,
        }
    return repaired.detach().cpu().contiguous(), info


def _run_single_sampler_config_hardened(label, config, prior_model, lik_model, precomp=None,
                                        ref_bank=None, ref_bank_source='None', n_ref_used=0):
    samps, ess_trace, info = _orig_run_single_sampler_config(
        label, config, prior_model, lik_model, precomp,
        ref_bank=ref_bank, ref_bank_source=ref_bank_source, n_ref_used=n_ref_used,
    )
    samps, info = _repair_nonfinite_sampler_output(
        label, config, prior_model, ref_bank, samps, info, precomp,
    )
    return samps, ess_trace, info


sampling.run_single_sampler_config = _run_single_sampler_config_hardened
for _fn_name in ('run_tree_sampler_suite', 'run_standard_sampler_pipeline'):
    _fn = getattr(sampling, _fn_name, None)
    if _fn is not None and hasattr(_fn, '__globals__'):
        _fn.__globals__['run_single_sampler_config'] = _run_single_sampler_config_hardened


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
# Dashboard PDF utilities
# ==========================================
# Produces a single multipage PDF containing the scalar metrics tables and every
# figure produced by this script. Console progress logs are intentionally not
# captured into the dashboard.

import glob
import numbers
import re
import shutil
import textwrap
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

SAVE_DASHBOARD_PDF = True
DASHBOARD_SHOW_FIGURES = True
DASHBOARD_PDF_PATH = None  # Filled after init_run_results(), inside the active run-results directory.


def _dashboard_is_scalar_cell(x):
    if x is None:
        return True
    if isinstance(x, (str, bool)):
        return True
    if isinstance(x, numbers.Number):
        return True
    if isinstance(x, np.generic):
        return True
    return False


def _dashboard_format_cell(x, max_len=72):
    if x is None:
        return ""
    try:
        if isinstance(x, np.generic):
            x = x.item()
    except Exception:
        pass
    if isinstance(x, numbers.Number):
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return str(x)
            if abs(xf) >= 1e4 or (0 < abs(xf) < 1e-3):
                return f"{xf:.4e}"
            return f"{xf:.6g}"
        except Exception:
            return str(x)
    if isinstance(x, (list, tuple, dict, np.ndarray)):
        try:
            if isinstance(x, np.ndarray):
                return f"array{tuple(x.shape)}"
            return f"{type(x).__name__}[{len(x)}]"
        except Exception:
            return type(x).__name__
    s = str(x)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _dashboard_sanitize_df(df, include_index=True):
    df = pd.DataFrame(df).copy()
    if include_index and not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    for col in df.columns:
        df[col] = df[col].map(_dashboard_format_cell)
    df.columns = [_dashboard_format_cell(c, max_len=40) for c in df.columns]
    return df


def metrics_dict_to_scalar_df(metrics_dict, display_names=None):
    """Convert the metrics dictionary into a dashboard-friendly scalar table."""
    display_names = display_names or {}
    rows = []
    for label, data in metrics_dict.items():
        if not isinstance(data, dict):
            continue
        row = OrderedDict()
        row["Method"] = display_names.get(label, label)
        for key, val in data.items():
            if _dashboard_is_scalar_cell(val):
                row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)


def nested_dict_to_df(dct, row_name="Method", display_names=None):
    """Convert nested dict/list records into a table without large arrays."""
    display_names = display_names or {}
    if isinstance(dct, pd.DataFrame):
        return dct.copy()
    rows = []
    if isinstance(dct, dict):
        iterable = dct.items()
    else:
        iterable = enumerate(dct)
    for key, val in iterable:
        row = OrderedDict()
        row[row_name] = display_names.get(key, key)
        if isinstance(val, dict):
            for k, v in val.items():
                if _dashboard_is_scalar_cell(v):
                    row[k] = v
                elif isinstance(v, (list, tuple, np.ndarray)):
                    row[k] = _dashboard_format_cell(v)
                else:
                    row[k] = _dashboard_format_cell(v)
        else:
            row["value"] = _dashboard_format_cell(val)
        rows.append(row)
    return pd.DataFrame(rows)


def sampler_configs_to_df(configs):
    rows = []
    for label, cfg in configs.items():
        row = OrderedDict()
        row["Method"] = label
        row.update(cfg)
        rows.append(row)
    return pd.DataFrame(rows)


class DashboardPDF:
    def __init__(self, path, title="Dashboard"):
        self.path = os.path.abspath(path)
        self.title = title
        self.enabled = bool(SAVE_DASHBOARD_PDF)
        self.pdf = PdfPages(self.path) if self.enabled else None
        self._seen_fig_ids = {id(plt.figure(num)) for num in plt.get_fignums()}
        self.figure_pages = 0
        self.table_pages = 0
        self.text_pages = 0

    def add_text_page(self, title, lines, footer=None, mono=False):
        if not self.enabled:
            return
        if isinstance(lines, str):
            lines = lines.splitlines()
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.055, 0.93, title, fontsize=18, fontweight="bold", va="top")
        y = 0.86
        fontsize = 9.2 if mono else 10.5
        family = "monospace" if mono else "sans-serif"
        for raw in lines:
            wrapped = textwrap.wrap(str(raw), width=112 if mono else 105) or [""]
            for line in wrapped:
                ax.text(0.06, y, line, fontsize=fontsize, family=family, va="top")
                y -= 0.033 if mono else 0.038
                if y < 0.08:
                    if footer:
                        ax.text(0.055, 0.035, footer, fontsize=8.5, alpha=0.65)
                    self.pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    self.text_pages += 1
                    fig = plt.figure(figsize=(11, 8.5))
                    fig.patch.set_facecolor("white")
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.axis("off")
                    ax.text(0.055, 0.93, title + " (cont.)", fontsize=18, fontweight="bold", va="top")
                    y = 0.86
        if footer:
            ax.text(0.055, 0.035, footer, fontsize=8.5, alpha=0.65)
        self.pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        self.text_pages += 1

    def add_dataframe(self, title, df, max_rows=28, max_cols=7, include_index=True):
        if not self.enabled:
            return
        df = _dashboard_sanitize_df(df, include_index=include_index)
        if df.empty:
            self.add_text_page(title, ["No rows available."])
            return

        # Keep the first column (usually Method / metric name) pinned on horizontal splits.
        first_col = [df.columns[0]]
        other_cols = list(df.columns[1:])
        cols_per_page = max(1, max_cols - 1)
        col_chunks = [other_cols[i:i + cols_per_page] for i in range(0, len(other_cols), cols_per_page)] or [[]]

        for ci, col_chunk in enumerate(col_chunks):
            cols = first_col + col_chunk
            df_col = df.loc[:, cols]
            for ri in range(0, len(df_col), max_rows):
                df_page = df_col.iloc[ri:ri + max_rows]
                fig, ax = plt.subplots(figsize=(11, 8.5))
                fig.patch.set_facecolor("white")
                ax.axis("off")
                suffix = ""
                if len(col_chunks) > 1:
                    suffix += f" - columns {ci + 1}/{len(col_chunks)}"
                if len(df_col) > max_rows:
                    suffix += f" - rows {ri + 1}-{min(ri + max_rows, len(df_col))}"
                ax.set_title(title + suffix, fontsize=15, fontweight="bold", pad=14)

                col_width = 1.0 / max(len(df_page.columns), 1)
                table = ax.table(
                    cellText=df_page.values,
                    colLabels=df_page.columns,
                    cellLoc="center",
                    colLoc="center",
                    loc="center",
                    colWidths=[col_width] * len(df_page.columns),
                )
                table.auto_set_font_size(False)
                table.set_fontsize(7.5 if len(df_page.columns) >= 6 else 8.5)
                table.scale(1.0, 1.25)
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_text_props(weight="bold")
                    if col == 0 and row > 0:
                        cell.set_text_props(ha="left")
                self.pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                self.table_pages += 1

    def _style_table_cells(self, table, header_fontsize=None, body_fontsize=None,
                           header_facecolor="0.92", first_col_left=True):
        """Apply consistent, readable table styling for dashboard pages."""
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("0.72")
            cell.set_linewidth(0.45)
            if row == 0:
                cell.set_facecolor(header_facecolor)
                cell.set_text_props(weight="bold", fontsize=header_fontsize)
            else:
                if row % 2 == 0:
                    cell.set_facecolor("0.985")
                if body_fontsize is not None:
                    cell.set_text_props(fontsize=body_fontsize)
                if first_col_left and col == 0:
                    cell.set_text_props(ha="left")

    def _add_table_block(self, ax, title, df, bbox, col_widths=None,
                         header_fontsize=8.2, body_fontsize=8.0):
        """Draw one compact table block inside an existing page."""
        df_fmt = _dashboard_sanitize_df(df, include_index=False)
        ax.text(bbox[0], bbox[1] + bbox[3] + 0.012, title,
                fontsize=11.5, fontweight="bold", va="bottom", ha="left")
        if df_fmt.empty:
            ax.text(bbox[0], bbox[1] + 0.5 * bbox[3], "No rows available.", fontsize=10)
            return
        n_cols = len(df_fmt.columns)
        if col_widths is None:
            first_w = 0.24 if n_cols > 1 else 1.0
            rest_w = (1.0 - first_w) / max(n_cols - 1, 1)
            col_widths = [first_w] + [rest_w] * (n_cols - 1)
        table = ax.table(
            cellText=df_fmt.values,
            colLabels=df_fmt.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=col_widths,
            bbox=bbox,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(body_fontsize)
        self._style_table_cells(table, header_fontsize=header_fontsize, body_fontsize=body_fontsize)

    def _rename_runinfo_columns(self, df):
        """Shorten run-info headers enough to fit while preserving content."""
        rename = {
            "display_name": "method label",
            "method": "sampler",
            "weight_mode": "weights",
            "mala_step_size": "MALA dt",
            "score_norm_initial": "score norm init",
            "score_norm_mean": "score norm mean",
            "score_norm_final": "score norm final",
            "score_norm_max": "score norm max",
            "pde_likelihood_evals": "PDE logL evals",
            "pde_score_evals": "PDE score evals",
            "pde_gn_hessian_evals": "PDE GN Hess evals",
            "pde_solve_count": "PDE solves",
            "runtime_seconds": "runtime (s)",
            "reference_method": "reference",
            "N_ref": "N ref",
            "mala_steps": "MALA steps",
            "mala_burnin": "MALA burnin",
        }
        return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    def add_results_tables(self, results_df, results_runinfo_df):
        """Add exactly two table pages: canonical metrics and readable run-info.

        The metrics page keeps the saved *_metrics.csv / tables.tex layout. The
        run-info page preserves the saved *_runinfo.csv contents, but splits the
        many accounting columns into three normal table blocks on one page so it
        remains legible instead of becoming a tiny one-line wide table.
        """
        if not self.enabled:
            return

        # Page 1: metric rows x sampler columns, matching *_metrics.csv / tables.tex.
        metrics_fmt = _dashboard_sanitize_df(results_df, include_index=True)
        fig, ax = plt.subplots(figsize=(12.5, 8.5))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        ax.set_title("Metrics table (saved *_metrics.csv / tables.tex layout)",
                     fontsize=18, fontweight="bold", pad=14)
        n_rows = max(len(metrics_fmt), 1)
        n_cols = max(len(metrics_fmt.columns), 1)
        first_w = 0.34 if n_cols > 1 else 0.95
        rest_w = (0.94 - first_w) / max(n_cols - 1, 1)
        col_widths = [first_w] + [rest_w] * (n_cols - 1)
        table = ax.table(
            cellText=metrics_fmt.values,
            colLabels=metrics_fmt.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=col_widths,
            bbox=[0.035, 0.055, 0.93, 0.86],
        )
        table.auto_set_font_size(False)
        # Larger text than the previous version; still adapts if many methods are added.
        body_fs = min(10.5, max(7.2, 120.0 / (n_rows + 0.75 * n_cols)))
        header_fs = min(10.5, body_fs + 0.5)
        table.set_fontsize(body_fs)
        self._style_table_cells(table, header_fontsize=header_fs, body_fontsize=body_fs)
        self.pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        self.table_pages += 1

        # Page 2: normal, readable run-info blocks on one page.
        runinfo = pd.DataFrame(results_runinfo_df).copy()
        target = ""
        if "target" in runinfo.columns and len(runinfo) > 0:
            target = str(runinfo["target"].iloc[0])
        runinfo = self._rename_runinfo_columns(runinfo)
        fig, ax = plt.subplots(figsize=(15.5, 10.0))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        title = "Run-info table (saved *_runinfo.csv, split for readability)"
        if target:
            title += f" - {target}"
        ax.set_title(title, fontsize=18, fontweight="bold", pad=14)

        def cols_present(cols):
            return [c for c in cols if c in runinfo.columns]

        config_cols = cols_present([
            "method label", "sampler", "weights", "N ref", "steps",
            "MALA steps", "MALA burnin", "MALA dt", "reference", "runtime (s)",
        ])
        score_cols = cols_present([
            "method label", "score_norm", "score norm init", "score norm mean",
            "score norm final", "score norm max",
        ])
        budget_cols = cols_present([
            "method label", "PDE logL evals", "PDE score evals", "PDE GN Hess evals", "PDE solves",
        ])

        # Normalized column widths for each block. First column gets label width;
        # remaining columns share the rest.
        def widths(n, first=0.24):
            if n <= 1:
                return [1.0]
            return [first] + [(1.0 - first) / (n - 1)] * (n - 1)

        if config_cols:
            self._add_table_block(
                ax, "Sampler configuration and runtime", runinfo[config_cols],
                bbox=[0.035, 0.635, 0.93, 0.265], col_widths=widths(len(config_cols), first=0.22),
                header_fontsize=8.8, body_fontsize=8.7,
            )
        if score_cols:
            self._add_table_block(
                ax, "Score-norm diagnostics", runinfo[score_cols],
                bbox=[0.035, 0.355, 0.93, 0.185], col_widths=widths(len(score_cols), first=0.30),
                header_fontsize=9.2, body_fontsize=9.0,
            )
        if budget_cols:
            self._add_table_block(
                ax, "PDE evaluation budget", runinfo[budget_cols],
                bbox=[0.035, 0.115, 0.93, 0.165], col_widths=widths(len(budget_cols), first=0.30),
                header_fontsize=9.2, body_fontsize=9.0,
            )

        # If future runinfo files add columns not covered above, surface them in a small note
        # instead of silently dropping them.
        used = set(config_cols + score_cols + budget_cols + ["target", "label"])
        extra = [c for c in runinfo.columns if c not in used]
        if extra:
            ax.text(0.035, 0.055, "Additional run-info columns: " + ", ".join(extra),
                    fontsize=8.0, alpha=0.75, ha="left", va="bottom")
        self.pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        self.table_pages += 1

    def _figure_sort_key(self, path):
        name = os.path.basename(path)
        m = re.search(r"_figure_(\d+)_", name)
        return (int(m.group(1)) if m else 10_000, name)

    def add_image_page(self, image_path):
        """Embed one saved PNG/JPG as a full dashboard page."""
        if not self.enabled or not os.path.exists(image_path):
            return
        img = mpimg.imread(image_path)
        h, w = img.shape[:2]
        aspect = w / max(h, 1)
        if aspect > 2.2:
            figsize = (18.0, 7.0)
        elif aspect > 1.35:
            figsize = (15.5, 9.0)
        else:
            figsize = (11.0, 10.0)
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.015, 0.015, 0.97, 0.97])
        ax.imshow(img)
        ax.axis("off")
        self.pdf.savefig(fig, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        self.figure_pages += 1

    def add_run_results_png_figures(self, run_results_dir):
        """Append all saved run-results PNG figures, sorted by figure number.

        This is intentionally based on the files saved by sampling.py's patched
        plt.show() hook, so dashboard coverage matches the normal run-results
        directory exactly: ESS, PCA, field reconstructions, wavefields, boundary
        traces, curvature spectra, and any future diagnostics.
        """
        if not self.enabled or not run_results_dir or not os.path.isdir(run_results_dir):
            return
        pngs = sorted(glob.glob(os.path.join(run_results_dir, "*.png")), key=self._figure_sort_key)
        for path in pngs:
            self.add_image_page(path)

    def add_figure(self, fig=None, close=False):
        if not self.enabled:
            return
        if fig is None:
            fig = plt.gcf()
        try:
            fig.savefig(self.pdf, format="pdf", bbox_inches="tight")
            self.figure_pages += 1
        except Exception as exc:
            self.add_text_page("Figure capture failed", [repr(exc)])
        if close:
            plt.close(fig)

    def capture_new_figures(self, close=False):
        if not self.enabled:
            return
        for num in list(plt.get_fignums()):
            fig = plt.figure(num)
            fig_id = id(fig)
            if fig_id not in self._seen_fig_ids:
                self.add_figure(fig, close=close)
                self._seen_fig_ids.add(fig_id)

    def close(self):
        if self.enabled and self.pdf is not None:
            self.pdf.close()
            self.pdf = None


def dashboard_copy_into_run_dir(dashboard_path, results_df_path=None):
    """Copy the dashboard into the run-results directory so it is included in the zip."""
    if not dashboard_path or not os.path.exists(dashboard_path) or results_df_path is None:
        return dashboard_path
    run_dir = os.path.dirname(os.path.abspath(results_df_path))
    os.makedirs(run_dir, exist_ok=True)
    dest = os.path.join(run_dir, os.path.basename(dashboard_path))
    if os.path.abspath(dest) != os.path.abspath(dashboard_path):
        shutil.copy2(dashboard_path, dest)
    return dest




################################################################################

# ==========================================
# 0. KL basis generation (match old script)
# ==========================================
os.makedirs('data', exist_ok=True)

N = 32
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ell = 0.1
sigma_prior = 1.0
q_max = 100

dists = cdist(coords, coords)
C = sigma_prior ** 2 * np.exp(-dists / ell)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/Darcy_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files (follow old I/O path exactly)
# ==========================================
num_observation = 120
num_holdout_observation = 30
num_truncated_series = 32
seed = 42

dimension_of_PoI = N * N

interior_mask = np.ones((N, N), dtype=bool)
interior_mask[0, :] = False
interior_mask[-1, :] = False
interior_mask[:, 0] = False
interior_mask[:, -1] = False
interior_indices = jnp.array(np.where(interior_mask.ravel())[0])

key = jax.random.PRNGKey(seed)
obs_indices_train = np.array(
    jax.random.choice(key, interior_indices, shape=(num_observation,), replace=False)
)
remaining_interior_indices = np.setdiff1d(np.asarray(interior_indices), obs_indices_train)
key_holdout = jax.random.PRNGKey(seed + 1)
obs_indices_holdout = np.array(
    jax.random.choice(key_holdout, jnp.array(remaining_interior_indices), shape=(num_holdout_observation,), replace=False)
)
obs_indices = obs_indices_train

# Load / truncate / resave exactly like the old script rather than using the
# in-memory eigendecomposition directly. This keeps the modular version aligned
# with the old data-generation path.
df_modes = pd.read_csv('data/Darcy_Basis_Modes.csv', header=None)
if isinstance(df_modes.iloc[0, 0], str):
    df_modes = pd.read_csv('data/Darcy_Basis_Modes.csv')

modes_raw = df_modes.to_numpy().flatten()
num_modes_available = modes_raw.size // dimension_of_PoI
full_basis = modes_raw.reshape((dimension_of_PoI, num_modes_available))
basis_truncated = full_basis[:, :num_truncated_series]

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices_train).to_csv('data/obs_locations.csv', index=False, header=False)

# Match old script's reload-from-disk path too.
df_Basis = pd.read_csv('data/Basis.csv', header=None)
df_obs = pd.read_csv('data/obs_locations.csv', header=None)

basis_raw = df_Basis.to_numpy().flatten()
if basis_raw.size % dimension_of_PoI == 1:
    basis_raw = basis_raw[1:]
basis_raw = basis_raw.astype(np.float64)

if basis_raw.size % dimension_of_PoI != 0:
    raise ValueError(
        f"Basis file size {basis_raw.size} is not divisible by grid size {dimension_of_PoI}."
    )

num_modes_in_file = basis_raw.size // dimension_of_PoI
full_basis = np.reshape(basis_raw, (dimension_of_PoI, num_modes_in_file))
Basis = jnp.array(full_basis[:, :num_truncated_series], dtype=jnp.float64)

obs_raw = df_obs.to_numpy().flatten()
if obs_raw.size == num_observation + 1:
    obs_raw = obs_raw[1:]
obs_raw = obs_raw.astype(int)
if obs_raw.size > num_observation:
    obs_raw = obs_raw[:num_observation]
elif obs_raw.size < num_observation:
    raise ValueError(f"Obs file only has {obs_raw.size} locations, need {num_observation}.")
obs_locations_train = jnp.array(obs_raw, dtype=int)
obs_locations_holdout = jnp.array(obs_indices_holdout, dtype=int)
obs_locations = obs_locations_train

# ==========================================
# 2. Physics: Darcy flow
# ==========================================
jax.config.update("jax_enable_x64", True)

NOISE_STD = 0.001

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

    rhs = f_darcy[_int_rows, _int_cols]
    return A, rhs


@jax.jit
def solve_forward(alpha):
    log_k = jnp.reshape(Basis @ alpha, (N, N))
    k_field = jnp.exp(log_k)
    A, rhs = _assemble_darcy_vectorized(k_field)
    p_int = jnp.linalg.solve(A, rhs)
    p_full = jnp.zeros(N * N, dtype=jnp.float64)
    p_full = p_full.at[int_flat].set(p_int)
    return p_full[obs_locations_train]


@jax.jit
def solve_forward_holdout(alpha):
    log_k = jnp.reshape(Basis @ alpha, (N, N))
    k_field = jnp.exp(log_k)
    A, rhs = _assemble_darcy_vectorized(k_field)
    p_int = jnp.linalg.solve(A, rhs)
    p_full = jnp.zeros(N * N, dtype=jnp.float64)
    p_full = p_full.at[int_flat].set(p_int)
    return p_full[obs_locations_holdout]


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
HESS_MIN = 1e-6
HESS_MAX = 1e6
GNL_PILOT_N = 1024
GNL_STIFF_LAMBDA_CUT = HESS_MAX
GNL_USE_DOMINANT_PARTICLE_NEWTON = True
DEFAULT_N_GEN = 2000
N_REF = 2000
BUILD_GNL_BANKS = False


# ==========================================
# Iterative transport benchmark configuration
# ==========================================
# This script runs Algorithm 1 (alternating likelihood-corrected transport) on
# the Darcy inverse problem.  Each method alternates a public transport node T_r
# with a hidden DRC ratio node R_r.  The ratio node stores likelihood-correction
# weights for the next transport reference, but only rounds listed in
# DISPLAY_TRANSPORT_ROUNDS are shown in metrics and plots.

def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


N_REF = _env_int('IP_ITER_N_REF', _env_int('IP_DARCY_ITER_N_REF', N_REF))
DEFAULT_N_GEN = _env_int('IP_ITER_DEFAULT_N_GEN', _env_int('IP_DARCY_ITER_DEFAULT_N_GEN', N_REF))
ITERATIVE_TRANSPORT_ROUNDS = _env_int('IP_ITER_ROUNDS', 3)
DISPLAY_TRANSPORT_ROUNDS = {1, 3}
TRANSPORT_STEPS = _env_int('IP_ITER_TRANSPORT_STEPS', 200)

DRC_RATIO_COMMON = dict(
    transition_w='ou',
    drc_pf_steps=_env_int('IP_ITER_DRC_PF_STEPS', 32),
    drc_divergence=os.environ.get('IP_ITER_DRC_DIVERGENCE', 'auto'),
    drc_div_probes=_env_int('IP_ITER_DRC_DIV_PROBES', 1),
    drc_eval_batch_size=_env_int('IP_ITER_DRC_EVAL_BATCH_SIZE', 32),
    drc_clip=_env_float('IP_ITER_DRC_CLIP', 20.0),
    drc_temperature=_env_float('IP_ITER_DRC_TEMPERATURE', 1.0),
    drc_fd_eps=_env_float('IP_ITER_DRC_FD_EPS', 1e-3),
)

METHOD_SPECS = OrderedDict([
    ('LFGI', {
        'init': 'ce_hlsi',
        'display': 'GN-LFGI',
    }),
    ('MatrixBlend', {
        'init': 'matrix_blend',
        'display': 'Local Matrix Blend',
    }),
    ('ScalarBlend', {
        'init': 'scalar_blend',
        'display': 'Local Scalar Blend',
    }),
    ('UnifMatrixBlend', {
        'init': 'uniform_matrix_blend',
        'display': 'Uniform Matrix Blend',
    }),
    ('UnifScalarBlend', {
        'init': 'uniform_scalar_blend',
        'display': 'Uniform Scalar Blend',
    }),
    ('Tweedie', {
        'init': 'tweedie',
        'display': 'Tweedie',
    }),
])


def make_iterative_transport_sampler_configs(method_specs, rounds=ITERATIVE_TRANSPORT_ROUNDS):
    """Build Algorithm 1 schedules: T_r transport, then hidden R_r ratio update."""
    configs = OrderedDict()
    for method_key, spec in method_specs.items():
        init_name = spec['init']
        display_base = spec['display']
        prev_ratio_label = None

        for round_idx in range(1, int(rounds) + 1):
            transport_label = f'{method_key}-T{round_idx}'
            transport_cfg = {
                'init': init_name,
                'init_weights': 'None' if round_idx == 1 else 'DRC',
                'transition_w': 'ou',
                'init_steps': TRANSPORT_STEPS,
                'mala_steps': 0,
                'mala_burnin': 0,
                'log_mean_ess': True,
                'include_results': round_idx in DISPLAY_TRANSPORT_ROUNDS,
                'display_name': f'{display_base} round {round_idx}',
            }
            if prev_ratio_label is not None:
                transport_cfg['ref_source'] = prev_ratio_label
            configs[transport_label] = transport_cfg

            ratio_label = f'{method_key}-R{round_idx}'
            ratio_cfg = {
                'ref_source': transport_label,
                'init': 'DRC-R',
                # The R-node uses the same base score estimator as the preceding
                # T-node, but builds it on the current unweighted endpoint cloud.
                # DRC weights are only consumed by the next transport node.
                'init_weights': 'None',
                'drc_score_init': init_name,
                'drc_score_init_weights': 'None',
                'include_results': False,
                'display_name': f'{display_base} ratio round {round_idx}',
            }
            ratio_cfg.update(DRC_RATIO_COMMON)
            configs[ratio_label] = ratio_cfg
            prev_ratio_label = ratio_label

    return configs


SAMPLER_CONFIGS = make_iterative_transport_sampler_configs(METHOD_SPECS)

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
run_ctx = init_run_results('darcy_iterative_transport')
DASHBOARD_PDF_PATH = os.path.join(
    run_ctx['run_results_dir'],
    f"{run_ctx['run_results_stem']}_summary_dashboard.pdf",
)

RUN_COMMAND_HINT = (
    'IP_ITER_N_REF={n_ref} IP_ITER_ROUNDS={rounds} '
    'IP_ITER_TRANSPORT_STEPS={steps} IP_ITER_DRC_PF_STEPS={pf_steps} '
    'python darcy_flow_iterative.py'
).format(
    n_ref=N_REF,
    rounds=ITERATIVE_TRANSPORT_ROUNDS,
    steps=TRANSPORT_STEPS,
    pf_steps=DRC_RATIO_COMMON['drc_pf_steps'],
)

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
y_holdout_clean_np = np.array(solve_forward_holdout(jnp.array(alpha_true_np)))
y_holdout_obs_np = y_holdout_clean_np + np.random.normal(0.0, NOISE_STD, size=y_holdout_clean_np.shape)


dashboard = DashboardPDF(
    DASHBOARD_PDF_PATH,
    title='Darcy flow iterative transport dashboard',
)
dashboard.add_text_page(
    'Darcy flow iterative transport dashboard',
    [
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        'This dashboard contains the two canonical saved-results tables plus every PNG diagnostic plot saved in the run directory. Public rows are transport outputs T_r; ratio nodes R_r are hidden and only carry likelihood-correction weights forward.',
        'Included PNG diagnostics: ESS vs diffusion time, PCA histograms, log-permeability reconstructions, pressure fields, and permeability fields.',
        'Tables are intentionally limited to two pages: metrics plus a readable split run-info page.',
        'Random progress output from precomputation / Hessian batching is intentionally excluded.',
        f"run_results_dir = {run_ctx['run_results_dir']}",
        '',
        f'seed = {seed}',
        f'ACTIVE_DIM = {ACTIVE_DIM}',
        f'N_REF = {N_REF}',
        f'DEFAULT_N_GEN = {DEFAULT_N_GEN}',
        f'ITERATIVE_TRANSPORT_ROUNDS = {ITERATIVE_TRANSPORT_ROUNDS}',
        f'DISPLAY_TRANSPORT_ROUNDS = {sorted(DISPLAY_TRANSPORT_ROUNDS)}',
        f'TRANSPORT_STEPS = {TRANSPORT_STEPS}',
        f'DRC_RATIO_COMMON = {DRC_RATIO_COMMON}',
        f'NOISE_STD = {NOISE_STD}',
        f'N = {N}, num_observation = {num_observation}, num_holdout_observation = {num_holdout_observation}',
        f'HESS_MIN = {HESS_MIN}, HESS_MAX = {HESS_MAX}',
        f'BUILD_GNL_BANKS = {BUILD_GNL_BANKS}',
        f'GNL_PILOT_N = {GNL_PILOT_N}',
        f'GNL_STIFF_LAMBDA_CUT = {GNL_STIFF_LAMBDA_CUT}',
        f'PLOT_NORMALIZER = {PLOT_NORMALIZER}',
        f'run command = {RUN_COMMAND_HINT}',
    ],
)

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

pipeline = run_standard_sampler_pipeline(
    prior_model,
    lik_model,
    SAMPLER_CONFIGS,
    n_ref=N_REF,
    build_gnl_banks=BUILD_GNL_BANKS,
    compute_pou=False,
)
precomp = pipeline['precomp']
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

try:
    metrics = compute_heldout_predictive_metrics(
        samples,
        metrics,
        heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
        batched_forward_eval_fn=lambda a_batch: np.asarray(
            batch_solve_forward_holdout(jnp.asarray(a_batch, dtype=jnp.float64))
        ),
        batched_forward_eval_batch_size=8,
        y_holdout_obs_np=y_holdout_obs_np,
        noise_std=NOISE_STD,
        display_names=display_names,
        min_valid=10,
    )
except Exception as exc:
    print(f"WARNING: held-out predictive metrics failed and will be skipped: {exc}")

mean_pressures = {}
mean_permeabilities = {}
norm_true_pressure = np.linalg.norm(true_pressure) + 1e-12

print('\n=== Darcy physical-space metrics ===')
print(f"{'Method':<32} | {'LogPerm RelL2(%)':<18} | {'Pearson':<10} | {'RMSE_a':<12} | {'PressureRel':<12} | {'SensorRel':<12} | {'HeldoutNLL':<12} | {'HeldoutZ2':<12}")
print('-' * 160)
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
        f"{display_names.get(label, label):<32} | {logperm_rel_pct:<18.4f} | "
        f"{metrics[label].get('Pearson_field', float('nan')):<10.4f} | {metrics[label]['RMSE_alpha']:<12.4e} | "
        f"{pressure_rel:<12.4e} | {metrics[label].get('FwdRelErr', float('nan')):<12.4e} | "
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
    target_name='Darcy flow log-permeability',
    display_names=display_names,
    reference_name=reference_title,
)

dashboard.add_results_tables(results_df, results_runinfo_df)

save_reproducibility_log(
    title='Darcy flow iterative transport reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'DEFAULT_N_GEN': DEFAULT_N_GEN,
        'GNL_PILOT_N': GNL_PILOT_N,
        'GNL_STIFF_LAMBDA_CUT': GNL_STIFF_LAMBDA_CUT,
        'GNL_USE_DOMINANT_PARTICLE_NEWTON': GNL_USE_DOMINANT_PARTICLE_NEWTON,
        'HESS_MAX': HESS_MAX,
        'HESS_MIN': HESS_MIN,
        'N': N,
        'NOISE_STD': NOISE_STD,
        'N_REF': N_REF,
        'ITERATIVE_TRANSPORT_ROUNDS': ITERATIVE_TRANSPORT_ROUNDS,
        'DISPLAY_TRANSPORT_ROUNDS': sorted(DISPLAY_TRANSPORT_ROUNDS),
        'TRANSPORT_STEPS': TRANSPORT_STEPS,
        'DRC_RATIO_COMMON': DRC_RATIO_COMMON,
        'METHOD_SPECS': METHOD_SPECS,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'USE_GAUSS_NEWTON_HESSIAN': True,
        'PLOT_NORMALIZER': PLOT_NORMALIZER,
        'plot_normalizer_key': plot_normalizer_key,
        'plot_normalizer_title': plot_normalizer_title,
        'dimension_of_PoI': dimension_of_PoI,
        'num_modes_available': num_modes_available,
        'num_observation': num_observation,
        'num_holdout_observation': num_holdout_observation,
        'num_truncated_series': num_truncated_series,
        'obs_indices_train': obs_indices_train,
        'obs_indices_holdout': obs_indices_holdout,
        'obs_locations_train': obs_locations_train,
        'obs_locations_holdout': obs_locations_holdout,
        'sigma_prior': sigma_prior,
        'ell': ell,
    },
    extra_sections={
        'saved_results_files': {
            'metrics_csv': results_df_path,
            'runinfo_csv': results_runinfo_df_path,
            'dashboard_pdf': DASHBOARD_PDF_PATH,
        },
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
print('\nVisualizing Darcy field reconstructions...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1

if n_cols > 1:
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))

    vis_anchor_key = plot_normalizer_key if plot_normalizer_key in mean_fields else next(iter(mean_fields.keys()))

    vmin = float(np.min(true_field))
    vmax = float(np.max(true_field))

    if vis_anchor_key in samples and vis_anchor_key in mean_fields:
        anchor_vis_samps = get_valid_samples(samples[vis_anchor_key])[:1000]
        if anchor_vis_samps.shape[0] > 0:
            anchor_vis_fields = reconstruct_log_permeability(anchor_vis_samps[:, :ACTIVE_DIM])
            max_err = max(1e-12, float(np.abs(mean_fields[vis_anchor_key] - true_field).max()))
            max_std = max(1e-12, float(np.std(anchor_vis_fields, axis=0).max()))
        else:
            max_err = 1e-12
            max_std = 1e-12
    else:
        max_err = 1e-12
        max_std = 1e-12

    if vis_anchor_key not in mean_fields:
        max_err = 1e-12
        max_std = 1e-12
        for label in methods_to_plot:
            mean_f = mean_fields[label]
            max_err = max(max_err, np.abs(mean_f - true_field).max())
            samps = get_valid_samples(samples[label])[:500]
            if samps.shape[0] > 0:
                fields = reconstruct_log_permeability(samps[:, :ACTIVE_DIM])
                max_std = max(max_std, np.std(fields, axis=0).max())

    im0 = axes[0, 0].imshow(true_field, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, 0].scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.7, label='Sensors')
    axes[0, 0].set_title('Ground Truth\nLog-Permeability $m(x)$', fontsize=18)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[3, 0].imshow(true_field, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[3, 0].set_title('Ground Truth', fontsize=14)
    axes[3, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    for i, label in enumerate(methods_to_plot):
        col = i + 1
        mean_f = mean_fields[label]

        axes[0, col].imshow(mean_f, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[0, col].scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.5)
        axes[0, col].set_title(f"{display_names.get(label, label)}\nMean Posterior", fontsize=18)
        axes[0, col].axis('off')

        err_f = np.abs(mean_f - true_field)
        axes[1, col].imshow(err_f, cmap='inferno', origin='lower', vmin=0, vmax=max_err)
        axes[1, col].set_title(f"Error Map\n(Max: {err_f.max():.2f})", fontsize=16)
        axes[1, col].axis('off')

        samps = get_valid_samples(samples[label])[:1000]
        if samps.shape[0] > 0:
            fields = reconstruct_log_permeability(samps[:, :ACTIVE_DIM])
            std_f = np.std(fields, axis=0)
        else:
            std_f = np.zeros_like(true_field)
        axes[2, col].imshow(std_f, cmap='viridis', origin='lower', vmin=0, vmax=max_std)
        axes[2, col].set_title(f"Uncertainty\n(Max std: {std_f.max():.2f})", fontsize=16)
        axes[2, col].axis('off')

        if samps.shape[0] > 0:
            sample_field = reconstruct_log_permeability(samps[:1, :ACTIVE_DIM])[0]
            axes[3, col].imshow(sample_field, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
            axes[3, col].set_title('Posterior Sample', fontsize=14)
        else:
            axes[3, col].text(0.5, 0.5, 'No valid\nsamples', ha='center', va='center', transform=axes[3, col].transAxes)
        axes[3, col].axis('off')

    plt.suptitle(f'Inverse Darcy flow iterative transport (d={ACTIVE_DIM})', fontsize=22, y=1.01)
    plt.tight_layout()
    plt.show()

    print('\nVisualizing pressure fields...')
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

    plt.suptitle(f'Inverse Darcy flow iterative transport (d={ACTIVE_DIM}): pressure field', fontsize=16, y=1.05)
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

    plt.suptitle(f'Inverse Darcy flow iterative transport (d={ACTIVE_DIM}): permeability field', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()
else:
    print('WARNING: no Darcy methods available for visualization.')

try:
    sampling._save_all_open_figures_to_run_results()
except Exception as exc:
    print(f'WARNING: final open-figure save before dashboard failed: {exc}')

if DASHBOARD_SHOW_FIGURES:
    dashboard.add_run_results_png_figures(run_ctx['run_results_dir'])
dashboard.close()
plt.close('all')
try:
    run_results_zip_path = zip_run_results_dir(extra_paths=[DASHBOARD_PDF_PATH])
except TypeError:
    run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Dashboard PDF: {DASHBOARD_PDF_PATH}')
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Darcy flow iterative transport comparison pipeline complete ===')
