# -*- coding: utf-8 -*-
"""Navier-Stokes inverse-problem iterative-transport benchmark.

Runs Algorithm 1, alternating likelihood-corrected transport, for GN-LFGI,
local matrix blend, local scalar blend, uniform matrix blend, uniform scalar
blend, and Tweedie. Each method performs three transport/ratio rounds and
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

import sys, importlib, linecache, os

################################################################################
# Make sure /content itself is before parent dirs.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Clear stale source-line cache and stale imported module.
linecache.clearcache()
if "sampling" in sys.modules:
    del sys.modules["sampling"]

# Slurm/repository default: import the shared module as <repo-root>/sampling.py.
# A suffixed module can still be selected explicitly with HLSI_SAMPLING_MODULE,
# but no suffixed helper is required for normal GitHub/Slurm use.
SAMPLING_MODULE_NAME = os.environ.get("HLSI_SAMPLING_MODULE", "sampling")
sampling = importlib.import_module(SAMPLING_MODULE_NAME)
importlib.reload(sampling)
# Preserve the historical module name for downstream `from sampling import ...`.
sys.modules["sampling"] = sampling

print("Using:", sampling.__file__)
print("DRC test:", sampling.canonicalize_init_weights("DRC"))


# The shared helper still uses the historical implementation name `ce_hlsi`
# internally.  For this iterative-transport script, every plot/table-facing
# occurrence of that method should be displayed as LFGI.  The score initializer
# remains `ce_hlsi` for compatibility with sampling.py; only labels are changed.
_orig_drc_method_pretty_name = getattr(sampling, "_drc_method_pretty_name", None)
if _orig_drc_method_pretty_name is not None:
    def _navier_lfgi_drc_method_pretty_name(label, cfg=None):
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

    sampling._drc_method_pretty_name = _navier_lfgi_drc_method_pretty_name
    for _fn_name in ("save_drc_energy_comparison_grid", "finalize_drc_energy_benchmark_plots"):
        _fn = getattr(sampling, _fn_name, None)
        if _fn is not None and hasattr(_fn, "__globals__"):
            _fn.__globals__["_drc_method_pretty_name"] = _navier_lfgi_drc_method_pretty_name


# Compatibility shim for current sampling.py versions. In the direct transport
# path, run_single_sampler_config calls resolve_hlsi_gate_law(init, init_weights,
# gate_rho=...), while the current helper signature expects the second positional
# argument to be gate_rho. DRC/density-only scripts avoid this path; iterative
# transport with a direct LFGI T-step hits it. Treat a positional init-weight
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
# node. The shared helper's reference-bank constructor correctly rejects a
# source with no finite samples, but that stops the full comparison. For this
# experiment we keep the run alive by replacing non-finite output rows at the
# boundary of each sampler node. Partial blow-ups are repaired from the finite
# rows of the same output; complete blow-ups fall back to the node's finite
# input reference bank, and finally to fresh prior samples if no finite input
# bank exists. The run-info table records when this happens.
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
    compute_map_laplace_density_baseline,
    make_density_manuscript_table,
    run_drc_pf_sensitivity_benchmark,
    init_run_results,
    make_physics_likelihood,
    make_posterior_score_fn,
    pearson_corr_array,
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

# ==========================================
# Config generator
# ==========================================
N = 32
# Moderately harder inverse-problem setting: fewer and mildly anisotropic
# sensors, a somewhat larger latent state, and a rougher anisotropic prior.
# This is intentionally harder than the original uploaded script but avoids
# the previous over-hardened regime where CE-HLSI bootstrap references could
# collapse to non-finite samples.
num_observation = 80
num_holdout_observation = 40
num_truncated_series = 40
num_modes_available = 120
seed = 42
prior_length_scale_x = 0.075
prior_length_scale_y = 0.13
INITIAL_VORTICITY_SCALE = 1.20

os.makedirs('data', exist_ok=True)
rng = np.random.default_rng(seed)


def build_anisotropic_periodic_fourier_basis(N, q_max=120, length_scale_x=0.075, length_scale_y=0.13):
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
        # Directional spectral envelope: high-kx modes survive much longer than high-ky modes.
        # This creates an anisotropic rough/smooth split, hence a more ill-conditioned posterior.
        amp = np.exp(-0.5 * ((2.0 * np.pi * length_scale_x * kx) ** 2 + (2.0 * np.pi * length_scale_y * ky) ** 2))
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


full_basis, basis_meta = build_anisotropic_periodic_fourier_basis(
    N=N,
    q_max=num_modes_available,
    length_scale_x=prior_length_scale_x,
    length_scale_y=prior_length_scale_y,
)
basis_truncated = full_basis[:, :num_truncated_series]
basis_modes_path = 'data/NavierStokes_Basis_Modes_generated.csv'
pd.DataFrame(full_basis).to_csv(basis_modes_path, index=False, header=False)
pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
# Mildly anisotropic sensor geometry.  Compared with the original random
# 120-sensor problem, this keeps fewer sensors and some directional structure;
# compared with the previous hard version, it restores enough spatial coverage
# that bootstrap reference banks remain numerically sane.
row_grid_np, col_grid_np = np.indices((N, N))
train_mask = (
    np.isin(col_grid_np, [int(0.55 * N), int(0.70 * N), int(0.85 * N)])
    | ((row_grid_np == int(0.25 * N)) & (col_grid_np >= int(0.25 * N)))
    | ((row_grid_np == int(0.58 * N)) & (col_grid_np <= int(0.80 * N)))
)
holdout_mask = (
    np.isin(col_grid_np, [int(0.18 * N), int(0.38 * N), int(0.62 * N)])
    | ((row_grid_np == int(0.75 * N)) & (col_grid_np <= int(0.75 * N)))
)
train_candidates = np.flatnonzero(train_mask.ravel())
holdout_candidates = np.setdiff1d(np.flatnonzero(holdout_mask.ravel()), train_candidates)
if train_candidates.size < num_observation:
    raise ValueError(f"Need at least {num_observation} training candidates, found {train_candidates.size}.")
if holdout_candidates.size < num_holdout_observation:
    raise ValueError(f"Need at least {num_holdout_observation} holdout candidates, found {holdout_candidates.size}.")
obs_indices_train = rng.choice(train_candidates, size=(num_observation,), replace=False)
obs_indices_holdout = rng.choice(holdout_candidates, size=(num_holdout_observation,), replace=False)
obs_indices = obs_indices_train
pd.DataFrame(obs_indices_train).to_csv('data/obs_locations.csv', index=False, header=False)

# ==========================================
# Physics
# ==========================================
jax.config.update("jax_enable_x64", True)
dimension_of_PoI = N ** 2
# Slightly lower viscosity and a slightly longer horizon than the original
# make the forward map harder, but not so chaotic/stiff that CE-HLSI
# bootstrapped references collapse.
nu = 7.5e-4
delta_t = 0.05
T_end = 10.5
num_time_steps = int(round(T_end / delta_t))

Basis = jnp.array(basis_truncated)
obs_locations = jnp.array(obs_indices, dtype=int)
holdout_locations = jnp.array(obs_indices_holdout, dtype=int)

x_1d = jnp.linspace(0.0, 1.0, N, endpoint=False)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d, indexing='ij')
freq_1d = jnp.fft.fftfreq(N, d=1.0 / N)
KX, KY = jnp.meshgrid(2.0 * jnp.pi * freq_1d, 2.0 * jnp.pi * freq_1d, indexing='ij')
K2 = KX ** 2 + KY ** 2
K2_safe = jnp.where(K2 == 0.0, 1.0, K2)
freq_abs_x = jnp.abs(jnp.fft.fftfreq(N, d=1.0 / N))
freq_abs_y = jnp.abs(jnp.fft.fftfreq(N, d=1.0 / N))
DEALIAS = ((freq_abs_x[:, None] <= (N / 3.0)) & (freq_abs_y[None, :] <= (N / 3.0))).astype(jnp.float64)
# Moderate multiscale anisotropic forcing.  This is stronger and less isotropic
# than the original single diagonal forcing, but backed off from the satanic
# version that made the likelihood too stiff.
forcing_field = (
    0.14 * jnp.sin(2.0 * jnp.pi * (1.5 * X_grid + 0.35 * Y_grid))
    + 0.08 * jnp.cos(2.0 * jnp.pi * (0.25 * X_grid + 2.5 * Y_grid))
    + 0.04 * jnp.sin(2.0 * jnp.pi * (3.0 * X_grid - 0.75 * Y_grid))
)
forcing_hat = jnp.fft.fftn(forcing_field)

# Mild anisotropic observation blur.  It adds ill-posedness relative to direct
# final-vorticity sensing, but it is much less aggressive than the previous
# filter and should leave bootstrap samples finite.
OBS_FILTER = jnp.exp(-0.5 * ((freq_abs_x[:, None] / 8.0) ** 2 + (freq_abs_y[None, :] / 4.0) ** 2))


def _observed_vorticity_field(omega_field):
    return jnp.fft.ifftn(jnp.fft.fftn(omega_field) * OBS_FILTER).real
CN_NUM = 1.0 - 0.5 * delta_t * nu * K2
CN_DEN = 1.0 + 0.5 * delta_t * nu * K2


def _latent_to_initial_vorticity(alpha):
    omega0 = INITIAL_VORTICITY_SCALE * jnp.reshape(Basis @ alpha, (N, N))
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
    omega_T = _observed_vorticity_field(solve_forward_full(alpha))
    return omega_T.reshape(-1)[obs_locations]


@jax.jit
def solve_forward_holdout(alpha):
    omega_T = _observed_vorticity_field(solve_forward_full(alpha))
    return omega_T.reshape(-1)[holdout_locations]


batch_solve_forward_holdout = jax.jit(jax.vmap(solve_forward_holdout))


# ==========================================
# Shared sampling / density-benchmark config
# ==========================================
ACTIVE_DIM = num_truncated_series
NOISE_STD = 0.005
HESS_MIN = -1e5
HESS_MAX = 5e7
GNL_PILOT_N = 1024
GNL_STIFF_LAMBDA_CUT = HESS_MAX
GNL_USE_DOMINANT_PARTICLE_NEWTON = True
BUILD_GNL_BANKS = False

PLOT_NORMALIZER = 'best'


# ==========================================
# Iterative transport benchmark configuration
# ==========================================
# This script runs Algorithm 1 (alternating likelihood-corrected transport) on
# the Navier-Stokes inverse problem. Each method alternates a public transport
# node T_r with a hidden DRC ratio node R_r. The ratio node stores likelihood-
# correction weights for the next transport reference, but only rounds listed in
# DISPLAY_TRANSPORT_ROUNDS are shown in metrics and plots.

def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


N_REF = _env_int('IP_ITER_N_REF', _env_int('IP_NAVIER_ITER_N_REF', 1000))
DEFAULT_N_GEN = _env_int('IP_ITER_DEFAULT_N_GEN', _env_int('IP_NAVIER_ITER_DEFAULT_N_GEN', N_REF))
ITERATIVE_TRANSPORT_ROUNDS = _env_int('IP_ITER_ROUNDS', 3)
DISPLAY_TRANSPORT_ROUNDS = {1, 3}
TRANSPORT_STEPS = _env_int('IP_ITER_TRANSPORT_STEPS', 200)
HELDOUT_BATCH_SIZE = _env_int('IP_ITER_HELDOUT_BATCH_SIZE', 8)

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
run_ctx = init_run_results('navier_stokes_iterative_transport')
DASHBOARD_PDF_PATH = os.path.join(
    run_ctx['run_results_dir'],
    f"{run_ctx['run_results_stem']}_summary_dashboard.pdf",
)

RUN_COMMAND_HINT = (
    'IP_ITER_N_REF={n_ref} IP_ITER_ROUNDS={rounds} '
    'IP_ITER_TRANSPORT_STEPS={transport_steps} IP_ITER_DRC_PF_STEPS={pf_steps} '
    'python navier_stokes.py'
).format(
    n_ref=N_REF,
    rounds=ITERATIVE_TRANSPORT_ROUNDS,
    transport_steps=TRANSPORT_STEPS,
    pf_steps=DRC_RATIO_COMMON['drc_pf_steps'],
)

# ==========================================
# Execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
alpha_true_np = np.random.randn(ACTIVE_DIM) * 0.60
y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
y_obs_np = y_clean_np + np.random.normal(0, NOISE_STD, size=y_clean_np.shape)
y_clean_holdout = solve_forward_holdout(jnp.array(alpha_true_np))
y_clean_holdout_np = np.array(y_clean_holdout)
y_holdout_obs_np = y_clean_holdout_np + np.random.normal(0, NOISE_STD, size=y_clean_holdout_np.shape)

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, _ = make_physics_likelihood(
    solve_forward,
    y_obs_np,
    NOISE_STD,
    use_gauss_newton_hessian=True,
    log_batch_size=50,
    grad_batch_size=25,
    hess_batch_size=2,
)
posterior_score_fn = make_posterior_score_fn(lik_model)


dashboard = DashboardPDF(
    DASHBOARD_PDF_PATH,
    title='Navier-Stokes iterative-transport dashboard',
)
dashboard.add_text_page(
    'Navier-Stokes iterative-transport dashboard',
    [
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        'This dashboard runs Algorithm 1, alternating likelihood-corrected transport, with the same base estimator used for each transport node and its associated ratio node.',
        'Displayed methods are transport rounds 1 and 3 only. Ratio nodes are hidden and only carry DRC likelihood-correction weights forward.',
        f"run_results_dir = {run_ctx['run_results_dir']}",
        '',
        f'run command = {RUN_COMMAND_HINT}',
        '',
        f'seed = {seed}',
        f'ACTIVE_DIM = {ACTIVE_DIM}',
        f'N_REF = {N_REF}',
        f'DEFAULT_N_GEN = {DEFAULT_N_GEN}',
        f'ITERATIVE_TRANSPORT_ROUNDS = {ITERATIVE_TRANSPORT_ROUNDS}',
        f'DISPLAY_TRANSPORT_ROUNDS = {sorted(DISPLAY_TRANSPORT_ROUNDS)}',
        f'TRANSPORT_STEPS = {TRANSPORT_STEPS}',
        f'DRC_RATIO_COMMON = {DRC_RATIO_COMMON}',
        f'METHOD_SPECS = {METHOD_SPECS}',
        f'NOISE_STD = {NOISE_STD}',
        f'N = {N}, num_observation = {num_observation}, num_holdout_observation = {num_holdout_observation}',
        f'prior_length_scale_x = {prior_length_scale_x}, prior_length_scale_y = {prior_length_scale_y}',
        f'INITIAL_VORTICITY_SCALE = {INITIAL_VORTICITY_SCALE}',
        f'nu = {nu}, delta_t = {delta_t}, T_end = {T_end}, num_time_steps = {num_time_steps}',
        f'HESS_MIN = {HESS_MIN}, HESS_MAX = {HESS_MAX}',
        f'BUILD_GNL_BANKS = {BUILD_GNL_BANKS}',
        f'GNL_PILOT_N = {GNL_PILOT_N}',
        f'GNL_STIFF_LAMBDA_CUT = {GNL_STIFF_LAMBDA_CUT}',
        f'PLOT_NORMALIZER = {PLOT_NORMALIZER}',
    ],
)

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
try:
    metrics = compute_heldout_predictive_metrics(
        samples,
        metrics,
        heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
        batched_forward_eval_fn=lambda a_batch: np.asarray(
            batch_solve_forward_holdout(jnp.asarray(a_batch, dtype=jnp.float64))
        ),
        batched_forward_eval_batch_size=HELDOUT_BATCH_SIZE,
        y_holdout_obs_np=y_holdout_obs_np,
        noise_std=NOISE_STD,
        display_names=display_names,
        min_valid=10,
    )
except Exception as exc:
    print(f"WARNING: held-out predictive metrics failed and will be skipped: {exc}")

Basis_np = np.array(Basis)
obs_locs_np = np.array(obs_locations)
obs_row = obs_locs_np // N
obs_col = obs_locs_np % N
holdout_locs_np = np.array(holdout_locations)
holdout_row = holdout_locs_np // N
holdout_col = holdout_locs_np % N
OBS_FILTER_NP = np.array(OBS_FILTER)


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
print(f"{'Method':<32} | {'Inv RelL2(%)':<12} | {'Pearson':<10} | {'Final RelL2':<12} | {'RMSE_alpha':<12} | {'FwdRelErr':<12} | {'HeldoutNLL':<12} | {'HeldoutZ2':<12}")
print('-' * 142)
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
    final_rel = float(np.linalg.norm(mean_final_fields[label] - true_final_field) / (np.linalg.norm(true_final_field) + 1e-12))
    metrics[label].update(dict(
        mean_latent=mean_latent,
        RMSE_alpha=float(np.sqrt(np.mean((mean_latent - alpha_true_np) ** 2))),
        RMSE_field=rmse_array(mean_field, true_field),
        RelL2_field=float(np.linalg.norm(mean_field - true_field) / norm_true),
        Pearson_field=pearson_corr_array(mean_field, true_field),
        RMSE_final_vorticity=rmse_array(mean_final_fields[label], true_final_field),
        RelL2_final_vorticity=final_rel,
        Pearson_final_vorticity=pearson_corr_array(mean_final_fields[label], true_final_field),
        SensorRMSE=rmse_array(y_pred, y_clean_np),
        SensorMaxAbsResidual=float(np.max(sensor_residuals[label])),
        FwdRelErr=fwd_rel,
    ))
    print(
        f"{display_names.get(label, label):<32} | {inv_rel_l2_pct:<12.4f} | "
        f"{metrics[label].get('Pearson_field', float('nan')):<10.4f} | "
        f"{metrics[label].get('RelL2_final_vorticity', float('nan')):<12.4e} | "
        f"{metrics[label].get('RMSE_alpha', float('nan')):<12.4e} | {fwd_rel:<12.4e} | "
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
    target_name='Navier-Stokes iterative transport',
    display_names=display_names,
    reference_name=reference_title,
)

dashboard.add_results_tables(results_df, results_runinfo_df)

save_reproducibility_log(
    title='Navier-Stokes iterative transport reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'N_REF': N_REF,
        'DEFAULT_N_GEN': DEFAULT_N_GEN,
        'ITERATIVE_TRANSPORT_ROUNDS': ITERATIVE_TRANSPORT_ROUNDS,
        'DISPLAY_TRANSPORT_ROUNDS': sorted(DISPLAY_TRANSPORT_ROUNDS),
        'TRANSPORT_STEPS': TRANSPORT_STEPS,
        'DRC_RATIO_COMMON': DRC_RATIO_COMMON,
        'METHOD_SPECS': METHOD_SPECS,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'RUN_COMMAND_HINT': RUN_COMMAND_HINT,
        'num_holdout_observation': num_holdout_observation,
        'HELDOUT_BATCH_SIZE': HELDOUT_BATCH_SIZE,
        'HESS_MIN': HESS_MIN,
        'HESS_MAX': HESS_MAX,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'GNL_PILOT_N': GNL_PILOT_N,
        'GNL_STIFF_LAMBDA_CUT': GNL_STIFF_LAMBDA_CUT,
        'GNL_USE_DOMINANT_PARTICLE_NEWTON': GNL_USE_DOMINANT_PARTICLE_NEWTON,
        'basis_modes_path': basis_modes_path,
        'prior_length_scale_x': prior_length_scale_x,
        'prior_length_scale_y': prior_length_scale_y,
        'INITIAL_VORTICITY_SCALE': INITIAL_VORTICITY_SCALE,
        'nu': nu,
        'delta_t': delta_t,
        'T_end': T_end,
        'num_time_steps': num_time_steps,
        'OBS_FILTER': OBS_FILTER_NP,
        'num_observation': num_observation,
        'sensor_geometry': 'mild anisotropic strips plus two cross rows',
        'NOISE_STD': NOISE_STD,
        'PLOT_NORMALIZER': PLOT_NORMALIZER,
        'plot_normalizer_key': plot_normalizer_key,
        'plot_normalizer_title': plot_normalizer_title,
        'DASHBOARD_PDF_PATH': DASHBOARD_PDF_PATH,
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
            'num_methods_with_mean_final_fields': len(mean_final_fields),
            'num_methods_with_ess_logs': len(ess_logs),
        },
    },
)


print('Visualizing anisotropic sensing geometry and observation filter...')
fig_geom, axes_geom = plt.subplots(1, 3, figsize=(13.5, 4.0))
mask_train = np.zeros((N, N), dtype=float)
mask_holdout = np.zeros((N, N), dtype=float)
mask_train[obs_row, obs_col] = 1.0
mask_holdout[holdout_row, holdout_col] = 1.0
axes_geom[0].imshow(mask_train, origin='lower', cmap='gray_r', vmin=0.0, vmax=1.0)
axes_geom[0].set_title('Training sensor mask')
axes_geom[0].axis('off')
axes_geom[1].imshow(mask_holdout, origin='lower', cmap='gray_r', vmin=0.0, vmax=1.0)
axes_geom[1].set_title('Held-out sensor mask')
axes_geom[1].axis('off')
im_filter = axes_geom[2].imshow(OBS_FILTER_NP, origin='lower', cmap='viridis')
axes_geom[2].set_title('Anisotropic observation blur filter')
axes_geom[2].axis('off')
plt.colorbar(im_filter, ax=axes_geom[2], fraction=0.046, pad=0.04)
plt.suptitle('Hard Navier-Stokes observation geometry', fontsize=15)
plt.tight_layout()
plt.show()


def _overlay(ax):
    ax.scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.6)


plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_initial_vorticity,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
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
max_resid = max(1e-12, max(float(sensor_residuals[label].max()) for label in sensor_residuals)) if sensor_residuals else 1.0
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

try:
    sampling._save_all_open_figures_to_run_results()
except Exception:
    pass

# Final artifact sweep: make sure figures created immediately before shutdown are
# saved into run_results before the dashboard and zip are finalized.
try:
    sampling._save_all_open_figures_to_run_results()
except Exception as exc:
    print(f'WARNING: final open-figure save before dashboard failed: {exc}')

dashboard.add_run_results_png_figures(run_ctx['run_results_dir'])
dashboard.close()

run_results_zip_path = zip_run_results_dir(extra_paths=[DASHBOARD_PDF_PATH])
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Dashboard PDF: {DASHBOARD_PDF_PATH}')
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Navier-Stokes iterative transport pipeline complete ===')
