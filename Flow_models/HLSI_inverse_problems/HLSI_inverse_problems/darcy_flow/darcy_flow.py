# -*- coding: utf-8 -*- 
"""Darcy flow inverse-problem density-evaluation benchmark.

Updated to match the current Navier-Stokes and Helmholtz density scripts:
build MALA posterior source/eval banks, fit Tweedie / scalar blend / LFGI
probability-flow normalized-density surrogates with independent score-signal
and gate particle banks, and evaluate density/energy diagnostics on a held-out
density-evaluation bank.

Typical repository/Slurm run:

    # Place this file at darcy_flow/darcy_flow.py.
    # Place the updated shared module at <repo-root>/sampling.py.
    sbatch --export=ALL,PROB=darcy_flow run_one.slurm

Useful overrides:

    export IP_DENSITY_N_REF_SIGNAL=5000
    export IP_DENSITY_N_REF_GATE=5000
    export IP_DENSITY_N_REF_EVAL=5000
    export IP_DENSITY_BANK_COUPLING=independent      # shared | prefix | independent
    export IP_DENSITY_EVAL_SOURCE=MALA-EVAL          # use a disjoint density-eval bank
    export IP_DENSITY_EVAL_BANK_COUPLING=independent
    export IP_DENSITY_MALA_INIT=map_laplace
    export IP_DENSITY_MALA_STEPS=600
    export IP_DENSITY_MALA_BURNIN=150
    export IP_DENSITY_MALA_DT=4e-5
    export IP_DENSITY_MAP_LAPLACE_STARTS=128
    export IP_DENSITY_MAP_LAPLACE_MAX_ITER=25
    export IP_DENSITY_BASELINES=map_laplace
    export IP_DENSITY_DRC_PF_STEPS=32
    export IP_DENSITY_DRC_PLOT_LAYOUT=comparison_grid
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
# DRC density benchmark configuration
# ==========================================
# This benchmark matches the updated Navier-Stokes/Helmholtz wiring:
#   1. build a MALA source bank for score signals and LFGI gate estimation,
#   2. optionally build an independent MALA-EVAL bank for held-out density eval,
#   3. run ratio-only DRC-R density nodes for Tweedie, Scalar Blend, Matrix Blend, and LFGI.
# Each density node computes probability-flow log q and compares -log q against
# the true unnormalized posterior energy -log pi at the configured eval points.
def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_float_or_none(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    if raw in {'none', 'null', 'nan', ''}:
        return None
    return float(raw)


def _env_bool(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {'0', 'false', 'no', 'off', 'none', ''}


def _env_is_set(name):
    return name in os.environ and str(os.environ.get(name, '')).strip() != ''


def _canonical_bank_coupling(value):
    raw = str(value).strip().lower().replace('-', '_')
    aliases = {
        'same': 'shared',
        'same_bank': 'shared',
        'shared_bank': 'shared',
        'share': 'shared',
        'reuse': 'shared',
        'prefix_slice': 'prefix',
        'disjoint_prefix': 'prefix',
        'split': 'independent',
        'split_bank': 'independent',
        'indep': 'independent',
        'separate': 'independent',
        'separate_bank': 'independent',
    }
    raw = aliases.get(raw, raw)
    if raw not in {'shared', 'prefix', 'independent'}:
        raise ValueError(
            f"Unknown bank coupling {value!r}; expected shared, prefix, or independent."
        )
    return raw


def _required_source_bank_size(n_signal, n_gate, score_gate_bank_coupling):
    coupling = _canonical_bank_coupling(score_gate_bank_coupling)
    if coupling == 'shared':
        return int(n_signal)
    if coupling == 'prefix':
        return int(max(n_signal, n_gate))
    return int(n_signal) + int(n_gate)


def _canonical_source_label(value):
    raw = str(value).strip()
    low = raw.lower().replace('_', '-').replace(' ', '-')
    if low in {'none', 'null', ''}:
        return 'None'
    if low == 'mala':
        return 'MALA'
    if low in {'mala-eval', 'malaeval'}:
        return 'MALA-EVAL'
    return raw


def _env_percentile_pair(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    parts = [float(x.strip()) for x in str(raw).replace(';', ',').split(',') if x.strip()]
    if len(parts) < 2:
        return default
    lo, hi = parts[:2]
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError(f'{name} must be two percentiles 0 <= lo < hi <= 100; got {raw!r}')
    return (lo, hi)


def _env_csv(name, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return tuple(default)
    parts = [x.strip() for x in str(raw).replace(';', ',').split(',') if x.strip()]
    return tuple(parts) if parts else tuple(default)


def _env_int_tuple(name, default):
    return tuple(int(float(x)) for x in _env_csv(name, default))


def _env_float_tuple_or_none(name, default=None):
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    parts = [x.strip() for x in str(raw).replace(';', ',').split(',') if x.strip()]
    if not parts or any(x.lower() in {'none', 'null'} for x in parts):
        return default
    return tuple(float(x) for x in parts)


# Darcy keeps its legacy IP_DENSITY_N_REF fallback, but now uses the same split
# bank controls as Navier-Stokes and Helmholtz. The score-signal bank feeds the
# frozen score field; the gate bank feeds LFGI gate estimation; the eval bank
# is held out for PF/DRC density-energy diagnostics.
N_REF_SIGNAL = _env_int('IP_DENSITY_N_REF_SIGNAL', _env_int('IP_DENSITY_N_REF', N_REF))
N_REF_GATE_ENV_SET = _env_is_set('IP_DENSITY_N_REF_GATE')
N_REF_GATE = _env_int('IP_DENSITY_N_REF_GATE', N_REF_SIGNAL)
N_REF_EVAL = _env_int('IP_DENSITY_N_REF_EVAL', N_REF_SIGNAL)
N_REF = N_REF_SIGNAL  # Backward-compatible alias used by reporting helpers.
DEFAULT_N_GEN = _env_int('IP_DENSITY_DEFAULT_N_GEN', N_REF_SIGNAL)

# Density-evaluation correction-factor benchmark. By default, the frozen score
# estimator is fitted from MALA while density values are evaluated on MALA-EVAL,
# so score-reference/gate particles and density-evaluation locations do not
# overlap unless the source/coupling flags explicitly request sharing.
DENSITY_REF_SOURCE = _canonical_source_label(os.environ.get('IP_DENSITY_REF_SOURCE', 'MALA'))
DENSITY_BANK_COUPLING = _canonical_bank_coupling(os.environ.get(
    'IP_DENSITY_BANK_COUPLING',
    os.environ.get('IP_DENSITY_GATE_BANK_COUPLING', 'independent'), #'independent'),
))
DENSITY_EVAL_SOURCE = _canonical_source_label(os.environ.get('IP_DENSITY_EVAL_SOURCE', 'MALA-EVAL'))
DENSITY_EVAL_BANK_COUPLING = _canonical_bank_coupling(
    os.environ.get('IP_DENSITY_EVAL_BANK_COUPLING', 'independent')
)
DENSITY_DRC_PF_STEPS = _env_int('IP_DENSITY_DRC_PF_STEPS', 64)
DENSITY_DRC_EVAL_BATCH_SIZE = _env_int('IP_DENSITY_DRC_EVAL_BATCH_SIZE', 32)
DENSITY_DRC_TMIN = _env_float('IP_DENSITY_DRC_TMIN', 10 ** (-2.5))
DENSITY_DRC_TMAX = _env_float('IP_DENSITY_DRC_TMAX', 5.0)
DENSITY_DRC_CLIP = _env_float_or_none('IP_DENSITY_DRC_CLIP', None)
DENSITY_DRC_TEMPERATURE = _env_float('IP_DENSITY_DRC_TEMPERATURE', 1.0)
DENSITY_DRC_ENERGY_PLOTS = _env_bool('IP_DENSITY_DRC_ENERGY_PLOTS', True)
DENSITY_DRC_PLOT_AXIS_MODE = os.environ.get('IP_DENSITY_DRC_PLOT_AXIS_MODE', 'robust')
DENSITY_DRC_RESIDUAL_AXIS_MODE = os.environ.get('IP_DENSITY_DRC_RESIDUAL_AXIS_MODE', 'robust')
DENSITY_DRC_RESIDUAL_KIND = os.environ.get('IP_DENSITY_DRC_RESIDUAL_KIND', 'affine_normalized')
DENSITY_DRC_AFFINE_FIT_SCOPE = os.environ.get('IP_DENSITY_DRC_AFFINE_FIT_SCOPE', 'central')
DENSITY_DRC_ROBUST_PERCENTILES = _env_percentile_pair('IP_DENSITY_DRC_ROBUST_PERCENTILES', (4.0, 96.0))
DENSITY_DRC_SAVE_RAW_PLOTS = _env_bool('IP_DENSITY_DRC_SAVE_RAW_PLOTS', False)
DENSITY_DRC_SAVE_LOGLOG_PLOTS = _env_bool('IP_DENSITY_DRC_SAVE_LOGLOG_PLOTS', False)
DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS = _env_bool('IP_DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS', False)
DENSITY_DRC_SAVE_LEGACY_ALIAS = _env_bool('IP_DENSITY_DRC_SAVE_LEGACY_ALIAS', True)
DENSITY_DRC_PLOT_LAYOUT = os.environ.get('IP_DENSITY_DRC_PLOT_LAYOUT', 'comparison_grid')
DENSITY_DRC_GRID_MAX_POINTS = _env_int('IP_DENSITY_DRC_GRID_MAX_POINTS', 5000)
DENSITY_DRC_GRID_SAVE_PDF = _env_bool('IP_DENSITY_DRC_GRID_SAVE_PDF', True)

# Divergence defaults matching the density scripts. LFGI and Tweedie dispatch
# to analytic implementations in sampling.py when available; scalar blend keeps
# Hutchinson by default for speed unless explicitly overridden.
DENSITY_TWEEDIE_DIVERGENCE = os.environ.get('IP_DENSITY_TWEEDIE_DIVERGENCE', 'auto')
DENSITY_BLEND_DIVERGENCE = os.environ.get('IP_DENSITY_BLEND_DIVERGENCE', 'auto')
DENSITY_MATRIX_BLEND_DIVERGENCE = os.environ.get('IP_DENSITY_MATRIX_BLEND_DIVERGENCE', DENSITY_BLEND_DIVERGENCE)
DENSITY_LFGI_DIVERGENCE = os.environ.get('IP_DENSITY_LFGI_DIVERGENCE', 'auto')
DENSITY_DIV_PROBES = _env_int('IP_DENSITY_DRC_DIV_PROBES', 1)

# Manuscript-table extras. These are intentionally thin wrappers around helper
# functions in sampling.py so Darcy produces the same paper metrics as the
# Navier-Stokes and Helmholtz density benchmarks: PF density rows, a MAP-Laplace
# Gaussian baseline row, optional known-Z columns, and an optional PF sensitivity
# plot/table.
DENSITY_BASELINES = _env_csv('IP_DENSITY_BASELINES', ('map_laplace',))
DENSITY_KNOWN_LOGZ = _env_float_or_none('IP_DENSITY_KNOWN_LOGZ', None)
DENSITY_RUN_PF_SENSITIVITY = _env_bool('IP_DENSITY_RUN_PF_SENSITIVITY', False)
DENSITY_PF_SENSITIVITY_LABELS = _env_csv('IP_DENSITY_PF_SENSITIVITY_LABELS', ('DENS-LFGI', 'DENS-Tweedie'))
DENSITY_PF_SENSITIVITY_STEPS = _env_int_tuple('IP_DENSITY_PF_SENSITIVITY_STEPS', (32, 64, 128))
DENSITY_PF_SENSITIVITY_TMINS = _env_float_tuple_or_none('IP_DENSITY_PF_SENSITIVITY_TMINS', None)

# Legacy ALT-DRC bootstrap settings are retained for manual experiments, but the
# active paper-facing density benchmark below uses MALA / MALA-EVAL by default.
BOOT_N_REF = _env_int('IP_DENSITY_BOOT_N_REF', N_REF_SIGNAL)
BOOT_INIT_STEPS = _env_int('IP_DENSITY_BOOT_INIT_STEPS', 200)
BOOT_DRC_PF_STEPS = _env_int('IP_DENSITY_BOOT_DRC_PF_STEPS', 32)
BOOT_DRC_DIVERGENCE = os.environ.get('IP_DENSITY_BOOT_DRC_DIVERGENCE', 'analytic')
BOOT_DRC_DIV_PROBES_DEFAULT = 0 if BOOT_DRC_DIVERGENCE.strip().lower() in {'analytic', 'closed_form', 'closedform', 'exact', 'auto'} else 1
BOOT_DRC_DIV_PROBES = _env_int('IP_DENSITY_BOOT_DRC_DIV_PROBES', BOOT_DRC_DIV_PROBES_DEFAULT)
BOOT_DRC_EVAL_BATCH_SIZE = _env_int('IP_DENSITY_BOOT_DRC_EVAL_BATCH_SIZE', 32)
BOOT_DRC_CLIP = _env_float_or_none('IP_DENSITY_BOOT_DRC_CLIP', 20.0)
BOOT_DRC_TEMPERATURE = _env_float('IP_DENSITY_BOOT_DRC_TEMPERATURE', 1.0)

# MALA source-bank configuration. DENSITY_SOURCE_REQUIRED_N is the minimum number
# of source particles needed to provide disjoint score/gate slices from
# DENSITY_REF_SOURCE under the selected score/gate coupling.
DENSITY_SOURCE_REQUIRED_N = _required_source_bank_size(
    N_REF_SIGNAL, N_REF_GATE, DENSITY_BANK_COUPLING,
)
MALA_N_SAMPLES = _env_int('IP_DENSITY_MALA_N_SAMPLES', DENSITY_SOURCE_REQUIRED_N)
MALA_STEPS = _env_int('IP_DENSITY_MALA_STEPS', 700)
MALA_BURNIN = _env_int('IP_DENSITY_MALA_BURNIN', 200)
MALA_DT = _env_float('IP_DENSITY_MALA_DT', 4.0e-5)
# Default to target-side MAP/Laplace proxy initialization for MALA.
MALA_INIT = os.environ.get('IP_DENSITY_MALA_INIT', 'prior')
MALA_PRECOND = _env_bool('IP_DENSITY_MALA_PRECOND', False)
MALA_EVAL_N_SAMPLES = _env_int('IP_DENSITY_MALA_EVAL_N_SAMPLES', N_REF_EVAL)
MALA_EVAL_STEPS = _env_int('IP_DENSITY_MALA_EVAL_STEPS', MALA_STEPS)
MALA_EVAL_BURNIN = _env_int('IP_DENSITY_MALA_EVAL_BURNIN', MALA_BURNIN)
MALA_EVAL_DT = _env_float('IP_DENSITY_MALA_EVAL_DT', MALA_DT)
MALA_EVAL_INIT = os.environ.get('IP_DENSITY_MALA_EVAL_INIT', MALA_INIT)
MALA_EVAL_PRECOND = _env_bool('IP_DENSITY_MALA_EVAL_PRECOND', MALA_PRECOND)
MAP_LAPLACE_STARTS = _env_int('IP_DENSITY_MAP_LAPLACE_STARTS', 128)
MAP_LAPLACE_MAX_ITER = _env_int('IP_DENSITY_MAP_LAPLACE_MAX_ITER', 25)
MAP_LAPLACE_TOL = _env_float('IP_DENSITY_MAP_LAPLACE_TOL', 1e-5)
MAP_LAPLACE_RIDGE = _env_float('IP_DENSITY_MAP_LAPLACE_RIDGE', 1e-6)
MAP_LAPLACE_MAX_STEP_NORM = _env_float('IP_DENSITY_MAP_LAPLACE_MAX_STEP_NORM', 2.0)
MAP_LAPLACE_BACKTRACK_STEPS = _env_int('IP_DENSITY_MAP_LAPLACE_BACKTRACK_STEPS', 8)
HELDOUT_BATCH_SIZE = _env_int('IP_DENSITY_HELDOUT_BATCH_SIZE', 8)

if MALA_N_SAMPLES < DENSITY_SOURCE_REQUIRED_N:
    raise ValueError(
        f'MALA_N_SAMPLES={MALA_N_SAMPLES} must be >= {DENSITY_SOURCE_REQUIRED_N} '
        f'for N_REF_SIGNAL={N_REF_SIGNAL}, N_REF_GATE={N_REF_GATE}, '
        f'and DENSITY_BANK_COUPLING={DENSITY_BANK_COUPLING!r}.'
    )
if DENSITY_EVAL_SOURCE == 'MALA-EVAL' and MALA_EVAL_N_SAMPLES < N_REF_EVAL:
    raise ValueError(
        f'MALA_EVAL_N_SAMPLES={MALA_EVAL_N_SAMPLES} must be >= N_REF_EVAL={N_REF_EVAL}.'
    )

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
run_ctx = init_run_results('darcy_density_eval_bench')
DASHBOARD_PDF_PATH = os.path.join(
    run_ctx['run_results_dir'],
    f"{run_ctx['run_results_stem']}_summary_dashboard.pdf",
)

RUN_COMMAND_HINT = (
    'IP_DENSITY_N_REF_SIGNAL={n_signal} IP_DENSITY_N_REF_GATE={n_gate} '
    'IP_DENSITY_N_REF_EVAL={n_eval} IP_DENSITY_BANK_COUPLING={bank_coupling} '
    'IP_DENSITY_EVAL_SOURCE={eval_source} IP_DENSITY_MALA_N_SAMPLES={mala_n} '
    'IP_DENSITY_MALA_EVAL_N_SAMPLES={mala_eval_n} IP_DENSITY_MALA_STEPS={mala_steps} '
    'IP_DENSITY_MALA_BURNIN={burnin} IP_DENSITY_MALA_DT={dt:g} '
    'IP_DENSITY_MALA_INIT={mala_init} IP_DENSITY_MAP_LAPLACE_STARTS={map_starts} '
    'IP_DENSITY_MAP_LAPLACE_MAX_ITER={map_iter} IP_DENSITY_DRC_PF_STEPS={pf_steps} '
    'IP_DENSITY_DRC_PLOT_LAYOUT={layout} IP_DENSITY_BASELINES={baselines} '
    'IP_DENSITY_RUN_PF_SENSITIVITY={pf_sens} python darcy_flow.py'
).format(
    n_signal=N_REF_SIGNAL,
    n_gate=N_REF_GATE,
    n_eval=N_REF_EVAL,
    bank_coupling=DENSITY_BANK_COUPLING,
    eval_source=DENSITY_EVAL_SOURCE,
    mala_n=MALA_N_SAMPLES,
    mala_eval_n=MALA_EVAL_N_SAMPLES,
    mala_steps=MALA_STEPS,
    burnin=MALA_BURNIN,
    dt=MALA_DT,
    mala_init=MALA_INIT,
    map_starts=MAP_LAPLACE_STARTS,
    map_iter=MAP_LAPLACE_MAX_ITER,
    pf_steps=DENSITY_DRC_PF_STEPS,
    layout=DENSITY_DRC_PLOT_LAYOUT,
    baselines=','.join(DENSITY_BASELINES),
    pf_sens=int(bool(DENSITY_RUN_PF_SENSITIVITY)),
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
    title='Darcy flow density-evaluation dashboard',
)
dashboard.add_text_page(
    'Darcy flow density-evaluation dashboard',
    [
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        'This dashboard matches the updated Navier-Stokes/Helmholtz density-evaluation benchmark, with configurable score-signal, gate, and held-out density-evaluation banks. By default, density values are evaluated on MALA-EVAL rather than on the score-reference bank.',
        'Primary density diagnostic: true posterior energy -log pi(x) versus probability-flow estimated energy -log q(x), plus affine-calibrated residuals in one publication-style comparison grid.',
        'The comparison-grid plot uses LFGI robust axes to align all rows.',
        'Tables are intentionally limited to two pages: metrics plus a readable split run-info page.',
        'Random progress output from precomputation / Hessian batching is intentionally excluded.',
        f"run_results_dir = {run_ctx['run_results_dir']}",
        '',
        f'seed = {seed}',
        f'ACTIVE_DIM = {ACTIVE_DIM}',
        f'run command = {RUN_COMMAND_HINT}',
        '',
        f'N_REF_SIGNAL = {N_REF_SIGNAL}',
        f'N_REF_GATE = {N_REF_GATE}',
        f'N_REF_EVAL = {N_REF_EVAL}',
        f'DENSITY_REF_SOURCE = {DENSITY_REF_SOURCE}',
        f'DENSITY_BANK_COUPLING = {DENSITY_BANK_COUPLING}',
        f'DENSITY_EVAL_SOURCE = {DENSITY_EVAL_SOURCE}',
        f'DENSITY_EVAL_BANK_COUPLING = {DENSITY_EVAL_BANK_COUPLING}',
        f'MALA_N_SAMPLES = {MALA_N_SAMPLES}',
        f'MALA_EVAL_N_SAMPLES = {MALA_EVAL_N_SAMPLES}',
        f'MALA_STEPS = {MALA_STEPS}',
        f'MALA_BURNIN = {MALA_BURNIN}',
        f'MALA_DT = {MALA_DT}',
        f'MALA_INIT = {MALA_INIT}',
        f'MALA_PRECOND = {MALA_PRECOND}',
        f'MALA_EVAL_INIT = {MALA_EVAL_INIT}',
        f'MALA_EVAL_PRECOND = {MALA_EVAL_PRECOND}',
        f'MAP_LAPLACE_STARTS = {MAP_LAPLACE_STARTS}',
        f'MAP_LAPLACE_MAX_ITER = {MAP_LAPLACE_MAX_ITER}',
        f'MAP_LAPLACE_TOL = {MAP_LAPLACE_TOL}',
        f'MAP_LAPLACE_RIDGE = {MAP_LAPLACE_RIDGE}',
        f'DENSITY_DRC_PF_STEPS = {DENSITY_DRC_PF_STEPS}',
        f'DENSITY_TWEEDIE_DIVERGENCE = {DENSITY_TWEEDIE_DIVERGENCE}',
        f'DENSITY_BLEND_DIVERGENCE = {DENSITY_BLEND_DIVERGENCE}',
        f'DENSITY_MATRIX_BLEND_DIVERGENCE = {DENSITY_MATRIX_BLEND_DIVERGENCE}',
        f'DENSITY_LFGI_DIVERGENCE = {DENSITY_LFGI_DIVERGENCE}',
        f'DENSITY_BASELINES = {DENSITY_BASELINES}',
        f'DENSITY_KNOWN_LOGZ = {DENSITY_KNOWN_LOGZ}',
        f'DENSITY_RUN_PF_SENSITIVITY = {DENSITY_RUN_PF_SENSITIVITY}',
        f'DENSITY_PF_SENSITIVITY_STEPS = {DENSITY_PF_SENSITIVITY_STEPS}',
        f'DENSITY_PF_SENSITIVITY_TMINS = {DENSITY_PF_SENSITIVITY_TMINS}',
        f'DENSITY_DRC_EVAL_BATCH_SIZE = {DENSITY_DRC_EVAL_BATCH_SIZE}',
        f'DENSITY_DRC_TMIN = {DENSITY_DRC_TMIN}',
        f'DENSITY_DRC_TMAX = {DENSITY_DRC_TMAX}',
        f'DENSITY_DRC_CLIP = {DENSITY_DRC_CLIP}',
        f'DENSITY_DRC_PLOT_AXIS_MODE = {DENSITY_DRC_PLOT_AXIS_MODE}',
        f'DENSITY_DRC_RESIDUAL_AXIS_MODE = {DENSITY_DRC_RESIDUAL_AXIS_MODE}',
        f'DENSITY_DRC_RESIDUAL_KIND = {DENSITY_DRC_RESIDUAL_KIND}',
        f'DENSITY_DRC_AFFINE_FIT_SCOPE = {DENSITY_DRC_AFFINE_FIT_SCOPE}',
        f'DENSITY_DRC_ROBUST_PERCENTILES = {DENSITY_DRC_ROBUST_PERCENTILES}',
        f'DENSITY_DRC_SAVE_RAW_PLOTS = {DENSITY_DRC_SAVE_RAW_PLOTS}',
        f'DENSITY_DRC_SAVE_LOGLOG_PLOTS = {DENSITY_DRC_SAVE_LOGLOG_PLOTS}',
        f'DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS = {DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS}',
        f'DENSITY_DRC_PLOT_LAYOUT = {DENSITY_DRC_PLOT_LAYOUT}',
        f'DENSITY_DRC_GRID_MAX_POINTS = {DENSITY_DRC_GRID_MAX_POINTS}',
        f'DENSITY_DRC_GRID_SAVE_PDF = {DENSITY_DRC_GRID_SAVE_PDF}',
        f'DEFAULT_N_GEN = {DEFAULT_N_GEN}',
        f'NOISE_STD = {NOISE_STD}',
        f'N = {N}, num_observation = {num_observation}, num_holdout_observation = {num_holdout_observation}',
        f'HESS_MIN = {HESS_MIN}, HESS_MAX = {HESS_MAX}',
        f'BUILD_GNL_BANKS = {BUILD_GNL_BANKS}',
        f'GNL_PILOT_N = {GNL_PILOT_N}',
        f'GNL_STIFF_LAMBDA_CUT = {GNL_STIFF_LAMBDA_CUT}',
        f'PLOT_NORMALIZER = {PLOT_NORMALIZER}',
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


def _boot_drc_r_config(ref_source, label):
    return {
        'display_name': label,
        'ref_source': ref_source,
        'init': 'DRC-R',
        'init_weights': 'None',
        'drc_score_init': 'ce_hlsi',
        'drc_score_init_weights': 'None',
        'transition_w': 'ou',
        'n_ref': BOOT_N_REF,
        'include_results': False,
        'drc_pf_steps': BOOT_DRC_PF_STEPS,
        'drc_divergence': BOOT_DRC_DIVERGENCE,
        'drc_div_probes': BOOT_DRC_DIV_PROBES,
        'drc_eval_batch_size': BOOT_DRC_EVAL_BATCH_SIZE,
        'drc_clip': BOOT_DRC_CLIP,
        'drc_temperature': BOOT_DRC_TEMPERATURE,
        'drc_fd_eps': 1e-3,
    }


def _density_eval_config(ref_source, score_init, divergence, label, display_name):
    return {
        'display_name': display_name,
        'ref_source': ref_source,
        'init': 'DRC-R',
        'init_weights': 'None',
        'drc_score_init': score_init,
        # The MALA source bank is the empirical posterior/reference bank. Do not
        # likelihood-reweight again when fitting each frozen score family.
        'drc_score_init_weights': 'None',
        'transition_w': 'ou',
        'n_ref': N_REF_SIGNAL,
        'n_ref_signal': N_REF_SIGNAL,
        'n_ref_gate': N_REF_GATE,
        'score_gate_bank_coupling': DENSITY_BANK_COUPLING,
        'drc_eval_source': DENSITY_EVAL_SOURCE,
        'drc_eval_n_ref': N_REF_EVAL,
        'drc_eval_bank_coupling': DENSITY_EVAL_BANK_COUPLING,
        'include_results': False,
        'drc_pf_steps': DENSITY_DRC_PF_STEPS,
        'drc_divergence': divergence,
        'drc_div_probes': DENSITY_DIV_PROBES,
        'drc_eval_batch_size': DENSITY_DRC_EVAL_BATCH_SIZE,
        'drc_clip': DENSITY_DRC_CLIP,
        'drc_temperature': DENSITY_DRC_TEMPERATURE,
        'drc_fd_eps': 1e-3,
        'drc_tmin': DENSITY_DRC_TMIN,
        'drc_tmax': DENSITY_DRC_TMAX,
        'drc_store_details': True,
        'drc_energy_benchmark': True,
        'drc_energy_plots': DENSITY_DRC_ENERGY_PLOTS,
        'drc_energy_plot_axis_mode': DENSITY_DRC_PLOT_AXIS_MODE,
        'drc_energy_residual_axis_mode': DENSITY_DRC_RESIDUAL_AXIS_MODE,
        'drc_energy_residual_kind': DENSITY_DRC_RESIDUAL_KIND,
        'drc_energy_affine_fit_scope': DENSITY_DRC_AFFINE_FIT_SCOPE,
        'drc_energy_robust_percentiles': DENSITY_DRC_ROBUST_PERCENTILES,
        'drc_energy_save_raw_plots': DENSITY_DRC_SAVE_RAW_PLOTS,
        'drc_energy_save_loglog_plots': DENSITY_DRC_SAVE_LOGLOG_PLOTS,
        'drc_energy_save_logratio_residual_plots': DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS,
        'drc_energy_save_legacy_alias': DENSITY_DRC_SAVE_LEGACY_ALIAS,
        'drc_energy_plot_layout': DENSITY_DRC_PLOT_LAYOUT,
        'drc_energy_grid_method_order': ('DENS-LFGI', 'DENS-MatrixBlend', 'DENS-ScalarBlend', 'DENS-Tweedie', 'DENS-MAP-Laplace'),
        'drc_energy_grid_axis_reference': 'DENS-LFGI',
        'drc_energy_grid_max_points': DENSITY_DRC_GRID_MAX_POINTS,
        'drc_energy_grid_save_pdf': DENSITY_DRC_GRID_SAVE_PDF,
    }


SAMPLER_CONFIGS = OrderedDict([
    ('MALA', {
        'display_name': 'MALA reference',
        'init': MALA_INIT,
        'init_weights': 'None',
        'transition_w': 'ou',
        'n_ref': DENSITY_SOURCE_REQUIRED_N,
        'n_samples': MALA_N_SAMPLES,
        'init_steps': 0,
        'mala_steps': MALA_STEPS,
        'mala_burnin': MALA_BURNIN,
        'mala_dt': MALA_DT,
        'precond_mala': MALA_PRECOND,
        'map_laplace_starts': MAP_LAPLACE_STARTS,
        'map_laplace_max_iter': MAP_LAPLACE_MAX_ITER,
        'map_laplace_tol': MAP_LAPLACE_TOL,
        'map_laplace_ridge': MAP_LAPLACE_RIDGE,
        'map_laplace_max_step_norm': MAP_LAPLACE_MAX_STEP_NORM,
        'map_laplace_backtrack_steps': MAP_LAPLACE_BACKTRACK_STEPS,
        'log_mean_ess': False,
        'include_results': True,
        'is_reference': True,
    }),
])

if DENSITY_EVAL_SOURCE == 'MALA-EVAL':
    SAMPLER_CONFIGS['MALA-EVAL'] = {
        'display_name': 'MALA density-eval bank',
        'init': MALA_EVAL_INIT,
        'init_weights': 'None',
        'transition_w': 'ou',
        'n_ref': N_REF_EVAL,
        'n_samples': MALA_EVAL_N_SAMPLES,
        'init_steps': 0,
        'mala_steps': MALA_EVAL_STEPS,
        'mala_burnin': MALA_EVAL_BURNIN,
        'mala_dt': MALA_EVAL_DT,
        'precond_mala': MALA_EVAL_PRECOND,
        'map_laplace_starts': MAP_LAPLACE_STARTS,
        'map_laplace_max_iter': MAP_LAPLACE_MAX_ITER,
        'map_laplace_tol': MAP_LAPLACE_TOL,
        'map_laplace_ridge': MAP_LAPLACE_RIDGE,
        'map_laplace_max_step_norm': MAP_LAPLACE_MAX_STEP_NORM,
        'map_laplace_backtrack_steps': MAP_LAPLACE_BACKTRACK_STEPS,
        'log_mean_ess': False,
        'include_results': False,
        'is_reference': False,
    }

SAMPLER_CONFIGS.update(OrderedDict([
    ('DENS-Tweedie', _density_eval_config(
        DENSITY_REF_SOURCE, 'tweedie', DENSITY_TWEEDIE_DIVERGENCE,
        'DENS-Tweedie', 'Tweedie',
    )),
    ('DENS-ScalarBlend', _density_eval_config(
        DENSITY_REF_SOURCE, 'scalar_blend', DENSITY_BLEND_DIVERGENCE,
        'DENS-ScalarBlend', 'Scalar Blend',
    )),
    ('DENS-MatrixBlend', _density_eval_config(
        DENSITY_REF_SOURCE, 'matrix_blend', DENSITY_MATRIX_BLEND_DIVERGENCE,
        'DENS-MatrixBlend', 'MATRIX BLEND',
    )),
    ('DENS-LFGI', _density_eval_config(
        DENSITY_REF_SOURCE, 'ce_hlsi', DENSITY_LFGI_DIVERGENCE,
        'DENS-LFGI', 'LFGI',
    )),
]))

RUN_COMMAND_HINT = (
    'IP_DENSITY_N_REF_SIGNAL={n_signal} IP_DENSITY_N_REF_GATE={n_gate} '
    'IP_DENSITY_N_REF_EVAL={n_eval} IP_DENSITY_BANK_COUPLING={bank_coupling} '
    'IP_DENSITY_EVAL_SOURCE={eval_source} IP_DENSITY_MALA_N_SAMPLES={mala_n} '
    'IP_DENSITY_MALA_EVAL_N_SAMPLES={mala_eval_n} IP_DENSITY_MALA_STEPS={mala_steps} '
    'IP_DENSITY_MALA_BURNIN={burnin} IP_DENSITY_MALA_DT={dt:g} '
    'IP_DENSITY_MALA_INIT={mala_init} IP_DENSITY_MAP_LAPLACE_STARTS={map_starts} '
    'IP_DENSITY_MAP_LAPLACE_MAX_ITER={map_iter} IP_DENSITY_DRC_PF_STEPS={pf_steps} '
    'IP_DENSITY_DRC_PLOT_LAYOUT={layout} IP_DENSITY_BASELINES={baselines} '
    'IP_DENSITY_RUN_PF_SENSITIVITY={pf_sens} python darcy_flow.py'
).format(
    n_signal=N_REF_SIGNAL,
    n_gate=N_REF_GATE,
    n_eval=N_REF_EVAL,
    bank_coupling=DENSITY_BANK_COUPLING,
    eval_source=DENSITY_EVAL_SOURCE,
    mala_n=MALA_N_SAMPLES,
    mala_eval_n=MALA_EVAL_N_SAMPLES,
    mala_steps=MALA_STEPS,
    burnin=MALA_BURNIN,
    dt=MALA_DT,
    mala_init=MALA_INIT,
    map_starts=MAP_LAPLACE_STARTS,
    map_iter=MAP_LAPLACE_MAX_ITER,
    pf_steps=DENSITY_DRC_PF_STEPS,
    layout=DENSITY_DRC_PLOT_LAYOUT,
    baselines=','.join(DENSITY_BASELINES),
    pf_sens=int(bool(DENSITY_RUN_PF_SENSITIVITY)),
)






pipeline = run_standard_sampler_pipeline(
    prior_model,
    lik_model,
    SAMPLER_CONFIGS,
    n_ref=N_REF_SIGNAL,
    build_gnl_banks=BUILD_GNL_BANKS,
    compute_pou=True,
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

# Add closed-form density baselines on the same held-out density-eval bank.
# Keep this deliberately small: for the manuscript comparison we need the
# MAP-Laplace row to test whether Darcy nonlinearity separates LFGI
# from a local Gaussian approximation.
def _first_density_eval_bank(precomp_dict):
    for lab in ('DENS-LFGI', 'DENS-MatrixBlend', 'DENS-Tweedie', 'DENS-ScalarBlend'):
        bank = precomp_dict.get('eval_banks', {}).get(lab)
        if bank is not None:
            return bank
    for lab in ('DENS-LFGI', 'DENS-MatrixBlend', 'DENS-Tweedie', 'DENS-ScalarBlend'):
        det = precomp_dict.get('drc_details', {}).get(lab)
        if det is not None:
            return {
                'X_ref': det['X_ref'],
                'log_lik_ref': det['log_lik'],
                'bank_name': det.get('eval_bank_name', f'{lab}_details_eval'),
                'eval_only': True,
            }
    return None

baseline_eval_bank = _first_density_eval_bank(precomp)
if baseline_eval_bank is not None and any(str(b).lower().replace('-', '_') in {'map_laplace', 'laplace_map', 'map'} for b in DENSITY_BASELINES):
    try:
        map_df, map_details, map_component = compute_map_laplace_density_baseline(
            baseline_eval_bank,
            prior_model,
            lik_model,
            label='DENS-MAP-Laplace',
            n_starts=MAP_LAPLACE_STARTS,
            max_iter=MAP_LAPLACE_MAX_ITER,
            tol=MAP_LAPLACE_TOL,
            ridge=MAP_LAPLACE_RIDGE,
            max_step_norm=MAP_LAPLACE_MAX_STEP_NORM,
            backtrack_steps=MAP_LAPLACE_BACKTRACK_STEPS,
            batch_size=max(1, DENSITY_DRC_EVAL_BATCH_SIZE),
            save_dir=run_ctx['run_results_dir'],
            run_stem=run_ctx['run_results_stem'],
            make_plots=False,
            precomp=precomp,
            known_logZ=DENSITY_KNOWN_LOGZ,
            plot_axis_mode=DENSITY_DRC_PLOT_AXIS_MODE,
            residual_axis_mode=DENSITY_DRC_RESIDUAL_AXIS_MODE,
            robust_percentiles=DENSITY_DRC_ROBUST_PERCENTILES,
            residual_kind=DENSITY_DRC_RESIDUAL_KIND,
            affine_fit_scope=DENSITY_DRC_AFFINE_FIT_SCOPE,
            verbose=True,
        )
        display_names['DENS-MAP-Laplace'] = 'MAP-Laplace'
        print('\n=== Added MAP-Laplace density baseline ===')
        print(map_df.to_string(index=False))
    except Exception as exc:
        print(f"WARNING: MAP-Laplace density baseline failed and will be skipped: {exc}")
elif baseline_eval_bank is None:
    print('WARNING: no density eval bank found; MAP-Laplace density baseline skipped.')



def _darcy_density_label_alias(label):
    """Normalize legacy Darcy density labels before manuscript plots are written."""
    label_s = str(label)
    if label_s == 'DENS-CE-HLSI':
        return 'DENS-LFGI'
    if label_s in {'DENS-CenteredBlend', 'DENS-CenteredMatrixBlend', 'DENS-Matrix-Blend'}:
        return 'DENS-MatrixBlend'
    return label_s.replace('CE-HLSI', 'LFGI')


def _darcy_alias_drc_details(details):
    aliased = OrderedDict()
    for lab, val in (details or {}).items():
        new_lab = _darcy_density_label_alias(lab)
        if new_lab not in aliased:
            aliased[new_lab] = val
    return aliased

# The standard DRC comparison grid is created inside run_standard_sampler_pipeline,
# before the closed-form MAP-Laplace baseline row exists. Regenerate the grid here
# after inserting DENS-MAP-Laplace so the figure matches the metric table.
density_grid_method_order = ('DENS-LFGI', 'DENS-MatrixBlend', 'DENS-ScalarBlend', 'DENS-Tweedie', 'DENS-MAP-Laplace')
if 'DENS-MAP-Laplace' in _darcy_alias_drc_details(precomp.get('drc_details', {})):
    try:
        all_drc_details = _darcy_alias_drc_details(precomp.get('drc_details', {}))
        grid_labels = [lab for lab in density_grid_method_order if lab in all_drc_details]
        grid_labels += [lab for lab in all_drc_details.keys() if lab not in grid_labels and str(lab).startswith('DENS-')]
        details_for_grid = OrderedDict((lab, all_drc_details[lab]) for lab in grid_labels)
        cfg_for_grid = {lab: dict(SAMPLER_CONFIGS.get(lab, {})) for lab in grid_labels}
        cfg_for_grid.setdefault('DENS-MAP-Laplace', {})['display_name'] = 'MAP-Laplace'
        refreshed_grid = sampling.save_drc_energy_comparison_grid(
            details_for_grid,
            cfg_by_label=cfg_for_grid,
            save_dir=run_ctx['run_results_dir'],
            run_stem=run_ctx['run_results_stem'],
            method_order=density_grid_method_order,
            axis_reference_label='DENS-LFGI' if 'DENS-LFGI' in details_for_grid else None,
            plot_axis_mode=DENSITY_DRC_PLOT_AXIS_MODE,
            residual_axis_mode=DENSITY_DRC_RESIDUAL_AXIS_MODE,
            robust_percentiles=DENSITY_DRC_ROBUST_PERCENTILES,
            affine_fit_scope=DENSITY_DRC_AFFINE_FIT_SCOPE,
            residual_kind=DENSITY_DRC_RESIDUAL_KIND,
            max_points=DENSITY_DRC_GRID_MAX_POINTS,
            save_pdf=DENSITY_DRC_GRID_SAVE_PDF,
        )
        precomp['drc_energy_comparison_grid'] = refreshed_grid
        print('Refreshed density-energy comparison grid with MAP-Laplace row.')
    except Exception as exc:
        print(f'WARNING: failed to refresh density-energy comparison grid with MAP-Laplace row: {exc}')

# Optional tiny PF-discretization sensitivity plot for the manuscript sanity
# check. This reuses frozen score specs and held-out eval banks; it does not
# rebuild score/gate banks.
pf_sensitivity_df = pd.DataFrame()
pf_sensitivity_fig_path = None
if DENSITY_RUN_PF_SENSITIVITY:
    try:
        pf_sensitivity_df, pf_sensitivity_fig_path = run_drc_pf_sensitivity_benchmark(
            precomp,
            SAMPLER_CONFIGS,
            prior_model,
            lik_model,
            labels=DENSITY_PF_SENSITIVITY_LABELS,
            pf_steps_list=DENSITY_PF_SENSITIVITY_STEPS,
            tmin_list=DENSITY_PF_SENSITIVITY_TMINS,
            save_dir=run_ctx['run_results_dir'],
            run_stem=run_ctx['run_results_stem'],
            batch_size=max(1, DENSITY_DRC_EVAL_BATCH_SIZE),
            robust_percentiles=DENSITY_DRC_ROBUST_PERCENTILES,
            affine_fit_scope=DENSITY_DRC_AFFINE_FIT_SCOPE,
            known_logZ=DENSITY_KNOWN_LOGZ,
            make_plot=True,
        )
        if not pf_sensitivity_df.empty:
            pf_cols = ['method', 'pf_steps', 't_min', 'affine_energy_rmse', 'slope_normalized_energy_rmse', 'raw_logw_ess', 'pointwise_nll']
            pf_cols = [c for c in pf_cols if c in pf_sensitivity_df.columns]
            dashboard.add_dataframe(
                'Density PF discretization sensitivity',
                pf_sensitivity_df[pf_cols],
                max_rows=30,
                max_cols=7,
                include_index=False,
            )
            if pf_sensitivity_fig_path:
                dashboard.add_image_page(pf_sensitivity_fig_path)
    except Exception as exc:
        print(f'WARNING: density PF sensitivity benchmark failed and will be skipped: {exc}')

def _make_darcy_table4_density_table(density_df, display_names=None, method_order=None):
    """Darcy Table 4 view: central posterior-bulk density metrics plus NLL std."""
    if density_df is None or len(density_df) == 0:
        return pd.DataFrame()
    display_names = display_names or {}
    method_order = tuple(method_order or ())
    df = pd.DataFrame(density_df).copy()
    if 'label' in df.columns:
        df['Method'] = df['label'].map(lambda x: display_names.get(str(x), str(x)))
    elif 'Method' not in df.columns:
        df['Method'] = [display_names.get(str(i), str(i)) for i in range(len(df))]
    column_map = OrderedDict([
        ('Method', 'Method'),
        ('central_energy_neglogq_spearman', 'Spearman'),
        ('central_affine_energy_slope', 'Central slope'),
        ('central_affine_energy_r2', 'Central R2'),
        ('central_affine_energy_rmse', 'Central RMSE'),
        ('pointwise_nll', 'Pointwise NLL'),
        ('pointwise_nll_std', 'Pointwise NLL std'),
    ])
    cols = [c for c in column_map if c in df.columns]
    out = df[cols].rename(columns={c: column_map[c] for c in cols})
    if method_order and 'Method' in out.columns:
        order_labels = [display_names.get(str(x), str(x)) for x in method_order]
        order_map = {lab: i for i, lab in enumerate(order_labels)}
        out['_order'] = out['Method'].map(lambda x: order_map.get(str(x), len(order_map) + 100))
        out = out.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
    return out


# Surface the density/energy benchmark produced by ratio-only nodes plus any
# closed-form baseline rows added above.
drc_energy_tables = precomp.get('drc_energy_benchmarks', {})
if drc_energy_tables:
    drc_energy_df = pd.concat(list(drc_energy_tables.values()), ignore_index=True)
    manuscript_density_df = _make_darcy_table4_density_table(
        drc_energy_df,
        display_names=display_names,
        method_order=('DENS-Tweedie', 'DENS-ScalarBlend', 'DENS-MatrixBlend', 'DENS-LFGI', 'DENS-MAP-Laplace'),
    )
    if manuscript_density_df.empty:
        manuscript_density_df = make_density_manuscript_table(
            drc_energy_df,
            display_names=display_names,
            method_order=('DENS-Tweedie', 'DENS-ScalarBlend', 'DENS-MatrixBlend', 'DENS-LFGI', 'DENS-MAP-Laplace'),
            include_known_z=DENSITY_KNOWN_LOGZ is not None,
        )
    print('\n=== Density/energy benchmark on configured density-eval bank ===')
    print(drc_energy_df.to_string(index=False))
    print('\n=== Manuscript density table ===')
    print(manuscript_density_df.to_string(index=False))
    manuscript_density_df_path = os.path.join(
        run_ctx['run_results_dir'],
        f"{run_ctx['run_results_stem']}_manuscript_density_table.csv",
    )
    manuscript_density_df.to_csv(manuscript_density_df_path, index=False)
    print(f'Saved manuscript density table to {manuscript_density_df_path}')
    dashboard.add_dataframe(
        'Manuscript density table',
        manuscript_density_df,
        max_rows=20,
        max_cols=8,
        include_index=False,
    )
    dashboard.add_dataframe(
        'Full density/energy benchmark on configured density-eval bank',
        drc_energy_df,
        max_rows=24,
        max_cols=8,
        include_index=False,
    )
else:
    drc_energy_df = pd.DataFrame()
    manuscript_density_df = pd.DataFrame()
    manuscript_density_df_path = None
    print("\nWARNING: no density/energy benchmark was found in precomp['drc_energy_benchmarks'].")

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
        batched_forward_eval_batch_size=HELDOUT_BATCH_SIZE,
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
print(f"{'Method':<24} | {'LogPerm RelL2(%)':<18} | {'Pearson':<10} | {'RMSE_a':<12} | {'PressureRel':<12} | {'SensorRel':<12}")
print('-' * 113)
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
        f"{display_names.get(label, label):<24} | {logperm_rel_pct:<18.4f} | "
        f"{metrics[label]['Pearson_field']:<10.4f} | {metrics[label]['RMSE_alpha']:<12.4e} | {pressure_rel:<12.4e} | {metrics[label]['FwdRelErr']:<12.4e} | "
        f"{metrics[label].get('HeldoutPredNLL', np.nan):<12.4e} | {metrics[label].get('HeldoutStdResSq', np.nan):<12.4e}"
    )

plot_pca_histograms(
    samples,
    alpha_true_np,
    display_names=display_names,
)

results_df, results_runinfo_df, results_df_path, results_runinfo_df_path = save_results_tables(
    metrics,
    sampler_run_info,
    n_ref=N_REF_SIGNAL,
    target_name='Darcy flow log-permeability',
    display_names=display_names,
    reference_name=reference_title,
)

dashboard.add_results_tables(results_df, results_runinfo_df)

save_reproducibility_log(
    title='Darcy flow density-evaluation reproducibility log',
    config={
        'seed': seed,
        'ACTIVE_DIM': ACTIVE_DIM,
        'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
        'C': C,
        'DEFAULT_N_GEN': DEFAULT_N_GEN,
        'GNL_PILOT_N': GNL_PILOT_N,
        'GNL_STIFF_LAMBDA_CUT': GNL_STIFF_LAMBDA_CUT,
        'GNL_USE_DOMINANT_PARTICLE_NEWTON': GNL_USE_DOMINANT_PARTICLE_NEWTON,
        'HESS_MAX': HESS_MAX,
        'HESS_MIN': HESS_MIN,
        'N': N,
        'NOISE_STD': NOISE_STD,
        'N_REF': N_REF,
        'N_REF_SIGNAL': N_REF_SIGNAL,
        'N_REF_GATE': N_REF_GATE,
        'N_REF_EVAL': N_REF_EVAL,
        'DENSITY_SOURCE_REQUIRED_N': DENSITY_SOURCE_REQUIRED_N,
        'DENSITY_REF_SOURCE': DENSITY_REF_SOURCE,
        'DENSITY_BANK_COUPLING': DENSITY_BANK_COUPLING,
        'DENSITY_EVAL_SOURCE': DENSITY_EVAL_SOURCE,
        'DENSITY_EVAL_BANK_COUPLING': DENSITY_EVAL_BANK_COUPLING,
        'MALA_N_SAMPLES': MALA_N_SAMPLES,
        'MALA_EVAL_N_SAMPLES': MALA_EVAL_N_SAMPLES,
        'MALA_STEPS': MALA_STEPS,
        'MALA_BURNIN': MALA_BURNIN,
        'MALA_DT': MALA_DT,
        'MALA_INIT': MALA_INIT,
        'MALA_PRECOND': MALA_PRECOND,
        'MALA_EVAL_INIT': MALA_EVAL_INIT,
        'MALA_EVAL_PRECOND': MALA_EVAL_PRECOND,
        'MAP_LAPLACE_STARTS': MAP_LAPLACE_STARTS,
        'MAP_LAPLACE_MAX_ITER': MAP_LAPLACE_MAX_ITER,
        'MAP_LAPLACE_TOL': MAP_LAPLACE_TOL,
        'MAP_LAPLACE_RIDGE': MAP_LAPLACE_RIDGE,
        'MAP_LAPLACE_MAX_STEP_NORM': MAP_LAPLACE_MAX_STEP_NORM,
        'MAP_LAPLACE_BACKTRACK_STEPS': MAP_LAPLACE_BACKTRACK_STEPS,
        'BOOT_N_REF': BOOT_N_REF,
        'BOOT_INIT_STEPS': BOOT_INIT_STEPS,
        'BOOT_DRC_PF_STEPS': BOOT_DRC_PF_STEPS,
        'BOOT_DRC_DIVERGENCE': BOOT_DRC_DIVERGENCE,
        'BOOT_DRC_DIV_PROBES': BOOT_DRC_DIV_PROBES,
        'BOOT_DRC_EVAL_BATCH_SIZE': BOOT_DRC_EVAL_BATCH_SIZE,
        'BOOT_DRC_CLIP': BOOT_DRC_CLIP,
        'BOOT_DRC_TEMPERATURE': BOOT_DRC_TEMPERATURE,
        'DENSITY_REF_SOURCE': DENSITY_REF_SOURCE,
        'DENSITY_DRC_PF_STEPS': DENSITY_DRC_PF_STEPS,
        'DENSITY_DRC_EVAL_BATCH_SIZE': DENSITY_DRC_EVAL_BATCH_SIZE,
        'DENSITY_DRC_TMIN': DENSITY_DRC_TMIN,
        'DENSITY_DRC_TMAX': DENSITY_DRC_TMAX,
        'DENSITY_DRC_CLIP': DENSITY_DRC_CLIP,
        'DENSITY_DRC_TEMPERATURE': DENSITY_DRC_TEMPERATURE,
        'DENSITY_DRC_PLOT_AXIS_MODE': DENSITY_DRC_PLOT_AXIS_MODE,
        'DENSITY_DRC_RESIDUAL_AXIS_MODE': DENSITY_DRC_RESIDUAL_AXIS_MODE,
        'DENSITY_DRC_RESIDUAL_KIND': DENSITY_DRC_RESIDUAL_KIND,
        'DENSITY_DRC_AFFINE_FIT_SCOPE': DENSITY_DRC_AFFINE_FIT_SCOPE,
        'DENSITY_DRC_ROBUST_PERCENTILES': DENSITY_DRC_ROBUST_PERCENTILES,
        'DENSITY_DRC_SAVE_RAW_PLOTS': DENSITY_DRC_SAVE_RAW_PLOTS,
        'DENSITY_DRC_SAVE_LOGLOG_PLOTS': DENSITY_DRC_SAVE_LOGLOG_PLOTS,
        'DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS': DENSITY_DRC_SAVE_LOGRATIO_RESIDUAL_PLOTS,
        'DENSITY_DRC_SAVE_LEGACY_ALIAS': DENSITY_DRC_SAVE_LEGACY_ALIAS,
        'DENSITY_DRC_PLOT_LAYOUT': DENSITY_DRC_PLOT_LAYOUT,
        'DENSITY_DRC_GRID_MAX_POINTS': DENSITY_DRC_GRID_MAX_POINTS,
        'DENSITY_DRC_GRID_SAVE_PDF': DENSITY_DRC_GRID_SAVE_PDF,
        'DENSITY_TWEEDIE_DIVERGENCE': DENSITY_TWEEDIE_DIVERGENCE,
        'DENSITY_BLEND_DIVERGENCE': DENSITY_BLEND_DIVERGENCE,
        'DENSITY_LFGI_DIVERGENCE': DENSITY_LFGI_DIVERGENCE,
        'DENSITY_DIV_PROBES': DENSITY_DIV_PROBES,
        'DENSITY_BASELINES': DENSITY_BASELINES,
        'DENSITY_KNOWN_LOGZ': DENSITY_KNOWN_LOGZ,
        'DENSITY_RUN_PF_SENSITIVITY': DENSITY_RUN_PF_SENSITIVITY,
        'DENSITY_PF_SENSITIVITY_LABELS': DENSITY_PF_SENSITIVITY_LABELS,
        'DENSITY_PF_SENSITIVITY_STEPS': DENSITY_PF_SENSITIVITY_STEPS,
        'DENSITY_PF_SENSITIVITY_TMINS': DENSITY_PF_SENSITIVITY_TMINS,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'USE_GAUSS_NEWTON_HESSIAN': True,
        'X': X,
        'Y': Y,
        'd_lat': ACTIVE_DIM,
        'dimension_of_PoI': dimension_of_PoI,
        'display_names': display_names,
        'interior_indices': interior_indices,
        'interior_mask': interior_mask,
        'n_int': n_int,
        'num_modes_available': num_modes_available,
        'num_observation': num_observation,
        'num_holdout_observation': num_holdout_observation,
        'num_truncated_series': num_truncated_series,
        'obs_col': obs_col,
        'obs_indices': obs_indices,
        'obs_indices_train': obs_indices_train,
        'obs_indices_holdout': obs_indices_holdout,
        'obs_locations': obs_locations,
        'obs_locations_train': obs_locations_train,
        'obs_locations_holdout': obs_locations_holdout,
        'obs_row': obs_row,
        'sampler_run_info': sampler_run_info,
        'sigma_prior': sigma_prior,
        'ell': ell,
    },
    extra_sections={
        'saved_results_files': {
            'metrics_csv': results_df_path,
            'runinfo_csv': results_runinfo_df_path,
            'dashboard_pdf': DASHBOARD_PDF_PATH,
            'manuscript_density_table_csv': manuscript_density_df_path,
            'pf_sensitivity_figure': pf_sensitivity_fig_path,
        },
        'drc_density_energy_benchmark': (
            drc_energy_df.to_dict('records') if isinstance(drc_energy_df, pd.DataFrame) and not drc_energy_df.empty else []
        ),
        'manuscript_density_table': (
            manuscript_density_df.to_dict('records') if isinstance(manuscript_density_df, pd.DataFrame) and not manuscript_density_df.empty else []
        ),
        'density_pf_sensitivity': (
            pf_sensitivity_df.to_dict('records') if isinstance(pf_sensitivity_df, pd.DataFrame) and not pf_sensitivity_df.empty else []
        ),
    },
)

# ==========================================
# 4. Problem-specific visualization (restore old layout / scaling)
# ==========================================
print('\nVisualizing Darcy field reconstructions...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1

# --- Figure 1: Log-permeability reconstruction ---
fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))

vis_anchor_key = 'MALA (prior)' if 'MALA (prior)' in mean_fields else next(iter(mean_fields.keys()))

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

plt.suptitle(f'Inverse Darcy flow (d={ACTIVE_DIM})', fontsize=22, y=1.01)
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

plt.suptitle(f'Inverse Darcy flow (d={ACTIVE_DIM}): pressure field', fontsize=16, y=1.05)
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

plt.suptitle(f'Inverse Darcy flow (d={ACTIVE_DIM}): permeability field', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

try:
    sampling._save_all_open_figures_to_run_results()
except Exception as exc:
    print(f'WARNING: final open-figure save before dashboard failed: {exc}')

if DASHBOARD_SHOW_FIGURES:
    dashboard.add_run_results_png_figures(run_ctx['run_results_dir'])
dashboard.close()
plt.close('all')
# sampling.py now accepts extra_paths in the artifact-complete zip helper. The
# fallback keeps this script runnable with older helper versions, but complete
# artifact zipping requires the updated sampling.py from the Navier-Stokes patch.
try:
    run_results_zip_path = zip_run_results_dir(extra_paths=[DASHBOARD_PDF_PATH])
except TypeError:
    run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Dashboard PDF: {DASHBOARD_PDF_PATH}')
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Darcy flow ALT-DRC density comparison pipeline complete ===')
