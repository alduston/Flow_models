# -*- coding: utf-8 -*-
import gc
import os
import sys
import time
from collections import OrderedDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

THIS_DIR = os.getcwd()                                    #if on colab 
#THIS_DIR = os.path.dirname(os.path.abspath(__file__))     #if not on colab
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
import sys, importlib, linecache, os

################################################################################
# Make sure /content itself is before parent dirs.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Clear stale source-line cache and stale imported module.
linecache.clearcache()
if "sampling" in sys.modules:
    del sys.modules["sampling"]

import sampling
importlib.reload(sampling)

print("Using:", sampling.__file__)
print("DRC test:", sampling.canonicalize_init_weights("DRC"))

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
# 0. KL BASIS GENERATION
# ==========================================
os.makedirs('data', exist_ok=True)

N = 48
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
np.savetxt('data/AcousticFWI_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# CONFIGURATION GENERATOR (ACOUSTIC FWI)
# ==========================================
num_truncated_series = 32
num_holdout_observation = 64
seed = 42

BACKGROUND_VELOCITY = 1.0
VELOCITY_LOG_PERTURB_SCALE = 0.32
MODEL_XMIN = 0.10
MODEL_XMAX = 0.90
MODEL_YMIN = 0.16
MODEL_YMAX = 0.88
MODEL_MASK_SOFTNESS = 0.030

N_SOURCES = 8
SOURCE_DEPTH = 0.12
RECEIVER_DEPTH = 0.09
SOURCE_WIDTH = 0.020
SOURCE_COL_INDICES = np.round(np.linspace(4, N - 5, N_SOURCES)).astype(int)
RECEIVER_COL_INDICES = np.arange(1, N - 1)
N_RECEIVERS = RECEIVER_COL_INDICES.size

N_TIME_STEPS = 240
DT = 3.5e-3
RECORD_STRIDE = 4
RICKER_FREQ = 9.0
SPONGE_WIDTH_CELLS = 5
SPONGE_MAX_DAMP = 18.0


def _smooth_box_mask(xx, yy, x_lo, x_hi, y_lo, y_hi, softness):
    sig = lambda z: 1.0 / (1.0 + np.exp(-z / softness))
    return sig(xx - x_lo) * sig(x_hi - xx) * sig(yy - y_lo) * sig(y_hi - yy)


model_support_mask = _smooth_box_mask(
    X, Y, MODEL_XMIN, MODEL_XMAX, MODEL_YMIN, MODEL_YMAX, MODEL_MASK_SOFTNESS
)

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
holdout_step_indices_np = np.setdiff1d(np.arange(N_TIME_STEPS, dtype=int), record_step_indices_np)
holdout_times_np = times_np[holdout_step_indices_np]
N_HOLDOUT_RECORD_STEPS = holdout_step_indices_np.size

ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
dist_to_edge = np.minimum.reduce([ii, jj, (N - 1) - ii, (N - 1) - jj]).astype(np.float64)
sponge_profile = np.clip((SPONGE_WIDTH_CELLS - dist_to_edge) / max(SPONGE_WIDTH_CELLS, 1), 0.0, None)
damping_field = SPONGE_MAX_DAMP * sponge_profile ** 2

num_observation = N_SOURCES * N_RECEIVERS * N_RECORD_STEPS
dimension_of_PoI = N * N

df_modes = pd.read_csv('data/AcousticFWI_Basis_Modes.csv', header=None)
modes_raw = df_modes.to_numpy().flatten().astype(np.float64)
num_modes_available = modes_raw.size // dimension_of_PoI
full_basis = modes_raw.reshape((dimension_of_PoI, num_modes_available))
basis_truncated = full_basis[:, :num_truncated_series]

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(receiver_flat_indices).to_csv('data/obs_locations.csv', index=False, header=False)
pd.DataFrame(source_centers, columns=['x', 'y']).to_csv('data/source_locations.csv', index=False)

key_holdout = jax.random.PRNGKey(seed + 1)
num_holdout_candidates = N_SOURCES * N_RECEIVERS * N_HOLDOUT_RECORD_STEPS
obs_indices_holdout = np.array(
    jax.random.choice(key_holdout, jnp.arange(num_holdout_candidates), shape=(num_holdout_observation,), replace=False)
)

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
holdout_step_indices_jax = jnp.array(holdout_step_indices_np, dtype=int)
obs_indices_holdout_jax = jnp.array(obs_indices_holdout, dtype=int)

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
def _simulate_single_shot_gather_holdout(velocity_field, source_idx):
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
    rec_sub = rec_all[holdout_step_indices_jax, :]
    return rec_sub.T


@jax.jit
def solve_forward(alpha):
    _, velocity_field = _alpha_to_raw_and_velocity(alpha)
    shot_ids = jnp.arange(N_SOURCES)
    gathers = jax.vmap(lambda s_idx: _simulate_single_shot_gather(velocity_field, s_idx))(shot_ids)
    return _flatten_measurements_by_source(gathers)


@jax.jit
def solve_forward_holdout(alpha):
    _, velocity_field = _alpha_to_raw_and_velocity(alpha)
    shot_ids = jnp.arange(N_SOURCES)
    gathers = jax.vmap(lambda s_idx: _simulate_single_shot_gather_holdout(velocity_field, s_idx))(shot_ids)
    holdout_vec = _flatten_measurements_by_source(gathers)
    return holdout_vec[obs_indices_holdout_jax]


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
DEFAULT_N_GEN = 4000
N_REF = 4000
BUILD_GNL_BANKS = False
N_SHOT_GATHER_PLOT_SOURCES = 3

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
run_results_info = init_run_results('acoustic_fwi_hlsi')
DASHBOARD_PDF_PATH = os.path.join(
    run_results_info['run_results_dir'],
    f"{run_results_info['run_results_stem']}_summary_dashboard.pdf",
)

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



def _make_synthetic_raw_truth():
    lens_fast = 1.55 * np.exp(-(((X - 0.33) / 0.10) ** 2 + ((Y - 0.38) / 0.14) ** 2) / 2.0)
    lens_slow = -1.25 * np.exp(-(((X - 0.68) / 0.12) ** 2 + ((Y - 0.56) / 0.11) ** 2) / 2.0)
    ridge_center = 0.62 - 0.09 * np.sin(2.1 * np.pi * (X - 0.08))
    channel = -0.95 * np.exp(-0.5 * ((Y - ridge_center) / 0.050) ** 2) * np.exp(-0.5 * ((X - 0.56) / 0.28) ** 2)
    ripple = 0.18 * np.sin(2.0 * np.pi * X) * np.sin(1.5 * np.pi * Y)
    raw = (lens_fast + lens_slow + channel + ripple) * model_support_mask
    return raw.astype(np.float64)


raw_truth_np = _make_synthetic_raw_truth()
alpha_true_np = _project_raw_field_to_latent(raw_truth_np)
mode_decay = np.linspace(1.0, 0.65, ACTIVE_DIM)
alpha_true_np = alpha_true_np + 0.04 * np.random.randn(ACTIVE_DIM) * mode_decay

y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
NOISE_STD = 0.05 * np.std(y_clean_np)
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)
y_holdout_clean_np = np.array(solve_forward_holdout(jnp.array(alpha_true_np)))
y_holdout_obs_np = y_holdout_clean_np + np.random.normal(0.0, NOISE_STD, size=y_holdout_clean_np.shape)

dashboard = DashboardPDF(
    DASHBOARD_PDF_PATH,
    title='Acoustic FWI HLSI dashboard',
)
dashboard.add_text_page(
    'Acoustic FWI HLSI dashboard',
    [
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        'This dashboard contains the two canonical saved-results tables plus every PNG diagnostic plot saved in the run directory.',
        'Tables are intentionally limited to two pages: metrics plus a readable split run-info page.',
        'Random progress output from precomputation / Hessian batching is intentionally excluded.',
        f"run_results_dir = {run_results_info['run_results_dir']}",
        '',
        f'seed = {seed}',
        f'ACTIVE_DIM = {ACTIVE_DIM}',
        f'N_REF = {N_REF}',
        f'DEFAULT_N_GEN = {DEFAULT_N_GEN}',
        f'NOISE_STD = {NOISE_STD}',
        f'N = {N}, N_SOURCES = {N_SOURCES}, N_RECEIVERS = {N_RECEIVERS}, N_RECORD_STEPS = {N_RECORD_STEPS}',
        f'N_SHOT_GATHER_PLOT_SOURCES = {N_SHOT_GATHER_PLOT_SOURCES}',
        f'HESS_MIN = {HESS_MIN}, HESS_MAX = {HESS_MAX}',
        f'PLOT_NORMALIZER = {PLOT_NORMALIZER}',
    ],
)

def batched_solve_forward_holdout_np(a_batch, chunk_size=8):
    a_batch = np.asarray(a_batch, dtype=np.float64)
    outs = []
    for i in range(0, a_batch.shape[0], chunk_size):
        chunk = jnp.asarray(a_batch[i:i + chunk_size], dtype=jnp.float64)
        outs.append(np.asarray(jax.vmap(solve_forward_holdout)(chunk)))
    return np.concatenate(outs, axis=0)

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
    ('CE-HLSI1', {'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI2', {'ref_source': 'CE-HLSI1', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI3', {'ref_source': 'CE-HLSI2', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI4', {'ref_source': 'CE-HLSI3', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    ('DRC-CE-HLSI1', {'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('DRC-CE-HLSI2', {'ref_source': 'DRC-CE-HLSI1', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
    ('DRC-CE-HLSI3', {'ref_source': 'DRC-CE-HLSI2', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
    ('DRC-CE-HLSI4', {'ref_source': 'DRC-CE-HLSI3', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
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

metrics = compute_heldout_predictive_metrics(
    samples,
    metrics,
    heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
    batched_forward_eval_fn=batched_solve_forward_holdout_np,
    y_holdout_obs_np=y_holdout_obs_np,
    noise_std=NOISE_STD,
    display_names=display_names,
    min_valid=10,
)

print('\n=== Acoustic FWI field/data metrics ===')
print(f"{'Method':<24} | {'RelL2_c (%)':<12} | {'Pearson':<10} | {'RMSE_a':<12} | {'FwdRel':<12} | {'HeldoutNLL':<12} | {'HeldoutZ2':<12}")
print('-' * 135)
norm_true = np.linalg.norm(true_field) + 1e-12
for label in mean_fields:
    data = metrics[label]
    inv_rel_l2_pct = 100.0 * data['RelL2_field']
    print(
        f"{display_names.get(label, label):<24} | {inv_rel_l2_pct:<12.4f} | {data['Pearson_field']:<10.4f} | {data['RMSE_alpha']:<12.4e} | {data['FwdRelErr']:<12.4e} | "
        f"{data.get('HeldoutPredNLL', np.nan):<12.4e} | {data.get('HeldoutStdResSq', np.nan):<12.4e}"
    )

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
    target_name='Acoustic FWI',
    display_names=display_names,
    reference_name=reference_title,
)

dashboard.add_results_tables(results_df, results_runinfo_df)

config_dict = {
    'seed': seed,
    'ACTIVE_DIM': ACTIVE_DIM,
    'N_REF': N_REF,
    'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
    'N_SHOT_GATHER_PLOT_SOURCES': N_SHOT_GATHER_PLOT_SOURCES,
    'PLOT_NORMALIZER': PLOT_NORMALIZER,
    'HESS_MIN': HESS_MIN,
    'HESS_MAX': HESS_MAX,
    'NOISE_STD': NOISE_STD,
    'num_observation': num_observation,
    'num_holdout_observation': num_holdout_observation,
    'num_truncated_series': num_truncated_series,
    'num_modes_available': num_modes_available,
    'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
}

save_reproducibility_log(
    title='Acoustic FWI HLSI run reproducibility log',
    config=config_dict,
    extra_sections={
        'saved_results_files': {'metrics_csv': results_df_path, 'runinfo_csv': results_runinfo_df_path, 'dashboard_pdf': DASHBOARD_PDF_PATH},
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
    ax.scatter(receiver_col, receiver_row, c='lime', s=10, marker='s', alpha=0.8)
    ax.scatter(source_cols, source_rows, c='cyan', s=40, marker='*', alpha=0.9)
    ax.contour(model_support_mask_np, levels=[0.5], colors='white', linewidths=0.9)


fig_field, axes_field = plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_velocity_field,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_raw,
    reference_bottom_title='Ground Truth\nRaw latent field $m(x)$',
    field_cmap='viridis',
    sample_cmap='viridis',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_field,
    overlay_method_fn=_overlay_field,
    suptitle=f'Acoustic FWI (d={ACTIVE_DIM}): velocity reconstruction',
    field_name='Velocity $c(x)$',
)
axes_field[0, 0].legend(['Receivers', 'Sources'], fontsize=8, loc='upper right')
if DASHBOARD_SHOW_FIGURES:
    plt.show()

# ==========================================
# Figure 2: selected-source shot gathers + residuals
# ==========================================
print('\nVisualizing selected acoustic shot gathers and residuals...')

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

# Select a representative subset of source gathers for the signed gather/residual grid.
# N_SHOT_GATHER_PLOT_SOURCES controls the number of individual source gathers shown;
# with the default value 3, the figure has 6 rows: 3 gather rows and 3 residual rows.
n_shot_gather_plot_sources = int(np.clip(int(N_SHOT_GATHER_PLOT_SOURCES), 1, N_SOURCES))
shot_plot_indices = np.unique(
    np.round(np.linspace(0, N_SOURCES - 1, n_shot_gather_plot_sources)).astype(int)
)
# np.unique can only reduce the count when a pathological value slips through; fill
# from the left if needed so the requested row count remains stable.
if shot_plot_indices.size < n_shot_gather_plot_sources:
    for candidate in range(N_SOURCES):
        if candidate not in set(shot_plot_indices.tolist()):
            shot_plot_indices = np.append(shot_plot_indices, candidate)
        if shot_plot_indices.size >= n_shot_gather_plot_sources:
            break
shot_plot_indices = np.sort(shot_plot_indices[:n_shot_gather_plot_sources])

# Use a single symmetric scale for all displayed signed gather panels and a
# separate symmetric scale for displayed signed residual panels.
all_signed_gathers = [true_meas[shot_plot_indices], (obs_meas - true_meas)[shot_plot_indices]]
for pred_all in predicted_meas_by_label.values():
    if pred_all is not None:
        all_signed_gathers.append(pred_all[shot_plot_indices])
        all_signed_gathers.append((pred_all - true_meas)[shot_plot_indices])

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
            [(obs_meas - true_meas)[shot_plot_indices].ravel()] +
            [((pred_all - true_meas)[shot_plot_indices]).ravel() for pred_all in predicted_meas_by_label.values() if pred_all is not None]
        )),
        99.2,
    ))
)

n_cols = len(methods_to_plot) + 1
n_panel_rows = 2 * len(shot_plot_indices)
fig2, axes2 = plt.subplots(
    n_panel_rows,
    n_cols,
    figsize=(4.15 * n_cols, 2.18 * n_panel_rows),
    sharex='col',
    sharey='row',
)
axes2 = np.atleast_2d(axes2)

panel_row_titles = []
for src_idx in shot_plot_indices:
    panel_row_titles.extend([
        f'Source-{src_idx} shot gather',
        f'Source-{src_idx} residual',
    ])

last_src_idx = int(shot_plot_indices[-1])
for plot_pos, src_idx in enumerate(shot_plot_indices):
    row_g = 2 * plot_pos
    row_r = row_g + 1
    src_idx = int(src_idx)
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
    if src_idx == last_src_idx:
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
    if src_idx == last_src_idx:
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
        if src_idx == last_src_idx:
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
        if src_idx == last_src_idx:
            axes2[row_r, col].set_xlabel('Receiver x', fontsize=11)

# Put row labels on the left margin for readability in the grid.
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
    f'Acoustic shot gathers and residual images ({len(shot_plot_indices)} of {N_SOURCES} sources shown: {shot_plot_indices.tolist()})',
    fontsize=16,
    y=1.002,
)
plt.tight_layout()
if DASHBOARD_SHOW_FIGURES:
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
    'Source-0 frequency-slice amplitude / phase receiver traces and residuals',
    fontsize=16,
    y=1.10,
)
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
if DASHBOARD_SHOW_FIGURES:
    plt.show()

# ==========================================
# Figure 4: GN curvature spectrum
# ==========================================
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
ax4.set_title('Acoustic FWI Gauss-Newton curvature spectrum', fontsize=15)
ax4.grid(True, which='both', alpha=0.25)
ax4.legend(fontsize=9)
plt.tight_layout()
if DASHBOARD_SHOW_FIGURES:
    plt.show()

try:
    sampling._save_all_open_figures_to_run_results()
except Exception:
    pass

dashboard.add_run_results_png_figures(run_results_info['run_results_dir'])
dashboard.close()
plt.close('all')
# The dashboard already lives in the active run-results directory, so zip_run_results_dir()
# includes it alongside the PNGs, CSVs, and reproducibility log.
run_results_zip_path = zip_run_results_dir()
print(f'Dashboard PDF: {DASHBOARD_PDF_PATH}')
print(f'Run-results zip: {run_results_zip_path}')
