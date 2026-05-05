# -*- coding: utf-8 -*-
import gc
import os
import sys
from collections import OrderedDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

THIS_DIR = os.getcwd() # if on collab
#THIS_DIR = os.path.dirname(os.path.abspath(__file__)) # if not on collab

REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jax import lax
from scipy.spatial.distance import cdist


#############################################

import sys, importlib, linecache, os

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



############################################


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



# ==========================================
# 0. KL basis generation
# ==========================================
os.makedirs('data', exist_ok=True)

N = 32
x = np.linspace(0.0, 1.0, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing='ij')
coords = np.column_stack([X.ravel(), Y.ravel()])

ELL = 0.1
SIGMA_PRIOR = 1.0
q_max = 100

dists = cdist(coords, coords)
C = SIGMA_PRIOR ** 2 * np.exp(-dists / ELL)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/AllenCahn_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files
# ==========================================
num_observation = 40
num_holdout_observation = 40
num_truncated_series = 32
seed = 42

dimension_of_PoI = N * N
num_modes_available = Basis_Modes.shape[1]
basis_truncated = Basis_Modes[:, :num_truncated_series]

key = jax.random.PRNGKey(seed)
obs_indices_train = np.array(
    jax.random.choice(key, jnp.arange(dimension_of_PoI), shape=(num_observation,), replace=False)
)
remaining_indices = np.setdiff1d(np.arange(dimension_of_PoI), obs_indices_train)
key_holdout = jax.random.PRNGKey(seed + 1)
obs_indices_holdout = np.array(
    jax.random.choice(key_holdout, jnp.array(remaining_indices), shape=(num_holdout_observation,), replace=False)
)
obs_indices = obs_indices_train

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(obs_indices_train).to_csv('data/obs_locations.csv', index=False, header=False)

# ==========================================
# 2. Physics: periodic Allen–Cahn dynamics
# ==========================================
jax.config.update("jax_enable_x64", True)

Basis = jnp.array(basis_truncated, dtype=jnp.float64)
obs_locations_train = jnp.array(obs_indices_train, dtype=int)
obs_locations_holdout = jnp.array(obs_indices_holdout, dtype=int)
obs_locations = obs_locations_train

x_1d = jnp.linspace(0.0, 1.0, N, endpoint=False)
X_grid, Y_grid = jnp.meshgrid(x_1d, x_1d, indexing='ij')

OBS_TIME = 0.33
ALLEN_CAHN_EPS = 0.045
ALLEN_CAHN_DT = 5.0e-4
ALLEN_CAHN_STEPS = int(round(OBS_TIME / ALLEN_CAHN_DT))
NOISE_STD = 0.01

kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=1.0 / N)
ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=1.0 / N)
KX, KY = jnp.meshgrid(kx_1d, ky_1d, indexing='ij')
K2 = KX ** 2 + KY ** 2
IMEX_DENOM = 1.0 / (1.0 + ALLEN_CAHN_DT * (ALLEN_CAHN_EPS ** 2) * K2)


@jax.jit
def _allen_cahn_step(u_field):
    reaction_rhs = u_field + ALLEN_CAHN_DT * (u_field - u_field ** 3)
    reaction_hat = jnp.fft.fftn(reaction_rhs)
    next_hat = reaction_hat * IMEX_DENOM
    next_field = jnp.real(jnp.fft.ifftn(next_hat))
    return next_field


@jax.jit
def _propagate_allen_cahn_obs_time(u0_field):
    def body_fn(_, u):
        return _allen_cahn_step(u)
    return lax.fori_loop(0, ALLEN_CAHN_STEPS, body_fn, u0_field)


def _propagate_allen_cahn(u0_field, t=OBS_TIME):
    """Propagate periodic Allen–Cahn dynamics to time t."""
    if abs(float(t) - float(OBS_TIME)) < 1e-15:
        return _propagate_allen_cahn_obs_time(u0_field)
    n_steps = max(1, int(round(float(t) / ALLEN_CAHN_DT)))
    u = u0_field
    for _ in range(n_steps):
        u = _allen_cahn_step(u)
    return u


@jax.jit
def solve_forward(alpha):
    u0 = jnp.reshape(Basis @ alpha, (N, N))
    uT = _propagate_allen_cahn_obs_time(u0)
    return uT.reshape(-1)[obs_locations_train]


@jax.jit
def solve_forward_holdout(alpha):
    u0 = jnp.reshape(Basis @ alpha, (N, N))
    uT = _propagate_allen_cahn_obs_time(u0)
    return uT.reshape(-1)[obs_locations_holdout]


def solve_full_state(alpha, t=OBS_TIME):
    u0 = jnp.reshape(Basis @ alpha, (N, N))
    return _propagate_allen_cahn(u0, t)


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
DEFAULT_N_GEN = 1000
N_REF = 1000
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
run_ctx = init_run_results('allen_cahn_hlsi')
DASHBOARD_PDF_PATH = os.path.join(
    run_ctx['run_results_dir'],
    f"{run_ctx['run_results_stem']}_summary_dashboard.pdf",
)

# ==========================================
# 3. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)


def make_structured_truth_coefficients(active_dim=ACTIVE_DIM):
    """Build a phase-separated synthetic initial condition and project it into the KL basis."""
    X_np = np.array(X_grid)
    Y_np = np.array(Y_grid)
    r1 = np.sqrt((X_np - 0.30) ** 2 + (Y_np - 0.36) ** 2)
    r2 = np.sqrt((X_np - 0.69) ** 2 + (Y_np - 0.63) ** 2)
    level_set = (
        1.25 * np.tanh((0.17 - r1) / 0.035)
        - 1.05 * np.tanh((0.14 - r2) / 0.040)
        + 0.55 * np.cos(2.0 * np.pi * X_np) * np.cos(2.0 * np.pi * Y_np)
        - 0.20 * np.sin(4.0 * np.pi * (X_np - 0.35 * Y_np))
    )
    truth_field = np.tanh(1.3 * level_set)
    B = basis_truncated[:, :active_dim]
    coeffs, *_ = np.linalg.lstsq(B, truth_field.reshape(-1), rcond=None)
    return coeffs.astype(np.float64), truth_field.astype(np.float64)


alpha_true_np, true_u0_target = make_structured_truth_coefficients(ACTIVE_DIM)
y_clean = solve_forward(jnp.array(alpha_true_np))
y_clean_np = np.array(y_clean)
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)
y_holdout_clean_np = np.array(solve_forward_holdout(jnp.array(alpha_true_np)))
y_holdout_obs_np = y_holdout_clean_np + np.random.normal(0.0, NOISE_STD, size=y_holdout_clean_np.shape)

dashboard = DashboardPDF(
    DASHBOARD_PDF_PATH,
    title='Allen-Cahn IC HLSI dashboard',
)
dashboard.add_text_page(
    'Allen-Cahn IC HLSI dashboard',
    [
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        'This dashboard contains the two canonical saved-results tables plus every PNG diagnostic plot saved in the run directory.',
        'Included PNG diagnostics: ESS vs diffusion time, PCA histograms, field/state visualizations, and sensor residual maps.',
        'Tables are intentionally limited to two pages: metrics plus a readable split run-info page.',
        'Random progress output from precomputation / Hessian batching is intentionally excluded.',
        f"run_results_dir = {run_ctx['run_results_dir']}",
        '',
        f'seed = {seed}',
        f'ACTIVE_DIM = {ACTIVE_DIM}',
        f'N_REF = {N_REF}',
        f'DEFAULT_N_GEN = {DEFAULT_N_GEN}',
        f'NOISE_STD = {NOISE_STD}',
        f'N = {N}, num_observation = {num_observation}, num_holdout_observation = {num_holdout_observation}',
        f'OBS_TIME = {OBS_TIME}',
        f'ALLEN_CAHN_EPS = {ALLEN_CAHN_EPS}',
        f'ALLEN_CAHN_DT = {ALLEN_CAHN_DT}',
        f'ALLEN_CAHN_STEPS = {ALLEN_CAHN_STEPS}',
        f'HESS_MIN = {HESS_MIN}, HESS_MAX = {HESS_MAX}',
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


SAMPLER_CONFIGS = OrderedDict([
    ('CE-HLSI1', {'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI2', {'ref_source': 'CE-HLSI1', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI3', {'ref_source': 'CE-HLSI2', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    ('DRC-CE-HLSI1', {'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('DRC-CE-HLSI2', {'ref_source': 'DRC-CE-HLSI1', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
    ('DRC-CE-HLSI3', {'ref_source': 'DRC-CE-HLSI2', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
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
obs_locs_np = np.array(obs_locations)
obs_row = obs_locs_np // N
obs_col = obs_locs_np % N


def reconstruct_initial_condition(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    d_lat = latents.shape[1]
    B = Basis_np[:, :d_lat]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))



def latent_to_initial_condition(alpha):
    return reconstruct_initial_condition(np.asarray(alpha)[None, :])[0]



def solve_state_field(alpha_vec, t=OBS_TIME):
    return np.array(solve_full_state(jnp.array(alpha_vec), t=t))



def sensor_vector_to_grid(vec, fill_value=np.nan):
    grid = np.full((N, N), fill_value, dtype=np.float64)
    grid.flat[obs_locs_np] = np.asarray(vec)
    return grid


true_u0 = latent_to_initial_condition(alpha_true_np)
true_uT = solve_state_field(alpha_true_np, t=OBS_TIME)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_u0,
    field_from_latent_fn=latent_to_initial_condition,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_clean_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

metrics = compute_heldout_predictive_metrics(
    samples,
    metrics,
    heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
    batched_forward_eval_fn=lambda a_batch: np.asarray(
        batch_solve_forward_holdout(jnp.asarray(a_batch, dtype=jnp.float64))
    ),
    y_holdout_obs_np=y_holdout_obs_np,
    noise_std=NOISE_STD,
    display_names=display_names,
    min_valid=10,
)

mean_final_states = {}
norm_true_uT = np.linalg.norm(true_uT) + 1e-12

print('\n=== Allen–Cahn field/state metrics ===')
print(f"{'Method':<24} | {'IC RelL2(%)':<12} | {'Pearson':<10} | {'RMSE_a':<12} | {'StateRel':<12} | {'SensorRel':<12} | {'HeldoutNLL':<12} | {'HeldoutZ2':<12}")
print('-' * 135)
for label in [lab for lab in samples.keys() if lab in mean_fields]:
    mean_latent = np.asarray(metrics[label]['mean_latent'])
    mean_uT = solve_state_field(mean_latent, t=OBS_TIME)
    mean_final_states[label] = mean_uT
    final_rel = float(np.linalg.norm(mean_uT - true_uT) / norm_true_uT)
    metrics[label]['RMSE_final'] = rmse_array(mean_uT, true_uT)
    metrics[label]['RelL2_final'] = final_rel
    ic_rel_l2_pct = 100.0 * float(metrics[label]['RelL2_field'])
    print(
        f"{display_names.get(label, label):<24} | {ic_rel_l2_pct:<12.4f} | {metrics[label].get('Pearson_field', float('nan')):<10.4f} | "
        f"{metrics[label]['RMSE_alpha']:<12.4e} | {final_rel:<12.4e} | {metrics[label]['FwdRelErr']:<12.4e} | "
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
    target_name='Allen–Cahn initial condition',
    display_names=display_names,
    reference_name=reference_title,
)

dashboard.add_results_tables(results_df, results_runinfo_df)

save_reproducibility_log(
    title='Allen–Cahn IC HLSI run reproducibility log',
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
        'num_holdout_observation': num_holdout_observation,
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
        'OBS_TIME': OBS_TIME,
        'ALLEN_CAHN_EPS': ALLEN_CAHN_EPS,
        'ALLEN_CAHN_DT': ALLEN_CAHN_DT,
        'ALLEN_CAHN_STEPS': ALLEN_CAHN_STEPS,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
    },
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
            'num_methods_with_mean_final_states': len(mean_final_states),
            'num_methods_with_ess_logs': len(ess_logs),
        },
    },
)

# ==========================================
# 4. Problem-specific visualization
# ==========================================

def _overlay_sensors(ax):
    ax.scatter(obs_col, obs_row, c='lime', s=8, marker='.', alpha=0.7)


fig_field, axes_field = plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_initial_condition,
    display_names=display_names,
    true_field=true_u0,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_u0,
    reference_bottom_title='Ground Truth\nInitial condition $u_0(x)$',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_sensors,
    overlay_method_fn=_overlay_sensors,
    suptitle=f'Allen–Cahn IC inversion (d={ACTIVE_DIM}): initial condition reconstruction',
    field_name='Initial condition $u_0(x)$',
)
plt.show()

print('\nVisualizing Allen–Cahn states at observation time...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

im_true_final = axes2[0].imshow(true_uT, cmap='RdBu_r', origin='lower')
axes2[0].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.7, label='Sensors')
axes2[0].set_title(f'Ground Truth\nAllen–Cahn state $u(x,T)$, $T={OBS_TIME:.2f}$', fontsize=14)
axes2[0].axis('off')
axes2[0].legend(fontsize=8, loc='upper right')
plt.colorbar(im_true_final, ax=axes2[0], fraction=0.046, pad=0.04)

vmin_final = float(np.min(true_uT))
vmax_final = float(np.max(true_uT))
for i, label in enumerate(methods_to_plot):
    col = i + 1
    mean_uT = mean_final_states.get(label)
    if mean_uT is None:
        axes2[col].axis('off')
        continue
    axes2[col].imshow(mean_uT, cmap='RdBu_r', origin='lower', vmin=vmin_final, vmax=vmax_final)
    axes2[col].scatter(obs_col, obs_row, c='red', s=12, marker='.', alpha=0.5)
    axes2[col].set_title(f"{display_names.get(label, label)}\nAllen–Cahn state", fontsize=14)
    axes2[col].axis('off')

plt.suptitle(
    f'Allen–Cahn IC inversion (d={ACTIVE_DIM}): Allen–Cahn state at observation time',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

print('\nVisualizing sensor residual maps...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

obs_grid = sensor_vector_to_grid(y_obs_np)
im_obs = axes3[0].imshow(obs_grid, cmap='coolwarm', origin='lower')
axes3[0].set_title('Noisy observations\nat sensors', fontsize=14)
axes3[0].axis('off')
plt.colorbar(im_obs, ax=axes3[0], fraction=0.046, pad=0.04)

residual_scale = 1e-12
residual_grids = {}
for label in methods_to_plot:
    mean_lat = np.asarray(metrics[label]['mean_latent'])
    y_pred = np.array(solve_forward(jnp.array(mean_lat)))
    resid_grid = sensor_vector_to_grid(y_pred - y_obs_np, fill_value=0.0)
    residual_grids[label] = resid_grid
    residual_scale = max(residual_scale, float(np.max(np.abs(resid_grid))))

for i, label in enumerate(methods_to_plot):
    col = i + 1
    im_res = axes3[col].imshow(
        residual_grids[label],
        cmap='RdBu_r',
        origin='lower',
        vmin=-residual_scale,
        vmax=residual_scale,
    )
    axes3[col].set_title(f"{display_names.get(label, label)}\nSensor residuals", fontsize=14)
    axes3[col].axis('off')
    plt.colorbar(im_res, ax=axes3[col], fraction=0.046, pad=0.04)

plt.suptitle(
    f'Allen–Cahn IC inversion (d={ACTIVE_DIM}): sensor-space residuals',
    fontsize=16,
    y=1.05,
)
plt.tight_layout()
plt.show()

if DASHBOARD_SHOW_FIGURES:
    dashboard.add_run_results_png_figures(run_ctx['run_results_dir'])
dashboard.close()
plt.close('all')
# The dashboard lives in the active run-results directory, so zip_run_results_dir()
# includes it alongside the PNGs, CSVs, and reproducibility log.
run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Dashboard PDF: {DASHBOARD_PDF_PATH}')
print(f'Run-results zip: {run_results_zip_path}')
print('\n=== Allen–Cahn IC HLSI pipeline complete ===')
