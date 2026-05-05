# -*- coding: utf-8 -*-
import gc
import os
import sys
from collections import OrderedDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

THIS_DIR = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
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


'''
import sampling as sampling_utils
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
'''


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
N_HOLDOUT_RECEIVERS = None
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
receiver_boundary_pos = np.unique(receiver_boundary_pos)
receiver_flat_indices = boundary_indices_ordered[receiver_boundary_pos]
remaining_boundary_pos = np.setdiff1d(np.arange(n_boundary), receiver_boundary_pos)
if N_HOLDOUT_RECEIVERS is None:
    N_HOLDOUT_RECEIVERS = remaining_boundary_pos.size
else:
    N_HOLDOUT_RECEIVERS = min(int(N_HOLDOUT_RECEIVERS), remaining_boundary_pos.size)
holdout_boundary_pos = remaining_boundary_pos[:N_HOLDOUT_RECEIVERS]
holdout_receiver_flat_indices = boundary_indices_ordered[holdout_boundary_pos]
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
num_holdout_observation = N_SOURCES * 2 * N_HOLDOUT_RECEIVERS
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
holdout_receiver_indices_jax = jnp.array(holdout_receiver_flat_indices, dtype=int)
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


def _flatten_measurements_by_source_generic(meas_complex, n_receivers):
    parts = []
    for s in range(N_SOURCES):
        parts.append(jnp.real(meas_complex[:n_receivers, s]))
        parts.append(jnp.imag(meas_complex[:n_receivers, s]))
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
def solve_forward_holdout(alpha):
    _, _, n2_field = _alpha_to_raw_and_contrast(alpha)
    A = _assemble_helmholtz_operator(n2_field)
    U_total = jnp.linalg.solve(A, source_terms_jax)
    U_scat = U_total - BACKGROUND_FIELDS
    meas = U_scat[holdout_receiver_indices_jax, :]
    return _flatten_measurements_by_source_generic(meas, holdout_receiver_indices_jax.shape[0])


batch_solve_forward_holdout = jax.jit(jax.vmap(solve_forward_holdout))


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
DEFAULT_N_GEN = 1500
N_REF = 1500
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
run_results_info = init_run_results('helmholtz_scattering_hlsi')
DASHBOARD_PDF_PATH = os.path.join(
    run_results_info['run_results_dir'],
    f"{run_results_info['run_results_stem']}_summary_dashboard.pdf",
)

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
NOISE_STD = 1e-4
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)
y_clean_holdout = solve_forward_holdout(jnp.array(alpha_true_np))
y_clean_holdout_np = np.array(y_clean_holdout)
y_holdout_obs_np = y_clean_holdout_np + np.random.normal(0.0, NOISE_STD, size=y_clean_holdout_np.shape)
HELDOUT_BATCH_SIZE = 1

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
   #('MALA', {'init': 'prior', 'init_steps': 0, 'mala_steps': 1, 'mala_burnin': 0, 'mala_dt': 1e-4, 'is_reference': True}),

    #('HLSI', {'init': 'HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('WC-HLSI', {'init': 'HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    #('MALA_HLSI', {'ref_source': 'MALA', 'init': 'HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('HLSI_HLSI', {'ref_source': 'HLSI', 'init': 'HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('WC-HLSI_HLSI', {'ref_source': 'WC-HLSI', 'init': 'HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('HLSI_HLSI', {'ref_source': 'HLSI', 'init': 'HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    #('MALA_CE-HLSI', {'ref_source': 'MALA', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('HLSI_CE-HLSI', {'ref_source': 'HLSI', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('WC-HLSI_CE-HLSI', {'ref_source': 'WC-HLSI', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI_CE-HLSI', {'ref_source': 'CE-HLSI', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    ('CE-HLSI_CE-HLSI_CE-HLSI', {'ref_source': 'CE-HLSI_CE-HLSI', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('HLSI_HLSI_HLSI', {'ref_source': 'HLSI_HLSI', 'init': ' HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('CE-HLSI_HLSI_HLSI_HLSI', {'ref_source': 'CE-HLSI_HLSI_HLSI', 'init': ' HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    #('CE-HLSI_HLSI_HLSI_HLSI_HSLI', {'ref_source': 'CE-HLSI_HLSI_HLSI_HLSI', 'init': ' HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),


])
dashboard = DashboardPDF(
    DASHBOARD_PDF_PATH,
    title="Helmholtz scattering HLSI dashboard",
)
dashboard.add_text_page(
    "Helmholtz scattering HLSI dashboard",
    [
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "This dashboard contains the two canonical saved-results tables plus every PNG diagnostic plot saved in the run directory.",
        "Tables are intentionally limited to two pages: metrics plus a readable split run-info page.",
        "Random progress output from precomputation / Hessian batching is intentionally excluded.",
        f"run_results_dir = {run_results_info['run_results_dir']}",
        "",
        f"seed = {seed}",
        f"ACTIVE_DIM = {ACTIVE_DIM}",
        f"N_REF = {N_REF}",
        f"DEFAULT_N_GEN = {DEFAULT_N_GEN}",
        f"NOISE_STD = {NOISE_STD}",
        f"HELMHOLTZ_K = {HELMHOLTZ_K}",
        f"N = {N}, N_SOURCES = {N_SOURCES}, N_RECEIVERS = {N_RECEIVERS}, N_HOLDOUT_RECEIVERS = {N_HOLDOUT_RECEIVERS}",
        f"HESS_MIN = {HESS_MIN}, HESS_MAX = {HESS_MAX}",
        f"PLOT_NORMALIZER = {PLOT_NORMALIZER}",
    ],
)
pipeline = run_standard_sampler_pipeline(prior_model, lik_model, SAMPLER_CONFIGS, n_ref=N_REF, build_gnl_banks=BUILD_GNL_BANKS, compute_pou=True)
samples = pipeline['samples']
ess_logs = pipeline['ess_logs']
sampler_run_info = pipeline['sampler_run_info']
display_names = pipeline['display_names']
reference_key = pipeline['reference_key']
reference_title = pipeline['reference_title']

summarize_sampler_run(sampler_run_info)
plot_mean_ess_logs(ess_logs, display_names=display_names)
metrics = compute_latent_metrics(samples, reference_key, alpha_true_np, prior_model, lik_model, posterior_score_fn, display_names=display_names)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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
support_mask_np = np.array(support_mask_jax)
receiver_row = receiver_flat_indices // N
receiver_col = receiver_flat_indices % N
holdout_receiver_row = holdout_receiver_flat_indices // N
holdout_receiver_col = holdout_receiver_flat_indices % N


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



BOUNDARY_TRACE_COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown',
    'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray', 'tab:red',
]
BOUNDARY_TRACE_MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '<', '>', 'h']


def trace_style_for_index(index):
    """Fixed-order trace style so CE-HLSI bootstrap variants get distinct colors."""
    color = BOUNDARY_TRACE_COLORS[index % len(BOUNDARY_TRACE_COLORS)]
    marker = BOUNDARY_TRACE_MARKERS[index % len(BOUNDARY_TRACE_MARKERS)]
    return dict(
        color=color,
        linestyle='--',
        linewidth=1.55,
        alpha=0.92,
        zorder=6 + index,
        marker=marker,
        markersize=3.2,
        markerfacecolor='white',
        markeredgewidth=0.8,
        markevery=4,
    )


def trace_style_for_label(label):
    # Backward-compatible fallback for any old call sites. New boundary-trace
    # plotting uses trace_style_for_index(...) to guarantee distinct colors.
    return trace_style_for_index(0)


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

dashboard.add_results_tables(results_df, results_runinfo_df)

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
        'HELDOUT_BATCH_SIZE': HELDOUT_BATCH_SIZE,
        'num_holdout_observation': num_holdout_observation,
        'N_HOLDOUT_RECEIVERS': N_HOLDOUT_RECEIVERS,
        'num_observation': num_observation,
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
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
try:
    sampling_utils._save_all_open_figures_to_run_results()
except Exception:
    pass
if DASHBOARD_SHOW_FIGURES:
    plt.show()
plt.close(fig2)

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
for trace_idx, label in enumerate(methods_to_plot[:4]):
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    y_pred = unpack_measurement_vector(np.array(solve_forward(jnp.array(mean_lat))))[0]
    pretty_label = display_names.get(label, label)
    model_trace_data[pretty_label] = {'real': np.real(y_pred).copy(), 'imag': np.imag(y_pred).copy(), 'style': trace_style_for_index(trace_idx)}
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
for h, l in zip(handles, labels):
    if l not in legend_map:
        legend_map[l] = h
fig3.legend(legend_map.values(), legend_map.keys(), loc='upper center', ncol=min(6, len(legend_map)), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
fig3.suptitle('Source-0 scattered-field boundary traces', fontsize=16, y=1.08)
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
try:
    sampling_utils._save_all_open_figures_to_run_results()
except Exception:
    pass
if DASHBOARD_SHOW_FIGURES:
    plt.show()
plt.close(fig3)

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
try:
    sampling_utils._save_all_open_figures_to_run_results()
except Exception:
    pass
if DASHBOARD_SHOW_FIGURES:
    plt.show()
plt.close(fig4)

dashboard.add_run_results_png_figures(run_results_info['run_results_dir'])
dashboard.close()
# The dashboard already lives in the active run-results directory, so zip_run_results_dir()
# includes it alongside the PNGs, CSVs, and reproducibility log.
run_results_zip_path = zip_run_results_dir()
print(f'Dashboard PDF: {DASHBOARD_PDF_PATH}')
print(f'Run-results zip: {run_results_zip_path}')
