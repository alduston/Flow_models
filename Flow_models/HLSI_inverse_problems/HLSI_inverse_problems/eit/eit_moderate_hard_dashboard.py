# -*- coding: utf-8 -*-
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

import sampling
importlib.reload(sampling)

print("Using:", sampling.__file__)
try:
    print("DRC test:", sampling.canonicalize_init_weights("DRC"))
except Exception as _exc:
    print(f"DRC canonicalization check skipped: {_exc}")

from sampling import (
    GaussianPrior,
    compute_heldout_predictive_metrics,
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
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x)
coords = np.column_stack([X.ravel(), Y.ravel()])

ELL_X = 0.045
ELL_Y = 0.090
SIGMA_PRIOR = 1.0
q_max = 120

# Moderately harder anisotropic KL prior.  Compared with the original isotropic
# EIT setup, this preserves more high-frequency structure along x than y, which
# produces a rough/smooth directional split and a more ill-conditioned posterior
# without making the PDE solve itself pathological.
dx = coords[:, 0][:, None] - coords[:, 0][None, :]
dy = coords[:, 1][:, None] - coords[:, 1][None, :]
r_aniso = np.sqrt((dx / ELL_X) ** 2 + (dy / ELL_Y) ** 2)
C = SIGMA_PRIOR ** 2 * np.exp(-r_aniso)
C = 0.5 * (C + C.T) + 1e-10 * np.eye(C.shape[0])
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
Basis_Modes = eigvecs[:, :q_max] * np.sqrt(eigvals[:q_max])
np.savetxt('data/EIT_Basis_Modes.csv', Basis_Modes, delimiter=',')

# ==========================================
# 1. Configuration / data files
# ==========================================
num_truncated_series = 40
seed = 42
N_CURRENT_PATTERNS = 12

# Clustered sensor layout: dense coverage on a few boundary arcs and sparse/no
# coverage elsewhere. This is intended to create locally stiff likelihoods near
# instrumented regions while preserving genuine posterior ambiguity away from
# the sensor clusters.
SENSOR_LAYOUT_NAME = 'moderately_harder_clustered_3arc_anisotropic'
SENSOR_CLUSTER_SPECS = (
    {'center_frac': 0.08, 'half_width_frac': 0.065, 'count': 12},
    {'center_frac': 0.41, 'half_width_frac': 0.065, 'count': 12},
    {'center_frac': 0.74, 'half_width_frac': 0.065, 'count': 12},
)
SENSOR_BACKBONE_COUNT = 4
N_ELECTRODES = int(sum(spec['count'] for spec in SENSOR_CLUSTER_SPECS) + SENSOR_BACKBONE_COUNT)
dimension_of_PoI = N * N
num_modes_available = Basis_Modes.shape[1]


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


def _select_evenly_spaced_positions(positions, count):
    positions = np.array(sorted(set(int(p) for p in np.asarray(positions).tolist())), dtype=int)
    if count <= 0 or positions.size == 0:
        return np.array([], dtype=int)
    if count >= positions.size:
        return positions.copy()

    raw_idx = np.linspace(0, positions.size - 1, count)
    selected = []
    used = set()
    for idx in np.round(raw_idx).astype(int):
        idx = int(np.clip(idx, 0, positions.size - 1))
        pos = int(positions[idx])
        if pos not in used:
            selected.append(pos)
            used.add(pos)

    if len(selected) < count:
        remaining = np.array([p for p in positions.tolist() if int(p) not in used], dtype=int)
        if remaining.size > 0:
            extra = _select_evenly_spaced_positions(remaining, count - len(selected))
            for pos in extra.tolist():
                pos = int(pos)
                if pos not in used:
                    selected.append(pos)
                    used.add(pos)

    return np.array(sorted(selected), dtype=int)


def _circular_arc_positions(n_boundary, center_frac, half_width_frac):
    center = float(center_frac % 1.0) * n_boundary
    positions = np.arange(n_boundary, dtype=float)
    clockwise = np.mod(positions - center, n_boundary)
    counterclockwise = np.mod(center - positions, n_boundary)
    circular_distance = np.minimum(clockwise, counterclockwise)
    mask = circular_distance <= (float(half_width_frac) * n_boundary)
    return np.where(mask)[0].astype(int)


def _build_clustered_electrode_positions(n_boundary, cluster_specs, backbone_count=0):
    selected = []
    used = set()
    target_count = int(sum(int(spec['count']) for spec in cluster_specs) + int(backbone_count))

    for spec in cluster_specs:
        arc_positions = _circular_arc_positions(
            n_boundary,
            center_frac=spec['center_frac'],
            half_width_frac=spec['half_width_frac'],
        )
        arc_positions = np.array([p for p in arc_positions.tolist() if int(p) not in used], dtype=int)
        chosen = _select_evenly_spaced_positions(arc_positions, int(spec['count']))
        for pos in chosen.tolist():
            pos = int(pos)
            if pos not in used:
                selected.append(pos)
                used.add(pos)

    if backbone_count > 0:
        remaining = np.array([p for p in range(n_boundary) if p not in used], dtype=int)
        backbone = _select_evenly_spaced_positions(remaining, int(backbone_count))
        for pos in backbone.tolist():
            pos = int(pos)
            if pos not in used:
                selected.append(pos)
                used.add(pos)

    if len(selected) < target_count:
        remaining = np.array([p for p in range(n_boundary) if p not in used], dtype=int)
        filler = _select_evenly_spaced_positions(remaining, target_count - len(selected))
        for pos in filler.tolist():
            pos = int(pos)
            if pos not in used:
                selected.append(pos)
                used.add(pos)

    return np.array(sorted(selected), dtype=int)


boundary_indices_ordered = _ordered_boundary_indices(N)
n_boundary = len(boundary_indices_ordered)
electrode_boundary_pos = _build_clustered_electrode_positions(
    n_boundary,
    cluster_specs=SENSOR_CLUSTER_SPECS,
    backbone_count=SENSOR_BACKBONE_COUNT,
)
if electrode_boundary_pos.size != N_ELECTRODES:
    raise ValueError(f'Expected {N_ELECTRODES} electrodes, got {electrode_boundary_pos.size}')

electrode_flat_indices = boundary_indices_ordered[electrode_boundary_pos]
_train_electrode_set = set(int(i) for i in electrode_flat_indices.tolist())
_holdout_electrode_candidates = np.array(
    [int(i) for i in boundary_indices_ordered.tolist() if int(i) not in _train_electrode_set],
    dtype=int,
)
heldout_electrode_flat_indices = np.array(sorted(_holdout_electrode_candidates.tolist()), dtype=int)
N_HOLDOUT_ELECTRODES = int(heldout_electrode_flat_indices.size)

print(
    f"EIT sensor layout '{SENSOR_LAYOUT_NAME}': {N_ELECTRODES} training sensors across "
    f"{len(SENSOR_CLUSTER_SPECS)} dense clusters + {SENSOR_BACKBONE_COUNT} backbone sensors; "
    f"{N_HOLDOUT_ELECTRODES} boundary nodes held out."
)

boundary_theta = 2.0 * np.pi * np.arange(n_boundary) / n_boundary
current_patterns = np.zeros((N_CURRENT_PATTERNS, n_boundary), dtype=np.float64)
for l in range(N_CURRENT_PATTERNS):
    # Mildly more demanding multiscale boundary currents.  The frequencies skip
    # upward to create stiffer localized sensitivities, but each pattern remains
    # zero-net-current and RMS-normalized for Neumann compatibility.
    k = l + 1 + (l // 3)
    pat = np.cos(k * boundary_theta) + 0.30 * np.sin((2 * k + 1) * boundary_theta + 0.17 * l)
    pat = pat - pat.mean()
    pat = pat / (np.sqrt(np.mean(pat ** 2)) + 1e-12)
    current_patterns[l] = pat

num_observation = N_CURRENT_PATTERNS * N_ELECTRODES
basis_truncated = Basis_Modes[:, :num_truncated_series]

pd.DataFrame(basis_truncated).to_csv('data/Basis.csv', index=False, header=False)
pd.DataFrame(electrode_flat_indices).to_csv('data/obs_locations.csv', index=False, header=False)
pd.DataFrame(heldout_electrode_flat_indices).to_csv('data/heldout_obs_locations.csv', index=False, header=False)

# ==========================================
# 2. Physics: EIT forward model
# ==========================================
jax.config.update("jax_enable_x64", True)

Basis = jnp.array(basis_truncated, dtype=jnp.float64)
obs_locations = jnp.array(electrode_flat_indices, dtype=int)
heldout_obs_locations = jnp.array(heldout_electrode_flat_indices, dtype=int)
current_patterns_jax = jnp.array(current_patterns, dtype=jnp.float64)
boundary_indices_jax = jnp.array(boundary_indices_ordered, dtype=int)

h = 1.0 / (N - 1)
NOISE_STD = 1.5e-5

_xface_left = (jnp.arange(N - 1)[:, None] * N + jnp.arange(N)[None, :]).ravel()
_xface_right = _xface_left + N
_yface_bot = (jnp.arange(N)[:, None] * N + jnp.arange(N - 1)[None, :]).ravel()
_yface_top = _yface_bot + 1
_boundary_ds = h * jnp.ones(n_boundary, dtype=jnp.float64)
_GAUGE_NODE = 0


def _assemble_eit_neumann(sigma_field):
    """
    Assemble the full Neumann stiffness matrix with a gauge pin at node 0.
    """
    n_total = N * N
    h2 = h * h

    sigma_xp = 2.0 * sigma_field[:-1, :] * sigma_field[1:, :] / (sigma_field[:-1, :] + sigma_field[1:, :] + 1e-30)
    sigma_yp = 2.0 * sigma_field[:, :-1] * sigma_field[:, 1:] / (sigma_field[:, :-1] + sigma_field[:, 1:] + 1e-30)

    kx = sigma_xp.ravel() / h2
    ky = sigma_yp.ravel() / h2

    A = jnp.zeros((n_total, n_total), dtype=jnp.float64)
    A = A.at[_xface_left, _xface_left].add(kx)
    A = A.at[_xface_right, _xface_right].add(kx)
    A = A.at[_xface_left, _xface_right].add(-kx)
    A = A.at[_xface_right, _xface_left].add(-kx)

    A = A.at[_yface_bot, _yface_bot].add(ky)
    A = A.at[_yface_top, _yface_top].add(ky)
    A = A.at[_yface_bot, _yface_top].add(-ky)
    A = A.at[_yface_top, _yface_bot].add(-ky)

    A = A.at[_GAUGE_NODE, :].set(0.0)
    A = A.at[:, _GAUGE_NODE].set(0.0)
    A = A.at[_GAUGE_NODE, _GAUGE_NODE].set(1.0)
    return A


@jax.jit
def _build_eit_rhs(current_pattern_boundary):
    rhs = jnp.zeros(N * N, dtype=jnp.float64)
    flux = current_pattern_boundary * _boundary_ds
    rhs = rhs.at[boundary_indices_jax].add(flux)
    rhs = rhs.at[_GAUGE_NODE].set(0.0)
    return rhs


_RHS_ALL_PATTERNS = jnp.stack(
    [_build_eit_rhs(current_patterns_jax[l]) for l in range(N_CURRENT_PATTERNS)],
    axis=1,
)


@jax.jit
def solve_forward(alpha):
    log_sigma = jnp.reshape(Basis @ alpha, (N, N))
    sigma_field = jnp.exp(log_sigma)
    A = _assemble_eit_neumann(sigma_field)
    U = jnp.linalg.solve(A, _RHS_ALL_PATTERNS)
    V = U[obs_locations, :]
    V = V - jnp.mean(V, axis=0, keepdims=True)
    return V.T.ravel()


@jax.jit
def solve_forward_holdout(alpha):
    log_sigma = jnp.reshape(Basis @ alpha, (N, N))
    sigma_field = jnp.exp(log_sigma)
    A = _assemble_eit_neumann(sigma_field)
    U = jnp.linalg.solve(A, _RHS_ALL_PATTERNS)
    V = U[heldout_obs_locations, :]
    V = V - jnp.mean(V, axis=0, keepdims=True)
    return V.T.ravel()


@jax.jit
def solve_single_pattern(alpha, pattern_idx):
    log_sigma = jnp.reshape(Basis @ alpha, (N, N))
    sigma_field = jnp.exp(log_sigma)
    A = _assemble_eit_neumann(sigma_field)
    rhs = _build_eit_rhs(current_patterns_jax[pattern_idx])
    u = jnp.linalg.solve(A, rhs)
    return u.reshape(N, N)


# ==========================================
# Shared sampling configuration
# ==========================================
ACTIVE_DIM = num_truncated_series
PLOT_NORMALIZER = 'best'
HESS_MIN = 5e-5
HESS_MAX = 5e6
GNL_PILOT_N = 1024
GNL_STIFF_LAMBDA_CUT = HESS_MAX
GNL_USE_DOMINANT_PARTICLE_NEWTON = True
DEFAULT_N_GEN = 400
N_REF = 5000
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
run_ctx = init_run_results('eit_moderate_hard_hlsi')
DASHBOARD_PDF_PATH = os.path.join(
    run_ctx['run_results_dir'],
    f"{run_ctx['run_results_stem']}_summary_dashboard.pdf",
)
dashboard = DashboardPDF(
    DASHBOARD_PDF_PATH,
    title='Moderately hard anisotropic EIT HLSI dashboard',
)
dashboard.add_text_page(
    'Moderately hard anisotropic EIT HLSI dashboard',
    [
        'This run keeps the active sampler configuration unchanged while making the EIT inverse problem moderately harder.',
        'Hardening knobs: anisotropic KL prior, 40 latent modes, clustered boundary electrodes, fewer current patterns, tighter observation noise, and multiscale boundary currents.',
        'The dashboard contains the canonical saved-results tables plus every PNG diagnostic plot saved in the run directory.',
    ],
)

# ==========================================
# 3. Experiment execution
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

np.random.seed(seed)
alpha_true_np = np.random.randn(ACTIVE_DIM) * 0.60
y_clean_np = np.array(solve_forward(jnp.array(alpha_true_np)))
y_obs_np = y_clean_np + np.random.normal(0.0, NOISE_STD, size=y_clean_np.shape)
y_holdout_clean_np = np.array(solve_forward_holdout(jnp.array(alpha_true_np)))
y_holdout_obs_np = y_holdout_clean_np + np.random.normal(0.0, NOISE_STD, size=y_holdout_clean_np.shape)

_batched_solve_forward_holdout = jax.jit(jax.vmap(solve_forward_holdout))
HELDOUT_BATCH_SIZE = 2

prior_model = GaussianPrior(dim=ACTIVE_DIM)
lik_model, lik_aux = make_physics_likelihood(
    solve_forward,
    y_obs_np,
    NOISE_STD,
    use_gauss_newton_hessian=True,
    log_batch_size=25,
    grad_batch_size=10,
    hess_batch_size=1,
)
posterior_score_fn = make_posterior_score_fn(lik_model)

print('\n=== True-point GN Hessian / Jacobian diagnostics ===')
try:
    J_true_np = np.asarray(lik_aux['solve_forward_jac_jax'](jnp.asarray(alpha_true_np, dtype=jnp.float64)))
    svals_true = np.linalg.svd(J_true_np, compute_uv=False)
    gn_eigs_true = 1.0 + (svals_true ** 2) / (NOISE_STD ** 2)
    gn_cond_true = float(np.max(gn_eigs_true) / max(np.min(gn_eigs_true), 1e-300)) if gn_eigs_true.size else np.nan
    print(f"  J_true shape: {J_true_np.shape}")
    print(f"  singular values: min={np.min(svals_true):.4e}, median={np.median(svals_true):.4e}, max={np.max(svals_true):.4e}")
    print(f"  GN precision eigs: min={np.min(gn_eigs_true):.4e}, median={np.median(gn_eigs_true):.4e}, max={np.max(gn_eigs_true):.4e}, cond={gn_cond_true:.4e}")

    fig_diag, axes_diag = plt.subplots(1, 2, figsize=(12, 4))
    axes_diag[0].semilogy(np.arange(1, len(svals_true) + 1), np.sort(svals_true)[::-1], marker='o', linewidth=1.2, markersize=3)
    axes_diag[0].set_title('EIT forward-Jacobian singular values at truth')
    axes_diag[0].set_xlabel('index')
    axes_diag[0].set_ylabel('singular value')
    axes_diag[0].grid(True, alpha=0.3)
    axes_diag[1].semilogy(np.arange(1, len(gn_eigs_true) + 1), np.sort(gn_eigs_true)[::-1], marker='o', linewidth=1.2, markersize=3)
    axes_diag[1].set_title('GN posterior precision spectrum at truth')
    axes_diag[1].set_xlabel('index')
    axes_diag[1].set_ylabel(r'$1 + s_i^2 / \sigma^2$')
    axes_diag[1].grid(True, alpha=0.3)
    fig_diag.suptitle('Moderately hard EIT: local ill-conditioning diagnostic', fontsize=14)
    fig_diag.tight_layout()
    plt.show()
except Exception as exc:
    J_true_np = None
    svals_true = np.array([])
    gn_eigs_true = np.array([])
    gn_cond_true = np.nan
    print(f"  WARNING: true-point GN diagnostics failed: {exc}")

'''
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


SAMPLER_CONFIGS = OrderedDict([
    ('MALA', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'precond_mala': False, 'is_reference': True}),
    ('Precond MALA', { 'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 4e-3, 'precond_mala': True, 'is_reference': True,}),
    ('Ref_Laplace', {'init': 'Ref_Laplace', 'init_weights': 'WC', 'init_steps': 0, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),
    ('Tweedie', {'init': 'tweedie', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('Blend', {'init': 'blend', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),

    ('HLSI', {'init': 'HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI', {'init': 'HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI', {'init': 'HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),
    ('CE-HLSI', {'init': 'CE-HLSI', 'init_weights': 'L', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-WC-HLSI', {'init': 'CE-HLSI', 'init_weights': 'WC', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-PoU-HLSI', {'init': 'CE-HLSI', 'init_weights': 'PoU', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': False}),

    ('KAPPA05_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.5}),
    ('KAPPA10_PoU', {'init': 'GATE-HLSI', 'init_weights': 'PoU', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),

    ('KAPPA05_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 0.5, 'log_mean_ess': True}),
    ('KAPPA10_WC', {'init': 'GATE-HLSI', 'init_weights': 'WC', 'gate_rho': 1.0, 'gate_beta': 1.0, 'gate_kappa': 1.0}),
])
'''

SAMPLER_CONFIGS = OrderedDict([
    ('MALA (prior)', {'init': 'prior', 'init_steps': 0, 'mala_steps': 500, 'mala_burnin': 100, 'mala_dt': 1e-4, 'is_reference': True}),
    ('HLSI-OU', {'init': 'HLSI', 'init_weights': 'L', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('HLSI-Surr', {'init': 'HLSI', 'init_weights': 'L', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI-OU', {'init': 'CE-HLSI', 'init_weights': 'L', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI-Surr', {'init': 'CE-HLSI', 'init_weights': 'L', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI-OU', {'init': 'HLSI', 'init_weights': 'WC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('WC-HLSI-Surr', {'init': 'HLSI', 'init_weights': 'WC', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI-OU', {'init': 'HLSI', 'init_weights': 'PoU', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('PoU-HLSI-Surr', {'init': 'HLSI', 'init_weights': 'PoU', 'transition_w': 'surrogate', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
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
electrode_row = electrode_flat_indices // N
electrode_col = electrode_flat_indices % N
d_lat = ACTIVE_DIM


def reconstruct_log_conductivity(latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    B = Basis_np[:, :latents.shape[1]]
    fields_flat = latents @ B.T
    return fields_flat.reshape((-1, N, N))


def latent_to_log_conductivity(alpha):
    return reconstruct_log_conductivity(np.asarray(alpha)[None, :])[0]


def solve_potential_field(alpha_vec, pattern_idx=0):
    return np.array(solve_single_pattern(jnp.array(alpha_vec), pattern_idx))


true_field = latent_to_log_conductivity(alpha_true_np)
true_potential = solve_potential_field(alpha_true_np, pattern_idx=0)

mean_fields, metrics = compute_field_summary_metrics(
    samples,
    metrics,
    alpha_true_np,
    true_field,
    field_from_latent_fn=latent_to_log_conductivity,
    forward_eval_fn=lambda a: np.array(solve_forward(jnp.array(a))),
    y_ref_np=y_clean_np,
    display_names=display_names,
    min_valid=10,
    d_lat=ACTIVE_DIM,
)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    metrics = compute_heldout_predictive_metrics(
        samples,
        metrics,
        heldout_forward_eval_fn=lambda a: np.array(solve_forward_holdout(jnp.array(a))),
        batched_forward_eval_fn=lambda alpha_batch: np.asarray(
            _batched_solve_forward_holdout(jnp.asarray(alpha_batch, dtype=jnp.float64))
        ),
        batched_forward_eval_batch_size=HELDOUT_BATCH_SIZE,
        y_holdout_obs_np=y_holdout_obs_np,
        noise_std=NOISE_STD,
        display_names=display_names,
        min_valid=10,
    )
except Exception as exc:
    print(f"WARNING: held-out predictive metrics failed and will be skipped: {exc}")

print('\n=== EIT field/data metrics ===')
print(f"{'Method':<24} | {'RelL2_m (%)':<12} | {'Pearson':<10} | {'RMSE_a':<12} | {'FwdRel':<12}")
print('-' * 84)
for label in mean_fields:
    data = metrics[label]
    inv_rel_l2_pct = 100.0 * data['RelL2_field']
    print(f"{display_names.get(label, label):<24} | {inv_rel_l2_pct:<12.4f} | {data['Pearson_field']:<10.4f} | {data['RMSE_alpha']:<12.4e} | {data['FwdRelErr']:<12.4e}")

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
    target_name='Moderately hard anisotropic EIT log-conductivity',
    display_names=display_names,
    reference_name=reference_title,
)
dashboard.add_results_tables(results_df, results_runinfo_df)

save_reproducibility_log(
    title='Moderately hard anisotropic EIT HLSI run reproducibility log',
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
        'num_truncated_series': num_truncated_series,
        'num_modes_available': num_modes_available,
        'ELL_X': ELL_X,
        'ELL_Y': ELL_Y,
        'SIGMA_PRIOR': SIGMA_PRIOR,
        'N_CURRENT_PATTERNS': N_CURRENT_PATTERNS,
        'SENSOR_LAYOUT_NAME': SENSOR_LAYOUT_NAME,
        'SENSOR_CLUSTER_SPECS': SENSOR_CLUSTER_SPECS,
        'SENSOR_BACKBONE_COUNT': SENSOR_BACKBONE_COUNT,
        'N_ELECTRODES': N_ELECTRODES,
        'N_HOLDOUT_ELECTRODES': N_HOLDOUT_ELECTRODES,
        'electrode_boundary_pos': electrode_boundary_pos,
        'heldout_electrode_flat_indices': heldout_electrode_flat_indices,
        'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
        'HELDOUT_BATCH_SIZE': HELDOUT_BATCH_SIZE,
        'DASHBOARD_PDF_PATH': DASHBOARD_PDF_PATH,
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
            'gn_cond_true': gn_cond_true,
            'gn_min_eig_true': float(np.min(gn_eigs_true)) if len(gn_eigs_true) else np.nan,
            'gn_max_eig_true': float(np.max(gn_eigs_true)) if len(gn_eigs_true) else np.nan,
        },
    },
)

# ==========================================
# 4. Problem-specific visualization
# ==========================================

def _overlay_electrodes(ax):
    ax.scatter(electrode_col, electrode_row, c='lime', s=10, marker='s', alpha=0.8)


fig_field, axes_field = plot_field_reconstruction_grid(
    samples,
    mean_fields,
    reconstruct_log_conductivity,
    display_names=display_names,
    true_field=true_field,
    plot_normalizer_key=plot_normalizer_key,
    reference_bottom_panel=true_field,
    reference_bottom_title='Ground Truth\nLog-conductivity $m(x)$',
    field_cmap='RdBu_r',
    sample_cmap='RdBu_r',
    bottom_cmap='RdBu_r',
    overlay_reference_fn=_overlay_electrodes,
    overlay_method_fn=_overlay_electrodes,
    suptitle=f'Inverse EIT (d={ACTIVE_DIM}): log-conductivity reconstruction',
    field_name='Log-conductivity $m(x)$',
)
plt.show()

print('\nVisualizing electric potential fields (pattern 1)...')
methods_to_plot = [label for label in samples.keys() if label in mean_fields]
n_cols = len(methods_to_plot) + 1
fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

pot_vmin = float(np.min(true_potential))
pot_vmax = float(np.max(true_potential))
axes2[0].imshow(true_potential, cmap='RdBu_r', origin='lower', vmin=pot_vmin, vmax=pot_vmax)
axes2[0].scatter(electrode_col, electrode_row, c='yellow', s=18, marker='s', alpha=0.8)
axes2[0].set_title('Ground Truth\nPotential $u(x)$', fontsize=14)
axes2[0].axis('off')

for i, label in enumerate(methods_to_plot):
    col = i + 1
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        axes2[col].axis('off')
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    try:
        u_field = solve_potential_field(mean_lat, pattern_idx=0)
        axes2[col].imshow(u_field, cmap='RdBu_r', origin='lower', vmin=pot_vmin, vmax=pot_vmax)
        axes2[col].scatter(electrode_col, electrode_row, c='yellow', s=10, marker='s', alpha=0.6)
        axes2[col].set_title(f"{display_names.get(label, label)}\nPotential", fontsize=14)
    except Exception:
        axes2[col].set_title(f"{display_names.get(label, label)}\n(solve failed)", fontsize=14)
    axes2[col].axis('off')

plt.suptitle(f'Inverse EIT (d={ACTIVE_DIM}): electric potential (pattern 1)', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

print('\nVisualizing conductivity fields $\\sigma(x)=e^{m(x)}$...')
fig3, axes3 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
true_conductivity = np.exp(true_field)
cond_vmin = float(np.min(true_conductivity))
cond_vmax = float(np.max(true_conductivity))
axes3[0].imshow(true_conductivity, cmap='magma', origin='lower', vmin=cond_vmin, vmax=cond_vmax)
axes3[0].set_title('Ground Truth\n$\\sigma(x)=e^{m(x)}$', fontsize=14)
axes3[0].axis('off')

for i, label in enumerate(methods_to_plot):
    col = i + 1
    cond_f = np.exp(mean_fields[label])
    axes3[col].imshow(cond_f, cmap='magma', origin='lower', vmin=cond_vmin, vmax=cond_vmax)
    axes3[col].set_title(f"{display_names.get(label, label)}\n$\\sigma(x)=e^{{m(x)}}$", fontsize=14)
    axes3[col].axis('off')

plt.suptitle(f'Inverse EIT (d={ACTIVE_DIM}): conductivity field', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

print('\nVisualizing boundary voltage profiles...')
theta_electrodes = 2.0 * np.pi * electrode_boundary_pos / n_boundary
method_voltage_preds = {}
for label in methods_to_plot:
    samps_clean = get_valid_samples(samples[label])
    if samps_clean.shape[0] < 10:
        continue
    mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
    try:
        method_voltage_preds[label] = np.array(solve_forward(jnp.array(mean_lat)))
    except Exception:
        pass

preferred_order = [
    'HLSI', 'WC-HLSI', 'PoU-HLSI', 'CE-HLSI', 'CE-WC-HLSI', 'CE-PoU-HLSI', 'MALA (prior)'
]
labels_for_voltage_plot = [lab for lab in preferred_order if lab in method_voltage_preds]
if len(labels_for_voltage_plot) == 0:
    labels_for_voltage_plot = list(method_voltage_preds.keys())

n_patterns_show = min(4, N_CURRENT_PATTERNS)
pattern_indices = list(range(n_patterns_show))
n_cols_plot = min(2, n_patterns_show)
n_rows_plot = int(np.ceil(n_patterns_show / n_cols_plot))
fig4, axes4 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(7 * n_cols_plot, 4.5 * n_rows_plot), squeeze=False)

for ax, pattern_idx in zip(axes4.ravel(), pattern_indices):
    sl = slice(pattern_idx * N_ELECTRODES, (pattern_idx + 1) * N_ELECTRODES)
    y_obs_pat = y_obs_np[sl]
    y_clean_pat = y_clean_np[sl]
    ax.plot(theta_electrodes, y_clean_pat, 'k-o', label='Clean', linewidth=2, markersize=4)
    ax.plot(theta_electrodes, y_obs_pat, 'r.', label='Noisy obs', markersize=5, alpha=0.7)
    for label in labels_for_voltage_plot:
        y_pred_pat = method_voltage_preds[label][sl]
        ax.plot(theta_electrodes, y_pred_pat, '--', label=display_names.get(label, label), linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Boundary angle (rad)', fontsize=12)
    ax.set_ylabel('Voltage', fontsize=12)
    ax.set_title(f'Boundary voltage profile (pattern {pattern_idx + 1})', fontsize=14)
    ax.grid(True, alpha=0.3)

for ax in axes4.ravel()[len(pattern_indices):]:
    ax.axis('off')

handles, labels = axes4.ravel()[0].get_legend_handles_labels()
fig4.legend(handles, labels, loc='upper center', ncol=min(len(labels), 4), fontsize=10)
fig4.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


print('\nVisualizing electrode/current-pattern geometry...')
fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
all_bxy = np.column_stack([boundary_indices_ordered % N, boundary_indices_ordered // N])
heldout_row = heldout_electrode_flat_indices // N
heldout_col = heldout_electrode_flat_indices % N
axes5[0].scatter(all_bxy[:, 0], all_bxy[:, 1], c='0.85', s=10, label='Boundary')
axes5[0].scatter(heldout_col, heldout_row, c='tab:blue', s=16, alpha=0.45, label='Held-out boundary')
axes5[0].scatter(electrode_col, electrode_row, c='tab:red', s=30, marker='s', label='Training electrodes')
axes5[0].set_title(f"Electrode layout: {N_ELECTRODES} train / {N_HOLDOUT_ELECTRODES} held out")
axes5[0].set_xlim(-1, N)
axes5[0].set_ylim(-1, N)
axes5[0].set_aspect('equal')
axes5[0].invert_yaxis()
axes5[0].grid(alpha=0.15)
axes5[0].legend(fontsize=8, loc='upper right')
for k in range(min(6, N_CURRENT_PATTERNS)):
    axes5[1].plot(boundary_theta, current_patterns[k], linewidth=1.3, label=f'pattern {k + 1}')
axes5[1].set_title('First multiscale boundary current patterns')
axes5[1].set_xlabel('Boundary angle (rad)')
axes5[1].set_ylabel('Current density')
axes5[1].grid(alpha=0.3)
axes5[1].legend(fontsize=8, ncol=2)
fig5.suptitle('Moderately hard anisotropic EIT sensing geometry', fontsize=14)
fig5.tight_layout()
plt.show()

dashboard.add_run_results_png_figures(run_ctx['run_results_dir'])
dashboard.close()

run_results_zip_path = zip_run_results_dir()
print(f"Run-results directory: {run_ctx['run_results_dir']}")
print(f'Run-results zip: {run_results_zip_path}')
print(f'Dashboard PDF: {DASHBOARD_PDF_PATH}')
print('\n=== EIT HLSI pipeline complete ===')
