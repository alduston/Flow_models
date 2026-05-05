# -*- coding: utf-8 -*-
"""
HLSI sampler experiment for Assignment 4 / Problem II Helmholtz physics.

This is an adaptation of the existing helmholtz.py HLSI experiment to the HW4
setup:
    - unit disk Omega = {||x|| <= 1}, approximated by a masked Cartesian grid;
    - real scattered-field Helmholtz equation with homogeneous Neumann boundary;
    - tanh(m) medium reparameterization;
    - true field m=0.2 inside sqrt((x-0.1)^2 + 2(y+0.2)^2)<0.5, 0 outside;
    - three point-like incident spherical waves with e_j ~ N(0,1);
    - either one superposed source, or three independent source experiments;
    - boundary observations with 1% iid Gaussian noise.

Run from the same directory as sampling.py:
    python helmholtz_hw_hlsi.py

Notes:
    * This is not intended to be a replacement for the required FEniCS homework
      notebook. It is a deliberately apples-to-oranges posterior-sampling
      sanity check against the same physics/data regime.
    * If the course starter code uses specific point-source locations, replace
      SOURCE_POINTS below with those locations.
"""

import gc
import os
import sys
from collections import OrderedDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

THIS_DIR = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
for path in (THIS_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

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






def _stdout_format_metric_cell(x):
    """Compact numeric formatting for full metric tables printed to stdout."""
    try:
        if isinstance(x, np.generic):
            x = x.item()
    except Exception:
        pass
    if x is None:
        return ""
    if isinstance(x, numbers.Number):
        xf = float(x)
        if not np.isfinite(xf):
            return str(x)
        if abs(xf) >= 1e4 or (0 < abs(xf) < 1e-3):
            return f"{xf:.6e}"
        return f"{xf:.6f}"
    return str(x)


def _format_df_for_stdout(df):
    out = pd.DataFrame(df).copy()
    for col in out.columns:
        out[col] = out[col].map(_stdout_format_metric_cell)
    return out


def print_full_metrics_to_stdout(results_df, results_runinfo_df=None, title='Run'):
    """Print the exact saved metrics table plus a method-wise recovery summary."""
    if results_df is None or len(results_df) == 0:
        print(f"\n=== Full metrics table: {title} ===")
        print("No metrics available.")
        return

    print(f"\n=== Full saved metrics table: {title} ===")
    print("Rows match the saved *_metrics.csv / dashboard metrics table.")
    print(_format_df_for_stdout(results_df).to_string())

    preferred_rows = [
        'RMSE_alpha', 'RelL2_alpha', 'MMD_to_reference', 'KSD', 'KLdiag',
        'RMSE_field', 'Pearson_field', 'RelL2_field', 'InverseRelL2_percent',
        'RMSE_tanh_m', 'RelL2_tanh_m', 'TanhMRelL2_percent',
        'FwdRelErr', 'HeldoutPredNLL', 'HeldoutStdResSq', 'HeldoutStdResRMS',
        'HeldoutPredNumValid',
    ]
    available_rows = [row for row in preferred_rows if row in results_df.index]
    if available_rows:
        method_df = results_df.loc[available_rows].T
        method_df.index.name = 'Method'
        print(f"\n=== Method-wise recovery / calibration summary: {title} ===")
        print(_format_df_for_stdout(method_df).to_string())

    if results_runinfo_df is not None and len(results_runinfo_df) > 0:
        runinfo = pd.DataFrame(results_runinfo_df).copy()
        cols = [c for c in [
            'display_name', 'method', 'weight_mode', 'N_ref', 'steps', 'mala_steps',
            'mala_burnin', 'mala_step_size', 'score_norm', 'score_norm_initial',
            'score_norm_mean', 'score_norm_final', 'score_norm_max',
            'pde_likelihood_evals', 'pde_score_evals', 'pde_gn_hessian_evals',
            'pde_solve_count', 'runtime_seconds',
        ] if c in runinfo.columns]
        if cols:
            print(f"\n=== Run-info summary: {title} ===")
            print(_format_df_for_stdout(runinfo[cols]).to_string(index=False))


def create_summary_dashboard_for_current_run(run_name, source_mode, k0, results_df, results_runinfo_df,
                                             dashboard_config=None, extra_lines=None):
    """Create one Helmholtz HW dashboard PDF for the current sampling.py run directory."""
    run_dir = getattr(sampling, 'RUN_RESULTS_DIR', None)
    run_stem = getattr(sampling, 'RUN_RESULTS_STEM', run_name)
    if run_dir is None or run_stem is None:
        print('WARNING: Could not create summary dashboard because sampling.RUN_RESULTS_DIR is unset.')
        return None

    if hasattr(sampling, '_save_all_open_figures_to_run_results'):
        try:
            sampling._save_all_open_figures_to_run_results()
        except Exception as exc:
            print(f'WARNING: could not force-save open figures before dashboard creation: {exc}')

    dashboard_path = os.path.join(run_dir, f'{run_stem}_summary_dashboard.pdf')
    dashboard_title = f'HW4 Helmholtz HLSI dashboard: {run_name}'
    dash = DashboardPDF(dashboard_path, title=dashboard_title)

    created = datetime.now().isoformat(timespec='seconds')
    cfg = dashboard_config or {}
    lines = [
        dashboard_title,
        f'Created: {created}',
        '',
        'This dashboard contains the two canonical saved-results tables plus every PNG diagnostic plot saved in this experiment run directory.',
        'Expected diagnostic plots: ESS vs t, field visualization, PCA histograms, scattered-field recovery, boundary trace, and GN curvature spectrum.',
        'Random progress output from precomputation / Hessian batching is intentionally excluded.',
        '',
        f'run_results_dir = {run_dir}',
        f'run_name = {run_name}',
        f'source_mode = {source_mode}',
        f'k0 = {k0:g}',
    ]
    for key, val in cfg.items():
        if key in {'SAMPLER_CONFIGS'}:
            continue
        lines.append(f'{key} = {_dashboard_format_cell(val, max_len=120)}')
    if extra_lines:
        lines.append('')
        lines.extend([str(x) for x in extra_lines])

    dash.add_text_page(dashboard_title, lines)
    dash.add_results_tables(results_df, results_runinfo_df)
    if DASHBOARD_SHOW_FIGURES:
        dash.add_run_results_png_figures(run_dir)
    dash.close()
    print(f'Saved summary dashboard to {dashboard_path}')
    return dashboard_path


################################################################################
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

################################################################################

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

# =============================================================================
# User-facing configuration
# =============================================================================
seed = 42

# Masked Cartesian discretization of the unit disk.  N=35 is intentionally
# moderate because HLSI needs many forward/Jacobian/Hessian evaluations.
N = 35
DOMAIN_RADIUS = 1.0

# KL latent parameterization of m(x).  The HW truth is discontinuous, so the
# latent truth below is a least-squares projection.  Field metrics are computed
# against the actual discontinuous truth, not only this projected alpha.
num_truncated_series = 48
ell = 0.32
sigma_prior = 0.22

# HW Problem II defaults.
HELMHOLTZ_K_DEFAULT = 5.0
NOISE_REL = 0.01

# The assignment PDF does not specify the actual point-source coordinates; the
# starter code may. These are outside the unit disk and match the simple defaults
# used in the earlier SD notebook adaptation.
SOURCE_POINTS = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
N_HW_SOURCES = 3
SOURCE_SEED = 42

# Boundary receiver subsampling.  Use <= number of boundary grid nodes.  Remaining
# boundary nodes become held-out receivers for posterior predictive diagnostics.
N_RECEIVERS = 96
N_HOLDOUT_RECEIVERS = None

# Dense real Helmholtz solves can be numerically close to resonance on a coarse
# graph-Laplacian disk. Keep this at 0.0 for the closest HW match. If JAX reports
# singular solves, try 1e-8 or 1e-6.
HELMHOLTZ_SOLVE_SHIFT = 0.0

# Sampler controls. Matched to the pasted active CE-HLSI bootstrap run.
ACTIVE_DIM = num_truncated_series
DEFAULT_N_GEN = 7000
N_REF = 7000
BUILD_GNL_BANKS = False
# Match the pasted active-run script: pipeline precomputes PoU-capable banks even
# though the active CE bootstrap chain uses L / None weights.
INCLUDE_POU = True
PLOT_NORMALIZER = 'best'
HESS_MIN = 1e-10
HESS_MAX = 1e10
GNL_PILOT_N = 512
GNL_STIFF_LAMBDA_CUT = HESS_MAX
HELDOUT_BATCH_SIZE = 1

# By default this runs both the literal HW single-superposed-source case and the
# independent-source variant from Problem II.3.  Comment out a line if you only
# want one experiment.
RUN_EXPERIMENTS = [
    ('hw_superposed_k5', 'superposed', 5.0),
    ('hw_independent_k5', 'separate', 5.0),
    # ('hw_superposed_k10', 'superposed', 10.0),
]


# =============================================================================
# Geometry, basis, and finite-difference unit-disk operators
# =============================================================================

def _build_disk_geometry(n=N, radius=DOMAIN_RADIUS, n_receivers=N_RECEIVERS,
                         n_holdout_receivers=N_HOLDOUT_RECEIVERS):
    x = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(x, x, indexing='xy')
    h = float(x[1] - x[0])
    mask = (X ** 2 + Y ** 2) <= (radius + 1e-12) ** 2

    active_rows, active_cols = np.where(mask)
    coords = np.column_stack([X[mask], Y[mask]])
    n_active = coords.shape[0]

    active_id = -np.ones((n, n), dtype=np.int64)
    active_id[active_rows, active_cols] = np.arange(n_active, dtype=np.int64)

    # Graph finite-volume Laplacian with omitted exterior neighbors = zero flux
    # across the masked boundary, i.e. a simple homogeneous Neumann approximation.
    edges_i, edges_j = [], []
    for r, c in zip(active_rows, active_cols):
        idx = active_id[r, c]
        for dr, dc in ((1, 0), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and active_id[rr, cc] >= 0:
                jdx = active_id[rr, cc]
                edges_i.append(idx)
                edges_j.append(jdx)

    # Boundary active nodes are those with at least one missing coordinate neighbor.
    boundary = np.zeros(n_active, dtype=bool)
    for r, c in zip(active_rows, active_cols):
        idx = active_id[r, c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= n or cc < 0 or cc >= n or active_id[rr, cc] < 0:
                boundary[idx] = True
                break

    boundary_idx = np.where(boundary)[0]
    angles = np.arctan2(coords[boundary_idx, 1], coords[boundary_idx, 0])
    boundary_idx = boundary_idx[np.argsort(angles)]
    n_boundary = boundary_idx.size

    n_receivers = min(int(n_receivers), n_boundary)
    receiver_pos = np.round(np.arange(n_receivers) * n_boundary / n_receivers).astype(int)
    receiver_pos = np.unique(np.clip(receiver_pos, 0, n_boundary - 1))
    receiver_idx = boundary_idx[receiver_pos]

    remaining_pos = np.setdiff1d(np.arange(n_boundary), receiver_pos)
    if n_holdout_receivers is None:
        holdout_pos = remaining_pos
    else:
        holdout_pos = remaining_pos[:min(int(n_holdout_receivers), remaining_pos.size)]
    holdout_idx = boundary_idx[holdout_pos]

    def active_to_grid(active_values, outside_value=0.0):
        arr = np.full((n, n), outside_value, dtype=np.float64)
        arr[active_rows, active_cols] = np.asarray(active_values, dtype=np.float64).reshape(-1)
        return arr

    def active_indices_to_grid_rc(active_indices):
        active_indices = np.asarray(active_indices, dtype=int)
        return active_rows[active_indices], active_cols[active_indices]

    return dict(
        n=n,
        radius=radius,
        x=x,
        X=X,
        Y=Y,
        h=h,
        mask=mask,
        active_rows=active_rows,
        active_cols=active_cols,
        active_id=active_id,
        coords=coords,
        n_active=n_active,
        edges_i=np.asarray(edges_i, dtype=np.int64),
        edges_j=np.asarray(edges_j, dtype=np.int64),
        boundary_idx=boundary_idx,
        receiver_idx=receiver_idx,
        holdout_idx=holdout_idx,
        active_to_grid=active_to_grid,
        active_indices_to_grid_rc=active_indices_to_grid_rc,
    )


def _assemble_negative_laplacian_jax(geom):
    n_active = int(geom['n_active'])
    h = float(geom['h'])
    weight = 1.0 / (h * h)
    A = jnp.zeros((n_active, n_active), dtype=jnp.float64)
    i = jnp.asarray(geom['edges_i'], dtype=jnp.int32)
    j = jnp.asarray(geom['edges_j'], dtype=jnp.int32)
    A = A.at[i, i].add(weight)
    A = A.at[j, j].add(weight)
    A = A.at[i, j].add(-weight)
    A = A.at[j, i].add(-weight)
    return A


def _make_kl_basis(geom):
    os.makedirs('data', exist_ok=True)
    coords = geom['coords']
    dists = cdist(coords, coords)
    C = sigma_prior ** 2 * np.exp(-dists / ell)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    q_max = max(num_truncated_series, min(100, eigvecs.shape[1]))
    basis_modes = eigvecs[:, :q_max] * np.sqrt(np.maximum(eigvals[:q_max], 0.0))
    basis_truncated = basis_modes[:, :num_truncated_series]
    np.savetxt('data/HW4_Helmholtz_Disk_Basis_Modes.csv', basis_modes, delimiter=',')
    pd.DataFrame(basis_truncated).to_csv('data/HW4_Helmholtz_Disk_Basis.csv', index=False, header=False)
    return basis_modes, basis_truncated, eigvals


def _true_m_active(geom):
    x = geom['coords'][:, 0]
    y = geom['coords'][:, 1]
    inside = np.sqrt((x - 0.1) ** 2 + 2.0 * (y + 0.2) ** 2) < 0.5
    return 0.2 * inside.astype(np.float64)


def _project_to_basis(basis, field_active):
    alpha, *_ = np.linalg.lstsq(np.asarray(basis), np.asarray(field_active), rcond=None)
    return alpha.astype(np.float64)


def _incident_fields_active(geom, k0, mode):
    rng = np.random.RandomState(SOURCE_SEED)
    coeffs = rng.normal(0.0, 1.0, size=N_HW_SOURCES)
    coords = geom['coords']
    fields = []
    for (xs, ys), ej in zip(SOURCE_POINTS, coeffs):
        r = np.sqrt((coords[:, 0] - xs) ** 2 + (coords[:, 1] - ys) ** 2)
        r = np.maximum(r, 1e-12)
        fields.append(-ej * np.cos(k0 * r) / (4.0 * np.pi * r))
    fields = np.stack(fields, axis=1)  # [n_active, 3]
    if mode == 'superposed':
        return fields.sum(axis=1, keepdims=True), coeffs
    if mode == 'separate':
        return fields, coeffs
    raise ValueError("source mode must be 'superposed' or 'separate'")


def _flatten_by_source(meas):
    # meas shape [n_receivers, n_patterns]. Return source-major vector.
    return jnp.ravel(jnp.transpose(meas))


def _unpack_measurement_vector(y, n_patterns, n_receivers):
    return np.asarray(y).reshape(n_patterns, n_receivers)


# =============================================================================
# Experiment construction and execution
# =============================================================================



def make_sampler_configs(include_pou=INCLUDE_POU):
    """Active sampler config matched to the pasted Helmholtz run script.

    The pasted version intentionally comments out the broad baseline sweep and
    keeps only the CE-HLSI self-bootstrap chain active:
        1. CE-HLSI from the original prior/reference bank;
        2. CE-HLSI initialized from the CE-HLSI output bank;
        3. CE-HLSI initialized from the second-stage CE-HLSI output bank.

    Keep the method labels exactly aligned with the pasted script so downstream
    run-info tables, dashboards, and comparisons use the same names.
    """
    return OrderedDict([
    ('CE-HLSI1', {'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI2', {'ref_source': 'CE-HLSI1', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'include_results': False}),
    ('CE-HLSI3', {'ref_source': 'CE-HLSI2', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('CE-HLSI4', {'ref_source': 'CE-HLSI3', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'include_results': False}),
    ('CE-HLSI5', {'ref_source': 'CE-HLSI4', 'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),

    ('DRC-CE-HLSI1', {'init': 'CE-HLSI', 'init_weights': 'None', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True}),
    ('DRC-CE-HLSI2', {'ref_source': 'DRC-CE-HLSI1', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'include_results': False, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
    ('DRC-CE-HLSI3', {'ref_source': 'DRC-CE-HLSI2', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
    ('DRC-CE-HLSI4', {'ref_source': 'DRC-CE-HLSI3', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'include_results': False, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
    ('DRC-CE-HLSI5', {'ref_source': 'DRC-CE-HLSI3', 'init': 'CE-HLSI', 'init_weights': 'DRC', 'transition_w': 'ou', 'init_steps': 200, 'mala_steps': 0, 'mala_burnin': 0, 'log_mean_ess': True, 'drc_pf_steps': 32, 'drc_div_probes': 1, 'drc_eval_batch_size': 32, 'drc_clip': 20.0, 'drc_temperature': 1.0, 'drc_fd_eps': 1e-3}),
    ])



def run_hw_hlsi_experiment(run_name, source_mode, k0):
    print('\n' + '=' * 96)
    print(f'Running HW4 Helmholtz HLSI experiment: {run_name} | mode={source_mode} | k0={k0:g}')
    print('=' * 96)

    geom = _build_disk_geometry()
    _, basis_truncated, eigvals = _make_kl_basis(geom)
    true_m = _true_m_active(geom)
    alpha_true_np = _project_to_basis(basis_truncated, true_m)
    projected_true_m = basis_truncated @ alpha_true_np
    incident_np, source_coeffs = _incident_fields_active(geom, k0, source_mode)

    n_active = int(geom['n_active'])
    n_patterns = int(incident_np.shape[1])
    receiver_idx = np.asarray(geom['receiver_idx'], dtype=np.int64)
    holdout_idx = np.asarray(geom['holdout_idx'], dtype=np.int64)
    n_receivers = int(receiver_idx.size)
    n_holdout = int(holdout_idx.size)

    Basis = jnp.asarray(basis_truncated, dtype=jnp.float64)
    incident_jax = jnp.asarray(incident_np, dtype=jnp.float64)
    receiver_idx_jax = jnp.asarray(receiver_idx, dtype=jnp.int32)
    holdout_idx_jax = jnp.asarray(holdout_idx, dtype=jnp.int32)
    NEG_LAPLACIAN = _assemble_negative_laplacian_jax(geom)
    IDENTITY = jnp.eye(n_active, dtype=jnp.float64)

    @jax.jit
    def alpha_to_m(alpha):
        return Basis @ alpha

    @jax.jit
    def solve_fields_from_m(m_active):
        q = jnp.tanh(m_active)
        A = NEG_LAPLACIAN - (k0 ** 2) * jnp.diag(1.0 - q) + HELMHOLTZ_SOLVE_SHIFT * IDENTITY
        rhs = (k0 ** 2) * q[:, None] * incident_jax
        return jnp.linalg.solve(A, rhs)  # scattered field, shape [n_active, n_patterns]

    @jax.jit
    def solve_forward(alpha):
        fields = solve_fields_from_m(alpha_to_m(alpha))
        meas = fields[receiver_idx_jax, :]
        return _flatten_by_source(meas)

    @jax.jit
    def solve_forward_from_m(m_active):
        fields = solve_fields_from_m(m_active)
        meas = fields[receiver_idx_jax, :]
        return _flatten_by_source(meas)

    @jax.jit
    def solve_forward_holdout(alpha):
        fields = solve_fields_from_m(alpha_to_m(alpha))
        meas = fields[holdout_idx_jax, :]
        return _flatten_by_source(meas)

    batch_solve_forward_holdout = jax.jit(jax.vmap(solve_forward_holdout))

    @jax.jit
    def solve_fields_from_alpha(alpha):
        return solve_fields_from_m(alpha_to_m(alpha))

    # Synthetic data from the actual HW discontinuous m field, not from the KL projection.
    y_clean_np = np.asarray(solve_forward_from_m(jnp.asarray(true_m, dtype=jnp.float64)))
    noise_std = float(NOISE_REL * max(np.max(np.abs(y_clean_np)), 1e-12))
    rng = np.random.RandomState(seed)
    y_obs_np = y_clean_np + rng.normal(0.0, noise_std, size=y_clean_np.shape)

    if n_holdout > 0:
        y_clean_holdout_np = np.asarray(
            _flatten_by_source(solve_fields_from_m(jnp.asarray(true_m, dtype=jnp.float64))[holdout_idx_jax, :])
        )
        y_holdout_obs_np = y_clean_holdout_np + rng.normal(0.0, noise_std, size=y_clean_holdout_np.shape)
    else:
        y_clean_holdout_np = np.zeros((0,), dtype=np.float64)
        y_holdout_obs_np = np.zeros((0,), dtype=np.float64)

    os.makedirs('data', exist_ok=True)
    pd.DataFrame(receiver_idx).to_csv(f'data/{run_name}_receiver_active_indices.csv', index=False, header=False)
    pd.DataFrame(holdout_idx).to_csv(f'data/{run_name}_holdout_active_indices.csv', index=False, header=False)

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
    init_run_results(run_name)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prior_model = GaussianPrior(dim=ACTIVE_DIM)
    lik_model, lik_aux = make_physics_likelihood(
        solve_forward,
        y_obs_np,
        noise_std,
        use_gauss_newton_hessian=True,
        log_batch_size=50,
        grad_batch_size=25,
        hess_batch_size=1,
    )
    posterior_score_fn = make_posterior_score_fn(lik_model)

    SAMPLER_CONFIGS = make_sampler_configs(include_pou=INCLUDE_POU)
    pipeline = run_standard_sampler_pipeline(
        prior_model,
        lik_model,
        SAMPLER_CONFIGS,
        n_ref=N_REF,
        build_gnl_banks=BUILD_GNL_BANKS,
        compute_pou=INCLUDE_POU,
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

    def reconstruct_m_active(alpha):
        alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)[:ACTIVE_DIM]
        return np.asarray(basis_truncated @ alpha, dtype=np.float64)

    def reconstruct_m_grid(alpha):
        return geom['active_to_grid'](reconstruct_m_active(alpha))

    def reconstruct_m_grid_batch(alpha_batch):
        arr = np.asarray(alpha_batch, dtype=np.float64)
        if arr.ndim == 1:
            return reconstruct_m_grid(arr)
        return np.stack([reconstruct_m_grid(a[:ACTIVE_DIM]) for a in arr], axis=0)

    true_m_grid = geom['active_to_grid'](true_m)
    projected_true_m_grid = geom['active_to_grid'](projected_true_m)

    mean_fields_active, metrics = compute_field_summary_metrics(
        samples,
        metrics,
        alpha_true_np,
        true_m,
        reconstruct_m_active,
        forward_eval_fn=lambda a: np.asarray(solve_forward(jnp.asarray(a, dtype=jnp.float64))),
        y_ref_np=y_clean_np,
        display_names=display_names,
        min_valid=10,
        d_lat=ACTIVE_DIM,
    )

    # Add contrast metrics for q=tanh(m), since that is the actual PDE coefficient.
    q_true = np.tanh(true_m)
    q_norm = np.linalg.norm(q_true) + 1e-12
    for label, m_mean in mean_fields_active.items():
        q_mean = np.tanh(m_mean)
        metrics.setdefault(label, {})
        metrics[label]['RMSE_tanh_m'] = float(np.sqrt(np.mean((q_mean - q_true) ** 2)))
        metrics[label]['RelL2_tanh_m'] = float(np.linalg.norm(q_mean - q_true) / q_norm)
        metrics[label]['InverseRelL2_percent'] = 100.0 * float(metrics[label].get('RelL2_field', np.nan))
        metrics[label]['TanhMRelL2_percent'] = 100.0 * float(metrics[label].get('RelL2_tanh_m', np.nan))

    if y_holdout_obs_np.size > 0:
        try:
            metrics = compute_heldout_predictive_metrics(
                samples,
                metrics,
                heldout_forward_eval_fn=lambda a: np.asarray(solve_forward_holdout(jnp.asarray(a, dtype=jnp.float64))),
                batched_forward_eval_fn=lambda a_batch: np.asarray(
                    batch_solve_forward_holdout(jnp.asarray(a_batch, dtype=jnp.float64))
                ),
                batched_forward_eval_batch_size=HELDOUT_BATCH_SIZE,
                y_holdout_obs_np=y_holdout_obs_np,
                noise_std=noise_std,
                display_names=display_names,
                min_valid=10,
            )
        except Exception as exc:
            print(f"WARNING: held-out predictive metrics failed and will be skipped: {exc}")

    results_df, results_runinfo_df, results_df_path, results_runinfo_df_path = save_results_tables(
        metrics,
        sampler_run_info,
        n_ref=N_REF,
        target_name=f'HW4 Helmholtz unit disk ({source_mode}, k0={k0:g})',
        display_names=display_names,
        reference_name=reference_title,
    )

    print_full_metrics_to_stdout(
        results_df,
        results_runinfo_df,
        title=f'{run_name} ({source_mode}, k0={k0:g})',
    )

    mean_fields_grid = {label: geom['active_to_grid'](field) for label, field in mean_fields_active.items()}
    plot_normalizer_key = resolve_plot_normalizer(
        PLOT_NORMALIZER,
        list(mean_fields_grid.keys()),
        display_names=display_names,
        metrics_dict=metrics,
        fallback=reference_key,
    )
    plot_normalizer_title = display_names.get(plot_normalizer_key, plot_normalizer_key)
    print(f"Plot normalizer: {plot_normalizer_title} ({plot_normalizer_key})")

    receiver_rows, receiver_cols = geom['active_indices_to_grid_rc'](receiver_idx)
    holdout_rows, holdout_cols = geom['active_indices_to_grid_rc'](holdout_idx)

    def _overlay_disk(ax):
        ax.contour(geom['mask'].astype(float), levels=[0.5], colors='white', linewidths=1.2)
        ax.scatter(receiver_cols, receiver_rows, c='lime', s=11, marker='s', alpha=0.78)
        if holdout_idx.size > 0:
            ax.scatter(holdout_cols, holdout_rows, c='cyan', s=6, marker='.', alpha=0.45)

    plot_field_reconstruction_grid(
        samples,
        mean_fields_grid,
        reconstruct_m_grid_batch,
        display_names=display_names,
        true_field=true_m_grid,
        plot_normalizer_key=plot_normalizer_key,
        reference_bottom_panel=projected_true_m_grid,
        reference_bottom_title='KL projection\nof HW truth',
        field_cmap='viridis',
        sample_cmap='viridis',
        bottom_cmap='viridis',
        overlay_reference_fn=_overlay_disk,
        overlay_method_fn=_overlay_disk,
        suptitle=f'HW4 Problem II Helmholtz HLSI ({source_mode}, k0={k0:g}, d={ACTIVE_DIM})',
        field_name='m(x)',
    )

    try:
        plot_pca_histograms(
            samples,
            alpha_true=alpha_true_np,
            display_names=display_names,
            normalizer=PLOT_NORMALIZER,
            metrics_dict=metrics,
            fallback_key=reference_key,
        )
    except Exception as exc:
        print(f"WARNING: PCA histograms failed and will be skipped: {exc}")

    # Wavefield panels: scattered field for pattern 0.  This is real-valued HW u.
    methods_to_plot = [label for label in samples.keys() if label in mean_fields_active]
    n_cols = len(methods_to_plot) + 1
    true_fields = np.asarray(solve_fields_from_m(jnp.asarray(true_m, dtype=jnp.float64)))
    true_u0 = geom['active_to_grid'](true_fields[:, 0])
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(4.1 * n_cols, 4.2))
    if n_cols == 1:
        axes2 = [axes2]
    vabs = max(1e-12, float(np.max(np.abs(true_u0))))
    im = axes2[0].imshow(true_u0, cmap='RdBu_r', origin='lower', vmin=-vabs, vmax=vabs)
    _overlay_disk(axes2[0])
    axes2[0].set_title('Ground truth\nscattered field u')
    axes2[0].axis('off')
    plt.colorbar(im, ax=axes2[0], fraction=0.046, pad=0.04)
    for col, label in enumerate(methods_to_plot, start=1):
        samps_clean = get_valid_samples(samples[label])
        if samps_clean.shape[0] < 10:
            axes2[col].axis('off')
            continue
        mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
        u_mean = np.asarray(solve_fields_from_alpha(jnp.asarray(mean_lat, dtype=jnp.float64)))[:, 0]
        u_grid = geom['active_to_grid'](u_mean)
        axes2[col].imshow(u_grid, cmap='RdBu_r', origin='lower', vmin=-vabs, vmax=vabs)
        _overlay_disk(axes2[col])
        axes2[col].set_title(f"{display_names.get(label, label)}\nu pattern 0")
        axes2[col].axis('off')
    fig2.suptitle(f'Scattered field recovery, pattern 0 ({source_mode}, k0={k0:g})')
    plt.tight_layout()
    plt.show()

    # Boundary trace plot for pattern 0.
    theta_receivers = np.arctan2(geom['coords'][receiver_idx, 1], geom['coords'][receiver_idx, 0])
    order = np.argsort(theta_receivers)
    theta_sorted = theta_receivers[order]
    y_true_by_src = _unpack_measurement_vector(y_clean_np, n_patterns, n_receivers)
    y_obs_by_src = _unpack_measurement_vector(y_obs_np, n_patterns, n_receivers)
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax3a.plot(theta_sorted, y_true_by_src[0, order], 'k-', linewidth=2.2, label='Clean')
    ax3a.scatter(theta_sorted, y_obs_by_src[0, order], c='tab:red', s=12, alpha=0.5, label='Noisy obs')
    ax3b.axhline(0.0, color='0.35', linestyle='--', linewidth=1.0)
    trace_methods = methods_to_plot[:5]
    for label in trace_methods:
        samps_clean = get_valid_samples(samples[label])
        if samps_clean.shape[0] < 10:
            continue
        mean_lat = np.mean(samps_clean, axis=0)[:ACTIVE_DIM]
        y_pred = np.asarray(solve_forward(jnp.asarray(mean_lat, dtype=jnp.float64)))
        y_pred_by_src = _unpack_measurement_vector(y_pred, n_patterns, n_receivers)
        pretty = display_names.get(label, label)
        ax3a.plot(theta_sorted, y_pred_by_src[0, order], linewidth=1.6, label=pretty)
        ax3b.plot(theta_sorted, np.abs(y_pred_by_src[0, order] - y_true_by_src[0, order]), linewidth=1.4, label=pretty)
    ax3a.set_title('Boundary scattered-field trace, pattern 0')
    ax3a.set_ylabel('u on boundary')
    ax3b.set_ylabel('|prediction - clean|')
    ax3b.set_xlabel('boundary angle')
    for ax in (ax3a, ax3b):
        ax.grid(True, alpha=0.25)
    ax3a.legend(ncol=min(4, len(trace_methods) + 2), fontsize=9)
    plt.tight_layout()
    plt.show()

    # GN curvature spectrum at the projected truth and at the selected normalizer.
    try:
        fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5))
        true_curv = np.linalg.eigvalsh(-np.asarray(
            lik_aux['hess_lik_gn_jax'](jnp.asarray(alpha_true_np), jnp.asarray(y_obs_np), noise_std)
        ))
        true_curv = np.clip(np.sort(true_curv)[::-1], 1e-16, None)
        ax4.semilogy(np.arange(1, true_curv.size + 1), true_curv, marker='o', linewidth=2, label='Projected-truth GN spectrum')
        if plot_normalizer_key in metrics and 'mean_latent' in metrics[plot_normalizer_key]:
            ref_alpha = np.asarray(metrics[plot_normalizer_key]['mean_latent'])[:ACTIVE_DIM]
            ref_curv = np.linalg.eigvalsh(-np.asarray(
                lik_aux['hess_lik_gn_jax'](jnp.asarray(ref_alpha), jnp.asarray(y_obs_np), noise_std)
            ))
            ref_curv = np.clip(np.sort(ref_curv)[::-1], 1e-16, None)
            ax4.semilogy(np.arange(1, ref_curv.size + 1), ref_curv, marker='s', linewidth=2,
                         label=f'{display_names.get(plot_normalizer_key, plot_normalizer_key)} GN spectrum')
        ax4.axhline(HESS_MIN, linestyle='--', linewidth=1.2, label='HESS_MIN')
        ax4.axhline(HESS_MAX, linestyle='--', linewidth=1.2, label='HESS_MAX')
        ax4.set_xlabel('Eigenvalue rank')
        ax4.set_ylabel('GN curvature magnitude')
        ax4.set_title(f'GN curvature spectrum ({source_mode}, k0={k0:g})')
        ax4.grid(True, which='both', alpha=0.25)
        ax4.legend(fontsize=9)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"WARNING: curvature spectrum failed and will be skipped: {exc}")

    dashboard_config = OrderedDict([
        ('seed', seed),
        ('N', N),
        ('n_active_disk_nodes', n_active),
        ('num_truncated_series', num_truncated_series),
        ('ell', ell),
        ('sigma_prior', sigma_prior),
        ('k0', k0),
        ('source_mode', source_mode),
        ('N_RECEIVERS', n_receivers),
        ('N_HOLDOUT_RECEIVERS', n_holdout),
        ('NOISE_REL', NOISE_REL),
        ('noise_std_absolute', noise_std),
        ('ACTIVE_DIM', ACTIVE_DIM),
        ('DEFAULT_N_GEN', DEFAULT_N_GEN),
        ('N_REF', N_REF),
        ('BUILD_GNL_BANKS', BUILD_GNL_BANKS),
        ('INCLUDE_POU', INCLUDE_POU),
        ('PLOT_NORMALIZER', PLOT_NORMALIZER),
        ('HESS_MIN', HESS_MIN),
        ('HESS_MAX', HESS_MAX),
        ('plot_normalizer_key', plot_normalizer_key),
        ('plot_normalizer_title', plot_normalizer_title),
    ])
    dashboard_path = create_summary_dashboard_for_current_run(
        run_name,
        source_mode,
        k0,
        results_df,
        results_runinfo_df,
        dashboard_config=dashboard_config,
        extra_lines=[
            f'metrics_csv = {results_df_path}',
            f'runinfo_csv = {results_runinfo_df_path}',
        ],
    )

    save_reproducibility_log(
        title=f'HW4 Helmholtz unit-disk HLSI run: {run_name}',
        config={
            'N': N,
            'n_active_disk_nodes': n_active,
            'num_truncated_series': num_truncated_series,
            'ell': ell,
            'sigma_prior': sigma_prior,
            'k0': k0,
            'source_mode': source_mode,
            'SOURCE_POINTS': SOURCE_POINTS,
            'source_coeffs': source_coeffs,
            'N_RECEIVERS': n_receivers,
            'N_HOLDOUT_RECEIVERS': n_holdout,
            'NOISE_REL': NOISE_REL,
            'noise_std_absolute': noise_std,
            'HELMHOLTZ_SOLVE_SHIFT': HELMHOLTZ_SOLVE_SHIFT,
            'ACTIVE_DIM': ACTIVE_DIM,
            'DEFAULT_N_GEN': DEFAULT_N_GEN,
            'N_REF': N_REF,
            'BUILD_GNL_BANKS': BUILD_GNL_BANKS,
            'INCLUDE_POU': INCLUDE_POU,
            'PLOT_NORMALIZER': PLOT_NORMALIZER,
            'HESS_MIN': HESS_MIN,
            'HESS_MAX': HESS_MAX,
            'GNL_PILOT_N': GNL_PILOT_N,
            'GNL_STIFF_LAMBDA_CUT': GNL_STIFF_LAMBDA_CUT,
            'SAMPLER_CONFIGS': SAMPLER_CONFIGS,
            'basis_top_eigvals': eigvals[:min(10, eigvals.size)],
            'alpha_true_projection_norm': float(np.linalg.norm(alpha_true_np)),
            'projected_truth_rel_l2_m': float(np.linalg.norm(projected_true_m - true_m) / (np.linalg.norm(true_m) + 1e-12)),
        },
        extra_sections={
            'saved_results_files': {
                'metrics_csv': results_df_path,
                'runinfo_csv': results_runinfo_df_path,
                'summary_dashboard_pdf': dashboard_path,
            },
            'summary_stats': {
                'reference_key': reference_key,
                'reference_title': reference_title,
                'plot_normalizer_key': plot_normalizer_key,
                'plot_normalizer_title': plot_normalizer_title,
                'num_methods_evaluated': len(results_df.columns),
                'num_methods_with_samples': len(samples),
                'num_methods_with_mean_fields': len(mean_fields_active),
                'num_methods_with_ess_logs': len(ess_logs),
            },
        },
    )

    zip_path = zip_run_results_dir()
    print(f'Summary dashboard PDF: {dashboard_path}')
    print(f'Run-results zip: {zip_path}')
    return zip_path


if __name__ == '__main__':
    for run_name, source_mode, k0 in RUN_EXPERIMENTS:
        run_hw_hlsi_experiment(run_name, source_mode, k0)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
