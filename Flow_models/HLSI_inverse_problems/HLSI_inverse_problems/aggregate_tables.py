#!/usr/bin/env python3
"""Aggregate inverse-problem run tables, including Darcy density Table 4 metrics.

The legacy path is preserved: run-level ``*_metrics.csv`` files with shape
``metric | sampler1 | sampler2 | ...`` are aggregated to mean/std summaries.

The current Darcy/Helmholtz/Navier-Stokes density scripts also write one-row
``*_drc_energy_<method>.csv`` files and a compact
``*_manuscript_density_table.csv``.  This script prefers the full DRC-energy
CSVs when present because they contain the central Darcy Table 4 metrics and
run-level uncertainty columns such as ``pointwise_nll_std``.  It falls back to
compact manuscript density tables when the full files are unavailable.
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import pandas as pd

PROBLEMS = [
    "advect_diff",
    "allen_cahn",
    "darcy_flow",
    "eit",
    "heat",
    "helmholtz",
    "navier_stokes",
    "poisson",
    "afwi",
]

NON_METRIC_COLUMNS = {
    "label",
    "method",
    "Method",
    "method_family",
    "plot_axis_mode",
    "residual_axis_mode",
    "affine_fit_scope",
    "residual_kind",
    "eval_bank_name",
    "baseline_runtime_seconds",
    "source_file",
}

# Table 4 in the current draft uses posterior-bulk / central Darcy density
# diagnostics.  Prefer the full central columns from *_drc_energy_*.csv.  The
# compact manuscript table aliases are fallbacks for old run directories.
TABLE4_METRICS = OrderedDict([
    ("spearman", {
        "display": "Spearman",
        "aliases": [
            "central_energy_neglogq_spearman",
            "Spearman",
            "energy_neglogq_spearman",
        ],
    }),
    ("central_slope", {
        "display": "Central slope",
        "aliases": [
            "central_affine_energy_slope",
            "Central slope",
            "Affine slope",
            "affine_energy_slope",
        ],
    }),
    ("central_r2", {
        "display": "Central R2",
        "aliases": [
            "central_affine_energy_r2",
            "Central R2",
            "Central $R^2$",
            "Affine $R^2$",
            "affine_energy_r2",
        ],
    }),
    ("central_rmse", {
        "display": "Central RMSE",
        "aliases": [
            "central_affine_energy_rmse",
            "Central RMSE",
            "Affine RMSE",
            "affine_energy_rmse",
        ],
    }),
    ("pointwise_nll", {
        "display": "Pointwise NLL",
        "aliases": [
            "pointwise_nll",
            "Pointwise NLL",
            "neglogq_mean",
        ],
        "std_aliases": [
            "pointwise_nll_std",
            "neglogq_std",
            "Pointwise NLL std",
        ],
    }),
])

METHOD_ORDER = ["Tweedie", "Scalar Blend", "LFGI-GN", "MAP-Laplace"]


def sanitize(name):
    """Make a name safe for CSV column names while remaining readable."""
    return (
        str(name).strip()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("–", "_")
        .replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("^", "")
    )


def canonical_method_name(name):
    """Normalize historical/internal density labels to manuscript-facing names."""
    s = str(name).strip()
    low = s.lower().replace("_", "-")
    if low.startswith("dens-"):
        low = low[5:]
    if "tweedie" in low:
        return "Tweedie"
    if "scalar" in low and "blend" in low:
        return "Scalar Blend"
    if "ce-hlsi" in low or "hlsi" in low or "lfgi" in low:
        return "LFGI-GN"
    if "map" in low and "laplace" in low:
        return "MAP-Laplace"
    # Strip common density suffixes while preserving unknown method names.
    s = s.replace("DENS-", "").replace(" PF", "")
    return s


def method_sort_key(method):
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), str(method))


def format_number(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1e4 or (0.0 < abs(x) < 1e-3):
        return "{:.3e}".format(x)
    return "{:.4g}".format(x)


def format_mean_std(mean, std):
    if pd.isna(mean):
        return ""
    std = 0.0 if pd.isna(std) else float(std)
    return "{} ± {}".format(format_number(mean), format_number(std))


def find_metric_csvs(problem_dir):
    """Recursively find legacy latent/posterior metrics CSV files."""
    run_results = problem_dir / "run_results"
    if not run_results.exists():
        return []
    csvs = []
    for p in run_results.rglob("*_metrics.csv"):
        if not p.is_file():
            continue
        name = p.name
        if "meta_metrics" in name or "manuscript_density" in name or "drc_energy" in name:
            continue
        csvs.append(p)
    return sorted(csvs)


def find_density_csvs(problem_dir):
    """Find full DRC-energy density CSVs and compact manuscript density tables."""
    run_results = problem_dir / "run_results"
    if not run_results.exists():
        return [], []
    raw = []
    compact = []
    for p in run_results.rglob("*.csv"):
        if not p.is_file():
            continue
        name = p.name
        if "meta_" in name:
            continue
        if "_drc_energy_" in name:
            raw.append(p)
        elif name.endswith("_manuscript_density_table.csv"):
            compact.append(p)
    return sorted(raw), sorted(compact)


def load_and_normalize_metrics(csv_path):
    """Load one legacy metrics CSV and return metric | sampler | value rows."""
    df = pd.read_csv(csv_path)
    if "metric" not in df.columns:
        raise ValueError("{} is missing a 'metric' column".format(csv_path))

    long_df = df.melt(id_vars="metric", var_name="sampler", value_name="value")
    long_df["source_file"] = str(csv_path)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["metric", "sampler", "value"])
    return long_df


def aggregate_long_table(combined, group_cols):
    stats = (
        combined.groupby(group_cols)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs"})
    )
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def aggregate_legacy_metrics(problem_dir, output_dir):
    problem_name = problem_dir.name
    csv_paths = find_metric_csvs(problem_dir)
    if not csv_paths:
        return {"n_files": 0}

    long_tables = [load_and_normalize_metrics(path) for path in csv_paths]
    combined = pd.concat(long_tables, ignore_index=True)
    stats = aggregate_long_table(combined, ["metric", "sampler"])

    mean_wide = stats.pivot(index="metric", columns="sampler", values="mean")
    std_wide = stats.pivot(index="metric", columns="sampler", values="std")
    n_wide = stats.pivot(index="metric", columns="sampler", values="n_runs")
    sampler_order = list(mean_wide.columns)

    numeric_cols = {}
    formatted_cols = {}
    for sampler in sampler_order:
        safe = sanitize(sampler)
        numeric_cols["{}__mean".format(safe)] = mean_wide[sampler]
        numeric_cols["{}__std".format(safe)] = std_wide[sampler]
        numeric_cols["{}__n_runs".format(safe)] = n_wide[sampler]
        formatted_cols[sampler] = [
            format_mean_std(m, s) for m, s in zip(mean_wide[sampler].tolist(), std_wide[sampler].tolist())
        ]

    numeric_out = pd.DataFrame(numeric_cols, index=mean_wide.index).reset_index()
    formatted_out = pd.DataFrame(formatted_cols, index=mean_wide.index).reset_index()

    unique_counts = sorted(set(int(x) for x in stats["n_runs"].dropna().tolist()))
    if len(unique_counts) == 1:
        formatted_out.insert(1, "n_runs", unique_counts[0])

    tidy_out = stats.sort_values(["metric", "sampler"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = output_dir / "{}_meta_metrics_numeric.csv".format(problem_name)
    formatted_path = output_dir / "{}_meta_metrics_formatted.csv".format(problem_name)
    tidy_path = output_dir / "{}_meta_metrics_tidy.csv".format(problem_name)

    numeric_out.to_csv(str(numeric_path), index=False)
    formatted_out.to_csv(str(formatted_path), index=False)
    tidy_out.to_csv(str(tidy_path), index=False)

    return {
        "n_files": len(csv_paths),
        "numeric": numeric_path,
        "formatted": formatted_path,
        "tidy": tidy_path,
    }


def row_metric_value(row, aliases):
    for alias in aliases:
        if alias in row.index:
            val = pd.to_numeric(pd.Series([row[alias]]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val), alias
    return None, None


def load_density_raw_table(csv_path):
    """Load one *_drc_energy_<method>.csv into method | metric | value rows."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        method_source = row.get("label", row.get("Method", csv_path.stem))
        method = canonical_method_name(method_source)
        # Full metric dump: keep all numeric scalar columns except known metadata.
        for col in df.columns:
            if col in NON_METRIC_COLUMNS or str(col).endswith("_std"):
                continue
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            within_std = None
            std_col = "{}_std".format(col)
            if std_col in df.columns:
                std_val = pd.to_numeric(pd.Series([row[std_col]]), errors="coerce").iloc[0]
                if pd.notna(std_val):
                    within_std = float(std_val)
            rows.append({
                "method": method,
                "metric": str(col),
                "value": float(value),
                "within_run_std": within_std,
                "source_metric": str(col),
                "source_file": str(csv_path),
            })
        # Also guarantee the Table 4 alias metrics exist even if the raw column
        # names differ across script versions.
        for canonical, spec in TABLE4_METRICS.items():
            value, source_metric = row_metric_value(row, spec["aliases"])
            if value is None:
                continue
            within_std = None
            for std_alias in spec.get("std_aliases", []):
                if std_alias in row.index:
                    std_val = pd.to_numeric(pd.Series([row[std_alias]]), errors="coerce").iloc[0]
                    if pd.notna(std_val):
                        within_std = float(std_val)
                        break
            # Avoid double-counting when the raw file already uses the
            # canonical metric name (notably pointwise_nll).
            if str(source_metric) == canonical:
                continue
            rows.append({
                "method": method,
                "metric": canonical,
                "value": float(value),
                "within_run_std": within_std,
                "source_metric": str(source_metric),
                "source_file": str(csv_path),
            })
    return pd.DataFrame(rows)


def compact_column_to_metric(column):
    col = str(column).strip()
    for canonical, spec in TABLE4_METRICS.items():
        if col in spec["aliases"]:
            return canonical
    return col


def load_density_compact_table(csv_path):
    """Load a compact *_manuscript_density_table.csv as a fallback."""
    df = pd.read_csv(csv_path)
    if df.empty or "Method" not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        method = canonical_method_name(row.get("Method", ""))
        std_alias_names = {
            alias
            for spec in TABLE4_METRICS.values()
            for alias in spec.get("std_aliases", [])
        }
        for col in df.columns:
            if col == "Method" or col in std_alias_names:
                continue
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            metric = compact_column_to_metric(col)
            within_std = None
            for spec in TABLE4_METRICS.values():
                if metric == compact_column_to_metric(spec["display"]):
                    for std_alias in spec.get("std_aliases", []):
                        if std_alias in df.columns:
                            std_val = pd.to_numeric(pd.Series([row[std_alias]]), errors="coerce").iloc[0]
                            if pd.notna(std_val):
                                within_std = float(std_val)
                                break
                    break
            rows.append({
                "method": method,
                "metric": metric,
                "value": float(value),
                "within_run_std": within_std,
                "source_metric": str(col),
                "source_file": str(csv_path),
            })
    return pd.DataFrame(rows)


def aggregate_density_problem(problem_dir, output_dir):
    problem_name = problem_dir.name
    raw_paths, compact_paths = find_density_csvs(problem_dir)
    source_kind = "raw_drc_energy" if raw_paths else "compact_manuscript"
    paths = raw_paths if raw_paths else compact_paths
    if not paths:
        return {"n_files": 0, "source_kind": source_kind}

    loaders = load_density_raw_table if raw_paths else load_density_compact_table
    frames = [loaders(path) for path in paths]
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return {"n_files": 0, "source_kind": source_kind}
    combined = pd.concat(frames, ignore_index=True)

    stats = aggregate_long_table(combined, ["method", "metric"])
    if "within_run_std" in combined.columns:
        wstats = (
            combined.dropna(subset=["within_run_std"])
            .groupby(["method", "metric"])["within_run_std"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={
                "mean": "within_run_std_mean",
                "std": "within_run_std_std",
                "count": "within_run_std_count",
            })
        )
        stats = stats.merge(wstats, on=["method", "metric"], how="left")
        stats["within_run_std_std"] = stats["within_run_std_std"].fillna(0.0)
    else:
        stats["within_run_std_mean"] = pd.NA
        stats["within_run_std_std"] = pd.NA
        stats["within_run_std_count"] = 0

    stats = stats.sort_values(["method", "metric"], key=lambda col: col.map(method_sort_key) if col.name == "method" else col)

    # Table 4 matrix, using canonical metrics and manuscript method order.
    method_names = sorted(stats["method"].unique().tolist(), key=method_sort_key)
    numeric_rows = []
    formatted_rows = []
    for method in method_names:
        numeric_row = OrderedDict([("Method", method)])
        formatted_row = OrderedDict([("Method", method)])
        run_counts = []
        for canonical, spec in TABLE4_METRICS.items():
            # Use canonical rows when present, otherwise fall back to the first
            # matching source metric in the aggregate stats.
            metric_rows = stats[(stats["method"] == method) & (stats["metric"] == canonical)]
            if metric_rows.empty:
                aliases = spec["aliases"]
                metric_rows = stats[(stats["method"] == method) & (stats["metric"].isin(aliases))]
            display = spec["display"]
            if metric_rows.empty:
                numeric_row[display + "__mean"] = pd.NA
                numeric_row[display + "__std"] = pd.NA
                numeric_row[display + "__n_runs"] = pd.NA
                numeric_row[display + "__within_run_std_mean"] = pd.NA
                formatted_row[display] = ""
                continue
            row = metric_rows.iloc[0]
            mean = row["mean"]
            std = row["std"]
            n_runs = int(row["n_runs"])
            run_counts.append(n_runs)
            numeric_row[display + "__mean"] = mean
            numeric_row[display + "__std"] = std
            numeric_row[display + "__n_runs"] = n_runs
            numeric_row[display + "__within_run_std_mean"] = row.get("within_run_std_mean", pd.NA)
            formatted_row[display] = format_mean_std(mean, std)
        if run_counts:
            unique_counts = sorted(set(run_counts))
            formatted_row["n_runs"] = unique_counts[0] if len(unique_counts) == 1 else ";".join(map(str, unique_counts))
        numeric_rows.append(numeric_row)
        formatted_rows.append(formatted_row)

    numeric_table4 = pd.DataFrame(numeric_rows)
    formatted_table4 = pd.DataFrame(formatted_rows)
    if "n_runs" in formatted_table4.columns:
        cols = ["Method", "n_runs"] + [c for c in formatted_table4.columns if c not in {"Method", "n_runs"}]
        formatted_table4 = formatted_table4[cols]

    output_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = output_dir / "{}_density_metrics_tidy.csv".format(problem_name)
    table4_numeric_path = output_dir / "{}_density_table4_numeric.csv".format(problem_name)
    table4_formatted_path = output_dir / "{}_density_table4_formatted.csv".format(problem_name)

    stats.to_csv(str(tidy_path), index=False)
    numeric_table4.to_csv(str(table4_numeric_path), index=False)
    formatted_table4.to_csv(str(table4_formatted_path), index=False)

    return {
        "n_files": len(paths),
        "source_kind": source_kind,
        "tidy": tidy_path,
        "table4_numeric": table4_numeric_path,
        "table4_formatted": table4_formatted_path,
    }


def aggregate_problem(problem_dir, output_dir):
    return {
        "legacy": aggregate_legacy_metrics(problem_dir, output_dir),
        "density": aggregate_density_problem(problem_dir, output_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate inverse-problem run CSVs.  This preserves the legacy "
            "*_metrics.csv mean/std summaries and also aggregates density "
            "*_drc_energy_*.csv files for Darcy Table 4 uncertainty."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Top-level directory that contains advect_diff/, darcy_flow/, etc. (default: current directory)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory where meta CSVs will be written (default: <root>/meta_results)",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=PROBLEMS,
        help="Subset of problem directories to process",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir is not None else (root / "meta_results")

    print("Searching under: {}".format(root))
    print("Writing summaries to: {}".format(outdir))

    any_found = False
    for problem in args.problems:
        problem_dir = root / problem
        try:
            result = aggregate_problem(problem_dir, outdir)
        except Exception as e:
            print("[fail] {}: {}".format(problem, e))
            continue

        legacy = result.get("legacy", {})
        density = result.get("density", {})
        if legacy.get("n_files", 0) == 0 and density.get("n_files", 0) == 0:
            print("[skip] {}: no metrics or density CSVs found under {}".format(problem, problem_dir / "run_results"))
            continue

        any_found = True
        if legacy.get("n_files", 0):
            print("[ok]   {}: aggregated {} legacy metrics files".format(problem, legacy["n_files"]))
            print("       numeric   -> {}".format(legacy["numeric"]))
            print("       formatted -> {}".format(legacy["formatted"]))
            print("       tidy      -> {}".format(legacy["tidy"]))
        if density.get("n_files", 0):
            print("[ok]   {}: aggregated {} density files ({})".format(problem, density["n_files"], density["source_kind"]))
            print("       tidy      -> {}".format(density["tidy"]))
            print("       table4 numeric   -> {}".format(density["table4_numeric"]))
            print("       table4 formatted -> {}".format(density["table4_formatted"]))

    if not any_found:
        print("No metrics CSV files were found. Check the root path and directory structure.")


if __name__ == "__main__":
    main()
