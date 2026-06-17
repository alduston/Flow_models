#!/usr/bin/env python3

import argparse
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


def sanitize(name):
    """Make a sampler name safe for CSV column names while remaining readable."""
    return (
        str(name).strip()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def find_metric_csvs(problem_dir):
    """Recursively find metrics CSV files under <problem>/run_results."""
    run_results = problem_dir / "run_results"
    if not run_results.exists():
        return []

    csvs = []
    for p in run_results.rglob("*_metrics.csv"):
        if p.is_file() and "meta_metrics" not in p.name:
            csvs.append(p)
    return sorted(csvs)


def load_and_normalize_metrics(csv_path):
    """Load one metrics CSV and return a standardized long-form table.

    Expected input shape:
        metric | sampler1 | sampler2 | ...

    Returned shape:
        metric | sampler | value
    """
    df = pd.read_csv(csv_path)
    if "metric" not in df.columns:
        raise ValueError("{} is missing a 'metric' column".format(csv_path))

    long_df = df.melt(id_vars="metric", var_name="sampler", value_name="value")
    long_df["source_file"] = str(csv_path)

    # Coerce numeric metric values; drop anything non-numeric/empty.
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["metric", "sampler", "value"])
    return long_df


def aggregate_problem(problem_dir, output_dir):
    problem_name = problem_dir.name
    csv_paths = find_metric_csvs(problem_dir)

    if not csv_paths:
        return 0, None, None, None

    long_tables = [load_and_normalize_metrics(path) for path in csv_paths]
    combined = pd.concat(long_tables, ignore_index=True)

    # pandas on some clusters is too old for groupby(..., dropna=False)
    stats = (
        combined.groupby(["metric", "sampler"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_runs"})
    )
    stats["std"] = stats["std"].fillna(0.0)

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

        formatted_vals = []
        for m, s in zip(mean_wide[sampler].tolist(), std_wide[sampler].tolist()):
            if pd.isna(m):
                formatted_vals.append("")
            else:
                s_val = 0.0 if pd.isna(s) else s
                formatted_vals.append("{:.6g} ± {:.6g}".format(m, s_val))
        formatted_cols[sampler] = formatted_vals

    numeric_out = pd.DataFrame(numeric_cols, index=mean_wide.index).reset_index()
    formatted_out = pd.DataFrame(formatted_cols, index=mean_wide.index).reset_index()

    # Optional convenience column: common run count if all samplers/metrics agree.
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

    return len(csv_paths), numeric_path, formatted_path, tidy_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate run-level *_metrics.csv files for each inverse-problem directory "
            "and write mean/std summary CSVs."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Top-level directory that contains advect_diff/, afwi/, ... (default: current directory)",
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
            n_files, numeric_path, formatted_path, tidy_path = aggregate_problem(problem_dir, outdir)
        except Exception as e:
            print("[fail] {}: {}".format(problem, e))
            continue

        if n_files == 0:
            print("[skip] {}: no *_metrics.csv files found under {}".format(problem, problem_dir / "run_results"))
            continue

        any_found = True
        print("[ok]   {}: aggregated {} files".format(problem, n_files))
        print("       numeric   -> {}".format(numeric_path))
        print("       formatted -> {}".format(formatted_path))
        print("       tidy      -> {}".format(tidy_path))

    if not any_found:
        print("No metrics CSV files were found. Check the root path and directory structure.")


if __name__ == "__main__":
    main()
