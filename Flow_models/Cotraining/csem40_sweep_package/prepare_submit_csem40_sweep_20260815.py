#!/usr/bin/env python3
"""Generate and optionally submit a bundled 40-epoch CSEM screen.

The default pairwise design covers every listed hyperparameter value and every
two-factor value combination.  It is substantially more informative than a
one-factor-at-a-time sweep while avoiding the literal Cartesian product.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
from typing import Any


DEFAULT_CONFIG = "csem_fmnist_40ep_sweep_space_20260815.json"
SBATCH_JOB_RE = re.compile(r"Submitted\s+batch\s+job\s+(\d+)", re.IGNORECASE)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def stable_hash(value: Any, length: int = 10) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def safe_label(value: str, limit: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return (cleaned or "run")[:limit]


def merge_args(*mappings: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in mappings:
        result.update(mapping)
    return result


def args_to_cli(arguments: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in arguments.items():
        option = "--" + key
        if isinstance(value, bool):
            cli.append(option if value else "--no-" + key)
        elif value is None:
            continue
        elif isinstance(value, list):
            for item in value:
                cli.extend((option, str(item)))
        else:
            cli.extend((option, str(value)))
    return cli


def candidate_pair_tokens(candidate: tuple[int, ...]) -> set[tuple[int, int, int, int]]:
    return {
        (left, candidate[left], right, candidate[right])
        for left in range(len(candidate))
        for right in range(left + 1, len(candidate))
    }


def pair_token(
    first_factor: int,
    first_value: int,
    second_factor: int,
    second_value: int,
) -> tuple[int, int, int, int]:
    if first_factor < second_factor:
        return (first_factor, first_value, second_factor, second_value)
    return (second_factor, second_value, first_factor, first_value)


def pairwise_candidates(factors: list[dict[str, Any]], seed: int) -> list[tuple[int, ...]]:
    """Build a deterministic IPOG-style all-pairs covering design.

    This grows rows one factor at a time instead of enumerating the full
    Cartesian product.  That matters once CSEM and terminal-KL coefficients are
    crossed independently: the full product is intentionally large, while the
    all-pairs design remains small.
    """
    cardinalities = [len(factor["values"]) for factor in factors]
    if any(cardinality < 1 for cardinality in cardinalities):
        raise ValueError("Every factor must have at least one value.")
    factor_count = len(factors)
    if factor_count == 0:
        return [tuple()]
    if factor_count == 1:
        return [(value,) for value in range(cardinalities[0])]

    rng = random.Random(seed)
    factor_order = sorted(
        range(factor_count), key=lambda index: (-cardinalities[index], index)
    )
    first, second = factor_order[:2]
    rows: list[dict[int, int]] = [
        {first: first_value, second: second_value}
        for first_value in range(cardinalities[first])
        for second_value in range(cardinalities[second])
    ]
    rng.shuffle(rows)
    assigned = [first, second]
    covered = {
        pair_token(first, row[first], second, row[second]) for row in rows
    }

    for current in factor_order[2:]:
        value_usage = [0 for _ in range(cardinalities[current])]

        # Horizontal growth: assign the value that covers the most new pairs
        # with values already present in each row.
        for row in rows:
            value_order = list(range(cardinalities[current]))
            rng.shuffle(value_order)
            scored = []
            for current_value in value_order:
                gain = sum(
                    pair_token(previous, row[previous], current, current_value)
                    not in covered
                    for previous in assigned
                )
                scored.append((gain, -value_usage[current_value], current_value))
            _, _, chosen = max(scored)
            row[current] = chosen
            value_usage[chosen] += 1
            for previous in assigned:
                covered.add(
                    pair_token(previous, row[previous], current, chosen)
                )

        def missing_current_pairs() -> list[tuple[int, int, int, int]]:
            missing: list[tuple[int, int, int, int]] = []
            for previous in assigned:
                for previous_value in range(cardinalities[previous]):
                    for current_value in range(cardinalities[current]):
                        token = pair_token(
                            previous, previous_value, current, current_value
                        )
                        if token not in covered:
                            missing.append(token)
            return missing

        # Vertical growth: add rows until every still-missing pair involving
        # the new factor is covered. Each added row opportunistically covers
        # one missing pair for every prior factor.
        missing = missing_current_pairs()
        while missing:
            rng.shuffle(missing)
            seed_token = missing[0]
            if seed_token[0] == current:
                current_value = seed_token[1]
                seed_previous = seed_token[2]
                seed_previous_value = seed_token[3]
            else:
                seed_previous = seed_token[0]
                seed_previous_value = seed_token[1]
                current_value = seed_token[3]
            new_row = {
                current: current_value,
                seed_previous: seed_previous_value,
            }
            for previous in assigned:
                if previous in new_row:
                    continue
                candidates = list(range(cardinalities[previous]))
                rng.shuffle(candidates)
                candidates.sort(
                    key=lambda previous_value: (
                        pair_token(
                            previous,
                            previous_value,
                            current,
                            current_value,
                        )
                        not in covered
                    ),
                    reverse=True,
                )
                new_row[previous] = candidates[0]
            rows.append(new_row)
            value_usage[current_value] += 1
            for previous in assigned:
                covered.add(
                    pair_token(
                        previous,
                        new_row[previous],
                        current,
                        current_value,
                    )
                )
            missing = missing_current_pairs()
        assigned.append(current)

    selected = [tuple(row[index] for index in range(factor_count)) for row in rows]
    expected = {
        (left, left_value, right, right_value)
        for left in range(factor_count)
        for right in range(left + 1, factor_count)
        for left_value in range(cardinalities[left])
        for right_value in range(cardinalities[right])
    }
    actual: set[tuple[int, int, int, int]] = set()
    for candidate in selected:
        actual.update(candidate_pair_tokens(candidate))
    if expected != actual:
        raise RuntimeError(
            f"Internal all-pairs verification failed: {len(expected - actual)} "
            "pairs remain uncovered."
        )
    return selected


def full_candidates(factors: list[dict[str, Any]]) -> list[tuple[int, ...]]:
    return list(
        itertools.product(*(range(len(factor["values"])) for factor in factors))
    )


def factor_case(
    factors: list[dict[str, Any]], candidate: tuple[int, ...]
) -> tuple[dict[str, Any], dict[str, str]]:
    arguments: dict[str, Any] = {}
    assignments: dict[str, str] = {}
    for factor, value_index in zip(factors, candidate):
        value = factor["values"][value_index]
        assignments[str(factor["name"])] = str(value["label"])
        arguments.update(value.get("args", {}))
    return arguments, assignments


def make_run_records(
    config: dict[str, Any],
    strategy: str,
    trainer_path: Path,
    basedir: Path,
    sweep_root: Path,
    seed: int,
    allow_large_sweep: bool,
    maximum_runs: int,
) -> list[dict[str, Any]]:
    fixed_args = dict(config.get("fixed_args", {}))
    factors = list(config.get("factors", []))
    if strategy == "pairwise":
        candidates = pairwise_candidates(factors, seed)
    elif strategy == "full":
        full_count = math.prod(len(factor["values"]) for factor in factors)
        if full_count > maximum_runs and not allow_large_sweep:
            raise RuntimeError(
                f"Full Cartesian design contains {full_count:,} factor rows, "
                f"above --max-runs={maximum_runs}. Pass --allow-large-sweep only "
                "if that literal product is intentional."
            )
        candidates = full_candidates(factors)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    proposed: list[dict[str, Any]] = []
    for explicit in config.get("explicit_runs", []):
        proposed.append(
            {
                "design_source": "explicit",
                "design_label": str(explicit["label"]),
                "factor_assignments": {},
                "args": merge_args(fixed_args, explicit.get("args", {})),
            }
        )
    for candidate in candidates:
        factor_args, assignments = factor_case(factors, candidate)
        proposed.append(
            {
                "design_source": strategy,
                "design_label": "pair_" + stable_hash(assignments, 8),
                "factor_assignments": assignments,
                "args": merge_args(fixed_args, factor_args),
            }
        )

    # Preserve named controls first and discard exact duplicate argument sets.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in proposed:
        argument_hash = stable_hash(record["args"], 16)
        if argument_hash in seen:
            continue
        seen.add(argument_hash)
        unique.append(record)

    if len(unique) > maximum_runs and not allow_large_sweep:
        raise RuntimeError(
            f"Design contains {len(unique)} runs, above --max-runs={maximum_runs}. "
            "Use pairwise coverage, raise --max-runs, or pass --allow-large-sweep."
        )

    records: list[dict[str, Any]] = []
    for index, record in enumerate(unique, start=1):
        run_hash = stable_hash(record["args"], 10)
        run_id = (
            f"r{index:03d}_{safe_label(record['design_label'], 36)}_{run_hash}"
        )
        result_dir = sweep_root / "results" / run_id
        run_log = sweep_root / "logs" / f"{run_id}.out"
        run_args = dict(record["args"])
        run_args["master-results-dir"] = str(result_dir)
        # Resolve "python" at execution time after the Slurm job activates the
        # persistent hlsi venv. Do not bake the login-node interpreter path in.
        command = ["python", "-u", str(trainer_path)] + args_to_cli(run_args)
        records.append(
            {
                "run_index": index,
                "run_id": run_id,
                "design_source": record["design_source"],
                "design_label": record["design_label"],
                "factor_assignments": record["factor_assignments"],
                "args": run_args,
                "command": command,
                "command_shell": shlex.join(command),
                "basedir": str(basedir),
                "result_dir": str(result_dir),
                "run_log": str(run_log),
                "status_path": str(sweep_root / "status" / f"{run_id}.json"),
            }
        )
    return records


def write_manifest_csv(path: Path, records: list[dict[str, Any]]) -> None:
    factor_names = sorted(
        {name for record in records for name in record["factor_assignments"]}
    )
    argument_names = sorted({name for record in records for name in record["args"]})
    fields = [
        "run_index",
        "run_id",
        "design_source",
        "design_label",
        *[f"factor__{name}" for name in factor_names],
        *[f"arg__{name}" for name in argument_names],
        "result_dir",
        "run_log",
        "command_shell",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                key: record[key]
                for key in (
                    "run_index",
                    "run_id",
                    "design_source",
                    "design_label",
                    "result_dir",
                    "run_log",
                    "command_shell",
                )
            }
            row.update(
                {
                    f"factor__{name}": record["factor_assignments"].get(name, "")
                    for name in factor_names
                }
            )
            row.update(
                {
                    f"arg__{name}": record["args"].get(name, "")
                    for name in argument_names
                }
            )
            writer.writerow(row)


def slurm_text(
    *,
    bundle_index: int,
    bundle_path: Path,
    worker_path: Path,
    sweep_root: Path,
    slurm: dict[str, Any],
) -> str:
    output_path = sweep_root / "logs" / f"slurm-csem40-b{bundle_index:03d}-%j.out"
    mail_lines = ""
    if slurm.get("mail_user"):
        mail_lines += f"#SBATCH --mail-user={slurm['mail_user']}\n"
    if slurm.get("mail_type"):
        mail_lines += f"#SBATCH --mail-type={slurm['mail_type']}\n"
    return f"""#!/bin/bash
#SBATCH -J c40b{bundle_index:03d}
#SBATCH -p {slurm.get('partition', 'gh')}
#SBATCH -N {int(slurm.get('nodes', 1))}
#SBATCH -n {int(slurm.get('tasks', 1))}
#SBATCH -t {slurm.get('time', '06:00:00')}
#SBATCH -o {output_path}
{mail_lines}set -uo pipefail

echo "== CSEM 40-epoch sweep bundle {bundle_index:03d}"
echo "== Host: $(hostname)"
echo "== Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "== Start: $(date)"

module purge
module load gcc/13.2.0
module load python3/3.11.8

if [ -z "${{SCRATCH:-}}" ]; then
    echo "[error] SCRATCH is not set."
    exit 2
fi
VENV_DIR="$SCRATCH/venvs/hlsi"
if [ ! -d "$VENV_DIR" ]; then
    echo "[error] Expected venv at $VENV_DIR not found."
    exit 2
fi
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi || true
python - <<'PY'
import sys
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(2)
print("CUDA device:", torch.cuda.get_device_name(0))
PY
rc=$?
if [ $rc -ne 0 ]; then
    exit $rc
fi

python -u {shlex.quote(str(worker_path))} --bundle {shlex.quote(str(bundle_path))}
rc=$?
echo "== End: $(date), rc=$rc"
exit $rc
"""


def generate(
    config_path: Path,
    basedir_override: Path | None,
    sweep_root_override: Path | None,
    strategy: str,
    seed: int,
    runs_per_bundle_override: int | None,
    allow_large_sweep: bool,
    maximum_runs: int,
) -> tuple[Path, list[Path], list[dict[str, Any]]]:
    config = read_json(config_path)
    basedir = (basedir_override or Path(config["default_basedir"])).resolve()
    sweep_root = (
        sweep_root_override or (basedir / str(config["sweep_name"]))
    ).resolve()
    if sweep_root.exists():
        raise FileExistsError(
            f"Sweep root already exists; refusing to overwrite: {sweep_root}"
        )
    trainer_path = (basedir / str(config["trainer"])).resolve()
    worker_path = (basedir / str(config["worker"])).resolve()
    for required in (basedir, trainer_path, worker_path):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    records = make_run_records(
        config,
        strategy,
        trainer_path,
        basedir,
        sweep_root,
        seed,
        allow_large_sweep,
        maximum_runs,
    )
    for directory in (
        "bundles",
        "logs",
        "results",
        "slurm_jobs",
        "status",
        "summary",
    ):
        (sweep_root / directory).mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "config_path": str(config_path.resolve()),
        "basedir": str(basedir),
        "sweep_root": str(sweep_root),
        "strategy": strategy,
        "seed": seed,
        "run_count": len(records),
        "runs": records,
    }
    write_json(sweep_root / "run_manifest.json", manifest)
    write_manifest_csv(sweep_root / "run_manifest.csv", records)

    slurm_config = dict(config.get("slurm", {}))
    runs_per_bundle = int(
        runs_per_bundle_override
        if runs_per_bundle_override is not None
        else slurm_config.get("runs_per_bundle", 2)
    )
    if runs_per_bundle < 1:
        raise ValueError("runs_per_bundle must be >= 1")

    slurm_paths: list[Path] = []
    bundle_records: list[dict[str, Any]] = []
    for offset in range(0, len(records), runs_per_bundle):
        bundle_index = len(bundle_records) + 1
        selected_runs = records[offset : offset + runs_per_bundle]
        bundle_path = sweep_root / "bundles" / f"bundle_{bundle_index:03d}.json"
        bundle = {
            "schema_version": 1,
            "bundle_index": bundle_index,
            "sweep_root": str(sweep_root),
            "runs": selected_runs,
        }
        write_json(bundle_path, bundle)
        slurm_path = sweep_root / "slurm_jobs" / f"csem40_bundle_{bundle_index:03d}.slurm"
        slurm_path.write_text(
            slurm_text(
                bundle_index=bundle_index,
                bundle_path=bundle_path,
                worker_path=worker_path,
                sweep_root=sweep_root,
                slurm=slurm_config,
            ),
            encoding="utf-8",
        )
        slurm_path.chmod(0o750)
        slurm_paths.append(slurm_path)
        bundle_records.append(
            {
                "bundle_index": bundle_index,
                "bundle_path": str(bundle_path),
                "slurm_path": str(slurm_path),
                "run_ids": [run["run_id"] for run in selected_runs],
            }
        )
    write_json(sweep_root / "bundle_manifest.json", {"bundles": bundle_records})
    return sweep_root, slurm_paths, records


def submit(slurm_paths: list[Path], sweep_root: Path) -> None:
    submissions: list[dict[str, Any]] = []
    for slurm_path in slurm_paths:
        completed = subprocess.run(
            ["sbatch", str(slurm_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        combined_output = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        matches = SBATCH_JOB_RE.findall(combined_output)
        if completed.returncode != 0 or not matches:
            write_json(
                sweep_root / "submission_manifest.json",
                {"submissions": submissions, "failed_output": combined_output},
            )
            raise RuntimeError(
                f"Could not submit/parse job id for {slurm_path}.\n{combined_output}"
            )
        job_id = matches[-1]
        record = {
            "slurm_path": str(slurm_path),
            "job_id": job_id,
            "raw_output": combined_output,
        }
        submissions.append(record)
        write_json(
            sweep_root / "submission_manifest.json",
            {"submissions": submissions},
        )
        print(f"submitted {slurm_path.name}: job {job_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--basedir", type=Path, default=None)
    parser.add_argument("--sweep-root", type=Path, default=None)
    parser.add_argument("--strategy", choices=("pairwise", "full"), default="pairwise")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--runs-per-bundle", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=256)
    parser.add_argument("--allow-large-sweep", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--submit-existing",
        type=Path,
        default=None,
        metavar="SWEEP_ROOT",
        help="Submit already-generated slurm jobs without regenerating anything.",
    )
    args = parser.parse_args()

    if args.submit_existing is not None:
        sweep_root = args.submit_existing.resolve()
        bundle_manifest = read_json(sweep_root / "bundle_manifest.json")
        slurm_paths = [
            Path(bundle["slurm_path"]) for bundle in bundle_manifest["bundles"]
        ]
        submit(slurm_paths, sweep_root)
        return 0

    config_path = args.config.resolve()
    sweep_root, slurm_paths, records = generate(
        config_path,
        args.basedir,
        args.sweep_root,
        args.strategy,
        args.seed,
        args.runs_per_bundle,
        args.allow_large_sweep,
        args.max_runs,
    )
    print(f"generated {len(records)} runs in {len(slurm_paths)} bundles")
    print(f"sweep root: {sweep_root}")
    print(f"manifest:   {sweep_root / 'run_manifest.csv'}")
    if args.submit:
        submit(slurm_paths, sweep_root)
    else:
        print("generation only; inspect the manifest, then submit with:")
        print(
            f"  {shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
            f"--submit-existing {shlex.quote(str(sweep_root))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
