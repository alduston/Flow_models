# -*- coding: utf-8 -*-
"""Generate an independent, diagnostics-gated NUTS reference for Darcy.

This program is deliberately an *offline oracle builder*, not a compute-frontier
competitor.  It reuses the exact Darcy problem construction from
``darcy_mala_lfgi_dirichlet_flop_frontier_20260821_v4.py`` but samples the
posterior with NumPyro NUTS.  No LFGI/GAD particles, banks, scores, gates, or
transport outputs are used in initialization or sampling.

The default workflow is intentionally conservative:

* reconstruct the exact synthetic data used by the frontier script;
* find the MAP from several truth-blind initial points;
* compute an exact posterior Hessian at the best MAP and sample in locally
  Laplace-whitened coordinates;
* run four independently adapted dense-mass NUTS chains, saving each completed
  chain separately so a stopped job can be resumed;
* certify the pooled cloud only after rank-normalized R-hat, bulk/tail ESS,
  divergence, tree-depth, acceptance, and E-BFMI checks pass.

On success, ``reference_samples.npz`` contains a ``samples`` array directly
accepted by the Darcy frontier script's ``--reference-samples`` option.  If
diagnostics fail, the samples are retained as
``reference_samples_uncertified.npz`` and the program exits nonzero rather than
silently presenting them as ground truth.

Required runtime dependencies are JAX, NumPyro, SciPy, ArviZ, and the imports
required by the Darcy frontier module (including its local ``gad_sampling.py``).

Example
-------

    python darcy_nuts_offline_reference_20260821_v2.py \
        --output-dir run_results/darcy_nuts_reference_seed42

Resume completed chains after a scheduler interruption:

    python darcy_nuts_offline_reference_20260821_v2.py \
        --output-dir run_results/darcy_nuts_reference_seed42 --resume

For a multi-node scheduler, the same run can be split safely into one setup
stage, one independently restartable process per chain, and one finalization
stage:

    python darcy_nuts_offline_reference_20260821_v2.py \
        --output-dir run_results/darcy_nuts_reference_seed42 --prepare-only
    python darcy_nuts_offline_reference_20260821_v2.py \
        --output-dir run_results/darcy_nuts_reference_seed42 \
        --resume --chain-index 0
    # Run --chain-index 1, 2, and 3 concurrently on other GPU nodes.
    python darcy_nuts_offline_reference_20260821_v2.py \
        --output-dir run_results/darcy_nuts_reference_seed42 \
        --resume --finalize-only

Then run the compute frontier with:

    python darcy_mala_lfgi_dirichlet_flop_frontier_20260821_v4.py \
        --compute-axis flops --gate-methods lfgi,GAD-lfgi \
        --reference-samples \
        run_results/darcy_nuts_reference_seed42/reference_samples.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("JAX_ENABLE_X64", "true")

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised in the target env
    raise SystemExit(
        "JAX is required. Run this script in the same JAX environment as the "
        "Darcy frontier experiment."
    ) from exc

jax.config.update("jax_enable_x64", True)

try:
    import arviz as az
except ImportError as exc:  # pragma: no cover - exercised in the target env
    raise SystemExit(
        "ArviZ is required for rank-normalized R-hat and bulk/tail ESS. "
        "Install it in the sampling environment (for example: pip install arviz)."
    ) from exc

try:
    import numpyro
    from numpyro.infer import MCMC, NUTS
except ImportError as exc:  # pragma: no cover - exercised in the target env
    raise SystemExit(
        "NumPyro is required for NUTS. Install it without replacing the working "
        "JAX build in the Darcy environment (for example: pip install numpyro)."
    ) from exc

import numpy as np
from scipy.optimize import minimize


SCRIPT_VERSION = "20260821_v2"
REFERENCE_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_darcy = here / "darcy_mala_lfgi_dirichlet_flop_frontier_20260821_v4.py"
    parser = argparse.ArgumentParser(
        description=(
            "Build a high-quality, independent NUTS posterior reference for "
            "the Darcy compute-frontier benchmark."
        )
    )
    parser.add_argument(
        "--darcy-script",
        type=str,
        default=str(default_darcy),
        help=(
            "Darcy frontier script exporting build_darcy_problem. Its defaults "
            "and data-generation convention are reproduced exactly."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible setup/chain files already present in --output-dir.",
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Construct and save the problem, truth-blind MAP, and Laplace "
            "whitening, then exit before sampling."
        ),
    )
    stages.add_argument(
        "--chain-index",
        type=int,
        default=None,
        help=(
            "Run or load only this zero-based chain and exit. Intended for one "
            "GPU process per chain after --prepare-only. Requires --resume."
        ),
    )
    stages.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Load all completed chain files, run diagnostics, and export the "
            "reference without starting any missing chain. Requires --resume."
        ),
    )

    # These must match the benchmark run whose functionals will be scored.
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-observations", type=int, default=120)
    parser.add_argument("--n-holdout", type=int, default=30)
    parser.add_argument("--noise-std", type=float, default=1e-3)

    # Offline-oracle defaults: intentionally much longer than the frontier MALA.
    parser.add_argument("--sampler-seed", type=int, default=20260821)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=3000)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument(
        "--thinning",
        type=int,
        default=1,
        help=(
            "Must remain 1 for a certified run. Retaining every draw gives more "
            "reliable ESS/MCSE diagnostics than post-hoc thinning."
        ),
    )
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--max-tree-depth-warmup", type=int, default=10)
    parser.add_argument("--max-tree-depth", type=int, default=12)
    parser.add_argument(
        "--diagonal-mass",
        action="store_true",
        help="Use diagonal rather than the recommended dense adapted mass matrix.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Truth-blind MAP and local whitening configuration.
    parser.add_argument("--map-starts", type=int, default=4)
    parser.add_argument("--map-start-scale", type=float, default=0.5)
    parser.add_argument("--map-maxiter", type=int, default=1000)
    parser.add_argument("--map-gtol", type=float, default=1e-7)
    parser.add_argument("--map-ftol", type=float, default=1e-12)
    parser.add_argument(
        "--map-energy-window",
        type=float,
        default=100.0,
        help="Retain distinct local optima within this potential-energy gap.",
    )
    parser.add_argument("--map-distinct-distance", type=float, default=1e-3)
    parser.add_argument("--hessian-eigenvalue-floor", type=float, default=1e-6)
    parser.add_argument("--hessian-max-condition", type=float, default=1e8)
    parser.add_argument(
        "--laplace-scale",
        type=float,
        default=1.0,
        help="Multiplier on the exact-Hessian local covariance square root.",
    )
    parser.add_argument(
        "--init-overdispersion",
        type=float,
        default=1.5,
        help="Standard deviation of chain starts in Laplace-whitened coordinates.",
    )

    # Conservative reference-certification criteria.
    parser.add_argument("--max-rhat", type=float, default=1.01)
    parser.add_argument("--min-bulk-ess", type=float, default=1000.0)
    parser.add_argument("--min-tail-ess", type=float, default=400.0)
    parser.add_argument("--max-divergences", type=int, default=0)
    parser.add_argument("--max-tree-depth-fraction", type=float, default=0.01)
    parser.add_argument("--min-ebfmi", type=float, default=0.30)
    parser.add_argument("--min-mean-accept", type=float, default=0.80)
    parser.add_argument(
        "--allow-failed-diagnostics",
        action="store_true",
        help=(
            "Also write reference_samples.npz after failed checks. The file is "
            "marked uncertified; use only for deliberate sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--functional-chunk-size",
        type=int,
        default=512,
        help="Host-side chunk size for permeability-field functional evaluation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.grid_size < 5:
        raise ValueError("--grid-size must be at least 5.")
    if args.latent_dim <= 0 or args.latent_dim > args.grid_size ** 2:
        raise ValueError("--latent-dim must lie in [1, grid_size**2].")
    n_interior = (args.grid_size - 2) ** 2
    if args.n_observations <= 0 or args.n_holdout <= 0:
        raise ValueError("Observation counts must be positive.")
    if args.n_observations + args.n_holdout > n_interior:
        raise ValueError("Training plus held-out locations exceed the interior grid.")
    if args.noise_std <= 0.0:
        raise ValueError("--noise-std must be positive.")
    if args.num_chains < 4:
        raise ValueError("At least four chains are required for a reference run.")
    if args.num_warmup < 500 or args.num_samples < 500:
        raise ValueError("Reference warmup and retained samples must each be >= 500.")
    if args.thinning != 1:
        raise ValueError(
            "Certified reference runs intentionally require --thinning 1; retain "
            "every post-warmup transition and let ESS quantify autocorrelation."
        )
    if not 0.5 < args.target_accept < 1.0:
        raise ValueError("--target-accept must lie in (0.5,1).")
    if args.max_tree_depth_warmup < 1 or args.max_tree_depth < 1:
        raise ValueError("NUTS tree depths must be positive.")
    if args.map_starts < 1 or args.map_maxiter < 1:
        raise ValueError("MAP optimization controls must be positive.")
    if args.map_start_scale < 0.0 or args.map_energy_window < 0.0:
        raise ValueError("MAP start scale and energy window must be nonnegative.")
    if args.hessian_eigenvalue_floor <= 0.0:
        raise ValueError("--hessian-eigenvalue-floor must be positive.")
    if args.hessian_max_condition <= 1.0:
        raise ValueError("--hessian-max-condition must exceed one.")
    if args.laplace_scale <= 0.0 or args.init_overdispersion <= 0.0:
        raise ValueError("Laplace and initialization scales must be positive.")
    if args.max_rhat <= 1.0 or args.min_bulk_ess <= 0.0 or args.min_tail_ess <= 0.0:
        raise ValueError("Invalid R-hat/ESS certification thresholds.")
    if args.max_divergences < 0:
        raise ValueError("--max-divergences must be nonnegative.")
    if not 0.0 <= args.max_tree_depth_fraction <= 1.0:
        raise ValueError("--max-tree-depth-fraction must lie in [0,1].")
    if args.min_ebfmi <= 0.0 or not 0.0 < args.min_mean_accept < 1.0:
        raise ValueError("Invalid E-BFMI/acceptance thresholds.")
    if args.functional_chunk_size < 1:
        raise ValueError("--functional-chunk-size must be positive.")
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires an explicit --output-dir.")
    staged = args.prepare_only or args.chain_index is not None or args.finalize_only
    if staged and not args.output_dir:
        raise ValueError("Staged execution requires an explicit --output-dir.")
    if (args.chain_index is not None or args.finalize_only) and not args.resume:
        raise ValueError("--chain-index and --finalize-only require --resume.")
    if args.chain_index is not None and not 0 <= args.chain_index < args.num_chains:
        raise ValueError(
            f"--chain-index must lie in [0,{args.num_chains - 1}] for this run."
        )


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_arrays(metadata: Mapping[str, object], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(canonical_json(dict(metadata)).encode("utf-8"))
    for array in arrays:
        arr = np.ascontiguousarray(np.asarray(array))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_darcy_module(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Darcy script not found: {path}")
    module_name = f"_darcy_reference_problem_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
    if not hasattr(module, "build_darcy_problem"):
        raise AttributeError(
            f"{path.name} does not export build_darcy_problem; use the v4 frontier script."
        )
    return module


def build_problem_and_data(args: argparse.Namespace, module):
    problem = module.build_darcy_problem(
        grid_size=args.grid_size,
        latent_dim=args.latent_dim,
        n_observations=args.n_observations,
        n_holdout=args.n_holdout,
        seed=args.data_seed,
    )
    # This exactly mirrors the v4 frontier.  alpha_true is recorded for auditing
    # but is never used to initialize MAP optimization or NUTS.
    data_rng = np.random.default_rng(args.data_seed)
    alpha_true = data_rng.normal(0.0, 0.5, size=args.latent_dim)
    y_clean = np.asarray(
        problem.solve_forward(jnp.asarray(alpha_true, dtype=jnp.float64)),
        dtype=np.float64,
    )
    y_obs = y_clean + data_rng.normal(0.0, args.noise_std, size=y_clean.shape)
    problem_meta = {
        "data_seed": int(args.data_seed),
        "grid_size": int(args.grid_size),
        "latent_dim": int(args.latent_dim),
        "n_observations": int(args.n_observations),
        "n_holdout": int(args.n_holdout),
        "noise_std": float(args.noise_std),
    }
    fingerprint = hash_arrays(
        problem_meta,
        [
            problem.basis_np,
            problem.obs_locations_train,
            problem.obs_locations_holdout,
            alpha_true,
            y_obs,
        ],
    )
    return problem, alpha_true, y_clean, y_obs, fingerprint


def configuration_signature(args: argparse.Namespace, darcy_sha256: str) -> str:
    ignored = {
        "output_dir",
        "resume",
        "progress_bar",
        "allow_failed_diagnostics",
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "max_divergences",
        "max_tree_depth_fraction",
        "min_ebfmi",
        "min_mean_accept",
        "functional_chunk_size",
        "prepare_only",
        "chain_index",
        "finalize_only",
    }
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in ignored and key != "darcy_script"
    }
    payload["darcy_script_sha256"] = darcy_sha256
    payload["script_version"] = SCRIPT_VERSION
    return hash_json(payload)


def make_alpha_potential(problem, y_obs: np.ndarray, noise_std: float):
    y = jnp.asarray(y_obs, dtype=jnp.float64)
    sigma = jnp.asarray(float(noise_std), dtype=jnp.float64)

    def potential(alpha):
        alpha = jnp.asarray(alpha, dtype=jnp.float64)
        residual = (problem.solve_forward(alpha) - y) / sigma
        return 0.5 * jnp.vdot(alpha, alpha) + 0.5 * jnp.vdot(residual, residual)

    return potential


def find_map_and_laplace(
    args: argparse.Namespace,
    potential_alpha,
) -> Dict[str, object]:
    d = int(args.latent_dim)
    value_grad_jax = jax.jit(jax.value_and_grad(potential_alpha))
    # Force compilation once at the correct dimension.
    warm_value, warm_grad = value_grad_jax(jnp.zeros(d, dtype=jnp.float64))
    _ = (jax.device_get(warm_value), jax.device_get(warm_grad))

    def objective(x_np: np.ndarray):
        value, gradient = value_grad_jax(jnp.asarray(x_np, dtype=jnp.float64))
        scalar = float(np.asarray(jax.device_get(value)))
        grad = np.asarray(jax.device_get(gradient), dtype=np.float64)
        if not math.isfinite(scalar) or not np.isfinite(grad).all():
            raise FloatingPointError("Non-finite value/gradient during MAP optimization.")
        return scalar, grad

    rng = np.random.default_rng(args.sampler_seed + 17_301)
    starts = [np.zeros(d, dtype=np.float64)]
    starts.extend(
        rng.normal(0.0, args.map_start_scale, size=d).astype(np.float64)
        for _ in range(args.map_starts - 1)
    )
    solutions = []
    print(f"\n=== Truth-blind multi-start MAP ({len(starts)} starts) ===")
    for index, start in enumerate(starts):
        started = time.perf_counter()
        try:
            initial_energy, _ = objective(start)
        except (FloatingPointError, ValueError) as exc:
            elapsed = time.perf_counter() - started
            print(
                f"MAP start {index + 1}/{len(starts)} is non-finite and was skipped: {exc}"
            )
            solutions.append(
                {
                    "start_index": index,
                    "initial_energy": float("inf"),
                    "energy": float("inf"),
                    "gradient_norm": float("inf"),
                    "success": False,
                    "status": -1,
                    "message": str(exc),
                    "iterations": -1,
                    "function_evaluations": -1,
                    "elapsed_seconds": elapsed,
                    "position": start,
                }
            )
            continue
        try:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": int(args.map_maxiter),
                    "gtol": float(args.map_gtol),
                    "ftol": float(args.map_ftol),
                    "maxls": 50,
                },
            )
        except (FloatingPointError, ValueError) as exc:
            elapsed = time.perf_counter() - started
            print(
                f"MAP start {index + 1}/{len(starts)} failed after {elapsed:.1f}s: {exc}"
            )
            solutions.append(
                {
                    "start_index": index,
                    "initial_energy": initial_energy,
                    "energy": float("inf"),
                    "gradient_norm": float("inf"),
                    "success": False,
                    "status": -1,
                    "message": str(exc),
                    "iterations": -1,
                    "function_evaluations": -1,
                    "elapsed_seconds": elapsed,
                    "position": start,
                }
            )
            continue
        elapsed = time.perf_counter() - started
        x = np.asarray(result.x, dtype=np.float64)
        try:
            energy, gradient = objective(x)
        except (FloatingPointError, ValueError) as exc:
            print(
                f"MAP start {index + 1}/{len(starts)} ended at a non-finite "
                f"position and was skipped: {exc}"
            )
            solutions.append(
                {
                    "start_index": index,
                    "initial_energy": initial_energy,
                    "energy": float("inf"),
                    "gradient_norm": float("inf"),
                    "success": False,
                    "status": int(getattr(result, "status", -1)),
                    "message": str(exc),
                    "iterations": int(getattr(result, "nit", -1)),
                    "function_evaluations": int(getattr(result, "nfev", -1)),
                    "elapsed_seconds": elapsed,
                    "position": start,
                }
            )
            continue
        record = {
            "start_index": index,
            "initial_energy": initial_energy,
            "energy": energy,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", -1)),
            "function_evaluations": int(getattr(result, "nfev", -1)),
            "elapsed_seconds": elapsed,
            "position": x,
        }
        solutions.append(record)
        print(
            f"MAP start {index + 1}/{len(starts)}: U={energy:.6e}, "
            f"|grad|={record['gradient_norm']:.3e}, success={record['success']}, "
            f"nit={record['iterations']}, wall={elapsed:.1f}s"
        )

    finite = [item for item in solutions if math.isfinite(float(item["energy"]))]
    if not finite:
        raise RuntimeError("Every MAP optimization produced a non-finite result.")
    finite.sort(key=lambda item: float(item["energy"]))
    best = finite[0]
    map_position = np.asarray(best["position"], dtype=np.float64)
    best_energy = float(best["energy"])

    usable = []
    for item in finite:
        if float(item["energy"]) - best_energy > args.map_energy_window:
            continue
        candidate = np.asarray(item["position"], dtype=np.float64)
        if all(
            np.linalg.norm(candidate - np.asarray(other["position"]))
            > args.map_distinct_distance
            for other in usable
        ):
            usable.append(item)
    if not usable:
        usable = [best]
    print(
        f"Selected MAP U={best_energy:.6e}; retained {len(usable)} distinct "
        "candidate basin(s) for overdispersed chain starts."
    )

    print("Computing exact posterior Hessian at the selected MAP...")
    hessian_jax = jax.jit(jax.hessian(potential_alpha))(
        jnp.asarray(map_position, dtype=jnp.float64)
    )
    hessian = np.asarray(jax.device_get(hessian_jax), dtype=np.float64)
    hessian = 0.5 * (hessian + hessian.T)
    if not np.isfinite(hessian).all():
        raise FloatingPointError("Exact MAP Hessian contains non-finite entries.")
    raw_eigvals, eigvecs = np.linalg.eigh(hessian)
    spectral_max = max(float(np.max(np.abs(raw_eigvals))), args.hessian_eigenvalue_floor)
    effective_floor = max(
        float(args.hessian_eigenvalue_floor),
        spectral_max / float(args.hessian_max_condition),
    )
    clipped_eigvals = np.maximum(raw_eigvals, effective_floor)
    laplace_factor = (
        eigvecs
        @ np.diag(float(args.laplace_scale) / np.sqrt(clipped_eigvals))
    )
    if not np.isfinite(laplace_factor).all():
        raise FloatingPointError("Laplace whitening factor is non-finite.")
    print(
        "MAP Hessian spectrum: "
        f"raw=[{raw_eigvals.min():.3e}, {raw_eigvals.max():.3e}], "
        f"regularized floor={effective_floor:.3e}, "
        f"condition={clipped_eigvals.max() / clipped_eigvals.min():.3e}"
    )
    return {
        "map_position": map_position,
        "map_energy": best_energy,
        "map_gradient_norm": float(best["gradient_norm"]),
        "candidate_positions": np.stack(
            [np.asarray(item["position"], dtype=np.float64) for item in usable]
        ),
        "candidate_energies": np.asarray(
            [float(item["energy"]) for item in usable], dtype=np.float64
        ),
        "hessian": hessian,
        "hessian_eigenvalues_raw": raw_eigvals,
        "hessian_eigenvalues_regularized": clipped_eigvals,
        "laplace_factor": laplace_factor,
        "map_records": [
            {key: value for key, value in item.items() if key != "position"}
            for item in solutions
        ],
    }


def save_setup(
    path: Path,
    setup: Mapping[str, object],
    *,
    problem_fingerprint: str,
    config_signature: str,
) -> None:
    atomic_npz(
        path,
        map_position=np.asarray(setup["map_position"]),
        map_energy=np.asarray(setup["map_energy"]),
        map_gradient_norm=np.asarray(setup["map_gradient_norm"]),
        candidate_positions=np.asarray(setup["candidate_positions"]),
        candidate_energies=np.asarray(setup["candidate_energies"]),
        hessian=np.asarray(setup["hessian"]),
        hessian_eigenvalues_raw=np.asarray(setup["hessian_eigenvalues_raw"]),
        hessian_eigenvalues_regularized=np.asarray(
            setup["hessian_eigenvalues_regularized"]
        ),
        laplace_factor=np.asarray(setup["laplace_factor"]),
        problem_fingerprint=np.asarray(problem_fingerprint),
        configuration_signature=np.asarray(config_signature),
        map_records_json=np.asarray(canonical_json(json_ready(setup["map_records"]))),
    )


def load_setup(
    path: Path,
    *,
    problem_fingerprint: str,
    config_signature: str,
) -> Dict[str, object]:
    archive = np.load(path, allow_pickle=False)
    if str(archive["problem_fingerprint"].item()) != problem_fingerprint:
        raise RuntimeError("Existing setup.npz belongs to a different Darcy problem.")
    if str(archive["configuration_signature"].item()) != config_signature:
        raise RuntimeError("Existing setup.npz has incompatible sampler configuration.")
    return {
        "map_position": archive["map_position"],
        "map_energy": float(archive["map_energy"]),
        "map_gradient_norm": float(archive["map_gradient_norm"]),
        "candidate_positions": archive["candidate_positions"],
        "candidate_energies": archive["candidate_energies"],
        "hessian": archive["hessian"],
        "hessian_eigenvalues_raw": archive["hessian_eigenvalues_raw"],
        "hessian_eigenvalues_regularized": archive[
            "hessian_eigenvalues_regularized"
        ],
        "laplace_factor": archive["laplace_factor"],
        "map_records": json.loads(str(archive["map_records_json"].item())),
    }


def make_whitened_potential(potential_alpha, setup: Mapping[str, object]):
    map_position = jnp.asarray(setup["map_position"], dtype=jnp.float64)
    laplace_factor = jnp.asarray(setup["laplace_factor"], dtype=jnp.float64)

    def potential_z(z):
        alpha = map_position + laplace_factor @ jnp.asarray(z, dtype=jnp.float64)
        return potential_alpha(alpha)

    return potential_z


def make_chain_initial_positions(
    args: argparse.Namespace,
    setup: Mapping[str, object],
) -> np.ndarray:
    map_position = np.asarray(setup["map_position"], dtype=np.float64)
    factor = np.asarray(setup["laplace_factor"], dtype=np.float64)
    candidates = np.asarray(setup["candidate_positions"], dtype=np.float64)
    rng = np.random.default_rng(args.sampler_seed + 29_011)
    positions = []
    for chain_index in range(args.num_chains):
        center = candidates[chain_index % len(candidates)]
        center_z = np.linalg.solve(factor, center - map_position)
        jitter = rng.normal(0.0, args.init_overdispersion, size=args.latent_dim)
        positions.append(center_z + jitter)
    return np.stack(positions).astype(np.float64)


def stabilize_chain_initial_positions(
    positions: np.ndarray,
    potential_z,
    *,
    max_shrinks: int = 12,
) -> np.ndarray:
    """Shrink only pathological starts toward the MAP until U and grad are finite."""
    value_grad = jax.jit(jax.value_and_grad(potential_z))
    stabilized = np.asarray(positions, dtype=np.float64).copy()
    for chain_index in range(stabilized.shape[0]):
        candidate = stabilized[chain_index]
        for shrink_index in range(max_shrinks + 1):
            value, gradient = value_grad(jnp.asarray(candidate, dtype=jnp.float64))
            value_np = float(np.asarray(jax.device_get(value)))
            gradient_np = np.asarray(jax.device_get(gradient), dtype=np.float64)
            if math.isfinite(value_np) and np.isfinite(gradient_np).all():
                stabilized[chain_index] = candidate
                if shrink_index:
                    print(
                        f"Shrank chain {chain_index} initialization toward the MAP "
                        f"{shrink_index} time(s) to obtain finite geometry."
                    )
                break
            candidate = 0.5 * candidate
        else:
            raise FloatingPointError(
                f"Could not obtain a finite initialization for chain {chain_index}."
            )
    return stabilized


def chain_path(output_dir: Path, chain_index: int) -> Path:
    return output_dir / f"chain_{chain_index:02d}.npz"


def save_chain(
    path: Path,
    *,
    samples: np.ndarray,
    z_samples: np.ndarray,
    extras: Mapping[str, np.ndarray],
    init_z: np.ndarray,
    elapsed_seconds: float,
    problem_fingerprint: str,
    config_signature: str,
) -> None:
    payload = {
        "samples": np.asarray(samples, dtype=np.float64),
        "z_samples": np.asarray(z_samples, dtype=np.float64),
        "init_z": np.asarray(init_z, dtype=np.float64),
        "elapsed_seconds": np.asarray(float(elapsed_seconds)),
        "problem_fingerprint": np.asarray(problem_fingerprint),
        "configuration_signature": np.asarray(config_signature),
    }
    for name, values in extras.items():
        payload[f"extra__{name.replace('.', '__')}"] = np.asarray(values)
    atomic_npz(path, **payload)


def load_chain(
    path: Path,
    *,
    problem_fingerprint: str,
    config_signature: str,
    expected_samples: int,
    latent_dim: int,
) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["problem_fingerprint"].item()) != problem_fingerprint:
            raise RuntimeError(f"{path.name} belongs to a different Darcy problem.")
        if str(archive["configuration_signature"].item()) != config_signature:
            raise RuntimeError(f"{path.name} has incompatible sampler configuration.")
        samples = np.asarray(archive["samples"], dtype=np.float64)
        z_samples = np.asarray(archive["z_samples"], dtype=np.float64)
        init_z = np.asarray(archive["init_z"], dtype=np.float64)
        extras = {}
        for name in archive.files:
            if name.startswith("extra__"):
                extras[name[len("extra__") :].replace("__", ".")] = np.asarray(
                    archive[name]
                )
        elapsed_seconds = float(archive["elapsed_seconds"])
    expected_shape = (expected_samples, latent_dim)
    if samples.shape != expected_shape or z_samples.shape != expected_shape:
        raise RuntimeError(
            f"{path.name} sample shapes are alpha={samples.shape}, z={z_samples.shape}; "
            f"expected {expected_shape}."
        )
    if init_z.shape != (latent_dim,):
        raise RuntimeError(
            f"{path.name} initial position has shape {init_z.shape}; expected "
            f"({latent_dim},)."
        )
    if not np.isfinite(samples).all() or not np.isfinite(z_samples).all():
        raise FloatingPointError(f"{path.name} contains non-finite particles.")
    required = {
        "diverging",
        "potential_energy",
        "energy",
        "accept_prob",
        "num_steps",
        "adapt_state.step_size",
    }
    missing = required - set(extras)
    if missing:
        raise RuntimeError(
            f"{path.name} lacks diagnostics: " + ", ".join(sorted(missing))
        )
    malformed = {
        name: np.asarray(values).shape
        for name, values in extras.items()
        if np.asarray(values).shape != (expected_samples,)
    }
    if malformed:
        raise RuntimeError(f"{path.name} has malformed diagnostic arrays: {malformed}.")
    return {
        "samples": samples,
        "z_samples": z_samples,
        "init_z": init_z,
        "elapsed_seconds": elapsed_seconds,
        "extras": extras,
    }


def run_one_chain(
    args: argparse.Namespace,
    *,
    chain_index: int,
    potential_z,
    setup: Mapping[str, object],
    init_z: np.ndarray,
) -> Dict[str, object]:
    kernel = NUTS(
        potential_fn=potential_z,
        dense_mass=not args.diagonal_mass,
        target_accept_prob=float(args.target_accept),
        max_tree_depth=(
            int(args.max_tree_depth_warmup),
            int(args.max_tree_depth),
        ),
        find_heuristic_step_size=True,
        regularize_mass_matrix=True,
        forward_mode_differentiation=False,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(args.num_warmup),
        num_samples=int(args.num_samples),
        thinning=int(args.thinning),
        num_chains=1,
        chain_method="sequential",
        progress_bar=bool(args.progress_bar),
    )
    key = jax.random.PRNGKey(args.sampler_seed + 100_003 * (chain_index + 1))
    print(
        f"\n=== NUTS chain {chain_index + 1}/{args.num_chains} ===\n"
        f"warmup={args.num_warmup}, retained={args.num_samples}, "
        f"target_accept={args.target_accept:.3f}, "
        f"dense_mass={not args.diagonal_mass}"
    )
    started = time.perf_counter()
    mcmc.run(
        key,
        init_params=jnp.asarray(init_z, dtype=jnp.float64),
        extra_fields=(
            "potential_energy",
            "energy",
            "accept_prob",
            "num_steps",
            "adapt_state.step_size",
        ),
    )
    jax.block_until_ready(mcmc.last_state.z)
    elapsed = time.perf_counter() - started
    z_by_chain = np.asarray(
        jax.device_get(mcmc.get_samples(group_by_chain=True)), dtype=np.float64
    )
    if z_by_chain.shape != (1, args.num_samples, args.latent_dim):
        raise RuntimeError(f"Unexpected NUTS sample shape: {z_by_chain.shape}.")
    z_samples = z_by_chain[0]
    map_position = np.asarray(setup["map_position"], dtype=np.float64)
    factor = np.asarray(setup["laplace_factor"], dtype=np.float64)
    samples = map_position[None, :] + z_samples @ factor.T
    extras_raw = mcmc.get_extra_fields(group_by_chain=True)
    extras = {
        name: np.asarray(jax.device_get(values))[0]
        for name, values in extras_raw.items()
    }
    required = {
        "diverging",
        "potential_energy",
        "energy",
        "accept_prob",
        "num_steps",
        "adapt_state.step_size",
    }
    missing = required - set(extras)
    if missing:
        raise RuntimeError(
            "NumPyro did not return required diagnostics: " + ", ".join(sorted(missing))
        )
    if not np.isfinite(samples).all():
        raise FloatingPointError("NUTS returned non-finite posterior samples.")
    print(
        f"Chain {chain_index + 1} complete: wall={elapsed / 60.0:.1f} min, "
        f"divergences={int(np.asarray(extras['diverging']).sum())}, "
        f"mean_accept={float(np.mean(extras['accept_prob'])):.3f}"
    )
    return {
        "samples": samples,
        "z_samples": z_samples,
        "extras": extras,
        "init_z": np.asarray(init_z, dtype=np.float64),
        "elapsed_seconds": elapsed,
    }


def permeability_draw_mean(
    samples: np.ndarray,
    basis: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    output = np.empty(samples.shape[0], dtype=np.float64)
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(samples.shape[0], start + chunk_size)
        log_k = samples[start:stop] @ basis.T
        output[start:stop] = np.mean(np.exp(np.clip(log_k, -60.0, 60.0)), axis=1)
    return output


def permeability_field_mean(
    samples: np.ndarray,
    basis: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    total = np.zeros(basis.shape[0], dtype=np.float64)
    count = 0
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(samples.shape[0], start + chunk_size)
        log_k = samples[start:stop] @ basis.T
        total += np.exp(np.clip(log_k, -60.0, 60.0)).sum(axis=0)
        count += stop - start
    return total / float(count)


def build_derived_draws(
    samples_by_chain: np.ndarray,
    basis: np.ndarray,
    potential_energy: np.ndarray,
    *,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    basis_mean = np.mean(basis, axis=0)
    gram = (basis.T @ basis) / float(basis.shape[0])
    alpha_norm = np.linalg.norm(samples_by_chain, axis=2)
    log_k_spatial_mean = np.einsum("cnd,d->cn", samples_by_chain, basis_mean)
    log_k_spatial_rms = np.sqrt(
        np.maximum(
            np.einsum("cni,ij,cnj->cn", samples_by_chain, gram, samples_by_chain),
            0.0,
        )
    )
    permeability_spatial_mean = np.stack(
        [
            permeability_draw_mean(chain, basis, chunk_size=chunk_size)
            for chain in samples_by_chain
        ]
    )
    return {
        "alpha_norm": alpha_norm,
        "log_k_spatial_mean": log_k_spatial_mean,
        "log_k_spatial_rms": log_k_spatial_rms,
        "permeability_spatial_mean": permeability_spatial_mean,
        "potential_energy": np.asarray(potential_energy, dtype=np.float64),
    }


def arviz_diagnostics(
    samples_by_chain: np.ndarray,
    derived: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    posterior = {"alpha": np.asarray(samples_by_chain, dtype=np.float64)}
    posterior.update({key: np.asarray(value) for key, value in derived.items()})
    inference_data = az.from_dict(posterior=posterior)
    rhat_data = az.rhat(inference_data, method="rank")
    bulk_data = az.ess(inference_data, method="bulk")
    tail_data = az.ess(inference_data, method="tail")
    mcse_data = az.mcse(inference_data, method="mean")

    rows: List[Dict[str, object]] = []
    for variable in posterior:
        rhat = np.asarray(rhat_data[variable].values, dtype=np.float64)
        bulk = np.asarray(bulk_data[variable].values, dtype=np.float64)
        tail = np.asarray(tail_data[variable].values, dtype=np.float64)
        mcse = np.asarray(mcse_data[variable].values, dtype=np.float64)
        shape = rhat.shape
        for flat_index in range(max(1, rhat.size)):
            coordinate = None if shape == () else np.unravel_index(flat_index, shape)
            rows.append(
                {
                    "variable": variable,
                    "coordinate": "" if coordinate is None else ",".join(map(str, coordinate)),
                    "rank_rhat": float(rhat.reshape(-1)[flat_index]),
                    "bulk_ess": float(bulk.reshape(-1)[flat_index]),
                    "tail_ess": float(tail.reshape(-1)[flat_index]),
                    "mcse_mean": float(mcse.reshape(-1)[flat_index]),
                }
            )
    all_rhat = np.asarray([row["rank_rhat"] for row in rows], dtype=np.float64)
    all_bulk = np.asarray([row["bulk_ess"] for row in rows], dtype=np.float64)
    all_tail = np.asarray([row["tail_ess"] for row in rows], dtype=np.float64)
    finite_mask = np.isfinite(all_rhat) & np.isfinite(all_bulk) & np.isfinite(all_tail)
    if finite_mask.any():
        max_rhat = float(np.max(all_rhat[finite_mask]))
        min_bulk = float(np.min(all_bulk[finite_mask]))
        min_tail = float(np.min(all_tail[finite_mask]))
    else:
        max_rhat = float("nan")
        min_bulk = float("nan")
        min_tail = float("nan")
    return (
        {
            "finite": bool(finite_mask.all()),
            "nonfinite_entries": int((~finite_mask).sum()),
            "max_rank_rhat": max_rhat,
            "min_bulk_ess": min_bulk,
            "min_tail_ess": min_tail,
        },
        rows,
    )


def kernel_diagnostics(
    extras_by_name: Mapping[str, np.ndarray],
    *,
    max_tree_depth: int,
) -> Dict[str, object]:
    diverging = np.asarray(extras_by_name["diverging"], dtype=bool)
    accept = np.asarray(extras_by_name["accept_prob"], dtype=np.float64)
    num_steps = np.asarray(extras_by_name["num_steps"], dtype=np.int64)
    energy = np.asarray(extras_by_name["energy"], dtype=np.float64)
    step_size = np.asarray(
        extras_by_name["adapt_state.step_size"], dtype=np.float64
    )
    arrays_finite = bool(
        np.isfinite(accept).all()
        and np.isfinite(energy).all()
        and np.isfinite(step_size).all()
    )
    max_steps = 2 ** int(max_tree_depth) - 1
    per_chain = []
    for chain_index in range(diverging.shape[0]):
        chain_energy = energy[chain_index]
        energy_variance = float(np.var(chain_energy, ddof=1))
        ebfmi = (
            float(np.mean(np.diff(chain_energy) ** 2)) / energy_variance
            if energy_variance > 0.0
            else 0.0
        )
        per_chain.append(
            {
                "chain": chain_index,
                "divergences": int(diverging[chain_index].sum()),
                "mean_accept_prob": float(np.mean(accept[chain_index])),
                "tree_depth_saturation_fraction": float(
                    np.mean(num_steps[chain_index] >= max_steps)
                ),
                "mean_num_steps": float(np.mean(num_steps[chain_index])),
                "max_num_steps": int(np.max(num_steps[chain_index])),
                "ebfmi": ebfmi,
                "adapted_step_size": float(np.median(step_size[chain_index])),
            }
        )
    summary = {
        "finite": arrays_finite
        and all(math.isfinite(float(item["ebfmi"])) for item in per_chain),
        "total_divergences": int(diverging.sum()),
        "max_tree_depth_saturation_fraction": float(
            max(item["tree_depth_saturation_fraction"] for item in per_chain)
        ),
        "min_ebfmi": float(min(item["ebfmi"] for item in per_chain)),
        "min_mean_accept_prob": float(
            min(item["mean_accept_prob"] for item in per_chain)
        ),
        "per_chain": per_chain,
    }
    return summary


def reference_functionals(
    samples: np.ndarray,
    *,
    problem,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    mean = np.mean(samples, axis=0)
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
    log_k_mean = problem.basis_np @ mean
    log_k_variance = np.einsum(
        "gi,ij,gj->g", problem.basis_np, covariance, problem.basis_np
    )
    log_k_std = np.sqrt(np.maximum(log_k_variance, 0.0))
    k_mean = permeability_field_mean(
        samples, problem.basis_np, chunk_size=chunk_size
    )
    holdout = np.asarray(
        problem.solve_forward_holdout(jnp.asarray(mean, dtype=jnp.float64)),
        dtype=np.float64,
    )
    return {
        "mean": mean,
        "covariance": covariance,
        "log_k_mean": log_k_mean,
        "log_k_std": log_k_std,
        "k_mean": k_mean,
        "holdout_pressure_at_mean": holdout,
    }


def relative_l2(value: np.ndarray, reference: np.ndarray, floor: float = 1e-12) -> float:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(value - reference) / (np.linalg.norm(reference) + floor))


def functional_chain_agreement(
    samples_by_chain: np.ndarray,
    *,
    problem,
    chunk_size: int,
) -> Dict[str, object]:
    pooled = reference_functionals(
        samples_by_chain.reshape(-1, samples_by_chain.shape[-1]),
        problem=problem,
        chunk_size=chunk_size,
    )
    covariance = pooled["covariance"]
    eigvals, eigvecs = np.linalg.eigh(0.5 * (covariance + covariance.T))
    ridge = 1e-10 + 1e-8 * float(np.trace(covariance)) / covariance.shape[0]
    eigvals = np.maximum(eigvals, ridge)
    precision_sqrt = (eigvecs / np.sqrt(eigvals)[None, :]) @ eigvecs.T
    rows = []
    for chain_index, samples in enumerate(samples_by_chain):
        summary = reference_functionals(
            samples, problem=problem, chunk_size=chunk_size
        )
        mean_error = precision_sqrt @ (summary["mean"] - pooled["mean"])
        rows.append(
            {
                "chain": chain_index,
                "mean_whitened_rmse": float(
                    np.linalg.norm(mean_error) / math.sqrt(samples.shape[1])
                ),
                "covariance_relative_frobenius": float(
                    np.linalg.norm(summary["covariance"] - pooled["covariance"], ord="fro")
                    / (np.linalg.norm(pooled["covariance"], ord="fro") + 1e-12)
                ),
                "log_k_mean_relative_l2": relative_l2(
                    summary["log_k_mean"], pooled["log_k_mean"]
                ),
                "log_k_std_relative_l2": relative_l2(
                    summary["log_k_std"], pooled["log_k_std"]
                ),
                "k_mean_relative_l2": relative_l2(
                    summary["k_mean"], pooled["k_mean"]
                ),
                "holdout_pressure_relative_l2": relative_l2(
                    summary["holdout_pressure_at_mean"],
                    pooled["holdout_pressure_at_mean"],
                ),
            }
        )
    maxima = {
        key: float(max(row[key] for row in rows))
        for key in rows[0]
        if key != "chain"
    }
    return {"per_chain": rows, "maxima": maxima, "pooled": pooled}


def write_diagnostic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = ["variable", "coordinate", "rank_rhat", "bulk_ess", "tail_ess", "mcse_mean"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    darcy_path = Path(args.darcy_script).resolve()
    darcy_sha256 = hash_file(darcy_path)
    config_signature = configuration_signature(args, darcy_sha256)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = Path("run_results").resolve() / f"darcy_nuts_reference_{stamp}"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --resume or a new path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {output_dir}")
    print(f"Darcy definition: {darcy_path}")
    print(f"JAX backend/devices: {jax.default_backend()} / {jax.devices()}")

    darcy_module = load_darcy_module(darcy_path)
    problem, alpha_true, y_clean, y_obs, problem_fingerprint = build_problem_and_data(
        args, darcy_module
    )
    potential_alpha = make_alpha_potential(problem, y_obs, args.noise_std)

    run_config_path = output_dir / "run_config.json"
    previous = None
    if args.resume:
        if run_config_path.exists():
            previous = json.loads(run_config_path.read_text(encoding="utf-8"))
            if previous.get("problem_fingerprint") != problem_fingerprint:
                raise RuntimeError(
                    "Resume directory was created for a different Darcy problem."
                )
            if previous.get("configuration_signature") != config_signature:
                raise RuntimeError(
                    "Resume directory has incompatible sampler configuration."
                )
        elif args.chain_index is not None or args.finalize_only:
            raise FileNotFoundError(
                f"Staged worker requires the prepared configuration: {run_config_path}"
            )
        elif any(output_dir.iterdir()):
            raise RuntimeError(
                "Refusing to resume a nonempty directory without run_config.json. "
                "Choose an empty directory or recover the original configuration file."
            )

    config = {
        "script_version": SCRIPT_VERSION,
        "reference_format_version": REFERENCE_FORMAT_VERSION,
        "created_at": (
            previous.get("created_at")
            if previous is not None and previous.get("created_at")
            else datetime.now().isoformat(timespec="seconds")
        ),
        "last_resumed_at": (
            datetime.now().isoformat(timespec="seconds") if previous is not None else None
        ),
        "arguments": vars(args),
        "darcy_script": str(darcy_path),
        "darcy_script_sha256": darcy_sha256,
        "problem_fingerprint": problem_fingerprint,
        "configuration_signature": config_signature,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "arviz_version": az.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "truth_usage_policy": (
            "alpha_true is used only to reconstruct the benchmark observations "
            "and is never supplied to MAP optimization or NUTS initialization."
        ),
    }
    # The preparation process owns shared run metadata. Distributed chain
    # workers only read it, avoiding cross-node atomic-renaming races.
    if previous is None:
        atomic_json(run_config_path, config)
        atomic_npz(
            output_dir / "problem_data.npz",
            alpha_true=alpha_true,
            y_clean=y_clean,
            y_obs=y_obs,
            basis=problem.basis_np,
            obs_locations_train=problem.obs_locations_train,
            obs_locations_holdout=problem.obs_locations_holdout,
            problem_fingerprint=np.asarray(problem_fingerprint),
        )

    setup_path = output_dir / "setup.npz"
    if args.resume and setup_path.exists():
        print("Loading compatible MAP/Laplace setup from setup.npz...")
        setup = load_setup(
            setup_path,
            problem_fingerprint=problem_fingerprint,
            config_signature=config_signature,
        )
    elif args.chain_index is not None or args.finalize_only:
        raise FileNotFoundError(
            f"Staged worker requires the prepared setup file: {setup_path}"
        )
    else:
        setup = find_map_and_laplace(args, potential_alpha)
        save_setup(
            setup_path,
            setup,
            problem_fingerprint=problem_fingerprint,
            config_signature=config_signature,
        )

    if args.prepare_only:
        print(f"\nPreparation complete: {setup_path}")
        print("No NUTS chain was started (--prepare-only).")
        return

    if args.finalize_only:
        potential_z = None
        initial_positions = None
        selected_chain_indices = range(args.num_chains)
    else:
        potential_z = make_whitened_potential(potential_alpha, setup)
        initial_positions = make_chain_initial_positions(args, setup)
        initial_positions = stabilize_chain_initial_positions(
            initial_positions,
            potential_z,
        )
        selected_chain_indices = (
            [args.chain_index]
            if args.chain_index is not None
            else range(args.num_chains)
        )

    chain_results = []
    for chain_index in selected_chain_indices:
        path = chain_path(output_dir, chain_index)
        if args.finalize_only and not path.exists():
            raise FileNotFoundError(
                f"Cannot finalize: required {path.name} is missing. Rerun its "
                f"worker with --resume --chain-index {chain_index}."
            )
        if args.resume and path.exists():
            print(f"Loading completed {path.name}...")
            result = load_chain(
                path,
                problem_fingerprint=problem_fingerprint,
                config_signature=config_signature,
                expected_samples=args.num_samples,
                latent_dim=args.latent_dim,
            )
        else:
            result = run_one_chain(
                args,
                chain_index=chain_index,
                potential_z=potential_z,
                setup=setup,
                init_z=initial_positions[chain_index],
            )
            save_chain(
                path,
                samples=result["samples"],
                z_samples=result["z_samples"],
                extras=result["extras"],
                init_z=result["init_z"],
                elapsed_seconds=result["elapsed_seconds"],
                problem_fingerprint=problem_fingerprint,
                config_signature=config_signature,
            )
        chain_results.append(result)

    if args.chain_index is not None:
        print(
            f"\nDistributed worker complete: chain {args.chain_index} is saved at "
            f"{chain_path(output_dir, args.chain_index)}"
        )
        return

    samples_by_chain = np.stack([result["samples"] for result in chain_results])
    z_samples_by_chain = np.stack([result["z_samples"] for result in chain_results])
    flat_samples = samples_by_chain.reshape(-1, args.latent_dim)
    if not np.isfinite(flat_samples).all():
        raise FloatingPointError("Combined reference cloud contains non-finite values.")

    extra_names = set(chain_results[0]["extras"])
    if any(set(result["extras"]) != extra_names for result in chain_results):
        raise RuntimeError("NUTS chains returned inconsistent diagnostic fields.")
    extras_by_name = {
        name: np.stack([np.asarray(result["extras"][name]) for result in chain_results])
        for name in sorted(extra_names)
    }
    derived = build_derived_draws(
        samples_by_chain,
        problem.basis_np,
        extras_by_name["potential_energy"],
        chunk_size=args.functional_chunk_size,
    )
    convergence, diagnostic_rows = arviz_diagnostics(samples_by_chain, derived)
    kernel = kernel_diagnostics(
        extras_by_name, max_tree_depth=args.max_tree_depth
    )
    agreement = functional_chain_agreement(
        samples_by_chain,
        problem=problem,
        chunk_size=args.functional_chunk_size,
    )

    checks = {
        "convergence_diagnostics_finite": convergence["finite"],
        "kernel_diagnostics_finite": kernel["finite"],
        "rank_rhat": convergence["max_rank_rhat"] <= args.max_rhat,
        "bulk_ess": convergence["min_bulk_ess"] >= args.min_bulk_ess,
        "tail_ess": convergence["min_tail_ess"] >= args.min_tail_ess,
        "divergences": kernel["total_divergences"] <= args.max_divergences,
        "tree_depth": (
            kernel["max_tree_depth_saturation_fraction"]
            <= args.max_tree_depth_fraction
        ),
        "ebfmi": kernel["min_ebfmi"] >= args.min_ebfmi,
        "acceptance": kernel["min_mean_accept_prob"] >= args.min_mean_accept,
    }
    certified = bool(all(checks.values()))
    diagnostics = {
        "certified": certified,
        "checks": checks,
        "thresholds": {
            "max_rhat": args.max_rhat,
            "min_bulk_ess": args.min_bulk_ess,
            "min_tail_ess": args.min_tail_ess,
            "max_divergences": args.max_divergences,
            "max_tree_depth_fraction": args.max_tree_depth_fraction,
            "min_ebfmi": args.min_ebfmi,
            "min_mean_accept": args.min_mean_accept,
        },
        "convergence": convergence,
        "kernel": kernel,
        "functional_chain_agreement": {
            "per_chain": agreement["per_chain"],
            "maxima": agreement["maxima"],
        },
        "num_chains": args.num_chains,
        "samples_per_chain": args.num_samples,
        "total_reference_particles": int(flat_samples.shape[0]),
        "total_chain_wall_seconds": float(
            sum(float(result["elapsed_seconds"]) for result in chain_results)
        ),
    }
    atomic_json(output_dir / "nuts_diagnostics.json", diagnostics)
    write_diagnostic_csv(output_dir / "nuts_parameter_diagnostics.csv", diagnostic_rows)

    pooled_functionals = agreement["pooled"]
    atomic_npz(
        output_dir / "reference_functionals.npz",
        **{key: np.asarray(value) for key, value in pooled_functionals.items()},
    )
    source = (
        f"Independent NumPyro NUTS: {args.num_chains} chains x "
        f"{args.num_samples} retained draws; dense_mass={not args.diagonal_mass}; "
        f"target_accept={args.target_accept}; certified={certified}"
    )
    reference_payload = {
        "samples": flat_samples,
        "samples_by_chain": samples_by_chain,
        "z_samples_by_chain": z_samples_by_chain,
        "source": np.asarray(source),
        "certified": np.asarray(certified),
        "problem_fingerprint": np.asarray(problem_fingerprint),
        "configuration_signature": np.asarray(config_signature),
        "data_seed": np.asarray(args.data_seed),
        "sampler_seed": np.asarray(args.sampler_seed),
        "alpha_true": alpha_true,
        "map_position": np.asarray(setup["map_position"]),
    }
    uncertified_path = output_dir / "reference_samples_uncertified.npz"
    atomic_npz(uncertified_path, **reference_payload)
    canonical_path = output_dir / "reference_samples.npz"
    if certified or args.allow_failed_diagnostics:
        atomic_npz(canonical_path, **reference_payload)
    elif canonical_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        withdrawn = output_dir / f"reference_samples_withdrawn_{timestamp}.npz"
        os.replace(canonical_path, withdrawn)
        print(
            "A canonical reference from an earlier certification policy was "
            f"withdrawn to {withdrawn.name} because the current checks failed."
        )

    print("\n=== Offline NUTS reference diagnostics ===")
    print(f"Rank-normalized R-hat max: {convergence['max_rank_rhat']:.5f}")
    print(f"Bulk ESS min:              {convergence['min_bulk_ess']:.1f}")
    print(f"Tail ESS min:              {convergence['min_tail_ess']:.1f}")
    print(f"Divergences:               {kernel['total_divergences']}")
    print(
        "Max tree-depth fraction:    "
        f"{100.0 * kernel['max_tree_depth_saturation_fraction']:.3f}%"
    )
    print(f"Minimum E-BFMI:            {kernel['min_ebfmi']:.3f}")
    print(f"Minimum mean acceptance:   {kernel['min_mean_accept_prob']:.3f}")
    print("Checks: " + ", ".join(f"{key}={'PASS' if value else 'FAIL'}" for key, value in checks.items()))

    if certified:
        print(f"\nCERTIFIED reference: {canonical_path}")
        print(f"Diagnostics: {output_dir / 'nuts_diagnostics.json'}")
        print(f"Functionals: {output_dir / 'reference_functionals.npz'}")
        return
    if args.allow_failed_diagnostics:
        print(
            "\nWARNING: diagnostics failed, but --allow-failed-diagnostics wrote "
            f"an explicitly uncertified canonical file: {canonical_path}"
        )
        return
    print(
        "\nREFERENCE NOT CERTIFIED. Samples were preserved at "
        f"{uncertified_path}. Increase warmup/draws, raise target acceptance, "
        "or inspect the failed chains; reference_samples.npz was not written."
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
