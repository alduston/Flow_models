#!/usr/bin/env python3
"""Train one detached-tail geometry cell, then run frozen oracle/NFE diagnostics."""
from __future__ import annotations
import argparse, csv, json, math, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep")
MANIFEST = "cifar_detached_tail_Tgrid_v1_manifest.csv"
TARGET = "csem_detached_tail_Tgrid_v1.py"
RESULTS_ROOT = "cifar_detached_tail_Tgrid_v1_results"
LOG_ROOT = "cifar_detached_tail_Tgrid_v1_config_logs"
STATUS_ROOT = "cifar_detached_tail_Tgrid_v1_status"
N_CELLS = 18
EVAL_EPOCH_LABEL = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cell(path: Path, cell_id: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    hit = [r for r in rows if int(r["cell_id"]) == cell_id]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one row for cell {cell_id}, found {len(hit)}")
    return hit[0]


def read_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return int(json.loads(path.read_text()).get("returncode", 999999)) == 0
    except Exception:
        return False


def tee_process(command: list[str], log_path: Path, cwd: Path) -> tuple[int, float, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    t0 = time.monotonic()
    with log_path.open("w", buffering=1) as log:
        log.write("COMMAND:\n" + shlex.join(command) + "\n\n")
        proc = subprocess.Popen(
            command, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
    return rc, time.monotonic() - t0, started, utc_now()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def validate_contract(row: dict[str, str]) -> None:
    T = float(row["T_full"])
    TK = float(row["T_K"])
    delta = float(row["delta_T"])
    if T not in (1.45, 1.60, 1.75):
        raise RuntimeError(f"Unexpected T: {row}")
    if delta not in (0.30, 0.40, 0.50):
        raise RuntimeError(f"Unexpected delta_T: {row}")
    if not math.isclose(T - TK, delta, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(f"delta_T != T-T_K: {row}")
    if not math.isclose(float(row["csem_w"]), 0.05, abs_tol=1e-12):
        raise RuntimeError(f"CSEM must be fixed at .05: {row}")
    if not math.isclose(float(row["terminal_kl_w"]), 0.60, abs_tol=1e-12):
        raise RuntimeError(f"terminal KL must be fixed at .60: {row}")
    if int(row["epochs"]) != 500 or int(row["refine_epochs"]) != 0:
        raise RuntimeError(f"Training protocol must be 500 + 0: {row}")
    if int(row["seed"]) not in (42, 43):
        raise RuntimeError(f"Unexpected seed: {row}")


def shared_args(row: dict[str, str]) -> list[str]:
    return [
        "--dataset", "CIFAR", "--model-preset", "auto", "--arms", "terminal_kl",
        "--seed", row["seed"],
        "--score-time-weighting", "canonical",
        "--score-head-time-weighting", "unweighted-eps",
        "--csem-w", row["csem_w"], "--terminal-kl-w", row["terminal_kl_w"],
        "--score-head-loss-w", "1.0",
        "--T-terminal", row["T_K"], "--T", row["T_full"],
        "--cfg-strength", "3.0", "--canonical-lr-scale", "1.0",
        "--encoder-score-warmup-epochs", "0", "--csem-ramp-epochs", "0",
        "--score-tracking-steps", "0", "--grad-diagnostics-every", "0",
        "--logvar-min=-30.0", "--logvar-max", "20.0",
        "--no-fail-on-nonfinite", "--no-bespoke-fid-classifier",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", type=int, required=True)
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    args = ap.parse_args()
    if args.cell_id < 0 or args.cell_id >= N_CELLS:
        raise SystemExit(f"cell-id must be in 0..{N_CELLS - 1}")

    base = args.base_dir.resolve()
    manifest = base / MANIFEST
    target = base / TARGET
    for p in (manifest, target):
        if not p.is_file():
            raise SystemExit(f"Missing required file: {p}")
    row = load_cell(manifest, args.cell_id)
    validate_contract(row)

    rep_root = base / RESULTS_ROOT / row["result_name"]
    train_root = rep_root / "training"
    mech_root = rep_root / "mechanism_eval"
    train_ckpt = train_root / "run_terminal_kl" / "checkpoints"
    log_root = base / LOG_ROOT / row["result_name"]
    status_root = base / STATUS_ROOT / row["result_name"]
    overall_status = status_root / "overall.json"
    train_status = status_root / "training.json"
    mech_status = status_root / "mechanism_eval.json"

    expected_curve = mech_root / "run_terminal_kl" / "dataframes" / f"oracle_sampling_decomposition_ep{EVAL_EPOCH_LABEL}.csv"
    expected_profile = mech_root / "run_terminal_kl" / "dataframes" / f"oracle_score_time_profile_ep{EVAL_EPOCH_LABEL}.csv"
    if read_ok(overall_status) and expected_curve.is_file() and expected_profile.is_file():
        print(f"[skip] {row['rep_id']} already complete.")
        return 0

    print("=" * 110)
    print("CIFAR DETACHED-TAIL HYPOTHESIS — T x DELTA_T, 2 SEEDS")
    print(f"cell / rep        = {args.cell_id:02d} / {row['rep_id']}")
    print(f"seed              = {row['seed']}")
    print(f"T / delta_T / TK  = {row['T_full']} / {row['delta_T']} / {row['T_K']}")
    print(f"TK/T              = {row['TK_fraction']}")
    print(f"csem / boundaryKL = {row['csem_w']} / {row['terminal_kl_w']} (FIXED)")
    print(f"training          = {row['epochs']} joint + {row['refine_epochs']} refinement")
    print("hypothesis        = tail length, not absolute TK, is the stable Gaussianization control")
    print("=" * 110)

    train_ckpts = [train_ckpt / "vae_cotrained.pt", train_ckpt / "unet_lsi.pt"]
    train_complete = read_ok(train_status) and all(p.is_file() for p in train_ckpts)
    if not train_complete:
        if train_root.exists():
            raise SystemExit(
                f"[blocked] Partial/existing training directory: {train_root}\n"
                "Remove/rename it explicitly before restarting this cell."
            )
        train_cmd = [
            sys.executable, "-u", str(target), *shared_args(row),
            "--epochs", row["epochs"], "--refine-epochs", row["refine_epochs"],
            "--eval-every", row["eval_every"], "--eval-samples", row["eval_samples"],
            "--no-eval-oracle-diagnostics", "--no-eval-oracle-transport-decomposition",
            "--master-results-dir", str(train_root),
        ]
        print("\n== Stage 1/2: 500-epoch geometry training from scratch ==")
        rc, elapsed, started, finished = tee_process(train_cmd, log_root / "training.log", base)
        write_status(train_status, {
            "stage": "training", "cell_id": args.cell_id, "rep_id": row["rep_id"],
            "seed": int(row["seed"]), "T_K": float(row["T_K"]),
            "T_full": float(row["T_full"]), "delta_T": float(row["delta_T"]),
            "returncode": rc, "elapsed_seconds": elapsed,
            "started_utc": started, "finished_utc": finished, "command": train_cmd,
        })
        if rc != 0:
            write_status(overall_status, {"returncode": rc, "failed_stage": "training", "rep_id": row["rep_id"]})
            return rc
        missing = [str(p) for p in train_ckpts if not p.is_file()]
        if missing:
            write_status(overall_status, {"returncode": 3, "failed_stage": "checkpoint_validation", "missing": missing})
            raise SystemExit("[error] training succeeded but checkpoint(s) missing:\n  " + "\n  ".join(missing))
    else:
        print("[skip] training already complete; reusing this sweep's checkpoint.")

    mech_complete = read_ok(mech_status) and expected_curve.is_file() and expected_profile.is_file()
    if not mech_complete:
        if mech_root.exists():
            raise SystemExit(
                f"[blocked] Partial/existing mechanism directory: {mech_root}\n"
                "Remove/rename only that directory before rerunning; training is retained."
            )
        mech_cmd = [
            sys.executable, "-u", str(target), *shared_args(row),
            "--epochs", "1", "--refine-epochs", "0", "--eval-every", "0", "--eval-samples", "1000",
            "--eval-oracle-diagnostics", "--eval-oracle-full-train-reference",
            "--oracle-profile-query-samples", row["oracle_profile_query_samples"],
            "--oracle-profile-time-points", row["oracle_profile_time_points"],
            "--oracle-profile-batch-size", "16", "--oracle-reference-batch-size", "2048",
            "--oracle-sampling-samples", row["oracle_sampling_samples"],
            "--oracle-sampling-batch-size", "16",
            "--oracle-sampling-step-grid", row["oracle_step_grid"],
            "--eval-oracle-transport-decomposition", "--no-eval-oracle-standard-samplers",
            "--evaluation-only-checkpoint-dir", str(train_ckpt),
            "--oracle-eval-epoch-label", str(EVAL_EPOCH_LABEL),
            "--master-results-dir", str(mech_root),
        ]
        print("\n== Stage 2/2: frozen oracle/modelability/NFE tail audit ==")
        rc, elapsed, started, finished = tee_process(mech_cmd, log_root / "mechanism_eval.log", base)
        write_status(mech_status, {
            "stage": "mechanism_eval", "cell_id": args.cell_id, "rep_id": row["rep_id"],
            "seed": int(row["seed"]), "T_K": float(row["T_K"]),
            "T_full": float(row["T_full"]), "delta_T": float(row["delta_T"]),
            "returncode": rc, "elapsed_seconds": elapsed,
            "started_utc": started, "finished_utc": finished,
            "oracle_step_grid": row["oracle_step_grid"], "command": mech_cmd,
        })
        if rc != 0:
            write_status(overall_status, {"returncode": rc, "failed_stage": "mechanism_eval", "rep_id": row["rep_id"]})
            return rc
    else:
        print("[skip] mechanism evaluation already complete.")

    missing = [str(p) for p in (expected_curve, expected_profile) if not p.is_file()]
    if missing:
        write_status(overall_status, {"returncode": 4, "failed_stage": "output_validation", "missing": missing})
        raise SystemExit("[error] expected mechanism output(s) missing:\n  " + "\n  ".join(missing))

    train_elapsed = json.loads(train_status.read_text()).get("elapsed_seconds", 0.0)
    mech_elapsed = json.loads(mech_status.read_text()).get("elapsed_seconds", 0.0)
    write_status(overall_status, {
        "returncode": 0, "cell_id": args.cell_id, "rep_id": row["rep_id"],
        "seed": int(row["seed"]), "T_K": float(row["T_K"]),
        "T_full": float(row["T_full"]), "delta_T": float(row["delta_T"]),
        "csem_w": float(row["csem_w"]), "terminal_kl_w": float(row["terminal_kl_w"]),
        "training_elapsed_seconds": train_elapsed, "mechanism_elapsed_seconds": mech_elapsed,
        "finished_utc": utc_now(),
    })
    print(f"[ok] {row['rep_id']} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
