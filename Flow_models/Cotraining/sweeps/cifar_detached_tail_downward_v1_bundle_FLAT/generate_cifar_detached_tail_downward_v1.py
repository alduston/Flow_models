#!/usr/bin/env python3
"""Generate the CIFAR detached-tail downward-extension manifest.

Primary scientific grid:
  T in {1.35, 1.45, 1.55}
  DeltaT = T - T_K in {0.10, 0.20, 0.30, 0.40}
  fixed w_C=0.05, w_K=0.60
  seeds {42,43}
  -> 24 primary trainings

KL-substitution control slice:
  T=1.45
  DeltaT in {0.10,0.20,0.30,0.40}
  fixed w_C=0.05, w_K=0.40
  seeds {42,43}
  -> 8 control trainings

Total: 32 independent 500-epoch trainings, zero score-only refinement.
"""
from __future__ import annotations
from pathlib import Path
import csv

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep")
OUT = BASE_DIR / "cifar_detached_tail_downward_v1_manifest.csv"

PRIMARY_T = (1.35, 1.45, 1.55)
DELTA_VALUES = (0.10, 0.20, 0.30, 0.40)
SEEDS = (42, 43)
CSEM_W = 0.05
PRIMARY_KL = 0.60
CONTROL_T = 1.45
CONTROL_KL = 0.40

FIELDS = [
    "cell_id", "sweep_group", "rep_id", "rep_role", "seed",
    "T_K", "T_full", "delta_T", "tail_fraction", "TK_fraction",
    "csem_w", "terminal_kl_w",
    "epochs", "refine_epochs", "eval_every", "eval_samples",
    "oracle_step_grid", "oracle_sampling_samples",
    "oracle_profile_query_samples", "oracle_profile_time_points",
    "result_name",
]


def ftag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def make_row(cell: int, group: str, T: float, delta: float, wK: float, seed: int) -> dict[str, str]:
    TK = T - delta
    if not (0.0 < TK < T):
        raise RuntimeError(f"Invalid geometry T={T}, delta={delta}, TK={TK}")
    prefix = "down" if group == "primary" else "ctrl"
    rep = f"{prefix}_T{ftag(T)}_d{ftag(delta)}_TK{ftag(TK)}_k{ftag(wK)}_s{seed}"
    role = (
        "downward detached-tail extension at wK=.60"
        if group == "primary"
        else "wK=.40 control at T=1.45 testing KL-tail substitution"
    )
    return {
        "cell_id": f"{cell:02d}",
        "sweep_group": group,
        "rep_id": rep,
        "rep_role": role,
        "seed": str(seed),
        "T_K": f"{TK:.2f}",
        "T_full": f"{T:.2f}",
        "delta_T": f"{delta:.2f}",
        "tail_fraction": f"{delta/T:.8f}",
        "TK_fraction": f"{TK/T:.8f}",
        "csem_w": f"{CSEM_W:.2f}",
        "terminal_kl_w": f"{wK:.2f}",
        "epochs": "500",
        "refine_epochs": "0",
        "eval_every": "0",
        "eval_samples": "10000",
        "oracle_step_grid": "5,10,25,50",
        "oracle_sampling_samples": "256",
        "oracle_profile_query_samples": "64",
        "oracle_profile_time_points": "16",
        "result_name": rep,
    }


def main() -> int:
    rows: list[dict[str, str]] = []
    cell = 0
    for T in PRIMARY_T:
        for delta in DELTA_VALUES:
            for seed in SEEDS:
                rows.append(make_row(cell, "primary", T, delta, PRIMARY_KL, seed))
                cell += 1
    for delta in DELTA_VALUES:
        for seed in SEEDS:
            rows.append(make_row(cell, "kl_control", CONTROL_T, delta, CONTROL_KL, seed))
            cell += 1

    if len(rows) != 32:
        raise RuntimeError(f"Expected 32 rows, made {len(rows)}")
    if sum(r["sweep_group"] == "primary" for r in rows) != 24:
        raise RuntimeError("Expected 24 primary cells")
    if sum(r["sweep_group"] == "kl_control" for r in rows) != 8:
        raise RuntimeError("Expected 8 KL-control cells")

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    print(f"Wrote {len(rows)} cells to {OUT}")
    print("Primary cells 00-23: wK=.60, T={1.35,1.45,1.55}, dT={.10,.20,.30,.40}, seeds 42/43")
    print("Control cells 24-31: wK=.40, T=1.45, dT={.10,.20,.30,.40}, seeds 42/43")
    for r in rows:
        print(
            f"cell {r['cell_id']} | {r['sweep_group']:10s} | T={r['T_full']} | "
            f"dT={r['delta_T']} | TK={r['T_K']} | k={r['terminal_kl_w']} | seed={r['seed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
