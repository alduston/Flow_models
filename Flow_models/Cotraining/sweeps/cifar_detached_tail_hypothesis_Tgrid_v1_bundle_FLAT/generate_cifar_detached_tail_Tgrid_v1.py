#!/usr/bin/env python3
"""Generate the 18-cell CIFAR detached-tail hypothesis manifest.

Scientific design from CSEM report §17.2:
  T in {1.45, 1.60, 1.75}
  DeltaT = T - T_K in {0.30, 0.40, 0.50}
  fixed near-optimal w_C=0.05, w_K=0.60
  two independent seeds {42,43}
  exactly 500 joint epochs, zero score-only refinement
"""
from __future__ import annotations
from pathlib import Path
import csv

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep")
OUT = BASE_DIR / "cifar_detached_tail_Tgrid_v1_manifest.csv"

T_VALUES = (1.45, 1.60, 1.75)
DELTA_VALUES = (0.30, 0.40, 0.50)
SEEDS = (42, 43)
CSEM_W = 0.05
TERMINAL_KL_W = 0.60

FIELDS = [
    "cell_id", "rep_id", "rep_role", "seed",
    "T_K", "T_full", "delta_T", "tail_fraction", "TK_fraction",
    "csem_w", "terminal_kl_w",
    "epochs", "refine_epochs", "eval_every", "eval_samples",
    "oracle_step_grid", "oracle_sampling_samples",
    "oracle_profile_query_samples", "oracle_profile_time_points",
    "result_name",
]


def ftag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def main() -> int:
    rows = []
    cell = 0
    for T in T_VALUES:
        for delta in DELTA_VALUES:
            TK = T - delta
            if not (0.0 < TK < T):
                raise RuntimeError(f"Invalid geometry T={T}, delta={delta}, TK={TK}")
            for seed in SEEDS:
                rep = f"tail_T{ftag(T)}_d{ftag(delta)}_TK{ftag(TK)}_s{seed}"
                rows.append({
                    "cell_id": f"{cell:02d}",
                    "rep_id": rep,
                    "rep_role": "detached-tail geometry test: fixed CSEM/KL, vary full horizon and tail length",
                    "seed": str(seed),
                    "T_K": f"{TK:.2f}",
                    "T_full": f"{T:.2f}",
                    "delta_T": f"{delta:.2f}",
                    "tail_fraction": f"{delta/T:.8f}",
                    "TK_fraction": f"{TK/T:.8f}",
                    "csem_w": f"{CSEM_W:.2f}",
                    "terminal_kl_w": f"{TERMINAL_KL_W:.2f}",
                    "epochs": "500",
                    "refine_epochs": "0",
                    "eval_every": "0",
                    "eval_samples": "10000",
                    "oracle_step_grid": "5,10,25,50",
                    "oracle_sampling_samples": "256",
                    "oracle_profile_query_samples": "64",
                    "oracle_profile_time_points": "16",
                    "result_name": rep,
                })
                cell += 1

    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 rows, made {len(rows)}")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} cells to {OUT}")
    for r in rows:
        print(
            f"cell {r['cell_id']} | T={r['T_full']} | dT={r['delta_T']} | "
            f"TK={r['T_K']} | seed={r['seed']} | c={r['csem_w']} | k={r['terminal_kl_w']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
