#!/usr/bin/env python3
"""Generate the five-representation from-scratch oracle/NFE mechanism manifest."""
from pathlib import Path
import csv

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
OUT = BASE_DIR / "cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_manifest.csv"

REPS = [
    ("two_stage_c0_k40", "no CSEM representation shaping / two-stage-like control", 0.00, 0.40),
    ("weak_c2_k40", "weak CSEM shaping at matched terminal KL", 0.02, 0.40),
    ("sweet_c5_k40", "near-optimal CSEM shaping at matched terminal KL", 0.05, 0.40),
    ("strong_c8_k40", "strong score-friendly CSEM shaping at matched terminal KL", 0.08, 0.40),
    ("highkl_c5_k80", "same CSEM as sweet spot with stronger terminal Gaussianization", 0.05, 0.80),
]

FIELDS = [
    "cell_id","rep_id","rep_role","T_K","T_full","csem_w","terminal_kl_w",
    "epochs","refine_epochs","eval_every","eval_samples","oracle_step_grid",
    "oracle_sampling_samples","oracle_profile_query_samples","oracle_profile_time_points",
    "result_name",
]


def main() -> int:
    rows=[]
    for i,(rep,role,csem,kl) in enumerate(REPS):
        rows.append({
            "cell_id": f"{i:02d}",
            "rep_id": rep,
            "rep_role": role,
            "T_K": "1.20",
            "T_full": "1.60",
            "csem_w": f"{csem:.2f}",
            "terminal_kl_w": f"{kl:.2f}",
            "epochs": "500",
            "refine_epochs": "100",
            "eval_every": "50",
            "eval_samples": "10000",
            "oracle_step_grid": "5,10,25,50",
            "oracle_sampling_samples": "256",
            "oracle_profile_query_samples": "64",
            "oracle_profile_time_points": "16",
            "result_name": rep,
        })
    with OUT.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} representation cells to {OUT}")
    for r in rows:
        print(
            f"cell {r['cell_id']} | {r['rep_id']} | csem={r['csem_w']} | "
            f"KL={r['terminal_kl_w']} | RK4={r['oracle_step_grid']}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
