#!/usr/bin/env python3
"""Generate the 20-cell CSEM-vs-standard-two-stage Pareto dominance manifest."""
from __future__ import annotations
from pathlib import Path
import csv

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep")
OUT = BASE_DIR / "cifar_csem_vs_twostage_pareto_T1p6_20run_v1_manifest.csv"

# Exactly the protocol proposed in-chat:
#   standard two-stage LDM: T_K=0, csem=0, sweep time-zero KL beta_0
#   CSEM: T_K=1.2, T=1.6, boundary KL=.40, sweep csem strength
# Two paired seeds per operating point = 20 independent 500-epoch trainings.
SEEDS = (42, 43)
TWO_STAGE_BETA0 = (0.00, 0.01, 0.03, 0.07, 0.15, 0.30)
CSEM_WEIGHTS = (0.02, 0.04, 0.08, 0.15)

FIELDS = [
    "cell_id", "family", "rep_id", "rep_role", "seed",
    "pareto_lever", "pareto_value", "T_K", "T_full", "csem_w", "terminal_kl_w",
    "epochs", "refine_epochs", "eval_every", "eval_samples",
    "oracle_step_grid", "oracle_sampling_samples", "oracle_profile_query_samples",
    "oracle_profile_time_points", "result_name",
]


def fmt_code(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", "p")


def main() -> int:
    rows: list[dict[str, str]] = []

    for beta0 in TWO_STAGE_BETA0:
        for seed in SEEDS:
            rep = f"twostage_tk0_b{fmt_code(beta0)}_s{seed}"
            rows.append({
                "family": "two_stage",
                "rep_id": rep,
                "rep_role": "standard two-stage LDM: reconstruction + beta0*K0; detached score training",
                "seed": str(seed),
                "pareto_lever": "beta0_time_zero_KL",
                "pareto_value": f"{beta0:.6f}",
                "T_K": "0.0",
                "T_full": "1.60",
                "csem_w": "0.0",
                "terminal_kl_w": f"{beta0:.6f}",
                "epochs": "500",
                "refine_epochs": "0",
                # Final evaluation only: all 20 runs spend their training budget on the 500 epochs.
                "eval_every": "0",
                "eval_samples": "10000",
                "oracle_step_grid": "5,10,25,50",
                "oracle_sampling_samples": "256",
                "oracle_profile_query_samples": "64",
                "oracle_profile_time_points": "16",
                "result_name": rep,
            })

    for csem in CSEM_WEIGHTS:
        for seed in SEEDS:
            rep = f"csem_tk1p2_c{fmt_code(csem)}_k0p400_s{seed}"
            rows.append({
                "family": "csem",
                "rep_id": rep,
                "rep_role": "partial-joint CSEM at T_K=1.2 with matched boundary KL=.40",
                "seed": str(seed),
                "pareto_lever": "csem_weight",
                "pareto_value": f"{csem:.6f}",
                "T_K": "1.20",
                "T_full": "1.60",
                "csem_w": f"{csem:.6f}",
                "terminal_kl_w": "0.400000",
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

    if len(rows) != 20:
        raise RuntimeError(f"Scientific contract violated: expected exactly 20 cells, got {len(rows)}")

    # Stable cell IDs after family-specific construction.
    for i, row in enumerate(rows):
        row["cell_id"] = f"{i:02d}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} Pareto cells to {OUT}")
    for r in rows:
        print(
            f"cell {r['cell_id']} | {r['family']:9s} | seed={r['seed']} | "
            f"lever={r['pareto_lever']}={r['pareto_value']} | "
            f"T_K={r['T_K']} csem={r['csem_w']} KL={r['terminal_kl_w']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
