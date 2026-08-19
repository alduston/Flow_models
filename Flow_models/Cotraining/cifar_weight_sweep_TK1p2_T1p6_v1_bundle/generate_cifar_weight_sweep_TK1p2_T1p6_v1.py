#!/usr/bin/env python3
"""Generate the 20-cell CIFAR CSEM/KL weight sweep at fixed T_K=1.2, T=1.6."""

from pathlib import Path
import csv

BASE_DIR = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0')
OUT = BASE_DIR / 'cifar_weight_sweep_TK1p2_T1p6_v1_manifest.csv'
CSEM_VALUES = [0.00, 0.02, 0.04, 0.06, 0.08]
TERMINAL_KL_VALUES = [0.10, 0.20, 0.30, 0.40]

FIELDS = ['cell_id', 'T_K', 'T_full', 'csem_w', 'terminal_kl_w', 'score_head_loss_w', 'outer_time_weighting', 'head_time_weighting', 'epochs', 'refine_epochs', 'eval_every', 'eval_samples', 'cfg_strength', 'result_name']


def tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def main() -> int:
    rows = []
    cell_id = 0
    for csem_w in CSEM_VALUES:
        for terminal_kl_w in TERMINAL_KL_VALUES:
            rows.append({
                "cell_id": f"{cell_id:02d}",
                "T_K": "1.20",
                "T_full": "1.60",
                "csem_w": f"{csem_w:.2f}",
                "terminal_kl_w": f"{terminal_kl_w:.2f}",
                "score_head_loss_w": "1.0",
                "outer_time_weighting": "canonical",
                "head_time_weighting": "unweighted-eps",
                "epochs": "500",
                "refine_epochs": "100",
                "eval_every": "50",
                "eval_samples": "10000",
                "cfg_strength": "3.0",
                "result_name": f"cw_{tag(csem_w)}_kl_{tag(terminal_kl_w)}",
            })
            cell_id += 1

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} sweep cells to {OUT}")
    for row in rows:
        print(
            f"cell {row['cell_id']} | csem_w={row['csem_w']} | "
            f"terminal_kl_w={row['terminal_kl_w']} | "
            f"T_K={row['T_K']} | T={row['T_full']} | result={row['result_name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
