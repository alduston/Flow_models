#!/usr/bin/env python3
"""Generate the CIFAR T_K sweep manifest with fixed full horizon T=1.75."""

from pathlib import Path
import csv

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
OUT = BASE_DIR / "cifar_TK_sweep_T1p75_v1_manifest.csv"
TK_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

FIELDS = ['cell_id', 'T_K', 'T_full', 'csem_w', 'terminal_kl_w', 'score_head_loss_w', 'outer_time_weighting', 'head_time_weighting', 'epochs', 'refine_epochs', 'eval_every', 'eval_samples', 'cfg_strength', 'result_name']


def tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def main() -> int:
    rows = []
    for i, tk in enumerate(TK_VALUES):
        rows.append({
            "cell_id": f"{i:02d}",
            "T_K": f"{tk:.2f}",
            "T_full": "1.75",
            "csem_w": "0.10",
            "terminal_kl_w": "0.30",
            "score_head_loss_w": "1.0",
            "outer_time_weighting": "canonical",
            "head_time_weighting": "unweighted-eps",
            "epochs": "500",
            "refine_epochs": "100",
            "eval_every": "50",
            "eval_samples": "10000",
            "cfg_strength": "3.0",
            "result_name": f"TK_{tag(tk)}",
        })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} sweep cells to {OUT}")
    for row in rows:
        print(
            f"cell {row['cell_id']} | T_K={row['T_K']} | "
            f"T={row['T_full']} | result={row['result_name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
