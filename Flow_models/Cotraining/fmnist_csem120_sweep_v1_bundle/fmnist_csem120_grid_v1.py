#!/usr/bin/env python3
"""
Build the exact 120-cell FashionMNIST CSEM screening manifest.

Grid:
  outer representation metric:  unweighted-eps, canonical
  score-head metric:             unweighted-eps, canonical
  shared csem_w = terminal_kl_w: 0.05, 0.10, 0.20, 0.40, 0.60, 1.00
  score-head LR:                 5e-5, 1e-4, 2e-4, 4e-4, 8e-4

Bundling:
  20 ordinary Slurm jobs.
  Each bundle fixes (outer metric, head metric, score-head LR)
  and runs the six representation coefficients sequentially.
  No Slurm job arrays are used.
"""

from __future__ import annotations

import csv
from pathlib import Path

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep")
MANIFEST_NAME = "fmnist_csem120_manifest_v1.csv"

OUTER_METRICS = ("unweighted-eps", "canonical")
HEAD_METRICS = ("unweighted-eps", "canonical")
REP_WEIGHTS = (0.05, 0.10, 0.20, 0.40, 0.60, 1.00)
SCORE_HEAD_LRS = (5e-5, 1e-4, 2e-4, 4e-4, 8e-4)

T_TERMINAL = 1.50
EPOCHS = 120
REFINE_EPOCHS = 0
EVAL_EVERY = 60
EVAL_SAMPLES = 2000
CFG_STRENGTH = 3.0
SCORE_HEAD_LOSS_W = 1.0


def weight_label(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def lr_label(x: float) -> str:
    # 5e-05 -> 5em05, 0.0001 -> 1em04, etc.
    s = f"{x:.0e}".replace("+", "")
    return s.replace("e-", "em").replace("e", "e")


def metric_label(metric: str) -> str:
    return "u" if metric == "unweighted-eps" else "c"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    config_id = 0
    bundle_id = 0

    # One bundle = one (outer metric, head metric, score-head LR),
    # with the six representation weights run sequentially.
    for outer in OUTER_METRICS:
        for head in HEAD_METRICS:
            for lr in SCORE_HEAD_LRS:
                for slot, rep_w in enumerate(REP_WEIGHTS):
                    result_name = (
                        f"cfg_{config_id:03d}"
                        f"_o{metric_label(outer)}"
                        f"_h{metric_label(head)}"
                        f"_lam{weight_label(rep_w)}"
                        f"_lr{lr_label(lr)}"
                    )
                    rows.append(
                        {
                            "config_id": f"{config_id:03d}",
                            "bundle_id": f"{bundle_id:02d}",
                            "slot_in_bundle": str(slot),
                            "outer_time_weighting": outer,
                            "head_time_weighting": head,
                            "rep_weight": f"{rep_w:.8g}",
                            "lr_score_head": f"{lr:.8g}",
                            "score_head_loss_w": f"{SCORE_HEAD_LOSS_W:.1f}",
                            "T_terminal": f"{T_TERMINAL:.2f}",
                            "epochs": str(EPOCHS),
                            "refine_epochs": str(REFINE_EPOCHS),
                            "eval_every": str(EVAL_EVERY),
                            "eval_samples": str(EVAL_SAMPLES),
                            "cfg_strength": f"{CFG_STRENGTH:.1f}",
                            "result_name": result_name,
                        }
                    )
                    config_id += 1
                bundle_id += 1

    assert len(rows) == 120, len(rows)
    assert bundle_id == 20, bundle_id
    assert all(
        sum(r["bundle_id"] == f"{b:02d}" for r in rows) == 6
        for b in range(20)
    )
    return rows


def build_manifest(path: Path | None = None) -> Path:
    path = path or (BASE_DIR / MANIFEST_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    path = build_manifest()
    rows = build_rows()
    print(f"Wrote {len(rows)} configurations to {path}")
    print("Bundles: 20 ordinary jobs x 6 sequential runs/job")
    print("Per run: 120 epochs; eval at epochs 60 and 120; 2000 eval samples")
    print("Fixed: FMNIST, terminal_kl arm, T=1.50, CFG=3.0, score-head-loss-w=1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
