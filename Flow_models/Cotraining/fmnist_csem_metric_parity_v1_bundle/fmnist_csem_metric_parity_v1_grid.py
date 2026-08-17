#!/usr/bin/env python3
"""Generate the fmnist_csem_metric_parity_v1 manifest.

Design:
  - 1 unweighted reference: csem_w=0.60, terminal_kl_w=0.60
  - 16 canonical cells:
      csem_w in [0.025, 0.05, 0.075, 0.1]
      terminal_kl_w in [0.3, 0.5, 0.7, 0.9]
  - score-head metric fixed to unweighted-eps
  - score-head LR fixed to 8e-4
  - all endpoint oracle/time-profile diagnostics enabled by the runner

The goal is to find canonical cells that simultaneously match the reference in
reconstruction FID and terminal q_T-to-Gaussian mismatch, then compare generation.
"""

from pathlib import Path
import csv

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep")
OUT = BASE_DIR / "fmnist_csem_metric_parity_v1_manifest.csv"

CSEM_GRID = [0.025, 0.05, 0.075, 0.1]
TERMINAL_GRID = [0.3, 0.5, 0.7, 0.9]


def ftag(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> int:
    rows = []
    cfg = 0
    common = dict(
        head_time_weighting="unweighted-eps",
        lr_score_head=8e-4,
        score_head_loss_w=1.0,
        T_terminal=1.50,
        epochs=120,
        refine_epochs=0,
        eval_every=0,
        eval_samples=2000,
        cfg_strength=3.0,
        oracle_profile_query_samples=256,
        oracle_profile_time_points=32,
        oracle_profile_batch_size=16,
        oracle_reference_batch_size=2048,
        oracle_sampling_samples=2000,
        oracle_sampling_batch_size=32,
        oracle_sampling_steps=25,
    )

    rows.append(dict(
        config_id=f"{cfg:03d}",
        bundle_id=f"{cfg//2:02d}",
        slot_in_bundle=cfg % 2,
        role="unweighted_reference",
        outer_time_weighting="unweighted-eps",
        csem_w=0.60,
        terminal_kl_w=0.60,
        result_name="metric_parity_cfg_000_U_cw0p6_kw0p6_lr8em4",
        **common,
    ))
    cfg += 1

    for cw in CSEM_GRID:
        for kw in TERMINAL_GRID:
            rows.append(dict(
                config_id=f"{cfg:03d}",
                bundle_id=f"{cfg//2:02d}",
                slot_in_bundle=cfg % 2,
                role="canonical_grid",
                outer_time_weighting="canonical",
                csem_w=cw,
                terminal_kl_w=kw,
                result_name=(
                    f"metric_parity_cfg_{cfg:03d}_C_"
                    f"cw{ftag(cw)}_kw{ftag(kw)}_lr8em4"
                ),
                **common,
            ))
            cfg += 1

    fields = [
        "config_id","bundle_id","slot_in_bundle","role",
        "outer_time_weighting","head_time_weighting",
        "csem_w","terminal_kl_w","lr_score_head","score_head_loss_w",
        "T_terminal","epochs","refine_epochs","eval_every","eval_samples",
        "cfg_strength","oracle_profile_query_samples","oracle_profile_time_points",
        "oracle_profile_batch_size","oracle_reference_batch_size",
        "oracle_sampling_samples","oracle_sampling_batch_size",
        "oracle_sampling_steps","result_name",
    ]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    assert len(rows) == 17
    print(f"Wrote {len(rows)} cells to {OUT}")
    print("Reference: U, csem=.60, terminal=.60, LR=8e-4")
    print("Canonical grid:", CSEM_GRID, "x", TERMINAL_GRID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
