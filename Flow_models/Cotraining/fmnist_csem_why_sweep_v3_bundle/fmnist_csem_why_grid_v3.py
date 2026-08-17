#!/usr/bin/env python3
"""
Generate the focused FMNIST CSEM "why canonical?" diagnostic manifest.

Scientific design
-----------------
Four reconstruction-FID-matched pairs are re-run.  In every pair:
  * score-head time metric is fixed to unweighted-eps;
  * unweighted outer uses representation coefficient lambda=0.60;
  * canonical outer uses lambda=0.10;
  * csem_w == terminal_kl_w inside each run;
  * T=1.50, 120 co-training epochs, no refinement;
  * standard endpoint evaluation uses 2000 samples, CFG=3;
  * exact aggregate-score diagnostics are enabled at the final epoch only.

The four pairs vary only score-head LR:
  1e-4, 2e-4, 4e-4, 8e-4.

These were already close in reconstruction FID in the parent 120-cell sweep:
  pair 0: 12.1066 vs 11.6443
  pair 1: 11.9660 vs 11.5300
  pair 2: 11.8361 vs 11.5137
  pair 3: 11.6825 vs 11.5007
(unweighted outer vs canonical outer respectively).
"""

from pathlib import Path
import csv

BASE_DIR = Path(
    "/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
)
MANIFEST_NAME = "fmnist_csem_why_manifest_v3.csv"

PAIR_DATA = [
    # pair_id, lr, prior recon U, prior recon C, prior mean gen FID U, prior mean gen FID C
    (0, 1e-4, 12.1066174333, 11.6443297888, 16.5508524497, 21.2921043797),
    (1, 2e-4, 11.9659551912, 11.5300443368, 15.8951458404, 20.6090672827),
    (2, 4e-4, 11.8360717763, 11.5137034968, 15.7665227814, 20.8381638114),
    (3, 8e-4, 11.6825151317, 11.5007157054, 15.4860409757, 20.2495970353),
]

FIELDS = [
    "config_id",
    "pair_id",
    "slot_in_pair",
    "outer_time_weighting",
    "head_time_weighting",
    "rep_weight",
    "lr_score_head",
    "score_head_loss_w",
    "T_terminal",
    "epochs",
    "refine_epochs",
    "eval_every",
    "eval_samples",
    "cfg_strength",
    "oracle_profile_query_samples",
    "oracle_profile_time_points",
    "oracle_profile_batch_size",
    "oracle_reference_batch_size",
    "oracle_sampling_samples",
    "oracle_sampling_batch_size",
    "oracle_sampling_steps",
    "prior_recon_fid",
    "prior_mean_gen_fid",
    "result_name",
]


def main() -> int:
    rows = []
    cfg_id = 0
    for pair_id, lr, recon_u, recon_c, gen_u, gen_c in PAIR_DATA:
        specs = [
            ("unweighted-eps", 0.60, recon_u, gen_u, "U"),
            ("canonical",      0.10, recon_c, gen_c, "C"),
        ]
        for slot, (outer, rep_w, recon, gen, short) in enumerate(specs):
            lr_tag = f"{lr:.0e}".replace("-", "m").replace("+", "").replace("e-0", "em").replace("e-", "em")
            # explicit, readable result name; config id prevents collisions.
            result_name = (
                f"why_cfg_{cfg_id:03d}_pair{pair_id}_{short}_"
                f"lam{rep_w:.2f}_lr{lr:.0e}"
            ).replace(".", "p").replace("e-", "em")
            rows.append({
                "config_id": f"{cfg_id:03d}",
                "pair_id": f"{pair_id:02d}",
                "slot_in_pair": slot,
                "outer_time_weighting": outer,
                "head_time_weighting": "unweighted-eps",
                "rep_weight": rep_w,
                "lr_score_head": lr,
                "score_head_loss_w": 1.0,
                "T_terminal": 1.50,
                "epochs": 120,
                "refine_epochs": 0,
                "eval_every": 0,          # endpoint only: oracle diagnostics are expensive
                "eval_samples": 2000,
                "cfg_strength": 3.0,
                "oracle_profile_query_samples": 256,
                "oracle_profile_time_points": 32,
                "oracle_profile_batch_size": 16,
                "oracle_reference_batch_size": 2048,
                "oracle_sampling_samples": 2000,
                "oracle_sampling_batch_size": 32,
                "oracle_sampling_steps": 25,
                "prior_recon_fid": recon,
                "prior_mean_gen_fid": gen,
                "result_name": result_name,
            })
            cfg_id += 1

    assert len(rows) == 8
    for p in range(4):
        assert sum(int(r["pair_id"]) == p for r in rows) == 2

    out = BASE_DIR / MANIFEST_NAME
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")
    for r in rows:
        print(
            f"pair {r['pair_id']} slot {r['slot_in_pair']} "
            f"{r['outer_time_weighting']:14s} "
            f"lambda={float(r['rep_weight']):.2f} "
            f"lr={float(r['lr_score_head']):.1e} "
            f"prior recon={float(r['prior_recon_fid']):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
