#!/usr/bin/env python3
"""Generate the 18-cell CIFAR fine sweep manifest."""

from pathlib import Path
import csv

BASE_DIR = Path("/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0")
OUT = BASE_DIR / "cifar_TKxCSEM_fine_T1p75_v1_manifest.csv"
TK_VALUES = [0.7, 0.85, 1.0, 1.15, 1.3, 1.45]
CSEM_VALUES = [0.075, 0.1, 0.15]
FIELDS = ['cell_id', 'group_id', 'group_tag', 'T_K', 'T_full', 'csem_w', 'terminal_kl_w', 'score_head_loss_w', 'outer_time_weighting', 'head_time_weighting', 'epochs', 'refine_epochs', 'eval_every', 'eval_samples', 'cfg_strength', 'result_name']

def tag(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", "p")

def main() -> int:
    rows = []
    cell = 0
    for gid, tk in enumerate(TK_VALUES):
        gtag = f"TK_{tag(tk)}"
        for cw in CSEM_VALUES:
            rows.append({
                "cell_id": f"{cell:02d}",
                "group_id": f"{gid:02d}",
                "group_tag": gtag,
                "T_K": f"{tk:.2f}",
                "T_full": "1.75",
                "csem_w": f"{cw:.3f}",
                "terminal_kl_w": "0.30",
                "score_head_loss_w": "1.0",
                "outer_time_weighting": "canonical",
                "head_time_weighting": "unweighted-eps",
                "epochs": "500",
                "refine_epochs": "0",
                "eval_every": "50",
                "eval_samples": "10000",
                "cfg_strength": "3.0",
                "result_name": f"{gtag}/csem_{tag(cw,3)}",
            })
            cell += 1

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} cells in {len(TK_VALUES)} groups to {OUT}")
    for gid, tk in enumerate(TK_VALUES):
        vals = [r["csem_w"] for r in rows if int(r["group_id"]) == gid]
        print(f"group {gid:02d} | T_K={tk:.2f} | csem={','.join(vals)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
