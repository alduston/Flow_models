#!/usr/bin/env python3
"""Generate the 20-cell frozen-checkpoint oracle-score NFE mechanism sweep."""

from pathlib import Path
import csv

BASE_DIR = Path('/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0')
OUT = BASE_DIR / 'cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_manifest.csv'

# Five frozen representations chosen to separate CSEM shaping from terminal-KL
# Gaussianization.  All were trained at T_K=1.2, T=1.6 with the same CIFAR preset.
REPRESENTATIONS = [
    {
        'rep_id': 'two_stage_c0_k40',
        'rep_role': 'no CSEM / two-stage-like control',
        'source_results_root': 'cifar_weight_sweep_TK1p2_T1p6_v1_results',
        'source_result_name': 'cw_0p00_kl_0p40',
        'csem_w': '0.00', 'terminal_kl_w': '0.40',
    },
    {
        'rep_id': 'weak_c2_k40',
        'rep_role': 'weak CSEM shaping at matched KL',
        'source_results_root': 'cifar_weight_sweep_TK1p2_T1p6_v1_results',
        'source_result_name': 'cw_0p02_kl_0p40',
        'csem_w': '0.02', 'terminal_kl_w': '0.40',
    },
    {
        'rep_id': 'sweet_c5_k40',
        'rep_role': 'near-optimal CSEM shaping at matched KL',
        'source_results_root': 'cifar_highkl_weight_sweep_TK1p2_T1p6_v2_results',
        'source_result_name': 'cw_0p05_kl_0p40',
        'csem_w': '0.05', 'terminal_kl_w': '0.40',
    },
    {
        'rep_id': 'strong_c8_k40',
        'rep_role': 'strong score-friendly CSEM shaping at matched KL',
        'source_results_root': 'cifar_highkl_weight_sweep_TK1p2_T1p6_v2_results',
        'source_result_name': 'cw_0p08_kl_0p40',
        'csem_w': '0.08', 'terminal_kl_w': '0.40',
    },
    {
        'rep_id': 'highkl_c5_k80',
        'rep_role': 'same CSEM as sweet spot, stronger terminal Gaussianization',
        'source_results_root': 'cifar_highkl_weight_sweep_TK1p2_T1p6_v2_results',
        'source_result_name': 'cw_0p05_kl_0p80',
        'csem_w': '0.05', 'terminal_kl_w': '0.80',
    },
]

RK4_STEPS = [5, 10, 25, 50]
FIELDS = [
    'cell_id', 'rep_id', 'rep_role', 'source_results_root', 'source_result_name',
    'T_K', 'T_full', 'csem_w', 'terminal_kl_w', 'rk4_steps', 'rk4_nfe',
    'oracle_sampling_samples', 'oracle_profile_query_samples',
    'oracle_profile_time_points', 'result_name',
]


def main() -> int:
    rows = []
    cell = 0
    for rep in REPRESENTATIONS:
        for steps in RK4_STEPS:
            rows.append({
                'cell_id': f'{cell:02d}',
                **rep,
                'T_K': '1.20',
                'T_full': '1.60',
                'rk4_steps': str(steps),
                'rk4_nfe': str(4 * steps),
                'oracle_sampling_samples': '256',
                'oracle_profile_query_samples': '64',
                'oracle_profile_time_points': '16',
                'result_name': f"{rep['rep_id']}_rk4s{steps:03d}",
            })
            cell += 1

    assert len(rows) == 20
    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f'Wrote {len(rows)} oracle-NFE cells to {OUT}')
    for r in rows:
        print(
            f"cell {r['cell_id']} | {r['rep_id']} | steps={r['rk4_steps']} "
            f"NFE={r['rk4_nfe']} | source={r['source_result_name']}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
