CIFAR deployment-only evaluation sweep v1
========================================

Purpose
-------
No-training optimization of frozen checkpoints from cifar_detached_tail_downward_v1_results.
Primary = both seeds of T=1.35, T_K=1.05, DeltaT=.30, wC=.05, wK=.60.

Default PRIMARY screen
----------------------
CFG:         0,1,1.5,2,2.5,3,3.5,4,5,6
Temperature: 0.90,0.95,1.00,1.05,1.10
RK4 steps:   8,12,18,25,40,60  => NFE 32,48,72,100,160,240
Samples:     2500 per CFG x temperature configuration
Jobs:        12 = 2 seeds x 6 RK4 budgets

Each job loads the checkpoint once and evaluates all 50 CFG x temperature cells using the same fixed test/noise/label banks. No model weights are updated.

Optional ROBUSTNESS group
-------------------------
T=1.45,T_K=1.15,Delta=.30 seeds 42/43 and T=1.55,T_K=1.35,Delta=.20 seeds 42/43, each at NFE 72,100,160.

Copy all files into:
/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep/

The copied old results tree must remain named:
cifar_detached_tail_downward_v1_results/

Run
---
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep
python submit_cifar_deployment_eval_v1.py --group primary --dry-run
python submit_cifar_deployment_eval_v1.py --group primary

Compile partial or full results:
python compile_cifar_deployment_eval_v1.py

Optional robustness:
python submit_cifar_deployment_eval_v1.py --group robustness --dry-run
python submit_cifar_deployment_eval_v1.py --group robustness

Important outputs
-----------------
cifar_deployment_eval_v1_compiled/deployment_eval_all.csv
cifar_deployment_eval_v1_compiled/primary_seed_aggregates.csv
cifar_deployment_eval_v1_compiled/best_cfg_temp_by_nfe.csv
cifar_deployment_eval_v1_compiled/top25_primary_configs.csv
cifar_deployment_eval_v1_compiled/primary_cfg_temp_nfe_surface.csv

Interpretation
--------------
- temperature != 1 improving FID indicates exploitable radial terminal-start mismatch;
- CFG optimum away from 3 means prior comparisons were evaluated off their deployment optimum;
- NFE tests whether the CFG/temperature optimum is stable to integration accuracy;
- 2500 samples is a screen: confirm the top basin at 10k before claiming a new record.

Implementation
--------------
- evaluation-only checkpoint load is mandatory;
- no training/refinement steps;
- full-training oracle q_t construction disabled;
- invariant LSI-gap skipped; sample panels disabled;
- FID/KID/SW2 use the same evaluator family as the source sweep.
