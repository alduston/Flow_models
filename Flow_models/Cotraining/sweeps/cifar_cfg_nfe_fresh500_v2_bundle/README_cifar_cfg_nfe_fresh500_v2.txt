CIFAR CFG x NFE fresh-training certification audit v2
=====================================================

Purpose
-------
Rerun the deployment audit without relying on any pre-existing checkpoint.
Each seed trains the current best CIFAR configuration FROM SCRATCH for 500 joint epochs,
with zero score-only refinement, and then evaluates the full CFG x NFE grid immediately
using the in-memory EMA model from that same process.

This intentionally removes the old evaluation-only checkpoint-load path.

Chosen training configuration
-----------------------------
Current best short-horizon geometry from the consolidated knowledge state:
  T_K = 1.05
  T   = 1.35
  DeltaT = 0.30
  w_C = 0.05
  w_K = 0.60
  outer score-time weighting = canonical
  score-head weighting       = unweighted-eps
  score-head loss weight     = 1.0
  epochs / refine            = 500 / 0
  seeds                      = 42, 43

The historical two-seed Gaussian-start FID for this cell was about 9.52.

Certification evaluation contract
---------------------------------
  samples/config = 10,000
  temperature    = 1.0 FIXED (not swept)
  CFG            = {2.0, 2.5, 3.0, 3.5, 4.0}
  RK4 steps      = {18, 25, 40}
  NFE            = {72, 100, 160}
  total configs  = 15 per freshly trained seed

The direct baseline-reproduction cell is CFG=3.0, temperature=1.0, NFE=100.

Important safety/correctness changes
------------------------------------
1. No old checkpoints are imported.
2. Training and all 15 evaluations happen in one process per seed, so the evaluated model
   is the fresh in-memory EMA model.
3. The source fixes CFG semantics: CFG=0 is the unconditional/null-label field, CFG=1 is
   the ordinary conditional field. (The certification grid itself begins at CFG=2.)
4. Fixed evaluation noise/label/test banks are shared across configurations within each seed.
5. Temperature is fixed to 1.0 and absent as a sweep axis.
6. Evaluation-only oracle diagnostics, LSI-gap work, and sample panels are disabled.

Install
-------
Extract/copy all bundle files into:
/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep/

Dry run
-------
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_cfg_nfe_fresh500_v2.py --dry-run

Submit both fresh seeds
-----------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_cfg_nfe_fresh500_v2.py

Submit only one seed if needed
------------------------------
# cell 0 -> seed 42; cell 1 -> seed 43
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_cfg_nfe_fresh500_v2.py --cells 0

Compile completed results
-------------------------
"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_cfg_nfe_fresh500_v2.py

Key outputs
-----------
cifar_cfg_nfe_fresh500_v2_compiled/cfg_nfe_all.csv
cifar_cfg_nfe_fresh500_v2_compiled/cfg_nfe_seed_aggregates.csv
cifar_cfg_nfe_fresh500_v2_compiled/best_cfg_by_nfe.csv
cifar_cfg_nfe_fresh500_v2_compiled/best_nfe_by_cfg.csv
cifar_cfg_nfe_fresh500_v2_compiled/ranked_cfg_nfe_configs.csv
cifar_cfg_nfe_fresh500_v2_compiled/baseline_cfg3_nfe100.csv

Expected certification check
----------------------------
The CFG=3, NFE=100 row should return to the ordinary roughly 9.5-FID / roughly .002-KID
regime within normal two-seed/evaluation variation. If it does not, treat the deployment evaluator
or current training/evaluation source as suspect before interpreting CFG/NFE differences.
