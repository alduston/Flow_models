CIFAR CSEM / TERMINAL-KL HIGH-KL WEIGHT SWEEP — FIXED T_K=1.2, T=1.6
============================================================================

Scientific grid
---------------
Hold fixed:

    T_K = 1.2
    T   = 1.6

Sweep the 5 x 4 Cartesian product:

    csem_w        in {.03, .04, .05, .06, .08}
    terminal_kl_w in {.40, .50, .60, .80}

This gives exactly 20 runs.  The .04/.40 cell overlaps the previous sweep.

Fixed training/evaluation setup
-------------------------------
  dataset/preset              CIFAR / auto -> cifar_golden
  outer representation metric canonical
  score-head metric           unweighted-eps
  score_head_loss_w           1.0
  canonical_lr_scale          1.0
  cotrain                     500 epochs
  score-only refinement       100 epochs
  eval_every                  50
  eval_samples                10000
  CFG                         3.0
  bespoke FID classifier      OFF
  fail-on-nonfinite           OFF
  logvar clamp                [-30,20]

Four-way evaluation
-------------------
Every ordinary evaluation retains the existing four-way learned-score diagnostic:

  1. empirical class-conditional q_TK at T_K=1.2
  2. Gaussian N(0,I) at T_K=1.2
  3. empirical class-conditional q_T at T=1.6
  4. Gaussian N(0,I) at T=1.6

The T-horizon RK4 step count is automatically increased by the driver to match
integration-grid/NFE density relative to T_K.

Job packing — IMPORTANT regression fix
--------------------------------------
The submitter packs up to 3 sequential cells into each Slurm job, giving exactly
7 jobs for all 20 cells:

  job 1: cells 00,01,02
  job 2: cells 03,04,05
  job 3: cells 06,07,08
  job 4: cells 09,10,11
  job 5: cells 12,13,14
  job 6: cells 15,16,17
  job 7: cells 18,19

The comma-separated cell list is NOT embedded in --export.  Instead the submitter
sets CSEM_WEIGHT_SWEEP_CELL_IDS in the sbatch process environment and invokes
sbatch with --export=ALL.  This avoids the earlier Slurm parsing bug that reduced
each 3-cell group to only its first cell.

Run location
------------
Copy all files to:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0

Generate / verify manifest
--------------------------
"$SCRATCH/venvs/hlsi/bin/python" generate_cifar_highkl_weight_sweep_TK1p2_T1p6_v2.py

Dry-run submission
------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_highkl_weight_sweep_TK1p2_T1p6_v2.py --cells all --dry-run

The dry run must print exactly these seven groups:
0,1,2 / 3,4,5 / 6,7,8 / 9,10,11 / 12,13,14 / 15,16,17 / 18,19

Submit all 20 runs as 7 jobs
----------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_highkl_weight_sweep_TK1p2_T1p6_v2.py --cells all

Compile when complete
---------------------
"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_highkl_weight_sweep_TK1p2_T1p6_v2.py

Primary compiled output
-----------------------
cifar_highkl_weight_sweep_TK1p2_T1p6_v2_compiled/weight_sweep_summary.csv

Namespace
---------
All manifests, result roots, status roots, logs, receipts, and compiled outputs
use the cifar_highkl_weight_sweep_TK1p2_T1p6_v2 namespace, so this sweep can coexist with the completed v1 sweep.
