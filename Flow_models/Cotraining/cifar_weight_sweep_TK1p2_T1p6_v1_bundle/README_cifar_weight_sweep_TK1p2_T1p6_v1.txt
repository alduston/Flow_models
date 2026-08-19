CIFAR CSEM / TERMINAL-KL WEIGHT SWEEP — FIXED T_K=1.2, T=1.6
================================================================

Scientific grid
---------------
Hold the representation/KL horizon and full score horizon fixed:

    T_K = 1.2
    T   = 1.6

Sweep the 5 x 4 Cartesian product:

    csem_w        in {0, .02, .04, .06, .08}
    terminal_kl_w in {.10, .20, .30, .40}

This gives exactly 20 runs.

Fixed training setup retained from the supplied sweep bundle
--------------------------------------------------------------
  dataset/preset              CIFAR / auto -> cifar_golden
  outer representation metric canonical
  score-head metric           unweighted-eps
  score_head_loss_w           1.0
  score-head LR               omitted -> CIFAR preset base LDM LR = 1e-4
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
The included driver is the supplied four-way T/T_K diagnostic version.
At ordinary evaluation it runs the learned score from all four starts:

  1. empirical class-conditional q_TK at T_K=1.2
  2. Gaussian N(0,I) at T_K=1.2
  3. empirical class-conditional q_T at T=1.6
  4. Gaussian N(0,I) at T=1.6

The T_K arm uses the driver's default 25 RK4 steps. With the CIFAR log_t grid,
the driver automatically raises the T=1.6 arm to 26 RK4 steps so the longer
horizon does not receive a lower integration-grid/NFE density.

No expensive exact aggregate-score oracle suite is enabled; this is the ordinary
four-way learned-score initialization comparison requested for every evaluation.

Job packing
-----------
The submitter packs up to 3 sequential sweep cells into each ordinary Slurm job.
For all 20 cells this produces exactly 7 jobs:

  job 1: cells 00,01,02
  job 2: cells 03,04,05
  job 3: cells 06,07,08
  job 4: cells 09,10,11
  job 5: cells 12,13,14
  job 6: cells 15,16,17
  job 7: cells 18,19

If one cell fails, the Slurm job continues to the remaining packed cells and
returns nonzero at the end. Each cell keeps its own result directory, config log,
and status JSON.

Files
-----
csem_split_new_weight_sweep_TK1p2_T1p6_v1.py
    Unmodified copy of the supplied four-way evaluation driver, renamed only to
    keep this sweep bundle namespace independent.

cifar_weight_sweep_TK1p2_T1p6_v1_manifest.csv
generate_cifar_weight_sweep_TK1p2_T1p6_v1.py
run_cifar_weight_sweep_TK1p2_T1p6_v1_cell.py
cifar_weight_sweep_TK1p2_T1p6_v1_cell_job.slurm
submit_cifar_weight_sweep_TK1p2_T1p6_v1.py
compile_cifar_weight_sweep_TK1p2_T1p6_v1.py
README_cifar_weight_sweep_TK1p2_T1p6_v1.txt
CHECKSUMS_cifar_weight_sweep_TK1p2_T1p6_v1.txt

Run location
------------
Copy the bundle files to:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0

Generate / verify manifest
--------------------------
"$SCRATCH/venvs/hlsi/bin/python" generate_cifar_weight_sweep_TK1p2_T1p6_v1.py

Dry-run submission
------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_weight_sweep_TK1p2_T1p6_v1.py --dry-run

For --cells all, the dry run should print exactly 7 sbatch commands with groups
0,1,2 / 3,4,5 / ... / 18,19.

Submit all 20 runs as 7 jobs
----------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_weight_sweep_TK1p2_T1p6_v1.py --cells all

Selective resubmission examples
-------------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_weight_sweep_TK1p2_T1p6_v1.py --cells 4
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_weight_sweep_TK1p2_T1p6_v1.py --cells 6-11
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_weight_sweep_TK1p2_T1p6_v1.py --cells 1,7,13,19

Selected cells are re-packed into groups of at most three in the order supplied
by the parsed cell set.

Compile when complete
---------------------
"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_weight_sweep_TK1p2_T1p6_v1.py

Primary output
--------------
cifar_weight_sweep_TK1p2_T1p6_v1_compiled/weight_sweep_summary.csv

The summary includes the four initialization/horizon arms for FID, KID, SW2 and
diversity, plus useful Gaussian-minus-oracle and T-minus-T_K FID contrasts.

Result directories
------------------
cifar_weight_sweep_TK1p2_T1p6_v1_results/cw_0p00_kl_0p10/
...
cifar_weight_sweep_TK1p2_T1p6_v1_results/cw_0p08_kl_0p40/

Each one-cell runner refuses to overwrite an existing partial result directory.
