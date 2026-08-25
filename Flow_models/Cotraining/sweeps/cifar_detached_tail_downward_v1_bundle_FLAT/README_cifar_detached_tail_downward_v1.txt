CIFAR CSEM DETACHED-TAIL DOWNWARD EXTENSION V1
================================================

Purpose
-------
The preceding T x DeltaT experiment found that DeltaT=.30 was the best tested value for Gaussian-T FID,
oracle-qT FID, learned-vs-oracle score error, exact-field path variation, and low-NFE exact dynamics at
all T in {1.45,1.60,1.75}. Because .30 was the lower boundary of that grid, this sweep extends the tail
axis downward and directly tests the new hypothesis that stronger terminal KL can substitute for tail length.

Primary question
----------------
How short can the detached tail become before Gaussianization becomes the limiting error channel?

Primary grid (24 trainings)
---------------------------
    T in {1.35, 1.45, 1.55}
    DeltaT = T - T_K in {.10, .20, .30, .40}
    w_C = .05
    w_K = .60
    seeds = {42,43}

Corresponding T_K values:

                 dT=.10   dT=.20   dT=.30   dT=.40
    T=1.35        1.25     1.15     1.05      .95
    T=1.45        1.35     1.25     1.15     1.05
    T=1.55        1.45     1.35     1.25     1.15

KL-tail substitution control (8 trainings)
-------------------------------------------
At the common T=1.45 slice, repeat the same four tail lengths with

    w_C = .05
    w_K = .40
    seeds = {42,43}

This gives a direct within-experiment comparison of w_K=.40 vs .60 while holding T, DeltaT, CSEM,
architecture, training length, and seeds fixed.

Total bundle design
-------------------
    24 primary + 8 control = 32 independent trainings.

Fixed training/evaluation protocol
----------------------------------
    dataset/preset              CIFAR / auto -> cifar_golden
    outer representation route canonical physical-time
    score-head route            unweighted-epsilon
    score-head loss weight      1.0
    joint epochs                500
    score-only refinement       0
    final ordinary eval         10,000 samples, CFG=3
    encoder warmup/ramp         0 / 0
    extra score tracking        0
    logvar clamp                [-30,20]

Each frozen representation then receives the same oracle/NFE mechanism audit as the preceding tail sweep:
    exact empirical conditional oracle vs learned score
    exact q_h vs Gaussian initialization
    horizons h in {T_K,T}
    RK4 steps {5,10,25,50} = {20,40,100,200} NFE
    time-resolved exact-field / learned-vs-oracle profile

Decision rules
--------------
1. SHORT-TAIL BOUNDARY
   The desired result is a crossover: as DeltaT decreases, representation/oracle/modelability metrics should
   continue improving until Gaussian-start mismatch begins worsening materially. If Gaussian-T FID or the
   initialization diagnostics turn upward at .10 or .20, the detached-tail boundary has been localized.
   If .10 is still best, the optimum remains below the tested range.

2. KL-TAIL SUBSTITUTION
   At T=1.45, compare the w_K=.40 and .60 curves. Evidence that stronger terminal KL substitutes for tail
   length means the w_K=.60 curve should tolerate a shorter DeltaT before Gaussian-start/init metrics degrade.

Operational groups
------------------
The submitter supports:
    --group primary   -> cells 00-23 only
    --group control   -> cells 24-31 only
    --group all       -> all 32 cells

Thus the 24 decisive downward-extension jobs can be run first; the eight KL-control jobs can be submitted
separately without changing the manifest.

Files
-----
  csem_detached_tail_downward_v1.py
      Same corrected CSEM training/evaluation source used in the successful preceding detached-tail sweep.

  generate_cifar_detached_tail_downward_v1.py
      Recreates the exact 32-cell manifest.

  cifar_detached_tail_downward_v1_manifest.csv
      Pre-generated manifest.

  run_cifar_detached_tail_downward_v1_cell.py
      Trains one representation from scratch, then runs the frozen oracle/NFE audit.

  cifar_detached_tail_downward_v1_cell_job.slurm
      One-GPU Vista gh Slurm job.

  submit_cifar_detached_tail_downward_v1.py
      Submit selected cells or primary/control/all groups.

  compile_cifar_detached_tail_downward_v1.py
      Compiles the scientific outputs and directly compares w_K=.40 vs .60 at T=1.45.

  CHECKSUMS_cifar_detached_tail_downward_v1.txt

Install location
----------------
Copy/extract all bundle files directly into:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep

Dry run primary
---------------
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep

"$SCRATCH/venvs/hlsi/bin/python" \
  submit_cifar_detached_tail_downward_v1.py --group primary --cells all --dry-run

Submit primary 24
-----------------
"$SCRATCH/venvs/hlsi/bin/python" \
  submit_cifar_detached_tail_downward_v1.py --group primary --cells all

Submit KL control 8
-------------------
"$SCRATCH/venvs/hlsi/bin/python" \
  submit_cifar_detached_tail_downward_v1.py --group control --cells all

Submit everything at once
-------------------------
"$SCRATCH/venvs/hlsi/bin/python" \
  submit_cifar_detached_tail_downward_v1.py --group all --cells all

Selected cells
--------------
Examples:
    --cells 0-7
    --cells 8,9,24,25
The --group filter and --cells selection are intersected.

Resume / failure behavior
-------------------------
Successful training checkpoints are reused if only mechanism evaluation fails. Partial training/mechanism
folders are never silently overwritten; remove or rename only the relevant partial folder before resubmission.

Compile
-------
After desired jobs finish:

"$SCRATCH/venvs/hlsi/bin/python" \
  compile_cifar_detached_tail_downward_v1.py

Use --allow-incomplete if compiling primary before the optional controls finish.

Headline compiled outputs
-------------------------
  downward_tail_geometry_seed_aggregates.csv
      Two-seed means/stds for every (group, w_K, T, DeltaT) geometry.

  primary_summary_by_delta.csv
      Primary w_K=.60 behavior averaged across T at each tail length.

  primary_best_delta_by_T_and_metric.csv
      Winning tail length at each primary T for deployment, initialization, representation, modelability,
      and exact-field diagnostics; includes a flag when .10 is still the lower-boundary optimum.

  primary_tail_stability_summary.csv
      Whether the preferred short tail is stable across T and whether all T values select <=.20 or exactly .10.

  kl_tail_comparison_T1p45.csv
      Direct matched T=1.45 comparison of w_K=.40 vs .60 across DeltaT.

  kl_control_T1p45_geometry.csv
      Two-seed aggregate control slice.

  best_delta_by_group_T_and_metric.csv
      Complete best-tail table for both primary and control groups.

  downward_tail_cell_summary.csv
      One row per independent training.

Raw compiled audit tables are also retained:
  oracle_nfe_curve_long.csv
  oracle_field_profile_all.csv
  training_eval_all.csv
  training_loss_all.csv
  run_status.csv
  missing_or_failed.csv

Plots
-----
  downward_gaussian_fid.png
  downward_init_gap.png
  downward_oracle_fid.png
  downward_reconstruction.png
  downward_modelability.png
  downward_exact_field.png
  kl_tail_tradeoff_gaussian_fid.png
  kl_tail_tradeoff_init_gap.png
  kl_tail_tradeoff_endpoint_init.png
