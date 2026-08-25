CIFAR CSEM DETACHED-TAIL HYPOTHESIS SWEEP V1
=============================================

Purpose
-------
Directly test §17.2 of the consolidated CSEM state-of-knowledge report:

    DeltaT := T - T_K

may be a more stable control variable than absolute T_K. Existing best regions came from
T=1.75, T_K≈1.30 (tail≈.45) and T=1.60, T_K=1.20 (tail=.40), but those studies differed in
weights and refinement. This sweep removes that confound.

Scientific design
-----------------
Full horizon:
    T in {1.45, 1.60, 1.75}

Detached score-only Gaussianization tail:
    DeltaT in {0.30, 0.40, 0.50}

Therefore T_K=T-DeltaT is:

                 Delta=.30   Delta=.40   Delta=.50
    T=1.45          1.15         1.05         0.95
    T=1.60          1.30         1.20         1.10
    T=1.75          1.45         1.35         1.25

Every geometry is run with seeds {42,43}: 9 geometries x 2 seeds = 18 independent trainings.

Fixed controls
--------------
    dataset/preset              CIFAR / auto -> cifar_golden
    CSEM weight w_C             .05
    boundary/terminal KL w_K    .60
    outer representation route canonical physical-time
    score-head route            unweighted-epsilon
    score-head loss weight      1.0
    joint epochs                500
    score-only refinement       0
    final ordinary eval         10,000 samples, CFG=3
    encoder warmup/ramp         0 / 0
    extra tracking steps        0
    logvar clamp                [-30,20]

Why .05 / .60?
---------------
The consolidated report identifies w_C≈.04-.05 and w_K≈.5-.6 as the current fixed-horizon basin,
with (.05,.60) the best individual observed cell. Holding both fixed ensures that T and DeltaT are
the only experimental axes.

Why zero refinement?
--------------------
Score-only refinement does not move the representation but can mask geometry-dependent joint tracking.
For a direct geometry test every run stops after the same 500 joint epochs. Frozen oracle diagnostics
then separately expose representation, score-model, initialization, and finite-NFE channels.

Per-cell evaluation
-------------------
After training, each frozen representation receives the same mechanism audit used in the recent oracle
program:

  * exact empirical conditional oracle vs learned score
  * exact q_h vs Gaussian initialization
  * h in {T_K,T}
  * RK4 steps {5,10,25,50} = {20,40,100,200} NFE
  * time-resolved exact-field / learned-vs-oracle profile

The primary detached-tail outputs are:

  1. Gaussian-T FID
  2. oracle-q_T FID
  3. Gaussian-minus-oracle initialization gap at T
  4. reconstruction FID
  5. K_{T_K}, latent RMS, posterior variance
  6. learned-vs-oracle score error
  7. exact-field path-rate and finite-NFE dynamics

Interpretation
--------------
The detached-tail hypothesis is supported if, as T changes and absolute T_K moves substantially, the
best deployable/initialization region repeatedly tracks the same DeltaT (especially .40-.50). A clean
result would look like:

  * DeltaT=.30: tail too short -> residual Gaussian-start gap;
  * DeltaT=.40/.50: gap closes with good oracle/reconstruction channel;
  * any longer-coupling cost appears mainly in oracle/reconstruction rather than initialization.

It is weakened if the best point instead tracks a nearly fixed absolute T_K, or if optimal DeltaT shifts
strongly with T.

This design also directly asks whether T=1.45 can retain a good Gaussian start under the stronger
w_K=.60 anchor, without separately sweeping KL. A positive result motivates the dedicated Priority-7
shorter-T x stronger-KL experiment.

Files
-----
  csem_detached_tail_Tgrid_v1.py
      Main CSEM training/evaluation source, inherited from the corrected Pareto source.

  generate_cifar_detached_tail_Tgrid_v1.py
      Recreates the exact 18-cell manifest.

  cifar_detached_tail_Tgrid_v1_manifest.csv
      Pre-generated manifest.

  run_cifar_detached_tail_Tgrid_v1_cell.py
      One self-contained cell: train from scratch, then frozen mechanism evaluation.

  cifar_detached_tail_Tgrid_v1_cell_job.slurm
      One-GPU Vista gh Slurm job.

  submit_cifar_detached_tail_Tgrid_v1.py
      Submit all or selected cells without Slurm arrays.

  compile_cifar_detached_tail_Tgrid_v1.py
      Compiles raw outputs into T x DeltaT tables, two-seed aggregates, best-tail summaries, and plots.

  CHECKSUMS_cifar_detached_tail_Tgrid_v1.txt

Install location
----------------
Copy/extract ALL files directly into:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep

Dry run
-------
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep

"$SCRATCH/venvs/hlsi/bin/python" generate_cifar_detached_tail_Tgrid_v1.py

"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_detached_tail_Tgrid_v1.py \
    --cells all --dry-run

Submit all 18
-------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_detached_tail_Tgrid_v1.py \
    --cells all

Selected cells
--------------
Examples:

  --cells 0-5
  --cells 0,1,8,9

Cell ordering is T-major, DeltaT-middle, seed-minor. The manifest is the authority.

Resume / failure behavior
-------------------------
A successful training checkpoint is reused if only mechanism evaluation failed. A partial training or
mechanism directory is never silently overwritten; remove/rename the relevant partial directory before
resubmission.

Compile
-------
After all jobs finish:

"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_detached_tail_Tgrid_v1.py

Headline compiled outputs
-------------------------
  detached_tail_geometry_seed_aggregates.csv
      Two-seed mean/std for each of the nine geometries.

  detached_tail_summary_by_delta.csv
      Across-T behavior at each fixed tail length.

  detached_tail_summary_by_T.csv
      Across-tail behavior at each full horizon.

  best_delta_by_T_and_metric.csv
      For every T and headline metric, which DeltaT wins.

  detached_tail_stability_summary.csv
      Compact direct test of whether the winning DeltaT is stable across T.

  detached_tail_cell_summary.csv
      One row per independent training.

  oracle_nfe_curve_long.csv
  oracle_field_profile_all.csv
  training_eval_all.csv
  training_loss_all.csv
  run_status.csv
  missing_or_failed.csv

Plots
-----
  tail_gaussian_fid.png
  tail_init_gap.png
  tail_oracle_fid.png
  tail_reconstruction.png
  tail_modelability.png
  tail_exact_field.png
