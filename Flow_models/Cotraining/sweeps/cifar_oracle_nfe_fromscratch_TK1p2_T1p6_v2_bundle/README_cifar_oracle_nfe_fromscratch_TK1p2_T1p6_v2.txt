CIFAR CSEM ORACLE-SCORE NFE MECHANISM SWEEP V2 — TRAIN FROM SCRATCH
==================================================================

Purpose
-------
Separate three possible sources of CSEM sampling gains:

  1. OPTIMAL-FIELD DYNAMICS
     Does CSEM reshape the exact reverse score field so that finite-NFE
     integration is intrinsically easier?

  2. SCORE MODELING
     Does CSEM primarily make the optimal score easier for the learned DiT to
     approximate?

  3. TERMINAL GAUSSIANIZATION
     Does CSEM / terminal KL mainly make Gaussian initialization closer to q_h?

Nothing in this bundle assumes that checkpoints from an earlier sweep still
exist.  Every representation is trained from scratch by this sweep.

Fixed training setup
--------------------
  dataset/preset              CIFAR / auto -> cifar_golden
  T_K                         1.2
  T                           1.6
  outer representation route canonical
  score-head route            unweighted-epsilon
  score_head_loss_w           1.0
  cotrain                     500 epochs
  score-only refinement       100 epochs
  training eval_every         50
  training eval_samples       10000
  CFG                         3.0
  canonical_lr_scale          1.0
  encoder warmup/ramp         0 / 0
  extra score tracking        0
  logvar clamp                [-30,20]

Five representations (five training jobs)
------------------------------------------
  0  csem=0.00, KL=.40   no-CSEM / two-stage-like representation control
     (the score head is still trained concurrently, but the representation receives
      zero CSEM gradient; oracle-score results therefore isolate the resulting
      representation independently of how that score head was trained)
  1  csem=0.02, KL=.40   weak CSEM shaping
  2  csem=0.05, KL=.40   near-optimal CSEM shaping at matched KL
  3  csem=0.08, KL=.40   strong score-friendly shaping at matched KL
  4  csem=0.05, KL=.80   same CSEM, stronger Gaussianization control

Why five jobs instead of twenty independent trainings?
-------------------------------------------------------
Each representation is trained ONCE.  Immediately afterward, that exact frozen
checkpoint is evaluated at RK4 steps

    {5, 10, 25, 50} = {20, 40, 100, 200} RK4 NFE.

This still gives 5 x 4 = 20 NFE curve points, but NFE is not confounded with
independent training noise.  It is both cheaper and scientifically cleaner.

Mechanism matrix inside every representation
---------------------------------------------
At both h=T_K=1.2 and h=T=1.6, for every NFE point:

  exact empirical conditional oracle score + exact q_h initialization
  learned conditional score               + exact q_h initialization
  exact empirical conditional oracle score + Gaussian initialization
  learned conditional score               + Gaussian initialization

The exact oracle uses the empirical class-conditional aggregate posterior over
the full training set and no CFG.  This keeps the oracle a genuine score field.

Direct trajectory diagnostics
-----------------------------
All NFE points share the same selected posterior components, z0 draws, forward
noise, Gaussian bank, and labels.  Therefore the evaluator logs paired latent
errors that do not rely on FID as a proxy:

  endpoint_rms_to_maxnfe
      endpoint RMS relative to the same channel at 50 RK4 steps.
      For oracle + q_h this is the clean finite-integration / exact-field
      dynamical-difficulty diagnostic.

  endpoint_rms_learned_vs_oracle
      learned-vs-exact endpoint RMS at identical initialization and NFE.
      This isolates propagated score-model error.

  endpoint_rms_gaussian_vs_qh
      Coupling-dependent paired-bank endpoint RMS for Gaussian-init vs q_h-init at
      identical score/NFE.  Treat this as a secondary diagnostic; latent SW2 between
      q_h and Gaussian and the oracle Gaussian-vs-q_h distribution metrics are the
      cleaner terminal-mismatch measures.

The driver also records image FID/KID/diversity and latent SW2 for each channel.

Exact-field profile
-------------------
Once per representation the evaluation records a time-resolved profile including
learned-vs-oracle score error plus exact score/drift magnitude and path variation
quantities.  These help explain WHY a representation has a better or worse
oracle-NFE curve (smaller field, slower turning, lower temporal/path variation,
etc.).

Run location
------------
Copy all bundle files to:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0

Dry run
-------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2.py --cells all --dry-run

Submit all five representation jobs
-----------------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2.py --cells all

Submission safety
-----------------
There is ONE representation cell per Slurm job.  No comma-separated cell list is
used at all, so the earlier --export comma-parsing failure mode cannot recur.
The cell ID is inserted into the sbatch process environment and propagated with
--export=ALL.

Resume behavior
---------------
If training succeeded but mechanism evaluation failed, rerunning the same cell
skips training and reuses the checkpoint produced by THIS sweep.  A partial
mechanism_eval directory is never silently overwritten; remove/rename only that
directory before retrying.

Compile
-------
"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2.py

Primary compiled files
----------------------
  oracle_nfe_curve_long.csv
  oracle_nfe_mechanism_wide.csv
  oracle_field_profile_all.csv
  training_eval_all.csv
  training_loss_all.csv
  run_status.csv
  missing_or_failed.csv
