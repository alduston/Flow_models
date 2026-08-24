CIFAR CSEM vs STANDARD TWO-STAGE LDM PARETO DOMINANCE SWEEP V1
================================================================

PURPOSE
-------
Directly test whether CSEM has a better reconstruction-fidelity / score-regularity
Pareto frontier than the ordinary time-zero-KL lever available to a standard
VAE-based two-stage LDM.

This is NOT the earlier no-CSEM-at-T_K=1.2 control.  The standard-LDM arm here is
the exact T_K=0 endpoint:

    L_rep = L_recon + beta0 * E_x KL(q_phi(z0|x) || N(0,I))

with no diffusion-derived gradient reaching the VAE.  The score head is trained
on detached latents over the full [t_min,T] diffusion path.  T_K=0 is represented
exactly; it is never approximated by t_min.

RUN LOCATION
------------
Place every file in this bundle directly in:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep

THE 20 TRAININGS
----------------
All runs:
  dataset / preset              CIFAR / auto -> cifar_golden
  T                             1.6
  score-head route              unweighted-epsilon on detached/full-horizon latents
  score_head_loss_w             1.0
  epochs                        exactly 500
  score-only refinement         0
  ordinary evaluation           final epoch only
  ordinary eval samples         10000
  CFG                           3.0
  seeds                         42, 43

A. STANDARD TWO-STAGE LDM FRONTIER — 12 runs
   T_K                           0 exactly
   CSEM weight                   0 exactly
   beta0 = time-zero KL weight   {0, .01, .03, .07, .15, .30}
   seeds                         {42,43} at every beta0

B. CSEM FRONTIER — 8 runs
   T_K                           1.2
   terminal/boundary KL weight  .40
   CSEM weight                  {.02, .04, .08, .15}
   seeds                         {42,43} at every CSEM weight

Total: 6*2 + 4*2 = 20 independent 500-epoch trainings.

WHY THE TRAINING SCRIPT IS PATCHED
----------------------------------
The previous oracle/NFE source required T_K > t_min and therefore could not run
the exact standard-LDM endpoint.  csem_pareto_T1p6_20run_v1.py changes that
boundary carefully:

  * T_K=0 is explicitly legal only as the exact standard-two-stage endpoint.
  * csem_w must be exactly zero when T_K=0.
  * representation reconstruction is performed from z0 at decoder time t=0.
  * the active KL is K0, the ordinary VAE KL, evaluated analytically at t=0.
  * the representation-side score/CSEM measure is disabled (zero weight and no
    diffusion schedule is used to define the VAE objective).
  * the score head still trains independently on detached latents over [t_min,T].
  * frozen oracle diagnostics skip the meaningless reverse-from-T_K=0 trajectory;
    direct q0 reconstruction remains present and full-T curves are retained.
  * full-T ordinary sampler NFE is fixed across both families because T=1.6 is
    fixed; T_K is not allowed to silently change the deployable numerical budget.
  * --seed is exposed so paired seeds are genuinely independent trainings.

FROZEN MECHANISM AUDIT AFTER EACH TRAINING
------------------------------------------
After the 500-epoch checkpoint is frozen, each job evaluates:

  RK4 steps {5,10,25,50} = {20,40,100,200} NFE

at the full T=1.6 horizon for every family, crossing:

  exact empirical conditional oracle score + exact q_T initialization
  learned conditional score               + exact q_T initialization
  exact empirical conditional oracle score + Gaussian initialization
  learned conditional score               + Gaussian initialization

CSEM cells additionally retain the T_K=1.2 mechanism channels.  Two-stage cells
skip T_K reverse integration because T_K=0 is already q0 and there is no reverse
path to integrate.

The exact-field profile records learned-vs-oracle score error, exact score/drift
magnitude, temporal/path variation, and related quantities on the same frozen
representation.

PRIMARY PARETO TEST
-------------------
The compiler treats these as the headline axes:

  x: reconstruction FID

  intrinsic y:
     integrated_oracle_cond_drift_path_rate_rms
     = physical-time integral of recorded exact conditional drift path-rate

  learned-modelability y:
     cond_learned_oracle_score_logtime_mean
     = held-out learned-vs-exact conditional score MSE under the profile's
       approximately log-time measure

  propagated finite-model y:
     endpoint_model_rms_T_200nfe
     = learned-vs-oracle endpoint RMS at identical q_T initialization and 200 NFE

Gaussian-start FID is compiled as a secondary deployable-performance axis, not as
the definition of score regularity.

The compiler also performs local interpolation only over the reconstruction-FID
range where both seed-averaged frontiers overlap.  In
pareto_overlap_interpolation.csv:

  delta_log10_csem_minus_two_stage < 0

means CSEM has LOWER / BETTER score difficulty at matched reconstruction for that
metric.  Treat interpolation as a compact diagnostic, not a substitute for the
raw two-seed points and error bars.

FILES
-----
  csem_pareto_T1p6_20run_v1.py
      Patched experiment source with exact T_K=0 standard-LDM semantics and --seed.

  cifar_csem_vs_twostage_pareto_T1p6_20run_v1_manifest.csv
      Static 20-cell manifest.

  generate_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.py
      Recreates the exact 20-cell manifest if needed.

  run_cifar_csem_vs_twostage_pareto_T1p6_20run_v1_cell.py
      Runs one cell: 500-epoch training from scratch, then frozen mechanism audit.

  cifar_csem_vs_twostage_pareto_T1p6_20run_v1_cell_job.slurm
      One-GPU GH-node Slurm job, 24-hour wall clock.

  submit_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.py
      Submits one independent Slurm job per requested cell; no arrays and no
      comma-packed --export values.

  compile_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.py
      Compiles raw outputs, cell summaries, two-seed aggregates, overlap test,
      and Pareto plots.

  CHECKSUMS_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.txt
      SHA256 hashes for the bundle.

DRY RUN
-------
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep

"$SCRATCH/venvs/hlsi/bin/python" \
  submit_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.py \
  --cells all --dry-run

SUBMIT ALL 20
-------------
"$SCRATCH/venvs/hlsi/bin/python" \
  submit_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.py \
  --cells all

Useful partial submissions:

  --cells 0-11      all standard two-stage cells
  --cells 12-19     all CSEM cells
  --cells 0,1       beta0=0, both seeds

Each submitted cell gets its own Slurm job and submission receipt.

RESUME / FAILURE BEHAVIOR
-------------------------
The runner will not silently overwrite partial scientific output.

  * If a cell completed fully, rerunning it skips it.
  * If training completed but mechanism evaluation failed, the training
    checkpoint is retained. Remove/rename only that cell's mechanism_eval
    directory and resubmit the cell.
  * If a partial training directory exists without a successful training status,
    the runner blocks. Remove/rename it explicitly before restarting that cell.

COMPILE AFTER THE SWEEP
-----------------------
"$SCRATCH/venvs/hlsi/bin/python" \
  compile_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.py

For a partial diagnostic compilation:

"$SCRATCH/venvs/hlsi/bin/python" \
  compile_cifar_csem_vs_twostage_pareto_T1p6_20run_v1.py --allow-incomplete

HEADLINE COMPILED OUTPUTS
-------------------------
  pareto_cell_summary.csv
      One row per independent training with reconstruction, latent, exact-field,
      learned-oracle, endpoint-model-error, and generation metrics.

  pareto_seed_aggregates.csv
      Two-seed mean/std at each of the 10 operating points.

  pareto_overlap_interpolation.csv
      Matched-reconstruction local interpolation of exact/modelability frontiers.
      Negative delta means CSEM is better.

  pareto_exact_field.png
      reconstruction FID vs integrated exact conditional drift path-rate

  pareto_score_modelability.png
      reconstruction FID vs learned-oracle conditional score MSE

  pareto_endpoint_model_error.png
      reconstruction FID vs propagated learned-oracle endpoint RMS

  pareto_generation.png
      reconstruction FID vs Gaussian-start FID (secondary)

Also retained:
  oracle_nfe_curve_long.csv
  oracle_nfe_mechanism_wide.csv
  oracle_field_profile_all.csv
  training_eval_all.csv
  training_loss_all.csv
  run_status.csv
  missing_or_failed.csv

SCIENTIFIC INTERPRETATION
-------------------------
The decisive result is not which family has the single best FID cell.  The target
claim is Pareto dominance in the reconstruction / score-regularity exchange.

Strong support for CSEM:
  At the same reconstruction FID, CSEM has consistently lower exact-field
  regularity cost AND lower learned-vs-oracle score error over the common range.

Optimization-only advantage:
  Exact-field curves overlap but learned-vs-oracle error favors CSEM.

No Pareto advantage:
  Properly tuned beta0 cells trace the same or a better intrinsic exact-field
  frontier at matched reconstruction.

The two-seed design is enough to expose gross frontier separation, but it should
not be read as a high-powered test of very small differences.
