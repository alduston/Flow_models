CSEM PAPER FINAL SWEEP — V2 CORRECTED CONFIGURATION AUDIT
==========================================================

This bundle supersedes V1 for any reruns. It preserves the V1 script filenames
so the launch commands stay the same, but should live in its own directory.

Critical corrections versus V1:
  1. CIFAR final CSEM and CIFAR T/lambda_K sensitivity use the late-August
     leading anchor: OU-partial GroupNorm at T_K with rho_m=alpha_K^2 and
     rho_v=alpha_K^4, together with terminal KL w_K=.60 at the center point.
  2. The final CIFAR geometry is T_K=1.05, T=1.35, w_C=.05, canonical outer
     weighting, unweighted-eps score-head weighting, CFG=2.5, temperature 1.
  3. CIFAR headline sampling uses RK4 25 steps = 100 NFE.
  4. Two raw-mean terminal-KL control seeds are included on the same code path
     to re-certify the OU-partial paired gain and diagnose the old control drift.
  5. Independent CSEM/Tweedie evaluation now passes the training loader to
     full-reference diagnostics; this fixes the V1 oracle_reference_loader error.
  6. Dataset download/extraction is serialized with a filesystem lock into
     data_cache_v2, preventing concurrent CIFAR jobs from corrupting ./data.

Expected/intentional instability:
  - anchor_none is SUPPOSED to blow up in latent scale. The late-August factorial
    had Gaussian FID >300 and terminal-KL diagnostics around 6e3 with no endpoint
    Gaussianization, while oracle-q_TK and reconstruction remained good.
  - naive_tweedie_cotrain is also an intentional failure control (collapse).
Do not interpret either as evidence that the optimized OU-partial+KL arm lacks
scale control.

CSEM PAPER FINAL SWEEP V1
=========================

Purpose
-------
This bundle is a self-contained final-paper sweep built from the latest certified
CSEM two-horizon / terminal-anchor code path recovered from the August 2026 runs.
It is designed to populate the result placeholders in the current CSEM_main paper:

  1. Independent Tweedie vs Independent CSEM vs Co-trained CSEM + K_{T_K}
     on CIFAR-10 and FashionMNIST, including epoch curves and final metrics.
  2. Scale-anchor ablation on CIFAR-10:
       terminal K_{T_K}, legacy encoder-mean GroupNorm, no anchor.
  3. Compact terminal-horizon / lambda_K sensitivity around the best certified
     CIFAR operating point.
  4. Heun-SDE vs RK4-ODE sampler-order comparison at matched 20-step budgets for
     the final CIFAR CSEM model and the naive Tweedie co-training control.
  5. Naive DSM/Tweedie co-training collapse traces and generation metrics.

The bundle intentionally DOES NOT run a dense 2D T x lambda_K sweep. That is the
one requested item reduced to a compact cross (4 extra cells plus the central
main cell) so the final bundle stays tractable. It also runs the collapse control
only on CIFAR, where the modern code path and terminal-anchor evidence are strongest.

Scientific configuration choices
--------------------------------
CIFAR final CSEM cell (certified fresh-training geometry):
  T_K                    = 1.05
  T                      = 1.35
  Delta T                = 0.30
  w_C                    = 0.05
  lambda_K               = 0.60
  outer metric           = canonical physical-time CSEM
  score-head metric      = unweighted epsilon
  score-head loss weight = 1.0
  joint epochs/refine    = 500 / 0
  CFG                    = 2.5
  ordinary RK4 steps     = 18 (72 NFE)
  seeds                  = 42, 43
  eval samples           = 10,000

This is the current certified raw-mean terminal-KL configuration. The later
OU-partial+KL result is NOT used as the paper's final arm because its absolute
headline improvement was explicitly still provisional and the current paper
specifies raw encoder means with no normalization.

CIFAR independent baselines:
  exact T_K=0 two-stage limit
  time-zero VAE KL beta_0 = 0.07
  500 VAE-only epochs + 500 frozen-VAE score epochs
  both CSEM and Tweedie heads are trained on the SAME fixed VAE in one run
  CFG/evaluation settings match the co-trained arm

beta_0=.07 is selected from the completed true two-stage Pareto audit because it
had the best mean Gaussian-start FID among the measured standard two-stage cells
(very close to .01 and .15). The point of this paper sweep is a strong baseline,
not reconstruction matching to a specific CSEM cell.

FashionMNIST final CSEM cell:
  T_K = T               = 1.50
  w_C                   = 0.10
  lambda_K              = 0.60
  outer/head metric     = canonical / unweighted epsilon
  joint epochs/refine   = 400 / 0
  CFG                   = 3.0
  RK4 steps             = 25
  seed                  = 42
  eval samples          = 5,000

FMNIST has not had the same late two-horizon terminal-anchor optimization as
CIFAR. This choice keeps the validated WHY-v3 canonical representation setting
(w_C=.10, T=1.5) but decouples and strengthens the terminal anchor to .60, exactly
the confound identified by the FMNIST mechanism study. The independent FMNIST
baseline reuses beta_0=.07 for a common strong two-stage comparator; this value
should not be described as a separately optimized FMNIST beta.

Scale-anchor ablation
---------------------
The terminal arm is reused from CIFAR main seed 42. Dedicated cells run:
  - Architectural normalization: historical per-sample mean GroupNorm, no K_TK.
  - No anchor: raw mean, no GroupNorm, no K_TK.
Everything else is held at the final CIFAR geometry and training recipe.

Terminal horizon / anchor sensitivity
-------------------------------------
The central point is the CIFAR main cell (T=1.35, lambda_K=.60). Four extra cells:
  T in {1.25, 1.45} at lambda_K=.60,
  lambda_K in {.40, .80} at T=1.35,
with T_K=1.05 and w_C=.05 fixed. These are final-only 5,000-sample evaluations.
This is deliberately a compact cross rather than a dense 2D surface.

Naive Tweedie co-training control
---------------------------------
CIFAR only, 100 joint epochs, evaluation every 10 epochs.
The active encoder-shaping score target is standard unweighted DSM/Tweedie epsilon.
No CSEM-specific terminal anchor and no mean normalization are applied. The run is
shortened because the failure mode is expected to appear early and the paper needs
collapse dynamics rather than an optimized DSM alternative.

Files
-----
csem_paper_core_v1.py
    Self-contained training/evaluation source. Based on the fresh-training certified
    two-horizon code path. Two small paper-specific patches are included:
      (a) optional Heun/RK4 matched-step evaluation;
      (b) co-training evaluation selects the actually active head, which is required
          for the Tweedie-control cell.

csem_paper_suite_v1.py
    Constructs the exact scientific configs for each paper cell and dispatches the
    existing train_vae_cotrained_cond implementation.

csem_paper_final_v1_manifest.csv
    13 dedicated jobs covering the paper comparisons.

run_csem_paper_final_v1_cell.py
csem_paper_final_v1_cell_job.slurm
submit_csem_paper_final_v1.py
    Vista/gh execution layer. No Slurm arrays are used.

compile_csem_paper_final_v1.py
plot_csem_paper_final_v1.py
    After the sweep, compile raw outputs into paper-facing CSVs and PNGs.

PAPER_PLACEHOLDER_COVERAGE.csv
    Maps each current paper placeholder to cells and compiled outputs.

Install on Vista
----------------
Recommended directory:
  /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/sweeps/csem_paper_final_v1

Unzip the bundle there and enter the directory.

Validate + dry run
------------------
  "$SCRATCH/venvs/hlsi/bin/python" validate_csem_paper_final_v1.py
  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py --dry-run

You can dry-run / submit by family:
  --cells main
  --cells scale_anchor
  --cells horizon_anchor
  --cells tweedie_collapse

Submit all 13 jobs
------------------
  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py

Submit a subset
---------------
  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py --cells 0-5
  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py --cells scale_anchor

Compile and plot after completion
---------------------------------
  "$SCRATCH/venvs/hlsi/bin/python" compile_csem_paper_final_v1.py
  "$SCRATCH/venvs/hlsi/bin/python" plot_csem_paper_final_v1.py

Primary compiled outputs
------------------------
csem_paper_compiled_v1/main_final_metrics_by_seed.csv
csem_paper_compiled_v1/main_final_metrics_aggregate.csv
csem_paper_compiled_v1/main_epoch_curves_long.csv
csem_paper_compiled_v1/scale_anchor_final.csv
csem_paper_compiled_v1/terminal_horizon_anchor_sensitivity.csv
csem_paper_compiled_v1/naive_tweedie_collapse_final.csv
csem_paper_compiled_v1/solver_order_comparison.csv
csem_paper_compiled_v1/all_loss_history.csv
csem_paper_compiled_v1/all_eval_records.csv
csem_paper_compiled_v1/run_status.csv

Paper-ready figures are written to:
  csem_paper_figures_v1/

Important runtime note
----------------------
The two independent-baseline cells are the expensive jobs: each does 500 VAE-only
CIFAR epochs followed by 500 frozen-VAE score epochs while training both CSEM and
Tweedie heads on the same fixed representation. The Slurm wrapper therefore requests
48 hours. If Vista's current gh walltime policy is lower, reduce the requested walltime
or split those two cells; no scientific code change is required.

Resubmission safety
-------------------
A successful status JSON + existing result directory is skipped. An existing result
directory without a successful status is treated as a partial run and is NOT
silently overwritten; rename/remove that one result directory before retrying.
