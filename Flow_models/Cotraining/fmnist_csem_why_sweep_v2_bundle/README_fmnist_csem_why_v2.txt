CSEM FMNIST — WHY-v2: Time-Resolved Canonical-vs-Unweighted Mechanism Sweep

PURPOSE
-------
WHY-v2 is the same focused 8-run reconstruction-matched diagnostic experiment
as WHY-v1, but makes the score-error dependence on diffusion time t a
first-class output.

The scientific question is now explicitly:

  At matched reconstruction quality, HOW do the two learned representations
  change
      (a) intrinsic CSEM target variance,
      (b) actual learned-vs-aggregate score error, and
      (c) CFG-guided score error
  as functions of diffusion time?

We want to know whether the canonical representation's worse sampling is
associated with a particular time band rather than with one aggregate scalar.

EXPERIMENT DESIGN
-----------------
8 runs = 4 matched reconstruction pairs x 2 outer representation metrics.

Fixed:
  dataset                         FashionMNIST
  arm                             terminal_kl
  score-head time weighting       unweighted-eps
  score-head loss weight          1
  T_terminal                      1.50
  co-training epochs              120
  refinement                      0
  endpoint evaluation             final epoch only
  standard eval samples           2000
  CFG                             3.0
  seed                            42

Outer representation pair:
  U: --score-time-weighting unweighted-eps
     csem_w = terminal_kl_w = 0.60

  C: --score-time-weighting canonical
     csem_w = terminal_kl_w = 0.10

Four score-head learning rates:
  pair 0   1e-4
  pair 1   2e-4
  pair 2   4e-4
  pair 3   8e-4

Parent-sweep reconstruction FID, U vs C:
  pair 0   12.1066 vs 11.6443
  pair 1   11.9660 vs 11.5300
  pair 2   11.8361 vs 11.5137
  pair 3   11.6825 vs 11.5007

FULL AGGREGATE ORACLE
---------------------
At the final epoch the VAE encodes the entire FashionMNIST training set.
Every posterior Gaussian component is retained. OracleScoreModel therefore
computes the score of the full empirical aggregated posterior mixture, not a
small evaluation subset.

TIME-RESOLVED SCORE PROFILE
---------------------------
Each run evaluates 32 schedule locations. Because the experiment uses a log_t
OU schedule, these points are approximately evenly spaced in log(t), giving
dense relative coverage of the difficult small-time region without neglecting
the terminal end.

At every queried t, WHY-v2 records:

Time geometry
  t
  log_t
  alpha
  sigma
  sigma_sq
  SNR
  log_SNR
  trapezoid physical-dt node width
  canonical epsilon-space node mass dt/sigma^2
  normalized canonical node-mass fraction

Unconditional field
  uncond_component_residual_eps
  uncond_intrinsic_var_eps
  uncond_learned_oracle_eps

  uncond_component_residual_score
  uncond_intrinsic_var_score
  uncond_learned_oracle_score

Conditional field
  cond_component_residual_eps
  cond_intrinsic_var_eps
  cond_learned_oracle_eps

  cond_component_residual_score
  cond_intrinsic_var_score
  cond_learned_oracle_score

Actual CFG=3 field
  guided_learned_oracle_eps
  guided_learned_oracle_score

The important decomposition is, at each t,

  component residual(t)
      ≈ intrinsic CSEM variance(t)
        + learned-vs-aggregate-oracle error(t).

The finite-sample closure residual is logged explicitly as a sanity check.

PER-T CONTRIBUTIONS
-------------------
For every error metric m(t), the raw profile also records:

  m_logtime_contribution
      contribution to the historical equal-index/log-time average.

  m_physical_contribution
      trapezoid dt * m(t), i.e. the node's contribution to the physical-time
      integral.

  m_physical_abs_fraction
      fraction of the run's total absolute physical-time contribution.

Thus we can distinguish:
  - a large error at tiny t that occupies little physical dt;
  - a moderate error over a broad mid-time band;
  - terminal-time errors;
  - errors that are emphasized heavily by the canonical 1/sigma^2 conversion.

LOG-TIME BAND BREAKDOWN
-----------------------
The compiler additionally collapses the 32 points into 6 fixed equal-width
log(t) bands.

For each band and each metric it records:
  mean error in that band
  physical-time integral over that band
  fraction of the full physical-time error attributable to the band

The paired table then reports:
  U
  C
  C-U
  C/U

This is the fastest way to see whether the canonical representation's excess
actual score error is concentrated at small, intermediate, or late t.

ORACLE TRANSPORT DECOMPOSITION
------------------------------
WHY-v2 retains the WHY-v1 endpoint reverse-process ablation:

  direct_q0_train

  oracle_qT_rk4_ode_25
      exact aggregate score, exact forward-noised qT initialization

  oracle_gaussian_rk4_ode_25
      exact aggregate score, N(0,I) initialization

  learned_gaussian_uncond_rk4_ode_25
      learned score, same N(0,I) initial bank

FID, KID, latent SW2, and LPIPS diversity are measured for each stage.

FILES
-----
csem_split_metric_why_v2.py
    Updated unique training/evaluation driver.

fmnist_csem_why_grid_v2.py
fmnist_csem_why_manifest_v2.csv

fmnist_csem_why_pair_runner_v2.py
fmnist_csem_why_pair_job_v2.slurm
submit_fmnist_csem_why_pairs_v2.py

compile_fmnist_csem_why_v2.py

plot_fmnist_csem_why_time_profiles_v2.py
    Optional plotting utility. Produces log-t/log-error plots for each matched
    pair for unconditional actual score error, intrinsic variance,
    conditional actual score error, and CFG=3 actual score error.

SETUP
-----
Place all files in:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep/

Generate/verify manifest:

  "$SCRATCH/venvs/hlsi/bin/python" fmnist_csem_why_grid_v2.py

Dry run:

  "$SCRATCH/venvs/hlsi/bin/python" \
      submit_fmnist_csem_why_pairs_v2.py --dry-run

Submit all four ordinary pair jobs:

  "$SCRATCH/venvs/hlsi/bin/python" \
      submit_fmnist_csem_why_pairs_v2.py --pairs all

No Slurm arrays are used.

COMPILE
-------
After completion:

  "$SCRATCH/venvs/hlsi/bin/python" compile_fmnist_csem_why_v2.py

The most important time-profile outputs are:

  fmnist_csem_why_compiled_v2/oracle_score_time_profile_pair_wide.csv

      One row per (pair,t). This is the main raw U-vs-C time-profile file.

  fmnist_csem_why_compiled_v2/oracle_score_time_bins.csv

      Six-band summaries for each individual run.

  fmnist_csem_why_compiled_v2/oracle_score_time_bin_pair_deltas.csv

      Six-band U/C/C-U/C-U ratio comparison. This is the first file to inspect
      for "where in t is canonical different?"

  fmnist_csem_why_compiled_v2/all_oracle_score_time_profiles.csv

      Complete raw per-run profile.

  fmnist_csem_why_compiled_v2/oracle_profile_pair_deltas.csv

      Generic pairwise delta table including every numeric profile field.

Optional plots:

  "$SCRATCH/venvs/hlsi/bin/python" \
      plot_fmnist_csem_why_time_profiles_v2.py

This creates:

  fmnist_csem_why_compiled_v2/time_profile_plots_v2/

RECOMMENDED FILES TO UPLOAD FOR ANALYSIS
----------------------------------------
1. oracle_score_time_profile_pair_wide.csv
2. oracle_score_time_bin_pair_deltas.csv
3. oracle_sampling_pair_deltas.csv
4. pair_endpoint_wide.csv
5. endpoint_loss_epoch120.csv
6. run_status.csv
7. missing_or_failed.csv

NAMESPACE NOTE
--------------
All WHY-v2 filenames and result directories are new. WHY-v1 and the original
120-cell sweep are left untouched.
