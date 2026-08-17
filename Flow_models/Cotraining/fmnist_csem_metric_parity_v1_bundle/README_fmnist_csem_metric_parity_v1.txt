CSEM FMNIST — METRIC PARITY V1
================================

Question
--------
The previous matched-reconstruction experiment showed:

  canonical outer weighting:
    - lowers intrinsic CSEM variance,
    - lowers actual learned-vs-aggregate score error,
    - transports perfectly well with the exact score when initialized from q_T,

but at reconstruction parity it had a much larger q_T-to-Gaussian mismatch because
the previous experiment tied:

    csem_w == terminal_kl_w.

This experiment decouples those two coefficients.

Scientific target
-----------------
Find canonical representations that simultaneously match the unweighted reference in:

  1. reconstruction FID,
  2. terminal component KL,
  3. empirical latent SW2(q_T, N(0,I)),

then compare generated FID/KID/SW2 and the exact score-error profiles.

Fixed score-head setup
----------------------
The score-head time metric is fixed to unweighted-eps.
The score-head LR is fixed to 8e-4.
There is no head-metric sweep.

Reference cell
--------------
  outer metric      unweighted-eps
  csem_w            0.60
  terminal_kl_w     0.60

Canonical calibration surface
-----------------------------
  csem_w in          [0.025, 0.05, 0.075, 0.1]
  terminal_kl_w in   [0.3, 0.5, 0.7, 0.9]

This gives 16 canonical cells + 1 reference = 17 total configurations.

The grid is deliberately two-dimensional:
  - moving vertically changes the strength of the canonical representation objective;
  - moving horizontally changes only the terminal Gaussian anchor.

Hence fixed-csem rows directly test whether restoring q_T≈N(0,I) repairs sampling while
leaving the canonical score-estimation geometry largely intact.

Training/evaluation
-------------------
  FMNIST / fmnist_reference
  terminal_kl arm only
  T = 1.5
  120 epochs
  no refinement
  endpoint evaluation only
  2000 standard evaluation samples
  CFG = 3
  full-training-set empirical aggregate score oracle
  32-point score-error profile in t
  256 oracle query components per t
  2000-sample RK4-25 oracle transport decomposition
  seed 42
  no bespoke FID classifier
  no warmup/ramp/tracking diagnostics

Compute layout
--------------
17 cells are packed into 9 ordinary Vista jobs, at most 2 sequential cells per job.
No Slurm arrays are used.

Files
-----
csem_split_metric_metric_parity_v1.py
fmnist_csem_metric_parity_v1_grid.py
fmnist_csem_metric_parity_v1_manifest.csv
fmnist_csem_metric_parity_v1_bundle_runner.py
fmnist_csem_metric_parity_v1_bundle_job.slurm
submit_fmnist_csem_metric_parity_v1.py
compile_fmnist_csem_metric_parity_v1.py
plot_fmnist_csem_metric_parity_v1.py
README_fmnist_csem_metric_parity_v1.txt

Install location
----------------
Copy all files to:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep/

Generate / verify manifest
--------------------------
"$SCRATCH/venvs/hlsi/bin/python" fmnist_csem_metric_parity_v1_grid.py

Dry run
-------
"$SCRATCH/venvs/hlsi/bin/python" submit_fmnist_csem_metric_parity_v1.py --dry-run

Submit all 9 jobs
-----------------
"$SCRATCH/venvs/hlsi/bin/python" submit_fmnist_csem_metric_parity_v1.py --bundles all

Submit a subset
---------------
"$SCRATCH/venvs/hlsi/bin/python" submit_fmnist_csem_metric_parity_v1.py --bundles 0-3
"$SCRATCH/venvs/hlsi/bin/python" submit_fmnist_csem_metric_parity_v1.py --bundles 4,7

Compile
-------
"$SCRATCH/venvs/hlsi/bin/python" compile_fmnist_csem_metric_parity_v1.py

Primary compiled outputs
------------------------
fmnist_csem_metric_parity_v1_compiled/canonical_parity_ranking.csv
fmnist_csem_metric_parity_v1_compiled/top5_canonical_parity_candidates.csv
fmnist_csem_metric_parity_v1_compiled/metric_parity_surface.csv
fmnist_csem_metric_parity_v1_compiled/canonical_time_profiles_vs_reference.csv
fmnist_csem_metric_parity_v1_compiled/endpoint_eval_epoch120.csv
fmnist_csem_metric_parity_v1_compiled/endpoint_loss_epoch120.csv
fmnist_csem_metric_parity_v1_compiled/all_oracle_sampling_decomposition.csv
fmnist_csem_metric_parity_v1_compiled/run_status.csv
fmnist_csem_metric_parity_v1_compiled/missing_or_failed.csv

The compiler identifies the unweighted reference from the new run itself. It does not
hard-code the previous run's reference metrics.

Default parity-selection box
----------------------------
  reconstruction FID absolute difference <= 0.5
  terminal KL ratio in [0.8, 1.25]
  terminal SW2 ratio in [0.8, 1.25]

It also gives every canonical cell a continuous parity_score so the closest point is
still obvious if the coarse grid does not land exactly inside the box.

Plot surfaces
-------------
"$SCRATCH/venvs/hlsi/bin/python" plot_fmnist_csem_metric_parity_v1.py

This writes separate figures for:
  reconstruction FID
  terminal KL
  terminal SW2
  joint parity score
  mean Heun/RK4 generation FID

Interpretation
--------------
The decisive comparison is NOT equal numerical csem_w.

It is:
  canonical and unweighted at equal reconstruction quality AND equal terminal mismatch.

If a canonical cell reaches that joint parity and then matches/beats the unweighted
reference in generation, the earlier canonical sampling deficit was a terminal-anchor
confound rather than a failure of canonical score geometry.

If canonical remains worse at joint parity, the remaining diagnostics tell us whether
the residual comes from late-t score error, CFG, or some other representation geometry.
