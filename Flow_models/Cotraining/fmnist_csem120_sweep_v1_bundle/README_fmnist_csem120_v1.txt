CSEM FMNIST 120-cell screening sweep — v1
==========================================

Expected directory:
  /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep/

The directory already contains:
  csem_split_metric.py

Copy these new files beside it:
  fmnist_csem120_grid_v1.py
  fmnist_csem120_bundle_runner_v1.py
  fmnist_csem120_bundle_job_v1.slurm
  submit_fmnist_csem120_bundles_v1.py
  fmnist_csem120_manifest_v1.csv   (pre-generated; grid builder can regenerate it)

Sweep grid
----------
2 outer metrics:
  unweighted-eps, canonical

2 score-head metrics:
  unweighted-eps, canonical

6 matched representation coefficients:
  csem_w = terminal_kl_w in {0.05, 0.10, 0.20, 0.40, 0.60, 1.00}

5 independent score-head learning rates:
  {5e-5, 1e-4, 2e-4, 4e-4, 8e-4}

Total:
  2 x 2 x 6 x 5 = 120 configurations

Bundling
--------
There are 20 ordinary Slurm jobs, NOT an array.
Each job fixes:
  (outer metric, score-head metric, score-head LR)
and runs the 6 representation coefficients sequentially.

At the conservative budget of 60 s/epoch:
  120 epochs/run = 2 hours/run
  6 runs/job = 12 hours/job
The Slurm walltime is set to 16:00:00 to leave room for evaluation/setup/zipping.

Fixed screening settings
------------------------
dataset: FMNIST
model preset: auto -> fmnist_reference
arm: terminal_kl only
epochs: 120
refinement: 0
evaluation: epochs 60 and 120
eval samples: 2000
T_terminal: 1.50
CFG: 3.0
score-head-loss-w: 1.0
bespoke FID classifier: disabled
canonical LR compensation: 1.0 (none)
encoder warmup: 0
CSEM ramp: 0
extra score tracking: 0
grad diagnostics: 0
fail-on-nonfinite: disabled, matching the supplied current Slurm run
seed: 42, fixed internally by csem_split_metric.py

Setup
-----
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep/

# Optional: regenerate the exact 120-row manifest.
python3 fmnist_csem120_grid_v1.py

# Strongly recommended before submission:
python3 submit_fmnist_csem120_bundles_v1.py --dry-run

# Submit all 20 ordinary jobs:
python3 submit_fmnist_csem120_bundles_v1.py

# Submit/re-submit only selected bundles:
python3 submit_fmnist_csem120_bundles_v1.py --bundles 3,7-9

Outputs
-------
Per-configuration results:
  fmnist_csem120_results_v1/

Per-configuration full logs:
  fmnist_csem120_config_logs_v1/

Per-configuration status JSON:
  fmnist_csem120_status_v1/

Slurm bundle logs:
  slurm_logs_csem120_v1/

Submission receipts:
  submission_receipts_csem120_v1/

Resubmission behavior
---------------------
If a configuration has a status JSON with returncode 0 and its results directory
still exists, the bundle runner skips it. This lets you safely re-submit a bundle
whose later configurations failed.

If a results directory exists without a successful status record, the runner
refuses to overwrite it. Remove or rename that partial configuration directory
explicitly before retrying it.

TACC sbatch parsing
-------------------
The submitter does not assume clean sbatch stdout. It searches the full banner
and output for:
  Submitted batch job <jobid>
and also accepts a bare parsable numeric job-id line as a fallback.
