# Canonical CSEM 40-epoch screening sweep

This package generates a deterministic, bundled TACC sweep for the failure
window observed during epochs 11--40. The default design contains 46
configurations: six named controls plus 40 all-pairs configurations. Every
listed value and every two-factor value combination is covered, including all
30 CSEM-weight x terminal-KL-weight combinations. Each configuration runs both
`terminal_kl` and `norm` from the same seed (92 arm runs total).

The literal Cartesian product contains 12,441,600 configurations and is guarded
against accidental submission.  The pairwise design is the intended first
screen.

## Files to place together

- `csem_canonical_stable_20260813.py` -- extended trainer implementation.
- `csem_canonical_sweep_train_20260815.py` -- unique sweep entry point.
- `csem_fmnist_40ep_sweep_space_20260815.json` -- declarative search space.
- `prepare_submit_csem40_sweep_20260815.py` -- generator and submitter.
- `run_csem40_bundle_20260815.py` -- compute-node bundle worker.
- `analyze_csem40_sweep_20260815.py` -- aggregator, rankings, and plots.

Copy all six files into:

```text
/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist_run1
```

The filenames are intentionally unique; none reuses an earlier sweep or Slurm
script name.

## Generate first, inspect, then submit

On a Vista login node:

```bash
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist_run1

python3 prepare_submit_csem40_sweep_20260815.py \
    --config csem_fmnist_40ep_sweep_space_20260815.json \
    --strategy pairwise
```

This creates, without submitting:

```text
csem_fmnist_canonical_40ep_screen_20260815/
  run_manifest.csv
  run_manifest.json
  bundle_manifest.json
  bundles/
  slurm_jobs/
  logs/
  status/
  results/
  summary/
```

Inspect `run_manifest.csv`.  Then submit all generated bundles:

```bash
python3 prepare_submit_csem40_sweep_20260815.py \
    --submit-existing \
    /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist_run1/csem_fmnist_canonical_40ep_screen_20260815
```

Alternatively, generation and submission can be performed together with
`--submit`, but the two-step workflow makes the job count and commands visible
before anything reaches the scheduler.

The submit parser searches the complete TACC banner for
`Submitted batch job <id>`, so the Vista preamble does not cause the earlier
job-ID parsing failure.

## What is swept

- CSEM coefficient: 0.05, 0.092, exactly 1/7, 0.25, 0.5, 1.0.
- Terminal-KL coefficient: 0 (no anchor), 0.15, 0.5, 1.0, 1.5.
- Terminal horizon: 1.0, 1.5, 2.0.
- Encoder LR: 5e-5, 7.5e-5, 1e-4, 2e-4, 5e-4.
- Decoder LR: 7.5e-5, 2e-4, 5e-4.
- Score-head LR: 3e-5, 1e-4, 2e-4, 3.33e-4.
- AdamW beta2: 0.9, 0.95, 0.99, 0.999. This tests how quickly second-moment
  estimates adapt when encoder CSEM gradients enter after warmup.
- Warmup: immediate ramp, frozen encoder for 5 or 10 epochs, and the old
  moving-encoder/detached-CSEM warmup.
- Same-head tracking: disabled, one step every two batches, one per batch, or
  two per batch.
- Log-variance floor: -12, -8, -6.
- VAE/score/discriminator clipping policies: (1,1,1), (5,1,1), (1,5,1),
  (5,5,5).
- GAN disabled, starting at epoch 40, or starting at epoch 25.

Named controls additionally reproduce the stabilized run, test the proposed
`beta=0.15` split-LR configuration, test full-speed canonical training, run
the historical unweighted objective at `T=1.5`, and test an exact sevenfold
CSEM reduction both without and with sevenfold score-head LR compensation.

The pairwise rows use direct encoder, decoder, co-trained score-head, and
auxiliary tracking-head LRs, so `canonical-lr-scale` is fixed to 1 there. The
named stable reproduction retains `canonical-lr-scale=0.15` for exact command
provenance. Direct LR flags take precedence, so every optimizer's effective LR
is visible in the manifest.

## Important comparability choices

- Training stops at epoch 40.
- The cosine LR horizon remains 700 epochs.  Therefore epoch 40 has the same LR
  as epoch 40 of the intended long run rather than being annealed to 1e-6.
- Both scale-anchor arms run for every configuration.
- The two objective coefficients are crossed independently. Rows where they
  match test the canonical mathematical synchronization; mismatched rows are
  explicit optimization ablations.
- Evaluation occurs only at epoch 40 with 2,000 samples.
- Component gradients are sampled every 200 batches.
- Every run has its own results directory, stdout log, and atomic status file.
- Seed 42 is reset identically for both arms of every configuration.
- Two configurations are bundled sequentially per six-hour GPU job, producing
  23 Slurm jobs for the default 46-configuration design.
- A failed configuration does not prevent the next configuration in its bundle
  from running.

The supplied stable log was about 44 seconds per arm-epoch. On that hardware,
the training portion is roughly one GPU-hour per two-arm configuration, or
about 46 GPU-hours across the screen, plus final evaluations. Two configurations
per six-hour allocation leaves substantial margin. If a different GPU is much
slower, generate with `--runs-per-bundle 1` instead (46 shorter Slurm jobs).

## Analyze results

The analyzer can be rerun while jobs are still active; it consumes completed
or in-progress loss histories:

```bash
source "$SCRATCH/venvs/hlsi/bin/activate"

python analyze_csem40_sweep_20260815.py \
    /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist_run1/csem_fmnist_canonical_40ep_screen_20260815
```

Principal outputs are placed in `summary/`:

- `all_run_arm_summary.csv` -- one row per configuration and arm.
- `leaderboard_terminal_kl.csv` and `leaderboard_norm.csv`.
- `all_trajectories.csv` and `all_evaluations.csv`.
- `factor_effects.csv` -- marginal effect of every factor value.
- `paired_arm_deltas.csv` -- within-configuration terminal-KL minus
  normalization contrasts.
- `README.md` -- concise ranked report.
- scatter plots and top-configuration trajectory plots.

The balanced rank combines reconstruction FID, generated FID, reconstruction
degradation, variance-floor saturation, score clipping, raw score MSE, and
posterior stability.  Separate quality and stability ranks are also retained;
the scalar balanced score should not be treated as the only scientific
criterion. The intended follow-up is to rerun a small Pareto set (not just the
single rank winner) at multiple seeds for 700 epochs and 10,000-sample
evaluation.

## Resume behavior

Completed runs are skipped by the bundle worker.  A result directory without a
valid completed status is never overwritten automatically.  This prevents a
resubmission from destroying partial diagnostics.  To rerun a failed case,
move its result directory and status file to an archival location first, then
submit the relevant generated Slurm file again.

## Literal full Cartesian product

The following is implemented but intentionally guarded and is not operationally
recommended as the first pass:

```bash
python3 prepare_submit_csem40_sweep_20260815.py \
    --config csem_fmnist_40ep_sweep_space_20260815.json \
    --strategy full \
    --allow-large-sweep
```

It would create 12,441,600 factor configurations before named controls and
deduplication, each containing two arms. Use the 46-configuration all-pairs
screen to narrow the ranges first.
