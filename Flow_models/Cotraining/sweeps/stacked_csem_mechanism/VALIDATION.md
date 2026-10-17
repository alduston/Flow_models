# Validation record

All checks below passed locally with Python 3.12.14, Torch 2.14.1+cpu and torchvision 0.29.1+cpu.

- Exact two-block baseline target, loss, reconstruction and gradient parity.
- Encoder-only score-gradient scaling at 0, 0.5, 2, 4, with unchanged head gradients.
- Two/three-block Gaussian teacher identities and every posterior-DAG validity pattern.
- Nonlinear three-block mean AND variance derivatives checked against independent prior-space Gauss-Hermite density quadrature/finite differences.
- Independent learned-witness lower bound detects a known Gaussian regression bias and respects a zero-bias control.
- Real-network decoupled zero correction, invertible 256-coordinate packing, all posterior topologies.
- Full 192x18 head forward/backward at both depths, including identical initial head tensors.
- Core and learned-witness evaluations preserve model weights, existing parameter gradients, CPU RNG and modes.
- Bitwise interrupted/resumed joint training with active GAN, and frozen corrected-head training; optimizer/scheduler/discriminator/EMA/RNG states checked.
- Uncommitted-log rollback and recovery from a final-checkpoint/snapshot interruption.
- Synthetic train/sample/score/witness/checkpoint pipelines for two-block full/decoupled and three-block full/Markov/siblings/decoupled models.
- Three-block finite-bank reference pipeline, both class and null conditioning, nested image banks and raw/summary outputs; decoupled single-component identities at both depths.
- Nonempty mechanism report, plots and paired frozen-head valid-risk compilation, plus empty/partial report handling.
- Bash syntax and a dry-run submission plan for all 54 jobs, including the 24 frozen-source dependencies.
- Gaussian probability-flow covariance solver reproduces the valid-score/oracle-initialized covariance with maximum relative error 5.57e-10 over the included cases.

The final change to use a training-log-time grid for pressure probes was followed by targeted isolation and full/decoupled three-block pipeline checks.

These are analytic/software checks, not full-model scientific outcomes. GPU execution, prepared data and weight availability, real 10,000-sample FIDs and the 500-epoch runs remain cluster work. Synthetic pipeline FIDs are explicitly labeled feature-surrogate values. Raw CPU validation logs and exact Gaussian calculations are in `validation/`; `validation.json` is machine-readable. No cluster jobs were submitted here.
