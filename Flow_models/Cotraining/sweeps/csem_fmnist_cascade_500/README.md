# FashionMNIST CSEM: eight-run cascade sweep, 500 epochs

Fresh training only. Every arm is a stochastic diagonal-Gaussian VAE, uses CSEM
representation weighting and the analytic lower-boundary B_t penalty, and has
256 latent coordinates. The reference two-block model is the recovered
coarse-to-fine residual trainer with **noisy-root reconstruction restored**.
The exact recovered trainer and the attached CIFAR compiler are retained under
`pinned/`; this bundle includes every required Python source.

## Run on Vista / Frontera

Extract the ZIP, enter `csem_fmnist_cascade_500/`, activate your existing `hlsi`
environment, then:

```bash
python paper.py validate
python paper.py prepare-data --weights
python paper.py submit --cells all --dry-run
python paper.py submit --cells all
```

`prepare-data --weights` pre-caches FashionMNIST, Inception and LPIPS/VGG on the
login node. Training jobs also validate/prepare the dataset under a shared lock;
there is no dependency on a preparation job ID. Network access is needed only
when data or pretrained weights are absent. The original launcher settings are
preserved: partition `gh`, one node/task, 48 hours, gcc/13.2.0, python3/3.11.8,
`$SCRATCH/venvs/hlsi`. Set `VENV_DIR` for another existing environment, or use
`submit --partition ... --time ... --account ...` for site overrides. The job
executes a real full-size CUDA model/autograd/sampling smoke before training.

**Cache name:** `data_cache_fmnist/`. It contains
`FMNIST/FashionMNIST/raw/`. To reuse a prepared sweep cache, pass its parent:

```bash
python paper.py prepare-data --data-root ../other_sweep/data_cache_fmnist --weights
python paper.py submit --cells all --data-root ../other_sweep/data_cache_fmnist
```

An older standalone `./data/FashionMNIST/raw/` cache has a different layout;
copy its `FashionMNIST/` directory into `data_cache_fmnist/FMNIST/` before the
prepare command. Existing checkpoints are not imported into this fresh sweep.

## Eight runs

| Cell | Variant | Seed | Root / child coordinates | Spatial grids | Score depth |
|---:|---|---:|---|---|---|
| 0 | Flat diagonal-Gaussian baseline + B | 42 | 256 | 4 channels at 8×8 | 8 |
| 1 | Recovered optimized residual cascade + B | 42 | 64 / 192 | 4 channels at 4×4; 3 at 8×8 | 8 + 8 |
| 2 | Flat baseline replication | 43 | 256 | 4 channels at 8×8 | 8 |
| 3 | Optimized cascade replication | 43 | 64 / 192 | 4 channels at 4×4; 3 at 8×8 | 8 + 8 |
| 4 | Cascade with matched total trunk depth | 42 | 64 / 192 | 4×4 → 8×8 | 4 + 4 |
| 5 | Balanced stochastic allocation | 42 | 128 / 128 | 8 channels at 4×4; 2 at 8×8 | 8 + 8 |
| 6 | Cascade at one spatial resolution | 42 | 64 / 192 | 1 channel at 8×8; 3 at 8×8 | 8 + 8 |
| 7 | Cascade with bottom-up child posterior | 42 | 64 / 192 | 4×4 → 8×8 | 8 + 8 |

Cells 0/1 and 2/3 are paired by training seed. They answer whether the original
flat-versus-optimized difference repeats. Cells 4–7 change one structural axis
relative to cell 1 and are exploratory single-seed runs:

- **4: score allocation/capacity.** The successful trainer defaults to eight
  trunk layers per head. Halving both heads separates that score-capacity
  contribution from the headline comparison. Matching total trunk depth does
  not exactly match parameters or FLOPs: the conditional child has cross-attention.
- **5: latent allocation.** Give the coarse root half rather than a quarter of
  the stochastic budget. This tests whether root-prior quality is constrained
  by the 64-coordinate bottleneck.
- **6: spatial organization.** Preserve 64/192 coordinates while moving the
  root to 8×8. The root posterior stem uses stride 1 rather than 2, and the
  top-down path retains the same convolution/residual/attention modules with
  scale-1 rather than scale-2 interpolation. This tests the coarse-to-fine bias
  against a same-resolution cascade; spatial attention cost also changes.
- **7: inference hierarchy.** Change `q(u2|x,z1)` to `q(u2|x)` by removing
  top-down input from the child posterior. The learned diffusion prior
  `p(u2|z1)`, clean-root score context, residual child baseline, and top-down
  decoder remain conditional. This isolates the benefit of root-dependent
  inference from a cascade with a factorized innovation posterior.

All cascade arms retain full-root spatial injection and cross-attention at
every child score layer. They sample the root first, then the residual child
with that fixed generated root. These are learned diffusion priors for both
stochastic blocks; no child prior is replaced by an untrained Gaussian.

## Shared FMNIST recipe

| Setting | Value |
|---|---|
| Epochs / refinement | 500 / none |
| Batch / VAE feature width | 128 / base_ch 32 |
| DiT width / attention heads | 192 / 6 |
| VAE LR / effective head LR setting | 5e−4 / 2e−4 |
| Actual score optimizer LR | 2e−4 / 0.6, as in recovered trainer |
| Scheduler | 500-epoch cosine; minimum LR 1e−6 |
| Encoder CSEM / score-head coefficient | 0.6 / 0.6 |
| Terminal KL coefficient | 1.0 |
| B coefficient | `auto`, resolved from representation CSEM and the actual grid |
| OU schedule | 1000 log-time nodes; t_min=2e−5; T=2.0 |
| LPIPS / GAN | 0.85 / 0.0025; discriminator starts epoch 25 |
| EMA / class-label dropout | 0.9997 / 0.10 |
| AdamW / clipping | Inherited betas/decay; one joint norm-1 clip |
| Evaluation cadence | Every 50 epochs through 500 |
| Primary FID | 10000 test images, legacy Inception, Gaussian full-T, CFG=3, RK4-20 per block |
| Readout time | 1e−4 |
| Checkpoint cadence | Every 10 epochs, every evaluation, and epoch 500 |

The successful noisy-root run's prior gradient logs identify encoder CSEM 0.6.
The earlier proposed `--csem-rep-weight 1.2` test is not used here. Scientific
settings are explicit in `manifest.csv`, rather than relying on trainer defaults.
This is a controlled comparison inside the recovered trainer. The older roughly
5-FID result used a different training duration/protocol and is contextual, not
an extra matched control arm.

The recovered trainer samples log-time indices uniformly and uses an epsilon
loss. Preserve its own small-time representation metric:

```text
log_floor = log(sigma_min² / alpha_min²)
B = 0.5 * mean_coordinates softplus(log_floor - conditional_logvar)
band_weight = csem_rep_weight * sigma_min² / (N * first_grid_spacing)
```

For this grid and coefficient, `band_weight` is approximately **0.10354**.
It is not automatically 0.6 and is independent of terminal KL 1.0. CSEM, KL
and B average over all 256 coordinates; e.g. the 64/192 model uses weights
0.25/0.75, and the 128/128 model uses 0.5/0.5. B reaches posterior/ancestral
parameters and has no score-head gradient. The child rate/diffusion/B terms
operate in normalized innovation coordinates `u2`, not raw child coordinates.

**Reconstruction uses noisy roots in every arm.** Both blocks are corrupted at
the same sampled OU time; the decoder recomputes its entire root feature path
and child location/scale from the corrupted root. Clean-root features are used
only for child posterior inference and exact conditional CSEM score training.
There is no image-feature skip into generation. Child score context is kept
clean because corrupting it while retaining the inherited conditional Gaussian
target would change the estimated score problem and need a separate derivation.

## Precedent and scope

The architectural axes draw on residual parameterization/top-down inference in
[NVAE](https://arxiv.org/abs/2007.03898) and the multiresolution generative hierarchy
in [VDVAE](https://arxiv.org/abs/2011.10650). They are design hypotheses for this
small CSEM model, not reproductions of those full systems. The cascade literature
also identifies lower-stage conditioning mismatch as important:
[Cascaded Diffusion Models](https://jmlr.csail.mit.edu/papers/v23/21-0635.html).
Here we preserve the proven noisy-root reconstruction fix and track
encoded-root/generated-child quality; that fix is not an assertion that this
implementation reproduces the paper's diffusion conditioning augmentation.

The root and child affine residual map stays
`raw_z2 = b(root) + exp(log_d(root))*u2` with bounded log scale. The generative
baseline depends only on the root. A conditionally diagonal posterior is retained
in all arms so the analytic conditional CSEM epsilon target remains valid.

The flat model has about 3.680M VAE and 5.489M score parameters. The optimized
model has about 6.376M VAE and 12.167M score parameters. Cell 4 has about 6.236M
score parameters. Parameter counts are logged in config/evaluation records;
the main comparison measures the complete optimized architecture, not isolated
block count at exactly equal model capacity. Runtime is measured separately.

Two training seeds are a useful repeatability check, not a precise significance
estimate. Four exploratory architectures cannot establish their own seed
robustness. The fixed epoch-500 endpoint is primary; best-over-epochs is secondary.
The new epoch-300 checkpoint uses the 500-epoch cosine trajectory and therefore
does not recreate an earlier training run with a 300-epoch cosine trajectory.

## Diagnostics and result compilation

```bash
python compile_results.py
python compile_results.py --strict
# Equivalent:
python paper.py report --strict
```

The compiler retains the analogous CIFAR bundle's cross-cell CSV, Markdown,
PNG and multipage-PDF structure, adapted to native FMNIST logs and all eight arms:

- `compiled/SUMMARY.md`, `leaderboard.csv`, `run_status.csv`.
- `fid_primary_curve.csv`, `fid_primary_wide.csv`, `all_eval_records.csv`,
  `all_loss_history.csv`, `gaussian_fullT_metrics_long.csv`.
- `paired_seed_contrasts.csv`: both flat/cascade seed pairs at shared epochs.
- `architecture_contrasts.csv`, `architecture_seed_summary.csv`.
- `sampling_checks.csv`, `child_usage_checks.csv`, `gradient_checks.csv`,
  `csem_routing.csv`.
- `figures/`: six plots; `eight_cell_review.pdf`: combined six-page review.

Every evaluation measures posterior reconstruction FID and, for cascades,
encoded-root/generated-child FID. Fixed-root child zero/shuffle/two-draw probes,
root/child losses and variances, and pre-update objective-gradient diagnostics
are inherited from the successful trainer.

Primary FID always uses the same 10000-example, CFG-3, RK4-20-per-block protocol.
The **separate 1024-example paired subset** checks CFG 1 versus 3, 20 versus
40 steps per block, and 10 steps per block in a cascade. Flat 20 versus cascade
10 matches 80 total RK4 stages; network FLOPs remain different. Those subset
FIDs are never substituted into the primary leaderboard. Generation+decode
time uses synchronized CUDA timing and excludes feature extraction/reconstruction
diagnostics. It includes all sequential generation solves and the image decoder.

Missing epoch-500 results remain blank, including during partial compilation.
No partial result is reported as a completed endpoint. `--strict` requires
complete status, all ten scheduled evaluations and finite final diagnostics.

## Status, resume, and useful commands

```bash
python paper.py submit --cells paired
python paper.py submit --cells variants
python paper.py smoke --cell-id 6 --device cuda
python paper.py smoke --device cpu
python paper.py run --cell-id 1 --resume
python paper.py submit --cells all --restart-incomplete --dry-run
```

Completed/active jobs are skipped. Submission and per-cell run locks prevent
duplicate writers. A checkpoint is atomically replaced only after that epoch's
evaluation completes. Resume rolls CSVs back to the committed checkpoint epoch
and restores optimizer, scheduler, EMA and RNG states. Source/manifest and recipe
guards reject changed architecture, coefficient or duration. Incomplete runs
without a checkpoint need `--restart` (or `submit --restart-incomplete`); the
previous output is archived under `failed_attempts/`. Completed runs are retained.

The source exposes `train_hier_csem_fmnist.py` for standalone use, but `paper.py`
is the intended sweep entry point and ensures manifest/status consistency. This
bundle also ships `.gitignore`, so result/cache files stay out of source control.

## Validation

`paper.py validate` requires no GPU and checks eight-arm settings, source hashes,
the coordinate budget, CLI flags, Slurm parser/selection, and endpoint contracts.
`paper.py smoke` uses actual full-size PyTorch models, analytic-score checks,
coordinate weighting, noisy-root routing, band-only gradient routing, one
optimizer update and real sequential RK4 sampling. `tests.default_parity` compares
the default model/loss tensors to the pinned recovered source, and
`tests.trainer_resume_test` checks interrupted versus uninterrupted optimization.
See `VALIDATION.md` for checks performed before packaging. Cluster training and
measured FashionMNIST FID are the work this ZIP schedules, not locally generated
results.
