# Validation record

Prepared 13 October 2026. Offline runtime: Python 3.12, PyTorch 2.14.1+cpu,
torchvision 0.29.1+cpu. Production jobs reuse the installed cluster CUDA pair.

Passed checks:

- **Original CIFAR recipe parity:** all existing nonarchitectural resolved
  settings match the original CIFAR optimized cell (seeds/output identities
  excluded). `core.py` and `suite.py` retain their exact recovered bytes.
  Canonical representation CSEM 0.05, terminal KL 0.60, B coefficient 0.05,
  T_K/T 1.05/1.35, OU-partial anchoring, batch 256, VAE/DiT capacity, learning
  rates, losses, full evaluation grid and 800-epoch cosine duration are checked.
- **Eight full-size CPU model checks:** actual RGB CIFAR model construction,
  fixed-draw conditional Gaussian sampling, root packing isometry, Gaussian
  targets checked against a log-density derivative, clean-root ancestry,
  newly recomputed noisy-root decoder features, and direct child injection.
  Tested coordinate-weighted B with no head gradient, canonical outer score
  gradients, detached independent inner head gradients, original gradient
  replacement and clipping, an AdamW update, finite values and real sequential
  reverse solves. All eight passed. Counts are in `smoke_validation.json`.
- **Actual architecture-note parity:** extracted the supplied direct-FMNIST
  VAE classes from the pinned source, scaled shapes to CIFAR, mapped equivalent
  model weights, and obtained bit-identical direct decoder outputs and raw
  conditional child-posterior outputs. CIFAR's explicit per-factor OU-partial
  mean anchor is applied separately at the encoder boundary.
- **Guarded trainer execution:** the actual adapted CIFAR trainer performed
  two synthetic minibatch optimizer updates at reduced feature width, retained
  positive B coefficient/logging and independent gradient routing, and saved
  its full optimizer/scheduler/EMA/RNG checkpoint. Resuming the same completed
  configuration preserved histories and model/scheduler states. This check
  disabled LPIPS/GAN and substituted evaluation values; production retains
  their original coefficients. It was an implementation check, not training
  evidence for CIFAR quality.
- **Evaluator IO:** actual two-block generation, posterior reconstruction,
  fixed-root conditional generation, child interventions, independent child
  draws, timing and score-gap calculations under a tiny synthetic feature
  extractor. RNG was preserved and parameter `.grad` fields untouched.
  These synthetic feature scores are not Inception/CIFAR FIDs.
- **Compiler:** empty results, eight completed fixture cells, and partial/error
  fixture results. Verified 32 paired contrasts, exclusion of T_K/oracle metrics
  from the primary table, all CSVs, six figures/PDF pages, missing epoch-800
  handling and strict-mode rejection. Fixture output is excluded from the ZIP.
- **Cluster contracts:** source hashes, eight-cell selection, Slurm job-ID
  parsing, eight-job submission dry run, Python parsing and `bash -n job.slurm`.

No cluster jobs were submitted. CUDA behavior, full-data CIFAR training with
LPIPS/GAN and 10000-image Inception measurements remain for the cluster run.
The ZIP contains no checkpoints, synthetic FID results, cache or completed
status files.

Optional deeper checks in the activated environment:

```bash
python -c 'import tests; tests.recipe_checks(".")'
python -c 'import tests; tests.architecture_note_parity(".")'
python -c 'import tests; tests.trainer_io_test(".")'
python -c 'import tests; tests.evaluator_io_test(".")'
```
