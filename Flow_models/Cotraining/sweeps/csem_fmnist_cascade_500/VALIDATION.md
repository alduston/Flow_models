# Validation performed before packaging

Date: 2026-10-13. Local model runtime: PyTorch 2.14.1+cpu,
torchvision 0.29.1+cpu, Python 3.12. The bundle uses the existing cluster CUDA
environment and does not pin those CPU wheels for training.

Passed:

1. **Eight full-size model smokes.** Real VAE/DiT construction for every arm,
   synthetic 32×32 grayscale input, analytic Gaussian epsilon targets checked
   against a log-density derivative, correct coordinate/B weighting, noisy-root
   reconstruction routing, top-down versus bottom-up inference dependency,
   B-only VAE gradient routing, finite joint gradients, an optimizer update,
   and real sequential RK4 solves. Counts and band coefficients are recorded
   in `smoke_validation.json`. These values are test losses, not data/FID results.
2. **Recovered-source parity.** The default optimized arm has bit-identical
   VAE/head initial states, RNG state after construction, reconstruction tensors,
   CSEM loss, KL, B, and automatic B coefficient compared with the recovered
   trainer in `pinned/` under fixed draws.
3. **Actual training and resume.** Full-size hierarchical model, two synthetic
   one-minibatch epochs, inherited AdamW and cosine scheduler. An interruption
   after epoch 1 plus resume produced bit-identical VAE, heads, EMA, scheduler
   and Torch RNG to uninterrupted training. Verified atomic checkpoints,
   rollback of an uncommitted CSV row, positive B coefficient, and rejection
   of a changed architectural recipe. LPIPS and GAN were disabled in this
   synthetic IO test; the sweep manifest retains their original coefficients.
4. **Evaluation IO.** Real flat, same-resolution and bottom-up models ran the
   generation/reconstruction path, timing, same-start CFG/step checks, equal
   total-step control, child replacement/two-draw probes, and online-gradient
   audits. Checked output CSV schemas, clean-root conditional generation,
   finite values, unchanged RNG, and untouched parameter `.grad` buffers.
   This check used synthetic low-dimensional features instead of downloaded
   Inception weights; it is not a measured FashionMNIST FID result.
5. **Compiler.** Empty results, all eight completed synthetic fixtures, and
   partial/error fixtures. Verified exact paired differences, endpoint handling,
   strict rejection of missing epoch-500 records, all cross-cell CSV exports,
   six PNG figures, and a six-page PDF. Visually inspected the comparison plot.
   Fixture files are excluded from the deliverable.
6. **Cluster contracts.** `paper.py validate`, eight-job submission dry run,
   Slurm banner/parsable-ID handling, cell selection, source hashes, manifest
   coordinate budgets, Python parsing, and `bash -n job.slurm`.

No cluster jobs were submitted from this workspace. CUDA execution, full-data
training with LPIPS/GAN, and 10000-image Inception FIDs will be measured by the
scheduled sweep. No trained checkpoints, synthetic FID fixtures, dataset cache,
or fake completed statuses are included in the ZIP.

The deeper optional offline checks can be rerun in the activated environment:

```bash
python -c 'import tests; tests.default_parity(".")'
python -c 'import tests; tests.trainer_resume_test(".")'
python -c 'import tests; tests.evaluation_io_test(".")'
```
