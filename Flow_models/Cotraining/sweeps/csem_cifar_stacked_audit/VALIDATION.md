# Validation performed — 21 October 2026

All checks below passed locally on CPU with PyTorch 2.14.1+cpu / torchvision 0.29.1+cpu. These versions describe the validation environment, not a requirement to upgrade the cluster environment.

- Immutable-source checksum checks for recovered CIFAR core, config resolver, IO, direct VAE, hierarchy helpers and boundary adapter.
- Recipe equality against the recovered optimized flat+B control for every joint arm, with only predeclared architecture/pressure/diagnostic/evaluation changes. The head coefficient and learning rate, horizons, anchor, terminal penalty and B coefficient remain fixed.
- Guarded AST adapters compile the actual pinned training and evaluation functions; unexpected source shapes are rejected.
- Reduced-capacity real RGB VAE / hierarchical posterior / decoder / factored DiT graphs, including coherent Gaussian samples and direct-code decoding.
- Same-seed initial head state equality across flat and stacked arms; head architecture has the original CIFAR capacity in production.
- On common posterior/noise draws, r=0 removes score-derived VAE gradients, r=4 quadruples them, while head gradients are identical to r=1.
- Exact root pack/unpack round trip, and numerically zero corrected-teacher defect in the conditionally decoupled control.
- Corrected-teacher conditional-mean identity against an analytic correlated Gaussian model with one root and three child coordinates; Monte Carlo tolerance specified in `tests.py`.
- Held-out diagnostics / witness fitting and raw paired per-image exports run on synthetic RGB samples.
- Actual adapted routed trainer updates on synthetic data: **bitwise equality** of VAE, online head and EMA between uninterrupted training and interrupted / resumed training, including pending evaluation.
- Actual original flat+B trainer versus new flat path: **bitwise equality** of VAE, online head and EMA after common synthetic training updates.
- Actual naive, corrected and DSM frozen optimizer updates and **bitwise equality** of uninterrupted / resumed frozen head and EMA.
- Actual coherent A/B posterior-bank evaluator and joint RK4 / Heun paths, with synthetic features standing in for Inception; legacy metric columns and disabled invalid joint-mixture gap are checked. These numerical outputs are not FID measurements.
- Compiler fixture: all 24 endpoints are correctly recognized, paired valid-risk deltas have the expected sign/value, missing final epochs are rejected in strict mode, and partial endpoints are not promoted.
- Launcher dry-run covers all 24 cells and their frozen-source dependencies; Vista banner and `ID;cluster` job-ID parsing is retained and tested.

Not executed locally: CUDA, real CIFAR training, LPIPS/GAN training with pretrained features, real 10,000-image Inception FID, Slurm submission, or cluster wall-clock measurement. The job script performs a full-capacity CUDA model/gradient smoke in its own process before training. Synthetic training checks use smaller networks, no GAN/LPIPS loss and a stubbed evaluator so they do not require downloads.

To repeat:

```bash
python paper.py validate
python tests.py
python paper.py smoke --cell-id 6 --device cuda
```

`paper.py validate` checks package / recipe / adapters. `tests.py` runs the deeper CPU checks. CPU `smoke` uses reduced capacity; CUDA `smoke` uses the production CIFAR model dimensions. The optional torchmetrics warning is inherited; this bundle's FID is computed by the original NumPy/SciPy routine on extracted legacy Inception features.
