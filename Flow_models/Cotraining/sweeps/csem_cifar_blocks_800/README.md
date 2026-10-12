# CIFAR-10 CSEM hierarchical cotraining — repaired five-cell sweep

This is a replacement for the original six-cell CIFAR bundle. **Three-block
models are removed.** Cell 5 is an entirely new, incompatible v3 cascade,
reconstructed from the October 11 FMNIST residual-cascade experiments and
`hierarchical_iterative_CSEM_cotraining_note.tex`.

## Run all five cells

Activate the original `hlsi` environment on Vista, extract the complete zip,
enter its inner directory and run:

```bash
python paper.py validate
python paper.py submit --cells all
```

For a dry run, add `--dry-run`. The submit command schedules **five** independent
Slurm GPU jobs; it does not execute training on the login node. Each job prepares
or verifies CIFAR data under the existing shared cache lock. No separate prep
job or dependency is necessary. To reuse a cache in another directory, pass
`--data-root /path/to/data_cache_v6` to `submit`.

| Cell | Architecture | B at t_min | Latent layout | DiT depth |
|---|---|---|---|---|
| 0 | Original flat CSEM | No | 8 × 8 × 8 | 12 |
| 1 | Original two-block conditional CSEM | No | (4+4) × 8 × 8 | 6+6 |
| 3 | Original flat CSEM | Yes | 8 × 8 × 8 | 12 |
| 4 | Original two-block conditional CSEM | Yes | (4+4) × 8 × 8 | 6+6 |
| 5 | New v3 two-stage normalized residual cascade | Yes | root 8 × 4 × 4; child 6 × 8 × 8 | 6+6 |

All cases use 512 latent coordinates, 800 joint epochs, CIFAR-10 seed 42,
OU-partial means, the pinned 0.05 encoder CSEM coefficient, the independent
score-head objective and optimizer, 0.6 terminal KL, full evaluation grids and
original data augmentation/LPIPS/GAN/EMA/scheduler. The analytic `B` coefficient
is 0.05 in cells 3–5. **All baseline `core.py`, `suite.py`, `paper_io.py` and
source-hash compatibility checks remain unchanged.** The new cascade has larger
parameter count, so this is a matched latent-budget/DiT-depth, not perfectly
matched FLOP or parameter, comparison.

## Why v3 differs from the unsuccessful optimized cascade

The original `2:6` cascade logged FID 92.52 at epoch 50, 54.09 at 100, 61.24
at 150, and 82.34 at 200 (RK4-25, CFG 2.5). Its reconstruction loss improved
to ~0.022 by 200, suggesting a severe learned-prior/decoder compatibility
problem rather than simply a reconstruction-capacity failure. At epoch 200,
posterior variance mean/median were 0.416/0.076, respectively (aggregated
across coordinates). In comparison, the flat control progressed to FID 11.23
at epoch 250, with posterior variance mean/median 0.05464/0.05500.
These observations do **not** establish the root cause: the old optimized log
does not provide sufficiently detailed root-vs-child FID or child-use metrics.

The FMNIST evidence motivating v3 is stronger than the slogan 'noisy root':
**the old CIFAR optimized version already reconstructed from jointly noisy
latents**, so that aspect alone does not explain the 80-FID failure. The new
model incorporates the *full* demonstrated formulation:

- An 8 × 4 × 4 Gaussian root inferred from downsampled CIFAR features, with
  additional residual/attention processing; it is exactly packed to 2 × 8 × 8
  for compatibility with the unchanged training/sampling core.
- A **single sampled clean root** used in the conditional child posterior and
  analytic CSEM supervision. A conditional child *innovation*, not the raw
  child decoder feature, is diffused and subject to terminal KL and B.
- Shared root-dependent 4→8 top-down blocks, an initialized bounded log scale
  and location baseline, and residual child feature fusion.
- Spatial root injection **and per-layer cross-attention to all 4 × 4 root
  tokens** in the child's DiT. No image features leak into the learned score
  model.
- Reconstruction from D(z1_t, u2_t, t) with **all** root-derived decoder
  features recomputed from the noisy root. Conditional score/analytic target
  context remains the clean sampled root, as mathematically required.
- Exactly two sequential reverse diffusion solves, root first and conditional
  child second, followed by decoding.
- Cutoff closure B at 0.05 with posterior-coordinate normalization across the
  full 512-coordinate latent bank and unchanged gradient routing.

The new `optimized_v3` cell has **distinct result directory and status file**
(`status/cell_005_v3.json`) to prevent older cell-5 Slurm jobs/checkpoints from
silently satisfying its completion checks. Old v2 checkpoints **cannot** be
resumed as v3. If running in the old live directory, cancel obsolete v2 jobs
to avoid wasting GPU resources; the new job uses a separate result name.

## Diagnostics and tests

Per-block training metrics include `score_1/2`, `kl_1/2`, `var_1/2`, while
the v3 evaluation records *paired* posterior-child zeroed, shuffled and
noise-replaced decode MSE at fixed encoded roots. Hierarchical evaluation
also measures **encoded root + generated conditional child** alongside fully
generated FID, helping discriminate root-prior mismatch from child-prior error.
Check `results/<result_name>/dataframes/eval_metrics.csv`,
`loss_history.csv`, and per-evaluation `samples/.../metrics.json`.

Recommended commands:

```bash
python paper.py validate
python paper.py smoke --cell-id 5 --device cuda
python paper.py submit --cells all --dry-run
python paper.py submit --cells all
python paper.py report --cfg 2.5
```

Offline validation and real PyTorch CPU smoke tests of all five cells (forward,
CSEM/decoder gradients, root conditioning, sequential reverse solves) were run while packaging. **No new full CIFAR training or FID result is claimed.**
The GPU Slurm smoke runs again before each of the 800-epoch jobs.
