# CIFAR-10 CSEM: six 800-epoch cascade runs

This extends the original one/two/three-block CIFAR-10 sweep with exactly three
new runs. Every run uses RGB CIFAR-10, seed 42, 800 cotraining epochs and no
head-only refinement. Original cells 0–2 retain their scientific settings,
result directory names and status/config hashes.

| Cell | Model | Latents | Score depths | Band coefficient |
|---|---|---|---|---:|
| 0 | Original one-block CSEM | 8 x 8 x 8 | 12 | 0 |
| 1 | Original two-block conditional CSEM | 4 x 8 x 8 + 4 x 8 x 8 | 6 + 6 | 0 |
| 2 | Original three-block conditional CSEM | 3 x 8 x 8 + 3 x 8 x 8 + 2 x 8 x 8 | 4 + 4 + 4 | 0 |
| 3 | One-block CSEM + B | 8 x 8 x 8 | 12 | 0.05 |
| 4 | Two-block CSEM + B | 4 x 8 x 8 + 4 x 8 x 8 | 6 + 6 | 0.05 |
| 5 | Optimized two-block CSEM + B | root 8 x 4 x 4 + residual child 6 x 8 x 8 | 6 + 6 | 0.05 |

All arms have **512 stochastic latent coordinates** and twelve total DiT layers.
The optimized arm adds VAE posterior/top-down modules; it is a fixed latent
budget/depth comparison, not an exactly matched parameter-count experiment.
Actual VAE and score parameter counts are printed by smoke tests and recorded
for hierarchical training. Both optimized blocks have learned diffusion priors.

## Add the three new runs to an existing sweep

The full archive contains the `csem_cifar_blocks_800/` folder. The smaller
`csem_cifar_blocks_800_update.zip` contains only changed/new files at its root;
extract that archive **inside** your existing `csem_cifar_blocks_800` directory.
Then activate your existing `hlsi` environment:

```bash
python paper.py validate
python paper.py submit --cells new --dry-run
python paper.py submit --cells new
```

`new`, `band`, and `3-5` all select cells 3, 4, 5. `baseline` selects 0–2;
`all`/`main` select all six. Existing completed/active jobs are skipped using the
same status files. The three new result directories are distinct from the old
ones. Existing old checkpoints remain compatible; new band runs start fresh.

The complete archive can alternatively be extracted as a standalone sweep.
To reuse an existing prepared dataset cache:

```bash
python paper.py submit --cells new --data-root ../csem_paper_final_v6/data_cache_v6
```

The cache parent contains `CIFAR/`. Each selected training job validates/prepares
it under the shared lock. No separate preparation job is submitted. The original
Vista/Frontera launcher uses the `gh` partition, one node/task, and 48-hour limit.
It runs a real CUDA smoke check of its exact cell before training. The updated
Slurm job-ID parser accepts site banners and standard/parsable output.

Useful commands:

```bash
python paper.py smoke --cell-id 3 --device cuda
python paper.py smoke --cell-id 4 --device cuda
python paper.py smoke --cell-id 5 --device cuda
python paper.py run --cell-id 5 --resume
python paper.py report --cfg 2.5
python paper.py report --cfg 3.0
```

Reports separate curves by **cell ID**, including the different two-block runs.
They use measured results only and can read in-progress histories.

## Matched analytic lower-boundary penalty

For the actual representation schedule's first node, define

    floor = log(sigma_min^2 / alpha_min^2)
    B_prime = 0.5 * mean_coordinates softplus(floor - conditional_logvar)
    band_objective = active_encoder_csem_coefficient * B_prime

Cells 3–5 set `tmin_band_weight=matched`: the resolved coefficient is **0.05**,
matching representation CSEM. It is independent of the **0.60 terminal KL**
coefficient and the **1.0 detached score-head** coefficient. There is no
additional time-proposal factor: the canonical CSEM estimator already estimates
the physical-time integral. The original recipe has no coefficient warm-up.

The nonnegative B_prime differs from the conditional entropy change B by an
encoder-independent constant. It conservatively completes the missing
information charge below the cutoff; it does not restore omitted score-error
terms. Its mean over all 512 coordinates uses the same normalization as CSEM/KL.
Larger blocks therefore contribute in proportion to their coordinate counts.

The correction is deterministic given posterior log variances. It reaches the
encoder and ancestral posterior dependencies, with no score-head gradient.
Original score-gradient replacement, optimizers, clipping, EMA and checkpoints
remain in place. The original core, suite and paper_io files remain unchanged;
guarded AST adapters add the band term only to new cells and refuse to adapt a
trainer with an unexpected shape. Resume guards reject changed band, cutoff,
block-layout or architecture settings.

Loss histories add:

- `band`: coordinate-mean unweighted B_prime.
- `band_objective`, `band_weight`: its actual weighted contribution/coefficient.
- `band_i`: conditional band penalty for each block.
- `below_i`: fraction with log variance below the effective diffusion-noise floor.
- `deep_i`: fraction with standard deviation more than tenfold below that floor.

The effective standard-deviation floor is sigma_min/alpha_min and is recorded
in the resolved configuration. With OU t_min=3e-5 it is approximately 0.007746.

## Optimized two-block architecture (cell 5)

1. The original bottom-up image backbone extracts 8 x 8 image features. A new
   downsampling posterior head produces the root Gaussian at 4 x 4.
2. Sample the clean root once. A shared generative top-down path maps it from
   4 x 4 to 8 x 8. Child posterior parameters combine those ancestral features
   with the bottom-up image features; they remain conditionally diagonal Gaussian.
3. The child score head models an innovation u_2 conditional on the complete
   clean root. The generative path predicts location b(root) and positive scale
   d(root), and reconstructs the raw child as z_2 = b(root) + d(root) * u_2.
4. Inject root information at 4 x 4 and the child at 8 x 8. The inherited residual
   decoder then proceeds to 16 x 16 and 32 x 32. Image features enter posterior
   parameter computation; generation reconstructs all decoder features from latents.
5. Apply CSEM, terminal KL and B to the innovation coordinates. Generation solves
   the root diffusion first and the child conditional diffusion second, using
   fixed generated root context. Encoded-root/generated-child diagnostics use
   the same conditional model.

For compatibility with the pinned trainer/noise banks, an invertible pixel
permutation packs the 8-channel 4 x 4 root as 2 channels at 8 x 8. The root DiT
actually uses 4 x 4 (16 patch-1 tokens), while the child uses 8 x 8 (64 tokens).
This permutation preserves Gaussian targets, coordinate norms, KL and the
512-coordinate budget exactly. It is not upsampling the root stochastic state
or adding more latent variables.

Keep six DiT layers per head, the full root as side information, and the original
sampler grids in this first experiment. This reduces root network spatial work
without changing solver accuracy budgets. Hierarchical evaluation records
per-sampler generation-plus-decoding seconds and actual block spatial sizes.
The measurements include the same ancestral/root and decoder work used by FID.

Class conditioning/CFG, label dropout, loss strengths and training horizons are
held at the original settings to make the architectural comparison interpretable.
Coarse/detail roles are architectural biases, not a guaranteed semantic split.

## Shared recipe and evaluation

| Setting | Value |
|---|---|
| Dataset / seed / batch | CIFAR RGB / 42 / 256 |
| Representation / score-sampling horizon | T_K=1.05 / T=1.35 |
| Schedule | OU log_t, t_min=3e-5, 1000 discrete nodes |
| Mean anchor | OU-partial, anchor time 1.05; per factor in cascades |
| Encoder CSEM / terminal KL | 0.05 / 0.60 |
| Representation / head time weighting | canonical / unweighted-eps |
| VAE / head LR | 2.5e-4 / 1e-4 |
| LR schedule | 800-epoch cosine, minimum 1e-6 |
| Optimizer / clipping | Original AdamW and one combined norm-1 clip |
| Precision auxiliary / cosine | 0.0025 / 0 |
| LPIPS / GAN | 0.85 / 0.0025; discriminator starts at epoch 25 |
| DiT hidden width / attention heads | 384 / 6; factored head |
| EMA / label dropout | 0.9997 / 0.10 |
| Reconstruction | Original time-conditioned decoding of corrupted latents |
| Evaluation | Every 50 epochs, including epoch 800; 10,000 test images |
| Generation grids | CFG 2.5/3.0; RK4-20, RK4-25, Heun-20 per block |

Cascades cost sequential reverse solves. The same step count per block is not
matched total sampling cost to a flat model; use the timing columns alongside
FID. The original evaluation backend is legacy torchvision Inception, not an
assertion of equivalence to every external FID implementation.

## Validation

`paper.py validate` checks original source hashes and recipes, six-cell
selection, old status-hash compatibility, the exact band formula/slope, guarded
trainer adaptation, fixed-draw conditional coupling, dimension weighting,
sequential sampling, resume guards and endpoint completeness. `paper.py smoke`
uses real PyTorch model/autograd graphs and reverse sampling, including the
optimized spatial packing and band-only gradient routing. CUDA/800-epoch
training and numerical endpoint FID require the cluster run.

Before delivery, all three new cells passed full-size CPU autograd/sampling
smokes. Separate reduced-width synthetic-data trainer checks executed two real
optimizer steps per new cell and verified the band coefficient, coordinate
weighting, loss-history logging, checkpoint save and checkpoint reload. These
checks validate implementation; they do not measure CIFAR quality.
