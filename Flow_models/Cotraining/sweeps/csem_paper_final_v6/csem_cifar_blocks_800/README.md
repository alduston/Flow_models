# CIFAR-10 CSEM: one, two and three latent blocks

Exactly **three training jobs**, all on RGB CIFAR-10, seed 42, with **800
cotraining epochs**. No FMNIST, extra seed, independent-prior phase, refinement
phase or separate dataset-preparation job is submitted.

| Cell | Model | Latent channels at 8 x 8 | DiT depth per head | Epochs |
|---|---|---|---|---:|
| 0 | Original flat CIFAR CSEM | 8 | 12 | 800 |
| 1 | Two-block conditional CSEM | 4 + 4 | 6 + 6 | 800 |
| 2 | Three-block conditional CSEM | 3 + 3 + 2 | 4 + 4 + 4 | 800 |

Eight total latent channels and twelve total DiT layers are held fixed. The
VAE's image encoder/decoder backbone is the original CIFAR architecture in all
three arms. Hierarchical arms add small ancestor-to-posterior adapters and one
score context projection per descendant. Separate time/class embeddings and
head modules add parameter overhead, so matched total depth is not an exactly
matched parameter count. Actual counts are printed by the smoke test and saved
for hierarchical training in `paper_cell_config.json`.

## Run on Vista / Frontera

Extract the archive under your existing sweeps directory:

```bash
unzip csem_cifar_blocks_800.zip
cd csem_cifar_blocks_800
module purge
module load gcc/13.2.0
module load python3/3.11.8
source "$SCRATCH/venvs/hlsi/bin/activate"
python paper.py validate
python paper.py submit --dry-run
python paper.py submit
```

The dry run prints exactly three `sbatch` commands. Jobs use the supplied
`gh` partition, one node/task and 48-hour walltime from your attached launcher.
Each job first runs a real CUDA graph/sampling smoke test in a separate process,
then validates/prepares the shared CIFAR cache under the original dataset lock.
Only one job downloads/extracts if the data is missing. Existing complete splits
are reused; training never downloads or modifies a live cache.

To reuse the CIFAR cache of the existing V6 sweep, substitute:

```bash
python paper.py submit --data-root ../csem_paper_final_v6/data_cache_v6
```

Use the actual path of your already-prepared cache if that folder differs. The
cache parent must contain the `CIFAR/` subdirectory. Checkpoints, status files,
results and Slurm logs remain isolated in this new sweep directory. The old
sweep's results and checkpoints are not imported or modified.

Direct commands are also available:

```bash
python paper.py run --cell-id 0 --resume
python paper.py run --cell-id 1 --resume
python paper.py run --cell-id 2 --resume
python paper.py smoke --device cuda
python paper.py prepare-data
python paper.py report --cfg 2.5
python paper.py report --cfg 3.0
```

`report` can read live in-progress evaluation histories. It writes measured
curves/latest tables and PNG/PDF FID plots under `compiled/`; missing results
are left missing. These commands do not submit additional jobs.

## The one-block baseline is the attached CIFAR main cell 0

Cell 0 calls the original `suite.run_cell` and original
`core.train_vae_cotrained_cond`, with the same config resolver, model classes,
training objective, initialization, optimizer, scheduler and evaluator. With
an explicit cache path it calls the same trainer with only `data_root` changed.
This is the attached optimized 800-epoch recipe requested as the approximately
8.85-FID reference. That number is the supplied historical reference; the
attachments contain configuration/source, not a new numerical reproduction.

| Setting | Value in all three jobs |
|---|---|
| Dataset / seed / batch | CIFAR RGB / 42 / 256 |
| Representation horizon / score-sampling horizon | T_K = 1.05 / T = 1.35 |
| Diffusion schedule | OU, log_t; t_min = 3e-5; 1000 discrete training nodes |
| Mean-scale anchor | OU-partial, anchor time 1.05 |
| Encoder CSEM weight / terminal KL weight | 0.05 / 0.60 |
| Representation / inner-head time weights | canonical / unweighted-eps |
| VAE / head learning rates | 2.5e-4 / 1e-4 |
| LR schedule | Cosine over 800 epochs, minimum 1e-6 |
| Joint optimizer / clipping | AdamW, weight decay 1e-4, betas (0.9, 0.95); one joint norm-1 clip |
| Gradient routing | Original representation gradients; independent full-horizon head gradients replace head grads |
| Precision auxiliary / cosine loss | 0.0025 / 0 |
| LPIPS / GAN weights | 0.85 / 0.0025 |
| Discriminator | Original PatchGAN; starts at epoch 25; Adam LR 1e-4 |
| DiT width / attention heads / factored head | 384 / 6 / enabled |
| VAE base width / residual blocks | 64 / 2; original encoder/decoder attention and projection options |
| EMA / label dropout | 0.9997 / 0.10 |
| Reconstruction | Original time-conditioned decoding of OU-corrupted latents |
| Evaluation | Every 50 epochs and endpoint; 10,000 CIFAR test examples |
| Gaussian sampling grid | CFG 2.5 and 3.0; RK4-20, RK4-25, Heun-20; temperature 1 |
| Head-only refinement | 0 epochs |

The uploaded `core.py` had an import-time error: the default value of
`setup_run_results_dir(..., wipe=...)` referenced undefined `cfg`. This copy
changes that default to `True`. The training function already passes `wipe`
explicitly. This is the only change to the original engine file; all model,
training and evaluation code remains unchanged. Validation restores the old
default in memory and checks the complete source hash against the attachment.
`suite.py` and `paper_io.py` are unchanged byte for byte. `provenance.json` and
`baseline_manifest.csv` record the exact source and baseline row.

## What “hierarchical” means in this sweep

This extends the two-block FMNIST experiment, rather than using unmatched
Gaussian priors for descendant latents. At one fixed spatial scale:

    q(z1,...,zB | x) = q1(z1 | x) prod_{i=2}^B qi(zi | x,z<i).

Each factor is a diagonal Gaussian. The original eight-channel encoder emits
base means/logvariances, partitioned into the indicated blocks. For i > 1, a
small adapter of the realized clean ancestors changes that factor's mean and
logvariance. Descendant adapters are zero-initialized. OU-partial normalization
is applied to each final conditional mean **before** that block is drawn.
Global normalization after drawing ancestors would otherwise invalidate the
root's Gaussian sampling interpretation. Blockwise normalization is therefore
an explicit architectural difference in the hierarchical arms.

Generation is:

    learned root reverse diffusion -> learned conditional reverse diffusion
    for block 2 -> learned conditional reverse diffusion for block 3 -> decoder.

Every modeled block gets a CSEM-trained diffusion prior; no clean downstream
aggregate is replaced with an assumed Gaussian. There are no image-feature
skip connections into the decoder or score heads. Image information enters
reconstruction only through sampled modeled latents. Later posterior factors
may see x, because their conditional aggregate distribution is itself learned
by a conditional diffusion head.

For a fixed image and realized clean ancestors, the exact epsilon target is:

    C_i = sigma_t (y_i - alpha_t mu_i) /
          (alpha_t^2 exp(logvar_i) + sigma_t^2).

Head i sees only (y_i, t, class label, clean z<i). It never sees x or later
latents. Regression over images at fixed clean parents identifies the
conditional aggregate score. The CSEM representation loss differentiates
through parameters, target variance, noisy sample and clean-parent context.
The independent full-horizon inner regression detaches all these encoder
quantities, including the parent context, before replacing score-parameter
gradients. Auxiliary factored-head precision loss is preserved.

Concatenating block outputs before coordinate-mean MSE/KL reproduces the
original normalization: each block receives weight C_i/8. The CSEM/KL weights
are not multiplied by the number of blocks. Each conditional diffusion starts
from Gaussian noise at T=1.35, with the corresponding conditional terminal
anchor at T_K=1.05. As in the flat diffusion baseline, these finite-horizon
Gaussian starts are approximations; terminal KL does not prove exact endpoint
matching. Here that diffusion-truncation approximation is present for **each**
conditional head, not solely the root. It is not a Gaussian prior substitution
at a clean descendant layer.

This is a small fixed-resolution stochastic hierarchy, not a reproduction of
NVAE's multiscale hierarchy or LSGM's jointly diffused top-scale group set.
The experiment measures the effect of this conditional factorization at
matched total latent width and transformer depth. It does not claim to capture
all possible hierarchical-VAE improvements.

## Evaluation and compute interpretation

The flat cell keeps the original evaluator, including its existing T_K/full-T
and oracle-initialization diagnostics. Hierarchical cells evaluate the shared
full-T Gaussian grid with the original torchvision Inception feature extractor
and FID/KID/SW2 functions. Full generated images use the same fixed Gaussian
bank and uniformly sampled label bank as the flat evaluator. Hierarchical
posterior reconstructions use fixed posterior draws with correct parent-child
coupling. Joint Gaussian oracle initialization is not asserted for the hierarchy.

All arms report full generated FID and posterior reconstruction FID. The two
hierarchical arms additionally encode the root and generate **every descendant**
from its learned prior, reporting `fid_encoded_root_generated_children_cfg...`.
This distinguishes decoder improvements from reliance on posterior-only detail.
Per-block training `score_i`, `kl_i` and `var_i` are logged. The conditional
score-risk diagnostic has its own column and is not equated to a flat joint
score functional.

All plots use a single chosen CFG across all three arms, and identify RK4-20
as **20 reverse steps per block**. Hierarchy therefore has sequential sampling
cost: 20/40/60 total block steps, respectively, although the per-head DiT depths
are split 12 / 6+6 / 4+4+4. This is not equal sampling latency. Both evaluated
CFG values remain available; the report does not independently pick each
model's best CFG. Legacy Inception FID here preserves your existing metric
convention and should not be silently compared with a different FID backend.

## Resume, failures and checks

Full-state checkpoints are atomic and written every ten epochs and around
evaluations. They include online/EMA heads, VAE, discriminator, optimizers,
scheduler, RNG state and histories. Interrupted evaluation is retried before
the next training epoch. Hierarchy additionally rejects changed block splits,
head depths and hierarchy versions. Completed and active Slurm jobs are skipped.

Rerun `python paper.py submit` after a walltime interruption or failure to
resume unfinished cells. For an attempt without a usable checkpoint:

```bash
python paper.py submit --cells 1 --restart-incomplete
```

The previous attempt is archived under `failed_attempts/`. Completed cells are
not overwritten. `--cells 0`, `--cells 1`, `--cells 2`, and `--cells 0-2` select
individual cells; `all`, `main`, and `missing` select all unfinished cells.

Offline validation checks source/config equivalence, exactly three submissions,
preserved mathematical loss/optimizer/routing/clipping statements, posterior
ancestor coupling and fixed noise, learned-prior ancestral sampling, coordinate
weighting, resume compatibility and 800-epoch evaluation completeness. Python
files compile and the Slurm launcher passes shell syntax validation.

Torch and CUDA were unavailable in the artifact-creation environment, so real
model/autograd smoke tests, cluster execution and numerical FID reproduction
were not run there. Every cluster job runs the supplied CUDA smoke test before
training. You can also run it manually; it uses synthetic images only and does
not produce experimental FID results. The standard `hlsi` environment must
contain the same torch, torchvision, lpips, numpy, scipy, pandas, matplotlib
and tqdm dependencies used by your attached sweep.

Files: `paper.py`, `manifest.csv`, `job.slurm`, unchanged `suite.py` and
`paper_io.py`, minimally repaired `core.py`, `hier_models.py`, `hier_engine.py`,
`hier_eval.py`, `tests.py`, `baseline_manifest.csv`, `provenance.json`, this README.
