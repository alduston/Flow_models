# CIFAR-10 direct-child CSEM cascade: eight fresh 800-epoch runs

This carries the new **direct stochastic child** architecture from the supplied
note (the architecture reported at FMNIST FID 4.8983) into the recorded optimized
RGB CIFAR recipe. The CIFAR baseline is now **two blocks**, with the same child
variable Z2 used by the posterior, CSEM score head, terminal KL, B_t and decoder.
No root-dependent child location/scale or residual-innovation map exists.

All runs train from scratch for **800 cotraining epochs**, with no score-only
refinement. Each job runs one cell on one GH node/GPU, plus a separate-process
CUDA smoke and periodic evaluations. Do not extract this over an old sweep:
the archive has a new directory and unique result names/checkpoint identities.

## Launch

Extract, enter `csem_cifar_direct_cascade_800/`, activate your existing `hlsi`
environment, then:

```bash
python paper.py validate
python paper.py prepare-data
python paper.py submit --cells all --dry-run
python paper.py submit --cells all
```

The launcher retains the Vista/Frontera `gh` partition, one node/task,
gcc/13.2.0, python3/3.11.8 and `$SCRATCH/venvs/hlsi`. Set `VENV_DIR` to use
another existing environment. The default time request is **24 hours**, reduced
from the old CIFAR bundle's 48 hours. The prior 12-hour recommendation applied
to the smaller 500-epoch FMNIST sweep; CIFAR uses larger networks, 800 epochs,
and the original full evaluation suite. This time limit is a configurable cap,
not a measured runtime prediction:

```bash
python paper.py submit --cells all --time 12:00:00
python paper.py submit --cells all --partition gh --time 24:00:00
```

Training checkpoints every ten epochs and around each evaluation. Timed-out
incomplete jobs can be resubmitted with the same command; they resume from their
full-state checkpoint. Active/completed jobs are skipped. No preparation job
dependency is used, avoiding a stalled dependency when a prepare job fails.

## Eight cells centered on the direct cascade

| Cell | Architecture | Seed | Root / child coordinates | Actual stochastic shapes | Score trunk depths |
|---:|---|---:|---|---|---|
| 0 | **Direct-child baseline** | 42 | **128 / 384** | **8×4×4; 6×8×8** | **6 + 6** |
| 1 | Direct-child baseline repeat | 43 | 128 / 384 | 8×4×4; 6×8×8 | 6 + 6 |
| 2 | Flat diagonal-Gaussian control + B | 42 | 512 | 8×8×8 | 12 |
| 3 | Flat control repeat | 43 | 512 | 8×8×8 | 12 |
| 4 | Balanced direct cascade | 42 | 256 / 256 | 16×4×4; 4×8×8 | 6 + 6 |
| 5 | Fine-heavy direct cascade | 42 | 64 / 448 | 4×4×4; 7×8×8 | 6 + 6 |
| 6 | Same-resolution direct cascade | 42 | 128 / 384 | 2×8×8; 6×8×8 | 6 + 6 |
| 7 | Child-deeper direct cascade | 42 | 128 / 384 | 8×4×4; 6×8×8 | 4 + 8 |

Every cascade retains direct Z2, top-down conditional Gaussian inference,
full-root context at every child DiT block, noisy-root reconstruction, and two
learned sequential diffusion priors. There are no three-block, no-B, clean-root
reconstruction, or older affine-innovation arms. Cells 0/2 and 1/3 provide
paired seed comparisons; cells 4–7 alter one structural axis relative to cell 0.
They test root bottleneck allocation, spatial organization, and allocation of
the original twelve score layers. The two-seed comparison is a repeatability
check; the four exploratory variants are single-seed runs.

All arms have **512 modeled coordinates and twelve total score trunk layers**.
The baseline uses 6+6, as specified by the original CIFAR cascade budget, rather
than copying FMNIST's 8+8 depths. Cross-attention and multiscale VAE modules still
change parameter count and FLOPs. Actual parameter counts are recorded; matching
coordinate count/trunk depth is not exact parameter or latency matching.

## CIFAR recipe retained from the original bundle

`core.py` and `suite.py` are byte-for-byte the recovered CIFAR sources.
Validation compares every existing resolved nonarchitectural setting to the
original CIFAR optimized cell. The new wrappers change architecture, logging
and orchestration; they preserve time sampling, losses, auxiliary precision,
gradient replacement, optimizers, EMA and checkpoint mechanics.

| Setting | Value |
|---|---|
| Dataset / input | CIFAR-10 RGB, 32×32 |
| Batch | 256 |
| VAE backbone | base_ch 64; original residual/attention capacity |
| DiT width / heads | 384 / 6; factored natural-parameter prediction |
| Representation horizon T_K / full score horizon T | **1.05 / 1.35** |
| OU grid | log_t, 1000 nodes, t_min=3e−5 |
| Mean anchor | **OU-partial at 1.05**, per conditional factor |
| Encoder CSEM / terminal KL | **0.05 / 0.60** |
| Representation / head time weighting | **canonical / unweighted-epsilon** |
| Detached score-head coefficient | **1.0** |
| B coefficient | **0.05**, matched to representation CSEM |
| VAE / score-head LR | **2.5e−4 / 1e−4** |
| Scheduler | **800-epoch cosine**, minimum LR 1e−6 |
| Auxiliary precision / cosine loss | 0.0025 / 0 |
| LPIPS / GAN | 0.85 / 0.0025; GAN begins epoch 25 |
| Label dropout / EMA | 0.10 / 0.9997 |
| Optimizer / clip | Original AdamW; joint norm-1 cotrain clipping |
| Refinement | None |
| Evaluation | Every 50 epochs, through 800; 10000 test images |
| Full evaluation suite | CFG 2.5 and 3; RK4-20, RK4-25, Heun-20 per block |
| Primary comparison | Full-T Gaussian start, temperature 1, RK4-20, CFG 2.5 |
| Feature backend | Original legacy torchvision Inception |

These are the recorded CIFAR settings, not the FMNIST loss coefficients or
schedule. In particular, FMNIST's local continuation factor (~0.1034) belongs
to its unit-epsilon/log-time representation loss. CIFAR's outer representation
loss is canonical, so its B continuation coefficient remains **0.05**. For each
conditional Gaussian:

```text
floor = log(sigma_min² / alpha_min²)
B = 0.5 * mean_coordinates softplus(floor - conditional_logvar)
weighted_B = active_encoder_CSEM_coefficient * B
```

B is the note's nonnegative upper bound on the information lost below t_min,
not an exact omitted-information formula. Coordinate means match the existing
CSEM/KL normalization: the baseline weights root/child contributions 0.25/0.75;
the balanced arm uses 0.5/0.5 and the fine-heavy arm 0.125/0.875. B reaches
conditional posterior and ancestral parameters, with no score-head gradient.

## Architecture and gradient routes

1. The original CIFAR bottom-up encoder produces 8×8 image features. A
   residual/attention stage predicts the root's Gaussian parameters at 4×4
   (8×8 in cell 6). Apply the inherited OU-partial mean anchor and draw the root.
2. Normalized residual/attention blocks and nearest-neighbor upsampling construct
   deterministic top-down features from that **same clean root sample**.
3. Combine image and top-down features to predict direct conditional Z2 moments.
   Apply the same per-factor mean anchor and draw Z2. The child score head sees
   only its corrupted Z2, time, class label and the full clean root; it does not
   see image features or posterior moments. Moments enter its analytic target.
4. Reconstruction consumes **both OU-corrupted codes** at the representation
   time. It recomputes all top-down decoder features from the corrupted root and
   injects Z2 directly before the inherited 8→16→32 decoder. No clean feature
   cache or image-feature skip enters reconstruction/generation.
5. Sampling completes the root diffusion first, then the conditional child
   diffusion with the completed root fixed. Decode those same generated codes.

The root is stored as packed channels at 8×8 for the original CIFAR tensor/bank
interface: an invertible pixel permutation maps an 8×4×4 root to 2×8×8. This
preserves coordinate norms, diagonal Gaussian targets, KL and B. Root DiT and
decoder genuinely operate at 4×4; packing is not an extra stochastic layer or
interpolated root state. Cell 6 uses the identity packing at 8×8.

The separate CIFAR optimization routes are preserved:

- **Representation:** canonical CSEM on [t_min,T_K], differentiated through
  noisy inputs, conditional target precision, sampled-root dependency and clean
  root score context. Shared posterior/top-down parameters remain differentiable.
- **Head:** independent full-horizon [t_min,T] epsilon regression with detached
  latent/moment/context tensors. Its parameter gradients replace the outer
  head gradients before the original joint clipping step. The auxiliary factored
  precision branch is retained. This detached regression route does not remove
  the encoder's child-CSEM/root gradient on the representation route.

The reference 4.8983 result is **FMNIST evidence for the architectural choice**,
not a claimed CIFAR measurement. The archived note and actual direct-FMNIST
source are included under `pinned/` for provenance. A parity check maps the
note's model modules onto the CIFAR port and verifies identical direct decoding
and raw child-posterior output under common weights, apart from CIFAR's explicit
per-factor OU-partial encoder anchor.

## Cache, status and checkpoints

The default cache retains the original name **`data_cache_v6`**, containing
`CIFAR/cifar-10-batches-py/`. Reuse a prepared cache by passing its parent:

```bash
python paper.py prepare-data --data-root ../csem_paper_final_v6/data_cache_v6
python paper.py submit --cells all --data-root ../csem_paper_final_v6/data_cache_v6
```

Dataset preparation is staged under a shared lock. Make sure your `hlsi`
environment has LPIPS and the inherited Inception/VGG weight cache available;
missing pretrained weights require network access. Full training refuses to
quietly skip the inherited perceptual objective.

Scientific source hashes and resolved recipe hashes travel into status and
full-state checkpoints. Resume rejects architecture/coefficient/layout/source
changes. The original atomic checkpoints retain optimizer, cosine scheduler,
EMA, discriminator and RNG state, plus histories. A checkpoint written before
an evaluation carries `pending_eval`; resuming it completes that evaluation
before more training. Unique result directories keep older affine-cascade
checkpoints separate. `--restart-incomplete` archives failed output; completed
cells are retained.

Useful commands:

```bash
python paper.py submit --cells baseline       # only the two new baseline seeds
python paper.py submit --cells paired         # baseline + flat controls
python paper.py submit --cells variants       # cells 4–7
python paper.py smoke --cell-id 0 --device cuda
python paper.py run --cell-id 0 --resume
python paper.py submit --cells all --restart-incomplete --dry-run
```

## Compile results

```bash
python compile_results.py
python compile_results.py --strict
# Equivalent:
python paper.py report --strict
```

The adapted cross-cell compiler creates `compiled/SUMMARY.md`, a leaderboard,
all-loss/all-evaluation tables, full-T Gaussian metrics, FID curves, paired-seed
contrasts, architecture contrasts, six PNG figures and `eight_cell_review.pdf`.
It reads both in-progress and finished histories. Missing epoch-800 endpoints
stay missing; partial latest values are separate. Strict mode requires every
scheduled evaluation and finite final sampling/conditional diagnostics.

Evaluation retains posterior reconstruction FID, encoded-root/generated-child
FID, root/child CSEM/KL/variance histories, child zero/shuffle/noise interventions,
and two generated child draws at fixed roots. CUDA-synchronized generation+
decoding time is recorded separately from feature extraction and diagnostics.
Terminal KL shares and raw CSEM losses are not estimates of blockwise mutual
information. Full generation versus fixed-root FID is a useful diagnostic and
does not give a unique causal decomposition.

Baseline two-block sampling has two serial reverse solves. Twenty steps per
block is a shared per-factor accuracy setting, not matched total inference
cost to one flat solve. Use measured timing alongside FID and consult the
retained solver/CFG comparisons. This bundle keeps the original CIFAR sampler
suite and score normalization rather than substituting a new protocol.

See `VALIDATION.md` for performed checks. Full CIFAR training, CUDA execution
and real 10000-image FIDs are to be measured on the cluster.
