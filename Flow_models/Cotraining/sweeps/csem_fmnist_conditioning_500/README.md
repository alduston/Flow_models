# FMNIST CSEM conditioning comparison: five 500-epoch runs

This replaces the attached older residual-cascade sweep with the recovered
**direct-latent** two-block and three-block trainers used in the recent runs.
The two-block reference is the architecture associated with the 4.8983 FID
endpoint. The full-ancestry three-block reference is the recently run
64/64/128-coordinate model. Their original sources are included in `pinned/`.
All jobs train from scratch for 500 epochs, with seed 42 and 256 coordinates.

## Cluster launch

Extract the ZIP and enter `csem_fmnist_conditioning_500/`. Activate the existing
`hlsi` environment, then run:

```bash
python paper.py validate
python paper.py prepare-data --weights
python paper.py submit --cells all --dry-run
python paper.py submit --cells all
```

The submit command launches **five independent jobs** on partition `gh`, one
node/task, with a 48-hour walltime. `job.slurm` retains gcc/13.2.0,
python3/3.11.8 and `$SCRATCH/venvs/hlsi`. Set `VENV_DIR` to use another existing
environment, or supply `--partition`, `--time`, and `--account` to `submit`.
The job runs an offline CUDA smoke in a separate process before training.

The default data cache is **`data_cache_fmnist/`**, containing
`FMNIST/FashionMNIST/raw/`. A prepared cache from another bundle can be reused:

```bash
python paper.py prepare-data --data-root ../other_sweep/data_cache_fmnist --weights
python paper.py submit --cells all --data-root ../other_sweep/data_cache_fmnist
```

For an older `data/FashionMNIST/raw/` cache, copy `FashionMNIST/` into
`data_cache_fmnist/FMNIST/`. Preparation validates the processed data under a
shared lock. `--weights` caches LPIPS/VGG and Inception on the login node.
All trainer/reference/launcher/compiler sources are included; datasets and
pretrained weights are prepared separately.

## The five experiments

Every prior factor also conditions on class label C. Label dropout is unchanged.

| Cell | Name | Prior factorization | Coordinates | Score depth |
|---:|---|---|---|---|
| 0 | `one_block` | p(z1\|C) | 256 | 8 |
| 1 | `two_block` | p(z1\|C) p(z2\|z1,C) | 64:192 | 8:8 |
| 2 | `three_full` | p(z1\|C) p(z2\|z1,C) p(z3\|z1,z2,C) | 64:64:128 | 8:8:8 |
| 3 | `three_markov` | p(z1\|C) p(z2\|z1,C) p(z3\|z2,C) | 64:64:128 | 8:8:8 |
| 4 | `three_siblings` | p(z1\|C) p(z2\|z1,C) p(z3\|z1,C) | 64:64:128 | 8:8:8 |

One block uses the original four-channel 8x8 FMNIST VAE and one diffusion head.
The two-block model has a four-channel 4x4 root and three-channel 8x8 direct
child. Three blocks have a four-channel 4x4 root, one-channel 8x8 middle code,
and two-channel 8x8 final code. The decoder consumes those modeled codes
directly; the older residual-to-affine-child transformation is absent.

With two ordered blocks, full ancestry, a chain and root conditioning are all
the same edge z1 -> z2. Independent blocks would be a separate non-cascade
ablation. The three requested connected three-block graphs are included;
this is not an exhaustive enumeration of every disconnected graph or ordering.

**What changes across the three-block arms:** only the ancestors admitted to
the final diffusion score prior, at both training and generation. The posterior
remains q(z1|x), q(z2|x,z1), q(z3|x,z1,z2), and the decoder is identical. This
isolates the generative-conditioning choice while allowing joint CSEM training
to reshape the representation. The Markov and sibling score heads marginalize
the omitted ancestor. Conditional Gaussian component targets remain valid:
averaging them given the retained inputs yields the appropriate marginal score.
Omitting an ancestor from the score head does not eliminate every gradient path
through the full-ancestry inference network or analytic target.

## Preserved scientific recipe

| Setting | Value |
|---|---|
| Joint training / subsequent refinement | 500 epochs / none |
| Batch / workers | 128 / 2 |
| VAE preset | original `fmnist_reference`, base width 32 |
| DiT width / heads / depth | 192 / 6 / 8 per stochastic block |
| VAE LR / head LR setting | 5e-4 / 2e-4 |
| Actual score optimizer LR | 2e-4 / 0.6 |
| Scheduler | 500-epoch cosine, minimum LR 1e-6 |
| CSEM coefficient: heads / representation | 0.6 / 0.6 |
| Terminal KL / lower-boundary B | 1.0 / original `auto` formula |
| LPIPS / GAN / discriminator start | 0.85 / 0.0025 / epoch 25 |
| EMA / class-label dropout | 0.9997 / 0.10 |
| OU time grid | 1000 log-time nodes, 2e-5 through 2.0 |
| Reconstruction | all latent blocks corrupted at the same sampled time |
| Decoder root features | recomputed from corrupted root |
| Posterior inference / conditional score inputs | clean sampled ancestors |
| Primary evaluation | every 50 epochs, including epoch 500 |
| Primary FID | 10000 images, legacy Inception, CFG 3, RK4-20 per generated block |
| Decode time | 1e-4 |

CSEM, terminal KL and B average over latent coordinates, giving block weights
1; 0.25/0.75; and 0.25/0.25/0.5 respectively. `auto` B uses the original
epsilon-loss quadrature formula, about 0.10354 at representation coefficient 0.6.
It does not use the terminal KL coefficient as its weight.

The recovered standalone defaults specified 700 epochs; the requested/reference
experiment explicitly supplies 500, including the cosine horizon. Original
non-default historical CLI overrides cannot be inferred from source alone;
this sweep preserves the recovered default scientific recipe, with 500 epochs.
The validation compares launchable computations directly to those pinned sources.

Operational additions are prepared-data IO, strict resume guards, source hashes,
CPU-safe restoration of RNG bytes, atomic checkpoints every 10 epochs, and
committing evaluation epochs only after all diagnostics finish. These additions
do not change the fresh-run loss, optimizer or RNG trajectory. Diagnostics
restore training RNG, modes and parameter gradients.

## Added measurements at every evaluation

### All encoded/generated FID combinations

`eval.csv` retains the original primary/staged metrics. The new
`eval_fid_combinations.csv` contains **2, 4 or 8 combinations** for one, two or
three blocks. Each uses **10000 images** and the primary Inception/CFG/RK4 recipe.
`E` denotes a fixed joint posterior sample from the same real image; `G` denotes
a reverse-generated code using the parent codes actually present in that hybrid.
Each block uses the same Gaussian-start bank across hybrids.

| Three-block mask | Meaning |
|---|---|
| EEE | all encoded: posterior reconstruction |
| EEG | encoded z1,z2; generated z3 |
| EGE | encoded z1,z3; generated z2 |
| EGG | encoded root; generated descendants |
| GEE | generated root; encoded z2,z3 |
| GEG | generated root,z3; encoded middle code |
| GGE | generated root,middle; encoded final code |
| GGG | full ancestral generation |

EEE, EEG, EGG and GGG form the usual encoded-prefix substitution ladder.
The remaining hybrids are labeled `conditioning_stress`: an encoded descendant
need not be consistent with its generated parent. They are useful interventions,
not the ordinary ancestral distribution. FID differences are not additive error
contributions. Hybrid grids and the first 256 image arrays are also saved.

### Bayes/excess CSEM risk estimates

`eval_csem_risk_queries.csv` logs the individual probes;
`eval_csem_risk.csv` reports means and standard errors by block, head source,
reference-bank size and time bin, plus a time-averaged row (`time_bin=-1`).

Default: 128 held-out queries, 8 stratified log-grid-index bins, and nested
reference banks of **64 and 256 training images per class**. Both online and EMA
heads are evaluated at CFG 1 against the same probes. All risks are in epsilon
units, per coordinate; the online/EMA distinction is recorded explicitly.

Let U be the analytic component denoising target, f the learned prediction,
and m=E[U|Y,retained ancestors,C] the population conditional Bayes predictor.
Then the true decomposition is

```text
E||f-U||² = E||m-U||² + E||f-m||².
```

The diagnostic computes a finite-reference approximation m_hat using an explicit
conditional Gaussian mixture. Its fields are:

- `raw_csem_risk`: mean ||f-U||² on held-out probes.
- `bayes_risk_proxy`: mean ||m_hat-U||², an estimated irreducible cost.
- `excess_risk_proxy`: mean ||f-m_hat||², estimated score approximation error.
- `orthogonality_residual`: mean 2(f-m_hat)·(m_hat-U).
- `decomposition_error`: numerical residual of the exact three-term sample identity.
- `cross_fitted_excess_proxy`: mean (f-m_A)·(f-m_B) using split banks.
- `split_oracle_disagreement`: mean ||m_A-m_B||².
- `ess`, `ancestor_ess`, `max_responsibility`, `low_ess`: conditioning coverage.

Observed ancestors receive the correct posterior likelihood weights. For the
full final head these are q1(z1|x)q2(z2|x,z1). For Markov, the hidden root is
sampled from q1(.|x), then weighted by q2(z2|x,z1). For siblings, the observed
root is weighted by q1(z1|x), and hidden z2 is sampled from q2(.|x,z1). Both
omitted-ancestor integrations use Monte Carlo; these are not exact Bayes risks.

Interpret a Bayes/excess split together with bank-size stability, split-bank
agreement and ESS. Low ESS can make the oracle unreliable in sharp/high-dimensional
latent regimes. A nonzero cross term signals finite-reference/sampling error;
the split-bank estimate can be signed. These fields expose that uncertainty
rather than attributing every raw-loss decrease to better learned scores.

### Gradient interference and block-specific guidance

- `eval_gradient_checks.csv` retains weighted objective gradients before clipping,
  per-posterior norms, reconstruction/CSEM cosines and predicted clipping factors.
  Three-block arms add `cosine_csem2_csem3_z2_posterior` and
  `cosine_csem2_csem3_vae`. A negative cosine directly detects conflicting gradients;
  it does not by itself measure the eventual Adam update or establish causality.
- `eval_block_cfg.csv` varies one block's CFG at a time over **1, 3 and 5**, holding
  the others at 3. It uses 1024 paired images, fixed encoded ancestors and starts,
  and reports full-generation, encoded-root and encoded-z1,z2 FIDs as applicable,
  plus MSE to the default. These subset FIDs are compared within this CSV.
- `eval_sampling_checks.csv` keeps uniform CFG 1/3 and RK4-20/40 paired checks.
- `eval_child_usage.csv` keeps zero/shuffle interventions and repeated conditional
  draws with encoded root or encoded z1,z2, including MSE and LPIPS when available.
- `loss.csv` keeps individual score losses, KLs, posterior variances and B statistics.
- `csem_target_samples.npz` records exact supervision arrays at the endpoint.

The additional all-hybrid FIDs and guidance probes increase evaluation time.
The training update budget is the same in every arm.

## Resume and compile

```bash
python paper.py submit --cells missing
python paper.py run --cell-id 3 --resume
python paper.py report
python paper.py report --strict
```

`submit --cells baseline` selects 0–2; `--cells variants` selects 3–4;
`--cells three` selects 2–4. Explicit IDs and ranges are supported.
Checkpoints resume only with matching source/configuration. `--restart` for
`run`, or `--restart-incomplete` for `submit`, archives the existing incomplete
attempt before fresh training. Standalone older checkpoints are not imported.

The compiler produces `report/summary.csv`, all merged native/diagnostic CSVs,
FID contrasts against the two-block and three-full controls, `report.md`, and
`comparison.pdf`. Strict completion checks require every evaluation epoch,
all hybrids and the risk/guidance tables.

This is a single-seed comparison with the original per-head capacity. Deeper
models have more score parameters and more reverse solves. It tests the requested
configurations; it does not independently identify hierarchy depth from capacity.

## Validation

See `VALIDATION.md` and `smoke_validation.json`. All five full-size models pass
CPU forward/backward, sampling, topology routing, hybrid, risk and gradient checks.
The two reference arms match the pinned model/target/loss/gradient/sampling
computations bit-for-bit. Synthetic interrupted/resumed runs test checkpoint
and log recovery. Pretrained Inception FID and 500-epoch training are left to
the cluster run; the local evaluation smoke uses a feature surrogate.

`python paper.py validate --models` repeats the CPU model/diagnostic checks.
