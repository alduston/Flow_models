# FMNIST: matched flat, two-block, and three-block CSEM cotraining

This bundle tests whether stochastic posterior stacking improves generation
when the score model, latent dimension, decoder, training recipe, and VAE
parameter budget are controlled. It includes 20 jobs: four arms at five paired
seeds (42–46). All jobs use r=0.5, the setting that produced the historical 4.3.
No cluster jobs have been submitted or trained by creating this bundle.

## Run on Vista

Unzip and enter `fmnist_capacity_matched`. Activate the existing hlsi CUDA
environment used for the FMNIST audit. Keep its torch/torchvision versions;
`requirements.txt` lists supporting packages, not a request to upgrade Torch.

```bash
python prepare_cache.py --weights
python paper.py validate --models
python paper.py submit --cells all --dry-run
python paper.py submit --cells all
```

Submission creates one Slurm job per cell on `gh`, one node per job, with a
48-hour limit. As in the original Vista bundle, GPU visibility is supplied by
the partition. Pass `--account YOUR_ALLOCATION` if required for your allocation.
The job script defaults to `$SCRATCH/venvs/hlsi`; export `VENV_DIR` to reuse a
different environment. Export `CSEM_DATA_ROOT` to reuse an existing prepared
FMNIST cache; preparation honors the same variable. `TORCH_HOME` must point to
the same checkpoint cache in preparation and compute jobs if you override it.

```bash
python paper.py plan
python paper.py submit --cells confirmation      # only new seeds 44,45,46
python paper.py report                          # valid while jobs are running
python paper.py report --strict                 # require all 20 jobs complete
```

Repeat the submit command to resume interrupted jobs. Active and validated
completed jobs are skipped. Outputs and guarded checkpoints live under
`results/<arm>_s<seed>`. The job script forwards Slurm's early warning to Python,
which checkpoints after the current epoch. It returns 75 for an orderly partial
run; submitting again resumes rather than restarting. Sources/configuration
are hashed: editing a running experiment requires a separate output directory.
`--restart` on a direct `run` archives the old attempt; it never silently merges
recipes. Use an extracted copy separate from the earlier mechanism study.

## Exact historical anchor

Original cell **3**, `b2_full_r0.5_s43`, epoch **500**:

- Generation FID: **4.3040228295** at CFG 3; **5.3848377148** at CFG 1.
- Posterior reconstruction FID: approximately **2.7380**.
- Other seed (42), same arm: CFG-3 FID **4.5634474009**.
- Two-seed CFG-3 mean: **4.4337351152**.

These are historical measurements from the delivered consolidated report, not
results of this new bundle. `stack2_s43` reproduces the original architecture
and recipe. Exact numerical reproduction of historical FID still depends on
the original hardware/software and stochastic execution; a 4.304 result is not
guaranteed. The model initialization, sampled losses, gradients and one update
were checked against the original source locally (see `VALIDATION.md`).

`provenance/baseline_4p3.json` records the source identity/hash and historical
readouts; `provenance/original_manifest.json` preserves the original manifest.
`original_models.py`, `original_experiments.py`, and `pinned/` retain reference
source. The recovered source archive SHA256 is:

`988b10e4b1f924ec60e8582b82dcb389fb530a3f9b94dc5edb09ab09e3497ad8`

## What is held fixed

| Setting | Every arm |
|---|---|
| Training | 500 joint epochs, batch 128, no warmup, seeds 42–46 |
| Score model | One joint DiT, width 192, depth 18, 6 heads, patch size 1; ordinary epsilon output, **not factored** |
| Score parameters | 12,159,556; identical initial tensors at a common seed |
| Coordinates | 256, packed by an exact permutation into 4×8×8 |
| Forward process | OU; 1,000 log-time points from 2e-5 to T=2 |
| CSEM | Unweighted epsilon MSE; head coefficient 0.6; encoder multiplier r=0.5, effective coefficient 0.3 |
| Optimizer | AdamW, betas (0.9,0.95), weight decay 1e-4 |
| Learning rates | VAE 5e-4; actual head LR 2e-4 / 0.6 = 3.333333e-4; cosine to 1e-6 |
| Image objective | Noisy reconstruction MSE + LPIPS(VGG) 0.85 + GAN 0.0025, GAN starts epoch 25 |
| Other terms | Terminal coefficient 1; inherited automatic small-time band coefficient, fixed across r/arms |
| Clipping / EMA | Original combined norm-1 clipping; head EMA 0.9997; online VAE |
| Decoder | Same complete image-generating computation and same initial decoder tensors |
| Primary evaluation | Epoch 500; full-T Gaussian start; temperature 1; CFG 3; RK4-20; 10,000 test-set legacy Inception features; decode time 1e-4 |
| Secondary evaluation | CFG 1 at the same epochs; reconstruction FID; original smaller sampling checks |
| Evaluation schedule | FID every 50 epochs; diagnostics every 25; witnesses/sampling checks every 100 |

The band coefficient is resolved by the original function on the actual time
grid and recorded in each `config.json`; it is not replaced by a rounded number
from the report. The FID backend is the original torchvision Inception feature
protocol, not a new standard-FID implementation. The inherited optional
`torchmetrics` warning does not disable this bundle's NumPy/SciPy FID routine.

## Capacity and the four arms

| Arm | Stochastic blocks | VAE parameters | Score parameters | Total |
|---|---:|---:|---:|---:|
| `flat` | 1 | 6,368,847 | 12,159,556 | 18,528,403 |
| `stack2` | 2 | 6,368,847 | 12,159,556 | 18,528,403 |
| `flat_three_encoder` | 1 | 6,368,817 | 12,159,556 | 18,528,373 |
| `stack3` | 3 | 6,368,817 | 12,159,556 | 18,528,373 |

There are no padding/dummy parameters. Every VAE parameter participates in the
objective and receives a gradient in the routing check. The 30-parameter
cross-family difference is 0.00047% of the VAE budget; the three-block model has
the smaller budget. Decoder, common image encoder, and root-feature parameters
are exactly matched, not merely matched in total count. See
`parameter_audit.json` for counts and initialization hashes.

### Why the flat arm really is one Gaussian

For two-block storage, the original posterior is

`q(a,b|x) = q(a|x) q(b|x,a)`, with independent Gaussian innovations.

The flat control uses the **same network** but supplies the deterministic root
mean to the child conditioner:

`q_flat(a,b|x) = q(a|x) q(b|x,mu_a(x))`.

Every mean and variance is then a deterministic function of x alone. Therefore
the complete 256-coordinate vector is one diagonal Gaussian conditional on x.
There is no sampled-parent dependence and its CSEM teacher is valid for that
flat posterior. The root and child storage shapes are retained solely to keep
the original decoder and all its capacity identical. This is a stronger
architecture control than comparing against the old 8-layer-head flat model.

The flat model deliberately retains the multiscale deterministic decoder.
This experiment isolates **stochastic posterior hierarchy**, not the benefit
of a deeper/different image decoder. Code fields `blocks=2` for `flat` and
`blocks=3` for `flat_three_encoder` describe storage groups needed by inherited
probes. **Use `stochastic_blocks`, not `blocks`, to identify stochastic depth.**

### Three-block budget allocation

`stack3` uses 64 root + 64 second + 128 third coordinates. It retains the
original two-block root and the exact original decoder; concatenating the last
two codes gives the original three-channel decoder child input.

The original conditional-posterior network has 779,142 parameters. It is
replaced by two narrower conditioners (hidden widths 87 and 75), totaling
779,112. The third receives image features, root features, and the actual
sampled second code directly. Widths were selected solely by parameter-count
arithmetic, before training; no FID search determined them. Thus the three-block
arm is a controlled new architecture, **not** the old larger three-block VAE.

Splitting an encoder budget changes its allocation. `flat_three_encoder` uses
exactly this same three-stage encoder/decoder and tensors, with all sampled
parent inputs replaced by recursively computed deterministic posterior means.
It is also one diagonal Gaussian given x. This extra control prevents a gain
from the new encoder allocation being misidentified as a stochastic-depth gain.

## Prespecified comparisons and interpretation

Primary endpoint: **epoch 500, CFG 3, RK4-20, Gaussian T2, 10,000 samples**.
The two planned hierarchy contrasts are:

1. `stack2 - flat`: exact VAE architecture/parameter/decoder/score match.
2. `stack3 - flat_three_encoder`: exact architecture/parameter match within the
   three-stage encoder family.

Secondary: `stack3 - flat` and `stack3 - stack2`, which match the VAE budget to
30 parameters but also change its allocation. CFG 1 and reconstruction FID are
secondary. Never select a different CFG, epoch, solver or seed for each arm.
A lower reconstruction FID alone does not demonstrate better generation.

All five paired seeds are reported. Seeds 42/43 were already observed when
r=0.5 was selected; the compiler also reports **new seeds 44/45/46 separately**.
Means, SDs and Student-t intervals describe variation across training seeds;
they are not confidence intervals for an individual FID estimate. Intervals
are pointwise, unadjusted for multiplicity, with only three new confirmation
seeds. The compiler does not automatically declare statistical superiority.

A robust negative paired delta on new seeds supports stacking **at this fixed
recipe and budget**. A tie or reversal undermines the attribution of the old
4.3 gain to stochastic stacking. Neither outcome establishes universal ranking
across all flat parameterizations, pressure settings, or datasets. An endpoint
far from the historical stack2 result should trigger a reproduction check
before interpreting the architecture comparisons.

## Reading live results

`python paper.py report` reads current CSVs even if no job has completed. It
produces:

- `compiled/status.csv`: observed training epochs and validated completion.
- `compiled/latest.csv`: each run's latest primary CFG-3 measurement.
- `compiled/eval_history.csv`: full evaluation history, including distinct
  reconstruction values and smaller oracle/sampling checks.
- `compiled/paired_latest.csv`: most recent common epoch per seed/comparison/CFG.
- `compiled/paired_history.csv`: all strictly matched primary evaluation pairs.
- `compiled/endpoint_arms.csv` and `endpoint_pairs.csv`: fixed-epoch-500 means,
  variation, intervals, and available/expected seed counts.
- `compiled/report.md` and `summary.json`: readable and machine-readable summary.

**A is the first named arm, B is the second; delta = A minus B. Negative favors A.**
Different epochs, sample counts, feature backends, solvers, and starts are never
subtracted. Synthetic smoke FIDs never enter primary comparisons. An observed
endpoint measurement does not certify job completion. `report --strict` requires
validated completed jobs, not just a printed final epoch.

The original dependence, corrected-teacher, score-risk, gradient-pressure and
witness diagnostics remain available for interpretation, but are not used to
select recipes or redefine the primary generation comparison. Equal parameter
counts and epochs do not imply identical FLOPs or wall time; per-epoch training
seconds and runtime versions/device are recorded.
