# Naive stacked CSEM: cluster mechanism study

This bundle investigates **why one joint diffusion head trained on naive stacked
Gaussian component targets works well**, despite its generally biased ancestor
targets. It is not a best-FID search. The study follows the observed first
225 epochs: parent-dependent child means grow early, decline modestly later,
remain useful for reconstruction, and face opposing score/image gradients.

It adapts the supplied bundle's `gh` Slurm partition, GCC/Python modules,
`$SCRATCH/venvs/hlsi` environment, prepared FashionMNIST cache, per-cell jobs,
and compile/report workflow. The old bundle's cascade priors are **not** used:
every arm here has one joint head and one joint reverse solve.

## Start on the cluster

Extract in a shared work directory that survives compute-node exits. From
inside `stacked_csem_mechanism`:

```bash
module load gcc/13.2.0 python3/3.11.8
source "$SCRATCH/venvs/hlsi/bin/activate"
python -m pip install -r requirements.txt

# Run on a login/preparation node with internet access. Training never downloads data.
python paper.py prepare-data --weights
python paper.py validate
python paper.py submit --cells pilot --dry-run
python paper.py submit --cells pilot
```

The pilot is six 500-epoch, seed-42 runs: two-block pressure 0/1/4,
two-block decoupled control, and three-block full pressure 1/4. They have the
full research diagnostics, not a reduced training recipe.

```bash
# Expand to all paired main contrasts (22 joint-training jobs total).
# Active and completed pilot cells are skipped.
python paper.py submit --cells core

# Run matched frozen-encoder teacher comparisons after their source jobs finish.
# Sources that are still queued/running get afterok dependencies.
python paper.py submit --cells frozen --dependencies

# Optional boundary-weight and clipping-mediation controls (8 joint jobs).
python paper.py submit --cells robustness

# Or submit the complete predefined 54-job study, including dependencies.
python paper.py submit --cells all --dependencies
```

`--cells` also accepts IDs/ranges, e.g. `--cells 4,8,12,14`.
See `python paper.py plan`, `manifest.csv`, and `RESEARCH_PLAN.md`.
`experiments.py` is the executable source of truth; the JSON/CSV manifests are
exported views. Make study changes there **before** starting jobs, then regenerate
with `python -c "import experiments; experiments.export('.')"`.
All main/robustness runs train for **500 epochs**; the 24 frozen-head
continuations train for **50 additional head-only epochs**.
No jobs were submitted from this environment.

## Environment and cache overrides

```bash
export VENV_DIR="$SCRATCH/venvs/hlsi"
export CSEM_DATA_ROOT="$SCRATCH/data_cache_fmnist"
export TORCH_HOME="$SCRATCH/torch_cache"
python paper.py prepare-data --weights --data-root "$CSEM_DATA_ROOT"
python paper.py submit --cells core --data-root "$CSEM_DATA_ROOT" --account YOUR_ALLOCATION
```

Keep these paths consistent when resubmitting. The default data cache is
`./data_cache_fmnist`; its layout matches the old bundle's
`data_cache_fmnist/FMNIST/FashionMNIST/...`.
An intact old cache can be reused by pointing `--data-root` at its **parent cache**
directory. Preparation checks both splits under a file lock and only stages a
download if validation fails. Training opens validated files without downloading.
Pretrained Inception and LPIPS/VGG weights must also be available on compute nodes
through the same `TORCH_HOME` or existing Torch cache.

`job.slurm` preserves the supplied default `gh`, one-node, one-task layout and
48-hour limit. Change via `submit --partition ... --time ... --account ...`.
The current implementation uses one visible GPU. Check your allocation/site
policy for whole-node charging before launching many jobs. Set `SKIP_MODULES=1`
if your shell already supplies the correct modules; `VENV_DIR` still selects
the Python environment. `requirements.txt` intentionally does not replace the
cluster's CUDA Torch/torchvision pair.

## Scientific comparisons

| Group | Joint recipes per seed | Seeds | Question |
|---|---|---|---|
| Encoder pressure | Two-block full, multiplier 0, 0.5, 1, 2, 4 | 42, 43 | Does CSEM reshape dependence and bias? |
| Two-block zero-bias control | Per-image decoupled Gaussian posterior | 42, 43 | Is useful coupling necessary for quality? |
| Depth and dependency | Three-block full (1 and 4), Markov (1), siblings (1), decoupled (1) | 42, 43 | Which ancestor targets become biased? |
| Boundary control | Two/three-block full, multiplier 4 with co-scaled B | 42, 43 | Is the effect actually caused by the small-time penalty? |
| Clipping control | Two-block full, multipliers 1 and 4 with separate clipping | 42, 43 | Does global clipping mediate the pressure effect? |
| Frozen teacher | Naive / analytic corrected / DSM on 2/3-block full sources at pressure 1/4 | 42, 43 | Is bias consequential once the representation is fixed? |

The encoder gradient multiplier changes **only the VAE derivative of the score
term**. Head coefficient remains 0.6; head optimizer LR remains `2e-4/0.6`.
Terminal coefficient is 1. The primary pressure comparisons keep the automatic
lower-boundary penalty coefficient fixed at the baseline value. Its co-scaling
is tested separately. All global/separate clipping factors and unclipped
head/VAE norms are recorded.

Both depths use **256 coordinates**, packed into **4x8x8** by an exact coordinate
permutation: 64+192 versus 64+64+128. Both have the same joint DiT, width 192,
depth 18, six heads, and identical initial head tensors at the same seed.
The three-block VAE has extra posterior/decoder modules. Therefore a raw
two-versus-three comparison includes a VAE-capacity/allocation change; the
within-depth topology controls and pressure contrasts are the cleaner tests.

`gaussian_sandbox.py` also solves an independent linear-Gaussian hierarchy:
exact naive/valid regression fields, bias/curl, and probability-flow covariance
evolution from Gaussian versus oracle-q_T starts. This tests consequences of
biased supervision with network fitting and decoder effects removed. Its
outputs are labeled synthetic calculations, separate from FMNIST results.

Posterior topology is varied, not score-head visibility. A Markov third
posterior is `q(C|X,B)`, siblings is `q(C|X,A)`, and the decoupled control is
`q(A|X)q(B|X)q(C|X)`. Deterministic image-dependent parent anchors remain
available, so decoupling removes stochastic-parent dependence without discarding
image evidence. Decoder architecture is unchanged within each depth.

## What is measured

Score, correction and curl quantities use epsilon units (`epsilon=-sigma*score`).
Squared quantities convert to score units by dividing by sigma squared. Fixed
time bins are retained separately; a small curl alone does not establish an
accurate score.

**Every 25 epochs, plus initialization and endpoint** (`diagnostics/`):

- Per-image Gaussian symmetric KL over independent ancestor draws, mean-parent
  SNR, log-variance dependence, parent/child covariance, child off-diagonal
  covariance, and per-edge whitened local Jacobian norms.
- Posterior variance and mean energies; clean decoder same-image swaps and zero
  interventions. Shared-image aggregate correlation is kept separate from
  per-image dependence.
- Local edge-gate derivatives of score, reconstruction MSE, LPIPS, terminal,
  boundary B, GAN and weighted total; actual encoder score/image gradient cosine.
  These gate probes use one fixed small batch, stratified across the training
  log-time grid, and have no replicated SE.
- A chain-rule decomposition of score pressure into head-input, teacher-observation,
  teacher-mean and teacher-variance routes, with a numerical sum residual.
  This tests which route drives the sign instead of presuming every score
  pressure is an irreducible-variance incentive.
- Valid analytic corrected and raw DSM teachers; naive versus valid target
  risk, common valid excess risk, online/EMA paired differences, correction
  energy, descendant correction contributions, global curl and each cross-block
  curl pair, by block, time and
  class/null conditioning. Both mean and variance derivatives are retained.
- Time-resolved correction summaries now appear in stdout as well as files.

**Every 100 epochs and endpoint** (`witness/`): a small ridge/random-feature
predictor learns the naive-minus-corrected teacher defect on independent
training-image draws. Untouched test-image draws estimate

\[
2E[\delta\cdot w(Y,c)]-E\|w(Y,c)\|^2
\;\leq\; E\|E[\delta\mid Y,c]\|^2
\;\leq\;E\|\delta\|^2.
\]

This provides a **variational lower bound on squared bias without a density
bank**. The right bound is pathwise defect energy. The learned head's residual
is also projected along this learned defect direction. Positive, resolved lower
bounds detect bias; negative/inconclusive ones do not prove independence or
accurate scores. We retain individual image-level observations. Fixed-probe
SEs/95% intervals are not training-seed uncertainty or simultaneous confidence
across all times/arms. Feature bandwidth/ridge are fixed, not selected on test
bounds. Read `THEORY.md` for scope and the three-block corrected teacher.

**Every 50 epochs** (`eval.csv`): 10,000-sample generated and posterior
reconstruction legacy Inception FID at **both CFG 1 and CFG 3**, with RK4-20.
Every 100 epochs, paired **1024-sample checks** compare CFG 1/3, RK4-20/50,
and Gaussian versus encoder-OU `q_T` initialization. Subset FIDs are compared
only within the same sample-count protocol. Image grids and initial arrays
are saved. Conditional sampling is not fabricated by clamping joint reverse
coordinates; incoherent E/G hybrids from the cascade bundle are not used here.

**Frozen continuations** start from the same final source EMA head and VAE.
They use identical minibatch/noise streams, fresh AdamW at 2e-5, a 50-epoch
cosine to 2e-6, EMA 0.999, and no encoder/discriminator/decoder updates. Their
teachers are naive, exact analytic corrected, or raw DSM. Epoch-zero and
endpoint CFG1/3 FIDs, common valid risks and endpoint witnesses are retained.
Comparing raw teacher MSEs across these branches would mix their noise floors;
the common valid risk probes are the intended score comparison.

## Optional expensive score reference

Archive snapshots exist at joint epochs 100, 200, 300, 400, 500. Start with
selected baseline/strong-pressure cells rather than auditing every recipe.

```bash
# Inside a GPU allocation; per-class image banks, not a total-image count.
python paper.py audit --cell-id 4 --epoch 200 \
  --banks 64,256 --particles 16 --queries 16 --times .01,.2,1,2

# Submit the default full-time audit as its own GPU job.
python paper.py submit --cells 4,8,12,14 --audit-epoch 500
```

The reference uses sequential Gaussian conditional-posterior proposals, two
independent particle panels, nested image banks, half-particle checks, ESS,
and maximum responsibility. It reports finite-bank head error and naive-field
bias by block/time/conditioning. A resolution screen failing at a low-noise
time remains a failed screen, not a trained-model failure. The smallest bank
has no smaller-bank comparison and is deliberately not certified. Passing
screens remain heuristics, not certified full-population score accuracy.

## Resume, signals and reports

```bash
# Resubmission resumes the latest committed epoch; completed cells are skipped.
python paper.py submit --cells core
python paper.py run --cell-id 4 --resume
python paper.py report

# Archive an incomplete attempt before starting that cell from scratch.
python paper.py run --cell-id 4 --restart

# Strict requires all 54 predefined jobs, including optional groups, to finish.
python paper.py report --strict
```

Checkpoints are atomic every five epochs, also at epoch zero, endpoint and a
graceful stop. Slurm's five-minute early warning is forwarded to Python, which
finishes/commits the current epoch. A hard kill can replay up to five epochs;
uncommitted CSV rows and later diagnostics/snapshots are removed on resume.
The job exits 75 for a committed early stop; resubmit it. Dependent frozen jobs
use `afterok`, so they require a successfully completed parent and may need
resubmission after an interrupted parent job. They are never initialized from
an incomplete source.

Source/configuration hashes, source-snapshot SHA256, file locks and atomic status
records prevent silent recipe changes and duplicate active runs. Keep code and
data paths fixed for a live study. Checkpoints include optimizers, scheduler,
discriminator, online/EMA models and CPU/CUDA/NumPy/Python RNG states.
The bundle checkpoint format differs from the earlier standalone Colab script;
these are fresh controlled jobs, not automatic imports of the interrupted run.

`report/` contains merged native/diagnostic CSVs, latest-observed-epoch summaries,
matched-seed/common-epoch FID contrasts, dependence trajectories, learned bias
brackets and paired frozen-head valid-risk contrasts. Partial runs are
labeled; no best-checkpoint selection is used. Keep the raw `results/` directories
for paired per-image reanalysis (`*_probes.npz`).

## Offline verification

```bash
python paper.py validate
python paper.py validate --models
python paper.py smoke --cell-id 12 --device cuda
python gaussian_sandbox.py --output report/gaussian_sandbox
```

`validate` tests analytic Gaussian/DAG identities, a nonlinear mean-and-variance
quadrature identity, the learned-bias bound, exact baseline loss/gradient parity,
encoder-only scaling and real-network decoupled controls. `--models` adds
diagnostic isolation, joint/GAN and frozen-corrected interrupted resume, and
small real-network synthetic pipelines across all topologies. Synthetic FID uses
an explicitly labeled feature surrogate; it is not a model-quality result.
See `VALIDATION.md` for the checks actually run while preparing this archive.
