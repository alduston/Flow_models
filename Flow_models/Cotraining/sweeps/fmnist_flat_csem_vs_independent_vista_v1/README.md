# Fashion-MNIST: matched flat CSEM versus two-stage LDMs

This bundle preserves the **flat_three_encoder** architecture and joint recipe
identified in the supplied plan with **seed-42 FID 4.3229685350** and historical
five-seed mean **4.5597724371 +/- 0.1684896094 SD**. It does not use the separate
two-block 4.304 run or the smaller `flat` encoder. Those endpoint values are
provenance reported in the supplied plan, not results newly measured here.

The comparison has three output regimes:

| Regime | Representation | Prior |
|---|---|---|
| `cotrained_csem` | 500 joint epochs, original CSEM/terminal/band recipe | Original joint analytic CSEM |
| `indep_tweedie` | 500 epochs of image loss + beta0 time-zero KL | Fresh DiT, 500 DSM epochs |
| `indep_csem` | Exactly the same frozen VAE as paired DSM | Fresh equally initialized DiT, 500 analytic CSEM epochs |

No cluster training is performed by downloading this archive. All training
starts from scratch: the supplied baseline archive contains source and recipes,
not the completed 4.323 model weights.

## Upload, unpack, and prepare

Upload `fmnist_flat_csem_vs_independent_vista_v1.zip` to the directory requested
in the task, then run on a Vista login node:

```bash
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/sweeps
unzip fmnist_flat_csem_vs_independent_vista_v1.zip
cd fmnist_flat_csem_vs_independent_vista_v1

module purge
module load gcc/13.2.0
module load python3/3.11.8
export VENV_DIR="$SCRATCH/venvs/hlsi"
source "$VENV_DIR/bin/activate"

# Keep the existing working CUDA Torch/torchvision pair.
python -m pip install -r requirements.txt
export CSEM_DATA_ROOT="$PWD/data_cache_fmnist"
export TORCH_HOME="$SCRATCH/torch_cache"
python paper.py init
python paper.py prepare-data --weights
python paper.py plan
```

If your hlsi environment lives elsewhere, set `VENV_DIR` to that existing
environment. `TORCH_HOME` and `CSEM_DATA_ROOT` are inherited by jobs. Existing
valid caches can be reused by setting those paths before the preparation command.
There is no Torch install/upgrade step in this bundle. The inherited weight
cache preparer streams downloads without importing Torch on the login node.

Set `VISTA_ACCOUNT` if your allocation is not the cluster default, for example:

```bash
export VISTA_ACCOUNT=DMS22021
```

Use your current allocation if different. Optionally set
`VISTA_SU_PER_NODE_HOUR` to its known billing factor; no factor is assumed.

## 1. Submit the timed GPU pilot

```bash
python paper.py submit --phase pilot --dry-run
python paper.py submit --phase pilot
```

This is a single `gh` node/job, initially limited to two hours. It first checks
production capacity and gradient paths, then runs interruption/resume and reload
checks on reduced synthetic models. Those metrics are marked synthetic and
cannot enter selection or publication outputs. It then times all four stages
with **production models, real data, ten training minibatches each, and the full
10,000-image evaluation**. GAN is activated immediately for this timing pilot
only. Normal training retains the original epoch-25 GAN start.

The pilot writes `pilot/timing.json` only after all checks/stages finish.
To inspect its completion:

```bash
squeue -u "$USER"
python paper.py status
```

The pilot does not tune using the test set. Its timing models and seed42 data
are excluded from final results.

## 2. Submit the complete staged comparison

After the pilot completes:

```bash
python paper.py submit --phase all --dry-run
python paper.py submit --phase all
```

Both commands print the pilot-based projected cost. Real production submission
requires a completed pilot. Default wall limits use 1.7 times the measured
estimate, rounded up, capped at 48 hours; `--time 48:00:00` supplies an explicit
wall limit if desired. Wall limits are not billed runtime estimates.

The default full DAG has **45 training tasks plus three selection/report jobs**
(48 queued jobs, in addition to the pilot):

1. **Screen:** seed42 joint reference and six independent VAEs at
   beta0 = 0, .01, .03, .07, .15, .30; all run 500 epochs. Only validation
   reconstruction/representation diagnostics select two positive beta points.
2. **Shortlist:** keep the closest reconstruction match and a point on the
   positive-beta reconstruction/KL frontier. Prefer the opposite side of the
   reference distortion when available. Beta=0 remains a diagnostic control.
   The automatic selector stops if no positive point is within 50% of the joint
   reconstruction FID, so a poor screen cannot silently become the final baseline.
3. **Development:** train the two shortlisted seed43 VAEs. Reuse the corresponding
   seed42 screen VAEs. Train both prior objectives at both candidate LRs,
   **3.333333e-4 and 8e-4**, on development seeds42/43: 16 priors, 500 epochs each.
4. **Freeze:** choose one common beta0/LR by mean epoch500 CFG3 validation FID
   across both objectives and both seeds. The paired priors' checkpoint and
   initialization hashes must match. The shortlist and final choice are immutable.
5. **Confirmation:** seeds47--51, 5 joint runs + 5 VAEs + 10 priors = **20 tasks**,
   yielding 15 generative models. Each independent prior depends on its exact,
   successfully completed VAE. Fresh jobs cannot start before parameter selection.
6. **Report:** require all 20 stages and all five seed identities, then produce
   curves, uncertainty tables, paired differences, sample grids and raw exports.

All dependencies use `afterok`, and training verifies successful parent-stage
metadata and hashes even if launched manually. Job names use symbolic beta slots
until the validation shortlist is available; each run's `config.json` records
the resolved beta, source checkpoint hash, selection hash and exact recipe.

If compute is tight, **before the first production submission**, use
`python paper.py init --fixed-lr-only` in a fresh extracted directory instead of
the default init. This freezes the prior LR at 3.333333e-4 and reduces development
to eight priors: 37 training tasks plus three control/report jobs. It is a
predeclared cost-reduction option, not the default optimized comparison.

## Monitoring, resuming, and isolated stages

```bash
squeue -u "$USER"
python paper.py status
python paper.py report                  # complete runs only; clearly labelled live n/5
python paper.py submit --phase all      # skips verified completed and active jobs
```

Checkpoints include model, optimizer, discriminator, scheduler, score EMA,
global RNG, named prior RNGs, loader RNG, exact stage/configuration, parent hash
and timing. An interrupted stage resumes at the last committed epoch. The
early Slurm signal requests a checkpoint at the next epoch boundary. A clean
partial stop returns75, so its dependent jobs will wait rather than consume an
incomplete representation.

After a failed/timed-out parent, replace its own campaign's blocked descendants:

```bash
python paper.py submit --phase all --repair-blocked
```

This repairs only jobs recorded in this directory's submission ledger. It can
cancel pending jobs waiting on old dependencies and queues replacements with new
parent IDs. It never cancels unrelated jobs. Running jobs are reused.
Do not edit code, frozen selections, or manifests after jobs start; use a new
bundle directory for a changed experiment. Invalid completed outputs fail loudly.

Stage-only submission is available with `--phase screen`, `--phase development`,
or `--phase final`, provided the omitted parents are already successful/queued.
All-phase submission is the straightforward workflow.

The optional full-data historical provenance run can be launched inside an
existing GPU allocation with:

```bash
python paper.py run --task historical_joint_s42
```

This is separate from confirmation and does not feed validation selection. It
uses the same training recipe and all60k training images, with the common new
evaluation RNG contract. Exact equality with the old single-run FID is not an
acceptance criterion. Inherited fixed-minibatch routing/capacity checks in the
pilot validate the implementation before the full campaign.

## Evaluation and results

All three new comparison arms train on the **same fixed stratified 50k subset**
of the original60k training images; the remaining10k form a validation reference.
The official10k test images are reserved for confirmation reporting. This is the
split change explicitly prescribed by the supplied plan. The model and joint
loss recipe remain those of the4.323 result; a repeated4.323 value on this split
is not promised.

- Fixed CFG3 primary and CFG1 secondary, Gaussian start, temperature1, T=2,
  t_min=2e-5, decode_time=1e-4, 20 RK4 integration steps.
- The unchanged legacy torchvision Inception extractor and inherited PSD
  covariance FID calculation are used throughout. No new feature backend.
- Evaluation every50 joint/prior epochs, including the fixed500 endpoint;
  independent VAEs report reconstruction diagnostics at the same interval.
- One fixed evaluation start/posterior stream across regimes, seeds and epochs,
  isolated from training; class counts/order follow the common reference labels.
- Actual head forward calls are counted. With the inherited two-pass CFG path,
  RK4-20 uses160 head forwards per sample even at CFG1; it is not20 NFE.
- Endpoint KID uses20 fixed subsets of1000, raw polynomial MMD^2 units. Subset SD
  is separate from training-seed uncertainty.
- Posterior-sample reconstruction FID is a reference, not a guaranteed lower bound.
- EMA is applied only to the score head; VAE state is online/frozen as appropriate.
- Saved-checkpoint evaluation checks identical tensors and full RK4 solves on
  fixed inputs, including CFG1/null-label and CFG3 paths.

```bash
python paper.py report --strict
```

`compiled/` then contains:

- `fmnist_comparison.pdf`, `.svg`, `.png`: two-panel FID versus epoch and total
  training GPU-hours, with pointwise95% Student-t intervals across five seeds.
- `per_seed_metrics.csv`, `curve_summary.csv`, `endpoint_summary.csv/.json`,
  `paired_differences.csv`, `timing.csv`, `status.csv` and a readable `report.md`.
- `common_compute_summary.csv`: comparison only within overlapping observed
  compute ranges for every seed, linearly interpolating between recorded points;
  no extrapolation. Empty if no common range exists.
- `website_training_curves.csv`: `regime,seed,epoch,fid`, CFG3 only, with duplicate
  and seed checks performed by this compiler.
- `representative_seed47/`: prespecified class-balanced images and reconstruction
  references, chosen without best-seed selection.

Independent compute curves include the **full500-epoch VAE cost for each usable
pipeline**, even though the campaign shares that checkpoint. Evaluation time and
allocation runtime are exported separately from training. Generation curves start
at the first measured prior checkpoint after pretraining, not at zero cost.
Both independent stages receive500 epochs; co-training receives500 joint epochs.
This is a fully trained baseline comparison, not an equal-epoch/equal-compute claim.

The historical seeds42--46 are excluded from fresh confidence intervals. The
endpoint differences are seed-paired; overlapping curve bands are not used as a
significance test. An incomplete report displays observed/expected counts and
never writes the strict completion marker.

## Source and checks

`provenance/capacity_matched_source/` is the complete supplied baseline source.
The architecture, packing, original Gaussian-loss helpers, decoder and sampler
remain byte-identical in their root files. `provenance/flat_three_encoder_cell2.json`
is the selected recipe. `RECIPE_DIFF.md` describes intentional differences.

Run `python paper.py check --device cpu` for the model/routing/frozen-state and
interruption/resume suite. The production pilot repeats it on CUDA. The local
validation record in `validation/` describes checks actually run; it does not
claim a Vista production run or a newly achieved FID.

Use `sha256sum -c SHA256SUMS` immediately after extraction to verify archive contents.
