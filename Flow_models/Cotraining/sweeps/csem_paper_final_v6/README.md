# CSEM paper sweep V6

Nine files, one command-line entry point. No patches, duplicate audits, old wrappers,
archived logs or example result files are shipped. The modern engine and the recovered
August-13 FMNIST engine remain separate so their training objectives are preserved.

## Install and run on Frontera

Upload `csem_paper_final_v6_bundle.zip` to the sweeps directory, then:

```bash
cd /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/sweeps
unzip csem_paper_final_v6_bundle.zip
cd csem_paper_final_v6
module purge
module load gcc/13.2.0
module load python3/3.11.8
source "$SCRATCH/venvs/hlsi/bin/activate"
python paper.py validate
python paper.py import-v5 --source ../csem_paper_final_v5
python paper.py submit --cells main --dry-run
python paper.py submit --cells main
```

`main` selects cells 0–7 and skips imported/completed cells. With the results you
uploaded, it submits **six** training jobs: 0, 1, 2, 3, 6, 7. Each independent job
produces BOTH independently trained heads. Submission first creates one dataset
preparation job; training starts only after that job succeeds. Preparation and
training use the same existing `hlsi` environment and `gh` partition.

To submit all known failed jobs from the uploaded status file, including ablations:

```bash
python paper.py submit --cells 0-3,6-7,11,17-18
```

Cell 16 had no final status in the uploaded results. If its V5 job finishes,
rerun `import-v5` before scheduling it. If it has stopped without valid results:

```bash
python paper.py submit --cells 16
```

Alternatively, `python paper.py submit --cells missing` schedules every unfinished
cell in V6, including cell 16. Check that any older V5 jobs have finished first;
active V6 jobs are checked with `squeue` and skipped automatically. Submission
never overwrites or cancels a V5 job.

For only the independent baselines:

```bash
python paper.py submit --cells independent
```

## Baselines and paper comparisons

| Cells | Dataset | Seeds | VAE-only epochs | Frozen-VAE prior epochs | Outputs |
|---|---|---|---:|---:|---|
| 2, 3 | CIFAR | 42, 43 | 800 | 800 | Independent CSEM; Independent Tweedie |
| 6, 7 | FMNIST | 42, 43 | 700 | 700 | Independent CSEM; Independent Tweedie |

The two heads have separate parameters and optimizers and train on paired samples
from one frozen VAE. CSEM uses the analytic conditional epsilon expectation;
Tweedie/DSM uses the sampled noise target. Neither prior changes the VAE. The
baseline KL is the conventional time-zero KL: CIFAR beta=0.07, FMNIST beta=0.01.
The frozen-prior LR is 1e-4 for CIFAR and 8e-4 for FMNIST, with the existing cosine
decay to 1e-7. The baseline horizon is 1.35 for CIFAR and 2 for FMNIST.

The main co-training recipes, 800-epoch CIFAR schedule and 700-epoch recovered
FMNIST schedule are unchanged from V5. Result directories now have unique names
derived from dataset, mode, seed and cell ID. `source_v5_result_name` in the
manifest maps every old directory to its new name.

`legacy_fmnist.py` retains V5's `provenance/csem_new_fmnist_aug13.py` verbatim
(SHA256 `0e4d76e8dae5cae41ec8bd109c2ac76e3dbcfcc795ef55b11386f4723cc258da`).
That source was recovered from the August-12 upload `csem_new (1).py` and matches
the August-13 terminal-KL Gaussian-start run's CLI and logging. Its original
terminal run directory was `Cotraining/fmnist_run3/run_results_fmnist_scale_norm_vs_terminal_kl/run_terminal_kl`.
The recovered recipe uses horizon 2, unweighted epsilon CSEM coefficient .6,
terminal KL 1, head LR 2e-4/.6 with the original joint gradient coefficient .6,
shared time/noise draws and original joint clipping. These differ from the older
canonical horizon-1.5/weight-.1/KL-.3 recipe. The exact historical terminal-KL
FID 5.04 was never located; the separate August-18 approximately-4.97 oracle-start
result is not a Gaussian-start headline.

While jobs run, or after they finish:

```bash
python paper.py report
```

This writes CSV tables to `compiled/` and PNG/PDF figures to `figures/`, including:

- Co-trained CSEM, Independent CSEM and Independent Tweedie versus **prior-training
  epoch**, for both datasets: FID, KID, feature SW2, latent SW2 squared and CSEM gap
  in score units. Independent VAE-only epochs are excluded from that x-axis.
- The old-paper paired panel: FMNIST FID and CIFAR KID versus prior-training epoch.
- Scale-anchor dynamics, posterior variance/collapse controls, horizon/anchor
  sensitivity, CIFAR LR/loss trajectories and Heun/RK4 solver comparisons.

Main curves use Gaussian starts, temperature 1, RK4-20 and 10,000 evaluation
samples. For each dataset, `best-final` chooses between the evaluated CFG
candidates using mean completed co-trained endpoint FID, then applies that ONE
CFG to all three regimes, both seeds and every epoch. This is selection within
the existing grid, not proof of a global optimum. An unfinished co-training
selection is labeled provisional. `--cfg-policy old-paper` uses FMNIST CFG=1.5
and CIFAR CFG=3; `--cfg-policy recipe` uses FMNIST CFG=3 and CIFAR CFG=2.5.
`--cifar-treatment raw` switches the co-trained CIFAR comparison to the raw-mean
terminal-KL controls. Both flags are accepted by `report` and `compile`.

Bands show across-seed standard deviation where both seeds are available.
`main_curve_coverage.csv` and `figure_coverage.json` expose missing regime/seed
results. Live CSVs contribute curves; only successful endpoint-complete runs
enter final tables. Missing curves/metrics are never filled with invented values.
Legacy horizon ablations used 5,000 samples while the center used 10,000; these
sample counts remain explicit and are plotted as separate series.

## Repairs and reuse of existing results

1. **Independent evaluation:** the Gaussian-only baseline previously requested an
   oracle-qT sampler without constructing its bank. The sampler plan now also
   determines bank requirements. Gaussian-only baselines request no oracle banks;
   explicit oracle comparisons retain the appropriate bank(s).
2. **CIFAR cache:** one preparation job validates both splits, downloads/extracts
   into fresh staging and publishes a complete snapshot under a dataset lock.
   Corrupt snapshots are quarantined and clean downloads are retried. Training
   opens validated data with `download=False`, under a shared lock, and uses the
   new `data_cache_v6/` directory. It never repairs or modifies V5's live cache.
3. **FMNIST metric names:** the old replacement accidentally inserted ASCII
   control-A and erased CFG tokens. New evaluation labels retain CFG3 and CFG1.5
   distinctly. The importer repairs V5 labels using the verified evaluator order:
   first column CFG3, duplicate `.1` CFG1.5; original Heun-50 columns are CFG3 only.
   Unknown ambiguous labels are rejected. Empty cross-dataset columns in combined
   exports are removed before importing, without removing any observed value.
4. **Resume:** modern runs save full state every ten epochs, at the frozen-VAE
   boundary, and before/after evaluation. This includes both online/EMA priors,
   both prior optimizers/schedulers, VAE, discriminator, RNG and metric histories.
   Interrupted evaluations are retried before the next training epoch. Paired
   evaluation rows are deduplicated, and changed training configurations are
   rejected on resume. Live CSV updates are atomic.
5. **Reporting:** reads both final and in-progress loss/evaluation files, requires
   frozen-VAE `Refine` tags for independent curves, preserves CFG/solver/init
   labels, clears stale empty tables and retains seed-specific solver figures.
   A successful main job must contain both required heads, all evaluated epochs
   and finite paper metrics; incomplete baselines cannot be reported as complete.

`import-v5` reads the old result/status directories, or their compiled CSV exports.
It copies completed compatible histories and corrected labels into V6 without
changing old files or measured values. It does not copy model checkpoints or
infer training states from CSVs. `import_report.json` records exactly what was
imported and which headers were repaired. The uploaded results successfully
imported cells **4,5,8,9,10,12,13,14,15,19,20,21**. The independent V5 jobs never
produced evaluation records. Without actual complete-state checkpoints, those
failed runs require training again; their loss CSVs alone cannot restore weights.

V6 Slurm jobs resume automatically when a complete-state checkpoint exists.
For a direct run: `python paper.py run --cell-id 6 --resume`. To restart an
unrecoverable V6 attempt, use `--restart` or submit with `--restart-incomplete`;
the previous attempt is moved to `failed_attempts/`, not deleted. The historical
FMNIST engine has no full-state resume; completed V5 FMNIST runs can be imported.

## Files and validation

| File | Role |
|---|---|
| `paper.py` | Validate, import, prepare, submit, run, compile and plot |
| `suite.py` | Resolve recipes and adapt historical evaluation |
| `core.py` | Modern training/evaluation engine |
| `legacy_fmnist.py` | Unchanged recovered August-13 training source |
| `paper_io.py` | Sampler, header, dataset and resume contracts |
| `manifest.csv` | 22 cells and explicit baseline outputs |
| `job.slurm` | Shared preparation/training launcher |
| `tests.py` | CPU regression tests |
| `README.md` | Commands, configuration and audit |

Validation passed: all 22 resolved recipes; actual evaluator sampler planning;
CFG preservation/recovery; corrupt-pickle repair, retry and concurrent preparation;
actual phase checkpoint/evaluation functions with injected interruption; frozen
VAE routes; complete/partial table compilation; all three paper regimes and
PNG/PDF rendering. The actual uploaded results imported all 12 successful cells
and recovered the FMNIST RK4-20 endpoint FIDs 5.170576 (seed 42) and 5.168397
(seed 43) at CFG3. The Slurm script passes shell syntax validation.

These are CPU contract and reporting checks. CUDA training, actual torchvision
downloads and numerical FID reproduction were not run in this environment and
remain to be exercised by the cluster jobs. `python paper.py validate` reruns the
CPU checks without needing Torch; fixture values are temporary and are never
presented as experimental measurements.
