# LSGM → CSEM: matched CIFAR-10 runs with 1, 2 and 4 latent groups

This is the surgery described in `SURGERY_PLAN.md`, applied to vendored
[NVlabs/LSGM](https://github.com/NVlabs/LSGM) at commit
`5eae2f385c014f2250c3130152b6be711f6a3a5a`. It uses LSGM's NVAE and NCSN++
code, not the encoder, DiT, LPIPS, GAN or evaluator in the previous 800-epoch
bundle. Keep `lsgm/LICENSE` with the code; upstream is licensed for
non-commercial research. No dataset, pretrained weights or trained checkpoint
is included.

The requested CSEM cells are **B1-C, B2-C, B4-C**. They have shared NVAE
pretraining cells A1/A2/A4 and separate final evaluation cells C1-C/C2-C/C4-C.
Matched joint LSGM baselines and the plan's calibration/ablation cells are also
included. The manifest has 26 cells in total; nothing is automatically submitted.

## Run the three CSEM configurations

Unzip, enter this directory, and use your existing CUDA-enabled `hlsi` venv.
Install only the additional dependencies; retain the matched torch/torchvision
builds already installed for your cluster.

```bash
source "$SCRATCH/venvs/hlsi/bin/activate"
python -m pip install -r requirements-extra.txt
python paper.py validate
python paper.py prepare-data
```

Prepare the original LSGM FID weights/statistics once. `prepare-fid` downloads
the pytorch-fid Inception weights and computes statistics from all 50,000
**training** images, using the original LSGM Inception implementation.

```bash
# Login/network node: download weights without needing a GPU.
python paper.py prepare-fid --weights-only
# Then run this on an allocated CUDA node with the same venv:
python paper.py prepare-fid
```

On Vista, the SLURM script retains `gh`, one node, one task and 48 hours. It
loads gcc/13.2.0, python3/3.11.8 and the same `$SCRATCH/venvs/hlsi` venv as the
old bundle. If you want the custom CUDA FIR kernel, set `LSGM_CUDA_MODULE` to
the installed module matching your torch CUDA build before submitting. Module
names vary between cluster installations; the script does not invent one.

```bash
# Optional: select a CUDA module installed on YOUR cluster.
# export LSGM_CUDA_MODULE=cuda/<installed-version>

python paper.py submit --phase A --cells csem
# Once all three A cells have COMPLETED successfully:
python paper.py submit --phase B --cells csem
# Once their B cells have COMPLETED successfully:
python paper.py submit --phase C --cells csem
python paper.py report
```

Each submission stage selects exactly three cells for `--cells csem`.
Phase B/C jobs are gated on successful prerequisite status and a checkpoint;
there are no SLURM dependencies or automatically launched follow-up jobs.
Re-run a submission command after prerequisites finish. Already completed and
active jobs are skipped. Each job runs a synthetic CUDA graph smoke test in a
separate process before entering the real trainer. Dataset downloading is
restricted to preparation; training only validates and reads the prepared cache.

For the **matched LSGM + CSEM comparison**, use `--cells main` instead of
`--cells csem`: three A jobs, six B jobs, six C jobs. Every pair starts from
the same corresponding A checkpoint. Calibration is a separate selection:

```bash
python paper.py submit --phase A --cells calibration
# After A0 completes:
python paper.py submit --phase B --cells calibration
# After B0 completes:
python paper.py submit --phase C --cells calibration
```

`tier1` selects main + calibration. `cascade-controls` selects B2-LC/B4-LC
and their C evaluations. `ablations` selects B1-Cnb/B1-C2H and their C
evaluations. `all` selects the entire plan. Explicit comma-separated IDs work.
Optional controls reuse the already completed A checkpoints.

```bash
python paper.py commands --phase B --cells csem
python paper.py submit --phase B --cells main --dry-run
python paper.py run --cell-id B2-C --resume
python paper.py smoke --cell-id B4-C --device cuda
# Expensive production-width, batch-2 check on a CUDA node:
python paper.py smoke --cell-id B4-C --device cuda --full-size
```

## Protocol and capacity

The plan deliberately replaces the old **800 epochs** with LSGM's two-stage
ablation protocol: **400 NVAE epochs + 1,000 end-to-end epochs**. Unconditional
CIFAR-10, DML reconstruction likelihood, no class labels or CFG.

| Configuration | Groups × channels | NVAE parameters | Prior parameters | Prior width |
|---|---:|---:|---:|---:|
| B1-L / B1-C | 1 × 180 | 41,592,384 | 137,883,496 | 512 |
| B2-L | 2 × 90 | 70,508,920 | 137,883,496 | 512 joint |
| B2-C / B2-LC | 2 × 90 | 70,508,920 | 132,958,024 | 352 per head |
| B4-L | 4 × 45 | 128,560,692 | 137,883,496 | 512 joint |
| B4-C / B4-LC | 4 × 45 | 128,560,692 | 144,273,000 | 256 per head |
| B0 calibration | 4 × 45, nf=2 | 130,622,772 | 137,883,496 | 512 joint |

Every latent is 180 × 16 × 16 in total. Main cells disable normalizing flows,
so each posterior conditional is exactly diagonal Gaussian. Ancestors follow
NVAE's decoder order. Cascades expose the same parameter/spectral/EMA interface
as a single prior and need G sequential ODE solves. Actual prior capacity is
approximately matched: 0.964× for G=2 and 1.046× for G=4. The NVAE itself grows
with group count; it is matched **within each paired comparison**, not across G.
`architecture_counts.json` contains full production-shape parameter counts.

Stage 1 retains NVAE's Adamax optimizer, LR 1e-2 → 1e-4, batch 256, 5 warmup
epochs, spectral/BN regularization 1e-2, KL annealing to 1.0 and best-validation
checkpoint selection. Its training file is unchanged.

Stage 2 retains NCSN++ with two residual cells per scale, three scales, dropout
0.2, mixed prediction, FIR, VPSDE beta 0.1 → 20, `time_eps=0.01`, independently
sampled `drop_all_iw` prior and `ll_iw` encoder routes, batch 128, AMP,
spectral/BN regularization 1e-2, Adam/FusedAdam, LR 1e-4 → 1e-5 for the VAE,
constant prior LR 1e-4, EMA 0.9999, no gradient clipping, and the plan's
continued KL annealing from 0.7 to 1.0. Apex is optional and its use is recorded.
An explicit full-size smoke uses the same widths but batch 2; ordinary smokes
reduce widths and latent channels and never produce scientific training results.

Final Phase C uses the **final** `checkpoint.pt` with EMA prior weights:
50,000 ODE samples, original LSGM train-set FID statistics, RK45,
`ode_eps=1e-6`, tolerance 1e-5, and full test-set ODE NLL. Training monitoring
remains 2,500-sample FID and ten validation batches every 50 epochs.
Small-sample monitoring FIDs and final 50k FIDs stay in separate columns.

## Objective and routing

For each Gaussian conditional with mean μ and log standard deviation ℓ,
`v_comp = m(t)^2 exp(2ℓ) + var(t)` and the CSEM epsilon target is
`sqrt(var(t)) (z_t − m(t) μ) / v_comp`.

The encoder receives reconstruction plus its scheduled KL coefficient times:

1. Likelihood-weighted CSEM regression (`g²/(2 var)` with LSGM importance sampling).
2. Per-component terminal KL to N(0,I), with coefficient 1.
3. The band closure `0.5 softplus(log(var_eps) − 2 log(m_eps) − 2ℓ)`.

The target, noised latents and clean ancestor context are live on this route.
The prior route retains LSGM's independent, full-horizon, unweighted-epsilon
importance-sampled objective, with all VAE moments, latents and ancestors
detached. Its final gradients equal the gradients of this p loss alone.
CSEM does **not** add encoder entropy again: that cancellation is already
inside the analytic target objective. LSGM arms retain DSM plus their original
entropy/cross-entropy terms. Cascaded LSGM is an explicit factorization control.

B1-Cnb turns off the band term. B1-C2H truncates only the representation route
to t_K=0.4544 (with a correctly density-adjusted uniform proposal) and uses
terminal coefficient 12; the head/sampling horizon remains t=1.

## Compatibility changes beyond the plan's pseudocode

All changes are in `patches/lsgm_csem.patch`, with per-file upstream and patched
hashes in `provenance.json`. `paper_io.py` is copied byte-for-byte from the input.
`training_obj_disjoint.py`, `train_vae.py`, diffusion implementations, NCSN++
layers and NVAE layers remain unchanged. The new objective is `csem_lsgm.py`.

* The modern torch in-place error is repaired by delaying the VAE optimizer
  step until both backwards finish when p/q objectives differ. This applies to
  **every matched arm**, including LSGM. Losses and pre-update gradients retain
  upstream semantics; exact upstream first-loss parity is checked. The original
  update order failed the real-model torch 2.7 test.
* Upstream EMA inherited Optimizer without initializing its hook metadata;
  modern `state_dict`/schedulers fail. Its constructor now initializes that
  metadata before restoring the same underlying state/groups. The EMA formula
  and decay remain unchanged. EMA save/load/swap is tested.
* `lsgm_runner.py` restores the pre-torch-2.6 `torch.load(weights_only=False)`
  behavior for **trusted local LSGM checkpoints containing argparse.Namespace**.
  This allows leaving `train_vae.py` unchanged. Use the runner through paper.py.
* `_accumulate`, SyncBN backward's count argument, and SciPy `sqrtm` are updated
  for current dependencies. On a single GPU NVAE takes its original framework
  BN+Swish path; the synchronized backward branch is used at world size >1.
* Parameter-count logging uses Python `sum` over tensor `numel()` values. Newer
  NumPy rejects upstream's `np.sum(generator)` before training starts. The same
  auxiliary-parameter exclusion is retained, and the GPU smoke now exercises
  this shared A/B/C startup helper.
* FIR/fused CUDA extensions are lazy rather than compiled at import. Backend
  `auto` tries CUDA and explicitly warns/logs a native PyTorch fallback if JIT
  fails. Set `LSGM_OP_BACKEND=cuda` for a strict custom-op requirement, or
  `native` to use the same native implementation for every arm. Successful
  completion records the actual backend. Do not mix backends between arms.
* Cascades use ODE previews during training; the upstream discretized preview
  is skipped. Asking a cascade for discretized FID/NLL raises a clear error.
* Conditional ODE NLL supports multiple trace-estimator samples and reports
  uncertainty for the sum over heads, rather than returning missing std values.

Published nf=2 FIDs are **reference values**, not measured results in this
bundle. B0 exists to calibrate the port on your hardware. A0/B0 should be run
before interpreting comparison results; no reproduction claim is made here.

## Resume, records and reporting

`run` locks each cell, preserves the manifest configuration hash, requires
successful prerequisites, and resumes an existing checkpoint by default.
Optimizer, scheduler, scaler, spectral-norm and EMA state use upstream resume
behavior. RNG state is not saved by upstream, so resumption is not bitwise
identical to uninterrupted training.

Stage B checkpoints are saved every ten epochs and at the final epoch. Stage A
keeps LSGM's **best-validation** checkpoint; resuming A can roll back to a much
earlier improvement. The completion marker certifies the entire stage-1 epoch
budget finished, while the shared best checkpoint can have an earlier epoch.
Final B certification requires epoch 1000 and the final evaluation row.

Use `--restart-incomplete` on submission, or `--restart` on `run`, to archive an
incomplete attempt into `failed_attempts/` and start afresh. Completed runs are
protected. Failures record a nonzero return code and are never treated as
completed. Run commands are stored verbatim in each result's config record.

Each B cell records `eval_in_progress.csv`, `diagnostics.csv`, parameter counts,
environment/backend information and console output. C cells record
`final_metrics.json`: final FID/NLL, NFE per head and total, sampling seconds,
and model capacity. Group diagnostics report sampled per-image rep/K/B terms,
posterior sigma quantiles, cutoff/deep-cutoff fractions, mixing coefficients,
and iteration time. Expensive sigma quantiles are sampled every 200 steps, then
averaged per epoch. Upstream active-unit and per-group KL logs remain in
TensorBoard. `report` writes `compiled/latest.csv`, `curves.csv` and FID curves;
missing values remain missing.

## Validation limits

```bash
python paper.py validate          # no torch needed: hashes, syntax, all parsers, gating
python paper.py validate --torch  # scientific checks + reduced-width CPU model graphs
```

Delivered validation covers Monte Carlo Rao–Blackwell targets/variance, 20,000
Gaussian band bounds, the finite-horizon population identity, cascade call order,
provided-noise reproducibility, first-loss parity, p→VAE isolation, live ancestor
conditioning, p-only final head gradients, five actual model training steps,
and EMA/checkpoint state round trips. Real RK45 cascade sampling, conditional
prior likelihood and NVAE decoding are also checked. CPU checks adapt CUDA
allocation and BN inside the **test process only**, retaining production code.

This workspace had no CUDA GPU. CUDA kernels, full-batch AMP, the production
training budget and FID reproduction must therefore be checked on the cluster.
`validation_results.json` records exactly which checks ran; CPU smoke values
are test diagnostics, not CIFAR results.
