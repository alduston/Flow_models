# CIFAR-10 stacked CSEM mechanism audit

A focused cross-dataset replication of the completed FMNIST stacked mechanism study. This is a **new cluster bundle**, not measured CIFAR results. Extract into a new directory, not over an earlier sweep.

Recovered sources:

- Latest matching CIFAR bundle: `csem_cifar_direct_cascade_800.zip`, 13 October 2026, including its optimized **flat one-block control**.
- FMNIST mechanism bundle: `stacked_csem_mechanism.zip`, 17 October 2026, interpreted using the attached consolidated report dated 21 October (all 54 FMNIST cells completed).
- The older conditional-prior CIFAR runs were **cascades**, not naive joint stacks. This bundle implements one joint noisy-code DiT and one simultaneous reverse solve.

## Launch

Use your existing Vista `hlsi` environment and cached CIFAR / VGG / Inception weights. Do not upgrade a working cluster torch/torchvision pair merely to match the local validation environment.

```bash
unzip csem_cifar_stacked_audit.zip
cd csem_cifar_stacked_audit
source "$SCRATCH/venvs/hlsi/bin/activate"
python paper.py validate
python paper.py prepare-data --data-root ../csem_cifar_direct_cascade_800/data_cache_v6
python paper.py submit --cells all --data-root ../csem_cifar_direct_cascade_800/data_cache_v6 --dry-run
python paper.py submit --cells all --data-root ../csem_cifar_direct_cascade_800/data_cache_v6
```

`--cells all` submits the 12 core jobs first, then 12 frozen jobs with `afterok` dependencies on the correct source jobs. Alternatively submit `--cells core`, inspect its results, and submit `--cells frozen` later. `--cells baseline` selects flat seeds 42/43; `--cells pilot` selects one seed from each core arm. Numeric ranges are supported. Dry runs display dependency placeholders when source jobs have not yet been submitted.

Slurm defaults: `gh`, one node/task, 24-hour cap, gcc/13.2.0, python3/3.11.8, `$SCRATCH/venvs/hlsi`. Override with `--time`, `--partition`, `--account` or `VENV_DIR`. Runtime is not measured here. Every job runs a separate-process CUDA graph smoke first. Re-submit the same command after a time limit: active/completed jobs are skipped, incomplete runs resume full state. Joint checkpoints every ten epochs and around evaluation; frozen checkpoints every epoch. Changing scientific sources/manifest invalidates resume. Source/representation final epoch is fixed, never chosen by best FID.

## The 24 cells

Every core arm uses seeds 42 and 43 and 800 joint epochs. Every frozen arm uses 50 additional head-only epochs.

| Cells | Arm | Purpose |
|---|---|---|
| 0–1 | Optimized flat, one block | CIFAR baseline with its existing B penalty |
| 2–3 | Two-block full, r=0 | Head continues to learn; remove only score-derived encoder gradients |
| 4–5 | Two-block full, r=0.5 | Moderate pressure suggested by FMNIST |
| 6–7 | Two-block full, r=1 | Nominal CIFAR pressure |
| 8–9 | Two-block full, r=4 | Strong-pressure stress test |
| 10–11 | Two-block decoupled, r=1 | Conditional-independence / zero-teacher-defect control |
| 12–14 | Source 6; naive / corrected / DSM | Fixed representation teacher intervention |
| 15–17 | Source 7; naive / corrected / DSM | Second seed |
| 18–20 | Source 8; naive / corrected / DSM | Strong-pressure source |
| 21–23 | Source 9; naive / corrected / DSM | Second seed |

No third-block, topology grid, clipping grid, boundary-weight grid or architecture sweep is included.

## CIFAR settings preserved

The recovered `core.py`, `suite.py`, `paper_io.py`, `direct_models.py`, `hier_models.py` and `band_regularizer.py` are byte-identical. Guarded adapters change only posterior construction/coherent sampling, encoder gradient scaling, evaluation banks and added diagnostics.

| Setting | Value |
|---|---|
| Input / batch | CIFAR-10 RGB 32×32; 256 |
| Latent coordinates | 512 in every arm; flat 8×8×8; stack root 8×4×4 (128), child 6×8×8 (384) |
| Joint head | One **384-wide, depth-12, 6-head** factored DiT in every arm |
| Root packing | Exact pixel permutation into 2×8×8; child occupies remaining six channels |
| T_K / T | **1.05 / 1.35** |
| OU times | 1,000 log-time nodes, t_min=3e−5 |
| Mean anchor | OU-partial at 1.05, per Gaussian factor |
| Encoder CSEM / head coefficient | **0.05 × r / 1.0** |
| VAE / head LR | **2.5e−4 / 1e−4**, unchanged when r varies |
| Representation / head weighting | Canonical / unweighted epsilon; independent full-horizon detached head regression |
| Terminal factor penalty / B | **0.60 / 0.05**, fixed across r |
| Precision auxiliary / cosine | 0.0025 / 0 |
| LPIPS / GAN | 0.85 / 0.0025, GAN starts epoch 25 |
| Optimizer / clipping | Original AdamW, original joint norm-1 clip after head-gradient replacement |
| Schedule / EMA | 800-epoch cosine to 1e−6 / head EMA .9997; online VAE |
| Refinement after joint phase | None |
| Evaluation | Every 50 epochs, 10,000 images, original legacy torchvision Inception |
| Primary endpoint | **Epoch 800, full-T Gaussian start, temperature 1, RK4-20, CFG 2.5** |
| Secondary checks | CFG **1** and 3; RK4-25 and Heun-20 at each guidance level; inherited Gaussian T_K check |

The joint head sees only all noisy coordinates, time, and class/null label. No clean parent, posterior moments or image enters it. Both latent blocks evolve together in a **single reverse solve**, then the inherited direct-child decoder consumes those same generated coordinates. The 12-layer joint head exactly matches the flat score architecture/parameter budget, and same-seed initial head tensors match across flat/stacked arms. Hierarchical VAE construction runs under an RNG fork after consuming the original flat construction stream. The hierarchical VAE is larger; actual VAE/head counts are recorded, so this is not an exact total-capacity match.

At r=0 terminal KL, boundary, reconstruction, LPIPS and GAN remain active, and the score head learns on the moving encoder. It is not a separately trained ordinary VAE/LDM baseline. CSEM feedback scales all noisy-input, teacher and ancestry routes; inner head gradients, its LR and its coefficient stay fixed. B remains 0.05, as in the main FMNIST pressure comparison.

The decoupled control substitutes the deterministic root mean for the sampled root only in child inference. The decoder still consumes both noisy codes. This removes coupling **given X**, not aggregate dependence through image content.

## Frozen teacher contract

All three continuations of each source load the identical final EMA head and freeze the **entire** VAE, including decoder. They use identical batch/noise/dropout streams, fresh AdamW, full-horizon unweighted-epsilon regression, one head-only norm-1 clip and a 50-epoch cosine schedule.

Frozen LR is the recovered **CIFAR refinement LR 1.5e−5**, minimum 1e−6. EMA stays at CIFAR .9997. No FMNIST .6 loss coefficient, head-LR division, T=2 horizon, .1034 boundary coefficient or FMNIST optimizer schedule is imported.

The natural-parameter auxiliary term is disabled **in all three frozen branches** so the intended intervention is the epsilon teacher. The naive branch controls for that shared change and for resetting optimizer/schedule. Compare valid versus matched naive continuation, and naive continuation versus unchanged source separately.

- `naive`: original factorwise analytic epsilon target, evaluated using the coherent sampled clean parent.
- `corrected`: analytic conditional-posterior-transport target, with descendant mean/variance derivatives through the inferred root. Its conditional mean is the valid joint epsilon field.
- `dsm`: actual diffusion innovation, also a valid joint epsilon target.

Neither valid target is a population-converged learned score. Fifty head-only epochs give a matched finite-budget intervention.

## Diagnostics and exports

`audit/epoch_XXXX.csv` contains held-out class/null risks at t=.01,.2,1.05,1.35. `valid_risk = mean(p²−2pT_corrected)` and DSM analogue are proper risks up to the teacher-independent constant. Comparing predictions on the **same frozen representation and same probes** is valid; comparing these absolute values across different encoders is not an accuracy ranking.

- Conditional symmetric KL is measured over eight independent root draws per held-out image, at initialization and every evaluation. Persistent coupling and changes in conditional variance are distinct observations.
- Bias witnesses fit the defect `C_naive−T_corrected` on 2,048 training images and evaluate on 1,024 held-out test images at t=.2, initially and every 200 epochs. Class and null witnesses are separate. Linear + 64 Fourier features, ridge .001, signed lower bound `2 E[delta·w]−E||w||²`; negative lower bounds are retained. The full pathwise defect is only an upper bound, not proof of conditional-mean bias.
- Witness head-error projection distinguishes teacher defect from bias inherited by the head. Raw per-image arrays and fixed probe seeds are exported.
- Frozen source and continued heads are evaluated together on common probes. The compiler checks source-risk array identity before taking teacher-versus-naive paired risk differences.
- Existing histories retain reconstruction, posterior variance, terminal factor penalty, naive weighted/unweighted loss and clipping statistics. B per-block diagnostics remain available.

The inherited flat Gaussian-mixture oracle and mixture-based LSI/score gaps are disabled: a hierarchical posterior is not that mixture. Corresponding legacy accuracy columns are NaN, not fabricated joint-score errors. The hierarchy's sum of OU-marginalized conditional-factor penalties is preserved as its terminal regularizer; it is **not claimed to equal KL of the jointly noised aggregate**.

```bash
python paper.py report
python paper.py report --strict
```

Outputs in `compiled/`: endpoint inventory, all histories/diagnostics, paired FID contrasts, paired fixed-model valid-risk contrasts with per-image SE, across-seed aggregates, FID trajectories and `SUMMARY.md`. Missing final epochs stay missing; partial latest values are labeled separately. Strict mode requires every scheduled final endpoint and FID contract. It does not convert per-image SE into training-seed uncertainty.

Read `RESEARCH_PLAN.md` for the decision rules and `VALIDATION.md` for what has actually been checked. No cluster training or real CIFAR FID was executed in this workspace.
