# Updated Fashion-MNIST CSEM comparison: Vista run-bundle plan

Date: October 28, 2026  
Status: experiment and implementation plan; no new bundle or cluster jobs created.

## Objective

Produce an updated, reproducible Fashion-MNIST figure comparing the optimized **flat CSEM co-training** architecture against the same-capacity independently trained latent diffusion model with either **Tweedie/DSM** or **analytic CSEM** supervision. Use repeated training seeds, fixed evaluation settings, and explicit accounting for the two-stage training cost.

The three regimes answer two questions: does swapping the score target help on a fixed representation, and how much does the complete co-training procedure improve generation relative to independently learned representations? The latter is a comparison of training procedures, including their representation regularizers; it does not isolate a single gradient term.

## 1. Recovered source and exact 4.3 result

Use **`fmnist_capacity_matched_4p3_baseline.zip`**, extracted directory **`fmnist_capacity_matched/`**, as the implementation base. Its model version is `fmnist_capacity_matched_v1`.

The completed **`fmnist_compiled_results.zip`** identifies the relevant run unambiguously:

| Item | Verified value |
|---|---|
| Run | **`flat_three_encoder_s42`**, manifest cell 2 |
| Configuration | `arm=flat_three_encoder`, `capacity_family=budget_three`, `topology=decoupled` |
| Actual stochastic depth | **One** image-conditional diagonal Gaussian |
| Endpoint | Epoch 500, CFG 3, Gaussian start, temperature 1, RK4-20, 10,000 images |
| Seed-42 generated FID | **4.3229685350** |
| Seed-42 posterior reconstruction FID | **2.8529478747** |
| Five-seed generated FID, seeds 42–46 | **4.5597724371 ± 0.1684896094 SD** |
| Five-seed 95% Student-t interval for mean FID | [4.3505649551, 4.7689799192] |
| Five-seed reconstruction FID | **2.8577967668 ± 0.0505912910 SD** |

The approximately 4.3 number is a **single-seed endpoint**, not the seed-average performance. Do not replace the new curve's endpoint average by 4.323.

The earlier `b2_full_r0.5_s43` result, FID 4.3040, belongs to a **two-block** model. It is provenance for the recipe, not the flat architecture to use here. The separate `flat` arm also differs from `flat_three_encoder`; its five-seed CFG-3 mean is 4.58937.

### Architecture to preserve

- `models.py::BudgetThreeVAE` with the decoupled `original_models.py::Posterior` wrapper and the inherited multiscale decoder.
- 256 latent coordinates: 64 root coordinates at 4×4, then 64 and 128 coordinates at 8×8. Storage shapes are `(4,4,4)`, `(1,8,8)`, `(2,8,8)`; pack into 4×8×8 through the existing exact permutation.
- Child posterior inputs use recursively computed **deterministic posterior means**, rather than sampled parent codes. Consequently all means and variances depend on the image alone. Three storage groups/encoder stages do not make this a stochastic three-block model.
- Conditioner hidden widths **87 and 75**, retaining the parameter-budget allocation of the capacity-matched experiment.
- One joint class-conditional DiT: **width 192, depth 18, six attention heads, patch size 1**.
- **Ordinary epsilon output, not a factored head.** Do not substitute the older eight-layer flat prior or a new factored parameterization.
- VAE parameters **6,368,817**; score parameters **12,159,556**; combined **18,528,373** (excluding discriminator).
- No normalization of the encoder mean: inherited `use_norm=False`. Internal feature GroupNorm layers remain.

The updated website's factored-head schematic therefore does not exactly depict this measured implementation. After the experiment, label it as conceptual or revise it before associating the new result with that architecture.

## 2. Freeze the co-training recipe

Copy manifest cell 2 and change only run/output identifiers and prescribed seeds. Preserve the actual computation, reductions, and gradient routing.

| Setting | Fixed co-training choice |
|---|---|
| Dataset/preprocessing | Inherited FMNIST loader and `fmnist_reference` preset; 32×32, one channel; preserve exact normalization/resizing |
| Training | 500 joint epochs; batch 128; two loader workers; no encoder warmup |
| OU process | alpha(t)=exp(-t), sigma²(t)=1-exp(-2t) |
| Time sampling | Uniform index sampling over 1,000 log-spaced points, t_min=2e-5 to T=2 |
| Horizons | Fully joint inherited recipe; effectively T_K=T=2, no detached long-time tail |
| CSEM metric | Unweighted epsilon MSE, with the original mean reductions |
| Score coefficient | `score_weight=0.6` |
| Encoder routing | `rep_multiplier=r=0.5`; effective CSEM encoder coefficient 0.3 |
| Terminal anchor | `terminal_weight=1.0`, at T=2 |
| Small-time band | `tmin_band_weight=auto`, `band_mode=fixed`; preserve original resolution function |
| Image loss | Inherited noisy reconstruction MSE + 0.85 LPIPS(VGG) + 0.0025 GAN |
| Adversarial schedule | Starts epoch 25; discriminator LR 1e-4 |
| Optimizer | AdamW, betas (0.9,0.95), weight decay 1e-4 |
| VAE LR | 5e-4 |
| Head LR | Configured 2e-4, **actual optimizer LR 2e-4/0.6 = 3.333333e-4** |
| LR schedule | Inherited cosine schedule to 1e-6 |
| Clipping | Combined VAE+head global norm 1 |
| EMA | Score head only, decay 0.9997; online VAE |
| Label dropout | 0.1; retain original null-label convention |
| Sampling/decode | Full-T Gaussian initialization; temperature 1; decode time 1e-4 |
| Primary evaluation | CFG 3, 20 RK4 integration steps, 10,000 samples, legacy Inception backend |
| Secondary evaluation | CFG 1 at the same checkpoints; posterior reconstruction FID |

Important implementation details:

1. `r=0.5` scales **both input and target gradient routes into the VAE**. It does not halve the score-head gradients or multiply the entire scalar loss by 0.5.
2. The small-time band remains fixed under r changes in this recipe. Record its resolved value from the actual schedule in `config.json`; do not recompute it using 0.3 merely because the encoder CSEM gradient coefficient is 0.3.
3. Retain the original per-coordinate terminal-KL normalization and the latent-coordinate weighting across storage groups.
4. Do not import the newer canonical/two-horizon CIFAR settings into this FMNIST reproduction. They motivate controls but did not produce the 4.323 result.
5. `rk4_steps=20` is an integration-step setting. Record actual score-network evaluations separately rather than labelling it “20 NFE.”

## 3. Three training regimes

| Public label / output regime | Representation training | Score-prior training |
|---|---|---|
| Co-trained CSEM / `cotrained_csem` | Exact recipe above, 500 epochs | Joint analytic CSEM training |
| Two-stage, denoising target / `indep_tweedie` | Independent VAE, then freeze encoder and decoder | Fresh DiT trained with standard diffusion-noise target |
| Two-stage, CSEM target / `indep_csem` | **The identical frozen VAE checkpoint used by Independent Tweedie** | Fresh identically initialized DiT trained with analytic component target |

### Independent stage 1: a properly regularized VAE

Use the same flat architecture, decoder, image preprocessing, LPIPS/GAN settings, and representation optimizer. Train **500 VAE epochs**, without score-derived gradients or a learned score model:

`L_VAE = L_image + beta0 * KL(q_phi(z0|x) || N(0,I)).`

Compute the exact diagonal-Gaussian **time-zero** KL. Use the coordinate-mean reduction corresponding to the inherited loss scaling, and document the formula. Do not use terminal KL at T=2 as a substitute for ordinary VAE KL, and do not create the baseline solely by setting r=0 or CSEM weight=0 while leaving the old representation regularizer unchanged.

For this architecture-matched comparison, retain the inherited time-conditioned/noisy reconstruction pipeline in **all three regimes**, including stage 1. Only the representation regularizer and score coupling change. Document this explicitly as the matched two-stage VAE/LDM protocol; it is not a claim that every conventional LDM uses this decoder-training procedure. A conventional clean-latent reconstruction protocol can be a separately labelled sensitivity experiment, not a silent difference in the main comparison.

The independent VAE has no CSEM small-time band or CSEM terminal anchor: its regularizer is the stated time-zero KL. Retain common numerical log-variance clamps, but do not impose a new hard posterior-variance floor.

The consolidated report's true-two-stage audit motivates a **beta0 screen over {0, 0.01, 0.03, 0.07, 0.15, 0.30}**. These were CIFAR settings, not certified FMNIST optima. In that audit, beta0=0 was dominated by a small positive KL; using only an unregularized baseline would be weak evidence. Check reduction conventions before transferring these numerical coefficients.

### Independent stage 2: isolate the supervision change

Freeze VAE parameters **and state**, use evaluation mode, and draw fresh posterior samples/noise during prior training. Do not train only on posterior means.

For each image, write the packed flat posterior as diagonal Gaussian with mean mu and variance v. With a=exp(-t), s²=1-exp(-2t):

- Draw z0=mu+sqrt(v)*eta and z_t=a*z0+s*epsilon, with independent standard Gaussian eta and epsilon.
- **Independent Tweedie/DSM:** target is epsilon.
- **Independent CSEM:** target is s*(z_t-a*mu)/(a²*v+s²), coordinatewise.
- Both losses are unweighted epsilon MSE with the same reduction, log-time grid, forward process, class dropout, and DiT architecture.
- Share initial head tensors, data-order streams, and forward draws within each seed pair wherever possible. Use separate named RNG streams so additional teacher operations cannot desynchronize sampling.
- Detach all encoder-derived tensors and verify no VAE gradients or state updates occur.
- Since the posterior is flat/decoupled, the analytic component teacher is exact; a conditional-hierarchy teacher correction is unnecessary.
- Train each prior for **500 epochs**, without additional score-only refinement.

Tune the frozen-prior LR on development seeds. Candidate **3.333333e-4** matches the actual co-trained head optimizer LR; **8e-4** is a historical FMNIST frozen-prior candidate recovered from prior project context, not a confirmed optimum for this larger DiT. Use unit prior-loss coefficient in stage 2, so no hidden 0.6 compensation applies. Preserve AdamW/EMA/cosine settings otherwise.

The main estimator-swap pair must use the **same beta0, VAE checkpoint, LR, schedule, and evaluation**. If target-specific LR tuning gives a materially different result, report those individually tuned endpoints separately; do not attribute that difference entirely to the target.

## 4. Development, fresh seeds, and compute accounting

### Development before the final campaign

1. Use historical seeds 42/43 only for reproduction and baseline selection. Reproduce the seed-42 joint run sufficiently to validate losses, routing, and evaluator; the five-seed historical mean is the realistic reference, not a guaranteed 4.323.
2. Screen the beta0 grid using seed 42 and representation diagnostics. Keep at most two plausible regularized points spanning reconstruction fidelity near the co-trained reference and the nondominated branch. If neither covers a useful reconstruction range, extend the screen before freezing.
3. Train both prior objectives for the retained points and the two candidate LRs on development seeds 42/43. Use staged short runs for screening, but complete the finalists to 500 prior epochs before selecting.
4. Select one common beta0/LR pair by **mean validation generation FID averaged across both prior objectives and development seeds**. Also retain the reconstruction-matched comparison as a secondary diagnostic. Do not pick the baseline solely because its reconstruction is best or because it makes CSEM look strongest.
5. Use a fixed validation split from training data for selection; reserve the standard test reference for final reporting. Historical full-data reproduction is a separate provenance check. Freeze the selected training split/protocol for every final arm, and label any resulting departure from the historical full-data setup.
6. Do not retune CFG, horizons, r, architecture, or training endpoint during the fresh-seed campaign.

If development cost is limiting, first freeze the shared LR at 3.333333e-4 and prioritize the beta0 screen. Keep the 8e-4 branch optional and separately budgeted.

### Final repeated-seed campaign

Use **five fresh paired seeds 47–51** for the main uncertainty report, after verifying they were not used in development. All seeds 42–46 have already been observed in architecture/recipe assessment; preserve their results as historical evidence rather than presenting them as fresh confirmation.

For one selected independent-VAE recipe, the final campaign consists of:

- 5 joint CSEM trainings.
- 5 independent VAE trainings.
- 10 frozen-prior trainings: two objectives branching from each VAE.
- **20 training tasks**, yielding 15 final generative models. A common stage-1 checkpoint is reused for the paired prior objectives.

The independent pipeline receives **500 VAE + 500 prior epochs**, whereas co-training receives 500 joint epochs. This is a generous fully trained baseline comparison, **not an equal-epoch or equal-compute claim**.

Record GPU allocation/device, training seconds, optimizer steps, data passes, evaluation seconds, and end-to-end runtime separately. For each independently usable baseline, its end-to-end cost includes the full VAE cost even when that checkpoint was shared to save campaign resources.

Show generation FID versus cumulative training GPU-hours as the compute comparison; include stage-1 cost before the independent prior curve begins. Also export the familiar prior-training-epoch plot, clearly identifying the extra 500-epoch pretraining cost. Compare quality at common observed compute budgets without extrapolation or treating overlapping confidence bands as a significance test.

Use a timed Vista pilot to set wall limits and SU estimates; do not infer run duration from GAD workloads or equate a Slurm wall limit with actual cost. Print projected total cost before submission.

## 5. Evaluation, uncertainty, and figure specification

### Fixed evaluation contract

- Same 10,000-image reference, generated sample count, preprocessing, legacy torchvision Inception feature extraction, covariance/FID routine, and class-frequency schedule in every primary row.
- Primary: CFG 3, Gaussian start at T=2, temperature 1, 20 RK4 steps, decode time 1e-4, EMA score head plus the corresponding online/frozen VAE.
- Secondary: CFG 1 using the same checkpoints. Do not apply the newer CIFAR CFG-2.5 result to FMNIST.
- Evaluate every 50 joint/prior epochs, with endpoint fixed at 500. No best-epoch or best-seed selection.
- Use fixed evaluation RNG streams across matching arms and checkpoints. Separate them from training RNG state.
- Save posterior-sample reconstruction FID per representation; label it a reconstruction reference, not a mathematically guaranteed FID lower bound. Mean-latent reconstructions, if shown, get their own label.
- Add KID with an explicit subset protocol as a secondary metric if supported by the common evaluator. Do not reuse old inconsistent KID values or confuse KID subset variability with training-seed uncertainty.
- Verify in-process and checkpoint-reloaded evaluation agree on identical saved inputs. The notes record earlier evaluation/checkpoint anomalies, so this is a meaningful preflight check.

### Uncertainty

At each common checkpoint show the mean and a **pointwise 95% Student-t interval across five training seeds**. At the endpoint provide n, mean, SD, and interval. The interval measures training-seed variation under this evaluation contract; it does not remove finite-sample FID bias.

Report seed-paired differences for Independent CSEM minus Independent Tweedie, and Co-trained CSEM minus each baseline, with paired Student-t intervals. Retain seed identities in all exports; require all five prescribed seeds for the final figure. An incomplete live plot must display its observed/expected seed count.

Historical seed 42 is not a sixth fresh replicate. Do not combine multiple CFG settings, checkpoints, or repeated sampling passes as independent training seeds.

### Deliverable figure

Create a compact two-panel main figure:

1. **FID versus prior-training epoch**, three regime curves with 95% bands; for co-training the x-axis is joint epoch. Caption explicitly states the independent VAE pretraining budget.
2. **FID versus end-to-end training GPU-hours**, including independent VAE pretraining cost.

Place reconstruction references in a companion panel/table or as clearly labelled per-regime reference lines. Include representative class-balanced samples and reconstructions from a prespecified seed, not the best-FID seed. Provide a small endpoint table with generation FID, reconstruction FID, KID if available, training cost, and n.

Export PDF/SVG for the manuscript and PNG/SVG for the website. Preserve raw per-seed metrics so figures can be regenerated without retraining.

## 6. Implementation plan for the updated Vista bundle

Proposed name: **`fmnist_flat_csem_vs_independent_vista_v1.zip`**. This is a new bundle name, not an existing artifact.

1. **Freeze provenance.** Vendor the recovered capacity-matched source, manifest cell 2, parameter audit, and completed endpoint readouts. Keep source/config hashes and a concise recipe-diff document.
2. **Retain model code.** Reuse `models.py`, `original_models.py`, pinned architecture code, packing, decoder, and sampler. Preserve the `flat_three_encoder` capacity audit. Add explicit `stochastic_blocks=1` assertions.
3. **Add explicit stages.** Implement `joint`, `vae_pretrain`, `prior_dsm`, and `prior_csem`. Do not repurpose the original short frozen-head refinements as a standard two-stage baseline without changing their initialization, source checkpoint, duration, and optimizer semantics.
4. **Validate gradient paths.** Joint mode must match original loss/gradients for a fixed minibatch; r must scale VAE gradients only. Pretraining must have no learned-score dependence. Frozen modes must leave every VAE tensor unchanged. Analytic teachers must match the flat Gaussian formula.
5. **Handle checkpoints correctly.** Persist optimizers, schedulers, EMA, RNG streams, stage/epoch, source checkpoint hashes, and decoder state. Test interruption/resume and reject incompatible recipes. Add completion records only after the final evaluation is committed.
6. **Use one evaluator.** Extend the inherited evaluator rather than replacing its FID backend. Check CFG-1/null-label semantics, RK4 step counting, EMA handling, sample count, and test/validation separation.
7. **Add manifests and Slurm dependencies.** Development and confirmation manifests are separate. Independent prior jobs depend on their exact VAE checkpoint with successful-stage checks; stage identity and beta0 belong in names. Resume/skipping uses manifest and checkpoint metadata, not directory-name seed inference.
8. **Reuse Vista workflow.** Base launcher on the existing `paper.py`/`job.slurm`: `gh`, one node/job, inherited GPU visibility, existing CUDA-enabled `hlsi` environment. Preserve Torch/torchvision versions; prepare FMNIST, LPIPS and Inception caches before compute jobs. Keep account and environment paths configurable.
9. **Compile robustly.** Export histories, endpoints, paired contrasts, reconstruction references, timing, status, and machine-readable summaries. Never mix validation with test, synthetic smoke metrics with production, or different CFG/solver/sample-count contracts.
10. **Package documentation.** Include dry-run, preflight, development submit, frozen-final submit, resume, strict compilation, and figure-regeneration commands. Return executable commands when the bundle is actually implemented; no submission is part of this planning task.

## 7. Website integration after completion

The supplied website already expects regime keys `cotrained_csem`, `indep_csem`, and `indep_tweedie` in `tools/plot_training_curves.py`. Export its simple CSV view (`regime,seed,epoch,fid`) from the richer metrics table, keeping only the primary evaluation contract.

Before using that plotter for final figures, add duplicate (regime, seed, epoch) rejection, seed-completeness checks, finite-value validation, and explicit compute-axis support. Its current implementation reads every row as a replicate without checking seed identity.

Replace `assets/img/csem_fid_fmnist.png`, the research results table, and homepage FMNIST headline from the **same compiled endpoint summary**. Leave CIFAR numbers separate until an analogous updated comparison exists. Do not place a new 4.3 single-run result beside legacy-architecture 6.84/6.69 baselines as if they form a controlled comparison.

Keep the public caption precise: one diagonal-Gaussian latent representation, matched VAE/DiT capacity, fixed CFG and sampler, independent VAE pretraining cost, five fresh seeds, and pointwise intervals. The target conclusion is a measured comparison of optimized training procedures, with the independent target-swap pair isolating the supervision change.

## Sources inspected

- `fmnist_capacity_matched_4p3_baseline.zip`: README, manifest cell 2, `models.py`, `original_models.py`, `trainer.py`, and pinned loss helpers.
- `fmnist_compiled_results.zip`: `compiled/latest.csv`, `endpoint_arms.csv`, and `report.md`; all 20 historical jobs validated complete.
- Attached `CSEM_consolidated_state_of_knowledge (1)(3).tex`: true two-stage beta0/Pareto audit, representation versus terminal-initialization controls, and evaluator lessons. Most newer numerical operating points concern CIFAR.
- Attached `CSEM_main (1)(2).tex`: three-regime definition and distinction between representation and sampling horizons.
- Attached `alduston.github.io-updated.zip`: plotting interface and website asset path.

