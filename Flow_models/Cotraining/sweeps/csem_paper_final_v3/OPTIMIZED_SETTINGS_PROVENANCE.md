# CSEM paper-final V3: optimized-settings provenance

## Why V3 exists
V1/V2 accidentally mixed the most recent CIFAR recipe with a non-optimal FMNIST score-head configuration and an obsolete freeze-then-refine implementation of the independent baselines. V3 resolves those issues by using dataset-specific settings supported by the most recent completed sweeps.

## Located FMNIST ~FID-5 run
The latest exact FMNIST artifact located in the August lineage is the Aug-18 `canonical_oracleqt_v1` run. The included provenance files are exact snapshots of its source/evaluation/loss outputs.

At epoch 400 its reported metrics are:
- VAE reconstruction FID: 4.08320453
- Heun-50, CFG 3, oracle-q_T-class init FID: 5.07160008
- RK4-25, CFG 3, oracle-q_T-class init FID: 4.97398564

Important: **4.97399 is an oracle-q_T-class initialization result, not Gaussian-start FID.** The old CSEM-main headline of ~5.03 was a separate Gaussian-start, CFG-1.5 comparison. V3 therefore does two things: (1) cell 8 reproduces the Aug-18 evaluation contract at CFG 3 as a provenance check; (2) main FMNIST cells 4-5 use the same recovered training recipe but the old-main comparison CFG 1.5, so the paper comparison remains deployable and directly comparable across regimes.

### Recovered training settings
- dataset/preset: FMNIST / `fmnist_reference`
- architecture: latent channels 4; VAE base channels 32; DiT 192 x 8, 6 heads, patch 1; batch 128
- horizon: T_K = T = 1.5; log-t OU; t_min = 2e-5
- joint epochs: 400; no score-only refinement
- representation time weighting: canonical
- score-head time weighting: unweighted-eps
- CSEM representation coefficient: w_C = 0.10
- terminal component KL coefficient: w_K = 0.30
- score-head loss coefficient: 1.0
- score-head LR: 8e-4
- VAE LR: 5e-4 with cosine schedule over the run
- perceptual weight: 0.85
- PatchGAN generator weight: 0.0025; discriminator starts epoch 25; discriminator LR 1e-4
- no encoder-mean GroupNorm; no OU-partial anchor; terminal KL is the scale anchor
- fixed eval banks

The terminal coefficient w_K=.30 is recoverable directly from the stored first-epoch objective: `loss - recon - .85*perc - .10*score_lsi = .30*terminal_kl` to floating-point error.

## Optimized CIFAR co-training
The Aug-29 state-of-knowledge recommendation is used unchanged:
- T_K=1.05, T=1.35 (detached tail .30)
- w_C=.05, w_K=.60
- OU-partial GroupNorm at T_K, rho_m=alpha_K^2 and rho_v=alpha_K^4
- canonical representation weighting; unweighted-eps score-head weighting
- 500 joint epochs, no refinement
- score-head LR 1e-4
- CFG 2.5; RK4-25 = 100 NFE headline evaluation
- seeds 42 and 43

## Independent baseline protocol
V3 no longer uses the V1/V2 `VAE stage + equally long score-only refine` workaround. It uses the exact T_K=0 standard-LDM path already present in the modern core:
- no diffusion/CSEM gradient reaches the VAE
- K_0 is ordinary VAE KL
- the analytic-CSEM and Tweedie heads are trained side-by-side on detached latents from the **same VAE and same minibatches** during the same epoch clock
- no score-only refinement phase

This is the protocol used by the later true two-stage Pareto audit and gives a cleaner estimator comparison.

### CIFAR independent VAE
The modern two-seed Pareto sweep tested beta0={.01,.03,.07,.15,.30}; beta0=.07 gave the best mean Gaussian-start FID among those regularized two-stage cells (12.40, essentially tied with .01/.15) while remaining a well-regularized interior point. V3 uses beta0=.07, 500 epochs, T=1.35, score LR 1e-4, CFG2.5.

### FMNIST independent VAE
The historical FMNIST comparison code used standard q0 KL beta0=.01. V3 keeps beta0=.01 but updates the score-head optimization to the later FMNIST-supported LR 8e-4, trains both heads simultaneously on the same detached representation, uses 400 epochs and evaluates at CFG1.5.

## Intentional non-optimized controls
Cells `anchor_none` and `naive_tweedie_cotrain` are intentionally pathological ablations. A scale explosion in `anchor_none` or collapse in `naive_tweedie_cotrain` is expected and must not be interpreted as a main-method failure.
