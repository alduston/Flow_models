# Source lineage and V3 decisions

The runnable core descends from the late-August two-horizon CSEM code and retains the V2 dataset-lock, OU-partial anchor, solver-suite, and oracle-loader fixes.

## Changes from V2

1. **FMNIST co-trained main settings were corrected.** V2 had `w_K=.60` and score-head LR `2e-4`, neither of which matches the located Aug-18 ~FID-5 run. V3 uses `w_C=.10`, `w_K=.30`, canonical outer weighting, unweighted-eps head, score-head LR `8e-4`, T=1.5, 400 joint epochs, no refinement.
2. **Independent baselines now use the exact T_K=0 modern two-stage path.** There is no diffusion-derived VAE gradient; q0 KL is the VAE regularizer; both CSEM and Tweedie heads train on the same detached latent stream during the same epoch clock. The obsolete equal-length freeze-then-refine stage is removed.
3. **Dataset-specific two-stage KL is used.** CIFAR beta0=.07 from the modern two-stage Pareto sweep; FMNIST beta0=.01 from the historical independent-VAE comparison lineage.
4. **Dataset-specific score-head LR is used.** CIFAR 1e-4; FMNIST 8e-4.
5. **FMNIST paper comparison returns to CFG1.5**, matching the old CSEM-main comparison. A separate provenance cell evaluates the recovered Aug-18 model contract at CFG3.
6. CIFAR main remains the Aug-29 optimized OU-partial+terminal-KL recipe: `(T_K,T,w_C,w_K)=(1.05,1.35,.05,.60)`, CFG2.5, 100 NFE.

## What the located FID ~5 actually was

The exact Aug-18 `eval_metrics.csv` reaches RK4-25 FID `4.973985635966699` at epoch 400, but the metric key is `...cfg3_0_initoracleqtclass`: this is oracle-q_T-class initialization. It is not the same quantity as the old CSEM-main Gaussian-start headline FID 5.03 at CFG1.5. V3 preserves that distinction rather than silently conflating the two.

Exact source/eval/loss snapshots are bundled under `provenance/`.
