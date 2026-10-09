# V4 dataset settings

See `FMNIST_CONFIGURATION_AUDIT.md` for the source evidence, settings comparison, fixes, and limitations.

FMNIST main cells4/5 now execute the recovered August13 Gaussian-start terminal-KL training implementation at T2, w_C=.6 (unweighted epsilon), w_K=1, 700 epochs, head LR2e-4/.6 with joint coefficient.6, CFG3, N10000, RK4-25. The recovered log reaches5.13 Heun and5.24 final RK4. An exact terminal-KL FID5.04 is not certified.

The Aug18 canonical .1/.3 T1.5 run produced oracle-q_T-class FID4.97; it was the wrong source for a deployable ~5 Gaussian-start recipe. It is retained only as optional replication cell8, with shared samples and N10000 restored.

CIFAR main remains the Aug29 recipe: T_K1.05/T1.35, canonical w_C=.05, w_K=.60, OU-partial anchor, 500 epochs, score LR1e-4, CFG2.5, RK4-25. Independent baselines retain beta0=.07 CIFAR/.01 FMNIST. Their tracking head uses the full-horizon input, receives the same configured LR as its peer, and is evaluated at every checkpoint.

Historical evidence is included verbatim under `provenance/`; these results are not fresh V4 measurements.
