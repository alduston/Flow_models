# FMNIST configuration audit — 9 October 2026

The V3 sweep did not reproduce the recovered Gaussian-start terminal-KL experiment. Its FMNIST recipe was taken from a different, oracle-terminal-initialized canonical experiment. V4 restores the closest verified deployable terminal-KL run and its original joint optimizer implementation. An exact terminal-KL FID of 5.04 was **not** found; no new FID is claimed here.

## Recovered evidence

1. **August 13 Gaussian-start terminal KL.** `provenance/fmnist_aug13_original_log.txt` is the original uploaded log (`Pasted text (2)(20260813-225431).txt`). It explicitly identifies dataset FMNIST, `use_latent_norm=False`, `kl_reg_type=terminal`, `kl_w=1.0`, T=2.0, 700 joint epochs, zero refinement, 10,000 evaluation examples, CFG=3, and the compact 4-channel / VAE-base-32 / DiT-192×8 architecture. At epoch 650 it reports Heun-SDE-50 FID **5.13**, RK4-25 **5.25**. At epoch 700 it reports **5.22 / 5.24**. Its output directory was `Cotraining/fmnist_run3/run_results_fmnist_scale_norm_vs_terminal_kl/run_terminal_kl`, zipped as `run_terminal_kl_20260813_121645.zip`.
2. **Recovered source.** `provenance/csem_new_fmnist_aug13.py` preserves the archived `csem_new (1).py` uploaded August 12. Its CLI, preset, optimizer, metric code, and log messages match this run. The log does not contain a source checksum, so this is a matching recovered source, not a cryptographically verified copy of the cluster file. V4 executes this source's training function for FMNIST main cells. The only runtime substitution is the newer locked dataset loader, which preserves transforms, batch size, shuffle, drop-last, and worker count; it prevents concurrent downloads.
3. **August 18 canonical oracle-start run.** The original bundled source and exact evaluation CSV remain under `provenance/`. At epoch 400 they give reconstruction 4.0832, Heun-50 5.0716, RK4-25 4.9740, with `initoracleqtclass` explicitly in the metric names. These are learned-network reverse samples initialized from class-conditional empirical q_T, not deployment from N(0,I). The August 18 launch log says 10,000 evaluation examples; V3 incorrectly set its purported replication cell to 2,000.
4. **Older paper headline.** Historical paper excerpts report 5.03 at FMNIST epoch 320, CFG1.5, RK4-20. That is a separate experiment predating the August terminal-KL comparison. Its exact terminal-KL provenance was not recovered and must not be used to assign settings to the August run.
5. **Current supplied results.** FMNIST V3 cells 4/5 finish at Gaussian-start RK4-25 FID 11.1615 / 10.4388, reconstruction 5.8971 / 5.8916, and oracle-q_T FID 7.8011 / 7.6924. Poorer reconstruction and oracle-start performance show that the difference is not solely terminal initialization. No controlled experiment here identifies the individual causal contribution of each discrepancy.

## Configuration discrepancies and corrections

| Setting | V3 FMNIST main | Recovered Aug-13 / V4 main |
|---|---|---|
| Representation / full horizon | 1.5 / 1.5 | 2.0 / 2.0 |
| Outer time weighting | canonical physical-time | unweighted epsilon |
| CSEM representation weight | 0.10 | 0.60 |
| Terminal KL weight | 0.30 | 1.00 |
| Joint epochs / cosine LR horizon | 400 / 400 | 700 / 700 |
| Head optimizer initial LR | 8e-4 | 2e-4 / .6 = 3.333333e-4 |
| Head gradient coefficient | independent loss coefficient 1 | joint CSEM coefficient .6 |
| Head time/noise sample | separate full-horizon sample | same sample as representation/reconstruction |
| Gradient routing | replace head gradients using detached inner objective | original joint loss backward |
| Gradient clipping | global VAE + replaced-head norm clip | original global VAE + joint-head norm clip |
| CFG | 1.5 | 3.0 |
| Evaluation count | 5,000 | 10,000 |
| Headline sampler | RK4-25 | RK4-25; historical Heun-SDE-50 also retained |
| Mean GroupNorm / partial anchor | off / off | off / off |

Unchanged shared choices include VAE LR5e-4, GAN/discriminator settings, perceptual coefficient .85, DiT/VAE architecture, log-t grid with t_min2e-5, encoder posterior sampling, EMA .9997, fixed evaluation banks, and decode time1e-4. TF32/high matmul precision is enabled on CUDA to match the recovered source's CLI.

The weights cannot be compared in isolation: time weighting and T changed. The mean contribution to K_T has strength proportional to w_K exp(-2T), which is .014936 for V3 and .018316 for the recovered run. Thus changing w_K from .3 to1 at different horizons is not a factor-of-three change in its effective mean force.

## Additional real bugs fixed

- In the modern exact T_K=0 independent-pair mode, the tracking Tweedie head was trained on representation inputs at t=0 with the representation noise as target. It now uses the detached full-horizon input, time, noise target, and corresponding weights. Previously only the CSEM active head learned from the full-horizon draw.
- Co-training checkpoints never evaluated the tracking head. V4 evaluates its EMA and writes `Ctrl_Diff` records, making Independent Tweedie available to the paper compiler. Both heads have the same configured score LR; the tracking optimizer previously retained preset `lr_ldm` even when `lr_score_head` was overridden.
- Metric lookup now excludes nonfinite entries and Gaussian-T_K columns, requires the requested solver step count, and never silently substitutes a different step count. Combined CSVs contain many all-NaN columns from other configurations; choosing the shortest column name without checking its value could drop valid results.
- The retained canonical August-18 replication cell explicitly reuses the representation time/noise draw for the detached inner regression. It still computes a separate forward graph for its auxiliary loss, avoiding the previously fixed shared-autograd-graph failure. This is a restoration of its sampling contract, not a claim of identical training trajectories.

## What to rerun

From the extracted bundle directory:

```bash
python verify_fmnist_v4.py
python validate_csem_paper_final_v1.py
python submit_csem_paper_final_v1.py --cells 4-7 --dry-run
python submit_csem_paper_final_v1.py --cells 4-7
```

Cells4/5 restore the Gaussian-start terminal-KL recipe for seeds42/43. Cells6/7 compare the repaired detached CSEM/Tweedie heads using one shared standard VAE per seed at T2, 700 epochs, CFG3, and N10000. Their beta0=.01 and head LR8e-4 are retained from the FMNIST baseline/score-head lineage; they are not claimed to be a newly optimized global optimum. Cell8 remains an optional canonical/oracle provenance control, excluded from the headline table.

Independent CIFAR cells2/3 also require rerunning to obtain corrected Tweedie training/evaluation. The CIFAR main co-training recipe is unchanged. New result names prevent V3 completed statuses from suppressing changed jobs. Keep existing outputs for comparison; do not resume these changed configurations from V3 checkpoints. There are no Slurm arrays.

After completion:

```bash
python compile_csem_paper_final_v1.py
python plot_csem_paper_final_v1.py
```

## Validation and limits

- All Python files compile; Slurm shell syntax and selected submission dry-run pass.
- All18 resolved manifest configurations pass checks.
- The recovered source CLI was executed with training intercepted and its complete terminal-arm config captured in `provenance/fmnist_aug13_resolved_source_configs.json`. Every original terminal-arm config value except output paths matches V4 cell4; the source optimizer is used directly.
- CPU regression checks execute the actual tracking preparation/loss block with a nonzero head-time sample and zero-time representation sample; the resulting target and loss verify the corrected full-horizon training input.
- Compiler checks cover all-NaN cross-configuration columns, oracle/T_K exclusion, exact solver selection, and both independent regimes. An end-to-end synthetic compile verifies all three FMNIST regimes.
- This workspace has no Torch/CUDA. No training run or fresh FID reproduction was performed. FID5.04, the new seed43 outcome, and the reproduced absolute scores remain unverified. The supplied paper placeholders are not filled with historical values.
