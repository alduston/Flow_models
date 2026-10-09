# Paper coverage and CIFAR schedule audit

V5 addresses the two follow-up requests. It retains the recovered August13
FMNIST joint-training objective and source unchanged, and adds paper evaluation.
The exact recalled terminal-KL5.04 remains unverified; recovered Gaussian metrics
are Heun5.13 at650 and RK4-25 5.24 at700. No new FID is asserted.

## Old-paper comparison

The attached PDF page10 specifies a VAE-only stage followed by priors trained on
one frozen VAE, and RK4-20 epoch curves at a common dataset-level CFG. V4 instead
trained detached priors while the independent VAE was changing; that protocol
was not the same experiment. V5 freezes score heads during VAE training, then
freezes the VAE during paired CSEM/Tweedie prior training. Both heads use the
same latent draws, diffusion times, noise, LR, epoch budget and evaluation banks.
The x-axis is prior-training epoch (VAE pretraining is separately costed):
CIFAR800+800 and FMNIST700+700, vs800/700 joint epochs for co-training.
This matches the old paper's score-training clock, not equal total compute.

Every main run records Gaussian-start RK4-20/25 at the recovered recipe CFG and
the old paper CFG. Final co-trained seed-mean RK4-20 FID chooses the reporting
CFG from that TWO-VALUE grid. The same choice is applied to every epoch and
both baselines. `--cfg-policy old-paper` fixes1.5/3; `recipe` fixes3/2.5.
These are best-supported recipes/candidate selection, not proven global optima.

The historical FMNIST evaluator runs unchanged first. Extra paper evaluations
are generated from a checked AST adaptation of that evaluator and execute
inside saved/restored Python/NumPy/Torch/CUDA RNG state. Original training
source, objective, clipping, optimizer, and original metric values are preserved.
The original provenance source bytes are unchanged. The additional evaluation
cost is real; use the 48h Slurm allocation and inspect runtime on your GPU.

## New-paper plot coverage

| Paper item | Experiment/compiled data | Plot artifact stem (PNG and PDF) |
|---|---|---|
| Six main rows and estimator comparison | 0-7, main_final_metrics_by_seed/aggregate.csv | main_{fmnist,cifar}_{fid,kid,sw2,csem_gap}_vs_epoch |
| Old paper Figure1 layout | same main curves, exact RK4-20 | main_old_paper_epoch_comparison |
| Latent distance diagnostic | separate paper_latent_sw2 column | main_{dataset}_latent_sw2_vs_epoch |
| Controlled scale-anchor table | 9,10,16, scale_anchor_final.csv | scale_anchor_dynamics and scale_anchor_{metric} |
| Optimized sampling T / lambda_K | 0,11-14, terminal_horizon_anchor_sensitivity.csv | sensitivity_T, sensitivity_lambdaK |
| Raw terminal T_K / lambda_K | 16,18-21, raw_anchor_sensitivity.csv | raw_sensitivity_TK, raw_sensitivity_lambdaK |
| Solver table FID/KID | main CSEM / raw / naive cells, solver_order_comparison.csv | solver_{mode}_{dataset} |
| Naive DSM generation table |15, naive_tweedie_collapse_final.csv + solver table | solver_naive_tweedie_cotrain_cifar |
| Posterior mean and variance collapse |15, all_loss_history.csv | tweedie_collapse_mean_variance, tweedie_collapse_{posterior_var,posterior_std,latent_rms} |

Main CSVs include reconstruction FID/KID/feature-SW2 references. Main lines are
solid for co-trained CSEM and dashed for independent heads, matching the old
PDF. Bands are across-seed SD only when there are multiple seeds. Missing data
are explicitly reported, never filled with invented observations. All control
panels include latent scale, terminal KL, reconstruction and score diagnostics.

Two metric bugs are corrected: legacy `lsi_gap_unet_uncond` sums epsilon-space
errors without dividing by sigma^2, so it is not the paper's score-unit gap.
V5 retains it and adds `csem_gap_score_uncond` with the actual conversion.
The legacy SW2 was in latent coordinates. V5 separately evaluates real/generated
FID-feature SW2 (square root of mean projected squared distance) using256 fixed
unit projections. The preserved legacy latent SW2 column is squared distance
and its plot is labeled SW2².
The component-gap time grid is the configured diffusion grid; it is unconditional
(no CFG transformation), held-out, deterministic per seed, and sums latent
coordinates. It is not a learned-vs-oracle aggregate-score MSE.

Heun-SDE20 and RK4-ODE20 share checkpoint, initial-noise/label bank and step
budget, but require nominally40 vs80 score calls; this is not an equal-NFE
comparison. Stochastic Heun paths also differ from deterministic RK4 trajectories.
The naive DSM control keeps reconstruction, decoder, adversarial, AdamW and EMA
settings; the intentional unweighted epsilon loss coefficient1 and absent anchor
are recorded, not described as the same CSEM objective.

The optimized CIFAR recipe applies OU-partial normalization plus terminal KL.
The new tex explicitly defines its final method as raw encoder means. V5 keeps
these scientifically distinct: default optimized main curves use0/1, whereas
`--cifar-treatment raw` uses16/17. Central scale-ablation panels ONLY use raw-KL,
mean-GN/no-KL and raw/no-KL, all with the same800-epoch budgets and other settings.
OU-partial is not silently pooled into this controlled three-treatment table.
Likewise T_full and the terminal boundary T_K are separately swept/labeled;
the lambda_K*alpha(T_K)^2 force concerns the latter, not fixed-T_K sampling tails.

## Is CIFAR bottomed out?

Only seed43 completed the attached main CIFAR run. Seed42 failed dataset access.
The ten-thousand-sample, Gaussian-start, CFG2.5 observations are:

| Epoch | RK4-25 FID | RK4-20 FID | RK4-25 KID | Recon FID |
|---:|---:|---:|---:|---:|
|100|15.8087|15.8159|.004859|7.7439|
|200|11.4488|11.4558|.002973|6.6995|
|300|10.2863|10.2918|.002429|5.9703|
|400|9.5542|9.5587|.002086|5.5509|
|500|9.1105|9.1130|.001739|5.2461|

The latest100 epochs improve FID by0.444 (4.64%), KID by about16.6%, and the VAE
floor by0.305. There is diminishing return but no measured plateau. A3.86 FID
gap to reconstruction remains. RK4-20 and25 are nearly identical: more RK4 steps
are not the evident bottleneck. Heun20 at500 is9.8154, so solver choice matters.
There is one completed seed and five checkpoints; no statistical plateau test
or globally optimal epoch/LR claim is justified.

The manifest stopped at500, but its actual cosine scheduler horizon was800.
Ending there abandoned the last300 annealing epochs. V5 extends CIFAR main and
matched ablations to800, retains exactly that horizon and base LRs, and evaluates
every50 epochs rather than100.
There is no premature LR restart, no second arbitrary decay layered on the cosine,
and no automatic1000-epoch extension that would change the early trajectory.

| Completed epoch / next epoch LR | VAE | Score |
|---:|---:|---:|
|500|7.786e-5|3.156e-5|
|600|3.747e-5|1.550e-5|
|640|2.478e-5|1.045e-5|
|700|1.048e-5|4.768e-6|
|750|3.392e-6|1.951e-6|
|800|1e-6|1e-6|

A fit a+b/epoch to300/400/500 extrapolates8.45 at800; three-point fits are fragile
and ignore the impending LR floor. Roughly8.5-9 is a plausible target range,
not a forecast interval or a result. Best budget decision now: complete800 and
retain the best measured checkpoint. If750/800 still consistently improve by
~0.1FID per50 epochs across seeds, consider a separate200-epoch, low-LR experiment
(VAE1e-5, score5e-6, decay to1e-6) to1000. That restart is deliberately NOT enabled
by default before seeing the800-epoch endpoints. Do not extrapolate cosine past
its800 minimum: the native cosine would rise again.

## Continuation and verification

`continue_cifar_v5.py` permits a completed V3 epoch500 weight-only warm start with
an explicit flag; the original Adam moments, online prior, discriminator and RNG
are unrecoverable. It uses the original cosine phase, not a new300-epoch schedule,
and labels that limitation. Fresh runs remain preferable for clean full-history
alternate-CFG and new-metric curves. V5 modern jobs additionally retain complete
training state and best recipe-CFG Gaussian RK4-20 state; unfinished main CIFAR
jobs can restore those states into a fresh output directory with the helper.
This checkpoint addition does not apply to the untouched historical FMNIST engine
or frozen-VAE refinement stage.

CPU checks:22 resolved configurations; actual detached tracking-target routing;
legacy evaluation adapter compilation; compiler strict Gaussian/CFG/step filters;
six main regimes/metadata; correct score/feature/latent metric fields; central
three-arm ablation; raw sensitivity5points; all plots in PNG/PDF; raw-method and
old-paper compiler switches. Fixtures are synthetic and deleted. Source syntax,
shell syntax, archive CRC, provenance-source hash and bundle checksums are checked.
Torch/CUDA are unavailable here, so no GPU training, checkpoint round trip, or
new FID reproduction was possible. V3/V4 audit files document previous bundle
states; this V5 audit/README supersede their plot/baseline/schedule instructions.
