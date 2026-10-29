# Recipe preservation and intentional differences

## Unchanged co-training implementation

The root `models.py`, `original_models.py`, `vae_three.py`, `fmnist_reference.py`
and pinned implementation files are copied verbatim from the supplied
capacity-matched archive. The selected configuration is manifest cell2,
`flat_three_encoder_s42`, changed only as documented below.

One image-conditional diagonal Gaussian over256 coordinates is stored as
64+64+128 coordinates in three deterministic encoder stages. Conditioner widths
are87/75. Posterior parents are deterministic means, not sampled ancestors. The
unchanged multiscale decoder is time-conditioned. `use_norm=False` applies to
the encoder mean; internal GroupNorm remains.

DiT: width192, depth18, six heads, patch1, ordinary epsilon output. Counts are
6,368,817 VAE parameters plus12,159,556 score parameters =18,528,373, excluding D.
Assertions enforce one stochastic block and ordinary epsilon output.

Joint loss is exactly the inherited mean-reduced noisy reconstruction MSE plus
0.85 LPIPS(VGG), 0.0025 GAN, 0.6 epsilon-CSEM, terminal KL weight1, and the resolved
automatic fixed small-time band. Both input and analytic-target VAE gradient
routes use r=.5, giving encoder score coefficient.3; the head coefficient is.6.
The band is not rescaled to.3. DiT optimizer LR is2e-4/.6, VAE LR5e-4.
AdamW(.9,.95), weight decay1e-4, cosine minimum1e-6, combined global clip1,
head-only EMA.9997, label dropout.1, original null label10. GAN starts epoch25,
discriminator LR1e-4. Batch128; two workers;500 epochs; no warmup or refinement.
OU/log-time grid:1000 points,2e-5 to2; fully coupled T_K=T. No new horizons,
factored heads, variance floors, score metrics or CIFAR-specific settings.

## Intentional protocol changes

1. Per plan, new comparison arms share a50k training/10k validation split,
   stratified by class with seed20261028. Only final reporting uses test10k.
   Historical4.323 used all60k training data. The optional historical run restores
   that training split and is excluded from confirmation.
2. Training-seed selection is separated: development42/43; fresh confirmation47--51.
3. Evaluation uses one explicit CPU evaluation RNG across all arms/epochs/seeds,
   instead of the original training-seed-specific CPU/CUDA global evaluation RNG.
   This changes the specific finite evaluation draws while retaining sampler,
   sample count, feature backend, covariance routine and score settings.
4. The new campaign omits expensive witness/finite-bank diagnostic sweeps that
   do not affect training. It adds exact time-zero KL/variance/latent-scale
   diagnostics, stage timing, fixed-checkpoint parity, paired statistics and KID.
5. Full-state checkpoints commit every epoch; immutable model snapshots every50.
   The scheduler, EMA and all RNG streams resume exactly. Uncommitted metric rows
   are removed before resuming an interrupted epoch.
6. Sampling is still the inherited RK4 sampler. Actual score forward calls are
   measured, including the unconditional and conditional CFG calls.

## Independent representation/prior definitions

Stage1 is not `r=0`. It removes the learned score, CSEM band and terminal anchor.
Its loss is the SAME time-conditioned noisy reconstruction image objective plus

    beta0 / (2 B D) * sum_(batch,coordinate) [mu^2 + exp(logvar) - 1 - logvar].

For storage groups of sizes64,64,128, group means are weighted1/4,1/4,1/2. This
is exact time-zero diagonal KL, not terminal KL. Inherited logvar clamps remain;
there is no added hard floor. The same decoder/noisy reconstruction protocol is
used by all arms; it is a matched two-stage protocol, not a claim that every LDM
uses this decoder-training objective.

Stage2 loads only the stage1 VAE, leaves its tensors/state in eval mode, and starts
an independently initialized DiT (never a jointly trained score head). The two
objectives share that VAE checkpoint and initial head tensors. Independent named
RNG streams control data order, time indices, posterior eta, diffusion epsilon,
and label dropout. Encoder parameters/tensors are detached; every frozen VAE
state tensor is hashed and checked at epoch boundaries.

Both objectives use z0=mu+sqrt(v)*eta, zt=a*z0+s*epsilon. DSM target=epsilon;
CSEM target=s*(zt-a*mu)/(a*a*v+s*s). The teacher is exact because the posterior
is decoupled. The unit-coefficient epsilon MSE uses the same mean reduction in
both arms. Both use the same selected LR, AdamW/EMA,500-epoch cosine schedule
to1e-6 and clipping. There is no hidden.6 compensation in either frozen prior.

## Selection and scope of optimality

The six beta points are transferred candidates from the attached CIFAR audit,
not established FMNIST optima. The two LR candidates include the actual joint
head LR and the historical8e-4 frozen-prior candidate. The campaign chooses the
common endpoint mean over both objectives and development seeds, never one
objective's most favorable baseline or a best training epoch.

The automatic reconstruction shortlist is a reproducible diagnostic heuristic,
not a proof of global optimality. Full screen diagnostics, both retained points
and all development outcomes are preserved. If it fails its reconstruction
coverage check, extend the screen in a separate version before confirmation.
Frozen selections reject later changes; there is no test-driven retuning.

The independent target swap isolates supervision. Comparing independent versus
co-trained models compares training procedures and representation regularizers.
It does not identify a single gradient term as the entire causal effect.
