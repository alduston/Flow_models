CSEM PAPER-FINAL V5 — paper figures and CIFAR continuation
=========================================================
Read PAPER_AND_CIFAR_AUDIT_V5.md for evidence, limitations, and plot mappings.
Runner filenames retain _v1 for compatibility; extract to a fresh V5 folder.
No arrays. Every changed run has a new result name and refuses overwrites.

GPU environment: PyTorch/torchvision/CUDA, lpips, numpy, pandas, scipy,
matplotlib, tqdm. Slurm uses $SCRATCH/venvs/hlsi. LPIPS preflight is mandatory.

python verify_paper_v5.py                       # CPU static/fixture checks
python validate_csem_paper_final_v1.py          # real Torch config preflight
python submit_csem_paper_final_v1.py --cells 0-7 --dry-run
python submit_csem_paper_final_v1.py --cells 0-7 # main comparison, both datasets
python submit_csem_paper_final_v1.py --cells 9-21 # all controls/sensitivity

0-1  CIFAR optimized OU-partial + terminal KL; 800 epochs, original800 cosine
2-3  CIFAR independent VAE800 then frozen-VAE priors800; CSEM+Tweedie paired
4-5  FMNIST recovered Aug13 Gaussian terminal-KL joint trainer; 700 epochs
6-7  FMNIST independent VAE700 then frozen-VAE priors700; CSEM+Tweedie paired
8    optional Aug18 shared-sample configuration check; historical4.97 was oracle
9-10 matched per-mean GN/no-anchor CIFAR controls, 800 epochs
11-14 sampling-horizon T_full / KL-weight sensitivity for optimized OU-partial
15   naive unweighted DSM co-training collapse diagnostic, 100epochs
16-17 raw-mean terminal-KL CIFAR controls, 800epochs; match the main tex definition
18-21 raw-mean T_K / KL-weight controls; separate anchor horizon from sampling T

python compile_csem_paper_final_v1.py
python plot_csem_paper_final_v1.py

Default compiler: RK4-20; selects among the EVALUATED CFG candidates on final
co-trained seed-mean FID, then holds CFG fixed for every epoch and all priors.
Candidates: FMNIST3/1.5; CIFAR2.5/3. These are not an exhaustive optimization.
PNG and vector PDF figures land in csem_paper_figures_v1. selected_cfg.json
records selection; figure_coverage.json reports unavailable data/curves.
Original RK4-25 and FMNIST Heun50 metrics are retained in all_eval_records.csv.

To use exactly the old paper's fixed CFG values (FMNIST1.5, CIFAR3):
python compile_csem_paper_final_v1.py --cfg-policy old-paper
python plot_csem_paper_final_v1.py

The new tex's main method explicitly has RAW means. Optimized CIFAR is OU-partial.
For a main table/figure faithful to that raw-method definition (cells16/17):
python compile_csem_paper_final_v1.py --cifar-treatment raw
python plot_csem_paper_final_v1.py
Do not label the default optimized CIFAR treatment as unnormalized raw means.
Use separate base directories if retaining figures from multiple compiler policies.

CONTINUE THE EXISTING CIFAR500 RUN (GPU python in an allocated cluster job):
python continue_cifar_v5.py --source-results /absolute/path/to/cifar_main_csem_oupartial_s43 --seed 43 --allow-weight-only

This uses the NEW seed43 main result name and cannot coexist with a fresh job
for that same cell in the same destination. It continues epochs501-800 along
the existing800 cosine tail, starting VAE LR~7.79e-5 and score LR~3.16e-5.
V3 saves only EMA prior/VAE weights: AdamW moments, online score, discriminator
and RNG cannot be recovered. This is an approximate warm start, prominently
labeled in CONTINUATION_PROVENANCE.txt. Older rows cannot acquire new alternate
CFG/SW2/gap metrics retroactively. Use a fresh0-7 sweep for complete comparable
curves at all epochs. Choose --cfg-policy recipe to retain historical CIFAR2.5
curves when comparing an approximate continuation with the existing run.

V5 modern co-training saves training_state_latest.pt and training_state_best_fid.pt
plus best_fid.json at evaluation checkpoints. To continue an unfinished V5 main
CIFAR run into a fresh V5 destination, use continue_cifar_v5.py without the
--allow-weight-only flag; the full-state path restores online/EMA models, AdamW,
discriminator, cosine phase and RNG. It is a continuation, not score-only refine.

No training or FID reproduction was run here: Torch/CUDA are absent. CPU config,
actual tracking-target, evaluator-adapter, compiler and plot fixture checks pass.
