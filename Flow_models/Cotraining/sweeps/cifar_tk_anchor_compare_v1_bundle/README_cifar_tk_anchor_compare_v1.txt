CIFAR CSEM T_K-aware GroupNorm / terminal-KL comparison v1
=============================================================

PURPOSE
-------
Compare the historical encoder-mean GroupNorm scale anchor against the current
terminal component KL and two coherent T_K-aware continuations of the old GN idea.
The experiment is an 8-arm anchor x KL design with 3 fresh seeds per arm = 24 runs.

FIXED SCIENTIFIC CONTRACT
-------------------------
Dataset/model: CIFAR / current cifar_golden architecture
T_K=1.05, T=1.35, DeltaT=.30
w_CSEM=.05
canonical representation route; unweighted-eps score-head route; score-head loss w=1
500 joint epochs; 0 score-only refinement
CFG=2.5
RK4 steps=25 = NFE 100
Gaussian-start temperature=1.0 (ordinary sampler; no temperature sweep)
10,000 eval samples
seeds 42,43,44
No checkpoint reload; each cell trains from scratch.

THE 8 ARMS
----------
A_current_kl:                 no GN-like anchor + terminal K_T KL w=.60
B_unanchored:                 no GN-like anchor + no terminal KL
C_historical_gn0:             historical hard GN(mu_0) + no terminal KL
D_historical_gn0_plus_kl:     historical hard GN(mu_0) + terminal KL w=.60
E_ou_visible_tk:              OU-visible GN penalty at T_K + no terminal KL
F_ou_visible_tk_plus_kl:      OU-visible GN penalty at T_K + terminal KL w=.60
G_ou_partial_tk:              OU-partial-GN layer at T_K + no terminal KL
H_ou_partial_tk_plus_kl:      OU-partial-GN layer at T_K + terminal KL w=.60

ANCHOR DEFINITIONS
------------------
Historical GN0:
  exactly the existing one-group, affine=False GroupNorm applied to encoder mu.

OU-visible GN penalty:
  Per example, let m and v be the biased empirical mean/variance across the latent
  tensor. With alpha=exp(-T_K), sigma^2=1-alpha^2,
      m_K = alpha m,
      v_K = sigma^2 + alpha^2 v.
  Add
      0.5 * mean_x [ m_K^2 + v_K - log(v_K) - 1 ].
  Its zero set at T_K=0 is m=0,v=1 (the historical GN moment manifold), and it
  vanishes as T_K -> infinity. The coefficient is FIXED AT 1.0 in this sweep and
  is recorded explicitly as ou_visible_anchor_w. This is a soft anchor; its scalar
  strength is not claimed to be calibrated to the hard GN layer.

OU-partial-GN layer:
  Let rho_m=alpha^2 and rho_v=alpha^4. It partially corrects the original mean and
  variance defect by these visibility factors. The implementation is constructed
  to equal historical GN0 at T_K=0 (same eps convention) and become exactly the
  identity as T_K -> infinity. No additional loss coefficient is introduced.

IMPORTANT INTERPRETATION
------------------------
The primary clean comparisons are paired by seed:
  A-B: terminal KL contribution without anchor
  D-C: terminal KL contribution conditional on GN0
  F-E: terminal KL contribution conditional on OU-visible anchor
  H-G: terminal KL contribution conditional on OU-partial anchor
  C-B / E-B / G-B: each anchor contribution without terminal KL

Do not over-interpret the OU-visible arm's absolute ranking as a pure location
comparison against hard GN: it is deliberately a soft family with fixed coefficient 1.
The OU-partial arm is the stronger apples-to-apples temporal continuation of the
historical hard layer.

RUN ON THE CLUSTER
------------------
Copy all bundle files into, e.g.:
  /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/tk_anchor_sweep/

Then:
  python submit_cifar_tk_anchor_compare_v1.py --dry-run
  python submit_cifar_tk_anchor_compare_v1.py

Subset examples:
  python submit_cifar_tk_anchor_compare_v1.py --cells 0-2 --dry-run   # arm A, 3 seeds
  python submit_cifar_tk_anchor_compare_v1.py --cells 0-5             # arms A+B

Compile partial or complete results:
  python compile_cifar_tk_anchor_compare_v1.py

SLURM NOTE
----------
The job file intentionally uses partition gh with NO #SBATCH --gpus or --gres line,
matching the known-good cluster job pattern. It verifies CUDA visibility at runtime.

OUTPUTS
-------
cifar_tk_anchor_compare_v1_compiled/
  final_eval_by_seed.csv
  final_training_by_seed.csv
  arm_seed_aggregates.csv
  training_seed_aggregates.csv
  paired_fid_contrasts.csv
  missing_cells.csv
