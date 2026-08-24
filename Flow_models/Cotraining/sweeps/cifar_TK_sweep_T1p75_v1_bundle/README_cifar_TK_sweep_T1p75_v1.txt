CIFAR CSEM T_K SWEEP — FIXED T=1.75
=======================================

Scientific question
-------------------
Hold the full score-training / Gaussian sampling horizon fixed at

    T = 1.75

and sweep the representation/KL horizon

    T_K in {0, .25, .5, .75, 1.0, 1.25, 1.5, 1.75}.

This directly tests the interpretation of T_K as the amount of the diffusion
path allowed to shape the latent representation while T controls the full
score model and Gaussianization/reversal horizon.

Fixed training setup
--------------------
Matches the previous CIFAR T_K=1.5, T=2.0 run except for the requested
T=1.75 and training budget.

  dataset/preset              CIFAR / auto -> cifar_golden
  outer representation metric canonical
  score-head metric           unweighted-eps
  csem_w                      0.10
  terminal_kl_w               0.30
  score_head_loss_w           1.0
  score-head LR               omitted -> CIFAR preset base LDM LR = 1e-4
  canonical_lr_scale          1.0
  cotrain                     500 epochs
  score-only refinement       100 epochs
  eval_every                  50
  eval_samples                10000
  CFG                         3.0
  refinement LR               CIFAR preset 1.5e-5
  CIFAR VAE LR                2.5e-4
  CIFAR LR scheduler horizon  historical preset 800 epochs
  bespoke FID classifier      OFF
  fail-on-nonfinite           OFF
  logvar clamp                [-30,20]

Evaluation
----------
At every evaluation the script retains the fast two-mode comparison:

  1. learned-score RK4-25 initialized from empirical class-conditional q_TK
  2. learned-score RK4-25 initialized from N(0,I) at fixed T=1.75

plus the reconstruction baseline.

For T_K=0, reversal from q_TK=q_0 to q_0 has zero path length, so the first
control is implemented as direct aggregate-q_0 decoding.  The Gaussian-at-T
sampler is unchanged.

Exact T_K=0 semantics
---------------------
The updated training driver supports zero exactly.

At T_K=0:
  * no diffusion-derived CSEM gradient reaches the encoder;
  * reconstruction is performed at z_0;
  * K_0 is the ordinary componentwise VAE KL
        E_x KL(q_phi(z_0|x) || N(0,I));
  * the score network still trains throughout [t_min,1.75] on detached latents;
  * the 100-epoch refinement is likewise score-only on the full horizon.

Thus this cell is the two-stage-LDM optimization endpoint, not an approximation
using T_K=t_min.

Files
-----
csem_split_metric_TK_sweep_cifar_v1.py
    Updated two-horizon driver with exact T_K=0 support.

cifar_TK_sweep_T1p75_v1_manifest.csv
generate_cifar_TK_sweep_T1p75_v1.py
run_cifar_TK_sweep_T1p75_v1_cell.py
cifar_TK_sweep_T1p75_v1_cell_job.slurm
submit_cifar_TK_sweep_T1p75_v1.py
compile_cifar_TK_sweep_T1p75_v1.py
README_cifar_TK_sweep_T1p75_v1.txt

Run location
------------
Copy all bundle files to:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0

Generate / verify manifest
--------------------------
"$SCRATCH/venvs/hlsi/bin/python" generate_cifar_TK_sweep_T1p75_v1.py

Dry-run submission
------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TK_sweep_T1p75_v1.py --dry-run

Submit all eight ordinary jobs
------------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TK_sweep_T1p75_v1.py --cells all

No Slurm arrays are used.

Selective resubmission examples
-------------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TK_sweep_T1p75_v1.py --cells 0
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TK_sweep_T1p75_v1.py --cells 3-5
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TK_sweep_T1p75_v1.py --cells 1,7

Compile when complete
---------------------
"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_TK_sweep_T1p75_v1.py

Primary output
--------------
cifar_TK_sweep_T1p75_v1_compiled/TK_sweep_summary.csv

Also useful:
cifar_TK_sweep_T1p75_v1_compiled/all_eval_records.csv
cifar_TK_sweep_T1p75_v1_compiled/all_loss_history.csv
cifar_TK_sweep_T1p75_v1_compiled/cotrain_endpoint_eval_epoch500.csv
cifar_TK_sweep_T1p75_v1_compiled/cotrain_endpoint_loss_epoch500.csv
cifar_TK_sweep_T1p75_v1_compiled/final_eval_epoch600.csv
cifar_TK_sweep_T1p75_v1_compiled/final_loss_epoch600.csv
cifar_TK_sweep_T1p75_v1_compiled/run_status.csv
cifar_TK_sweep_T1p75_v1_compiled/missing_or_failed.csv

Results are stored independently in
-----------------------------------
cifar_TK_sweep_T1p75_v1_results/TK_0p00/
cifar_TK_sweep_T1p75_v1_results/TK_0p25/
...
cifar_TK_sweep_T1p75_v1_results/TK_1p75/

Each runner refuses to overwrite an existing partial result directory.
