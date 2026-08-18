CIFAR FINE T_K x CSEM SWEEP — FIXED T=1.75
================================================

Scientific grid
---------------
One seed, 18 total configurations:

    T_K in {0.70, 0.85, 1.00, 1.15, 1.30, 1.45}
    csem_w in {0.075, 0.100, 0.150}

Fixed:

    T                    = 1.75
    terminal_kl_w        = 0.30
    score_head_loss_w    = 1.0
    outer weighting      = canonical
    score-head weighting = unweighted-eps
    epochs               = 500
    refinement epochs    = 0
    eval_every           = 50
    eval_samples         = 10000
    CFG                  = 3.0
    canonical_lr_scale   = 1.0
    encoder warmup       = 0
    CSEM ramp            = 0
    extra score tracking = 0
    grad diagnostics     = off
    logvar clamp         = [-30,20]
    fail-on-nonfinite    = off
    bespoke FID          = off

CIFAR-specific preset behavior
------------------------------
    dataset/preset       = CIFAR / auto -> cifar_golden
    VAE base LR          = 2.5e-4
    base LDM LR          = 1.0e-4
    score-head LR        = omitted explicitly, therefore preset 1.0e-4
    architecture         = cifar_golden

Slurm packing
-------------
There are SIX ordinary Slurm jobs, not 18 and not an array.

Each Slurm job fixes one T_K and runs the three csem_w values sequentially:

    group 0: T_K=0.70 -> csem_w=.075, .100, .150
    group 1: T_K=0.85 -> csem_w=.075, .100, .150
    group 2: T_K=1.00 -> csem_w=.075, .100, .150
    group 3: T_K=1.15 -> csem_w=.075, .100, .150
    group 4: T_K=1.30 -> csem_w=.075, .100, .150
    group 5: T_K=1.45 -> csem_w=.075, .100, .150

Each job requests 24:00:00 on gh.

The group runner is restart-aware:
  * completed cells with status rc=0 are skipped;
  * an existing partial/unverified result directory is NOT overwritten;
  * if one cell returns nonzero, the group stops immediately rather than
    burning GPU time on the remaining two configurations.

Evaluation
----------
Each cell keeps the same fast comparison used in the previous T_K sweep:

    reconstruction baseline
    RK4-25 from empirical class-conditional q_TK
    RK4-25 from Gaussian N(0,I) at fixed T=1.75

No Heun evaluation and no refinement phase.

Files
-----
csem_split_metric_TKxCSEM_fine_cifar_v1.py
    Fresh-namespace copy of the corrected T/T_K sweep driver.
    Includes the factored-head graph-sharing fix:
    the detached full-horizon score-head auxiliary loss is recomputed on the
    independent inner forward graph rather than reusing the outer aux_loss_lam.

cifar_TKxCSEM_fine_T1p75_v1_manifest.csv
    All 18 scientific cells and their 6 packing groups.

generate_cifar_TKxCSEM_fine_T1p75_v1.py
    Regenerates the manifest.

run_cifar_TKxCSEM_fine_T1p75_v1_group.py
    Runs one T_K group (three csem_w values sequentially).

cifar_TKxCSEM_fine_T1p75_v1_group_job.slurm
    Generic 24-hour gh job used for each group.

submit_cifar_TKxCSEM_fine_T1p75_v1.py
    Submits 6 ordinary jobs and robustly parses Vista's
    "Submitted batch job <id>" output.

compile_cifar_TKxCSEM_fine_T1p75_v1.py
    Collects all cells, trajectories, status, and endpoint metrics.

plot_cifar_TKxCSEM_fine_T1p75_v1.py
    Plots Gaussian FID, oracle-q_TK FID, reconstruction FID, and the
    Gaussian-minus-oracle gap versus T_K, one curve per csem_w.

Run location
------------
Copy/unzip all bundle files into:

    /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0

1. Verify/regenerate manifest
-----------------------------
"$SCRATCH/venvs/hlsi/bin/python" generate_cifar_TKxCSEM_fine_T1p75_v1.py

2. Dry-run the actual commands for one packed group
---------------------------------------------------
"$SCRATCH/venvs/hlsi/bin/python" run_cifar_TKxCSEM_fine_T1p75_v1_group.py --group-id 0 --dry-run

3. Dry-run all six Slurm submissions
------------------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TKxCSEM_fine_T1p75_v1.py --dry-run

4. Submit all six jobs
----------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TKxCSEM_fine_T1p75_v1.py --groups all

Selective submission / retry
----------------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TKxCSEM_fine_T1p75_v1.py --groups 2
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TKxCSEM_fine_T1p75_v1.py --groups 1,4
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_TKxCSEM_fine_T1p75_v1.py --groups 3-5

Group IDs map to T_K as:
    0 -> .70
    1 -> .85
    2 -> 1.00
    3 -> 1.15
    4 -> 1.30
    5 -> 1.45

Results layout
--------------
cifar_TKxCSEM_fine_T1p75_v1_results/
    TK_0p70/
        csem_0p075/
        csem_0p100/
        csem_0p150/
    TK_0p85/
        ...
    ...
    TK_1p45/
        csem_0p075/
        csem_0p100/
        csem_0p150/

Per-cell logs:
    cifar_TKxCSEM_fine_T1p75_v1_config_logs/

Per-cell JSON status:
    cifar_TKxCSEM_fine_T1p75_v1_status/

Slurm logs:
    slurm_logs_cifar_TKxCSEM_fine_T1p75_v1/

Submission receipts:
    submission_receipts_cifar_TKxCSEM_fine_T1p75_v1/

5. Compile after completion
---------------------------
"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_TKxCSEM_fine_T1p75_v1.py

Primary compiled output
-----------------------
cifar_TKxCSEM_fine_T1p75_v1_compiled/fine_sweep_summary.csv

Also generated:
    all_eval_records.csv
    all_loss_history.csv
    run_status.csv
    missing_or_failed.csv
    best_by_csem.csv
    best_overall.csv

The endpoint summary includes:
    T_K
    csem_w
    reconstruction FID
    oracle-q_TK RK4 FID
    Gaussian-at-T RK4 FID
    Gaussian-minus-oracle FID gap
    terminal KL
    latent RMS
    posterior variance
    score-head MSE
    clipping / optimization diagnostics where available

6. Plot
-------
"$SCRATCH/venvs/hlsi/bin/python" plot_cifar_TKxCSEM_fine_T1p75_v1.py

Plots are written into:
    cifar_TKxCSEM_fine_T1p75_v1_compiled/

Static safety checks
--------------------
The Slurm preflight:
  * forces the hlsi venv Python;
  * checks CUDA visibility;
  * py-compiles the driver and group runner;
  * verifies that the target contains the independent inner factored-head
    auxiliary loss fix;
  * verifies the T/T_K split-routing markers;
  * explicitly rejects the old graph-sharing pattern if found.

No Slurm arrays are used.
