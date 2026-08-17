FMNIST CSEM WHY-v3 — NumPy 2.x Fix

WHY-v3 is scientifically identical to WHY-v2.  The only training/evaluation
change is a bug fix in the endpoint time-profile aggregation.

BUG IN WHY-v2
-------------
WHY-v2 contained:

    trapz = getattr(np, "trapezoid", np.trapz)

Python evaluates function-call arguments before executing getattr.  Therefore
the default expression was evaluated even when numpy.trapezoid existed.
NumPy 2.x has removed the legacy trapz alias, so endpoint evaluation raised
AttributeError only after the full 120-epoch training run had completed.

The variable was dead code: WHY-v2 already computes physical-time integrals
using explicit trapezoid node widths:

    integral = sum(metric(t_i) * dt_width_i)

WHY-v3 simply removes the obsolete fallback.  No objective, training dynamics,
diagnostic definition, matched pair, or hyperparameter has changed.

WHY THE FAILED V2 RUNS CANNOT BE POSTPROCESSED
-----------------------------------------------
In the training script, final checkpoints are saved after the evaluation loop.
The WHY-v2 exception occurred inside final evaluation, before those saves.
Therefore the failed v2 result directories do not contain a guaranteed final
VAE/score checkpoint suitable for an oracle-only recovery pass.

Use the fresh WHY-v3 namespaces and rerun the affected configurations.

PREVENTING ANOTHER LATE FAILURE
-------------------------------
The v3 Slurm script performs cheap preflights before launching training:

  * imports NumPy and Torch;
  * verifies numpy.trapezoid exists and executes a tiny trapezoid calculation;
  * verifies CUDA visibility;
  * scans the v3 target source and aborts if a legacy np.trapz reference exists.

FILES / NAMESPACES
------------------
csem_split_metric_why_v3.py
fmnist_csem_why_grid_v3.py
fmnist_csem_why_manifest_v3.csv
fmnist_csem_why_pair_runner_v3.py
fmnist_csem_why_pair_job_v3.slurm
submit_fmnist_csem_why_pairs_v3.py
compile_fmnist_csem_why_v3.py
plot_fmnist_csem_why_time_profiles_v3.py

Fresh outputs:
  fmnist_csem_why_results_v3/
  fmnist_csem_why_config_logs_v3/
  fmnist_csem_why_status_v3/
  slurm_logs_csem_why_v3/
  fmnist_csem_why_compiled_v3/

The v1/v2 files and partial v2 result directories are never overwritten.

CANCEL STILL-RUNNING WHY-v2 JOBS
---------------------------------
First inspect:

    squeue -u "$USER" -o "%.18i %.20j %.2t %.10M"

The v2 submitter used job names cwhyT0, cwhyT1, cwhyT2, cwhyT3.
If any are still running/pending, cancel only those:

    squeue -u "$USER" -h -o "%A %j" \
      | awk '$2 ~ /^cwhyT[0-3]$/ {print $1}' \
      | xargs -r scancel

VERIFY V3 MANIFEST
------------------
From:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep/

run:

    "$SCRATCH/venvs/hlsi/bin/python" fmnist_csem_why_grid_v3.py

DRY RUN
-------
    "$SCRATCH/venvs/hlsi/bin/python" \
        submit_fmnist_csem_why_pairs_v3.py --dry-run

SUBMIT
------
All four matched pairs:

    "$SCRATCH/venvs/hlsi/bin/python" \
        submit_fmnist_csem_why_pairs_v3.py --pairs all

Or just pair 0:

    "$SCRATCH/venvs/hlsi/bin/python" \
        submit_fmnist_csem_why_pairs_v3.py --pairs 0

COMPILE
-------
After completion:

    "$SCRATCH/venvs/hlsi/bin/python" compile_fmnist_csem_why_v3.py

The principal time-resolved files remain:

  fmnist_csem_why_compiled_v3/oracle_score_time_profile_pair_wide.csv
  fmnist_csem_why_compiled_v3/oracle_score_time_bin_pair_deltas.csv
  fmnist_csem_why_compiled_v3/oracle_sampling_pair_deltas.csv
  fmnist_csem_why_compiled_v3/pair_endpoint_wide.csv

PLOT
----
    "$SCRATCH/venvs/hlsi/bin/python" \
        plot_fmnist_csem_why_time_profiles_v3.py
