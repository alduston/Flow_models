CSEM PAPER-FINAL COMPARISON SWEEP -- V3
=======================================

Purpose
-------
Generate the paper's matched three-regime comparisons and the remaining scale/
horizon/sampler/collapse ablations using the best-supported dataset-specific
settings rather than forcing one recipe across FMNIST and CIFAR.

IMPORTANT PATH NOTE
-------------------
This bundle is path-independent. The submitter resolves BASE_DIR from the
location of submit_csem_paper_final_v1.py itself. Recommended install:

  /work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/sweeps/csem_paper_final_v3/

but the directory name is not hard-coded.

Main comparison jobs
--------------------
CIFAR:
  cells 0-1  optimized co-trained CSEM, seeds 42/43
  cells 2-3  exact T_K=0 independent VAE with paired CSEM/Tweedie heads, seeds 42/43

FMNIST:
  cells 4-5  recovered Aug-18 co-trained recipe, seeds 42/43, evaluated at old-main CFG 1.5
  cells 6-7  standard beta0=.01 independent VAE with paired CSEM/Tweedie heads, seeds 42/43
  cell 8     exact Aug-18 CFG3 replication/provenance check

CIFAR paper ablations:
  cell 9     historical hard-GN scale anchor
  cell 10    no-anchor failure control (scale explosion expected)
  cells 11-14  T / terminal-KL sensitivity around optimized OU-partial recipe
  cell 15    naive Tweedie co-training collapse control
  cells 16-17 raw-mean + terminal-KL paired controls

The exact settings/provenance and the important distinction between the
Aug-18 oracle-q_T FID 4.97399 and the old-main Gaussian FID ~5.03 are documented
in OPTIMIZED_SETTINGS_PROVENANCE.md.

Validation
----------
From this directory:

  "$SCRATCH/venvs/hlsi/bin/python" validate_csem_paper_final_v1.py
  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py --dry-run

Submit all:

  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py

Submit only main comparison:

  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py --cells main

Submit selected cells, e.g. FMNIST main only:

  "$SCRATCH/venvs/hlsi/bin/python" submit_csem_paper_final_v1.py --cells 4-7

Compile after jobs finish:

  "$SCRATCH/venvs/hlsi/bin/python" compile_csem_paper_final_v1.py
  "$SCRATCH/venvs/hlsi/bin/python" plot_csem_paper_final_v1.py

Primary outputs
---------------
  csem_paper_compiled_v1/main_final_metrics_by_seed.csv
  csem_paper_compiled_v1/main_final_metrics_aggregate.csv
  csem_paper_compiled_v1/main_epoch_curves_long.csv
  csem_paper_compiled_v1/scale_anchor_final.csv
  csem_paper_compiled_v1/terminal_horizon_anchor_sensitivity.csv
  csem_paper_compiled_v1/solver_order_comparison.csv
  csem_paper_compiled_v1/naive_tweedie_collapse_final.csv
  csem_paper_figures_v1/

Main-comparison semantics
-------------------------
Each independent-pair job produces BOTH Independent CSEM and Independent
Tweedie from the same exact standard-VAE representation. The score heads see
identical minibatches/latent samples and are trained over the same epoch clock.
This is preferable to comparing two separately trained VAEs.

Do not interpret the deliberate no-anchor/Tweedie-collapse controls as main
models. See OPTIMIZED_SETTINGS_PROVENANCE.md for expected failure signatures.
