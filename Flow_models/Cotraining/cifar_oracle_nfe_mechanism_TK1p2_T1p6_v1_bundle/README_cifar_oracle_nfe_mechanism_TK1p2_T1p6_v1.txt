CIFAR ORACLE-SCORE NFE MECHANISM SWEEP — FROZEN CHECKPOINTS
============================================================

Scientific question
-------------------
Does CSEM improve generation primarily because it makes the learned score easier
to estimate, or because it changes the latent representation so that the *exact*
reverse score field itself has easier finite-NFE dynamics?

This sweep is evaluation-only.  It reloads checkpoints from the two completed
weight sweeps and performs NO additional training.  Consequently every NFE point
for a representation uses exactly the same frozen VAE and learned score network.

20-cell design
--------------
Five frozen representations x four RK4 budgets = 20 cells.

Representation axis:

  rep_id                 csem_w   terminal_kl_w   purpose
  --------------------   ------   -------------   -------------------------------
  two_stage_c0_k40        0.00         0.40       no-CSEM / two-stage-like control
  weak_c2_k40             0.02         0.40       weak CSEM, same KL
  sweet_c5_k40            0.05         0.40       near-optimal CSEM, same KL
  strong_c8_k40           0.08         0.40       strong score-friendly CSEM, same KL
  highkl_c5_k80           0.05         0.80       same CSEM, stronger Gaussianization

NFE axis:

  RK4 steps   score-network evaluations (NFE)
       5                    20
      10                    40
      25                   100
      50                   200

Every representation was trained at:

  T_K = 1.2
  T   = 1.6
  canonical outer representation weighting
  unweighted-epsilon score-head weighting
  CIFAR golden architecture/preset

Frozen source checkpoints
-------------------------
The submitter checks these before submitting any jobs.  Required source roots are:

  cifar_weight_sweep_TK1p2_T1p6_v1_results/
  cifar_highkl_weight_sweep_TK1p2_T1p6_v2_results/

Within each selected result it requires:

  run_terminal_kl/checkpoints/vae_cotrained.pt
  run_terminal_kl/checkpoints/unet_lsi.pt

If any required source checkpoint is missing, submission aborts before the first
sbatch call.  The evaluator also independently refuses to fall back to random
weights.

Mechanism comparison inside every cell
--------------------------------------
For BOTH h=T_K=1.2 and h=T=1.6, using the same paired latent/noise banks:

  oracle + q_h
      exact empirical class-conditional q_h initialization and exact empirical
      class-conditional score.  This is the clean finite-NFE dynamics channel.

  learned + q_h
      same exact q_h initialization, learned conditional score.  Difference from
      oracle+q_h isolates score-model approximation propagated through sampling.

  oracle + Gaussian
      Gaussian initialization, exact conditional score.  Difference from
      oracle+q_h isolates terminal initialization mismatch.

  learned + Gaussian
      Gaussian initialization and learned conditional score.  Both practical
      errors are present.

  direct q_0
      direct empirical posterior sample decoded without reverse integration;
      this is the representation/decoder reference floor.

The oracle sampling curve deliberately uses the exact class-conditional score
WITHOUT CFG.  This keeps the oracle field a genuine score field rather than the
CFG-modified vector field.  The selected labels are drawn together with the
empirical posterior components, so their mixture remains the dataset mixture.

Oracle implementation note
--------------------------
For class-conditional oracle calls, the driver now evaluates only reference
components from the requested class before forming likelihood matrices.  This is
algebraically identical to the previous full-N-then-mask implementation but is
~10x cheaper on balanced CIFAR.  A numerical equivalence unit test was run when
the bundle was built (max absolute discrepancy ~2.4e-7 on a random toy mixture).

Optimal-field time profile
--------------------------
At 16 coupled OU times using 64 query components, every cell also records:

  * learned-vs-exact score error, unconditional and conditional;
  * intrinsic CSEM component variance;
  * exact conditional and unconditional score RMS;
  * exact probability-flow ODE drift RMS;
  * conditional-vs-unconditional exact-score separation;
  * adjacent-time exact score variation along coupled forward OU paths;
  * adjacent-time exact drift variation;
  * adjacent-time drift-direction cosine.

These diagnostics require essentially no additional oracle evaluations beyond the
existing learned-vs-oracle profile because the exact scores are already present.
They help explain *why* one representation may need fewer NFEs.

Evaluation sample sizes
-----------------------
  ordinary real-feature bank:       1000 test examples
  oracle-NFE sampling diagnostic:    256 paired samples
  oracle time-profile queries:        64 samples x 16 times

With 256 samples, image FID/KID are useful as paired directional diagnostics but
should not be read to hundredths.  Latent SW2 and the shape of the NFE curves are
the cleaner mechanism measurements.

Files
-----
  csem_oracle_nfe_mechanism_TK1p2_T1p6_v1.py
      Evaluation-enabled CSEM driver.  Loads frozen checkpoints, runs no training,
      contains the oracle NFE decomposition and field-geometry diagnostics.

  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_manifest.csv
  generate_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.py
  run_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_cell.py
  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_cell_job.slurm
  submit_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.py
  compile_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.py
  README_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.txt
  CHECKSUMS_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.txt

Run location
------------
Copy the bundle files to:

/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0

Optional dry run
----------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.py --cells all --dry-run

Submit all 20 cells
-------------------
"$SCRATCH/venvs/hlsi/bin/python" submit_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.py --cells all

Submission grouping / previous Slurm bug
----------------------------------------
The 20 cells are submitted as seven ordinary jobs:

  0,1,2
  3,4,5
  6,7,8
  9,10,11
  12,13,14
  15,16,17
  18,19

The submitter does NOT place the comma-containing cell list inside --export.
Instead it sets CSEM_ORACLE_NFE_CELL_IDS in sbatch's process environment and uses
--export=ALL.  A fake-sbatch regression test verified that all seven jobs receive
the complete values above, specifically guarding against the earlier failure mode
where only cells 0,3,6,... ran.

Compile
-------
"$SCRATCH/venvs/hlsi/bin/python" compile_cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1.py

Primary compiled outputs
------------------------
  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_compiled/oracle_nfe_curve_long.csv
  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_compiled/oracle_nfe_mechanism_wide.csv
  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_compiled/oracle_field_profile_all.csv
  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_compiled/oracle_field_profile_summary.csv
  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_compiled/run_status.csv
  cifar_oracle_nfe_mechanism_TK1p2_T1p6_v1_compiled/missing_or_failed.csv

Interpretive signatures
-----------------------
DYNAMICAL CSEM ADVANTAGE:
  oracle+q_h NFE curves improve substantially with CSEM, especially at low NFE,
  even though the score is exact.  Exact-field drift/path-variation diagnostics
  should ideally move consistently with that change.

PRIMARILY SCORE-ESTIMATION ADVANTAGE:
  oracle+q_h curves are similar across representations, while learned+q_h is much
  better for positive CSEM and learned-vs-oracle score error falls strongly.

TERMINAL-GAUSSIANIZATION ADVANTAGE:
  oracle+Gaussian improves relative to oracle+q_h when terminal KL increases,
  without a corresponding improvement in oracle+q_h dynamics.

MIXED MECHANISM:
  both oracle finite-NFE dynamics and learned-vs-oracle gaps improve.  This is a
  plausible and scientifically interesting outcome rather than a failure of the
  decomposition.
