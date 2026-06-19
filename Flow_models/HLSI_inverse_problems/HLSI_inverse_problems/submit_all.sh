#!/bin/bash
# Submit one Frontera rtx job per HLSI problem.
# Run from the HLSI_inverse_problems_frontera directory on a login node.
# Usage: bash submit_all.sh

set -euo pipefail

ALL_PROBLEMS=(
  "advect_diff"
  "allen_cahn"
  "darcy_flow"
  "eit"
  "heat"
  "helmholtz"
  "navier_stokes"
  "poisson"
  "afwi"
  "modulus"
  "helmholtz_alt"
  "known_z_calibration"
  "known_z_calibration2"
)

PROBLEMS=(
"darcy_flow"
"known_z_calibration"
"known_z_calibration2"
)

for p in "${PROBLEMS[@]}"; do
  echo "Submitting $p..."
  sbatch -J "hlsi-$p" --export=ALL,PROB="$p" run_one.slurm
done

echo ""
echo "All submitted. Check with: squeue -u \$USER"
