#!/bin/bash
# Submit one Frontera rtx job per HLSI problem.
# Run from the HLSI_inverse_problems_frontera directory on a login node.
# Usage: bash submit_all.sh

set -euo pipefail

PROBLEMS=(
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
)

SUB_PROBLEMS=(
)

for p in "${PROBLEMS[@]}"; do
  echo "Submitting $p..."
  sbatch -J "hlsi-$p" --export=ALL,PROB="$p" run_one.slurm
done

echo ""
echo "All submitted. Check with: squeue -u \$USER"
