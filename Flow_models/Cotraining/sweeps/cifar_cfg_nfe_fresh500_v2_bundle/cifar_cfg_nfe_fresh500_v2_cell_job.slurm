#!/bin/bash
#SBATCH -J csem_cfg_nfe_f500
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs_cifar_cfg_nfe_fresh500_v2/%x_%j.out
#SBATCH -e slurm_logs_cifar_cfg_nfe_fresh500_v2/%x_%j.err

set -euo pipefail

BASE_DIR="${BASE_DIR:-/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/eval_sweep}"
CELL_ID="${CELL_ID:?CELL_ID must be exported by the submitter}"
PYTHON_BIN="${PYTHON_BIN:-$SCRATCH/venvs/hlsi/bin/python}"

cd "$BASE_DIR"
mkdir -p slurm_logs_cifar_cfg_nfe_fresh500_v2

"$PYTHON_BIN" -u run_cifar_cfg_nfe_fresh500_v2_cell.py \
  --cell-id "$CELL_ID" \
  --base-dir "$BASE_DIR" \
  --python "$PYTHON_BIN"
