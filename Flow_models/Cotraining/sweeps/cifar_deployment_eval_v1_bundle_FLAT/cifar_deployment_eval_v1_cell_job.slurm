#!/bin/bash
#SBATCH -J csem-eval
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=START,END,FAIL
set -euo pipefail
: "${CELL_ID:?CELL_ID must be exported by submitter}"
: "${BASE_DIR:?BASE_DIR must be exported by submitter}"
module purge
module load gcc/13.2.0
module load python3/3.11.8
source "$SCRATCH/venvs/hlsi/bin/activate"
cd "$BASE_DIR"
echo "== Job: $SLURM_JOB_ID"; echo "== Host: $(hostname)"; echo "== Cell: $CELL_ID"; echo "== CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('torch',torch.__version__); print('cuda',torch.cuda.is_available()); assert torch.cuda.is_available(); print('gpu',torch.cuda.get_device_name(0))"
python -u run_cifar_deployment_eval_v1_cell.py --cell-id "$CELL_ID" --base-dir "$BASE_DIR"
