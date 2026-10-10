#!/bin/bash
#SBATCH -J csem_v6
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail
BASEDIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TASK="${TASK:-run}"
module purge
module load gcc/13.2.0
module load python3/3.11.8
: "${SCRATCH:?SCRATCH unset}"
VENV_DIR="$SCRATCH/venvs/hlsi"
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY="$VENV_DIR/bin/python3"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASEDIR"
if [ "$TASK" = prepare ]; then
  DATASETS="${DATASETS:-CIFAR:FMNIST}"
  "$PY" -u paper.py prepare-data --datasets "${DATASETS//:/,}"
  exit $?
fi
: "${CELL_ID:?CELL_ID unset}"
"$PY" - <<'PYENV'
import sys, torch, torchvision, lpips
print('Python:', sys.executable, 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
if not torch.cuda.is_available(): raise SystemExit('CUDA GPU is not visible')
print('GPU:', torch.cuda.get_device_name(0))
PYENV
ARGS=(--cell-id "$CELL_ID" --base-dir "$BASEDIR" --resume)
if [ "${RESTART:-0}" = 1 ]; then ARGS+=(--restart); fi
echo "Start cell=$CELL_ID job=${SLURM_JOB_ID:-unknown} $(date)"
"$PY" -u paper.py run "${ARGS[@]}"
