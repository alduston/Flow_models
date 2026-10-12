#!/bin/bash
#SBATCH -J lsgm_csem_cifar
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
set -euo pipefail
BASEDIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
module purge
module load gcc/13.2.0
module load python3/3.11.8
# Set LSGM_CUDA_MODULE to the installed CUDA module matching torch.version.cuda.
# If unset, native FIR is allowed; auto mode logs any JIT fallback explicitly.
if [ -n "${LSGM_CUDA_MODULE:-}" ]; then module load "$LSGM_CUDA_MODULE"; fi
: "${SCRATCH:?SCRATCH unset}"
VENV_DIR="$SCRATCH/venvs/hlsi"
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY="$VENV_DIR/bin/python3"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$SCRATCH/torch_extensions/lsgm}"
export PYTHONPATH="$BASEDIR/lsgm:$BASEDIR:${PYTHONPATH:-}"
export LSGM_OP_BACKEND="${LSGM_OP_BACKEND:-auto}"
export MAX_JOBS="${MAX_JOBS:-8}"
cd "$BASEDIR"
: "${CELL_ID:?CELL_ID unset}"
"$PY" - <<'PYENV'
import sys, torch, torchvision, tensorboardX, lmdb, torchdiffeq
print('Python:', sys.executable, 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
if not torch.cuda.is_available(): raise SystemExit('CUDA GPU is not visible')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PYENV
# Separate synthetic smoke process: no data download, checkpoint or RNG changes.
"$PY" -u paper.py smoke --cell-id "$CELL_ID" --device cuda
ARGS=(--cell-id "$CELL_ID" --base-dir "$BASEDIR" --resume)
if [ "${RESTART:-0}" = 1 ]; then ARGS+=(--restart); fi
if [ -n "${CSEM_DATA_ROOT:-}" ]; then ARGS+=(--data-root "$CSEM_DATA_ROOT"); fi
"$PY" -u paper.py run "${ARGS[@]}"
