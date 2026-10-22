#!/bin/bash
#SBATCH -J fmnist_capacity_matched
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail
BASEDIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ "${SKIP_MODULES:-0}" != 1 ]; then
    module purge
    module load gcc/13.2.0
    module load python3/3.11.8
fi
if [ -z "${VENV_DIR:-}" ]; then
    : "${SCRATCH:?SCRATCH unset; set VENV_DIR to your hlsi environment}"
    VENV_DIR="$SCRATCH/venvs/hlsi"
fi
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY="$VENV_DIR/bin/python3"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASEDIR"
: "${CELL_ID:?CELL_ID unset}"
"$PY" - <<'PYENV'
import torch,torchvision,lpips
print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,flush=True)
if not torch.cuda.is_available():raise SystemExit('CUDA GPU not visible')
print('GPU:',torch.cuda.get_device_name(0),flush=True)
PYENV
ARGS=(--cell-id "$CELL_ID" --base-dir "$BASEDIR" --resume)
if [ -n "${CSEM_DATA_ROOT:-}" ]; then ARGS+=(--data-root "$CSEM_DATA_ROOT"); fi
if [ -n "${AUDIT_EPOCH:-}" ]; then
    exec "$PY" -u paper.py audit "${ARGS[@]}" --epoch "$AUDIT_EPOCH"
fi
# Bash receives Slurm's early warning and forwards it to the Python process.
# Avoid exec: --signal=B targets the batch shell, not the child Python PID.
"$PY" -u paper.py run "${ARGS[@]}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1 TERM
set +e
while true; do
    wait "$TRAIN_PID"
    RC=$?
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then break; fi
done
exit "$RC"
