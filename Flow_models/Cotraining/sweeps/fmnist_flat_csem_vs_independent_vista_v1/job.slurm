#!/bin/bash
#SBATCH -J fmnist_three_way
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --signal=B:USR1@600
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
    : "${SCRATCH:?Set VENV_DIR to the existing hlsi environment}"
    VENV_DIR="$SCRATCH/venvs/hlsi"
fi
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY="$VENV_DIR/bin/python3"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASEDIR"
: "${TASK:?TASK unset}"
ARGS=(run --task "$TASK")
if [ -n "${CSEM_DATA_ROOT:-}" ]; then ARGS+=(--data-root "$CSEM_DATA_ROOT"); fi
"$PY" -u paper.py "${ARGS[@]}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1 TERM
set +e
while true; do
    wait "$TRAIN_PID"
    RC=$?
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then break; fi
done
exit "$RC"
