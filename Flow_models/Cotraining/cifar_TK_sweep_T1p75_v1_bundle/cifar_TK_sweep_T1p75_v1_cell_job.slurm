#!/bin/bash
#SBATCH -J csemCFTKs
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o slurm-csem-cifar-TK-sweep-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0"
RUNNER="$BASEDIR/run_cifar_TK_sweep_T1p75_v1_cell.py"
TARGET="$BASEDIR/csem_split_metric_TK_sweep_cifar_v1.py"
MANIFEST="$BASEDIR/cifar_TK_sweep_T1p75_v1_manifest.csv"

echo "== CIFAR T_K sweep, fixed T=1.75"
echo "== Host: $(hostname)"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Cell ID: ${CSEM_TK_SWEEP_CELL_ID:-UNSET}"
echo "== Start: $(date)"

if [ -z "${CSEM_TK_SWEEP_CELL_ID:-}" ]; then
    echo "[error] CSEM_TK_SWEEP_CELL_ID is not set."
    exit 2
fi

module purge
module load gcc/13.2.0
module load python3/3.11.8

if [ -z "${SCRATCH:-}" ]; then
    echo "[error] SCRATCH is not set."
    exit 2
fi

VENV_DIR="$SCRATCH/venvs/hlsi"
if [ ! -d "$VENV_DIR" ]; then
    echo "[error] Expected venv at $VENV_DIR not found."
    exit 2
fi

source "$VENV_DIR/bin/activate"

if [ -x "$VENV_DIR/bin/python" ]; then
    PY="$VENV_DIR/bin/python"
elif [ -x "$VENV_DIR/bin/python3" ]; then
    PY="$VENV_DIR/bin/python3"
else
    echo "[error] No executable Python inside $VENV_DIR/bin"
    exit 2
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for f in "$RUNNER" "$TARGET" "$MANIFEST"; do
    if [ ! -f "$f" ]; then
        echo "[error] Missing required file: $f"
        exit 2
    fi
done

cd "$BASEDIR"

echo "== Forced Python: $PY"
"$PY" - <<'PY'
import sys
import numpy as np
import torch

print("Python:", sys.executable)
print("NumPy:", np.__version__, np.__file__)
print("Torch:", torch.__version__, torch.__file__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA GPU is not visible.")

print("GPU:", torch.cuda.get_device_name(0))
print("GPU GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
rc=$?
if [ $rc -ne 0 ]; then exit $rc; fi

"$PY" -m py_compile "$TARGET" "$RUNNER"
rc=$?
if [ $rc -ne 0 ]; then
    echo "[error] Python preflight compilation failed."
    exit $rc
fi

echo "== Scientific contract:"
echo "   T fixed             = 1.75"
echo "   T_K sweep           = 0,.25,.5,.75,1,1.25,1.5,1.75"
echo "   outer/head          = canonical / unweighted-eps"
echo "   csem / terminal KL  = 0.10 / 0.30"
echo "   cotrain / refine    = 500 / 100"
echo "   eval                = every 50, 10000 samples, RK4 comparison"
echo "   CFG                 = 3.0"
echo "   score-head LR       = CIFAR preset default 1e-4"
echo "   no Slurm arrays"

start=$SECONDS

"$PY" -u "$RUNNER"     --cell-id "$CSEM_TK_SWEEP_CELL_ID"     --base-dir "$BASEDIR"
rc=$?

elapsed=$((SECONDS-start))
echo "== End: $(date)"
echo "== rc=$rc elapsed=$((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"
exit $rc
