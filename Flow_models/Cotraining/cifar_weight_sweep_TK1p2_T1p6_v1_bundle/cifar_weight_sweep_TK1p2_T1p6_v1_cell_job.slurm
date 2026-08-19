#!/bin/bash
#SBATCH -J csemCFwgt
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o slurm-csem-cifar-weight-sweep-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0"
RUNNER="$BASEDIR/run_cifar_weight_sweep_TK1p2_T1p6_v1_cell.py"
TARGET="$BASEDIR/csem_split_new_weight_sweep_TK1p2_T1p6_v1.py"
MANIFEST="$BASEDIR/cifar_weight_sweep_TK1p2_T1p6_v1_manifest.csv"

echo "== CIFAR CSEM/KL weight sweep, fixed T_K=1.2, T=1.6"
echo "== Host: $(hostname)"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Cell IDs: ${CSEM_WEIGHT_SWEEP_CELL_IDS:-UNSET}"
echo "== Start: $(date)"

if [ -z "${CSEM_WEIGHT_SWEEP_CELL_IDS:-}" ]; then
    echo "[error] CSEM_WEIGHT_SWEEP_CELL_IDS is not set."
    exit 2
fi

IFS=',' read -r -a CELLS <<< "$CSEM_WEIGHT_SWEEP_CELL_IDS"
if [ "${#CELLS[@]}" -lt 1 ] || [ "${#CELLS[@]}" -gt 3 ]; then
    echo "[error] Expected 1-3 cells per job, got ${#CELLS[@]}: $CSEM_WEIGHT_SWEEP_CELL_IDS"
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
"$PY" - <<'PYENV'
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
PYENV
rc=$?
if [ $rc -ne 0 ]; then exit $rc; fi

"$PY" -m py_compile "$TARGET" "$RUNNER"
rc=$?
if [ $rc -ne 0 ]; then
    echo "[error] Python preflight compilation failed."
    exit $rc
fi

echo "== Scientific contract:"
echo "   T_K / T             = 1.2 / 1.6"
echo "   csem_w grid         = 0,.02,.04,.06,.08"
echo "   terminal_kl_w grid  = .10,.20,.30,.40"
echo "   outer/head          = canonical / unweighted-eps"
echo "   cotrain / refine    = 500 / 100"
echo "   eval                = every 50, 10000 samples"
echo "   4-way eval          = oracle/Gaussian init at T_K and T, matched NFE density"
echo "   CFG                 = 3.0"
echo "   cells in this job   = $CSEM_WEIGHT_SWEEP_CELL_IDS"
echo "   no Slurm arrays"

start=$SECONDS
overall_rc=0

for cell in "${CELLS[@]}"; do
    echo
    echo "================================================================================"
    echo "== Starting sweep cell $cell at $(date)"
    echo "================================================================================"
    "$PY" -u "$RUNNER" --cell-id "$cell" --base-dir "$BASEDIR"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[failed] cell $cell rc=$rc; continuing remaining cells in this job."
        overall_rc=$rc
    fi
done

elapsed=$((SECONDS-start))
echo "== End: $(date)"
echo "== overall_rc=$overall_rc elapsed=$((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"
exit $overall_rc
