#!/bin/bash
#SBATCH -J csemONFE2
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o slurm-csem-oracle-nfe-v2-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0"
RUNNER="$BASEDIR/run_cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_cell.py"
TARGET="$BASEDIR/csem_oracle_nfe_fromscratch_TK1p2_T1p6_v2.py"
MANIFEST="$BASEDIR/cifar_oracle_nfe_fromscratch_TK1p2_T1p6_v2_manifest.csv"

echo "== CIFAR oracle/NFE mechanism sweep v2 — train from scratch"
echo "== Host: $(hostname)"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Representation cell: ${CSEM_ORACLE_NFE_REP_CELL:-UNSET}"
echo "== Start: $(date)"

if [ -z "${CSEM_ORACLE_NFE_REP_CELL:-}" ]; then
    echo "[error] CSEM_ORACLE_NFE_REP_CELL is not set."
    exit 2
fi

module purge
module load gcc/13.2.0
module load python3/3.11.8

if [ -z "${SCRATCH:-}" ]; then echo "[error] SCRATCH is not set."; exit 2; fi
VENV_DIR="$SCRATCH/venvs/hlsi"
if [ ! -d "$VENV_DIR" ]; then echo "[error] Expected venv at $VENV_DIR not found."; exit 2; fi
source "$VENV_DIR/bin/activate"
if [ -x "$VENV_DIR/bin/python" ]; then PY="$VENV_DIR/bin/python"; else PY="$VENV_DIR/bin/python3"; fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for f in "$RUNNER" "$TARGET" "$MANIFEST"; do
    if [ ! -f "$f" ]; then echo "[error] Missing required file: $f"; exit 2; fi
done
cd "$BASEDIR"

"$PY" - <<'PYENV'
import sys, numpy as np, torch
print("Python:",sys.executable)
print("NumPy:",np.__version__)
print("Torch:",torch.__version__)
print("CUDA build:",torch.version.cuda)
print("CUDA available:",torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("[error] CUDA GPU is not visible.")
print("GPU:",torch.cuda.get_device_name(0))
print("GPU GiB:",torch.cuda.get_device_properties(0).total_memory/2**30)
PYENV
rc=$?; if [ $rc -ne 0 ]; then exit $rc; fi

"$PY" -m py_compile "$TARGET" "$RUNNER"
rc=$?; if [ $rc -ne 0 ]; then exit $rc; fi

echo "== Scientific contract:"
echo "   representation is TRAINED FROM SCRATCH in this job"
echo "   500 cotrain + 100 score-only refinement; same CIFAR routing as weight sweeps"
echo "   T_K / T = 1.2 / 1.6"
echo "   after training, freeze checkpoint and evaluate RK4 steps 5,10,25,50"
echo "   oracle vs learned x exact q_h vs Gaussian x h={T_K,T}"
echo "   paired endpoint errors to max-NFE trajectory are logged"
echo "   exact-field time profile is logged once per representation"

start=$SECONDS
"$PY" -u "$RUNNER" --cell-id "$CSEM_ORACLE_NFE_REP_CELL" --base-dir "$BASEDIR"
rc=$?
elapsed=$((SECONDS-start))
echo "== End: $(date)"
echo "== rc=$rc elapsed=$((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"
exit $rc
