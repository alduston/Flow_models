#!/bin/bash
#SBATCH -J csemPareto
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o slurm-csem-pareto-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/sweep"
RUNNER="$BASEDIR/run_cifar_csem_vs_twostage_pareto_T1p6_20run_v1_cell.py"
TARGET="$BASEDIR/csem_pareto_T1p6_20run_v1.py"
MANIFEST="$BASEDIR/cifar_csem_vs_twostage_pareto_T1p6_20run_v1_manifest.csv"

echo "== CIFAR CSEM vs standard two-stage Pareto dominance sweep v1"
echo "== Host: $(hostname)"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Pareto cell: ${CSEM_PARETO_CELL:-UNSET}"
echo "== Start: $(date)"

if [ -z "${CSEM_PARETO_CELL:-}" ]; then
    echo "[error] CSEM_PARETO_CELL is not set."
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
print("Python:", sys.executable)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("[error] CUDA GPU is not visible.")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU GiB:", torch.cuda.get_device_properties(0).total_memory/2**30)
PYENV
rc=$?; if [ $rc -ne 0 ]; then exit $rc; fi

"$PY" -m py_compile "$TARGET" "$RUNNER"
rc=$?; if [ $rc -ne 0 ]; then exit $rc; fi

echo "== Scientific contract:"
echo "   exactly one independent 500-epoch training; NO score-only refinement"
echo "   family A: exact standard LDM endpoint T_K=0, csem=0, beta0 in {0,.01,.03,.07,.15,.30}"
echo "   family B: CSEM T_K=1.2, T=1.6, K_TK weight=.40, csem in {.02,.04,.08,.15}"
echo "   paired seeds {42,43}; 12 + 8 = 20 total trainings"
echo "   final ordinary eval only, then frozen oracle/learned mechanism audit"
echo "   RK4 steps {5,10,25,50} = {20,40,100,200} NFE at full T"
echo "   exact T_K=0 never approximated by t_min"

start=$SECONDS
"$PY" -u "$RUNNER" --cell-id "$CSEM_PARETO_CELL" --base-dir "$BASEDIR"
rc=$?
elapsed=$((SECONDS-start))
echo "== End: $(date)"
echo "== rc=$rc elapsed=$((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"
exit $rc
