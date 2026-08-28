#!/bin/bash
#SBATCH -J csem_tk_anchor
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs_cifar_tk_anchor_compare_v1/%x_%j.out
#SBATCH -e slurm_logs_cifar_tk_anchor_compare_v1/%x_%j.err
set -uo pipefail
BASEDIR="${BASE_DIR:-/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/tk_anchor_sweep}"
RUNNER="$BASEDIR/run_cifar_tk_anchor_compare_v1_cell.py"; TARGET="$BASEDIR/csem_tk_anchor_compare_cifar_v1.py"; MANIFEST="$BASEDIR/cifar_tk_anchor_compare_v1_manifest.csv"; CELL_ID="${CELL_ID:-}"
echo "== CSEM T_K anchor sweep v1 | host=$(hostname) | job=${SLURM_JOB_ID:-unknown} | cell=${CELL_ID:-UNSET} | $(date)"
if [ -z "$CELL_ID" ]; then echo '[error] CELL_ID unset'; exit 2; fi
module purge; module load gcc/13.2.0; module load python3/3.11.8
if [ -z "${SCRATCH:-}" ]; then echo '[error] SCRATCH unset'; exit 2; fi
VENV_DIR="$SCRATCH/venvs/hlsi"; if [ ! -d "$VENV_DIR" ]; then echo "[error] missing $VENV_DIR"; exit 2; fi
source "$VENV_DIR/bin/activate"; PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY="$VENV_DIR/bin/python3"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASEDIR/slurm_logs_cifar_tk_anchor_compare_v1"; cd "$BASEDIR"
for f in "$RUNNER" "$TARGET" "$MANIFEST"; do [ -f "$f" ] || { echo "[error] missing $f"; exit 2; }; done
"$PY" - <<'PYENV'
import sys, torch, numpy as np
print('Python:',sys.executable); print('NumPy:',np.__version__); print('Torch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit('[error] CUDA GPU not visible')
print('GPU:',torch.cuda.get_device_name(0)); print('GPU GiB:',torch.cuda.get_device_properties(0).total_memory/2**30)
PYENV
rc=$?; [ $rc -eq 0 ] || exit $rc
"$PY" -m py_compile "$TARGET" "$RUNNER" || exit $?
echo '== Contract: 8 anchors/KL arms x seeds 42,43,44 = 24 fresh trainings; fixed TK=1.05 T=1.35 wC=.05; 500 joint, no refine; CFG2.5; RK4 25 steps=NFE100; 10k eval.'
start=$SECONDS; "$PY" -u "$RUNNER" --cell-id "$CELL_ID" --base-dir "$BASEDIR" --python "$PY"; rc=$?; elapsed=$((SECONDS-start))
echo "== End $(date) rc=$rc elapsed=$((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"; exit $rc
