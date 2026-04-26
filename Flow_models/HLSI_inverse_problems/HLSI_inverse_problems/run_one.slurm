#!/bin/bash
#SBATCH -J hlsi
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

# PROB is passed in via sbatch --export; bail if not set
if [ -z "${PROB:-}" ]; then
  echo "[error] PROB not set. Submit with: sbatch --export=ALL,PROB=<name> run_one.slurm"
  exit 2
fi

echo "== Problem: $PROB"
echo "== Host: $(hostname)"
echo "== TACC_SYSTEM: ${TACC_SYSTEM:-unknown}"
echo "== Start time: $(date)"

# --- VISTA modules (must match how the venv was built) ---
module purge
module load gcc/13.2.0
module load python3/3.11.8
python3 -V

# --- persistent venv on Vista ---
VENV_DIR="$SCRATCH/venvs/hlsi"
if [ ! -d "$VENV_DIR" ]; then
  echo "[error] Expected venv at $VENV_DIR not found."
  exit 2
fi
source "$VENV_DIR/bin/activate"
echo "== Python: $(python -V)"
echo "== Which python: $(which python)"

# --- runtime environment ---
# Cap CPU threading (Grace has 72 cores; uncapped BLAS/OMP will thrash)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
# Headless matplotlib so plt.show() can never block the job
export MPLBACKEND=Agg
# Fail loudly if JAX can't see the GPU instead of silently falling back to CPU
export JAX_PLATFORMS=cuda

# --- base dir and target ---
# Default to the directory from which sbatch was called; this should be the
# repo root that contains sampling.py and the per-experiment subdirectories.
BASEDIR="${SLURM_SUBMIT_DIR:-/work2/10812/ald4435/frontera/Flow_models/Flow_models/HLSI_inverse_problems/HLSI_inverse_problems}"
PROB_DIR="$BASEDIR/$PROB"
TARGET="$PROB_DIR/${PROB}.py"
export PYTHONPATH="$BASEDIR:${PYTHONPATH:-}"

if [ ! -f "$TARGET" ]; then
  echo "[error] Could not find target script at: $TARGET"
  exit 2
fi

echo "== GPUs:"
nvidia-smi || true

cd "$PROB_DIR"
echo "== cwd: $(pwd)"
echo "== target: $TARGET"
echo "== [$(date)] Running $PROB =="

start=$SECONDS
python -u "$TARGET"
rc=$?
elapsed=$((SECONDS - start))

if [ $rc -eq 0 ]; then
  echo "== [$PROB] OK  (elapsed: ${elapsed}s)"
else
  echo "== [$PROB] FAILED rc=$rc  (elapsed: ${elapsed}s)"
fi

exit $rc
