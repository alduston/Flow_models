#!/bin/bash
#SBATCH -J hlsi
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL
# #SBATCH -A YOUR_TACC_PROJECT

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

# --- Frontera modules (must match how the venv was built) ---
module purge
module load gcc/9.1.0
module load python3/3.9.2
python3 -V

# --- persistent venv on Frontera ---
VENV_DIR="/work2/10812/ald4435/frontera/venvs/flow_frontera_py39/"
if [ ! -d "$VENV_DIR" ]; then
  echo "[error] Expected venv at $VENV_DIR not found."
  exit 2
fi
source "$VENV_DIR/bin/activate"
echo "== Python: $(python -V)"
echo "== Which python: $(which python)"

# --- base dir and target ---
# Default to the directory from which sbatch was called; this should be the
# repo root that contains sampling.py and the per-experiment subdirectories.
BASEDIR="${SLURM_SUBMIT_DIR:-/work2/10812/ald4435/frontera/Flow_models/Flow_models/HLSI_inverse_problems/HLSI_inverse_problems_frontera}"
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
