#!/bin/bash
#SBATCH -J csemWhyTime
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -o slurm-csem-why-time-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
RUNNER="$BASEDIR/fmnist_csem_why_pair_runner_v2.py"
TARGET="$BASEDIR/csem_split_metric_why_v2.py"

echo "== CSEM WHY matched-pair diagnostic"
echo "== Host: $(hostname)"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Pair ID: ${CSEM_WHY_PAIR_ID:-UNSET}"
echo "== Start: $(date)"

if [ -z "${CSEM_WHY_PAIR_ID:-}" ]; then
    echo "[error] CSEM_WHY_PAIR_ID is not set."
    exit 2
fi

# Keep the same Vista environment contract that is now working for the parent sweep.
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
    VENV_PYTHON="$VENV_DIR/bin/python"
elif [ -x "$VENV_DIR/bin/python3" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python3"
else
    echo "[error] No executable Python inside $VENV_DIR/bin."
    exit 2
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for f in "$RUNNER" "$TARGET" "$BASEDIR/fmnist_csem_why_manifest_v2.csv"; do
    if [ ! -f "$f" ]; then
        echo "[error] Missing required file: $f"
        exit 2
    fi
done

cd "$BASEDIR"

echo "== Forced Python: $VENV_PYTHON"
"$VENV_PYTHON" -V
nvidia-smi || true

"$VENV_PYTHON" - <<'PY'
import sys, torch
print("== sys.executable:", sys.executable)
print("== torch:", torch.__version__, torch.__file__)
print("== CUDA build:", torch.version.cuda)
print("== CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA GPU is not visible.")
print("== GPU:", torch.cuda.get_device_name(0))
print("== GPU GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
rc=$?
if [ $rc -ne 0 ]; then
    exit $rc
fi

echo "== Diagnostic contract:"
echo "   2 sequential runs: reconstruction-matched U/C outer pair"
echo "   score-head metric fixed to unweighted-eps"
echo "   120 epochs, endpoint eval only, 2000 standard eval samples"
echo "   full FashionMNIST training mixture defines exact aggregate oracle"
echo "   32-point log-time score profile with per-t oracle error / intrinsic variance / physical contribution"
echo "   2000-sample RK4-25 oracle transport decomposition"
echo "   no Slurm arrays"

start=$SECONDS
"$VENV_PYTHON" -u "$RUNNER" \
    --pair-id "$CSEM_WHY_PAIR_ID" \
    --base-dir "$BASEDIR"
rc=$?
elapsed=$((SECONDS-start))

echo "== Pair ${CSEM_WHY_PAIR_ID} rc=$rc elapsed=$((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"
echo "== End: $(date)"
exit $rc
