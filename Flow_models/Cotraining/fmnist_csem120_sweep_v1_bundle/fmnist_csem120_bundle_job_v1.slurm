#!/bin/bash
#SBATCH -J c120bundle
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -o slurm-csem120-bundle-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
RUNNER="$BASEDIR/fmnist_csem120_bundle_runner_v1.py"

echo "== Job: CSEM FMNIST 120-cell screening bundle"
echo "== Host: $(hostname)"
echo "== TACC_SYSTEM: ${TACC_SYSTEM:-unknown}"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Bundle ID: ${CSEM_BUNDLE_ID:-UNSET}"
echo "== Start time: $(date)"

if [ -z "${CSEM_BUNDLE_ID:-}" ]; then
    echo "[error] CSEM_BUNDLE_ID is not set."
    exit 2
fi

# ----------------------------------------------------------------------
# Modules / Python environment — mirrors the current FMNIST run.
# ----------------------------------------------------------------------
module purge
module load gcc/13.2.0
module load python3/3.11.8
python3 -V

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

echo "== Python: $(python -V)"
echo "== Which python: $(which python)"

# ----------------------------------------------------------------------
# Runtime environment.
# ----------------------------------------------------------------------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "$BASEDIR" ]; then
    echo "[error] Base directory not found: $BASEDIR"
    exit 2
fi

if [ ! -f "$BASEDIR/csem_split_metric.py" ]; then
    echo "[error] Training script not found: $BASEDIR/csem_split_metric.py"
    exit 2
fi

if [ ! -f "$RUNNER" ]; then
    echo "[error] Bundle runner not found: $RUNNER"
    exit 2
fi

if [ ! -f "$BASEDIR/fmnist_csem120_manifest_v1.csv" ]; then
    echo "[error] Manifest not found."
    echo "        Run: python3 $BASEDIR/fmnist_csem120_grid_v1.py"
    exit 2
fi

cd "$BASEDIR"

echo "== cwd: $(pwd)"
echo "== GPU:"
nvidia-smi || true

# Fail early if the venv cannot see the GH GPU.
python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"[error] Could not import torch: {exc}")
    sys.exit(2)

print("== PyTorch:", torch.__version__)
print("== torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("[error] PyTorch cannot see a CUDA GPU.")
    sys.exit(2)

print("== CUDA device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"== GPU memory: {props.total_memory / 2**30:.1f} GiB")
PY

rc=$?
if [ $rc -ne 0 ]; then
    exit $rc
fi

echo "== Screening contract:"
echo "   6 sequential runs in this ordinary Slurm job"
echo "   120 co-training epochs/run, 0 refinement"
echo "   eval at epochs 60 and 120, 2000 samples"
echo "   terminal_kl arm only; T=1.50; CFG=3.0"
echo "   score-head-loss-w=1.0"
echo "   common seed=42 (fixed inside csem_split_metric.py)"
echo "== Starting bundle ${CSEM_BUNDLE_ID} =="

start=$SECONDS
python -u "$RUNNER" --bundle-id "$CSEM_BUNDLE_ID" --base-dir "$BASEDIR"
rc=$?
elapsed=$((SECONDS - start))

hours=$((elapsed / 3600))
mins=$(((elapsed % 3600) / 60))
secs=$((elapsed % 60))

if [ $rc -eq 0 ]; then
    echo "== Bundle ${CSEM_BUNDLE_ID} OK (elapsed: ${hours}h ${mins}m ${secs}s)"
else
    echo "== Bundle ${CSEM_BUNDLE_ID} completed with failures rc=$rc (elapsed: ${hours}h ${mins}m ${secs}s)"
fi

echo "== End time: $(date)"
exit $rc
