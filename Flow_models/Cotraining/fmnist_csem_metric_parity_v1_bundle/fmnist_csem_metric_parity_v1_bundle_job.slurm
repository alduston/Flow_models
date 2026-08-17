#!/bin/bash
#SBATCH -J csemMP
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -o slurm-csem-metric-parity-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/fmnist/120_sweep"
RUNNER="$BASEDIR/fmnist_csem_metric_parity_v1_bundle_runner.py"
TARGET="$BASEDIR/csem_split_metric_metric_parity_v1.py"
MANIFEST="$BASEDIR/fmnist_csem_metric_parity_v1_manifest.csv"

echo "== CSEM metric-parity v1"
echo "== Host: $(hostname)"
echo "== Job: ${SLURM_JOB_ID:-unknown}"
echo "== Bundle: ${CSEM_METRIC_PARITY_BUNDLE_ID:-UNSET}"
echo "== Start: $(date)"

if [ -z "${CSEM_METRIC_PARITY_BUNDLE_ID:-}" ]; then
    echo "[error] CSEM_METRIC_PARITY_BUNDLE_ID is not set."
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
if [ -x "$VENV_DIR/bin/python" ]; then
    PY="$VENV_DIR/bin/python"
elif [ -x "$VENV_DIR/bin/python3" ]; then
    PY="$VENV_DIR/bin/python3"
else
    echo "[error] No Python executable in $VENV_DIR/bin"
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

"$PY" - <<'PY'
import numpy as np, torch, sys
print("Python:", sys.executable)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__, "CUDA build", torch.version.cuda)
if not hasattr(np, "trapezoid"):
    raise SystemExit("numpy.trapezoid unavailable")
print("trapezoid smoke:", np.trapezoid(np.array([1.,2.,3.]), np.array([0.,1.,2.])))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY
rc=$?
if [ $rc -ne 0 ]; then exit $rc; fi

if grep -n 'np\.trapz' "$TARGET"; then
    echo "[error] Legacy np.trapz reference found."
    exit 2
fi

"$PY" -u "$RUNNER"     --bundle-id "$CSEM_METRIC_PARITY_BUNDLE_ID"     --base-dir "$BASEDIR"
rc=$?

echo "== End: $(date), rc=$rc"
exit $rc
