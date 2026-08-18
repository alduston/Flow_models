#!/bin/bash
#SBATCH -J cfFineTK
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o slurm-csem-cifar-fine-TKxCSEM-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_Tk_T_0"
RUNNER="$BASEDIR/run_cifar_TKxCSEM_fine_T1p75_v1_group.py"
TARGET="$BASEDIR/csem_split_metric_TKxCSEM_fine_cifar_v1.py"
MANIFEST="$BASEDIR/cifar_TKxCSEM_fine_T1p75_v1_manifest.csv"

echo "== CIFAR fine T_K x CSEM sweep"
echo "== Host: $(hostname)"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Group ID: ${CSEM_FINE_GROUP_ID:-UNSET}"
echo "== Start: $(date)"

if [ -z "${CSEM_FINE_GROUP_ID:-}" ]; then
    echo "[error] CSEM_FINE_GROUP_ID is not set."
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
from pathlib import Path
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

target = Path("csem_split_metric_TKxCSEM_fine_cifar_v1.py")
src = target.read_text()
required = [
    "aux_head_w * aux_loss_lam_inner",
    "representation_active = T_K > 0.0",
    '"--T", "--T-score", "--T-sampling"',
]
for marker in required:
    if marker not in src:
        raise SystemExit(f"[error] target missing required split-routing marker: {marker}")
if "score_loss_lsi_inner + aux_head_w * aux_loss_lam\n" in src:
    raise SystemExit("[error] target appears to contain the old factored-head graph-sharing bug.")
print("Static T/T_K + factored-head routing contract: OK")
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
echo "   T fixed               = 1.75"
echo "   one T_K per job       = group-selected from .70,.85,1.00,1.15,1.30,1.45"
echo "   three sequential CSEM = .075,.10,.15"
echo "   terminal KL weight    = .30"
echo "   outer/head weighting  = canonical / unweighted-eps"
echo "   epochs/refine         = 500 / 0"
echo "   eval                  = every 50, 10000 samples"
echo "   CFG                    = 3.0"
echo "   score-head LR         = CIFAR preset default 1e-4"
echo "   no Slurm arrays"

start=$SECONDS
"$PY" -u "$RUNNER"     --group-id "$CSEM_FINE_GROUP_ID"     --base-dir "$BASEDIR"
rc=$?

elapsed=$((SECONDS-start))
echo "== End: $(date)"
echo "== rc=$rc elapsed=$((elapsed/3600))h $(((elapsed%3600)/60))m $((elapsed%60))s"
exit $rc
