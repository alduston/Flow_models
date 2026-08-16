#!/bin/bash
#SBATCH -J csemCFapV1
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o slurm-csem-cifar-apples-v1-%j.out
#SBATCH --mail-user=alois.duston@utexas.edu
#SBATCH --mail-type=END,FAIL,START

set -uo pipefail

echo "== Job: CIFAR CSEM apples-to-apples canonical-encoder / epsilon-head terminal-KL"
echo "== Host: $(hostname)"
echo "== TACC_SYSTEM: ${TACC_SYSTEM:-unknown}"
echo "== Job ID: ${SLURM_JOB_ID:-unknown}"
echo "== Start time: $(date)"

# ----------------------------------------------------------------------
# Modules / Python environment -- same environment as the historical run.
# ----------------------------------------------------------------------
module purge
module load gcc/13.2.0
module load python3/3.11.8
python3 -V

VENV_DIR="$SCRATCH/venvs/hlsi"
if [ ! -d "$VENV_DIR" ]; then
    echo "[error] Expected venv at $VENV_DIR not found."
    exit 2
fi

source "$VENV_DIR/bin/activate"

echo "== Python: $(python -V)"
echo "== Which python: $(which python)"

# ----------------------------------------------------------------------
# Runtime environment -- same as historical run.
# ----------------------------------------------------------------------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ----------------------------------------------------------------------
# Experiment location.
#
# New directory/name avoids collisions with cifar_split_metric0 and prior
# run_comparison.slurm files. Override without editing with CSEM_RUN_DIR.
# ----------------------------------------------------------------------
DEFAULT_BASEDIR="/work/10812/ald4435/frontera/Flow_models/Flow_models/Cotraining/cifar/cifar_split_metric1"
BASEDIR="${CSEM_RUN_DIR:-$DEFAULT_BASEDIR}"
SCRIPT_NAME="csem_split_metric_apples_v1.py"
TARGET="$BASEDIR/$SCRIPT_NAME"
RESULTS_NAME="run_results_cifar_split_apples_v1_terminal_w0p6_T1p50_800e_ref100"
RESULTS_DIR="$BASEDIR/$RESULTS_NAME"

if [ ! -d "$BASEDIR" ]; then
    echo "[error] Run directory does not exist: $BASEDIR"
    echo "        Override with CSEM_RUN_DIR if needed."
    exit 2
fi

if [ ! -f "$TARGET" ]; then
    echo "[error] Could not find target script at: $TARGET"
    exit 2
fi

if [ -e "$RESULTS_DIR" ]; then
    echo "[error] Results path already exists; refusing to overwrite: $RESULTS_DIR"
    echo "        Change RESULTS_NAME before resubmitting."
    exit 2
fi

export PYTHONPATH="$BASEDIR:${PYTHONPATH:-}"

echo "== GPUs:"
nvidia-smi || true

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

cd "$BASEDIR"

echo "== cwd: $(pwd)"
echo "== target: $TARGET"
echo "== Matched configuration:"
echo "   dataset/preset       = CIFAR / auto -> cifar_golden"
echo "   arm                  = terminal_kl only"
echo "   gradient routing     = canonical CSEM -> encoder/VAE; unweighted eps-MSE -> score head"
echo "   objective weights    = CSEM 0.6 / terminal K_T 0.6"
echo "   terminal T           = 1.50"
echo "   cotrain/refine       = preset defaults 800 / 100 epochs"
echo "   evaluation           = preset default every 100 epochs, 10000 samples"
echo "   CFG strength         = 2.0"
echo "   VAE/base LDM LR      = preset 2.5e-4 / 1.0e-4"
echo "   score-head LR        = legacy rule lr_ldm/csem_w = 1.6666667e-4"
echo "   canonical LR scaling = no explicit override (scale=1.0, hence no LR change)"
echo "   PatchGAN             = historical defaults: w=0.0025, lr=1e-4, start epoch 25, UNCLIPPED"
echo "   cotrain grad clip    = historical ONE joint VAE+score global norm clip at 1.0"
echo "   encoder warmup/ramp  = 0 / 0"
echo "   extra score tracking = 0"
echo "   grad diagnostics     = disabled"
echo "   logvar clamp         = historical [-30, 20]"
echo "   finite-abort safety  = disabled to match historical behavior"
echo "   results              = $RESULTS_DIR"
echo "== [$(date)] Starting matched CIFAR run =="

start=$SECONDS

# Intentionally DO NOT override LR, clipping, GAN start/LR, epoch budget,
# refinement budget, evaluation cadence, canonical-lr-scale, or logvar bounds.
# Those are restored to the historical csem_new.py contract by preset/defaults
# plus the two tiny code-level clipping reversions in csem_split_metric_apples_v1.py.
python -u "$TARGET" \
    --dataset CIFAR \
    --model-preset auto \
    --terminal-kl-w 0.6 \
    --csem-w 0.6 \
    --score-time-weighting canonical \
    --score-head-time-weighting unweighted-eps \
    --T-terminal 1.50 \
    --cfg-strength 2.00 \
    --encoder-score-warmup-epochs 0 \
    --csem-ramp-epochs 0 \
    --score-tracking-steps 0 \
    --grad-diagnostics-every 0 \
    --logvar-min=-30.0 \
    --logvar-max 20.0 \
    --no-fail-on-nonfinite \
    --no-bespoke-fid-classifier \
    --arms terminal_kl \
    --master-results-dir "$RESULTS_NAME"

rc=$?
elapsed=$((SECONDS - start))

hours=$((elapsed / 3600))
mins=$(((elapsed % 3600) / 60))
secs=$((elapsed % 60))

if [ $rc -eq 0 ]; then
    echo "== CIFAR matched split-routing run OK (elapsed: ${hours}h ${mins}m ${secs}s)"
else
    echo "== CIFAR matched split-routing run FAILED rc=$rc (elapsed: ${hours}h ${mins}m ${secs}s)"
fi

echo "== End time: $(date)"
exit $rc
