#!/usr/bin/env bash
#SBATCH --job-name=detlab-qual
#SBATCH --partition=gpuqs
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

: "${TUTORIAL_ROOT:?Set TUTORIAL_ROOT to the DeepDataMiningLearning repository root}"
: "${COCO_ROOT:?Set COCO_ROOT to the COCO val2017 image directory}"
: "${COCO_ANN:?Set COCO_ANN to instances_val2017.json}"
: "${KITTI_ROOT:?Set KITTI_ROOT to the KITTI root}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from detector-lab}"
PYTHON_BIN="${PYTHON_BIN:-/app/rl9.x/ml/torch/2.6/bin/python}"
PYTHON_PACKAGES="${PYTHON_PACKAGES:-$PROJECT_ROOT/../python-packages}"
QUAL_OUT="${QUAL_OUT:-$PROJECT_ROOT/qualitative}"
SCAN_LIMIT="${SCAN_LIMIT:-1500}"
SCORE_THR="${SCORE_THR:-0.25}"

export PYTHONPATH="$TUTORIAL_ROOT:$PYTHON_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$PROJECT_ROOT/logs/slurm" "$QUAL_OUT"
cd "$PROJECT_ROOT"

echo "Start: $(date -Is)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("This Slurm task has no visible CUDA GPU")

x = torch.ones(1024, device="cuda:0")
y = (x * 2.0).sum()
torch.cuda.synchronize()
print(
    "CUDA smoke test OK:",
    torch.cuda.get_device_name(0),
    "result=",
    float(y.item()),
)
PY

cmd=(
    "$PYTHON_BIN"
    scripts/qualitative_examples.py
    --tutorial-root "$TUTORIAL_ROOT"
    --coco-root "$COCO_ROOT"
    --coco-ann "$COCO_ANN"
    --kitti-root "$KITTI_ROOT"
    --out-dir "$QUAL_OUT"
    --device cuda
    --score-thr "$SCORE_THR"
    --scan-limit "$SCAN_LIMIT"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

required=(
    "$QUAL_OUT/selection.json"
    "$QUAL_OUT/qualitative_metrics.csv"
)

for f in "${required[@]}"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: expected qualitative output is missing: $f" >&2
        exit 3
    fi
done

panel_count="$(find "$QUAL_OUT" -maxdepth 1 -name '*.png' -type f | wc -l)"
if [[ "$panel_count" -lt 8 ]]; then
    echo "ERROR: expected at least 8 qualitative PNG panels, found $panel_count" >&2
    exit 4
fi

echo "Verified qualitative outputs: $panel_count PNG panels"
echo "Finish: $(date -Is)"
