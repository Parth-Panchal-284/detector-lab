#!/usr/bin/env bash
#SBATCH --job-name=detlab-latency
#SBATCH --partition=gpuqs
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
: "${TUTORIAL_ROOT:?Set TUTORIAL_ROOT to the professor repository root}"
: "${KITTI_ROOT:?Set KITTI_ROOT to the KITTI dataset root}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the detector-lab checkout}"
PYTHON_BIN="${PYTHON_BIN:-/app/rl9.x/ml/torch/2.6/bin/python}"
PYTHON_PACKAGES="${PYTHON_PACKAGES:-$PROJECT_ROOT/../python-packages}"
LATENCY_IMAGE="${LATENCY_IMAGE:-$KITTI_ROOT/training/image_2/000000.png}"
export PYTHONPATH="$PYTHON_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$LATENCY_IMAGE" ]]; then
    echo "Latency image not found: $LATENCY_IMAGE" >&2
    exit 2
fi

cd "$TUTORIAL_ROOT"
echo "Start: $(date -Is)"
echo "Host: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
"$PYTHON_BIN" -c 'import sys, torch, torchvision; print("python", sys.version); print("torch", torch.__version__, torch.__file__); print("torchvision", torchvision.__version__, torchvision.__file__); print("cuda", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "This Slurm task has no visible CUDA GPU"'
"$PYTHON_BIN" -m DeepDataMiningLearning.ngdet.latency \
    --models \
        torchvision:fasterrcnn_resnet50_fpn_v2 \
        yolo:yolo11m.pt \
        hf_detr:PekingU/rtdetr_r50vd \
    --image "$LATENCY_IMAGE" --accel fp32 fp16 --iters 40
echo "Finish: $(date -Is)"
