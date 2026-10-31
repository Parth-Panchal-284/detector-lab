#!/usr/bin/env bash
#SBATCH --job-name=detlab-eval
#SBATCH --partition=gpuqs
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail

: "${TUTORIAL_ROOT:?Set TUTORIAL_ROOT to the professor repository root (contains DeepDataMiningLearning/)}"
: "${COCO_ROOT:?Set COCO_ROOT to the COCO val2017 image directory}"
: "${COCO_ANN:?Set COCO_ANN to instances_val2017.json}"
: "${KITTI_ROOT:?Set KITTI_ROOT to the KITTI root with training/image_2 and training/label_2}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the detector-lab checkout}"
PYTHON_BIN="${PYTHON_BIN:-/app/rl9.x/ml/torch/2.6/bin/python}"
PYTHON_PACKAGES="${PYTHON_PACKAGES:-$PROJECT_ROOT/../python-packages}"
export PYTHONPATH="$PYTHON_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
MAX_IMAGES="${MAX_IMAGES:-150}"
SCORE_THR="${SCORE_THR:-0.25}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"

# The table is deliberately plain text so the run matrix is easy to audit/edit.
IFS=$'\t' read -r task_id model_tag model_spec dataset < <(
    awk -F '\t' -v task="$SLURM_ARRAY_TASK_ID" '$1 !~ /^#/ && $1 == task {print $1 "\t" $2 "\t" $3 "\t" $4}' \
        "$PROJECT_ROOT/configs/jobs.tsv"
)
if [[ -z "${model_spec:-}" || -z "${dataset:-}" ]]; then
    echo "No job configuration for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

cd "$TUTORIAL_ROOT"
OUT_DIR="$RESULTS_DIR/${model_tag}_${dataset}"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" --version > "$OUT_DIR/python-version.txt" 2>&1
"$PYTHON_BIN" -m pip freeze > "$OUT_DIR/python-packages.txt"

args=(
    -m DeepDataMiningLearning.ngdet.run_eval
    --models "$model_spec"
    --datasets "$dataset"
    --max-images "$MAX_IMAGES"
    --score-thr "$SCORE_THR"
    --device cuda
    --out-dir "$OUT_DIR"
)
if [[ "$dataset" == "coco" ]]; then
    args+=(--coco-ann "$COCO_ANN")
fi
args+=(--roots "coco=$COCO_ROOT" "kitti=$KITTI_ROOT")

echo "Start: $(date -Is)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys, torch, torchvision; print("python", sys.version); print("torch", torch.__version__, torch.__file__); print("torchvision", torchvision.__version__, torchvision.__file__); print("cuda", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "This Slurm task has no visible CUDA GPU"'
printf 'Command:'; printf ' %q' "$PYTHON_BIN" "${args[@]}"; printf '\n'
"$PYTHON_BIN" "${args[@]}"
echo "Finish: $(date -Is)"
