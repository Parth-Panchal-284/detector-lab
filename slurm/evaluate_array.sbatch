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

: "${TUTORIAL_ROOT:?Set TUTORIAL_ROOT to the DeepDataMiningLearning repository root}"
: "${COCO_ROOT:?Set COCO_ROOT to the COCO val2017 image directory}"
: "${COCO_ANN:?Set COCO_ANN to instances_val2017.json}"
: "${KITTI_ROOT:?Set KITTI_ROOT to the KITTI root}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from detector-lab}"
PYTHON_BIN="${PYTHON_BIN:-/app/rl9.x/ml/torch/2.6/bin/python}"
PYTHON_PACKAGES="${PYTHON_PACKAGES:-$PROJECT_ROOT/../python-packages}"

export PYTHONPATH="$PYTHON_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"

MAX_IMAGES="${MAX_IMAGES:-150}"
SCORE_THR="${SCORE_THR:-0.25}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"

IFS=$'\t' read -r \
    task_id \
    model_tag \
    model_spec \
    dataset < <(
        awk \
            -F '\t' \
            -v task="$SLURM_ARRAY_TASK_ID" \
            '$1 !~ /^#/ && $1 == task {
                print $1 "\t" $2 "\t" $3 "\t" $4
            }' \
            "$PROJECT_ROOT/configs/jobs.tsv"
    )

if [[ -z "${model_spec:-}" || -z "${dataset:-}" ]]; then
    echo \
        "No job configuration for array task " \
        "${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

cd "$TUTORIAL_ROOT"

OUT_DIR="$RESULTS_DIR/${model_tag}_${dataset}"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" --version \
    > "$OUT_DIR/python-version.txt" 2>&1

"$PYTHON_BIN" -m pip freeze \
    > "$OUT_DIR/python-packages.txt"

args=(
    -m
    DeepDataMiningLearning.ngdet.run_eval
    --models
    "$model_spec"
    --datasets
    "$dataset"
    --max-images
    "$MAX_IMAGES"
    --score-thr
    "$SCORE_THR"
    --device
    cuda
    --out-dir
    "$OUT_DIR"
)

if [[ "$dataset" == "coco" ]]; then
    args+=(
        --coco-ann
        "$COCO_ANN"
    )
fi

args+=(
    --roots
    "coco=$COCO_ROOT"
    "kitti=$KITTI_ROOT"
)

echo "Start: $(date -Is)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" -c '
import sys
import torch
import torchvision

print("python", sys.version)
print("torch", torch.__version__, torch.__file__)
print(
    "torchvision",
    torchvision.__version__,
    torchvision.__file__,
)
print("cuda", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("NO CUDA")
'

# A simple is_available() check is not sufficient on the
# cluster. Some unhealthy GPUs are visible to PyTorch but
# fail when CUDA actually creates a context.
echo "Running CUDA smoke test..."

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "This Slurm task has no visible CUDA GPU"
    )

device = torch.device("cuda:0")

# Force CUDA context creation and execute real GPU work.
x = torch.ones(1024, device=device)
y = (x * 2.0).sum()

torch.cuda.synchronize()

print(
    "CUDA smoke test OK:",
    torch.cuda.get_device_name(0),
    "result=",
    float(y.item()),
)
PY

# Record the start time immediately before evaluation so
# an old metrics.json cannot make a failed rerun look valid.
EVAL_START_EPOCH="$(date +%s)"

printf 'Command:'
printf ' %q' "$PYTHON_BIN" "${args[@]}"
printf '\n'

"$PYTHON_BIN" "${args[@]}"

METRICS_FILE="$OUT_DIR/metrics.json"

# run_eval currently catches per-model exceptions, so it can
# exit with code 0 even if inference itself failed. Verify that
# this invocation actually produced a fresh metrics file.
if [[ ! -s "$METRICS_FILE" ]]; then
    echo \
        "ERROR: evaluation finished without metrics.json:" \
        "$METRICS_FILE" >&2
    exit 3
fi

METRICS_MTIME="$(stat -c %Y "$METRICS_FILE")"

if (( METRICS_MTIME < EVAL_START_EPOCH )); then
    echo \
        "ERROR: metrics.json was not updated by this run:" \
        "$METRICS_FILE" >&2
    exit 4
fi

echo "Verified fresh metrics: $METRICS_FILE"
echo "Finish: $(date -Is)"
