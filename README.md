# Detector Lab

Reproducible zero-shot evaluation of three modern 2D object-detector families on COCO and KITTI, with accuracy and inference-speed experiments run on the SJSU HPC GPU cluster.

## Experiment question

How well do fundamentally different COCO-pretrained detector families transfer to autonomous-driving imagery without fine-tuning, and what speed-accuracy tradeoffs emerge?

The evaluation pipeline builds on the open-source [`ngdet` framework](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md) and uses its shared `driving3` taxonomy, dataset adapters, COCO-style evaluator, and latency benchmark. The experiments were run against `DeepDataMiningLearning` commit `cc64ca3ed3aa1ce7df3cf381c492bc7a9826b749`.

## Models

| Family | Model | Model spec |
| --- | --- | --- |
| Two-stage CNN | Faster R-CNN ResNet-50-FPN v2 | `torchvision:fasterrcnn_resnet50_fpn_v2` |
| One-stage CNN | YOLO11m | `yolo:yolo11m.pt` |
| Transformer | RT-DETR R50 | `hf_detr:PekingU/rtdetr_r50vd` |

All checkpoints are pretrained. No model was fine-tuned on KITTI.

## Datasets and evaluation protocol

The same three models were evaluated on two domains:

- **COCO val2017:** all 5,000 validation images.
- **KITTI train split:** 5,984 labeled images from local `training/image_2` and `training/label_2`, as resolved by the upstream evaluation harness.

The tutorial maps native dataset/model labels into the shared taxonomy:

- `vehicle`
- `person`
- `cyclist`

All reported accuracy values are therefore **driving3 COCO-style metrics**, not standard 80-class COCO leaderboard metrics.

The final full-dataset runs used:

- `MAX_IMAGES=0`
- `SCORE_THR=0.25`
- one NVIDIA A100-PCIE-40GB per Slurm task
- the same evaluation harness for all six model/dataset combinations
- native framework/model preprocessing rather than forcing one common resize

### Native inference preprocessing

| Model | Input processing / post-processing |
| --- | --- |
| Faster R-CNN | Torchvision native resize: shorter edge 800 px, maximum longer edge 1333 px; internal RPN and box-head NMS; project output filter `score_thr=0.25`. |
| YOLO11m | Ultralytics native prediction path with `imgsz=640`, minimum-rectangle padding where applicable, confidence threshold 0.25, and default NMS IoU threshold 0.7. |
| RT-DETR R50 | Hugging Face RT-DETR image processor at 640x640; query-based end-to-end prediction with score filtering and no conventional NMS stage. |

Because preprocessing is model-native, the experiment compares deployable pretrained inference pipelines rather than artificially forcing identical tensor shapes.

## Final full-dataset accuracy results

| Model | COCO mAP | KITTI mAP | Absolute drop | Relative drop |
| --- | ---: | ---: | ---: | ---: |
| Faster R-CNN | 0.4832 | 0.2346 | 0.2486 | 51.5% |
| YOLO11m | 0.4692 | 0.2247 | 0.2446 | 52.1% |
| RT-DETR R50 | **0.5014** | **0.2397** | 0.2617 | 52.2% |

### COCO object-scale performance

| Model | AP-small | AP-medium | AP-large |
| --- | ---: | ---: | ---: |
| Faster R-CNN | **0.2798** | 0.4907 | 0.6493 |
| YOLO11m | 0.2034 | 0.4900 | 0.6952 |
| RT-DETR R50 | 0.2517 | **0.5114** | **0.7103** |

### KITTI per-class AP@[0.50:0.95]

| Model | Vehicle | Person | Cyclist |
| --- | ---: | ---: | ---: |
| Faster R-CNN | 0.4479 | **0.2547** | 0.0012 |
| YOLO11m | 0.4458 | 0.2273 | 0.0008 |
| RT-DETR R50 | **0.4836** | 0.2341 | **0.0015** |

The near-zero cyclist AP across all three detector families is an important limitation of the cross-dataset evaluation. COCO represents a bicycle and rider differently from KITTI's combined `Cyclist` annotation, so the unified class mapping does not make the bounding-box definitions geometrically equivalent.

## A100 latency benchmark

End-to-end `predict()` latency includes preprocessing and post-processing.

| Model | Precision | Mean latency | P90 latency | FPS |
| --- | --- | ---: | ---: | ---: |
| Faster R-CNN | FP32 | 21.2 ms | 22.0 ms | 47.2 |
| Faster R-CNN | FP16 | 17.7 ms | 17.7 ms | 56.4 |
| YOLO11m | FP32 | **10.5 ms** | **10.8 ms** | **95.1** |
| YOLO11m | FP16 | 10.6 ms | 10.9 ms | 94.7 |
| RT-DETR R50 | FP32 | 28.6 ms | 28.7 ms | 34.9 |
| RT-DETR R50 | FP16 | 30.4 ms | 30.4 ms | 32.9 |

## Main observations

1. **Domain shift dominates architecture differences.** All three models lose approximately 51.5-52.2% of mAP when moving from COCO to KITTI without retraining.
2. **RT-DETR has the highest accuracy**, reaching 0.5014 mAP on COCO and 0.2397 on KITTI, but its KITTI advantage is small.
3. **YOLO11m has the strongest speed-accuracy tradeoff.** It reaches 95.1 FPS in FP32, about 2.0x Faster R-CNN throughput and 2.7x RT-DETR throughput, while remaining close in mAP.
4. **Small objects remain difficult.** All models are substantially weaker on small objects than large objects; YOLO11m shows the largest COCO small-vs-large gap.
5. **FP16 is not universally faster.** Faster R-CNN improved by about 20%, YOLO11m was essentially unchanged, and RT-DETR was slightly slower in this end-to-end benchmark.
6. **Taxonomy alignment does not guarantee annotation alignment.** The systematic cyclist failure shows that cross-dataset metrics can be constrained by annotation semantics and box geometry, not only visual domain shift.

## Repository structure

```text
configs/jobs.tsv                  Six model x dataset configurations
scripts/download_datasets.sh      Dataset download helper
scripts/analyze_results.py        Aggregates final metrics and creates report plots
scripts/qualitative_examples.py   Selects representative failure cases and renders comparison panels
slurm/evaluate_array.sbatch       GPU evaluation array job
slurm/latency.sbatch              FP32/FP16 latency benchmark
slurm/qualitative.sbatch          Qualitative-analysis GPU job
analysis/                         Compact final CSVs, summary, plots, and qualitative findings
qualitative/                      GT + three-model comparison panels and case-level diagnostics
RUN_LOG.md                        Job IDs, environment, failures, and final metrics
```

Large raw predictions, downloaded datasets, Slurm logs, and model checkpoints are intentionally excluded from Git.

## HPC environment

The completed runs used the SJSU HPC `gpuqs` partition and the cluster's Python 3.12 Torch runtime:

```text
Python       3.12.12
PyTorch      2.15.0+cu132
Torchvision  0.29.2+cu132
Transformers 4.57.3
Ultralytics  8.4.193
pycocotools  2.0.11
GPU          NVIDIA A100-PCIE-40GB
```

Per-run `pip freeze` and Python-version captures are stored alongside the raw HPC outputs.

## Reproducing the full evaluation

Clone this repository and the upstream DeepDataMiningLearning repository as siblings, then set the dataset paths:

```bash
git clone https://github.com/Parth-Panchal-284/detector-lab.git
git clone https://github.com/lkk688/DeepDataMiningLearning.git

cd DeepDataMiningLearning
git checkout cc64ca3ed3aa1ce7df3cf381c492bc7a9826b749
cd ../detector-lab
mkdir -p logs/slurm results_full

export TUTORIAL_ROOT="$PWD/../DeepDataMiningLearning"
export COCO_ROOT="$PWD/../datasets/coco/val2017"
export COCO_ANN="$PWD/../datasets/coco/annotations/instances_val2017.json"
export KITTI_ROOT="$PWD/../datasets/kitti"
export MAX_IMAGES=0
export SCORE_THR=0.25
export RESULTS_DIR="$PWD/results_full"

sbatch --array=0-5%3 slurm/evaluate_array.sbatch
```

The evaluation wrapper performs a real CUDA smoke test and verifies that every successful Slurm task creates a fresh `metrics.json`. This prevents a model-level CUDA failure from being mistaken for a successful evaluation.

## Reproducing latency

```bash
cd detector-lab

export TUTORIAL_ROOT="$PWD/../DeepDataMiningLearning"
export KITTI_ROOT="$PWD/../datasets/kitti"

sbatch slurm/latency.sbatch
```

The successful benchmark was Slurm job `99987`.

## Generating final tables and plots

Use the same Python 3.12 interpreter as the GPU jobs. The login shell's plain `python` may point to an older interpreter that cannot import the installed Matplotlib version.

```bash
cd detector-lab

/app/rl9.x/ml/torch/2.6/bin/python \
  scripts/analyze_results.py \
  --results-dir results_full \
  --latency-json \
  ../DeepDataMiningLearning/DeepDataMiningLearning/ngdet/output/latency.json \
  --out-dir analysis
```

Generated report artifacts:

```text
analysis/full_metrics.csv
analysis/latency_metrics.csv
analysis/summary.md
analysis/map_coco_vs_kitti.png
analysis/generalization_drop.png
analysis/kitti_per_class_ap.png
analysis/object_scale_coco.png
analysis/object_scale_kitti.png
analysis/speed_accuracy_fp32.png
```

See `RUN_LOG.md` for the successful job IDs and the cluster failures encountered during the full-dataset evaluation.
