# Run log

Fill this in from the Slurm logs and generated `metrics.json` files. Keep the raw outputs in your HPC results directory and copy only the compact files needed for the report to your submission location.

## Environment

| Item | Value |
| --- | --- |
| HPC partition / GPU model | Pending |
| Python / PyTorch / torchvision | Pending (captured per run under `results/`) |
| Ultralytics / Transformers | Pending |
| Dataset locations and versions | Pending |
| Taxonomy | `driving3` (vehicle, person, cyclist), verify against harness default |
| Score threshold | `0.25` |

## Evaluation runs

| Slurm job | Task | Model | Dataset / split | Images | Status | mAP@[.50:.95] | AP50 | AP75 | Notes |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| Pending | 0 | Faster R-CNN R50-FPN v2 | COCO val2017 | Pending | Pending | | | | |
| Pending | 1 | YOLO11m | COCO val2017 | Pending | Pending | | | | |
| Pending | 2 | RT-DETR R50 | COCO val2017 | Pending | Pending | | | | |
| Pending | 3 | Faster R-CNN R50-FPN v2 | KITTI training split | Pending | Pending | | | | |
| Pending | 4 | YOLO11m | KITTI training split | Pending | Pending | | | | |
| Pending | 5 | RT-DETR R50 | KITTI training split | Pending | Pending | | | | |

## Latency run

| Slurm job | GPU | Model | Precision | Mean / p50 / p90 (ms) | FPS | Notes |
| --- | --- | --- | --- | --- | ---: | --- |
| Pending | Pending | Faster R-CNN / YOLO11m / RT-DETR | FP32, FP16 | Pending | Pending | |

## Deviations and failures

Record changed image limits, thresholds, failed/restarted jobs, dataset issues, and any differences from the planned protocol here.
