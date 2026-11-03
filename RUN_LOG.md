# Run log

This file records the experiments used for the final object-detection study. Raw Slurm logs and model predictions remain on HPC; compact final tables and figures are stored under `analysis/`.

## Final environment

| Item | Value |
| --- | --- |
| HPC partition | `gpuqs` |
| GPU | NVIDIA A100-PCIE-40GB |
| Python | 3.12.12 (`/app/rl9.x/ml/torch/2.6/bin/python`) |
| PyTorch | 2.15.0+cu132 |
| Torchvision | 0.29.2+cu132 |
| Transformers | 4.57.3 |
| Ultralytics | 8.4.193 |
| pycocotools | 2.0.11 |
| Matplotlib | 3.11.2 |
| Taxonomy | `driving3`: vehicle, person, cyclist |
| Prediction score threshold | 0.25 |
| COCO split | `val2017`, 5,000 images |
| KITTI split | `train`, 5,984 labeled images from local KITTI training data |
| Training / fine-tuning | None; pretrained checkpoints used zero-shot |
| Upstream framework commit | `cc64ca3ed3aa1ce7df3cf381c492bc7a9826b749` |

Package versions were captured independently in each result directory using `pip freeze`.

## Final full-dataset evaluation runs

| Slurm task | Model | Dataset | Images | Status | mAP@[.50:.95] | AP50 | AP75 | Runtime reported by evaluator |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `100615_0` | Faster R-CNN R50-FPN v2 | COCO val2017 | 5,000 | Completed | 0.4832 | 0.7370 | 0.5137 | 241.1 s |
| `100646_1` | YOLO11m | COCO val2017 | 5,000 | Completed | 0.4692 | 0.6397 | 0.5064 | 87.2 s |
| `100628_2` | RT-DETR R50 | COCO val2017 | 5,000 | Completed | **0.5014** | 0.7259 | **0.5264** | 169.9 s |
| `100628_3` | Faster R-CNN R50-FPN v2 | KITTI | 5,984 | Completed | 0.2346 | 0.4342 | 0.2339 | 197.1 s |
| `100615_4` | YOLO11m | KITTI | 5,984 | Completed | 0.2247 | 0.4123 | 0.2240 | 337.1 s |
| `100646_5` | RT-DETR R50 | KITTI | 5,984 | Completed | **0.2397** | **0.4408** | **0.2344** | 281.0 s |

The evaluator runtime above is not used as the controlled latency measurement because these full-dataset runs include dataset/evaluation overhead and were scheduled independently. The dedicated latency experiment below is the speed comparison used in the report.

## Full-dataset object-scale metrics

### COCO

| Model | AP-small | AP-medium | AP-large |
| --- | ---: | ---: | ---: |
| Faster R-CNN | 0.2798 | 0.4907 | 0.6493 |
| YOLO11m | 0.2034 | 0.4900 | 0.6952 |
| RT-DETR R50 | 0.2517 | 0.5114 | 0.7103 |

### KITTI

| Model | AP-small | AP-medium | AP-large |
| --- | ---: | ---: | ---: |
| Faster R-CNN | 0.1193 | 0.2279 | 0.3086 |
| YOLO11m | 0.1094 | 0.2127 | 0.3027 |
| RT-DETR R50 | 0.1299 | 0.2288 | 0.3142 |

## KITTI per-class AP@[.50:.95]

| Model | Vehicle | Person | Cyclist |
| --- | ---: | ---: | ---: |
| Faster R-CNN | 0.4479 | 0.2547 | 0.0012 |
| YOLO11m | 0.4458 | 0.2273 | 0.0008 |
| RT-DETR R50 | 0.4836 | 0.2341 | 0.0015 |

The systematic near-zero cyclist AP is treated as a taxonomy/annotation-geometry limitation in the final analysis: a COCO bicycle box and a KITTI combined rider+bicycle `Cyclist` box are not geometrically equivalent even after mapping both into a common label.

## Cross-dataset generalization

| Model | COCO mAP | KITTI mAP | Absolute drop | Relative drop |
| --- | ---: | ---: | ---: | ---: |
| Faster R-CNN | 0.4832 | 0.2346 | 0.2486 | 51.5% |
| YOLO11m | 0.4692 | 0.2247 | 0.2446 | 52.1% |
| RT-DETR R50 | 0.5014 | 0.2397 | 0.2617 | 52.2% |

## Dedicated latency run

Successful Slurm job: `99987`.

Device: NVIDIA A100-PCIE-40GB. The ngdet latency benchmark measures end-to-end `predict()` time including preprocessing and post-processing, with warmup followed by 40 measured iterations.

| Model | Precision | Mean ms | p50 ms | p90 ms | FPS | FP32-relative speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN | FP32 | 21.2 | 20.1 | 22.0 | 47.2 | 1.00x |
| Faster R-CNN | FP16 | 17.7 | 17.7 | 17.7 | 56.4 | 1.20x |
| YOLO11m | FP32 | 10.5 | 10.5 | 10.8 | 95.1 | 1.00x |
| YOLO11m | FP16 | 10.6 | 10.5 | 10.9 | 94.7 | 1.00x |
| RT-DETR R50 | FP32 | 28.6 | 28.6 | 28.7 | 34.9 | 1.00x |
| RT-DETR R50 | FP16 | 30.4 | 30.3 | 30.4 | 32.9 | 0.94x |

## Preliminary subset run

Before the full validation pass, all six model/dataset combinations were successfully evaluated on the first 150 images using Slurm job `99906`. Those results were useful for validating the pipeline but are not used as the final headline numbers.

The subset was noticeably optimistic on COCO:

| Experiment | 150-image mAP | Full-set mAP |
| --- | ---: | ---: |
| Faster R-CNN / COCO | 0.5707 | 0.4832 |
| RT-DETR / COCO | 0.596 | 0.5014 |
| Faster R-CNN / KITTI | ~0.235 | 0.2346 |
| YOLO11m / KITTI | ~0.229 | 0.2247 |

This motivated using the complete available evaluation splits in the final report.

## Cluster failures and reruns

The full-dataset evaluation exposed intermittent unhealthy A100 devices on the cluster. These were infrastructure failures, not detector-code failures.

### Job `100615`

- Task 0, Faster R-CNN / COCO: completed.
- Task 4, YOLO11m / KITTI: completed.
- Tasks 1, 2, and 5: failed because the allocated CUDA device was not usable/visible to PyTorch.
- Task 3: the model reached CUDA initialization but failed with `CUDA_ERROR_UNKNOWN` / device unavailable.

### Job `100628`

Reran tasks 1, 2, 3, and 5.

- Task 2, RT-DETR / COCO: completed.
- Task 3, Faster R-CNN / KITTI: completed.
- Tasks 1 and 5 landed on `cs002` GPU 0 and failed with `CUDA_ERROR_UNKNOWN`. `nvidia-smi` showed 73 volatile uncorrectable ECC errors on that physical GPU.

At this point it was also discovered that the upstream framework's `run_eval.py` catches a per-model exception and can return process exit code 0 even when the model evaluation failed. Therefore Slurm could display `COMPLETED` without a fresh `metrics.json`.

### Job `100642`

Reran tasks 1 and 5 on `cs001` after strengthening the Slurm wrapper. Both correctly failed during the new CUDA smoke test because the allocated physical GPU was unhealthy.

### Job `100646`

Reran only the two missing experiments on idle A100 node `cs003`, one at a time:

- `100646_1`: YOLO11m / COCO, completed in 2:02 Slurm elapsed time.
- `100646_5`: RT-DETR / KITTI, completed in 4:50 Slurm elapsed time.

After this job, all six `results_full/*/metrics.json` files were present.

## Slurm-wrapper robustness change

`slurm/evaluate_array.sbatch` was updated after the cluster failures to:

1. verify `torch.cuda.is_available()`;
2. allocate a CUDA tensor and synchronize to force real CUDA-context creation;
3. record the evaluation start timestamp;
4. require a non-empty `metrics.json` after `run_eval.py` returns; and
5. require that the metrics file was freshly updated by the current invocation.

This prevents a silently caught detector exception or an unhealthy GPU from being recorded as a successful experiment.

## Analysis generation

The final aggregation command was:

```bash
/app/rl9.x/ml/torch/2.6/bin/python \
  scripts/analyze_results.py \
  --results-dir results_full \
  --latency-json \
  ../DeepDataMiningLearning/DeepDataMiningLearning/ngdet/output/latency.json \
  --out-dir analysis
```

Generated files:

```text
analysis/full_metrics.csv
analysis/generalization_drop.png
analysis/kitti_per_class_ap.png
analysis/latency_metrics.csv
analysis/map_coco_vs_kitti.png
analysis/object_scale_coco.png
analysis/object_scale_kitti.png
analysis/speed_accuracy_fp32.png
analysis/summary.md
```

The plain login-shell `python` was too old for the installed Matplotlib (`match` syntax). Using the same Python 3.12 interpreter as the GPU jobs resolved this without changing the analysis code.

## Qualitative failure analysis

A separate reproducible qualitative pass selected representative scenes from both datasets using ground-truth composition only, then ran all three detectors on the same images. The script records class-aware TP/FP/FN diagnostics at IoU 0.50 and saves four-column comparison panels (ground truth, Faster R-CNN, YOLO11m, RT-DETR).

Key artifacts:

```text
scripts/qualitative_examples.py
slurm/qualitative.sbatch
analysis/qualitative_findings.md
qualitative/selection.json
qualitative/qualitative_metrics.csv
qualitative/*.png
```

The final report uses four representative cases: COCO crowded, COCO small-object, KITTI cyclist, and KITTI person. These support the written analysis of precision/recall tradeoffs, small-object misses, cyclist annotation mismatch, and clutter-driven false positives.

