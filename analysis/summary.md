# Full-dataset detector evaluation summary

All accuracy values use the unified driving3 taxonomy (vehicle, person, cyclist) and COCO-style evaluation.

## Main accuracy results

| Model | COCO mAP | KITTI mAP | Absolute drop | Relative drop |
|---|---:|---:|---:|---:|
| Faster R-CNN | 0.4832 | 0.2346 | 0.2486 | 51.5% |
| YOLO11m | 0.4692 | 0.2247 | 0.2446 | 52.1% |
| RT-DETR R50 | 0.5014 | 0.2397 | 0.2617 | 52.2% |

## COCO full-set metrics

| Model | mAP | AP50 | AP75 | AP-S | AP-M | AP-L |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 0.4832 | 0.7370 | 0.5137 | 0.2798 | 0.4907 | 0.6493 |
| YOLO11m | 0.4692 | 0.6397 | 0.5064 | 0.2034 | 0.4900 | 0.6952 |
| RT-DETR R50 | 0.5014 | 0.7259 | 0.5264 | 0.2517 | 0.5114 | 0.7103 |

## KITTI full-set metrics

| Model | mAP | AP50 | AP75 | AP-S | AP-M | AP-L |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 0.2346 | 0.4342 | 0.2339 | 0.1193 | 0.2279 | 0.3086 |
| YOLO11m | 0.2247 | 0.4123 | 0.2240 | 0.1094 | 0.2127 | 0.3027 |
| RT-DETR R50 | 0.2397 | 0.4408 | 0.2344 | 0.1299 | 0.2288 | 0.3142 |

## KITTI per-class AP50:95

| Model | Vehicle | Person | Cyclist |
|---|---:|---:|---:|
| Faster R-CNN | 0.4479 | 0.2547 | 0.0012 |
| YOLO11m | 0.4458 | 0.2273 | 0.0008 |
| RT-DETR R50 | 0.4836 | 0.2341 | 0.0015 |

## Dedicated A100 latency benchmark

| Model | Precision | Mean ms | P90 ms | FPS | Speedup |
|---|---|---:|---:|---:|---:|
| Faster R-CNN | fp32 | 21.2 | 22.0 | 47.2 | x1.00 |
| Faster R-CNN | fp16 | 17.7 | 17.7 | 56.4 | x1.20 |
| YOLO11m | fp32 | 10.5 | 10.8 | 95.1 | x1.00 |
| YOLO11m | fp16 | 10.6 | 10.9 | 94.7 | x1.00 |
| RT-DETR R50 | fp32 | 28.6 | 28.7 | 34.9 | x1.00 |
| RT-DETR R50 | fp16 | 30.4 | 30.4 | 32.9 | x0.94 |

## Key quantitative observations

- All three models lose roughly 51.5% to 52.2% of mAP when moving from COCO to KITTI without retraining.
- RT-DETR R50 has the highest COCO mAP (0.5014); RT-DETR R50 also has the highest KITTI mAP (0.2397).
- KITTI cyclist AP is near zero for every detector (0.0008 to 0.0015), which should be discussed as a cross-dataset taxonomy/box-definition limitation.
- YOLO11m is the fastest FP32 model in the dedicated A100 benchmark at 95.1 FPS.
