#!/usr/bin/env python3
"""Generate reproducible qualitative detector-comparison panels.

The script uses the DeepDataMiningLearning repository's ngdet adapters so ground truth and predictions
share the same driving3 taxonomy. It selects representative images from COCO
and KITTI using only ground-truth composition, then runs Faster R-CNN,
YOLO11m, and RT-DETR on exactly the same images.

Outputs
-------
<out-dir>/selection.json
    Which dataset indices were selected and why.
<out-dir>/qualitative_metrics.csv
    Per-image TP/FP/FN, precision, recall, and F1 at class-aware IoU >= 0.50.
<out-dir>/*.png
    Four-column panels: Ground truth | Faster R-CNN | YOLO11m | RT-DETR.

These per-image IoU metrics are diagnostic only. The official quantitative
results remain the full COCO-style evaluation in results_full/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_SPECS = [
    (
        "fasterrcnn",
        "Faster R-CNN",
        "torchvision:fasterrcnn_resnet50_fpn_v2",
    ),
    (
        "yolo11m",
        "YOLO11m",
        "yolo:yolo11m.pt",
    ),
    (
        "rtdetr",
        "RT-DETR R50",
        "hf_detr:PekingU/rtdetr_r50vd",
    ),
]

SCENARIOS = ("cyclist", "person", "small", "crowded")
CLASS_COLORS = {
    "vehicle": (20, 120, 255),
    "person": (255, 80, 80),
    "cyclist": (20, 180, 90),
}
GT_COLOR = (255, 210, 0)
TEXT_COLOR = (245, 245, 245)
BG_COLOR = (22, 22, 22)


@dataclass
class Candidate:
    dataset: str
    index: int
    gt_count: int
    small_count: int
    class_counts: Dict[str, int]
    score_by_scenario: Dict[str, float]


@dataclass
class SimpleDetection:
    boxes: np.ndarray
    scores: np.ndarray
    names: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create qualitative GT/model comparison panels."
    )
    p.add_argument(
        "--tutorial-root",
        required=True,
        help="DeepDataMiningLearning repository root.",
    )
    p.add_argument("--coco-root", required=True)
    p.add_argument("--coco-ann", required=True)
    p.add_argument("--kitti-root", required=True)
    p.add_argument("--out-dir", default="qualitative")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument(
        "--scan-limit",
        type=int,
        default=1500,
        help=(
            "Maximum images scanned per dataset for candidate selection. "
            "Use 0 to scan the entire dataset."
        ),
    )
    p.add_argument(
        "--panel-width",
        type=int,
        default=520,
        help="Width of each column in the saved comparison panels.",
    )
    return p.parse_args()


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def candidate_from_sample(dataset: str, index: int, sample, taxonomy) -> Candidate:
    names = [taxonomy.classes[int(v)] for v in sample.gt_labels]
    counts = Counter(names)
    small_count = sum(box_area(b) < (32.0 * 32.0) for b in sample.gt_boxes)
    gt_count = len(sample.gt_boxes)

    # Large multipliers make each scenario's defining property dominate the
    # tie-breakers. Selection uses GT only, so model failures do not influence
    # which examples are shown.
    scores = {
        "cyclist": counts.get("cyclist", 0) * 10000 + gt_count,
        "person": counts.get("person", 0) * 1000 + gt_count,
        "small": small_count * 1000 + gt_count,
        "crowded": gt_count,
    }
    return Candidate(
        dataset=dataset,
        index=index,
        gt_count=gt_count,
        small_count=small_count,
        class_counts={k: int(v) for k, v in counts.items()},
        score_by_scenario=scores,
    )


def select_candidates(dataset_name: str, dataset, taxonomy, scan_limit: int):
    n = len(dataset)
    n_scan = n if scan_limit <= 0 else min(n, scan_limit)
    candidates: List[Candidate] = []

    print(f"Scanning {n_scan}/{n} {dataset_name} images for qualitative cases...")
    for i in range(n_scan):
        sample = dataset[i]
        candidates.append(candidate_from_sample(dataset_name, i, sample, taxonomy))
        if (i + 1) % 250 == 0 or i + 1 == n_scan:
            print(f"  {dataset_name}: scanned {i + 1}/{n_scan}")

    selected: Dict[str, Candidate] = {}
    used = set()
    for scenario in SCENARIOS:
        ranked = sorted(
            candidates,
            key=lambda c: (
                c.score_by_scenario[scenario],
                c.gt_count,
                -c.index,
            ),
            reverse=True,
        )

        # Prefer a semantically valid example for class-specific scenarios.
        chosen = None
        for c in ranked:
            if c.index in used:
                continue
            if scenario in ("cyclist", "person"):
                if c.class_counts.get(scenario, 0) <= 0:
                    continue
            if scenario == "small" and c.small_count <= 0:
                continue
            chosen = c
            break

        # Graceful fallback if a limited scan did not contain that condition.
        if chosen is None:
            chosen = next((c for c in ranked if c.index not in used), ranked[0])

        selected[scenario] = chosen
        used.add(chosen.index)

    return selected


def copy_detection(det) -> SimpleDetection:
    return SimpleDetection(
        boxes=np.asarray(det.boxes, dtype=np.float32).reshape(-1, 4).copy(),
        scores=np.asarray(det.scores, dtype=np.float32).reshape(-1).copy(),
        names=[str(v) for v in det.names],
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_detection(sample, det: SimpleDetection, taxonomy, iou_thr: float = 0.5):
    gt_boxes = np.asarray(sample.gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_names = [taxonomy.classes[int(v)] for v in sample.gt_labels]

    order = np.argsort(-det.scores) if len(det.scores) else np.array([], dtype=int)
    matched_gt = set()
    tp = 0
    fp = 0

    for pi in order:
        pname = det.names[int(pi)]
        candidates = [
            gi
            for gi, gname in enumerate(gt_names)
            if gname == pname and gi not in matched_gt
        ]
        if not candidates:
            fp += 1
            continue

        overlaps = iou_matrix(det.boxes[[pi]], gt_boxes[candidates])[0]
        best_local = int(np.argmax(overlaps))
        best_iou = float(overlaps[best_local])
        if best_iou >= iou_thr:
            matched_gt.add(candidates[best_local])
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def clipped_box(box, width: int, height: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def label_box(draw, xyxy, text: str, color, font):
    x1, y1, x2, y2 = xyxy
    draw.rectangle(xyxy, outline=color, width=3)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ty = max(0, y1 - th - 6)
    draw.rectangle((x1, ty, x1 + tw + 8, ty + th + 6), fill=color)
    draw.text((x1 + 4, ty + 3), text, fill=(0, 0, 0), font=font)


def draw_ground_truth(image: Image.Image, sample, taxonomy) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(max(12, int(min(out.size) * 0.018)))
    for box, label in zip(sample.gt_boxes, sample.gt_labels):
        name = taxonomy.classes[int(label)]
        xyxy = clipped_box(box, *out.size)
        label_box(draw, xyxy, f"GT {name}", GT_COLOR, font)
    return out


def draw_predictions(image: Image.Image, det: SimpleDetection) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(max(12, int(min(out.size) * 0.018)))
    for box, score, name in zip(det.boxes, det.scores, det.names):
        color = CLASS_COLORS.get(name, (190, 190, 190))
        xyxy = clipped_box(box, *out.size)
        label_box(draw, xyxy, f"{name} {float(score):.2f}", color, font)
    return out


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    w, h = image.size
    new_h = max(1, int(round(h * width / w)))
    return image.resize((width, new_h), Image.Resampling.LANCZOS)


def add_header(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header_h = 66
    out = Image.new("RGB", (image.width, image.height + header_h), BG_COLOR)
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    title_font = load_font(21)
    sub_font = load_font(15)
    draw.text((12, 7), title, fill=TEXT_COLOR, font=title_font)
    if subtitle:
        draw.text((12, 36), subtitle, fill=(200, 200, 200), font=sub_font)
    return out


def make_panel(
    sample,
    taxonomy,
    model_outputs: Dict[str, SimpleDetection],
    metrics: Dict[str, Dict[str, float]],
    scenario: str,
    dataset_name: str,
    index: int,
    panel_width: int,
) -> Image.Image:
    cols = []

    gt = resize_to_width(draw_ground_truth(sample.image, sample, taxonomy), panel_width)
    gt_counts = Counter(taxonomy.classes[int(v)] for v in sample.gt_labels)
    gt_sub = ", ".join(f"{k}={v}" for k, v in sorted(gt_counts.items())) or "no driving3 GT"
    cols.append(add_header(gt, "Ground truth", gt_sub))

    for key, display_name, _ in MODEL_SPECS:
        det = model_outputs[key]
        m = metrics[key]
        rendered = resize_to_width(draw_predictions(sample.image, det), panel_width)
        subtitle = (
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
            f"P={m['precision']:.2f} R={m['recall']:.2f}"
        )
        cols.append(add_header(rendered, display_name, subtitle))

    max_h = max(c.height for c in cols)
    canvas = Image.new("RGB", (panel_width * len(cols), max_h + 54), BG_COLOR)
    for i, col in enumerate(cols):
        canvas.paste(col, (i * panel_width, 54))

    draw = ImageDraw.Draw(canvas)
    header_font = load_font(24)
    draw.text(
        (14, 12),
        f"{dataset_name.upper()} | {scenario} case | eval index {index}",
        fill=TEXT_COLOR,
        font=header_font,
    )
    return canvas


def write_selection(path: Path, selected_by_dataset: Dict[str, Dict[str, Candidate]]):
    payload = {}
    for dataset_name, selected in selected_by_dataset.items():
        payload[dataset_name] = {}
        for scenario, c in selected.items():
            payload[dataset_name][scenario] = {
                "eval_index": c.index,
                "gt_count": c.gt_count,
                "small_count": c.small_count,
                "class_counts": c.class_counts,
                "selection_score": c.score_by_scenario[scenario],
            }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    tutorial_root = str(Path(args.tutorial_root).resolve())
    if tutorial_root not in sys.path:
        sys.path.insert(0, tutorial_root)

    # Import after inserting the DeepDataMiningLearning repository root into sys.path.
    import torch
    from DeepDataMiningLearning.ngdet.datasets import EvalDataset
    from DeepDataMiningLearning.ngdet.detectors.base import build_detector
    from DeepDataMiningLearning.ngdet.taxonomy import Taxonomy

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = Taxonomy.from_preset("driving3")
    datasets = {
        "coco": EvalDataset(
            "coco",
            args.coco_root,
            taxonomy,
            max_images=None,
            ann_file=args.coco_ann,
        ),
        "kitti": EvalDataset(
            "kitti",
            args.kitti_root,
            taxonomy,
            max_images=None,
        ),
    }

    selected_by_dataset = {
        name: select_candidates(name, ds, taxonomy, args.scan_limit)
        for name, ds in datasets.items()
    }
    write_selection(out_dir / "selection.json", selected_by_dataset)

    # Materialize only the selected samples. This avoids re-reading images for
    # each model and guarantees every detector receives exactly the same PIL RGB.
    samples = {}
    for dataset_name, selected in selected_by_dataset.items():
        for scenario, candidate in selected.items():
            samples[(dataset_name, scenario)] = datasets[dataset_name][candidate.index]

    predictions: Dict[Tuple[str, str], Dict[str, SimpleDetection]] = {
        key: {} for key in samples
    }

    for model_key, display_name, model_spec in MODEL_SPECS:
        print(f"\nLoading {display_name}: {model_spec}")
        detector = build_detector(
            model_spec,
            taxonomy,
            device=args.device,
            score_thr=args.score_thr,
        )

        for sample_key, sample in samples.items():
            dataset_name, scenario = sample_key
            print(
                f"  {display_name:<14} {dataset_name:<5} "
                f"{scenario:<8} idx={sample.image_id}"
            )
            predictions[sample_key][model_key] = copy_detection(
                detector.predict(sample.image)
            )

        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = out_dir / "qualitative_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "scenario",
                "eval_index",
                "model",
                "num_gt",
                "num_pred",
                "tp",
                "fp",
                "fn",
                "precision_iou50",
                "recall_iou50",
                "f1_iou50",
            ],
        )
        writer.writeheader()

        for (dataset_name, scenario), sample in samples.items():
            model_metrics = {}
            for model_key, display_name, _ in MODEL_SPECS:
                det = predictions[(dataset_name, scenario)][model_key]
                m = match_detection(sample, det, taxonomy, iou_thr=0.5)
                model_metrics[model_key] = m
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "scenario": scenario,
                        "eval_index": sample.image_id,
                        "model": display_name,
                        "num_gt": len(sample.gt_boxes),
                        "num_pred": len(det.boxes),
                        "tp": m["tp"],
                        "fp": m["fp"],
                        "fn": m["fn"],
                        "precision_iou50": f"{m['precision']:.6f}",
                        "recall_iou50": f"{m['recall']:.6f}",
                        "f1_iou50": f"{m['f1']:.6f}",
                    }
                )

            panel = make_panel(
                sample=sample,
                taxonomy=taxonomy,
                model_outputs=predictions[(dataset_name, scenario)],
                metrics=model_metrics,
                scenario=scenario,
                dataset_name=dataset_name,
                index=sample.image_id,
                panel_width=args.panel_width,
            )
            panel_path = out_dir / (
                f"{dataset_name}_{scenario}_idx{sample.image_id:05d}.png"
            )
            panel.save(panel_path, quality=95)
            print(f"Saved {panel_path}")

    print(f"\nSaved selection metadata: {out_dir / 'selection.json'}")
    print(f"Saved diagnostic metrics: {csv_path}")
    print("Qualitative analysis complete.")


if __name__ == "__main__":
    main()
