#!/usr/bin/env python3
"""Aggregate detector-lab full-evaluation results and make report figures.

Expected result layout:

results_full/
  fasterrcnn_coco/metrics.json
  fasterrcnn_kitti/metrics.json
  yolo11m_coco/metrics.json
  yolo11m_kitti/metrics.json
  rtdetr_coco/metrics.json
  rtdetr_kitti/metrics.json

The latency JSON is optional. If provided, the script also creates a
speed-vs-accuracy figure and latency summary table. The dedicated latency
benchmark should be used for speed analysis; the per-run `_meta.seconds`
values are intentionally not treated as latency measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MODEL_ORDER = ["fasterrcnn", "yolo11m", "rtdetr"]
DATASET_ORDER = ["coco", "kitti"]
CLASS_ORDER = ["vehicle", "person", "cyclist"]

MODEL_LABELS = {
    "fasterrcnn": "Faster R-CNN",
    "yolo11m": "YOLO11m",
    "rtdetr": "RT-DETR R50",
}

MODEL_SPECS = {
    "fasterrcnn": "torchvision:fasterrcnn_resnet50_fpn_v2",
    "yolo11m": "yolo:yolo11m.pt",
    "rtdetr": "hf_detr:PekingU/rtdetr_r50vd",
}


SUMMARY_FIELDS = [
    "model_key",
    "model",
    "model_spec",
    "dataset",
    "num_images",
    "score_thr",
    "mAP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
    "AP_vehicle",
    "AP_person",
    "AP_cyclist",
]


LATENCY_FIELDS = [
    "model_key",
    "model",
    "accel",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "fps",
    "speedup",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate detector-lab evaluation metrics and generate "
            "report-ready plots."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_full"),
        help="Directory containing the six full result folders.",
    )
    parser.add_argument(
        "--latency-json",
        type=Path,
        default=None,
        help=(
            "Optional latency.json from the dedicated ngdet latency "
            "benchmark."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis"),
        help="Output directory for tables and plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_model_key(folder_name: str) -> str:
    for key in MODEL_ORDER:
        if folder_name.startswith(f"{key}_"):
            return key
    raise ValueError(
        f"Cannot infer model key from result folder: {folder_name}"
    )


def infer_dataset(folder_name: str) -> str:
    for dataset in DATASET_ORDER:
        if folder_name.endswith(f"_{dataset}"):
            return dataset
    raise ValueError(
        f"Cannot infer dataset from result folder: {folder_name}"
    )


def validate_metric_record(
    record: dict[str, Any],
    model_key: str,
    dataset: str,
    path: Path,
) -> None:
    required = [
        "mAP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
        "per_class",
        "_meta",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            f"{path} is missing required keys: {missing}"
        )

    meta = record["_meta"]
    if meta.get("dataset") != dataset:
        raise ValueError(
            f"Dataset mismatch in {path}: expected {dataset}, "
            f"found {meta.get('dataset')}"
        )

    expected_spec = MODEL_SPECS[model_key]
    if meta.get("model") != expected_spec:
        raise ValueError(
            f"Model mismatch in {path}: expected {expected_spec}, "
            f"found {meta.get('model')}"
        )

    missing_classes = [
        name for name in CLASS_ORDER
        if name not in record["per_class"]
    ]
    if missing_classes:
        raise ValueError(
            f"{path} is missing per-class AP values: "
            f"{missing_classes}"
        )


def load_metrics(results_dir: Path) -> list[dict[str, Any]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    rows: list[dict[str, Any]] = []

    for model_key in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            folder = results_dir / f"{model_key}_{dataset}"
            path = folder / "metrics.json"

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing required full-run metrics: {path}"
                )

            payload = load_json(path)
            if not isinstance(payload, list) or len(payload) != 1:
                raise ValueError(
                    f"Expected one metrics record in {path}; "
                    f"found {type(payload).__name__} with "
                    f"length {len(payload) if isinstance(payload, list) else 'n/a'}"
                )

            record = payload[0]
            validate_metric_record(
                record,
                model_key,
                dataset,
                path,
            )

            meta = record["_meta"]
            per_class = record["per_class"]

            row = {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "model_spec": meta["model"],
                "dataset": dataset,
                "num_images": int(meta["num_images"]),
                "score_thr": float(meta["score_thr"]),
                "mAP": float(record["mAP"]),
                "AP50": float(record["AP50"]),
                "AP75": float(record["AP75"]),
                "AP_small": float(record["AP_small"]),
                "AP_medium": float(record["AP_medium"]),
                "AP_large": float(record["AP_large"]),
                "AR_1": float(record["AR_1"]),
                "AR_10": float(record["AR_10"]),
                "AR_100": float(record["AR_100"]),
                "AR_small": float(record["AR_small"]),
                "AR_medium": float(record["AR_medium"]),
                "AR_large": float(record["AR_large"]),
                "AP_vehicle": float(per_class["vehicle"]),
                "AP_person": float(per_class["person"]),
                "AP_cyclist": float(per_class["cyclist"]),
            }
            rows.append(row)

    return rows


def latency_model_key(model_name: str) -> str | None:
    if model_name.startswith("torchvision:fasterrcnn"):
        return "fasterrcnn"
    if model_name.startswith("yolo:yolo11m"):
        return "yolo11m"
    if model_name.startswith("hf_detr:"):
        return "rtdetr"
    return None


def find_latency_json(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit

    candidates = [
        Path("latency.json"),
        Path("analysis_inputs/latency.json"),
        Path(
            "../DeepDataMiningLearning/"
            "DeepDataMiningLearning/ngdet/output/latency.json"
        ),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def load_latency(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(
            f"Latency JSON does not exist: {path}"
        )

    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected latency JSON to contain a list: {path}"
        )

    rows: list[dict[str, Any]] = []
    for item in payload:
        model_name = str(item.get("model", ""))
        model_key = latency_model_key(model_name)
        if model_key is None:
            continue

        rows.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "accel": str(item["accel"]),
                "mean_ms": float(item["mean_ms"]),
                "p50_ms": float(item["p50_ms"]),
                "p90_ms": float(item["p90_ms"]),
                "fps": float(item["fps"]),
                "speedup": float(item["speedup"]),
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def metric_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["model_key"], row["dataset"]): row
        for row in rows
    }


def latency_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["model_key"], row["accel"]): row
        for row in rows
    }


def write_markdown_summary(
    path: Path,
    metrics: list[dict[str, Any]],
    latency: list[dict[str, Any]],
) -> None:
    lookup = metric_lookup(metrics)
    lat_lookup = latency_lookup(latency)

    lines: list[str] = []
    lines.append("# Full-dataset detector evaluation summary")
    lines.append("")
    lines.append(
        "All accuracy values use the unified driving3 taxonomy "
        "(vehicle, person, cyclist) and COCO-style evaluation."
    )
    lines.append("")
    lines.append("## Main accuracy results")
    lines.append("")
    lines.append(
        "| Model | COCO mAP | KITTI mAP | Absolute drop | "
        "Relative drop |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    for model_key in MODEL_ORDER:
        coco = lookup[(model_key, "coco")]["mAP"]
        kitti = lookup[(model_key, "kitti")]["mAP"]
        abs_drop = coco - kitti
        rel_drop = abs_drop / coco if coco else math.nan
        lines.append(
            f"| {MODEL_LABELS[model_key]} | {coco:.4f} | "
            f"{kitti:.4f} | {abs_drop:.4f} | "
            f"{rel_drop * 100:.1f}% |"
        )

    lines.append("")
    lines.append("## COCO full-set metrics")
    lines.append("")
    lines.append(
        "| Model | mAP | AP50 | AP75 | AP-S | AP-M | AP-L |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model_key in MODEL_ORDER:
        row = lookup[(model_key, "coco")]
        lines.append(
            f"| {row['model']} | {row['mAP']:.4f} | "
            f"{row['AP50']:.4f} | {row['AP75']:.4f} | "
            f"{row['AP_small']:.4f} | "
            f"{row['AP_medium']:.4f} | "
            f"{row['AP_large']:.4f} |"
        )

    lines.append("")
    lines.append("## KITTI full-set metrics")
    lines.append("")
    lines.append(
        "| Model | mAP | AP50 | AP75 | AP-S | AP-M | AP-L |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model_key in MODEL_ORDER:
        row = lookup[(model_key, "kitti")]
        lines.append(
            f"| {row['model']} | {row['mAP']:.4f} | "
            f"{row['AP50']:.4f} | {row['AP75']:.4f} | "
            f"{row['AP_small']:.4f} | "
            f"{row['AP_medium']:.4f} | "
            f"{row['AP_large']:.4f} |"
        )

    lines.append("")
    lines.append("## KITTI per-class AP50:95")
    lines.append("")
    lines.append("| Model | Vehicle | Person | Cyclist |")
    lines.append("|---|---:|---:|---:|")
    for model_key in MODEL_ORDER:
        row = lookup[(model_key, "kitti")]
        lines.append(
            f"| {row['model']} | {row['AP_vehicle']:.4f} | "
            f"{row['AP_person']:.4f} | "
            f"{row['AP_cyclist']:.4f} |"
        )

    if latency:
        lines.append("")
        lines.append("## Dedicated A100 latency benchmark")
        lines.append("")
        lines.append(
            "| Model | Precision | Mean ms | P90 ms | FPS | Speedup |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for model_key in MODEL_ORDER:
            for accel in ("fp32", "fp16"):
                row = lat_lookup.get((model_key, accel))
                if row is None:
                    continue
                lines.append(
                    f"| {row['model']} | {accel} | "
                    f"{row['mean_ms']:.1f} | "
                    f"{row['p90_ms']:.1f} | "
                    f"{row['fps']:.1f} | "
                    f"x{row['speedup']:.2f} |"
                )

    lines.append("")
    lines.append("## Key quantitative observations")
    lines.append("")

    drops = []
    for model_key in MODEL_ORDER:
        coco = lookup[(model_key, "coco")]["mAP"]
        kitti = lookup[(model_key, "kitti")]["mAP"]
        drops.append((coco - kitti) / coco * 100.0)

    lines.append(
        "- All three models lose roughly "
        f"{min(drops):.1f}% to {max(drops):.1f}% of mAP when "
        "moving from COCO to KITTI without retraining."
    )

    coco_best = max(
        MODEL_ORDER,
        key=lambda key: lookup[(key, "coco")]["mAP"],
    )
    kitti_best = max(
        MODEL_ORDER,
        key=lambda key: lookup[(key, "kitti")]["mAP"],
    )
    lines.append(
        f"- {MODEL_LABELS[coco_best]} has the highest COCO mAP "
        f"({lookup[(coco_best, 'coco')]['mAP']:.4f}); "
        f"{MODEL_LABELS[kitti_best]} also has the highest KITTI "
        f"mAP ({lookup[(kitti_best, 'kitti')]['mAP']:.4f})."
    )

    kitti_cyclist = [
        lookup[(key, "kitti")]["AP_cyclist"]
        for key in MODEL_ORDER
    ]
    lines.append(
        "- KITTI cyclist AP is near zero for every detector "
        f"({min(kitti_cyclist):.4f} to "
        f"{max(kitti_cyclist):.4f}), which should be discussed "
        "as a cross-dataset taxonomy/box-definition limitation."
    )

    if latency:
        fp32_rows = [
            lat_lookup[(key, "fp32")]
            for key in MODEL_ORDER
            if (key, "fp32") in lat_lookup
        ]
        if fp32_rows:
            fastest = max(fp32_rows, key=lambda row: row["fps"])
            lines.append(
                f"- {fastest['model']} is the fastest FP32 model "
                f"in the dedicated A100 benchmark at "
                f"{fastest['fps']:.1f} FPS."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_map_by_dataset(
    out_dir: Path,
    metrics: list[dict[str, Any]],
) -> None:
    lookup = metric_lookup(metrics)
    x = list(range(len(MODEL_ORDER)))
    width = 0.36

    coco = [lookup[(key, "coco")]["mAP"] for key in MODEL_ORDER]
    kitti = [lookup[(key, "kitti")]["mAP"] for key in MODEL_ORDER]

    plt.figure(figsize=(8.0, 4.8))
    plt.bar(
        [value - width / 2 for value in x],
        coco,
        width,
        label="COCO",
    )
    plt.bar(
        [value + width / 2 for value in x],
        kitti,
        width,
        label="KITTI",
    )
    plt.xticks(x, [MODEL_LABELS[key] for key in MODEL_ORDER])
    plt.ylabel("mAP@[0.50:0.95]")
    plt.title("Full-dataset accuracy under domain shift")
    plt.ylim(0.0, max(coco) * 1.18)
    plt.legend()

    for index, value in enumerate(coco):
        plt.text(
            index - width / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )
    for index, value in enumerate(kitti):
        plt.text(
            index + width / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    save_figure(out_dir / "map_coco_vs_kitti.png")


def plot_generalization_drop(
    out_dir: Path,
    metrics: list[dict[str, Any]],
) -> None:
    lookup = metric_lookup(metrics)
    drops = []
    for key in MODEL_ORDER:
        coco = lookup[(key, "coco")]["mAP"]
        kitti = lookup[(key, "kitti")]["mAP"]
        drops.append((coco - kitti) / coco * 100.0)

    plt.figure(figsize=(7.2, 4.6))
    bars = plt.bar(
        [MODEL_LABELS[key] for key in MODEL_ORDER],
        drops,
    )
    plt.ylabel("Relative mAP drop (%)")
    plt.title("COCO to KITTI zero-shot generalization gap")
    plt.ylim(0.0, max(drops) * 1.18)

    for bar, value in zip(bars, drops):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
        )

    save_figure(out_dir / "generalization_drop.png")


def plot_kitti_per_class(
    out_dir: Path,
    metrics: list[dict[str, Any]],
) -> None:
    lookup = metric_lookup(metrics)
    x = list(range(len(MODEL_ORDER)))
    width = 0.24

    plt.figure(figsize=(8.2, 4.8))

    for class_index, class_name in enumerate(CLASS_ORDER):
        offset = (class_index - 1) * width
        values = [
            lookup[(key, "kitti")][f"AP_{class_name}"]
            for key in MODEL_ORDER
        ]
        plt.bar(
            [value + offset for value in x],
            values,
            width,
            label=class_name.capitalize(),
        )

    plt.xticks(x, [MODEL_LABELS[key] for key in MODEL_ORDER])
    plt.ylabel("AP@[0.50:0.95]")
    plt.title("KITTI per-class performance")
    plt.ylim(0.0, 0.55)
    plt.legend()

    save_figure(out_dir / "kitti_per_class_ap.png")


def plot_object_scale(
    out_dir: Path,
    metrics: list[dict[str, Any]],
) -> None:
    lookup = metric_lookup(metrics)
    scale_fields = [
        ("AP_small", "Small"),
        ("AP_medium", "Medium"),
        ("AP_large", "Large"),
    ]

    x = list(range(len(MODEL_ORDER)))
    width = 0.24

    for dataset in DATASET_ORDER:
        plt.figure(figsize=(8.2, 4.8))
        for index, (field, label) in enumerate(scale_fields):
            offset = (index - 1) * width
            values = [
                lookup[(key, dataset)][field]
                for key in MODEL_ORDER
            ]
            plt.bar(
                [value + offset for value in x],
                values,
                width,
                label=label,
            )

        plt.xticks(
            x,
            [MODEL_LABELS[key] for key in MODEL_ORDER],
        )
        plt.ylabel("AP@[0.50:0.95]")
        plt.title(f"Object-scale AP on {dataset.upper()}")
        plt.ylim(0.0, 0.80)
        plt.legend()
        save_figure(
            out_dir / f"object_scale_{dataset}.png"
        )


def plot_speed_accuracy(
    out_dir: Path,
    metrics: list[dict[str, Any]],
    latency: list[dict[str, Any]],
) -> None:
    if not latency:
        return

    metrics_lookup = metric_lookup(metrics)
    lat_lookup = latency_lookup(latency)

    points = []
    for key in MODEL_ORDER:
        latency_row = lat_lookup.get((key, "fp32"))
        if latency_row is None:
            continue
        points.append(
            (
                key,
                latency_row["fps"],
                metrics_lookup[(key, "coco")]["mAP"],
            )
        )

    if not points:
        return

    plt.figure(figsize=(7.2, 4.8))
    for key, fps, map_value in points:
        plt.scatter(fps, map_value, s=90)
        plt.annotate(
            MODEL_LABELS[key],
            (fps, map_value),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.xlabel("FP32 FPS on NVIDIA A100")
    plt.ylabel("COCO driving3 mAP@[0.50:0.95]")
    plt.title("Speed-accuracy tradeoff")

    save_figure(out_dir / "speed_accuracy_fp32.png")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.results_dir)
    latency_path = find_latency_json(args.latency_json)
    latency = load_latency(latency_path)

    write_csv(
        out_dir / "full_metrics.csv",
        metrics,
        SUMMARY_FIELDS,
    )

    if latency:
        write_csv(
            out_dir / "latency_metrics.csv",
            latency,
            LATENCY_FIELDS,
        )

    write_markdown_summary(
        out_dir / "summary.md",
        metrics,
        latency,
    )

    plot_map_by_dataset(out_dir, metrics)
    plot_generalization_drop(out_dir, metrics)
    plot_kitti_per_class(out_dir, metrics)
    plot_object_scale(out_dir, metrics)
    plot_speed_accuracy(out_dir, metrics, latency)

    print(f"Loaded 6 full evaluations from: {args.results_dir}")
    if latency_path is not None:
        print(f"Loaded latency benchmark from: {latency_path}")
    else:
        print("Latency JSON not found; skipped latency outputs.")
    print(f"Saved analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()
