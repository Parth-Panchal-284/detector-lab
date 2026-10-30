#!/usr/bin/env bash
# Download the exact COCO/KITTI folders expected by the ngdet tutorial.
# Run from an approved transfer/login node, not inside a GPU job.

set -euo pipefail

DATA_ROOT="${1:-${DATA_ROOT:-}}"
if [[ -z "$DATA_ROOT" ]]; then
    echo "Usage: $0 /path/to/persistent-or-scratch/detector-lab-data" >&2
    echo "Choose a location with enough quota and at least about 30 GB free." >&2
    exit 2
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 2; }

DOWNLOAD_DIR="$DATA_ROOT/downloads"
mkdir -p "$DOWNLOAD_DIR" "$DATA_ROOT/coco" "$DATA_ROOT/kitti"

fetch() {
    local url="$1" destination="$2"
    if [[ -s "$destination" ]] && unzip -tq "$destination" >/dev/null 2>&1; then
        echo "Using verified archive: $destination"
    else
        echo "Downloading $url"
        curl --fail --location --retry 5 --retry-delay 3 --continue-at - \
            --output "$destination" "$url"
    fi
}

unpack() {
    local archive="$1" destination="$2"
    echo "Checking archive: $archive"
    unzip -tq "$archive" >/dev/null
    echo "Extracting into $destination"
    unzip -oq "$archive" -d "$destination"
}

# COCO 2017 validation images and instance annotations.
if [[ ! -d "$DATA_ROOT/coco/val2017" || ! -f "$DATA_ROOT/coco/annotations/instances_val2017.json" ]]; then
    fetch "https://images.cocodataset.org/zips/val2017.zip" \
        "$DOWNLOAD_DIR/coco_val2017.zip"
    fetch "https://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
        "$DOWNLOAD_DIR/coco_annotations_trainval2017.zip"
    unpack "$DOWNLOAD_DIR/coco_val2017.zip" "$DATA_ROOT/coco"
    unpack "$DOWNLOAD_DIR/coco_annotations_trainval2017.zip" "$DATA_ROOT/coco"
else
    echo "COCO validation images and annotations already exist; skipping extraction."
fi

# KITTI 2D object-detection training images and labels, hosted by the official
# KITTI benchmark's download endpoint. We use only the labeled training split.
if [[ ! -d "$DATA_ROOT/kitti/training/image_2" || ! -d "$DATA_ROOT/kitti/training/label_2" ]]; then
    fetch "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip" \
        "$DOWNLOAD_DIR/kitti_data_object_image_2.zip"
    fetch "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip" \
        "$DOWNLOAD_DIR/kitti_data_object_label_2.zip"
    unpack "$DOWNLOAD_DIR/kitti_data_object_image_2.zip" "$DATA_ROOT/kitti"
    unpack "$DOWNLOAD_DIR/kitti_data_object_label_2.zip" "$DATA_ROOT/kitti"
else
    echo "KITTI training images and labels already exist; skipping extraction."
fi

echo
echo "Dataset setup complete. Expected paths:"
echo "  COCO_ROOT=$DATA_ROOT/coco/val2017"
echo "  COCO_ANN=$DATA_ROOT/coco/annotations/instances_val2017.json"
echo "  KITTI_ROOT=$DATA_ROOT/kitti"
test -d "$DATA_ROOT/coco/val2017" || { echo "Missing COCO val2017 folder" >&2; exit 1; }
test -f "$DATA_ROOT/coco/annotations/instances_val2017.json" || { echo "Missing COCO annotation JSON" >&2; exit 1; }
test -d "$DATA_ROOT/kitti/training/image_2" || { echo "Missing KITTI training images" >&2; exit 1; }
test -d "$DATA_ROOT/kitti/training/label_2" || { echo "Missing KITTI training labels" >&2; exit 1; }
