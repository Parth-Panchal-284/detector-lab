# Detector Lab

Reproducible, zero-shot evaluation of modern 2D object detectors on COCO and KITTI, with runs scheduled on the SJSU HPC GPU cluster.

## Experiment question

How do three detector families compare on COCO and on the driving-domain KITTI dataset when using pretrained checkpoints without fine-tuning? We will compare accuracy and inference speed, then inspect class and object-scale errors to understand cross-domain degradation.

The experiment follows the professor's [`ngdet` tutorial](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md), which provides dataset adapters, a shared driving taxonomy, COCO-style metrics, and a report generator. The selected models are:

| Family | Model spec | Role |
| --- | --- | --- |
| Two-stage CNN | `torchvision:fasterrcnn_resnet50_fpn_v2` | Region-proposal baseline |
| One-stage CNN | `yolo:yolo11m.pt` | YOLO speed/accuracy baseline |
| Transformer | `hf_detr:PekingU/rtdetr_r50vd` | Query-based real-time detector |

Each model is evaluated on both datasets (six model/dataset runs). Checkpoints are pretrained; there is no fine-tuning in this experiment. Results cited from the professor's tutorial are background only; all result tables in the submission must come from our own HPC runs.

## Evaluation protocol

- Use the tutorial's default `driving3` unified labels: vehicle, person, cyclist. Record the COCO/KITTI-to-unified mapping and discuss taxonomy/box-definition differences.
- Use the same `--score-thr` for the three closed-set models, the same image subset per dataset, and the same evaluation code. The starter job uses 150 images per dataset for a time-bounded main comparison; first run a 20-image smoke test. If HPC allocation allows, rerun with all images (`MAX_IMAGES=0`) for stronger final numbers.
- Report mAP@[.50:.95], AP50, AP75, AP by object size, per-class AP, recall, and model/runtime details when emitted by the harness. Measure inference latency separately with the tutorial's `ngdet.latency` command on one allocated GPU.
- Archive exact commands, job logs, package versions, GPU model, and the generated `metrics.json`/report. Do not commit downloaded datasets, checkpoints, or generated result files.

## Files

- `configs/jobs.tsv`: the six model/dataset pairs scheduled by the Slurm array.
- `slurm/evaluate_array.sbatch`: one GPU per task; set dataset and tutorial-repository paths as environment variables before submission.
- `slurm/latency.sbatch`: optional same-GPU inference speed comparison across the three models.
- `scripts/download_datasets.sh`: resumable download and extraction of the required COCO/KITTI splits.
- `RUN_LOG.md`: fillable record of environment, job IDs, metrics, and deviations.

## HPC setup and run

Clone both this repository and the professor's repository on HPC. Keep datasets and outputs on the storage location allowed by the SJSU HPC instructions (often `$SCRATCH`); do not assume node-local storage persists after a job.

```bash
git clone https://github.com/Parth-Panchal-284/detector-lab.git
git clone https://github.com/lkk688/DeepDataMiningLearning.git
cd detector-lab
mkdir -p logs/slurm results
```

Create/activate a Python environment following the tutorial and the cluster's CUDA/PyTorch guidance. The tutorial lists `torch`, `torchvision`, `transformers`, `ultralytics`, `pycocotools`, `opencv-python`, `matplotlib`, and `pyarrow`. Verify that the chosen Python can import PyTorch and see the allocated CUDA device before submitting.

Set paths for your own HPC storage (examples only; replace them with your real paths):

```bash
export TUTORIAL_ROOT="$PWD/../DeepDataMiningLearning"
export COCO_ROOT="/path/to/coco/val2017"
export COCO_ANN="/path/to/coco/annotations/instances_val2017.json"
export KITTI_ROOT="/path/to/KITTI"
export MAX_IMAGES=20       # smoke test first; use 150 for the main run, 0 for all
export SCORE_THR=0.25
```

If the data are not already available in shared storage, choose an HPC-approved persistent or scratch location with enough quota and roughly 30 GB free, then run the downloader on a transfer/login node approved for external downloads:

```bash
cd ~/Downloads/CMPE249/detector-lab
df -h "$HOME"
bash scripts/download_datasets.sh "$HOME/detector-lab-data"
```

If SJSU HPC provides a scratch/transfer location, use that instead of `$HOME`; confirm the correct location and transfer policy with the cluster guidance. The KITTI download is large (the left color training images are about 12 GB compressed). COCO and KITTI are downloaded from their official dataset endpoints; follow their terms of use. The script checks/extracts archives and prints the expected paths when done.

Then set the paths from that script's output. The KITTI root should contain `training/image_2` and `training/label_2`, per the tutorial. The COCO paths should point to `val2017/` and `instances_val2017.json`.

Submit up to three tasks concurrently (adjust to cluster policy/availability):

```bash
sbatch --array=0-5%3 slurm/evaluate_array.sbatch
```

After the smoke test succeeds, set `MAX_IMAGES=150` and submit again for the main run. For a full validation pass, set `MAX_IMAGES=0`. Track jobs with `squeue -u "$USER"`; inspect logs under `logs/slurm/` and results under `results/`.

Once the evaluation jobs finish, the optional latency job measures all three models on the same allocated A100 with warmup and repeated iterations:

```bash
sbatch slurm/latency.sbatch
```

It writes the latency report to the professor repository's `ngdet/output/` directory. Copy the resulting report/JSON into `results/latency/` if you want it archived with this project's run artifacts (generated outputs are ignored by Git).

Before interpreting metrics, confirm that each result has both relevant classes and valid predictions. The harness maps classes into a shared taxonomy, but a COCO bicycle box and a KITTI cyclist box do not describe exactly the same object extent; document this limitation.

## Reproducibility checklist

For every run, record model/checkpoint, dataset and split, image count, score threshold, hardware, Python/framework versions, exact Slurm command, and output paths. Include failed or restarted jobs in the run log and use only completed `metrics.json` files in final tables.

## Push changes from this checkout

```bash
git add .gitignore README.md RUN_LOG.md configs slurm
git commit -m "Add HPC object detection experiment scaffold"
git push -u origin main
```
