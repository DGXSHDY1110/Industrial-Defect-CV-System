#!/usr/bin/env python3
"""Evaluate YOLOv8n detector on MVTec test/val set and generate metrics."""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CFG = PROJECT_ROOT / "configs" / "dataset" / "yolo_mvtec.yaml"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "outputs" / "checkpoints" / "yolo" / "yolov8n_mvtec" / "weights" / "best.pt"
)
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "yolo"
CASE_DIR = REPORT_DIR / "cases"
SUCCESS_DIR = CASE_DIR / "success"
FAILURE_DIR = CASE_DIR / "failure"


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_best_weights(checkpoint_dir: Path) -> Path:
    """Automatically discover best.pt if the user didn't provide explicit weights."""
    candidate = checkpoint_dir / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    # fallback: any best.pt inside checkpoint_dir
    for pt in sorted(checkpoint_dir.rglob("best.pt"), reverse=True):
        return pt
    raise FileNotFoundError(f"No best.pt found under {checkpoint_dir}")


def collect_metrics_from_results(results) -> dict:
    """Extract scalar metrics from ultralytics Results objects."""
    metrics = {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "num_images": 0,
        "num_gt_boxes": 0,
        "num_pred_boxes": 0,
    }
    if not results:
        return metrics

    total_gt = 0
    total_pred = 0

    for res in results:
        if res.boxes is not None:
            total_pred += len(res.boxes)
        # Ground-truth counts are not directly available from predict results,
        # so we estimate from the first result's path if we have a val run.
        # We'll rely on the model.val() path for proper mAP.

    # Use model.val() for authoritative metrics
    return metrics


def generate_metrics_md(metrics: dict, report_dir: Path) -> Path:
    """Write metrics.md Markdown report."""
    md_path = report_dir / "metrics.md"
    lines = [
        "# YOLOv8n MVTec Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Detection Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@50 | {metrics.get('mAP50', 0):.4f} |",
        f"| mAP@50-95 | {metrics.get('mAP50_95', 0):.4f} |",
        f"| Precision | {metrics.get('precision', 0):.4f} |",
        f"| Recall | {metrics.get('recall', 0):.4f} |",
        f"| F1 Score | {metrics.get('f1', 0):.4f} |",
        f"| Images Evaluated | {metrics.get('num_images', 0)} |",
        f"| GT Boxes | {metrics.get('num_gt_boxes', 0)} |",
        f"| Pred Boxes | {metrics.get('num_pred_boxes', 0)} |",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote metrics.md -> %s", md_path)
    return md_path


def copy_artifacts(val_dir: Path, report_dir: Path) -> None:
    """Copy PR curve, confusion matrix, and other visual artifacts into reports/yolo/."""
    # PR curve
    pr_src = val_dir / "PR_curve.png"
    if pr_src.exists():
        shutil.copy2(pr_src, report_dir / "PR_curve.png")
        logging.info("Copied PR_curve.png")

    # Confusion matrix
    cm_src = val_dir / "confusion_matrix.png"
    if cm_src.exists():
        shutil.copy2(cm_src, report_dir / "confusion_matrix.png")
        logging.info("Copied confusion_matrix.png")

    # Results plot
    results_src = val_dir / "results.png"
    if results_src.exists():
        shutil.copy2(results_src, report_dir / "results.png")
        logging.info("Copied results.png")

    # Validation batch labels
    for batch_img in sorted(val_dir.glob("val_batch*.jpg")):
        shutil.copy2(batch_img, report_dir / batch_img.name)
        logging.info("Copied %s", batch_img.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n on MVTec")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CFG,
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device or 'cpu'",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="Image size for evaluation",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(REPORT_DIR)

    # Resolve weights – auto-discover if needed
    weights_path = args.weights
    if not weights_path.exists():
        logging.info("Weights not found at %s, searching checkpoint dir ...", weights_path)
        checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "yolo" / "yolov8n_mvtec"
        weights_path = find_best_weights(checkpoint_dir)
    logging.info("Using weights: %s", weights_path)

    data_yaml = str(args.data.resolve())
    logging.info("Dataset config: %s", data_yaml)

    model = YOLO(str(weights_path))

    # Run validation for authoritative metrics
    logging.info("Running model.val() on split=%s ...", args.split)
    val_results = model.val(
        data=data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    # Collect metrics
    metrics = {
        "mAP50": float(val_results.box.map50) if val_results.box.map50 is not None else 0.0,
        "mAP50_95": float(val_results.box.map) if val_results.box.map is not None else 0.0,
        "precision": float(val_results.box.mp) if val_results.box.mp is not None else 0.0,
        "recall": float(val_results.box.mr) if val_results.box.mr is not None else 0.0,
        "num_images": 0,
        "num_gt_boxes": 0,
        "num_pred_boxes": 0,
    }

    # Compute F1
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = (
            2.0 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        )
    else:
        metrics["f1"] = 0.0

    # Count images evaluated from the dataset yaml
    ds = load_yaml(args.data)
    data_root = Path(ds["path"])
    split_key = args.split
    if split_key == "val":
        # In ultralytics, val split may be mapped to 'val'
        pass
    img_dir = data_root / "images" / args.split
    if img_dir.exists():
        image_files = list(img_dir.glob("*.*"))
        metrics["num_images"] = len(image_files)
    else:
        logging.warning("Image directory %s not found; num_images set to 0", img_dir)

    # Also read ground-truth box counts
    label_dir = data_root / "labels" / args.split
    if label_dir.exists():
        gt_total = 0
        for lbl in label_dir.glob("*.txt"):
            lines = lbl.read_text(encoding="utf-8").strip().split("\n")
            gt_total += len([l for l in lines if l.strip()])
        metrics["num_gt_boxes"] = gt_total
    else:
        logging.warning("Label directory %s not found", label_dir)

    # Prediction box counts from val results (rough estimate)
    if hasattr(val_results, "boxes") and val_results.boxes is not None:
        metrics["num_pred_boxes"] = len(val_results.boxes)

    # Save metrics.json
    metrics_path = REPORT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logging.info("Saved metrics.json -> %s", metrics_path)

    # Generate metrics.md
    generate_metrics_md(metrics, REPORT_DIR)

    # Copy visual artifacts from the val run directory
    # ultralytics saves val results under runs/val/ by default
    runs_dir = PROJECT_ROOT / "runs" / "val"
    if runs_dir.exists():
        # Find the latest val directory
        val_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if val_dirs:
            latest_val_dir = val_dirs[0]
            logging.info("Copying artifacts from %s", latest_val_dir)
            copy_artifacts(latest_val_dir, REPORT_DIR)
    else:
        logging.warning("No runs/val/ directory found; skipping artifact copy")

    logging.info("Evaluation complete.")
    logging.info("Metrics summary: mAP@50=%.4f  mAP@50-95=%.4f  F1=%.4f",
                 metrics["mAP50"], metrics["mAP50_95"], metrics["f1"])


if __name__ == "__main__":
    main()