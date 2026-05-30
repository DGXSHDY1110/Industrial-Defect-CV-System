#!/usr/bin/env python3
"""Generate inspection reports: metrics summary + success/failure case images."""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
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


# ---------------------------------------------------------------------------
# Utility functions ported from analyze_yolo_cases.py
# ---------------------------------------------------------------------------

def load_dataset_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_yolo_labels(label_path):
    labels = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    labels.append((cls_id, cx, cy, w, h))
    return labels


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def compute_iou(box_a, box_b):
    x1, y1, x2, y2 = box_a
    xa, ya, xb, yb = box_b
    ix1 = max(x1, xa)
    iy1 = max(y1, ya)
    ix2 = min(x2, xb)
    iy2 = min(y2, yb)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (xb - xa) * (yb - ya)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def draw_boxes(img, gt_boxes, pred_boxes, case_type, reason, max_iou, max_conf):
    h, w = img.shape[:2]
    for (cx, cy, bw, bh) in gt_boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for (px1, py1, px2, py2, conf, _) in pred_boxes:
        cv2.rectangle(img, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
        cv2.putText(img, f"Pred {conf:.2f}", (int(px1), int(py1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    cv2.putText(img, f"{case_type} | {reason}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"IoU: {max_iou:.3f}  Conf: {max_conf:.3f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def classify_case(gt_count, pred_count, max_iou, max_conf, iou_thresh, conf_thresh):
    has_gt = gt_count > 0
    has_pred = pred_count > 0
    if has_gt and max_iou >= iou_thresh and max_conf >= conf_thresh:
        return "success", "good_detection"
    if has_gt and not has_pred:
        return "failure", "false_negative"
    if has_pred and max_iou < iou_thresh:
        return "failure", "low_iou"
    if has_gt and max_iou >= iou_thresh and max_conf < conf_thresh:
        return "failure", "low_confidence"
    if not has_gt and has_pred:
        return "failure", "false_positive"
    return "failure", "borderline_case"


# ---------------------------------------------------------------------------
# Main report pipeline
# ---------------------------------------------------------------------------

def setup_logging(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def find_best_weights(checkpoint_dir: Path) -> Path:
    candidate = checkpoint_dir / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    for pt in sorted(checkpoint_dir.rglob("best.pt"), reverse=True):
        return pt
    raise FileNotFoundError(f"No best.pt found under {checkpoint_dir}")


def generate_metrics_summary_md(report_dir: Path) -> None:
    """If metrics.json exists from eval step, write a Markdown summary."""
    metrics_path = report_dir / "metrics.json"
    if not metrics_path.exists():
        logging.info("metrics.json not found, skipping summary.")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    md_path = report_dir / "metrics.md"
    lines = [
        "# YOLOv8n MVTec Report",
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


def extract_cases(
    weights: Path,
    data: Path,
    split: str,
    conf_thresh: float,
    iou_thresh: float,
    num_success: int,
    num_failure: int,
) -> None:
    """Run prediction on the specified split and write success/failure case images."""
    SUCCESS_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model: %s", weights)
    model = YOLO(str(weights))

    logging.info("Loading dataset: %s", data)
    ds = load_dataset_yaml(data)
    data_root = Path(ds["path"])
    img_rel = ds.get(split)
    if img_rel is None:
        img_rel = f"images/{split}"
    img_dir = data_root / img_rel
    label_dir = data_root / img_rel.replace("images", "labels")

    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    image_files = sorted(img_dir.glob("*.*"))
    logging.info("Found %d images in %s", len(image_files), img_dir)

    logging.info("Running predictions on %s set ...", split)
    results = model.predict(
        source=str(img_dir), conf=conf_thresh, iou=iou_thresh,
        device=0, verbose=False,
    )

    result_map = {}
    for res in results:
        result_map[Path(res.path).name] = res

    success_cases = []
    failure_cases = []

    for img_file in image_files:
        img_name = img_file.name
        stem = img_file.stem

        label_file = label_dir / f"{stem}.txt"
        gt_labels = read_yolo_labels(label_file)
        gt_count = len(gt_labels)

        res = result_map.get(img_name)
        pred_boxes_raw = []
        if res is not None and res.boxes is not None:
            for i in range(len(res.boxes)):
                xyxy = res.boxes.xyxy[i].cpu().numpy()
                conf = float(res.boxes.conf[i])
                cls_id = int(res.boxes.cls[i])
                pred_boxes_raw.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls_id))
        pred_count = len(pred_boxes_raw)

        img_arr = cv2.imread(str(img_file))
        if img_arr is None:
            logging.info("  Skipping unreadable image: %s", img_file)
            continue
        h_img, w_img = img_arr.shape[:2]

        max_iou = 0.0
        max_conf = 0.0
        for (_, cx, cy, bw, bh) in gt_labels:
            gt_xyxy = yolo_to_xyxy(cx, cy, bw, bh, w_img, h_img)
            for (px1, py1, px2, py2, pconf, _) in pred_boxes_raw:
                pred_xyxy = (int(px1), int(py1), int(px2), int(py2))
                iou_val = compute_iou(gt_xyxy, pred_xyxy)
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_conf = pconf

        if gt_count == 0 and pred_count > 0:
            max_conf = max(b[4] for b in pred_boxes_raw)

        case_type, reason = classify_case(
            gt_count, pred_count, max_iou, max_conf, iou_thresh, conf_thresh
        )

        entry = {
            "image_path": str(img_file),
            "stem": stem,
            "img": img_arr,
            "gt_labels": gt_labels,
            "pred_boxes": pred_boxes_raw,
            "case_type": case_type,
            "reason": reason,
            "max_iou": max_iou,
            "max_conf": max_conf,
        }
        if case_type == "success":
            success_cases.append(entry)
        else:
            failure_cases.append(entry)

    success_cases.sort(key=lambda x: x["max_iou"], reverse=True)
    priority = {"false_negative": 0, "low_iou": 1, "low_confidence": 2, "false_positive": 3}
    failure_cases.sort(key=lambda x: (priority.get(x["reason"], 99), -x["max_iou"]))

    csv_path = CASE_DIR / "case_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "image_path", "case_type", "reason",
                         "gt_count", "pred_count", "max_iou", "max_conf", "output_image"])
        cid = 1

        for entry in success_cases[:num_success]:
            img_copy = entry["img"].copy()
            draw_boxes(img_copy, entry["gt_labels"], entry["pred_boxes"],
                       entry["case_type"], entry["reason"],
                       entry["max_iou"], entry["max_conf"])
            out_name = f"success_{cid:03d}_{entry['stem']}.jpg"
            out_path = SUCCESS_DIR / out_name
            cv2.imwrite(str(out_path), img_copy)
            writer.writerow([cid, entry["image_path"], entry["case_type"], entry["reason"],
                             len(entry["gt_labels"]), len(entry["pred_boxes"]),
                             f"{entry['max_iou']:.4f}", f"{entry['max_conf']:.4f}", str(out_path)])
            logging.info("  S%03d: %s IoU=%.3f Conf=%.3f", cid, entry["reason"], entry["max_iou"], entry["max_conf"])
            cid += 1

        for entry in failure_cases[:num_failure]:
            img_copy = entry["img"].copy()
            draw_boxes(img_copy, entry["gt_labels"], entry["pred_boxes"],
                       entry["case_type"], entry["reason"],
                       entry["max_iou"], entry["max_conf"])
            out_name = f"failure_{cid:03d}_{entry['stem']}.jpg"
            out_path = FAILURE_DIR / out_name
            cv2.imwrite(str(out_path), img_copy)
            writer.writerow([cid, entry["image_path"], entry["case_type"], entry["reason"],
                             len(entry["gt_labels"]), len(entry["pred_boxes"]),
                             f"{entry['max_iou']:.4f}", f"{entry['max_conf']:.4f}", str(out_path)])
            logging.info("  F%03d: %s IoU=%.3f Conf=%.3f", cid, entry["reason"], entry["max_iou"], entry["max_conf"])
            cid += 1

    logging.info("Saved cases. CSV: %s", csv_path)
    logging.info("Success cases: %d/%d saved", min(len(success_cases), num_success), len(success_cases))
    logging.info("Failure cases: %d/%d saved", min(len(failure_cases), num_failure), len(failure_cases))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YOLO evaluation report & case images")
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
        default="val",
        choices=["val", "test"],
        help="Dataset split to run predictions on",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--num_success", type=int, default=10, help="Number of success case images")
    parser.add_argument("--num_failure", type=int, default=5, help="Number of failure case images")
    parser.add_argument(
        "--skip-metrics-md",
        action="store_true",
        help="Do not regenerate metrics.md",
    )
    args = parser.parse_args()

    setup_logging(REPORT_DIR)

    # Resolve weights
    weights_path = args.weights
    if not weights_path.exists():
        logging.info("Weights not found at %s, searching checkpoint dir ...", weights_path)
        checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "yolo" / "yolov8n_mvtec"
        weights_path = find_best_weights(checkpoint_dir)
    logging.info("Using weights: %s", weights_path)

    # Generate metrics summary from existing metrics.json
    if not args.skip_metrics_md:
        generate_metrics_summary_md(REPORT_DIR)

    # Extract success/failure cases
    logging.info("Extracting %d success and %d failure cases ...", args.num_success, args.num_failure)
    extract_cases(
        weights=weights_path,
        data=args.data,
        split=args.split,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        num_success=args.num_success,
        num_failure=args.num_failure,
    )

    logging.info("Report generation complete.")


if __name__ == "__main__":
    main()