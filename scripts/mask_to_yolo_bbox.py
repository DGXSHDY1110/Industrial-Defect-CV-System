"""
mask_to_yolo_bbox.py - Convert MVTec AD segmentation masks to YOLO bounding box labels.

This script reads a binary mask image, finds connected defect regions, filters small
ones by area, and writes YOLO-format bounding box labels.

Can be used standalone via CLI or imported by prepare_mvtec.py.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def mask_to_bboxes(mask_path, min_area=10):
    """
    Read a binary mask and return a list of bounding boxes.

    Parameters
    ----------
    mask_path : str or Path
        Path to the grayscale mask image.
    min_area : int
        Minimum area (pixels) for a connected component to be kept.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Each bbox as (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        logger.error("Mask file not found: %s", mask_path)
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error("Failed to read mask image: %s", mask_path)
        raise ValueError(f"Failed to read mask image: {mask_path}")

    # Threshold: any pixel > 0 is a defect
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Use connected components with stats for area filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    bboxes = []
    image_h, image_w = mask.shape

    # Label 0 is the background
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x_min = stats[label_id, cv2.CC_STAT_LEFT]
        y_min = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        x_max = x_min + width
        y_max = y_min + height

        # Clamp to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_w, x_max)
        y_max = min(image_h, y_max)

        # Skip degenerate boxes
        if x_max <= x_min or y_max <= y_min:
            continue

        bboxes.append((x_min, y_min, x_max, y_max))

    return bboxes


def bbox_to_yolo(bbox, image_width, image_height):
    """
    Convert pixel bbox to YOLO normalized format.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        (x_min, y_min, x_max, y_max) in pixel coordinates.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    tuple[float, float, float, float]
        (x_center, y_center, width, height) all normalized to [0, 1].
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min

    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    norm_w = w / image_width
    norm_h = h / image_height

    return (x_center, y_center, norm_w, norm_h)


def write_yolo_label(label_path, bboxes, image_width, image_height, class_id=0):
    """
    Write YOLO label file.

    Parameters
    ----------
    label_path : str or Path
        Output path for the .txt label file.
    bboxes : list[tuple[int, int, int, int]]
        List of pixel-space bounding boxes.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.
    class_id : int
        Integer class ID (default 0 for single-class defect).
    """
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as f:
        for bbox in bboxes:
            x_c, y_c, w, h = bbox_to_yolo(bbox, image_width, image_height)

            # Validate YOLO coordinate ranges
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
                logger.warning(
                    "Skipping invalid YOLO bbox: class=%d x_c=%.4f y_c=%.4f w=%.4f h=%.4f "
                    "(pixel bbox: %s, image: %dx%d)",
                    class_id, x_c, y_c, w, h, bbox, image_width, image_height,
                )
                continue

            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def draw_bboxes(image_path, bboxes, output_path):
    """
    Draw bounding boxes on an image and save to disk.

    Parameters
    ----------
    image_path : str or Path
        Path to the source image.
    bboxes : list[tuple[int, int, int, int]]
        List of pixel-space bounding boxes.
    output_path : str or Path
        Path where the visualization image will be saved.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Failed to read image for visualization: %s", image_path)
        return

    for (x_min, y_min, x_max, y_max) in bboxes:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)

    # If no bboxes, add a "normal" label
    if not bboxes:
        h, w = image.shape[:2]
        cv2.putText(
            image, "NORMAL", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )

    cv2.imwrite(str(output_path), image)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MVTec AD mask to YOLO bbox label."
    )
    parser.add_argument("--image_path", required=True, help="Path to the source image.")
    parser.add_argument("--mask_path", default=None, help="Path to the mask image (defect only).")
    parser.add_argument("--label_path", required=True, help="Path for the output .txt label file.")
    parser.add_argument("--vis_path", default=None, help="Path for the visualization output image.")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID (default 0).")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum CC area (default 10).")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    label_path = Path(args.label_path)

    # Read image dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Failed to read image: %s", image_path)
        sys.exit(1)
    image_h, image_w = image.shape[:2]

    bboxes = []
    if args.mask_path:
        bboxes = mask_to_bboxes(args.mask_path, min_area=args.min_area)

    write_yolo_label(label_path, bboxes, image_w, image_h, class_id=args.class_id)

    if args.vis_path:
        draw_bboxes(image_path, bboxes, args.vis_path)

    logger.info(
        "Processed %s: %d bboxes → %s",
        image_path.name, len(bboxes), label_path,
    )


if __name__ == "__main__":
    main()