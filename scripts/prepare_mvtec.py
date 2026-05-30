"""
prepare_mvtec.py - Convert MVTec AD subset to YOLOv8 detection dataset format.

Reads MVTec AD raw data for specified categories, splits normal/defect images
into train/val/test sets, converts masks to YOLO bounding box labels, and
generates dataset.yaml plus summary reports.

Usage:
    python scripts/prepare_mvtec.py \
        --raw_dir data/raw/mvtec_ad \
        --out_dir data/processed \
        --categories bottle capsule metal_nut \
        --val_ratio 0.2 \
        --seed 42 \
        --min_area 10 \
        --single_class \
        --vis_samples 50 \
        --overwrite
"""

import argparse
import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# Import from sibling script
from mask_to_yolo_bbox import mask_to_bboxes, write_yolo_label, draw_bboxes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MVTecRecord:
    """Represents one image from the MVTec AD dataset."""
    category: str
    defect_type: str
    image_path: Path
    mask_path: Path | None
    source_split: str       # "train" or "test" (MVTec original split)
    target_split: str | None = None  # "train" / "val" / "test" after conversion
    is_defect: bool = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)


def validate_raw_dataset(raw_dir: Path, categories: list[str]) -> None:
    """Check that the raw MVTec AD directory contains required category folders."""
    missing = []
    for cat in categories:
        cat_dir = raw_dir / cat
        if not cat_dir.is_dir():
            missing.append(cat)
        elif not (cat_dir / "train" / "good").is_dir():
            missing.append(f"{cat}/train/good")
        elif not (cat_dir / "test").is_dir():
            missing.append(f"{cat}/test")
    if missing:
        logger.error("Missing raw data directories: %s", missing)
        raise FileNotFoundError(f"Missing directories: {missing}")
    logger.info("Raw dataset validated: %s", categories)


def create_output_dirs(out_dir: Path) -> None:
    dirs = [
        out_dir / "images" / s for s in ("train", "val", "test")
    ] + [
        out_dir / "labels" / s for s in ("train", "val", "test")
    ] + [
        out_dir / "visualizations" / s for s in ("train", "val", "test")
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories created under %s", out_dir)


def collect_mvtec_records(category_dir: Path, category: str) -> list[MVTecRecord]:
    """Scan a single MVTec AD category directory and return all records."""
    records = []

    # 1. Normal training images: train/good/*.png
    train_good_dir = category_dir / "train" / "good"
    if train_good_dir.is_dir():
        for img_path in sorted(train_good_dir.glob("*.png")):
            records.append(MVTecRecord(
                category=category,
                defect_type="good",
                image_path=img_path,
                mask_path=None,
                source_split="train",
                is_defect=False,
            ))

    # 2. Normal test images: test/good/*.png
    test_good_dir = category_dir / "test" / "good"
    if test_good_dir.is_dir():
        for img_path in sorted(test_good_dir.glob("*.png")):
            records.append(MVTecRecord(
                category=category,
                defect_type="good",
                image_path=img_path,
                mask_path=None,
                source_split="test",
                is_defect=False,
            ))

    # 3. Defect images: test/<defect_type>/*.png + ground_truth/<defect_type>/*_mask.png
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"
    if test_dir.is_dir():
        for defect_type_dir in sorted(test_dir.iterdir()):
            if not defect_type_dir.is_dir():
                continue
            defect_type = defect_type_dir.name
            if defect_type == "good":
                continue  # already handled above

            for img_path in sorted(defect_type_dir.glob("*.png")):
                stem = img_path.stem
                # Try standard naming first: {stem}_mask.png
                mask_path = gt_dir / defect_type / f"{stem}_mask.png"

                # If not found, try glob match
                if not mask_path.exists():
                    candidates = sorted(
                        (gt_dir / defect_type).glob(f"{stem}*")
                    )
                    if candidates:
                        mask_path = candidates[0]
                        logger.debug(
                            "Matched mask via glob: %s → %s", img_path.name, mask_path.name
                        )
                    else:
                        logger.warning(
                            "No mask found for defect image: %s (category %s)",
                            img_path, category,
                        )
                        # Still create record but with missing mask
                        records.append(MVTecRecord(
                            category=category,
                            defect_type=defect_type,
                            image_path=img_path,
                            mask_path=None,
                            source_split="test",
                            is_defect=True,
                        ))
                        continue

                records.append(MVTecRecord(
                    category=category,
                    defect_type=defect_type,
                    image_path=img_path,
                    mask_path=mask_path,
                    source_split="test",
                    is_defect=True,
                ))

    return records


def build_filename(record: MVTecRecord) -> str:
    """Build unique target filename from record metadata."""
    stem = record.image_path.stem
    return f"{record.category}__{record.source_split}__{record.defect_type}__{stem}.png"


def split_records(
    records: list[MVTecRecord],
    val_ratio: float,
    def_train: float = 0.7,
    def_val: float = 0.2,
) -> None:
    """
    Assign target_split to each record in-place.

    Rules:
    - Normal train/good → (1-val_ratio) to train, val_ratio to val
    - Normal test/good → all to test
    - Defect test/<defect_type> → 70%/20%/10% train/val/test (per defect type)
    """
    # Normal train/good
    normal_train = [r for r in records if r.defect_type == "good" and r.source_split == "train"]
    random.shuffle(normal_train)
    n_val = max(1, int(len(normal_train) * val_ratio))
    for i, r in enumerate(normal_train):
        r.target_split = "val" if i < n_val else "train"

    # Normal test/good
    for r in records:
        if r.defect_type == "good" and r.source_split == "test":
            r.target_split = "test"

    # Defect: group by (category, defect_type)
    defect_groups: dict[tuple[str, str], list[MVTecRecord]] = {}
    for r in records:
        if r.is_defect:
            key = (r.category, r.defect_type)
            defect_groups.setdefault(key, []).append(r)

    for key, group in defect_groups.items():
        random.shuffle(group)
        n = len(group)
        if n == 1:
            group[0].target_split = "train"
        elif n == 2:
            group[0].target_split = "train"
            group[1].target_split = "val"
        elif n == 3:
            group[0].target_split = "train"
            group[1].target_split = "val"
            group[2].target_split = "test"
        else:
            n_test = max(1, round(n * 0.1))
            n_val_def = max(1, round(n * def_val))
            n_train_def = n - n_test - n_val_def
            # Ensure at least 1 per split when possible
            if n_train_def < 1:
                n_train_def = n - 2
                n_val_def = 1
                n_test = 1
            splits = (
                ["train"] * n_train_def
                + ["val"] * n_val_def
                + ["test"] * n_test
            )
            for r, s in zip(group, splits):
                r.target_split = s

    # Verify every record has a target_split
    unassigned = [r for r in records if r.target_split is None]
    if unassigned:
        logger.warning("%d records unassigned, putting in train", len(unassigned))
        for r in unassigned:
            r.target_split = "train"


def process_records(
    records: list[MVTecRecord],
    out_dir: Path,
    min_area: int,
    class_id: int,
    vis_samples: int,
) -> dict:
    """
    Copy images, generate labels, and create visualizations.

    Returns conversion metadata dict.
    """
    summary_records = []
    total_bboxes = 0
    empty_labels = 0
    missing_masks = []
    invalid_bboxes = []
    vis_counters = {"train": 0, "val": 0, "test": 0}

    for r in records:
        split = r.target_split
        new_name = build_filename(r)
        new_stem = Path(new_name).stem

        src_img = r.image_path
        dst_img = out_dir / "images" / split / new_name
        dst_label = out_dir / "labels" / split / f"{new_stem}.txt"

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Read image dimensions
        image = cv2.imread(str(dst_img))
        if image is None:
            logger.error("Failed to read copied image: %s", dst_img)
            continue
        img_h, img_w = image.shape[:2]

        # Generate label
        bboxes = []
        if r.is_defect:
            if r.mask_path and r.mask_path.exists():
                try:
                    bboxes = mask_to_bboxes(r.mask_path, min_area=min_area)
                except Exception as e:
                    logger.warning("Failed to process mask %s: %s", r.mask_path, e)
                    missing_masks.append(str(r.mask_path))
            else:
                logger.warning(
                    "Missing mask for defect image: %s (category %s, defect %s)",
                    r.image_path, r.category, r.defect_type,
                )
                missing_masks.append(str(r.image_path))

        # Filter invalid bboxes (those outside [0,1] after normalization)
        valid_bboxes = []
        for bb in bboxes:
            x_c = ((bb[0] + bb[2]) / 2.0) / img_w
            y_c = ((bb[1] + bb[3]) / 2.0) / img_h
            bw = (bb[2] - bb[0]) / img_w
            bh = (bb[3] - bb[1]) / img_h
            if 0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < bw <= 1 and 0 < bh <= 1:
                valid_bboxes.append(bb)
            else:
                invalid_bboxes.append({
                    "image": str(dst_img),
                    "bbox_pixel": bb,
                    "yolo": (x_c, y_c, bw, bh),
                })

        write_yolo_label(dst_label, valid_bboxes, img_w, img_h, class_id=class_id)
        total_bboxes += len(valid_bboxes)
        if len(valid_bboxes) == 0:
            empty_labels += 1

        # Visualization
        vis_dst = None
        if vis_counters[split] < vis_samples:
            vis_dst = out_dir / "visualizations" / split / f"{new_stem}.png"
            draw_bboxes(str(dst_img), valid_bboxes, vis_dst)
            vis_counters[split] += 1

        summary_records.append({
            "category": r.category,
            "defect_type": r.defect_type,
            "source_image": str(r.image_path),
            "source_mask": str(r.mask_path) if r.mask_path else None,
            "target_image": str(dst_img),
            "target_label": str(dst_label),
            "target_split": split,
            "is_defect": r.is_defect,
            "num_bboxes": len(valid_bboxes),
        })

    return {
        "total_bboxes": total_bboxes,
        "empty_label_files": empty_labels,
        "missing_masks": missing_masks,
        "invalid_bboxes": invalid_bboxes,
        "records": summary_records,
    }


def write_dataset_yaml(out_dir: Path, class_names: list[str]) -> None:
    """Write YOLOv8 dataset.yaml."""
    yaml_path = out_dir / "dataset.yaml"
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")
    logger.info("dataset.yaml written → %s", yaml_path)


def compute_split_summary(records: list[MVTecRecord], categories: list[str]) -> dict:
    """Compute per-split and per-category statistics."""
    summary: dict = {
        "categories": categories,
        "splits": {
            "train": {"images": 0, "normal": 0, "defect": 0, "labels": 0},
            "val": {"images": 0, "normal": 0, "defect": 0, "labels": 0},
            "test": {"images": 0, "normal": 0, "defect": 0, "labels": 0},
        },
        "by_category": {c: {"train": 0, "val": 0, "test": 0} for c in categories},
    }

    for r in records:
        s = r.target_split
        if s not in summary["splits"]:
            continue
        summary["splits"][s]["images"] += 1
        summary["splits"][s]["labels"] += 1
        if r.is_defect:
            summary["splits"][s]["defect"] += 1
        else:
            summary["splits"][s]["normal"] += 1

        summary["by_category"][r.category][s] += 1

    return summary


def run_sanity_checks(out_dir: Path) -> list[str]:
    """
    Run dataset sanity checks and return list of issues.

    Checks:
    1. Image/label count match per split
    2. Every image has a corresponding label file
    3. All label lines have valid format and coords
    4. No zero-byte images
    """
    issues = []

    for split in ("train", "val", "test"):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split

        imgs = sorted(img_dir.glob("*.png"))
        lbls = sorted(lbl_dir.glob("*.txt"))

        # 1. Count match
        if len(imgs) != len(lbls):
            issues.append(
                f"[{split}] Count mismatch: {len(imgs)} images vs {len(lbls)} labels"
            )

        # 2. Every image has a label
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        only_img = img_stems - lbl_stems
        only_lbl = lbl_stems - img_stems
        if only_img:
            issues.append(f"[{split}] Images without label: {sorted(only_img)[:10]}...")
        if only_lbl:
            issues.append(f"[{split}] Labels without image: {sorted(only_lbl)[:10]}...")

        # 3. Validate label contents
        for lbl_path in lbls:
            try:
                lines = lbl_path.read_text().strip().splitlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"[{split}] Invalid label format in {lbl_path.name}: {line}")
                        continue
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if cls_id != 0:
                        issues.append(f"[{split}] Unexpected class_id {cls_id} in {lbl_path.name}")
                    if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        issues.append(
                            f"[{split}] Invalid coords in {lbl_path.name}: "
                            f"x_c={x_c:.6f} y_c={y_c:.6f} w={w:.6f} h={h:.6f}"
                        )
            except Exception as e:
                issues.append(f"[{split}] Error reading {lbl_path}: {e}")

        # 4. Check images are readable
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                issues.append(f"[{split}] Unreadable image: {img_path.name}")

    if not issues:
        logger.info("Sanity checks: ALL PASSED")
    else:
        logger.warning("Sanity checks: %d ISSUE(S) FOUND", len(issues))
        for issue in issues:
            logger.warning("  - %s", issue)

    return issues


def print_final_report(
    summary: dict,
    total_bboxes: int,
    empty_labels: int,
    missing_masks: list,
    issues: list,
    out_dir: Path,
    categories: list[str],
) -> None:
    """Print a human-readable conversion report."""
    print()
    print("[MVTec → YOLO Conversion Done]")
    print()
    print(f"Raw dir:")
    print(f"  data/raw/mvtec_ad")
    print()
    print(f"Output dir:")
    print(f"  {out_dir}")
    print()
    print(f"Categories:")
    print(f"  {', '.join(categories)}")
    print()
    print("Split summary:")
    for split in ("train", "val", "test"):
        s = summary["splits"][split]
        print(f"  {split:5s}: {s['images']:4d} images, {s['defect']:4d} defect, {s['normal']:4d} normal")
    print()
    print("Labels:")
    total_labels = sum(s["labels"] for s in summary["splits"].values())
    print(f"  total label files: {total_labels}")
    print(f"  total bboxes     : {total_bboxes}")
    print(f"  empty labels     : {empty_labels}")
    print()
    print(f"Missing masks:")
    print(f"  {len(missing_masks)}")
    if missing_masks:
        for m in missing_masks[:5]:
            print(f"    - {m}")
        if len(missing_masks) > 5:
            print(f"    ... and {len(missing_masks) - 5} more")
    print()
    print(f"Sanity check issues: {len(issues)}")
    print()
    print(f"Dataset yaml:")
    print(f"  {out_dir / 'dataset.yaml'}")
    print()
    print("Next command:")
    print(f"  yolo detect train data={out_dir / 'dataset.yaml'} model=yolov8n.pt imgsz=768 epochs=50 batch=8")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare MVTec AD data for YOLOv8 detection training."
    )
    parser.add_argument("--raw_dir", required=True, help="Path to raw MVTec AD directory.")
    parser.add_argument("--out_dir", required=True, help="Path to output YOLO dataset directory.")
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "capsule", "metal_nut"],
        help="Space-separated list of categories (default: bottle capsule metal_nut).",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Ratio of normal train images to use for val (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--min_area", type=int, default=10,
                        help="Minimum connected component area for bbox (default: 10).")
    parser.add_argument("--single_class", action="store_true", default=True,
                        help="Use single 'defect' class (default: True).")
    parser.add_argument("--vis_samples", type=int, default=50,
                        help="Max visualization images per split (default: 50).")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing output directory.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    categories = args.categories

    # Safety: check for existing data unless --overwrite is set
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            logger.error(
                "Output directory %s exists and is not empty. Use --overwrite to replace.",
                out_dir,
            )
            sys.exit(1)
        else:
            logger.warning("Overwriting existing output directory: %s", out_dir)
            shutil.rmtree(out_dir)

    set_seed(args.seed)

    # Validate raw data
    validate_raw_dataset(raw_dir, categories)

    # Create output directories
    create_output_dirs(out_dir)

    # Collect all records
    all_records: list[MVTecRecord] = []
    for cat in categories:
        cat_dir = raw_dir / cat
        recs = collect_mvtec_records(cat_dir, cat)
        logger.info("Collected %d records for category %s", len(recs), cat)
        all_records.extend(recs)

    logger.info("Total records collected: %d", len(all_records))

    # Assign splits
    split_records(all_records, val_ratio=args.val_ratio)

    # Process: copy images, generate labels, vis
    class_names = ["defect"] if args.single_class else ["defect"]  # single-class only for now
    class_id = 0

    conv_info = process_records(
        all_records, out_dir, min_area=args.min_area,
        class_id=class_id, vis_samples=args.vis_samples,
    )

    # Write dataset.yaml
    write_dataset_yaml(out_dir, class_names)

    # Write summaries
    split_summary = compute_split_summary(all_records, categories)
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2, default=str)
    logger.info("split_summary.json written")

    conversion_summary = {
        "total_images": len(all_records),
        "total_labels": len(all_records),
        "total_bboxes": conv_info["total_bboxes"],
        "empty_label_files": conv_info["empty_label_files"],
        "missing_masks": conv_info["missing_masks"],
        "invalid_bboxes": conv_info["invalid_bboxes"],
        "records": conv_info["records"],
    }
    with open(out_dir / "conversion_summary.json", "w") as f:
        json.dump(conversion_summary, f, indent=2, default=str)
    logger.info("conversion_summary.json written")

    # Sanity checks
    issues = run_sanity_checks(out_dir)

    # Final report
    print_final_report(
        split_summary,
        conv_info["total_bboxes"],
        conv_info["empty_label_files"],
        conv_info["missing_masks"],
        issues,
        out_dir,
        categories,
    )


if __name__ == "__main__":
    main()