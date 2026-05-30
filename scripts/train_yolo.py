#!/usr/bin/env python3
"""
Train YOLOv8 detector for Industrial-Defect-CV-System.

Key path rule:
- Ultralytics `project` must be an absolute directory.
- Ultralytics `name` must be a simple run name, not a path.
- Correct output:
  outputs/checkpoints/yolo/<run_name>/
"""

import argparse
import os
import re
from pathlib import Path

from ultralytics import YOLO


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_in_project(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def validate_run_name(run_name: str) -> str:
    """
    Ultralytics `name` must not contain path separators.
    Otherwise it may create paths like:
    runs/detect/outputs/checkpoints/yolo/xxx
    """
    if "/" in run_name or "\\" in run_name:
        raise ValueError(
            f"--run-name must be a simple name, not a path: {run_name}\n"
            f"Use --project to control output directory."
        )

    if not re.match(r"^[A-Za-z0-9_.-]+$", run_name):
        raise ValueError(
            f"--run-name contains unsafe characters: {run_name}\n"
            f"Allowed: letters, numbers, underscore, hyphen, dot."
        )

    return run_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for industrial defect detection."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dataset.yaml",
        help="Path to YOLO dataset.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO checkpoint, e.g. yolov8n.pt or yolov8s.pt.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="Training image size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device id, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="outputs/checkpoints/yolo",
        help="Output root directory for YOLO runs. Do not include run name here.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="yolov8n_mvtec",
        help="Simple run name under project directory.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer, e.g. AdamW, SGD, auto.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=True,
        help="Overwrite existing run directory if it exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved paths and config, do not train.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = get_project_root()
    os.chdir(project_root)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    data_yaml = resolve_in_project(project_root, args.data)
    output_project = resolve_in_project(project_root, args.project)
    run_name = validate_run_name(args.run_name)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    output_project.mkdir(parents=True, exist_ok=True)
    final_output_dir = output_project / run_name

    print("=" * 100)
    print("[YOLOv8 Training]")
    print(f"Project root      : {project_root}")
    print(f"Data yaml         : {data_yaml}")
    print(f"Model             : {args.model}")
    print(f"Image size        : {args.imgsz}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch             : {args.batch}")
    print(f"Device            : {args.device}")
    print(f"Workers           : {args.workers}")
    print(f"Ultralytics project: {output_project}")
    print(f"Ultralytics name   : {run_name}")
    print(f"Expected output    : {final_output_dir}")
    print("=" * 100)

    if args.dry_run:
        print("[Dry Run] Path resolution is valid. Training skipped.")
        return

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        cos_lr=True,
        patience=args.patience,
        seed=args.seed,
        project=str(output_project),
        name=run_name,
        exist_ok=args.exist_ok,
        plots=True,
    )

    actual_save_dir = Path(getattr(results, "save_dir", final_output_dir)).resolve()

    print("=" * 100)
    print("[Done] YOLOv8 training finished.")
    print(f"Expected output: {final_output_dir}")
    print(f"Actual output  : {actual_save_dir}")

    wrong_prefix = project_root / "runs" / "detect"
    try:
        actual_save_dir.relative_to(wrong_prefix)
        print("[Warning] Output is still under runs/detect. Please check project/name arguments.")
    except ValueError:
        print("[OK] Output path is not under runs/detect.")

    print("=" * 100)


if __name__ == "__main__":
    main()
