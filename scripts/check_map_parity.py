#!/usr/bin/env python3
from __future__ import annotations

"""Compare lean YOLOv10 COCO mAP against the official repo's reported value.

This script reads the target mAP50-95 from yolov10-official/logs/<model>.csv and
then runs lean-yolo's COCO evaluation to verify parity within a tolerance.

Usage
  PYTHONPATH=yolov10-official \
  ./.venv/bin/python scripts/check_map_parity.py \
      --data-root data/coco \
      --model yolov10n \
      --device cuda \
      --imgsz 640 \
      --tolerance 0.01

Notes
- Expects the official repo at ./yolov10-official with logs/<model>.csv present.
- Uses default weight loading (downloads allowed) for the lean model.
- For speed, this runs a full-val check; use val.py with --max-images for quick sanity.
"""

import argparse
import csv
import os
from pathlib import Path

from lean_yolo.engine.eval import validate_coco


def parse_args():
    ap = argparse.ArgumentParser(description="Check mAP parity with official YOLOv10 logs")
    ap.add_argument("--data-root", default="data/coco", help="COCO root (full val) or subset root")
    ap.add_argument("--model", default="yolov10n", help="Model name (e.g., yolov10n)")
    ap.add_argument("--device", default="cuda", help="Device, e.g., cpu or cuda")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--tolerance", type=float, default=0.01, help="Allowed abs diff in mAP50-95")
    return ap.parse_args()


def read_official_map(model: str) -> float:
    csv_path = Path("yolov10-official") / "logs" / f"{model}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Official logs not found: {csv_path}")
    last_row = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    if not last_row:
        raise RuntimeError("Official logs CSV appears empty.")
    # Column name in logs
    key = "metrics/mAP50-95(B)"
    if key not in last_row:
        raise KeyError(f"Column '{key}' not in official logs.")
    return float(last_row[key])


def main():
    args = parse_args()
    # Read official target mAP
    target = read_official_map(args.model)
    # Run lean eval with default weights (downloads allowed)
    stats = validate_coco(
        model_name=args.model,
        weights="DEFAULT",
        data_root=args.data_root,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.001,
        iou=0.65,
        max_images=None,
    )
    got = stats.get("mAP50-95", 0.0)
    diff = abs(got - target)
    print({"target": round(target, 5), "got": round(got, 5), "diff": round(diff, 5)})
    if diff > args.tolerance:
        raise SystemExit(
            f"mAP50-95 mismatch: target={target:.5f}, got={got:.5f}, tol={args.tolerance:.3f}"
        )


if __name__ == "__main__":
    main()

