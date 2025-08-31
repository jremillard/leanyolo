#!/usr/bin/env python3
from __future__ import annotations

import argparse

from lean_yolo.data.coco import ensure_coco_val
from lean_yolo.engine.eval import validate_coco


def parse_args():
    ap = argparse.ArgumentParser(description="Lean YOLOv10 COCO validation")
    ap.add_argument("--data-root", default="data/coco", help="COCO root directory")
    ap.add_argument("--download", action="store_true", help="Download COCO val2017 if missing")
    ap.add_argument("--model", default="yolov10s", help="Model name")
    ap.add_argument("--weights", default="DEFAULT", help="Weights key or None for random init")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.65, help="IoU threshold")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    ap.add_argument("--max-images", type=int, default=None, help="Validate on first N images")
    ap.add_argument("--save-json", default=None, help="Optional: path to save detections JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.download:
        ensure_coco_val(args.data_root, download=True)
    stats = validate_coco(
        model_name=args.model,
        weights=None if args.weights in {"", "none", "None", "NONE"} else args.weights,
        data_root=args.data_root,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_images=args.max_images,
        save_json=args.save_json,
    )
    print({k: round(v, 5) for k, v in stats.items()})


if __name__ == "__main__":
    main()

