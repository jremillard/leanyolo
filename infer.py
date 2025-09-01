#!/usr/bin/env python3
from __future__ import annotations

import argparse

from leanyolo.engine.infer import infer_paths


def parse_args():
    ap = argparse.ArgumentParser(description="leanyolo YOLOv10 inference")
    ap.add_argument("--source", required=True, help="Image path or directory")
    ap.add_argument("--model", default="yolov10s", help="Model name")
    ap.add_argument("--weights", default="PRETRAINED_COCO", help="Weights key or None")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    ap.add_argument("--save-dir", default="runs/infer/exp", help="Save directory")
    ap.add_argument("--classes-ann", default=None, help="Optional: COCO-style annotations JSON to derive class names")
    return ap.parse_args()


def main():
    args = parse_args()
    # Optional classes from COCO annotations
    class_names = None
    if args.classes_ann:
        import json
        with open(args.classes_ann, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cats = sorted(data.get('categories', []), key=lambda c: c.get('id', 0))
        class_names = [c.get('name', str(i)) for i, c in enumerate(cats)]

    _ = infer_paths(
        source=args.source,
        model_name=args.model,
        weights=None if args.weights in {"", "none", "None", "NONE"} else args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=args.save_dir,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
