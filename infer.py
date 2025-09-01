#!/usr/bin/env python3
from __future__ import annotations

"""
Inference script and example pipeline.

This script is designed to be copied and tweaked:
- Reads an image or directory, applies letterbox, and runs YOLOv10
- Decodes detections, rescales back, and writes visualization images
- Uses PyTorch-native `get_model` API (no YAML)
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names
from leanyolo.utils.box_ops import unletterbox_coords
from leanyolo.utils.letterbox import letterbox
from leanyolo.utils.viz import draw_detections


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


def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img).to(device)
    x = x.permute(2, 0, 1).float()
    return x.unsqueeze(0)


def infer_paths(
    source: str,
    model_name: str = "yolov10s",
    weights: str | None = "PRETRAINED_COCO",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    save_dir: str = "runs/infer/exp",
    class_names: List[str] | None = None,
) -> List[Tuple[str, torch.Tensor]]:
    """Run inference on a path or directory and save visualizations.

    Returns a list of (input_path, detections_tensor) pairs. Designed as
    reference code you can copy into your apps.
    """
    device_t = torch.device(device)
    cn = class_names or coco80_class_names()
    model = get_model(
        model_name,
        weights=weights,
        class_names=cn,
        input_norm_subtract=[0.0, 0.0, 0.0],
        input_norm_divide=[255.0, 255.0, 255.0],
    )
    model.to(device_t).eval()

    p = Path(source)
    paths: Iterable[Path]
    if p.is_dir():
        paths = sorted([x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        paths = [p]

    os.makedirs(save_dir, exist_ok=True)

    results: List[Tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        for ipath in paths:
            img = _imread_rgb(str(ipath))
            lb_img, gain, pad = letterbox(img, new_shape=imgsz)
            x = _to_tensor(lb_img, device_t)
            # Set decode thresholds and run: forward gives raw; decode_forward returns detections
            model.post_conf_thresh = conf
            model.post_iou_thresh = iou
            raw = model(x)
            dets = model.decode_forward(raw)[0][0]
            # Scale back to original image size
            if dets.numel() > 0:
                dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=img.shape[:2])

            # Save visualization
            rgb = img
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vis = draw_detections(bgr, dets, class_names=cn)
            out_path = os.path.join(save_dir, ipath.name)
            cv2.imwrite(out_path, vis)
            results.append((str(ipath), dets))

    return results

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
