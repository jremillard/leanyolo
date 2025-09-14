#!/usr/bin/env python3
"""
Inference script and example pipeline.

This script is designed to be copied and tweaked:
- Reads an image or directory, applies letterbox, and runs YOLOv10
- Decodes detections, rescales back, and writes visualization images
- Uses PyTorch-native `get_model` API (no YAML)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

import sys
from pathlib import Path as _Path

# Ensure repo root on path for 'leanyolo' imports when run from any CWD
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names
from leanyolo.utils.box_ops import unletterbox_coords
from leanyolo.utils.letterbox import letterbox
from leanyolo.utils.viz import draw_detections
from leanyolo.models.yolov10.postprocess import (
    decode_v10_official_topk as _decode_topk,
    decode_v10_predictions as _decode_nms,
)


def parse_args():
    ap = argparse.ArgumentParser(description="leanyolo YOLOv10 inference")
    ap.add_argument("--source", required=True, help="Image path or directory")
    ap.add_argument("--model", default="yolov10s", help="Model name")
    ap.add_argument("--weights", default="PRETRAINED_COCO", help="Weights key or None")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    ap.add_argument("--decode", choices=["topk", "nms"], default="topk", help="Decode mode: official top-k or class-wise NMS")
    ap.add_argument("--max-dets", type=int, default=300, help="Maximum detections per image after decode")
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
    decode: str = "topk",
    max_dets: int = 300,
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
            # Forward gives raw; decode based on requested mode
            raw = model(x)
            if decode == "topk":
                # Official top-k decode uses one-to-one branch when available
                if isinstance(raw, dict):
                    seq = raw.get("one2one", raw.get("one2many"))
                else:
                    seq = getattr(model, "_eval_branches", {}).get("one2one", raw)
                dets = _decode_topk(seq, num_classes=len(cn), strides=(8, 16, 32), max_det=max_dets)[0][0]
            else:
                # Class-wise NMS decode uses one-to-many branch when available
                if isinstance(raw, dict):
                    seq = raw.get("one2many", raw.get("one2one"))
                else:
                    seq = getattr(model, "_eval_branches", {}).get("one2many", raw)
                dets = _decode_nms(
                    seq,
                    num_classes=len(cn),
                    strides=(8, 16, 32),
                    conf_thresh=conf,
                    iou_thresh=iou,
                    max_det=max_dets,
                )[0][0]
            # Scale back to original image size
            if dets.numel() > 0:
                dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=img.shape[:2])

            # Save visualization
            rgb = img
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vis = draw_detections(bgr, dets, class_names=cn)
            out_path = os.path.join(save_dir, ipath.name)
            cv2.imwrite(out_path, vis)
            # Print status for coding agents
            print(f"Infer: input='{ipath}', output='{out_path}'")
            if dets.numel() == 0:
                print("  detections: 0")
            else:
                print(f"  detections: {dets.shape[0]}")
                for i in range(dets.shape[0]):
                    x1, y1, x2, y2, score, cls_idx = dets[i].tolist()
                    cls_idx_int = int(cls_idx)
                    cls_name = cn[cls_idx_int] if 0 <= cls_idx_int < len(cn) else str(cls_idx_int)
                    print(
                        "  box[{}]: x1={:.1f}, y1={:.1f}, x2={:.1f}, y2={:.1f}, score={:.3f}, cls='{}' ({})".format(
                            i, x1, y1, x2, y2, score, cls_name, cls_idx_int
                        )
                    )
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

    results = infer_paths(
        source=args.source,
        model_name=args.model,
        weights=None if args.weights in {"", "none", "None", "NONE"} else args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        decode=args.decode,
        max_dets=args.max_dets,
        save_dir=args.save_dir,
        class_names=class_names,
    )
    # Print a brief summary
    total = sum(int(d.shape[0]) for _, d in results)
    print(f"Done: {len(results)} image(s), total detections={total}")


if __name__ == "__main__":
    main()
