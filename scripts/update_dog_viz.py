#!/usr/bin/env python3
from __future__ import annotations

"""Update the classic YOLO dog image visualization in the repo root.

This script:
- Ensures dog.jpg exists (downloads if missing)
- Runs leanyolo inference
- Draws detections using leanyolo.utils.viz.draw_detections
- Saves visualization as dog_viz.jpg in the repo root
- Removes any other dog_* files, keeping only dog.jpg and dog_viz.jpg

Usage:
  PYTHONPATH=.:yolov10-official ./.venv/bin/python scripts/update_dog_viz.py

Notes:
- Default model is yolov10s with official weights. Change MODEL below if desired.
- We set a reasonable confidence threshold to avoid clutter.
"""

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import torch

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names
from leanyolo.utils.letterbox import letterbox
from leanyolo.utils.postprocess import decode_predictions
from leanyolo.utils.box_ops import unletterbox_coords
from leanyolo.utils.viz import draw_detections


DOG_URL = "https://github.com/pjreddie/darknet/raw/master/data/dog.jpg"
DOG_PATH = Path("dog.jpg")
OUT_PATH = Path("dog_viz.jpg")
MODEL = "yolov10l"
IMGSZ = 640
CONF = 0.25
IOU = 0.45


def ensure_dog(path: Path = DOG_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DOG_URL) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main() -> None:
    ensure_dog()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cn = coco80_class_names()
    model = get_model(MODEL, weights="PRETRAINED_COCO", class_names=cn).to(device).eval()

    # Load image (RGB path in pipeline)
    bgr = cv2.imread(str(DOG_PATH), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(DOG_PATH))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    lb_img, gain, pad = letterbox(rgb, new_shape=IMGSZ)
    x = torch.from_numpy(lb_img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x)
    dets = decode_predictions(
        preds,
        num_classes=len(cn),
        strides=(8, 16, 32),
        conf_thresh=CONF,
        iou_thresh=IOU,
        img_size=(IMGSZ, IMGSZ),
    )[0][0]
    if dets.numel() > 0:
        dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=rgb.shape[:2])

    # Draw on original BGR and save
    vis = draw_detections(bgr, dets, class_names=cn)
    cv2.imwrite(str(OUT_PATH), vis)

    # Cleanup: keep only dog.jpg and dog_viz.jpg
    keep = {str(DOG_PATH.resolve()), str(OUT_PATH.resolve())}
    for p in Path(".").glob("dog*"):
        rp = str(p.resolve())
        if rp not in keep:
            try:
                p.unlink()
            except Exception:
                pass

    print("Updated:", OUT_PATH)


if __name__ == "__main__":
    main()
