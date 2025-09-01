from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

from ..models import get_model
from ..utils.postprocess import decode_predictions
from ..utils.box_ops import unletterbox_coords
from ..utils.letterbox import letterbox
from ..utils.viz import draw_detections


def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_square(img: np.ndarray, size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img.shape[:2]
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, (h, w)


def _to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img).to(device)
    x = x.permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)


def infer_paths(
    source: str,
    model_name: str = "yolov10s",
    weights: str | None = "DEFAULT",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    save_dir: str = "runs/infer/exp",
    class_names: List[str] | None = None,
) -> List[Tuple[str, torch.Tensor]]:
    device_t = torch.device(device)
    model = get_model(model_name, weights=weights)
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
            preds = model(x)
            dets_per_img = decode_predictions(preds, num_classes=80, strides=(8, 16, 32), conf_thresh=conf, iou_thresh=iou, img_size=(imgsz, imgsz))
            dets = dets_per_img[0][0]
            # Scale back to original image size
            if dets.numel() > 0:
                dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=img.shape[:2])

            # Save visualization
            rgb = img
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vis = draw_detections(bgr, dets, class_names=class_names)
            out_path = os.path.join(save_dir, ipath.name)
            cv2.imwrite(out_path, vis)
            results.append((str(ipath), dets))

    return results
