from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..models import get_model
from ..data.coco import coco80_class_names
from ..utils.postprocess import decode_predictions
from ..utils.box_ops import unletterbox_coords
from ..utils.letterbox import letterbox
from ..data.coco import ensure_coco_val, load_coco_categories, list_images


@torch.no_grad()
def validate_coco(
    *,
    model_name: str = "yolov10s",
    weights: str | None = "PRETRAINED_COCO",
    data_root: str = "data/coco",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.65,
    device: str = "cpu",
    max_images: int | None = None,
    save_json: str | None = None,
) -> Dict[str, float]:
    device_t = torch.device(device)
    root = Path(data_root)
    subset_ann = root / "annotations.json"
    subset_imgs = root / "images"
    if subset_ann.exists() and subset_imgs.exists():
        images_dir, ann_json = subset_imgs, subset_ann
    else:
        images_dir, ann_json = ensure_coco_val(data_root, download=False)
    img_paths = list_images(images_dir)
    if max_images is not None:
        img_paths = img_paths[:max_images]

    coco = COCO(str(ann_json))
    cat_ids = load_coco_categories(ann_json)

    cn = coco80_class_names()
    model = get_model(
        model_name,
        weights=weights,
        class_names=cn,
        input_norm_subtract=[0.0, 0.0, 0.0],
        input_norm_divide=[255.0, 255.0, 255.0],
    )
    model.to(device_t).eval()

    results = []
    for p in img_paths:
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        orig_shape = img.shape[:2]
        lb_img, gain, pad = letterbox(img, new_shape=imgsz)
        x = torch.from_numpy(lb_img).to(device_t).permute(2, 0, 1).float().unsqueeze(0)

        preds = model(x)
        dets = decode_predictions(preds, num_classes=len(cn), strides=(8, 16, 32), conf_thresh=conf, iou_thresh=iou, img_size=(imgsz, imgsz))[0][0]
        if dets.numel() == 0:
            continue
        # Scale boxes back
        dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=orig_shape)
        # Convert to COCO json
        # COCO expects [x, y, w, h] with category_id being dataset category IDs
        image_id = int(Path(p).stem)
        for x1, y1, x2, y2, score, cls in dets.cpu().numpy():
            w, h = x2 - x1, y2 - y1
            cls = int(cls)
            cat_id = cat_ids[cls] if cls < len(cat_ids) else cat_ids[-1]
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    if not results:
        return {"mAP50-95": 0.0}

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(save_json).write_text(json.dumps(results))

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mAP .5:.95
    stats = {
        "mAP50-95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
    }
    return stats
