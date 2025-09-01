from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco import load_coco_categories
from ..utils.letterbox import letterbox


@dataclass
class CocoSample:
    image: np.ndarray  # RGB HxWx3
    boxes: torch.Tensor  # Nx4 xyxy
    labels: torch.Tensor  # N


class CocoDetection(Dataset):
    def __init__(self, images_dir: str | Path, ann_path: str | Path, imgsz: int = 640, augment: bool = False) -> None:
        self.images_dir = Path(images_dir)
        self.ann_path = Path(ann_path)
        self.imgsz = int(imgsz)
        self.augment = bool(augment)

        with open(self.ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Build image id -> file_name
        self.images = {img["id"]: img for img in data.get("images", [])}
        # Build annotations per image
        ann_per_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            img_id = ann["image_id"]
            ann_per_img.setdefault(img_id, []).append(ann)
        # Sequence of image ids actually used (those that exist on disk)
        self.ids: List[int] = []
        for img_id, img_info in self.images.items():
            p = self.images_dir / img_info["file_name"]
            if p.exists():
                self.ids.append(img_id)
        self.ids.sort()
        # Category mapping to contiguous
        cats = data.get("categories", [])
        # Keep their provided ids and order
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(cats, key=lambda x: x["id"]))}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # type: ignore[override]
        img_id = self.ids[idx]
        info = self.images[img_id]
        path = self.images_dir / info["file_name"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Load anns
        with open(self.ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anns = [a for a in data.get("annotations", []) if a.get("image_id") == img_id and not a.get("iscrowd", 0)]
        boxes_xywh = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes_xywh.append([x, y, w, h])
            labels.append(self.cat_id_to_idx[a["category_id"]])
        boxes_xywh = np.array(boxes_xywh, dtype=np.float32) if boxes_xywh else np.zeros((0, 4), dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Letterbox
        lb_img, gain, pad = letterbox(rgb, new_shape=self.imgsz)
        gx, gy = gain
        px, py = pad
        # Convert boxes to xyxy in letterboxed space
        boxes = boxes_xywh.copy()
        boxes[:, 0] = boxes[:, 0] * gx + px
        boxes[:, 1] = boxes[:, 1] * gy + py
        boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] * gx)
        boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] * gy)

        x = torch.from_numpy(lb_img).permute(2, 0, 1).contiguous().float()
        target = {
            "boxes": torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels_np).long(),
        }
        return x, target


def coco_collate(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    xs = [b[0] for b in batch]
    ts = [b[1] for b in batch]
    return torch.stack(xs, dim=0), ts

