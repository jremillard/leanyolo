from __future__ import annotations

import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests


COCO_VAL_IMAGES_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        chunk = 1 << 20
        done = 0
        with open(dst, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if not part:
                    continue
                f.write(part)
                done += len(part)


def ensure_coco_val(root: str | Path, *, download: bool = True) -> Tuple[Path, Path]:
    """Ensure COCO val2017 images and annotations exist under root.

    Returns (images_dir, annotations_json)
    """
    root = Path(root)
    images_dir = root / "images" / "val2017"
    ann_dir = root / "annotations"
    ann_json = ann_dir / "instances_val2017.json"

    if not images_dir.exists() and download:
        zip_path = root / "val2017.zip"
        _download(COCO_VAL_IMAGES_ZIP, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root / "images")
        zip_path.unlink(missing_ok=True)

    if not ann_json.exists() and download:
        zip_path = root / "annotations_trainval2017.zip"
        _download(COCO_ANN_ZIP, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)
        zip_path.unlink(missing_ok=True)

    if not images_dir.exists() or not ann_json.exists():
        raise FileNotFoundError("COCO val2017 not found; set download=True or provide data directory.")

    return images_dir, ann_json


def load_coco_categories(ann_json: Path) -> List[int]:
    data = json.loads(Path(ann_json).read_text())
    cats = sorted((c["id"] for c in data.get("categories", [])))
    return cats


def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in Path(images_dir).glob("*.jpg")])

