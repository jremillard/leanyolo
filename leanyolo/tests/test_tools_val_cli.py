from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_img(path: Path, w: int = 32, h: int = 24) -> None:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def _make_coco_single(img_name: str) -> dict:
    return {
        "info": {"year": 2024, "version": "1.0"},
        "licenses": [],
        "images": [{"id": 1, "file_name": img_name, "width": 32, "height": 24}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }


def test_val_cli_minimal(tmp_path: Path):
    from tools import val as tool

    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    img = images / "x.jpg"
    _write_img(img)
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(_make_coco_single(img.name)))

    argv = [
        "val.py",
        "--images", str(images),
        "--ann", str(ann),
        "--weights", "none",
        "--device", "cpu",
        "--imgsz", "64",
        "--max-images", "1",
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old

    # No exception means minimal pipeline executed
    # Optionally assert no crash by checking nothing
