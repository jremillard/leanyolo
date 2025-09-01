import json
import os
from pathlib import Path

import cv2
import numpy as np

from val import validate_coco


def _make_synthetic_coco(tmp_path: Path, n: int = 2):
    root = tmp_path / "coco"
    (root / "images" / "val2017").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    images = []
    for i in range(n):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        image_id = 100000 + i
        p = root / "images" / "val2017" / f"{image_id}.jpg"
        cv2.imwrite(str(p), img)
        images.append({"id": image_id, "file_name": f"{image_id}.jpg", "height": 64, "width": 64})

    categories = [{"id": i + 1, "name": f"c{i+1}", "supercategory": "none"} for i in range(80)]
    ann = {"info": {}, "licenses": [], "images": images, "annotations": [], "categories": categories}
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps(ann))
    return root


def test_validate_coco_runs_on_synthetic(tmp_path):
    data_root = _make_synthetic_coco(tmp_path)
    stats = validate_coco(
        model_name="yolov10s",
        weights=None,  # random weights
        data_root=str(data_root),
        imgsz=64,
        device="cpu",
        max_images=2,
    )
    assert "mAP50-95" in stats
    assert -1.0 <= stats["mAP50-95"] <= 1.0
