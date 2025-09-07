from __future__ import annotations

import json
import sys
from pathlib import Path


def _make_coco_root(root: Path) -> None:
    imgs = root / "images" / "val2017"
    ann_dir = root / "annotations"
    imgs.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    # Create a placeholder image file with correct extension (content unused here)
    (imgs / "000000000001.jpg").write_bytes(b"JPG")
    (imgs / "000000000002.jpg").write_bytes(b"JPG")
    data = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 10, "height": 10},
            {"id": 2, "file_name": "000000000002.jpg", "width": 10, "height": 10},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "obj"}],
    }
    (ann_dir / "instances_val2017.json").write_text(json.dumps(data))


def test_prepare_coco_sanity_subset_cli(tmp_path: Path):
    from tools import prepare_coco as tool

    root = tmp_path / "coco"
    _make_coco_root(root)

    argv = [
        "prepare_coco.py",
        "--root", str(root),
        "--sanity", "1",
        "--sanity-name", "coco-sanity1",
        "--no-link",
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old

    subset = root / "coco-sanity1"
    assert (subset / "images").exists()
    assert (subset / "annotations.json").exists()

