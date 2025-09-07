from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


def _write_img(path: Path, w: int = 32, h: int = 24) -> None:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def _make_coco(images: list[tuple[int, str]], anns: list[dict] | None = None) -> dict:
    return {
        "info": {"year": 2024, "version": "1.0"},
        "licenses": [],
        "images": [
            {"id": img_id, "file_name": fname, "width": 32, "height": 24}
            for img_id, fname in images
        ],
        "annotations": anns or [],
        "categories": [{"id": 1, "name": "object"}],
    }


def _one_box_ann(img_id: int) -> dict:
    return {"id": 1, "image_id": img_id, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "area": 100.0, "iscrowd": 0}


def test_train_cli_missing_required_args_exits():
    from tools import train as tool

    old = sys.argv[:]
    sys.argv = ["train.py"]
    try:
        with pytest.raises(SystemExit) as e:
            tool.main()
        assert e.value.code == 2
    finally:
        sys.argv = old


def test_train_cli_invalid_model_choice_exits(tmp_path: Path):
    from tools import train as tool

    # Minimal valid dataset to reach argparse validation
    tdir = tmp_path / "train_images"
    tdir.mkdir(parents=True, exist_ok=True)
    _write_img(tdir / "a.jpg")
    tann = tmp_path / "train.json"
    tann.write_text(json.dumps(_make_coco([(1, "a.jpg")], anns=[_one_box_ann(1)])))

    argv = [
        "train.py",
        "--train-images", str(tdir),
        "--train-ann", str(tann),
        "--model", "yolov10z",  # invalid choice
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as e:
            tool.main()
        assert e.value.code == 2
    finally:
        sys.argv = old


def test_train_cli_missing_train_paths_raises(tmp_path: Path):
    from tools import train as tool

    # Nonexistent ann path triggers FileNotFoundError
    argv = [
        "train.py",
        "--train-images", str(tmp_path / "train_images_dne"),
        "--train-ann", str(tmp_path / "train.json_dne"),
        "--weights", "none",
        "--device", "cpu",
        "--epochs", "1",
        "--batch-size", "1",
        "--workers", "0",
        "--no-tqdm",
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        with pytest.raises(FileNotFoundError):
            tool.main()
    finally:
        sys.argv = old


def test_train_cli_missing_val_paths_raises(tmp_path: Path):
    from tools import train as tool

    # Build train dataset
    tdir = tmp_path / "train_images"
    tdir.mkdir(parents=True, exist_ok=True)
    _write_img(tdir / "a.jpg")
    tann = tmp_path / "train.json"
    tann.write_text(json.dumps(_make_coco([(1, "a.jpg")], anns=[_one_box_ann(1)])))

    # Provide nonexistent val paths
    argv = [
        "train.py",
        "--train-images", str(tdir),
        "--train-ann", str(tann),
        "--val-images", str(tmp_path / "val_images_dne"),
        "--val-ann", str(tmp_path / "val.json_dne"),
        "--weights", "none",
        "--device", "cpu",
        "--epochs", "1",
        "--batch-size", "1",
        "--workers", "0",
        "--no-tqdm",
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        with pytest.raises(FileNotFoundError):
            tool.main()
    finally:
        sys.argv = old


def test_train_cli_invalid_weights_file_raises(tmp_path: Path):
    from tools import train as tool

    # Build minimal train dataset
    tdir = tmp_path / "train_images"
    tdir.mkdir(parents=True, exist_ok=True)
    _write_img(tdir / "a.jpg")
    tann = tmp_path / "train.json"
    tann.write_text(json.dumps(_make_coco([(1, "a.jpg")], anns=[_one_box_ann(1)])))

    # Create an incompatible checkpoint (empty dict)
    bad = tmp_path / "bad.pt"
    torch.save({}, bad)

    argv = [
        "train.py",
        "--train-images", str(tdir),
        "--train-ann", str(tann),
        "--weights", str(bad),
        "--device", "cpu",
        "--epochs", "1",
        "--batch-size", "1",
        "--workers", "0",
        "--no-tqdm",
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        with pytest.raises(Exception):
            tool.main()
    finally:
        sys.argv = old

