from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


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
    # simple centered box
    return {"id": 1, "image_id": img_id, "category_id": 1, "bbox": [0.0, 0.0, 32.0, 24.0], "area": 768.0, "iscrowd": 0}


def test_train_cli_minimal_no_val(tmp_path: Path):
    from tools import train as tool

    # Build tiny train dataset
    tdir = tmp_path / "train_images"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "a.jpg").write_bytes(b"JPG")
    _write_img(tdir / "a.jpg")
    tann = tmp_path / "train.json"
    tann.write_text(json.dumps(_make_coco([(1, "a.jpg")], anns=[_one_box_ann(1)])))

    savedir = tmp_path / "runs" / "exp"
    argv = [
        "train.py",
        "--train-images", str(tdir),
        "--train-ann", str(tann),
        "--weights", "none",
        "--device", "cpu",
        "--imgsz", "64",
        "--epochs", "1",
        "--batch-size", "1",
        "--workers", "0",
        "--no-tqdm",
        "--save-dir", str(savedir),
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old

    assert (savedir / "ckpt.pt").exists()
    assert any(p.name.startswith("epoch") and p.suffix == ".pt" for p in savedir.iterdir())


def test_train_cli_with_val(tmp_path: Path):
    from tools import train as tool

    # Train dataset
    tdir = tmp_path / "train_images"
    tdir.mkdir(parents=True, exist_ok=True)
    _write_img(tdir / "a.jpg")
    tann = tmp_path / "train.json"
    tann.write_text(json.dumps(_make_coco([(1, "a.jpg")], anns=[_one_box_ann(1)])))

    # Val dataset
    vdir = tmp_path / "val_images"
    vdir.mkdir(parents=True, exist_ok=True)
    _write_img(vdir / "b.jpg")
    vann = tmp_path / "val.json"
    vann.write_text(json.dumps(_make_coco([(2, "b.jpg")], anns=[])))

    savedir = tmp_path / "runs" / "exp"
    argv = [
        "train.py",
        "--train-images", str(tdir),
        "--train-ann", str(tann),
        "--val-images", str(vdir),
        "--val-ann", str(vann),
        "--weights", "none",
        "--device", "cpu",
        "--imgsz", "64",
        "--epochs", "1",
        "--batch-size", "1",
        "--workers", "0",
        "--no-tqdm",
        "--save-dir", str(savedir),
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old

    # Artifacts exist
    assert (savedir / "ckpt.pt").exists()
    assert any(p.name.startswith("epoch") and p.suffix == ".pt" for p in savedir.iterdir())


def test_train_cli_weights_file_and_model_choice(tmp_path: Path):
    from tools import train as tool
    from leanyolo.models import get_model
    from leanyolo.data.coco import coco80_class_names
    import torch

    # Build tiny dataset
    tdir = tmp_path / "train_images"
    tdir.mkdir(parents=True, exist_ok=True)
    _write_img(tdir / "a.jpg")
    tann = tmp_path / "train.json"
    tann.write_text(json.dumps(_make_coco([(1, "a.jpg")], anns=[_one_box_ann(1)])))

    # Create a state_dict checkpoint for yolov10n
    # Match number of classes to our toy dataset (1 class)
    m = get_model("yolov10n", weights=None, class_names=["object"]) 
    ckpt = tmp_path / "init_state.pt"
    torch.save(m.state_dict(), ckpt)

    savedir = tmp_path / "runs" / "exp"
    argv = [
        "train.py",
        "--train-images", str(tdir),
        "--train-ann", str(tann),
        "--model", "yolov10n",
        "--weights", str(ckpt),
        "--device", "cpu",
        "--imgsz", "64",
        "--epochs", "1",
        "--batch-size", "1",
        "--workers", "0",
        "--no-tqdm",
        "--save-dir", str(savedir),
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old

    assert (savedir / "ckpt.pt").exists()


def test_train_cli_freeze_headreset_eval_flags(tmp_path: Path):
    from tools import train as tool

    # Train dataset
    tdir = tmp_path / "train_images"
    tdir.mkdir(parents=True, exist_ok=True)
    _write_img(tdir / "a.jpg")
    tann = tmp_path / "train.json"
    tann.write_text(json.dumps(_make_coco([(1, "a.jpg")], anns=[_one_box_ann(1)])))

    # Val dataset
    vdir = tmp_path / "val_images"
    vdir.mkdir(parents=True, exist_ok=True)
    _write_img(vdir / "b.jpg")
    vann = tmp_path / "val.json"
    vann.write_text(json.dumps(_make_coco([(2, "b.jpg")], anns=[])))

    savedir = tmp_path / "runs" / "exp2"
    argv = [
        "train.py",
        "--train-images", str(tdir),
        "--train-ann", str(tann),
        "--val-images", str(vdir),
        "--val-ann", str(vann),
        "--weights", "none",
        "--device", "cpu",
        "--imgsz", "64",
        "--epochs", "1",
        "--batch-size", "1",
        "--workers", "0",
        "--no-tqdm",
        "--log-interval", "1",
        "--lr", "1e-5",
        "--freeze-backbone",
        "--head-reset",
        "--eval-conf", "0.5",
        "--eval-iou", "0.5",
        "--eval-progress",
        "--save-dir", str(savedir),
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old

    assert (savedir / "ckpt.pt").exists()
