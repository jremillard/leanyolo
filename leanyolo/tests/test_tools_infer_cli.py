from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def _make_img(path: Path, w: int = 32, h: int = 24) -> None:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = (0, 0, 0)
    cv2.imwrite(str(path), arr)


def test_infer_cli_minimal(tmp_path: Path):
    from tools import infer as tool

    img = tmp_path / "im.jpg"
    outdir = tmp_path / "out"
    _make_img(img)

    argv = [
        "infer.py",
        "--source", str(img),
        "--weights", "none",
        "--device", "cpu",
        "--imgsz", "64",
        "--save-dir", str(outdir),
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old

    # Output image should exist
    assert (outdir / img.name).exists()

