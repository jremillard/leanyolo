#!/usr/bin/env python3
"""Convert PRETRAINED_COCO weights to a plain state_dict checkpoint.

This script instantiates a lean YOLO model via `get_model(weights='PRETRAINED_COCO')`
and saves its `state_dict()` to the given path. The resulting `.pt` file is a
standard PyTorch state_dict compatible with `get_model(weights='/path/to/file.pt')`.

Usage:
  ./.venv/bin/python scripts/convert_official_weights.py \
      --model yolov10n \
      --out weights/yolov10n.state_dict.pt

Notes:
- Works for any variant that has a PRETRAINED_COCO entry in the weights registry.
- Does not require ultralytics; resolution and mapping are handled by leanyolo.
- For offline use, place the original weights file in a directory and set
  `LEANYOLO_WEIGHTS_DIR` to that directory; filenames should match the registry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names


def convert(model_name: str, out_path: str) -> str:
    """Resolve PRETRAINED_COCO weights via get_model and save a state_dict.

    Args:
        model_name: e.g., 'yolov10n', 'yolov10s', ...
        out_path: destination .pt file path
    Returns:
        The saved path as a string.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    class_names: List[str] = coco80_class_names()
    model = get_model(
        model_name,
        weights="PRETRAINED_COCO",
        class_names=class_names,
    )

    # Save a plain, strict-compatible state_dict
    torch.save(model.state_dict(), str(out_p))
    return str(out_p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert YOLOv10 PRETRAINED_COCO weights to a plain state_dict .pt")
    p.add_argument("--model", required=True, choices=[
        "yolov10n","yolov10s","yolov10m","yolov10b","yolov10l","yolov10x"
    ])
    p.add_argument("--out", required=True, help="Output .pt path for state_dict")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = convert(args.model, args.out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
