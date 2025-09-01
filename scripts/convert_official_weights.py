#!/usr/bin/env python3
from __future__ import annotations

"""Convert official THU-MIG YOLOv10 weights to a lean checkpoint.

This produces a file loadable by leanyolo.get_model(weights="/path/to/ckpt.pt")
without requiring ultralytics at runtime.

Usage:
  ./.venv/bin/python scripts/convert_official_weights.py \
      --model yolov10n \
      --out weights/yolov10n.lean.pt

Notes:
- Requires ultralytics installed only for the conversion step.
- The output file contains: {model_name, class_names, input_norm_subtract,
  input_norm_divide, state_dict} and is safe to load without ultralytics.
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch

from leanyolo.models import get_model, get_model_weights
from leanyolo.data.coco import coco80_class_names
from leanyolo.utils.remap import remap_official_yolov10_to_lean


def ensure_official_weight(model_name: str) -> str:
    entry = get_model_weights(model_name)().get(model_name, "PRETRAINED_COCO")
    cache_dir = os.environ.get(
        "LEANYOLO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "leanyolo")
    )
    os.makedirs(cache_dir, exist_ok=True)
    wpath = os.path.join(cache_dir, entry.filename or f"{model_name}.pt")
    if not os.path.exists(wpath):
        entry._download_to(entry.url, wpath, progress=True)
    return wpath


def load_official_ckpt(path: str):
    # Prefer the local yolov10-official repo (git-ignored sibling) for loader code
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    off = repo_root / "yolov10-official"
    if off.exists():
        sys.path.insert(0, str(off))
    # Use ultralytics attempt_load_one_weight to parse the checkpoint safely
    import ultralytics.nn.tasks as tasks  # type: ignore
    from ultralytics.nn.tasks import attempt_load_one_weight  # type: ignore

    # Force legacy torch.load behavior compatible with official checkpoints
    tasks.torch_safe_load = lambda weight: (
        torch.load(weight, map_location="cpu", weights_only=False),
        weight,
    )
    _model_obj, ckpt = attempt_load_one_weight(path, device="cpu", inplace=False, fuse=False)
    return ckpt


def convert(model_name: str, out_path: str) -> str:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    wpath = ensure_official_weight(model_name)
    ckpt = load_official_ckpt(wpath)

    class_names: List[str] = coco80_class_names()
    # Build lean model (no weights) for mapping
    model = get_model(
        model_name,
        weights=None,
        class_names=class_names,
    )
    # Map to our state dict
    mapped = remap_official_yolov10_to_lean(ckpt, model)

    # Save lean checkpoint
    out = {
        "leanyolo_version": "0.1",
        "model_name": model_name,
        "class_names": class_names,
        "input_norm_subtract": [0.0, 0.0, 0.0],
        "input_norm_divide": [255.0, 255.0, 255.0],
        "state_dict": mapped,
    }
    torch.save(out, str(out_p))
    return str(out_p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert official THU-MIG YOLOv10 weights to lean ckpt")
    p.add_argument("--model", required=True, choices=[
        "yolov10n","yolov10s","yolov10m","yolov10b","yolov10l","yolov10x"
    ])
    p.add_argument("--out", required=True, help="Output .pt path for lean checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = convert(args.model, args.out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
