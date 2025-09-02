#!/usr/bin/env python3
"""Prepare the Aquarium dataset for training (Windows-friendly, no symlinks).

This script unzips a Kaggle Aquarium dataset archive and arranges it into a
COCO-style layout under data/aquarium with plain files (no soft/hard links),
so it works cross-platform, including Windows.

Target layout:
  <root>/
    images/train/*.jpg
    images/val/*.jpg
    train.json
    val.json

Usage:
  ./.venv/bin/python scripts/prepare_acquirium.py \
    --zip data/AquariumDataset.zip --root data/aquarium --clean

Notes:
  - If the JSON "images[].file_name" contains subdirectories (e.g., "train/img.jpg"),
    this script rewrites them to basenames to match the new images/ path.
  - Existing destination files are overwritten when --clean is used; otherwise,
    only missing files are copied.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
import zipfile
from typing import Optional, Tuple, Dict, Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def find_split_dirs(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Heuristically locate train/val image directories inside extracted content."""
    train = None
    val = None
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        name = d.name.lower()
        if name in {"train", "training"}:
            # Must contain image files
            if any(p.suffix.lower() in IMAGE_EXTS for p in d.glob("*")):
                train = d
        if name in {"val", "valid", "validation"}:
            if any(p.suffix.lower() in IMAGE_EXTS for p in d.glob("*")):
                val = d
    return train, val


def find_coco_json(dir_path: Path) -> Optional[Path]:
    # Priority: _annotations.coco.json within the directory
    cand = dir_path / "_annotations.coco.json"
    if cand.exists():
        return cand
    # Fallback: any *.json inside that looks like COCO (has images/annotations keys)
    for p in dir_path.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                return p
        except Exception:
            continue
    return None


def ensure_clean_dir(d: Path, *, clean: bool) -> None:
    if clean and d.exists():
        try:
            if d.is_symlink():
                d.unlink()
            elif d.is_dir():
                shutil.rmtree(d)
            else:
                d.unlink()
        except FileNotFoundError:
            pass
    d.mkdir(parents=True, exist_ok=True)


def rewrite_coco_filenames(json_path: Path, out_json: Path, *, strip_dirs: bool) -> Dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if strip_dirs:
        for img in data.get("images", []):
            fn = img.get("file_name", "")
            img["file_name"] = Path(fn).name  # basename only
    out_json.write_text(json.dumps(data), encoding="utf-8")
    return data


def copy_images(src_dir: Path, dst_dir: Path, *, overwrite: bool = False) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.glob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            dst = dst_dir / p.name
            if overwrite or not dst.exists():
                shutil.copy2(p, dst)
                n += 1
    return n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare Aquarium dataset (Windows-friendly; no symlinks)")
    ap.add_argument("--zip", default="data/AquariumDataset.zip", help="Path to Aquarium zip archive")
    ap.add_argument("--root", default="data/aquarium", help="Output dataset root directory")
    ap.add_argument("--clean", action="store_true", help="Clean existing output directories before copying")
    ap.add_argument("--keep-extract", action="store_true", help="Keep extracted raw directory after prepare")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip)
    root = Path(args.root)
    raw = root / "raw_extracted"

    if not zip_path.exists():
        print(f"[error] Zip not found: {zip_path}", file=sys.stderr)
        sys.exit(2)

    # Extract
    print(f"[info] Extracting: {zip_path} -> {raw}")
    extract_zip(zip_path, raw)

    # Locate split dirs and JSONs
    train_dir, val_dir = find_split_dirs(raw)
    if train_dir is None or val_dir is None:
        print("[error] Could not find train/val image folders in archive.", file=sys.stderr)
        print(f" Scanned under: {raw}", file=sys.stderr)
        sys.exit(3)
    train_json = find_coco_json(train_dir)
    val_json = find_coco_json(val_dir)
    if train_json is None or val_json is None:
        print("[error] Could not find COCO JSONs under the split folders.", file=sys.stderr)
        print(f" train dir: {train_dir}\n val dir: {val_dir}", file=sys.stderr)
        sys.exit(4)

    # Prepare destination layout
    images_train = root / "images" / "train"
    images_val = root / "images" / "val"
    ensure_clean_dir(images_train, clean=args.clean)
    ensure_clean_dir(images_val, clean=args.clean)

    # Copy images
    n_train = copy_images(train_dir, images_train, overwrite=args.clean)
    n_val = copy_images(val_dir, images_val, overwrite=args.clean)
    print(f"[info] Copied images: train={n_train}, val={n_val}")

    # Rewrite and write JSONs at root (strip nested subdirs in file_name)
    out_train_json = root / "train.json"
    out_val_json = root / "val.json"
    train_meta = rewrite_coco_filenames(train_json, out_train_json, strip_dirs=True)
    val_meta = rewrite_coco_filenames(val_json, out_val_json, strip_dirs=True)

    # Basic sanity: ensure image filenames exist after rewrite
    def missing_count(meta: Dict[str, Any], img_dir: Path) -> int:
        miss = 0
        for im in meta.get("images", []):
            p = img_dir / Path(im.get("file_name", "")).name
            if not p.exists():
                miss += 1
        return miss

    missing_train = missing_count(train_meta, images_train)
    missing_val = missing_count(val_meta, images_val)
    if missing_train or missing_val:
        print(
            f"[warn] Missing files referenced in JSON after prepare: train={missing_train}, val={missing_val}.\n"
            "      Check that the archive contains matching images and annotations.")

    # Optionally clean extracted raw directory
    if not args.keep_extract:
        try:
            shutil.rmtree(raw)
        except Exception:
            pass

    # Final summary
    print("[done] Prepared Aquarium under:", root)
    print(" Images:", images_train, images_val)
    print(" Annotations:", out_train_json, out_val_json)
    print(" Ready for scripts/train.py and scripts/transfer_learn_aquarium.py")


if __name__ == "__main__":
    main()
