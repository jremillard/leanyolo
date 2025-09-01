#!/usr/bin/env python3
from __future__ import annotations

"""Download and prepare the Aquarium dataset.

This script attempts to fetch the dataset from Kaggle or use a local zip file
and sets up a COCO-style directory under data/aquarium/:

data/aquarium/
  images/train/ ... .jpg
  images/val/   ... .jpg
  train.json
  val.json

Notes:
- Kaggle API requires KAGGLE_USERNAME and KAGGLE_KEY (or kaggle.json).
- If you already have AquariumDataset.zip under data/, pass --zip to reuse it.
- If the dataset provides COCO JSONs, they will be linked/copied. Otherwise,
  you'll need to convert annotations manually.
"""

import argparse
import os
from pathlib import Path
import zipfile


def maybe_extract_zip(zip_path: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst)
    return dst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/aquarium", help="Output root directory")
    p.add_argument("--zip", default="data/AquariumDataset.zip", help="Optional local zip path")
    p.add_argument("--kaggle-dataset", default="sharansmenon/aquarium-dataset", help="Kaggle dataset ID")
    args = p.parse_args()

    root = Path(args.root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    zip_path = Path(args.zip)
    if zip_path.exists():
        print(f"[info] Using local zip: {zip_path}")
        maybe_extract_zip(zip_path, raw)
    else:
        print("[info] Attempting Kaggle download...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(args.kaggle_dataset, path=str(raw), unzip=True)
        except Exception as e:
            print("[error] Kaggle download failed. Provide --zip or configure Kaggle API.")
            print(e)
            return

    # Try to locate COCO-style annotations and split folders in extracted content
    # Heuristics:
    # - JSONs may be named `_annotations.coco.json` inside split dirs
    # - Split directories commonly named `train`, `valid`/`val`
    train_json = None
    val_json = None
    images_train = None
    images_val = None
    for d in raw.rglob("*"):
        if not d.is_dir():
            continue
        dn = d.name.lower()
        if dn in ("train", "training"):
            images_train = d
            cand = d / "_annotations.coco.json"
            if cand.exists():
                train_json = cand
        if dn in ("val", "valid", "validation"):
            images_val = d
            cand = d / "_annotations.coco.json"
            if cand.exists():
                val_json = cand
    # Fallback: scan for any jsons mentioning train/val
    if train_json is None or val_json is None:
        for pth in raw.rglob("*.json"):
            name = pth.name.lower()
            if train_json is None and "train" in name:
                train_json = pth
            if val_json is None and ("val" in name or "valid" in name):
                val_json = pth

    # Prepare output layout
    (root / "images").mkdir(parents=True, exist_ok=True)
    out_train = root / "images" / "train"
    out_val = root / "images" / "val"
    # If symlinking entire dirs is possible, replace existing empty dirs with symlinks
    def _link_dir(src: Path, dst: Path) -> None:
        if src is None:
            return
        try:
            if dst.exists() and dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()
            if not dst.exists():
                os.symlink(str(src.resolve()), str(dst), target_is_directory=True)
                return
        except Exception:
            pass
        # Fallback: ensure dst exists, copy files if empty
        dst.mkdir(parents=True, exist_ok=True)
        if not any(dst.iterdir()):
            for p in src.glob("*.jpg"):
                try:
                    os.link(str(p.resolve()), str(dst / p.name))
                except Exception:
                    import shutil
                    shutil.copy2(p, dst)

    # Symlink images if found
    if images_train and images_train.exists():
        _link_dir(images_train, out_train)
    if images_val and images_val.exists():
        _link_dir(images_val, out_val)

    # Link annotations if found
    if train_json:
        try:
            os.symlink(str(train_json.resolve()), str(root / "train.json"))
        except Exception:
            (root / "train.json").write_text(train_json.read_text())
    if val_json:
        try:
            os.symlink(str(val_json.resolve()), str(root / "val.json"))
        except Exception:
            (root / "val.json").write_text(val_json.read_text())

    print("[done] Prepared under:", root)
    print("Images:", out_train, out_val)
    print("Annotations:", root / "train.json", root / "val.json")
    print("If annotations are not COCO-style, please convert them and rerun.")


if __name__ == "__main__":
    main()
