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

    # Try to locate COCO-style annotations in the extracted content
    # Heuristic: look for *train*.json and *val*.json under raw
    train_json = None
    val_json = None
    images_train = None
    images_val = None
    for pth in raw.rglob("*.json"):
        name = pth.name.lower()
        if "train" in name and ("ann" in name or "json" in name):
            train_json = pth
        if ("val" in name or "valid" in name) and ("ann" in name or "json" in name):
            val_json = pth
    for d in raw.rglob("*"):
        if d.is_dir():
            dn = d.name.lower()
            if dn in ("train", "training"):
                images_train = d
            if dn in ("val", "valid", "validation"):
                images_val = d

    # Prepare output layout
    (root / "images").mkdir(parents=True, exist_ok=True)
    out_train = root / "images" / "train"
    out_val = root / "images" / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # Symlink images if found
    if images_train and images_train.exists():
        try:
            os.symlink(str(images_train.resolve()), str(out_train), target_is_directory=True)
        except Exception:
            pass
    if images_val and images_val.exists():
        try:
            os.symlink(str(images_val.resolve()), str(out_val), target_is_directory=True)
        except Exception:
            pass

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

