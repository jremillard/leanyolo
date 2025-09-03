#!/usr/bin/env python3
"""Download COCO val2017 and optionally create a small sanity subset.

Examples
- Download full val split only:
  ./tools/prepare_coco.py --root data/coco --download

- Download and build a 50-image sanity subset:
  ./tools/prepare_coco.py --root data/coco --download --sanity 50 --sanity-name coco-sanity50
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from leanyolo.data.coco import ensure_coco_val, list_images


def _filter_annotations(src_ann: Path, keep_image_ids: List[int]) -> Dict:
    data = json.loads(Path(src_ann).read_text())
    keep = set(keep_image_ids)
    images = [im for im in data.get("images", []) if im.get("id") in keep]
    anns = [an for an in data.get("annotations", []) if an.get("image_id") in keep]
    # Keep all categories for simplicity; COCO eval can handle empty-category images
    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": data.get("categories", []),
    }
    return out


def _build_sanity_subset(root: Path, n: int, name: str, link: bool = True) -> Path:
    images_dir = root / "images" / "val2017"
    ann_json = root / "annotations" / "instances_val2017.json"
    if not images_dir.exists() or not ann_json.exists():
        raise FileNotFoundError("Missing COCO val2017; run with --download or provide root.")

    subset_dir = root / name
    imgs_out = subset_dir / "images"
    ann_out = subset_dir / "annotations.json"
    subset_dir.mkdir(parents=True, exist_ok=True)
    imgs_out.mkdir(parents=True, exist_ok=True)

    imgs = list_images(images_dir)[:n]
    # Map from file name to image id by scanning annotations’ images
    data = json.loads(Path(ann_json).read_text())
    name_to_id = {im.get("file_name"): im.get("id") for im in data.get("images", [])}
    keep_ids = []
    for p in imgs:
        if link:
            dst = imgs_out / p.name
            try:
                if not dst.exists():
                    os.symlink(os.path.abspath(p), dst)
            except OSError:
                # Fallback to copy if symlink not permitted
                import shutil
                shutil.copy2(p, dst)
        else:
            import shutil
            shutil.copy2(p, imgs_out / p.name)
        keep_ids.append(name_to_id[p.name])

    out = _filter_annotations(ann_json, keep_ids)
    ann_out.write_text(json.dumps(out))
    return subset_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare COCO val2017 and optional sanity subset")
    ap.add_argument("--root", default="data/coco", help="COCO root directory")
    ap.add_argument("--download", action="store_true", help="Download val2017 if missing")
    ap.add_argument("--sanity", type=int, default=0, help="Create a subset with first N images")
    ap.add_argument("--sanity-name", default="coco-sanity50", help="Subset directory name under root")
    ap.add_argument("--no-link", action="store_true", help="Copy files instead of symlink")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    ensure_coco_val(root, download=args.download)
    if args.sanity and args.sanity > 0:
        subset = _build_sanity_subset(root, args.sanity, args.sanity_name, link=not args.no_link)
        print(f"Sanity subset created at: {subset}")
    else:
        print(f"COCO val2017 is ready under: {root}")


if __name__ == "__main__":
    main()
