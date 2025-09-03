#!/usr/bin/env python3
"""Download all official YOLOv10 pretrained weights into the cache.

This script resolves the PRETRAINED_COCO weights for every YOLOv10 variant and
downloads them into the default cache directory used by the project.

Cache directory
- Default: ~/.cache/leanyolo
- Override: set env var LEANYOLO_CACHE_DIR to a writable path

Usage
  ./.venv/bin/python tools/download_all_pretrained.py
"""
from __future__ import annotations

import os

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'leanyolo' is importable when this
# script is run from outside the repo root (e.g., in containers or CI).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from leanyolo.models.registry import list_models, get_model_weights


def _default_cache_dir() -> str:
    return os.environ.get(
        "LEANYOLO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "leanyolo")
    )


def main() -> None:
    cache_dir = _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache: {cache_dir}")

    names = list(list_models())
    ok = []
    failed = []
    for name in names:
        try:
            entry = get_model_weights(name)().get(name, "PRETRAINED_COCO")
            _ = entry.get_state_dict(progress=True)
            print(f"[ok] {name}: {entry.filename}")
            ok.append(name)
        except Exception as e:  # pragma: no cover - network/filesystem dependent
            print(f"[fail] {name}: {e}")
            failed.append((name, str(e)))

    print(f"\nSummary: {len(ok)} ok, {len(failed)} failed")
    if failed:
        for n, msg in failed:
            print(f" - {n}: {msg}")


if __name__ == "__main__":
    main()
