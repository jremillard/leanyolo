from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest


def _make_coco(images):
    return {
        "images": [
            {
                "id": i + 1,
                "file_name": fn,
                "width": 10,
                "height": 10,
            }
            for i, fn in enumerate(images)
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "fish"}],
    }


def _build_zip(zip_path: Path, with_splits=True, with_json=True) -> None:
    with zipfile.ZipFile(zip_path, "w") as z:
        if with_splits:
            # Train
            z.writestr("train/img1.jpg", b"JPG")
            if with_json:
                z.writestr(
                    "train/_annotations.coco.json",
                    json.dumps(_make_coco(["train/img1.jpg"])),
                )
            # Val
            z.writestr("val/img2.jpg", b"JPG")
            if with_json:
                z.writestr(
                    "val/_annotations.coco.json",
                    json.dumps(_make_coco(["val/img2.jpg"])),
                )
        else:
            z.writestr("README.txt", b"no splits here")


def _run_prepare(tmp_path: Path, *, keep_extract=False, clean=False, zip_has_json=True, zip_has_splits=True):
    from tools import prepare_acquirium as prep

    root = tmp_path / "aquarium_out"
    zip_path = tmp_path / "AquariumDataset.zip"
    _build_zip(zip_path, with_splits=zip_has_splits, with_json=zip_has_json)

    argv = [
        "prepare_acquirium.py",
        "--zip",
        str(zip_path),
        "--root",
        str(root),
    ]
    if keep_extract:
        argv.append("--keep-extract")
    if clean:
        argv.append("--clean")

    # Monkeypatch sys.argv to simulate CLI
    import sys

    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        prep.main()
    except SystemExit as e:
        return e.code, root
    finally:
        sys.argv = old_argv
    return 0, root


def test_prepare_basic_and_json_rewrite(tmp_path: Path):
    code, root = _run_prepare(tmp_path)
    assert code == 0
    # Images copied
    assert (root / "images" / "train" / "img1.jpg").exists()
    assert (root / "images" / "val" / "img2.jpg").exists()
    # JSON rewritten to basenames
    tj = json.loads((root / "train.json").read_text())
    vj = json.loads((root / "val.json").read_text())
    assert tj["images"][0]["file_name"] == "img1.jpg"
    assert vj["images"][0]["file_name"] == "img2.jpg"
    # Raw extracted removed by default
    assert not (root / "raw_extracted").exists()


def test_prepare_keep_extract(tmp_path: Path):
    code, root = _run_prepare(tmp_path, keep_extract=True)
    assert code == 0
    assert (root / "raw_extracted").exists()


def test_prepare_clean_removes_extra(tmp_path: Path):
    # First run to create layout
    code, root = _run_prepare(tmp_path)
    assert code == 0
    extra = root / "images" / "train" / "extra.jpg"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"X")
    assert extra.exists()
    # Second run with --clean should remove extra
    code, _ = _run_prepare(tmp_path, clean=True)
    assert code == 0
    assert not extra.exists()


def test_prepare_missing_zip(tmp_path: Path):
    from tools import prepare_acquirium as prep
    import sys

    root = tmp_path / "aquarium_out"
    argv = [
        "prepare_acquirium.py",
        "--zip",
        str(tmp_path / "does_not_exist.zip"),
        "--root",
        str(root),
    ]
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as e:
            prep.main()
        assert e.value.code == 2
    finally:
        sys.argv = old_argv


def test_prepare_missing_split_dirs(tmp_path: Path):
    # Zip without train/val dirs
    from tools import prepare_acquirium as prep
    import sys

    root = tmp_path / "aquarium_out"
    zip_path = tmp_path / "AquariumDataset.zip"
    _build_zip(zip_path, with_splits=False)
    argv = [
        "prepare_acquirium.py",
        "--zip",
        str(zip_path),
        "--root",
        str(root),
    ]
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as e:
            prep.main()
        assert e.value.code == 3
    finally:
        sys.argv = old_argv


def test_prepare_missing_jsons(tmp_path: Path):
    # Zip with images but no COCO jsons
    code, _ = _run_prepare(tmp_path, zip_has_json=False)
    assert code == 4

