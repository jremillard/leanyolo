from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("LEANYOLO_WEIGHTS_DIR"),
    reason="Requires LEANYOLO_WEIGHTS_DIR with PRETRAINED_COCO weights present",
)
def test_convert_official_cli_with_env(tmp_path: Path, monkeypatch):
    from tools import convert_official_weights as tool

    # Force CPU
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    out = tmp_path / "yolov10n.state_dict.pt"
    argv = [
        "convert_official_weights.py",
        "--model", "yolov10n",
        "--out", str(out),
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old
    assert out.exists()

