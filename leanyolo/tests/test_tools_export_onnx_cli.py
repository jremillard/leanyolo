from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _has_onnx() -> bool:
    return importlib.util.find_spec("onnx") is not None


@pytest.mark.skipif(not _has_onnx(), reason="onnx is not installed")
def test_export_onnx_cli_minimal(tmp_path: Path):
    from tools import export_onnx as tool

    out = tmp_path / "m.onnx"
    argv = [
        "export_onnx.py",
        "--model", "yolov10n",
        "--weights", "none",
        "--output", str(out),
        "--batch", "1",
        "--imgsz", "64",
        "--max-dets", "10",
        "--opset", "19",
        "--decode", "topk",
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        tool.main()
    finally:
        sys.argv = old
    assert out.exists()
    assert Path(str(out) + ".json").exists()

