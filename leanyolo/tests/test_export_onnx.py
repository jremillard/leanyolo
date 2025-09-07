from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names


@pytest.mark.parametrize("model_name", ["yolov10n"])  # keep small and fast
def test_export_onnx_roundtrip(tmp_path: Path, model_name: str):
    imgsz = 320
    max_dets = 50
    batch = 2
    cn = coco80_class_names()
    model = get_model(
        model_name,
        weights=None,  # random init is fine for structural check
        class_names=cn,
        input_norm_subtract=[0.0, 0.0, 0.0],
        input_norm_divide=[255.0, 255.0, 255.0],
    ).eval()

    onnx_path = tmp_path / f"{model_name}.onnx"

    # Export
    from leanyolo.models.yolov10.export import export_onnx as _export_onnx

    _export_onnx(
        model,
        str(onnx_path),
        dummy_batch=batch,
        imgsz=imgsz,
        opset=19,
        half=False,
        max_dets=max_dets,
        conf=0.10,  # low conf so we have nonzero dets from random
    )
    assert onnx_path.exists(), "ONNX file not created"

    # Try to validate shapes via onnxruntime if available
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np
    except Exception:
        pytest.skip("onnxruntime not available; skipping runtime roundtrip")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # CPU provider
    x = torch.rand((batch, 3, imgsz, imgsz)).float()
    outs = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    dets, num = outs
    assert dets.shape == (batch, max_dets, 6)
    assert num.shape == (batch,)

    # Light parity check against PyTorch wrapper
    from leanyolo.models.yolov10.export import build_export_wrapper

    with torch.inference_mode():
        w = build_export_wrapper(model, imgsz=imgsz, max_dets=max_dets, conf=0.10)
        dets_pt, num_pt = w(x)
    assert dets_pt.shape == dets.shape
    assert tuple(num_pt.shape) == num.shape


@pytest.mark.parametrize("model_name", ["yolov10n"])  # small variant
def test_export_onnx_with_nms(tmp_path: Path, model_name: str):
    imgsz = 320
    max_dets = 50
    batch = 1
    cn = coco80_class_names()
    model = get_model(model_name, weights=None, class_names=cn,
                      input_norm_subtract=[0.0, 0.0, 0.0], input_norm_divide=[255.0, 255.0, 255.0]).eval()

    onnx_path = tmp_path / f"{model_name}_nms.onnx"
    from leanyolo.models.yolov10.export import export_onnx as _export_onnx
    _export_onnx(model, str(onnx_path), dummy_batch=batch, imgsz=imgsz, opset=19,
                 half=False, max_dets=max_dets, conf=0.10, decode='nms', iou=0.45, pre_topk=200)
    assert onnx_path.exists(), "ONNX (nms) file not created"
