from __future__ import annotations

import os
import pytest
import torch

from .common import ensure_dirs, refs_dir, ref_path, load_inputs, load_tensor, compare_tensors, rubric_for


def _maybe_skip_if_missing_decoded(model_name: str) -> None:
    d = refs_dir(model_name)
    p = ref_path(model_name, "decoded_topk")
    if not os.path.exists(p):
        pytest.skip(
            f"Missing decoded_topk reference for '{model_name}' in '{d}'. "
            "Regenerate refs: python -m leanyolo.tests.fidelity.generate_references --sizes "
            f"{model_name[-1]} --img 320"
        )


def _decode_lean_topk(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Use one-to-one branch to mirror official eval decode
    with torch.no_grad():
        c3, c4, c5 = model.backbone(x)
        p3, p4, p5 = model.neck(c3, c4, c5)
        r = model.head.forward_feat((p3, p4, p5), model.head.one2one_cv2, model.head.one2one_cv3)  # type: ignore[attr-defined]
        from leanyolo.models.yolov10.postprocess import decode_v10_official_topk

        dets = decode_v10_official_topk(r, num_classes=len(model.class_names), strides=(8, 16, 32), max_det=300)
    return dets[0]


def _run_decode_parity(model_name: str):
    ensure_dirs()
    _maybe_skip_if_missing_decoded(model_name)
    rub = rubric_for(model_name)

    x = load_inputs(320)
    from leanyolo.models import get_model
    from leanyolo.data.coco import coco80_class_names

    m = get_model(
        model_name,
        weights="PRETRAINED_COCO",
        class_names=coco80_class_names(),
        input_norm_subtract=[0.0],
        input_norm_divide=[1.0],
    ).eval()

    out = _decode_lean_topk(m, x)
    ref = load_tensor(ref_path(model_name, "decoded_topk"))

    # Align counts by top-K on confidence for stable comparison
    if out.numel() == 0 and ref.numel() == 0:
        return
    assert out.shape[1] == 6 and ref.shape[1] == 6
    k = min(out.shape[0], ref.shape[0])
    so = torch.argsort(out[:, 4], descending=True)[:k]
    sr = torch.argsort(ref[:, 4], descending=True)[:k]
    oo = out[so]
    rr = ref[sr]

    for j in range(6):
        cmp = compare_tensors(oo[:, j], rr[:, j], rub.end2end_tol)
        assert cmp["allclose"], f"decoded col {j} mismatch: {cmp}"


@pytest.mark.fidelity
def test_decode_parity_yolov10n():
    _run_decode_parity("yolov10n")


@pytest.mark.fidelity
def test_decode_parity_yolov10s():
    _run_decode_parity("yolov10s")


@pytest.mark.fidelity
def test_decode_parity_yolov10m():
    _run_decode_parity("yolov10m")


@pytest.mark.fidelity
def test_decode_parity_yolov10l():
    _run_decode_parity("yolov10l")


@pytest.mark.fidelity
def test_decode_parity_yolov10x():
    _run_decode_parity("yolov10x")

