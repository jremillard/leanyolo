from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import pytest
import torch

from lean_yolo.models import get_model
from lean_yolo.utils.remap import remap_official_yolov10_to_lean, extract_state_dict
from .common import (
    ensure_dirs,
    load_inputs,
    load_tensor,
    ref_path,
    refs_dir,
    rubric_for,
)
from .report_utils import record_report


REF_KEYS = (
    "backbone_c3",
    "backbone_c4",
    "backbone_c5",
    "neck_p3",
    "neck_p4",
    "neck_p5",
    "head_p3",
    "head_p4",
    "head_p5",
)


def _maybe_skip_if_missing_refs(model_name: str) -> None:
    d = refs_dir(model_name)
    missing = [k for k in REF_KEYS if not os.path.exists(ref_path(model_name, k))]
    if missing:
        pytest.skip(
            f"Missing reference outputs for '{model_name}' in '{d}'. "
            "Run: python -m lean_yolo.tests.fidelity.generate_references --sizes "
            f"{model_name[-1]} --img 320"
        )


def _run_lean_components(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        c3, c4, c5 = model.backbone(x)
        p3, p4, p5 = model.neck(c3, c4, c5)
        h: List[torch.Tensor] = model.head((p3, p4, p5))
    return {
        "backbone_c3": c3,
        "backbone_c4": c4,
        "backbone_c5": c5,
        "neck_p3": p3,
        "neck_p4": p4,
        "neck_p5": p5,
        "head_p3": h[0],
        "head_p4": h[1],
        "head_p5": h[2],
    }


def run_fidelity_for_variant(model_name: str) -> None:
    ensure_dirs()
    _maybe_skip_if_missing_refs(model_name)

    # Load rubric (tolerances)
    rub = rubric_for(model_name)

    # Load inputs and model
    x = load_inputs(320)
    # Build lean model and load official weights via official loader to avoid torch.load safety issues
    m = get_model(model_name, weights=None)
    # Use official loader to get checkpoint then remap
    # Add official repo to path
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    off = os.path.join(repo_root, "yolov10-official")
    if off not in sys.path:
        sys.path.insert(0, off)
    import ultralytics.nn.tasks as tasks  # type: ignore
    from ultralytics.nn.tasks import attempt_load_one_weight  # type: ignore
    # Resolve weight file path using our registry metadata (no torch.load)
    from lean_yolo.models import get_model_weights

    entry = get_model_weights(model_name)().get(model_name, "DEFAULT")
    wdir = os.environ.get("LEAN_YOLO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "lean_yolo"))
    wpath = os.path.join(wdir, entry.filename or f"{model_name}.pt")
    if not os.path.exists(wpath):
        os.makedirs(wdir, exist_ok=True)
        entry._download_to(entry.url, wpath, progress=True)
    # Monkeypatch official torch_safe_load to force weights_only=False
    tasks.torch_safe_load = lambda weight: (torch.load(weight, map_location="cpu", weights_only=False), weight)
    # Load ckpt and map
    _model_obj, ckpt = attempt_load_one_weight(wpath, device="cpu", inplace=True, fuse=False)
    mapped = remap_official_yolov10_to_lean(ckpt, m)
    m.load_state_dict(mapped, strict=False)

    # Compute lean outputs
    outs = _run_lean_components(m, x)

    # Load references
    refs = {k: load_tensor(ref_path(model_name, k)) for k in REF_KEYS}

    # Compare per-component with rubric tolerances
    from .common import compare_tensors

    results: Dict[str, Dict] = {
        "model": model_name,
        "img": 320,
        "rubric": {
            "backbone": rub.backbone_tol.__dict__,
            "neck": rub.neck_tol.__dict__,
            "head": rub.head_tol.__dict__,
            "end2end": rub.end2end_tol.__dict__,
        },
        "components": {},
        "overall_pass": True,
    }

    # Backbone
    for k in ("backbone_c3", "backbone_c4", "backbone_c5"):
        cmp = compare_tensors(outs[k], refs[k], rub.backbone_tol)
        results["components"][k] = cmp
        assert cmp["shape_match"], f"Shape mismatch for {k}"
        assert cmp["allclose"], f"Numerical mismatch for {k}: {cmp}"

    # Neck
    for k in ("neck_p3", "neck_p4", "neck_p5"):
        cmp = compare_tensors(outs[k], refs[k], rub.neck_tol)
        results["components"][k] = cmp
        assert cmp["shape_match"], f"Shape mismatch for {k}"
        assert cmp["allclose"], f"Numerical mismatch for {k}: {cmp}"

    # Head (training-style outputs)
    for k in ("head_p3", "head_p4", "head_p5"):
        cmp = compare_tensors(outs[k], refs[k], rub.head_tol)
        results["components"][k] = cmp
        assert cmp["shape_match"], f"Shape mismatch for {k}"
        assert cmp["allclose"], f"Numerical mismatch for {k}: {cmp}"

    # Record report artifact
    _ = record_report(model_name, results)


# Thin wrappers to integrate with pytest discovery per variant


@pytest.mark.fidelity
def test_fidelity_yolov10n():
    run_fidelity_for_variant("yolov10n")


@pytest.mark.fidelity
def test_fidelity_yolov10s():
    run_fidelity_for_variant("yolov10s")


@pytest.mark.fidelity
def test_fidelity_yolov10m():
    run_fidelity_for_variant("yolov10m")


@pytest.mark.fidelity
def test_fidelity_yolov10l():
    run_fidelity_for_variant("yolov10l")


@pytest.mark.fidelity
def test_fidelity_yolov10x():
    run_fidelity_for_variant("yolov10x")
