from __future__ import annotations

import os
import time
from typing import Dict, List

import torch

from .common import ensure_dirs, load_inputs, ref_path, load_tensor, compare_tensors, rubric_for, model_variants
from .report_utils import record_report


def _load_lean_with_official_weights(model_name: str):
    from lean_yolo.models import get_model, get_model_weights
    from lean_yolo.utils.remap import remap_official_yolov10_to_lean

    m = get_model(model_name, weights=None)

    # Ensure official repo importable
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    off = os.path.join(repo_root, "yolov10-official")
    if off not in sys.path:
        sys.path.insert(0, off)
    import ultralytics.nn.tasks as tasks  # type: ignore
    from ultralytics.nn.tasks import attempt_load_one_weight  # type: ignore

    entry = get_model_weights(model_name)().get(model_name, "DEFAULT")
    wdir = os.environ.get("LEAN_YOLO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "lean_yolo"))
    wpath = os.path.join(wdir, entry.filename or f"{model_name}.pt")
    if not os.path.exists(wpath):
        os.makedirs(wdir, exist_ok=True)
        entry._download_to(entry.url, wpath, progress=True)
    tasks.torch_safe_load = lambda weight: (torch.load(weight, map_location="cpu", weights_only=False), weight)
    _model_obj, ckpt = attempt_load_one_weight(wpath, device="cpu", inplace=True, fuse=False)
    mapped = remap_official_yolov10_to_lean(ckpt, m)
    m.load_state_dict(mapped, strict=False)
    return m


def _run_components(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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


def run_all(variants=None):
    ensure_dirs()
    variants = variants or model_variants()
    summary: Dict[str, Dict[str, bool]] = {}
    for name in variants:
        rub = rubric_for(name)
        try:
            x = load_inputs(320)
            model = _load_lean_with_official_weights(name)
            outs = _run_components(model, x)
            refs = {k: load_tensor(ref_path(name, k)) for k in (
                "backbone_c3","backbone_c4","backbone_c5","neck_p3","neck_p4","neck_p5","head_p3","head_p4","head_p5"
            )}

            res = {
                "model": name,
                "img": 320,
                "components": {},
                "rubric": {
                    "backbone": rub.backbone_tol.__dict__,
                    "neck": rub.neck_tol.__dict__,
                    "head": rub.head_tol.__dict__,
                },
                "overall_pass": True,
            }
            ok = True
            for k, tol in (
                ("backbone_c3", rub.backbone_tol), ("backbone_c4", rub.backbone_tol), ("backbone_c5", rub.backbone_tol),
                ("neck_p3", rub.neck_tol), ("neck_p4", rub.neck_tol), ("neck_p5", rub.neck_tol),
                ("head_p3", rub.head_tol), ("head_p4", rub.head_tol), ("head_p5", rub.head_tol),
            ):
                cmp = compare_tensors(outs[k], refs[k], tol)
                res["components"][k] = cmp
                ok = ok and bool(cmp["shape_match"]) and bool(cmp["allclose"])
            res["overall_pass"] = ok
            summary[name] = {"pass": ok}
            record_report(name, res)
        except Exception as e:
            summary[name] = {"pass": False, "error": str(e)}
    # Write consolidated summary
    ts = time.strftime("%Y%m%d-%H%M%S")
    from .common import REPORTS_DIR, write_json
    write_json(os.path.join(REPORTS_DIR, f"summary-{ts}.json"), summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    s = run_all()
    print(s)

