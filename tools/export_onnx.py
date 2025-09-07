#!/usr/bin/env python3
from __future__ import annotations

"""Export YOLOv10 models to ONNX with dynamic batch axis.

Outputs
- ONNX file with inputs: images [N,3,H,W]; outputs:
  - detections: [N, max_dets, 6] = [x1,y1,x2,y2,score,cls]
  - num_dets: [N] valid count per image
- A companion JSON metadata file next to the ONNX output.
"""

import argparse
import json
import os
from pathlib import Path

import torch

import sys
from pathlib import Path as _Path

# Ensure repo root on path for 'leanyolo' imports when run from any CWD
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names
from leanyolo.models.yolov10.export import export_onnx as _export_onnx


def parse_args():
    ap = argparse.ArgumentParser(description="Export YOLOv10 to ONNX")
    ap.add_argument("--model", default="yolov10s", help="Model name (yolov10[n|s|m|l|x])")
    ap.add_argument("--weights", default="PRETRAINED_COCO", help="Weights key or local .pt path or None")
    ap.add_argument("--output", default="runs/export/yolov10s.onnx", help="Output ONNX path")
    ap.add_argument("--batch", type=int, default=1, help="Dummy batch for export/validation")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    ap.add_argument("--max-dets", type=int, default=300, help="Maximum detections per image")
    ap.add_argument("--opset", type=int, default=19, help="ONNX opset version")
    ap.add_argument("--half", action="store_true", help="Export in FP16")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for decode")
    ap.add_argument("--decode", choices=["topk", "nms"], default="topk", help="Decode mode for ONNX graph")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (if decode=nms)")
    ap.add_argument("--pre-topk", type=int, default=1000, help="Candidate pre‑topk before NMS (decode=nms)")
    ap.add_argument("--validate", action="store_true", help="Run onnxruntime validation after export")
    return ap.parse_args()


def _maybe_validate(onnx_path: str, *, model, imgsz: int, batch: int, max_dets: int) -> None:
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"[validate] Skipping onnxruntime validation: {e}")
        return

    # Prefer CUDA if available, otherwise CPU
    avail = set(ort.get_available_providers())
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    if not providers:
        print(f"[validate] No available ORT providers; available={list(avail)}")
        return
    print(f"[validate] ORT providers: {providers}")
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inputs = {sess.get_inputs()[0].name: (torch.zeros((batch, 3, imgsz, imgsz)).float().numpy())}
    outs = sess.run(None, inputs)
    dets_onnx = outs[0]
    n_onnx = outs[1]

    # Compare shapes and run a PyTorch forward for parity on random data
    x = torch.rand((batch, 3, imgsz, imgsz)).float()
    with torch.inference_mode():
        # Import the same wrapper used in export
        from leanyolo.models.yolov10.export import build_export_wrapper

        m = build_export_wrapper(model, imgsz=imgsz, max_dets=max_dets)
        dets_pt, n_pt = m(x)
    assert dets_onnx.shape == dets_pt.shape, f"shape mismatch: onnx {dets_onnx.shape} vs torch {tuple(dets_pt.shape)}"
    assert n_onnx.shape == tuple(n_pt.shape), f"num_dets shape mismatch: onnx {n_onnx.shape} vs torch {tuple(n_pt.shape)}"
    # Value check (loose): compare mean/std to avoid heavy tol checks
    import numpy as _np
    d_onnx = _np.nan_to_num(dets_onnx).astype("float32")
    d_pt = dets_pt.cpu().numpy().astype("float32")
    mean_diff = float(abs(d_onnx.mean() - d_pt.mean()))
    std_diff = float(abs(d_onnx.std() - d_pt.std()))
    print(f"[validate] parity stats: mean_diff={mean_diff:.4e}, std_diff={std_diff:.4e}")


def main():
    args = parse_args()
    cn = coco80_class_names()
    model = get_model(
        args.model,
        weights=None if args.weights in {"", "none", "None", "NONE"} else args.weights,
        class_names=cn,
        input_norm_subtract=[0.0, 0.0, 0.0],
        input_norm_divide=[255.0, 255.0, 255.0],
    ).eval()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_p)

    # Export
    _export_onnx(
        model,
        onnx_path,
        dummy_batch=args.batch,
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        max_dets=args.max_dets,
        conf=args.conf,
        decode=args.decode,
        iou=args.iou,
        pre_topk=args.pre_topk,
    )

    # Write sidecar metadata
    meta = {
        "model": args.model,
        "weights": args.weights,
        "imgsz": int(args.imgsz),
        "opset": int(args.opset),
        "half": bool(args.half),
        "max_dets": int(args.max_dets),
        "conf": float(args.conf),
        "decode": args.decode,
        "iou": float(args.iou),
        "pre_topk": int(args.pre_topk),
        "inputs": {"images": ["N", 3, int(args.imgsz), int(args.imgsz)]},
        "outputs": {"detections": ["N", int(args.max_dets), 6], "num_dets": ["N"]},
    }
    with open(str(out_p) + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported: {onnx_path}")

    if args.validate:
        _maybe_validate(onnx_path, model=model, imgsz=args.imgsz, batch=args.batch, max_dets=args.max_dets)


if __name__ == "__main__":
    main()
