from __future__ import annotations

"""
Generate reference outputs from the official YOLOv10 implementation for a set of
pretrained weights and deterministic inputs. These references are used by the
fidelity tests to verify the leanyolo implementation.

Usage:
  python -m leanyolo.tests.fidelity.generate_references --sizes n s m l x --img 320

Environment:
  - LEANYOLO_CACHE_DIR may be set to point to weights cache.
  - LEANYOLO_WEIGHTS_DIR may be set to a directory holding weight files.
"""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _add_official_to_path() -> None:
    root = _repo_root()
    candidates = [
        os.path.join(root, "references", "yolov10", "THU-MIG.yoloe"),
    ]
    for off in candidates:
        if os.path.isdir(off) and off not in os.sys.path:
            os.sys.path.insert(0, off)
            return


def _y10_yaml_for(size: str) -> str:
    return f"ultralytics/cfg/models/v10/yolov10{size}.yaml"


def _variant_name(size: str) -> str:
    return f"yolov10{size}"


def _weights_path_for(model_name: str) -> str:
    # Use the project's weight resolver metadata to download to cache if missing, without torch.load.
    from leanyolo.models import get_model_weights

    weights_enum = get_model_weights(model_name)()
    entry = weights_enum.get(model_name, "PRETRAINED_COCO")
    cache_dir = os.environ.get("LEANYOLO_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "leanyolo"))
    filename = entry.filename or f"{model_name}.pt"
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        # Avoid torch.load here; use the entry's download helper
        entry._download_to(entry.url, path, progress=True)
    return path


def _deterministic_input(HW: int) -> torch.Tensor:
    # Stable pseudo-random tensor with fixed seed
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, HW, HW, generator=g)
    return x


def _hook_indices() -> Tuple[Sequence[int], Sequence[int]]:
    # Indices valid across sizes per official YAMLs
    # Backbone c3,c4,c5 at [4,6,10]; Neck p3,p4,p5 at [16,19,22]
    return (4, 6, 10), (16, 19, 22)


def _collect_official_outputs(size: str, img: int, weights_path: str) -> Dict[str, torch.Tensor]:
    _add_official_to_path()
    # Import official tasks and monkeypatch torch_safe_load to use weights_only=False for PyTorch>=2.6
    import ultralytics.nn.tasks as tasks  # type: ignore
    from ultralytics.nn.tasks import YOLOv10DetectionModel, attempt_load_one_weight
    # Monkeypatch: force legacy torch.load behavior compatible with official checkpoints
    def _tsafe(weight: str):
        ckpt = torch.load(weight, map_location="cpu", weights_only=False)
        return ckpt, weight

    try:
        tasks.torch_safe_load = _tsafe  # type: ignore[attr-defined]
    except Exception:
        pass

    yaml_path = _y10_yaml_for(size)
    model = YOLOv10DetectionModel(yaml_path, ch=3, nc=80, verbose=False)
    _m, ckpt = attempt_load_one_weight(weights_path, device="cpu", inplace=True, fuse=False)
    model.load(ckpt, verbose=False)
    model.eval()

    # Hooks for backbone and neck
    feats: Dict[int, torch.Tensor] = {}
    c_idx, n_idx = _hook_indices()
    hooks = [model.model[i].register_forward_hook(lambda m, inp, out, i=i: feats.__setitem__(i, out)) for i in (*c_idx, *n_idx)]

    x = _deterministic_input(img)
    with torch.no_grad():
        out_full = model(x)  # outputs depend on official version; for Detect models returns train outputs

    for h in hooks:
        h.remove()

    # out_full for YOLOv10 returns dict with 'one2many' (tuple: (decoded, raw_list)) in eval mode.
    # Normalize to a list of three raw tensors [P3, P4, P5] from the second element.
    if isinstance(out_full, dict):
        o = out_full.get("one2many", None)
        if o is None:
            raise RuntimeError("Unexpected official model output dict structure; expected key 'one2many'.")
        if isinstance(o, (list, tuple)) and len(o) == 2 and isinstance(o[1], (list, tuple)):
            head_out = list(o[1])  # raw per-level outputs
        elif isinstance(o, (list, tuple)):
            # Some variants may return only raw list in eval; accept directly
            head_out = list(o)
        else:
            raise RuntimeError(f"Unexpected 'one2many' structure: type={type(o)}")
    elif isinstance(out_full, (list, tuple)):
        head_out = list(out_full)
    else:
        # Fallback: if it's a single tensor, split by stride (not expected for v10)
        raise RuntimeError(f"Unexpected official model output type: {type(out_full)}")

    c3, c4, c5 = feats[c_idx[0]], feats[c_idx[1]], feats[c_idx[2]]
    p3, p4, p5 = feats[n_idx[0]], feats[n_idx[1]], feats[n_idx[2]]

    return {
        "input": x,
        "backbone_c3": c3,
        "backbone_c4": c4,
        "backbone_c5": c5,
        "neck_p3": p3,
        "neck_p4": p4,
        "neck_p5": p5,
        "head_p3": head_out[0],
        "head_p4": head_out[1],
        "head_p5": head_out[2],
    }


def _save_refs(model_name: str, refs: Dict[str, torch.Tensor], img: int, weights_path: str) -> None:
    from .common import save_tensor, refs_dir, ref_path, ensure_dirs, write_json

    ensure_dirs()
    d = refs_dir(model_name)
    os.makedirs(d, exist_ok=True)
    meta = {
        "model": model_name,
        "img": img,
        "weights_path": weights_path,
        "dtype": str(refs["input"].dtype),
        "shapes": {k: tuple(map(int, v.shape)) for k, v in refs.items()},
    }
    write_json(os.path.join(d, "meta.json"), meta)
    for k, v in refs.items():
        save_tensor(ref_path(model_name, k), v)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="*", default=["n", "s", "m", "l", "x"], help="Model sizes to process")
    p.add_argument("--img", type=int, default=320, help="Input size (square)")
    args = p.parse_args(argv)

    for size in args.sizes:
        model_name = _variant_name(size)
        print(f"[gen] {model_name} @ {args.img}x{args.img}")
        wpath = _weights_path_for(model_name)
        refs = _collect_official_outputs(size, args.img, wpath)
        _save_refs(model_name, refs, args.img, wpath)
        print(f"[gen] Saved references for {model_name}")


if __name__ == "__main__":  # pragma: no cover
    main()
