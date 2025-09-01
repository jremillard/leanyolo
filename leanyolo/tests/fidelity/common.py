from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .rubric import Rubric, Tolerance, get_rubric


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUTS_DIR = os.path.join(DATA_DIR, "inputs")
REFS_DIR = os.path.join(DATA_DIR, "refs")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def ensure_dirs() -> None:
    for d in (DATA_DIR, INPUTS_DIR, REFS_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)


def input_path(name: str) -> str:
    return os.path.join(INPUTS_DIR, name)


def refs_dir(model_name: str) -> str:
    return os.path.join(REFS_DIR, model_name)


def ref_path(model_name: str, key: str) -> str:
    return os.path.join(refs_dir(model_name), f"{key}.pt")


def save_tensor(path: str, t: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(t.cpu(), path)


def load_tensor(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def compare_tensors(a: torch.Tensor, b: torch.Tensor, tol: Tolerance) -> Dict[str, float | bool]:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    same_shape = tuple(a.shape) == tuple(b.shape)
    if not same_shape:
        return {
            "shape_match": False,
            "allclose": False,
            "max_abs_err": float("inf"),
            "mae": float("inf"),
            "mse": float("inf"),
            "numel": float(a.numel()),
        }
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mae = float(diff.mean().item()) if diff.numel() else 0.0
    mse = float((diff**2).mean().item()) if diff.numel() else 0.0
    allc = torch.allclose(a, b, rtol=tol.rtol, atol=tol.atol)
    if tol.max_abs is not None:
        allc = allc and (max_abs <= tol.max_abs)
    return {
        "shape_match": True,
        "allclose": bool(allc),
        "max_abs_err": max_abs,
        "mae": mae,
        "mse": mse,
        "numel": float(a.numel()),
    }


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def model_variants() -> Sequence[str]:
    return ("yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x")


def load_inputs(size: int = 320) -> torch.Tensor:
    p = input_path(f"x_{size}.pt")
    if not os.path.exists(p):
        # Create deterministically on the fly
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 3, size, size, generator=g)
        save_tensor(p, x)
        return x
    return load_tensor(p)


def rubric_for(model_name: str) -> Rubric:
    return get_rubric(model_name)
