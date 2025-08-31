from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Type

import torch
import torch.nn as nn

from .yolov10.model import YOLOv10
import warnings
from ..utils.weights import WeightsEntry, WeightsResolver
from ..utils.remap import (
    adapt_state_dict_for_lean,
    remap_official_yolov10_to_lean,
    extract_state_dict,
)


# Model registry (name -> builder)
_MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {}


def _register_model(name: str):
    def deco(fn: Callable[..., nn.Module]):
        _MODEL_BUILDERS[name] = fn
        return fn

    return deco


# Configuration for YOLOv10 sizes (depth/width multipliers)
@dataclass(frozen=True)
class ModelSpec:
    depth_mult: float
    width_mult: float
    max_channels: int = 1024
    variant: str = "s"  # one of n,s,m,b,l,x


@dataclass(frozen=True)
class Y10Config:
    # Backbone channels by stage index
    CH: Dict[int, int]
    # Neck/head channels by stage index
    HCH: Dict[int, int]
    # Repeats per stage index (2,4,6,8,13,16,19,22)
    reps: Dict[int, int]
    # Module types per key: 'C2f' or 'C2fCIB'
    types: Dict[str, str]
    # Depthwise repvgg-like kernel fusion flags (per key)
    lk: Dict[str, bool]


_YOLOV10_SPECS: Dict[str, ModelSpec] = {
    # Official scales extracted from THU-MIG/yolov10 YAMLs
    # [depth_mult, width_mult, max_channels]
    "yolov10n": ModelSpec(depth_mult=0.33, width_mult=0.25, max_channels=1024, variant="n"),
    "yolov10s": ModelSpec(depth_mult=0.33, width_mult=0.50, max_channels=1024, variant="s"),
    "yolov10m": ModelSpec(depth_mult=0.67, width_mult=0.75, max_channels=768, variant="m"),
    "yolov10b": ModelSpec(depth_mult=0.67, width_mult=1.00, max_channels=512, variant="b"),
    "yolov10l": ModelSpec(depth_mult=1.00, width_mult=1.00, max_channels=512, variant="l"),
    "yolov10x": ModelSpec(depth_mult=1.00, width_mult=1.25, max_channels=512, variant="x"),
}


def _build_y10_config(variant: str) -> Y10Config:
    # Exact per-variant channels from official YAMLs
    CH = {
        "n": {0: 16, 1: 32, 2: 32, 3: 64, 4: 64, 5: 128, 6: 128, 7: 256, 8: 256, 9: 256, 10: 256},
        "s": {0: 32, 1: 64, 2: 64, 3: 128, 4: 128, 5: 256, 6: 256, 7: 512, 8: 512, 9: 512, 10: 512},
        "m": {0: 48, 1: 96, 2: 96, 3: 192, 4: 192, 5: 384, 6: 384, 7: 576, 8: 576, 9: 576, 10: 576},
        "b": {0: 64, 1: 128, 2: 128, 3: 256, 4: 256, 5: 512, 6: 512, 7: 512, 8: 512, 9: 512, 10: 512},
        "l": {0: 64, 1: 128, 2: 128, 3: 256, 4: 256, 5: 512, 6: 512, 7: 512, 8: 512, 9: 512, 10: 512},
        "x": {0: 80, 1: 160, 2: 160, 3: 320, 4: 320, 5: 640, 6: 640, 7: 640, 8: 640, 9: 640, 10: 640},
    }[variant]
    HCH = {
        "n": {13: 128, 16: 64, 19: 128, 22: 256},
        "s": {13: 256, 16: 128, 19: 256, 22: 512},
        "m": {13: 384, 16: 192, 19: 384, 22: 576},
        "b": {13: 512, 16: 256, 19: 512, 22: 512},
        "l": {13: 512, 16: 256, 19: 512, 22: 512},
        "x": {13: 640, 16: 320, 19: 640, 22: 640},
    }[variant]
    reps = {
        2: {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant],
        4: {"n": 2, "s": 2, "m": 4, "b": 4, "l": 6, "x": 6}[variant],
        6: {"n": 2, "s": 2, "m": 4, "b": 4, "l": 6, "x": 6}[variant],
        8: {"n": 1, "s": 1, "m": 2, "b": 2, "l": 1, "x": 3}[variant],
        13: {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant],
        16: {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant],
        19: {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant],
        22: {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant],
    }
    types = {
        # Backbone
        "c6": "C2fCIB" if variant == "x" else "C2f",
        "c8": "C2f" if variant == "n" else "C2fCIB",
        # Neck
        "p5_p4": "C2fCIB" if variant in {"b", "l", "x"} else "C2f",
        "p3_p4": "C2fCIB" if variant in {"m", "b", "l", "x"} else "C2f",
        "p4_p5": "C2fCIB",
    }
    lk = {
        # Use RepVGGDW-like dual-branch in these blocks
        "c8": variant in {"s", "x"},
        "p5_p4": variant in {"x"},
        "p4_p5": variant in {"n", "s", "x"},
    }
    return Y10Config(CH=CH, HCH=HCH, reps=reps, types=types, lk=lk)


@_register_model("yolov10n")
def _build_yolov10n(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10n"]
    cfg = _build_y10_config(spec.variant)
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec, cfg=cfg)


@_register_model("yolov10s")
def _build_yolov10s(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10s"]
    cfg = _build_y10_config(spec.variant)
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec, cfg=cfg)


@_register_model("yolov10m")
def _build_yolov10m(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10m"]
    cfg = _build_y10_config(spec.variant)
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec, cfg=cfg)


@_register_model("yolov10b")
def _build_yolov10b(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10b"]
    cfg = _build_y10_config(spec.variant)
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec, cfg=cfg)


@_register_model("yolov10l")
def _build_yolov10l(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10l"]
    cfg = _build_y10_config(spec.variant)
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec, cfg=cfg)


@_register_model("yolov10x")
def _build_yolov10x(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10x"]
    cfg = _build_y10_config(spec.variant)
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec, cfg=cfg)


# Weights registry per model name
class _YOLOv10Weights(WeightsResolver):
    """Weight resolver for YOLOv10 families.

    This uses placeholders for URLs to avoid implicit downloads. Provide a
    local path or set the LEAN_YOLO_WEIGHTS_DIR env var to enable loading.
    """

    # Official YOLOv10 weights from THU-MIG GitHub releases (v1.1)
    # Do NOT use Ultralytics weights here.
    MODEL_TO_WEIGHTS: Dict[str, Dict[str, WeightsEntry]] = {
        "yolov10n": {
            "DEFAULT": WeightsEntry(
                name="yolov10n.DEFAULT",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
                filename="yolov10n.pt",
                sha256="61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10s": {
            "DEFAULT": WeightsEntry(
                name="yolov10s.DEFAULT",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
                filename="yolov10s.pt",
                sha256="96af3fc7c7169abcc4867f3e3088b761bb33cf801283c2ec05f9703d63a0ba77",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10m": {
            "DEFAULT": WeightsEntry(
                name="yolov10m.DEFAULT",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
                filename="yolov10m.pt",
                sha256="ff2c559f11d13701abc4e0345f82851d146ecfe7035efaafcc08475cfd8b5f2d",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10b": {
            "DEFAULT": WeightsEntry(
                name="yolov10b.DEFAULT",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
                filename="yolov10b.pt",
                sha256="3846434cbf0016b663a1ccd6d843c48468f6852f4feeddcb9f67f9182168c142",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10l": {
            "DEFAULT": WeightsEntry(
                name="yolov10l.DEFAULT",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
                filename="yolov10l.pt",
                sha256="83769ec3cbc61f18113f612f8bdcf922396628d620682bb72966e9b148004b8b",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10x": {
            "DEFAULT": WeightsEntry(
                name="yolov10x.DEFAULT",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt",
                filename="yolov10x.pt",
                sha256="6e6eae65e6c268c49a25849922e0c75a5c707d626d67170d16a97813b0f8eb79",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
    }

    def list(self, model_name: str) -> Iterable[str]:
        return self.MODEL_TO_WEIGHTS.get(model_name, {}).keys()

    def get(self, model_name: str, key: str) -> WeightsEntry:
        mapping = self.MODEL_TO_WEIGHTS.get(model_name)
        if not mapping or key not in mapping:
            raise KeyError(f"No weights '{key}' for model '{model_name}'.")
        return mapping[key]


def list_models() -> Iterable[str]:
    return tuple(_MODEL_BUILDERS.keys())


def get_model(
    name: str,
    *,
    weights: Optional[str] = None,
    num_classes: int = 80,
    in_channels: int = 3,
    exact: bool = True,
) -> nn.Module:
    """Create a model by name, optionally loading weights.

    Args:
        name: Model name (e.g., 'yolov10s').
        weights: Optional weight key (e.g., 'DEFAULT'). If provided, tries to
                 load via the weight resolver. See README for offline options.
        num_classes: Number of classes (default 80 for COCO).
        in_channels: Input channels (default 3).

    Returns:
        torch.nn.Module: Instantiated model.
    """
    if name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    model = _MODEL_BUILDERS[name](num_classes=num_classes, in_channels=in_channels)
    if weights:
        try:
            entry = _YOLOv10Weights().get(name, weights)
            loaded_obj = entry.get_state_dict(progress=True)
            # Attempt high-fidelity mapping from official to lean architecture
            mapped = remap_official_yolov10_to_lean(loaded_obj, model)
            # Fallback to simple unwrap if mapping yielded nothing
            state_dict = mapped if mapped else adapt_state_dict_for_lean(loaded_obj)
            # Load and collect stats
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            try:
                src_sd = extract_state_dict(loaded_obj)
                src_total = sum(1 for v in src_sd.values() if isinstance(v, torch.Tensor)) or 1
                used_src = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
                dst_total = sum(1 for _ in model.state_dict().keys()) or 1
                dst_loaded = dst_total - len(missing)
                pct_src = 100.0 * used_src / src_total
                pct_dst = 100.0 * dst_loaded / dst_total
                warnings.warn(
                    f"Weights loaded: {used_src}/{src_total} from file ({pct_src:.1f}%), "
                    f"filled model: {dst_loaded}/{dst_total} params ({pct_dst:.1f}%).",
                    RuntimeWarning,
                )
                # Note: No runtime dependency or fallback to the official implementation.
                # The lean implementation must achieve exact compatibility on its own.
            except Exception:
                pass
            if unexpected:
                warnings.warn(
                    f"Unexpected keys when loading weights: {sorted(unexpected)[:10]}...",
                    RuntimeWarning,
                )
            if missing:
                warnings.warn(
                    f"Missing keys when loading weights: {sorted(missing)[:10]}...",
                    RuntimeWarning,
                )
        except Exception as e:  # pragma: no cover - environment dependent
            warnings.warn(
                f"Could not load weights '{weights}' for '{name}': {e}. "
                "Proceeding with randomly initialized weights.",
                RuntimeWarning,
            )
    return model


def get_model_weights(name: str) -> Type[_YOLOv10Weights]:
    """Return the weights resolver type for a model name.

    Usage:
        weights_enum = get_model_weights('yolov10s')
        weights = weights_enum().get('yolov10s', 'DEFAULT')
        model.load_state_dict(weights.get_state_dict())
    """
    if name not in _YOLOV10_SPECS:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return _YOLOv10Weights
