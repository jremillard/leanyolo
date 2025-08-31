from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Type

import torch
import torch.nn as nn

from .yolov10.model import YOLOv10
import warnings
from ..utils.weights import WeightsEntry, WeightsResolver
from ..utils.remap import adapt_state_dict_for_lean


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


_YOLOV10_SPECS: Dict[str, ModelSpec] = {
    # Conservative, readable defaults; not guaranteed to match official exactly
    "yolov10n": ModelSpec(depth_mult=0.33, width_mult=0.25, max_channels=512),
    "yolov10s": ModelSpec(depth_mult=0.33, width_mult=0.50, max_channels=768),
    "yolov10m": ModelSpec(depth_mult=0.67, width_mult=0.75, max_channels=1024),
    "yolov10b": ModelSpec(depth_mult=0.80, width_mult=1.00, max_channels=1024),
    "yolov10l": ModelSpec(depth_mult=1.00, width_mult=1.00, max_channels=1024),
    "yolov10x": ModelSpec(depth_mult=1.25, width_mult=1.25, max_channels=1280),
}


@_register_model("yolov10n")
def _build_yolov10n(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10n"]
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec)


@_register_model("yolov10s")
def _build_yolov10s(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10s"]
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec)


@_register_model("yolov10m")
def _build_yolov10m(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10m"]
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec)


@_register_model("yolov10b")
def _build_yolov10b(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10b"]
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec)


@_register_model("yolov10l")
def _build_yolov10l(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10l"]
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec)


@_register_model("yolov10x")
def _build_yolov10x(num_classes: int = 80, in_channels: int = 3) -> nn.Module:
    spec = _YOLOV10_SPECS["yolov10x"]
    return YOLOv10(num_classes=num_classes, in_channels=in_channels, spec=spec)


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
            state_dict = entry.get_state_dict(progress=True)
            # Lightweight remap for official checkpoints (unwrap + strip prefixes)
            state_dict = adapt_state_dict_for_lean(state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
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
    return _YOLOV10Weights
