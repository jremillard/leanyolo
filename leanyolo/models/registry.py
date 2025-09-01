from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Type, Sequence

import torch
import torch.nn as nn

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


_VARIANTS = ("yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x")


@_register_model("yolov10n")
def _build_yolov10n(class_names: Sequence[str], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10n import YOLOv10n
    return YOLOv10n(class_names=class_names, in_channels=in_channels)


@_register_model("yolov10s")
def _build_yolov10s(class_names: Sequence[str], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10s import YOLOv10s
    return YOLOv10s(class_names=class_names, in_channels=in_channels)


@_register_model("yolov10m")
def _build_yolov10m(class_names: Sequence[str], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10m import YOLOv10m
    return YOLOv10m(class_names=class_names, in_channels=in_channels)


@_register_model("yolov10b")
def _build_yolov10b(class_names: Sequence[str], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10b import YOLOv10b
    return YOLOv10b(class_names=class_names, in_channels=in_channels)


@_register_model("yolov10l")
def _build_yolov10l(class_names: Sequence[str], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10l import YOLOv10l
    return YOLOv10l(class_names=class_names, in_channels=in_channels)


@_register_model("yolov10x")
def _build_yolov10x(class_names: Sequence[str], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10x import YOLOv10x
    return YOLOv10x(class_names=class_names, in_channels=in_channels)


# Weights registry per model name
class _YOLOv10Weights(WeightsResolver):
    """Weight resolver for YOLOv10 families.

    This uses placeholders for URLs to avoid implicit downloads. Provide a
    local path or set the LEANYOLO_WEIGHTS_DIR env var to enable loading.
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
    weights: Optional[str],
    class_names: Sequence[str],
) -> nn.Module:
    """Create a model by name, optionally loading weights.

    Input format:
    - Tensor layout: CHW, shape (N, C, H, W)
    - Color order: RGB (not BGR). Tip: If loading images with OpenCV (BGR), convert to RGB first

    Args:
        name: Model name (e.g., 'yolov10s').
        weights: Weight key (e.g., 'DEFAULT') or None to skip loading. Must be provided explicitly.
        class_names: Sequence of class names for the dataset; length defines number of classes.

    Returns:
        torch.nn.Module: Instantiated model.
    """
    if name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    # Models expect 3-channel RGB input; hard-code in_channels=3
    model = _MODEL_BUILDERS[name](class_names=class_names, in_channels=3)
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
    if name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return _YOLOv10Weights
