from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Type, Sequence
import os

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
def _build_yolov10n(class_names: Sequence[str], input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10n import YOLOv10n
    return YOLOv10n(class_names=class_names, in_channels=in_channels, input_norm_subtract=input_norm_subtract, input_norm_divide=input_norm_divide)


@_register_model("yolov10s")
def _build_yolov10s(class_names: Sequence[str], input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10s import YOLOv10s
    return YOLOv10s(class_names=class_names, in_channels=in_channels, input_norm_subtract=input_norm_subtract, input_norm_divide=input_norm_divide)


@_register_model("yolov10m")
def _build_yolov10m(class_names: Sequence[str], input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10m import YOLOv10m
    return YOLOv10m(class_names=class_names, in_channels=in_channels, input_norm_subtract=input_norm_subtract, input_norm_divide=input_norm_divide)


@_register_model("yolov10b")
def _build_yolov10b(class_names: Sequence[str], input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10b import YOLOv10b
    return YOLOv10b(class_names=class_names, in_channels=in_channels, input_norm_subtract=input_norm_subtract, input_norm_divide=input_norm_divide)


@_register_model("yolov10l")
def _build_yolov10l(class_names: Sequence[str], input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10l import YOLOv10l
    return YOLOv10l(class_names=class_names, in_channels=in_channels, input_norm_subtract=input_norm_subtract, input_norm_divide=input_norm_divide)


@_register_model("yolov10x")
def _build_yolov10x(class_names: Sequence[str], input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float], in_channels: int = 3) -> nn.Module:
    from .yolov10.yolov10x import YOLOv10x
    return YOLOv10x(class_names=class_names, in_channels=in_channels, input_norm_subtract=input_norm_subtract, input_norm_divide=input_norm_divide)


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
            "PRETRAINED_COCO": WeightsEntry(
                name="yolov10n.PRETRAINED_COCO",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
                filename="yolov10n.pt",
                sha256="61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10s": {
            "PRETRAINED_COCO": WeightsEntry(
                name="yolov10s.PRETRAINED_COCO",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
                filename="yolov10s.pt",
                sha256="96af3fc7c7169abcc4867f3e3088b761bb33cf801283c2ec05f9703d63a0ba77",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10m": {
            "PRETRAINED_COCO": WeightsEntry(
                name="yolov10m.PRETRAINED_COCO",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
                filename="yolov10m.pt",
                sha256="ff2c559f11d13701abc4e0345f82851d146ecfe7035efaafcc08475cfd8b5f2d",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10b": {
            "PRETRAINED_COCO": WeightsEntry(
                name="yolov10b.PRETRAINED_COCO",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
                filename="yolov10b.pt",
                sha256="3846434cbf0016b663a1ccd6d843c48468f6852f4feeddcb9f67f9182168c142",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10l": {
            "PRETRAINED_COCO": WeightsEntry(
                name="yolov10l.PRETRAINED_COCO",
                url="https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
                filename="yolov10l.pt",
                sha256="83769ec3cbc61f18113f612f8bdcf922396628d620682bb72966e9b148004b8b",
                metadata={"task": "detection", "dataset": "coco", "source": "THU-MIG/yolov10@v1.1"},
            )
        },
        "yolov10x": {
            "PRETRAINED_COCO": WeightsEntry(
                name="yolov10x.PRETRAINED_COCO",
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
    input_norm_subtract: Optional[Sequence[float]] = None,
    input_norm_divide: Optional[Sequence[float]] = None,
) -> nn.Module:
    """Create a model by name, optionally loading weights.

    Input format:
    - Tensor layout: CHW, shape (N, C, H, W)
    - Color order: RGB (not BGR). Tip: If loading images with OpenCV (BGR), convert to RGB first

    Pretrained weight resolution, download, and caching:
    - Pass ``weights='PRETRAINED_COCO'`` to load the official YOLOv10 weights for the
      given variant. Resolution order is handled by ``WeightsEntry.get_state_dict``:
        1) If a ``local_path`` is provided (not used here), load directly from it.
        2) If ``LEANYOLO_WEIGHTS_DIR`` is set, try ``$LEANYOLO_WEIGHTS_DIR/<filename>``.
        3) Otherwise, use a cache directory and download when missing.
    - Cache location: by default ``~/.cache/leanyolo`` or the directory specified by
      ``LEANYOLO_CACHE_DIR``. The expected filename is derived from the weight entry
      (e.g., ``yolov10s.pt``).
    - Integrity verification: when a file is found in the cache or downloaded, its
      SHA-256 is computed and compared against the known digest from the registry.
      If the hash does not match, the file is removed and loading fails with a clear
      error to prevent using a corrupted or tampered checkpoint.
    - You can also provide a direct filesystem path to a checkpoint via the ``weights``
      argument (instead of ``'PRETRAINED_COCO'``); in that case no download/caching is
      performed and the file is loaded directly after basic sanity checks.

    Args:
        name: Model name (e.g., 'yolov10s').
        weights: Weight key (must be 'PRETRAINED_COCO') or None to skip loading.
        class_names: Sequence of class names for the dataset; length defines number of classes.
        input_norm_subtract: Optional per-channel mean to subtract. If None, uses [0,0,0]. Length must be 3 or 1.
        input_norm_divide: Optional per-channel divisor. If None, uses [255,255,255]. Length must be 3 or 1.

    Returns:
        torch.nn.Module: Instantiated model.
    """
    if name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    # Provide defaults for normalization vectors (divide by 255; no subtract)
    if input_norm_subtract is None:
        input_norm_subtract = (0.0, 0.0, 0.0)
    if input_norm_divide is None:
        input_norm_divide = (255.0, 255.0, 255.0)
    def _to3(x: Sequence[float]) -> Sequence[float]:
        if len(x) == 1:
            return [float(x[0])] * 3
        if len(x) != 3:
            raise ValueError("subtract_mean/divide must have length 1 or 3")
        return [float(v) for v in x]
    sub3 = _to3(input_norm_subtract)
    div3 = _to3(input_norm_divide)
    # Models expect 3-channel RGB input; hard-code in_channels=3
    model = _MODEL_BUILDERS[name](class_names=class_names, input_norm_subtract=sub3, input_norm_divide=div3, in_channels=3)
    if weights is not None:
        # Local checkpoint path support
        if isinstance(weights, str) and os.path.isfile(weights):
            try:
                ckpt = torch.load(weights, map_location="cpu", weights_only=False)
                if not isinstance(ckpt, dict):
                    raise ValueError("checkpoint must be a dict")
                mname = ckpt.get("model_name")
                cls = ckpt.get("class_names")
                sd = ckpt.get("state_dict")
                if mname != name:
                    raise ValueError(f"Checkpoint model_name '{mname}' does not match requested '{name}'")
                if not isinstance(cls, (list, tuple)) or list(cls) != list(class_names):
                    raise ValueError("Checkpoint class_names do not match provided class_names")
                if not isinstance(sd, dict):
                    raise ValueError("Checkpoint missing state_dict")
                model.load_state_dict(sd, strict=True)
                return model
            except Exception as e:
                raise ValueError(f"Failed to load local checkpoint '{weights}': {e}")
        if weights != "PRETRAINED_COCO":
            raise ValueError("weights must be a filename, 'PRETRAINED_COCO', or None")
        try:
            entry = _YOLOv10Weights().get(name, "PRETRAINED_COCO")
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
        weights = weights_enum().get('yolov10s', 'PRETRAINED_COCO')
        model.load_state_dict(weights.get_state_dict())
    """
    if name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return _YOLOv10Weights
