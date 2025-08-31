from __future__ import annotations

from typing import Dict, Tuple

import torch


POSSIBLE_STATE_KEYS = (
    "state_dict",
    "model",
    "ema_state_dict",
    "model_state",
    "net",
)


def extract_state_dict(obj: Dict) -> Dict[str, torch.Tensor]:
    """Extract a flat state_dict from various checkpoint formats.

    Tries common wrapper keys and falls back to the object itself if it looks like
    a state_dict (str->Tensor mapping).
    """
    # If it's a model-like object with state_dict method
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try:
            sd = obj.state_dict()
            if isinstance(sd, dict) and sd:
                return sd
        except Exception:
            pass

    if isinstance(obj, dict):
        # If it already looks like a state_dict
        if all(isinstance(k, str) for k in obj.keys()) and any(
            isinstance(v, (torch.Tensor, dict)) for v in obj.values()
        ):
            pass

        for k in POSSIBLE_STATE_KEYS:
            v = obj.get(k)
            if v is None:
                continue
            # If a nested model-like object
            if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
                try:
                    sd = v.state_dict()
                    if isinstance(sd, dict) and sd:
                        return sd
                except Exception:
                    pass
            # If nested dict
            if isinstance(v, dict) and v:
                inner = v
                for k2 in POSSIBLE_STATE_KEYS:
                    vv = inner.get(k2)
                    if hasattr(vv, "state_dict") and callable(getattr(vv, "state_dict")):
                        try:
                            sd = vv.state_dict()
                            if isinstance(sd, dict) and sd:
                                return sd
                        except Exception:
                            pass
                    if isinstance(vv, dict) and vv:
                        inner = vv
                        break
                return inner
    return obj


def strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common prefixes like 'module.' or 'model.' from keys.
    Leaves the last-appearing variant if multiple prefixes exist.
    """
    prefixes = ("module.", "model.", "model.model.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        kk = k
        changed = True
        # Iteratively strip known prefixes
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p) :]
                    changed = True
        out[kk] = v
    return out


def adapt_state_dict_for_lean(loaded: Dict) -> Dict[str, torch.Tensor]:
    """Lightweight adapter for official YOLOv10 weights.

    This function only unwraps wrapper keys and strips common prefixes.
    It does not attempt deep architectural remapping.
    """
    sd = extract_state_dict(loaded)
    # Only keep tensor-like entries (skip metadata entries if present)
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    sd = strip_common_prefixes(sd)
    return sd


def _flatten_state_by_order(sd: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, torch.Tensor], ...]:
    # Return in stable order
    return tuple((k, sd[k]) for k in sd.keys())


def remap_by_shape(src_sd: Dict[str, torch.Tensor], dst_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Heuristic mapping of weights by matching tensor shapes in order.

    Args:
        src_sd: source state dict (e.g., official model.state_dict()).
        dst_sd: destination model.state_dict() (from our lean model).
    Returns:
        Dict suitable for `load_state_dict` on destination.
    """
    src_items = _flatten_state_by_order(src_sd)
    dst_items = _flatten_state_by_order(dst_sd)

    out: Dict[str, torch.Tensor] = {}
    si = 0
    for dk, dv in dst_items:
        # Advance src until shapes match or exhaust
        while si < len(src_items) and src_items[si][1].shape != dv.shape:
            si += 1
        if si >= len(src_items):
            break
        sk, sv = src_items[si]
        if sv.shape == dv.shape:
            out[dk] = sv
            si += 1
    return out


def remap_official_yolov10_to_lean(loaded_obj: Dict, dst_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Map an official YOLOv10 checkpoint (THU-MIG/Ultralytics format) to our lean model.

    Strategy:
    - Extract nested state dict from the checkpoint (handles object wrappers).
    - Strip common prefixes (module./model.).
    - Heuristically align by matching shapes in iteration order.
    """
    src_sd = adapt_state_dict_for_lean(loaded_obj)
    dst_sd = dst_model.state_dict()
    mapped = remap_by_shape(src_sd, dst_sd)
    return mapped
