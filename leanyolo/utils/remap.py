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


def _module_like_to_state_dict(mod, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Extract state_dict from an nn.Module-like object without calling methods.

    Traverses attributes `_parameters`, `_buffers`, and `_modules` recursively
    to build a flat dict of tensors keyed by hierarchical names. This works for
    safely-unpickled objects that mimic nn.Module structure but don't implement
    a functional state_dict method (e.g., when loaded with weights_only=True and
    stubbed classes).
    """
    out: Dict[str, torch.Tensor] = {}
    try:
        params = getattr(mod, "_parameters", None)
        if isinstance(params, dict):
            for k, v in params.items():
                if isinstance(v, torch.Tensor):
                    out[prefix + k] = v
        buffers = getattr(mod, "_buffers", None)
        if isinstance(buffers, dict):
            for k, v in buffers.items():
                if isinstance(v, torch.Tensor):
                    out[prefix + k] = v
        children = getattr(mod, "_modules", None)
        if isinstance(children, dict):
            for name, child in children.items():
                child_prefix = prefix + (name + "." if prefix or name else "")
                out.update(_module_like_to_state_dict(child, child_prefix))
    except Exception:
        pass
    return out


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
    # Fallback: handle module-like objects without calling methods
    ml = _module_like_to_state_dict(obj)
    if ml:
        return ml

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
            ml = _module_like_to_state_dict(v)
            if ml:
                return ml
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
                    ml2 = _module_like_to_state_dict(vv)
                    if ml2:
                        return ml2
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
    """Lightweight adapter for official weights.

    Unwraps wrapper keys and strips common prefixes. Does not attempt
    architecture-specific remapping.
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


## Note: YOLOv10-specific remapping moved to leanyolo.models.yolov10.remap
