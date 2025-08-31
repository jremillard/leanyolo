from __future__ import annotations

from typing import Dict

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
    if isinstance(obj, dict):
        # If it already looks like a state_dict
        if all(isinstance(k, str) for k in obj.keys()) and any(
            isinstance(v, (torch.Tensor, dict)) for v in obj.values()
        ):
            # If nested dicts under top-level, we still proceed; remapper will handle
            pass

        for k in POSSIBLE_STATE_KEYS:
            v = obj.get(k)
            if isinstance(v, dict) and v:
                # v might itself have one of the keys (nested)
                inner = v
                for k2 in POSSIBLE_STATE_KEYS:
                    vv = inner.get(k2)
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

