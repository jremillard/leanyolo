from __future__ import annotations

from typing import Dict

import torch

from .keymap import remap_official_keys_by_name
from leanyolo.utils.remap import extract_state_dict, strip_common_prefixes, remap_by_shape


def remap_official_yolov10_to_lean(loaded_obj: Dict, dst_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Map an official YOLOv10 checkpoint (THU-MIG/Ultralytics format) to our lean model.

    Strategy:
    - Extract nested state dict from the checkpoint (handles object wrappers).
    - Strip common prefixes (module./model.).
    - Deterministically map by layer indices, then fill remaining by shape.
    """
    raw_src = extract_state_dict(loaded_obj)
    dst_sd = dst_model.state_dict()

    # Deterministic mapping by layer indices/prefix names
    nm_raw = remap_official_keys_by_name(raw_src, dst_sd)
    name_mapped = {k: v for k, v in nm_raw.items() if isinstance(v, torch.Tensor) and v.shape == dst_sd[k].shape}

    # Fallback: strip prefixes then fill remaining by shape
    stripped = strip_common_prefixes({k: v for k, v in raw_src.items() if isinstance(v, torch.Tensor)})
    remaining_dst = {k: v for k, v in dst_sd.items() if k not in name_mapped}
    shape_fill = remap_by_shape(stripped, remaining_dst)

    out = dict(name_mapped)
    out.update(shape_fill)

    # Fill fused RepVGGDW conv1 branches when official checkpoint is fused (no conv1 keys)
    for dk in dst_sd.keys():
        if ".cv1.2.conv1.conv.weight" in dk and dk not in out:
            base = dk.replace("conv1.conv.weight", "conv.conv.weight")
            if base in out:
                wshape = dst_sd[dk].shape
                out[dk] = torch.zeros(wshape)
                bn_w = dk.replace("conv.weight", "bn.weight")
                bn_b = dk.replace("conv.weight", "bn.bias")
                bn_rm = dk.replace("conv.weight", "bn.running_mean")
                bn_rv = dk.replace("conv.weight", "bn.running_var")
                if bn_w in dst_sd and bn_w not in out:
                    out[bn_w] = torch.ones_like(dst_sd[bn_w])
                    out[bn_b] = torch.zeros_like(dst_sd[bn_b])
                    out[bn_rm] = torch.zeros_like(dst_sd[bn_rm])
                    out[bn_rv] = torch.ones_like(dst_sd[bn_rv])

    return out

