from __future__ import annotations

from typing import Dict


BACKBONE_MAP = {
    0: "backbone.cv0",
    1: "backbone.cv1",
    2: "backbone.c2",
    3: "backbone.cv3",
    4: "backbone.c4",
    5: "backbone.sc5",
    6: "backbone.c6",
    7: "backbone.sc7",
    8: "backbone.c8",
    9: "backbone.sppf9",
    10: "backbone.psa10",
}

NECK_MAP = {
    13: "neck.p5_p4_c2f",
    16: "neck.p4_p3_c2f",
    17: "neck.p3_down",
    19: "neck.p3_p4_c2f",
    20: "neck.p4_down",
    22: "neck.p4_p5_c2f",
}

HEAD_MAP = {
    23: "head",  # v10Detect head module
}


def remap_official_keys_by_name(src_sd: Dict[str, object], dst_state_keys: Dict[str, object]) -> Dict[str, object]:
    """Map official YOLOv10 checkpoint keys to our lean model keys by layer index mapping.

    This function translates prefixes like 'model.4.' to 'backbone.c4.' etc.
    Only keys present in dst_state_keys are returned.
    """
    out: Dict[str, object] = {}

    def try_add(idx: int, prefix: str, key: str, val: object) -> None:
        new_key = key.replace(f"model.{idx}.", prefix + ".", 1)
        if new_key in dst_state_keys:
            out[new_key] = val

    for k, v in src_sd.items():
        if not k.startswith("model."):
            continue
        try:
            idx_str, rest = k.split(".", 2)[1:3]  # 'model', idx, rest
            idx = int(idx_str)
        except Exception:
            continue

        if idx in BACKBONE_MAP:
            try_add(idx, BACKBONE_MAP[idx], k, v)
        elif idx in NECK_MAP:
            try_add(idx, NECK_MAP[idx], k, v)
        elif idx in HEAD_MAP:
            try_add(idx, HEAD_MAP[idx], k, v)
        else:
            continue

    return out
