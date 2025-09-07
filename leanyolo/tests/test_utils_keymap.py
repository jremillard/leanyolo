import pytest


def _run_keymap_test(remap_func):
    # Destination keys mimic a subset of a model.state_dict() keys
    dst_keys = {
        # direct map for backbone.c4
        "backbone.c4.cv.weight": 0,
        # alt_map case for RepVGGDW fused naming inside cv1.2.*
        "backbone.cv1.cv1.2.conv.conv.weight": 0,
    }

    # Source checkpoint style keys
    src = {
        "model.4.cv.weight": "B",  # direct mapping via BACKBONE_MAP[4] -> backbone.c4
        "model.1.cv1.2.conv.weight": "A",  # requires alt_map suffix rewrite
        "model.999.ignored.weight": "X",  # no mapping for this idx
        "some.other": "Y",  # not a model.* key
    }

    out = remap_func(src, dst_keys)

    # Direct mapping present
    assert out.get("backbone.c4.cv.weight") == "B"
    # Alt-map mapping hit
    assert out.get("backbone.cv1.cv1.2.conv.conv.weight") == "A"
    # Unmappable are dropped
    assert "model.999.ignored.weight" not in out
    assert "some.other" not in out


def test_utils_keymap_remap():
    from leanyolo.utils.keymap import remap_official_keys_by_name

    _run_keymap_test(remap_official_keys_by_name)


def test_model_keymap_remap():
    from leanyolo.models.yolov10.keymap import remap_official_keys_by_name

    _run_keymap_test(remap_official_keys_by_name)

