import torch

from leanyolo.utils.remap import adapt_state_dict_for_lean


def test_adapt_state_dict_strips_wrappers_and_prefixes():
    base = {
        "backbone.stem.conv.weight": torch.randn(16, 3, 3, 3),
        "backbone.stem.bn.weight": torch.randn(16),
    }
    # Wrap under nested keys and add prefixes
    ckpt = {
        "state_dict": {
            "model": {
                "module.backbone.stem.conv.weight": base["backbone.stem.conv.weight"],
                "module.backbone.stem.bn.weight": base["backbone.stem.bn.weight"],
            }
        }
    }
    out = adapt_state_dict_for_lean(ckpt)
    assert set(out.keys()) == set(base.keys())
    assert torch.allclose(out["backbone.stem.conv.weight"], base["backbone.stem.conv.weight"])  # identity
