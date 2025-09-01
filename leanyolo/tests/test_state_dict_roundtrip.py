import os
import tempfile

import pytest
import torch

from leanyolo.models.registry import get_model


@pytest.mark.parametrize(
    "model_name",
    ("yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"),
)
def test_state_dict_save_load_forward_match(model_name: str):
    # Use a tiny spatial size to keep the test fast for all variants
    b, c, h, w = 1, 3, 128, 128
    x = torch.rand(b, c, h, w)

    class_names = ["cat", "dog"]  # small nc keeps head lightweight

    # Build model with random weights and neutral normalization (no scale/shift)
    m1 = get_model(
        name=model_name,
        weights=None,
        class_names=class_names,
        input_norm_subtract=(0.0,),
        input_norm_divide=(1.0,),
    )
    m1.eval()

    with torch.inference_mode():
        out1 = m1(x)

    # Save a plain state_dict
    fd, path = tempfile.mkstemp(suffix=f"_{model_name}.pt")
    os.close(fd)
    try:
        torch.save(m1.state_dict(), path)

        # Load into a fresh model strictly via local path
        m2 = get_model(
            name=model_name,
            weights=path,
            class_names=class_names,
            input_norm_subtract=(0.0,),
            input_norm_divide=(1.0,),
        )
        m2.eval()

        with torch.inference_mode():
            out2 = m2(x)

        # Raw training-style outputs must match exactly
        assert isinstance(out1, list) and isinstance(out2, list)
        assert len(out1) == len(out2) == 3
        for a, b in zip(out1, out2):
            assert a.shape == b.shape
            assert torch.allclose(a, b, rtol=0.0, atol=0.0)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

