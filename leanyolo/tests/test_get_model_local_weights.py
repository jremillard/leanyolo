import os
import tempfile

import torch

from leanyolo.models.registry import get_model


def _tmpfile(suffix: str = ".pt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def test_loads_plain_state_dict_ok():
    # Build a small model and save a plain state_dict
    class_names = ["a", "b"]
    m = get_model(
        name="yolov10n",
        weights=None,
        class_names=class_names,
    )
    sd = m.state_dict()
    path = _tmpfile()
    try:
        torch.save(sd, path)

        # Now request the model again, loading from the saved plain state_dict
        m2 = get_model(
            name="yolov10n",
            weights=path,
            class_names=class_names,
        )
        # Basic check: params match
        for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
            assert k1 == k2
            assert torch.allclose(v1, v2)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def test_raises_on_incompatible_state_dict():
    # Create a mismatched checkpoint by saving a model with a different class count
    m_wrong = get_model(
        name="yolov10n",
        weights=None,
        class_names=["a", "b", "c"],  # 3 classes vs 2 below
    )
    path = _tmpfile()
    try:
        torch.save(m_wrong.state_dict(), path)

        # Attempt to load into a 2-class model should raise
        try:
            _ = get_model(
                name="yolov10n",
                weights=path,
                class_names=["a", "b"],
            )
            assert False, "Expected ValueError for incompatible local weights"
        except ValueError as e:
            # Ensure the message indicates incompatibility without conversion
            assert "compatible with this library version" in str(e)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

