import sys
import types
import tempfile
from pathlib import Path

import torch

from leanyolo.utils.weights import WeightsEntry
from leanyolo.utils.remap import extract_state_dict


# Define a top-level stub class so it is picklable
class YOLOv10DetectionModel:
    __module__ = "ultralytics.nn.tasks"

    def __init__(self, model):
        self._parameters = {}
        self._buffers = {}
        self._modules = {"model": model}


def _install_ultralytics_stub_for_dump():
    # Create a minimal ultralytics-like module tree to embed in a checkpoint
    root = types.ModuleType("ultralytics")
    nn = types.ModuleType("ultralytics.nn")
    tasks = types.ModuleType("ultralytics.nn.tasks")
    root.nn = nn
    nn.tasks = tasks
    sys.modules["ultralytics"] = root
    sys.modules["ultralytics.nn"] = nn
    sys.modules["ultralytics.nn.tasks"] = tasks

    tasks.YOLOv10DetectionModel = YOLOv10DetectionModel
    return root


def test_weights_entry_loads_ultralytics_style_checkpoint(tmp_path):
    # 1) Build a tiny nn.Sequential to act as 'model'
    seq = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 1),
        torch.nn.BatchNorm2d(3),
    )

    # 2) Install stub to create a pickled object with ultralytics globals
    stub_root = _install_ultralytics_stub_for_dump()

    try:
        ckpt = {"model": stub_root.nn.tasks.YOLOv10DetectionModel(seq)}
        path = Path(tmp_path) / "fake_official.pt"
        torch.save(ckpt, path)
    finally:
        # Remove stubs to simulate environment without ultralytics at load time
        for k in ["ultralytics.nn.tasks", "ultralytics.nn", "ultralytics"]:
            sys.modules.pop(k, None)

    # 3) Use WeightsEntry to safely load without ultralytics being installed
    entry = WeightsEntry(name="test", url=None, filename=str(path.name))
    obj = entry.get_state_dict(local_path=str(path))
    # 4) Extract and verify a plausible state_dict
    sd = extract_state_dict(obj)
    assert isinstance(sd, dict)
    # Should include conv and bn parameters
    keys = set(sd.keys())
    assert any(k.endswith(".weight") for k in keys)
    assert any("running_mean" in k for k in keys)
