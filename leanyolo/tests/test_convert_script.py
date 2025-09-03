import importlib.util
import os
from pathlib import Path

import torch

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names


def _load_convert_module():
    script_path = Path(__file__).resolve().parents[2] / "tools" / "convert_official_weights.py"
    spec = importlib.util.spec_from_file_location("convert_official_weights", str(script_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def test_convert_script_roundtrip(tmp_path, monkeypatch):
    # Prepare a fake official weight file via env dir to avoid network
    env_dir = tmp_path / "env"
    out_dir = tmp_path / "out"
    env_dir.mkdir()
    out_dir.mkdir()

    class_names = coco80_class_names()

    # Seed model to generate a source state_dict that the registry will resolve
    m_seed = get_model(
        name="yolov10n",
        weights=None,
        class_names=class_names,
    )
    seed_sd = m_seed.state_dict()
    # Place as the expected filename in the env dir so PRETRAINED_COCO resolves locally
    env_path = env_dir / "yolov10n.pt"
    torch.save(seed_sd, env_path)

    monkeypatch.setenv("LEANYOLO_WEIGHTS_DIR", str(env_dir))

    # Import the conversion script and run convert
    conv = _load_convert_module()
    out_path = out_dir / "yolov10n.state_dict.pt"
    saved = conv.convert("yolov10n", str(out_path))
    assert Path(saved).exists()

    # Saved file should be a plain state_dict
    loaded = torch.load(saved, map_location="cpu")
    assert isinstance(loaded, dict)
    assert all(isinstance(v, torch.Tensor) for v in loaded.values())

    # Build models via (a) PRETRAINED_COCO and (b) the converted state_dict path
    m_a = get_model(name="yolov10n", weights="PRETRAINED_COCO", class_names=class_names)
    m_b = get_model(name="yolov10n", weights=saved, class_names=class_names)
    m_a.eval(); m_b.eval()

    x = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        out_a = m_a(x)
        out_b = m_b(x)

    assert isinstance(out_a, list) and isinstance(out_b, list)
    assert len(out_a) == len(out_b) == 3
    for a, b in zip(out_a, out_b):
        assert a.shape == b.shape
        assert torch.allclose(a, b, rtol=0.0, atol=0.0)
