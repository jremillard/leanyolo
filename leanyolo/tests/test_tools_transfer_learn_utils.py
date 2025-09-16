from __future__ import annotations

import torch


def test_resolve_device_cpu(monkeypatch):
    from tools import transfer_learn_aquarium as tool

    device, label, warn = tool.resolve_device("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"
    assert label == "cpu"
    assert warn is None


def test_resolve_device_cuda_fallback_when_unavailable(monkeypatch):
    from tools import transfer_learn_aquarium as tool

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, label, warn = tool.resolve_device("cuda")
    assert device.type == "cpu"
    assert label == "cpu"
    assert warn is not None and "CUDA" in warn


def test_resolve_device_invalid_string(monkeypatch):
    from tools import transfer_learn_aquarium as tool

    device, label, warn = tool.resolve_device("not-a-device")
    assert device.type == "cpu"
    assert label == "cpu"
    assert warn is not None and "Invalid" in warn
