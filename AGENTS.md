# leanyolo Development Guide

## Project Overview

Clean, minimal PyTorch implementation focused on YOLOv10. The goal is a faithful, readable YOLOv10 port that can load official pretrained weights across standard sizes using typical PyTorch conventions (no YAML configs).

Status: this repository currently contains documentation and planning. Code scaffolding (modules/CLI) is in progress.

Note: For a fuller overview, usage examples (Python API and CLI), and the latest details, also read `README.md` and `todo.md`.

Key capabilities
- YOLOv10-only: backbone, neck, detection head
- Official weight loading for sizes: `n`, `s`, `m`, `b`, `l`, `x`
- PyTorch-native configuration: construct models and pass args in Python or via CLI flags
- Inference, validation (COCO-style mAP .5:.95), training baseline, and export (TorchScript/ONNX)

## Tech Stack

- Python 3.9+ (3.10/3.11 recommended)
- PyTorch (CPU or CUDA build appropriate for your system)
- torchvision, torchaudio (per PyTorch install channel)
- numpy, pillow, opencv-python, tqdm, matplotlib, pycocotools
- COCO dataset format for data and class definitions

Note: Pretrained weights come from the official THU-MIG/yolov10 releases. This project prioritizes exact compatibility without any runtime dependency on the official implementation. The official repository is used only as a reference for side‑by‑side testing, creating unit tests, and study.

## Project Structure

Planned layout (WIP — not yet present in the repo):
```
leanyolo/
  README.md
  LICENSE
  leanyolo/
    __init__.py
    models/yolov10/{backbone.py, neck.py, head.py, model.py}
    data/{dataset.py, transforms.py, collate.py}
    engine/{train.py, eval.py, infer.py}
    utils/{metrics.py, box_ops.py, losses.py, viz.py}
    tests/{test_backbone.py, test_neck.py, test_head.py, test_model.py, test_output_parity.py}
  train.py
  val.py
  infer.py
  export.py
  requirements.txt
```

## Development Guidelines

### Key Principles

- YOLOv10-first focus: implement backbone, neck, head, and end-to-end forward
- Faithful parity: match official implementation outputs and parameter shapes
- PyTorch-native API: avoid YAML; use Python constructors and simple CLI flags
- Readability over magic: clear modules, explicit shapes, and simple data flow
- Reproducibility: CPU reference outputs and sanity checks for weight parity

### Best Practices

- Keep PRs minimal and focused; include rationale and docs updates when behavior changes
- Add or update unit tests for modules touched (backbone/neck/head/model)
- Verify weight loading: parameter count parity and dry-forward tests
- Use COCO JSON format and standard directory structure for datasets

### End-of-Task QA Checklist (Required)

Run all checks inside this repo’s virtual environment where PyTorch and test deps are already installed. Do not use the system Python.

- Environment: activate `.venv` (or call binaries explicitly). Quick verify:
  - `./.venv/bin/python -c "import sys; print(sys.executable)"` -> path ends with `/.venv/bin/python`
  - `./.venv/bin/python -c "import torch, torchvision; print(torch.__version__, bool(torch.cuda.is_available()))"`
- Unit tests: `./.venv/bin/pytest -q` or `./.venv/bin/pytest -q -m "not fidelity"`; ensure all pass.
- Fidelity tests: `./.venv/bin/pytest -q -m fidelity` with references present. Ensure official weights are available offline:
- Set `LEANYOLO_WEIGHTS_DIR` to a directory containing `yolov10{n,s,m,b,l,x}.pt`, or
- Ensure `LEANYOLO_CACHE_DIR` is writeable and allow a one-time download of official weights.
- Weight loading: verify `get_model(name, weights="PRETRAINED_COCO", class_names=coco80_class_names())` works for all variants without errors; review missing/unexpected key warnings.
- API compatibility: confirm `leanyolo.models.get_model`, detection head outputs, and CLI entrypoints behave as documented; e.g., run `test_eval_synthetic` and other API-focused tests.

Document any intentional behavior changes and update README/examples accordingly.

### Running Tests

Always run tests using the virtualenv binaries to avoid picking up the system Python:

- Preflight environment:
  - `./.venv/bin/python -c "import sys; print(sys.executable)"`  # must point into .venv
  - `./.venv/bin/python -c "import torch; print(torch.__version__)"`

- Full suite: `./.venv/bin/pytest -q`
- Unit-only (exclude fidelity): `./.venv/bin/pytest -q -m "not fidelity"`
- Fidelity-only: `./.venv/bin/pytest -q -m fidelity`

Requirements for fidelity tests:
- Reference tensors present under `leanyolo/tests/data/refs/<model>/` (see `leanyolo/tests/fidelity/README.md`).
- Official weights available offline via `LEANYOLO_WEIGHTS_DIR` or writable cache via `LEANYOLO_CACHE_DIR`.
- The `yolov10-official/` repo is present as a sibling folder; it is imported as source for weight loading utilities.

Tips:
- To force CPU for deterministic parity snapshots: prefix with `CUDA_VISIBLE_DEVICES=""`.

## Environment Setup

### Environment Policy (Required)

- All Python commands and unit tests for this repository must be executed inside the local virtual environment `.venv`.
- When benchmarking or generating parity data from the official YOLOv10 implementation, all Python commands must be executed inside that repo’s dedicated environment `.venv-ref`.
- Do not use the system Python. Either activate the environment or call the interpreter explicitly.

Examples
- Repo dev (this repo): `source .venv/bin/activate` then run `python`, `pip`, `pytest`; or use explicit paths `./.venv/bin/python`, `./.venv/bin/pytest`.
- Official benchmarking (yolov10-official): `source .venv-ref/bin/activate` or `./.venv-ref/bin/python ...`.

Quick checks
```
./.venv/bin/python -c "import sys; print(sys.executable)"   # should resolve to .venv
./.venv/bin/pytest -q                                       # run tests (when available)
```

### Development Requirements

- Python 3.9+ (3.10/3.11 recommended)
- PyTorch (CPU or CUDA build matching your hardware/driver)
- Git

### Installation Steps

Create and activate a virtual environment (Linux/macOS):

```
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch (CUDA build by default):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Notes
- This installs CUDA-enabled wheels (cu12x). They run on CPU as well; GPU is used if available.
- If you need CPU-only wheels, switch to the CPU index.

Install project dependencies:

```
pip install -r requirements.txt
```

Optional quick check in Python (placeholder; modules not yet implemented):

```python
import torch
print(torch.__version__)
print("Torch installed and GPU available?", torch.cuda.is_available())
```

Dataset layout (COCO):

```
data/
  images/
    train2017/ ... .jpg
    val2017/   ... .jpg
  annotations/
    instances_train2017.json
    instances_val2017.json
```

Notes
- Class names are read from COCO JSON files
- YOLO text format is not supported

## Reference Implementation Setup (Official YOLOv10)

Use the official YOLOv10 repo to generate reference outputs for parity checks.

Location and ignore rules
- Clone the reference repo into a sibling folder under this project root: `yolov10-official/`.
- Create its own virtual environment inside that folder: `.venv-ref/`.
- Both `yolov10-official/` and `.venv-ref/` are git-ignored in this repo.

1) Clone the official repository (under this repo, but git-ignored):

```
git clone https://github.com/THU-MIG/yolov10.git yolov10-official
cd yolov10-official
```

2) Create a clean environment inside the reference repo (CPU shown for reproducibility):

```
python -m venv .venv-ref
source .venv-ref/bin/activate
```

3) Install dependencies (match your torch build as needed):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

4) Obtain official weights for the target size (n/s/m/b/l/x) following the repo’s README instructions.

5) Generate reference outputs on CPU for deterministic comparison (examples; consult the official README for exact commands):

```
# Example: run validation or inference to produce outputs
# The exact entrypoints/flags may differ; see yolov10-official/README.md
python val.py --weights yolov10s.pt --img 640 --device cpu --batch 1
python detect.py --weights yolov10s.pt --source path/to/images --img 640 --device cpu
```

6) Save intermediate tensors or end-to-end predictions as needed for parity tests in this repo’s `leanyolo/tests` workflow. Use consistent image sizes and preprocessing.

Tips
- Prefer CPU for parity snapshots to avoid minor CUDA nondeterminism for unit tests.
- Keep torch, torchvision, and numpy versions pinned when regenerating references.
- Track exact commands and seeds for reproducibility.

Note on pretrained weights
- When downloading pretrained weights via this repo’s APIs, sources are the official THU-MIG/yolov10 GitHub releases (e.g., tag v1.1). Do not fetch or reference Ultralytics weight files.
