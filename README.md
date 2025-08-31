# lean-yolo

 Faithful PyTorch implementation of YOLOv10 with exact compatibility for official pretrained weights across all sizes (n, s, m, b, l, x). This project has no runtime dependency on the official repository; the official repo is used only for side‑by‑side testing, study, and unit test generation.

## Status
- Core code implemented: model registry, architecture, exact weight loader, inference/validation CLIs, and tests.

## Scope
- YOLOv10-only: backbone, neck, detection head.
- Object detection only: this project focuses solely on detection. Segmentation, pose, and OBB are out of scope.
- Official weights: load checkpoints for sizes `n`, `s`, `m`, `b`, `l`, `x`.
- PyTorch-native configuration: build models and pass args in Python or via CLI flags; no YAML files.

## Features 
- Load official YOLOv10 weights with exact parity across sizes.
- Inference: images and folders save visualizations.
- Validation: COCO-style metrics (mAP .5:.95), PR curves.
- Training: baseline reproduction with AMP/EMA/grad accumulation.
- Export: TorchScript and ONNX.

## Getting Started

- Prerequisites: Python 3.9+ (3.10/3.11 recommended), Git

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
- This installs CUDA-enabled wheels (cu12x). They run on CPU too; GPU is used if available.
- If you explicitly want CPU-only wheels, use the CPU index instead.

Install project dependencies:
```
pip install -r requirements.txt
```

## Python API

Build a model using PyTorch-style API with model registry pattern:
```python
import torch
from lean_yolo.models import get_model, get_model_weights, list_models

# List available models (similar to torchvision.models)
all_models = list_models()  # Returns ['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x']
print(f"Available models: {all_models}")

# Load a model with pretrained weights (weights="DEFAULT" loads official weights)
model = get_model("yolov10s", weights="DEFAULT")
model.eval()

# Alternative: load with specific configuration
model = get_model("yolov10s", num_classes=80, weights=None)  # No pretrained weights
model.eval()

# Load weights separately if needed
weights_enum = get_model_weights("yolov10s")  # Returns the weights enum class
weights = weights_enum.DEFAULT  # Access pretrained weights
model.load_state_dict(weights.get_state_dict(progress=True))

# Forward a dummy tensor
x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    out = model(x)
```

Weight loading notes
- Official sources only: downloads use the YOLOv10 official releases from `THU-MIG/yolov10` (e.g., v1.1 assets like `yolov10s.pt`). Ultralytics weights are not used here.
- Offline options:
  - Set `LEAN_YOLO_WEIGHTS_DIR=/path/to/weights` and place files like `yolov10s.pt` there, or
  - Pass a local file path to `get_state_dict(local_path=...)`.
- If weights cannot be found, the model initializes randomly and a warning is emitted.


## CLI 

CLI entrypoints
- `infer.py`: basic inference with letterbox preprocessing and NMS
- `val.py`: COCO validation (downloads val2017 on demand)

Notes
- Training requires COCO JSON annotation format (standard COCO dataset structure).
- Class names are automatically extracted from the COCO JSON files.
- For COCO, the standard 80-class list is used.

Preprocessing
- Inference/validation use letterbox resize with padding (114,114,114) for parity with official preprocessing.

## Datasets

COCO JSON format is the primary supported format. The project expects standard COCO dataset structure:

```
data/
  images/
    train2017/ ... .jpg
    val2017/   ... .jpg
  annotations/
    instances_train2017.json
    instances_val2017.json
```

The COCO JSON files provide class definitions, object annotations, and image metadata.

YOLO text format is not supported.

## YOLOv10 Compatibility

Supported sizes : `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, `yolov10x`.

 Weight loading
- Exact loading of all official THU-MIG release weights with the lean implementation. The official repository is not imported at runtime.

## Roadmap (YOLOv10-first)
- Implement YOLOv10 modules and forward pass
- Create unit tests for each module (backbone, neck, head)
- Generate reference outputs from official repo (using CPU for reproducibility)
- Validate module outputs match official implementation with test fixtures
- Load official weights for n/s/m/b/l/x (verify parameter count parity)
- Complete end-to-end output verification against official implementation
- Inference pipeline and visualizations
- Validation metrics (COCO mAP) and val CLI
- Optional training loop (reproduce baseline)
- Export (ONNX/TorchScript) and quick perf checks

## License
MIT License — see `LICENSE` for details.

## References
- [YOLOv10 Official Repository](https://github.com/THU-MIG/yolov10)

- [YOLOv10 Paper](https://arxiv.org/abs/2405.14458)
- [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv7 Paper](https://arxiv.org/abs/2207.02696)
- [YOLOv6 Paper](https://arxiv.org/abs/2209.02976)
- [YOLOv5 Technical Report](https://github.com/ultralytics/yolov5)
- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [YOLOv2 Paper](https://arxiv.org/abs/1612.08242)
- [YOLOv1 Paper](https://arxiv.org/abs/1506.02640)

## Acknowledgements
- Inspired by the YOLO family and community implementations.
- Thanks to the PyTorch community and contributors to related tooling.
