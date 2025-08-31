# lean-yolo

Clean, minimal PyTorch implementation focused on YOLOv10. Goal: a faithful, readable YOLOv10 port that can load official pretrained weights across standard sizes using typical PyTorch conventions (no YAML configs).

## Scope
- YOLOv10-only: backbone, neck, detection head.
- Official weights: load checkpoints for sizes `n`, `s`, `m`, `b`, `l`, `x`.
- PyTorch-native configuration: build models and pass args in Python or via CLI flags; no YAML files.

## Features 
- Load official YOLOv10 weights (direct or light key remap).
- Inference: images and folders save visualizations.
- Validation: COCO-style metrics (mAP .5:.95), PR curves.
- Training: baseline reproduction with AMP/EMA/grad accumulation.
- Export: TorchScript and ONNX.

## Getting Started

Prerequisites
- Python 3.9+ (3.10/3.11 recommended)
- PyTorch (CPU or CUDA build for your hardware)
- Git

Create an environment (venv on Linux)
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies (minimal set; adjust PyTorch index for your CUDA/CPU)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow opencv-python tqdm matplotlib pycocotools
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


## CLI 

Typical flags configure everything directly; pass paths and sizes explicitly.

Train
```
python train.py \
  --model yolov10 \
  --weights weights/yolov10s.pt \
  --train-coco data/annotations/instances_train2017.json \
  --train-images data/images/train2017 \
  --val-coco data/annotations/instances_val2017.json \
  --val-images data/images/val2017 \
  --epochs 300 --batch 16 --img 640 --device 0
```

Validate
```
python val.py \
  --model yolov10 \
  --weights runs/exp0/weights/best.pt \
  --val-coco data/annotations/instances_val2017.json \
  --val-images data/images/val2017 \
  --img 640 --batch 16 --device 0
```

Inference
```
python infer.py \
  --model yolov10 \
  --weights runs/exp0/weights/best.pt \
  --source path/to/images_or_video \
  --coco-classes data/annotations/instances_val2017.json \
  --img 640 --conf 0.25 --device 0
```

Export
```
python export.py --model yolov10s --weights best.pt --format onnx --img 640 --device cpu
```

Notes
- Training requires COCO JSON annotation format (standard COCO dataset structure).
- Class names are automatically extracted from the COCO JSON files.
- For COCO, the standard 80-class list is used.

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
- Drop-in when key names/shape match.
- Light adapter for minor key name differences or head bias shapes.
- Sanity checks: parameter count parity and dry-forward tests.

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
