# leanYOLO

This project (leanYOLO) goal is to provide PyTorch implementations of YOLOs that is easy to integrate, easy to understand.

## Status
- Core code implemented: model registry, architecture, exact weight loader, inference/validation CLIs, and tests.

## Scope
- YOLOv10-only: backbone, neck, detection head.
- Object detection only: this project focuses solely on detection. Segmentation, pose, and OBB are out of scope.
- Official weights: load checkpoints for sizes `n`, `s`, `m`, `b`, `l`, `x`.

## YOLO History

YOLO (You Only Look Once) is a real-time object deep learning detection algorithm in computer vision that predicts 
bounding boxes and class probabilities directly from full images in a single pass.

YOLO's focus is on speed, accuracy, and efficiency for object detection tasks, often used in applications running on edge devices or that require high speed.

## Getting Started

- Prerequisites: Python 3.9+ (3.10/3.11 recommended), Git

Create and activate a virtual environment (Linux/macOS):
```
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch with CUDA in the venv
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:
```
pip install -r requirements.txt
```

## Python API

Build a model using PyTorch-style API with model registry pattern:
```python
import torch
from lean_yolo.models import get_model, get_model_weights, list_models
from leanyolo.data.coco import coco80_class_names

# List available models (similar to torchvision.models)
all_models = list_models()  # Returns ['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x']
print(f"Available models: {all_models}")

# Load a model with pretrained weights (weights="PRETRAINED_COCO" loads official weights)
# Use YOLO-standard input normalization: subtract [0,0,0], divide by [255,255,255]
model = get_model(
    "yolov10s",
    weights="PRETRAINED_COCO",
    class_names=coco80_class_names(),
    input_norm_subtract=[0, 0, 0],
    input_norm_divide=[255, 255, 255],
)
model.eval()

# Alternative: load with specific configuration
# To skip normalization inside the model, use [0,0,0] and [1,1,1]
model = get_model(
    "yolov10s",
    weights=None,
    class_names=coco80_class_names(),
    input_norm_subtract=[0, 0, 0],
    input_norm_divide=[1, 1, 1],
)  # No pretrained weights
model.eval()

# Load weights separately if needed
weights_enum = get_model_weights("yolov10s")  # Returns the resolver type
weights_entry = weights_enum().get("yolov10s", "PRETRAINED_COCO")  # Pretrained COCO
model.load_state_dict(weights_entry.get_state_dict(progress=True))

# Forward a dummy tensor (model applies normalization). In eval mode, YOLOv10
# models return decoded detections per image: list of [N,6] tensors
# [x1, y1, x2, y2, score, cls].
x = torch.zeros(1, 3, 640, 640)
model.post_conf_thresh = 0.25  # confidence threshold (after sigmoid)
model.post_iou_thresh = 0.45   # IoU threshold for NMS (higher keeps more overlaps)
with torch.no_grad():
    raw = model(x)  # [P3,P4,P5] raw tensors in training/eval
    dets_per_img = model.decode_forward(raw)
    dets = dets_per_img[0][0]  # [N,6]
    # x1,y1 = top-left; x2,y2 = bottom-right (pixels in input letterbox space)
```

Weight loading notes
- Official sources only: downloads use the YOLOv10 official releases from `THU-MIG/yolov10` (e.g., v1.1 assets like `yolov10s.pt`). Ultralytics weights are not used here.
- Offline options:
  - Set `LEAN_YOLO_WEIGHTS_DIR=/path/to/weights` and place files like `yolov10s.pt` there, or
  - Pass a local file path to `get_state_dict(local_path=...)`.
- If weights cannot be found, the model initializes randomly and a warning is emitted.


## CLI 

CLI entrypoints (copy/paste friendly)
- `infer.py`: basic inference with letterbox preprocessing, decode, NMS, and visualization
- `val.py`: COCO validation (downloads val2017 on demand) and JSON export
- `tools/train.py`: baseline trainer showing data loading, loss, scheduler, eval, and checkpointing
- `tools/prepare_acquirium.py`: unzip + arrange Aquarium into COCO layout (Windows‑friendly; no symlinks)
- `tools/transfer_learn_aquarium.py`: transfer learning example with AMP, warmup+cosine LR, light aug, gradual unfreeze

These scripts are intentionally simple and serve as example code. Feel free to
copy/paste and adapt them for your own datasets and pipelines.

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

Supported models : `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, `yolov10x`.

 Weight loading
- Exact loading of all official THU-MIG release weights with the lean implementation. The official repository is not imported at runtime.

## Validation

COCO mAP@0.5:0.95 comparison on val2017:

| Model    | Official mAP | LeanYOLO mAP | Difference |
|----------|--------------|--------------|------------|
| yolov10n | 0.38480      | 0.38115      | 0.00365    |
| yolov10s | 0.45866      | 0.45344      | 0.00522    |
| yolov10m | 0.50999      | 0.49910      | 0.01089    |
| yolov10b | 0.52303      | 0.51141      | 0.01162    |
| yolov10l | 0.53018      | 0.51902      | 0.01116    |
| yolov10x | 0.54231      | 0.52934      | 0.01297    |

mAP@0.5:0.95 is the mean Average Precision (mAP) evaluated at Intersection over Union (IoU) thresholds ranging from 0.5 to 0.95 in steps of 0.05. This metric assesses object detection performance by averaging precision across multiple recall levels and IoU thresholds, providing a comprehensive measure of accuracy for COCO dataset evaluations.


## Ultralytics

Ultralytics' is providing user-friendly APIs, pre-trained models, and a large community ecosystem. Ultralytics maintains implementations of YOLOv3 through YOLOv12 in a single repository, preserving the Darknet convention of configuring models with YAML config files even in their PyTorch implementations. While many versions of YOLO are functionally obsolete for most applications, they remain in the Ultralytics repo for historical and research purposes. This backward compatibility approach, while serving R&D needs for exploring new model architectures and conducting comparative studies, makes the official Ultralytics repository more complex and unfamiliar to engineers who primarily work with standard PyTorch patterns and conventions. Also Ultralytics use AGPL-3.0 (requiring open-sourcing of derivative works) with commercial licenses available.

## References

Based on the descriptions above, the original YOLO papers are important for completeness.

| Model | Year | Framework | Paper | Repository |
|-------|------|-----------|-------|------------|
| YOLOv1 | 2015 | Darknet | [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) | [Repository](https://github.com/pjreddie/darknet) |
| YOLOv2 | 2016 | Darknet | [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) | [Repository](https://github.com/pjreddie/darknet) |
| YOLOv3 | 2018 | Darknet | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) | [Repository](https://github.com/pjreddie/darknet) |
| YOLOv4 | 2020 | Darknet | [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) | [Repository](https://github.com/AlexeyAB/darknet) |
| YOLOv5 | 2020 | PyTorch | [Ultralytics YOLOv5 Documentation](https://docs.ultralytics.com/models/yolov5/) | [Repository](https://github.com/ultralytics/yolov5) |
| PP-YOLO | 2020 | PaddlePaddle | [PP-YOLO: An Effective and Efficient Implementation of Object Detector](https://arxiv.org/abs/2007.12099) | [Repository](https://github.com/PaddlePaddle/PaddleDetection) |
| YOLOv6 | 2022 | PyTorch | [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976) | [Repository](https://github.com/meituan/YOLOv6) |
| YOLOv7 | 2022 | PyTorch | [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) | [Repository](https://github.com/WongKinYiu/yolov7) |
| DAMO-YOLO | 2022 | PyTorch | [DAMO-YOLO: A Report on Real-Time Object Detection Design](https://arxiv.org/abs/2211.15444) | [Repository](https://github.com/tinyvision/DAMO-YOLO) |
| YOLOv8 | 2023 | PyTorch | [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/) | [Repository](https://github.com/ultralytics/ultralytics) |
| YOLO-NAS | 2023 | PyTorch | Deci YOLO-NAS: Neural Architecture Search for Object Detection | [Repository](https://github.com/Deci-AI/super-gradients) |
| YOLO-World | 2024 | PyTorch | [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270) | [Repository](https://github.com/AILab-CVC/YOLO-World) |
| YOLOv9 | 2024 | PyTorch | [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616) | [Repository](https://github.com/WongKinYiu/yolov9) |
| YOLOv10 | 2024 | PyTorch | [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458) | [Repository](https://github.com/THU-MIG/yolov10) |
| YOLOv11 | 2024 | PyTorch | [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolov11/) | [Repository](https://github.com/ultralytics/ultralytics) |
| YOLOv12 | 2025 | PyTorch | [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524) | [Repository](https://github.com/ultralytics/ultralytics) |

## License
MIT License — see `LICENSE` for details.
## Transfer Learning (Aquarium)

Train YOLOv10 on the Kaggle Aquarium dataset (COCO-style):

1) Prepare dataset from local zip (Windows‑friendly; no symlinks):

```
./.venv/bin/python tools/prepare_acquirium.py --root data/aquarium --zip data/AquariumDataset.zip --clean
```

This prepares:
```
data/aquarium/
  images/train/ ...
  images/val/   ...
  train.json
  val.json
```

2) Train (best‑practice transfer learning script)

Recommended command (CUDA, 50 epochs, 640 resolution, AMP on by default):

```
./.venv/bin/python tools/transfer_learn_aquarium.py \
  --root data/aquarium \
  --device cuda \
  --model yolov10m \
  --weights PRETRAINED_COCO \
  --imgsz 640 \
  --epochs 50 \
  --batch-size 32 \
  --save-dir runs/transfer/aquarium_yolov10m_640
```

Notes:
- AMP enabled by default; disable with `--no-amp` if needed.
- Backbone/neck frozen and head reset by default; unfreezes at epoch 5 (configurable).
- If you see OOM, reduce `--batch-size` (16–24 for 16 GB is typical at 640). 

Alternate baseline trainer (simple loop), if you prefer:

```
./.venv/bin/python tools/train.py \
  --train-images data/aquarium/images/train \
  --train-ann data/aquarium/train.json \
  --val-images data/aquarium/images/val \
  --val-ann data/aquarium/val.json \
  --model yolov10n \
  --weights PRETRAINED_COCO \
  --imgsz 640 \
  --epochs 20 \
  --batch-size 16 \
  --device cuda \
  --freeze-backbone \
  --head-reset \
  --save-dir runs/train/aquarium_n
```

3) Inference with your trained checkpoint

Use the best epoch from the log (e.g., epoch035.pt), make sure `--model` matches the trained variant:

```
./.venv/bin/python infer.py \
  --source data/aquarium/images/val \
  --model yolov10m \
  --weights runs/transfer/aquarium_yolov10m_640/epoch035.pt \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.65 \
  --device cuda \
  --classes-ann data/aquarium/val.json \
  --save-dir runs/infer/aquarium_yolov10m_640_e35
```

Latest run snapshot (yolov10m, 640, 50 epochs):
- Best mAP50‑95 ≈ 0.389 at epoch 35, stable plateau ~0.38–0.39.
- Smooth loss decline; no obvious overfitting. Consider early‑stopping around best epoch.

```
./.venv/bin/python val.py \
  --images data/aquarium/images/val \
  --ann data/aquarium/val.json \
  --model yolov10n \
  --weights runs/train/aquarium_n/ckpt.pt \
  --imgsz 640 \
  --device cuda \
  --save-viz-dir runs/val/aquarium_viz
```

Notes:
- The model normalizes inputs internally (divide by 255). No external scaling needed.
- Full mAP evaluation (COCO-style) runs after each epoch on the validation set.
- The training script saves checkpoints per epoch and a final `ckpt.pt`, which can be loaded via `get_model(name, weights="/path/to/ckpt.pt", class_names=...)`.
