# leanYOLO

The goal of this project is to provide a lean, easy-to-understand, and easy-to-integrate PyTorch implementation of YOLO models.

## Scope & Status

This project focuses on providing a minimal, yet functional, implementation of YOLOv10 for object detection.

-   **Core Implementation:** Model registry, architecture, official weight loading, inference/validation CLIs, and tests are complete.
-   **YOLOv10 Only:** Implements the YOLOv10 backbone, neck, and detection head.
-   **Object Detection Only:** Segmentation, pose estimation, and oriented bounding box (OBB) detection are out of scope.
-   **Official Weights:** Supports loading official pretrained checkpoints for `n`, `s`, `m`, `b`, `l`, and `x` models.

## A Brief History of YOLO

YOLO (You Only Look Once) is a family of real-time object detection algorithms in computer vision. It is known for its ability to predict bounding boxes and class probabilities from full images in a single pass, making it a popular choice for applications requiring high speed and efficiency, especially on edge devices.

## Getting Started

### Prerequisites

-   Python 3.9+ (3.10/3.11 recommended)
-   Git

### Installation

1.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install PyTorch with CUDA support:**

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

3.  **Install project dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Testing

See [sqa.md](sqa.md) for the full Software Quality Assurance plan, including environment validation, unit tests, functional tool checks, and integration scenarios.

## Python API

Use the PyTorch-style API to build and use models.

```python
import torch
from leanyolo.models import get_model, list_models
from leanyolo.data.coco import coco80_class_names

# List available models (similar to torchvision.models)
print(f"Available models: {list_models()}")
# Output: ['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x']

# Load a model with pretrained COCO weights
# The model handles input normalization (division by 255) internally.
model = get_model(
    "yolov10s",
    weights="PRETRAINED_COCO",
    class_names=coco80_class_names(),
)
model.eval()

# Example: Forward a dummy tensor
# In eval mode, the model returns a list of decoded detections per image.
# Each detection is a [N, 6] tensor: [x1, y1, x2, y2, score, class_index].
x = torch.zeros(1, 3, 640, 640)
model.post_conf_thresh = 0.25  # Confidence threshold
model.post_iou_thresh = 0.45   # IoU threshold for Non-Maximum Suppression (NMS)

with torch.no_grad():
    detections = model(x)
    print(detections[0].shape) # Shape of detections for the first image
```

### Weight Loading

-   **Official Sources:** Weights are downloaded from the official `THU-MIG/yolov10` repository releases.
-   **Offline Usage:**
    -   Set the `LEAN_YOLO_WEIGHTS_DIR` environment variable to a directory containing the `.pt` weight files.
    -   Alternatively, pass a local file path directly to `get_model(weights="/path/to/weights.pt", ...)`.
-   **Initialization:** If weights are not found or specified, the model will be initialized with random weights, and a warning will be issued.

## Command-Line Interface (CLI)

This project provides simple and adaptable CLI scripts for common tasks.

-   `tools/infer.py`: Perform inference on images with letterbox preprocessing, decoding, NMS, and visualization.
-   `tools/val.py`: Run COCO validation and export results to a JSON file.
-   `tools/train.py`: A baseline training script demonstrating data loading, loss calculation, scheduling, evaluation, and checkpointing.
-   `tools/prepare_aquarium.py`: Unzip and arrange the Aquarium dataset into a COCO-style layout.
-   `tools/transfer_learn_aquarium.py`: An example of transfer learning with AMP, learning rate scheduling, and gradual unfreezing.
-   `tools/export_onnx.py`: Export YOLOv10 to ONNX with dynamic batch and fixed detections output.

These scripts are designed to be easy to copy and modify for your own projects.

## Datasets

The primary supported annotation format is **COCO JSON**. The project expects the standard COCO dataset structure:

```
data/
  images/
    train2017/
      - image1.jpg
      - ...
    val2017/
      - image2.jpg
      - ...
  annotations/
    - instances_train2017.json
    - instances_val2017.json
```

*YOLO text format is not supported.*

## YOLOv10 Compatibility

This project supports `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, and `yolov10x` models.

-   **Weight Loading:** Ensures exact loading of all official `THU-MIG/yolov10` release weights without importing the official repository at runtime.
-   **Validation:** The validation process uses the official YOLOv10 NMS-free top-k decoding on the one-to-one branch to match the reference evaluation setup.

### Validation Results

COCO mAP@0.5:0.95 comparison on `val2017`:

| Model    | Official mAP | LeanYOLO mAP | Difference |
|:---------|:-------------|:-------------|:-----------|
| yolov10n | 0.38480      | 0.38470      | 0.00010    |
| yolov10s | 0.45866      | 0.46115      | 0.00249    |
| yolov10m | 0.50999      | 0.50909      | 0.00090    |
| yolov10b | 0.52303      | 0.52188      | 0.00115    |
| yolov10l | 0.53018      | 0.52868      | 0.00150    |
| yolov10x | 0.54231      | 0.54127      | 0.00104    |

*mAP@0.5:0.95 is the mean Average Precision over IoU thresholds from 0.5 to 0.95.*

## Transfer Learning Example: Aquarium Dataset

This example demonstrates how to train YOLOv10 on the Kaggle Aquarium dataset.

### 1. Prepare the Dataset

Unzip the dataset and organize it into a COCO-style directory structure.

```bash
python tools/prepare_aquarium.py --root data/aquarium --zip data/AquariumDataset.zip --clean
```

This command creates the following structure:
```
data/aquarium/
  images/
    train/
    val/
  train.json
  val.json
```

### 2. Train the Model

This project provides two training scripts: a best-practice transfer learning script and a simpler baseline trainer.

**Recommended: Transfer Learning Script**

This script uses AMP, a cosine learning rate scheduler, and gradual unfreezing.

```bash
python tools/transfer_learn_aquarium.py \
  --root data/aquarium \
  --device cuda \
  --model yolov10m \
  --weights PRETRAINED_COCO \
  --imgsz 640 \
  --epochs 50 \
  --batch-size 32 \
  --save-dir runs/transfer/aquarium_yolov10m_640
```
*Note: If you encounter out-of-memory (OOM) errors, try reducing the `--batch-size`.*

**Alternative: Baseline Trainer**

```bash
python tools/train.py \
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
  --save-dir runs/train/aquarium_n
```

### 3. Run Inference

Use your trained checkpoint to run inference on the validation set.

```bash
python tools/infer.py \
  --source data/aquarium/images/val \
  --model yolov10m \
  --weights runs/transfer/aquarium_yolov10m_640/best.pt \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.45 \
  --device cuda \
  --save-dir runs/infer/aquarium_yolov10m_640_best
```

## Export to ONNX

Export a YOLOv10 model to ONNX with dynamic batch input and fixed‑shape detections per image.

```bash
# NMS-free (top-k) decode (default)
python tools/export_onnx.py \
  --model yolov10s \
  --weights PRETRAINED_COCO \
  --imgsz 640 \
  --max-dets 300 \
  --opset 19 \
  --output runs/export/yolov10s.onnx \
  --decode topk \
  --validate

# Class-wise NMS inside ONNX graph (uses ONNX NonMaxSuppression)
python tools/export_onnx.py \
  --model yolov10s \
  --weights PRETRAINED_COCO \
  --imgsz 640 \
  --max-dets 300 \
  --opset 19 \
  --output runs/export/yolov10s_nms.onnx \
  --decode nms --iou 0.45 --pre-topk 1000
```

Outputs:
- Input: `images` with shape `[N, 3, 640, 640]`
- Outputs:
  - `detections`: `[N, max_dets, 6]` with rows `[x1,y1,x2,y2,score,cls]`
  - `num_dets`: `[N]` valid count per image

Notes:
- Input dtype matches export: float32 (or float16 with `--half`), RGB CHW. The graph applies normalization internally via registered buffers (default is divide‑by‑255). Feed values in [0,255] and do not pre‑divide. If you prefer to feed [0,1] tensors, export a model constructed with `input_norm_divide=[1,1,1]` so the graph skips scaling.
- Choose `--decode topk` to match official YOLOv10’s NMS‑free evaluation or `--decode nms` for class‑wise NMS using ONNX NonMaxSuppression. In both cases, `--conf` and `--max-dets` are enforced inside the graph.

## A Note on Ultralytics

The Ultralytics repository provides user-friendly APIs and a large ecosystem for many YOLO versions (YOLOv3-YOLOv12). It uses YAML configuration files, a convention inherited from Darknet. While this approach is valuable for research and backward compatibility, it can add complexity for developers accustomed to standard PyTorch patterns. Additionally, Ultralytics uses the AGPL-3.0 license, which requires derivative works to be open-sourced, with commercial licenses available separately.

## References

For completeness, here is a list of the original YOLO papers and their implementations.

| Model      | Year | Paper                                                                                               | Repository                                               |
|:-----------|:-----|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------|
| YOLOv1     | 2015 | [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)            | [pjreddie/darknet](https://github.com/pjreddie/darknet)     |
| YOLOv2     | 2016 | [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)                               | [pjreddie/darknet](https://github.com/pjreddie/darknet)     |
| YOLOv3     | 2018 | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)                               | [pjreddie/darknet](https://github.com/pjreddie/darknet)     |
| YOLOv4     | 2020 | [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)            | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)     |
| YOLOv5     | 2020 | [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/models/yolov5/)                                | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
| PP-YOLO    | 2020 | [PP-YOLO: An Effective and Efficient Implementation of Object Detector](https://arxiv.org/abs/2007.12099) | [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) |
| YOLOv6     | 2022 | [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976) | [meituan/YOLOv6](https://github.com/meituan/YOLOv6)         |
| YOLOv7     | 2022 | [YOLOv7: Trainable bag-of-freebies...](https://arxiv.org/abs/2207.02696)                               | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)   |
| DAMO-YOLO  | 2022 | [DAMO-YOLO: A Report on Real-Time Object Detection Design](https://arxiv.org/abs/2211.15444)           | [tinyvision/DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) |
| YOLOv8     | 2023 | [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/)                                | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| YOLO-NAS   | 2023 | [Deci YOLO-NAS: Neural Architecture Search for Object Detection](https://github.com/Deci-AI/super-gradients) | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) |
| YOLO-World | 2024 | [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270)            | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) |
| YOLOv9     | 2024 | [YOLOv9: Learning What You Want to Learn...](https://arxiv.org/abs/2402.13616)                         | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)   |
| YOLOv10    | 2024 | [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)                      | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)       |
| YOLOv11    | 2024 | [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/models/yolov11/)                                | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| YOLOv12    | 2025 | [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)              | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
