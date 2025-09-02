# leanyolo TODO

Status (2025‑09‑01)
- Dataset prep: `scripts/prepare_acquirium.py` added (Windows‑friendly; no symlinks). Done.
- Transfer script: `scripts/transfer_learn_aquarium.py` added with AMP default, warmup+cosine LR, light aug, gradual unfreeze, detailed logging. Done.
- Baseline trainer moved to `scripts/train.py`. Done.
- Eval helper: `evaluate_coco` renamed to `evaluate` (COCO‑format mAP). Done.

Recent results
- Aquarium (COCO‑format), yolov10m @ 640, 50 epochs, batch 32, AMP: best mAP50‑95 ≈ 0.389 at epoch 35.

Next improvements
- Save best checkpoint automatically (best.pt) based on mAP.
- Early stopping on plateau (patience N epochs).
- Cache COCO ground truth object and filename→id map across epochs to reduce eval overhead.
- CLI to control eval post‑NMS max detections (default 300) for faster mid‑run evals.
- Gradient accumulation option to simulate larger batch sizes.
- Fix run folder naming to reflect actual model variant consistently.
- Optional eval‑phase progress logging (every K images) for transparency during COCOeval.

Backlog
- Export (ONNX/TorchScript) and quick perf checks for YOLOv10.
- Optional richer data augmentations (HSV jitter, mosaic) gated behind flags.
