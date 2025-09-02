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

Linter findings (major/high‑value)
- Narrow broad exception handling (W0718): leanyolo/utils/weights.py, leanyolo/utils/remap.py. Replace generic Exception with specific exceptions and handle/report accordingly.
- Complexity hotspots in remapping code: leanyolo/utils/remap.py triggers too many branches/returns/nested blocks (R0911/R0912/R1702) and large local variable counts. Consider splitting into focused helpers with unit tests.
- Data loader complexity: leanyolo/data/coco_simple.py has too many locals (R0914). Refactor long __getitem__ into smaller helpers (decode_anns, letterbox_boxes) with docstrings.
- Ambiguous variable name (E741): leanyolo/models/yolov10/postprocess.py uses `l` for left distance. Rename to `left` (and `right`, `top`, `bottom`) to reduce confusion.
 
- Docstrings: many public modules/classes/functions lack docstrings. Add brief docstrings for public APIs and core utilities.

Scripts lint highlights
 
- Broad exceptions in scripts/update_dog_viz.py; narrow exception types and log context.
- Long lines and large functions in scripts/transfer_learn_aquarium.py and scripts/train.py; consider minor refactors for readability.

Backlog
- Export (ONNX/TorchScript) and quick perf checks for YOLOv10.
- Optional richer data augmentations (HSV jitter, mosaic) gated behind flags.
