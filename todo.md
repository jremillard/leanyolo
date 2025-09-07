# leanyolo TODO

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
 
- Broad exceptions in tools/update_dog_viz.py; narrow exception types and log context.
- Long lines and large functions in tools/transfer_learn_aquarium.py and tools/train.py; consider minor refactors for readability.

Backlog
- Export (ONNX/TorchScript) and quick perf checks for YOLOv10.
- Optional richer data augmentations (HSV jitter, mosaic) gated behind flags.

## Export ONNX Plan (Pending)

Scope
- End-to-end YOLOv10 export to ONNX producing final post-NMS detections.

Constraints and Defaults
- Outputs: final boxes after YOLO-style class-wise NMS (matches Python).
- Opset: default 19 (current widely-supported), overridable via CLI.
- Shapes: dynamic batch axis; static HxW (no dynamic spatial size).
- Image size: default 640x640; channels-first NCHW.
- Precision: float32 by default; optional fp16 (`--half`) export.
- Preprocessing: identical to Python inference (scaling/normalization, channel order).
- Verification: high confidence on CPU and GPU with onnxruntime providers.

Deliverables
- `tools/export_onnx.py` CLI for exporting checkpoints to ONNX.
- `leanyolo/leanyolo/models/yolov10/model.py`: `export_onnx()` helper.
- Postprocess graph components (decode + NMS) exportable to ONNX.
- Unit tests validating exportability and PyTorch↔ONNX parity.
- README section documenting usage and validation steps.

Planned Steps (all pending)
1) Audit YOLOv10 ops for ONNX
   - Identify layers/ops used in backbone/neck/head/postprocess.
   - Replace/export-safe variations where needed (e.g., avoid Python loops, ensure SiLU/Concat/Reshape are traceable).
   - Confirm no control flow or data-dependent shape ops block export.

2) Outputs: final post-NMS boxes
   - Define ONNX outputs as `detections: (B, N, 6)` where each row is `[x1,y1,x2,y2,score,class]` with `N = --max-dets` and zero-padding when fewer detections.
   - Add `num_dets: (B,)` to report valid counts per image.

3) Choose stable opset (default 19)
   - Default to opset 19; expose `--opset` to override.
   - Validate ops availability (NonMaxSuppression, Cast, TopK, Gather, Slice, Where) under chosen opset.

4) Implement decode + class-wise YOLO NMS (export-friendly)
   - Move/guard decode logic into an export path to avoid unsupported ops and Python-side indexing tricks.
   - Implement class-wise NMS using ONNX `NonMaxSuppression`:
     - Option A: per-class NMS loop via scripted/rolled graph components (concatenate results), keeping shapes static by padding per class then merging.
     - Option B: class-offset trick (add large class-dependent offsets to boxes) to enable class-wise behavior with a single NMS pass; verify numeric stability.
   - Enforce `--max-dets`, `--conf`, and `--iou` thresholds inside the graph; pad/sort to fixed `N`.

5) Implement CLI: `tools/export_onnx.py`
   - Args: `--weights PATH` (or `--model yolov10[n|s|m|l|x]`), `--batch INT` (dummy batch for export/validation), `--imgsz 640`, `--max-dets 300`, `--opset 19`, `--half`, `--conf 0.25`, `--iou 0.7`, `--output PATH`.
   - Behavior: constructs model, loads weights, switches to eval, freezes, and calls `export_onnx()` with dynamic batch axis enabled.
   - Writes metadata (inputs/outputs, opset, image size, thresholds) alongside the ONNX file.

6) Add `model.export_onnx()` with dynamic batch
   - Signature: `export_onnx(path, dummy_batch, imgsz, opset=19, half=False, max_dets=300, conf=0.25, iou=0.7)`.
   - Uses `torch.onnx.export` with `training=False`, `do_constant_folding=True`, and `dynamic_axes` for batch.
   - Inputs: dummy tensor shape `(dummy_batch, 3, imgsz, imgsz)` and dtype per `half`; exported ONNX has dynamic batch.
   - Ensures decode + NMS are part of the graph to return final detections.

7) Match preprocessing and dtypes
   - Mirror Python preprocessing: scale to [0,1], channel order, normalization; document exact expectations.
   - Decide whether to include normalization inside the graph; if not, document clear pre-processing required by runtime.

8) Support dynamic batch axis
   - Export with `dynamic_axes={"images": {0: "batch"}, "detections": {0: "batch"}, "num_dets": {0: "batch"}}`.
   - Validate inference works on multiple batch sizes (e.g., 1, 4, 8) with identical outputs to PyTorch.

9) Validate with onnxruntime (CPU and CUDA)
   - Add verification routine invoked by CLI flag `--validate`:
     - Generate synthetic batches of different sizes and compare PyTorch vs ONNX outputs: shapes, dtypes, and values (tolerances: atol ~1e-3 fp32, ~1e-2 fp16).
     - Run with `CPUExecutionProvider`; if available, also with `CUDAExecutionProvider`.
     - Log provider availability and any fallbacks.

10) Add PyTorch↔ONNX parity tests
   - `leanyolo/tests/test_export_onnx.py`:
     - Export a small variant at `imgsz=640`, `dummy_batch=1`, `max_dets=50` (dynamic batch enabled).
     - Verify exported graph loads and runs on CPU; compare against PyTorch for fixed seed images at batch sizes 1 and >1.
     - Conditionally run GPU provider validation if available; skip otherwise.

11) Document export and runtime usage
   - README: add “Export to ONNX” section with exact commands and notes on inputs/outputs and preprocessing.
   - Include guidance for loading the model in onnxruntime (CPU/GPU) with example code.

Acceptance Criteria
- Export produces an ONNX with dynamic batch input `(_,3,640,640)` and returns final detections as `(_,N,6)` with `num_dets`.
- Parity tests pass within tolerances on CPU; optional GPU validation passes when available.
- CLI successfully exports from a real checkpoint and logs opset, shapes, and thresholds.
- Preprocessing semantics match Python inference path; documented clearly.
/