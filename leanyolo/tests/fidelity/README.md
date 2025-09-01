# Fidelity Testing Framework

Purpose: verify that the leanyolo implementation reproduces the official YOLOv10 outputs across all official pretrained weights (n, s, m, l, x), focusing purely on functional correctness.

## Rubric

- Structural Fidelity: exact tensor shape matches for backbone (C3/C4/C5), neck (P3/P4/P5), and head outputs per level.
- Numerical Accuracy: torch.allclose checks with per-component tolerances (rtol/atol) and optional max_abs caps.
- Weight Coverage: separate suites for `yolov10n`, `yolov10s`, `yolov10m`, `yolov10l`, `yolov10x`.
- Full Pipeline: end-to-end training-style outputs from the detect head compared against references.

Default tolerances (per `rubric.py`):
- Backbone: rtol=1e-4, atol=1e-4, max_abs=5e-4
- Neck: rtol=1e-4, atol=1e-4, max_abs=5e-4
- Head: rtol=1e-4, atol=1e-4, max_abs=5e-4
- End-to-end: rtol=1e-4, atol=1e-4, max_abs=5e-4

Adjust per weight if needed by editing `RUBRICS` in `rubric.py`.

## Components Under Test

- Backbone: outputs C3, C4, C5
- Neck: outputs P3, P4, P5 (PAN-FPN)
- Head: training-style outputs per level (P3, P4, P5); raw logits/regressions, no decode/NMS

Official reference extraction hooks use stable indices from the YAML graphs: backbone at layers [4, 6, 10] and neck at [16, 19, 22] for all sizes.

## Data and Artifacts

- Inputs: deterministic tensors saved under `leanyolo/tests/data/inputs/x_320.pt` (auto-generated on first run if missing).
- References: per-model tensors saved under `leanyolo/tests/data/refs/<model>/`:
  - `backbone_c3.pt`, `backbone_c4.pt`, `backbone_c5.pt`
  - `neck_p3.pt`, `neck_p4.pt`, `neck_p5.pt`
  - `head_p3.pt`, `head_p4.pt`, `head_p5.pt`
  - `meta.json` with shapes, dtype, weights path, and image size
- Reports: JSON summaries saved to `leanyolo/tests/reports/<model>-<timestamp>.json`.

## Generating Reference Outputs

Prerequisites:
- Install PyTorch and the project requirements.
- Ensure `yolov10-official/` is present (already vendored in this repo).

Command:
```
python -m leanyolo.tests.fidelity.generate_references --sizes n s m l x --img 320
```

Notes:
- The generator resolves official THU-MIG `*.pt` weights via the leanyolo weights registry. You may set `LEANYOLO_WEIGHTS_DIR` to a local directory containing weight files to avoid network.
- References are written under `leanyolo/tests/data/refs/`.

## Running Tests

```
pytest -k fidelity -q
```

If references are missing, tests are skipped with an instruction to generate them.

## Updating for New Weights

When official weights are updated:
1. Update SHA256 in `leanyolo/models/registry.py` if needed.
2. Regenerate references:
   ```
   python -m leanyolo.tests.fidelity.generate_references --sizes n s m l x --img 320
   ```
3. Inspect reports under `leanyolo/tests/reports/`.

## Compatibility Matrix

This framework ties a leanyolo commit to specific weight checksums and reference outputs. Maintain a simple matrix in your release notes or CI to track which leanyolo version was validated against which weight SHA:

| leanyolo commit | yolov10n SHA256 | yolov10s SHA256 | yolov10m SHA256 | yolov10l SHA256 | yolov10x SHA256 |
|------------------|------------------|------------------|------------------|------------------|------------------|
| <git-sha>        | <sha>            | <sha>            | <sha>            | <sha>            | <sha>            |
