# Software Quality Assurance Plan – lean-YOLO

## 1. Purpose
Provide repeatable tests that verify correctness and stability of the lean-YOLO implementation, its standalone tools, and integration workflows (including transfer learning).

## 2. Test Environment
- Python 3.9+ (virtual environment recommended)
- PyTorch + dependencies from `requirements.txt`
- Optional CUDA-capable GPU
- Sample data:
  - `leanyolo/tests/data` (unit tests)
  - COCO-style dataset (e.g., Aquarium) for functional/integration tests
  - `dog.jpg` for inference demo

## 3. Test Matrix

### 3.0 Environment & Documentation Validation
| ID | Test Point | Steps | Expected Result |
|----|------------|-------|----------------|
| **EN-001** | Fresh env + imports | Delete `.venv` → `python -m venv .venv` → install torch per README → `./.venv/bin/python -m pip install -r requirements.txt` → `./.venv/bin/python tools/check_imports.py` | Prints `OK` and exits 0; README commands work without using sudo |

### 3.1 Unit Tests (`pytest -q -m "not fidelity"`)
| ID | Test Point | Steps | Expected Result |
|----|------------|-------|----------------|
| **UT-001** | `get_model` rejects unknown weight keys | Call `get_model("yolov10s", weights="DEFAULT")` | Raises `ValueError` |
| **UT-002** | `get_model` accepts `None` | Call with `weights=None` and COCO class names | Model returns without error, `class_names` length 80 |
| **UT-003** | Input normalization broadcast & validation | Call with `input_norm_subtract=[10.0]`; call with mismatched lengths | Broadcast to 3 channels; second call raises `ValueError` |
| **UT-004** | Plain state-dict load | Save state dict to temp file, reload with `get_model` | Parameters match original |
| **UT-005** | Incompatible state-dict detection | Save model with 3 classes, load into 2-class model | Raises `ValueError` mentioning incompatibility |
| **UT-006** | Weight download | Run `pytest leanyolo/tests/test_weights_download.py` | Pretrained weights downloaded or pulled from cache |
| **UT-007** | Safe unpickle | Run `pytest leanyolo/tests/test_weights_safe_unpickle.py` | Loading malicious pickle raises `ValueError` |
| **UT-008** | State-dict roundtrip | Run `pytest leanyolo/tests/test_state_dict_roundtrip.py` | Save→load yields identical params |
| **UT-009** | Letterbox transformations | Run `pytest leanyolo/tests/test_letterbox.py` | Output sizes, gains, pads match formulas |
| **UT-010** | Loss calculations | Run `pytest leanyolo/tests/test_losses_v10.py` | Expected tensor shapes/values, no NaNs |
| **UT-011** | Post-processing / NMS | Run `pytest leanyolo/tests/test_postprocess.py` | Correct number/order of boxes |
| **UT-012** | Remap official weights | Run `pytest leanyolo/tests/test_remap.py` | Remapped weights equal references |
| **UT-013** | Layer parity (backbone/neck/head) | Run fidelity tests under `-m fidelity` | Outputs match reference tensors |
| **UT-014** | Synthetic evaluation | Run `pytest leanyolo/tests/test_eval_synthetic.py` | mAP numbers consistent with snapshot |
| **UT-015** | Official remap parity | Run `pytest leanyolo/tests/test_remap_official.py` | Model layers match official parameters |

### 3.2 Functional Tests for Tools
| ID | Utility | Steps | Expected Result |
|----|---------|-------|----------------|
| **FT-001** | `tools/prepare_aquarium.py` | Run with sample Aquarium ZIP | COCO-style folder structure produced |
| **FT-002** | `tools/train.py` | Train on small dataset for 1 epoch (`--batch-size 2`) | Checkpoint + log files appear in `runs/train/*` |
| **FT-003** | `tools/val.py` | Validate pretrained model on small dataset | JSON results and console summary generated |
| **FT-004** | `tools/infer.py` | Run on `dog.jpg` with pretrained weights | Output image saved to `runs/infer/*`, console lists detections |
| **FT-005** | `tools/transfer_learn_aquarium.py` | Train for 3 epochs on Aquarium data | Log shows decreasing loss, new checkpoint saved |
| **FT-006** | `tools/download_all_pretrained.py` | Execute without arguments | All six YOLOv10 weight files downloaded |
| **FT-007** | `tools/convert_official_weights.py` | Convert official checkpoint to lean format | Converted `.pt` file loads successfully with `get_model` |

### 3.3 Integration Tests (Full Cycle)
| ID | Scenario | Steps | Expected Result |
|----|----------|-------|----------------|
| **IT-001** | Baseline train→val→infer | Prepare dataset → run `train.py` (1-2 epochs) → `val.py` → `infer.py` on sample image | Training completes; validation JSON shows expected mAP; inference outputs annotated image |
| **IT-002** | Transfer learning loop | Run `prepare_aquarium.py` → `transfer_learn_aquarium.py` (≥5 epochs) → `val.py` | Validation mAP improves vs. baseline recorded in IT-001 |
| **IT-003** | Weight conversion + evaluation | Convert official weights → load with `val.py` → check mAP parity vs. official results | mAP within ±0.5 of official reference |
| **IT-004** | Map parity checker | Run `tools/check_map_parity.py` after inference & validation | Report indicates parity across model sizes |

### 3.4 Reporting
- Capture console logs and generated artifacts.
- Store test run summaries under `reports/` for regression tracking.

## 4. Maintenance
- Update this plan when new models, tools, or datasets are added.
- Ensure `README.md` and `AGENTS.md` reference this SQA plan so contributors know it exists.
