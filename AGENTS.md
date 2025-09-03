# Repository Guidelines

## Project Structure & Module Organization
- Code lives under `leanyolo/` (modules are WIP but follow this layout):
```
leanyolo/
  leanyolo/
    models/yolov10/{backbone.py, neck.py, head.py, model.py}
    data/{dataset.py, transforms.py, collate.py}
    utils/{metrics.py, box_ops.py, losses.py, viz.py}
    tests/{test_*.py, fidelity/}
  tools/
    train.py  (baseline trainer)
    transfer_learn_aquarium.py  (example transfer learning script)
    prepare_coco.py  prepare_acquirium.py  ...
  runs/  (outputs: logs, checkpoints, visualizations)
  references/  (papers + official repos)
  requirements.txt
```
- YOLOv10-only: backbone, neck, head, and end-to-end model.
- No YAML configs; prefer Python constructors and CLI flags.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv` (optional activate: `source .venv/bin/activate`).
- Install PyTorch (CUDA by default): `./.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`.
- Install deps: `./.venv/bin/python -m pip install -r requirements.txt`.
- Quick checks: `./.venv/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"`.
- Run tests: `./.venv/bin/python -m pytest -q`; unit-only: `./.venv/bin/python -m pytest -q -m "not fidelity"`; fidelity: `./.venv/bin/python -m pytest -q -m fidelity`.

## Coding Style & Naming Conventions
- Python 3.9+; 4-space indent; PEP 8; type hints encouraged.
- Modules: `lower_snake.py`; classes: `CapWords`; functions/vars: `lower_snake`; constants: `UPPER_SNAKE`.
- Prefer explicit tensor shapes/dtypes in docstrings and comments.
- Keep APIs PyTorch-native; avoid hidden magic and side effects.

## Testing Guidelines
- Framework: `pytest`. Place unit tests in `leanyolo/tests/` as `test_*.py`.
- Use marker `fidelity` for parity tests; run unit-only with `-m "not fidelity"`.
- For deterministic snapshots, force CPU: `CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m pytest -q`.
- Weights/config: set `LEANYOLO_WEIGHTS_DIR` (offline weights) or ensure `LEANYOLO_CACHE_DIR` is writable.
- When touching backbone/neck/head/model, add/adjust tests and verify weight loading and a dry forward.

## Commit & Pull Request Guidelines
- Small, focused checkins with rationale; update docs when behavior changes.
- Include tests for affected modules and show weight-loading logs (missing/unexpected keys reviewed).
- Commit style: concise imperative subject; Conventional Commits (feat/fix/docs/refactor/test) recommended.
- checkin must pass CI and note any intentional behavior differences.
- Do not commit datasets or weights; keep large binaries out of git.

## Security & Configuration Tips
- Use a single venv: `.venv`; never rely on system Python.
- Official repos live under `references/` (managed by scripts); do not vendor as top-level folders.
- Avoid network in tests unless explicitly cached; prefer CPU for parity.
- Document exact commands/seeds when regenerating references.

## Docs & References Verification
- When `README.md` is changed (especially the References table), run:
- Verify only: `./.venv/bin/python tools/download_references.py --verify-only`

## References Directory
- Path: `references/`
- Papers: `references/<yolo_version>/<paper_id>/data` with TeX/PDF/HTML saved and TeX extracted.
- Official repo: `references/yolov10/THU-MIG.yoloe` (cloned from https://github.com/THU-MIG/yoloe)
- Re-generate or verify: `./.venv/bin/python tools/download_references.py` or `--verify-only`.
