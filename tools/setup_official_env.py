#!/usr/bin/env python3
"""Create an isolated virtual environment for the official THU-MIG.yoloe repo.

Why:
- Keep the official repository’s Python dependencies separate from LeanYOLO.
- Avoid accidental cross-imports when experimenting interactively.

What it does:
- Creates a venv at references/yolov10/THU-MIG.yoloe/.venv-official
- Upgrades pip, installs requirements.txt from the official repo

Torch install:
- This script does not install torch by default. Install a CPU or CUDA build
  explicitly to match your system if needed. Example (CUDA 12.1):
    <venv>/bin/python -m pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121

Usage:
  ./.venv/bin/python tools/setup_official_env.py \
    --repo references/yolov10/THU-MIG.yoloe

After setup, use:
  REPO=references/yolov10/THU-MIG.yoloe
  $REPO/.venv-official/bin/python -c "import ultralytics; print('OK')"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main(argv=None):
    p = argparse.ArgumentParser(description="Set up isolated venv for official THU-MIG.yoloe repo")
    p.add_argument("--repo", default="references/yolov10/THU-MIG.yoloe", help="Path to official repo root")
    p.add_argument("--python", default=sys.executable, help="Python interpreter to use for venv creation")
    args = p.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not repo.exists() or not (repo / "ultralytics").exists():
        print(f"[error] Repo not found at {repo}. Run tools/download_references.py first.")
        sys.exit(2)

    venv_dir = repo / ".venv-official"
    if not venv_dir.exists():
        run([args.python, "-m", "venv", str(venv_dir)])
    py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    # Upgrade pip
    run([str(py), "-m", "pip", "install", "-U", "pip"])

    # Install official requirements (lightweight; does not include torch)
    req = repo / "requirements.txt"
    if req.exists():
        run([str(py), "-m", "pip", "install", "-r", str(req)])
    else:
        print(f"[warn] requirements.txt not found under {repo}")

    print("[done] Official venv ready:", venv_dir)
    print("Install torch suitable for your system if needed. Example:")
    print(f"  {py} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")


if __name__ == "__main__":  # pragma: no cover
    main()

