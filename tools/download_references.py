#!/usr/bin/env python3
"""Download references (papers + official repos) and verify.

What it does:
- Downloads original TeX sources from arXiv when available (PDF fallback).
- Unpacks TeX tarballs and organizes everything as:
  ``references/<yolo_name>/<paper_id>/data``
- Clones official YOLOv10 repository into:
  ``references/yolov10/THU-MIG.yoloe``
- Verifies that the set of YOLO models matches the README "References" table.
- Verifies that TeX sources are extracted and PDFs/HTML are present when TeX is unavailable.
- Verifies the presence of the official YOLOv10 repository.

Usage:
    python tools/download_references.py [--out-dir references] [--verify-only]
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

import requests

# Mapping of YOLO version/name -> URLs for tex and pdf.
# tex may be ``None`` if no source tarball is provided.
# Mapping of normalized YOLO name -> URLs for tex and pdf.
PAPERS: Dict[str, Dict[str, Optional[str]]] = {
    "yolov1": {
        "tex": "https://arxiv.org/e-print/1506.02640",
        "pdf": "https://arxiv.org/pdf/1506.02640.pdf",
    },
    "yolov2": {
        "tex": "https://arxiv.org/e-print/1612.08242",
        "pdf": "https://arxiv.org/pdf/1612.08242.pdf",
    },
    "yolov3": {
        "tex": "https://arxiv.org/e-print/1804.02767",
        "pdf": "https://arxiv.org/pdf/1804.02767.pdf",
    },
    "yolov4": {
        "tex": "https://arxiv.org/e-print/2004.10934",
        "pdf": "https://arxiv.org/pdf/2004.10934.pdf",
    },
    "yolov5": {
        "tex": None,
        "pdf": None,
        "html": "https://docs.ultralytics.com/models/yolov5/",
    },
    "pp-yolo": {
        "tex": "https://arxiv.org/e-print/2007.12099",
        "pdf": "https://arxiv.org/pdf/2007.12099.pdf",
    },
    "yolov6": {
        "tex": "https://arxiv.org/e-print/2209.02976",
        "pdf": "https://arxiv.org/pdf/2209.02976.pdf",
    },
    "yolov7": {
        "tex": "https://arxiv.org/e-print/2207.02696",
        "pdf": "https://arxiv.org/pdf/2207.02696.pdf",
    },
    "damo-yolo": {
        "tex": "https://arxiv.org/e-print/2211.15444",
        "pdf": "https://arxiv.org/pdf/2211.15444.pdf",
    },
    "yolov8": {
        "tex": None,
        "pdf": None,
        "html": "https://docs.ultralytics.com/models/yolov8/",
    },
    "yolo-nas": {
        "tex": None,
        "pdf": None,
    },
    "yolo-world": {
        "tex": "https://arxiv.org/e-print/2401.17270",
        "pdf": "https://arxiv.org/pdf/2401.17270.pdf",
    },
    "yolov9": {
        "tex": "https://arxiv.org/e-print/2402.13616",
        "pdf": "https://arxiv.org/pdf/2402.13616.pdf",
    },
    "yolov10": {
        "tex": "https://arxiv.org/e-print/2405.14458",
        "pdf": "https://arxiv.org/pdf/2405.14458.pdf",
    },
    "yolov11": {
        "tex": None,
        "pdf": None,
        "html": "https://docs.ultralytics.com/models/yolov11/",
    },
    "yolov12": {
        "tex": "https://arxiv.org/e-print/2502.12524",
        "pdf": "https://arxiv.org/pdf/2502.12524.pdf",
    },
}

def _norm_name(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(" ", "-")
    return s


def _arxiv_id_from_url(url: str) -> Optional[str]:
    # Match arXiv IDs in e-print or pdf URL forms, e.g., 1506.02640
    m = re.search(r"/(?:e-print|pdf)/([0-9]{4}\.[0-9]{5})(?:\.pdf)?", url)
    return m.group(1) if m else None


def _dest_for(name: str, paper_id: Optional[str], out_dir: Path) -> Path:
    pid = paper_id if paper_id else "docs"
    return out_dir / name / pid / "data"


def _safe_extract_tar(fileobj: io.BufferedIOBase, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=fileobj, mode="r:*") as tf:
        for member in tf.getmembers():
            member_path = dest / member.name
            if not str(member_path.resolve()).startswith(str(dest.resolve())):
                continue
            tf.extract(member, dest)


def _extract_zip_bytes(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)


def download_file(url: str) -> Optional[bytes]:
    """Download ``url`` and return bytes on success; ``None`` otherwise."""
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.content
    except requests.RequestException:
        pass
    return None


def download_papers(out_dir: Path) -> None:
    """Download all YOLO papers into ``out_dir`` and organize outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, urls in PAPERS.items():
        name = _norm_name(name)
        tex_url = urls.get("tex")
        pdf_url = urls.get("pdf")

        # Prefer TeX sources when available
        if tex_url:
            data = download_file(tex_url)
            if data is not None:
                paper_id = _arxiv_id_from_url(tex_url)
                dest = _dest_for(name, paper_id, out_dir)
                try:
                    _safe_extract_tar(io.BytesIO(data), dest)
                    print(f"[{name}] TeX extracted -> {dest}")
                    continue
                except tarfile.TarError:
                    # fallthrough to try zip or pdf
                    pass
                try:
                    _extract_zip_bytes(data, dest)
                    print(f"[{name}] TeX (zip) extracted -> {dest}")
                    continue
                except zipfile.BadZipFile:
                    pass

        # Fallback to PDF
        if pdf_url:
            data = download_file(pdf_url)
            if data is not None:
                paper_id = _arxiv_id_from_url(pdf_url)
                dest = _dest_for(name, paper_id, out_dir)
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "paper.pdf").write_bytes(data)
                print(f"[{name}] PDF saved -> {dest / 'paper.pdf'}")
                continue

        # Fallback to HTML docs (save page for reference)
        html_url = urls.get("html") if isinstance(urls, dict) else None
        if html_url:
            data = download_file(html_url)
            if data is not None:
                dest = _dest_for(name, None, out_dir)
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "index.html").write_bytes(data)
                (dest / "SOURCE_URL.txt").write_text(html_url)
                print(f"[{name}] Docs saved -> {dest / 'index.html'}")
                continue

        # Ensure a placeholder exists for references with no public artifact
        dest = _dest_for(name, None, out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.txt").write_text(
            "No TeX/PDF available. See README References for source links.\n"
        )
        print(f"[{name}] no artifact; placeholder created at {dest}")


def parse_readme_models(readme_path: Path) -> List[str]:
    """Parse the README References table and return normalized model names."""
    txt = readme_path.read_text(encoding="utf-8", errors="ignore")
    block_re = re.compile(r"\n\| Model \|.*?\n(.*?)\n\n", re.S)
    m = block_re.search(txt)
    if not m:
        return []
    rows = [r.strip() for r in m.group(1).splitlines() if r.strip().startswith("|")]
    models: List[str] = []
    for r in rows:
        cols = [c.strip() for c in r.strip("|").split("|")]
        if not cols:
            continue
        # Skip markdown separator rows like |-----|----| ...
        if set(cols[0]) <= {"-"}:
            continue
        model = _norm_name(cols[0])  # first column: Model
        models.append(model)
    return models


def _clone_official_repos(out_dir: Path) -> None:
    """Clone official repo under references/yolov10/THU-MIG.yoloe only."""
    # THU-MIG/yoloe -> references/yolov10/THU-MIG.yoloe
    yoloe_dir = out_dir / "yolov10" / "THU-MIG.yoloe"
    if not yoloe_dir.exists():
        yoloe_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/THU-MIG/yoloe.git",
                str(yoloe_dir),
            ])
            print(f"[yolov10] related repo cloned -> {yoloe_dir}")
        except Exception as e:
            print(f"[yoloe] failed to clone repo: {e}")


def verify_against_readme(out_dir: Path, repo_root: Path) -> Tuple[bool, str]:
    """Verify downloads match README and structure.

    Returns (ok, report).
    """
    # Expected title tokens to validate TeX sources per model.
    EXPECTED_TOKENS: Dict[str, List[str]] = {
        "yolov1": ["you only look once", "real time object detection"],
        "yolov2": ["yolo9000", "better faster stronger"],
        "yolov3": ["yolov3", "incremental improvement"],
        "yolov4": ["yolov4", "optimal speed and accuracy"],
        "pp-yolo": ["pp yolo", "effective and efficient implementation"],
        "yolov6": ["yolov6", "single stage", "industrial applications"],
        "yolov7": ["yolov7", "trainable bag of freebies"],
        "damo-yolo": ["damo yolo", "real time object detection design"],
        "yolo-world": ["yolo world", "open vocabulary object detection"],
        "yolov9": ["yolov9", "programmable gradient information"],
        "yolov10": ["yolov10", "end to end object detection"],
        "yolov12": ["yolov12", "attention centric"],
        # Docs-only pages — validate presence of model keyword
        "yolov5": ["yolov5"],
        "yolov8": ["yolov8"],
        "yolov11": ["yolov11", "yolo11"],
    }
    readme_models = parse_readme_models(repo_root / "README.md")
    expected = set(readme_models)
    have = set(_norm_name(k) for k in PAPERS.keys())

    report_lines: List[str] = []
    ok = True

    # Compare sets of models
    missing_in_script = expected - have
    extra_in_script = have - expected
    if missing_in_script:
        ok = False
        report_lines.append(f"Missing in script: {sorted(missing_in_script)}")
    if extra_in_script:
        # Not a hard error but flag
        report_lines.append(f"Extra in script: {sorted(extra_in_script)}")

    # Verify per-model directory structure and content
    for model in sorted(have & expected):
        model_dir = out_dir / model
        if not model_dir.exists():
            ok = False
            report_lines.append(f"{model}: directory missing at {model_dir}")
            continue
        # Expect at least one paper-id directory
        subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if not subdirs:
            ok = False
            report_lines.append(f"{model}: no paper_id subdirectory found")
            continue
        for sub in subdirs:
            # Skip repository clones (identified by presence of a .git directory)
            if (sub / ".git").exists():
                continue
            data_dir = sub / "data"
            if not data_dir.exists():
                ok = False
                report_lines.append(f"{model}: {sub.name} missing data/ directory")
                continue
            # Check if TeX extracted or PDF/HTML present or placeholder exists
            tex_files = list(data_dir.rglob("*.tex"))
            pdf_files = list(data_dir.glob("paper.pdf"))
            html_files = list(data_dir.glob("index.html"))
            placeholder = (data_dir / "README.txt").exists()
            if not tex_files and not pdf_files and not html_files and not placeholder:
                ok = False
                report_lines.append(
                    f"{model}: {sub.name} has no .tex, no paper.pdf, and no index.html"
                )
                continue

            # If TeX exists, validate expected tokens appear
            if tex_files and model in EXPECTED_TOKENS:
                try:
                    def _norm_text(s: str) -> str:
                        s = s.lower()
                        s = re.sub(r"[^a-z0-9]+", " ", s)
                        return s

                    # Concatenate contents of tex files (bounded)
                    buf_parts: List[str] = []
                    total = 0
                    for tf in tex_files:
                        try:
                            txt = tf.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            txt = tf.read_bytes()[:200000].decode("utf-8", errors="ignore")
                        buf_parts.append(txt)
                        total += len(txt)
                        if total > 500000:  # cap to 500KB to avoid excess
                            break
                    corpus = _norm_text("\n".join(buf_parts))
                    tokens = [
                        _norm_text(tkn) for tkn in EXPECTED_TOKENS.get(model, [])
                    ]
                    missing = [t for t in tokens if t and t not in corpus]
                    if missing:
                        ok = False
                        report_lines.append(
                            f"{model}: expected tokens missing in TeX: {missing[:3]}"
                        )
                except Exception as e:
                    ok = False
                    report_lines.append(f"{model}: error reading TeX for validation: {e}")

            # If HTML docs exist, validate presence of model tokens in HTML
            if html_files and model in EXPECTED_TOKENS:
                try:
                    def _norm_text(s: str) -> str:
                        s = s.lower()
                        s = re.sub(r"[^a-z0-9]+", " ", s)
                        return s

                    parts: List[str] = []
                    for hf in html_files:
                        txt = hf.read_text(encoding="utf-8", errors="ignore")
                        parts.append(txt)
                    corpus = _norm_text("\n".join(parts))
                    tokens = [
                        _norm_text(tkn) for tkn in EXPECTED_TOKENS.get(model, [])
                    ]
                    found_any = any(t for t in tokens if t and t in corpus)
                    if not found_any:
                        ok = False
                        report_lines.append(
                            f"{model}: expected tokens not found in HTML"
                        )
                except Exception as e:
                    ok = False
                    report_lines.append(f"{model}: error reading HTML for validation: {e}")

    # Verify official YOLOv10 repo presence and a key file
    # Verify THU-MIG.yoloe presence (requested path)
    yoloe_repo = out_dir / "yolov10" / "THU-MIG.yoloe"
    if not yoloe_repo.exists():
        ok = False
        report_lines.append(f"THU-MIG.yoloe repo missing at {yoloe_repo}")
    # Sanity check: expected YOLOv10 YAML exists within THU-MIG.yoloe
    key_file = yoloe_repo / "ultralytics" / "cfg" / "models" / "v10" / "yolov10s.yaml"
    if not key_file.exists():
        ok = False
        report_lines.append(f"THU-MIG.yoloe missing expected YAML at {key_file}")

    if not report_lines:
        report_lines.append("All good: README models matched and files verified.")

    return ok, "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO papers and verify")
    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "references"
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_dir,
        help="Directory to store papers (default: references/)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification without downloading",
    )
    args = parser.parse_args()

    if not args.verify_only:
        download_papers(args.out_dir)
        _clone_official_repos(args.out_dir)
    ok, report = verify_against_readme(args.out_dir, repo_root)
    print("\nVerification summary:\n" + report)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
