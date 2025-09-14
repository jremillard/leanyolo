#!/usr/bin/env python3
"""
Lightweight import resolvability checker for the repo.

Scans Python files for top-level import statements and verifies that
`importlib.util.find_spec(mod)` returns a spec without importing heavy modules.

Usage:
  ./.venv/bin/python tools/check_imports.py
  ./.venv/bin/python tools/check_imports.py --paths leanyolo tools

Exits with code 0 on success (all resolvable), 1 otherwise. Prints "OK" or a
compact list of missing module names.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
import importlib.util as import_util
from typing import Iterable, Set


def discover_py_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for child in p.rglob("*.py"):
                # Skip common noise
                if (
                    "__pycache__" in child.parts
                    or ".venv" in child.parts
                    or "tests" in child.parts
                    or "references" in child.parts
                ):
                    continue
                yield child


class ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.modules: Set[str] = set()

    def visit_If(self, node: ast.If) -> None:
        # Skip TYPE_CHECKING blocks
        if _contains_type_checking(node.test):
            for n in node.orelse:
                self.visit(n)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name and not name.startswith(":"):
                self.modules.add(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Skip relative imports
        if node.level and node.level > 0:
            return
        if node.module:
            name = node.module.split(".")[0]
            if name:
                self.modules.add(name)


def _contains_type_checking(expr: ast.AST) -> bool:
    # Matches "if TYPE_CHECKING:" or "if typing.TYPE_CHECKING:"
    if isinstance(expr, ast.Name) and expr.id == "TYPE_CHECKING":
        return True
    if isinstance(expr, ast.Attribute) and getattr(expr.value, "id", None) == "typing" and expr.attr == "TYPE_CHECKING":
        return True
    if isinstance(expr, ast.BoolOp):
        return any(_contains_type_checking(v) for v in expr.values)
    if isinstance(expr, ast.UnaryOp):
        return _contains_type_checking(expr.operand)
    if isinstance(expr, ast.BinOp):
        return _contains_type_checking(expr.left) or _contains_type_checking(expr.right)
    if isinstance(expr, ast.Compare):
        return _contains_type_checking(expr.left) or any(_contains_type_checking(c) for c in expr.comparators)
    return False


def collect_modules(paths: Iterable[Path]) -> Set[str]:
    mods: Set[str] = set()
    for file in discover_py_files(paths):
        try:
            text = file.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception:
            continue
        ic = ImportCollector()
        ic.visit(tree)
        mods |= ic.modules
    return mods


def check_resolvable(mods: Iterable[str]) -> Set[str]:
    missing: Set[str] = set()
    for m in sorted(set(mods)):
        try:
            spec = import_util.find_spec(m)
        except Exception:
            spec = None
        if spec is None:
            missing.add(m)
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check import resolvability in the repo")
    ap.add_argument("--paths", nargs="*", default=["leanyolo", "tools"], help="Paths to scan for imports")
    args = ap.parse_args(argv)

    # Ensure repo root is importable (when invoked as tools/check_imports.py)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    roots = [Path(p) for p in args.paths]
    mods = collect_modules(roots)
    # Always include the top-level package if present
    mods.add("leanyolo")
    missing = check_resolvable(mods)

    if not missing:
        print("OK")
        return 0
    print("Missing:", sorted(missing))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
