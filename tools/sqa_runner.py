#!/usr/bin/env python3
"""
SQA runner for lean-YOLO: runs one SQA test per Codex CLI invocation, with
concurrency, structured logging, and rollup reporting.

Usage examples:
  - List all discovered tests from sqa.md:
      ./\.venv/bin/python tools/sqa_runner.py list --plan-file sqa.md

  - Run all tests with default concurrency (2):
      ./\.venv/bin/python tools/sqa_runner.py run --plan main

  - Run only unit-test series in parallel (4 workers):
      ./\.venv/bin/python tools/sqa_runner.py run --plan main --tests UT-* --concurrency 4

  - Dry run to see the commands but not execute Codex:
      ./\.venv/bin/python tools/sqa_runner.py run --plan main --dry-run --tee

  - Reset (delete) run artifacts for a plan:
      ./\.venv/bin/python tools/sqa_runner.py reset --plan main --yes

Design notes:
  - Reads test matrix from a markdown table in sqa.md with columns:
    ID | Test Point | Steps | Expected Result
  - Builds two messages per test: the concrete steps to run, and an instruction to
    read sqa.md for compliance.
  - Runs an external Codex CLI command template per test with placeholders filled.
  - Only stdlib is used; unit tests cover parsing and status detection.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import fnmatch
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclasses.dataclass
class TestCase:
    test_id: str
    test_point: str
    steps: str
    expected: str


TABLE_HEADER_RE = re.compile(
    r"^\|\s*ID\s*\|\s*(?:Test\s*Point|Utility|Scenario)\s*\|\s*Steps\s*\|\s*Expected\s*Result\s*\|\s*$",
    re.IGNORECASE,
)
TABLE_SEP_RE = re.compile(r"^\|\s*-+\s*\|")
TEST_ID_RE = re.compile(r"([A-Z]{2}-\d{3})")


def _strip_md(s: str) -> str:
    """Remove common markdown wrappers like **bold**, `code`, and trim whitespace."""
    s = s.strip()
    # Remove leading/trailing pipes if present due to sloppy input
    s = s.strip("|")
    # Remove bold markers and inline code markers
    s = s.replace("**", "")
    s = s.replace("`", "")
    return s.strip()


def parse_sqa_md(path: Path) -> List[TestCase]:
    """Parse sqa.md and return a list of TestCase parsed from tables.

    Assumes tables include header: ID | Test Point | Steps | Expected Result
    and multiple such tables can appear in the document.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    cases: List[TestCase] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if TABLE_HEADER_RE.match(line):
            # Expect a separator line next
            i += 1
            if i >= len(lines) or not TABLE_SEP_RE.search(lines[i]):
                # Not a valid table; skip
                continue
            i += 1
            # Consume table rows until a blank line or non-table line
            while i < len(lines):
                row = lines[i]
                if not row.strip() or not row.strip().startswith("|"):
                    break
                # Split by pipes; keep things simple as plan text likely safe
                parts = [p.strip() for p in row.strip().split("|")][1:-1]
                if len(parts) < 4:
                    i += 1
                    continue
                raw_id, test_point, steps, expected = parts[:4]
                # Clean markdown wrappers
                tid = _strip_md(raw_id)
                m = TEST_ID_RE.search(tid)
                if not m:
                    i += 1
                    continue
                test_id = m.group(1)
                cases.append(
                    TestCase(
                        test_id=test_id,
                        test_point=_strip_md(test_point),
                        steps=_strip_md(steps),
                        expected=_strip_md(expected),
                    )
                )
                i += 1
            # Do not skip i increment here; loop continues
        else:
            i += 1
    return cases


def filter_tests(cases: Sequence[TestCase], patterns: Optional[Sequence[str]]) -> List[TestCase]:
    """Filter cases by simple glob patterns on test_id; if None, return all."""
    if not patterns:
        return list(cases)
    selected: List[TestCase] = []
    for c in cases:
        for pat in patterns:
            if fnmatch.fnmatchcase(c.test_id, pat):
                selected.append(c)
                break
    # Preserve order of original cases; remove duplicates
    uniq: List[TestCase] = []
    seen = set()
    for c in selected:
        if c.test_id not in seen:
            uniq.append(c)
            seen.add(c.test_id)
    return uniq


def build_prompts(case: TestCase, sqa_md_path: Path) -> Tuple[str, str]:
    test_msg = (
        f"Run SQA test {case.test_id} – {case.test_point}. "
        f"Steps: {case.steps}. "
        "Run commands from the repository root. "
        "Save outputs in place and summarize the result clearly. "
        "When done, print a final line: TEST STATUS: PASSED or FAILED."
    )
    read_msg = (
        f"Read {sqa_md_path}. Confirm steps and expected result align for {case.test_id} "
        f"and note any deviation. Expected Result: {case.expected}."
    )
    return test_msg, read_msg


def determine_status(stdout_text: str, exit_code: int, success_regex: Optional[str] = None) -> str:
    """Infer status from stdout and exit code.

    Returns one of: PASSED, FAILED, ERROR, UNKNOWN
    """
    # Explicit directive takes precedence
    matches = re.findall(r"TEST STATUS:\s*(PASSED|FAILED)", stdout_text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    if success_regex:
        if re.search(success_regex, stdout_text, re.IGNORECASE | re.MULTILINE):
            return "PASSED"
        return "FAILED" if exit_code != 0 else "UNKNOWN"

    # pytest-style hints
    if re.search(r"(\d+)\s+passed", stdout_text) and not re.search(r"\bfailed\b", stdout_text):
        if exit_code == 0:
            return "PASSED"

    if exit_code != 0:
        # Could be failure or error; we conservatively call FAILED
        return "FAILED"

    # No explicit indicators
    return "UNKNOWN"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


async def _stream_pipe(reader: asyncio.StreamReader, sink, tee: bool, prefix: str = ""):
    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            text = line.decode(errors="replace")
        except Exception:
            text = str(line)
        sink.write(text)
        sink.flush()
        if tee:
            out = sys.stdout if prefix == "STDOUT" else sys.stderr
            out.write(text)
            out.flush()


def _rm_tree(path: Path) -> None:
    if not path.exists():
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            Path(root, name).unlink(missing_ok=True)
        for name in dirs:
            Path(root, name).rmdir()
    path.rmdir()


async def run_one_test(
    plan: str,
    case: TestCase,
    cmd_template: str,
    sqa_md_path: Path,
    run_root: Path,
    timeout: int,
    tee: bool,
    success_regex: Optional[str] = None,
) -> dict:
    test_dir = run_root / case.test_id
    # Start clean if re-running
    if test_dir.exists():
        _rm_tree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    test_msg, read_msg = build_prompts(case, sqa_md_path)
    # Write prompt and command template
    (test_dir / "prompt.txt").write_text(
        f"TEST MESSAGE:\n{test_msg}\n\nREAD MESSAGE:\n{read_msg}\n", encoding="utf-8"
    )

    # Fill template
    combined_msg = f"{test_msg}\n\n{read_msg}"
    template_ctx = {
        "test": test_msg,
        "read": read_msg,
        "combined": combined_msg,
        "test_q": shlex.quote(test_msg),
        "read_q": shlex.quote(read_msg),
        "combined_q": shlex.quote(combined_msg),
        "plan_id": plan,
        "test_id": case.test_id,
        "sqa_md_path": str(sqa_md_path),
        "plan_dir": str(sqa_md_path.parent),
    }
    try:
        cmd_str = cmd_template.format(**template_ctx)
    except KeyError as e:
        raise SystemExit(f"Unknown placeholder in --cmd template: {e}")

    (test_dir / "cmd.txt").write_text(cmd_str + "\n", encoding="utf-8")

    # Launch subprocess
    start = _now_iso()
    proc = await asyncio.create_subprocess_shell(
        cmd_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path.cwd()),
    )

    stdout_path = test_dir / "stdout.log"
    stderr_path = test_dir / "stderr.log"

    with stdout_path.open("w", encoding="utf-8", buffering=1) as out:
        with stderr_path.open("w", encoding="utf-8", buffering=1) as err:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        _stream_pipe(proc.stdout, out, tee, prefix="STDOUT"),
                        _stream_pipe(proc.stderr, err, tee, prefix="STDERR"),
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                end = _now_iso()
                meta = {
                    "test_id": case.test_id,
                    "start": start,
                    "end": end,
                    "duration_sec": None,
                    "exit_code": None,
                    "timeout": True,
                    "status": "ERROR",
                }
                (test_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                (test_dir / "status.txt").write_text("ERROR\n", encoding="utf-8")
                return meta

    exit_code = await proc.wait()
    end = _now_iso()
    # Read stdout back for status determination
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    status = determine_status(stdout_text, exit_code, success_regex)

    meta = {
        "test_id": case.test_id,
        "start": start,
        "end": end,
        "duration_sec": None,  # duration can be computed from timestamps if needed
        "exit_code": exit_code,
        "timeout": False,
        "status": status,
    }
    (test_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (test_dir / "status.txt").write_text(f"{status}\n", encoding="utf-8")
    return meta


async def run_tests(
    plan: str,
    cases: Sequence[TestCase],
    cmd_template: str,
    sqa_md_path: Path,
    outdir: Path,
    concurrency: int,
    timeout: int,
    tee: bool,
    success_regex: Optional[str],
    dry_run: bool,
) -> Tuple[List[dict], Path]:
    # Use stable plan directory; test leaf directories hold results
    run_root = outdir / plan
    run_root.mkdir(parents=True, exist_ok=True)
    # Record sqa.md path for traceability
    (run_root / "sqa_md_path.txt").write_text(str(sqa_md_path) + "\n", encoding="utf-8")

    if dry_run:
        for c in cases:
            test_msg, read_msg = build_prompts(c, sqa_md_path)
            combined_msg = f"{test_msg}\n\n{read_msg}"
            template_ctx = {
                "test": test_msg,
                "read": read_msg,
                "combined": combined_msg,
                "test_q": shlex.quote(test_msg),
                "read_q": shlex.quote(read_msg),
                "combined_q": shlex.quote(combined_msg),
                "plan_id": plan,
                "test_id": c.test_id,
                "sqa_md_path": str(sqa_md_path),
                "plan_dir": str(sqa_md_path.parent),
            }
            cmd_str = cmd_template.format(**template_ctx)
            print(f"[DRY-RUN] {c.test_id}: {cmd_str}")
        return [], run_root

    sem = asyncio.Semaphore(concurrency)
    results: List[dict] = []
    report_lock = asyncio.Lock()

    async def _task(c: TestCase):
        async with sem:
            meta = await run_one_test(
                plan=plan,
                case=c,
                cmd_template=cmd_template,
                sqa_md_path=sqa_md_path,
                run_root=run_root,
                timeout=timeout,
                tee=tee,
                success_regex=success_regex,
            )
            results.append(meta)
            # Update report incrementally so partial results persist if interrupted
            async with report_lock:
                try:
                    write_report(results, run_root, plan, cmd_template)
                except Exception:
                    # Best-effort; continue even if report writing fails
                    pass

    await asyncio.gather(*[_task(c) for c in cases])
    return results, run_root


def write_report(results: Sequence[dict], run_root: Path, plan: str, cmd_template: str):
    status_counts = {"PASSED": 0, "FAILED": 0, "ERROR": 0, "UNKNOWN": 0}
    for r in results:
        status_counts[r.get("status", "UNKNOWN")] = status_counts.get(r.get("status", "UNKNOWN"), 0) + 1

    report_json = {
        "plan": plan,
        "run_dir": str(run_root),
        "generated_at": _now_iso(),
        "command_template": cmd_template,
        "total": len(results),
        "by_status": status_counts,
        "tests": results,
    }
    (run_root / "report.json").write_text(json.dumps(report_json, indent=2) + "\n", encoding="utf-8")

    # Markdown summary
    lines = []
    lines.append(f"# SQA Report for plan `{plan}`")
    lines.append("")
    lines.append(f"Run dir: `{run_root}`  ")
    lines.append(f"Generated: {report_json['generated_at']}")
    lines.append("")
    lines.append("| Test ID | Status | Exit Code | Meta Path |")
    lines.append("|---------|--------|-----------|-----------|")
    for r in results:
        tid = r.get("test_id")
        status = r.get("status")
        exit_code = r.get("exit_code")
        # meta.json lives at runs/sqa/<plan>/<timestamp>/<test_id>/meta.json
        # Derive relative path from run_root
        # Locate meta.json by searching directories
        meta_rel = f"{tid}/meta.json"
        lines.append(f"| {tid} | {status} | {exit_code} | {meta_rel} |")
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_list(args: argparse.Namespace) -> int:
    path = Path(args.plan_file)
    if not path.exists():
        print(f"sqa.md not found at: {path}", file=sys.stderr)
        return 2
    cases = parse_sqa_md(path)
    for c in cases:
        print(f"{c.test_id}\t{c.test_point}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    base = Path(args.outdir) / args.plan
    if not base.exists():
        print(f"No runs found for plan: {args.plan}")
        return 0
    if not args.yes:
        reply = input(f"Delete all runs under {base}? [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1
    # delete directory tree safely within workspace
    for root, dirs, files in os.walk(base, topdown=False):
        for name in files:
            Path(root, name).unlink(missing_ok=True)
        for name in dirs:
            Path(root, name).rmdir()
    base.rmdir()
    print(f"Deleted: {base}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan = args.plan
    sqa_md_path = Path(args.plan_file)
    if not sqa_md_path.exists():
        print(f"sqa.md not found at: {sqa_md_path}", file=sys.stderr)
        return 2

    cases = parse_sqa_md(sqa_md_path)
    if not cases:
        print("No tests discovered in sqa.md. Ensure it has the expected table headers.", file=sys.stderr)
        return 2

    patterns = []
    if args.tests:
        for part in args.tests.split(","):
            part = part.strip()
            if part:
                patterns.append(part)
    selected = filter_tests(cases, patterns or None)
    if not selected:
        print("No tests match the given filters.", file=sys.stderr)
        return 2

    # Prepare output dir
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run
    results: List[dict] = []
    run_root: Path
    try:
        results, run_root = asyncio.run(
            run_tests(
                plan=plan,
                cases=selected,
                cmd_template=args.cmd,
                sqa_md_path=sqa_md_path,
                outdir=outdir,
                concurrency=args.concurrency,
                timeout=args.timeout,
                tee=args.tee,
                success_regex=args.success_regex,
                dry_run=args.dry_run,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130

    if not args.dry_run:
        write_report(results, run_root, plan, args.cmd)
        # Non-zero exit if any failure/error
        bad = [r for r in results if r.get("status") in {"FAILED", "ERROR"}]
        return 1 if bad else 0
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SQA tests via Codex CLI with concurrency and logging.")
    sub = p.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List discovered test IDs from sqa.md")
    p_list.add_argument("--plan-file", default="sqa.md", help="Path to sqa.md file")
    p_list.set_defaults(func=cmd_list)

    # reset
    p_reset = sub.add_parser("reset", help="Delete runs for a plan under runs/sqa/<plan>")
    p_reset.add_argument("--plan", required=True, help="Plan label; used as output folder name")
    p_reset.add_argument("--outdir", default="runs/sqa", help="Base output directory")
    p_reset.add_argument("--yes", action="store_true", help="Do not prompt for confirmation")
    p_reset.set_defaults(func=cmd_reset)

    # run
    p_run = sub.add_parser("run", help="Run tests from sqa.md via Codex CLI")
    p_run.add_argument("--plan", required=True, help="Required plan label for outputs")
    p_run.add_argument("--tests", default=None, help="Comma-separated test IDs or globs (e.g., UT-001,UT-002 or UT-*)")
    p_run.add_argument("--concurrency", type=int, default=2, help="Number of concurrent Codex commands")
    p_run.add_argument("--plan-file", default="sqa.md", help="Path to sqa.md file")
    p_run.add_argument(
        "--cmd",
        # Fully open by default as requested: no sandbox or approvals
        default='codex --dangerously-bypass-approvals-and-sandbox exec -C . {combined_q}',
        help=(
            "Command template with placeholders: {test}, {read}, {combined}, "
            "{test_q}, {read_q}, {combined_q}, {plan_id}, {test_id}, {sqa_md_path}, {plan_dir}"
        ),
    )
    p_run.add_argument("--timeout", type=int, default=1800, help="Per-test timeout in seconds")
    p_run.add_argument("--outdir", default="runs/sqa", help="Base output directory")
    p_run.add_argument("--success-regex", default=None, help="Regex to detect success in stdout")
    p_run.add_argument("--dry-run", action="store_true", help="Show commands but do not execute Codex")
    p_run.add_argument("--tee", action="store_true", help="Echo subprocess output to console while logging")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
