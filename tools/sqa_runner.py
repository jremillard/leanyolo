#!/usr/bin/env python3
"""
SQA runner for lean-YOLO: runs one SQA test per Codex CLI invocation,
sequentially, with structured logging and rollup reporting.

Usage examples:
  - List all discovered tests from sqa.yaml:
      ./.venv/bin/python tools/sqa_runner.py list --plan-file sqa.yaml

  - Run all tests sequentially:
      ./.venv/bin/python tools/sqa_runner.py run --plan main

  - Filter tests by ID glob:
      ./.venv/bin/python tools/sqa_runner.py run --plan main --tests UT-*

  - Re-run only failed/missing from last run dir:
      ./.venv/bin/python tools/sqa_runner.py run --plan main --failed-missing

  - Dry run to see the commands but not execute Codex:
      ./.venv/bin/python tools/sqa_runner.py run --plan main --dry-run --tee

  - Reset (delete) run artifacts for a plan:
      ./.venv/bin/python tools/sqa_runner.py reset --plan main --yes

Design notes:
  - Reads test matrix from sqa.yaml with schema: { version, tests: [ {id, name, steps[], expected} ] }
  - Builds two messages per test: the concrete steps to run, and an instruction to
    read sqa.yaml for compliance.
  - Runs an external Codex CLI command template per test with placeholders filled.
  - Only stdlib plus PyYAML (safe_load) is used; unit tests cover parsing and status detection.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None


# Fixed Codex exec command template (no profiles/config overrides allowed)
CODEX_CMD_TEMPLATE = (
    "codex exec "
    "--model gpt-5 "
    "--sandbox workspace-write "
    # exec subcommand always runs with approval_policy=never; flag is not accepted
    "-c model_reasoning_effort=\"high\" "
    "-c model_reasoning_summary=\"detailed\" "
    "-c model_verbosity=\"high\" "
    "-c preferred_auth_method=\"chatgpt\" "
    # Make child processes non-interactive and avoid pagers/prompts
    "-c 'shell_environment_policy.set={{ CI = \"1\", "
    "GIT_TERMINAL_PROMPT = \"0\", "
    "PIP_NO_INPUT = \"1\", "
    "PIP_DISABLE_PIP_VERSION_CHECK = \"1\", "
    "DEBIAN_FRONTEND = \"noninteractive\", "
    "APT_LISTCHANGES_FRONTEND = \"none\", "
    "PAGER = \"cat\", GIT_PAGER = \"cat\", PYTHONPAGER = \"cat\", "
    "GIT_EDITOR = \"true\", EDITOR = \"true\", VISUAL = \"true\" }}' "
    "-c sandbox_workspace_write.network_access=true "
    "-C . {combined_q}"
)


@dataclasses.dataclass
class TestCase:
    test_id: str
    name: str
    steps: List[str]
    expected: str


TEST_ID_RE = re.compile(r"^[A-Z]{2}-\d{3}$")


def _strip_md(s: str) -> str:
    """Remove common markdown wrappers like **bold**, `code`, and trim whitespace."""
    s = s.strip()
    # Remove leading/trailing pipes if present due to sloppy input
    s = s.strip("|")
    # Remove bold markers and inline code markers
    s = s.replace("**", "")
    s = s.replace("`", "")
    return s.strip()


def parse_sqa_yaml(path: Path) -> List[TestCase]:
    if yaml is None:
        raise SystemExit("PyYAML is required to parse sqa.yaml. Please install pyyaml.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tests = data.get("tests", []) if isinstance(data, dict) else []
    cases: List[TestCase] = []
    for t in tests:
        tid = t.get("id")
        name = t.get("name", "")
        steps = t.get("steps", [])
        expected = t.get("expected", "")
        if not tid or not TEST_ID_RE.match(tid):
            continue
        if not isinstance(steps, list):
            # Allow a single string; convert to list
            steps = [str(steps)]
        cases.append(TestCase(test_id=tid, name=str(name), steps=[str(s) for s in steps], expected=str(expected)))
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


def build_prompts(case: TestCase, sqa_plan_path: Path) -> Tuple[str, str]:
    steps_str = " -> ".join(case.steps)
    test_msg = (
        f"Run SQA test {case.test_id} – {case.name}. "
        f"Steps: {steps_str}. "
        "Run commands from the repository root. "
        "Run non-interactively (no prompts, no pagers); if a command may prompt, add the appropriate non-interactive flag and/or redirect stdin from /dev/null. "
        "Save outputs in place and summarize the result clearly. "
        "When done, print a final line: TEST STATUS: PASSED or FAILED."
    )
    read_msg = (
        f"Read {sqa_plan_path}. Confirm steps and expected result align for {case.test_id} "
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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


async def _stream_pipe(reader: asyncio.StreamReader, sink, tee: bool, prefix: str = ""):
    """Stream data from reader to sink without line-size limits.

    Uses fixed-size chunks to avoid LimitOverrunError from very long lines.
    """
    while True:
        try:
            chunk = await reader.read(4096)
        except Exception:
            # In case of unexpected reader failure, stop gracefully
            break
        if not chunk:
            break
        try:
            text = chunk.decode(errors="replace")
        except Exception:
            text = str(chunk)
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


def _merge_move_dir(src: Path, dst: Path) -> None:
    """Merge contents of src into dst and remove src.

    Existing files are left intact; only missing files/dirs are moved.
    """
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root, name)
            d = target_root / name
            if not d.exists():
                try:
                    s.replace(d)
                except Exception:
                    # If move fails (e.g., cross-device), fallback to copy then unlink
                    try:
                        data = s.read_bytes()
                        d.write_bytes(data)
                        s.unlink(missing_ok=True)
                    except Exception:
                        pass
    # Attempt to remove the emptied source tree
    try:
        _rm_tree(src)
    except Exception:
        pass


async def run_one_test(
    plan: str,
    case: TestCase,
    cmd_template: str,
    sqa_plan_path: Path,
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

    test_msg, read_msg = build_prompts(case, sqa_plan_path)
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
        "sqa_plan_path": str(sqa_plan_path),
        "plan_dir": str(sqa_plan_path.parent),
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
    sqa_plan_path: Path,
    outdir: Path,
    timeout: int,
    tee: bool,
    success_regex: Optional[str],
    dry_run: bool,
) -> Tuple[List[dict], Path]:
    # Use stable plan directory; test leaf directories hold results
    run_root = outdir / plan
    run_root.mkdir(parents=True, exist_ok=True)
    # Record sqa.md path for traceability
    (run_root / "sqa_plan_path.txt").write_text(str(sqa_plan_path) + "\n", encoding="utf-8")

    if dry_run:
        total = len(cases)
        for i, c in enumerate(cases, start=1):
            print(f"[{i}/{total}] DRY-RUN {c.test_id} – {c.name}")
            test_msg, read_msg = build_prompts(c, sqa_plan_path)
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
                "sqa_plan_path": str(sqa_plan_path),
                "plan_dir": str(sqa_plan_path.parent),
            }
            cmd_str = cmd_template.format(**template_ctx)
            print(f"[DRY-RUN] {c.test_id}: {cmd_str}")
        return [], run_root

    results: List[dict] = []

    # Best-effort migration of any legacy top-level test directories before running
    try:
        for entry in outdir.iterdir():
            if entry.is_dir() and TEST_ID_RE.match(entry.name):
                dest = run_root / entry.name
                if dest.exists():
                    _merge_move_dir(entry, dest)
                else:
                    try:
                        entry.rename(dest)
                    except Exception:
                        _merge_move_dir(entry, dest)
    except FileNotFoundError:
        pass

    # Run tests sequentially, updating the report after each test
    total = len(cases)
    for i, c in enumerate(cases, start=1):
        print(f"[{i}/{total}] START {c.test_id} – {c.name}")
        t0 = time.perf_counter()
        meta = await run_one_test(
            plan=plan,
            case=c,
            cmd_template=cmd_template,
            sqa_plan_path=sqa_plan_path,
            run_root=run_root,
            timeout=timeout,
            tee=tee,
            success_regex=success_regex,
        )
        results.append(meta)
        dt = time.perf_counter() - t0
        test_dir = run_root / c.test_id
        print(
            f"[{i}/{total}] END   {c.test_id} -> {meta.get('status')} "
            f"(exit={meta.get('exit_code')}, {dt:.1f}s). Artifacts: {test_dir}"
        )
        # Re-check and migrate any stray test dirs that may have been created at outdir
        try:
            for entry in outdir.iterdir():
                if entry.is_dir() and TEST_ID_RE.match(entry.name):
                    dest = run_root / entry.name
                    if dest.exists():
                        _merge_move_dir(entry, dest)
                    else:
                        try:
                            entry.rename(dest)
                        except Exception:
                            _merge_move_dir(entry, dest)
        except FileNotFoundError:
            pass
        try:
            write_report(results, run_root, plan, cmd_template)
        except Exception:
            # Best-effort; continue even if report writing fails
            pass

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
        # meta.json lives at runs/sqa/<plan>/<test_id>/meta.json
        # Derive relative path from run_root
        meta_rel = f"{tid}/meta.json"
        lines.append(f"| {tid} | {status} | {exit_code} | {meta_rel} |")
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_list(args: argparse.Namespace) -> int:
    path = Path(args.plan_file)
    if not path.exists():
        print(f"sqa.yaml not found at: {path}", file=sys.stderr)
        return 2
    cases = parse_sqa_yaml(path)
    for c in cases:
        print(f"{c.test_id}\t{c.name}")
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
    sqa_plan_path = Path(args.plan_file)
    if not sqa_plan_path.exists():
        print(f"sqa.yaml not found at: {sqa_plan_path}", file=sys.stderr)
        return 2

    cases = parse_sqa_yaml(sqa_plan_path)
    if not cases:
        print("No tests discovered in sqa.yaml.", file=sys.stderr)
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
    print(f"Plan '{plan}': {len(selected)} test(s) selected.")

    # Prepare output dir
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure plan scoped directory exists and migrate any legacy outputs that may
    # have been written at the root (runs/sqa/<TEST_ID>) by older tools.
    run_root_planned = outdir / plan
    run_root_planned.mkdir(parents=True, exist_ok=True)
    try:
        for entry in outdir.iterdir():
            # Detect legacy test directories like UT-101, FT-003, etc.
            if entry.is_dir() and TEST_ID_RE.match(entry.name):
                dest = run_root_planned / entry.name
                if dest.exists():
                    _merge_move_dir(entry, dest)
                else:
                    try:
                        entry.rename(dest)
                    except Exception:
                        # Best-effort migration; continue if rename fails; fallback to merge
                        _merge_move_dir(entry, dest)
    except FileNotFoundError:
        pass

    # Optionally restrict to failed or missing tests in the previous run dir
    if args.failed_missing:
        run_root_prev = outdir / plan
        want_ids = {c.test_id for c in selected}
        filtered: List[TestCase] = []
        for c in selected:
            status_path = run_root_prev / c.test_id / "status.txt"
            if not status_path.exists():
                filtered.append(c)
                continue
            try:
                status = status_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                status = ""
            if status != "PASSED":
                filtered.append(c)
        selected = filtered
        if not selected:
            print(f"No failed or missing tests to run under {run_root_prev}.")
            return 0

    # Run
    results: List[dict] = []
    run_root: Path
    try:
        results, run_root = asyncio.run(
            run_tests(
                plan=plan,
                cases=selected,
                cmd_template=CODEX_CMD_TEMPLATE,
                sqa_plan_path=sqa_plan_path,
                outdir=outdir,
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
        write_report(results, run_root, plan, CODEX_CMD_TEMPLATE)
        # Non-zero exit if any failure/error
        bad = [r for r in results if r.get("status") in {"FAILED", "ERROR"}]
        return 1 if bad else 0
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SQA tests via Codex CLI sequentially with structured logging.")
    sub = p.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List discovered test IDs from sqa.yaml")
    p_list.add_argument("--plan-file", default="sqa.yaml", help="Path to sqa.yaml file")
    p_list.set_defaults(func=cmd_list)

    # reset
    p_reset = sub.add_parser("reset", help="Delete runs for a plan under runs/sqa/<plan>")
    p_reset.add_argument("--plan", required=True, help="Plan label; used as output folder name")
    p_reset.add_argument("--outdir", default="runs/sqa", help="Base output directory")
    p_reset.add_argument("--yes", action="store_true", help="Do not prompt for confirmation")
    p_reset.set_defaults(func=cmd_reset)

    # run
    p_run = sub.add_parser("run", help="Run tests from sqa.yaml via Codex CLI")
    p_run.add_argument("--plan", required=True, help="Required plan label for outputs")
    p_run.add_argument("--tests", default=None, help="Comma-separated test IDs or globs (e.g., UT-001,UT-002 or UT-*)")
    p_run.add_argument("--plan-file", default="sqa.yaml", help="Path to sqa.yaml file")
    p_run.add_argument(
        "--failed-missing",
        action="store_true",
        help="Only run tests that are missing or not PASSED under runs/sqa/<plan>",
    )
    # No ability to override the Codex command; flags are fixed in the runner
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
