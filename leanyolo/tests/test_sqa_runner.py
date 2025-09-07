import re
from pathlib import Path

import pytest


from tools.sqa_runner import (
    build_prompts,
    determine_status,
    filter_tests,
    parse_sqa_md,
)


def test_parse_ids_from_sqa_md():
    sqa_path = Path("sqa.md")
    assert sqa_path.exists(), "sqa.md must exist at repo root"
    cases = parse_sqa_md(sqa_path)
    ids = [c.test_id for c in cases]
    # Ensure overall coverage of sections
    for expected in ["EN-001", "UT-001", "FT-001", "IT-004"]:
        assert expected in ids
    # Sanity: each case has non-empty fields
    for c in cases:
        assert c.test_id and c.test_point and c.steps and c.expected


def test_filter_by_glob():
    cases = parse_sqa_md(Path("sqa.md"))
    only_ut = filter_tests(cases, ["UT-*"])
    assert only_ut, "UT-* should match at least one test"
    assert all(c.test_id.startswith("UT-") for c in only_ut)
    # Ensure we excluded EN and IT groups
    assert all(not c.test_id.startswith("EN-") for c in only_ut)


def test_prompt_generation_includes_steps_and_expected():
    cases = parse_sqa_md(Path("sqa.md"))
    # pick a known case
    target = next(c for c in cases if c.test_id == "UT-001")
    test_msg, read_msg = build_prompts(target, Path("sqa.md"))
    assert "Steps:" in test_msg
    assert target.steps in test_msg
    assert "TEST STATUS: PASSED or FAILED" in test_msg
    assert "Read sqa.md" in read_msg or "sqa.md" in read_msg
    assert target.expected in read_msg


@pytest.mark.parametrize(
    "stdout,exit_code,expected",
    [
        ("...\nTEST STATUS: PASSED\n", 0, "PASSED"),
        ("...\nTEST STATUS: FAILED\n", 1, "FAILED"),
        ("2 passed, 0 warnings in 0.10s\n", 0, "PASSED"),
        ("1 failed, 1 passed in 0.10s\n", 1, "FAILED"),
        ("no markers\n", 0, "UNKNOWN"),
    ],
)
def test_status_detection(stdout, exit_code, expected):
    status = determine_status(stdout, exit_code)
    assert status == expected

