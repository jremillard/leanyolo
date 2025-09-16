from pathlib import Path

import warnings

from leanyolo.utils import val_log as V


def test_ensure_csv_creates_and_appends(tmp_path):
    p = tmp_path / "val.csv"
    # Create new CSV
    V.ensure_csv(p)
    # Header matches canonical columns
    header = p.read_text(encoding="utf-8").splitlines()[0]
    assert ",".join(V.COLUMNS) == header

    # Append a minimal row
    V.append_row(p, {"timestamp": "t0", "model": "m0", "notes": "ok"})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # Check a few fields by position
    cells = lines[1].split(",")
    assert cells[0] == "t0"  # timestamp
    assert cells[V.COLUMNS.index("model")] == "m0"
    assert cells[V.COLUMNS.index("notes")] == "ok"


def test_ensure_csv_migrates_existing_header(tmp_path):
    p = tmp_path / "val_mig.csv"
    # Write an older/smaller header with one row
    old_cols = ["timestamp", "model", "commit"]
    p.write_text(
        ",".join(old_cols) + "\n" + ",".join(["t1", "m1", "c1"]) + "\n",
        encoding="utf-8",
    )

    # Ensure migration to canonical header
    V.ensure_csv(p)

    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(V.COLUMNS)
    row = lines[1].split(",")
    assert row[V.COLUMNS.index("timestamp")] == "t1"
    assert row[V.COLUMNS.index("model")] == "m1"
    assert row[V.COLUMNS.index("commit")] == "c1"
    # New columns default to empty string
    assert row[V.COLUMNS.index("map_50")] == ""


def test_collect_env_and_now_iso():
    env = V.collect_env_info(device="cpu")
    assert set(["commit", "host", "device", "device_name"]).issubset(env)
    # timestamp format quick check and ensure no deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error", DeprecationWarning)
        ts = V.now_iso()
    assert ts.endswith("Z") and "T" in ts

