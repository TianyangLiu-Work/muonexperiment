from pathlib import Path

import pandas as pd
import pytest

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


def test_fmt_handles_common_report_values() -> None:
    assert fmt(float("nan")) == "n/a"
    assert fmt(True) == "yes"
    assert fmt(False) == "no"
    assert fmt(3) == "3"
    assert fmt("Muon") == "Muon"
    assert fmt(1.234567, digits=3) == "1.23"


def test_markdown_table_formats_present_columns_only() -> None:
    frame = pd.DataFrame(
        [
            {"metric": "nrUpdate", "ratio": 2.01001, "supported": True},
            {"metric": "stUpdate", "ratio": float("nan"), "supported": False},
        ]
    )
    table = markdown_table(frame, ["metric", "ratio", "supported", "missing_column"])

    assert "missing_column" not in table
    assert "nrUpdate" in table
    assert "2.01" in table
    assert "n/a" in table
    assert "yes" in table
    assert "no" in table


def test_require_one_returns_unique_row_and_rejects_ambiguous_matches() -> None:
    frame = pd.DataFrame(
        [
            {"metric": "nrUpdate", "family": "All", "ratio_ci95_low": 1.9, "ratio_ci95_high": 2.1},
            {"metric": "nrUpdate", "family": "MF", "ratio_ci95_low": 1.1, "ratio_ci95_high": 1.3},
        ]
    )

    row = require_one(frame, metric="nrUpdate", family="All")
    assert row["ratio_ci95_low"] == 1.9
    assert ratio_ci(row) == "[1.9, 2.1]"

    with pytest.raises(ValueError, match="expected one row"):
        require_one(frame, metric="nrUpdate")
    with pytest.raises(ValueError, match="expected one row"):
        require_one(frame, metric="stUpdate")


def test_write_markdown_creates_parent_and_single_trailing_newline(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "note.md"

    write_markdown(output, "hello\n\n")

    assert output.read_text(encoding="utf-8") == "hello\n"
