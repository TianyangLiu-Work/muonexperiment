from __future__ import annotations

from pathlib import Path

import pandas as pd


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, str)):
        return str(value)
    return f"{float(value):.{digits}g}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.columns:
        display[column] = display[column].map(fmt)
    return display.to_markdown(index=False)


def require_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        mask &= frame[key].eq(value)
    matches = frame[mask]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {filters}, got {len(matches)}")
    return matches.iloc[0]


def ratio_ci(record: pd.Series, low: str = "ratio_ci95_low", high: str = "ratio_ci95_high") -> str:
    return f"[{fmt(record[low])}, {fmt(record[high])}]"


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
