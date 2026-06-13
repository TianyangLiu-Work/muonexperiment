from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.artifacts import ARTIFACT_DIRS, IGNORE_POLICY, KEY_TABLES
from e11_condition_geometry.reporting import markdown_table, write_markdown


JSON_PATH = Path("results/e11_artifact_manifest.json")
MARKDOWN_PATH = Path("discussion/e11_artifact_manifest.md")


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def human_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def csv_rows(path: Path) -> int:
    return len(pd.read_csv(path)) if path.exists() else -1


def main() -> None:
    artifact_dirs = [dict(item) for item in ARTIFACT_DIRS]
    for item in artifact_dirs:
        size = path_size_bytes(Path(item["path"]))
        item["size_bytes"] = size
        item["size"] = human_size(size)

    table_rows = [
        {
            "path": path,
            "rows": csv_rows(Path(path)),
            "size": human_size(path_size_bytes(Path(path))),
        }
        for path in KEY_TABLES
    ]

    ignore_policy = [dict(item) for item in IGNORE_POLICY]

    manifest = {
        "purpose": "E11 reproducibility and artifact-boundary manifest.",
        "artifact_dirs": artifact_dirs,
        "key_tables": table_rows,
        "ignore_policy": ignore_policy,
        "validation_command": "make e11-check",
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    artifact_frame = pd.DataFrame(artifact_dirs)
    table_frame = pd.DataFrame(table_rows)
    ignore_frame = pd.DataFrame(ignore_policy)
    text = f"""# E11 Artifact Manifest

This generated manifest documents the current reproducibility boundary for E11. It separates the head-to-tail paper evidence from background condition-geometry guardrails, records what should be committed as evidence, and lists which key CSV tables back the paper-facing claims.

## Artifact Directories

{markdown_table(artifact_frame, ["path", "role", "commit_policy", "size"])}

## Key Quantitative Tables

{markdown_table(table_frame, ["path", "rows", "size"])}

## Ignored Local Artifacts

{markdown_table(ignore_frame, ["path_or_pattern", "reason"])}

## Validation Command

```bash
{manifest["validation_command"]}
```

Machine-readable copy: `results/e11_artifact_manifest.json`.
"""
    write_markdown(MARKDOWN_PATH, text)
    print(f"saved artifact manifest to {MARKDOWN_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
