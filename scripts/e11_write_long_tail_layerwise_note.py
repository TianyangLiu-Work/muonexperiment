from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_layerwise import LongTailLayerwiseConfig


def _load_runner_module():
    path = ROOT / "scripts" / "e11_run_long_tail_layerwise.py"
    spec = importlib.util.spec_from_file_location("e11_run_long_tail_layerwise", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner_module()
    metrics_path = runner.OUTPUT_DIR / "metrics.csv"
    summary_path = runner.OUTPUT_DIR / "summary.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing long-tail layerwise metrics: {metrics_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing long-tail layerwise summary: {summary_path}")
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    figure_path = runner.write_figure(metrics, summary)
    runner.write_discussion(LongTailLayerwiseConfig(), summary, figure_path)
    print(f"wrote long-tail layerwise discussion from {summary_path}")


if __name__ == "__main__":
    main()
