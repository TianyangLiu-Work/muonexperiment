from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_one_step import LongTailOneStepConfig


def _load_runner_module():
    path = ROOT / "scripts" / "e11_run_long_tail_one_step.py"
    spec = importlib.util.spec_from_file_location("e11_run_long_tail_one_step", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner_module()
    step_metrics_path = runner.OUTPUT_DIR / "step_metrics.csv"
    pair_summary_path = runner.OUTPUT_DIR / "pair_summary.csv"
    if not step_metrics_path.is_file():
        raise FileNotFoundError(f"missing long-tail one-step steps: {step_metrics_path}")
    if not pair_summary_path.is_file():
        raise FileNotFoundError(f"missing long-tail one-step summary: {pair_summary_path}")
    step_metrics = pd.read_csv(step_metrics_path)
    pair_summary = pd.read_csv(pair_summary_path)
    figure_path = runner.write_figure(step_metrics, pair_summary)
    runner.write_discussion(LongTailOneStepConfig(), pair_summary, figure_path)
    print(f"wrote long-tail one-step discussion from {pair_summary_path}")


if __name__ == "__main__":
    main()
