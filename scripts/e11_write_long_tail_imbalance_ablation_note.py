from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_runner_module():
    path = ROOT / "scripts" / "e11_run_long_tail_imbalance_ablation.py"
    spec = importlib.util.spec_from_file_location("e11_run_long_tail_imbalance_ablation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner_module()
    step_path = runner.OUTPUT_DIR / "step_metrics.csv"
    summary_path = runner.OUTPUT_DIR / "summary.csv"
    if not step_path.is_file():
        raise FileNotFoundError(f"missing imbalance ablation step metrics: {step_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing imbalance ablation summary: {summary_path}")
    step_metrics = pd.read_csv(step_path)
    summary = pd.read_csv(summary_path)
    runner.write_outputs(step_metrics, summary)
    print(f"wrote imbalance ablation discussion from {summary_path}")


if __name__ == "__main__":
    main()
