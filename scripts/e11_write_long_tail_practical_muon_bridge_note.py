from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_muon_bridge import LongTailPracticalMuonBridgeConfig


def _load_runner_module():
    path = ROOT / "scripts" / "e11_run_long_tail_practical_muon_bridge.py"
    spec = importlib.util.spec_from_file_location("e11_run_long_tail_practical_muon_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner_module()
    step_metrics_path = runner.OUTPUT_DIR / "step_metrics.csv"
    step_summary_path = runner.OUTPUT_DIR / "step_summary.csv"
    summary_path = runner.OUTPUT_DIR / "summary.csv"
    for path in [step_metrics_path, step_summary_path, summary_path]:
        if not path.is_file():
            raise FileNotFoundError(f"missing long-tail practical Muon bridge artifact: {path}")
    step_metrics = pd.read_csv(step_metrics_path)
    step_summary = pd.read_csv(step_summary_path)
    summary = pd.read_csv(summary_path)
    figure_path = runner.write_figure(step_metrics, step_summary)
    runner.write_discussion(LongTailPracticalMuonBridgeConfig(), summary, step_summary, figure_path)
    print(f"wrote long-tail practical Muon bridge discussion from {summary_path}")


if __name__ == "__main__":
    main()
