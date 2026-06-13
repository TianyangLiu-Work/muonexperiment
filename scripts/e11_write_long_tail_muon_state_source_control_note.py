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
    path = ROOT / "scripts" / "e11_run_long_tail_muon_state_source_control.py"
    spec = importlib.util.spec_from_file_location("e11_run_long_tail_muon_state_source_control", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner_module()
    summary_path = runner.OUTPUT_DIR / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            "missing state-source control summary; run "
            "scripts/e11_run_long_tail_muon_state_source_control.py first"
        )
    summary = pd.read_csv(summary_path)
    figure_path = runner.write_figure(summary)
    runner.write_discussion(LongTailPracticalMuonBridgeConfig(), summary, figure_path)
    print("saved discussion/e11_long_tail_muon_state_source_control.md")


if __name__ == "__main__":
    main()
