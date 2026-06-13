from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_muon_bridge import LongTailMuonBridgeConfig


def _load_runner_module():
    path = ROOT / "scripts" / "e11_run_long_tail_muon_bridge.py"
    spec = importlib.util.spec_from_file_location("e11_run_long_tail_muon_bridge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner_module()
    pair_summary_path = runner.OUTPUT_DIR / "pair_summary.csv"
    if not pair_summary_path.is_file():
        raise FileNotFoundError(f"missing long-tail Muon bridge summary: {pair_summary_path}")
    pair_summary = pd.read_csv(pair_summary_path)
    figure_path = runner.write_figure(pair_summary)
    runner.write_discussion(LongTailMuonBridgeConfig(), pair_summary, figure_path)
    print(f"wrote long-tail Muon bridge discussion from {pair_summary_path}")


if __name__ == "__main__":
    main()
