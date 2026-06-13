from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.head_tail_interference import HeadTailConfig


def _load_runner_module():
    path = ROOT / "scripts" / "e11_run_head_tail_interference.py"
    spec = importlib.util.spec_from_file_location("e11_run_head_tail_interference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner_module()
    pair_summary_path = runner.OUTPUT_DIR / "pair_summary.csv"
    if not pair_summary_path.is_file():
        raise FileNotFoundError(f"missing head-to-tail summary: {pair_summary_path}")
    pair_summary = pd.read_csv(pair_summary_path)
    figure_path = runner.write_figure(pair_summary)
    runner.write_discussion(HeadTailConfig(), pair_summary, figure_path)
    print(f"wrote head-to-tail discussion from {pair_summary_path}")


if __name__ == "__main__":
    main()
