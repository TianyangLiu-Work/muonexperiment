from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e11_condition_geometry.long_tail_practical_training import LongTailPracticalTrainingConfig
from e11_run_long_tail_practical_training import OUTPUT_DIR, write_discussion, write_figure


def main() -> None:
    config_path = OUTPUT_DIR / "config.json"
    if config_path.exists():
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_config["seeds"] = tuple(raw_config["seeds"])
        raw_config["head_classes"] = tuple(raw_config["head_classes"])
        raw_config["tail_classes"] = tuple(raw_config["tail_classes"])
        config = LongTailPracticalTrainingConfig(**raw_config)
    else:
        config = LongTailPracticalTrainingConfig()
    step_metrics = pd.read_csv(OUTPUT_DIR / "step_metrics.csv")
    summary = pd.read_csv(OUTPUT_DIR / "summary.csv")
    figure_path = write_figure(step_metrics)
    write_discussion(config, summary, figure_path)
    print("rewrote long-tail practical training note")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
