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

from e11_condition_geometry.long_tail_practical_training import (
    LongTailPracticalTrainingConfig,
    LongTailPracticalTrainingLRSweepConfig,
)
from e11_run_long_tail_practical_training_lr_sweep import OUTPUT_DIR, write_discussion, write_figure


def _load_config() -> LongTailPracticalTrainingLRSweepConfig:
    path = OUTPUT_DIR / "config.json"
    if not path.exists():
        return LongTailPracticalTrainingLRSweepConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    base_raw = raw.get("base_config", {})
    for key in ["seeds", "head_classes", "tail_classes"]:
        if key in base_raw:
            base_raw[key] = tuple(base_raw[key])
    base_config = LongTailPracticalTrainingConfig(**base_raw)
    return LongTailPracticalTrainingLRSweepConfig(
        muon_lrs=tuple(raw.get("muon_lrs", (3e-3, 1e-2, 3e-2, 1e-1))),
        base_config=base_config,
    )


def main() -> None:
    config = _load_config()
    sweep = pd.read_csv(OUTPUT_DIR / "sweep_summary.csv")
    figure_path = write_figure(sweep)
    write_discussion(config, sweep, figure_path)
    print("rewrote practical training LR sweep note")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
