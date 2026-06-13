from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PaperStats:
    """Current paper-facing numeric anchors for the head-to-tail draft."""

    synthetic_positive: pd.Series
    synthetic_negative: pd.Series
    long_tail_one_step: pd.Series
    muon_bridge_polar_momentum: pd.Series
    muon_bridge_ns_momentum: pd.Series
    practical_muon_bridge_polar_momentum: pd.Series
    practical_muon_bridge_ns_momentum: pd.Series
    practical_training: pd.Series
    long_tail_forgetting: pd.Series
    layer_one: pd.Series
    layer_two: pd.Series


def _read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(root / relative_path)


def load_paper_stats(root: Path | str = ".") -> PaperStats:
    """Load the current head-to-tail paper anchors from generated E11 results."""

    root = Path(root)
    head_tail = _read_csv(root, "results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    one_step = _read_csv(root, "results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    muon_bridge = _read_csv(root, "results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = _read_csv(root, "results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    practical_training = _read_csv(root, "results/e11_long_tail_practical_training/summary.csv").iloc[0]
    forgetting = _read_csv(root, "results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = _read_csv(root, "results/e11_long_tail_layerwise/summary.csv")

    return PaperStats(
        synthetic_positive=head_tail.loc["high_head_rank_low_tail_srank"],
        synthetic_negative=head_tail.loc["low_head_rank_high_tail_srank"],
        long_tail_one_step=one_step,
        muon_bridge_polar_momentum=muon_bridge.loc["polar_momentum"],
        muon_bridge_ns_momentum=muon_bridge.loc["ns_momentum"],
        practical_muon_bridge_polar_momentum=practical_bridge.loc["polar_momentum"],
        practical_muon_bridge_ns_momentum=practical_bridge.loc["ns_momentum"],
        practical_training=practical_training,
        long_tail_forgetting=forgetting,
        layer_one=layerwise[layerwise["layer"].eq(1)].iloc[0],
        layer_two=layerwise[layerwise["layer"].eq(2)].iloc[0],
    )
