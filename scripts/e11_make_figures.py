from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.plots import make_all_figures
import pandas as pd


def main() -> None:
    config = default_config()
    steps = pd.read_csv(config.output_dir / "step_metrics.csv")
    layers = pd.read_csv(config.output_dir / "layer_metrics.csv")
    summaries = {
        "performance": pd.read_csv(config.output_dir / "performance_summary.csv"),
        "geometry": pd.read_csv(config.output_dir / "geometry_summary.csv"),
        "prediction": pd.read_csv(config.output_dir / "prediction_summary.csv"),
        "volatility": pd.read_csv(config.output_dir / "volatility_summary.csv"),
        "update_spectrum": pd.read_csv(config.output_dir / "update_spectrum_summary.csv"),
        "update_transmission": pd.read_csv(config.output_dir / "update_transmission_summary.csv"),
        "first_order_calibration": pd.read_csv(config.output_dir / "first_order_calibration_summary.csv"),
        "polar_alignment": pd.read_csv(config.output_dir / "polar_alignment_summary.csv"),
        "first_order_pair": pd.read_csv(config.output_dir / "first_order_pair_summary.csv"),
        "win_condition": pd.read_csv(config.output_dir / "win_condition_summary.csv"),
        "win_generalization": pd.read_csv(config.output_dir / "win_prediction_generalization_summary.csv"),
        "win_overlap": pd.read_csv(config.output_dir / "win_feature_overlap_summary.csv"),
    }
    paths = make_all_figures(steps, layers, summaries, config.figure_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
