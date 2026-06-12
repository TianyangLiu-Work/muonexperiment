from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.discussion import write_discussion
import pandas as pd


def main() -> None:
    config = default_config()
    performance = pd.read_csv(config.output_dir / "performance_summary.csv")
    geometry = pd.read_csv(config.output_dir / "geometry_summary.csv")
    prediction = pd.read_csv(config.output_dir / "prediction_summary.csv")
    volatility = pd.read_csv(config.output_dir / "volatility_summary.csv")
    update_spectrum = pd.read_csv(config.output_dir / "update_spectrum_summary.csv")
    update_transmission = pd.read_csv(config.output_dir / "update_transmission_summary.csv")
    first_order_calibration = pd.read_csv(config.output_dir / "first_order_calibration_summary.csv")
    polar_alignment = pd.read_csv(config.output_dir / "polar_alignment_summary.csv")
    first_order_pair = pd.read_csv(config.output_dir / "first_order_pair_summary.csv")
    win_condition = pd.read_csv(config.output_dir / "win_condition_summary.csv")
    win_generalization = pd.read_csv(config.output_dir / "win_prediction_generalization_summary.csv")
    win_overlap = pd.read_csv(config.output_dir / "win_feature_overlap_summary.csv")
    equal_update_volatility = pd.read_csv(Path("results/e11_equal_update") / "volatility_summary.csv")
    equal_update_spectrum = pd.read_csv(Path("results/e11_equal_update") / "update_spectrum_summary.csv")
    equal_update_transmission = pd.read_csv(Path("results/e11_equal_update") / "update_transmission_summary.csv")
    equal_update_first_order_calibration = pd.read_csv(Path("results/e11_equal_update") / "first_order_calibration_summary.csv")
    equal_update_polar_alignment = pd.read_csv(Path("results/e11_equal_update") / "polar_alignment_summary.csv")
    equal_update_first_order_pair = pd.read_csv(Path("results/e11_equal_update") / "first_order_pair_summary.csv")
    equal_update_win_condition = pd.read_csv(Path("results/e11_equal_update") / "win_condition_summary.csv")
    equal_update_win_generalization = pd.read_csv(Path("results/e11_equal_update") / "win_prediction_generalization_summary.csv")
    equal_update_win_overlap = pd.read_csv(Path("results/e11_equal_update") / "win_feature_overlap_summary.csv")
    figure_paths = {
        "loss_curves": config.figure_dir / "loss_curves.png",
        "geometry_separation": config.figure_dir / "geometry_separation.png",
        "predicted_decrease_vs_observed": config.figure_dir / "predicted_decrease_vs_observed.png",
        "condition_score_trajectories": config.figure_dir / "condition_score_trajectories.png",
        "mean_3d_condition_loss": config.figure_dir / "mean_3d_condition_loss.png",
        "layerwise_3d_condition_loss": config.figure_dir / "layerwise_3d_condition_loss.png",
        "volatility_robustness": config.figure_dir / "volatility_robustness.png",
        "update_spectrum_robustness": config.figure_dir / "update_spectrum_robustness.png",
        "update_transmission_heatmap": config.figure_dir / "update_transmission_heatmap.png",
        "first_order_calibration": config.figure_dir / "first_order_calibration.png",
        "polar_alignment_identity": config.figure_dir / "polar_alignment_identity.png",
        "first_order_pair_comparison": config.figure_dir / "first_order_pair_comparison.png",
        "win_condition_summary": config.figure_dir / "win_condition_summary.png",
        "win_prediction_generalization": config.figure_dir / "win_prediction_generalization.png",
        "win_feature_overlap": config.figure_dir / "win_feature_overlap.png",
        "equal_update_volatility_robustness": Path("figures/e11_equal_update") / "volatility_robustness.png",
        "equal_update_update_spectrum_robustness": Path("figures/e11_equal_update") / "update_spectrum_robustness.png",
        "equal_update_update_transmission_heatmap": Path("figures/e11_equal_update") / "update_transmission_heatmap.png",
        "equal_update_first_order_calibration": Path("figures/e11_equal_update") / "first_order_calibration.png",
        "equal_update_polar_alignment_identity": Path("figures/e11_equal_update") / "polar_alignment_identity.png",
        "equal_update_first_order_pair_comparison": Path("figures/e11_equal_update") / "first_order_pair_comparison.png",
        "equal_update_win_condition_summary": Path("figures/e11_equal_update") / "win_condition_summary.png",
        "equal_update_win_prediction_generalization": Path("figures/e11_equal_update") / "win_prediction_generalization.png",
        "equal_update_win_feature_overlap": Path("figures/e11_equal_update") / "win_feature_overlap.png",
        "candidate_initial_geometry_support": Path("figures/e11_candidate_scan") / "initial_geometry_support.png",
    }
    missing = [str(path) for path in figure_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing figures: {missing}")
    write_discussion(
        config.discussion_path,
        performance=performance,
        geometry=geometry,
        prediction=prediction,
        volatility=volatility,
        update_spectrum=update_spectrum,
        update_transmission=update_transmission,
        first_order_calibration=first_order_calibration,
        polar_alignment=polar_alignment,
        first_order_pair=first_order_pair,
        win_condition=win_condition,
        win_generalization=win_generalization,
        win_overlap=win_overlap,
        equal_update_volatility=equal_update_volatility,
        equal_update_spectrum=equal_update_spectrum,
        equal_update_transmission=equal_update_transmission,
        equal_update_first_order_calibration=equal_update_first_order_calibration,
        equal_update_polar_alignment=equal_update_polar_alignment,
        equal_update_first_order_pair=equal_update_first_order_pair,
        equal_update_win_condition=equal_update_win_condition,
        equal_update_win_generalization=equal_update_win_generalization,
        equal_update_win_overlap=equal_update_win_overlap,
        figure_paths=figure_paths,
    )
    print(f"saved discussion to {config.discussion_path}")


if __name__ == "__main__":
    main()
