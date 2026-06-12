from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.plots import make_all_figures
from e11_condition_geometry.runner import run_equal_update_experiment
from e11_condition_geometry.statistics import (
    activation_perturbation_summary,
    final_performance_summary,
    first_order_calibration_summary,
    first_order_pair_summary,
    first_order_win_condition_summary,
    win_feature_overlap_summary,
    win_prediction_generalization_summary,
    geometry_summary,
    polar_alignment_summary,
    prediction_summary,
    run_dynamics_summary,
    update_spectrum_summary,
    update_transmission_summary,
    volatility_summary,
)


def main() -> None:
    config = replace(
        default_config(),
        output_dir=Path("results/e11_equal_update"),
        figure_dir=Path("figures/e11_equal_update"),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    steps, layers = run_equal_update_experiment(config)
    performance = final_performance_summary(steps)
    geometry = geometry_summary(steps)
    prediction = prediction_summary(steps)
    dynamics = run_dynamics_summary(steps)
    volatility = volatility_summary(dynamics)
    update_spectrum = update_spectrum_summary(steps)
    activation_perturbation = activation_perturbation_summary(layers)
    update_transmission = update_transmission_summary(steps)
    first_order_calibration = first_order_calibration_summary(steps)
    polar_alignment = polar_alignment_summary(layers)
    first_order_pair = first_order_pair_summary(steps)
    win_condition = first_order_win_condition_summary(steps, layers)
    win_generalization = win_prediction_generalization_summary(steps, layers)
    win_overlap = win_feature_overlap_summary(steps, layers)
    steps.to_csv(config.output_dir / "step_metrics.csv", index=False)
    layers.to_csv(config.output_dir / "layer_metrics.csv", index=False)
    performance.to_csv(config.output_dir / "performance_summary.csv", index=False)
    geometry.to_csv(config.output_dir / "geometry_summary.csv", index=False)
    prediction.to_csv(config.output_dir / "prediction_summary.csv", index=False)
    dynamics.to_csv(config.output_dir / "run_dynamics_summary.csv", index=False)
    volatility.to_csv(config.output_dir / "volatility_summary.csv", index=False)
    update_spectrum.to_csv(config.output_dir / "update_spectrum_summary.csv", index=False)
    activation_perturbation.to_csv(config.output_dir / "activation_perturbation_summary.csv", index=False)
    update_transmission.to_csv(config.output_dir / "update_transmission_summary.csv", index=False)
    first_order_calibration.to_csv(config.output_dir / "first_order_calibration_summary.csv", index=False)
    polar_alignment.to_csv(config.output_dir / "polar_alignment_summary.csv", index=False)
    first_order_pair.to_csv(config.output_dir / "first_order_pair_summary.csv", index=False)
    win_condition.to_csv(config.output_dir / "win_condition_summary.csv", index=False)
    win_generalization.to_csv(config.output_dir / "win_prediction_generalization_summary.csv", index=False)
    win_overlap.to_csv(config.output_dir / "win_feature_overlap_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    paths = make_all_figures(
        steps,
        layers,
        {
            "performance": performance,
            "geometry": geometry,
            "prediction": prediction,
            "volatility": volatility,
            "update_spectrum": update_spectrum,
            "update_transmission": update_transmission,
            "first_order_calibration": first_order_calibration,
            "polar_alignment": polar_alignment,
            "first_order_pair": first_order_pair,
            "win_condition": win_condition,
            "win_generalization": win_generalization,
            "win_overlap": win_overlap,
        },
        config.figure_dir,
    )
    print(f"saved equal-update E11 results to {config.output_dir}")
    print(f"step rows={len(steps)}, layer rows={len(layers)}, runs={steps['run_id'].nunique()}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
