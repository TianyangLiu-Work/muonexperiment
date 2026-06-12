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
from e11_condition_geometry.config import ExperimentConfig, ProblemSpec
from e11_condition_geometry.plots import make_all_figures
from e11_condition_geometry.runner import run_equal_update_experiment
from e11_condition_geometry.statistics import (
    final_performance_summary,
    first_order_calibration_summary,
    first_order_pair_summary,
    first_order_win_condition_summary,
    geometry_summary,
    polar_alignment_summary,
    prediction_summary,
    run_dynamics_summary,
    update_spectrum_summary,
    update_transmission_summary,
    volatility_summary,
    win_feature_overlap_summary,
    win_prediction_generalization_summary,
)


OUTPUT_DIR = Path("results/e11_overlap_followup")
FIGURE_DIR = Path("figures/e11_overlap_followup")
DISCUSSION_PATH = Path("discussion/e11_overlap_followup.md")


def overlap_specs() -> tuple[ProblemSpec, ...]:
    learning_rates = (3e-3, 1e-2, 3e-2)
    base_specs = [
        (
            "MatrixFactorizationInput",
            {
                "steps": 10,
                "d": 20,
                "rank": 10,
                "kappa": 1.0,
                "num_factors": 10,
                "input_columns_multiplier": 2,
            },
        ),
        (
            "MatrixFactorizationInput",
            {
                "steps": 10,
                "d": 20,
                "rank": 10,
                "kappa": 10.0,
                "num_factors": 10,
                "input_columns_multiplier": 2,
            },
        ),
        (
            "MatrixSensing",
            {
                "steps": 5,
                "d": 20,
                "rank": 10,
                "kappa": 10.0,
                "measurement_multiplier": 2.0,
            },
        ),
        (
            "MatrixSensing",
            {
                "steps": 5,
                "d": 20,
                "rank": 5,
                "kappa": 1.0,
                "measurement_multiplier": 2.0,
            },
        ),
        (
            "MatrixSensing",
            {
                "steps": 5,
                "d": 20,
                "rank": 10,
                "kappa": 1.0,
                "measurement_multiplier": 0.5,
            },
        ),
        (
            "SmallMLPDigits",
            {
                "steps": 10,
                "d": 64,
                "rank": 10,
                "hidden_dim": 16,
                "num_samples": 128,
                "batch_size": 32,
            },
        ),
    ]
    specs: list[ProblemSpec] = []
    for family, kwargs in base_specs:
        for lr in learning_rates:
            setting_parts = [family.replace("MatrixFactorizationInput", "MF").replace("MatrixSensing", "MS")]
            for key in ["d", "rank", "kappa", "num_factors", "input_columns_multiplier", "measurement_multiplier", "hidden_dim", "num_samples"]:
                if key in kwargs:
                    setting_parts.append(f"{key}={kwargs[key]:g}" if isinstance(kwargs[key], float) else f"{key}={kwargs[key]}")
            setting_parts.append(f"lr={lr:.0e}")
            specs.append(
                ProblemSpec(
                    family=family,
                    setting="overlap " + " ".join(setting_parts),
                    lr=lr,
                    **kwargs,
                )
            )
    return tuple(specs)


def build_config() -> ExperimentConfig:
    base = default_config()
    return replace(
        base,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        discussion_path=DISCUSSION_PATH,
        specs=overlap_specs(),
    )


def summarize_and_save(config: ExperimentConfig, steps: pd.DataFrame, layers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dynamics = run_dynamics_summary(steps)
    summaries = {
        "performance": final_performance_summary(steps),
        "geometry": geometry_summary(steps),
        "prediction": prediction_summary(steps),
        "volatility": volatility_summary(dynamics),
        "update_spectrum": update_spectrum_summary(steps),
        "update_transmission": update_transmission_summary(steps),
        "first_order_calibration": first_order_calibration_summary(steps),
        "polar_alignment": polar_alignment_summary(layers),
        "first_order_pair": first_order_pair_summary(steps),
        "win_condition": first_order_win_condition_summary(steps, layers),
        "win_generalization": win_prediction_generalization_summary(steps, layers),
        "win_overlap": win_feature_overlap_summary(steps, layers),
        "run_dynamics": dynamics,
    }
    steps.to_csv(config.output_dir / "step_metrics.csv", index=False)
    layers.to_csv(config.output_dir / "layer_metrics.csv", index=False)
    for name, frame in summaries.items():
        frame.to_csv(config.output_dir / f"{name}_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return summaries


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.4g}")
    return display.to_markdown(index=False)


def write_discussion(config: ExperimentConfig, summaries: dict[str, pd.DataFrame], figure_paths: dict[str, Path]) -> None:
    pair = summaries["first_order_pair"]
    def ratio_line(metric: str, family: str) -> str:
        row = pair[(pair["metric"] == metric) & (pair["problem_family"] == family)].iloc[0]
        return (
            f"{family}: ratio={row['geomean_ratio_muon_over_adam']:.4g}, "
            f"95% CI=[{row['ratio_ci95_low']:.4g}, {row['ratio_ci95_high']:.4g}], "
            f"Muon-higher pairs={int(row['muon_higher_pairs'])}/{int(row['n_pairs'])}"
        )

    pair_table = markdown_table(
        pair[
            pair["metric"].isin(["update_grad_inner", "update_grad_cosine", "delta_loss"])
            & pair["problem_family"].isin(["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
        ],
        [
            "metric",
            "problem_family",
            "n_pairs",
            "muon_higher_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
        ],
    )
    performance_table = markdown_table(
        summaries["performance"],
        [
            "problem_family",
            "setting",
            "algo",
            "runs",
            "median_final_loss",
            "median_recovery",
            "median_delta_loss",
            "median_nrG",
            "median_stA",
            "median_condition_score",
        ],
    )
    calibration_table = markdown_table(
        summaries["first_order_calibration"][
            summaries["first_order_calibration"]["group"].isin(
                ["All", "Adam", "Muon", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
            )
        ],
        [
            "group",
            "points",
            "positive_points",
            "spearman_delta_vs_first_order",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "within_factor_2",
        ],
    )
    text = f"""# E11 Overlap Follow-Up

## Purpose

This follow-up runs a small set of settings selected from the initial-geometry overlap scan. The goal is to test Muon-vs-Adam one-step progress in settings whose spectral diagnostics are closer to the existing global support, rather than only extrapolating between the original separated families.

All runs use the equal-update control: Adam and Muon start from matched initialization, propose their own update direction, and are rescaled to the same global relative update norm at each step.

## Selected Settings

- MF-with-input: `d=20`, `rank=10`, 10 factors, `input_columns_multiplier=2`, `kappa in {{1, 10}}`.
- Matrix Sensing: `d=20`, selected rank/measurement settings from the scan.
- Small MLP digits: `hidden_dim=16`, `num_samples=128`.
- Learning rates: `3e-3`, `1e-2`, `3e-2`; seeds: 0 to 4.

## First-Order Pair Comparison

Ratios above 1 mean Muon is larger than Adam in the matched setting/seed/step pair.

Main observed pattern:

- `update_grad_inner`: {ratio_line("update_grad_inner", "MatrixFactorizationInput")}.
- `update_grad_inner`: {ratio_line("update_grad_inner", "MatrixSensing")}.
- `update_grad_inner`: {ratio_line("update_grad_inner", "SmallMLPDigits")}.
- `delta_loss`: {ratio_line("delta_loss", "MatrixFactorizationInput")}.
- `delta_loss`: {ratio_line("delta_loss", "MatrixSensing")}.
- `delta_loss`: {ratio_line("delta_loss", "SmallMLPDigits")}.

{pair_table}

![First-order pair comparison](../{figure_paths['first_order_pair_comparison']})

## First-Order Calibration

{calibration_table}

![First-order calibration](../{figure_paths['first_order_calibration']})

## Performance Context

{performance_table}

![Loss curves](../{figure_paths['loss_curves']})

## Current Interpretation

These overlap settings are not meant to replace the main experiment. They are a targeted check of whether the Matrix Sensing advantage persists when the spectral support is pulled toward the other families. The result is mixed but useful: Matrix Sensing still gives Muon a small first-order advantage, SmallMLPDigits becomes Muon-favorable after moving to a smaller hidden layer, and MF-with-input remains Adam-favorable in first-order descent. This suggests the useful predictor is not simply the problem-family label, but also not yet captured by the current rank/spectral features alone.

The key quantity remains `update_grad_inner = <G, W_t-W_(t+1)>`, because it is the direct first-order progress term.
"""
    config.discussion_path.parent.mkdir(parents=True, exist_ok=True)
    config.discussion_path.write_text(text, encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    steps, layers = run_equal_update_experiment(config)
    summaries = summarize_and_save(config, steps, layers)
    figure_paths = make_all_figures(steps, layers, summaries, config.figure_dir)
    write_discussion(config, summaries, figure_paths)
    print(f"saved overlap follow-up results to {config.output_dir}")
    print(f"step rows={len(steps)}, layer rows={len(layers)}, runs={steps['run_id'].nunique()}")
    print(f"discussion: {config.discussion_path}")


if __name__ == "__main__":
    main()
