import math

import pandas as pd

from scripts.e11_evaluate_cifar100_resnet_condition_score_heldouts import HeldoutSplit
from scripts.e11_evaluate_cifar100_resnet_condition_score_heldouts import load_frozen_fits
from scripts.e11_evaluate_cifar100_resnet_condition_score_heldouts import score_heldout_split
from scripts.e11_evaluate_cifar100_resnet_condition_score_heldouts import summarize_heldout_scores
from scripts.e11_freeze_condition_score_v4_validation import load_axis_frame
from scripts.e11_write_cifar100_resnet_condition_score_next import FEATURE_COLUMNS


def _layer_rows(warmup_steps: int) -> list[dict]:
    rows = []
    for index, value in enumerate([1.0, 2.0, 4.0, 8.0], start=1):
        rows.append(
            {
                "warmup_steps": warmup_steps,
                "layer_index": index,
                "parameter": f"layer{index}.weight",
                "mean_gradient_nuclear_rank": value,
                "mean_alignment_ratio_spectral_over_fro": 1.0,
                "mean_step_size_ratio_spectral_over_fro": 1.0,
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": 1.0,
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": 1.0,
                "mean_tail_accuracy_before": 0.5,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": value,
            }
        )
    return rows


def _coefficient_rows() -> pd.DataFrame:
    rows = [
        {
            "source_warmup_steps": 100,
            "term": "depth_intercept",
            "coefficient": 0.0,
            "feature_mean": math.nan,
            "feature_std": math.nan,
            "ridge_alpha": 1.0,
        },
        {
            "source_warmup_steps": 100,
            "term": "depth_slope_log_early_layer_prior",
            "coefficient": 0.0,
            "feature_mean": math.nan,
            "feature_std": math.nan,
            "ridge_alpha": 1.0,
        },
        {
            "source_warmup_steps": 100,
            "term": "residual_intercept",
            "coefficient": 0.0,
            "feature_mean": math.nan,
            "feature_std": math.nan,
            "ridge_alpha": 1.0,
        },
    ]
    for feature_name in FEATURE_COLUMNS:
        rows.append(
            {
                "source_warmup_steps": 100,
                "term": feature_name,
                "coefficient": 1.0 if feature_name == "log_gradient_nuclear_rank" else 0.0,
                "feature_mean": 0.0,
                "feature_std": 1.0,
                "ridge_alpha": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_heldout_condition_score_uses_frozen_coefficients_without_refit(tmp_path):
    split = HeldoutSplit(
        split_id="toy_heldout",
        role="primary_heldout_data",
        display_name="Toy held-out data",
        layer_summary_path=tmp_path / "layer_summary.csv",
        config_path=tmp_path / "config.json",
    )
    source = pd.DataFrame(_layer_rows(100))
    heldout = pd.DataFrame(_layer_rows(200))
    frozen_fits = load_frozen_fits(_coefficient_rows())

    pairs = score_heldout_split(split, heldout, source, frozen_fits)
    summary = summarize_heldout_scores(pairs)
    primary = summary[summary["score"].eq("condition_score_v2_calibrated_residual")].iloc[0]
    source_control = summary[summary["score"].eq("source_observed_drift_positive_control")].iloc[0]

    assert primary["split_id"] == "toy_heldout"
    assert primary["mean_spearman_score_vs_target_residual"] > 0.9
    assert source_control["mean_points"] == 4.0


def test_v4_axis_frame_aggregates_frobenius_amplitude_from_metrics(tmp_path):
    layer_summary = pd.DataFrame(
        [
            {
                "warmup_steps": 100,
                "layer_index": 1,
                "parameter": "conv1.weight",
                "shape": "64x3x3x3",
                "mean_gradient_nuclear_rank": 2.0,
                "mean_alignment_ratio_spectral_over_fro": 1.0,
                "mean_step_size_ratio_spectral_over_fro": 1.0,
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": 1.0,
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": 0.5,
                "mean_tail_accuracy_before": 0.5,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": 0.75,
            },
            {
                "warmup_steps": 100,
                "layer_index": 2,
                "parameter": "layer2.0.downsample.0.weight",
                "shape": "128x64x1x1",
                "mean_gradient_nuclear_rank": 3.0,
                "mean_alignment_ratio_spectral_over_fro": 1.0,
                "mean_step_size_ratio_spectral_over_fro": 1.0,
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": 1.0,
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": 0.25,
                "mean_tail_accuracy_before": 0.5,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": 0.5,
            },
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "seed": 0,
                "geometry": "frobenius",
                "warmup_steps": 100,
                "layer_index": 2,
                "parameter": "layer2.0.downsample.0.weight",
                "shape": "128x64x1x1",
                "scaled_jvp_tail_drift_fro": 2.0,
            },
            {
                "seed": 1,
                "geometry": "frobenius",
                "warmup_steps": 100,
                "layer_index": 2,
                "parameter": "layer2.0.downsample.0.weight",
                "shape": "128x64x1x1",
                "scaled_jvp_tail_drift_fro": 8.0,
            },
            {
                "seed": 0,
                "geometry": "frobenius",
                "warmup_steps": 100,
                "layer_index": 1,
                "parameter": "conv1.weight",
                "shape": "64x3x3x3",
                "scaled_jvp_tail_drift_fro": 1.0,
            },
            {
                "seed": 0,
                "geometry": "spectral",
                "warmup_steps": 100,
                "layer_index": 2,
                "parameter": "layer2.0.downsample.0.weight",
                "shape": "128x64x1x1",
                "scaled_jvp_tail_drift_fro": 100.0,
            },
        ]
    )
    layer_path = tmp_path / "layer_summary.csv"
    metrics_path = tmp_path / "metrics.csv"
    layer_summary.to_csv(layer_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    frame = load_axis_frame(layer_path, metrics_path)
    downsample = frame[frame["parameter"].eq("layer2.0.downsample.0.weight")].iloc[0]

    assert math.isclose(
        downsample["geomean_scaled_jvp_tail_drift_fro_frobenius_direction"],
        4.0,
    )
    assert math.isclose(downsample["log_scaled_jvp_fro_amplitude"], math.log(4.0))
    assert downsample["transport_stage"] == "layer2"
    assert downsample["transport_is_downsample"] == 1.0
