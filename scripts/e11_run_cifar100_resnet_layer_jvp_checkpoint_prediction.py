from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    dataset_display_name,
    model_display_name,
)
from e11_condition_geometry.reporting import fmt, markdown_table
from e11_condition_geometry.statistics import ci95, corr_ci95, log_ratio_ci95


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md")
DEFAULT_WARMUP_STEPS = (2000, 5000, 10000)
PREDICTORS = {
    "source_observed_drift_ratio": "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
    "source_scaled_jvp_ratio": "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "source_unit_jvp_ratio": "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "source_alignment_ratio": "mean_alignment_ratio_spectral_over_fro",
    "source_gradient_nuclear_rank": "mean_gradient_nuclear_rank",
    "architecture_early_layer_prior": "early_layer_prior",
}
PRE_REGISTERED_PREDICTORS = {
    "source_scaled_jvp_ratio",
    "source_unit_jvp_ratio",
    "source_alignment_ratio",
    "source_gradient_nuclear_rank",
}
POSITIVE_CONTROL_PREDICTORS = {
    "source_observed_drift_ratio",
    "architecture_early_layer_prior",
}


def load_layer_jvp_module():
    path = ROOT / "scripts" / "e11_run_cifar100_resnet_layer_jvp_tail_quality.py"
    spec = importlib.util.spec_from_file_location("e11_layer_jvp_tail_quality", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load layer JVP module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_warmup_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not steps:
        raise argparse.ArgumentTypeError("expected at least one warmup step")
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("warmup steps must be positive")
    return steps


def _finite_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-300))


def add_prediction_control_columns(layer_summary: pd.DataFrame) -> pd.DataFrame:
    layer_summary = layer_summary.copy()
    layer_summary["early_layer_prior"] = 1.0 / layer_summary["layer_index"].astype(float).clip(lower=1.0)
    return layer_summary


def log_positive(values: pd.Series) -> pd.Series:
    return values.astype(float).clip(lower=1e-300).map(math.log)


def residualize_against_early_layer_prior(frame: pd.DataFrame, column: str) -> tuple[float, float, pd.Series]:
    x = log_positive(frame["early_layer_prior"])
    y = log_positive(frame[column])
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    denominator = float((x_centered * x_centered).sum())
    slope = 0.0 if denominator <= 0.0 else float((x_centered * (y - y_mean)).sum() / denominator)
    intercept = y_mean - slope * x_mean
    return intercept, slope, y - (intercept + slope * x)


def residual_from_fit(frame: pd.DataFrame, column: str, intercept: float, slope: float) -> pd.Series:
    return log_positive(frame[column]) - (float(intercept) + float(slope) * log_positive(frame["early_layer_prior"]))


def layer_checkpoint_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (warmup_steps, layer_index, parameter), group in paired.groupby(
        ["warmup_steps", "layer_index", "parameter"],
        observed=True,
        sort=True,
    ):
        unit, unit_low, unit_high = log_ratio_ci95(group["jvp_tail_drift_sq_ratio_spectral_over_fro"])
        scaled, scaled_low, scaled_high = log_ratio_ci95(group["scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"])
        observed, observed_low, observed_high = log_ratio_ci95(group["observed_tail_drift_sq_ratio_spectral_over_fro"])
        rows.append(
            {
                "warmup_steps": int(warmup_steps),
                "layer_index": int(layer_index),
                "parameter": str(parameter),
                "shape": str(group["shape"].iloc[0]),
                "seeds": int(group["seed"].nunique()),
                "mean_gradient_nuclear_rank": float(group["gradient_nuclear_rank"].mean()),
                "mean_alignment_ratio_spectral_over_fro": float(
                    group["alignment_ratio_spectral_over_fro"].mean()
                ),
                "mean_step_size_ratio_spectral_over_fro": float(
                    group["step_size_ratio_spectral_over_fro"].mean()
                ),
                "mean_tail_accuracy_before": float(group["tail_accuracy_before"].mean()),
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": unit,
                "jvp_tail_drift_sq_ratio_ci95_low": unit_low,
                "jvp_tail_drift_sq_ratio_ci95_high": unit_high,
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": scaled,
                "scaled_jvp_tail_drift_sq_ratio_ci95_low": scaled_low,
                "scaled_jvp_tail_drift_sq_ratio_ci95_high": scaled_high,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": observed,
                "observed_tail_drift_sq_ratio_ci95_low": observed_low,
                "observed_tail_drift_sq_ratio_ci95_high": observed_high,
                "spectral_less_observed_tail_drift_fraction": float(
                    group["spectral_less_observed_tail_drift"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_checkpoints(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for warmup_steps, group in paired.groupby("warmup_steps", observed=True, sort=True):
        unit, unit_low, unit_high = log_ratio_ci95(group["jvp_tail_drift_sq_ratio_spectral_over_fro"])
        scaled, scaled_low, scaled_high = log_ratio_ci95(group["scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"])
        observed, observed_low, observed_high = log_ratio_ci95(group["observed_tail_drift_sq_ratio_spectral_over_fro"])
        rows.append(
            {
                "warmup_steps": int(warmup_steps),
                "seeds": int(group["seed"].nunique()),
                "parameters": int(group["parameter"].nunique()),
                "paired_points": int(len(group)),
                "mean_tail_accuracy_before": float(group["tail_accuracy_before"].mean()),
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": unit,
                "jvp_tail_drift_sq_ratio_ci95_low": unit_low,
                "jvp_tail_drift_sq_ratio_ci95_high": unit_high,
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": scaled,
                "scaled_jvp_tail_drift_sq_ratio_ci95_low": scaled_low,
                "scaled_jvp_tail_drift_sq_ratio_ci95_high": scaled_high,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": observed,
                "observed_tail_drift_sq_ratio_ci95_low": observed_low,
                "observed_tail_drift_sq_ratio_ci95_high": observed_high,
                "spectral_less_scaled_jvp_tail_drift_fraction": float(
                    group["spectral_less_scaled_jvp_tail_drift"].mean()
                ),
                "spectral_less_observed_tail_drift_fraction": float(
                    group["spectral_less_observed_tail_drift"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def prediction_pairs(layer_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    layer_summary = add_prediction_control_columns(layer_summary)
    checkpoints = sorted(int(value) for value in layer_summary["warmup_steps"].unique())
    for train_steps in checkpoints:
        train = layer_summary[layer_summary["warmup_steps"].eq(train_steps)].copy()
        for test_steps in checkpoints:
            if test_steps == train_steps:
                continue
            test = layer_summary[layer_summary["warmup_steps"].eq(test_steps)].copy()
            joined = train.merge(
                test[
                    [
                        "parameter",
                        "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
                        "observed_tail_drift_sq_ratio_ci95_high",
                    ]
                ],
                on="parameter",
                how="inner",
                suffixes=("_source", "_target"),
            )
            target = joined["geomean_observed_tail_drift_sq_ratio_spectral_over_fro_target"].clip(lower=1e-300)
            target_log = target.map(math.log)
            target_below_one = target < 1.0
            target_top_k = set(
                joined.nlargest(min(5, len(joined)), "geomean_observed_tail_drift_sq_ratio_spectral_over_fro_target")[
                    "parameter"
                ]
            )
            for predictor_name, column in PREDICTORS.items():
                source_column = f"{column}_source" if f"{column}_source" in joined.columns else column
                source = joined[source_column].astype(float).clip(lower=1e-300)
                source_log = source.map(math.log)
                spearman, spearman_low, spearman_high, points = corr_ci95(source_log, target_log, method="spearman")
                pearson, pearson_low, pearson_high, _ = corr_ci95(source_log, target_log, method="pearson")
                source_below_one = source < 1.0
                sign_accuracy = float((source_below_one == target_below_one).mean())
                predicted_top_k = set(joined.nlargest(min(5, len(joined)), source_column)["parameter"])
                rows.append(
                    {
                        "source_warmup_steps": int(train_steps),
                        "target_warmup_steps": int(test_steps),
                        "predictor": predictor_name,
                        "predictor_family": (
                            "pre_registered"
                            if predictor_name in PRE_REGISTERED_PREDICTORS
                            else "positive_control"
                        ),
                        "points": int(points),
                        "spearman_log_predictor_vs_log_target_observed": spearman,
                        "spearman_ci95_low": spearman_low,
                        "spearman_ci95_high": spearman_high,
                        "pearson_log_predictor_vs_log_target_observed": pearson,
                        "pearson_ci95_low": pearson_low,
                        "pearson_ci95_high": pearson_high,
                        "threshold_below_one_accuracy": sign_accuracy,
                        "target_observed_below_one_fraction": float(target_below_one.mean()),
                        "source_predictor_below_one_fraction": float(source_below_one.mean()),
                        "top5_risk_overlap_fraction": len(predicted_top_k & target_top_k)
                        / max(len(target_top_k), 1),
                    }
                )
    return pd.DataFrame(rows)


def summarize_prediction_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor, group in pairs.groupby("predictor", observed=True, sort=False):
        spearman, spearman_low, spearman_high = ci95(
            group["spearman_log_predictor_vs_log_target_observed"]
        )
        pearson, pearson_low, pearson_high = ci95(group["pearson_log_predictor_vs_log_target_observed"])
        top5, top5_low, top5_high = ci95(group["top5_risk_overlap_fraction"])
        threshold, threshold_low, threshold_high = ci95(group["threshold_below_one_accuracy"])
        rows.append(
            {
                "predictor": predictor,
                "predictor_family": str(group["predictor_family"].iloc[0]),
                "checkpoint_transfer_pairs": int(len(group)),
                "mean_spearman_log_predictor_vs_log_target_observed": spearman,
                "spearman_ci95_low": spearman_low,
                "spearman_ci95_high": spearman_high,
                "mean_pearson_log_predictor_vs_log_target_observed": pearson,
                "pearson_ci95_low": pearson_low,
                "pearson_ci95_high": pearson_high,
                "mean_top5_risk_overlap_fraction": top5,
                "top5_risk_overlap_ci95_low": top5_low,
                "top5_risk_overlap_ci95_high": top5_high,
                "mean_threshold_below_one_accuracy": threshold,
                "threshold_below_one_accuracy_ci95_low": threshold_low,
                "threshold_below_one_accuracy_ci95_high": threshold_high,
            }
        )
    return pd.DataFrame(rows)


RESIDUAL_PREDICTORS = {
    "source_observed_residual": "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
    "source_scaled_jvp_residual": "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "source_unit_jvp_residual": "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "source_gradient_nuclear_rank_residual": "mean_gradient_nuclear_rank",
}


def residual_prediction_pairs(layer_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    layer_summary = add_prediction_control_columns(layer_summary)
    checkpoints = sorted(int(value) for value in layer_summary["warmup_steps"].unique())
    for train_steps in checkpoints:
        train = layer_summary[layer_summary["warmup_steps"].eq(train_steps)].copy()
        observed_intercept, observed_slope, source_observed_residual = residualize_against_early_layer_prior(
            train,
            "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
        )
        source_residuals = {
            "source_observed_residual": source_observed_residual,
        }
        for predictor_name, column in RESIDUAL_PREDICTORS.items():
            if predictor_name == "source_observed_residual":
                continue
            _intercept, _slope, source_residuals[predictor_name] = residualize_against_early_layer_prior(
                train,
                column,
            )
        residual_frame = pd.DataFrame({"parameter": train["parameter"].to_numpy()})
        for predictor_name, values in source_residuals.items():
            residual_frame[predictor_name] = values.to_numpy(dtype=float)
        for test_steps in checkpoints:
            if test_steps == train_steps:
                continue
            test = layer_summary[layer_summary["warmup_steps"].eq(test_steps)].copy()
            joined = train[["parameter"]].merge(test, on="parameter", how="inner")
            target_residual = residual_from_fit(
                joined,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
                observed_intercept,
                observed_slope,
            )
            target_top_k = set(
                pd.DataFrame({"parameter": joined["parameter"], "target_residual": target_residual})
                .nlargest(min(5, len(joined)), "target_residual")["parameter"]
            )
            source_joined = residual_frame.merge(
                pd.DataFrame({"parameter": joined["parameter"], "target_residual": target_residual}),
                on="parameter",
                how="inner",
            )
            for predictor_name in RESIDUAL_PREDICTORS:
                source = source_joined[predictor_name].astype(float)
                target = source_joined["target_residual"].astype(float)
                spearman, spearman_low, spearman_high, points = corr_ci95(source, target, method="spearman")
                pearson, pearson_low, pearson_high, _ = corr_ci95(source, target, method="pearson")
                predicted_top_k = set(source_joined.nlargest(min(5, len(source_joined)), predictor_name)["parameter"])
                rows.append(
                    {
                        "source_warmup_steps": int(train_steps),
                        "target_warmup_steps": int(test_steps),
                        "predictor": predictor_name,
                        "predictor_family": (
                            "positive_control"
                            if predictor_name == "source_observed_residual"
                            else "architecture_adjusted"
                        ),
                        "points": int(points),
                        "source_depth_fit_intercept": float(observed_intercept),
                        "source_depth_fit_slope": float(observed_slope),
                        "spearman_residual_predictor_vs_residual_target_observed": spearman,
                        "spearman_ci95_low": spearman_low,
                        "spearman_ci95_high": spearman_high,
                        "pearson_residual_predictor_vs_residual_target_observed": pearson,
                        "pearson_ci95_low": pearson_low,
                        "pearson_ci95_high": pearson_high,
                        "top5_residual_risk_overlap_fraction": len(predicted_top_k & target_top_k)
                        / max(len(target_top_k), 1),
                    }
                )
    return pd.DataFrame(rows)


def summarize_residual_prediction_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor, group in pairs.groupby("predictor", observed=True, sort=False):
        spearman, spearman_low, spearman_high = ci95(
            group["spearman_residual_predictor_vs_residual_target_observed"]
        )
        pearson, pearson_low, pearson_high = ci95(
            group["pearson_residual_predictor_vs_residual_target_observed"]
        )
        top5, top5_low, top5_high = ci95(group["top5_residual_risk_overlap_fraction"])
        rows.append(
            {
                "predictor": predictor,
                "predictor_family": str(group["predictor_family"].iloc[0]),
                "checkpoint_transfer_pairs": int(len(group)),
                "mean_spearman_residual_predictor_vs_residual_target_observed": spearman,
                "spearman_ci95_low": spearman_low,
                "spearman_ci95_high": spearman_high,
                "mean_pearson_residual_predictor_vs_residual_target_observed": pearson,
                "pearson_ci95_low": pearson_low,
                "pearson_ci95_high": pearson_high,
                "mean_top5_residual_risk_overlap_fraction": top5,
                "top5_residual_risk_overlap_ci95_low": top5_low,
                "top5_residual_risk_overlap_ci95_high": top5_high,
            }
        )
    return pd.DataFrame(rows)


def run_checkpoint_prediction(
    base_config: Cifar100ResNetOneStepConfig,
    *,
    warmup_steps: tuple[int, ...],
    jvp_epsilon: float,
    max_matrix_parameters: int | None,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer_jvp = load_layer_jvp_module()
    metric_frames = []
    paired_frames = []

    for warmup_step in warmup_steps:
        config = replace(base_config, warmup_steps=int(warmup_step))
        if progress:
            print(f"[resnet-layer-jvp-predict] warmup_steps={warmup_step}", flush=True)
        metrics, paired, _summary, _overall = layer_jvp.run_layer_jvp_probe(
            config,
            jvp_epsilon=jvp_epsilon,
            max_matrix_parameters=max_matrix_parameters,
            progress=progress,
        )
        for frame in (metrics, paired):
            frame["warmup_steps"] = int(warmup_step)
        metric_frames.append(metrics)
        paired_frames.append(paired)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = pd.concat(metric_frames, ignore_index=True)
    paired = pd.concat(paired_frames, ignore_index=True)
    layer_summary = layer_checkpoint_table(paired)
    checkpoint_summary = summarize_checkpoints(paired)
    prediction_pair_summary = prediction_pairs(layer_summary)
    prediction_summary = summarize_prediction_pairs(prediction_pair_summary)
    return metrics, paired, layer_summary, checkpoint_summary, prediction_pair_summary, prediction_summary


def write_figure(
    checkpoint_summary: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    residual_prediction_summary: pd.DataFrame,
    figure_dir: Path,
    *,
    title: str = "CIFAR ResNet all-layer JVP checkpoint-prediction benchmark",
) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))

    ordered = checkpoint_summary.sort_values("warmup_steps")
    x = ordered["warmup_steps"].to_numpy(dtype=float)
    for prefix, label, color in [
        ("scaled_jvp", "scaled JVP", "#0072B2"),
        ("observed", "observed", "#D55E00"),
    ]:
        y = ordered[f"geomean_{prefix}_tail_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
        low = ordered[f"{prefix}_tail_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
        high = ordered[f"{prefix}_tail_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
        axes[0].errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("tail-rich ResNet warmup steps")
    axes[0].set_ylabel("spectral/Fro squared drift ratio")
    axes[0].set_title("Checkpoint-level all-layer JVP readout")
    axes[0].legend(frameon=False)

    ordered_pred = prediction_summary.sort_values("predictor")
    y_positions = list(range(len(ordered_pred)))
    estimates = ordered_pred["mean_spearman_log_predictor_vs_log_target_observed"].to_numpy(dtype=float)
    low = ordered_pred["spearman_ci95_low"].to_numpy(dtype=float)
    high = ordered_pred["spearman_ci95_high"].to_numpy(dtype=float)
    colors = [
        "#009E73" if predictor in PRE_REGISTERED_PREDICTORS else "#CC79A7"
        for predictor in ordered_pred["predictor"]
    ]
    axes[1].barh(y_positions, estimates, color=colors, alpha=0.85)
    axes[1].errorbar(
        estimates,
        y_positions,
        xerr=[estimates - low, high - estimates],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(ordered_pred["predictor"].tolist(), fontsize=8)
    axes[1].set_xlabel("held-out-checkpoint Spearman")
    axes[1].set_title("Layer-risk transfer")

    ordered_residual = residual_prediction_summary.sort_values("predictor")
    y_positions = list(range(len(ordered_residual)))
    estimates = ordered_residual[
        "mean_spearman_residual_predictor_vs_residual_target_observed"
    ].to_numpy(dtype=float)
    low = ordered_residual["spearman_ci95_low"].to_numpy(dtype=float)
    high = ordered_residual["spearman_ci95_high"].to_numpy(dtype=float)
    residual_colors = [
        "#CC79A7" if family == "positive_control" else "#56B4E9"
        for family in ordered_residual["predictor_family"]
    ]
    axes[2].barh(y_positions, estimates, color=residual_colors, alpha=0.88)
    axes[2].errorbar(
        estimates,
        y_positions,
        xerr=[estimates - low, high - estimates],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels(ordered_residual["predictor"].tolist(), fontsize=8)
    axes[2].set_xlabel("depth-adjusted Spearman")
    axes[2].set_title("Residual layer-risk transfer")

    fig.suptitle(title)
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_layer_jvp_checkpoint_prediction.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    base_config: Cifar100ResNetOneStepConfig,
    warmup_steps: tuple[int, ...],
    checkpoint_summary: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    residual_prediction_summary: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
    *,
    jvp_epsilon: float,
) -> None:
    scaled_row = prediction_summary[prediction_summary["predictor"].eq("source_scaled_jvp_ratio")].iloc[0]
    rank_row = prediction_summary[prediction_summary["predictor"].eq("source_gradient_nuclear_rank")].iloc[0]
    observed_row = prediction_summary[
        prediction_summary["predictor"].eq("source_observed_drift_ratio")
    ].iloc[0]
    early_layer_row = prediction_summary[
        prediction_summary["predictor"].eq("architecture_early_layer_prior")
    ].iloc[0]
    residual_observed_row = residual_prediction_summary[
        residual_prediction_summary["predictor"].eq("source_observed_residual")
    ].iloc[0]
    residual_scaled_row = residual_prediction_summary[
        residual_prediction_summary["predictor"].eq("source_scaled_jvp_residual")
    ].iloc[0]
    residual_nrank_row = residual_prediction_summary[
        residual_prediction_summary["predictor"].eq("source_gradient_nuclear_rank_residual")
    ].iloc[0]
    lines = [
        f"# E11 {dataset_display_name(base_config.dataset_name)} {model_display_name(base_config.model_arch)} All-Layer JVP Checkpoint-Prediction Benchmark",
        "",
        "This diagnostic turns the single-checkpoint all-layer JVP bridge into a",
        f"checkpoint-transfer prediction test. For each tail-rich {model_display_name(base_config.model_arch)} checkpoint,",
        "the script probes every Conv/Linear matrix weight, computes unit-JVP,",
        "matched-gain scaled-JVP, and observed layer-only drift ratios, then asks",
        "whether layer scores measured at one checkpoint predict observed layer risk",
        "at held-out checkpoints without fitting a new model. The original",
        "pre-registered predictors are retained, and two positive controls are",
        "added: source-checkpoint observed drift and an early-layer architecture",
        "prior. These controls test whether held-out layer-risk ordering is",
        "predictable at all, rather than attributing every failure to target noise.",
        "",
        "The same artifact also reports an architecture-adjusted residual test.",
        "For each source checkpoint, it fits source observed log drift from",
        "log early-layer prior, applies that source fit to the held-out target",
        "checkpoint, and asks which source residual scores predict target",
        "residual risk. This avoids fitting the depth correction on the target",
        "checkpoint itself.",
        "",
        f"- Warmup checkpoints: {', '.join(str(step) for step in warmup_steps)}",
        f"- Dataset/model: {dataset_display_name(base_config.dataset_name)} / {model_display_name(base_config.model_arch)}",
        f"- Seeds per checkpoint: {len(base_config.seeds)}",
        f"- Head train examples per class: {base_config.head_train_per_class}",
        f"- Tail train examples per class: {base_config.tail_train_per_class}",
        f"- Tail eval examples per class: {base_config.tail_eval_per_class}",
        f"- Target head first-order gain: {base_config.target_head_gain_fraction} * head-batch loss",
        f"- Finite-difference JVP epsilon: {jvp_epsilon}",
        f"- Device/dtype request: {base_config.device}/{base_config.dtype}",
        "",
        f"![CIFAR ResNet all-layer JVP checkpoint prediction](../{figure_path.as_posix()})",
        "",
        "## Checkpoint Summary",
        "",
        markdown_table(
            checkpoint_summary,
            [
                "warmup_steps",
                "seeds",
                "parameters",
                "paired_points",
                "mean_tail_accuracy_before",
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
                "scaled_jvp_tail_drift_sq_ratio_ci95_low",
                "scaled_jvp_tail_drift_sq_ratio_ci95_high",
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro",
                "observed_tail_drift_sq_ratio_ci95_low",
                "observed_tail_drift_sq_ratio_ci95_high",
            ],
        ),
        "",
        "## Held-Out Checkpoint Prediction Summary",
        "",
        markdown_table(
            prediction_summary,
            [
                "predictor",
                "predictor_family",
                "checkpoint_transfer_pairs",
                "mean_spearman_log_predictor_vs_log_target_observed",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "mean_top5_risk_overlap_fraction",
                "top5_risk_overlap_ci95_low",
                "top5_risk_overlap_ci95_high",
            ],
        ),
        "",
        "## Readout",
        "",
        f"- Source observed-drift positive control: Spearman "
        f"{fmt(observed_row['mean_spearman_log_predictor_vs_log_target_observed'])} "
        f"[{fmt(observed_row['spearman_ci95_low'])}, {fmt(observed_row['spearman_ci95_high'])}], "
        f"top-5 risk overlap {fmt(observed_row['mean_top5_risk_overlap_fraction'])}.",
        f"- Early-layer architecture prior: Spearman "
        f"{fmt(early_layer_row['mean_spearman_log_predictor_vs_log_target_observed'])} "
        f"[{fmt(early_layer_row['spearman_ci95_low'])}, {fmt(early_layer_row['spearman_ci95_high'])}], "
        f"top-5 risk overlap {fmt(early_layer_row['mean_top5_risk_overlap_fraction'])}.",
        f"- Pre-registered scaled-JVP transfer predictor: Spearman "
        f"{fmt(scaled_row['mean_spearman_log_predictor_vs_log_target_observed'])} "
        f"[{fmt(scaled_row['spearman_ci95_low'])}, {fmt(scaled_row['spearman_ci95_high'])}] "
        f"over {int(scaled_row['checkpoint_transfer_pairs'])} directed checkpoint-transfer pairs.",
        f"- Rank-only source predictor: Spearman "
        f"{fmt(rank_row['mean_spearman_log_predictor_vs_log_target_observed'])} "
        f"[{fmt(rank_row['spearman_ci95_low'])}, {fmt(rank_row['spearman_ci95_high'])}].",
        f"- Architecture-adjusted source observed residual: Spearman "
        f"{fmt(residual_observed_row['mean_spearman_residual_predictor_vs_residual_target_observed'])} "
        f"[{fmt(residual_observed_row['spearman_ci95_low'])}, {fmt(residual_observed_row['spearman_ci95_high'])}], "
        f"top-5 residual overlap {fmt(residual_observed_row['mean_top5_residual_risk_overlap_fraction'])}.",
        f"- Architecture-adjusted scaled-JVP residual: Spearman "
        f"{fmt(residual_scaled_row['mean_spearman_residual_predictor_vs_residual_target_observed'])} "
        f"[{fmt(residual_scaled_row['spearman_ci95_low'])}, {fmt(residual_scaled_row['spearman_ci95_high'])}].",
        f"- Architecture-adjusted gradient-rank residual: Spearman "
        f"{fmt(residual_nrank_row['mean_spearman_residual_predictor_vs_residual_target_observed'])} "
        f"[{fmt(residual_nrank_row['spearman_ci95_low'])}, {fmt(residual_nrank_row['spearman_ci95_high'])}].",
        "",
        "Interpretation: this is a checkpoint-transfer mechanism benchmark. The",
        "positive controls show that layer-risk ordering is stable enough to",
        "transfer across the tested tail-rich checkpoints. The current scaled-JVP",
        "readout transfers the below-one direction but not the layer ranking, so",
        "the missing ingredient is in the measurable condition score rather than",
        "only in target-checkpoint noise. The residual benchmark further shows",
        "that observed source residuals transfer after removing the early-layer",
        "prior, while the scaled-JVP residual remains inverted. It is still not a standard long-tailed",
        "classification benchmark or a final optimizer-performance result.",
        "",
        "Artifacts:",
        f"- [metrics.csv](../{(output_dir / 'metrics.csv').as_posix()})",
        f"- [paired_metrics.csv](../{(output_dir / 'paired_metrics.csv').as_posix()})",
        f"- [layer_summary.csv](../{(output_dir / 'layer_summary.csv').as_posix()})",
        f"- [checkpoint_summary.csv](../{(output_dir / 'checkpoint_summary.csv').as_posix()})",
        f"- [prediction_pairs.csv](../{(output_dir / 'prediction_pairs.csv').as_posix()})",
        f"- [prediction_summary.csv](../{(output_dir / 'prediction_summary.csv').as_posix()})",
        f"- [residual_prediction_pairs.csv](../{(output_dir / 'residual_prediction_pairs.csv').as_posix()})",
        f"- [residual_prediction_summary.csv](../{(output_dir / 'residual_prediction_summary.csv').as_posix()})",
        f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
    ]
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-name", choices=["CIFAR100", "CIFAR10"], default=None)
    parser.add_argument("--model-arch", choices=["resnet18", "resnet34", "resnet50"], default=None)
    parser.add_argument("--head-classes", default=None)
    parser.add_argument("--tail-classes", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup-steps", type=parse_warmup_steps, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--target-head-gain-fraction", type=float, default=0.005)
    parser.add_argument("--head-train-per-class", type=int, default=None)
    parser.add_argument("--tail-train-per-class", type=int, default=None)
    parser.add_argument("--tail-eval-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--jvp-epsilon", type=float, default=1e-4)
    parser.add_argument("--max-matrix-parameters", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--discussion-path", type=Path, default=DEFAULT_DISCUSSION_PATH)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing layer/checkpoint summaries and refresh prediction CSVs, figure, and discussion.",
    )
    return parser.parse_args()


def parse_class_list(value: str) -> tuple[int, ...]:
    classes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not classes:
        raise argparse.ArgumentTypeError("class list must not be empty")
    return classes


def config_from_args(args: argparse.Namespace) -> tuple[Cifar100ResNetOneStepConfig, tuple[int, ...], int | None]:
    config = Cifar100ResNetOneStepConfig()
    if args.smoke:
        config = replace(
            config,
            seeds=(0,),
            head_classes=tuple(range(5)),
            tail_classes=tuple(range(5, 10)),
            head_train_per_class=20,
            tail_train_per_class=5,
            tail_eval_per_class=5,
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
        warmup_steps = (2, 4)
        max_matrix_parameters = 2 if args.max_matrix_parameters is None else args.max_matrix_parameters
    else:
        config = replace(config, seeds=tuple(range(args.seeds)))
        warmup_steps = args.warmup_steps
        max_matrix_parameters = args.max_matrix_parameters
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.dataset_name is not None:
        config = replace(config, dataset_name=args.dataset_name)
    if args.model_arch is not None:
        config = replace(config, model_arch=args.model_arch)
    if args.head_classes is not None:
        config = replace(config, head_classes=parse_class_list(args.head_classes))
    if args.tail_classes is not None:
        config = replace(config, tail_classes=parse_class_list(args.tail_classes))
    if args.target_head_gain_fraction is not None:
        config = replace(config, target_head_gain_fraction=args.target_head_gain_fraction)
    if args.head_train_per_class is not None:
        config = replace(config, head_train_per_class=args.head_train_per_class)
    if args.tail_train_per_class is not None:
        config = replace(config, tail_train_per_class=args.tail_train_per_class)
    if args.tail_eval_per_class is not None:
        config = replace(config, tail_eval_per_class=args.tail_eval_per_class)
    if args.batch_size is not None:
        config = replace(config, warmup_batch_size=args.batch_size)
    if args.head_batch_size is not None:
        config = replace(config, head_batch_size=args.head_batch_size)
    if args.lr is not None:
        config = replace(config, lr=args.lr)
    if args.download is not None:
        config = replace(config, download=args.download)
    return config, warmup_steps, max_matrix_parameters


def main() -> None:
    args = parse_args()
    config, warmup_steps, max_matrix_parameters = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing:
        metrics = pd.read_csv(args.output_dir / "metrics.csv")
        paired = pd.read_csv(args.output_dir / "paired_metrics.csv")
        layer_summary = pd.read_csv(args.output_dir / "layer_summary.csv")
        checkpoint_summary = pd.read_csv(args.output_dir / "checkpoint_summary.csv")
        config_path = args.output_dir / "config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            config = Cifar100ResNetOneStepConfig(**existing.get("base_config", {}))
            warmup_steps = tuple(int(step) for step in existing.get("warmup_steps", warmup_steps))
            args.jvp_epsilon = float(existing.get("jvp_epsilon", args.jvp_epsilon))
            max_matrix_parameters = existing.get("max_matrix_parameters", max_matrix_parameters)
    else:
        metrics, paired, layer_summary, checkpoint_summary, prediction_pair_summary, prediction_summary = (
            run_checkpoint_prediction(
                config,
                warmup_steps=warmup_steps,
                jvp_epsilon=args.jvp_epsilon,
                max_matrix_parameters=max_matrix_parameters,
                progress=args.progress,
            )
        )
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        paired.to_csv(args.output_dir / "paired_metrics.csv", index=False)
        layer_summary.to_csv(args.output_dir / "layer_summary.csv", index=False)
        checkpoint_summary.to_csv(args.output_dir / "checkpoint_summary.csv", index=False)
    prediction_pair_summary = prediction_pairs(layer_summary)
    prediction_summary = summarize_prediction_pairs(prediction_pair_summary)
    residual_prediction_pair_summary = residual_prediction_pairs(layer_summary)
    residual_prediction_summary = summarize_residual_prediction_pairs(residual_prediction_pair_summary)
    prediction_pair_summary.to_csv(args.output_dir / "prediction_pairs.csv", index=False)
    prediction_summary.to_csv(args.output_dir / "prediction_summary.csv", index=False)
    residual_prediction_pair_summary.to_csv(args.output_dir / "residual_prediction_pairs.csv", index=False)
    residual_prediction_summary.to_csv(args.output_dir / "residual_prediction_summary.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "base_config": asdict(config),
                "warmup_steps": list(warmup_steps),
                "jvp_epsilon": float(args.jvp_epsilon),
                "max_matrix_parameters": max_matrix_parameters,
                "predictors": PREDICTORS,
                "pre_registered_predictors": {
                    key: value for key, value in PREDICTORS.items() if key in PRE_REGISTERED_PREDICTORS
                },
                "positive_control_predictors": {
                    key: value for key, value in PREDICTORS.items() if key in POSITIVE_CONTROL_PREDICTORS
                },
                "residual_predictors": RESIDUAL_PREDICTORS,
                "residual_adjustment": "source_checkpoint_log_observed_drift_on_log_early_layer_prior",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(
        checkpoint_summary,
        prediction_summary,
        residual_prediction_summary,
        args.figure_dir,
        title=(
            f"{dataset_display_name(config.dataset_name)} {model_display_name(config.model_arch)} "
            "all-layer JVP checkpoint-prediction benchmark"
        ),
    )
    write_discussion(
        config,
        warmup_steps,
        checkpoint_summary,
        prediction_summary,
        residual_prediction_summary,
        figure_path,
        args.output_dir,
        args.discussion_path,
        jvp_epsilon=args.jvp_epsilon,
    )
    print(
        f"saved {dataset_display_name(config.dataset_name)} {model_display_name(config.model_arch)} "
        f"layer-JVP checkpoint prediction to {args.output_dir}"
    )
    print(
        f"metric rows={len(metrics)}, paired rows={len(paired)}, "
        f"layer summaries={len(layer_summary)}, checkpoint summaries={len(checkpoint_summary)}"
    )
    print(f"figure: {figure_path}")
    print(f"discussion: {args.discussion_path}")
    print(prediction_summary.to_string(index=False))
    print(residual_prediction_summary.to_string(index=False))


if __name__ == "__main__":
    main()
