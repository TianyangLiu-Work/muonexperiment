from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown
from e11_condition_geometry.statistics import ci95, corr_ci95


INPUT_LAYER_SUMMARY = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv")
PROTOCOL_DIR = Path("results/e11_cifar100_resnet_condition_score_protocol")
OUTPUT_DIR = Path("results/e11_cifar100_resnet_condition_score_next")
FIGURE_DIR = Path("figures/e11_cifar100_resnet_condition_score_next")
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_condition_score_next.md")
HELDOUT_EVAL_DIR = OUTPUT_DIR / "heldout_score_evaluation"
HELDOUT_SCORE_SUMMARY = HELDOUT_EVAL_DIR / "heldout_score_summary.csv"
HELDOUT_GATE_REPORT = HELDOUT_EVAL_DIR / "heldout_gate_report.csv"
HELDOUT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md")
RIDGE_ALPHA = 1.0

FEATURE_COLUMNS = {
    "log_gradient_nuclear_rank": "mean_gradient_nuclear_rank",
    "log_alignment_ratio": "mean_alignment_ratio_spectral_over_fro",
    "log_step_size_ratio": "mean_step_size_ratio_spectral_over_fro",
    "log_unit_jvp_ratio": "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "log_scaled_jvp_ratio": "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
    "log_tail_accuracy_before": "mean_tail_accuracy_before",
}


def log_positive(values: pd.Series) -> pd.Series:
    return values.astype(float).clip(lower=1e-300).map(math.log)


def add_protocol_features(layer_summary: pd.DataFrame) -> pd.DataFrame:
    frame = layer_summary.copy()
    frame["early_layer_prior"] = 1.0 / frame["layer_index"].astype(float).clip(lower=1.0)
    frame["log_early_layer_prior"] = log_positive(frame["early_layer_prior"])
    frame["log_observed"] = log_positive(frame["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"])
    for feature_name, column in FEATURE_COLUMNS.items():
        frame[feature_name] = log_positive(frame[column])
    return frame


def fit_depth_baseline(source: pd.DataFrame) -> tuple[float, float, pd.Series]:
    x = source["log_early_layer_prior"].astype(float)
    y = source["log_observed"].astype(float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    denominator = float((x_centered * x_centered).sum())
    slope = 0.0 if denominator <= 0.0 else float((x_centered * (y - y_mean)).sum() / denominator)
    intercept = y_mean - slope * x_mean
    return intercept, slope, y - (intercept + slope * x)


def target_residual_from_source_depth(target: pd.DataFrame, intercept: float, slope: float) -> pd.Series:
    return target["log_observed"].astype(float) - (float(intercept) + float(slope) * target["log_early_layer_prior"].astype(float))


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)


def fit_calibrated_residual_score(source: pd.DataFrame, alpha: float = RIDGE_ALPHA) -> dict[str, object]:
    depth_intercept, depth_slope, source_residual = fit_depth_baseline(source)
    x = feature_matrix(source)
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    x_standardized = (x - feature_mean) / feature_std
    y = source_residual.to_numpy(dtype=float)
    y_mean = float(y.mean())
    y_centered = y - y_mean
    regularized = x_standardized.T @ x_standardized + float(alpha) * np.eye(x_standardized.shape[1])
    beta = np.linalg.solve(regularized, x_standardized.T @ y_centered)
    return {
        "depth_intercept": float(depth_intercept),
        "depth_slope": float(depth_slope),
        "residual_intercept": y_mean,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "beta": beta,
    }


def predict_residual(frame: pd.DataFrame, fit: dict[str, object]) -> np.ndarray:
    x = feature_matrix(frame)
    mean = np.asarray(fit["feature_mean"], dtype=float)
    std = np.asarray(fit["feature_std"], dtype=float)
    beta = np.asarray(fit["beta"], dtype=float)
    return ((x - mean) / std) @ beta + float(fit["residual_intercept"])


def standardized_theory_score(frame: pd.DataFrame, fit: dict[str, object]) -> np.ndarray:
    x = feature_matrix(frame)
    mean = np.asarray(fit["feature_mean"], dtype=float)
    std = np.asarray(fit["feature_std"], dtype=float)
    standardized = (x - mean) / std
    feature_names = list(FEATURE_COLUMNS)
    return (
        standardized[:, feature_names.index("log_gradient_nuclear_rank")]
        + standardized[:, feature_names.index("log_alignment_ratio")]
        - standardized[:, feature_names.index("log_scaled_jvp_ratio")]
    )


def coefficient_rows(source_step: int, fit: dict[str, object]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = [
        {
            "source_warmup_steps": int(source_step),
            "term": "depth_intercept",
            "coefficient": float(fit["depth_intercept"]),
            "feature_mean": math.nan,
            "feature_std": math.nan,
            "ridge_alpha": RIDGE_ALPHA,
        },
        {
            "source_warmup_steps": int(source_step),
            "term": "depth_slope_log_early_layer_prior",
            "coefficient": float(fit["depth_slope"]),
            "feature_mean": math.nan,
            "feature_std": math.nan,
            "ridge_alpha": RIDGE_ALPHA,
        },
        {
            "source_warmup_steps": int(source_step),
            "term": "residual_intercept",
            "coefficient": float(fit["residual_intercept"]),
            "feature_mean": math.nan,
            "feature_std": math.nan,
            "ridge_alpha": RIDGE_ALPHA,
        },
    ]
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "source_warmup_steps": int(source_step),
                "term": feature_name,
                "coefficient": float(np.asarray(fit["beta"], dtype=float)[index]),
                "feature_mean": float(np.asarray(fit["feature_mean"], dtype=float)[index]),
                "feature_std": float(np.asarray(fit["feature_std"], dtype=float)[index]),
                "ridge_alpha": RIDGE_ALPHA,
            }
        )
    return rows


def score_pairs(layer_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coefficients = []
    checkpoints = sorted(int(value) for value in layer_summary["warmup_steps"].unique())
    for source_step in checkpoints:
        source = layer_summary[layer_summary["warmup_steps"].eq(source_step)].copy()
        fit = fit_calibrated_residual_score(source)
        coefficients.extend(coefficient_rows(source_step, fit))
        source_fit_residual = pd.DataFrame(
            {
                "parameter": source["parameter"].to_numpy(),
                "source_observed_drift_positive_control": fit_depth_baseline(source)[2].to_numpy(dtype=float),
            }
        )
        for target_step in checkpoints:
            if target_step == source_step:
                continue
            target = layer_summary[layer_summary["warmup_steps"].eq(target_step)].copy()
            target = source[["parameter"]].merge(target, on="parameter", how="inner")
            target_residual = target_residual_from_source_depth(
                target,
                float(fit["depth_intercept"]),
                float(fit["depth_slope"]),
            )
            primary_residual = predict_residual(target, fit)
            predicted_log_observed = (
                float(fit["depth_intercept"])
                + float(fit["depth_slope"]) * target["log_early_layer_prior"].to_numpy(dtype=float)
                + primary_residual
            )
            source_control = source_fit_residual.merge(target[["parameter"]], on="parameter", how="inner")[
                "source_observed_drift_positive_control"
            ].to_numpy(dtype=float)
            score_values = {
                "condition_score_v2_calibrated_residual": primary_residual,
                "early_layer_prior": target["log_early_layer_prior"].to_numpy(dtype=float),
                "legacy_scaled_jvp_ratio": target["log_scaled_jvp_ratio"].to_numpy(dtype=float),
                "theory_sign_composite": standardized_theory_score(target, fit),
                "source_observed_drift_positive_control": source_control,
            }
            predicted_log_values = {
                "condition_score_v2_calibrated_residual": predicted_log_observed,
                "legacy_scaled_jvp_ratio": target["log_scaled_jvp_ratio"].to_numpy(dtype=float),
                "source_observed_drift_positive_control": (
                    source.set_index("parameter")
                    .loc[target["parameter"], "log_observed"]
                    .to_numpy(dtype=float)
                ),
            }
            target_below_one = target["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].astype(float) < 1.0
            target_top5 = set(
                pd.DataFrame({"parameter": target["parameter"], "target_residual": target_residual})
                .nlargest(min(5, len(target)), "target_residual")["parameter"]
            )
            for score_name, values in score_values.items():
                score_series = pd.Series(values)
                target_series = pd.Series(target_residual.to_numpy(dtype=float))
                spearman, spearman_low, spearman_high, points = corr_ci95(
                    score_series,
                    target_series,
                    method="spearman",
                )
                pearson, pearson_low, pearson_high, _ = corr_ci95(
                    score_series,
                    target_series,
                    method="pearson",
                )
                predicted_top5 = set(
                    pd.DataFrame({"parameter": target["parameter"], "score": values})
                    .nlargest(min(5, len(target)), "score")["parameter"]
                )
                if score_name in predicted_log_values:
                    predicted_below_one = predicted_log_values[score_name] < 0.0
                    threshold_accuracy = float((predicted_below_one == target_below_one.to_numpy()).mean())
                    predicted_below_one_fraction = float(predicted_below_one.mean())
                else:
                    threshold_accuracy = math.nan
                    predicted_below_one_fraction = math.nan
                rows.append(
                    {
                        "source_warmup_steps": int(source_step),
                        "target_warmup_steps": int(target_step),
                        "score": score_name,
                        "score_family": (
                            "primary_candidate"
                            if score_name == "condition_score_v2_calibrated_residual"
                            else "positive_control"
                            if score_name == "source_observed_drift_positive_control"
                            else "baseline"
                        ),
                        "points": int(points),
                        "spearman_score_vs_target_residual": spearman,
                        "spearman_ci95_low": spearman_low,
                        "spearman_ci95_high": spearman_high,
                        "pearson_score_vs_target_residual": pearson,
                        "pearson_ci95_low": pearson_low,
                        "pearson_ci95_high": pearson_high,
                        "top5_residual_risk_overlap_fraction": len(predicted_top5 & target_top5)
                        / max(len(target_top5), 1),
                        "threshold_below_one_accuracy": threshold_accuracy,
                        "target_observed_below_one_fraction": float(target_below_one.mean()),
                        "predicted_below_one_fraction": predicted_below_one_fraction,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(coefficients)


def summarize_score_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score, group in pairs.groupby("score", observed=True, sort=False):
        spearman, spearman_low, spearman_high = ci95(group["spearman_score_vs_target_residual"])
        pearson, pearson_low, pearson_high = ci95(group["pearson_score_vs_target_residual"])
        top5, top5_low, top5_high = ci95(group["top5_residual_risk_overlap_fraction"])
        threshold, threshold_low, threshold_high = ci95(group["threshold_below_one_accuracy"])
        if math.isfinite(top5_low):
            top5_low = max(0.0, top5_low)
        if math.isfinite(top5_high):
            top5_high = min(1.0, top5_high)
        if math.isfinite(threshold_low):
            threshold_low = max(0.0, threshold_low)
        if math.isfinite(threshold_high):
            threshold_high = min(1.0, threshold_high)
        rows.append(
            {
                "score": score,
                "score_family": str(group["score_family"].iloc[0]),
                "checkpoint_transfer_pairs": int(len(group)),
                "mean_spearman_score_vs_target_residual": spearman,
                "spearman_ci95_low": spearman_low,
                "spearman_ci95_high": spearman_high,
                "mean_pearson_score_vs_target_residual": pearson,
                "pearson_ci95_low": pearson_low,
                "pearson_ci95_high": pearson_high,
                "mean_top5_residual_risk_overlap_fraction": top5,
                "top5_residual_risk_overlap_ci95_low": top5_low,
                "top5_residual_risk_overlap_ci95_high": top5_high,
                "mean_threshold_below_one_accuracy": threshold,
                "threshold_below_one_accuracy_ci95_low": threshold_low,
                "threshold_below_one_accuracy_ci95_high": threshold_high,
            }
        )
    return pd.DataFrame(rows)


def gate_report(summary: pd.DataFrame) -> pd.DataFrame:
    by_score = summary.set_index("score")
    primary = by_score.loc["condition_score_v2_calibrated_residual"]
    early = by_score.loc["early_layer_prior"]
    threshold = float(primary["mean_threshold_below_one_accuracy"])
    residual_low = float(primary["spearman_ci95_low"])
    beats_early = float(primary["mean_spearman_score_vs_target_residual"]) > float(
        early["mean_spearman_score_vs_target_residual"]
    )
    rows = [
        {
            "gate_id": "legacy_checkpoint_residual_spearman",
            "scope": "retrospective ResNet18 checkpoint split",
            "status": "pass" if residual_low > 0.0 else "fail",
            "evidence": (
                f"primary residual Spearman={fmt(primary['mean_spearman_score_vs_target_residual'])} "
                f"CI=[{fmt(primary['spearman_ci95_low'])}, {fmt(primary['spearman_ci95_high'])}]"
            ),
        },
        {
            "gate_id": "legacy_checkpoint_threshold_accuracy",
            "scope": "retrospective ResNet18 checkpoint split",
            "status": "pass" if threshold >= 0.8 else "fail",
            "evidence": f"primary below-one threshold accuracy={fmt(threshold)}",
        },
        {
            "gate_id": "baseline_comparison",
            "scope": "retrospective ResNet18 checkpoint split",
            "status": "pass" if beats_early else "fail",
            "evidence": (
                f"primary Spearman={fmt(primary['mean_spearman_score_vs_target_residual'])}; "
                f"early-layer prior Spearman={fmt(early['mean_spearman_score_vs_target_residual'])}"
            ),
        },
    ]
    rows.extend(heldout_protocol_rows())
    return pd.DataFrame(rows)


def _gate_status(gates: pd.DataFrame, gate_id: str) -> tuple[str, str]:
    match = gates[gates["gate_id"].eq(gate_id)]
    if match.empty:
        return "not_run", f"{gate_id} is missing from the held-out gate report."
    row = match.iloc[0]
    return str(row["status"]), str(row["evidence"])


def heldout_protocol_rows() -> list[dict[str, str]]:
    if not HELDOUT_GATE_REPORT.exists():
        return [
            {
                "gate_id": "primary_heldout_architecture",
                "scope": "P0 protocol",
                "status": "not_run",
                "evidence": "ResNet34 held-out architecture split is registered but not generated.",
            },
            {
                "gate_id": "primary_heldout_data",
                "scope": "P0 protocol",
                "status": "not_run",
                "evidence": "CIFAR-10-LT held-out data split is registered but not generated.",
            },
            {
                "gate_id": "p0_predictive_condition_claim",
                "scope": "paper claim",
                "status": "not_ready",
                "evidence": "The retrospective checkpoint split is promising, but the registered architecture and data held-out splits are still missing.",
            },
        ]

    gates = pd.read_csv(HELDOUT_GATE_REPORT)
    arch_status, arch_evidence = _gate_status(gates, "primary_heldout_architecture_residual_spearman")
    data_status, data_evidence = _gate_status(gates, "primary_heldout_data_residual_spearman")
    p0_status, p0_evidence = _gate_status(gates, "p0_predictive_condition_heldout_claim")
    return [
        {
            "gate_id": "primary_heldout_architecture",
            "scope": "P0 protocol",
            "status": arch_status,
            "evidence": f"held-out evaluation: {arch_evidence}",
        },
        {
            "gate_id": "primary_heldout_data",
            "scope": "P0 protocol",
            "status": data_status,
            "evidence": f"held-out evaluation: {data_evidence}",
        },
        {
            "gate_id": "p0_predictive_condition_claim",
            "scope": "paper claim",
            "status": p0_status,
            "evidence": f"{p0_evidence} See {HELDOUT_DISCUSSION_PATH.as_posix()}.",
        },
    ]


def write_figure(summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("mean_spearman_score_vs_target_residual")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    y_positions = list(range(len(ordered)))
    estimates = ordered["mean_spearman_score_vs_target_residual"].to_numpy(dtype=float)
    low = ordered["spearman_ci95_low"].to_numpy(dtype=float)
    high = ordered["spearman_ci95_high"].to_numpy(dtype=float)
    colors = [
        "#009E73"
        if family == "primary_candidate"
        else "#CC79A7"
        if family == "positive_control"
        else "#0072B2"
        for family in ordered["score_family"]
    ]
    axes[0].barh(y_positions, estimates, color=colors, alpha=0.88)
    axes[0].errorbar(
        estimates,
        y_positions,
        xerr=[estimates - low, high - estimates],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(ordered["score"].tolist(), fontsize=8)
    axes[0].set_xlabel("held-out-checkpoint residual Spearman")
    axes[0].set_title("Residual risk prediction")

    threshold = summary.dropna(subset=["mean_threshold_below_one_accuracy"]).copy()
    threshold = threshold.sort_values("mean_threshold_below_one_accuracy")
    y_positions = list(range(len(threshold)))
    values = threshold["mean_threshold_below_one_accuracy"].to_numpy(dtype=float)
    axes[1].barh(y_positions, values, color="#D55E00", alpha=0.85)
    axes[1].axvline(0.8, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(threshold["score"].tolist(), fontsize=8)
    axes[1].set_xlim(0.0, 1.05)
    axes[1].set_xlabel("below-one threshold accuracy")
    axes[1].set_title("Direction threshold")

    fig.suptitle("Registered condition-score v2 retrospective checkpoint analysis")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_condition_score_next.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    figure_path: Path,
    discussion_path: Path,
) -> None:
    by_score = summary.set_index("score")
    primary = by_score.loc["condition_score_v2_calibrated_residual"]
    early = by_score.loc["early_layer_prior"]
    legacy = by_score.loc["legacy_scaled_jvp_ratio"]
    heldout_text = ""
    boundary_text = (
        "This is not yet the P0 predictive-condition result. It is a locked ResNet18 "
        "checkpoint-split analysis showing that the registered v2 score is worth running "
        "on the protocol's held-out architecture and held-out data splits. The paper must "
        "still call the P0 condition-score claim incomplete until those GPU/Slurm splits "
        "exist and pass the registered gates."
    )
    if HELDOUT_SCORE_SUMMARY.exists() and HELDOUT_GATE_REPORT.exists():
        heldout_summary = pd.read_csv(HELDOUT_SCORE_SUMMARY)
        heldout_gates = pd.read_csv(HELDOUT_GATE_REPORT)
        heldout_by_key = heldout_summary.set_index(["split_role", "score"])
        arch_primary = heldout_by_key.loc[
            ("primary_heldout_architecture", "condition_score_v2_calibrated_residual")
        ]
        data_primary = heldout_by_key.loc[
            ("primary_heldout_data", "condition_score_v2_calibrated_residual")
        ]
        data_legacy = heldout_by_key.loc[
            ("primary_heldout_data", "legacy_scaled_jvp_ratio")
        ]
        heldout_text = f"""
## Held-Out Evaluation

The registered held-out condition-score evaluation now fails the P0 residual-ranking gates. ResNet34 held-out architecture primary residual Spearman is {fmt(arch_primary['mean_spearman_score_vs_target_residual'])} [{fmt(arch_primary['spearman_ci95_low'])}, {fmt(arch_primary['spearman_ci95_high'])}], and CIFAR-10-LT held-out data primary residual Spearman is {fmt(data_primary['mean_spearman_score_vs_target_residual'])} [{fmt(data_primary['spearman_ci95_low'])}, {fmt(data_primary['spearman_ci95_high'])}]. The below-one threshold direction still passes ({fmt(arch_primary['mean_threshold_below_one_accuracy'])} and {fmt(data_primary['mean_threshold_below_one_accuracy'])}), while the CIFAR-10-LT legacy scaled-JVP ratio has residual Spearman {fmt(data_legacy['mean_spearman_score_vs_target_residual'])} [{fmt(data_legacy['spearman_ci95_low'])}, {fmt(data_legacy['spearman_ci95_high'])}].

{markdown_table(heldout_gates, ["gate_id", "scope", "status", "evidence"])}

Full held-out report: [{HELDOUT_DISCUSSION_PATH.name}](../{HELDOUT_DISCUSSION_PATH.as_posix()}).
"""
        boundary_text = (
            "This is not the P0 predictive-condition result. The retrospective "
            "checkpoint split passes, but the registered no-tuning held-out evaluation "
            "fails the residual-ranking gates on both the ResNet34 architecture split "
            "and the CIFAR-10-LT data split. The score still preserves the below-one "
            "threshold direction, so the current evidence supports a narrower "
            "directional guardrail rather than a held-out layer-risk ranking claim."
        )
    text = f"""# E11 CIFAR-100-LT ResNet Condition-Score v2 Retrospective Analysis

This generated analysis is the first executable step after the registered condition-score protocol. It uses the existing ResNet18 checkpoint-transfer `layer_summary.csv` as a locked retrospective split: each source checkpoint fits the calibrated residual score, then the frozen score predicts residual observed layer risk on the other checkpoints.

![CIFAR-100-LT ResNet condition-score v2 retrospective analysis](../{figure_path.as_posix()})

## Summary

{markdown_table(summary, ["score", "score_family", "checkpoint_transfer_pairs", "mean_spearman_score_vs_target_residual", "spearman_ci95_low", "spearman_ci95_high", "mean_top5_residual_risk_overlap_fraction", "mean_threshold_below_one_accuracy"])}

## Gate Report

{markdown_table(gates, ["gate_id", "scope", "status", "evidence"])}

## Readout

- Primary `condition_score_v2_calibrated_residual` residual Spearman is {fmt(primary['mean_spearman_score_vs_target_residual'])} [{fmt(primary['spearman_ci95_low'])}, {fmt(primary['spearman_ci95_high'])}] on the retrospective checkpoint split.
- The early-layer prior residual Spearman is {fmt(early['mean_spearman_score_vs_target_residual'])} [{fmt(early['spearman_ci95_low'])}, {fmt(early['spearman_ci95_high'])}].
- The legacy scaled-JVP residual score remains negative at {fmt(legacy['mean_spearman_score_vs_target_residual'])} [{fmt(legacy['spearman_ci95_low'])}, {fmt(legacy['spearman_ci95_high'])}].
{heldout_text}

## Boundary

{boundary_text}

Artifacts:
- [score_pairs.csv](../{(OUTPUT_DIR / 'score_pairs.csv').as_posix()})
- [score_summary.csv](../{(OUTPUT_DIR / 'score_summary.csv').as_posix()})
- [calibration_coefficients.csv](../{(OUTPUT_DIR / 'calibration_coefficients.csv').as_posix()})
- [gate_report.csv](../{(OUTPUT_DIR / 'gate_report.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
- [heldout_score_evaluation/heldout_score_summary.csv](../{HELDOUT_SCORE_SUMMARY.as_posix()})
- [heldout_score_evaluation/heldout_gate_report.csv](../{HELDOUT_GATE_REPORT.as_posix()})
"""
    write_markdown(discussion_path, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_summary = add_protocol_features(pd.read_csv(INPUT_LAYER_SUMMARY))
    score_registry = pd.read_csv(PROTOCOL_DIR / "score_registry.csv")
    split_registry = pd.read_csv(PROTOCOL_DIR / "split_registry.csv")
    acceptance_gates = pd.read_csv(PROTOCOL_DIR / "acceptance_gates.csv")
    pairs, coefficients = score_pairs(layer_summary)
    summary = summarize_score_pairs(pairs)
    gates = gate_report(summary)

    pairs.to_csv(OUTPUT_DIR / "score_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "score_summary.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "calibration_coefficients.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "input_layer_summary": INPUT_LAYER_SUMMARY.as_posix(),
                "protocol_dir": PROTOCOL_DIR.as_posix(),
                "ridge_alpha": RIDGE_ALPHA,
                "feature_columns": FEATURE_COLUMNS,
                "score_registry_rows": int(len(score_registry)),
                "split_registry_rows": int(len(split_registry)),
                "acceptance_gate_rows": int(len(acceptance_gates)),
                "analysis_scope": "locked retrospective ResNet18 checkpoint split; registered held-out architecture/data evaluation available separately when heldout_score_evaluation exists",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(summary, FIGURE_DIR)
    write_discussion(summary, gates, figure_path, DISCUSSION_PATH)
    print(f"saved condition-score v2 retrospective analysis to {OUTPUT_DIR}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(summary.to_string(index=False))
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
