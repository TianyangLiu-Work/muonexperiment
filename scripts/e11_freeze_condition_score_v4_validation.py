from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown
from e11_condition_geometry.statistics import ci95, corr_ci95
from scripts.e11_write_cifar100_resnet_condition_score_next import (
    add_protocol_features,
    fit_depth_baseline,
    target_residual_from_source_depth,
)


V4_DIR = Path("results/e11_condition_score_v4_protocol")
SOURCE_LAYER_SUMMARY = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv")
SOURCE_METRICS = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/metrics.csv")
VALIDATION_DIR = V4_DIR / "validation_cifar100_rotated"
VALIDATION_LAYER_SUMMARY = VALIDATION_DIR / "layer_summary.csv"
VALIDATION_METRICS = VALIDATION_DIR / "metrics.csv"
OUTPUT_DIR = V4_DIR / "validation_score_freeze"
DISCUSSION_PATH = Path("discussion/e11_condition_score_v4_validation_freeze.md")
FINAL_LAYER_SUMMARIES = (
    V4_DIR / "final_architecture_wide_resnet50_2" / "layer_summary.csv",
    V4_DIR / "final_data_cifar10_mixed" / "layer_summary.csv",
)

AXIS_FEATURES = (
    "log_scaled_jvp_ratio",
    "log_scaled_jvp_fro_amplitude",
    "log_gradient_nuclear_rank",
    "log_alignment_ratio",
    "log_step_size_ratio",
    "log_early_layer_prior",
    "transport_is_downsample",
    "transport_is_classifier",
)


@dataclass(frozen=True)
class CandidateScore:
    score_id: str
    role: str
    formula: str
    weights: dict[str, float]
    predicts_direction_threshold: bool = False


CANDIDATES = (
    CandidateScore(
        score_id="condition_score_v4_direction_axis_scaled_jvp_ratio",
        role="direction_guardrail",
        formula="raw log scaled-JVP spectral/Frobenius squared ratio",
        weights={"log_scaled_jvp_ratio": 1.0},
        predicts_direction_threshold=True,
    ),
    CandidateScore(
        score_id="condition_score_v4_fro_amplitude_axis",
        role="residual_candidate",
        formula="source-standardized Frobenius matched-head-gain scaled-JVP amplitude",
        weights={"log_scaled_jvp_fro_amplitude": 1.0},
    ),
    CandidateScore(
        score_id="condition_score_v4_two_axis_positive",
        role="residual_candidate",
        formula="source-standardized direction ratio plus Frobenius amplitude",
        weights={"log_scaled_jvp_ratio": 1.0, "log_scaled_jvp_fro_amplitude": 1.0},
    ),
    CandidateScore(
        score_id="condition_score_v4_two_axis_amplitude_minus_direction",
        role="residual_candidate",
        formula="source-standardized Frobenius amplitude minus direction ratio",
        weights={"log_scaled_jvp_fro_amplitude": 1.0, "log_scaled_jvp_ratio": -1.0},
    ),
    CandidateScore(
        score_id="condition_score_v4_two_axis_transport",
        role="residual_candidate",
        formula="two-axis score with generic downsample/classifier transport tags",
        weights={
            "log_scaled_jvp_ratio": 1.0,
            "log_scaled_jvp_fro_amplitude": 1.0,
            "transport_is_downsample": 0.5,
            "transport_is_classifier": 0.5,
        },
    ),
    CandidateScore(
        score_id="early_layer_prior",
        role="baseline",
        formula="raw log early-layer prior",
        weights={"log_early_layer_prior": 1.0},
    ),
)


SUMMARY_COLUMNS = [
    "score",
    "score_role",
    "validation_transfer_pairs",
    "mean_points",
    "mean_spearman_score_vs_target_residual",
    "spearman_ci95_low",
    "spearman_ci95_high",
    "mean_pearson_score_vs_target_residual",
    "pearson_ci95_low",
    "pearson_ci95_high",
    "mean_top5_residual_risk_overlap_fraction",
    "top5_residual_risk_overlap_ci95_low",
    "top5_residual_risk_overlap_ci95_high",
    "mean_threshold_below_one_accuracy",
    "threshold_below_one_accuracy_ci95_low",
    "threshold_below_one_accuracy_ci95_high",
]


def log_positive(values: pd.Series) -> pd.Series:
    return values.astype(float).clip(lower=1e-300).map(math.log)


def geomean(values: pd.Series) -> float:
    positive = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=1e-300)
    if positive.empty:
        return math.nan
    return float(np.exp(np.log(positive.to_numpy(dtype=float)).mean()))


def transport_stage(parameter: str) -> str:
    if parameter == "conv1.weight":
        return "stem"
    if parameter.startswith("layer"):
        return parameter.split(".", maxsplit=1)[0]
    if parameter in {"fc.weight", "classifier.weight"}:
        return "classifier"
    return "other"


def add_transport_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    parameter = frame["parameter"].astype(str)
    frame["transport_stage"] = parameter.map(transport_stage)
    frame["transport_is_downsample"] = parameter.str.contains("downsample").astype(float)
    frame["transport_is_classifier"] = parameter.isin({"fc.weight", "classifier.weight"}).astype(float)
    frame["transport_is_stem"] = parameter.eq("conv1.weight").astype(float)
    return frame


def add_fro_amplitude_axis(layer_summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    fro = metrics[metrics["geometry"].astype(str).eq("frobenius")].copy()
    group_cols = ["warmup_steps", "layer_index", "parameter", "shape"]
    amplitude = (
        fro.groupby(group_cols, observed=True, sort=False)["scaled_jvp_tail_drift_fro"]
        .apply(geomean)
        .reset_index(name="geomean_scaled_jvp_tail_drift_fro_frobenius_direction")
    )
    merged = layer_summary.merge(amplitude, on=group_cols, how="left", validate="one_to_one")
    merged["log_scaled_jvp_fro_amplitude"] = log_positive(
        merged["geomean_scaled_jvp_tail_drift_fro_frobenius_direction"]
    )
    return merged


def load_axis_frame(layer_summary_path: Path, metrics_path: Path) -> pd.DataFrame:
    layer_summary = pd.read_csv(layer_summary_path)
    metrics = pd.read_csv(metrics_path)
    frame = add_protocol_features(layer_summary)
    frame = add_fro_amplitude_axis(frame, metrics)
    frame = add_transport_features(frame)
    missing = [column for column in AXIS_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"v4 axis frame missing columns: {missing}")
    if frame["log_scaled_jvp_fro_amplitude"].isna().any():
        missing_rows = frame[frame["log_scaled_jvp_fro_amplitude"].isna()][
            ["warmup_steps", "parameter"]
        ].head()
        raise ValueError(f"v4 Frobenius amplitude axis has missing rows: {missing_rows.to_dict('records')}")
    return frame


def source_feature_stats(source: pd.DataFrame) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for column in AXIS_FEATURES:
        values = source[column].to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if not math.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[column] = (mean, std)
    return stats


def standardized_values(frame: pd.DataFrame, stats: dict[str, tuple[float, float]], column: str) -> np.ndarray:
    mean, std = stats[column]
    return (frame[column].to_numpy(dtype=float) - mean) / std


def score_values(candidate: CandidateScore, frame: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    if candidate.score_id in {"condition_score_v4_direction_axis_scaled_jvp_ratio", "early_layer_prior"}:
        feature = next(iter(candidate.weights))
        return frame[feature].to_numpy(dtype=float) * float(candidate.weights[feature])
    values = np.zeros(len(frame), dtype=float)
    for feature, weight in candidate.weights.items():
        values += float(weight) * standardized_values(frame, stats, feature)
    return values


def target_top_k(target: pd.DataFrame, target_residual: pd.Series) -> set[str]:
    return set(
        pd.DataFrame({"parameter": target["parameter"], "target_residual": target_residual})
        .nlargest(min(5, len(target)), "target_residual")["parameter"]
        .astype(str)
    )


def score_validation_pairs(source_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_steps = sorted(int(value) for value in source_frame["warmup_steps"].unique())
    target_steps = sorted(int(value) for value in validation_frame["warmup_steps"].unique())
    for source_step in source_steps:
        source = source_frame[source_frame["warmup_steps"].eq(source_step)].copy()
        depth_intercept, depth_slope, source_residual = fit_depth_baseline(source)
        stats = source_feature_stats(source)
        source_control = pd.DataFrame(
            {
                "parameter": source["parameter"].astype(str).to_numpy(),
                "source_observed_residual_positive_control": source_residual.to_numpy(dtype=float),
                "source_log_observed_positive_control": source["log_observed"].to_numpy(dtype=float),
            }
        )
        for target_step in target_steps:
            target = validation_frame[validation_frame["warmup_steps"].eq(target_step)].copy()
            target_residual = target_residual_from_source_depth(target, float(depth_intercept), float(depth_slope))
            target_below_one = (
                target["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].astype(float) < 1.0
            )
            target_top5 = target_top_k(target, target_residual)
            target_series = pd.Series(target_residual.to_numpy(dtype=float))
            for candidate in CANDIDATES:
                values = score_values(candidate, target, stats)
                score_series = pd.Series(values)
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
                    .astype(str)
                )
                if candidate.predicts_direction_threshold:
                    predicted_below_one = values < 0.0
                    threshold_accuracy = float((predicted_below_one == target_below_one.to_numpy()).mean())
                    predicted_below_one_fraction = float(predicted_below_one.mean())
                else:
                    threshold_accuracy = math.nan
                    predicted_below_one_fraction = math.nan
                rows.append(
                    {
                        "split_id": "v4_validation_cifar100lt_resnet18_rotated_partition",
                        "source_warmup_steps": int(source_step),
                        "target_warmup_steps": int(target_step),
                        "score": candidate.score_id,
                        "score_role": candidate.role,
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
                        "frozen_depth_intercept": float(depth_intercept),
                        "frozen_depth_slope": float(depth_slope),
                    }
                )

            matched = target[["parameter"]].astype({"parameter": str}).merge(
                source_control,
                on="parameter",
                how="inner",
            )
            if not matched.empty:
                target_matched = matched[["parameter"]].merge(target, on="parameter", how="inner")
                target_residual_matched = target_residual_from_source_depth(
                    target_matched,
                    float(depth_intercept),
                    float(depth_slope),
                )
                source_values = matched["source_observed_residual_positive_control"].to_numpy(dtype=float)
                source_series = pd.Series(source_values)
                target_series = pd.Series(target_residual_matched.to_numpy(dtype=float))
                spearman, spearman_low, spearman_high, points = corr_ci95(
                    source_series,
                    target_series,
                    method="spearman",
                )
                pearson, pearson_low, pearson_high, _ = corr_ci95(
                    source_series,
                    target_series,
                    method="pearson",
                )
                target_below_one_matched = (
                    target_matched["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].astype(float) < 1.0
                )
                predicted_below_one = matched["source_log_observed_positive_control"].to_numpy(dtype=float) < 0.0
                target_top5 = target_top_k(target_matched, target_residual_matched)
                predicted_top5 = set(
                    pd.DataFrame({"parameter": target_matched["parameter"], "score": source_values})
                    .nlargest(min(5, len(target_matched)), "score")["parameter"]
                    .astype(str)
                )
                rows.append(
                    {
                        "split_id": "v4_validation_cifar100lt_resnet18_rotated_partition",
                        "source_warmup_steps": int(source_step),
                        "target_warmup_steps": int(target_step),
                        "score": "source_observed_drift_positive_control",
                        "score_role": "positive_control",
                        "points": int(points),
                        "spearman_score_vs_target_residual": spearman,
                        "spearman_ci95_low": spearman_low,
                        "spearman_ci95_high": spearman_high,
                        "pearson_score_vs_target_residual": pearson,
                        "pearson_ci95_low": pearson_low,
                        "pearson_ci95_high": pearson_high,
                        "top5_residual_risk_overlap_fraction": len(predicted_top5 & target_top5)
                        / max(len(target_top5), 1),
                        "threshold_below_one_accuracy": float(
                            (predicted_below_one == target_below_one_matched.to_numpy()).mean()
                        ),
                        "target_observed_below_one_fraction": float(target_below_one_matched.mean()),
                        "predicted_below_one_fraction": float(predicted_below_one.mean()),
                        "frozen_depth_intercept": float(depth_intercept),
                        "frozen_depth_slope": float(depth_slope),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
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
                "score": str(score),
                "score_role": str(group["score_role"].iloc[0]),
                "validation_transfer_pairs": int(len(group)),
                "mean_points": float(group["points"].mean()),
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


def select_primary_score(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        return "pending_validation_output", "not_ready"
    candidates = summary[summary["score_role"].eq("residual_candidate")].copy()
    if candidates.empty:
        return "missing_residual_candidate", "fail"
    candidates["passes_residual_gate"] = candidates["spearman_ci95_low"].astype(float) > 0.0
    if not candidates["passes_residual_gate"].any():
        best = candidates.sort_values(
            ["spearman_ci95_low", "mean_spearman_score_vs_target_residual"],
            ascending=[False, False],
        ).iloc[0]
        return str(best["score"]), "validation_failed"
    best = candidates[candidates["passes_residual_gate"]].sort_values(
        ["spearman_ci95_low", "mean_spearman_score_vs_target_residual"],
        ascending=[False, False],
    ).iloc[0]
    return str(best["score"]), "frozen"


def freeze_registry(summary: pd.DataFrame, validation_generated: bool) -> pd.DataFrame:
    selected_score, freeze_status = select_primary_score(summary)
    rows: list[dict[str, object]] = [
        {
            "item": "v4 validation split output",
            "status": "generated" if validation_generated else "not_run",
            "evidence": VALIDATION_LAYER_SUMMARY.as_posix(),
        },
        {
            "item": "v4 candidate pool",
            "status": "registered",
            "evidence": "candidate formulas are fixed in scripts/e11_freeze_condition_score_v4_validation.py",
        },
        {
            "item": "v4 selected residual score",
            "status": freeze_status,
            "evidence": selected_score,
        },
        {
            "item": "v4 final split outputs",
            "status": "generated_before_freeze" if any(path.exists() for path in FINAL_LAYER_SUMMARIES) else "not_run",
            "evidence": "; ".join(path.as_posix() for path in FINAL_LAYER_SUMMARIES),
        },
    ]
    return pd.DataFrame(rows)


def formula_registry(summary: pd.DataFrame, validation_generated: bool) -> pd.DataFrame:
    selected_score, freeze_status = select_primary_score(summary)
    rows = [
        {
            "score_id": candidate.score_id,
            "role": candidate.role,
            "formula": candidate.formula,
            "feature_weights": json.dumps(candidate.weights, sort_keys=True),
            "normalization_source": "raw log axis"
            if candidate.score_id in {"condition_score_v4_direction_axis_scaled_jvp_ratio", "early_layer_prior"}
            else "source calibration step mean/std",
            "validation_selection_rule": (
                "eligible residual candidates require validation Spearman CI lower endpoint above zero; "
                "select highest lower endpoint, then highest mean Spearman"
                if candidate.role == "residual_candidate"
                else "reported but not eligible as the residual primary"
            ),
            "selected_for_final_evaluation": "yes"
            if validation_generated and freeze_status == "frozen" and candidate.score_id == selected_score
            else "no",
        }
        for candidate in CANDIDATES
    ]
    rows.append(
        {
            "score_id": "source_observed_drift_positive_control",
            "role": "positive_control",
            "formula": "source observed residual under source-fit depth baseline, matched by parameter name",
            "feature_weights": "{}",
            "normalization_source": "source observed drift; not eligible for final claim",
            "validation_selection_rule": "reported only as an upper-bound transfer control",
            "selected_for_final_evaluation": "no",
        }
    )
    rows.append(
        {
            "score_id": "condition_score_v4_validation_selected",
            "role": "primary_alias",
            "formula": selected_score,
            "feature_weights": "{}",
            "normalization_source": "points to selected residual candidate after validation",
            "validation_selection_rule": "pending until validation output exists; not final-ready unless status is frozen",
            "selected_for_final_evaluation": "yes" if validation_generated and freeze_status == "frozen" else "no",
        }
    )
    return pd.DataFrame(rows)


def gate_report(summary: pd.DataFrame, registry: pd.DataFrame, validation_generated: bool) -> pd.DataFrame:
    status_lookup = registry.set_index("item")["status"].to_dict()
    evidence_lookup = registry.set_index("item")["evidence"].to_dict()
    final_outputs_absent = status_lookup.get("v4 final split outputs") == "not_run"
    selected_status = status_lookup.get("v4 selected residual score")
    direction = summary[summary["score"].eq("condition_score_v4_direction_axis_scaled_jvp_ratio")]
    direction_pass = (
        not direction.empty
        and float(direction.iloc[0]["mean_threshold_below_one_accuracy"]) >= 0.8
        and float(direction.iloc[0]["threshold_below_one_accuracy_ci95_low"]) >= 0.8
    )
    selected_score = evidence_lookup.get("v4 selected residual score")
    selected_summary = summary[summary["score"].eq(selected_score)] if selected_score in set(summary["score"]) else pd.DataFrame()
    rows = [
        {
            "gate_id": "V4F-1-validation-output",
            "scope": "validation-only split",
            "status": "pass" if validation_generated else "not_run",
            "evidence": VALIDATION_LAYER_SUMMARY.as_posix(),
        },
        {
            "gate_id": "V4F-2-no-final-before-freeze",
            "scope": "unspent final splits",
            "status": "pass" if final_outputs_absent else "fail",
            "evidence": "no v4 final layer_summary.csv exists before a frozen validation score",
        },
        {
            "gate_id": "V4F-3-residual-score-freeze",
            "scope": "primary residual-ranking score",
            "status": "pass" if selected_status == "frozen" else "not_ready",
            "evidence": selected_score if selected_score is not None else "no selected score",
        },
        {
            "gate_id": "V4F-4-direction-threshold-guardrail",
            "scope": "direction axis",
            "status": "pass" if direction_pass else "not_run" if not validation_generated else "fail",
            "evidence": (
                "validation direction-axis threshold accuracy="
                f"{fmt(direction.iloc[0]['mean_threshold_below_one_accuracy'])} "
                f"CI=[{fmt(direction.iloc[0]['threshold_below_one_accuracy_ci95_low'])}, "
                f"{fmt(direction.iloc[0]['threshold_below_one_accuracy_ci95_high'])}]"
                if not direction.empty
                else "validation score rows missing"
            ),
        },
        {
            "gate_id": "V4F-5-selected-score-residual-spearman",
            "scope": "validation residual ranking",
            "status": "pass"
            if not selected_summary.empty and float(selected_summary.iloc[0]["spearman_ci95_low"]) > 0.0
            else "not_run"
            if not validation_generated
            else "fail",
            "evidence": (
                f"selected={selected_score}; Spearman="
                f"{fmt(selected_summary.iloc[0]['mean_spearman_score_vs_target_residual'])} "
                f"CI=[{fmt(selected_summary.iloc[0]['spearman_ci95_low'])}, "
                f"{fmt(selected_summary.iloc[0]['spearman_ci95_high'])}]"
                if not selected_summary.empty
                else "no selected validation summary row"
            ),
        },
        {
            "gate_id": "V4F-6-final-claim-readiness",
            "scope": "P0 predictive-condition claim",
            "status": "pass" if selected_status == "frozen" and direction_pass and final_outputs_absent else "not_ready",
            "evidence": "final splits remain blocked until this gate is pass in a committed artifact",
        },
    ]
    return pd.DataFrame(rows)


def write_discussion(
    summary: pd.DataFrame,
    registry: pd.DataFrame,
    formulas: pd.DataFrame,
    gates: pd.DataFrame,
    validation_generated: bool,
) -> None:
    if summary.empty:
        summary_text = "Validation score rows are not generated yet because the v4 validation Slurm output is missing."
    else:
        summary_text = markdown_table(
            summary,
            [
                "score",
                "score_role",
                "validation_transfer_pairs",
                "mean_spearman_score_vs_target_residual",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "mean_threshold_below_one_accuracy",
            ],
        )
    text = f"""# E11 Condition-Score V4 Validation Freeze

This generated artifact is the missing commit boundary between the registered
v4 protocol and any unspent v4 final split. It is intentionally allowed to look
at the validation-only rotated CIFAR-100-LT split, but it keeps all v2/v3 final
splits quarantined and refuses to mark a final-ready score unless the validation
residual-ranking and direction gates pass before final outputs exist.

## Freeze Registry

{markdown_table(registry, ["item", "status", "evidence"])}

## Candidate Formula Registry

{markdown_table(formulas, ["score_id", "role", "formula", "selected_for_final_evaluation"])}

## Validation Score Summary

{summary_text}

## Gate Report

{markdown_table(gates, ["gate_id", "scope", "status", "evidence"])}

## Claim Boundary

Current status: {"validation output exists" if validation_generated else "validation output is not run"}.

Allowed now: commit the v4 validation-freeze machinery and, if needed, submit
the validation-only Slurm job.

Blocked now: running or interpreting the unspent WideResNet50-2 and CIFAR-10
mixed final splits as P0 evidence before `V4F-6-final-claim-readiness` passes in
a committed artifact.

Artifacts:
- [score_formula_registry.csv](../{(OUTPUT_DIR / 'score_formula_registry.csv').as_posix()})
- [freeze_status.csv](../{(OUTPUT_DIR / 'freeze_status.csv').as_posix()})
- [validation_gate_report.csv](../{(OUTPUT_DIR / 'validation_gate_report.csv').as_posix()})
- [validation_score_pairs.csv](../{(OUTPUT_DIR / 'validation_score_pairs.csv').as_posix()})
- [validation_score_summary.csv](../{(OUTPUT_DIR / 'validation_score_summary.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_generated = VALIDATION_LAYER_SUMMARY.exists() and VALIDATION_METRICS.exists()
    if validation_generated:
        source_frame = load_axis_frame(SOURCE_LAYER_SUMMARY, SOURCE_METRICS)
        validation_frame = load_axis_frame(VALIDATION_LAYER_SUMMARY, VALIDATION_METRICS)
        pairs = score_validation_pairs(source_frame, validation_frame)
        summary = summarize_pairs(pairs)
    else:
        pairs = pd.DataFrame(
            columns=[
                "split_id",
                "source_warmup_steps",
                "target_warmup_steps",
                "score",
                "score_role",
                "points",
                "spearman_score_vs_target_residual",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "pearson_score_vs_target_residual",
                "pearson_ci95_low",
                "pearson_ci95_high",
                "top5_residual_risk_overlap_fraction",
                "threshold_below_one_accuracy",
                "target_observed_below_one_fraction",
                "predicted_below_one_fraction",
                "frozen_depth_intercept",
                "frozen_depth_slope",
            ]
        )
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    registry = freeze_registry(summary, validation_generated)
    formulas = formula_registry(summary, validation_generated)
    gates = gate_report(summary, registry, validation_generated)
    formulas.to_csv(OUTPUT_DIR / "score_formula_registry.csv", index=False)
    registry.to_csv(OUTPUT_DIR / "freeze_status.csv", index=False)
    pairs.to_csv(OUTPUT_DIR / "validation_score_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "validation_score_summary.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "validation_gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "source_layer_summary": SOURCE_LAYER_SUMMARY.as_posix(),
                "source_metrics": SOURCE_METRICS.as_posix(),
                "validation_layer_summary": VALIDATION_LAYER_SUMMARY.as_posix(),
                "validation_metrics": VALIDATION_METRICS.as_posix(),
                "validation_generated": validation_generated,
                "final_layer_summaries": [path.as_posix() for path in FINAL_LAYER_SUMMARIES],
                "axis_features": list(AXIS_FEATURES),
                "selection_rule": (
                    "among residual candidates, require validation Spearman CI lower endpoint above zero; "
                    "select highest lower endpoint, then highest mean Spearman"
                ),
                "direction_gate": "condition_score_v4_direction_axis_scaled_jvp_ratio threshold accuracy CI lower endpoint >= 0.8",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(summary, registry, formulas, gates, validation_generated)
    print(f"saved v4 validation-freeze artifact to {OUTPUT_DIR}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
