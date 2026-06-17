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
from scripts.e11_freeze_condition_score_v4_validation import (
    load_axis_frame,
    source_feature_stats,
    target_top_k,
)
from scripts.e11_write_cifar100_resnet_condition_score_next import (
    fit_depth_baseline,
    target_residual_from_source_depth,
)


SOURCE_LAYER_SUMMARY = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv")
SOURCE_METRICS = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/metrics.csv")
V5_PROTOCOL_DIR = Path("results/e11_condition_score_v5_protocol")
V5_THEORY_DIR = Path("results/e11_condition_score_v5_theory_protocol")
VALIDATION_DIR = V5_PROTOCOL_DIR / "validation_cifar100_mod4_partition"
VALIDATION_LAYER_SUMMARY = VALIDATION_DIR / "layer_summary.csv"
VALIDATION_METRICS = VALIDATION_DIR / "metrics.csv"
OUTPUT_DIR = V5_PROTOCOL_DIR / "validation_score_freeze"
DISCUSSION_PATH = Path("discussion/e11_condition_score_v5_validation_freeze.md")
FINAL_LAYER_SUMMARIES = (
    V5_PROTOCOL_DIR / "final_architecture_resnext50_32x4d" / "layer_summary.csv",
    V5_PROTOCOL_DIR / "final_data_cifar10_cross" / "layer_summary.csv",
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


@dataclass(frozen=True)
class CandidateScore:
    score_id: str
    role: str
    formula: str
    weights: dict[str, float]
    raw_axis: bool = False
    predicts_direction_threshold: bool = False


CANDIDATES = (
    CandidateScore(
        score_id="condition_score_v5_direction_axis_scaled_jvp_ratio",
        role="direction_guardrail",
        formula="raw log scaled-JVP spectral/Frobenius squared ratio",
        weights={"log_scaled_jvp_ratio": 1.0},
        raw_axis=True,
        predicts_direction_threshold=True,
    ),
    CandidateScore(
        score_id="condition_score_v5_raw_fro_amplitude_axis",
        role="diagnostic_residual_axis",
        formula="source-standardized raw Frobenius matched-head-gain scaled-JVP amplitude",
        weights={"log_scaled_jvp_fro_amplitude": 1.0},
    ),
    CandidateScore(
        score_id="condition_score_v5_transport_normalized_amplitude_minus_direction",
        role="residual_candidate",
        formula=(
            "source-standardized Frobenius amplitude minus direction ratio, with "
            "pre-registered downsample/classifier transport and early-depth nuisance penalties"
        ),
        weights={
            "log_scaled_jvp_fro_amplitude": 1.0,
            "log_scaled_jvp_ratio": -1.0,
            "log_early_layer_prior": -0.5,
            "transport_is_downsample": -0.25,
            "transport_is_classifier": -0.25,
        },
    ),
    CandidateScore(
        score_id="condition_score_v5_transport_defect_penalty",
        role="diagnostic_transport_axis",
        formula="absolute source-standardized amplitude defect plus downsample/classifier transport tags",
        weights={
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
        raw_axis=True,
    ),
)


def standardized_values(frame: pd.DataFrame, stats: dict[str, tuple[float, float]], column: str) -> np.ndarray:
    mean, std = stats[column]
    return (frame[column].to_numpy(dtype=float) - mean) / std


def score_values(candidate: CandidateScore, frame: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    if candidate.raw_axis:
        feature = next(iter(candidate.weights))
        return frame[feature].to_numpy(dtype=float) * float(candidate.weights[feature])
    if candidate.score_id == "condition_score_v5_transport_defect_penalty":
        amplitude_defect = np.abs(standardized_values(frame, stats, "log_scaled_jvp_fro_amplitude"))
        return (
            amplitude_defect
            + 0.5 * frame["transport_is_downsample"].to_numpy(dtype=float)
            + 0.5 * frame["transport_is_classifier"].to_numpy(dtype=float)
        )
    values = np.zeros(len(frame), dtype=float)
    for feature, weight in candidate.weights.items():
        if feature.startswith("transport_is_"):
            values += float(weight) * frame[feature].to_numpy(dtype=float)
        else:
            values += float(weight) * standardized_values(frame, stats, feature)
    return values


def empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
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


def score_validation_pairs(source_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_step in sorted(int(value) for value in source_frame["warmup_steps"].unique()):
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
        for target_step in sorted(int(value) for value in validation_frame["warmup_steps"].unique()):
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
                        "split_id": "v5_validation_cifar100lt_mod4_partition",
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
                spearman, spearman_low, spearman_high, points = corr_ci95(
                    pd.Series(source_values),
                    pd.Series(target_residual_matched.to_numpy(dtype=float)),
                    method="spearman",
                )
                pearson, pearson_low, pearson_high, _ = corr_ci95(
                    pd.Series(source_values),
                    pd.Series(target_residual_matched.to_numpy(dtype=float)),
                    method="pearson",
                )
                target_below_one_matched = (
                    target_matched["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].astype(float) < 1.0
                )
                predicted_below_one = matched["source_log_observed_positive_control"].to_numpy(dtype=float) < 0.0
                target_top5_matched = target_top_k(target_matched, target_residual_matched)
                predicted_top5 = set(
                    pd.DataFrame({"parameter": target_matched["parameter"], "score": source_values})
                    .nlargest(min(5, len(target_matched)), "score")["parameter"]
                    .astype(str)
                )
                rows.append(
                    {
                        "split_id": "v5_validation_cifar100lt_mod4_partition",
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
                        "top5_residual_risk_overlap_fraction": len(predicted_top5 & target_top5_matched)
                        / max(len(target_top5_matched), 1),
                        "threshold_below_one_accuracy": float(
                            (predicted_below_one == target_below_one_matched.to_numpy()).mean()
                        ),
                        "target_observed_below_one_fraction": float(target_below_one_matched.mean()),
                        "predicted_below_one_fraction": float(predicted_below_one.mean()),
                        "frozen_depth_intercept": float(depth_intercept),
                        "frozen_depth_slope": float(depth_slope),
                    }
                )
    return pd.DataFrame(rows) if rows else empty_pairs()


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


def freeze_status(summary: pd.DataFrame, validation_generated: bool) -> pd.DataFrame:
    selected_score, status = select_primary_score(summary)
    final_outputs_present = any(path.exists() for path in FINAL_LAYER_SUMMARIES)
    return pd.DataFrame(
        [
            {
                "item": "v5 validation split output",
                "status": "generated" if validation_generated else "not_run",
                "evidence": VALIDATION_LAYER_SUMMARY.as_posix(),
            },
            {
                "item": "v5 candidate pool",
                "status": "registered",
                "evidence": "candidate formulas are fixed in scripts/e11_freeze_condition_score_v5_validation.py",
            },
            {
                "item": "v5 transport-normalized residual score",
                "status": status,
                "evidence": selected_score,
            },
            {
                "item": "v5 final split outputs",
                "status": "generated_before_freeze" if final_outputs_present else "not_run",
                "evidence": "; ".join(path.as_posix() for path in FINAL_LAYER_SUMMARIES),
            },
            {
                "item": "v5 spent-final quarantine",
                "status": "enforced",
                "evidence": (V5_THEORY_DIR / "spent_evidence_policy.csv").as_posix(),
            },
        ]
    )


def formula_registry(summary: pd.DataFrame, validation_generated: bool) -> pd.DataFrame:
    selected_score, status = select_primary_score(summary)
    rows = [
        {
            "score_id": candidate.score_id,
            "role": candidate.role,
            "formula": candidate.formula,
            "feature_weights": json.dumps(candidate.weights, sort_keys=True),
            "theory_contract": "sandwiched tail-drift plus partition/architecture transport normalization",
            "normalization_source": "raw log axis" if candidate.raw_axis else "source calibration step mean/std",
            "uses_spent_final_rows": "no",
            "validation_selection_rule": (
                "eligible residual candidates require validation Spearman CI lower endpoint above zero; "
                "select highest lower endpoint, then highest mean Spearman"
                if candidate.role == "residual_candidate"
                else "reported but not eligible as the residual primary"
            ),
            "selected_for_final_evaluation": "yes"
            if validation_generated and status == "frozen" and candidate.score_id == selected_score
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
            "theory_contract": "upper-bound transfer control; not a score candidate",
            "normalization_source": "source observed drift; not eligible for final claim",
            "uses_spent_final_rows": "no final-target tuning",
            "validation_selection_rule": "reported only as an upper-bound transfer control",
            "selected_for_final_evaluation": "no",
        }
    )
    rows.append(
        {
            "score_id": "condition_score_v5_validation_selected",
            "role": "primary_alias",
            "formula": selected_score,
            "feature_weights": "{}",
            "theory_contract": "points to selected v5 residual candidate after validation",
            "normalization_source": "pending until validation output exists",
            "uses_spent_final_rows": "no",
            "validation_selection_rule": "not final-ready unless status is frozen",
            "selected_for_final_evaluation": "yes" if validation_generated and status == "frozen" else "no",
        }
    )
    return pd.DataFrame(rows)


def gate_report(summary: pd.DataFrame, status: pd.DataFrame, validation_generated: bool) -> pd.DataFrame:
    status_lookup = status.set_index("item")["status"].to_dict()
    evidence_lookup = status.set_index("item")["evidence"].to_dict()
    final_outputs_absent = status_lookup.get("v5 final split outputs") == "not_run"
    selected_status = status_lookup.get("v5 transport-normalized residual score")
    selected_score = evidence_lookup.get("v5 transport-normalized residual score")
    direction = summary[summary["score"].eq("condition_score_v5_direction_axis_scaled_jvp_ratio")]
    direction_pass = (
        not direction.empty
        and float(direction.iloc[0]["mean_threshold_below_one_accuracy"]) >= 0.8
        and float(direction.iloc[0]["threshold_below_one_accuracy_ci95_low"]) >= 0.8
    )
    selected_summary = summary[summary["score"].eq(selected_score)] if selected_score in set(summary["score"]) else pd.DataFrame()
    selected_residual_pass = (
        not selected_summary.empty
        and float(selected_summary.iloc[0]["spearman_ci95_low"]) > 0.0
        and selected_status == "frozen"
    )
    return pd.DataFrame(
        [
            {
                "gate_id": "V5F-1-validation-output",
                "scope": "validation-only split",
                "status": "pass" if validation_generated else "not_run",
                "evidence": VALIDATION_LAYER_SUMMARY.as_posix(),
            },
            {
                "gate_id": "V5F-2-no-final-before-freeze",
                "scope": "unspent final splits",
                "status": "pass" if final_outputs_absent else "fail",
                "evidence": "no v5 final layer_summary.csv exists before a frozen validation score",
            },
            {
                "gate_id": "V5F-3-spent-final-quarantine",
                "scope": "v2/v3/v4 final rows",
                "status": "pass" if status_lookup.get("v5 spent-final quarantine") == "enforced" else "fail",
                "evidence": "spent evidence policy forbids fitting, feature selection, threshold tuning, and final P0 evidence",
            },
            {
                "gate_id": "V5F-4-residual-score-freeze",
                "scope": "transport-normalized residual score",
                "status": "pass" if selected_residual_pass else selected_status,
                "evidence": selected_score,
            },
            {
                "gate_id": "V5F-5-direction-threshold-guardrail",
                "scope": "direction guardrail",
                "status": "pass" if direction_pass else ("not_run" if not validation_generated else "fail"),
                "evidence": "direction threshold accuracy lower endpoint must be at least 0.8",
            },
            {
                "gate_id": "V5F-6-final-claim-readiness",
                "scope": "P0 predictive-condition claim",
                "status": "pass" if selected_residual_pass and direction_pass and final_outputs_absent else "not_ready",
                "evidence": "new final splits may run only after validation freeze passes",
            },
        ]
    )


def write_discussion(
    formulas: pd.DataFrame,
    status: pd.DataFrame,
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    validation_generated: bool,
) -> None:
    if validation_generated and not summary.empty:
        selected = status.set_index("item").loc["v5 transport-normalized residual score", "evidence"]
        selected_summary = summary[summary["score"].eq(selected)]
        selected_text = (
            "No residual candidate passed validation."
            if selected_summary.empty
            else (
                f"Selected residual candidate `{selected}` has validation residual Spearman "
                f"{fmt(selected_summary.iloc[0]['mean_spearman_score_vs_target_residual'])} "
                f"[{fmt(selected_summary.iloc[0]['spearman_ci95_low'])}, {fmt(selected_summary.iloc[0]['spearman_ci95_high'])}]."
            )
        )
    else:
        selected_text = "Validation output is not present yet, so no v5 residual score is frozen."
    gate_lookup = gates.set_index("gate_id")["status"].to_dict()
    if gate_lookup.get("V5F-6-final-claim-readiness") == "pass":
        boundary_text = (
            "Final evaluation is now unblocked as a run, not as a claim: the "
            "ResNeXt50-32x4d and CIFAR-10 cross-partition final splits may be "
            "evaluated with the frozen validation-selected score. A P0 claim still "
            "requires both final splits to pass their registered residual and "
            "direction gates."
        )
    else:
        boundary_text = (
            "Blocked now: the v5 final architecture and data splits cannot support "
            "a P0 claim until this artifact reports a frozen residual score, a "
            "passing direction guardrail, and no generated final outputs before the freeze."
        )
    text = f"""# E11 Condition-Score V5 Validation Freeze

This generated artifact is the freeze boundary for the v5 theory protocol. It
is allowed to run before the validation Slurm job finishes; in that state it
writes `not_run`/`not_ready` rows and keeps the final splits blocked. Once the
validation-only mod-4 CIFAR-100-LT split exists, the same script evaluates the
registered v5 candidate pool and freezes a transport-normalized residual score
only if the validation residual-ranking gate passes.

## Formula Registry

{markdown_table(formulas, ["score_id", "role", "formula", "theory_contract", "normalization_source", "uses_spent_final_rows", "selected_for_final_evaluation"])}

## Freeze Status

{markdown_table(status, ["item", "status", "evidence"])}

## Validation Score Summary

{markdown_table(summary, SUMMARY_COLUMNS)}

## Gate Report

{markdown_table(gates, ["gate_id", "scope", "status", "evidence"])}

## Boundary

{selected_text}

{boundary_text}

Artifacts:
- [score_formula_registry.csv](../{(OUTPUT_DIR / 'score_formula_registry.csv').as_posix()})
- [freeze_status.csv](../{(OUTPUT_DIR / 'freeze_status.csv').as_posix()})
- [validation_gate_report.csv](../{(OUTPUT_DIR / 'validation_gate_report.csv').as_posix()})
- [validation_score_summary.csv](../{(OUTPUT_DIR / 'validation_score_summary.csv').as_posix()})
- [validation_score_pairs.csv](../{(OUTPUT_DIR / 'validation_score_pairs.csv').as_posix()})
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
        pairs = empty_pairs()
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    status = freeze_status(summary, validation_generated)
    formulas = formula_registry(summary, validation_generated)
    gates = gate_report(summary, status, validation_generated)
    formulas.to_csv(OUTPUT_DIR / "score_formula_registry.csv", index=False)
    status.to_csv(OUTPUT_DIR / "freeze_status.csv", index=False)
    pairs.to_csv(OUTPUT_DIR / "validation_score_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "validation_score_summary.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "validation_gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "validation_generated": bool(validation_generated),
                "validation_layer_summary": VALIDATION_LAYER_SUMMARY.as_posix(),
                "validation_metrics": VALIDATION_METRICS.as_posix(),
                "final_layer_summaries": [path.as_posix() for path in FINAL_LAYER_SUMMARIES],
                "candidate_scores": [candidate.score_id for candidate in CANDIDATES],
                "theory_protocol": "discussion/e11_condition_score_v5_theory_protocol.md",
                "selection_rule": "freeze highest residual-candidate CI lower endpoint above zero after validation; direction guardrail must pass separately",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(formulas, status, summary, gates, validation_generated)
    print(f"saved condition-score v5 validation freeze to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
