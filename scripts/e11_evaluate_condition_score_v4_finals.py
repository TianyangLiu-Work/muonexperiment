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
from scripts.e11_freeze_condition_score_v4_validation import CANDIDATES
from scripts.e11_freeze_condition_score_v4_validation import SOURCE_LAYER_SUMMARY
from scripts.e11_freeze_condition_score_v4_validation import SOURCE_METRICS
from scripts.e11_freeze_condition_score_v4_validation import V4_DIR
from scripts.e11_freeze_condition_score_v4_validation import load_axis_frame
from scripts.e11_freeze_condition_score_v4_validation import score_values
from scripts.e11_freeze_condition_score_v4_validation import source_feature_stats
from scripts.e11_freeze_condition_score_v4_validation import target_top_k
from scripts.e11_write_cifar100_resnet_condition_score_next import fit_depth_baseline
from scripts.e11_write_cifar100_resnet_condition_score_next import target_residual_from_source_depth


FREEZE_DIR = V4_DIR / "validation_score_freeze"
OUTPUT_DIR = V4_DIR / "final_score_evaluation"
DISCUSSION_PATH = Path("discussion/e11_condition_score_v4_final_evaluation.md")


@dataclass(frozen=True)
class FinalSplit:
    split_id: str
    role: str
    display_name: str
    layer_summary_path: Path
    metrics_path: Path
    config_path: Path


FINAL_SPLITS = (
    FinalSplit(
        split_id="v4_final_architecture_wide_resnet50_2_cifar100lt",
        role="fresh_final_heldout_architecture",
        display_name="WideResNet50-2 CIFAR-100-LT final architecture split",
        layer_summary_path=V4_DIR / "final_architecture_wide_resnet50_2" / "layer_summary.csv",
        metrics_path=V4_DIR / "final_architecture_wide_resnet50_2" / "metrics.csv",
        config_path=V4_DIR / "final_architecture_wide_resnet50_2" / "config.json",
    ),
    FinalSplit(
        split_id="v4_final_data_cifar10lt_mixed_partition",
        role="fresh_final_heldout_data_partition",
        display_name="CIFAR-10-LT mixed-partition final data split",
        layer_summary_path=V4_DIR / "final_data_cifar10_mixed" / "layer_summary.csv",
        metrics_path=V4_DIR / "final_data_cifar10_mixed" / "metrics.csv",
        config_path=V4_DIR / "final_data_cifar10_mixed" / "config.json",
    ),
)


PAIR_COLUMNS = [
    "split_id",
    "split_role",
    "split_display_name",
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


SUMMARY_COLUMNS = [
    "split_id",
    "split_role",
    "split_display_name",
    "score",
    "score_role",
    "final_transfer_pairs",
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


def selected_score_id() -> str:
    status = pd.read_csv(FREEZE_DIR / "freeze_status.csv").set_index("item")
    selected = str(status.loc["v4 selected residual score", "evidence"])
    selected_status = str(status.loc["v4 selected residual score", "status"])
    if selected_status != "frozen":
        raise ValueError(f"v4 final evaluation requires a frozen selected score, got {selected_status}")
    return selected


def candidate_by_id(score_id: str):
    for candidate in CANDIDATES:
        if candidate.score_id == score_id:
            return candidate
    raise ValueError(f"unknown v4 candidate score: {score_id}")


def score_final_split(
    split: FinalSplit,
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    primary_score: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary_candidate = candidate_by_id(primary_score)
    direction_candidate = candidate_by_id("condition_score_v4_direction_axis_scaled_jvp_ratio")
    early_candidate = candidate_by_id("early_layer_prior")
    scored_candidates = (primary_candidate, direction_candidate, early_candidate)
    source_steps = sorted(int(value) for value in source_frame["warmup_steps"].unique())
    target_steps = sorted(int(value) for value in target_frame["warmup_steps"].unique())

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
            target = target_frame[target_frame["warmup_steps"].eq(target_step)].copy()
            target_residual = target_residual_from_source_depth(target, float(depth_intercept), float(depth_slope))
            target_below_one = (
                target["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].astype(float) < 1.0
            )
            target_top5 = target_top_k(target, target_residual)
            target_series = pd.Series(target_residual.to_numpy(dtype=float))
            for candidate in scored_candidates:
                values = score_values(candidate, target, stats)
                spearman, spearman_low, spearman_high, points = corr_ci95(
                    pd.Series(values),
                    target_series,
                    method="spearman",
                )
                pearson, pearson_low, pearson_high, _ = corr_ci95(
                    pd.Series(values),
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
                        "split_id": split.split_id,
                        "split_role": split.role,
                        "split_display_name": split.display_name,
                        "source_warmup_steps": int(source_step),
                        "target_warmup_steps": int(target_step),
                        "score": candidate.score_id,
                        "score_role": "primary_candidate"
                        if candidate.score_id == primary_score
                        else candidate.role,
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
                target_top5 = target_top_k(target_matched, target_residual_matched)
                predicted_top5 = set(
                    pd.DataFrame({"parameter": target_matched["parameter"], "score": source_values})
                    .nlargest(min(5, len(target_matched)), "score")["parameter"]
                    .astype(str)
                )
                rows.append(
                    {
                        "split_id": split.split_id,
                        "split_role": split.role,
                        "split_display_name": split.display_name,
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
    for (split_id, score), group in pairs.groupby(["split_id", "score"], observed=True, sort=False):
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
                "split_id": str(split_id),
                "split_role": str(group["split_role"].iloc[0]),
                "split_display_name": str(group["split_display_name"].iloc[0]),
                "score": str(score),
                "score_role": str(group["score_role"].iloc[0]),
                "final_transfer_pairs": int(len(group)),
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


def gate_report(summary: pd.DataFrame, primary_score: str) -> pd.DataFrame:
    rows = []
    all_final_pass = True
    for split in FINAL_SPLITS:
        if not split.layer_summary_path.exists():
            all_final_pass = False
            rows.append(
                {
                    "gate_id": f"{split.role}_generated",
                    "scope": split.display_name,
                    "status": "not_run",
                    "evidence": f"missing final layer summary at {split.layer_summary_path}",
                }
            )
            continue
        split_summary = summary[summary["split_id"].eq(split.split_id)]
        primary = split_summary[split_summary["score"].eq(primary_score)]
        direction = split_summary[split_summary["score"].eq("condition_score_v4_direction_axis_scaled_jvp_ratio")]
        early = split_summary[split_summary["score"].eq("early_layer_prior")]
        source_control = split_summary[split_summary["score"].eq("source_observed_drift_positive_control")]
        baselines_reported = not direction.empty and not early.empty and not source_control.empty
        if primary.empty:
            all_final_pass = False
            rows.append(
                {
                    "gate_id": f"{split.role}_primary_generated",
                    "scope": split.display_name,
                    "status": "not_run",
                    "evidence": f"primary score {primary_score} row is missing",
                }
            )
            continue
        primary_row = primary.iloc[0]
        residual_pass = float(primary_row["spearman_ci95_low"]) > 0.0
        direction_row = direction.iloc[0] if not direction.empty else None
        threshold_pass = direction_row is not None and float(direction_row["mean_threshold_below_one_accuracy"]) >= 0.8
        all_final_pass = all_final_pass and residual_pass and threshold_pass and baselines_reported
        rows.extend(
            [
                {
                    "gate_id": f"{split.role}_residual_spearman",
                    "scope": split.display_name,
                    "status": "pass" if residual_pass else "fail",
                    "evidence": (
                        f"primary residual Spearman={fmt(primary_row['mean_spearman_score_vs_target_residual'])} "
                        f"CI=[{fmt(primary_row['spearman_ci95_low'])}, {fmt(primary_row['spearman_ci95_high'])}]"
                    ),
                },
                {
                    "gate_id": f"{split.role}_direction_threshold_accuracy",
                    "scope": split.display_name,
                    "status": "pass" if threshold_pass else "fail",
                    "evidence": (
                        "direction-axis below-one threshold accuracy="
                        f"{fmt(direction_row['mean_threshold_below_one_accuracy'])}"
                        if direction_row is not None
                        else "direction-axis row missing"
                    ),
                },
                {
                    "gate_id": f"{split.role}_baselines_reported",
                    "scope": split.display_name,
                    "status": "pass" if baselines_reported else "fail",
                    "evidence": "direction-axis, early-prior, and source-observed controls reported",
                },
            ]
        )
    rows.append(
        {
            "gate_id": "v4_p0_predictive_condition_claim",
            "scope": "v4 condition-score final evaluation",
            "status": "pass" if all_final_pass else "not_ready",
            "evidence": "Both unspent final splits must pass residual, direction, and reporting gates.",
        }
    )
    return pd.DataFrame(rows)


def write_discussion(summary: pd.DataFrame, gates: pd.DataFrame, primary_score: str) -> None:
    if summary.empty:
        summary_section = "No v4 final score rows are available yet."
    else:
        summary_section = markdown_table(
            summary,
            [
                "split_role",
                "score",
                "score_role",
                "final_transfer_pairs",
                "mean_spearman_score_vs_target_residual",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "mean_threshold_below_one_accuracy",
            ],
        )
    text = f"""# E11 Condition-Score V4 Final Evaluation

This generated evaluator applies the validation-frozen v4 score
`{primary_score}` to the two unspent final splits. It does not refit
coefficients, does not use v2/v3 final split outcomes, and reports direction,
early-layer, and source-observed controls beside the primary residual score.

## Final Score Summary

{summary_section}

## Gate Report

{markdown_table(gates, ["gate_id", "scope", "status", "evidence"])}

## Claim Boundary

`not_run` is the expected state until both final Slurm jobs finish. `not_ready`
remains the correct P0 state unless both unspent final splits pass all gates.

Artifacts:
- [final_score_pairs.csv](../{(OUTPUT_DIR / 'final_score_pairs.csv').as_posix()})
- [final_score_summary.csv](../{(OUTPUT_DIR / 'final_score_summary.csv').as_posix()})
- [final_gate_report.csv](../{(OUTPUT_DIR / 'final_gate_report.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_score = selected_score_id()
    source_frame = load_axis_frame(SOURCE_LAYER_SUMMARY, SOURCE_METRICS)
    frames = []
    for split in FINAL_SPLITS:
        if split.layer_summary_path.exists() and split.metrics_path.exists():
            target_frame = load_axis_frame(split.layer_summary_path, split.metrics_path)
            frames.append(score_final_split(split, source_frame, target_frame, primary_score))
    pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLUMNS)
    summary = summarize_pairs(pairs)
    gates = gate_report(summary, primary_score)
    pairs.to_csv(OUTPUT_DIR / "final_score_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "final_score_summary.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "final_gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "source_layer_summary": SOURCE_LAYER_SUMMARY.as_posix(),
                "source_metrics": SOURCE_METRICS.as_posix(),
                "freeze_dir": FREEZE_DIR.as_posix(),
                "primary_score": primary_score,
                "final_splits": [
                    {
                        "split_id": split.split_id,
                        "role": split.role,
                        "display_name": split.display_name,
                        "layer_summary_path": split.layer_summary_path.as_posix(),
                        "metrics_path": split.metrics_path.as_posix(),
                        "generated": split.layer_summary_path.exists() and split.metrics_path.exists(),
                    }
                    for split in FINAL_SPLITS
                ],
                "analysis_scope": "v4 final split evaluator; not_ready unless both unspent final splits pass",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(summary, gates, primary_score)
    print(f"saved v4 final evaluation to {OUTPUT_DIR}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
