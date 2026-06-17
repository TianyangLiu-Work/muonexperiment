from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown
from e11_condition_geometry.statistics import ci95, corr_ci95
from scripts.e11_evaluate_cifar100_resnet_condition_score_heldouts import load_frozen_fits
from scripts.e11_write_cifar100_resnet_condition_score_next import add_protocol_features
from scripts.e11_write_cifar100_resnet_condition_score_next import fit_depth_baseline
from scripts.e11_write_cifar100_resnet_condition_score_next import predict_residual
from scripts.e11_write_cifar100_resnet_condition_score_next import target_residual_from_source_depth


FRESH_PROTOCOL_DIR = Path("results/e11_condition_score_fresh_protocol")
SOURCE_LAYER_SUMMARY = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv")
V2_COEFFICIENTS_PATH = Path("results/e11_cifar100_resnet_condition_score_next/calibration_coefficients.csv")
OUTPUT_DIR = FRESH_PROTOCOL_DIR / "fresh_score_evaluation"
FIGURE_DIR = Path("figures/e11_condition_score_fresh_protocol/fresh_score_evaluation")
DISCUSSION_PATH = Path("discussion/e11_condition_score_fresh_evaluation.md")


@dataclass(frozen=True)
class FreshSplit:
    split_id: str
    role: str
    display_name: str
    layer_summary_path: Path
    config_path: Path


FRESH_SPLITS = (
    FreshSplit(
        split_id="fresh_final_architecture_resnet50_cifar100lt",
        role="fresh_final_heldout_architecture",
        display_name="Fresh ResNet50 CIFAR-100-LT architecture split",
        layer_summary_path=FRESH_PROTOCOL_DIR / "fresh_architecture_resnet50" / "layer_summary.csv",
        config_path=FRESH_PROTOCOL_DIR / "fresh_architecture_resnet50" / "config.json",
    ),
    FreshSplit(
        split_id="fresh_final_data_cifar10lt_alt_partition",
        role="fresh_final_heldout_data_partition",
        display_name="Fresh CIFAR-10-LT alternate-partition data split",
        layer_summary_path=FRESH_PROTOCOL_DIR / "fresh_data_cifar10_alt" / "layer_summary.csv",
        config_path=FRESH_PROTOCOL_DIR / "fresh_data_cifar10_alt" / "config.json",
    ),
)


SCORE_FAMILIES = {
    "condition_score_v3_zero_fit_scaled_jvp": "primary_candidate",
    "condition_score_v2_calibrated_residual_spent_baseline": "retired_baseline",
    "early_layer_prior": "baseline",
    "source_observed_drift_positive_control": "positive_control",
}


PAIR_COLUMNS = [
    "split_id",
    "split_role",
    "split_display_name",
    "source_warmup_steps",
    "target_warmup_steps",
    "score",
    "score_family",
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


def _target_top_k(target: pd.DataFrame, target_residual: pd.Series) -> set[str]:
    return set(
        pd.DataFrame({"parameter": target["parameter"], "target_residual": target_residual})
        .nlargest(min(5, len(target)), "target_residual")["parameter"]
        .astype(str)
    )


def _score_rows_for_frame(
    *,
    split: FreshSplit,
    source_step: int,
    target_step: int,
    target: pd.DataFrame,
    target_residual: pd.Series,
    score_values: dict[str, np.ndarray],
    predicted_log_values: dict[str, np.ndarray],
    frozen_depth_intercept: float,
    frozen_depth_slope: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_below_one = target["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"].astype(float) < 1.0
    target_top5 = _target_top_k(target, target_residual)
    target_series = pd.Series(target_residual.to_numpy(dtype=float))
    for score_name, values in score_values.items():
        score_series = pd.Series(np.asarray(values, dtype=float))
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
        if score_name in predicted_log_values:
            predicted_below_one = np.asarray(predicted_log_values[score_name], dtype=float) < 0.0
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
                "score": score_name,
                "score_family": SCORE_FAMILIES[score_name],
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
                "frozen_depth_intercept": float(frozen_depth_intercept),
                "frozen_depth_slope": float(frozen_depth_slope),
            }
        )
    return rows


def score_fresh_split(
    split: FreshSplit,
    target_layer_summary: pd.DataFrame,
    source_layer_summary: pd.DataFrame,
    v2_fits: dict[int, dict[str, object]],
) -> pd.DataFrame:
    target_frame = add_protocol_features(target_layer_summary)
    source_frame = add_protocol_features(source_layer_summary)
    rows: list[dict[str, object]] = []
    source_steps = sorted(int(value) for value in source_frame["warmup_steps"].unique())

    for source_step in source_steps:
        source = source_frame[source_frame["warmup_steps"].eq(source_step)].copy()
        if source.empty:
            continue
        depth_intercept, depth_slope, source_residual = fit_depth_baseline(source)
        source_control = pd.DataFrame(
            {
                "parameter": source["parameter"].astype(str).to_numpy(),
                "source_observed_residual_positive_control": source_residual.to_numpy(dtype=float),
                "source_log_observed_positive_control": source["log_observed"].to_numpy(dtype=float),
            }
        )
        v2_fit = v2_fits.get(int(source_step))
        for target_step in sorted(int(value) for value in target_frame["warmup_steps"].unique()):
            target = target_frame[target_frame["warmup_steps"].eq(target_step)].copy()
            target_residual = target_residual_from_source_depth(
                target,
                float(depth_intercept),
                float(depth_slope),
            )
            score_values = {
                "condition_score_v3_zero_fit_scaled_jvp": target["log_scaled_jvp_ratio"].to_numpy(dtype=float),
                "early_layer_prior": target["log_early_layer_prior"].to_numpy(dtype=float),
            }
            predicted_log_values = {
                "condition_score_v3_zero_fit_scaled_jvp": target["log_scaled_jvp_ratio"].to_numpy(dtype=float),
            }
            if v2_fit is not None:
                v2_residual = predict_residual(target, v2_fit)
                v2_predicted_log_observed = (
                    float(v2_fit["depth_intercept"])
                    + float(v2_fit["depth_slope"]) * target["log_early_layer_prior"].to_numpy(dtype=float)
                    + v2_residual
                )
                score_values["condition_score_v2_calibrated_residual_spent_baseline"] = v2_residual
                predicted_log_values[
                    "condition_score_v2_calibrated_residual_spent_baseline"
                ] = v2_predicted_log_observed
            rows.extend(
                _score_rows_for_frame(
                    split=split,
                    source_step=source_step,
                    target_step=target_step,
                    target=target,
                    target_residual=target_residual,
                    score_values=score_values,
                    predicted_log_values=predicted_log_values,
                    frozen_depth_intercept=float(depth_intercept),
                    frozen_depth_slope=float(depth_slope),
                )
            )

            matched = target[["parameter"]].astype({"parameter": str}).merge(
                source_control,
                on="parameter",
                how="inner",
            )
            if not matched.empty:
                target_matched = target[target["parameter"].astype(str).isin(set(matched["parameter"]))].copy()
                target_matched = matched[["parameter"]].merge(target_matched, on="parameter", how="inner")
                target_residual_matched = target_residual_from_source_depth(
                    target_matched,
                    float(depth_intercept),
                    float(depth_slope),
                )
                rows.extend(
                    _score_rows_for_frame(
                        split=split,
                        source_step=source_step,
                        target_step=target_step,
                        target=target_matched,
                        target_residual=target_residual_matched,
                        score_values={
                            "source_observed_drift_positive_control": matched[
                                "source_observed_residual_positive_control"
                            ].to_numpy(dtype=float)
                        },
                        predicted_log_values={
                            "source_observed_drift_positive_control": matched[
                                "source_log_observed_positive_control"
                            ].to_numpy(dtype=float)
                        },
                        frozen_depth_intercept=float(depth_intercept),
                        frozen_depth_slope=float(depth_slope),
                    )
                )
    return pd.DataFrame(rows)


def summarize_score_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "split_id",
                "split_role",
                "split_display_name",
                "score",
                "score_family",
                "split_transfer_pairs",
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
        )
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
                "score_family": str(group["score_family"].iloc[0]),
                "split_transfer_pairs": int(len(group)),
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


def gate_report(summary: pd.DataFrame, splits: tuple[FreshSplit, ...]) -> pd.DataFrame:
    rows = []
    all_primary_pass = True
    for split in splits:
        if not split.layer_summary_path.exists():
            all_primary_pass = False
            rows.append(
                {
                    "gate_id": f"{split.role}_generated",
                    "scope": split.display_name,
                    "status": "not_run",
                    "evidence": f"missing fresh layer summary at {split.layer_summary_path}",
                }
            )
            continue
        split_summary = summary[summary["split_id"].eq(split.split_id)]
        primary = split_summary[split_summary["score"].eq("condition_score_v3_zero_fit_scaled_jvp")]
        early = split_summary[split_summary["score"].eq("early_layer_prior")]
        v2_baseline = split_summary[
            split_summary["score"].eq("condition_score_v2_calibrated_residual_spent_baseline")
        ]
        source_control = split_summary[split_summary["score"].eq("source_observed_drift_positive_control")]
        baselines_reported = not early.empty and not v2_baseline.empty and not source_control.empty
        if primary.empty:
            all_primary_pass = False
            rows.append(
                {
                    "gate_id": f"{split.role}_primary_generated",
                    "scope": split.display_name,
                    "status": "not_run",
                    "evidence": "fresh primary score row is missing",
                }
            )
            continue
        primary_row = primary.iloc[0]
        residual_pass = float(primary_row["spearman_ci95_low"]) > 0.0
        threshold_pass = float(primary_row["mean_threshold_below_one_accuracy"]) >= 0.8
        early_comparison_pass = not early.empty and float(
            primary_row["mean_spearman_score_vs_target_residual"]
        ) > float(early.iloc[0]["mean_spearman_score_vs_target_residual"])
        all_primary_pass = (
            all_primary_pass
            and residual_pass
            and threshold_pass
            and early_comparison_pass
            and baselines_reported
        )
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
                    "gate_id": f"{split.role}_threshold_accuracy",
                    "scope": split.display_name,
                    "status": "pass" if threshold_pass else "fail",
                    "evidence": (
                        "primary below-one threshold accuracy="
                        f"{fmt(primary_row['mean_threshold_below_one_accuracy'])}"
                    ),
                },
                {
                    "gate_id": f"{split.role}_early_prior_comparison",
                    "scope": split.display_name,
                    "status": "pass" if early_comparison_pass else "fail",
                    "evidence": (
                        f"primary Spearman={fmt(primary_row['mean_spearman_score_vs_target_residual'])}; "
                        f"early-layer prior Spearman={fmt(early.iloc[0]['mean_spearman_score_vs_target_residual'])}"
                        if not early.empty
                        else "early-layer prior row missing"
                    ),
                },
                {
                    "gate_id": f"{split.role}_baselines_reported",
                    "scope": split.display_name,
                    "status": "pass" if baselines_reported else "fail",
                    "evidence": "early prior, retired v2 baseline, and source-observed positive control reported",
                },
            ]
        )
    rows.append(
        {
            "gate_id": "fresh_p0_predictive_condition_claim",
            "scope": "fresh condition-score protocol",
            "status": "pass" if all_primary_pass else "not_ready",
            "evidence": "All fresh final splits must pass residual, threshold, baseline-comparison, and reporting gates.",
        }
    )
    return pd.DataFrame(rows)


def write_figure(summary: pd.DataFrame, figure_dir: Path) -> Path | None:
    if summary.empty:
        return None
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values(["split_role", "mean_spearman_score_vs_target_residual"])
    fig, ax = plt.subplots(figsize=(11.0, max(4.2, 0.35 * len(ordered))))
    y_positions = list(range(len(ordered)))
    estimates = ordered["mean_spearman_score_vs_target_residual"].to_numpy(dtype=float)
    low = ordered["spearman_ci95_low"].to_numpy(dtype=float)
    high = ordered["spearman_ci95_high"].to_numpy(dtype=float)
    colors = [
        "#009E73"
        if family == "primary_candidate"
        else "#CC79A7"
        if family == "positive_control"
        else "#D55E00"
        if family == "retired_baseline"
        else "#0072B2"
        for family in ordered["score_family"]
    ]
    labels = [
        f"{row.split_role.replace('fresh_final_', '')}: {row.score}"
        for row in ordered.itertuples(index=False)
    ]
    ax.barh(y_positions, estimates, color=colors, alpha=0.86)
    ax.errorbar(
        estimates,
        y_positions,
        xerr=[estimates - low, high - estimates],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("fresh residual layer-risk Spearman")
    ax.set_title("Fresh condition-score protocol evaluation")
    fig.tight_layout()
    path = figure_dir / "condition_score_fresh_evaluation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    figure_path: Path | None,
    discussion_path: Path,
) -> None:
    if figure_path is None:
        figure_md = "Fresh score figure is not generated until at least one fresh final split exists."
    else:
        figure_md = f"![Fresh condition-score evaluation](../{figure_path.as_posix()})"
    summary_section = (
        "No fresh final score rows are available yet."
        if summary.empty
        else markdown_table(
            summary,
            [
                "split_role",
                "score",
                "score_family",
                "split_transfer_pairs",
                "mean_spearman_score_vs_target_residual",
                "spearman_ci95_low",
                "spearman_ci95_high",
                "mean_top5_residual_risk_overlap_fraction",
                "mean_threshold_below_one_accuracy",
            ],
        )
    )
    text = f"""# E11 Fresh Condition-Score Evaluation

This generated evaluator was frozen before the fresh final Slurm outputs existed. It applies the registered `condition_score_v3_zero_fit_scaled_jvp` primary score, reports the retired v2 calibrated residual baseline, and keeps the spent ResNet34/original-CIFAR-10 held-outs out of fitting and final evidence.

{figure_md}

## Fresh Score Summary

{summary_section}

## Gate Report

{markdown_table(gates, ["gate_id", "scope", "status", "evidence"])}

## Boundary

`not_run` is the pre-output state for missing fresh final layer summaries. Once those summaries exist, `not_ready` remains the correct state unless both fresh final splits pass residual-Spearman, threshold-direction, baseline-comparison, and baseline-reporting gates.

Artifacts:
- [fresh_score_pairs.csv](../{(OUTPUT_DIR / 'fresh_score_pairs.csv').as_posix()})
- [fresh_score_summary.csv](../{(OUTPUT_DIR / 'fresh_score_summary.csv').as_posix()})
- [fresh_gate_report.csv](../{(OUTPUT_DIR / 'fresh_gate_report.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(discussion_path, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_layer_summary = pd.read_csv(SOURCE_LAYER_SUMMARY)
    v2_fits = load_frozen_fits(pd.read_csv(V2_COEFFICIENTS_PATH))
    frames = []
    for split in FRESH_SPLITS:
        if split.layer_summary_path.exists():
            frames.append(
                score_fresh_split(
                    split,
                    pd.read_csv(split.layer_summary_path),
                    source_layer_summary,
                    v2_fits,
                )
            )
    pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIR_COLUMNS)
    summary = summarize_score_pairs(pairs)
    gates = gate_report(summary, FRESH_SPLITS)
    pairs.to_csv(OUTPUT_DIR / "fresh_score_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "fresh_score_summary.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "fresh_gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "source_layer_summary": SOURCE_LAYER_SUMMARY.as_posix(),
                "v2_coefficients_path": V2_COEFFICIENTS_PATH.as_posix(),
                "fresh_splits": [
                    {
                        "split_id": split.split_id,
                        "role": split.role,
                        "display_name": split.display_name,
                        "layer_summary_path": split.layer_summary_path.as_posix(),
                        "generated": split.layer_summary_path.exists(),
                    }
                    for split in FRESH_SPLITS
                ],
                "primary_score": "condition_score_v3_zero_fit_scaled_jvp",
                "primary_score_rule": "zero-fit log scaled-JVP ratio; no spent held-out fitting",
                "analysis_scope": "fresh final split evaluator; not_ready unless all fresh split gates pass",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(summary, FIGURE_DIR)
    write_discussion(summary, gates, figure_path, DISCUSSION_PATH)
    print(f"saved fresh condition-score evaluation to {OUTPUT_DIR}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
