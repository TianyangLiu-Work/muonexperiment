from __future__ import annotations

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
from scripts.e11_evaluate_condition_score_v4_finals import FINAL_SPLITS
from scripts.e11_evaluate_condition_score_v4_finals import selected_score_id
from scripts.e11_freeze_condition_score_v4_validation import CANDIDATES
from scripts.e11_freeze_condition_score_v4_validation import SOURCE_LAYER_SUMMARY
from scripts.e11_freeze_condition_score_v4_validation import SOURCE_METRICS
from scripts.e11_freeze_condition_score_v4_validation import V4_DIR
from scripts.e11_freeze_condition_score_v4_validation import load_axis_frame
from scripts.e11_freeze_condition_score_v4_validation import score_values
from scripts.e11_freeze_condition_score_v4_validation import source_feature_stats
from scripts.e11_write_cifar100_resnet_condition_score_next import fit_depth_baseline
from scripts.e11_write_cifar100_resnet_condition_score_next import target_residual_from_source_depth


FINAL_SCORE_SUMMARY_PATH = V4_DIR / "final_score_evaluation" / "final_score_summary.csv"
FINAL_GATE_REPORT_PATH = V4_DIR / "final_score_evaluation" / "final_gate_report.csv"
OUTPUT_DIR = Path("results/e11_condition_score_v4_failure_mechanism_audit")
FIGURE_DIR = Path("figures/e11_condition_score_v4_failure_mechanism_audit")
DISCUSSION_PATH = Path("discussion/e11_condition_score_v4_failure_mechanism_audit.md")


def parameter_stage(parameter: str) -> str:
    if parameter == "conv1.weight":
        return "stem"
    if parameter.startswith("layer"):
        return parameter.split(".", maxsplit=1)[0]
    if parameter in {"fc.weight", "classifier.weight"}:
        return "classifier"
    return "other"


def block_term(parameter: str) -> str:
    if "downsample" in parameter:
        return "downsample"
    if parameter in {"conv1.weight"}:
        return "stem_conv"
    if parameter in {"fc.weight", "classifier.weight"}:
        return "classifier"
    parts = parameter.split(".")
    return parts[-2] if len(parts) >= 2 else "other"


def candidate_lookup() -> dict[str, object]:
    return {candidate.score_id: candidate for candidate in CANDIDATES}


def top_k_mask(values: pd.Series, k: int = 5) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    top_index = values.nlargest(min(k, len(values))).index
    return values.index.isin(top_index)


def build_layer_diagnostics(source_frame: pd.DataFrame, primary_score: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidates = candidate_lookup()
    direction = candidates["condition_score_v4_direction_axis_scaled_jvp_ratio"]
    amplitude = candidates["condition_score_v4_fro_amplitude_axis"]
    primary = candidates[primary_score]
    early = candidates["early_layer_prior"]

    for split in FINAL_SPLITS:
        if not split.layer_summary_path.exists() or not split.metrics_path.exists():
            continue
        target_frame = load_axis_frame(split.layer_summary_path, split.metrics_path)
        for source_step in sorted(int(value) for value in source_frame["warmup_steps"].unique()):
            source = source_frame[source_frame["warmup_steps"].eq(source_step)].copy()
            depth_intercept, depth_slope, _source_residual = fit_depth_baseline(source)
            stats = source_feature_stats(source)
            for target_step in sorted(int(value) for value in target_frame["warmup_steps"].unique()):
                target = target_frame[target_frame["warmup_steps"].eq(target_step)].copy()
                target_residual = target_residual_from_source_depth(
                    target,
                    float(depth_intercept),
                    float(depth_slope),
                )
                frame = pd.DataFrame(
                    {
                        "split_id": split.split_id,
                        "split_role": split.role,
                        "source_warmup_steps": int(source_step),
                        "target_warmup_steps": int(target_step),
                        "parameter": target["parameter"].astype(str).to_numpy(),
                        "layer_index": target["layer_index"].astype(int).to_numpy(),
                        "stage": target["parameter"].astype(str).map(parameter_stage).to_numpy(),
                        "block_term": target["parameter"].astype(str).map(block_term).to_numpy(),
                        "target_residual_log_observed": target_residual.to_numpy(dtype=float),
                        "primary_amplitude_minus_direction_score": score_values(primary, target, stats),
                        "direction_log_scaled_jvp_ratio": score_values(direction, target, stats),
                        "fro_amplitude_score": score_values(amplitude, target, stats),
                        "early_layer_prior_score": score_values(early, target, stats),
                    }
                )
                for column in [
                    "target_residual_log_observed",
                    "primary_amplitude_minus_direction_score",
                    "direction_log_scaled_jvp_ratio",
                    "fro_amplitude_score",
                    "early_layer_prior_score",
                ]:
                    frame[f"{column}_top5"] = top_k_mask(frame[column], k=5)
                rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_score_columns(layer_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_columns = {
        "condition_score_v4_two_axis_amplitude_minus_direction": "primary_amplitude_minus_direction_score",
        "v4_direction_axis_scaled_jvp_ratio": "direction_log_scaled_jvp_ratio",
        "v4_residual_amplitude_axis_scaled_jvp_fro": "fro_amplitude_score",
        "early_layer_prior": "early_layer_prior_score",
    }
    for (split_id, split_role), split_group in layer_diagnostics.groupby(
        ["split_id", "split_role"],
        observed=True,
        sort=False,
    ):
        target = split_group["target_residual_log_observed"]
        target_top5 = split_group["target_residual_log_observed_top5"].astype(bool)
        for score_id, column in score_columns.items():
            spearman, spearman_low, spearman_high, points = corr_ci95(
                split_group[column],
                target,
                method="spearman",
            )
            pearson, pearson_low, pearson_high, _ = corr_ci95(
                split_group[column],
                target,
                method="pearson",
            )
            predicted_top5 = split_group[f"{column}_top5"].astype(bool)
            top5_overlap = float((target_top5 & predicted_top5).sum() / max(int(target_top5.sum()), 1))
            rows.append(
                {
                    "split_id": str(split_id),
                    "split_role": str(split_role),
                    "score_id": score_id,
                    "points": int(points),
                    "spearman_score_vs_target_residual": spearman,
                    "spearman_ci95_low": spearman_low,
                    "spearman_ci95_high": spearman_high,
                    "pearson_score_vs_target_residual": pearson,
                    "pearson_ci95_low": pearson_low,
                    "pearson_ci95_high": pearson_high,
                    "top5_residual_overlap_fraction": top5_overlap,
                }
            )
    return pd.DataFrame(rows)


def summarize_score_columns_by_transfer_pair(layer_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_columns = {
        "condition_score_v4_two_axis_amplitude_minus_direction": "primary_amplitude_minus_direction_score",
        "v4_direction_axis_scaled_jvp_ratio": "direction_log_scaled_jvp_ratio",
        "v4_residual_amplitude_axis_scaled_jvp_fro": "fro_amplitude_score",
        "early_layer_prior": "early_layer_prior_score",
    }
    for (split_id, split_role, source_step, target_step), group in layer_diagnostics.groupby(
        ["split_id", "split_role", "source_warmup_steps", "target_warmup_steps"],
        observed=True,
        sort=True,
    ):
        for score_id, column in score_columns.items():
            spearman, spearman_low, spearman_high, points = corr_ci95(
                group[column],
                group["target_residual_log_observed"],
                method="spearman",
            )
            rows.append(
                {
                    "split_id": str(split_id),
                    "split_role": str(split_role),
                    "source_warmup_steps": int(source_step),
                    "target_warmup_steps": int(target_step),
                    "score_id": score_id,
                    "points": int(points),
                    "spearman_score_vs_target_residual": spearman,
                    "spearman_ci95_low": spearman_low,
                    "spearman_ci95_high": spearman_high,
                }
            )
    return pd.DataFrame(rows)


def summarize_transfer_pairs(pair_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_id, split_role, score_id), group in pair_scores.groupby(
        ["split_id", "split_role", "score_id"],
        observed=True,
        sort=False,
    ):
        mean, low, high = ci95(group["spearman_score_vs_target_residual"])
        rows.append(
            {
                "split_id": str(split_id),
                "split_role": str(split_role),
                "score_id": str(score_id),
                "transfer_pairs": int(len(group)),
                "mean_spearman_score_vs_target_residual": mean,
                "spearman_ci95_low": low,
                "spearman_ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def build_top5_stage_summary(layer_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_id, split_role, stage, block), group in layer_diagnostics.groupby(
        ["split_id", "split_role", "stage", "block_term"],
        observed=True,
        sort=True,
    ):
        rows.append(
            {
                "split_id": str(split_id),
                "split_role": str(split_role),
                "stage": str(stage),
                "block_term": str(block),
                "rows": int(len(group)),
                "target_residual_top5_count": int(group["target_residual_log_observed_top5"].sum()),
                "primary_top5_count": int(group["primary_amplitude_minus_direction_score_top5"].sum()),
                "direction_top5_count": int(group["direction_log_scaled_jvp_ratio_top5"].sum()),
                "fro_amplitude_top5_count": int(group["fro_amplitude_score_top5"].sum()),
                "mean_target_residual": float(group["target_residual_log_observed"].mean()),
                "mean_primary_score": float(group["primary_amplitude_minus_direction_score"].mean()),
                "mean_direction_score": float(group["direction_log_scaled_jvp_ratio"].mean()),
                "mean_fro_amplitude_score": float(group["fro_amplitude_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_failure_outcome_matrix() -> pd.DataFrame:
    final_summary = pd.read_csv(FINAL_SCORE_SUMMARY_PATH)
    gates = pd.read_csv(FINAL_GATE_REPORT_PATH)
    gate_status = gates.set_index("gate_id")["status"].to_dict()
    rows = []
    for row in final_summary.itertuples(index=False):
        low = float(row.spearman_ci95_low)
        high = float(row.spearman_ci95_high)
        if math.isfinite(low) and low > 0.0:
            residual_status = "passes_residual_gate"
        elif math.isfinite(high) and high < 0.0:
            residual_status = "inverted_residual_ranking"
        elif math.isfinite(float(row.mean_spearman_score_vs_target_residual)):
            residual_status = "residual_inconclusive"
        else:
            residual_status = "not_applicable"
        rows.append(
            {
                "split_id": row.split_id,
                "split_role": row.split_role,
                "score": row.score,
                "score_role": row.score_role,
                "mean_spearman_score_vs_target_residual": row.mean_spearman_score_vs_target_residual,
                "spearman_ci95_low": row.spearman_ci95_low,
                "spearman_ci95_high": row.spearman_ci95_high,
                "mean_threshold_below_one_accuracy": row.mean_threshold_below_one_accuracy,
                "residual_gate_status": residual_status,
            }
        )
    rows.append(
        {
            "split_id": "v4_final_gate",
            "split_role": "p0_predictive_condition",
            "score": "condition_score_v4_two_axis_amplitude_minus_direction",
            "score_role": "primary_candidate",
            "mean_spearman_score_vs_target_residual": math.nan,
            "spearman_ci95_low": math.nan,
            "spearman_ci95_high": math.nan,
            "mean_threshold_below_one_accuracy": math.nan,
            "residual_gate_status": gate_status.get("v4_p0_predictive_condition_claim", "not_ready"),
        }
    )
    return pd.DataFrame(rows)


def build_obstruction_summary(axis_pair_summary: pd.DataFrame) -> pd.DataFrame:
    lookup = axis_pair_summary.set_index(["split_role", "score_id"])
    arch_primary = lookup.loc[
        ("fresh_final_heldout_architecture", "condition_score_v4_two_axis_amplitude_minus_direction")
    ]
    data_primary = lookup.loc[
        ("fresh_final_heldout_data_partition", "condition_score_v4_two_axis_amplitude_minus_direction")
    ]
    data_direction = lookup.loc[
        ("fresh_final_heldout_data_partition", "v4_direction_axis_scaled_jvp_ratio")
    ]
    data_amplitude = lookup.loc[
        ("fresh_final_heldout_data_partition", "v4_residual_amplitude_axis_scaled_jvp_fro")
    ]
    data_early = lookup.loc[
        ("fresh_final_heldout_data_partition", "early_layer_prior")
    ]
    return pd.DataFrame(
        [
            {
                "obstruction_id": "V4-O1-architecture-transfer-pass",
                "scope": "WideResNet50-2 final architecture split",
                "evidence": (
                    f"primary residual Spearman {fmt(arch_primary['mean_spearman_score_vs_target_residual'])} "
                    f"CI=[{fmt(arch_primary['spearman_ci95_low'])}, {fmt(arch_primary['spearman_ci95_high'])}]"
                ),
                "mechanistic_read": "The amplitude-minus-direction score can transfer through a wider bottleneck architecture when the data partition is unchanged.",
                "claim_effect": "positive architecture-transfer boundary, not a full predictive condition",
            },
            {
                "obstruction_id": "V4-O2-data-partition-reversal",
                "scope": "CIFAR-10 mixed final data split",
                "evidence": (
                    f"primary residual Spearman {fmt(data_primary['mean_spearman_score_vs_target_residual'])} "
                    f"CI=[{fmt(data_primary['spearman_ci95_low'])}, {fmt(data_primary['spearman_ci95_high'])}]"
                ),
                "mechanistic_read": "The validation-frozen primary score reverses under the mixed CIFAR-10 class partition and blocks the v4 P0 claim.",
                "claim_effect": "negative data-partition boundary",
            },
            {
                "obstruction_id": "V4-O3-direction-is-not-the-failure",
                "scope": "CIFAR-10 mixed final data split",
                "evidence": (
                    f"direction-axis Spearman {fmt(data_direction['mean_spearman_score_vs_target_residual'])}; "
                    "direction threshold accuracy is 1 in the final evaluator"
                ),
                "mechanistic_read": "The below-one spectral/Frobenius direction guardrail and residual ordering of the ratio axis survive the failed data split.",
                "claim_effect": "failure localizes to residual-risk scalar aggregation rather than direction threshold",
            },
            {
                "obstruction_id": "V4-O4-amplitude-depth-confound",
                "scope": "CIFAR-10 mixed final data split",
                "evidence": (
                    f"Frobenius-amplitude Spearman {fmt(data_amplitude['mean_spearman_score_vs_target_residual'])}; "
                    f"early-layer prior Spearman {fmt(data_early['mean_spearman_score_vs_target_residual'])}"
                ),
                "mechanistic_read": "The amplitude and depth-related axes carry the wrong residual ordering under the mixed partition, so subtracting the direction term amplifies the reversal.",
                "claim_effect": "requires a new theory term or a narrower data-partition claim boundary",
            },
        ]
    )


def write_stage_figure(top5_stage_summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    failed = top5_stage_summary[
        top5_stage_summary["split_id"].eq("v4_final_data_cifar10lt_mixed_partition")
    ].copy()
    failed["stage_block"] = failed["stage"] + "/" + failed["block_term"]
    ordered = failed.sort_values("target_residual_top5_count", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.32 * len(ordered))))
    y = np.arange(len(ordered))
    ax.barh(y - 0.24, ordered["target_residual_top5_count"], height=0.22, label="target residual top-5")
    ax.barh(y, ordered["primary_top5_count"], height=0.22, label="primary top-5")
    ax.barh(y + 0.24, ordered["direction_top5_count"], height=0.22, label="direction top-5")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["stage_block"], fontsize=8)
    ax.set_xlabel("top-5 count across source-target transfer pairs")
    ax.set_title("V4 CIFAR-10 mixed final split: residual versus score hot spots")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "v4_cifar10_mixed_reversal_top5.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    outcome: pd.DataFrame,
    obstruction: pd.DataFrame,
    axis_pair_summary: pd.DataFrame,
    top5_stage_summary: pd.DataFrame,
    figure_path: Path,
) -> None:
    primary_rows = outcome[
        outcome["score"].eq("condition_score_v4_two_axis_amplitude_minus_direction")
        & outcome["split_id"].str.startswith("v4_final_")
    ]
    failed_top5 = (
        top5_stage_summary[
            top5_stage_summary["split_id"].eq("v4_final_data_cifar10lt_mixed_partition")
        ]
        .sort_values("target_residual_top5_count", ascending=False)
        .head(10)
    )
    text = f"""# E11 Condition-Score V4 Failure Mechanism Audit

This generated audit treats the v4 final evaluation as a negative boundary, not
as a tuning target. The validation-frozen primary score
`{selected_score_id()}` passes the WideResNet50-2 final architecture split but
fails the CIFAR-10 mixed final data partition. The audit asks which registered
score axis is responsible for that reversal.

![V4 CIFAR-10 mixed reversal top-5](../{figure_path.as_posix()})

## Final Outcome Matrix

{markdown_table(primary_rows, ["split_role", "score", "mean_spearman_score_vs_target_residual", "spearman_ci95_low", "spearman_ci95_high", "residual_gate_status"])}

## Obstruction Summary

{markdown_table(obstruction, ["obstruction_id", "scope", "evidence", "mechanistic_read", "claim_effect"])}

## Axis Transfer Summary

{markdown_table(axis_pair_summary, ["split_role", "score_id", "transfer_pairs", "mean_spearman_score_vs_target_residual", "spearman_ci95_low", "spearman_ci95_high"])}

## Failed Split Hot Spots

{markdown_table(failed_top5, ["stage", "block_term", "target_residual_top5_count", "primary_top5_count", "direction_top5_count", "fro_amplitude_top5_count"])}

## Boundary

The failure is not a direction-threshold failure: the direction axis keeps the
below-one gate and has positive residual ordering on the CIFAR-10 mixed final
split. The failure is the validation-frozen scalar aggregation. In this data
partition, the Frobenius-amplitude/depth side of the score carries the wrong
residual ordering strongly enough that amplitude-minus-direction reverses the
ranking. These final rows are now spent for score fitting.

Artifacts:
- [final_outcome_matrix.csv](../{(OUTPUT_DIR / 'final_outcome_matrix.csv').as_posix()})
- [axis_pair_summary.csv](../{(OUTPUT_DIR / 'axis_pair_summary.csv').as_posix()})
- [axis_transfer_pair_scores.csv](../{(OUTPUT_DIR / 'axis_transfer_pair_scores.csv').as_posix()})
- [top5_stage_summary.csv](../{(OUTPUT_DIR / 'top5_stage_summary.csv').as_posix()})
- [obstruction_summary.csv](../{(OUTPUT_DIR / 'obstruction_summary.csv').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_frame = load_axis_frame(SOURCE_LAYER_SUMMARY, SOURCE_METRICS)
    primary_score = selected_score_id()
    layer_diagnostics = build_layer_diagnostics(source_frame, primary_score)
    axis_all_layer_summary = summarize_score_columns(layer_diagnostics)
    axis_transfer_pair_scores = summarize_score_columns_by_transfer_pair(layer_diagnostics)
    axis_pair_summary = summarize_transfer_pairs(axis_transfer_pair_scores)
    top5_stage_summary = build_top5_stage_summary(layer_diagnostics)
    final_outcome = build_failure_outcome_matrix()
    obstruction = build_obstruction_summary(axis_pair_summary)
    figure_path = write_stage_figure(top5_stage_summary, FIGURE_DIR)

    final_outcome.to_csv(OUTPUT_DIR / "final_outcome_matrix.csv", index=False)
    axis_all_layer_summary.to_csv(OUTPUT_DIR / "axis_all_layer_summary.csv", index=False)
    axis_transfer_pair_scores.to_csv(OUTPUT_DIR / "axis_transfer_pair_scores.csv", index=False)
    axis_pair_summary.to_csv(OUTPUT_DIR / "axis_pair_summary.csv", index=False)
    top5_stage_summary.to_csv(OUTPUT_DIR / "top5_stage_summary.csv", index=False)
    obstruction.to_csv(OUTPUT_DIR / "obstruction_summary.csv", index=False)
    write_discussion(final_outcome, obstruction, axis_pair_summary, top5_stage_summary, figure_path)
    print(f"saved v4 failure mechanism audit to {OUTPUT_DIR}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(obstruction.to_string(index=False))


if __name__ == "__main__":
    main()
