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
from e11_condition_geometry.statistics import corr_ci95
from scripts.e11_write_cifar100_resnet_condition_score_next import add_protocol_features
from scripts.e11_write_cifar100_resnet_condition_score_next import fit_depth_baseline
from scripts.e11_write_cifar100_resnet_condition_score_next import target_residual_from_source_depth


HELDOUT_SUMMARY_PATH = Path(
    "results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_score_summary.csv"
)
FRESH_SUMMARY_PATH = Path(
    "results/e11_condition_score_fresh_protocol/fresh_score_evaluation/fresh_score_summary.csv"
)
SOURCE_LAYER_SUMMARY_PATH = Path("results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv")
FRESH_RESNET50_LAYER_SUMMARY_PATH = Path(
    "results/e11_condition_score_fresh_protocol/fresh_architecture_resnet50/layer_summary.csv"
)
OUTPUT_DIR = Path("results/e11_condition_score_failure_mechanism_audit")
FIGURE_DIR = Path("figures/e11_condition_score_failure_mechanism_audit")
DISCUSSION_PATH = Path("discussion/e11_condition_score_failure_mechanism_audit.md")


def classify_residual(row: pd.Series) -> str:
    low = float(row["spearman_ci95_low"])
    high = float(row["spearman_ci95_high"])
    mean = float(row["mean_spearman_score_vs_target_residual"])
    if math.isfinite(low) and low > 0.0:
        return "passes_residual_gate"
    if math.isfinite(high) and high < 0.0:
        return "inverted_residual_ranking"
    if math.isfinite(mean):
        return "residual_inconclusive"
    return "not_applicable"


def classify_threshold(row: pd.Series) -> str:
    value = row.get("mean_threshold_below_one_accuracy", math.nan)
    if pd.isna(value):
        return "not_reported"
    return "passes_threshold_gate" if float(value) >= 0.8 else "fails_threshold_gate"


def normalized_score_name(score: str) -> str:
    if score == "condition_score_v2_calibrated_residual_spent_baseline":
        return "condition_score_v2_calibrated_residual"
    if score == "condition_score_v3_zero_fit_scaled_jvp":
        return "condition_score_v3_zero_fit_scaled_jvp"
    return score


def score_generation(score: str) -> str:
    if score == "condition_score_v3_zero_fit_scaled_jvp":
        return "fresh_v3_zero_fit"
    if score in {"condition_score_v2_calibrated_residual", "condition_score_v2_calibrated_residual_spent_baseline"}:
        return "spent_v2_calibrated"
    if score == "legacy_scaled_jvp_ratio":
        return "legacy_scaled_jvp"
    if score == "early_layer_prior":
        return "depth_baseline"
    if score == "source_observed_drift_positive_control":
        return "positive_control"
    return "other"


def build_score_outcome_matrix() -> pd.DataFrame:
    heldout = pd.read_csv(HELDOUT_SUMMARY_PATH).copy()
    heldout["evaluation_family"] = "registered_v2_heldout"
    heldout["split_display_name"] = heldout["split_role"].map(
        {
            "primary_heldout_architecture": "Registered ResNet34 CIFAR-100-LT architecture split",
            "primary_heldout_data": "Registered CIFAR-10-LT data-family split",
        }
    )
    fresh = pd.read_csv(FRESH_SUMMARY_PATH).copy()
    fresh["evaluation_family"] = "fresh_v3_protocol"
    combined = pd.concat([heldout, fresh], ignore_index=True, sort=False)
    combined["score_normalized"] = combined["score"].astype(str).map(normalized_score_name)
    combined["score_generation"] = combined["score"].astype(str).map(score_generation)
    combined["residual_gate_status"] = combined.apply(classify_residual, axis=1)
    combined["threshold_gate_status"] = combined.apply(classify_threshold, axis=1)
    combined["threshold_only_success"] = (
        combined["residual_gate_status"].ne("passes_residual_gate")
        & combined["threshold_gate_status"].eq("passes_threshold_gate")
    )
    columns = [
        "evaluation_family",
        "split_role",
        "split_display_name",
        "score",
        "score_normalized",
        "score_generation",
        "split_transfer_pairs",
        "mean_points",
        "mean_spearman_score_vs_target_residual",
        "spearman_ci95_low",
        "spearman_ci95_high",
        "mean_top5_residual_risk_overlap_fraction",
        "mean_threshold_below_one_accuracy",
        "residual_gate_status",
        "threshold_gate_status",
        "threshold_only_success",
    ]
    return combined[[column for column in columns if column in combined.columns]].sort_values(
        ["evaluation_family", "split_role", "score_generation", "score"]
    )


def _one_score(score_matrix: pd.DataFrame, split_role: str, score: str) -> pd.Series:
    match = score_matrix[
        score_matrix["split_role"].eq(split_role)
        & score_matrix["score"].eq(score)
    ]
    if len(match) != 1:
        raise ValueError(f"expected one score row for split_role={split_role}, score={score}; found {len(match)}")
    return match.iloc[0]


def build_obstruction_taxonomy(score_matrix: pd.DataFrame) -> pd.DataFrame:
    v2_arch = _one_score(score_matrix, "primary_heldout_architecture", "condition_score_v2_calibrated_residual")
    v2_arch_source = _one_score(score_matrix, "primary_heldout_architecture", "source_observed_drift_positive_control")
    v2_data = _one_score(score_matrix, "primary_heldout_data", "condition_score_v2_calibrated_residual")
    v2_data_legacy = _one_score(score_matrix, "primary_heldout_data", "legacy_scaled_jvp_ratio")
    fresh_arch_v3 = _one_score(
        score_matrix,
        "fresh_final_heldout_architecture",
        "condition_score_v3_zero_fit_scaled_jvp",
    )
    fresh_arch_v2 = _one_score(
        score_matrix,
        "fresh_final_heldout_architecture",
        "condition_score_v2_calibrated_residual_spent_baseline",
    )
    fresh_data_v3 = _one_score(
        score_matrix,
        "fresh_final_heldout_data_partition",
        "condition_score_v3_zero_fit_scaled_jvp",
    )
    fresh_data_v2 = _one_score(
        score_matrix,
        "fresh_final_heldout_data_partition",
        "condition_score_v2_calibrated_residual_spent_baseline",
    )
    return pd.DataFrame(
        [
            {
                "obstruction_id": "O1-v2-architecture-weak-transfer",
                "scope": "registered ResNet34 architecture split",
                "evidence": (
                    f"v2 residual Spearman {fmt(v2_arch['mean_spearman_score_vs_target_residual'])} "
                    f"CI=[{fmt(v2_arch['spearman_ci95_low'])}, {fmt(v2_arch['spearman_ci95_high'])}], "
                    f"while source-observed control is {fmt(v2_arch_source['mean_spearman_score_vs_target_residual'])}"
                ),
                "mechanistic_read": "The residual target is partly transferable across architecture, but the calibrated v2 feature map is not stable enough.",
                "claim_effect": "blocks a scalar v2 condition-score claim",
                "allowed_use": "negative architecture-transfer evidence only",
            },
            {
                "obstruction_id": "O2-v2-data-family-sign-reversal",
                "scope": "registered CIFAR-10-LT data-family split",
                "evidence": (
                    f"v2 residual Spearman {fmt(v2_data['mean_spearman_score_vs_target_residual'])}, "
                    f"legacy scaled-JVP residual Spearman {fmt(v2_data_legacy['mean_spearman_score_vs_target_residual'])}"
                ),
                "mechanistic_read": "The calibrated residual score learned a data-family-dependent sign/feature mix that the older JVP term did not share.",
                "claim_effect": "blocks calibrated residual transfer across data families",
                "allowed_use": "negative data-family evidence only",
            },
            {
                "obstruction_id": "O3-v3-resnet50-jvp-reversal",
                "scope": "fresh ResNet50 architecture split",
                "evidence": (
                    f"v3 zero-fit scaled-JVP residual Spearman {fmt(fresh_arch_v3['mean_spearman_score_vs_target_residual'])}, "
                    f"retired v2 baseline {fmt(fresh_arch_v2['mean_spearman_score_vs_target_residual'])}"
                ),
                "mechanistic_read": "The pure finite-difference scaled-JVP score preserves the below-one direction but reverses residual layer-risk ranking in bottleneck ResNet50.",
                "claim_effect": "blocks the fresh v3 P0 predictive-condition claim",
                "allowed_use": "spent obstruction for deriving, not tuning, the next score",
            },
            {
                "obstruction_id": "O4-v3-data-pass-is-not-universal",
                "scope": "fresh CIFAR-10 alternate partition",
                "evidence": (
                    f"v3 residual Spearman {fmt(fresh_data_v3['mean_spearman_score_vs_target_residual'])}, "
                    f"retired v2 baseline {fmt(fresh_data_v2['mean_spearman_score_vs_target_residual'])}"
                ),
                "mechanistic_read": "The finite-difference score can work on a data-partition shift while failing on architecture depth/parameterization.",
                "claim_effect": "supports a targeted data-partition result, not a general predictive condition",
                "allowed_use": "positive diagnostic boundary only",
            },
        ]
    )


def parameter_stage(parameter: str) -> str:
    if parameter == "conv1.weight":
        return "stem"
    if parameter.startswith("layer1."):
        return "layer1"
    if parameter.startswith("layer2."):
        return "layer2"
    if parameter.startswith("layer3."):
        return "layer3"
    if parameter.startswith("layer4."):
        return "layer4"
    if parameter == "fc.weight":
        return "classifier"
    return "other"


def block_term(parameter: str) -> str:
    if "downsample" in parameter:
        return "downsample"
    if parameter == "conv1.weight":
        return "stem_conv"
    if parameter == "fc.weight":
        return "classifier"
    return parameter.rsplit(".", 2)[-2]


def build_resnet50_stage_reversal() -> pd.DataFrame:
    source = add_protocol_features(pd.read_csv(SOURCE_LAYER_SUMMARY_PATH))
    target = add_protocol_features(pd.read_csv(FRESH_RESNET50_LAYER_SUMMARY_PATH))
    rows: list[dict[str, object]] = []
    for source_step in sorted(int(value) for value in source["warmup_steps"].unique()):
        source_step_frame = source[source["warmup_steps"].eq(source_step)].copy()
        if source_step_frame.empty:
            continue
        depth_intercept, depth_slope, _ = fit_depth_baseline(source_step_frame)
        for target_step in sorted(int(value) for value in target["warmup_steps"].unique()):
            target_step_frame = target[target["warmup_steps"].eq(target_step)].copy()
            target_residual = target_residual_from_source_depth(
                target_step_frame,
                float(depth_intercept),
                float(depth_slope),
            )
            diagnostic = pd.DataFrame(
                {
                    "source_warmup_steps": source_step,
                    "target_warmup_steps": target_step,
                    "parameter": target_step_frame["parameter"].astype(str),
                    "layer_index": target_step_frame["layer_index"].astype(int),
                    "stage": target_step_frame["parameter"].astype(str).map(parameter_stage),
                    "block_term": target_step_frame["parameter"].astype(str).map(block_term),
                    "primary_log_scaled_jvp": target_step_frame["log_scaled_jvp_ratio"].astype(float),
                    "target_residual_log_observed": target_residual.astype(float).to_numpy(),
                    "observed_log_drift": target_step_frame["log_observed"].astype(float),
                    "observed_below_one": target_step_frame[
                        "geomean_observed_tail_drift_sq_ratio_spectral_over_fro"
                    ].astype(float)
                    < 1.0,
                }
            )
            target_top = set(
                diagnostic.nlargest(min(5, len(diagnostic)), "target_residual_log_observed")["parameter"]
            )
            score_top = set(diagnostic.nlargest(min(5, len(diagnostic)), "primary_log_scaled_jvp")["parameter"])
            for (stage, term), group in diagnostic.groupby(["stage", "block_term"], sort=False, observed=True):
                spearman, low, high, points = corr_ci95(
                    group["primary_log_scaled_jvp"],
                    group["target_residual_log_observed"],
                    method="spearman",
                )
                rows.append(
                    {
                        "source_warmup_steps": source_step,
                        "target_warmup_steps": target_step,
                        "stage": stage,
                        "block_term": term,
                        "layers": int(len(group)),
                        "mean_primary_log_scaled_jvp": float(group["primary_log_scaled_jvp"].mean()),
                        "mean_target_residual_log_observed": float(
                            group["target_residual_log_observed"].mean()
                        ),
                        "spearman_primary_vs_target_residual": spearman,
                        "spearman_ci95_low": low,
                        "spearman_ci95_high": high,
                        "points": points,
                        "top5_target_residual_layers": int(group["parameter"].isin(target_top).sum()),
                        "top5_primary_score_layers": int(group["parameter"].isin(score_top).sum()),
                        "observed_below_one_fraction": float(group["observed_below_one"].mean()),
                    }
                )
    stage_rows = pd.DataFrame(rows)
    if stage_rows.empty:
        return stage_rows
    summary_rows = []
    for (stage, term), group in stage_rows.groupby(["stage", "block_term"], sort=False, observed=True):
        summary_rows.append(
            {
                "stage": stage,
                "block_term": term,
                "transfer_pairs": int(len(group)),
                "layers": int(group["layers"].median()),
                "mean_primary_log_scaled_jvp": float(group["mean_primary_log_scaled_jvp"].mean()),
                "mean_target_residual_log_observed": float(
                    group["mean_target_residual_log_observed"].mean()
                ),
                "mean_spearman_primary_vs_target_residual": float(
                    group["spearman_primary_vs_target_residual"].mean()
                ),
                "total_top5_target_residual_layers": int(group["top5_target_residual_layers"].sum()),
                "total_top5_primary_score_layers": int(group["top5_primary_score_layers"].sum()),
                "mean_observed_below_one_fraction": float(group["observed_below_one_fraction"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    stage_order = {"stem": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4, "classifier": 5}
    summary["_stage_order"] = summary["stage"].map(stage_order).fillna(99)
    return summary.sort_values(["_stage_order", "block_term"]).drop(columns=["_stage_order"])


def write_stage_figure(stage_reversal: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_frame = stage_reversal.copy()
    plot_frame["label"] = plot_frame["stage"].astype(str) + "/" + plot_frame["block_term"].astype(str)
    plot_frame = plot_frame.sort_values("mean_target_residual_log_observed")
    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.32 * len(plot_frame))))
    y = np.arange(len(plot_frame))
    ax.barh(y - 0.18, plot_frame["mean_target_residual_log_observed"], height=0.35, label="target residual")
    ax.barh(y + 0.18, plot_frame["mean_primary_log_scaled_jvp"], height=0.35, label="primary scaled-JVP")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["label"], fontsize=8)
    ax.set_xlabel("mean log score or residual")
    ax.set_title("Fresh ResNet50 stage-level scaled-JVP reversal")
    ax.legend(loc="best")
    fig.tight_layout()
    figure_path = FIGURE_DIR / "resnet50_stage_reversal.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_matrix = build_score_outcome_matrix()
    taxonomy = build_obstruction_taxonomy(score_matrix)
    stage_reversal = build_resnet50_stage_reversal()
    figure_path = write_stage_figure(stage_reversal)

    score_matrix.to_csv(OUTPUT_DIR / "score_outcome_matrix.csv", index=False)
    taxonomy.to_csv(OUTPUT_DIR / "split_obstruction_taxonomy.csv", index=False)
    stage_reversal.to_csv(OUTPUT_DIR / "resnet50_stage_reversal.csv", index=False)

    fresh_arch_primary = _one_score(
        score_matrix,
        "fresh_final_heldout_architecture",
        "condition_score_v3_zero_fit_scaled_jvp",
    )
    fresh_data_primary = _one_score(
        score_matrix,
        "fresh_final_heldout_data_partition",
        "condition_score_v3_zero_fit_scaled_jvp",
    )
    stage_excerpt = stage_reversal.sort_values(
        "mean_target_residual_log_observed",
        ascending=False,
    ).head(10)
    text = f"""# E11 Condition-Score Failure Mechanism Audit

This generated audit is diagnostic-only. It uses the spent registered v2 held-outs and the spent fresh v3 final splits to characterize failure modes, not to fit or select another score. The next predictive-condition attempt still needs a new frozen score and new unspent final splits.

![ResNet50 stage reversal](../{figure_path.as_posix()})

## Score Outcome Matrix

{markdown_table(score_matrix, ["evaluation_family", "split_role", "score", "score_generation", "mean_spearman_score_vs_target_residual", "spearman_ci95_low", "spearman_ci95_high", "mean_threshold_below_one_accuracy", "residual_gate_status", "threshold_only_success"])}

## Obstruction Taxonomy

{markdown_table(taxonomy, ["obstruction_id", "scope", "evidence", "mechanistic_read", "claim_effect", "allowed_use"])}

## ResNet50 Stage Reversal

Fresh ResNet50 is the current decisive failure: the primary v3 residual Spearman is {fmt(fresh_arch_primary['mean_spearman_score_vs_target_residual'])} with CI [{fmt(fresh_arch_primary['spearman_ci95_low'])}, {fmt(fresh_arch_primary['spearman_ci95_high'])}], while the fresh CIFAR-10 alternate partition is {fmt(fresh_data_primary['mean_spearman_score_vs_target_residual'])} with CI [{fmt(fresh_data_primary['spearman_ci95_low'])}, {fmt(fresh_data_primary['spearman_ci95_high'])}]. The issue is therefore not a universal lack of JVP signal; it is an architecture/parameterization reversal.

{markdown_table(stage_excerpt, ["stage", "block_term", "transfer_pairs", "layers", "mean_primary_log_scaled_jvp", "mean_target_residual_log_observed", "mean_spearman_primary_vs_target_residual", "total_top5_target_residual_layers", "total_top5_primary_score_layers"])}

## Theory Consequence

The theorem-facing score cannot be only a scalar finite-difference magnitude. A usable next condition must explain why the same scaled-JVP direction ranks residual risk correctly on the CIFAR-10 partition but reverses on ResNet50 bottleneck architecture. The minimum viable mathematical object should separate direction-threshold control from residual-ranking control, and it likely needs an architecture-normalized transport term for bottleneck/downsample parameterization rather than another unconstrained regression on the failed final splits.

## Claim Boundary

Allowed: cite these artifacts as an obstruction audit and as motivation for a new theorem-linked score.

Blocked: tuning a new score on these rows and presenting the same ResNet34, CIFAR-10, ResNet50, or CIFAR-10 alternate splits as clean final evidence.

Artifacts:
- [score_outcome_matrix.csv](../{(OUTPUT_DIR / 'score_outcome_matrix.csv').as_posix()})
- [split_obstruction_taxonomy.csv](../{(OUTPUT_DIR / 'split_obstruction_taxonomy.csv').as_posix()})
- [resnet50_stage_reversal.csv](../{(OUTPUT_DIR / 'resnet50_stage_reversal.csv').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)
    print(f"saved condition-score failure mechanism audit to {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
