from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_condition_score_ablation")
DISCUSSION_PATH = Path("discussion/e11_condition_score_ablation.md")


def ci(row: pd.Series) -> str:
    mean_key = "mean_spearman_score_vs_target_residual"
    if mean_key not in row.index:
        mean_key = "residual_spearman"
    return (
        f"{fmt(row[mean_key])} "
        f"[{fmt(row['spearman_ci95_low'])}, {fmt(row['spearman_ci95_high'])}]"
    )


def residual_status(row: pd.Series) -> str:
    low = float(row["spearman_ci95_low"])
    high = float(row["spearman_ci95_high"])
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "inconclusive"


def threshold_status(row: pd.Series) -> str:
    low = row.get("threshold_below_one_accuracy_ci95_low")
    high = row.get("threshold_below_one_accuracy_ci95_high")
    if pd.isna(low) or pd.isna(high):
        return "not_applicable"
    if float(low) >= 0.8:
        return "pass"
    if float(high) < 0.8:
        return "fail"
    return "inconclusive"


def one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        mask &= frame[key].eq(value)
    rows = frame[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {filters}, got {len(rows)}")
    return rows.iloc[0]


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "v2": pd.read_csv(
            "results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/"
            "heldout_score_summary.csv"
        ),
        "v3": pd.read_csv(
            "results/e11_condition_score_fresh_protocol/fresh_score_evaluation/"
            "fresh_score_summary.csv"
        ),
        "v4_axis": pd.read_csv(
            "results/e11_condition_score_v4_failure_mechanism_audit/axis_pair_summary.csv"
        ),
        "v5_validation": pd.read_csv(
            "results/e11_condition_score_v5_protocol/validation_score_freeze/"
            "validation_score_summary.csv"
        ),
        "v5_formulas": pd.read_csv(
            "results/e11_condition_score_v5_protocol/validation_score_freeze/"
            "score_formula_registry.csv"
        ),
        "v5_final": pd.read_csv(
            "results/e11_condition_score_v5_protocol/final_score_evaluation/"
            "final_score_summary.csv"
        ),
        "v5_final_gates": pd.read_csv(
            "results/e11_condition_score_v5_protocol/final_score_evaluation/"
            "final_gate_report.csv"
        ),
    }


def normalize_v2_v3(frame: pd.DataFrame, generation: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "generation": generation,
                "split_id": row["split_id"],
                "split_role": row["split_role"],
                "score_id": row["score"],
                "score_role": row["score_family"],
                "transfer_pairs": int(row["split_transfer_pairs"]),
                "residual_spearman": float(row["mean_spearman_score_vs_target_residual"]),
                "spearman_ci95_low": float(row["spearman_ci95_low"]),
                "spearman_ci95_high": float(row["spearman_ci95_high"]),
                "residual_status": residual_status(row),
                "threshold_accuracy": row.get("mean_threshold_below_one_accuracy"),
                "threshold_status": threshold_status(row),
                "leakage_status": "spent_final_row_diagnostic_only",
                "claim_use": "obstruction evidence only; not eligible for score fitting or selection",
            }
        )
    return pd.DataFrame(rows)


def normalize_v4(axis: pd.DataFrame) -> pd.DataFrame:
    role_lookup = {
        "condition_score_v4_two_axis_amplitude_minus_direction": "primary_candidate",
        "v4_direction_axis_scaled_jvp_ratio": "direction_axis",
        "v4_residual_amplitude_axis_scaled_jvp_fro": "raw_amplitude_axis",
        "early_layer_prior": "baseline",
    }
    rows: list[dict[str, object]] = []
    for _, row in axis.iterrows():
        rows.append(
            {
                "generation": "v4 frozen final axis audit",
                "split_id": row["split_id"],
                "split_role": row["split_role"],
                "score_id": row["score_id"],
                "score_role": role_lookup[str(row["score_id"])],
                "transfer_pairs": int(row["transfer_pairs"]),
                "residual_spearman": float(row["mean_spearman_score_vs_target_residual"]),
                "spearman_ci95_low": float(row["spearman_ci95_low"]),
                "spearman_ci95_high": float(row["spearman_ci95_high"]),
                "residual_status": residual_status(row),
                "threshold_accuracy": pd.NA,
                "threshold_status": "not_applicable",
                "leakage_status": "spent_final_row_diagnostic_only",
                "claim_use": "failure localization only; cannot tune v5 coefficients",
            }
        )
    return pd.DataFrame(rows)


def normalize_v5(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        score = str(row["score"])
        if score == "condition_score_v5_transport_normalized_amplitude_minus_direction":
            claim_use = "validation-frozen residual candidate; completed final P0 gates failed"
        elif score == "condition_score_v5_direction_axis_scaled_jvp_ratio":
            claim_use = "direction guardrail only; not a residual-risk scalar"
        elif score == "source_observed_drift_positive_control":
            claim_use = "positive control; not claim eligible"
        else:
            claim_use = "diagnostic or baseline axis"
        rows.append(
            {
                "generation": "v5 validation freeze",
                "split_id": "v5_validation_cifar100lt_mod4_partition",
                "split_role": "validation_only",
                "score_id": score,
                "score_role": row["score_role"],
                "transfer_pairs": int(row["validation_transfer_pairs"]),
                "residual_spearman": float(row["mean_spearman_score_vs_target_residual"]),
                "spearman_ci95_low": float(row["spearman_ci95_low"]),
                "spearman_ci95_high": float(row["spearman_ci95_high"]),
                "residual_status": residual_status(row),
                "threshold_accuracy": row.get("mean_threshold_below_one_accuracy"),
                "threshold_status": threshold_status(row),
                "leakage_status": "validation_only_no_final_rows",
                "claim_use": claim_use,
            }
        )
    return pd.DataFrame(rows)


def normalize_v5_final(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        score = str(row["score"])
        if score == "condition_score_v5_transport_normalized_amplitude_minus_direction":
            claim_use = "completed frozen final score; failed P0 gate family"
        elif score == "condition_score_v5_direction_axis_scaled_jvp_ratio":
            claim_use = "completed direction readout; one endpoint can pass while P0 remains blocked"
        elif score == "source_observed_drift_positive_control":
            claim_use = "completed positive control; not claim eligible"
        else:
            claim_use = "completed diagnostic or baseline axis"
        rows.append(
            {
                "generation": "v5 completed final evaluation",
                "split_id": row["split_id"],
                "split_role": row["split_role"],
                "score_id": score,
                "score_role": row["score_role"],
                "transfer_pairs": int(row["final_transfer_pairs"]),
                "residual_spearman": float(row["mean_spearman_score_vs_target_residual"]),
                "spearman_ci95_low": float(row["spearman_ci95_low"]),
                "spearman_ci95_high": float(row["spearman_ci95_high"]),
                "residual_status": residual_status(row),
                "threshold_accuracy": row.get("mean_threshold_below_one_accuracy"),
                "threshold_status": threshold_status(row),
                "leakage_status": "completed_final_row_diagnostic_only",
                "claim_use": claim_use,
            }
        )
    return pd.DataFrame(rows)


def build_score_ablation_summary(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [
        normalize_v2_v3(inputs["v2"], "v2 registered held-out"),
        normalize_v2_v3(inputs["v3"], "v3 fresh held-out"),
        normalize_v4(inputs["v4_axis"]),
        normalize_v5(inputs["v5_validation"]),
        normalize_v5_final(inputs["v5_final"]),
    ]
    return pd.concat([frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True)


def build_term_failure_ladder(summary: pd.DataFrame) -> pd.DataFrame:
    v4_data_primary = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_data_partition",
        score_id="condition_score_v4_two_axis_amplitude_minus_direction",
    )
    v4_data_direction = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_data_partition",
        score_id="v4_direction_axis_scaled_jvp_ratio",
    )
    v4_data_amplitude = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_data_partition",
        score_id="v4_residual_amplitude_axis_scaled_jvp_fro",
    )
    v4_data_early = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_data_partition",
        score_id="early_layer_prior",
    )
    v4_arch_primary = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_architecture",
        score_id="condition_score_v4_two_axis_amplitude_minus_direction",
    )
    v5_primary = one(
        summary,
        generation="v5 validation freeze",
        score_id="condition_score_v5_transport_normalized_amplitude_minus_direction",
    )
    v5_early = one(summary, generation="v5 validation freeze", score_id="early_layer_prior")
    v5_direction = one(
        summary,
        generation="v5 validation freeze",
        score_id="condition_score_v5_direction_axis_scaled_jvp_ratio",
    )
    v5_transport = one(
        summary,
        generation="v5 validation freeze",
        score_id="condition_score_v5_transport_defect_penalty",
    )
    v5_final_arch_direction = one(
        summary,
        generation="v5 completed final evaluation",
        split_role="v5_final_heldout_architecture",
        score_id="condition_score_v5_direction_axis_scaled_jvp_ratio",
    )
    v5_final_data_primary = one(
        summary,
        generation="v5 completed final evaluation",
        split_role="v5_final_heldout_data_partition",
        score_id="condition_score_v5_transport_normalized_amplitude_minus_direction",
    )
    return pd.DataFrame(
        [
            {
                "ladder_step": "L1-direction-guardrail-is-separate",
                "term_tested": "direction_ratio_guardrail",
                "evidence": f"v4 CIFAR-10 mixed direction axis has residual Spearman {ci(v4_data_direction)} while v5 direction threshold accuracy is {fmt(v5_direction['threshold_accuracy'])}",
                "decision": "keep as a below-one direction gate, not as a residual-risk ranking claim",
                "blocked_overclaim": "a passing direction threshold predicts which layer has largest residual drift",
            },
            {
                "ladder_step": "L2-raw-amplitude-needs-transport",
                "term_tested": "raw_residual_amplitude",
                "evidence": f"v4 CIFAR-10 mixed raw amplitude reverses at {ci(v4_data_amplitude)}",
                "decision": "raw amplitude remains diagnostic-only without transport normalization",
                "blocked_overclaim": "raw Frobenius tail-sensitivity amplitude transfers across partitions by itself",
            },
            {
                "ladder_step": "L3-depth-is-a-nuisance-baseline",
                "term_tested": "early_depth_nuisance",
                "evidence": f"v4 CIFAR-10 mixed early-layer prior reverses at {ci(v4_data_early)}; v5 primary {ci(v5_primary)} beats v5 early prior {ci(v5_early)} on validation",
                "decision": "report early_layer_prior as a baseline that the primary score must beat",
                "blocked_overclaim": "the score is meaningful if it only reproduces layer-depth structure",
            },
            {
                "ladder_step": "L4-v4-scalar-fails-data-transport",
                "term_tested": "amplitude_minus_direction_without_partition_transport",
                "evidence": f"v4 architecture split passes at {ci(v4_arch_primary)}, but v4 CIFAR-10 mixed data split fails at {ci(v4_data_primary)}",
                "decision": "treat v4 as an architecture-positive, data-partition-negative obstruction",
                "blocked_overclaim": "v4 establishes a broad natural-task predictive condition",
            },
            {
                "ladder_step": "L5-v5-candidate-is-frozen-not-proven",
                "term_tested": "transport_normalized_amplitude_minus_direction",
                "evidence": (
                    f"v5 validation primary residual Spearman is {ci(v5_primary)}; "
                    f"transport-defect axis alone is {ci(v5_transport)}; completed final failures are "
                    f"architecture direction {ci(v5_final_arch_direction)} and data residual {ci(v5_final_data_primary)}"
                ),
                "decision": "preserve the completed final failures as a negative boundary; any repaired score needs a new unspent protocol",
                "blocked_overclaim": "validation success alone proves unseen architecture/data residual-risk prediction",
            },
        ]
    )


def build_leakage_and_claim_boundary(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    formulas = inputs["v5_formulas"]
    if formulas["uses_spent_final_rows"].astype(str).str.contains("yes", case=False).any():
        raise AssertionError("v5 formulas unexpectedly use spent final rows")
    final_gate_status = inputs["v5_final_gates"].set_index("gate_id")["status"].astype(str).to_dict()
    p0_status = final_gate_status.get("v5_p0_predictive_condition_claim", "missing")
    arch_direction_status = final_gate_status.get(
        "v5_final_heldout_architecture_direction_threshold_accuracy", "missing"
    )
    data_residual_status = final_gate_status.get(
        "v5_final_heldout_data_partition_residual_spearman", "missing"
    )
    return pd.DataFrame(
        [
            {
                "boundary_id": "B1-spent-v2-v3-v4",
                "allowed_input": "v2/v3/v4 final summaries",
                "allowed_use": "diagnostic obstruction, failure localization, and reviewer caveat",
                "forbidden_use": "fitting, selecting, thresholding, or reweighting v5 scores",
                "claim_status": "diagnostic_only",
            },
            {
                "boundary_id": "B2-v5-validation",
                "allowed_input": "v5 validation-only mod-4 split",
                "allowed_use": "freeze or reject the registered v5 residual candidate",
                "forbidden_use": "claiming final held-out prediction before final outputs exist",
                "claim_status": "validation_frozen",
            },
            {
                "boundary_id": "B3-v5-finals",
                "allowed_input": "completed ResNeXt50-32x4d and CIFAR-10 cross-partition final outputs",
                "allowed_use": (
                    "one-shot frozen-score negative boundary: "
                    f"architecture_direction={arch_direction_status}; data_residual={data_residual_status}; P0={p0_status}"
                ),
                "forbidden_use": "changing coefficients, dropping a failed split, or lowering gates",
                "claim_status": "completed_final_failed_boundary",
            },
            {
                "boundary_id": "B4-paper-wording",
                "allowed_input": "score ablation plus v5 final evaluator",
                "allowed_use": "state why the paper treats v5 as a completed negative predictive-condition boundary",
                "forbidden_use": "turning a validation or direction guardrail into a benchmark or global theorem claim",
                "claim_status": "local_mechanism_only_after_final_failure",
            },
        ]
    )


def write_discussion(
    summary: pd.DataFrame,
    ladder: pd.DataFrame,
    leakage: pd.DataFrame,
) -> None:
    v4_data_direction = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_data_partition",
        score_id="v4_direction_axis_scaled_jvp_ratio",
    )
    v4_data_amplitude = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_data_partition",
        score_id="v4_residual_amplitude_axis_scaled_jvp_fro",
    )
    v4_data_primary = one(
        summary,
        generation="v4 frozen final axis audit",
        split_role="fresh_final_heldout_data_partition",
        score_id="condition_score_v4_two_axis_amplitude_minus_direction",
    )
    v5_primary = one(
        summary,
        generation="v5 validation freeze",
        score_id="condition_score_v5_transport_normalized_amplitude_minus_direction",
    )
    v5_early = one(summary, generation="v5 validation freeze", score_id="early_layer_prior")
    v5_final_arch_direction = one(
        summary,
        generation="v5 completed final evaluation",
        split_role="v5_final_heldout_architecture",
        score_id="condition_score_v5_direction_axis_scaled_jvp_ratio",
    )
    v5_final_data_primary = one(
        summary,
        generation="v5 completed final evaluation",
        split_role="v5_final_heldout_data_partition",
        score_id="condition_score_v5_transport_normalized_amplitude_minus_direction",
    )
    text = f"""# E11 Condition-Score Ablation Audit

This generated audit is a CPU-only theory-to-score ablation over already
committed condition-score evidence. It adds no new GPU result and makes no P0
predictive-condition claim. Its purpose is to separate three objects reviewers
can otherwise conflate: the below-one direction guardrail, residual layer-risk ranking,
and partition/architecture transport.

The key spent-evidence lesson is that the v4 CIFAR-10 mixed split keeps the
direction axis positive at {ci(v4_data_direction)}, while raw amplitude
reverses at {ci(v4_data_amplitude)} and the v4 amplitude-minus-direction scalar
fails at {ci(v4_data_primary)}. Thus direction success is not enough for
residual-risk prediction. The v5 validation split freezes the transport-normalized
candidate at {ci(v5_primary)}, above the early-layer baseline at {ci(v5_early)},
but the completed final architecture and data splits failed the registered P0
gate family: the ResNeXt50 direction-threshold axis fails at
{ci(v5_final_arch_direction)}, and the CIFAR-10 cross-partition residual score
reverses at {ci(v5_final_data_primary)}. Any repaired predictive-condition
wording therefore needs a new unspent protocol.

## Score-Axis Summary

{markdown_table(summary, ["generation", "split_role", "score_id", "score_role", "residual_spearman", "spearman_ci95_low", "spearman_ci95_high", "residual_status", "threshold_accuracy", "threshold_status", "claim_use"])}

## Term Failure Ladder

{markdown_table(ladder, ["ladder_step", "term_tested", "evidence", "decision", "blocked_overclaim"])}

## Leakage and Claim Boundary

{markdown_table(leakage, ["boundary_id", "allowed_input", "allowed_use", "forbidden_use", "claim_status"])}

## Boundary

Allowed now: cite this ablation as spent-evidence failure localization and as
the reason the completed v5 final evaluator must report direction, residual
ranking, baseline dominance, and transport status separately.

Blocked now: using any v2/v3/v4 final row to fit or reselect the v5 score;
claiming that the v5 validation result proves unseen natural-task residual-risk
prediction; merging the direction threshold into a final-accuracy or optimizer
benchmark claim.

Artifacts:
- [score_ablation_summary.csv](../results/e11_condition_score_ablation/score_ablation_summary.csv)
- [term_failure_ladder.csv](../results/e11_condition_score_ablation/term_failure_ladder.csv)
- [leakage_and_claim_boundary.csv](../results/e11_condition_score_ablation/leakage_and_claim_boundary.csv)
- [config.json](../results/e11_condition_score_ablation/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    inputs = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_score_ablation_summary(inputs)
    ladder = build_term_failure_ladder(summary)
    leakage = build_leakage_and_claim_boundary(inputs)
    summary.to_csv(OUTPUT_DIR / "score_ablation_summary.csv", index=False)
    ladder.to_csv(OUTPUT_DIR / "term_failure_ladder.csv", index=False)
    leakage.to_csv(OUTPUT_DIR / "leakage_and_claim_boundary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "artifact": "condition_score_ablation",
                "uses_new_gpu_results": False,
                "uses_spent_final_rows_for_tuning": False,
                "claim_status": "completed_final_failed_boundary",
                "discussion": DISCUSSION_PATH.as_posix(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(summary, ladder, leakage)
    print(f"saved condition-score ablation audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(ladder.to_string(index=False))


if __name__ == "__main__":
    main()
