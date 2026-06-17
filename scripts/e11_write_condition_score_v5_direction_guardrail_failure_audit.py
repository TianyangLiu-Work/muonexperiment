from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


FINAL_EVAL_DIR = Path("results/e11_condition_score_v5_protocol/final_score_evaluation")
REVIEWER_RESPONSE_DIR = Path("results/e11_condition_score_v5_protocol/reviewer_failure_response")
OUTPUT_DIR = Path("results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit")
DISCUSSION_PATH = Path("discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md")

PRIMARY_SCORE = "condition_score_v5_transport_normalized_amplitude_minus_direction"
DIRECTION_SCORE = "condition_score_v5_direction_axis_scaled_jvp_ratio"
EARLY_BASELINE = "early_layer_prior"
SOURCE_CONTROL = "source_observed_drift_positive_control"
ARCH_SPLIT = "v5_final_architecture_resnext50_32x4d_cifar100lt"
DATA_SPLIT = "v5_final_data_cifar10lt_cross_partition"
SPLITS = (
    {
        "split_key": "architecture",
        "split_id": ARCH_SPLIT,
        "split_role": "v5_final_heldout_architecture",
        "label": "ResNeXt50-32x4d CIFAR-100-LT final architecture split",
    },
    {
        "split_key": "data",
        "split_id": DATA_SPLIT,
        "split_role": "v5_final_heldout_data_partition",
        "label": "CIFAR-10-LT cross-partition final data split",
    },
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(FINAL_EVAL_DIR / "final_score_summary.csv")
    gates = pd.read_csv(FINAL_EVAL_DIR / "final_gate_report.csv")
    active_modes = pd.read_csv(REVIEWER_RESPONSE_DIR / "active_failure_modes.csv")
    return summary, gates, active_modes


def score_row(summary: pd.DataFrame, split_id: str, score: str) -> pd.Series:
    rows = summary[summary["split_id"].eq(split_id) & summary["score"].eq(score)]
    if len(rows) != 1:
        raise ValueError(f"expected one summary row for {split_id}/{score}, got {len(rows)}")
    return rows.iloc[0]


def gate_status(gates: pd.DataFrame, split_role: str, suffix: str) -> tuple[str, str]:
    rows = gates[
        gates["gate_id"].astype(str).str.startswith(split_role)
        & gates["gate_id"].astype(str).str.endswith(suffix)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one gate for {split_role}/{suffix}, got {len(rows)}")
    row = rows.iloc[0]
    return str(row["status"]), str(row["evidence"])


def build_score_axis_contrast(summary: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        split_key = split["split_key"]
        split_id = split["split_id"]
        split_role = split["split_role"]
        primary = score_row(summary, split_id, PRIMARY_SCORE)
        direction = score_row(summary, split_id, DIRECTION_SCORE)
        early = score_row(summary, split_id, EARLY_BASELINE)
        source = score_row(summary, split_id, SOURCE_CONTROL)
        residual_status, residual_evidence = gate_status(gates, split_role, "residual_spearman")
        direction_status, direction_evidence = gate_status(gates, split_role, "direction_threshold_accuracy")
        baseline_status, baseline_evidence = gate_status(gates, split_role, "baseline_dominance")
        controls_status, controls_evidence = gate_status(gates, split_role, "controls_reported")
        rows.extend(
            [
                {
                    "axis_id": f"{split_key}_primary_residual_ranking",
                    "split_id": split_id,
                    "split_role": split_role,
                    "split_display_name": split["label"],
                    "score": PRIMARY_SCORE,
                    "score_role": "primary_candidate",
                    "gate_status": residual_status,
                    "mean_spearman": primary["mean_spearman_score_vs_target_residual"],
                    "spearman_ci95_low": primary["spearman_ci95_low"],
                    "spearman_ci95_high": primary["spearman_ci95_high"],
                    "top5_overlap": primary["mean_top5_residual_risk_overlap_fraction"],
                    "threshold_accuracy": "",
                    "threshold_ci95_low": "",
                    "evidence": residual_evidence,
                },
                {
                    "axis_id": f"{split_key}_direction_threshold_guardrail",
                    "split_id": split_id,
                    "split_role": split_role,
                    "split_display_name": split["label"],
                    "score": DIRECTION_SCORE,
                    "score_role": "direction_guardrail",
                    "gate_status": direction_status,
                    "mean_spearman": direction["mean_spearman_score_vs_target_residual"],
                    "spearman_ci95_low": direction["spearman_ci95_low"],
                    "spearman_ci95_high": direction["spearman_ci95_high"],
                    "top5_overlap": direction["mean_top5_residual_risk_overlap_fraction"],
                    "threshold_accuracy": direction["mean_threshold_below_one_accuracy"],
                    "threshold_ci95_low": direction["threshold_below_one_accuracy_ci95_low"],
                    "evidence": direction_evidence,
                },
                {
                    "axis_id": f"{split_key}_early_layer_prior_baseline",
                    "split_id": split_id,
                    "split_role": split_role,
                    "split_display_name": split["label"],
                    "score": EARLY_BASELINE,
                    "score_role": "baseline",
                    "gate_status": baseline_status,
                    "mean_spearman": early["mean_spearman_score_vs_target_residual"],
                    "spearman_ci95_low": early["spearman_ci95_low"],
                    "spearman_ci95_high": early["spearman_ci95_high"],
                    "top5_overlap": early["mean_top5_residual_risk_overlap_fraction"],
                    "threshold_accuracy": "",
                    "threshold_ci95_low": "",
                    "evidence": baseline_evidence,
                },
                {
                    "axis_id": f"{split_key}_source_observed_positive_control",
                    "split_id": split_id,
                    "split_role": split_role,
                    "split_display_name": split["label"],
                    "score": SOURCE_CONTROL,
                    "score_role": "positive_control",
                    "gate_status": controls_status,
                    "mean_spearman": source["mean_spearman_score_vs_target_residual"],
                    "spearman_ci95_low": source["spearman_ci95_low"],
                    "spearman_ci95_high": source["spearman_ci95_high"],
                    "top5_overlap": source["mean_top5_residual_risk_overlap_fraction"],
                    "threshold_accuracy": source["mean_threshold_below_one_accuracy"],
                    "threshold_ci95_low": source["threshold_below_one_accuracy_ci95_low"],
                    "evidence": controls_evidence,
                },
            ]
        )
    return pd.DataFrame(rows)


def build_gate_boundary(gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gate in gates.itertuples(index=False):
        gate_id = str(gate.gate_id)
        status = str(gate.status)
        if gate_id == "v5_final_heldout_architecture_residual_spearman" and status == "pass":
            claim_effect = "architecture residual-ranking signal survives the ResNeXt50 final"
            blocks_p0 = "no_by_itself"
        elif gate_id == "v5_final_heldout_architecture_direction_threshold_accuracy" and status == "fail":
            claim_effect = "architecture direction-threshold guardrail fails and blocks the frozen-score P0 claim"
            blocks_p0 = "yes"
        elif gate_id == "v5_final_heldout_data_partition_residual_spearman" and status == "fail":
            claim_effect = "CIFAR-10 data-partition residual ranking reverses and independently blocks the P0 claim"
            blocks_p0 = "yes"
        elif gate_id == "v5_final_heldout_data_partition_direction_threshold_accuracy" and status == "pass":
            claim_effect = "data-partition direction threshold survives, so the data failure is residual-transport rather than direction-threshold failure"
            blocks_p0 = "no_by_itself"
        elif gate_id.endswith("baseline_dominance") and status == "pass":
            claim_effect = "failure is not explained by the early-layer nuisance baseline dominating the primary score"
            blocks_p0 = "no"
        elif gate_id.endswith("controls_reported") and status == "pass":
            claim_effect = "required controls are present for the generated split"
            blocks_p0 = "no"
        elif gate_id.endswith("generated") and status == "not_run":
            claim_effect = "remaining registered final split is still missing"
            blocks_p0 = "yes"
        elif gate_id == "v5_p0_predictive_condition_claim":
            claim_effect = "P0 remains not_ready under the frozen final gate family"
            blocks_p0 = "yes"
        else:
            claim_effect = "gate status is carried through without reinterpretation"
            blocks_p0 = "yes" if status in {"fail", "not_ready", "not_run"} else "no"
        rows.append(
            {
                "gate_id": gate_id,
                "status": status,
                "evidence": str(gate.evidence),
                "claim_effect": claim_effect,
                "blocks_p0": blocks_p0,
            }
        )
    return pd.DataFrame(rows)


def build_mechanistic_diagnosis(axis: pd.DataFrame, active_modes: pd.DataFrame) -> pd.DataFrame:
    arch_primary = axis[axis["axis_id"].eq("architecture_primary_residual_ranking")].iloc[0]
    arch_direction = axis[axis["axis_id"].eq("architecture_direction_threshold_guardrail")].iloc[0]
    arch_early = axis[axis["axis_id"].eq("architecture_early_layer_prior_baseline")].iloc[0]
    arch_source = axis[axis["axis_id"].eq("architecture_source_observed_positive_control")].iloc[0]
    data_primary = axis[axis["axis_id"].eq("data_primary_residual_ranking")].iloc[0]
    data_direction = axis[axis["axis_id"].eq("data_direction_threshold_guardrail")].iloc[0]
    data_early = axis[axis["axis_id"].eq("data_early_layer_prior_baseline")].iloc[0]
    data_source = axis[axis["axis_id"].eq("data_source_observed_positive_control")].iloc[0]
    active_mode_ids = set(active_modes["active_failure_mode_id"].astype(str))
    return pd.DataFrame(
        [
            {
                "diagnosis_id": "V5-DGF-1-architecture-residual-ranking-survives",
                "evidence": (
                    f"architecture primary residual Spearman {fmt(arch_primary['mean_spearman'])} "
                    f"CI=[{fmt(arch_primary['spearman_ci95_low'])}, {fmt(arch_primary['spearman_ci95_high'])}]"
                ),
                "mechanistic_read": "the transport-normalized scalar still ranks residual layer risk on the generated architecture final",
                "claim_effect": "support partial residual-ranking mechanism evidence only",
                "blocked_wording": "the generated split alone proves the v5 predictive condition",
                "active_failure_mode": "no",
            },
            {
                "diagnosis_id": "V5-DGF-2-architecture-direction-threshold-fails",
                "evidence": (
                    f"architecture direction threshold accuracy {fmt(arch_direction['threshold_accuracy'])} "
                    f"with CI low {fmt(arch_direction['threshold_ci95_low'])}"
                ),
                "mechanistic_read": "the below-one spectral-vs-Frobenius direction classifier is not sufficiently architecture-stable",
                "claim_effect": "blocks the frozen-score P0 claim under the registered final gates",
                "blocked_wording": "residual ranking is sufficient despite direction-threshold failure",
                "active_failure_mode": "yes" if "V5-RFR-4-direction-guardrail-failure" in active_mode_ids else "missing",
            },
            {
                "diagnosis_id": "V5-DGF-3-data-residual-ranking-reverses",
                "evidence": (
                    f"CIFAR-10 primary residual Spearman {fmt(data_primary['mean_spearman'])} "
                    f"CI=[{fmt(data_primary['spearman_ci95_low'])}, {fmt(data_primary['spearman_ci95_high'])}]"
                ),
                "mechanistic_read": "the same frozen scalar reverses under the CIFAR-10 class-partition transport test",
                "claim_effect": "independently blocks broad data-family predictive-condition wording",
                "blocked_wording": "the v5 score predicts unseen data-partition residual risk",
                "active_failure_mode": "yes" if "V5-RFR-2-data-transport-boundary" in active_mode_ids else "missing",
            },
            {
                "diagnosis_id": "V5-DGF-4-data-direction-threshold-survives",
                "evidence": (
                    f"CIFAR-10 direction threshold accuracy {fmt(data_direction['threshold_accuracy'])} "
                    f"with CI low {fmt(data_direction['threshold_ci95_low'])}"
                ),
                "mechanistic_read": "the data split keeps the below-one direction classifier while the residual ranker fails",
                "claim_effect": "separates residual transport failure from direction-threshold failure",
                "blocked_wording": "direction-threshold success rescues the failed residual-ranking gate",
                "active_failure_mode": "no",
            },
            {
                "diagnosis_id": "V5-DGF-5-failure-modes-are-orthogonal",
                "evidence": (
                    f"architecture residual={arch_primary['gate_status']}/direction={arch_direction['gate_status']}; "
                    f"data residual={data_primary['gate_status']}/direction={data_direction['gate_status']}"
                ),
                "mechanistic_read": "architecture transfer and data-partition transfer break different endpoints of the frozen score contract",
                "claim_effect": "requires endpoint-specific theory and cannot be repaired by reporting only one successful axis",
                "blocked_wording": "one passing endpoint establishes a broad predictive condition",
                "active_failure_mode": "yes" if "V5-RFR-current-p0-not-ready" in active_mode_ids else "missing",
            },
            {
                "diagnosis_id": "V5-DGF-6-controls-do-not-rescue-p0",
                "evidence": (
                    f"architecture early prior {fmt(arch_early['mean_spearman'])}, source control {fmt(arch_source['mean_spearman'])}; "
                    f"data early prior {fmt(data_early['mean_spearman'])}, source control {fmt(data_source['mean_spearman'])}"
                ),
                "mechanistic_read": "reported controls help localize the failures but do not turn either failed final gate into a pass",
                "claim_effect": "P0 remains not_ready with two registered final failures",
                "blocked_wording": "the v5 frozen score is an unseen-task predictive condition",
                "active_failure_mode": "no",
            },
        ]
    )


def build_next_protocol_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requirement_id": "V5-DGF-NP1-separate-endpoints",
                "requirement": "Keep residual ranking and below-one direction classification as separate registered endpoints.",
                "reason": "The architecture final passes residual ranking while failing the direction threshold, while the data final does the opposite.",
                "forbidden_shortcut": "collapse the failed direction gate into the residual-ranking result",
            },
            {
                "requirement_id": "V5-DGF-NP2-separate-transport-axes",
                "requirement": "Model architecture transport and data-partition transport as separate theory obligations.",
                "reason": "ResNeXt50 exposes direction-threshold failure; CIFAR-10 exposes residual-ranking reversal.",
                "forbidden_shortcut": "treat one split's pass as evidence for the other split's failed endpoint",
            },
            {
                "requirement_id": "V5-DGF-NP3-no-post-final-repair",
                "requirement": "Any sign, threshold, normalization, or feature repair needs a new validation/final protocol with unspent splits.",
                "reason": "Both current final splits are already observed under the frozen v5 protocol.",
                "forbidden_shortcut": "lower the 0.8 direction threshold or tune the score on these final splits",
            },
            {
                "requirement_id": "V5-DGF-NP4-direction-transport-term",
                "requirement": "Add a theory term or ablation explaining when the below-one direction threshold transports across architecture families.",
                "reason": "The scalar residual ranker can survive while the direction classifier does not.",
                "forbidden_shortcut": "claim a single transport-normalized scalar explains both phenomena",
            },
            {
                "requirement_id": "V5-DGF-NP5-preserve-negative-boundaries",
                "requirement": "Keep both the ResNeXt50 direction failure and CIFAR-10 residual reversal in the main ledger.",
                "reason": "Top-conference credibility depends on preserving registered negative outcomes.",
                "forbidden_shortcut": "drop either failed final gate because another endpoint later passes",
            },
        ]
    )


def write_discussion(
    axis: pd.DataFrame,
    gates: pd.DataFrame,
    diagnosis: pd.DataFrame,
    requirements: pd.DataFrame,
) -> None:
    arch_primary = axis[axis["axis_id"].eq("architecture_primary_residual_ranking")].iloc[0]
    arch_direction = axis[axis["axis_id"].eq("architecture_direction_threshold_guardrail")].iloc[0]
    data_primary = axis[axis["axis_id"].eq("data_primary_residual_ranking")].iloc[0]
    data_direction = axis[axis["axis_id"].eq("data_direction_threshold_guardrail")].iloc[0]
    text = f"""# E11 Condition-Score V5 Direction-Guardrail Failure Audit

This generated audit is a completed final boundary diagnosis for the v5
condition-score program. It reads only the frozen final evaluator summary and
gate report, plus the reviewer active-failure-mode table. It performs no score repair,
no threshold change, and no final-row tuning.

The completed final boundary says two things at once. On the ResNeXt50-32x4d
architecture final, residual ranking survives with primary Spearman
{fmt(arch_primary['mean_spearman'])} CI=[{fmt(arch_primary['spearman_ci95_low'])},
{fmt(arch_primary['spearman_ci95_high'])}], while the direction-threshold
guardrail fails with below-one threshold accuracy {fmt(arch_direction['threshold_accuracy'])}
and CI low {fmt(arch_direction['threshold_ci95_low'])}. On the CIFAR-10
cross-partition final, residual ranking reverses with primary Spearman
{fmt(data_primary['mean_spearman'])} CI=[{fmt(data_primary['spearman_ci95_low'])},
{fmt(data_primary['spearman_ci95_high'])}], while the direction threshold survives
with below-one threshold accuracy {fmt(data_direction['threshold_accuracy'])}.
Therefore P0 remains not_ready with orthogonal final failures. Any repair of this
failure needs a new unspent protocol.

## Score-Axis Contrast

{markdown_table(axis, ["axis_id", "split_role", "score_role", "gate_status", "mean_spearman", "spearman_ci95_low", "spearman_ci95_high", "top5_overlap", "threshold_accuracy", "threshold_ci95_low", "evidence"])}

## Gate Boundary Summary

{markdown_table(gates, ["gate_id", "status", "claim_effect", "blocks_p0", "evidence"])}

## Mechanistic Diagnosis

{markdown_table(diagnosis, ["diagnosis_id", "evidence", "mechanistic_read", "claim_effect", "blocked_wording", "active_failure_mode"])}

## Next Protocol Requirements

{markdown_table(requirements, ["requirement_id", "requirement", "reason", "forbidden_shortcut"])}

Artifacts:
- [score_axis_contrast.csv](../{(OUTPUT_DIR / 'score_axis_contrast.csv').as_posix()})
- [gate_boundary_summary.csv](../{(OUTPUT_DIR / 'gate_boundary_summary.csv').as_posix()})
- [mechanistic_diagnosis.csv](../{(OUTPUT_DIR / 'mechanistic_diagnosis.csv').as_posix()})
- [next_protocol_requirements.csv](../{(OUTPUT_DIR / 'next_protocol_requirements.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, gates, active_modes = load_inputs()
    axis = build_score_axis_contrast(summary, gates)
    gate_boundary = build_gate_boundary(gates)
    diagnosis = build_mechanistic_diagnosis(axis, active_modes)
    requirements = build_next_protocol_requirements()
    axis.to_csv(OUTPUT_DIR / "score_axis_contrast.csv", index=False)
    gate_boundary.to_csv(OUTPUT_DIR / "gate_boundary_summary.csv", index=False)
    diagnosis.to_csv(OUTPUT_DIR / "mechanistic_diagnosis.csv", index=False)
    requirements.to_csv(OUTPUT_DIR / "next_protocol_requirements.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "final_eval_dir": FINAL_EVAL_DIR.as_posix(),
                "reviewer_response_dir": REVIEWER_RESPONSE_DIR.as_posix(),
                "final_splits": [split["split_id"] for split in SPLITS],
                "primary_score": PRIMARY_SCORE,
                "current_p0_status": "not_ready",
                "analysis_scope": "completed final boundary audit; no final-row tuning or score repair",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(axis, gate_boundary, diagnosis, requirements)
    print(f"saved direction-guardrail failure audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
