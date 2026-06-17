from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


HELDOUT_SUMMARY_PATH = Path(
    "results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_score_summary.csv"
)
HELDOUT_GATES_PATH = Path(
    "results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_gate_report.csv"
)
RETROSPECTIVE_SUMMARY_PATH = Path("results/e11_cifar100_resnet_condition_score_next/score_summary.csv")
OUTPUT_DIR = Path("results/e11_condition_score_theory_bridge")
TARGET_REGISTER_PATH = OUTPUT_DIR / "score_target_register.csv"
PROTOCOL_REQUIREMENTS_PATH = OUTPUT_DIR / "fresh_protocol_requirements.csv"
OUTPUT_PATH = Path("discussion/e11_condition_score_theory_bridge.md")


def _row(summary: pd.DataFrame, split_role: str, score: str) -> pd.Series:
    match = summary[summary["split_role"].eq(split_role) & summary["score"].eq(score)]
    if len(match) != 1:
        raise ValueError(f"expected one row for split_role={split_role}, score={score}; found {len(match)}")
    return match.iloc[0]


def _score_line(row: pd.Series) -> str:
    return (
        f"Spearman={fmt(row['mean_spearman_score_vs_target_residual'])} "
        f"CI=[{fmt(row['spearman_ci95_low'])}, {fmt(row['spearman_ci95_high'])}], "
        f"threshold={fmt(row['mean_threshold_below_one_accuracy'])}"
    )


def build_target_register(heldout: pd.DataFrame, retrospective: pd.DataFrame) -> pd.DataFrame:
    arch_primary = _row(heldout, "primary_heldout_architecture", "condition_score_v2_calibrated_residual")
    data_primary = _row(heldout, "primary_heldout_data", "condition_score_v2_calibrated_residual")
    arch_source = _row(heldout, "primary_heldout_architecture", "source_observed_drift_positive_control")
    data_source = _row(heldout, "primary_heldout_data", "source_observed_drift_positive_control")
    data_legacy = _row(heldout, "primary_heldout_data", "legacy_scaled_jvp_ratio")
    retrospective_primary = retrospective.set_index("score").loc["condition_score_v2_calibrated_residual"]
    return pd.DataFrame(
        [
            {
                "target_id": "direction_threshold",
                "theory_object": "matched-head-gain below-one drift direction",
                "measured_target": "whether observed spectral/Frobenius tail-drift ratio is below one",
                "current_evidence": (
                    f"held-out threshold accuracy is {fmt(arch_primary['mean_threshold_below_one_accuracy'])} "
                    f"on ResNet34/CIFAR-100-LT and {fmt(data_primary['mean_threshold_below_one_accuracy'])} "
                    "on ResNet18/CIFAR-10-LT"
                ),
                "status": "supported_guardrail",
                "allowed_claim": "The current score preserves a held-out direction threshold in these splits.",
                "blocked_claim": "This threshold result does not rank which layers carry the largest residual risk.",
            },
            {
                "target_id": "residual_layer_ranking",
                "theory_object": "downstream-aware residual layer-risk ordering",
                "measured_target": "Spearman correlation with source-depth-adjusted observed log-drift residuals",
                "current_evidence": (
                    f"retrospective primary {_score_line(retrospective_primary)}; "
                    f"held-out architecture primary {_score_line(arch_primary)}; "
                    f"held-out data primary {_score_line(data_primary)}"
                ),
                "status": "blocked_by_heldout_failure",
                "allowed_claim": "The failed held-outs are evidence that v2 is not invariant across architecture/data families.",
                "blocked_claim": "The current v2 score predicts held-out residual layer-risk ranking.",
            },
            {
                "target_id": "source_observed_transfer_control",
                "theory_object": "upper-bound positive control for layer-risk transfer",
                "measured_target": "source observed residuals matched by parameter name across target split",
                "current_evidence": (
                    f"source observed control Spearman={fmt(arch_source['mean_spearman_score_vs_target_residual'])} "
                    f"CI=[{fmt(arch_source['spearman_ci95_low'])}, {fmt(arch_source['spearman_ci95_high'])}] "
                    "on architecture split, but "
                    f"{fmt(data_source['mean_spearman_score_vs_target_residual'])} "
                    f"CI=[{fmt(data_source['spearman_ci95_low'])}, {fmt(data_source['spearman_ci95_high'])}] "
                    "on data split"
                ),
                "status": "architecture_transfer_only",
                "allowed_claim": "Architecture and data transfer failures are distinct obstructions.",
                "blocked_claim": "A single source-checkpoint residual structure transfers across all held-out families.",
            },
            {
                "target_id": "legacy_jvp_counterexample",
                "theory_object": "finite-difference downstream JVP ratio",
                "measured_target": "legacy scaled-JVP residual ranking on the CIFAR-10-LT data split",
                "current_evidence": (
                    f"legacy scaled-JVP data residual Spearman={fmt(data_legacy['mean_spearman_score_vs_target_residual'])} "
                    f"CI=[{fmt(data_legacy['spearman_ci95_low'])}, {fmt(data_legacy['spearman_ci95_high'])}], "
                    f"top5 overlap={fmt(data_legacy['mean_top5_residual_risk_overlap_fraction'])}"
                ),
                "status": "v2_not_uniformly_better",
                "allowed_claim": "Any revised score must report when older JVP terms dominate the calibrated v2 score.",
                "blocked_claim": "The calibrated v2 score uniformly improves on the legacy scaled-JVP ratio.",
            },
        ]
    )


def build_protocol_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requirement_id": "R1-target-separation",
                "requirement": "Declare threshold-direction and residual-ranking as separate statistical targets.",
                "why_required": "The current held-outs pass threshold direction while failing residual ranking.",
                "acceptance_evidence": "A future report has separate rows, gates, and claims for threshold accuracy and residual Spearman.",
                "current_status": "satisfied_for_reporting",
            },
            {
                "requirement_id": "R2-heldout-quarantine",
                "requirement": "Do not use the failed ResNet34/CIFAR-100-LT or ResNet18/CIFAR-10-LT held-outs to fit, select, or tune the next score.",
                "why_required": "These splits have already diagnosed v2 failure and are no longer clean final held-outs.",
                "acceptance_evidence": "The next protocol states that these splits are diagnostic-only and records fresh final held-out split IDs.",
                "current_status": "required_for_next_protocol",
            },
            {
                "requirement_id": "R3-theory-derived-score",
                "requirement": "Define the next score from theorem quantities before looking at fresh held-out targets.",
                "why_required": "A top-tier predictive condition must not be a post-hoc regression on failed held-outs.",
                "acceptance_evidence": "A frozen score registry names the mathematical terms, signs, transforms, and any coefficients before final evaluation.",
                "current_status": "missing",
            },
            {
                "requirement_id": "R4-nested-calibration",
                "requirement": "If coefficients are learned, learn them only on calibration families with a nested validation split.",
                "why_required": "Retrospective checkpoint success did not survive architecture/data transfer.",
                "acceptance_evidence": "Calibration, validation, and final held-out split roles are disjoint and recorded in CSV.",
                "current_status": "missing",
            },
            {
                "requirement_id": "R5-fresh-heldout-gates",
                "requirement": "Evaluate fresh architecture and data held-outs with CI lower endpoint above zero for residual Spearman and threshold accuracy above 0.8.",
                "why_required": "The current P0 claim failed exactly these held-out residual-ranking gates.",
                "acceptance_evidence": "Both fresh split gate rows pass, with early-layer, source-observed, and legacy scaled-JVP comparisons reported.",
                "current_status": "missing",
            },
            {
                "requirement_id": "R6-negative-outcome-reporting",
                "requirement": "Report negative, inverted, or baseline-dominated outcomes without replacing the target after seeing results.",
                "why_required": "The CIFAR-10-LT data split shows a baseline dominating the calibrated v2 score.",
                "acceptance_evidence": "The claim ledger and gate report preserve failed gates and baseline wins.",
                "current_status": "satisfied_for_current_failure",
            },
        ]
    )


def main() -> None:
    heldout = pd.read_csv(HELDOUT_SUMMARY_PATH)
    gates = pd.read_csv(HELDOUT_GATES_PATH)
    retrospective = pd.read_csv(RETROSPECTIVE_SUMMARY_PATH)
    target_register = build_target_register(heldout, retrospective)
    protocol_requirements = build_protocol_requirements()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_register.to_csv(TARGET_REGISTER_PATH, index=False)
    protocol_requirements.to_csv(PROTOCOL_REQUIREMENTS_PATH, index=False)

    text = f"""# E11 Condition-Score Theory Bridge

This generated bridge turns the held-out condition-score failure into a theory-facing protocol. It separates the theorem-adjacent threshold direction from the stronger residual layer-risk ranking target, and it records what a fresh P0 predictive-condition attempt must do without tuning on the failed held-outs.

## Score Target Register

{markdown_table(target_register, ["target_id", "theory_object", "measured_target", "current_evidence", "status", "allowed_claim", "blocked_claim"])}

## Current Held-Out Gates

{markdown_table(gates, ["gate_id", "scope", "status", "evidence"])}

## Fresh Protocol Requirements

{markdown_table(protocol_requirements, ["requirement_id", "requirement", "why_required", "acceptance_evidence", "current_status"])}

## Theory Consequence

The current theorem-to-score bridge is not a scalar success story. A below-one drift-threshold guardrail survived both held-outs, but the residual-ranking target failed on both registered final splits. These held-out splits are now spent: they may define the obstruction and motivate a new theory-derived score, but they cannot be used to tune that score and then serve as clean P0 evidence.

## Paper Claim Boundary

Allowed: the present condition-score evidence supports a local matched-head-gain drift mechanism and a direction-threshold guardrail.

Blocked: the present condition-score evidence does not support a claim that `condition_score_v2_calibrated_residual` predicts held-out residual layer-risk ranking across architecture and data families.

Artifacts:
- [score_target_register.csv](../{TARGET_REGISTER_PATH.as_posix()})
- [fresh_protocol_requirements.csv](../{PROTOCOL_REQUIREMENTS_PATH.as_posix()})
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved condition-score theory bridge to {OUTPUT_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
