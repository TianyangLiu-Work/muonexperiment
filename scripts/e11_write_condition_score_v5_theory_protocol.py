from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_condition_score_v5_theory_protocol")
DISCUSSION_PATH = Path("discussion/e11_condition_score_v5_theory_protocol.md")
V4_AUDIT_DIR = Path("results/e11_condition_score_v4_failure_mechanism_audit")
V4_FINAL_DIR = Path("results/e11_condition_score_v4_protocol/final_score_evaluation")


def _axis_row(axis_pair_summary: pd.DataFrame, split_role: str, score_id: str) -> pd.Series:
    match = axis_pair_summary[
        axis_pair_summary["split_role"].eq(split_role)
        & axis_pair_summary["score_id"].eq(score_id)
    ]
    if len(match) != 1:
        raise ValueError(f"expected one row for split_role={split_role}, score_id={score_id}; found {len(match)}")
    return match.iloc[0]


def _outcome_row(outcome: pd.DataFrame, split_role: str, score: str) -> pd.Series:
    match = outcome[
        outcome["split_role"].eq(split_role)
        & outcome["score"].eq(score)
    ]
    if len(match) != 1:
        raise ValueError(f"expected one row for split_role={split_role}, score={score}; found {len(match)}")
    return match.iloc[0]


def load_v4_evidence() -> dict[str, pd.Series]:
    axis_pair_summary = pd.read_csv(V4_AUDIT_DIR / "axis_pair_summary.csv")
    outcome = pd.read_csv(V4_AUDIT_DIR / "final_outcome_matrix.csv")
    final_summary = pd.read_csv(V4_FINAL_DIR / "final_score_summary.csv")
    return {
        "data_primary_axis": _axis_row(
            axis_pair_summary,
            "fresh_final_heldout_data_partition",
            "condition_score_v4_two_axis_amplitude_minus_direction",
        ),
        "data_direction_axis": _axis_row(
            axis_pair_summary,
            "fresh_final_heldout_data_partition",
            "v4_direction_axis_scaled_jvp_ratio",
        ),
        "data_amplitude_axis": _axis_row(
            axis_pair_summary,
            "fresh_final_heldout_data_partition",
            "v4_residual_amplitude_axis_scaled_jvp_fro",
        ),
        "data_early_axis": _axis_row(axis_pair_summary, "fresh_final_heldout_data_partition", "early_layer_prior"),
        "arch_primary_axis": _axis_row(
            axis_pair_summary,
            "fresh_final_heldout_architecture",
            "condition_score_v4_two_axis_amplitude_minus_direction",
        ),
        "data_primary_outcome": _outcome_row(
            outcome,
            "fresh_final_heldout_data_partition",
            "condition_score_v4_two_axis_amplitude_minus_direction",
        ),
        "arch_primary_outcome": _outcome_row(
            outcome,
            "fresh_final_heldout_architecture",
            "condition_score_v4_two_axis_amplitude_minus_direction",
        ),
        "data_source_control": _outcome_row(
            outcome,
            "fresh_final_heldout_data_partition",
            "source_observed_drift_positive_control",
        ),
        "arch_source_control": _outcome_row(
            outcome,
            "fresh_final_heldout_architecture",
            "source_observed_drift_positive_control",
        ),
        "final_gate": _outcome_row(
            outcome,
            "p0_predictive_condition",
            "condition_score_v4_two_axis_amplitude_minus_direction",
        ),
        "final_data_direction": final_summary[
            final_summary["split_role"].eq("fresh_final_heldout_data_partition")
            & final_summary["score"].eq("condition_score_v4_direction_axis_scaled_jvp_ratio")
        ].iloc[0],
    }


def score_ci(row: pd.Series, column: str = "mean_spearman_score_vs_target_residual") -> str:
    return f"{fmt(row[column])} [{fmt(row['spearman_ci95_low'])}, {fmt(row['spearman_ci95_high'])}]"


def build_theory_term_register(evidence: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term_id": "sandwiched_tail_drift",
                "mathematical_object": "R_l(D)=||B_T,l D_l A_T,l||_F^2 after matching head gain",
                "measurable_proxy": "finite-difference observed tail drift and per-layer JVP tail drift",
                "theorem_role": "root quantity that connects a matrix-block theorem to layerwise network diagnostics",
                "current_evidence": (
                    f"source-observed control Spearman is {score_ci(evidence['arch_source_control'])} on WideResNet50-2 "
                    f"and {score_ci(evidence['data_source_control'])} on CIFAR-10 mixed"
                ),
                "v5_rule": "All residual-risk scores must be stated as approximations to this sandwiched quantity, not as rank-only proxies.",
            },
            {
                "term_id": "direction_ratio_guardrail",
                "mathematical_object": "log R_l(D_spectral)-log R_l(D_frobenius) at matched head gain",
                "measurable_proxy": "log scaled-JVP spectral/Frobenius tail-drift ratio",
                "theorem_role": "below-one direction target; separate from residual-risk ranking",
                "current_evidence": (
                    f"CIFAR-10 mixed direction-axis residual Spearman remains positive at {score_ci(evidence['data_direction_axis'])} "
                    f"and threshold accuracy is {fmt(evidence['final_data_direction']['mean_threshold_below_one_accuracy'])}"
                ),
                "v5_rule": "Keep this as a guardrail; do not subtract it into a residual-risk scalar without a frozen transport normalization.",
            },
            {
                "term_id": "raw_residual_amplitude",
                "mathematical_object": "log ||B_T,l D_fro,l A_T,l||_F^2 after source-depth adjustment",
                "measurable_proxy": "log Frobenius matched-head-gain scaled-JVP tail-drift amplitude",
                "theorem_role": "absolute downstream tail-sensitivity magnitude",
                "current_evidence": (
                    f"CIFAR-10 mixed Frobenius-amplitude Spearman is {score_ci(evidence['data_amplitude_axis'])}"
                ),
                "v5_rule": "Raw amplitude is not claim-eligible by itself because the v4 final data split shows sign reversal.",
            },
            {
                "term_id": "partition_transport_defect",
                "mathematical_object": "change in class-conditioned tail covariance/Jacobian weighting between source and target partitions",
                "measurable_proxy": "pre-update class-conditioned activation/JVP covariance transport, measured before final target residual labels",
                "theorem_role": "missing invariance term exposed by the CIFAR-10 mixed reversal",
                "current_evidence": (
                    f"v4 primary score is {score_ci(evidence['data_primary_axis'])} on CIFAR-10 mixed while the direction axis stays positive"
                ),
                "v5_rule": "A v5 score must either include a frozen partition-transport normalization or narrow the claim to fixed-partition architecture transfer.",
            },
            {
                "term_id": "architecture_transport_defect",
                "mathematical_object": "change in parameterization, bottleneck/downsample transport, and layer shape",
                "measurable_proxy": "stage/block/shape tags plus validation-frozen architecture normalization",
                "theorem_role": "prevents a scalar ratio from silently changing meaning across architectures",
                "current_evidence": (
                    f"v4 WideResNet50-2 primary score passes with {score_ci(evidence['arch_primary_axis'])}; "
                    "earlier v3 ResNet50 ratio-only evidence remains a spent architecture reversal"
                ),
                "v5_rule": "Use a fresh architecture split with a new implementation path; report architecture normalization before final evaluation.",
            },
            {
                "term_id": "early_depth_nuisance",
                "mathematical_object": "depth and stage-dependent residual-risk baseline unrelated to the spectral/Frobenius direction",
                "measurable_proxy": "early_layer_prior and source-fit depth residuals",
                "theorem_role": "nuisance term that must be beaten or explicitly absorbed",
                "current_evidence": f"CIFAR-10 mixed early-layer prior Spearman is {score_ci(evidence['data_early_axis'])}",
                "v5_rule": "Every final report must compare against early_layer_prior; failure to beat it blocks the predictive-condition claim.",
            },
        ]
    )


def build_score_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contract_id": "V5-C1-theorem-reduction",
                "requirement": "The score must reduce to a sandwiched tail-drift expression in the linear matrix-block setting.",
                "acceptance_gate": "The discussion states the matrix quantity, sign convention, and measurable network proxy before validation.",
                "blocks": "rank-only or layer-index-only predictors cannot be sold as theorem-derived scores.",
            },
            {
                "contract_id": "V5-C2-target-separation",
                "requirement": "Direction threshold and residual layer-risk ranking remain separate targets.",
                "acceptance_gate": "Final reports include separate direction-threshold accuracy and residual Spearman gates.",
                "blocks": "a positive below-one direction result cannot rescue a failed residual-ranking result.",
            },
            {
                "contract_id": "V5-C3-transport-normalized-amplitude",
                "requirement": "Any residual-amplitude term must use a pre-registered partition/architecture transport normalization term.",
                "acceptance_gate": "The normalization is computed without final target residual labels and frozen before final split evaluation.",
                "blocks": "the v4 amplitude/depth reversal cannot be fixed by changing coefficients on spent final rows.",
            },
            {
                "contract_id": "V5-C4-spent-final-quarantine",
                "requirement": "All v2, v3, and v4 final rows are diagnostic-only after their evaluations are visible.",
                "acceptance_gate": "The spent evidence policy forbids score fitting, feature selection, threshold tuning, and final P0 evidence use.",
                "blocks": "reusing the failed CIFAR-10 mixed final split as validation data.",
            },
            {
                "contract_id": "V5-C5-freeze-before-final",
                "requirement": "The scalar score, coefficients, transforms, and claim boundary are committed before new final Slurm jobs run.",
                "acceptance_gate": "A validation-freeze artifact exists in git and final output directories are absent at freeze time.",
                "blocks": "post-hoc final split score selection.",
            },
            {
                "contract_id": "V5-C6-narrow-claim-fallback",
                "requirement": "If transport-normalized residual ranking is not ready, the paper narrows to direction guardrail plus local mechanism.",
                "acceptance_gate": "The claim ledger says predictive-condition P0 is not_ready and names the missing transport term.",
                "blocks": "overclaiming a predictive condition when only the direction guardrail is supported.",
            },
        ]
    )


def build_spent_evidence_policy(evidence: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_id": "spent_v2_architecture_resnet34",
                "source_artifact": "results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation",
                "result_summary": "registered v2 architecture final was weak/inconclusive for residual ranking",
                "allowed_use": "negative boundary evidence and obstruction taxonomy",
                "forbidden_use": "score fitting, feature selection, threshold tuning, validation selection, final P0 evidence",
            },
            {
                "split_id": "spent_v2_data_cifar10lt_original",
                "source_artifact": "results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation",
                "result_summary": "registered v2 CIFAR-10-LT data final inverted residual ranking",
                "allowed_use": "negative data-family evidence and baseline comparison",
                "forbidden_use": "score fitting, feature selection, threshold tuning, validation selection, final P0 evidence",
            },
            {
                "split_id": "spent_v3_architecture_resnet50",
                "source_artifact": "results/e11_condition_score_fresh_protocol/fresh_score_evaluation",
                "result_summary": "fresh v3 ResNet50 final inverted the ratio-only residual ranking",
                "allowed_use": "bottleneck/downsample obstruction analysis",
                "forbidden_use": "score fitting, feature selection, threshold tuning, validation selection, final P0 evidence",
            },
            {
                "split_id": "spent_v3_data_cifar10lt_alt",
                "source_artifact": "results/e11_condition_score_fresh_protocol/fresh_score_evaluation",
                "result_summary": "fresh v3 CIFAR-10 alternate final passed but is paired with the failed ResNet50 final",
                "allowed_use": "positive diagnostic boundary only",
                "forbidden_use": "score fitting, feature selection, threshold tuning, validation selection, final P0 evidence",
            },
            {
                "split_id": "spent_v4_architecture_wide_resnet50_2",
                "source_artifact": "results/e11_condition_score_v4_protocol/final_score_evaluation",
                "result_summary": f"v4 architecture final passed residual ranking with {score_ci(evidence['arch_primary_outcome'])}",
                "allowed_use": "positive architecture-transfer boundary evidence",
                "forbidden_use": "score fitting, feature selection, threshold tuning, validation selection, final P0 evidence",
            },
            {
                "split_id": "spent_v4_data_cifar10lt_mixed",
                "source_artifact": "results/e11_condition_score_v4_protocol/final_score_evaluation",
                "result_summary": f"v4 CIFAR-10 mixed final inverted residual ranking with {score_ci(evidence['data_primary_outcome'])}",
                "allowed_use": "negative data-partition boundary and transport-defect diagnosis",
                "forbidden_use": "score fitting, feature selection, threshold tuning, validation selection, final P0 evidence",
            },
        ]
    )


def build_unspent_split_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_id": "v5_validation_cifar100lt_mod4_partition",
                "role": "validation_only",
                "dataset": "CIFAR-100-LT",
                "architecture": "ResNet18 CIFAR stem",
                "class_partition": "head=classes mod 4 in {0,1}; tail=classes mod 4 in {2,3}",
                "purpose": "freeze transport normalization and any scalar aggregation before final splits",
                "entrypoint": "scripts/slurm/e11_cifar100_resnet_condition_score_v5_validation_mod4_partition.sbatch",
                "planned_artifact_prefix": "results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition",
                "claim_allowed": "no",
            },
            {
                "split_id": "v5_final_architecture_resnext50_32x4d_cifar100lt",
                "role": "fresh_final_heldout_architecture",
                "dataset": "CIFAR-100-LT",
                "architecture": "ResNeXt50-32x4d CIFAR stem",
                "class_partition": "head=0-49; tail=50-99",
                "purpose": "new architecture family after spent ResNet34/ResNet50/WideResNet50-2 finals",
                "entrypoint": "scripts/slurm/e11_cifar100_resnet_condition_score_v5_architecture_resnext50_32x4d.sbatch",
                "planned_artifact_prefix": "results/e11_condition_score_v5_protocol/final_architecture_resnext50_32x4d",
                "claim_allowed": "yes only after v5 validation-freeze commit",
            },
            {
                "split_id": "v5_final_data_cifar10lt_cross_partition",
                "role": "fresh_final_heldout_data_partition",
                "dataset": "CIFAR-10-LT",
                "architecture": "ResNet18 CIFAR stem",
                "class_partition": "head=0,3,4,6,9; tail=1,2,5,7,8",
                "purpose": "new CIFAR-10 partition after spent original/alternate/mixed data finals",
                "entrypoint": "scripts/slurm/e11_cifar100_resnet_condition_score_v5_data_cifar10_cross.sbatch",
                "planned_artifact_prefix": "results/e11_condition_score_v5_protocol/final_data_cifar10_cross",
                "claim_allowed": "yes only after v5 validation-freeze commit",
            },
        ]
    )


def build_acceptance_gates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate_id": "V5-1-theory-reduction",
                "scope": "score definition",
                "requirement": "Every claim-eligible score term maps to a sandwiched tail-drift quantity or an explicitly named nuisance/transport term.",
                "pass_condition": "theory_term_register.csv includes sandwiched, direction, raw amplitude, partition transport, architecture transport, and depth nuisance rows",
            },
            {
                "gate_id": "V5-2-spent-quarantine",
                "scope": "all v5 score development",
                "requirement": "Spent v2/v3/v4 final rows are diagnostic-only.",
                "pass_condition": "spent_evidence_policy.csv forbids score fitting, feature selection, threshold tuning, validation selection, and final P0 evidence use for every spent row",
            },
            {
                "gate_id": "V5-3-transport-before-scalar",
                "scope": "validation freeze",
                "requirement": "Partition/architecture transport normalization is specified before any scalar residual-risk aggregation is claim-eligible.",
                "pass_condition": "a future validation-freeze artifact computes the transport term without final residual labels",
            },
            {
                "gate_id": "V5-4-freeze-before-final",
                "scope": "fresh final split evaluation",
                "requirement": "The scalar score and claim boundary are frozen before v5 final Slurm jobs run.",
                "pass_condition": "final output directories are absent at the freeze commit and present only after the frozen evaluator runs",
            },
            {
                "gate_id": "V5-5-final-residual-and-direction",
                "scope": "P0 predictive-condition claim",
                "requirement": "Both fresh final splits pass residual ranking and direction guardrail.",
                "pass_condition": "residual Spearman CI lower endpoint is above zero and direction threshold accuracy is at least 0.8 on every v5 final split",
            },
            {
                "gate_id": "V5-6-narrow-claim-if-not-ready",
                "scope": "paper claim boundary",
                "requirement": "If V5-5 fails or no transport-normalized scalar is frozen, the paper keeps the predictive-condition claim not_ready.",
                "pass_condition": "README and claim ledger state the supported direction/local-mechanism claim and the missing predictive-condition evidence",
            },
        ]
    )


def write_outputs(
    theory_terms: pd.DataFrame,
    score_contract: pd.DataFrame,
    spent_policy: pd.DataFrame,
    split_requirements: pd.DataFrame,
    gates: pd.DataFrame,
    evidence: dict[str, pd.Series],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    theory_terms.to_csv(OUTPUT_DIR / "theory_term_register.csv", index=False)
    score_contract.to_csv(OUTPUT_DIR / "score_contract.csv", index=False)
    spent_policy.to_csv(OUTPUT_DIR / "spent_evidence_policy.csv", index=False)
    split_requirements.to_csv(OUTPUT_DIR / "unspent_split_requirements.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "acceptance_gates.csv", index=False)

    text = f"""# E11 Condition-Score V5 Theory Protocol

This generated protocol is not a positive P0 result. It converts the v4 final
failure into a transport-normalized score contract for the next score revision. The main
scientific point is that v4 did not fail at the below-one direction guardrail:
on the CIFAR-10 mixed final split, the direction axis remains positive
({score_ci(evidence['data_direction_axis'])}) while the raw Frobenius-amplitude
axis reverses ({score_ci(evidence['data_amplitude_axis'])}). A claim-eligible
v5 score therefore needs a transport-normalized residual-amplitude term or a
narrower fixed-partition claim.

## Theory Term Register

{markdown_table(theory_terms, ["term_id", "mathematical_object", "measurable_proxy", "theorem_role", "current_evidence", "v5_rule"])}

## Score Contract

{markdown_table(score_contract, ["contract_id", "requirement", "acceptance_gate", "blocks"])}

## Spent Evidence Policy

{markdown_table(spent_policy, ["split_id", "source_artifact", "result_summary", "allowed_use", "forbidden_use"])}

## Unspent Split Requirements

{markdown_table(split_requirements, ["split_id", "role", "dataset", "architecture", "class_partition", "purpose", "entrypoint", "planned_artifact_prefix", "claim_allowed"])}

## Acceptance Gates

{markdown_table(gates, ["gate_id", "scope", "requirement", "pass_condition"])}

## Theory Consequence

For a linear sandwich block, the layer risk is a sandwiched quantity
`||B_T D_l A_T||_F^2` after matching head gain. A direction ratio can tell
whether the spectral/Frobenius direction is below one, but it cannot by itself
rank residual layer risk when the target partition changes the tail covariance
seen by `B_T` and `A_T`. The v4 data split is exactly this obstruction: the
direction guardrail survives, but raw amplitude/depth aggregation reverses.

## Claim Boundary

Allowed now: use v2/v3/v4 final evaluations as spent positive or negative
boundary evidence, and use the v5 protocol to run a new validation-freeze
followed by new final splits.

Blocked now: fitting, selecting, or thresholding a v5 score on any v2/v3/v4 final row, or claiming a predictive condition before a transport-normalized score passes new unspent final architecture and data splits.

Artifacts:
- [theory_term_register.csv](../{(OUTPUT_DIR / 'theory_term_register.csv').as_posix()})
- [score_contract.csv](../{(OUTPUT_DIR / 'score_contract.csv').as_posix()})
- [spent_evidence_policy.csv](../{(OUTPUT_DIR / 'spent_evidence_policy.csv').as_posix()})
- [unspent_split_requirements.csv](../{(OUTPUT_DIR / 'unspent_split_requirements.csv').as_posix()})
- [acceptance_gates.csv](../{(OUTPUT_DIR / 'acceptance_gates.csv').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    evidence = load_v4_evidence()
    theory_terms = build_theory_term_register(evidence)
    score_contract = build_score_contract()
    spent_policy = build_spent_evidence_policy(evidence)
    split_requirements = build_unspent_split_requirements()
    gates = build_acceptance_gates()
    write_outputs(theory_terms, score_contract, spent_policy, split_requirements, gates, evidence)
    print(f"saved condition-score v5 theory protocol to {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
