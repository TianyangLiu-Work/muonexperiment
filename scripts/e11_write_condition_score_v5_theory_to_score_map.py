from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_condition_score_v5_theory_to_score_map")
DISCUSSION_PATH = Path("discussion/e11_condition_score_v5_theory_to_score_map.md")
V5_THEORY_DIR = Path("results/e11_condition_score_v5_theory_protocol")
V5_FREEZE_DIR = Path("results/e11_condition_score_v5_protocol/validation_score_freeze")
V4_AUDIT_DIR = Path("results/e11_condition_score_v4_failure_mechanism_audit")


def ci_text(row: pd.Series, mean: str = "mean_spearman_score_vs_target_residual") -> str:
    return f"{fmt(row[mean])} [{fmt(row['spearman_ci95_low'])}, {fmt(row['spearman_ci95_high'])}]"


def one_axis(axis_pairs: pd.DataFrame, split_role: str, score_id: str) -> pd.Series:
    rows = axis_pairs[
        axis_pairs["split_role"].eq(split_role)
        & axis_pairs["score_id"].eq(score_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one axis row for {split_role}/{score_id}, found {len(rows)}")
    return rows.iloc[0]


def obstruction_text(obstructions: pd.DataFrame, obstruction_id: str) -> str:
    rows = obstructions[obstructions["obstruction_id"].eq(obstruction_id)]
    if len(rows) != 1:
        raise ValueError(f"expected one obstruction row for {obstruction_id}, found {len(rows)}")
    return str(rows.iloc[0]["evidence"])


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "theory_terms": pd.read_csv(V5_THEORY_DIR / "theory_term_register.csv"),
        "score_contract": pd.read_csv(V5_THEORY_DIR / "score_contract.csv"),
        "freeze_formulas": pd.read_csv(V5_FREEZE_DIR / "score_formula_registry.csv"),
        "freeze_status": pd.read_csv(V5_FREEZE_DIR / "freeze_status.csv"),
        "freeze_gates": pd.read_csv(V5_FREEZE_DIR / "validation_gate_report.csv"),
        "axis_pairs": pd.read_csv(V4_AUDIT_DIR / "axis_pair_summary.csv"),
        "obstructions": pd.read_csv(V4_AUDIT_DIR / "obstruction_summary.csv"),
    }


def build_theorem_proxy_map() -> pd.DataFrame:
    rows = [
        {
            "map_id": "M1-sandwiched-tail-risk",
            "theorem_quantity": "R_l^p(D)=||B_T,p,l D_l A_T,p,l||_F^2 after matched head gain",
            "measured_proxy": "geomean observed and finite-difference JVP tail-drift ratios",
            "score_feature": "log_scaled_jvp_fro_amplitude",
            "identifiability_assumption": "local linearization error is small enough that layerwise JVP amplitude orders the nonlinear one-step tail response",
            "transport_term": "partition_transport_defect and architecture_transport_defect",
            "validation_gate": "residual-candidate Spearman CI lower endpoint above zero on the validation split",
            "claim_boundary": "cannot claim residual-risk prediction from rank-only or direction-only evidence",
        },
        {
            "map_id": "M2-direction-ratio",
            "theorem_quantity": "log R_l^p(D_spectral)-log R_l^p(D_frobenius)",
            "measured_proxy": "log scaled-JVP spectral/Frobenius squared ratio",
            "score_feature": "log_scaled_jvp_ratio",
            "identifiability_assumption": "matched-head-gain scaling makes the two directions comparable within each target partition",
            "transport_term": "none for threshold sign; separate from residual ranking",
            "validation_gate": "below-one threshold accuracy lower endpoint at least 0.8",
            "claim_boundary": "a passing direction guardrail does not support a residual-risk claim if ranking fails",
        },
        {
            "map_id": "M3-source-depth-residual",
            "theorem_quantity": "residual layer risk after subtracting a source depth baseline",
            "measured_proxy": "target residual from source-fit depth intercept and slope",
            "score_feature": "log_early_layer_prior and frozen_depth_slope",
            "identifiability_assumption": "depth nuisance is stable enough to subtract without using final target labels",
            "transport_term": "early_depth_nuisance",
            "validation_gate": "primary score beats early_layer_prior on residual ranking",
            "claim_boundary": "failure to beat the early-layer baseline blocks the predictive-condition claim",
        },
        {
            "map_id": "M4-partition-transport",
            "theorem_quantity": "change in class-conditioned B_T,p,l and A_T,p,l weighting across partitions",
            "measured_proxy": "pre-update partition tags and validation-frozen transport penalties",
            "score_feature": "transport_is_downsample and transport_is_classifier plus future partition statistics",
            "identifiability_assumption": "transport features are computed before final target residual labels exist",
            "transport_term": "partition_transport_defect",
            "validation_gate": "validation-freeze artifact exists before final output directories",
            "claim_boundary": "spent CIFAR-10 final rows may motivate this claim boundary but cannot tune it",
        },
        {
            "map_id": "M5-architecture-transport",
            "theorem_quantity": "change in layer shape, bottleneck/downsample map, and classifier transport across architectures",
            "measured_proxy": "architecture and parameterization tags registered before the ResNeXt50 final split",
            "score_feature": "transport_is_downsample and transport_is_classifier",
            "identifiability_assumption": "architecture tags encode parameterization changes without inspecting final residual outcomes",
            "transport_term": "architecture_transport_defect",
            "validation_gate": "fresh ResNeXt50-32x4d final is evaluated only after validation freeze",
            "claim_boundary": "architecture final failure narrows the claim to the local mechanism or fixed architecture family",
        },
    ]
    return pd.DataFrame(rows)


def build_score_lineage(freeze_formulas: pd.DataFrame, freeze_status: pd.DataFrame) -> pd.DataFrame:
    selected = freeze_status.set_index("item").loc["v5 transport-normalized residual score"]
    lineage = {
        "condition_score_v5_direction_axis_scaled_jvp_ratio": (
            "direction_ratio_guardrail",
            "direction guardrail",
            "not eligible as a residual-risk scalar",
        ),
        "condition_score_v5_raw_fro_amplitude_axis": (
            "raw_residual_amplitude",
            "diagnostic residual axis",
            "spent v4 data evidence shows raw amplitude can reverse",
        ),
        "condition_score_v5_transport_normalized_amplitude_minus_direction": (
            "raw_residual_amplitude; direction_ratio_guardrail; partition_transport_defect; architecture_transport_defect; early_depth_nuisance",
            "primary residual candidate",
            "eligible only if validation residual Spearman CI lower endpoint is above zero",
        ),
        "condition_score_v5_transport_defect_penalty": (
            "partition_transport_defect; architecture_transport_defect",
            "diagnostic transport axis",
            "reported to localize transport failures, not a final residual score",
        ),
        "early_layer_prior": (
            "early_depth_nuisance",
            "baseline",
            "primary must beat this baseline",
        ),
        "source_observed_drift_positive_control": (
            "sandwiched_tail_drift",
            "upper-bound positive control",
            "uses observed source drift and is not claim-eligible",
        ),
        "condition_score_v5_validation_selected": (
            "validation-freeze alias",
            "primary alias",
            f"{selected['status']} / {selected['evidence']}",
        ),
    }
    rows: list[dict[str, object]] = []
    for _, formula in freeze_formulas.iterrows():
        terms, claim_role, restriction = lineage[str(formula["score_id"])]
        rows.append(
            {
                "score_id": formula["score_id"],
                "score_role": formula["role"],
                "theory_terms_used": terms,
                "claim_role": claim_role,
                "leakage_status": "no spent final rows",
                "current_validation_status": selected["status"]
                if formula["score_id"] == "condition_score_v5_validation_selected"
                else formula["selected_for_final_evaluation"],
                "restriction": restriction,
            }
        )
    return pd.DataFrame(rows)


def build_transport_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step_id": "T1-source-calibration",
                "operation": "Compute source feature means, variances, and depth baseline on the source checkpoint-transfer split.",
                "allowed_inputs": "source layer_summary.csv and source metrics.csv",
                "forbidden_inputs": "v2/v3/v4 final residual labels; v5 final residual labels",
                "output": "source-standardized amplitude, ratio, and early-depth features",
                "freeze_point": "before validation scoring",
                "gate": "source statistics are deterministic and regenerated by the freeze script",
            },
            {
                "step_id": "T2-transport-tags",
                "operation": "Apply pre-registered downsample/classifier transport penalties.",
                "allowed_inputs": "parameter names and architecture tags visible before final outcomes",
                "forbidden_inputs": "final split residual ranking",
                "output": "transport_is_downsample and transport_is_classifier penalties",
                "freeze_point": "in score_formula_registry.csv before final split output exists",
                "gate": "score_formula_registry.csv reports uses_spent_final_rows=no",
            },
            {
                "step_id": "T3-validation-freeze",
                "operation": "Use only the v5 validation split to freeze or reject the residual candidate.",
                "allowed_inputs": "v5 validation layer_summary.csv and metrics.csv",
                "forbidden_inputs": "v5 final architecture/data outputs",
                "output": "frozen or validation_failed residual score status",
                "freeze_point": "discussion/e11_condition_score_v5_validation_freeze.md",
                "gate": "V5F-4 residual-score-freeze and V5F-5 direction-threshold-guardrail",
            },
            {
                "step_id": "T4-final-evaluation",
                "operation": "Evaluate the frozen score without changing weights, thresholds, features, or claim boundary.",
                "allowed_inputs": "new ResNeXt50-32x4d and CIFAR-10 cross-partition final outputs",
                "forbidden_inputs": "any post-hoc final coefficient or feature selection",
                "output": "final residual and direction gate report",
                "freeze_point": "after T3 passes",
                "gate": "both final splits must pass residual Spearman and direction threshold gates",
            },
            {
                "step_id": "T5-negative-path",
                "operation": "If validation or final gates fail, preserve the failure as boundary evidence.",
                "allowed_inputs": "failed validation/final reports",
                "forbidden_inputs": "reranking scores on visible failed final rows",
                "output": "narrow fixed-partition/local-mechanism claim",
                "freeze_point": "immediately after a failed gate",
                "gate": "README and gap register keep predictive-condition claim not_ready",
            },
        ]
    )


def build_falsifiable_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_id": "V5-P1-direction-guardrail",
                "target": "validation and final splits",
                "claim_tested": "The below-one spectral/Frobenius direction remains a separate positive guardrail.",
                "pass_rule": "threshold accuracy lower endpoint is at least 0.8",
                "fail_interpretation": "the local direction mechanism does not transport to this split",
                "claim_effect": "blocks even the direction-guardrail extension",
            },
            {
                "prediction_id": "V5-P2-residual-validation",
                "target": "validation-only mod-4 CIFAR-100-LT split",
                "claim_tested": "Transport-normalized amplitude-minus-direction predicts residual layer risk before final outputs exist.",
                "pass_rule": "residual Spearman CI lower endpoint is above zero",
                "fail_interpretation": "the current transport terms are still insufficient",
                "claim_effect": "do not run final splits for P0 predictive-condition evidence",
            },
            {
                "prediction_id": "V5-P3-architecture-final",
                "target": "ResNeXt50-32x4d CIFAR-100-LT final split",
                "claim_tested": "The frozen score survives a new architecture family and implementation path.",
                "pass_rule": "residual Spearman CI lower endpoint above zero and direction gate passes",
                "fail_interpretation": "architecture transport defect remains unmodeled",
                "claim_effect": "narrow to fixed architecture or local mechanism",
            },
            {
                "prediction_id": "V5-P4-data-final",
                "target": "CIFAR-10 cross-partition final split",
                "claim_tested": "The frozen score survives a new data-family partition after prior CIFAR-10 finals are spent.",
                "pass_rule": "residual Spearman CI lower endpoint above zero and direction gate passes",
                "fail_interpretation": "partition transport defect remains unmodeled",
                "claim_effect": "no broad predictive-condition claim",
            },
            {
                "prediction_id": "V5-P5-baseline-dominance",
                "target": "validation and final residual summaries",
                "claim_tested": "The score contains information beyond the early-depth nuisance baseline.",
                "pass_rule": "primary residual score beats early_layer_prior on residual ranking and top-k overlap",
                "fail_interpretation": "the measurable score is mostly a depth/stage proxy",
                "claim_effect": "paper reports a nuisance-baseline failure, not a theorem-derived predictor",
            },
        ]
    )


def build_ablation_matrix(axis_pairs: pd.DataFrame, obstructions: pd.DataFrame) -> pd.DataFrame:
    data_role = "fresh_final_heldout_data_partition"
    arch_role = "fresh_final_heldout_architecture"
    data_primary = one_axis(axis_pairs, data_role, "condition_score_v4_two_axis_amplitude_minus_direction")
    data_direction = one_axis(axis_pairs, data_role, "v4_direction_axis_scaled_jvp_ratio")
    data_amplitude = one_axis(axis_pairs, data_role, "v4_residual_amplitude_axis_scaled_jvp_fro")
    data_early = one_axis(axis_pairs, data_role, "early_layer_prior")
    arch_primary = one_axis(axis_pairs, arch_role, "condition_score_v4_two_axis_amplitude_minus_direction")
    return pd.DataFrame(
        [
            {
                "ablation_id": "A1-direction-only",
                "isolated_or_removed_term": "direction_ratio_guardrail only",
                "spent_v4_observation": f"CIFAR-10 mixed direction axis stays positive: {ci_text(data_direction)}; {obstruction_text(obstructions, 'V4-O3-direction-is-not-the-failure')}",
                "expected_v5_failure_mode": "passes below-one direction while failing residual-risk ranking",
                "required_v5_report": "direction threshold reported separately from residual Spearman",
            },
            {
                "ablation_id": "A2-raw-amplitude-only",
                "isolated_or_removed_term": "raw_residual_amplitude without transport",
                "spent_v4_observation": f"CIFAR-10 mixed amplitude reverses: {ci_text(data_amplitude)}",
                "expected_v5_failure_mode": "partition changes flip amplitude/depth ordering",
                "required_v5_report": "raw amplitude remains diagnostic-only",
            },
            {
                "ablation_id": "A3-early-depth-only",
                "isolated_or_removed_term": "early_depth_nuisance baseline",
                "spent_v4_observation": f"CIFAR-10 mixed early-layer prior reverses: {ci_text(data_early)}",
                "expected_v5_failure_mode": "score is only a layer-depth proxy",
                "required_v5_report": "primary score must beat early_layer_prior",
            },
            {
                "ablation_id": "A4-v4-amplitude-minus-direction",
                "isolated_or_removed_term": "partition_transport_defect omitted from scalar aggregation",
                "spent_v4_observation": f"WideResNet50-2 passes at {ci_text(arch_primary)} but CIFAR-10 mixed fails at {ci_text(data_primary)}",
                "expected_v5_failure_mode": "architecture transfer may pass while data-partition transfer fails",
                "required_v5_report": "architecture and data finals are separate gates",
            },
            {
                "ablation_id": "A5-transport-normalized-candidate",
                "isolated_or_removed_term": "all v5 terms included before final",
                "spent_v4_observation": "spent rows motivate the term but are quarantined from fitting",
                "expected_v5_failure_mode": "validation_failed or final not_ready if transport is still insufficient",
                "required_v5_report": "preserve negative validation/final outcomes without score retuning",
            },
        ]
    )


def build_readiness_ledger(freeze_status: pd.DataFrame, freeze_gates: pd.DataFrame) -> pd.DataFrame:
    status = freeze_status.set_index("item")["status"].to_dict()
    gates = freeze_gates.set_index("gate_id")["status"].to_dict()
    validation_status = status["v5 validation split output"]
    residual_status = status["v5 transport-normalized residual score"]
    final_status = status["v5 final split outputs"]
    final_unblocked = (
        validation_status == "generated"
        and residual_status == "frozen"
        and final_status == "not_run"
        and gates["V5F-6-final-claim-readiness"] == "pass"
    )
    validation_failed = validation_status == "generated" and residual_status == "validation_failed"
    if final_unblocked:
        predictive_evidence = (
            "frozen validation-selected score is eligible for final evaluation runs; "
            "final split outputs are still absent, so the P0 predictive-condition claim remains not_ready"
        )
    elif validation_failed:
        predictive_evidence = (
            "validation output exists but the residual score failed its freeze gate, "
            "so final evaluation is not claim-eligible"
        )
    else:
        predictive_evidence = (
            "validation freeze is pending and final split outputs are not claim-eligible yet"
        )
    return pd.DataFrame(
        [
            {
                "item": "theory-to-score map",
                "status": "generated",
                "evidence": DISCUSSION_PATH.as_posix(),
                "blocks_p0_if_missing": "yes",
            },
            {
                "item": "v5 validation output",
                "status": status["v5 validation split output"],
                "evidence": "results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/layer_summary.csv",
                "blocks_p0_if_missing": "yes",
            },
            {
                "item": "v5 residual score freeze",
                "status": status["v5 transport-normalized residual score"],
                "evidence": "results/e11_condition_score_v5_protocol/validation_score_freeze/freeze_status.csv",
                "blocks_p0_if_missing": "yes",
            },
            {
                "item": "no final before freeze",
                "status": gates["V5F-2-no-final-before-freeze"],
                "evidence": "results/e11_condition_score_v5_protocol/final_* directories absent before freeze",
                "blocks_p0_if_missing": "yes",
            },
            {
                "item": "predictive-condition claim",
                "status": "not_ready",
                "evidence": predictive_evidence,
                "blocks_p0_if_missing": "yes",
            },
        ]
    )


def build_boundary_text(readiness: pd.DataFrame) -> str:
    readiness_lookup = readiness.set_index("item")["evidence"].to_dict()
    predictive_evidence = str(readiness_lookup["predictive-condition claim"])
    if predictive_evidence.startswith("frozen validation-selected score"):
        return """Allowed now: cite this map as the pre-final theory-to-score bridge, use spent
v4 evidence only as diagnostic motivation, and run the v5 final splits with the
frozen validation-selected residual score.

Blocked now: fitting, selecting, or reweighting any v5 score on v2/v3/v4 final
rows; claiming that the transport-normalized residual score predicts held-out
layer risk before both v5 final gates pass."""
    return """Allowed now: cite this map as the pre-final theory-to-score bridge, use spent
v4 evidence only as diagnostic motivation, and keep the v5 final splits blocked
until the validation-freeze artifact reports a frozen residual score and a
passing direction guardrail.

Blocked now: fitting, selecting, or reweighting any v5 score on v2/v3/v4 final
rows; claiming that the transport-normalized residual score predicts held-out
layer risk before the v5 validation and final gates exist."""


def write_discussion(
    theorem_proxy_map: pd.DataFrame,
    score_lineage: pd.DataFrame,
    transport_contract: pd.DataFrame,
    predictions: pd.DataFrame,
    ablations: pd.DataFrame,
    readiness: pd.DataFrame,
    boundary: str,
) -> None:
    text = f"""# E11 Condition-Score V5 Theory-to-Score Map

This generated artifact is a theory-to-measurement bridge for the v5
condition-score attempt. It does not make the P0 predictive-condition claim.
It states the mathematical quantity, the measurable proxy, the leakage
boundary, and the falsifiable validation/final gates that would be needed
before such a claim is supportable.

## Proposition: Transport-Stable Sandwich Residual

For a target partition or architecture `p`, write the matched-head-gain
layerwise tail risk as
`R_l^p(D)=||B_T,p,l D_l A_T,p,l||_F^2`. If a frozen transport term
`T_l(p,s)` satisfies
`|log R_l^p(D_fro)-log R_l^s(D_fro)-T_l(p,s)| <= epsilon_l` without using
final residual labels, and if the direction ratio
`log R_l^p(D_spectral)-log R_l^p(D_fro)` is evaluated as a separate guardrail,
then a source-standardized amplitude-minus-direction score can preserve
residual-risk ordering only up to the unresolved transport error
`epsilon_l`. When those errors exceed the residual pairwise gaps, sign
reversal is an expected failure mode rather than a statistical accident.

Proof sketch. The matrix-block theorem controls the sandwiched quantity
`B_T D A_T` after head-gain matching. Source-to-target transfer changes the
outer and inner tail operators, so raw amplitude and depth terms acquire a
transport error. Subtracting the direction ratio removes a different target:
the below-one direction comparison. Therefore residual ranking requires a
frozen transport correction and a separate direction guardrail.

## Theorem Proxy Map

{markdown_table(theorem_proxy_map, ["map_id", "theorem_quantity", "measured_proxy", "score_feature", "transport_term", "validation_gate", "claim_boundary"])}

## Score Lineage

{markdown_table(score_lineage, ["score_id", "score_role", "theory_terms_used", "claim_role", "leakage_status", "current_validation_status", "restriction"])}

## Transport Normalization Contract

{markdown_table(transport_contract, ["step_id", "operation", "allowed_inputs", "forbidden_inputs", "output", "freeze_point", "gate"])}

## Falsifiable Predictions

{markdown_table(predictions, ["prediction_id", "target", "claim_tested", "pass_rule", "fail_interpretation", "claim_effect"])}

## Required Ablation Matrix

{markdown_table(ablations, ["ablation_id", "isolated_or_removed_term", "spent_v4_observation", "expected_v5_failure_mode", "required_v5_report"])}

## Claim Readiness Ledger

{markdown_table(readiness, ["item", "status", "evidence", "blocks_p0_if_missing"])}

## Boundary

{boundary}

Artifacts:
- [theorem_proxy_map.csv](../results/e11_condition_score_v5_theory_to_score_map/theorem_proxy_map.csv)
- [score_lineage.csv](../results/e11_condition_score_v5_theory_to_score_map/score_lineage.csv)
- [transport_normalization_contract.csv](../results/e11_condition_score_v5_theory_to_score_map/transport_normalization_contract.csv)
- [falsifiable_predictions.csv](../results/e11_condition_score_v5_theory_to_score_map/falsifiable_predictions.csv)
- [ablation_matrix.csv](../results/e11_condition_score_v5_theory_to_score_map/ablation_matrix.csv)
- [claim_readiness_ledger.csv](../results/e11_condition_score_v5_theory_to_score_map/claim_readiness_ledger.csv)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    inputs = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    theorem_proxy_map = build_theorem_proxy_map()
    score_lineage = build_score_lineage(inputs["freeze_formulas"], inputs["freeze_status"])
    transport_contract = build_transport_contract()
    predictions = build_falsifiable_predictions()
    ablations = build_ablation_matrix(inputs["axis_pairs"], inputs["obstructions"])
    readiness = build_readiness_ledger(inputs["freeze_status"], inputs["freeze_gates"])
    boundary = build_boundary_text(readiness)

    theorem_proxy_map.to_csv(OUTPUT_DIR / "theorem_proxy_map.csv", index=False)
    score_lineage.to_csv(OUTPUT_DIR / "score_lineage.csv", index=False)
    transport_contract.to_csv(OUTPUT_DIR / "transport_normalization_contract.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "falsifiable_predictions.csv", index=False)
    ablations.to_csv(OUTPUT_DIR / "ablation_matrix.csv", index=False)
    readiness.to_csv(OUTPUT_DIR / "claim_readiness_ledger.csv", index=False)
    write_discussion(
        theorem_proxy_map,
        score_lineage,
        transport_contract,
        predictions,
        ablations,
        readiness,
        boundary,
    )
    print(f"saved condition-score v5 theory-to-score map to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(readiness.to_string(index=False))


if __name__ == "__main__":
    main()
