from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_DIR = Path("results/e11_condition_score_v4_protocol")
OUTPUT_PATH = Path("discussion/e11_condition_score_v4_protocol.md")


def spent_split_rows() -> list[dict[str, str]]:
    return [
        {
            "split_id": "spent_v2_primary_heldout_architecture_resnet34",
            "source_artifact": "results/e11_cifar100_resnet_condition_score_next/heldout_architecture",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet34 CIFAR stem",
            "reason_spent": "registered v2 final held-out split; residual-ranking gate failed or was inconclusive",
            "allowed_use": "negative evidence, obstruction analysis, sanity-check replication",
            "forbidden_use": "feature selection, coefficient fitting, threshold tuning, validation selection, final P0 evidence",
        },
        {
            "split_id": "spent_v2_primary_heldout_data_cifar10lt_resnet18",
            "source_artifact": "results/e11_cifar100_resnet_condition_score_next/heldout_data",
            "dataset": "CIFAR-10-LT",
            "architecture": "ResNet18 CIFAR stem",
            "reason_spent": "registered v2 final held-out split; residual-ranking gate failed with sign reversal",
            "allowed_use": "negative evidence, obstruction analysis, sanity-check replication",
            "forbidden_use": "feature selection, coefficient fitting, threshold tuning, validation selection, final P0 evidence",
        },
        {
            "split_id": "spent_v3_fresh_architecture_resnet50",
            "source_artifact": "results/e11_condition_score_fresh_protocol/fresh_architecture_resnet50",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet50 CIFAR stem",
            "reason_spent": "fresh v3 final split; zero-fit scaled-JVP residual-ranking gate failed with sign reversal",
            "allowed_use": "negative evidence, bottleneck/downsample obstruction analysis",
            "forbidden_use": "feature selection, coefficient fitting, threshold tuning, validation selection, final P0 evidence",
        },
        {
            "split_id": "spent_v3_fresh_data_cifar10lt_alt_partition",
            "source_artifact": "results/e11_condition_score_fresh_protocol/fresh_data_cifar10_alt",
            "dataset": "CIFAR-10-LT",
            "architecture": "ResNet18 CIFAR stem",
            "reason_spent": "fresh v3 final split; residual-ranking gate passed but was paired with failed ResNet50 final architecture split",
            "allowed_use": "positive diagnostic boundary and split-specific sanity check",
            "forbidden_use": "feature selection, coefficient fitting, threshold tuning, validation selection, final P0 evidence",
        },
    ]


def score_axis_rows() -> list[dict[str, str]]:
    return [
        {
            "score_id": "condition_score_v4_two_axis_transport_jvp",
            "role": "primary_next_protocol_candidate",
            "target": "separate direction threshold from residual layer-risk ranking",
            "formula": "report the pair (log_scaled_jvp_ratio, log_scaled_jvp_frobenius_amplitude) with registered architecture transport tags; no scalar collapse before validation",
            "theory_link": "ratio controls spectral-vs-Frobenius direction; amplitude and transport tags control absolute downstream tail sensitivity",
            "coefficient_rule": "zero-fit axes; any scalar aggregation must be frozen in a future validation-commit artifact before final splits run",
            "uses_spent_final_splits": "no",
            "eligible_for_final_p0_claim": "not until validation coefficients/signs are frozen before unspent final splits",
        },
        {
            "score_id": "v4_direction_axis_scaled_jvp_ratio",
            "role": "required_direction_guardrail",
            "target": "below-one spectral/Frobenius direction",
            "formula": "log(geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro)",
            "theory_link": "finite-difference estimate of matched-head-gain direction advantage",
            "coefficient_rule": "zero-fit; positive axis means larger spectral/Frobenius ratio",
            "uses_spent_final_splits": "no",
            "eligible_for_final_p0_claim": "yes only as a direction guardrail, not as residual-risk ranking",
        },
        {
            "score_id": "v4_residual_amplitude_axis_scaled_jvp_fro",
            "role": "required_residual_risk_axis",
            "target": "absolute downstream tail-sensitivity magnitude after source-depth adjustment",
            "formula": "log geomean scaled_jvp_tail_drift_fro for the Frobenius matched-head-gain direction",
            "theory_link": "residual risk is not identifiable from a spectral/Frobenius ratio alone when absolute tail sensitivity changes by layer",
            "coefficient_rule": "zero-fit axis; calibration may only choose normalization, not final split coefficients",
            "uses_spent_final_splits": "no",
            "eligible_for_final_p0_claim": "not alone; must be paired with direction axis and validation-frozen transport normalization",
        },
        {
            "score_id": "v4_architecture_transport_tags",
            "role": "registered_stratification_axis",
            "target": "bottleneck/downsample/stage transport mismatch",
            "formula": "stage, block term, fan-in/fan-out, kernel size, and downsample/classifier indicator from parameter shape/name",
            "theory_link": "ResNet50 failure audit shows ratio-only ranking reverses under bottleneck/downsample parameterization",
            "coefficient_rule": "tags are used for stratified reporting and pre-validation normalization only",
            "uses_spent_final_splits": "no",
            "eligible_for_final_p0_claim": "no, unless a validation-frozen scalar transform is committed before final splits",
        },
        {
            "score_id": "early_layer_prior",
            "role": "required_baseline",
            "target": "depth-only residual layer-risk ranking",
            "formula": "log(1 / layer_index)",
            "theory_link": "architecture-depth nuisance baseline",
            "coefficient_rule": "zero-fit",
            "uses_spent_final_splits": "no",
            "eligible_for_final_p0_claim": "no",
        },
        {
            "score_id": "source_observed_drift_positive_control",
            "role": "required_positive_control",
            "target": "upper-bound transfer control",
            "formula": "source observed layer-only matched-head-gain drift residual",
            "theory_link": "checks whether target residual ordering is transferable at all",
            "coefficient_rule": "zero-fit but uses observed source drift",
            "uses_spent_final_splits": "no final-target tuning",
            "eligible_for_final_p0_claim": "no",
        },
    ]


def split_rows() -> list[dict[str, str]]:
    return [
        {
            "split_id": "v4_calibration_cifar100lt_resnet18_source_checkpoint_transfer",
            "role": "calibration_only",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet18 CIFAR stem",
            "class_partition": "head=0-49, tail=50-99",
            "seeds": "10",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "yes for validation-bound normalization only",
            "final_claim_allowed": "no",
            "planned_artifact_prefix": "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction",
        },
        {
            "split_id": "v4_validation_cifar100lt_resnet18_rotated_partition",
            "role": "validation_only",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet18 CIFAR stem",
            "class_partition": "head=0-24,50-74; tail=25-49,75-99",
            "seeds": "10",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "yes for selecting/finalizing the scalar aggregation before final splits",
            "final_claim_allowed": "no",
            "planned_artifact_prefix": "results/e11_condition_score_v4_protocol/validation_cifar100_rotated",
        },
        {
            "split_id": "v4_final_architecture_wide_resnet50_2_cifar100lt",
            "role": "fresh_final_heldout_architecture",
            "dataset": "CIFAR-100-LT",
            "architecture": "WideResNet50-2 CIFAR stem",
            "class_partition": "head=0-49, tail=50-99",
            "seeds": "at least 2 due compute; 3 preferred",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "no",
            "final_claim_allowed": "yes only after v4 score aggregation is committed before this job runs",
            "planned_artifact_prefix": "results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2",
        },
        {
            "split_id": "v4_final_data_cifar10lt_mixed_partition",
            "role": "fresh_final_heldout_data_partition",
            "dataset": "CIFAR-10-LT",
            "architecture": "ResNet18 CIFAR stem",
            "class_partition": "head=0,1,4,7,8; tail=2,3,5,6,9",
            "seeds": "10",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "no",
            "final_claim_allowed": "yes as an unspent CIFAR-10 partition; broad data-family claims need another dataset",
            "planned_artifact_prefix": "results/e11_condition_score_v4_protocol/final_data_cifar10_mixed",
        },
    ]


def gate_rows() -> list[dict[str, str]]:
    return [
        {
            "gate_id": "V4-1-spent-final-quarantine",
            "scope": "all v4 score development",
            "requirement": "The v2/v3 final splits are diagnostic-only and cannot be used for v4 fitting, validation, selection, or final evidence.",
            "pass_condition": "The v4 report cites spent splits only in obstruction sections and all v4 fit/selection rows cite calibration/validation splits.",
        },
        {
            "gate_id": "V4-2-score-axis-separation",
            "scope": "v4 score definition",
            "requirement": "Direction threshold, residual amplitude, and architecture transport tags remain separate until a validation commit freezes any scalar aggregation.",
            "pass_condition": "The score registry includes separate direction, amplitude, and transport rows and no final split is run before the validation commit.",
        },
        {
            "gate_id": "V4-3-validation-freeze-before-final",
            "scope": "unspent final splits",
            "requirement": "Any scalar coefficient, transform, normalization, or score-selection rule is committed before v4 final split jobs run.",
            "pass_condition": "A validation-commit artifact exists in git before final_architecture_wide_resnet50_2 or final_data_cifar10_mixed outputs exist.",
        },
        {
            "gate_id": "V4-4-residual-ranking-success",
            "scope": "P0 predictive-condition claim",
            "requirement": "The final v4 scalar score predicts source-depth-adjusted observed residual layer risk on both unspent final splits.",
            "pass_condition": "Residual Spearman CI lower endpoint is above zero on WideResNet50-2 CIFAR-100-LT and CIFAR-10 mixed-partition final splits.",
        },
        {
            "gate_id": "V4-5-threshold-direction-guardrail",
            "scope": "direction guardrail",
            "requirement": "The direction axis preserves below-one spectral/Frobenius direction on both unspent final splits.",
            "pass_condition": "Below-one threshold accuracy is at least 0.8 on every unspent final split.",
        },
        {
            "gate_id": "V4-6-baselines-and-negative-reporting",
            "scope": "artifact review",
            "requirement": "Every final report includes early-layer, v3 ratio-only, v2 retired, and source-observed controls, including failures.",
            "pass_condition": "Generated score summary has all baseline rows and the gate report preserves failed gates without replacing the target.",
        },
        {
            "gate_id": "V4-7-claim-boundary",
            "scope": "paper wording",
            "requirement": "A CIFAR-10 mixed-partition pass is not described as broad data-family generalization.",
            "pass_condition": "Claim ledger distinguishes data-partition evidence from broader dataset-family evidence.",
        },
    ]


def status_rows() -> list[dict[str, str]]:
    return [
        {
            "item": "v4 spent-final quarantine",
            "status": "registered",
            "evidence": "spent_split_register.csv lists v2 and v3 final splits with forbidden v4 uses",
        },
        {
            "item": "v4 score-axis registry",
            "status": "registered_pending_validation_commit",
            "evidence": "score_axis_registry.csv separates direction ratio, residual amplitude, and architecture transport tags",
        },
        {
            "item": "WideResNet50-2 implementation path",
            "status": "registered",
            "evidence": "CIFAR ResNet builder and checkpoint-prediction parser support model_arch=wide_resnet50_2",
        },
        {
            "item": "v4 final held-out evidence",
            "status": "not_run",
            "evidence": "unspent WideResNet50-2 and CIFAR-10 mixed-partition final Slurm jobs have not been run",
        },
    ]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    spent = pd.DataFrame(spent_split_rows())
    score_axes = pd.DataFrame(score_axis_rows())
    splits = pd.DataFrame(split_rows())
    gates = pd.DataFrame(gate_rows())
    status = pd.DataFrame(status_rows())

    spent.to_csv(RESULT_DIR / "spent_split_register.csv", index=False)
    score_axes.to_csv(RESULT_DIR / "score_axis_registry.csv", index=False)
    splits.to_csv(RESULT_DIR / "unspent_split_registry.csv", index=False)
    gates.to_csv(RESULT_DIR / "acceptance_gates.csv", index=False)
    status.to_csv(RESULT_DIR / "protocol_status.csv", index=False)

    text = f"""# E11 Condition-Score V4 Protocol

This generated v4 protocol starts after the v2 registered held-out failure and the v3 fresh ResNet50 reversal. Its purpose is to prevent another post-hoc scalar score: v4 separates direction ratio, residual amplitude, and architecture transport before any new final split is run.

## Spent Final Split Register

{markdown_table(spent, ["split_id", "source_artifact", "dataset", "architecture", "reason_spent", "allowed_use", "forbidden_use"])}

## Score-Axis Registry

{markdown_table(score_axes, ["score_id", "role", "target", "formula", "theory_link", "coefficient_rule", "uses_spent_final_splits", "eligible_for_final_p0_claim"])}

## Unspent Split Registry

{markdown_table(splits, ["split_id", "role", "dataset", "architecture", "class_partition", "seeds", "warmup_steps", "score_tuning_allowed", "final_claim_allowed", "planned_artifact_prefix"])}

## Acceptance Gates

{markdown_table(gates, ["gate_id", "scope", "requirement", "pass_condition"])}

## Protocol Status

{markdown_table(status, ["item", "status", "evidence"])}

## Theory Consequence

The v3 failure mechanism audit shows that a ratio-only score can pass the below-one direction guardrail while reversing residual layer-risk ranking under ResNet50 bottleneck parameterization. V4 therefore treats a single scalar ratio as insufficient: a final scalar is allowed only after an explicit validation commit explains how residual amplitude and architecture transport enter the score.

## Claim Boundary

Allowed now: v4 is a registered protocol and implementation path for the next predictive-condition attempt.

Blocked now: claiming a v4 predictive condition, because no validation commit or unspent final split evaluation exists yet.

Artifacts:
- [spent_split_register.csv](../{(RESULT_DIR / 'spent_split_register.csv').as_posix()})
- [score_axis_registry.csv](../{(RESULT_DIR / 'score_axis_registry.csv').as_posix()})
- [unspent_split_registry.csv](../{(RESULT_DIR / 'unspent_split_registry.csv').as_posix()})
- [acceptance_gates.csv](../{(RESULT_DIR / 'acceptance_gates.csv').as_posix()})
- [protocol_status.csv](../{(RESULT_DIR / 'protocol_status.csv').as_posix()})
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved condition-score v4 protocol to {OUTPUT_PATH} and {RESULT_DIR}")


if __name__ == "__main__":
    main()
