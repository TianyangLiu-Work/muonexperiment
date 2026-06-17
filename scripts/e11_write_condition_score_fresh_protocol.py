from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_DIR = Path("results/e11_condition_score_fresh_protocol")
OUTPUT_PATH = Path("discussion/e11_condition_score_fresh_protocol.md")


def quarantine_rows() -> list[dict[str, str]]:
    return [
        {
            "split_id": "spent_primary_heldout_architecture_resnet34",
            "source_artifact": "results/e11_cifar100_resnet_condition_score_next/heldout_architecture",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet34 CIFAR stem",
            "reason_spent": "registered v2 final held-out split; residual-ranking gate failed",
            "allowed_use": "negative evidence, obstruction analysis, sanity-check replication",
            "forbidden_use": "feature selection, coefficient fitting, threshold tuning, final P0 claim evidence for a revised score",
        },
        {
            "split_id": "spent_primary_heldout_data_cifar10lt_resnet18",
            "source_artifact": "results/e11_cifar100_resnet_condition_score_next/heldout_data",
            "dataset": "CIFAR-10-LT",
            "architecture": "ResNet18 CIFAR stem",
            "reason_spent": "registered v2 final held-out split; residual-ranking gate failed",
            "allowed_use": "negative evidence, obstruction analysis, sanity-check replication",
            "forbidden_use": "feature selection, coefficient fitting, threshold tuning, final P0 claim evidence for a revised score",
        },
    ]


def score_freeze_rows() -> list[dict[str, str]]:
    return [
        {
            "score_id": "condition_score_v3_zero_fit_scaled_jvp",
            "role": "primary_fresh_theory_candidate",
            "target": "direction_threshold_and_residual_layer_ranking",
            "formula": "log(geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro)",
            "theory_link": "finite-difference estimate of downstream tail drift under matched head-gain scaling",
            "coefficient_rule": "zero-fit; coefficient on log scaled-JVP ratio is fixed to +1 before fresh held-outs",
            "uses_spent_heldouts": "no",
            "eligible_for_fresh_p0_claim": "yes, only on fresh splits and only if all gates pass",
        },
        {
            "score_id": "condition_score_v3_nested_jvp_residual",
            "role": "secondary_nested_calibration_candidate",
            "target": "residual_layer_ranking",
            "formula": "calibration-only linear score over log_scaled_jvp_ratio, log_unit_jvp_ratio, log_gradient_nuclear_rank, and log_alignment_ratio after depth residualization",
            "theory_link": "downstream JVP terms estimate tail sensitivity; rank and alignment terms are theorem-adjacent nuisance modifiers",
            "coefficient_rule": "fit only on calibration families, select only on validation families, then commit coefficients before final held-outs",
            "uses_spent_heldouts": "no",
            "eligible_for_fresh_p0_claim": "not until coefficients and validation choice are committed before final held-outs",
        },
        {
            "score_id": "early_layer_prior",
            "role": "required_baseline",
            "target": "residual_layer_ranking",
            "formula": "log(1 / layer_index)",
            "theory_link": "architecture-depth nuisance baseline, not a downstream-aware condition",
            "coefficient_rule": "zero-fit",
            "uses_spent_heldouts": "no",
            "eligible_for_fresh_p0_claim": "no",
        },
        {
            "score_id": "source_observed_drift_positive_control",
            "role": "required_positive_control",
            "target": "residual_layer_ranking",
            "formula": "source observed layer-only matched-head-gain drift residual",
            "theory_link": "upper-bound transfer control, not a measurable pre-target condition",
            "coefficient_rule": "zero-fit but uses observed source drift",
            "uses_spent_heldouts": "no final-target tuning",
            "eligible_for_fresh_p0_claim": "no",
        },
    ]


def fresh_split_rows() -> list[dict[str, str]]:
    return [
        {
            "split_id": "fresh_calibration_cifar100lt_resnet18_source_folds",
            "role": "calibration_only",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet18 CIFAR stem",
            "class_partition": "head=0-49, tail=50-99",
            "seeds": "10",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "yes for nested candidate only",
            "final_claim_allowed": "no",
            "planned_artifact_prefix": "results/e11_condition_score_fresh_protocol/calibration_resnet18",
        },
        {
            "split_id": "fresh_validation_cifar100lt_resnet18_alt_partition",
            "role": "validation_only",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet18 CIFAR stem",
            "class_partition": "head=even classes, tail=odd classes",
            "seeds": "10",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "yes for score selection, no coefficient refit after selection",
            "final_claim_allowed": "no",
            "planned_artifact_prefix": "results/e11_condition_score_fresh_protocol/validation_cifar100_alt",
        },
        {
            "split_id": "fresh_final_architecture_resnet50_cifar100lt",
            "role": "fresh_final_heldout_architecture",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet50 CIFAR stem",
            "class_partition": "head=0-49, tail=50-99",
            "seeds": "at least 3 due compute; 5 preferred",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "no",
            "final_claim_allowed": "yes if paired with fresh data split and all gates pass",
            "planned_artifact_prefix": "results/e11_condition_score_fresh_protocol/fresh_architecture_resnet50",
        },
        {
            "split_id": "fresh_final_data_cifar10lt_alt_partition",
            "role": "fresh_final_heldout_data_partition",
            "dataset": "CIFAR-10-LT",
            "architecture": "ResNet18 CIFAR stem",
            "class_partition": "head=0,2,4,6,8; tail=1,3,5,7,9",
            "seeds": "10",
            "warmup_steps": "2000/5000/10000",
            "score_tuning_allowed": "no",
            "final_claim_allowed": "yes as a fresh partition stress test; broader data-family claims still need another dataset",
            "planned_artifact_prefix": "results/e11_condition_score_fresh_protocol/fresh_data_cifar10_alt",
        },
    ]


def gate_rows() -> list[dict[str, str]]:
    return [
        {
            "gate_id": "F1-quarantine-enforced",
            "scope": "all revised scores",
            "requirement": "Spent ResNet34 and CIFAR-10-LT held-outs are not used for fitting, selecting, or tuning revised scores.",
            "pass_condition": "Quarantine register is present and future score reports cite no spent split as calibration, validation, or final evidence.",
        },
        {
            "gate_id": "F2-score-freeze-before-final",
            "scope": "fresh final held-outs",
            "requirement": "Score formula, feature transforms, coefficient signs, and learned coefficients are committed before final held-out jobs run.",
            "pass_condition": "Score-freeze registry and any learned coefficient table exist in git before fresh final outputs are generated.",
        },
        {
            "gate_id": "F3-target-separation",
            "scope": "claim wording",
            "requirement": "Direction-threshold and residual-ranking targets remain separate.",
            "pass_condition": "Gate report has separate threshold-accuracy and residual-Spearman rows; a threshold pass cannot rescue a ranking fail.",
        },
        {
            "gate_id": "F4-residual-ranking-success",
            "scope": "fresh P0 predictive-condition claim",
            "requirement": "Primary score predicts source-depth-adjusted observed residual layer risk on both fresh final splits.",
            "pass_condition": "Residual Spearman CI lower endpoint is above zero on fresh ResNet50 architecture and fresh CIFAR-10 alternate partition splits.",
        },
        {
            "gate_id": "F5-threshold-direction-success",
            "scope": "fresh direction guardrail",
            "requirement": "Primary score preserves below-one direction on both fresh final splits.",
            "pass_condition": "Below-one threshold accuracy is at least 0.8 on every fresh final split.",
        },
        {
            "gate_id": "F6-baselines-reported",
            "scope": "artifact review",
            "requirement": "Every fresh report includes early-layer prior, source-observed positive control, and v2/legacy scaled-JVP baselines.",
            "pass_condition": "Generated score summary has all required baseline rows, including negative or baseline-dominated outcomes.",
        },
    ]


def status_rows() -> list[dict[str, str]]:
    return [
        {
            "item": "spent-heldout quarantine",
            "status": "registered",
            "evidence": "quarantine_register.csv lists both failed v2 final held-out splits and forbidden uses",
        },
        {
            "item": "theory-derived score freeze registry",
            "status": "registered",
            "evidence": "score_freeze_registry.csv defines zero-fit scaled-JVP primary candidate and nested secondary candidate",
        },
        {
            "item": "fresh ResNet50 implementation path",
            "status": "registered",
            "evidence": "CIFAR ResNet model builder and checkpoint-prediction parser support model_arch=resnet50",
        },
        {
            "item": "fresh final held-out evidence",
            "status": "missing",
            "evidence": "fresh ResNet50 and fresh CIFAR-10 alternate-partition Slurm jobs have not been run",
        },
    ]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    quarantine = pd.DataFrame(quarantine_rows())
    score_freeze = pd.DataFrame(score_freeze_rows())
    splits = pd.DataFrame(fresh_split_rows())
    gates = pd.DataFrame(gate_rows())
    status = pd.DataFrame(status_rows())

    quarantine.to_csv(RESULT_DIR / "quarantine_register.csv", index=False)
    score_freeze.to_csv(RESULT_DIR / "score_freeze_registry.csv", index=False)
    splits.to_csv(RESULT_DIR / "fresh_split_registry.csv", index=False)
    gates.to_csv(RESULT_DIR / "acceptance_gates.csv", index=False)
    status.to_csv(RESULT_DIR / "protocol_status.csv", index=False)

    text = f"""# E11 Fresh Condition-Score Protocol

This generated protocol revision starts after the registered v2 held-out failure. Its purpose is not to rescue `condition_score_v2_calibrated_residual`; it quarantines the failed held-outs, freezes the next score definitions, and names fresh final splits that can support or reject a revised P0 predictive-condition claim.

## Quarantine Register

{markdown_table(quarantine, ["split_id", "source_artifact", "dataset", "architecture", "reason_spent", "allowed_use", "forbidden_use"])}

## Score Freeze Registry

{markdown_table(score_freeze, ["score_id", "role", "target", "formula", "theory_link", "coefficient_rule", "uses_spent_heldouts", "eligible_for_fresh_p0_claim"])}

## Fresh Split Registry

{markdown_table(splits, ["split_id", "role", "dataset", "architecture", "class_partition", "seeds", "warmup_steps", "score_tuning_allowed", "final_claim_allowed", "planned_artifact_prefix"])}

## Acceptance Gates

{markdown_table(gates, ["gate_id", "scope", "requirement", "pass_condition"])}

## Protocol Status

{markdown_table(status, ["item", "status", "evidence"])}

## Claim Boundary

The zero-fit scaled-JVP score is a theory-derived candidate because it directly measures the downstream finite-difference tail-drift ratio under matched head-gain scaling. It is not a positive result until the fresh ResNet50 architecture split and the fresh CIFAR-10 alternate-partition split exist and pass the residual-ranking and threshold gates. The spent ResNet34 and original CIFAR-10 held-outs may appear only as negative evidence and must not be used for revised score fitting or final claims.

Artifacts:
- [quarantine_register.csv](../{(RESULT_DIR / 'quarantine_register.csv').as_posix()})
- [score_freeze_registry.csv](../{(RESULT_DIR / 'score_freeze_registry.csv').as_posix()})
- [fresh_split_registry.csv](../{(RESULT_DIR / 'fresh_split_registry.csv').as_posix()})
- [acceptance_gates.csv](../{(RESULT_DIR / 'acceptance_gates.csv').as_posix()})
- [protocol_status.csv](../{(RESULT_DIR / 'protocol_status.csv').as_posix()})
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved fresh condition-score protocol to {OUTPUT_PATH} and {RESULT_DIR}")


if __name__ == "__main__":
    main()
