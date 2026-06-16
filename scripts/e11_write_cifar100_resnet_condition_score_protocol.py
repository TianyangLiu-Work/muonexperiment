from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_DIR = Path("results/e11_cifar100_resnet_condition_score_protocol")
OUTPUT_PATH = Path("discussion/e11_cifar100_resnet_condition_score_protocol.md")


def score_rows() -> list[dict[str, str]]:
    return [
        {
            "score_id": "source_observed_drift_positive_control",
            "role": "positive_control",
            "uses_observed_source_drift": "yes",
            "fit_rule": "No fitting; ranks target layers by measured observed drift at the source checkpoint.",
            "input_features": "source checkpoint observed layer-only matched-head-gain drift",
            "heldout_claim_allowed": "no",
            "reason": "This proves that layer-risk ordering is transferable, but it is not a standalone condition score.",
        },
        {
            "score_id": "early_layer_prior",
            "role": "architecture_prior_baseline",
            "uses_observed_source_drift": "no",
            "fit_rule": "No fitting; risk score is log(1 / layer_index).",
            "input_features": "layer_index",
            "heldout_claim_allowed": "baseline only",
            "reason": "This is the strongest current simple non-condition baseline and must be beaten before claiming a useful condition score.",
        },
        {
            "score_id": "legacy_scaled_jvp_ratio",
            "role": "locked_boundary_baseline",
            "uses_observed_source_drift": "no",
            "fit_rule": "No fitting; use log scaled finite-difference JVP drift ratio from the source checkpoint.",
            "input_features": "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
            "heldout_claim_allowed": "boundary only",
            "reason": "The current audit shows this score preserves the below-one direction but fails raw and residual layer-risk ranking.",
        },
        {
            "score_id": "condition_score_v2_calibrated_residual",
            "role": "primary_candidate",
            "uses_observed_source_drift": "calibration only",
            "fit_rule": "Fit one ridge model on calibration splits only to predict source observed log drift after subtracting the early-layer baseline; freeze coefficients before held-out evaluation.",
            "input_features": "log_gradient_nuclear_rank; log_alignment_ratio; log_step_size_ratio; log_unit_jvp_ratio; log_scaled_jvp_ratio; log_tail_accuracy_before",
            "heldout_claim_allowed": "yes, if all primary gates pass",
            "reason": "This makes the next score explicitly downstream-aware while preventing target-checkpoint or target-architecture leakage.",
        },
        {
            "score_id": "theory_sign_composite",
            "role": "secondary_zero_fit_candidate",
            "uses_observed_source_drift": "no",
            "fit_rule": "No fitting; standardized log_gradient_nuclear_rank plus log_alignment_ratio minus log_scaled_jvp_ratio.",
            "input_features": "mean_gradient_nuclear_rank; mean_alignment_ratio_spectral_over_fro; geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro",
            "heldout_claim_allowed": "secondary only",
            "reason": "This keeps a zero-fit theorem-adjacent candidate beside the calibrated score; it cannot override the primary candidate result.",
        },
    ]


def split_rows() -> list[dict[str, str]]:
    return [
        {
            "split_id": "legacy_resnet18_checkpoint_transfer_boundary",
            "role": "locked_retrospective_audit",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet18 CIFAR stem",
            "checkpoints": "2000/5000/10000 warmup steps",
            "seeds": "10",
            "score_tuning_allowed": "no",
            "target_used_for_tuning": "no",
            "compute_mode": "already generated",
            "planned_artifact_prefix": "results/e11_cifar100_resnet_condition_score_audit",
        },
        {
            "split_id": "calibration_cifar100lt_resnet18_source_folds",
            "role": "calibration_only",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet18 CIFAR stem",
            "checkpoints": "source folds drawn from training checkpoints only",
            "seeds": "at least 10",
            "score_tuning_allowed": "yes",
            "target_used_for_tuning": "no",
            "compute_mode": "GPU via Slurm",
            "planned_artifact_prefix": "results/e11_cifar100_resnet_condition_score_next/calibration",
        },
        {
            "split_id": "primary_heldout_checkpoint_resnet18",
            "role": "primary_heldout_checkpoint",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet18 CIFAR stem",
            "checkpoints": "held-out warmup steps not used for calibration",
            "seeds": "at least 10",
            "score_tuning_allowed": "no",
            "target_used_for_tuning": "no",
            "compute_mode": "GPU via Slurm",
            "planned_artifact_prefix": "results/e11_cifar100_resnet_condition_score_next/heldout_checkpoint",
        },
        {
            "split_id": "primary_heldout_architecture_resnet34",
            "role": "primary_heldout_architecture",
            "dataset": "CIFAR-100-LT",
            "architecture": "ResNet34 CIFAR stem",
            "checkpoints": "matched-quality warmup steps fixed before scoring",
            "seeds": "at least 5",
            "score_tuning_allowed": "no",
            "target_used_for_tuning": "no",
            "compute_mode": "GPU via Slurm",
            "planned_artifact_prefix": "results/e11_cifar100_resnet_condition_score_next/heldout_architecture",
        },
        {
            "split_id": "primary_heldout_data_cifar10lt_resnet18",
            "role": "primary_heldout_data",
            "dataset": "CIFAR-10-LT",
            "architecture": "ResNet18 CIFAR stem",
            "checkpoints": "matched-quality warmup steps fixed before scoring",
            "seeds": "at least 10",
            "score_tuning_allowed": "no",
            "target_used_for_tuning": "no",
            "compute_mode": "GPU via Slurm",
            "planned_artifact_prefix": "results/e11_cifar100_resnet_condition_score_next/heldout_data",
        },
        {
            "split_id": "escalation_imagenetlt_resnet50",
            "role": "optional_scale_escalation",
            "dataset": "ImageNet-LT-style split",
            "architecture": "ResNet50 or pretrained feature backbone",
            "checkpoints": "fixed before scoring",
            "seeds": "at least 3",
            "score_tuning_allowed": "no",
            "target_used_for_tuning": "no",
            "compute_mode": "GPU via Slurm",
            "planned_artifact_prefix": "results/e11_cifar100_resnet_condition_score_next/imagenetlt_escalation",
        },
    ]


def gate_rows() -> list[dict[str, str]]:
    return [
        {
            "gate_id": "G1-no-target-leakage",
            "scope": "all held-out splits",
            "requirement": "No target checkpoint, held-out architecture, or held-out data result can change score features, coefficient signs, hyperparameters, or reporting thresholds.",
            "pass_condition": "Protocol, score registry, split registry, and coefficients are committed before held-out result generation.",
        },
        {
            "gate_id": "G2-primary-residual-prediction",
            "scope": "P0 predictive condition claim",
            "requirement": "The primary candidate must predict residual observed layer risk after removing the early-layer prior.",
            "pass_condition": "Held-out residual Spearman CI lower endpoint is above 0 on checkpoint, architecture, and data splits.",
        },
        {
            "gate_id": "G3-threshold-direction",
            "scope": "matched-head-gain drift direction",
            "requirement": "The score must not only rank risk; it must preserve the below-one direction for candidate layers.",
            "pass_condition": "Below-one threshold accuracy is at least 0.8 on every primary held-out split.",
        },
        {
            "gate_id": "G4-baseline-comparison",
            "scope": "condition-score usefulness",
            "requirement": "The primary candidate must be reported against early_layer_prior, legacy_scaled_jvp_ratio, and source_observed_drift_positive_control.",
            "pass_condition": "If the candidate does not beat early_layer_prior on residual ranking, the paper must call the result a boundary or negative result.",
        },
        {
            "gate_id": "G5-no-performance-overclaim",
            "scope": "paper wording",
            "requirement": "Passing this protocol does not justify final tail-accuracy or practical Muon optimizer claims.",
            "pass_condition": "Paper-facing text keeps the claim to local matched-head-gain tail-example drift unless a separate tuned benchmark passes.",
        },
        {
            "gate_id": "G6-reporting-completeness",
            "scope": "artifact review",
            "requirement": "Every held-out report must include seeds, checkpoints, tail accuracy before update, residual score metrics, threshold metrics, and all failures.",
            "pass_condition": "Generated CSV, figure, and discussion artifacts exist and make e11-check validates them.",
        },
    ]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(score_rows())
    splits = pd.DataFrame(split_rows())
    gates = pd.DataFrame(gate_rows())
    scores.to_csv(RESULT_DIR / "score_registry.csv", index=False)
    splits.to_csv(RESULT_DIR / "split_registry.csv", index=False)
    gates.to_csv(RESULT_DIR / "acceptance_gates.csv", index=False)

    text = f"""# E11 CIFAR-100 ResNet Condition-Score Protocol

This generated protocol pre-registers the next downstream-aware condition-score benchmark for the head-to-tail paper. It exists because the current checkpoint-transfer audit is a boundary result: source-observed drift and the early-layer prior transfer layer-risk ranking, but the legacy scaled-JVP score does not.

## Claim Boundary

Passing this protocol can support a predictive local condition-score claim for matched-head-gain tail-example drift. It cannot support a final long-tail accuracy claim, a practical Muon optimizer-performance claim, or a claim that the synthetic theorem is already a universal real-task predictor.

## Score Registry

{markdown_table(scores, ["score_id", "role", "uses_observed_source_drift", "fit_rule", "input_features", "heldout_claim_allowed", "reason"])}

## Split Registry

{markdown_table(splits, ["split_id", "role", "dataset", "architecture", "checkpoints", "seeds", "score_tuning_allowed", "target_used_for_tuning", "compute_mode", "planned_artifact_prefix"])}

## Acceptance Gates

{markdown_table(gates, ["gate_id", "scope", "requirement", "pass_condition"])}

## Primary Readout

The primary readout is `condition_score_v2_calibrated_residual` on the three primary held-out splits. The paper may call the score predictive only if the held-out residual Spearman confidence interval lower endpoint is above zero, below-one threshold accuracy is at least 0.8, and the comparison against `early_layer_prior` is reported without cherry-picking.

## Leakage Rule

The legacy checkpoint-transfer tables may describe the failure that motivated this protocol, but they must not be used to adjust the primary candidate after this protocol is committed. Any new score after a failed held-out run must receive a new score id and a separate protocol revision.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved condition-score protocol to {OUTPUT_PATH} and {RESULT_DIR}")


if __name__ == "__main__":
    main()
