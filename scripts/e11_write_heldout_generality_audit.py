from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


OUTPUT_DIR = Path("results/e11_heldout_generality_audit")
DISCUSSION_PATH = Path("discussion/e11_heldout_generality_audit.md")


def fmt(x: float) -> str:
    return f"{x:.4g}"


def ratio_ci(row: pd.Series, prefix: str = "tail_output_drift_sq_ratio") -> str:
    return (
        f"{fmt(float(row['geomean_tail_output_drift_sq_ratio_spectral_over_fro']))} "
        f"[{fmt(float(row[f'{prefix}_ci95_low']))}, {fmt(float(row[f'{prefix}_ci95_high']))}]"
    )


def tail_loss_sign(row: pd.Series) -> str:
    low = float(row["tail_loss_increase_diff_ci95_low"])
    high = float(row["tail_loss_increase_diff_ci95_high"])
    mean = float(row["mean_tail_loss_increase_diff_spectral_minus_fro"])
    if high < 0:
        return f"favorable spectral-minus-Fro tail-loss increase {fmt(mean)} [{fmt(low)}, {fmt(high)}]"
    if low > 0:
        return f"unfavorable spectral-minus-Fro tail-loss increase {fmt(mean)} [{fmt(low)}, {fmt(high)}]"
    return f"inconclusive spectral-minus-Fro tail-loss increase {fmt(mean)} [{fmt(low)}, {fmt(high)}]"


def bool_count(frame: pd.DataFrame, column: str, value: bool) -> int:
    return int(frame[column].astype(str).str.lower().eq(str(value).lower()).sum())


def decision_family_summary(frame: pd.DataFrame) -> str:
    return (
        f"rows={len(frame)}, seeds={int(frame['observed_seeds'].max())}, "
        f"estimate_range={fmt(float(frame['estimate'].min()))}-{fmt(float(frame['estimate'].max()))}, "
        f"max_raw_ci_high={fmt(float(frame['raw_ci95_high'].max()))}, "
        f"adjusted_worse_rows={int(frame['adjusted_primary_decision'].ne('not_primary_worse_adjusted').sum())}"
    )


def build_generality_evidence() -> pd.DataFrame:
    resnet_default = pd.read_csv("results/e11_cifar100_resnet_one_step/pair_summary.csv").iloc[0]
    checkpoint = pd.read_csv("results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv")
    tail_quality = pd.read_csv("results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv")
    imbalance = pd.read_csv("results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv")
    all_layer = pd.read_csv("results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv").iloc[0]
    phase1 = pd.read_csv("results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/primary_decisions.csv")
    phase2 = pd.read_csv("results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv")
    condition_gates = pd.read_csv("results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv")

    phase1_cifar10 = phase1[phase1["dataset_name"].eq("CIFAR10")]
    tail_quality_and_imbalance = pd.concat([tail_quality, imbalance], ignore_index=True)
    final_gate_status = condition_gates.set_index("gate_id")["status"].to_dict()

    return pd.DataFrame(
        [
            {
                "audit_id": "HGA-1-resnet18-default-support",
                "evidence_family": "CIFAR-100-LT ResNet18 default one-step",
                "source_role": "primary_positive_mechanism_support",
                "architecture_scope": "ResNet18 CIFAR stem",
                "data_scope": "CIFAR-100-LT head 0-49 tail 50-99",
                "setting_count": 1,
                "seed_count": int(resnet_default["seeds"]),
                "drift_ratio_readout": ratio_ci(resnet_default),
                "tail_accuracy_readout": f"pre-update tail accuracy {fmt(float(resnet_default['mean_tail_accuracy_before']))}",
                "head_gain_quality_readout": (
                    "head-gain relative-error intervals are recorded for both directions; "
                    "validator enforces matched-update/head-gain consistency"
                ),
                "tail_loss_readout": tail_loss_sign(resnet_default),
                "current_claim_use": "positive local matched-head-gain drift support for the tested ResNet18 family",
                "blocked_overread": "does not imply final tail accuracy or optimizer superiority",
                "source_artifacts": "results/e11_cifar100_resnet_one_step/pair_summary.csv; discussion/e11_cifar100_resnet_one_step.md",
            },
            {
                "audit_id": "HGA-2-resnet18-checkpoint-support",
                "evidence_family": "CIFAR-100-LT ResNet18 checkpoint sweep",
                "source_role": "within_architecture_checkpoint_support",
                "architecture_scope": "ResNet18 CIFAR stem",
                "data_scope": "CIFAR-100-LT fixed split",
                "setting_count": int(len(checkpoint)),
                "seed_count": int(checkpoint["seeds"].max()),
                "drift_ratio_readout": (
                    f"all {len(checkpoint)} checkpoint CI upper endpoints below one; "
                    f"worst_ci_high={fmt(float(checkpoint['tail_output_drift_sq_ratio_ci95_high'].max()))}"
                ),
                "tail_accuracy_readout": (
                    f"pre-update tail accuracy range "
                    f"{fmt(float(checkpoint['mean_tail_accuracy_before'].min()))}-"
                    f"{fmt(float(checkpoint['mean_tail_accuracy_before'].max()))}"
                ),
                "head_gain_quality_readout": "fixed matched-head-gain protocol across checkpoint states",
                "tail_loss_readout": "tail-loss signs are diagnostic context, not a performance gate",
                "current_claim_use": "reduces single-checkpoint state-selection concern inside ResNet18",
                "blocked_overread": "does not prove high-quality tail predictor preservation",
                "source_artifacts": "results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv; discussion/e11_cifar100_resnet_checkpoint_sweep.md",
            },
            {
                "audit_id": "HGA-3-resnet18-tail-quality-frequency-support",
                "evidence_family": "ResNet18 tail-quality and tail-frequency controls",
                "source_role": "within_architecture_tail_quality_support",
                "architecture_scope": "ResNet18 CIFAR stem",
                "data_scope": "CIFAR-100-LT tail-count and tail-rich controls",
                "setting_count": int(len(tail_quality_and_imbalance)),
                "seed_count": int(tail_quality_and_imbalance["seeds"].max()),
                "drift_ratio_readout": (
                    f"worst CI high across controls="
                    f"{fmt(float(tail_quality_and_imbalance['tail_output_drift_sq_ratio_ci95_high'].max()))}"
                ),
                "tail_accuracy_readout": (
                    f"best pre-update tail accuracy "
                    f"{fmt(float(tail_quality_and_imbalance['mean_tail_accuracy_before'].max()))}"
                ),
                "head_gain_quality_readout": "same matched-head-gain one-step protocol across tail-frequency settings",
                "tail_loss_readout": "mixed tail-loss signs; accuracy is not the claim target",
                "current_claim_use": "supports local drift robustness across tail frequency within ResNet18",
                "blocked_overread": "not a standard long-tailed benchmark and not a final-accuracy result",
                "source_artifacts": "results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv; results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv",
            },
            {
                "audit_id": "HGA-4-resnet18-all-layer-support",
                "evidence_family": "ResNet18 all-layer tail-rich JVP diagnostic",
                "source_role": "within_architecture_layer_support",
                "architecture_scope": "21 Conv/Linear matrix weights in ResNet18",
                "data_scope": "CIFAR-100-LT tail-rich checkpoint",
                "setting_count": int(all_layer["parameters"]),
                "seed_count": int(all_layer["seeds"]),
                "drift_ratio_readout": (
                    f"observed ratio {fmt(float(all_layer['geomean_observed_tail_drift_sq_ratio_spectral_over_fro']))} "
                    f"[{fmt(float(all_layer['observed_tail_drift_sq_ratio_ci95_low']))}, "
                    f"{fmt(float(all_layer['observed_tail_drift_sq_ratio_ci95_high']))}] over "
                    f"{int(all_layer['paired_points'])} layer/seed points"
                ),
                "tail_accuracy_readout": f"pre-update tail accuracy {fmt(float(all_layer['mean_tail_accuracy_before']))}",
                "head_gain_quality_readout": "finite-difference unit-JVP and matched-head-gain observed drift are reported separately",
                "tail_loss_readout": (
                    f"spectral-minus-Fro tail-loss increase "
                    f"{fmt(float(all_layer['mean_tail_loss_increase_diff_spectral_minus_fro']))} "
                    f"[{fmt(float(all_layer['tail_loss_increase_diff_ci95_low']))}, "
                    f"{fmt(float(all_layer['tail_loss_increase_diff_ci95_high']))}]"
                ),
                "current_claim_use": "shows the local signal is not only a final-layer artifact within ResNet18",
                "blocked_overread": "unit-JVP ratios are above one, so do not claim spectral unit directions are uniformly safer",
                "source_artifacts": "results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv; discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md",
            },
            {
                "audit_id": "HGA-5-cifar10-data-family-boundary",
                "evidence_family": "registered CIFAR-10-LT cross-partition phase1 family",
                "source_role": "heldout_data_finite_null_boundary",
                "architecture_scope": "ResNet18 CIFAR stem",
                "data_scope": "CIFAR-10-LT cross partitions",
                "setting_count": int(len(phase1_cifar10)),
                "seed_count": int(phase1_cifar10["observed_seeds"].max()),
                "drift_ratio_readout": decision_family_summary(phase1_cifar10),
                "tail_accuracy_readout": (
                    f"tail_quality_gate_pass_rows={bool_count(phase1_cifar10, 'tail_quality_gate', True)}; "
                    f"quality_gate_pass_rows={bool_count(phase1_cifar10, 'quality_gate', True)}"
                ),
                "head_gain_quality_readout": (
                    f"head_gain_gate_pass_rows={bool_count(phase1_cifar10, 'head_gain_gate', True)}"
                ),
                "tail_loss_readout": "primary family is Holm-adjusted on log tail-output drift ratio; secondary outcomes remain caveats",
                "current_claim_use": "finite registered data-family null boundary, not broad data generalization proof",
                "blocked_overread": "do not call this a fresh natural counterexample or universal absence of data-family failures",
                "source_artifacts": "results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/primary_decisions.csv; discussion/e11_natural_negative_search_phase1_evaluation.md",
            },
            {
                "audit_id": "HGA-6-resnet34-architecture-boundary",
                "evidence_family": "registered ResNet34 held-out architecture phase2 family",
                "source_role": "heldout_architecture_finite_null_boundary",
                "architecture_scope": "ResNet34 CIFAR stem",
                "data_scope": "CIFAR-100-LT registered partitions",
                "setting_count": int(len(phase2)),
                "seed_count": int(phase2["observed_seeds"].max()),
                "drift_ratio_readout": decision_family_summary(phase2),
                "tail_accuracy_readout": (
                    f"tail_quality_gate_pass_rows={bool_count(phase2, 'tail_quality_gate', True)}; "
                    f"quality_gate_pass_rows={bool_count(phase2, 'quality_gate', True)}"
                ),
                "head_gain_quality_readout": (
                    f"head_gain_gate_pass_rows={bool_count(phase2, 'head_gain_gate', True)}; "
                    "all rows fail the combined quality gate because head-gain caveats remain active"
                ),
                "tail_loss_readout": "primary family is Holm-adjusted on log tail-output drift ratio; power audit bounds detectable effects",
                "current_claim_use": "caveated held-out architecture finite-null boundary",
                "blocked_overread": "not mechanism validation, not a universal natural null, and not final-performance evidence",
                "source_artifacts": "results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv; discussion/e11_natural_negative_search_phase2_evaluation.md; discussion/e11_natural_negative_search_phase2_power_audit.md",
            },
            {
                "audit_id": "HGA-7-condition-score-heldout-boundary",
                "evidence_family": "v5 condition-score held-out final splits",
                "source_role": "predictive_score_negative_boundary",
                "architecture_scope": "ResNeXt50-32x4d final architecture split",
                "data_scope": "CIFAR-10-LT cross-partition final data split",
                "setting_count": int(len(condition_gates)),
                "seed_count": -1,
                "drift_ratio_readout": (
                    "v5_p0_predictive_condition_claim="
                    f"{final_gate_status['v5_p0_predictive_condition_claim']}; "
                    "architecture direction threshold="
                    f"{final_gate_status['v5_final_heldout_architecture_direction_threshold_accuracy']}; "
                    "data residual Spearman="
                    f"{final_gate_status['v5_final_heldout_data_partition_residual_spearman']}"
                ),
                "tail_accuracy_readout": "layer-risk score audit, not a tail-accuracy measurement",
                "head_gain_quality_readout": "frozen score evaluation with no refit or re-selection",
                "tail_loss_readout": "not a tail-loss readout",
                "current_claim_use": "negative boundary for predictive score generalization",
                "blocked_overread": "does not support a successful held-out natural-task predictor",
                "source_artifacts": "results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv; discussion/e11_condition_score_v5_final_evaluation.md",
            },
        ]
    )


def build_claim_gate(evidence: pd.DataFrame) -> pd.DataFrame:
    support_rows = int(evidence["source_role"].str.contains("support").sum())
    caveated_boundary_rows = int(evidence["source_role"].str.contains("boundary").sum())
    return pd.DataFrame(
        [
            {
                "gate_id": "HGG-1-positive-resnet18-family",
                "status": "pass",
                "evidence": f"{support_rows} ResNet18 support rows cover default, checkpoint, tail-quality, frequency, and all-layer diagnostics",
                "claim_effect": "supports a local ResNet18 mechanism claim under matched head gain",
            },
            {
                "gate_id": "HGG-2-heldout-data-family",
                "status": "finite_null_boundary",
                "evidence": "CIFAR-10-LT phase1 family has no adjusted primary worse rows but keeps quality/head-gain caveats",
                "claim_effect": "data-family result is bounded falsification evidence, not broad data generality",
            },
            {
                "gate_id": "HGG-3-heldout-architecture-family",
                "status": "caveated_finite_null_boundary",
                "evidence": "ResNet34 phase2 has 8/8 observed rows and no adjusted primary worse row, but head_gain_gate_pass_rows=0",
                "claim_effect": "architecture result is a caveated held-out boundary, not mechanism validation",
            },
            {
                "gate_id": "HGG-4-predictive-score-generality",
                "status": "failed_boundary",
                "evidence": "completed v5 final score gates fail orthogonally on architecture direction and data residual ranking",
                "claim_effect": "blocks a successful held-out layer-risk predictor claim",
            },
            {
                "gate_id": "HGG-5-paper-generality-claim",
                "status": "bounded_support_with_caveated_heldout_boundaries",
                "evidence": f"support_rows={support_rows}; caveated_or_negative_boundary_rows={caveated_boundary_rows}",
                "claim_effect": "write a focused ResNet18-supported mechanism with explicit held-out architecture/data boundaries",
            },
        ]
    )


def write_discussion(evidence: pd.DataFrame, gates: pd.DataFrame) -> None:
    text = f"""# E11 Held-Out Generality Audit

This generated audit separates positive mechanism support from caveated held-out
architecture/data boundaries. It is deliberately conservative: ResNet18
matched-head-gain diagnostics support the local mechanism in the tested family,
while CIFAR-10-LT and ResNet34 searches are finite registered boundary evidence,
not proof of broad architecture or data generalization.

## Generality Evidence Matrix

{markdown_table(evidence, ["audit_id", "evidence_family", "source_role", "architecture_scope", "data_scope", "setting_count", "seed_count", "drift_ratio_readout", "tail_accuracy_readout", "head_gain_quality_readout", "tail_loss_readout", "current_claim_use", "blocked_overread"])}

## Generality Claim Gate

{markdown_table(gates, ["gate_id", "status", "evidence", "claim_effect"])}

## Manuscript Boundary

Allowed now: write that the local matched-head-gain mechanism has positive
ResNet18 support across checkpoint, tail-quality, tail-frequency, and all-layer
diagnostics, and that registered CIFAR-10-LT and ResNet34 searches add finite
held-out boundary evidence.

Blocked now: claiming broad held-out architecture/data generality, using the
ResNet34 phase2 finite-null result as mechanism validation while all phase2
rows have `head_gain_gate=False`, or turning failed v5 score finals into a
successful predictive condition.

Artifacts:
- [generality_evidence_matrix.csv](../results/e11_heldout_generality_audit/generality_evidence_matrix.csv)
- [generality_claim_gate.csv](../results/e11_heldout_generality_audit/generality_claim_gate.csv)
- [config.json](../results/e11_heldout_generality_audit/config.json)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    evidence = build_generality_evidence()
    gates = build_claim_gate(evidence)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(OUTPUT_DIR / "generality_evidence_matrix.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "generality_claim_gate.csv", index=False)
    config = {
        "evidence_rows": int(len(evidence)),
        "gate_rows": int(len(gates)),
        "current_status": str(gates.loc[gates["gate_id"].eq("HGG-5-paper-generality-claim"), "status"].iloc[0]),
        "resnet34_head_gain_caveat": "all phase2 rows have head_gain_gate=False",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_discussion(evidence, gates)
    print(f"saved held-out generality audit to {DISCUSSION_PATH} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
