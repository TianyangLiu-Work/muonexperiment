from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_reviewer_risk_audit.md")


def interval(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def main() -> None:
    synthetic = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    cifar_resnet = pd.read_csv("results/e11_cifar100_resnet_one_step/pair_summary.csv").iloc[0]
    cifar_resnet_rho002 = pd.read_csv("results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv").iloc[0]
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    state_control = pd.read_csv(
        "results/e11_long_tail_muon_state_source_control/summary.csv"
    ).set_index(["state_source", "direction"])
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")

    positive = synthetic[synthetic["setting"].eq("high_head_rank_low_tail_srank")].iloc[0]
    negative = synthetic[synthetic["setting"].eq("low_head_rank_high_tail_srank")].iloc[0]
    layer_1 = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_2 = layerwise[layerwise["layer"].eq(2)].iloc[0]

    risks = pd.DataFrame(
        [
            {
                "reviewer_objection": "The matched-head-gain result only shows lower tail-example logit drift, not better tail performance.",
                "risk_level": "high if overclaimed",
                "current_evidence": (
                    f"One-step squared drift ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}, "
                    f"but tail loss-increase diff={fmt(one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={interval(one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}. "
                    f"CIFAR-100-LT ResNet18 gives squared drift ratio="
                    f"{fmt(cifar_resnet['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')} "
                    f"and tail-loss increase diff spectral-minus-Fro="
                    f"{fmt(cifar_resnet['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={interval(cifar_resnet, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}, "
                    f"with target-head-gain 0.002 squared drift ratio="
                    f"{fmt(cifar_resnet_rho002['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_rho002, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"but tail-accuracy-drop diff CI={interval(cifar_resnet, 'tail_accuracy_drop_diff_ci95_low', 'tail_accuracy_drop_diff_ci95_high')} crosses zero. "
                    f"The small practical training run has lower tail eval loss ratio="
                    f"{fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}, "
                    f"higher tail margin diff={fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}, "
                    f"but tail accuracy diff={fmt(practical_training['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_accuracy_diff_ci95_low', 'final_tail_eval_accuracy_diff_ci95_high')}."
                ),
                "safe_response": "Make function drift the main measured quantity; present practical tail-loss/margin evidence as a small sanity check and separate it from tail accuracy.",
                "remaining_work": "Run retuned long-horizon benchmarks before making performance claims.",
            },
            {
                "reviewer_objection": "The theory is local and uses a matched-head-gain protocol rather than a real optimizer schedule.",
                "risk_level": "medium",
                "current_evidence": (
                    f"8-step forgetting still has lower final squared drift ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}, "
                    f"but final tail loss diff CI={interval(forgetting, 'final_tail_loss_increase_diff_ci95_low', 'final_tail_loss_increase_diff_ci95_high')} crosses zero."
                ),
                "safe_response": "State the result as a local mechanism diagnostic for head-only updates, not a convergence or final-performance theorem.",
                "remaining_work": "Add retuned longer-horizon runs if the manuscript claims training improvement.",
            },
            {
                "reviewer_objection": "The nrank-vs-ssrank condition may be constructed rather than predictive.",
                "risk_level": "medium",
                "current_evidence": (
                    f"Positive boundary: nrank={fmt(positive['mean_head_gradient_nuclear_rank'])} > "
                    f"ssrank={fmt(positive['mean_tail_downstream_aware_stable_rank'])}, squared drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}. "
                    f"Negative boundary: nrank={fmt(negative['mean_head_gradient_nuclear_rank'])} < "
                    f"ssrank={fmt(negative['mean_tail_downstream_aware_stable_rank'])}, squared drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "safe_response": "Call the synthetic result a mechanism sanity check, not a validated predictor for natural tasks.",
                "remaining_work": "Test a pre-specified boundary rule on held-out real tasks or architecture splits.",
            },
            {
                "reviewer_objection": "The layerwise mechanism is not simply lower tail sensitivity.",
                "risk_level": "low if stated clearly",
                "current_evidence": (
                    f"Layer 1 unit-JVP squared drift ratio={fmt(layer_1['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])} while observed squared drift ratio={fmt(layer_1['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}; "
                    f"layer 2 unit-JVP squared drift ratio={fmt(layer_2['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])} while observed squared drift ratio={fmt(layer_2['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "safe_response": "Say spectral/polar reduces drift through smaller matched-head-gain step size, not because unit polar directions are always safer for tail.",
                "remaining_work": "Repeat layerwise diagnostics in deeper ConvNet or Transformer-style models.",
            },
            {
                "reviewer_objection": "The current paper discusses Muon but proves a statement about ideal spectral-gradient/polar directions.",
                "risk_level": "high for Muon-specific claims",
                "current_evidence": (
                    f"The selected-state compatibility check gives polar(M_t) squared drift ratio={fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={interval(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}, "
                    f"and the short-trajectory NS(M_t) squared drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={interval(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')} across "
                    f"{int(practical_bridge.loc['ns_momentum', 'comparisons'])} state-step comparisons. "
                    f"On Fro/GD-generated states, NS(M_t) gives squared drift ratio="
                    f"{fmt(state_control.loc[('fro_gd_trajectory', 'ns_momentum'), 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={interval(state_control.loc[('fro_gd_trajectory', 'ns_momentum')], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "safe_response": "Treat Muon as motivation plus selected-state compatibility checks; mention the Fro/GD state-source control as reducing but not eliminating state-selection concern.",
                "remaining_work": "Add random-checkpoint state controls and real long-tail practical Muon benchmarks before claiming full Muon training behavior.",
            },
            {
                "reviewer_objection": "The empirical evidence is too small for a long-tail learning paper.",
                "risk_level": "medium",
                "current_evidence": (
                    f"Current real-data evidence now includes sklearn digits with {int(one_step['seeds'])} paired seeds and a CIFAR-100-LT ResNet18 one-step diagnostic with "
                    f"{int(cifar_resnet['seeds'])} seeds; the ResNet squared drift ratio is "
                    f"{fmt(cifar_resnet['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}. "
                    f"A smaller-head-gain check at 0.002 gives ratio="
                    f"{fmt(cifar_resnet_rho002['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_rho002, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "safe_response": "Present the manuscript as a theory-and-diagnostic mechanism paper with an architecture robustness check, not a full long-tail benchmark paper.",
                "remaining_work": "Add checkpoint-quality sweeps, ImageNet-LT or iNaturalist-style matched-head-gain diagnostics, and long-horizon practical baselines before claiming benchmark-level generality.",
            },
            {
                "reviewer_objection": "There are too many legacy E11 artifacts and the main claim may be hard to follow.",
                "risk_level": "medium",
                "current_evidence": "README and artifact manifest now separate head-to-tail paper evidence from legacy condition-geometry guardrails.",
                "safe_response": "Keep the paper narrative to core mechanism blocks plus one practical sanity check: synthetic boundary, one-step digits, fixed Muon-style compatibility, trajectory Muon-style compatibility plus state-source control, small practical training, 8-step forgetting, and layerwise JVP. Keep LR sensitivity as supporting robustness evidence.",
                "remaining_work": "Move legacy update-spectrum artifacts to appendix/background or omit them from the main manuscript.",
            },
        ]
    )

    claim_decisions = pd.DataFrame(
        [
            {
                "claim": "Spectral/polar directions can reduce held-out tail-example logit drift at matched head gain.",
                "decision": "main-paper claim",
                "reason": "One-step and 8-step diagnostics support it with paired confidence intervals.",
            },
            {
                "claim": "nrank(G_H) > ssrank(B_T,A_T) is the central mechanism condition.",
                "decision": "theorem-backed mechanism claim",
                "reason": "It follows from the worst-case sensitivity bound and is verified in the controlled positive/negative boundary probe.",
            },
            {
                "claim": "Lower drift implies better tail loss, margin, or accuracy.",
                "decision": "do not claim",
                "reason": "One-step and forgetting diagnostics do not automatically transfer to tail loss/margin, and the practical run still has no tail accuracy gain.",
            },
            {
                "claim": "The paper explains full Muon optimizer behavior.",
                "decision": "do not claim",
                "reason": "Fixed-checkpoint, short-trajectory, and small practical diagnostics are still not a real long-tail benchmark.",
            },
            {
                "claim": "The paper is a full long-tail classification benchmark.",
                "decision": "do not claim",
                "reason": "The CIFAR-100-LT ResNet run is still a local one-step diagnostic, not a tuned long-horizon benchmark.",
            },
        ]
    )

    text = f"""# E11 Reviewer Risk Audit

This generated audit lists likely reviewer objections for the current head-to-tail interference paper and the evidence-backed response. It is meant to keep the manuscript focused on defensible function-drift claims.

## Risk Table

{markdown_table(risks, ["reviewer_objection", "risk_level", "current_evidence", "safe_response", "remaining_work"])}

## Claim Decisions

{markdown_table(claim_decisions, ["claim", "decision", "reason"])}

## Recommended Manuscript Discipline

1. Put head-to-tail interference, matched-head-gain drift, and the nrank-vs-ssrank condition in the main paper.
2. Treat Muon as motivation plus selected-state compatibility checks unless real long-tail practical Muon benchmarks are added.
3. Do not state or imply that lower logit drift automatically improves tail loss, margin, or accuracy.
4. Keep legacy update-spectrum artifacts out of the main claim unless they are explicitly labeled as background guardrails.

## Sources

- [head-to-tail interference note](e11_head_tail_interference.md)
- [long-tailed one-step diagnostic](e11_long_tail_one_step.md)
- [long-tailed Muon-style compatibility diagnostic](e11_long_tail_muon_bridge.md)
- [long-tailed practical-Muon trajectory compatibility](e11_long_tail_practical_muon_bridge.md)
- [long-tailed practical training diagnostic](e11_long_tail_practical_training.md)
- [long-tailed practical training LR sensitivity](e11_long_tail_practical_training_lr_sweep.md)
- [head-only forgetting probe](e11_long_tail_forgetting.md)
- [long-tailed layerwise diagnostic](e11_long_tail_layerwise.md)
- [CIFAR-100-LT ResNet18 one-step diagnostic](e11_cifar100_resnet_one_step.md)
- [CIFAR-100-LT ResNet18 smaller-head-gain check](e11_cifar100_resnet_one_step_rho002.md)
- [artifact manifest](e11_artifact_manifest.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved reviewer risk audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
