from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_paper_readiness_audit.md")


def interval(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def paired_geomean_ratio(step_metrics: pd.DataFrame, numerator: str, denominator: str, column: str) -> float:
    wide = step_metrics.pivot(index="seed", columns="geometry", values=column)
    return math.exp((wide[numerator] / wide[denominator]).map(math.log).mean())


def main() -> None:
    synthetic = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv")
    one_step_steps = pd.read_csv("results/e11_long_tail_one_step/step_metrics.csv")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    cifar_resnet = pd.read_csv("results/e11_cifar100_resnet_one_step/pair_summary.csv").iloc[0]
    cifar_resnet_rho002 = pd.read_csv("results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv").iloc[0]
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    state_control = pd.read_csv(
        "results/e11_long_tail_muon_state_source_control/summary.csv"
    ).set_index(["state_source", "direction"])
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    practical_lr_sweep = pd.read_csv("results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv")
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")

    positive = synthetic[synthetic["setting"].eq("high_head_rank_low_tail_srank")].iloc[0]
    negative = synthetic[synthetic["setting"].eq("low_head_rank_high_tail_srank")].iloc[0]
    layer_1 = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_2 = layerwise[layerwise["layer"].eq(2)].iloc[0]
    lr_sweep = practical_lr_sweep.assign(muon_lr_rounded=practical_lr_sweep["muon_lr"].round(3)).set_index("muon_lr_rounded")
    alignment_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "alignment")
    update_fro_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "update_fro_norm")
    update_op_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "update_op_norm")

    claim_status = pd.DataFrame(
        [
            {
                "claim": "Matched-head-gain spectral/polar directions reduce held-out tail-example logit drift in the tested long-tail diagnostics.",
                "readiness": "current core empirical claim",
                "evidence": (
                    f"One-step digits squared tail-example logit drift ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"head-alignment ratio={fmt(alignment_ratio)}, "
                    f"Frobenius-norm ratio={fmt(update_fro_ratio)}, "
                    f"operator-norm ratio={fmt(update_op_ratio)}; "
                    f"8-step final ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}; "
                    f"drift-area ratio={fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"CI={interval(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}. "
                    f"CIFAR-100-LT ResNet18 squared drift ratio={fmt(cifar_resnet['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')} "
                    f"with spectral-lower fraction={fmt(cifar_resnet['spectral_less_tail_output_drift_fraction'])} over "
                    f"{int(cifar_resnet['seeds'])} seeds; the target-head-gain 0.002 check gives squared drift ratio="
                    f"{fmt(cifar_resnet_rho002['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_rho002, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "why_it_is_ready": "It is measured under the paper's matched-head-gain protocol with paired confidence intervals and explicit norm-specific scaling readouts.",
                "remaining_risk": "The strongest new evidence is still a local diagnostic, not a retuned long-horizon long-tail benchmark.",
            },
            {
                "claim": "The matched-head-gain drift readout survives a more appropriate CIFAR-100-LT ResNet architecture.",
                "readiness": "supporting robustness check",
                "evidence": (
                    f"ResNet18 with CIFAR stem, fixed BatchNorm at measurement time, and Conv/Linear matrix-weight interventions gives "
                    f"squared tail-example logit drift ratio={fmt(cifar_resnet['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"tail-loss increase diff spectral-minus-Fro={fmt(cifar_resnet['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={interval(cifar_resnet, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}; "
                    f"tail-accuracy-drop diff={fmt(cifar_resnet['mean_tail_accuracy_drop_diff_spectral_minus_fro'])} "
                    f"CI={interval(cifar_resnet, 'tail_accuracy_drop_diff_ci95_low', 'tail_accuracy_drop_diff_ci95_high')}; "
                    f"the target-head-gain 0.002 check gives squared drift ratio="
                    f"{fmt(cifar_resnet_rho002['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_rho002, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "why_it_is_ready": "It was rerun on GPU with a convolutional architecture and CIFAR-100-LT split, addressing the pure two-layer-MLP concern.",
                "remaining_risk": "Still a one-step local intervention; BatchNorm/bias are frozen during the diagnostic and this is not a full practical Muon run.",
            },
            {
                "claim": "The condition nrank(G_H) > ssrank(B_T,A_T) is a useful mechanism boundary.",
                "readiness": "controlled-theory sanity check",
                "evidence": (
                    f"Positive setting: nrank={fmt(positive['mean_head_gradient_nuclear_rank'])}, "
                    f"ssrank={fmt(positive['mean_tail_downstream_aware_stable_rank'])}, "
                    f"squared drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}. "
                    f"Negative setting: nrank={fmt(negative['mean_head_gradient_nuclear_rank'])}, "
                    f"ssrank={fmt(negative['mean_tail_downstream_aware_stable_rank'])}, "
                    f"squared drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "why_it_is_ready": "The synthetic construction flips the theory inequality and the observed tail drift direction flips with it.",
                "remaining_risk": "This is not yet a held-out predictor for natural tasks or larger architectures.",
            },
            {
                "claim": "The mechanism is norm-specific scaled head-gain efficiency, not lower unit-direction tail sensitivity.",
                "readiness": "current mechanism explanation",
                "evidence": (
                    f"Layer 1 unit-JVP squared drift ratio={fmt(layer_1['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"scaled/observed squared drift ratios={fmt(layer_1['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_1['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}. "
                    f"Layer 2 unit-JVP squared drift ratio={fmt(layer_2['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"scaled/observed squared drift ratios={fmt(layer_2['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_2['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "why_it_is_ready": "Both layers have unit-JVP squared drift ratio above 1 but scaled and observed squared drift ratios below 1; the one-step diagnostic also shows smaller operator norm but larger Frobenius norm after matching head gain.",
                "remaining_risk": "The ResNet run checks architecture-level drift, but the detailed unit-JVP/scaled-JVP decomposition is still only a two-layer digits MLP.",
            },
            {
                "claim": "Lower tail-example logit drift implies better tail loss, margin, or accuracy.",
                "readiness": "not supported",
                "evidence": (
                    f"One-step tail loss-increase diff spectral-minus-fro={fmt(one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={interval(one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}; "
                    f"8-step final tail loss diff={fmt(forgetting['mean_final_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={interval(forgetting, 'final_tail_loss_increase_diff_ci95_low', 'final_tail_loss_increase_diff_ci95_high')}."
                ),
                "why_it_is_ready": "The measured drift result does not transfer cleanly to tail loss/margin outcomes.",
                "remaining_risk": "Any final-performance claim requires retuned, longer-horizon training.",
            },
            {
                "claim": "The ideal polar direction has selected-state compatibility with Muon-style momentum directions.",
                "readiness": "supported compatibility check, not final-performance claim",
                "evidence": (
                    f"polar(M_t) squared drift ratio vs Fro/GD={fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={interval(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"short-trajectory NS(M_t) squared drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={interval(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')} "
                    f"over {int(practical_bridge.loc['ns_momentum', 'comparisons'])} state-step comparisons; "
                    f"Fro/GD-state NS(M_t) squared drift ratio="
                    f"{fmt(state_control.loc[('fro_gd_trajectory', 'ns_momentum'), 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={interval(state_control.loc[('fro_gd_trajectory', 'ns_momentum')], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "why_it_is_ready": "The compatibility check is measured at a fixed checkpoint, along a short practical NS-Muon-style trajectory, and on a Fro/GD trajectory state-source control.",
                "remaining_risk": "Muon-style momentum evidence is still local and mostly on scikit-learn digits; the new CIFAR-100-LT ResNet run is an ideal polar/spectral intervention, not a Muon optimizer run.",
            },
            {
                "claim": "A small practical NS-Muon-style training loop has lower measured drift/loss in this fixed diagnostic.",
                "readiness": "supported as sanity check, not benchmark",
                "evidence": (
                    f"final train loss ratio Muon/Adam={fmt(practical_training['geomean_final_train_loss_ratio_muon_over_adam'])} "
                    f"CI={interval(practical_training, 'final_train_loss_ratio_ci95_low', 'final_train_loss_ratio_ci95_high')}; "
                    f"tail eval loss ratio={fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}; "
                    f"tail margin diff={fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}; "
                    f"tail drift RMS ratio={fmt(practical_training['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_drift_rms_ratio_ci95_low', 'final_tail_eval_drift_rms_ratio_ci95_high')}; "
                    f"tail accuracy diff={fmt(practical_training['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_accuracy_diff_ci95_low', 'final_tail_eval_accuracy_diff_ci95_high')}. "
                    f"LR sweep: lr=0.003 train ratio={fmt(lr_sweep.loc[0.003, 'geomean_final_train_loss_ratio_muon_over_adam'])}; "
                    f"lr=0.1 tail loss/drift ratios="
                    f"{fmt(lr_sweep.loc[0.1, 'geomean_final_tail_eval_loss_ratio_muon_over_adam'])}/"
                    f"{fmt(lr_sweep.loc[0.1, 'geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])}."
                ),
                "why_it_is_ready": "It uses paired seeds, identical mini-batch/noise schedules, and explicit final train/head/tail metrics.",
                "remaining_risk": "The LR sweep is coarse and still on scikit-learn digits; the new ResNet result is one-step only, so practical optimizer benchmarking remains needed.",
            },
        ]
    )

    story = pd.DataFrame(
        [
            {
                "section": "Question",
                "content": "When tail examples are absent from a mini-batch, can spectral geometry reduce tail function drift at the same head gain?",
            },
            {
                "section": "Theory",
                "content": "The local interference coefficient compares matched-head-gain drift through K_T,N / ||g_H||_{N,*}.",
            },
            {
                "section": "Matrix condition",
                "content": "For J_T(D)=B_T D A_T, spectral geometry has the better worst-case bound when nrank(G_H) > ssrank(B_T,A_T).",
            },
            {
                "section": "Evidence",
                "content": "Synthetic boundary, one-step digits, CIFAR-100-LT ResNet18 with a smaller-head-gain robustness check, fixed-checkpoint and trajectory Muon-style compatibility checks, small practical training, 8-step forgetting, and layerwise JVP diagnostics support the drift mechanism and its scope.",
            },
            {
                "section": "Boundary",
                "content": "The paper should not claim broad long-tail classification performance or full Muon optimizer behavior.",
            },
        ]
    )

    next_experiments = pd.DataFrame(
        [
            {
                "priority": "partly complete; extend for stronger empirical paper",
                "experiment": "Real long-tail benchmark",
                "purpose": "Test whether matched-head-gain tail drift reduction appears beyond scikit-learn digits.",
                "minimum_standard": "The new CIFAR-100-LT ResNet diagnostic is a first pass; strengthen it with more seeds, stronger checkpoints, ImageNet-LT/iNaturalist-style data, and class-wise metrics.",
            },
            {
                "priority": "must-have for full empirical optimizer claim",
                "experiment": "Real long-tail practical Muon benchmark",
                "purpose": "Test whether the selected-state compatibility pattern appears under realistic data, larger networks, and final tail metrics.",
                "minimum_standard": "Run practical Muon and baselines on CIFAR-100-LT/ImageNet-LT-style data with matched-head-gain diagnostics sampled along trajectories plus final class-wise metrics.",
            },
            {
                "priority": "must-have for architecture claim",
                "experiment": "Larger-architecture layerwise diagnostic",
                "purpose": "Check whether the scaled head-gain mechanism persists across layers in deeper models.",
                "minimum_standard": "Per-layer nrank(G_H,l), ssrank(B_T,l,A_T,l), unit JVP, scaled JVP, observed drift, and layer contribution.",
            },
            {
                "priority": "should-have",
                "experiment": "Held-out boundary prediction benchmark",
                "purpose": "Determine whether nrank-vs-ssrank is predictive beyond constructed settings.",
                "minimum_standard": "Pre-specified held-out family/architecture split with balanced accuracy, AUC, and confidence intervals.",
            },
            {
                "priority": "should-have",
                "experiment": "Retuned longer-horizon training grid",
                "purpose": "Decide whether local drift ever accumulates into final tail performance.",
                "minimum_standard": "Report final tail loss/error with small LR grids separately from local drift claims.",
            },
        ]
    )

    text = f"""# E11 Paper-Readiness Audit

This generated audit now tracks the current head-to-tail interference paper, not the older update-spectrum paper direction. It separates claims that are strong enough for the present mechanism draft from claims that require additional experiments.

## Proposed Paper Thesis

In long-tailed small-batch training, head-only updates can perturb held-out tail predictions. Under a matched-head-gain protocol, idealized spectral/polar directions can reduce tail-example logit drift when the head gradient has enough nuclear rank relative to the tail downstream-aware stable rank.

## Claim Readiness

{markdown_table(claim_status, ["claim", "readiness", "evidence", "why_it_is_ready", "remaining_risk"])}

## Publishable Story Arc

{markdown_table(story, ["section", "content"])}

## Experiments Still Needed For A Stronger Paper

{markdown_table(next_experiments, ["priority", "experiment", "purpose", "minimum_standard"])}

## What Not To Claim

1. Do not claim spectral-gradient/polar geometry or Muon is generally better for long-tailed classification.
2. Do not claim lower tail-example logit drift automatically improves tail loss, margin, or accuracy.
3. Do not claim the current local polar(M_t)/Newton-Schulz compatibility checks prove full practical Muon training behavior.
4. Do not claim the synthetic nrank-vs-ssrank boundary is already predictive for unseen real tasks.

## Current Best Paper Title Direction

> Head-to-Tail Interference in Long-Tailed Small-Batch Training: A Function-Drift View of Spectral Gradient Geometry

## Sources

- [head-to-tail interference note](e11_head_tail_interference.md)
- [long-tailed one-step diagnostic](e11_long_tail_one_step.md)
- [long-tailed Muon-style compatibility diagnostic](e11_long_tail_muon_bridge.md)
- [long-tailed practical-Muon trajectory compatibility](e11_long_tail_practical_muon_bridge.md)
- [long-tailed practical training diagnostic](e11_long_tail_practical_training.md)
- [long-tailed practical training LR sensitivity](e11_long_tail_practical_training_lr_sweep.md)
- [head-only forgetting probe](e11_long_tail_forgetting.md)
- [long-tailed layerwise diagnostic](e11_long_tail_layerwise.md)
- [artifact manifest](e11_artifact_manifest.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved paper-readiness audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
