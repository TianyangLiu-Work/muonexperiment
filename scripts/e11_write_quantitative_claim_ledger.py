from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_quantitative_claim_ledger.md")


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def paired_geomean_ratio(step_metrics: pd.DataFrame, numerator: str, denominator: str, column: str) -> float:
    wide = step_metrics.pivot(index="seed", columns="geometry", values=column)
    return math.exp((wide[numerator] / wide[denominator]).map(math.log).mean())


def main() -> None:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    one_step_steps = pd.read_csv("results/e11_long_tail_one_step/step_metrics.csv")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    state_control = pd.read_csv(
        "results/e11_long_tail_muon_state_source_control/summary.csv"
    ).set_index(["state_source", "direction"])
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    practical_lr_sweep = pd.read_csv("results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv")
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")
    layer_one = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_two = layerwise[layerwise["layer"].eq(2)].iloc[0]

    positive = head_tail.loc["high_head_rank_low_tail_srank"]
    negative = head_tail.loc["low_head_rank_high_tail_srank"]
    lr_sweep = practical_lr_sweep.assign(muon_lr_rounded=practical_lr_sweep["muon_lr"].round(3)).set_index("muon_lr_rounded")
    alignment_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "alignment")
    update_fro_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "update_fro_norm")
    update_op_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "update_op_norm")

    ledger = pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "paper_status": "main mechanism",
                "claim": "The synthetic nrank-vs-ssrank condition predicts the sign of head-to-tail drift reduction.",
                "quantitative_evidence": (
                    f"Positive setting: nrank(G_H)={fmt(positive['mean_head_gradient_nuclear_rank'])}, "
                    f"ssrank(B_T,A_T)={fmt(positive['mean_tail_downstream_aware_stable_rank'])}, "
                    f"spectral/Frobenius squared drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(positive, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}. "
                    f"Negative setting: nrank(G_H)={fmt(negative['mean_head_gradient_nuclear_rank'])}, "
                    f"ssrank(B_T,A_T)={fmt(negative['mean_tail_downstream_aware_stable_rank'])}, "
                    f"squared drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(negative, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "allowed_wording": "The constructed boundary example is consistent with the rank/sensitivity sign condition.",
                "do_not_write": "Do not present the synthetic boundary as an out-of-sample predictor.",
                "evidence": "figures/e11_head_tail_interference/head_tail_drift_ratio.png; results/e11_head_tail_interference/pair_summary.csv",
            },
            {
                "claim_id": "C2",
                "paper_status": "main empirical diagnostic",
                "claim": "On long-tailed digits, spectral/polar one-step updates reduce held-out tail-example logit drift at matched head gain.",
                "quantitative_evidence": (
                    f"Drift-squared ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"head-alignment ratio={fmt(alignment_ratio)}; "
                    f"Frobenius-norm ratio={fmt(update_fro_ratio)}; "
                    f"operator-norm ratio={fmt(update_op_ratio)}; "
                    f"spectral-lower paired seeds={fmt(one_step['spectral_less_tail_output_drift_fraction'])} fraction over "
                    f"{int(one_step['seeds'])} seeds."
                ),
                "allowed_wording": "The one-step diagnostic gives direct evidence for lower matched-head-gain tail-example logit drift in the tested digits setting.",
                "do_not_write": "Do not say this proves better tail classification performance.",
                "evidence": "figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png; results/e11_long_tail_one_step/pair_summary.csv",
            },
            {
                "claim_id": "C3",
                "paper_status": "main caveat",
                "claim": "Lower one-step tail drift does not yet imply lower tail loss or better tail accuracy.",
                "quantitative_evidence": (
                    f"Tail-loss increase difference spectral-minus-Frobenius="
                    f"{fmt(one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"{ci(one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}; "
                    f"tail-accuracy-drop difference={fmt(one_step['mean_tail_accuracy_drop_diff_spectral_minus_fro'])} "
                    f"{ci(one_step, 'tail_accuracy_drop_diff_ci95_low', 'tail_accuracy_drop_diff_ci95_high')}."
                ),
                "allowed_wording": "The defensible performance-adjacent claim is about logits/function drift, not accuracy.",
                "do_not_write": "Do not imply final tail accuracy improves from this evidence alone.",
                "evidence": "results/e11_long_tail_one_step/pair_summary.csv",
            },
            {
                "claim_id": "C4",
                "paper_status": "main short-horizon diagnostic",
                "claim": "The head-only forgetting probe shows lower measured spectral tail drift over eight matched head-gain steps.",
                "quantitative_evidence": (
                    f"Final squared drift ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}; "
                    f"drift-area ratio={fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}."
                ),
                "allowed_wording": "The short-horizon forgetting diagnostic is consistent with persistent drift reduction.",
                "do_not_write": "Do not call this a fully trained long-tail benchmark.",
                "evidence": "figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png; results/e11_long_tail_forgetting/summary.csv",
            },
            {
                "claim_id": "C5",
                "paper_status": "main mechanism caveat",
                "claim": "Layerwise evidence says the drift reduction comes from norm-specific matched-head-gain scaling, not lower unit-direction tail sensitivity.",
                "quantitative_evidence": (
                    f"Layer 1 unit/scaled/observed squared drift ratios="
                    f"{fmt(layer_one['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_one['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_one['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}. "
                    f"Layer 2 unit/scaled/observed squared drift ratios="
                    f"{fmt(layer_two['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_two['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_two['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "allowed_wording": "Spectral directions use a smaller operator-norm step despite a larger Frobenius-norm step after matching head gain in this diagnostic.",
                "do_not_write": "Do not say spectral directions are intrinsically less tail-sensitive layerwise or uniformly smaller.",
                "evidence": "figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png; results/e11_long_tail_layerwise/summary.csv",
            },
            {
                "claim_id": "C6",
                "paper_status": "Muon-style compatibility",
                "claim": "Momentum polar and short practical NS-Muon trajectory states show selected-state compatibility with the local drift reduction.",
                "quantitative_evidence": (
                    f"polar(M_t) squared drift ratio vs Fro/GD="
                    f"{fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"short-trajectory NS(M_t) squared drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')} "
                    f"over {int(practical_bridge.loc['ns_momentum', 'comparisons'])} state-step comparisons. "
                    f"On Fro/GD-generated states, NS(M_t) squared drift ratio="
                    f"{fmt(state_control.loc[('fro_gd_trajectory', 'ns_momentum'), 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(state_control.loc[('fro_gd_trajectory', 'ns_momentum')], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "allowed_wording": "The local diagnostics are selected-state compatibility checks between ideal polar geometry and Muon-style momentum/NS directions at sampled states.",
                "do_not_write": "Do not claim complete practical Muon training behavior or final performance is explained.",
                "evidence": "figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png; figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png; figures/e11_long_tail_muon_state_source_control/long_tail_muon_state_source_control.png; results/e11_long_tail_practical_muon_bridge/summary.csv; results/e11_long_tail_muon_state_source_control/summary.csv",
            },
            {
                "claim_id": "C7",
                "paper_status": "small practical-training sanity check",
                "claim": "A fixed lightweight NS-Muon-style imbalanced-training run has lower measured drift/loss in this diagnostic, but not better tail accuracy.",
                "quantitative_evidence": (
                    f"final train loss ratio Muon/Adam="
                    f"{fmt(practical_training['geomean_final_train_loss_ratio_muon_over_adam'])} "
                    f"{ci(practical_training, 'final_train_loss_ratio_ci95_low', 'final_train_loss_ratio_ci95_high')}; "
                    f"tail eval loss ratio={fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"{ci(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}; "
                    f"tail margin diff={fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                    f"{ci(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}; "
                    f"tail drift RMS ratio={fmt(practical_training['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} "
                    f"{ci(practical_training, 'final_tail_eval_drift_rms_ratio_ci95_low', 'final_tail_eval_drift_rms_ratio_ci95_high')}; "
                    f"tail accuracy diff={fmt(practical_training['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])} "
                    f"{ci(practical_training, 'final_tail_eval_accuracy_diff_ci95_low', 'final_tail_eval_accuracy_diff_ci95_high')}. "
                    f"LR sweep: lr=0.003 under-trains with train ratio {fmt(lr_sweep.loc[0.003, 'geomean_final_train_loss_ratio_muon_over_adam'])}; "
                    f"lr=0.1 worsens tail loss/drift with ratios "
                    f"{fmt(lr_sweep.loc[0.1, 'geomean_final_tail_eval_loss_ratio_muon_over_adam'])}/"
                    f"{fmt(lr_sweep.loc[0.1, 'geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])}."
                ),
                "allowed_wording": "This small practical run is consistent with the local drift story and reinforces the separation between tail loss/margin and tail accuracy.",
                "do_not_write": "Do not call this a full Muon benchmark or evidence of general tail accuracy improvement.",
                "evidence": "figures/e11_long_tail_practical_training/long_tail_practical_training.png; figures/e11_long_tail_practical_training_lr_sweep/long_tail_practical_training_lr_sweep.png; results/e11_long_tail_practical_training/summary.csv; results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv",
            },
        ]
    )

    priority = pd.DataFrame(
        [
            {
                "priority": "write first",
                "claim_ids": "C1, C2, C4, C5",
                "reason": "These four claims form the current head-to-tail mechanism story.",
            },
            {
                "priority": "state explicitly as caveats",
                "claim_ids": "C3, C6, C7",
                "reason": "These keep the paper from overclaiming performance or full Muon training behavior.",
            },
        ]
    )

    text = f"""# E11 Quantitative Claim Ledger

This generated ledger is the paper-writing guardrail for the current head-to-tail interference paper. Each claim below must be stated with its quantitative anchor and caveat. Claims from the older condition-geometry project belong in appendix/guardrail discussion unless a current script adds them here.

## Claim Ledger

{markdown_table(ledger, ["claim_id", "paper_status", "claim", "quantitative_evidence", "allowed_wording", "do_not_write", "evidence"])}

## Writing Priority

{markdown_table(priority, ["priority", "claim_ids", "reason"])}

## Practical Rule

Use `C1 -> C2 -> C6 -> C7 -> C4 -> C5` as the main paper sequence. Put `C3` next to every empirical drift claim and keep `C6`/`C7` scoped below broad final-performance claims.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved quantitative claim ledger to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
