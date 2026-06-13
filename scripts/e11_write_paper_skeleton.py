from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_paper_skeleton.md")


def interval(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def main() -> None:
    synthetic = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")
    positive = synthetic[synthetic["setting"].eq("high_head_rank_low_tail_srank")].iloc[0]
    negative = synthetic[synthetic["setting"].eq("low_head_rank_high_tail_srank")].iloc[0]
    layer_1 = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_2 = layerwise[layerwise["layer"].eq(2)].iloc[0]

    paper_claims = pd.DataFrame([
        {
            "claim": "Head-only updates create a measurable head-to-tail function-drift problem.",
            "status": "paper framing claim",
            "evidence": "Long-tail digits probes explicitly separate head batch gain from held-out tail-example logit drift.",
            "figure_or_table": "one-step and 8-step tail response figures",
        },
        {
            "claim": "Matched-head-gain spectral/polar directions reduce tail-example logit drift in the tested diagnostics.",
            "status": "main empirical claim",
            "evidence": (
                f"One-step squared drift ratio {fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                f"{interval(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                f"8-step final ratio {fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                f"{interval(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}."
            ),
            "figure_or_table": "long-tail one-step and forgetting figures",
        },
        {
            "claim": "nrank(G_H) > ssrank(B_T,A_T) is the matrix-block mechanism boundary.",
            "status": "theorem-backed mechanism claim",
            "evidence": (
                f"Positive boundary ratio {fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}; "
                f"negative boundary ratio {fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}."
            ),
            "figure_or_table": "synthetic boundary figure and paper table",
        },
        {
            "claim": "Muon-style momentum polar is compatible with part of the ideal polar drift signal.",
            "status": "selected-state compatibility claim",
            "evidence": (
                f"polar(M_t) squared drift ratio {fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                f"{interval(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                f"NS(M_t) squared drift ratio {fmt(muon_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                f"{interval(muon_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
            ),
            "figure_or_table": "Muon-style compatibility diagnostic figure",
        },
        {
            "claim": "Short practical NS-Muon trajectory states show selected-state compatibility with the local polar mechanism.",
            "status": "trajectory compatibility claim",
            "evidence": (
                f"Across {int(practical_bridge.loc['ns_momentum', 'comparisons'])} state-step comparisons, "
                f"polar(M_t) squared drift ratio {fmt(practical_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                f"{interval(practical_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                f"NS(M_t) squared drift ratio {fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                f"{interval(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
            ),
            "figure_or_table": "practical Muon trajectory compatibility figure",
        },
        {
            "claim": "A small practical NS-Muon-style training run has lower tail loss and higher measured tail margin in a fixed diagnostic.",
            "status": "practical sanity-check claim",
            "evidence": (
                f"Final train loss ratio {fmt(practical_training['geomean_final_train_loss_ratio_muon_over_adam'])} "
                f"{interval(practical_training, 'final_train_loss_ratio_ci95_low', 'final_train_loss_ratio_ci95_high')}; "
                f"tail eval loss ratio {fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                f"{interval(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}; "
                f"tail margin diff {fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                f"{interval(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}; "
                f"tail drift RMS ratio {fmt(practical_training['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} "
                f"{interval(practical_training, 'final_tail_eval_drift_rms_ratio_ci95_low', 'final_tail_eval_drift_rms_ratio_ci95_high')}; "
                f"tail accuracy diff {fmt(practical_training['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])} "
                f"{interval(practical_training, 'final_tail_eval_accuracy_diff_ci95_low', 'final_tail_eval_accuracy_diff_ci95_high')}."
            ),
            "figure_or_table": "practical training diagnostic figure",
        },
        {
            "claim": "The observed layerwise mechanism is smaller matched-head-gain step size, not lower unit-direction tail sensitivity.",
            "status": "mechanism clarification",
            "evidence": (
                f"Layer 1 unit-JVP/scaled/observed squared drift ratios {fmt(layer_1['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                f"{fmt(layer_1['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                f"{fmt(layer_1['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}; "
                f"layer 2 {fmt(layer_2['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                f"{fmt(layer_2['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                f"{fmt(layer_2['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
            ),
            "figure_or_table": "layerwise diagnostic figure",
        },
        {
            "claim": "Lower tail-example logit drift does not automatically imply better tail loss, margin, or accuracy.",
            "status": "required caveat",
            "evidence": (
                f"One-step tail loss-increase diff {fmt(one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                f"{interval(one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}; "
                f"8-step tail loss diff {fmt(forgetting['mean_final_tail_loss_increase_diff_spectral_minus_fro'])} "
                f"{interval(forgetting, 'final_tail_loss_increase_diff_ci95_low', 'final_tail_loss_increase_diff_ci95_high')}."
            ),
            "figure_or_table": "paper claim-boundary table",
        },
    ])

    section_plan = pd.DataFrame([
        {"section": "Introduction", "purpose": "Motivate tail examples being absent for many head-only updates.", "must_include": "The objective is matched-head-gain tail function drift, not optimizer leaderboard performance."},
        {"section": "Setup", "purpose": "Define H, T, F_T, J_T, local linearization, and matched-head-gain protocol.", "must_include": "State the local assumptions and the distinction between logit drift and tail loss/accuracy."},
        {"section": "Theory", "purpose": "Derive the head-to-tail interference coefficient and the matrix-block spectral-vs-Frobenius condition.", "must_include": "K_T,N, I_N(T|H), B_T D A_T, nrank(G_H), ssrank(B_T,A_T), and the worst-case bound."},
        {"section": "Experiments", "purpose": "Test the mechanism with seven lightweight diagnostics plus an LR-sensitivity robustness check.", "must_include": "Synthetic boundary, one-step digits, fixed Muon-style compatibility, trajectory Muon-style compatibility, practical training diagnostic, 8-step forgetting, and layerwise JVP."},
        {"section": "Discussion", "purpose": "State what is and is not supported.", "must_include": "No broad Muon-training claim, no final-performance claim, and required real long-tail / practical-Muon follow-ups."},
    ])

    figure_plan = pd.DataFrame([
        {"slot": "Figure 1", "artifact": "figures/e11_head_tail_interference/head_tail_drift_ratio.png", "message": "The synthetic boundary flips with nrank(G_H) versus ssrank(B_T,A_T)."},
        {"slot": "Figure 2", "artifact": "figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png", "message": "On long-tailed digits, spectral/polar reduces held-out tail-example logit drift at matched head gain."},
        {"slot": "Figure 3", "artifact": "figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png", "message": "Momentum polar gives a selected-state compatibility check for Muon-style state; finite Newton-Schulz is weaker."},
        {"slot": "Figure 4", "artifact": "figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png", "message": "Short practical NS-Muon trajectory states show squared drift ratios compatible with the local matched-head-gain mechanism."},
        {"slot": "Figure 5", "artifact": "figures/e11_long_tail_practical_training/long_tail_practical_training.png", "message": "A small practical imbalanced-training sanity check has lower measured drift/loss in this fixed diagnostic, but not better tail accuracy."},
        {"slot": "Figure 6", "artifact": "figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png", "message": "The drift reduction persists over an 8-step head-only horizon."},
        {"slot": "Figure 7", "artifact": "figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png", "message": "Layerwise results show scaled head-gain efficiency rather than lower unit-direction tail sensitivity."},
        {"slot": "Table 1", "artifact": "paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex", "message": "Quantitative paper table for drift ratios, caveats, and layerwise mechanism."},
    ])

    text = f'''# E11 Paper Skeleton

This generated skeleton is not a draft paper. It is a maintainable bridge from the current head-to-tail evidence base to the manuscript in `paper/specgrad_activation_paper/main.tex`.

## Working Title

Head-to-Tail Interference in Long-Tailed Small-Batch Training: A Function-Drift View of Spectral Gradient Geometry

## Abstract Sketch

Long-tailed small-batch training creates long stretches of head-only updates between rare tail batches. This paper studies those updates as perturbations to held-out tail functions. We define a head-to-tail interference coefficient and show that, for matrix blocks with local tail map `J_T(D)=B_T D A_T`, spectral geometry has a smaller worst-case matched-head-gain drift bound when `nrank(G_H) > ssrank(B_T,A_T)`. Lightweight diagnostics on a synthetic boundary and long-tailed digits show that idealized spectral-gradient/polar directions reduce tail-example logit drift at matched head gain, while tail loss and margin do not automatically improve. Fixed-checkpoint and short-trajectory Muon-style diagnostics show selected-state compatibility between momentum/NS directions and the local polar mechanism. The paper is therefore a local mechanism study of spectral-gradient/polar geometry, not a broad optimizer-performance claim.

## Core Claims

{markdown_table(paper_claims, ["claim", "status", "evidence", "figure_or_table"])}

## Section Plan

{markdown_table(section_plan, ["section", "purpose", "must_include"])}

## Main Figure/Table Plan

{markdown_table(figure_plan, ["slot", "artifact", "message"])}

## Current Manuscript Gap

The current evidence is strong enough for a focused theory-and-diagnostic paper about head-to-tail function drift. It is not yet enough for a broad long-tail classification benchmark paper or a full Muon optimizer theory. The next paper-critical items are a real long-tail benchmark and larger-architecture layerwise diagnostics.

## Linked Evidence

- [paper-readiness audit](e11_paper_readiness_audit.md)
- [reviewer risk audit](e11_reviewer_risk_audit.md)
- [head-to-tail interference note](e11_head_tail_interference.md)
- [long-tailed one-step diagnostic](e11_long_tail_one_step.md)
- [long-tailed Muon-style compatibility diagnostic](e11_long_tail_muon_bridge.md)
- [long-tailed practical-Muon trajectory compatibility](e11_long_tail_practical_muon_bridge.md)
- [long-tailed practical training diagnostic](e11_long_tail_practical_training.md)
- [long-tailed practical training LR sensitivity](e11_long_tail_practical_training_lr_sweep.md)
- [head-only forgetting probe](e11_long_tail_forgetting.md)
- [long-tailed layerwise diagnostic](e11_long_tail_layerwise.md)
- [artifact manifest](e11_artifact_manifest.md)
'''
    write_markdown(OUTPUT_PATH, text)
    print(f"saved paper skeleton to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
