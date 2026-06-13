from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_main_figure_captions.md")


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

    captions = pd.DataFrame(
        [
            {
                "slot": "Figure 1",
                "artifact": "figures/e11_head_tail_interference/head_tail_drift_ratio.png",
                "caption": (
                    "Controlled head-to-tail boundary check. In the positive setting, "
                    f"nrank(G_H)={fmt(positive['mean_head_gradient_nuclear_rank'])} exceeds "
                    f"ssrank(B_T,A_T)={fmt(positive['mean_tail_downstream_aware_stable_rank'])} and the spectral/Frobenius "
                    f"squared tail-example logit drift ratio is {fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}. "
                    f"In the negative setting, nrank(G_H)={fmt(negative['mean_head_gradient_nuclear_rank'])} is below "
                    f"ssrank(B_T,A_T)={fmt(negative['mean_tail_downstream_aware_stable_rank'])} and the squared drift ratio is "
                    f"{fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "interpretation": "This is consistent with the sign of the theorem boundary in a controlled construction, not a predictive claim for all natural tasks.",
            },
            {
                "slot": "Figure 2",
                "artifact": "figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png",
                "caption": (
                    "One-step long-tailed digits diagnostic under matched head gain. The spectral/polar direction has "
                    f"squared tail-example logit drift ratio {fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{interval(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')} "
                    "relative to Frobenius/GD, with lower drift in all paired seeds."
                ),
                "interpretation": "The paper-facing quantity is held-out tail-example logit drift; the figure should not be read as a tail accuracy result.",
            },
            {
                "slot": "Figure 3",
                "artifact": "figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png",
                "caption": (
                    "Local Muon-style compatibility diagnostic. Replacing the ideal current-gradient polar direction by momentum polar "
                    f"gives squared tail-example logit drift ratio {fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')} "
                    "relative to Frobenius/GD. The finite Newton-Schulz momentum approximation gives squared drift ratio "
                    f"{fmt(muon_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(muon_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "interpretation": "This is a selected-state compatibility check between momentum polar and the local polar mechanism, but not full practical-Muon training or a final-performance claim.",
            },
            {
                "slot": "Figure 4",
                "artifact": "figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png",
                "caption": (
                    "Short practical NS-Muon trajectory compatibility diagnostic. Across sampled trajectory states, momentum polar gives "
                    f"squared tail-example logit drift ratio {fmt(practical_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(practical_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}, "
                    "and finite Newton-Schulz momentum gives squared drift ratio "
                    f"{fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "interpretation": "This extends the selected-state compatibility check along a short trajectory, but it is still not a final-performance benchmark.",
            },
            {
                "slot": "Figure 5",
                "artifact": "figures/e11_long_tail_practical_training/long_tail_practical_training.png",
                "caption": (
                    "Small practical imbalanced-training diagnostic. With identical noisy mini-batch schedules and paired seeds, "
                    f"the finite-Newton-Schulz Muon-style run has final train loss ratio "
                    f"{fmt(practical_training['geomean_final_train_loss_ratio_muon_over_adam'])} "
                    f"{interval(practical_training, 'final_train_loss_ratio_ci95_low', 'final_train_loss_ratio_ci95_high')}, "
                    f"tail eval loss ratio {fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"{interval(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}, "
                    f"tail margin difference {fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                    f"{interval(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}, "
                    f"and tail drift RMS ratio {fmt(practical_training['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} "
                    f"{interval(practical_training, 'final_tail_eval_drift_rms_ratio_ci95_low', 'final_tail_eval_drift_rms_ratio_ci95_high')}."
                ),
                "interpretation": "This is a lightweight practical sanity check consistent with the drift story; tail accuracy remains unchanged and this is not a benchmark claim.",
            },
            {
                "slot": "Figure 6",
                "artifact": "figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png",
                "caption": (
                    "Eight-step head-only forgetting probe. The final spectral/Frobenius squared tail-example logit drift ratio is "
                    f"{fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{interval(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}, "
                    f"and the drift-area ratio is {fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"{interval(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}."
                ),
                "interpretation": "This extends the local drift effect across a short head-only horizon, but final tail loss and margin remain separate caveats.",
            },
            {
                "slot": "Figure 7",
                "artifact": "figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png",
                "caption": (
                    "Layerwise tail-drift diagnostic. Unit-direction JVP ratios are above one in both layers "
                    f"({fmt(layer_1['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])} and "
                    f"{fmt(layer_2['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}), but scaled and observed ratios are below one "
                    f"(layer 1 {fmt(layer_1['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_1['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}; "
                    f"layer 2 {fmt(layer_2['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_2['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])})."
                ),
                "interpretation": "The mechanism is smaller matched-head-gain step size, not a claim that unit polar directions are always safer for tail outputs.",
            },
            {
                "slot": "Table 1",
                "artifact": "paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex",
                "caption": "Generated summary table for the current head-to-tail evidence, including synthetic boundary, one-step drift, Muon-style compatibility checks, small practical training, head-only forgetting, and layerwise diagnostics.",
                "interpretation": "The table should make the paper's claim boundary explicit: drift evidence is stronger than final-performance evidence.",
            },
        ]
    )

    text = f"""# E11 Main Figure Captions

This generated note drafts paper-safe captions for the current head-to-tail paper figures and table. Each caption includes the quantitative anchor and the intended interpretation boundary.

## Captions

{markdown_table(captions, ["slot", "artifact", "caption", "interpretation"])}

## Caption Discipline

1. Every main caption should include a quantitative anchor.
2. Captions should distinguish tail-example logit drift from tail loss, margin, accuracy, or final performance.
3. The wording should follow the claim boundaries in [e11_paper_readiness_audit.md](e11_paper_readiness_audit.md) and [e11_reviewer_risk_audit.md](e11_reviewer_risk_audit.md).
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved main figure captions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
