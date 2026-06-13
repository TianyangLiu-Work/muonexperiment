from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_evidence_index.md")


def link(path: str) -> str:
    return f"[{path}](../{path})"


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def main() -> None:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    practical_lr_sweep = pd.read_csv("results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv")
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")
    layer_one = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_two = layerwise[layerwise["layer"].eq(2)].iloc[0]

    positive = head_tail.loc["high_head_rank_low_tail_srank"]
    negative = head_tail.loc["low_head_rank_high_tail_srank"]
    lr_sweep = practical_lr_sweep.assign(muon_lr_rounded=practical_lr_sweep["muon_lr"].round(3)).set_index("muon_lr_rounded")

    evidence = pd.DataFrame(
        [
            {
                "claim": "The synthetic condition has the intended positive case.",
                "recommended_figure": link("figures/e11_head_tail_interference/head_tail_drift_ratio.png"),
                "source_data": link("results/e11_head_tail_interference/pair_summary.csv"),
                "quantitative_anchor": (
                    f"nrank(G_H)={fmt(positive['mean_head_gradient_nuclear_rank'])}; "
                    f"ssrank(B_T,A_T)={fmt(positive['mean_tail_downstream_aware_stable_rank'])}; "
                    f"drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(positive, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "how_to_read": "The spectral/polar direction causes less tail drift when head gradient nuclear rank exceeds the downstream tail sensitivity rank.",
                "caveat": "This is a constructed mechanism check, not a real training benchmark.",
            },
            {
                "claim": "The synthetic condition has the intended negative case.",
                "recommended_figure": link("figures/e11_head_tail_interference/head_tail_drift_ratio.png"),
                "source_data": link("results/e11_head_tail_interference/pair_summary.csv"),
                "quantitative_anchor": (
                    f"nrank(G_H)={fmt(negative['mean_head_gradient_nuclear_rank'])}; "
                    f"ssrank(B_T,A_T)={fmt(negative['mean_tail_downstream_aware_stable_rank'])}; "
                    f"drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(negative, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "how_to_read": "When the rank/sensitivity inequality is reversed, spectral/polar drift is larger rather than smaller.",
                "caveat": "The current negative case tests sign logic, not a learned predictor.",
            },
            {
                "claim": "Long-tailed one-step updates reduce tail logit drift at matched head gain.",
                "recommended_figure": link("figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png"),
                "source_data": link("results/e11_long_tail_one_step/pair_summary.csv"),
                "quantitative_anchor": (
                    f"drift ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"paired fraction={fmt(one_step['spectral_less_tail_output_drift_fraction'])}."
                ),
                "how_to_read": "Ratios below 1 mean spectral/polar updates move tail logits less than Frobenius/GD-style updates for the same first-order head gain.",
                "caveat": "Tail loss increases slightly more for spectral in this one-step table, so the result is about function drift.",
            },
            {
                "claim": "The head-only forgetting diagnostic shows lower measured tail drift across eight steps.",
                "recommended_figure": link("figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png"),
                "source_data": link("results/e11_long_tail_forgetting/summary.csv"),
                "quantitative_anchor": (
                    f"final drift ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}; "
                    f"area ratio={fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}."
                ),
                "how_to_read": "The drift gap is not only an isolated one-step artifact in this short head-only horizon.",
                "caveat": "This is still an 8-step diagnostic, not a full long-tail training run.",
            },
            {
                "claim": "The ideal polar direction has selected-state compatibility with Muon-style momentum directions.",
                "recommended_figure": link("figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png"),
                "source_data": link("results/e11_long_tail_muon_bridge/pair_summary.csv"),
                "quantitative_anchor": (
                    f"polar(M_t) drift ratio={fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"NS(M_t) drift ratio={fmt(muon_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(muon_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "how_to_read": "Momentum polar has a drift ratio below 1, while the finite Newton-Schulz approximation is weaker and its confidence interval crosses 1.",
                "caveat": "This is a selected-state local diagnostic at fixed checkpoints, not proof of full Muon training behavior.",
            },
            {
                "claim": "Muon-style directions show selected-state compatibility across a short practical NS-Muon trajectory.",
                "recommended_figure": link("figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png"),
                "source_data": link("results/e11_long_tail_practical_muon_bridge/summary.csv"),
                "quantitative_anchor": (
                    f"trajectory polar(M_t) drift ratio={fmt(practical_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(practical_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"trajectory NS(M_t) drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')} over "
                    f"{int(practical_bridge.loc['ns_momentum', 'comparisons'])} state-step comparisons."
                ),
                "how_to_read": "Sampled short-trajectory Muon-style states have matched-head-gain drift ratios below Fro/GD in this diagnostic.",
                "caveat": "This is still a local diagnostic on small digits, not a final optimizer-performance benchmark.",
            },
            {
                "claim": "A small practical imbalanced-training run is consistent with the tail-drift story.",
                "recommended_figure": link("figures/e11_long_tail_practical_training/long_tail_practical_training.png"),
                "source_data": link("results/e11_long_tail_practical_training/summary.csv"),
                "quantitative_anchor": (
                    f"train loss ratio={fmt(practical_training['geomean_final_train_loss_ratio_muon_over_adam'])} "
                    f"{ci(practical_training, 'final_train_loss_ratio_ci95_low', 'final_train_loss_ratio_ci95_high')}; "
                    f"tail eval loss ratio={fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"{ci(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}; "
                    f"tail margin diff={fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                    f"{ci(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}; "
                    f"tail drift RMS ratio={fmt(practical_training['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} "
                    f"{ci(practical_training, 'final_tail_eval_drift_rms_ratio_ci95_low', 'final_tail_eval_drift_rms_ratio_ci95_high')}."
                ),
                "how_to_read": "The finite-NS Muon-style run has lower train/head loss, a higher measured tail-margin reading, and lower measured tail drift in this fixed lightweight training loop.",
                "caveat": "Tail accuracy difference is 0 and hyperparameters are fixed; this is a sanity check, not a broad optimizer benchmark.",
            },
            {
                "claim": "The practical-training result has a coarse LR sensitivity check.",
                "recommended_figure": link("figures/e11_long_tail_practical_training_lr_sweep/long_tail_practical_training_lr_sweep.png"),
                "source_data": link("results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv"),
                "quantitative_anchor": (
                    f"lr=0.003 train ratio={fmt(lr_sweep.loc[0.003, 'geomean_final_train_loss_ratio_muon_over_adam'])}; "
                    f"lr=0.03 tail drift ratio={fmt(lr_sweep.loc[0.03, 'geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])}; "
                    f"lr=0.1 tail loss ratio={fmt(lr_sweep.loc[0.1, 'geomean_final_tail_eval_loss_ratio_muon_over_adam'])}."
                ),
                "how_to_read": "The selected lr=0.03 is a tested operating point between under-training and over-shooting the tail.",
                "caveat": "This is a coarse small-task grid; it reduces cherry-picking risk but does not replace retuned long-tail benchmarks.",
            },
            {
                "claim": "Layerwise unit-direction sensitivity is not lower for spectral/polar updates.",
                "recommended_figure": link("figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png"),
                "source_data": link("results/e11_long_tail_layerwise/summary.csv"),
                "quantitative_anchor": (
                    f"unit JVP ratios: layer 1={fmt(layer_one['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"layer 2={fmt(layer_two['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "how_to_read": "Ratios above 1 mean the raw unit spectral direction can be more tail-sensitive.",
                "caveat": "This is the main mechanism caveat: direction alone is not enough.",
            },
            {
                "claim": "Layerwise matched-head-gain scaling explains the observed lower drift.",
                "recommended_figure": link("figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png"),
                "source_data": link("results/e11_long_tail_layerwise/summary.csv"),
                "quantitative_anchor": (
                    f"scaled ratios: layer 1={fmt(layer_one['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"layer 2={fmt(layer_two['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}; "
                    f"observed ratios: layer 1={fmt(layer_one['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"layer 2={fmt(layer_two['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "how_to_read": "After matching the head gain, spectral/polar needs a smaller effective step and produces lower observed tail drift.",
                "caveat": "This is evidence for scaled head-gain efficiency, not a universal spectral-stability claim.",
            },
            {
                "claim": "The current evidence does not prove broad tail-accuracy or benchmark improvement.",
                "recommended_figure": link("figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png"),
                "source_data": link("results/e11_long_tail_one_step/pair_summary.csv"),
                "quantitative_anchor": (
                    f"tail-loss diff={fmt(one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"{ci(one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}; "
                    f"tail-accuracy-drop diff={fmt(one_step['mean_tail_accuracy_drop_diff_spectral_minus_fro'])} "
                    f"{ci(one_step, 'tail_accuracy_drop_diff_ci95_low', 'tail_accuracy_drop_diff_ci95_high')}."
                ),
                "how_to_read": "The one-step table is about tail logit drift, while the small practical training run is only a narrow tail-loss/margin diagnostic.",
                "caveat": "A stronger claim needs real long-tail benchmarks and larger practical Muon training ablations.",
            },
        ]
    )

    text = f"""# E11 Evidence Index

This generated index maps each current head-to-tail paper claim to the most relevant figure and CSV source. Older condition-geometry artifacts remain useful guardrails, but they are not the main evidence table for the active paper draft.

{markdown_table(evidence, ["claim", "recommended_figure", "source_data", "quantitative_anchor", "how_to_read", "caveat"])}
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved evidence index to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
