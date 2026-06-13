from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_main_paper_package.md")


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

    main_items = pd.DataFrame(
        [
            {
                "slot": "Figure 1",
                "artifact": "figures/e11_head_tail_interference/head_tail_drift_ratio.png",
                "claim": "The nrank-vs-ssrank condition has the correct sign in controlled positive and negative settings.",
                "quantitative_anchor": (
                    f"Positive squared drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}; "
                    f"negative squared drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "reader_takeaway": "The theorem boundary is not decorative; flipping the constructed geometry flips the drift direction.",
            },
            {
                "slot": "Figure 2",
                "artifact": "figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png",
                "claim": "Spectral/polar reduces held-out tail-example logit drift at matched head gain on long-tailed digits.",
                "quantitative_anchor": (
                    f"Squared tail-example logit drift ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{interval(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"20/20 paired seeds lower drift."
                ),
                "reader_takeaway": "The main empirical quantity is tail function drift, not final classification performance.",
            },
            {
                "slot": "Figure 3",
                "artifact": "figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png",
                "claim": "Momentum polar gives a selected-state compatibility check for Muon-style state.",
                "quantitative_anchor": (
                    f"polar(M_t) squared drift ratio={fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"NS(M_t) squared drift ratio={fmt(muon_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(muon_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "reader_takeaway": "Muon-style state is locally compatible in sampled states, but finite Newton-Schulz and trajectory-level behavior remain caveats.",
            },
            {
                "slot": "Figure 4",
                "artifact": "figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png",
                "claim": "Muon-style directions show selected-state compatibility across a short practical NS-Muon trajectory.",
                "quantitative_anchor": (
                    f"trajectory polar(M_t) squared drift ratio={fmt(practical_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(practical_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"trajectory NS(M_t) squared drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "reader_takeaway": "Practical Muon-style states show squared drift ratios compatible with the local polar mechanism along sampled short-trajectory states.",
            },
            {
                "slot": "Figure 5",
                "artifact": "figures/e11_long_tail_practical_training/long_tail_practical_training.png",
                "claim": "A small practical imbalanced-training run reports lower tail loss, higher tail margin, and lower late-stage head loss in a fixed diagnostic.",
                "quantitative_anchor": (
                    f"Train loss ratio={fmt(practical_training['geomean_final_train_loss_ratio_muon_over_adam'])} "
                    f"{interval(practical_training, 'final_train_loss_ratio_ci95_low', 'final_train_loss_ratio_ci95_high')}; "
                    f"tail eval loss ratio={fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"{interval(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}; "
                    f"tail margin diff={fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                    f"{interval(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}; "
                    f"tail drift RMS ratio={fmt(practical_training['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} "
                    f"{interval(practical_training, 'final_tail_eval_drift_rms_ratio_ci95_low', 'final_tail_eval_drift_rms_ratio_ci95_high')}."
                ),
                "reader_takeaway": "The practical result is a fixed lightweight sanity check; tail accuracy is unchanged and benchmark claims remain open.",
            },
            {
                "slot": "Figure 6",
                "artifact": "figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png",
                "claim": "The lower measured tail drift remains visible over a short head-only horizon.",
                "quantitative_anchor": (
                    f"Final squared drift ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{interval(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}; "
                    f"area ratio={fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"{interval(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}."
                ),
                "reader_takeaway": "The effect is visible beyond a single step, but tail loss and margin remain separate outcomes.",
            },
            {
                "slot": "Figure 7",
                "artifact": "figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png",
                "claim": "Layerwise drift reduction comes from matched-head-gain scaling, not safer unit directions.",
                "quantitative_anchor": (
                    f"Layer 1 unit/scaled/observed={fmt(layer_1['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_1['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_1['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}; "
                    f"layer 2={fmt(layer_2['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_2['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_2['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "reader_takeaway": "Spectral/polar directions are not uniformly lower-sensitivity; the head-gain normalization matters.",
            },
            {
                "slot": "Table 1",
                "artifact": "paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex",
                "claim": "Paper-facing quantitative table summarizing the mechanism, drift evidence, and caveats.",
                "quantitative_anchor": "Generated from the current head-to-tail result CSVs.",
                "reader_takeaway": "The table keeps claim scope explicit for readers and reviewers.",
            },
        ]
    )

    appendix_items = pd.DataFrame(
        [
            {"artifact": "discussion/e11_paper_readiness_audit.md", "role": "Claim readiness, missing experiments, and paper title direction for the current head-to-tail draft."},
            {"artifact": "discussion/e11_reviewer_risk_audit.md", "role": "Reviewer objections and safe responses for function-drift claims."},
            {"artifact": "discussion/e11_pasted_review_audit.md", "role": "Traceability from pasted reviewer-risk notes to current paper edits and residual risks."},
            {"artifact": "discussion/e11_completion_audit.md", "role": "Requirement-by-requirement completion evidence for the current final-report paper package."},
            {"artifact": "discussion/e11_end_of_draft_self_review.md", "role": "Five-dimension self-review and claim-evidence map for the current draft."},
            {"artifact": "figures/e11_long_tail_muon_state_source_control/long_tail_muon_state_source_control.png", "role": "Appendix control for Muon trajectory state-selection risk; evaluates Muon-style directions on Fro/GD-generated states."},
            {"artifact": "discussion/e11_long_tail_muon_state_source_control.md", "role": "Markdown summary and CSV links for the Muon state-source control."},
            {"artifact": "discussion/e11_long_tail_practical_training_lr_sweep.md", "role": "Supporting robustness check for the selected practical-training Muon learning rate."},
            {"artifact": "discussion/e11_activation_perturbation.md", "role": "Background activation-geometry evidence; not a current main result."},
            {"artifact": "discussion/e11_reproduction_checklist.md", "role": "Minimal reproduction and appendix/guardrail reproduction commands."},
            {"artifact": "discussion/e11_artifact_manifest.md", "role": "Commit boundary for paper evidence, generated results, figures, and ignored local caches."},
        ]
    )

    excluded = pd.DataFrame(
        [
            {"artifact": "equal-update update-spectrum figures", "reason": "Older condition-geometry guardrails; they are not the current head-to-tail paper's main evidence."},
            {"artifact": "legacy optimizer-switch and broad LR-sweep figures", "reason": "Useful overclaim controls, but too far from the current head-to-tail drift mechanism. This does not exclude the practical-training LR sensitivity note listed above."},
            {"artifact": "boundary predictor detailed tables", "reason": "The current paper does not claim a predictive boundary model for unseen tasks."},
        ]
    )

    text = f"""# E11 Main Paper Package

This generated note selects the smallest evidence package for the current head-to-tail interference manuscript. The main paper should be carried by seven figures plus one generated table, with older condition-geometry artifacts used only as appendix guardrails. The LR-sensitivity figure is supporting robustness evidence rather than a main figure.

## Main Figure/Table Package

{markdown_table(main_items, ["slot", "artifact", "claim", "quantitative_anchor", "reader_takeaway"])}

## Appendix Allocation

{markdown_table(appendix_items, ["artifact", "role"])}

## Claims To Exclude From Main Text

{markdown_table(excluded, ["artifact", "reason"])}

## Main-Text Claim Order

1. Long-tailed small-batch training creates head-only updates that can perturb held-out tail functions.
2. The local matched-head-gain theory predicts a spectral-vs-Frobenius drift boundary through `nrank(G_H) > ssrank(B_T,A_T)`.
3. Synthetic and long-tailed digits diagnostics provide direct evidence for lower matched-head-gain tail-example logit drift for idealized spectral/polar directions in the tested settings.
4. The Muon-style compatibility diagnostics provide selected fixed-state and trajectory-state checks.
5. The result is about function drift under local matched-head-gain comparisons, not broad Muon performance or final tail accuracy.

## Drafting Rule

If a sentence cannot be supported by Figure 1, Figure 2, Figure 3, Figure 4, Figure 5, Figure 6, Figure 7, or Table 1, it should probably be in the appendix or discussion rather than in the main result section.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved main paper package to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
