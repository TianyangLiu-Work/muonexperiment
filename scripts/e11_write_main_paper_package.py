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
    cifar_resnet_practical_bridge = pd.read_csv(
        "results/e11_cifar100_resnet_practical_muon_bridge/summary.csv"
    ).set_index(["state_source", "direction"])
    cifar_resnet_layer_jvp_checkpoint_prediction = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/prediction_summary.csv"
    ).set_index("predictor")
    cifar_resnet_layer_jvp_checkpoint_residual_prediction = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/residual_prediction_summary.csv"
    ).set_index("predictor")
    cifar_resnet_imbalance_sweep = pd.read_csv(
        "results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv"
    )
    cifar_resnet_lt_standard_eval = pd.read_csv(
        "results/e11_cifar100_resnet_lt_standard_eval/summary.csv"
    ).set_index("frequency_group")
    cifar_resnet_lt_recipe = pd.read_csv(
        "results/e11_cifar100_resnet_lt_recipe_benchmark/summary.csv"
    ).set_index(["recipe", "frequency_group"])
    cifar_resnet_lt_recipe_pairs = pd.read_csv(
        "results/e11_cifar100_resnet_lt_recipe_benchmark/pair_summary.csv"
    ).set_index(["recipe", "frequency_group"])
    cifar_resnet_lt_muon_final = pd.read_csv(
        "results/e11_cifar100_resnet_lt_muon_final_benchmark/summary.csv"
    ).set_index(["recipe", "frequency_group"])
    cifar_resnet_lt_muon_final_pairs = pd.read_csv(
        "results/e11_cifar100_resnet_lt_muon_final_benchmark/pair_summary.csv"
    ).set_index(["recipe", "frequency_group"])
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")
    positive = synthetic[synthetic["setting"].eq("high_head_rank_low_tail_srank")].iloc[0]
    negative = synthetic[synthetic["setting"].eq("low_head_rank_high_tail_srank")].iloc[0]
    layer_1 = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_2 = layerwise[layerwise["layer"].eq(2)].iloc[0]
    resnet_adam_ns = cifar_resnet_practical_bridge.loc[("adamw_matrix_trajectory", "ns_momentum")]
    resnet_muon_ns = cifar_resnet_practical_bridge.loc[("ns_muon_matrix_trajectory", "ns_momentum")]
    resnet_jvp_checkpoint_scaled = cifar_resnet_layer_jvp_checkpoint_prediction.loc["source_scaled_jvp_ratio"]
    resnet_jvp_checkpoint_observed = cifar_resnet_layer_jvp_checkpoint_prediction.loc[
        "source_observed_drift_ratio"
    ]
    resnet_jvp_checkpoint_early_layer = cifar_resnet_layer_jvp_checkpoint_prediction.loc[
        "architecture_early_layer_prior"
    ]
    resnet_jvp_checkpoint_observed_residual = cifar_resnet_layer_jvp_checkpoint_residual_prediction.loc[
        "source_observed_residual"
    ]
    resnet_jvp_checkpoint_scaled_residual = cifar_resnet_layer_jvp_checkpoint_residual_prediction.loc[
        "source_scaled_jvp_residual"
    ]
    resnet_imbalance_worst = cifar_resnet_imbalance_sweep.loc[
        cifar_resnet_imbalance_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    resnet_imbalance_best_tail_accuracy = cifar_resnet_imbalance_sweep.loc[
        cifar_resnet_imbalance_sweep["mean_tail_accuracy_before"].idxmax()
    ]
    resnet_lt_many = cifar_resnet_lt_standard_eval.loc["many"]
    resnet_lt_medium = cifar_resnet_lt_standard_eval.loc["medium"]
    resnet_lt_few = cifar_resnet_lt_standard_eval.loc["few"]
    resnet_recipe_sgd_all = cifar_resnet_lt_recipe.loc[("sgd_aug_ce", "all")]
    resnet_recipe_sgd_few = cifar_resnet_lt_recipe.loc[("sgd_aug_ce", "few")]
    resnet_recipe_sgd_few_diff = cifar_resnet_lt_recipe_pairs.loc[("sgd_aug_ce", "few")]
    resnet_muon_final_lr1e4_all = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "all")]
    resnet_muon_final_lr1e4_few = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "few")]
    resnet_muon_final_lr1e4_all_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "all")
    ]
    resnet_muon_final_lr1e4_few_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "few")
    ]

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
            {"artifact": "figures/e11_cifar100_resnet_layer_jvp_tail_quality/cifar100_resnet_layer_jvp_tail_quality.png", "role": "Appendix ResNet all-layer finite-difference JVP control at the tail-rich checkpoint; supports the architecture-level mechanism bridge without turning the paper into a benchmark claim."},
            {"artifact": "discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md", "role": "Markdown summary and CSV links for the ResNet all-layer JVP tail-quality diagnostic."},
            {
                "artifact": "figures/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/cifar100_resnet_layer_jvp_checkpoint_prediction.png",
                "role": (
                    "Appendix checkpoint-transfer boundary check for the all-layer ResNet JVP readout; "
                    f"source-observed/early-layer controls have held-out Spearman "
                    f"{fmt(resnet_jvp_checkpoint_observed['mean_spearman_log_predictor_vs_log_target_observed'])} "
                    f"{interval(resnet_jvp_checkpoint_observed, 'spearman_ci95_low', 'spearman_ci95_high')} and "
                    f"{fmt(resnet_jvp_checkpoint_early_layer['mean_spearman_log_predictor_vs_log_target_observed'])} "
                    f"{interval(resnet_jvp_checkpoint_early_layer, 'spearman_ci95_low', 'spearman_ci95_high')}, "
                    f"but scaled-JVP held-out Spearman is "
                    f"{fmt(resnet_jvp_checkpoint_scaled['mean_spearman_log_predictor_vs_log_target_observed'])} "
                    f"{interval(resnet_jvp_checkpoint_scaled, 'spearman_ci95_low', 'spearman_ci95_high')}; "
                    f"after early-layer residualization, observed residual Spearman is "
                    f"{fmt(resnet_jvp_checkpoint_observed_residual['mean_spearman_residual_predictor_vs_residual_target_observed'])} "
                    f"{interval(resnet_jvp_checkpoint_observed_residual, 'spearman_ci95_low', 'spearman_ci95_high')} "
                    f"while scaled-JVP residual Spearman is "
                    f"{fmt(resnet_jvp_checkpoint_scaled_residual['mean_spearman_residual_predictor_vs_residual_target_observed'])} "
                    f"{interval(resnet_jvp_checkpoint_scaled_residual, 'spearman_ci95_low', 'spearman_ci95_high')}."
                ),
            },
            {"artifact": "discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md", "role": "Markdown summary and CSV links for the held-out checkpoint-transfer JVP benchmark."},
            {
                "artifact": "figures/e11_cifar100_resnet_imbalance_sweep/cifar100_resnet_imbalance_sweep.png",
                "role": (
                    "Appendix CIFAR-100-LT ResNet18 tail-count imbalance sweep over "
                    f"{int(cifar_resnet_imbalance_sweep['tail_train_per_class'].nunique())} settings; "
                    f"worst drift ratio is "
                    f"{fmt(resnet_imbalance_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{interval(resnet_imbalance_worst, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')} "
                    f"at tail_train_per_class={int(resnet_imbalance_worst['tail_train_per_class'])}, "
                    f"and best tail accuracy is {fmt(resnet_imbalance_best_tail_accuracy['mean_tail_accuracy_before'])} "
                    f"{interval(resnet_imbalance_best_tail_accuracy, 'tail_accuracy_before_ci95_low', 'tail_accuracy_before_ci95_high')}; "
                    "tail-loss signs are mixed, so this remains local drift robustness evidence."
                ),
            },
            {"artifact": "discussion/e11_cifar100_resnet_imbalance_sweep.md", "role": "Markdown summary and CSV links for the CIFAR-100-LT ResNet18 tail-count imbalance sweep."},
            {
                "artifact": "figures/e11_cifar100_resnet_lt_standard_eval/cifar100_resnet_lt_standard_eval.png",
                "role": (
                    "Appendix standard CIFAR-100-LT IF=100 ResNet18 many/medium/few reporting baseline; "
                    f"balanced accuracies are {fmt(resnet_lt_many['mean_balanced_accuracy'])} "
                    f"{interval(resnet_lt_many, 'balanced_accuracy_ci95_low', 'balanced_accuracy_ci95_high')}, "
                    f"{fmt(resnet_lt_medium['mean_balanced_accuracy'])} "
                    f"{interval(resnet_lt_medium, 'balanced_accuracy_ci95_low', 'balanced_accuracy_ci95_high')}, and "
                    f"{fmt(resnet_lt_few['mean_balanced_accuracy'])} "
                    f"{interval(resnet_lt_few, 'balanced_accuracy_ci95_low', 'balanced_accuracy_ci95_high')}; "
                    "not a tuned optimizer benchmark."
                ),
            },
            {"artifact": "discussion/e11_cifar100_resnet_lt_standard_eval.md", "role": "Markdown summary and CSV links for the standard long-tail classification reporting baseline."},
            {
                "artifact": "figures/e11_cifar100_resnet_lt_recipe_benchmark/cifar100_resnet_lt_recipe_benchmark.png",
                "role": (
                    "Appendix CIFAR-100-LT IF=100 ResNet18 augmented recipe pilot; "
                    f"SGD-aug all/few balanced accuracies are {fmt(resnet_recipe_sgd_all['mean_balanced_accuracy'])} "
                    f"{interval(resnet_recipe_sgd_all, 'balanced_accuracy_ci95_low', 'balanced_accuracy_ci95_high')} and "
                    f"{fmt(resnet_recipe_sgd_few['mean_balanced_accuracy'])} "
                    f"{interval(resnet_recipe_sgd_few, 'balanced_accuracy_ci95_low', 'balanced_accuracy_ci95_high')}, "
                    f"with few-group diff vs AdamW-aug {fmt(resnet_recipe_sgd_few_diff['mean_balanced_accuracy_diff'])} "
                    f"{interval(resnet_recipe_sgd_few_diff, 'balanced_accuracy_diff_ci95_low', 'balanced_accuracy_diff_ci95_high')}; "
                    "pilot benchmark context only."
                ),
            },
            {"artifact": "discussion/e11_cifar100_resnet_lt_recipe_benchmark.md", "role": "Markdown summary and CSV links for the augmented recipe benchmark pilot."},
            {
                "artifact": "figures/e11_cifar100_resnet_lt_muon_final_benchmark/cifar100_resnet_lt_recipe_benchmark.png",
                "role": (
                    "Appendix CIFAR-100-LT ResNet18 NS-Muon final-training pilot; "
                    f"lr=1e-4 all/few balanced accuracies are {fmt(resnet_muon_final_lr1e4_all['mean_balanced_accuracy'])} "
                    f"{interval(resnet_muon_final_lr1e4_all, 'balanced_accuracy_ci95_low', 'balanced_accuracy_ci95_high')} and "
                    f"{fmt(resnet_muon_final_lr1e4_few['mean_balanced_accuracy'])} "
                    f"{interval(resnet_muon_final_lr1e4_few, 'balanced_accuracy_ci95_low', 'balanced_accuracy_ci95_high')}, "
                    f"with all/few diffs vs AdamW-aug {fmt(resnet_muon_final_lr1e4_all_diff['mean_balanced_accuracy_diff'])} "
                    f"{interval(resnet_muon_final_lr1e4_all_diff, 'balanced_accuracy_diff_ci95_low', 'balanced_accuracy_diff_ci95_high')} and "
                    f"{fmt(resnet_muon_final_lr1e4_few_diff['mean_balanced_accuracy_diff'])} "
                    f"{interval(resnet_muon_final_lr1e4_few_diff, 'balanced_accuracy_diff_ci95_low', 'balanced_accuracy_diff_ci95_high')}; "
                    "negative final-performance boundary for the tested finite-NS Muon recipe."
                ),
            },
            {"artifact": "discussion/e11_cifar100_resnet_lt_muon_final_benchmark.md", "role": "Markdown summary and CSV links for the NS-Muon final-training benchmark pilot."},
            {
                "artifact": "figures/e11_cifar100_resnet_practical_muon_bridge/cifar100_resnet_practical_muon_bridge.png",
                "role": (
                    "Appendix ResNet practical Muon bridge on AdamW- and NS-Muon-sampled matrix-weight states; "
                    f"NS(M_t) drift ratios are {fmt(resnet_adam_ns['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(resnet_adam_ns, 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')} and "
                    f"{fmt(resnet_muon_ns['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{interval(resnet_muon_ns, 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
            },
            {"artifact": "discussion/e11_cifar100_resnet_practical_muon_bridge.md", "role": "Markdown summary and CSV links for the ResNet practical Muon/AdamW trajectory bridge."},
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
