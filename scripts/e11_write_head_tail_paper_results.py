from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("paper/specgrad_activation_paper/tables/head_tail_empirical_results.tex")


def fmt(value: float) -> str:
    value = float(value)
    if abs(value) >= 100 or (abs(value) < 1e-3 and value != 0.0):
        return f"{value:.2e}"
    return f"{value:.4g}"


def ci(mean: float, low: float, high: float) -> str:
    return f"{fmt(mean)} [{fmt(low)}, {fmt(high)}]"


def main() -> None:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    cifar_resnet = pd.read_csv("results/e11_cifar100_resnet_one_step/pair_summary.csv").iloc[0]
    cifar_resnet_rho002 = pd.read_csv("results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv").iloc[0]
    cifar_resnet_checkpoint_sweep = pd.read_csv("results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv")
    cifar_resnet_fc_condition = pd.read_csv("results/e11_cifar100_resnet_fc_condition_scatter/summary.csv")
    cifar_resnet_tail_quality = pd.read_csv("results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv")
    cifar_resnet_imbalance_sweep = pd.read_csv("results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv")
    cifar_resnet_layer_jvp = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv"
    ).iloc[0]
    cifar_resnet_layer_jvp_summary = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_tail_quality/summary.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_prediction = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/prediction_summary.csv"
    ).set_index("predictor")
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
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    cifar_resnet_practical_bridge = pd.read_csv(
        "results/e11_cifar100_resnet_practical_muon_bridge/summary.csv"
    ).set_index(["state_source", "direction"])
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")

    synthetic_positive = head_tail.set_index("setting").loc["high_head_rank_low_tail_srank"]
    synthetic_negative = head_tail.set_index("setting").loc["low_head_rank_high_tail_srank"]
    cifar_resnet_checkpoint_worst = cifar_resnet_checkpoint_sweep.loc[
        cifar_resnet_checkpoint_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_checkpoint_best_tail_accuracy = cifar_resnet_checkpoint_sweep.loc[
        cifar_resnet_checkpoint_sweep["mean_tail_accuracy_before"].idxmax()
    ]
    cifar_resnet_fc_condition_worst = cifar_resnet_fc_condition.loc[
        cifar_resnet_fc_condition["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_fc_condition_weakest = cifar_resnet_fc_condition.loc[
        cifar_resnet_fc_condition["mean_condition_score_nrank_over_tail_srank"].idxmin()
    ]
    cifar_resnet_tail_quality_worst = cifar_resnet_tail_quality.loc[
        cifar_resnet_tail_quality["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_tail_quality_best_tail_accuracy = cifar_resnet_tail_quality.loc[
        cifar_resnet_tail_quality["mean_tail_accuracy_before"].idxmax()
    ]
    cifar_resnet_imbalance_worst = cifar_resnet_imbalance_sweep.loc[
        cifar_resnet_imbalance_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_imbalance_best_tail_accuracy = cifar_resnet_imbalance_sweep.loc[
        cifar_resnet_imbalance_sweep["mean_tail_accuracy_before"].idxmax()
    ]
    cifar_resnet_layer_jvp_supported_layers = int(
        (cifar_resnet_layer_jvp_summary["observed_tail_drift_sq_ratio_ci95_high"] < 1.0).sum()
    )
    cifar_resnet_layer_jvp_checkpoint_scaled = cifar_resnet_layer_jvp_checkpoint_prediction.loc[
        "source_scaled_jvp_ratio"
    ]
    cifar_resnet_lt_many = cifar_resnet_lt_standard_eval.loc["many"]
    cifar_resnet_lt_medium = cifar_resnet_lt_standard_eval.loc["medium"]
    cifar_resnet_lt_few = cifar_resnet_lt_standard_eval.loc["few"]
    cifar_resnet_recipe_sgd_all = cifar_resnet_lt_recipe.loc[("sgd_aug_ce", "all")]
    cifar_resnet_recipe_sgd_few = cifar_resnet_lt_recipe.loc[("sgd_aug_ce", "few")]
    cifar_resnet_recipe_sgd_few_diff = cifar_resnet_lt_recipe_pairs.loc[("sgd_aug_ce", "few")]
    cifar_resnet_recipe_cb_few_diff = cifar_resnet_lt_recipe_pairs.loc[("adamw_aug_cb_loss", "few")]
    cifar_resnet_muon_final_adamw_all = cifar_resnet_lt_muon_final.loc[("adamw_aug_ce", "all")]
    cifar_resnet_muon_final_adamw_few = cifar_resnet_lt_muon_final.loc[("adamw_aug_ce", "few")]
    cifar_resnet_muon_final_lr1e4_all = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "all")]
    cifar_resnet_muon_final_lr1e4_few = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "few")]
    cifar_resnet_muon_final_lr1e4_all_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "all")
    ]
    cifar_resnet_muon_final_lr1e4_few_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "few")
    ]
    layer1 = layerwise.set_index("layer").loc[1]
    layer2 = layerwise.set_index("layer").loc[2]
    cifar_resnet_practical_adam_ns = cifar_resnet_practical_bridge.loc[
        ("adamw_matrix_trajectory", "ns_momentum")
    ]
    cifar_resnet_practical_muon_ns = cifar_resnet_practical_bridge.loc[
        ("ns_muon_matrix_trajectory", "ns_momentum")
    ]

    lines = [
        "% Auto-generated by scripts/e11_write_head_tail_paper_results.py. Do not edit by hand.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Head-to-tail empirical evidence summary. Except for the practical imbalanced-training and CIFAR-100-LT final-training rows, which report actual training outcomes, ratios are spectral/polar or Muon-style directions divided by Frobenius/GD. Values below $1$ mean lower perturbation of logits on held-out tail examples.}",
        "\\label{tab:head-tail-results}",
        "\\footnotesize",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.34\\linewidth}>{\\raggedright\\arraybackslash}p{0.30\\linewidth}}",
        "\\toprule",
        "Experiment & Main tail-drift result & Key caveat \\\\",
        "\\midrule",
        (
            "Synthetic positive boundary & "
            + ci(
                synthetic_positive["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                synthetic_positive["tail_output_drift_sq_ratio_ci95_low"],
                synthetic_positive["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " & $\\nrank(G_H)=8.61>\\ssrank(B_T,A_T)=1.00$ \\\\"
        ),
        (
            "Synthetic negative boundary & "
            + ci(
                synthetic_negative["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                synthetic_negative["tail_output_drift_sq_ratio_ci95_low"],
                synthetic_negative["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " & $\\nrank(G_H)=1.39<\\ssrank(B_T,A_T)=10.0$ \\\\"
        ),
        (
            "Long-tailed digits one-step & "
            + ci(
                one_step["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                one_step["tail_output_drift_sq_ratio_ci95_low"],
                one_step["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " & tail loss diff = "
            + ci(
                one_step["mean_tail_loss_increase_diff_spectral_minus_fro"],
                one_step["tail_loss_increase_diff_ci95_low"],
                one_step["tail_loss_increase_diff_ci95_high"],
            )
            + " \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 one-step & "
            + ci(
                cifar_resnet["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet["tail_output_drift_sq_ratio_ci95_low"],
                cifar_resnet["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " & tail loss diff = "
            + ci(
                cifar_resnet["mean_tail_loss_increase_diff_spectral_minus_fro"],
                cifar_resnet["tail_loss_increase_diff_ci95_low"],
                cifar_resnet["tail_loss_increase_diff_ci95_high"],
            )
            + "; rho=0.002 ratio "
            + ci(
                cifar_resnet_rho002["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet_rho002["tail_output_drift_sq_ratio_ci95_low"],
                cifar_resnet_rho002["tail_output_drift_sq_ratio_ci95_high"],
            )
            + "; accuracy diff CI crosses 0 \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 checkpoint sweep & worst drift "
            + ci(
                cifar_resnet_checkpoint_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet_checkpoint_worst["tail_output_drift_sq_ratio_ci95_low"],
                cifar_resnet_checkpoint_worst["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " at "
            + fmt(cifar_resnet_checkpoint_worst["warmup_steps"])
            + " steps & best pre-update tail accuracy "
            + fmt(cifar_resnet_checkpoint_best_tail_accuracy["mean_tail_accuracy_before"])
            + " at "
            + fmt(cifar_resnet_checkpoint_best_tail_accuracy["warmup_steps"])
            + " steps; still weak-tail-predictor evidence \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 final-layer condition & worst drift "
            + ci(
                cifar_resnet_fc_condition_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet_fc_condition_worst["tail_output_drift_sq_ratio_ci95_low"],
                cifar_resnet_fc_condition_worst["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " at "
            + fmt(cifar_resnet_fc_condition_worst["warmup_steps"])
            + " steps & weakest mean nrank/srank score "
            + fmt(cifar_resnet_fc_condition_weakest["mean_condition_score_nrank_over_tail_srank"])
            + "; final-layer-only downstream-aware check \\\\"
        ),
        (
            "CIFAR-100 ResNet18 tail-quality control & worst drift "
            + ci(
                cifar_resnet_tail_quality_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet_tail_quality_worst["tail_output_drift_sq_ratio_ci95_low"],
                cifar_resnet_tail_quality_worst["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " & best pre-update tail accuracy "
            + ci(
                cifar_resnet_tail_quality_best_tail_accuracy["mean_tail_accuracy_before"],
                cifar_resnet_tail_quality_best_tail_accuracy["tail_accuracy_before_ci95_low"],
                cifar_resnet_tail_quality_best_tail_accuracy["tail_accuracy_before_ci95_high"],
            )
            + "; tail-rich checkpoint control \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 imbalance sweep & worst drift "
            + ci(
                cifar_resnet_imbalance_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet_imbalance_worst["tail_output_drift_sq_ratio_ci95_low"],
                cifar_resnet_imbalance_worst["tail_output_drift_sq_ratio_ci95_high"],
            )
            + " at tail train/class "
            + fmt(cifar_resnet_imbalance_worst["tail_train_per_class"])
            + " & best pre-update tail accuracy "
            + ci(
                cifar_resnet_imbalance_best_tail_accuracy["mean_tail_accuracy_before"],
                cifar_resnet_imbalance_best_tail_accuracy["tail_accuracy_before_ci95_low"],
                cifar_resnet_imbalance_best_tail_accuracy["tail_accuracy_before_ci95_high"],
            )
            + " at tail train/class "
            + fmt(cifar_resnet_imbalance_best_tail_accuracy["tail_train_per_class"])
            + "; tail-loss evidence is mixed \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 all-layer JVP & observed "
            + ci(
                cifar_resnet_layer_jvp["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet_layer_jvp["observed_tail_drift_sq_ratio_ci95_low"],
                cifar_resnet_layer_jvp["observed_tail_drift_sq_ratio_ci95_high"],
            )
            + "; scaled JVP "
            + ci(
                cifar_resnet_layer_jvp["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"],
                cifar_resnet_layer_jvp["scaled_jvp_tail_drift_sq_ratio_ci95_low"],
                cifar_resnet_layer_jvp["scaled_jvp_tail_drift_sq_ratio_ci95_high"],
            )
            + " & "
            + fmt(cifar_resnet_layer_jvp_supported_layers)
            + "/"
            + fmt(cifar_resnet_layer_jvp["parameters"])
            + " layer CI upper endpoints below 1; tail-rich local JVP diagnostic \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 checkpoint-transfer JVP & scaled-JVP threshold accuracy "
            + fmt(cifar_resnet_layer_jvp_checkpoint_scaled["mean_threshold_below_one_accuracy"])
            + "; held-out Spearman "
            + ci(
                cifar_resnet_layer_jvp_checkpoint_scaled[
                    "mean_spearman_log_predictor_vs_log_target_observed"
                ],
                cifar_resnet_layer_jvp_checkpoint_scaled["spearman_ci95_low"],
                cifar_resnet_layer_jvp_checkpoint_scaled["spearman_ci95_high"],
            )
            + " & "
            + fmt(cifar_resnet_layer_jvp_checkpoint_scaled["checkpoint_transfer_pairs"])
            + " directed checkpoint-transfer pairs; layer ranking does not transfer \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 standard reporting & many/medium/few balanced acc. "
            + ci(
                cifar_resnet_lt_many["mean_balanced_accuracy"],
                cifar_resnet_lt_many["balanced_accuracy_ci95_low"],
                cifar_resnet_lt_many["balanced_accuracy_ci95_high"],
            )
            + " / "
            + ci(
                cifar_resnet_lt_medium["mean_balanced_accuracy"],
                cifar_resnet_lt_medium["balanced_accuracy_ci95_low"],
                cifar_resnet_lt_medium["balanced_accuracy_ci95_high"],
            )
            + " / "
            + ci(
                cifar_resnet_lt_few["mean_balanced_accuracy"],
                cifar_resnet_lt_few["balanced_accuracy_ci95_low"],
                cifar_resnet_lt_few["balanced_accuracy_ci95_high"],
            )
            + " & IF=100 AdamW ResNet18 reporting baseline; no augmentation, tuning, or Muon comparison \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 recipe pilot & SGD-aug all/few balanced acc. "
            + ci(
                cifar_resnet_recipe_sgd_all["mean_balanced_accuracy"],
                cifar_resnet_recipe_sgd_all["balanced_accuracy_ci95_low"],
                cifar_resnet_recipe_sgd_all["balanced_accuracy_ci95_high"],
            )
            + " / "
            + ci(
                cifar_resnet_recipe_sgd_few["mean_balanced_accuracy"],
                cifar_resnet_recipe_sgd_few["balanced_accuracy_ci95_low"],
                cifar_resnet_recipe_sgd_few["balanced_accuracy_ci95_high"],
            )
            + " & few diff vs AdamW-aug "
            + ci(
                cifar_resnet_recipe_sgd_few_diff["mean_balanced_accuracy_diff"],
                cifar_resnet_recipe_sgd_few_diff["balanced_accuracy_diff_ci95_low"],
                cifar_resnet_recipe_sgd_few_diff["balanced_accuracy_diff_ci95_high"],
            )
            + "; class-balanced AdamW few diff "
            + ci(
                cifar_resnet_recipe_cb_few_diff["mean_balanced_accuracy_diff"],
                cifar_resnet_recipe_cb_few_diff["balanced_accuracy_diff_ci95_low"],
                cifar_resnet_recipe_cb_few_diff["balanced_accuracy_diff_ci95_high"],
            )
            + " \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 NS-Muon final pilot & NS-Muon lr=$10^{-4}$ all/few balanced acc. "
            + ci(
                cifar_resnet_muon_final_lr1e4_all["mean_balanced_accuracy"],
                cifar_resnet_muon_final_lr1e4_all["balanced_accuracy_ci95_low"],
                cifar_resnet_muon_final_lr1e4_all["balanced_accuracy_ci95_high"],
            )
            + " / "
            + ci(
                cifar_resnet_muon_final_lr1e4_few["mean_balanced_accuracy"],
                cifar_resnet_muon_final_lr1e4_few["balanced_accuracy_ci95_low"],
                cifar_resnet_muon_final_lr1e4_few["balanced_accuracy_ci95_high"],
            )
            + " & AdamW-aug all/few "
            + ci(
                cifar_resnet_muon_final_adamw_all["mean_balanced_accuracy"],
                cifar_resnet_muon_final_adamw_all["balanced_accuracy_ci95_low"],
                cifar_resnet_muon_final_adamw_all["balanced_accuracy_ci95_high"],
            )
            + " / "
            + ci(
                cifar_resnet_muon_final_adamw_few["mean_balanced_accuracy"],
                cifar_resnet_muon_final_adamw_few["balanced_accuracy_ci95_low"],
                cifar_resnet_muon_final_adamw_few["balanced_accuracy_ci95_high"],
            )
            + "; all/few diff "
            + ci(
                cifar_resnet_muon_final_lr1e4_all_diff["mean_balanced_accuracy_diff"],
                cifar_resnet_muon_final_lr1e4_all_diff["balanced_accuracy_diff_ci95_low"],
                cifar_resnet_muon_final_lr1e4_all_diff["balanced_accuracy_diff_ci95_high"],
            )
            + " / "
            + ci(
                cifar_resnet_muon_final_lr1e4_few_diff["mean_balanced_accuracy_diff"],
                cifar_resnet_muon_final_lr1e4_few_diff["balanced_accuracy_diff_ci95_low"],
                cifar_resnet_muon_final_lr1e4_few_diff["balanced_accuracy_diff_ci95_high"],
            )
            + "; negative final-performance boundary \\\\"
        ),
        (
            "CIFAR-100-LT ResNet18 practical Muon bridge & AdamW-state NS$(M_t)$ "
            + ci(
                cifar_resnet_practical_adam_ns["geomean_tail_output_drift_sq_ratio_vs_fro"],
                cifar_resnet_practical_adam_ns["tail_output_drift_sq_ratio_vs_fro_ci95_low"],
                cifar_resnet_practical_adam_ns["tail_output_drift_sq_ratio_vs_fro_ci95_high"],
            )
            + "; NS-Muon-state NS$(M_t)$ "
            + ci(
                cifar_resnet_practical_muon_ns["geomean_tail_output_drift_sq_ratio_vs_fro"],
                cifar_resnet_practical_muon_ns["tail_output_drift_sq_ratio_vs_fro_ci95_low"],
                cifar_resnet_practical_muon_ns["tail_output_drift_sq_ratio_vs_fro_ci95_high"],
            )
            + " & "
            + fmt(cifar_resnet_practical_adam_ns["comparisons"])
            + " comparisons per state source/direction; local trajectory-state compatibility only \\\\"
        ),
        (
            "8-step head-only forgetting & "
            + ci(
                forgetting["geomean_final_tail_output_drift_sq_ratio_spectral_over_fro"],
                forgetting["final_tail_output_drift_sq_ratio_ci95_low"],
                forgetting["final_tail_output_drift_sq_ratio_ci95_high"],
            )
            + "; area "
            + ci(
                forgetting["geomean_tail_output_drift_area_ratio_spectral_over_fro"],
                forgetting["tail_output_drift_area_ratio_ci95_low"],
                forgetting["tail_output_drift_area_ratio_ci95_high"],
            )
            + " & tail loss/margin diff CI crosses 0 \\\\"
        ),
        (
            "Muon-style compatibility: $\\polar(M_t)$ / NS$(M_t)$ & "
            + ci(
                muon_bridge.loc["polar_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"],
                muon_bridge.loc["polar_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_low"],
                muon_bridge.loc["polar_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_high"],
            )
            + " / "
            + ci(
                muon_bridge.loc["ns_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"],
                muon_bridge.loc["ns_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_low"],
                muon_bridge.loc["ns_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_high"],
            )
            + " & NS$(M_t)$ CI crosses 1; selected-state check only \\\\"
        ),
        (
            "Practical-Muon trajectory compatibility & "
            + ci(
                practical_bridge.loc["polar_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"],
                practical_bridge.loc["polar_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_low"],
                practical_bridge.loc["polar_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_high"],
            )
            + " / "
            + ci(
                practical_bridge.loc["ns_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"],
                practical_bridge.loc["ns_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_low"],
                practical_bridge.loc["ns_momentum", "tail_output_drift_sq_ratio_vs_fro_ci95_high"],
            )
            + " & 120 state-step comparisons; still local matched-head-gain \\\\"
        ),
        (
            "Practical imbalanced training & "
            + "train "
            + ci(
                practical_training["geomean_final_train_loss_ratio_muon_over_adam"],
                practical_training["final_train_loss_ratio_ci95_low"],
                practical_training["final_train_loss_ratio_ci95_high"],
            )
            + "; tail loss "
            + ci(
                practical_training["geomean_final_tail_eval_loss_ratio_muon_over_adam"],
                practical_training["final_tail_eval_loss_ratio_ci95_low"],
                practical_training["final_tail_eval_loss_ratio_ci95_high"],
            )
            + " & tail accuracy diff = "
            + ci(
                practical_training["mean_final_tail_eval_accuracy_diff_muon_minus_adam"],
                practical_training["final_tail_eval_accuracy_diff_ci95_low"],
                practical_training["final_tail_eval_accuracy_diff_ci95_high"],
            )
            + "; fixed lightweight hyperparameters \\\\"
        ),
        (
            "Layer 1 scaled JVP / observed & "
            + ci(
                layer1["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"],
                layer1["scaled_jvp_tail_drift_sq_ratio_ci95_low"],
                layer1["scaled_jvp_tail_drift_sq_ratio_ci95_high"],
            )
            + " / "
            + ci(
                layer1["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"],
                layer1["observed_tail_drift_sq_ratio_ci95_low"],
                layer1["observed_tail_drift_sq_ratio_ci95_high"],
            )
            + " & unit-JVP squared drift ratio above one before head-gain scaling \\\\"
        ),
        (
            "Layer 2 scaled JVP / observed & "
            + ci(
                layer2["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"],
                layer2["scaled_jvp_tail_drift_sq_ratio_ci95_low"],
                layer2["scaled_jvp_tail_drift_sq_ratio_ci95_high"],
            )
            + " / "
            + ci(
                layer2["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"],
                layer2["observed_tail_drift_sq_ratio_ci95_low"],
                layer2["observed_tail_drift_sq_ratio_ci95_high"],
            )
            + " & unit-JVP squared drift ratio above one before head-gain scaling \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
