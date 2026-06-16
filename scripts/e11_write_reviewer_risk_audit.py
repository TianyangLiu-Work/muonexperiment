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
    cifar_resnet_checkpoint_sweep = pd.read_csv("results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv")
    cifar_resnet_condition_proxy = pd.read_csv(
        "results/e11_cifar100_resnet_condition_proxy_scatter/summary.csv"
    ).set_index(["comparison", "correlation"])
    cifar_resnet_fc_condition = pd.read_csv("results/e11_cifar100_resnet_fc_condition_scatter/summary.csv")
    cifar_resnet_fc_condition_points = pd.read_csv(
        "results/e11_cifar100_resnet_fc_condition_scatter/condition_metrics.csv"
    )
    cifar_resnet_tail_quality = pd.read_csv("results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv")
    cifar_resnet_imbalance_sweep = pd.read_csv(
        "results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv"
    )
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
    state_control = pd.read_csv(
        "results/e11_long_tail_muon_state_source_control/summary.csv"
    ).set_index(["state_source", "direction"])
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")

    positive = synthetic[synthetic["setting"].eq("high_head_rank_low_tail_srank")].iloc[0]
    negative = synthetic[synthetic["setting"].eq("low_head_rank_high_tail_srank")].iloc[0]
    cifar_resnet_checkpoint_worst = cifar_resnet_checkpoint_sweep.loc[
        cifar_resnet_checkpoint_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_checkpoint_best_tail_accuracy = cifar_resnet_checkpoint_sweep.loc[
        cifar_resnet_checkpoint_sweep["mean_tail_accuracy_before"].idxmax()
    ]
    cifar_resnet_rank_proxy_pearson = cifar_resnet_condition_proxy.loc[
        ("mean_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio", "pearson")
    ]
    cifar_resnet_fc_condition_worst = cifar_resnet_fc_condition.loc[
        cifar_resnet_fc_condition["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_fc_condition_weakest = cifar_resnet_fc_condition.loc[
        cifar_resnet_fc_condition["mean_condition_score_nrank_over_tail_srank"].idxmin()
    ]
    cifar_resnet_fc_condition_favors_fraction = (
        cifar_resnet_fc_condition_points["condition_score_nrank_over_tail_srank"] > 1.0
    ).mean()
    cifar_resnet_fc_condition_favors_count = int(
        (cifar_resnet_fc_condition_points["condition_score_nrank_over_tail_srank"] > 1.0).sum()
    )
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
    cifar_resnet_jvp_checkpoint_scaled = cifar_resnet_layer_jvp_checkpoint_prediction.loc[
        "source_scaled_jvp_ratio"
    ]
    cifar_resnet_jvp_checkpoint_observed = cifar_resnet_layer_jvp_checkpoint_prediction.loc[
        "source_observed_drift_ratio"
    ]
    cifar_resnet_jvp_checkpoint_early_layer = cifar_resnet_layer_jvp_checkpoint_prediction.loc[
        "architecture_early_layer_prior"
    ]
    cifar_resnet_lt_many = cifar_resnet_lt_standard_eval.loc["many"]
    cifar_resnet_lt_medium = cifar_resnet_lt_standard_eval.loc["medium"]
    cifar_resnet_lt_few = cifar_resnet_lt_standard_eval.loc["few"]
    cifar_resnet_recipe_sgd_all = cifar_resnet_lt_recipe.loc[("sgd_aug_ce", "all")]
    cifar_resnet_recipe_sgd_few = cifar_resnet_lt_recipe.loc[("sgd_aug_ce", "few")]
    cifar_resnet_recipe_sgd_few_diff = cifar_resnet_lt_recipe_pairs.loc[("sgd_aug_ce", "few")]
    cifar_resnet_muon_final_lr1e4_all = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "all")]
    cifar_resnet_muon_final_lr1e4_few = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "few")]
    cifar_resnet_muon_final_lr1e4_all_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "all")
    ]
    cifar_resnet_muon_final_lr1e4_few_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "few")
    ]
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
                    f"the ResNet checkpoint sweep worst drift CI upper endpoint is "
                    f"{fmt(cifar_resnet_checkpoint_worst['tail_output_drift_sq_ratio_ci95_high'])}, "
                    f"but best pre-update tail accuracy is only "
                    f"{fmt(cifar_resnet_checkpoint_best_tail_accuracy['mean_tail_accuracy_before'])}. "
                    f"The tail-rich ResNet control raises pre-update tail accuracy to "
                    f"{fmt(cifar_resnet_tail_quality_best_tail_accuracy['mean_tail_accuracy_before'])} "
                    f"CI={interval(cifar_resnet_tail_quality_best_tail_accuracy, 'tail_accuracy_before_ci95_low', 'tail_accuracy_before_ci95_high')} "
                    f"and keeps worst squared drift ratio at "
                    f"{fmt(cifar_resnet_tail_quality_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_tail_quality_worst, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"the CIFAR-100-LT ResNet18 imbalance sweep covers "
                    f"{int(cifar_resnet_imbalance_sweep['tail_train_per_class'].nunique())} tail-count settings, "
                    f"with worst squared drift ratio "
                    f"{fmt(cifar_resnet_imbalance_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_imbalance_worst, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')} "
                    f"and best pre-update tail accuracy "
                    f"{fmt(cifar_resnet_imbalance_best_tail_accuracy['mean_tail_accuracy_before'])} "
                    f"CI={interval(cifar_resnet_imbalance_best_tail_accuracy, 'tail_accuracy_before_ci95_low', 'tail_accuracy_before_ci95_high')}; "
                    f"the all-layer JVP tail-quality diagnostic gives observed squared drift ratio "
                    f"{fmt(cifar_resnet_layer_jvp['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_layer_jvp, 'observed_tail_drift_sq_ratio_ci95_low', 'observed_tail_drift_sq_ratio_ci95_high')}; "
                    f"but tail-accuracy-drop diff CI={interval(cifar_resnet, 'tail_accuracy_drop_diff_ci95_low', 'tail_accuracy_drop_diff_ci95_high')} crosses zero. "
                    f"The small practical training run has lower tail eval loss ratio="
                    f"{fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_loss_ratio_ci95_low', 'final_tail_eval_loss_ratio_ci95_high')}, "
                    f"higher tail margin diff={fmt(practical_training['mean_final_tail_eval_margin_diff_muon_minus_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_margin_diff_ci95_low', 'final_tail_eval_margin_diff_ci95_high')}, "
                    f"but tail accuracy diff={fmt(practical_training['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])} "
                    f"CI={interval(practical_training, 'final_tail_eval_accuracy_diff_ci95_low', 'final_tail_eval_accuracy_diff_ci95_high')}. "
                    f"The standard CIFAR-100-LT ResNet18 reporting baseline gives many/medium/few balanced accuracy "
                    f"{fmt(cifar_resnet_lt_many['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_lt_medium['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_lt_few['mean_balanced_accuracy'])}. "
                    f"The augmented recipe pilot gives SGD-aug all/few balanced accuracy "
                    f"{fmt(cifar_resnet_recipe_sgd_all['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_recipe_sgd_few['mean_balanced_accuracy'])}, "
                    f"with few diff vs AdamW-aug "
                    f"{fmt(cifar_resnet_recipe_sgd_few_diff['mean_balanced_accuracy_diff'])}. "
                    f"The NS-Muon final-training pilot is negative: lr=1e-4 all/few balanced accuracy "
                    f"{fmt(cifar_resnet_muon_final_lr1e4_all['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_muon_final_lr1e4_few['mean_balanced_accuracy'])}, "
                    f"with all/few paired diffs vs AdamW-aug "
                    f"{fmt(cifar_resnet_muon_final_lr1e4_all_diff['mean_balanced_accuracy_diff'])}/"
                    f"{fmt(cifar_resnet_muon_final_lr1e4_few_diff['mean_balanced_accuracy_diff'])}."
                ),
                "safe_response": "Make function drift the main measured quantity; use the tail-rich ResNet control, standard reporting baseline, and negative NS-Muon final pilot to address measurement-surface objections, while keeping practical tail-loss/margin evidence separate from tail accuracy.",
                "remaining_work": "Run retuned long-horizon optimizer benchmarks before making performance claims.",
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
                    f"ssrank={fmt(negative['mean_tail_downstream_aware_stable_rank'])}, squared drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}. "
                    f"On the ResNet checkpoint sweep, mean gradient nuclear rank alone has Pearson correlation "
                    f"{fmt(cifar_resnet_rank_proxy_pearson['estimate'])} "
                    f"CI={interval(cifar_resnet_rank_proxy_pearson, 'ci95_low', 'ci95_high')} with log squared drift ratio. "
                    f"For the final layer, nrank(G_H)/srank(H_T) favors spectral in "
                    f"{cifar_resnet_fc_condition_favors_count}/{len(cifar_resnet_fc_condition_points)} points "
                    f"(fraction {fmt(cifar_resnet_fc_condition_favors_fraction)}), "
                    f"with weakest mean score={fmt(cifar_resnet_fc_condition_weakest['mean_condition_score_nrank_over_tail_srank'])} "
                    f"and worst squared drift ratio={fmt(cifar_resnet_fc_condition_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_fc_condition_worst, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}. "
                    f"The all-layer ResNet JVP diagnostic covers {int(cifar_resnet_layer_jvp['parameters'])} Conv/Linear weights and "
                    f"{int(cifar_resnet_layer_jvp['paired_points'])} paired layer/seed points; "
                    f"{cifar_resnet_layer_jvp_supported_layers}/{int(cifar_resnet_layer_jvp['parameters'])} observed per-layer CI upper endpoints are below one. "
                    f"In the held-out checkpoint-transfer benchmark, source-observed and early-layer controls have Spearman "
                    f"{fmt(cifar_resnet_jvp_checkpoint_observed['mean_spearman_log_predictor_vs_log_target_observed'])} "
                    f"CI={interval(cifar_resnet_jvp_checkpoint_observed, 'spearman_ci95_low', 'spearman_ci95_high')} and "
                    f"{fmt(cifar_resnet_jvp_checkpoint_early_layer['mean_spearman_log_predictor_vs_log_target_observed'])} "
                    f"CI={interval(cifar_resnet_jvp_checkpoint_early_layer, 'spearman_ci95_low', 'spearman_ci95_high')}, "
                    f"while scaled-JVP has Spearman "
                    f"{fmt(cifar_resnet_jvp_checkpoint_scaled['mean_spearman_log_predictor_vs_log_target_observed'])} "
                    f"CI={interval(cifar_resnet_jvp_checkpoint_scaled, 'spearman_ci95_low', 'spearman_ci95_high')}."
                ),
                "safe_response": "Call the synthetic result a mechanism sanity check, use the ResNet rank-only proxy as a caveat, and present the final-layer plus all-layer JVP diagnostics as downstream-aware natural-task bridges.",
                "remaining_work": "Improve the downstream-aware score and repeat the held-out benchmark on architecture or dataset splits before claiming a general boundary predictor.",
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
                    f"CI={interval(cifar_resnet_rho002, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"a checkpoint sweep over 250, 500, 1000, and 2000 warmup steps has worst ratio="
                    f"{fmt(cifar_resnet_checkpoint_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={interval(cifar_resnet_checkpoint_worst, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}. "
                    f"A tail-rich control reaches tail accuracy="
                    f"{fmt(cifar_resnet_tail_quality_best_tail_accuracy['mean_tail_accuracy_before'])} "
                    f"and worst drift ratio={fmt(cifar_resnet_tail_quality_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])}. "
                    f"The CIFAR-100-LT ResNet18 imbalance sweep adds four explicit tail-count settings with worst drift CI upper endpoint "
                    f"{fmt(cifar_resnet_imbalance_worst['tail_output_drift_sq_ratio_ci95_high'])} and best tail accuracy "
                    f"{fmt(cifar_resnet_imbalance_best_tail_accuracy['mean_tail_accuracy_before'])}. "
                    f"The all-layer JVP diagnostic adds observed ratio="
                    f"{fmt(cifar_resnet_layer_jvp['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])} "
                    f"over {int(cifar_resnet_layer_jvp['paired_points'])} layer/seed pairs. "
                    f"The standard IF=100 ResNet18 reporting run adds final many/medium/few balanced accuracy "
                    f"{fmt(cifar_resnet_lt_many['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_lt_medium['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_lt_few['mean_balanced_accuracy'])}. "
                    f"The augmented recipe pilot adds SGD-momentum all/few balanced accuracy "
                    f"{fmt(cifar_resnet_recipe_sgd_all['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_recipe_sgd_few['mean_balanced_accuracy'])}. "
                    f"The NS-Muon final-training pilot adds a negative boundary: lr=1e-4 all/few balanced accuracy "
                    f"{fmt(cifar_resnet_muon_final_lr1e4_all['mean_balanced_accuracy'])}/"
                    f"{fmt(cifar_resnet_muon_final_lr1e4_few['mean_balanced_accuracy'])}."
                ),
                "safe_response": "Present the manuscript as a theory-and-diagnostic mechanism paper with architecture, tail-quality, standard reporting, and recipe-pilot controls, not a tuned long-tail optimizer benchmark paper.",
                "remaining_work": "Add ImageNet-LT or iNaturalist-style matched-head-gain diagnostics, wider NS-Muon schedules, and long-horizon practical baselines before claiming benchmark-level generality.",
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
                "reason": "The CIFAR-100-LT ResNet standard run, augmented recipe pilot, and negative NS-Muon final pilot are benchmark context, not a tuned multi-dataset Muon/AdamW long-horizon benchmark.",
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
- [CIFAR-100-LT ResNet18 checkpoint-quality sweep](e11_cifar100_resnet_checkpoint_sweep.md)
- [CIFAR-100-LT ResNet18 condition-proxy scatter](e11_cifar100_resnet_condition_proxy_scatter.md)
- [CIFAR-100-LT ResNet18 final-layer condition scatter](e11_cifar100_resnet_fc_condition_scatter.md)
- [CIFAR-100 ResNet18 tail-quality control](e11_cifar100_resnet_tail_quality_control.md)
- [CIFAR-100-LT ResNet18 imbalance sweep](e11_cifar100_resnet_imbalance_sweep.md)
- [CIFAR-100-LT ResNet18 all-layer JVP tail-quality diagnostic](e11_cifar100_resnet_layer_jvp_tail_quality.md)
- [CIFAR-100-LT ResNet18 standard many/medium/few evaluation](e11_cifar100_resnet_lt_standard_eval.md)
- [CIFAR-100-LT ResNet18 recipe benchmark pilot](e11_cifar100_resnet_lt_recipe_benchmark.md)
- [CIFAR-100-LT ResNet18 NS-Muon final-training benchmark pilot](e11_cifar100_resnet_lt_muon_final_benchmark.md)
- [artifact manifest](e11_artifact_manifest.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved reviewer risk audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
