from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt


DISCUSSION_OUTPUT_PATH = Path("discussion/e11_paper_numbers.tex")
PAPER_OUTPUT_PATH = Path("paper/specgrad_activation_paper/tables/e11_paper_numbers.tex")


def macro(name: str, value: object) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def ci_text(row: pd.Series, low: str, high: str) -> str:
    return f"\\ensuremath{{\\text{{[{fmt(row[low])},\\hspace{{0.35em}}{fmt(row[high])}]}}}}"


def ci_macros(prefix: str, row: pd.Series, low: str, high: str) -> list[str]:
    return [
        macro(f"{prefix}CiLow", fmt(row[low])),
        macro(f"{prefix}CiHigh", fmt(row[high])),
        macro(f"{prefix}Ci", ci_text(row, low, high)),
    ]


def paired_ratio_summary(step_metrics: pd.DataFrame, numerator: str, denominator: str, column: str) -> pd.Series:
    wide = step_metrics.pivot(index="seed", columns="geometry", values=column)
    ratio = wide[numerator] / wide[denominator]
    log_ratio = ratio.map(math.log)
    mean = float(log_ratio.mean())
    sem = float(log_ratio.std(ddof=1) / math.sqrt(len(log_ratio)))
    return pd.Series(
        {
            "geomean": math.exp(mean),
            "ci95_low": math.exp(mean - 1.96 * sem),
            "ci95_high": math.exp(mean + 1.96 * sem),
        }
    )


def latex_sci(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    return f"{mantissa:.{digits}g}\\times 10^{{{exponent}}}"


def latex_math_fragment(value: object) -> str:
    text = str(value)
    if text.startswith("$") and text.endswith("$"):
        return text[1:-1]
    return text


def latex_texttt(value: object) -> str:
    text = str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_")
    return f"\\texttt{{{text}}}"


def main() -> None:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    head_tail_alignment = pd.read_csv("results/e11_head_tail_alignment_ablation/summary.csv").set_index("setting")
    one_step_metrics = pd.read_csv("results/e11_long_tail_one_step/step_metrics.csv")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    cifar_resnet = pd.read_csv("results/e11_cifar100_resnet_one_step/pair_summary.csv").iloc[0]
    cifar_resnet_rho002 = pd.read_csv("results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv").iloc[0]
    cifar_resnet_checkpoint_sweep = pd.read_csv(
        "results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv"
    )
    cifar_resnet_condition_proxy_points = pd.read_csv(
        "results/e11_cifar100_resnet_condition_proxy_scatter/scatter_points.csv"
    )
    cifar_resnet_condition_proxy = pd.read_csv(
        "results/e11_cifar100_resnet_condition_proxy_scatter/summary.csv"
    ).set_index(["comparison", "correlation"])
    cifar_resnet_fc_condition_summary = pd.read_csv(
        "results/e11_cifar100_resnet_fc_condition_scatter/summary.csv"
    )
    cifar_resnet_fc_condition_points = pd.read_csv(
        "results/e11_cifar100_resnet_fc_condition_scatter/condition_metrics.csv"
    )
    cifar_resnet_tail_quality = pd.read_csv(
        "results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv"
    )
    cifar_resnet_imbalance_sweep = pd.read_csv(
        "results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv"
    )
    cifar_resnet_layer_jvp = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv"
    ).iloc[0]
    cifar_resnet_layer_jvp_summary = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_tail_quality/summary.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_summary = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/checkpoint_summary.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_layer_summary = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_prediction_pairs = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/prediction_pairs.csv"
    )
    cifar_resnet_layer_jvp_checkpoint_prediction_summary = pd.read_csv(
        "results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/prediction_summary.csv"
    ).set_index("predictor")
    cifar_resnet_lt_standard_eval = pd.read_csv(
        "results/e11_cifar100_resnet_lt_standard_eval/summary.csv"
    ).set_index("frequency_group")
    cifar_resnet_lt_recipe_benchmark = pd.read_csv(
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
    imbalance = pd.read_csv("results/e11_long_tail_imbalance_ablation/summary.csv")
    checkpoint_sweep = pd.read_csv("results/e11_long_tail_checkpoint_sweep/summary.csv")
    class_partition_sweep = pd.read_csv("results/e11_long_tail_class_partition_sweep/summary.csv")
    rho_sweep = pd.read_csv("results/e11_long_tail_rho_sweep/summary.csv")
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    cifar_resnet_practical_bridge = pd.read_csv(
        "results/e11_cifar100_resnet_practical_muon_bridge/summary.csv"
    ).set_index(["state_source", "direction"])
    state_source_control = pd.read_csv(
        "results/e11_long_tail_muon_state_source_control/summary.csv"
    ).set_index(["state_source", "direction"])
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    practical_lr_sweep = pd.read_csv("results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv")
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")
    local_linearization = pd.read_csv("results/e11_local_linearization/summary.csv")

    positive = head_tail.loc["high_head_rank_low_tail_srank"]
    negative = head_tail.loc["low_head_rank_high_tail_srank"]
    alignment_positive = head_tail_alignment.loc["high_head_rank_low_tail_srank"]
    imbalance_worst = imbalance.loc[imbalance["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    imbalance_default = imbalance[imbalance["tail_train_per_class"].eq(40)].iloc[0]
    imbalance_strong = imbalance[imbalance["tail_train_per_class"].eq(10)].iloc[0]
    checkpoint_worst = checkpoint_sweep.loc[checkpoint_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    checkpoint_margin_worst = checkpoint_sweep.loc[checkpoint_sweep["margin_delta_sq_ratio_ci95_high"].idxmax()]
    checkpoint_tail_accuracy_min = checkpoint_sweep["mean_tail_accuracy_before"].min()
    checkpoint_tail_accuracy_max = checkpoint_sweep["mean_tail_accuracy_before"].max()
    checkpoint_positive_margin_min = checkpoint_sweep["mean_tail_positive_margin_fraction_before"].min()
    checkpoint_positive_margin_max = checkpoint_sweep["mean_tail_positive_margin_fraction_before"].max()
    class_partition_worst = class_partition_sweep.loc[
        class_partition_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    class_partition_centered_worst = class_partition_sweep.loc[
        class_partition_sweep["centered_tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    rho_worst = rho_sweep.loc[rho_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    rho_centered_worst = rho_sweep.loc[rho_sweep["centered_tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    rho_head_gain_error_high = rho_sweep[
        [
            "actual_head_gain_relative_error_frobenius_ci95_high",
            "actual_head_gain_relative_error_spectral_ci95_high",
        ]
    ].max(axis=1)
    rho_head_gain_error_worst = rho_sweep.loc[rho_head_gain_error_high.idxmax()]
    rho_head_gain_error_worst_value = float(rho_head_gain_error_high.loc[rho_head_gain_error_high.idxmax()])
    layer_one = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_two = layerwise[layerwise["layer"].eq(2)].iloc[0]
    lr_0003 = practical_lr_sweep[practical_lr_sweep["muon_lr"].round(3).eq(0.003)].iloc[0]
    lr_001 = practical_lr_sweep[practical_lr_sweep["muon_lr"].round(3).eq(0.010)].iloc[0]
    lr_01 = practical_lr_sweep[practical_lr_sweep["muon_lr"].round(3).eq(0.100)].iloc[0]
    local_linearization_worst = local_linearization.loc[
        local_linearization["relative_error_ci95_high"].idxmax()
    ]
    cifar_resnet_checkpoint_worst = cifar_resnet_checkpoint_sweep.loc[
        cifar_resnet_checkpoint_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_checkpoint_best_tail_accuracy = cifar_resnet_checkpoint_sweep.loc[
        cifar_resnet_checkpoint_sweep["mean_tail_accuracy_before"].idxmax()
    ]
    cifar_resnet_checkpoint_tail_accuracy_min = cifar_resnet_checkpoint_sweep[
        "mean_tail_accuracy_before"
    ].min()
    cifar_resnet_checkpoint_tail_accuracy_max = cifar_resnet_checkpoint_sweep[
        "mean_tail_accuracy_before"
    ].max()
    cifar_resnet_checkpoint_positive_margin_min = cifar_resnet_checkpoint_sweep[
        "mean_tail_positive_margin_fraction_before"
    ].min()
    cifar_resnet_checkpoint_positive_margin_max = cifar_resnet_checkpoint_sweep[
        "mean_tail_positive_margin_fraction_before"
    ].max()
    cifar_resnet_condition_rank_pearson = cifar_resnet_condition_proxy.loc[
        ("mean_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio", "pearson")
    ]
    cifar_resnet_condition_rank_spearman = cifar_resnet_condition_proxy.loc[
        ("mean_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio", "spearman")
    ]
    cifar_resnet_condition_tail_accuracy_spearman = cifar_resnet_condition_proxy.loc[
        ("tail_accuracy_before_vs_log_tail_drift_sq_ratio", "spearman")
    ]
    cifar_resnet_fc_condition_worst = cifar_resnet_fc_condition_summary.loc[
        cifar_resnet_fc_condition_summary["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_fc_condition_weakest_mean = cifar_resnet_fc_condition_summary.loc[
        cifar_resnet_fc_condition_summary["mean_condition_score_nrank_over_tail_srank"].idxmin()
    ]
    cifar_resnet_fc_condition_weakest_point = cifar_resnet_fc_condition_points.loc[
        cifar_resnet_fc_condition_points["condition_score_nrank_over_tail_srank"].idxmin()
    ]
    cifar_resnet_fc_condition_worst_theory = cifar_resnet_fc_condition_summary.loc[
        cifar_resnet_fc_condition_summary["mean_theory_ratio_tail_srank_over_nrank"].idxmax()
    ]
    cifar_resnet_fc_condition_favors_fraction = (
        cifar_resnet_fc_condition_points["condition_score_nrank_over_tail_srank"] > 1.0
    ).mean()
    cifar_resnet_tail_quality_worst = cifar_resnet_tail_quality.loc[
        cifar_resnet_tail_quality["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_tail_quality_best_tail_accuracy = cifar_resnet_tail_quality.loc[
        cifar_resnet_tail_quality["mean_tail_accuracy_before"].idxmax()
    ]
    cifar_resnet_tail_quality_tail_accuracy_min = cifar_resnet_tail_quality[
        "mean_tail_accuracy_before"
    ].min()
    cifar_resnet_tail_quality_tail_accuracy_max = cifar_resnet_tail_quality[
        "mean_tail_accuracy_before"
    ].max()
    cifar_resnet_imbalance_worst = cifar_resnet_imbalance_sweep.loc[
        cifar_resnet_imbalance_sweep["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_imbalance_best_tail_accuracy = cifar_resnet_imbalance_sweep.loc[
        cifar_resnet_imbalance_sweep["mean_tail_accuracy_before"].idxmax()
    ]
    cifar_resnet_imbalance_settings = int(cifar_resnet_imbalance_sweep["tail_train_per_class"].nunique())
    cifar_resnet_layer_jvp_worst_observed = cifar_resnet_layer_jvp_summary.loc[
        cifar_resnet_layer_jvp_summary["observed_tail_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_layer_jvp_worst_scaled = cifar_resnet_layer_jvp_summary.loc[
        cifar_resnet_layer_jvp_summary["scaled_jvp_tail_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_layer_jvp_supported_layers = int(
        (cifar_resnet_layer_jvp_summary["observed_tail_drift_sq_ratio_ci95_high"] < 1.0).sum()
    )
    cifar_resnet_layer_jvp_checkpoint_scaled = cifar_resnet_layer_jvp_checkpoint_prediction_summary.loc[
        "source_scaled_jvp_ratio"
    ]
    cifar_resnet_layer_jvp_checkpoint_observed = cifar_resnet_layer_jvp_checkpoint_prediction_summary.loc[
        "source_observed_drift_ratio"
    ]
    cifar_resnet_layer_jvp_checkpoint_early_layer = cifar_resnet_layer_jvp_checkpoint_prediction_summary.loc[
        "architecture_early_layer_prior"
    ]
    cifar_resnet_layer_jvp_checkpoint_unit = cifar_resnet_layer_jvp_checkpoint_prediction_summary.loc[
        "source_unit_jvp_ratio"
    ]
    cifar_resnet_layer_jvp_checkpoint_rank = cifar_resnet_layer_jvp_checkpoint_prediction_summary.loc[
        "source_gradient_nuclear_rank"
    ]
    cifar_resnet_lt_standard_many = cifar_resnet_lt_standard_eval.loc["many"]
    cifar_resnet_lt_standard_medium = cifar_resnet_lt_standard_eval.loc["medium"]
    cifar_resnet_lt_standard_few = cifar_resnet_lt_standard_eval.loc["few"]
    cifar_resnet_lt_standard_all = cifar_resnet_lt_standard_eval.loc["all"]
    cifar_resnet_lt_recipe_adamw_aug_many = cifar_resnet_lt_recipe_benchmark.loc[("adamw_aug_ce", "many")]
    cifar_resnet_lt_recipe_adamw_aug_medium = cifar_resnet_lt_recipe_benchmark.loc[("adamw_aug_ce", "medium")]
    cifar_resnet_lt_recipe_adamw_aug_few = cifar_resnet_lt_recipe_benchmark.loc[("adamw_aug_ce", "few")]
    cifar_resnet_lt_recipe_adamw_aug_all = cifar_resnet_lt_recipe_benchmark.loc[("adamw_aug_ce", "all")]
    cifar_resnet_lt_recipe_cb_few = cifar_resnet_lt_recipe_benchmark.loc[("adamw_aug_cb_loss", "few")]
    cifar_resnet_lt_recipe_sgd_many = cifar_resnet_lt_recipe_benchmark.loc[("sgd_aug_ce", "many")]
    cifar_resnet_lt_recipe_sgd_medium = cifar_resnet_lt_recipe_benchmark.loc[("sgd_aug_ce", "medium")]
    cifar_resnet_lt_recipe_sgd_few = cifar_resnet_lt_recipe_benchmark.loc[("sgd_aug_ce", "few")]
    cifar_resnet_lt_recipe_sgd_all = cifar_resnet_lt_recipe_benchmark.loc[("sgd_aug_ce", "all")]
    cifar_resnet_lt_recipe_sgd_few_diff = cifar_resnet_lt_recipe_pairs.loc[("sgd_aug_ce", "few")]
    cifar_resnet_lt_recipe_sgd_all_diff = cifar_resnet_lt_recipe_pairs.loc[("sgd_aug_ce", "all")]
    cifar_resnet_lt_recipe_cb_few_diff = cifar_resnet_lt_recipe_pairs.loc[("adamw_aug_cb_loss", "few")]
    cifar_resnet_lt_muon_final_adamw_all = cifar_resnet_lt_muon_final.loc[("adamw_aug_ce", "all")]
    cifar_resnet_lt_muon_final_adamw_few = cifar_resnet_lt_muon_final.loc[("adamw_aug_ce", "few")]
    cifar_resnet_lt_muon_final_lr1e4_all = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "all")]
    cifar_resnet_lt_muon_final_lr1e4_few = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr1e-4", "few")]
    cifar_resnet_lt_muon_final_lr3e5_all = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr3e-5", "all")]
    cifar_resnet_lt_muon_final_lr3e5_few = cifar_resnet_lt_muon_final.loc[("ns_muon_aug_lr3e-5", "few")]
    cifar_resnet_lt_muon_final_lr1e4_all_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "all")
    ]
    cifar_resnet_lt_muon_final_lr1e4_few_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr1e-4", "few")
    ]
    cifar_resnet_lt_muon_final_lr3e5_all_diff = cifar_resnet_lt_muon_final_pairs.loc[
        ("ns_muon_aug_lr3e-5", "all")
    ]
    cifar_resnet_practical_adam_polar_momentum = cifar_resnet_practical_bridge.loc[
        ("adamw_matrix_trajectory", "polar_momentum")
    ]
    cifar_resnet_practical_adam_ns_momentum = cifar_resnet_practical_bridge.loc[
        ("adamw_matrix_trajectory", "ns_momentum")
    ]
    cifar_resnet_practical_muon_polar_momentum = cifar_resnet_practical_bridge.loc[
        ("ns_muon_matrix_trajectory", "polar_momentum")
    ]
    cifar_resnet_practical_muon_ns_momentum = cifar_resnet_practical_bridge.loc[
        ("ns_muon_matrix_trajectory", "ns_momentum")
    ]
    one_step_alignment_ratio = paired_ratio_summary(one_step_metrics, "spectral", "frobenius", "alignment")
    one_step_update_fro_ratio = paired_ratio_summary(one_step_metrics, "spectral", "frobenius", "update_fro_norm")
    one_step_update_op_ratio = paired_ratio_summary(one_step_metrics, "spectral", "frobenius", "update_op_norm")

    lines = [
        "% Auto-generated by scripts/e11_write_paper_numbers.py.",
        "% Do not edit by hand; regenerate after updating results.",
        "% Scope: current head-to-tail interference paper numbers only.",
        "",
        "% Synthetic head-to-tail boundary",
        macro("EelevenHeadTailPositiveNrankG", fmt(positive["mean_head_gradient_nuclear_rank"])),
        macro("EelevenHeadTailPositiveSsrankBTA", fmt(positive["mean_tail_downstream_aware_stable_rank"])),
        macro(
            "EelevenHeadTailPositiveDriftRatio",
            fmt(positive["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenHeadTailPositiveDriftRatio",
            positive,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro("EelevenHeadTailNegativeNrankG", fmt(negative["mean_head_gradient_nuclear_rank"])),
        macro("EelevenHeadTailNegativeSsrankBTA", fmt(negative["mean_tail_downstream_aware_stable_rank"])),
        macro(
            "EelevenHeadTailNegativeDriftRatio",
            fmt(negative["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenHeadTailNegativeDriftRatio",
            negative,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        "",
        "% Synthetic singular-vector alignment ablation",
        macro("EelevenHeadTailAlignmentAblationSeeds", int(alignment_positive["seeds"])),
        macro(
            "EelevenHeadTailAlignmentPositiveTheoryRatio",
            fmt(alignment_positive["mean_theory_ratio_spectral_over_fro"]),
        ),
        macro(
            "EelevenHeadTailAlignmentPositiveDriftRatio",
            fmt(alignment_positive["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenHeadTailAlignmentPositiveDriftRatio",
            alignment_positive,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenHeadTailAlignmentPositiveMedianDriftRatio",
            fmt(alignment_positive["median_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        macro(
            "EelevenHeadTailAlignmentPositiveSpectralLowerFraction",
            fmt(alignment_positive["spectral_less_tail_output_drift_fraction"]),
        ),
        "",
        "% Long-tailed one-step diagnostic",
        macro("EelevenLongTailOneStepSeeds", int(one_step["seeds"])),
        macro(
            "EelevenLongTailOneStepDriftRatio",
            fmt(one_step["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepDriftRatio",
            one_step,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepCenteredDriftRatio",
            fmt(one_step["geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepCenteredDriftRatio",
            one_step,
            "centered_tail_output_drift_sq_ratio_ci95_low",
            "centered_tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTrueLogitDriftRatio",
            fmt(one_step["geomean_true_logit_delta_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTrueLogitDriftRatio",
            one_step,
            "true_logit_delta_sq_ratio_ci95_low",
            "true_logit_delta_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepCompetitorLogitDriftRatio",
            fmt(one_step["geomean_competitor_logit_delta_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepCompetitorLogitDriftRatio",
            one_step,
            "competitor_logit_delta_sq_ratio_ci95_low",
            "competitor_logit_delta_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepMarginDeltaRatio",
            fmt(one_step["geomean_margin_delta_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepMarginDeltaRatio",
            one_step,
            "margin_delta_sq_ratio_ci95_low",
            "margin_delta_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepSpectralLowerFraction",
            fmt(one_step["spectral_less_tail_output_drift_fraction"]),
        ),
        macro(
            "EelevenLongTailOneStepTailLossDiff",
            fmt(one_step["mean_tail_loss_increase_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailLossDiff",
            one_step,
            "tail_loss_increase_diff_ci95_low",
            "tail_loss_increase_diff_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailLossIncreaseFro",
            fmt(one_step["mean_tail_loss_increase_frobenius"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailLossIncreaseFro",
            one_step,
            "tail_loss_increase_frobenius_ci95_low",
            "tail_loss_increase_frobenius_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailLossIncreaseSpectral",
            fmt(one_step["mean_tail_loss_increase_spectral"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailLossIncreaseSpectral",
            one_step,
            "tail_loss_increase_spectral_ci95_low",
            "tail_loss_increase_spectral_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailMarginDropDiff",
            fmt(one_step["mean_tail_margin_drop_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailMarginDropDiff",
            one_step,
            "tail_margin_drop_diff_ci95_low",
            "tail_margin_drop_diff_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailMarginDropFro",
            fmt(one_step["mean_tail_margin_drop_frobenius"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailMarginDropFro",
            one_step,
            "tail_margin_drop_frobenius_ci95_low",
            "tail_margin_drop_frobenius_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailMarginDropSpectral",
            fmt(one_step["mean_tail_margin_drop_spectral"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailMarginDropSpectral",
            one_step,
            "tail_margin_drop_spectral_ci95_low",
            "tail_margin_drop_spectral_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailAccuracyDropDiff",
            fmt(one_step["mean_tail_accuracy_drop_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailAccuracyDropDiff",
            one_step,
            "tail_accuracy_drop_diff_ci95_low",
            "tail_accuracy_drop_diff_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailAccuracyDropFro",
            fmt(one_step["mean_tail_accuracy_drop_frobenius"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailAccuracyDropFro",
            one_step,
            "tail_accuracy_drop_frobenius_ci95_low",
            "tail_accuracy_drop_frobenius_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepTailAccuracyDropSpectral",
            fmt(one_step["mean_tail_accuracy_drop_spectral"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepTailAccuracyDropSpectral",
            one_step,
            "tail_accuracy_drop_spectral_ci95_low",
            "tail_accuracy_drop_spectral_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepHeadGainErrorFro",
            fmt(one_step["mean_actual_head_gain_relative_error_frobenius"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepHeadGainErrorFro",
            one_step,
            "actual_head_gain_relative_error_frobenius_ci95_low",
            "actual_head_gain_relative_error_frobenius_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepHeadGainErrorSpectral",
            fmt(one_step["mean_actual_head_gain_relative_error_spectral"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepHeadGainErrorSpectral",
            one_step,
            "actual_head_gain_relative_error_spectral_ci95_low",
            "actual_head_gain_relative_error_spectral_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepAlignmentRatio",
            fmt(one_step_alignment_ratio["geomean"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepAlignmentRatio",
            one_step_alignment_ratio,
            "ci95_low",
            "ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepUpdateFroNormRatio",
            fmt(one_step_update_fro_ratio["geomean"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepUpdateFroNormRatio",
            one_step_update_fro_ratio,
            "ci95_low",
            "ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepUpdateOpNormRatio",
            fmt(one_step_update_op_ratio["geomean"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepUpdateOpNormRatio",
            one_step_update_op_ratio,
            "ci95_low",
            "ci95_high",
        ),
        macro("EelevenLongTailOneStepTailLossBefore", fmt(one_step["mean_tail_loss_before"])),
        *ci_macros(
            "EelevenLongTailOneStepTailLossBefore",
            one_step,
            "tail_loss_before_ci95_low",
            "tail_loss_before_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailLossAfterFro", fmt(one_step["mean_tail_loss_after_frobenius"])),
        *ci_macros(
            "EelevenLongTailOneStepTailLossAfterFro",
            one_step,
            "tail_loss_after_frobenius_ci95_low",
            "tail_loss_after_frobenius_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailLossAfterSpectral", fmt(one_step["mean_tail_loss_after_spectral"])),
        *ci_macros(
            "EelevenLongTailOneStepTailLossAfterSpectral",
            one_step,
            "tail_loss_after_spectral_ci95_low",
            "tail_loss_after_spectral_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailMarginBefore", fmt(one_step["mean_tail_margin_before"])),
        *ci_macros(
            "EelevenLongTailOneStepTailMarginBefore",
            one_step,
            "tail_margin_before_ci95_low",
            "tail_margin_before_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailMarginAfterFro", fmt(one_step["mean_tail_margin_after_frobenius"])),
        *ci_macros(
            "EelevenLongTailOneStepTailMarginAfterFro",
            one_step,
            "tail_margin_after_frobenius_ci95_low",
            "tail_margin_after_frobenius_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailMarginAfterSpectral", fmt(one_step["mean_tail_margin_after_spectral"])),
        *ci_macros(
            "EelevenLongTailOneStepTailMarginAfterSpectral",
            one_step,
            "tail_margin_after_spectral_ci95_low",
            "tail_margin_after_spectral_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailAccuracyBefore", fmt(one_step["mean_tail_accuracy_before"])),
        *ci_macros(
            "EelevenLongTailOneStepTailAccuracyBefore",
            one_step,
            "tail_accuracy_before_ci95_low",
            "tail_accuracy_before_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailAccuracyAfterFro", fmt(one_step["mean_tail_accuracy_after_frobenius"])),
        *ci_macros(
            "EelevenLongTailOneStepTailAccuracyAfterFro",
            one_step,
            "tail_accuracy_after_frobenius_ci95_low",
            "tail_accuracy_after_frobenius_ci95_high",
        ),
        macro("EelevenLongTailOneStepTailAccuracyAfterSpectral", fmt(one_step["mean_tail_accuracy_after_spectral"])),
        *ci_macros(
            "EelevenLongTailOneStepTailAccuracyAfterSpectral",
            one_step,
            "tail_accuracy_after_spectral_ci95_low",
            "tail_accuracy_after_spectral_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepPositiveMarginFraction",
            fmt(one_step["mean_tail_positive_margin_fraction_before"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepPositiveMarginFraction",
            one_step,
            "tail_positive_margin_fraction_ci95_low",
            "tail_positive_margin_fraction_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepCertifiedFractionFro",
            fmt(one_step["mean_tail_margin_certified_preserved_fraction_frobenius"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepCertifiedFractionFro",
            one_step,
            "tail_margin_certified_preserved_fraction_frobenius_ci95_low",
            "tail_margin_certified_preserved_fraction_frobenius_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepCertifiedFractionSpectral",
            fmt(one_step["mean_tail_margin_certified_preserved_fraction_spectral"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepCertifiedFractionSpectral",
            one_step,
            "tail_margin_certified_preserved_fraction_spectral_ci95_low",
            "tail_margin_certified_preserved_fraction_spectral_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepPredictionChangedFro",
            fmt(one_step["mean_tail_prediction_changed_fraction_frobenius"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepPredictionChangedFro",
            one_step,
            "tail_prediction_changed_fraction_frobenius_ci95_low",
            "tail_prediction_changed_fraction_frobenius_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepPredictionChangedSpectral",
            fmt(one_step["mean_tail_prediction_changed_fraction_spectral"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepPredictionChangedSpectral",
            one_step,
            "tail_prediction_changed_fraction_spectral_ci95_low",
            "tail_prediction_changed_fraction_spectral_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepPositivePredictionChangedFro",
            fmt(one_step["mean_tail_positive_margin_prediction_changed_fraction_frobenius"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepPositivePredictionChangedFro",
            one_step,
            "tail_positive_margin_prediction_changed_fraction_frobenius_ci95_low",
            "tail_positive_margin_prediction_changed_fraction_frobenius_ci95_high",
        ),
        macro(
            "EelevenLongTailOneStepPositivePredictionChangedSpectral",
            fmt(one_step["mean_tail_positive_margin_prediction_changed_fraction_spectral"]),
        ),
        *ci_macros(
            "EelevenLongTailOneStepPositivePredictionChangedSpectral",
            one_step,
            "tail_positive_margin_prediction_changed_fraction_spectral_ci95_low",
            "tail_positive_margin_prediction_changed_fraction_spectral_ci95_high",
        ),
        "",
        "% CIFAR-100-LT ResNet18 one-step diagnostic",
        macro("EelevenCifarResNetOneStepSeeds", int(cifar_resnet["seeds"])),
        macro(
            "EelevenCifarResNetOneStepDriftRatio",
            fmt(cifar_resnet["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetOneStepDriftRatio",
            cifar_resnet,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetOneStepCenteredDriftRatio",
            fmt(cifar_resnet["geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetOneStepCenteredDriftRatio",
            cifar_resnet,
            "centered_tail_output_drift_sq_ratio_ci95_low",
            "centered_tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetOneStepMarginDeltaRatio",
            fmt(cifar_resnet["geomean_margin_delta_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetOneStepMarginDeltaRatio",
            cifar_resnet,
            "margin_delta_sq_ratio_ci95_low",
            "margin_delta_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetOneStepSpectralLowerFraction",
            fmt(cifar_resnet["spectral_less_tail_output_drift_fraction"]),
        ),
        macro(
            "EelevenCifarResNetOneStepTailLossDiff",
            fmt(cifar_resnet["mean_tail_loss_increase_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetOneStepTailLossDiff",
            cifar_resnet,
            "tail_loss_increase_diff_ci95_low",
            "tail_loss_increase_diff_ci95_high",
        ),
        macro(
            "EelevenCifarResNetOneStepTailMarginDropDiff",
            fmt(cifar_resnet["mean_tail_margin_drop_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetOneStepTailMarginDropDiff",
            cifar_resnet,
            "tail_margin_drop_diff_ci95_low",
            "tail_margin_drop_diff_ci95_high",
        ),
        macro(
            "EelevenCifarResNetOneStepTailAccuracyDropDiff",
            fmt(cifar_resnet["mean_tail_accuracy_drop_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetOneStepTailAccuracyDropDiff",
            cifar_resnet,
            "tail_accuracy_drop_diff_ci95_low",
            "tail_accuracy_drop_diff_ci95_high",
        ),
        macro(
            "EelevenCifarResNetOneStepTailAccuracyBefore",
            fmt(cifar_resnet["mean_tail_accuracy_before"]),
        ),
        *ci_macros(
            "EelevenCifarResNetOneStepTailAccuracyBefore",
            cifar_resnet,
            "tail_accuracy_before_ci95_low",
            "tail_accuracy_before_ci95_high",
        ),
        macro(
            "EelevenCifarResNetOneStepMeanNrG",
            fmt(cifar_resnet["mean_nrG"]),
        ),
        macro(
            "EelevenCifarResNetRho002DriftRatio",
            fmt(cifar_resnet_rho002["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetRho002DriftRatio",
            cifar_resnet_rho002,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetRho002TailLossDiff",
            fmt(cifar_resnet_rho002["mean_tail_loss_increase_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetRho002TailLossDiff",
            cifar_resnet_rho002,
            "tail_loss_increase_diff_ci95_low",
            "tail_loss_increase_diff_ci95_high",
        ),
        macro("EelevenCifarResNetCheckpointSweepSettings", int(len(cifar_resnet_checkpoint_sweep))),
        macro(
            "EelevenCifarResNetCheckpointSweepWorstWarmupSteps",
            int(cifar_resnet_checkpoint_worst["warmup_steps"]),
        ),
        macro(
            "EelevenCifarResNetCheckpointSweepWorstDriftRatio",
            fmt(cifar_resnet_checkpoint_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetCheckpointSweepWorstDriftRatio",
            cifar_resnet_checkpoint_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetCheckpointSweepBestTailAccuracyWarmupSteps",
            int(cifar_resnet_checkpoint_best_tail_accuracy["warmup_steps"]),
        ),
        macro(
            "EelevenCifarResNetCheckpointSweepBestTailAccuracy",
            fmt(cifar_resnet_checkpoint_best_tail_accuracy["mean_tail_accuracy_before"]),
        ),
        *ci_macros(
            "EelevenCifarResNetCheckpointSweepBestTailAccuracy",
            cifar_resnet_checkpoint_best_tail_accuracy,
            "tail_accuracy_before_ci95_low",
            "tail_accuracy_before_ci95_high",
        ),
        macro(
            "EelevenCifarResNetCheckpointSweepTailAccuracyRange",
            f"{fmt(cifar_resnet_checkpoint_tail_accuracy_min)}\\text{{ to }}{fmt(cifar_resnet_checkpoint_tail_accuracy_max)}",
        ),
        macro(
            "EelevenCifarResNetCheckpointSweepPositiveMarginRange",
            f"{fmt(cifar_resnet_checkpoint_positive_margin_min)}\\text{{ to }}{fmt(cifar_resnet_checkpoint_positive_margin_max)}",
        ),
        macro("EelevenCifarResNetConditionProxyPoints", int(len(cifar_resnet_condition_proxy_points))),
        macro(
            "EelevenCifarResNetConditionProxyRankPearson",
            fmt(cifar_resnet_condition_rank_pearson["estimate"]),
        ),
        *ci_macros(
            "EelevenCifarResNetConditionProxyRankPearson",
            cifar_resnet_condition_rank_pearson,
            "ci95_low",
            "ci95_high",
        ),
        macro(
            "EelevenCifarResNetConditionProxyRankSpearman",
            fmt(cifar_resnet_condition_rank_spearman["estimate"]),
        ),
        *ci_macros(
            "EelevenCifarResNetConditionProxyRankSpearman",
            cifar_resnet_condition_rank_spearman,
            "ci95_low",
            "ci95_high",
        ),
        macro(
            "EelevenCifarResNetConditionProxyTailAccuracySpearman",
            fmt(cifar_resnet_condition_tail_accuracy_spearman["estimate"]),
        ),
        *ci_macros(
            "EelevenCifarResNetConditionProxyTailAccuracySpearman",
            cifar_resnet_condition_tail_accuracy_spearman,
            "ci95_low",
            "ci95_high",
        ),
        macro("EelevenCifarResNetFcConditionSettings", int(len(cifar_resnet_fc_condition_summary))),
        macro("EelevenCifarResNetFcConditionPoints", int(len(cifar_resnet_fc_condition_points))),
        macro(
            "EelevenCifarResNetFcConditionWorstWarmupSteps",
            int(cifar_resnet_fc_condition_worst["warmup_steps"]),
        ),
        macro(
            "EelevenCifarResNetFcConditionWorstDriftRatio",
            fmt(cifar_resnet_fc_condition_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetFcConditionWorstDriftRatio",
            cifar_resnet_fc_condition_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetFcConditionWeakestMeanConditionScore",
            fmt(cifar_resnet_fc_condition_weakest_mean["mean_condition_score_nrank_over_tail_srank"]),
        ),
        macro(
            "EelevenCifarResNetFcConditionWeakestMeanConditionWarmupSteps",
            int(cifar_resnet_fc_condition_weakest_mean["warmup_steps"]),
        ),
        macro(
            "EelevenCifarResNetFcConditionWeakestPointConditionScore",
            fmt(cifar_resnet_fc_condition_weakest_point["condition_score_nrank_over_tail_srank"]),
        ),
        macro(
            "EelevenCifarResNetFcConditionWeakestPointWarmupSteps",
            int(cifar_resnet_fc_condition_weakest_point["warmup_steps"]),
        ),
        macro(
            "EelevenCifarResNetFcConditionWorstTheoryRatio",
            fmt(cifar_resnet_fc_condition_worst_theory["mean_theory_ratio_tail_srank_over_nrank"]),
        ),
        macro(
            "EelevenCifarResNetFcConditionFavorsSpectralFraction",
            fmt(cifar_resnet_fc_condition_favors_fraction),
        ),
        macro("EelevenCifarResNetTailQualitySettings", int(len(cifar_resnet_tail_quality))),
        macro(
            "EelevenCifarResNetTailQualityWorstWarmupSteps",
            int(cifar_resnet_tail_quality_worst["warmup_steps"]),
        ),
        macro(
            "EelevenCifarResNetTailQualityWorstDriftRatio",
            fmt(cifar_resnet_tail_quality_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetTailQualityWorstDriftRatio",
            cifar_resnet_tail_quality_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetTailQualityBestTailAccuracyWarmupSteps",
            int(cifar_resnet_tail_quality_best_tail_accuracy["warmup_steps"]),
        ),
        macro(
            "EelevenCifarResNetTailQualityBestTailAccuracy",
            fmt(cifar_resnet_tail_quality_best_tail_accuracy["mean_tail_accuracy_before"]),
        ),
        *ci_macros(
            "EelevenCifarResNetTailQualityBestTailAccuracy",
            cifar_resnet_tail_quality_best_tail_accuracy,
            "tail_accuracy_before_ci95_low",
            "tail_accuracy_before_ci95_high",
        ),
        macro(
            "EelevenCifarResNetTailQualityTailAccuracyRange",
            f"{fmt(cifar_resnet_tail_quality_tail_accuracy_min)}\\text{{ to }}{fmt(cifar_resnet_tail_quality_tail_accuracy_max)}",
        ),
        "",
        "% CIFAR-100-LT ResNet18 imbalance sweep",
        macro("EelevenCifarResNetImbalanceSweepSettings", cifar_resnet_imbalance_settings),
        macro(
            "EelevenCifarResNetImbalanceSweepWorstTailTrainPerClass",
            int(cifar_resnet_imbalance_worst["tail_train_per_class"]),
        ),
        macro(
            "EelevenCifarResNetImbalanceSweepWorstDriftRatio",
            fmt(cifar_resnet_imbalance_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetImbalanceSweepWorstDriftRatio",
            cifar_resnet_imbalance_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetImbalanceSweepBestTailTrainPerClass",
            int(cifar_resnet_imbalance_best_tail_accuracy["tail_train_per_class"]),
        ),
        macro(
            "EelevenCifarResNetImbalanceSweepBestTailAccuracy",
            fmt(cifar_resnet_imbalance_best_tail_accuracy["mean_tail_accuracy_before"]),
        ),
        *ci_macros(
            "EelevenCifarResNetImbalanceSweepBestTailAccuracy",
            cifar_resnet_imbalance_best_tail_accuracy,
            "tail_accuracy_before_ci95_low",
            "tail_accuracy_before_ci95_high",
        ),
        macro(
            "EelevenCifarResNetImbalanceSweepBestTailDriftRatio",
            fmt(
                cifar_resnet_imbalance_best_tail_accuracy[
                    "geomean_tail_output_drift_sq_ratio_spectral_over_fro"
                ]
            ),
        ),
        *ci_macros(
            "EelevenCifarResNetImbalanceSweepBestTailDriftRatio",
            cifar_resnet_imbalance_best_tail_accuracy,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro("EelevenCifarResNetLayerJvpSeeds", int(cifar_resnet_layer_jvp["seeds"])),
        macro("EelevenCifarResNetLayerJvpParameters", int(cifar_resnet_layer_jvp["parameters"])),
        macro("EelevenCifarResNetLayerJvpPairedPoints", int(cifar_resnet_layer_jvp["paired_points"])),
        macro(
            "EelevenCifarResNetLayerJvpTailAccuracyBefore",
            fmt(cifar_resnet_layer_jvp["mean_tail_accuracy_before"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpScaledRatio",
            fmt(cifar_resnet_layer_jvp["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpScaledRatio",
            cifar_resnet_layer_jvp,
            "scaled_jvp_tail_drift_sq_ratio_ci95_low",
            "scaled_jvp_tail_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpObservedRatio",
            fmt(cifar_resnet_layer_jvp["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpObservedRatio",
            cifar_resnet_layer_jvp,
            "observed_tail_drift_sq_ratio_ci95_low",
            "observed_tail_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpSpectralLowerFraction",
            fmt(cifar_resnet_layer_jvp["spectral_less_observed_tail_drift_fraction"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpSupportedLayers",
            cifar_resnet_layer_jvp_supported_layers,
        ),
        macro(
            "EelevenCifarResNetLayerJvpWorstObservedParameter",
            latex_texttt(cifar_resnet_layer_jvp_worst_observed["parameter"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpWorstObservedRatio",
            fmt(cifar_resnet_layer_jvp_worst_observed["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpWorstObservedRatio",
            cifar_resnet_layer_jvp_worst_observed,
            "observed_tail_drift_sq_ratio_ci95_low",
            "observed_tail_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpWorstScaledParameter",
            latex_texttt(cifar_resnet_layer_jvp_worst_scaled["parameter"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpWorstScaledRatio",
            fmt(cifar_resnet_layer_jvp_worst_scaled["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpWorstScaledRatio",
            cifar_resnet_layer_jvp_worst_scaled,
            "scaled_jvp_tail_drift_sq_ratio_ci95_low",
            "scaled_jvp_tail_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionCheckpoints",
            int(cifar_resnet_layer_jvp_checkpoint_summary["warmup_steps"].nunique()),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionLayerSummaries",
            int(len(cifar_resnet_layer_jvp_checkpoint_layer_summary)),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionTransferPairs",
            int(cifar_resnet_layer_jvp_checkpoint_scaled["checkpoint_transfer_pairs"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionDirectedRows",
            int(len(cifar_resnet_layer_jvp_checkpoint_prediction_pairs)),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionObservedSpearman",
            fmt(cifar_resnet_layer_jvp_checkpoint_observed["mean_spearman_log_predictor_vs_log_target_observed"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpCheckpointPredictionObservedSpearman",
            cifar_resnet_layer_jvp_checkpoint_observed,
            "spearman_ci95_low",
            "spearman_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionObservedTopFiveOverlap",
            fmt(cifar_resnet_layer_jvp_checkpoint_observed["mean_top5_risk_overlap_fraction"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionEarlyLayerSpearman",
            fmt(cifar_resnet_layer_jvp_checkpoint_early_layer["mean_spearman_log_predictor_vs_log_target_observed"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpCheckpointPredictionEarlyLayerSpearman",
            cifar_resnet_layer_jvp_checkpoint_early_layer,
            "spearman_ci95_low",
            "spearman_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionEarlyLayerTopFiveOverlap",
            fmt(cifar_resnet_layer_jvp_checkpoint_early_layer["mean_top5_risk_overlap_fraction"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionScaledSpearman",
            fmt(cifar_resnet_layer_jvp_checkpoint_scaled["mean_spearman_log_predictor_vs_log_target_observed"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpCheckpointPredictionScaledSpearman",
            cifar_resnet_layer_jvp_checkpoint_scaled,
            "spearman_ci95_low",
            "spearman_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionScaledThresholdAccuracy",
            fmt(cifar_resnet_layer_jvp_checkpoint_scaled["mean_threshold_below_one_accuracy"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionScaledTopFiveOverlap",
            fmt(cifar_resnet_layer_jvp_checkpoint_scaled["mean_top5_risk_overlap_fraction"]),
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionUnitSpearman",
            fmt(cifar_resnet_layer_jvp_checkpoint_unit["mean_spearman_log_predictor_vs_log_target_observed"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpCheckpointPredictionUnitSpearman",
            cifar_resnet_layer_jvp_checkpoint_unit,
            "spearman_ci95_low",
            "spearman_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLayerJvpCheckpointPredictionRankSpearman",
            fmt(cifar_resnet_layer_jvp_checkpoint_rank["mean_spearman_log_predictor_vs_log_target_observed"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLayerJvpCheckpointPredictionRankSpearman",
            cifar_resnet_layer_jvp_checkpoint_rank,
            "spearman_ci95_low",
            "spearman_ci95_high",
        ),
        "",
        "% CIFAR-100-LT ResNet18 standard many/medium/few reporting baseline",
        macro("EelevenCifarResNetLtStandardEvalSeeds", int(cifar_resnet_lt_standard_all["seeds"])),
        macro("EelevenCifarResNetLtStandardEvalClasses", int(cifar_resnet_lt_standard_all["classes"])),
        macro("EelevenCifarResNetLtStandardEvalManyClasses", int(cifar_resnet_lt_standard_many["classes"])),
        macro("EelevenCifarResNetLtStandardEvalMediumClasses", int(cifar_resnet_lt_standard_medium["classes"])),
        macro("EelevenCifarResNetLtStandardEvalFewClasses", int(cifar_resnet_lt_standard_few["classes"])),
        macro("EelevenCifarResNetLtStandardEvalImbalanceFactor", fmt(100.0)),
        macro(
            "EelevenCifarResNetLtStandardEvalManyBalancedAccuracy",
            fmt(cifar_resnet_lt_standard_many["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtStandardEvalManyBalancedAccuracy",
            cifar_resnet_lt_standard_many,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtStandardEvalMediumBalancedAccuracy",
            fmt(cifar_resnet_lt_standard_medium["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtStandardEvalMediumBalancedAccuracy",
            cifar_resnet_lt_standard_medium,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtStandardEvalFewBalancedAccuracy",
            fmt(cifar_resnet_lt_standard_few["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtStandardEvalFewBalancedAccuracy",
            cifar_resnet_lt_standard_few,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtStandardEvalAllBalancedAccuracy",
            fmt(cifar_resnet_lt_standard_all["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtStandardEvalAllBalancedAccuracy",
            cifar_resnet_lt_standard_all,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        "",
        "% CIFAR-100-LT ResNet18 augmented recipe benchmark pilot",
        macro("EelevenCifarResNetLtRecipeBenchmarkSeeds", int(cifar_resnet_lt_recipe_adamw_aug_all["seeds"])),
        macro(
            "EelevenCifarResNetLtRecipeAdamwAugManyBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_adamw_aug_many["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeAdamwAugManyBalancedAccuracy",
            cifar_resnet_lt_recipe_adamw_aug_many,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeAdamwAugMediumBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_adamw_aug_medium["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeAdamwAugMediumBalancedAccuracy",
            cifar_resnet_lt_recipe_adamw_aug_medium,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeAdamwAugFewBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_adamw_aug_few["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeAdamwAugFewBalancedAccuracy",
            cifar_resnet_lt_recipe_adamw_aug_few,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeAdamwAugAllBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_adamw_aug_all["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeAdamwAugAllBalancedAccuracy",
            cifar_resnet_lt_recipe_adamw_aug_all,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeCbFewBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_cb_few["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeCbFewBalancedAccuracy",
            cifar_resnet_lt_recipe_cb_few,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeSgdManyBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_sgd_many["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeSgdManyBalancedAccuracy",
            cifar_resnet_lt_recipe_sgd_many,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeSgdMediumBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_sgd_medium["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeSgdMediumBalancedAccuracy",
            cifar_resnet_lt_recipe_sgd_medium,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeSgdFewBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_sgd_few["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeSgdFewBalancedAccuracy",
            cifar_resnet_lt_recipe_sgd_few,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeSgdAllBalancedAccuracy",
            fmt(cifar_resnet_lt_recipe_sgd_all["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeSgdAllBalancedAccuracy",
            cifar_resnet_lt_recipe_sgd_all,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeSgdFewBalancedAccuracyDiff",
            fmt(cifar_resnet_lt_recipe_sgd_few_diff["mean_balanced_accuracy_diff"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeSgdFewBalancedAccuracyDiff",
            cifar_resnet_lt_recipe_sgd_few_diff,
            "balanced_accuracy_diff_ci95_low",
            "balanced_accuracy_diff_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeSgdAllBalancedAccuracyDiff",
            fmt(cifar_resnet_lt_recipe_sgd_all_diff["mean_balanced_accuracy_diff"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeSgdAllBalancedAccuracyDiff",
            cifar_resnet_lt_recipe_sgd_all_diff,
            "balanced_accuracy_diff_ci95_low",
            "balanced_accuracy_diff_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtRecipeCbFewBalancedAccuracyDiff",
            fmt(cifar_resnet_lt_recipe_cb_few_diff["mean_balanced_accuracy_diff"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtRecipeCbFewBalancedAccuracyDiff",
            cifar_resnet_lt_recipe_cb_few_diff,
            "balanced_accuracy_diff_ci95_low",
            "balanced_accuracy_diff_ci95_high",
        ),
        "",
        "% CIFAR-100-LT ResNet18 NS-Muon final-training benchmark pilot",
        macro("EelevenCifarResNetLtMuonFinalBenchmarkSeeds", int(cifar_resnet_lt_muon_final_adamw_all["seeds"])),
        macro(
            "EelevenCifarResNetLtMuonFinalAdamwAllBalancedAccuracy",
            fmt(cifar_resnet_lt_muon_final_adamw_all["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalAdamwAllBalancedAccuracy",
            cifar_resnet_lt_muon_final_adamw_all,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalAdamwFewBalancedAccuracy",
            fmt(cifar_resnet_lt_muon_final_adamw_few["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalAdamwFewBalancedAccuracy",
            cifar_resnet_lt_muon_final_adamw_few,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourAllBalancedAccuracy",
            fmt(cifar_resnet_lt_muon_final_lr1e4_all["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourAllBalancedAccuracy",
            cifar_resnet_lt_muon_final_lr1e4_all,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourFewBalancedAccuracy",
            fmt(cifar_resnet_lt_muon_final_lr1e4_few["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourFewBalancedAccuracy",
            cifar_resnet_lt_muon_final_lr1e4_few,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveAllBalancedAccuracy",
            fmt(cifar_resnet_lt_muon_final_lr3e5_all["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveAllBalancedAccuracy",
            cifar_resnet_lt_muon_final_lr3e5_all,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveFewBalancedAccuracy",
            fmt(cifar_resnet_lt_muon_final_lr3e5_few["mean_balanced_accuracy"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveFewBalancedAccuracy",
            cifar_resnet_lt_muon_final_lr3e5_few,
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourAllBalancedAccuracyDiff",
            fmt(cifar_resnet_lt_muon_final_lr1e4_all_diff["mean_balanced_accuracy_diff"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourAllBalancedAccuracyDiff",
            cifar_resnet_lt_muon_final_lr1e4_all_diff,
            "balanced_accuracy_diff_ci95_low",
            "balanced_accuracy_diff_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourFewBalancedAccuracyDiff",
            fmt(cifar_resnet_lt_muon_final_lr1e4_few_diff["mean_balanced_accuracy_diff"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalLrOneEMinusFourFewBalancedAccuracyDiff",
            cifar_resnet_lt_muon_final_lr1e4_few_diff,
            "balanced_accuracy_diff_ci95_low",
            "balanced_accuracy_diff_ci95_high",
        ),
        macro(
            "EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveAllBalancedAccuracyDiff",
            fmt(cifar_resnet_lt_muon_final_lr3e5_all_diff["mean_balanced_accuracy_diff"]),
        ),
        *ci_macros(
            "EelevenCifarResNetLtMuonFinalLrThreeEMinusFiveAllBalancedAccuracyDiff",
            cifar_resnet_lt_muon_final_lr3e5_all_diff,
            "balanced_accuracy_diff_ci95_low",
            "balanced_accuracy_diff_ci95_high",
        ),
        "",
        "% CIFAR-100-LT ResNet18 practical Muon trajectory bridge",
        macro(
            "EelevenCifarResNetPracticalMuonBridgeStateSources",
            int(cifar_resnet_practical_bridge.index.get_level_values("state_source").nunique()),
        ),
        macro(
            "EelevenCifarResNetPracticalMuonBridgeComparisonsPerDirection",
            int(cifar_resnet_practical_adam_ns_momentum["comparisons"]),
        ),
        macro(
            "EelevenCifarResNetPracticalAdamStatePolarMomentumDriftRatio",
            fmt(cifar_resnet_practical_adam_polar_momentum["geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetPracticalAdamStatePolarMomentumDriftRatio",
            cifar_resnet_practical_adam_polar_momentum,
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenCifarResNetPracticalAdamStateNsMomentumDriftRatio",
            fmt(cifar_resnet_practical_adam_ns_momentum["geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetPracticalAdamStateNsMomentumDriftRatio",
            cifar_resnet_practical_adam_ns_momentum,
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenCifarResNetPracticalAdamStateNsMomentumCosine",
            fmt(cifar_resnet_practical_adam_ns_momentum["mean_gradient_momentum_cosine"]),
        ),
        *ci_macros(
            "EelevenCifarResNetPracticalAdamStateNsMomentumCosine",
            cifar_resnet_practical_adam_ns_momentum,
            "gradient_momentum_cosine_ci95_low",
            "gradient_momentum_cosine_ci95_high",
        ),
        macro(
            "EelevenCifarResNetPracticalMuonStatePolarMomentumDriftRatio",
            fmt(cifar_resnet_practical_muon_polar_momentum["geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetPracticalMuonStatePolarMomentumDriftRatio",
            cifar_resnet_practical_muon_polar_momentum,
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenCifarResNetPracticalMuonStateNsMomentumDriftRatio",
            fmt(cifar_resnet_practical_muon_ns_momentum["geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenCifarResNetPracticalMuonStateNsMomentumDriftRatio",
            cifar_resnet_practical_muon_ns_momentum,
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenCifarResNetPracticalMuonStateNsMomentumCosine",
            fmt(cifar_resnet_practical_muon_ns_momentum["mean_gradient_momentum_cosine"]),
        ),
        *ci_macros(
            "EelevenCifarResNetPracticalMuonStateNsMomentumCosine",
            cifar_resnet_practical_muon_ns_momentum,
            "gradient_momentum_cosine_ci95_low",
            "gradient_momentum_cosine_ci95_high",
        ),
        "",
        "% Local linearization quality",
        macro(
            "EelevenLocalLinearizationMaxRelativeErrorCiHigh",
            latex_sci(float(local_linearization_worst["relative_error_ci95_high"])),
        ),
        macro(
            "EelevenLocalLinearizationMaxRelativeErrorDirection",
            latex_math_fragment(local_linearization_worst["display_name"]),
        ),
        "",
        "% Long-tailed imbalance ablation",
        macro("EelevenLongTailImbalanceAblationSettings", int(len(imbalance))),
        macro(
            "EelevenLongTailImbalanceWorstTailTrainPerClass",
            int(imbalance_worst["tail_train_per_class"]),
        ),
        macro(
            "EelevenLongTailImbalanceWorstRatio",
            fmt(imbalance_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailImbalanceWorstRatio",
            imbalance_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailImbalanceDefaultRatio",
            fmt(imbalance_default["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailImbalanceDefaultRatio",
            imbalance_default,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailImbalanceStrongRatio",
            fmt(imbalance_strong["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailImbalanceStrongRatio",
            imbalance_strong,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailImbalanceStrongPositiveMarginFraction",
            fmt(imbalance_strong["mean_tail_positive_margin_fraction_before"]),
        ),
        *ci_macros(
            "EelevenLongTailImbalanceStrongPositiveMarginFraction",
            imbalance_strong,
            "tail_positive_margin_fraction_ci95_low",
            "tail_positive_margin_fraction_ci95_high",
        ),
        "",
        "% Long-tailed checkpoint sweep",
        macro("EelevenLongTailCheckpointSweepSettings", int(len(checkpoint_sweep))),
        macro(
            "EelevenLongTailCheckpointSweepWorstWarmupSteps",
            int(checkpoint_worst["warmup_steps"]),
        ),
        macro(
            "EelevenLongTailCheckpointSweepWorstRatio",
            fmt(checkpoint_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailCheckpointSweepWorstRatio",
            checkpoint_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailCheckpointSweepWorstMarginWarmupSteps",
            int(checkpoint_margin_worst["warmup_steps"]),
        ),
        macro(
            "EelevenLongTailCheckpointSweepWorstMarginRatio",
            fmt(checkpoint_margin_worst["geomean_margin_delta_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailCheckpointSweepWorstMarginRatio",
            checkpoint_margin_worst,
            "margin_delta_sq_ratio_ci95_low",
            "margin_delta_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailCheckpointSweepTailAccuracyRange",
            f"{fmt(checkpoint_tail_accuracy_min)}\\text{{ to }}{fmt(checkpoint_tail_accuracy_max)}",
        ),
        macro(
            "EelevenLongTailCheckpointSweepPositiveMarginRange",
            f"{fmt(checkpoint_positive_margin_min)}\\text{{ to }}{fmt(checkpoint_positive_margin_max)}",
        ),
        "",
        "% Long-tailed class-partition sweep",
        macro("EelevenLongTailClassPartitionSweepSettings", int(len(class_partition_sweep))),
        macro(
            "EelevenLongTailClassPartitionWorstPartition",
            latex_texttt(class_partition_worst["partition_name"]),
        ),
        macro(
            "EelevenLongTailClassPartitionWorstRatio",
            fmt(class_partition_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailClassPartitionWorstRatio",
            class_partition_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailClassPartitionCenteredWorstPartition",
            latex_texttt(class_partition_centered_worst["partition_name"]),
        ),
        macro(
            "EelevenLongTailClassPartitionCenteredWorstRatio",
            fmt(
                class_partition_centered_worst[
                    "geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro"
                ]
            ),
        ),
        *ci_macros(
            "EelevenLongTailClassPartitionCenteredWorstRatio",
            class_partition_centered_worst,
            "centered_tail_output_drift_sq_ratio_ci95_low",
            "centered_tail_output_drift_sq_ratio_ci95_high",
        ),
        "",
        "% Long-tailed matched-gain rho sweep",
        macro("EelevenLongTailRhoSweepSettings", int(len(rho_sweep))),
        macro(
            "EelevenLongTailRhoSweepWorstFraction",
            fmt(rho_worst["target_head_gain_fraction"]),
        ),
        macro(
            "EelevenLongTailRhoSweepWorstRatio",
            fmt(rho_worst["geomean_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailRhoSweepWorstRatio",
            rho_worst,
            "tail_output_drift_sq_ratio_ci95_low",
            "tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailRhoSweepCenteredWorstFraction",
            fmt(rho_centered_worst["target_head_gain_fraction"]),
        ),
        macro(
            "EelevenLongTailRhoSweepCenteredWorstRatio",
            fmt(rho_centered_worst["geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailRhoSweepCenteredWorstRatio",
            rho_centered_worst,
            "centered_tail_output_drift_sq_ratio_ci95_low",
            "centered_tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailRhoSweepWorstHeadGainErrorFraction",
            fmt(rho_head_gain_error_worst["target_head_gain_fraction"]),
        ),
        macro(
            "EelevenLongTailRhoSweepWorstHeadGainErrorCiHigh",
            fmt(rho_head_gain_error_worst_value),
        ),
        "",
        "% Long-tailed Muon-style compatibility diagnostic",
        macro(
            "EelevenLongTailMuonBridgePolarMomentumDriftRatio",
            fmt(muon_bridge.loc["polar_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailMuonBridgePolarMomentumDriftRatio",
            muon_bridge.loc["polar_momentum"],
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenLongTailMuonBridgeNsMomentumDriftRatio",
            fmt(muon_bridge.loc["ns_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailMuonBridgeNsMomentumDriftRatio",
            muon_bridge.loc["ns_momentum"],
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenLongTailMuonBridgePolarMomentumCosine",
            fmt(muon_bridge.loc["polar_momentum", "mean_direction_cosine_to_polar_grad"]),
        ),
        *ci_macros(
            "EelevenLongTailMuonBridgePolarMomentumCosine",
            muon_bridge.loc["polar_momentum"],
            "direction_cosine_to_polar_grad_ci95_low",
            "direction_cosine_to_polar_grad_ci95_high",
        ),
        "",
        "% Long-tailed practical Muon trajectory compatibility diagnostic",
        macro(
            "EelevenLongTailPracticalMuonBridgeComparisons",
            int(practical_bridge.loc["polar_momentum", "comparisons"]),
        ),
        macro(
            "EelevenLongTailPracticalMuonBridgePolarMomentumDriftRatio",
            fmt(practical_bridge.loc["polar_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalMuonBridgePolarMomentumDriftRatio",
            practical_bridge.loc["polar_momentum"],
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenLongTailPracticalMuonBridgeNsMomentumDriftRatio",
            fmt(practical_bridge.loc["ns_momentum", "geomean_tail_output_drift_sq_ratio_vs_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalMuonBridgeNsMomentumDriftRatio",
            practical_bridge.loc["ns_momentum"],
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenLongTailPracticalMuonBridgeMomentumCosine",
            fmt(practical_bridge.loc["ns_momentum", "mean_gradient_momentum_cosine"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalMuonBridgeMomentumCosine",
            practical_bridge.loc["ns_momentum"],
            "gradient_momentum_cosine_ci95_low",
            "gradient_momentum_cosine_ci95_high",
        ),
        "",
        "% Long-tailed Muon state-source control",
        macro(
            "EelevenLongTailMuonStateSourceControlComparisons",
            int(state_source_control.loc[("fro_gd_trajectory", "ns_momentum"), "comparisons"]),
        ),
        macro(
            "EelevenLongTailMuonStateSourceControlFroPolarMomentumDriftRatio",
            fmt(
                state_source_control.loc[
                    ("fro_gd_trajectory", "polar_momentum"),
                    "geomean_tail_output_drift_sq_ratio_vs_fro",
                ]
            ),
        ),
        *ci_macros(
            "EelevenLongTailMuonStateSourceControlFroPolarMomentumDriftRatio",
            state_source_control.loc[("fro_gd_trajectory", "polar_momentum")],
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenLongTailMuonStateSourceControlFroNsMomentumDriftRatio",
            fmt(
                state_source_control.loc[
                    ("fro_gd_trajectory", "ns_momentum"),
                    "geomean_tail_output_drift_sq_ratio_vs_fro",
                ]
            ),
        ),
        *ci_macros(
            "EelevenLongTailMuonStateSourceControlFroNsMomentumDriftRatio",
            state_source_control.loc[("fro_gd_trajectory", "ns_momentum")],
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenLongTailMuonStateSourceControlMuonNsMomentumDriftRatio",
            fmt(
                state_source_control.loc[
                    ("muon_ns_trajectory", "ns_momentum"),
                    "geomean_tail_output_drift_sq_ratio_vs_fro",
                ]
            ),
        ),
        *ci_macros(
            "EelevenLongTailMuonStateSourceControlMuonNsMomentumDriftRatio",
            state_source_control.loc[("muon_ns_trajectory", "ns_momentum")],
            "tail_output_drift_sq_ratio_vs_fro_ci95_low",
            "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        ),
        macro(
            "EelevenLongTailMuonStateSourceControlFroNsLowerFraction",
            fmt(
                state_source_control.loc[
                    ("fro_gd_trajectory", "ns_momentum"),
                    "less_tail_drift_than_fro_fraction",
                ]
            ),
        ),
        "",
        "% Long-tailed practical training diagnostic",
        macro("EelevenLongTailPracticalTrainingSeeds", int(practical_training["seeds"])),
        macro("EelevenLongTailPracticalTrainingSteps", int(practical_training["train_steps"])),
        macro("EelevenLongTailPracticalTrainingAdamLr", fmt(practical_training["adam_lr"])),
        macro("EelevenLongTailPracticalTrainingMuonLr", fmt(practical_training["muon_lr"])),
        macro(
            "EelevenLongTailPracticalTrainingTrainLossRatio",
            fmt(practical_training["geomean_final_train_loss_ratio_muon_over_adam"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalTrainingTrainLossRatio",
            practical_training,
            "final_train_loss_ratio_ci95_low",
            "final_train_loss_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailPracticalTrainingHeadLossRatio",
            fmt(practical_training["geomean_final_head_loss_ratio_muon_over_adam"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalTrainingHeadLossRatio",
            practical_training,
            "final_head_loss_ratio_ci95_low",
            "final_head_loss_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailPracticalTrainingTailLossRatio",
            fmt(practical_training["geomean_final_tail_eval_loss_ratio_muon_over_adam"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalTrainingTailLossRatio",
            practical_training,
            "final_tail_eval_loss_ratio_ci95_low",
            "final_tail_eval_loss_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailPracticalTrainingTailDriftRmsRatio",
            fmt(practical_training["geomean_final_tail_eval_drift_rms_ratio_muon_over_adam"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalTrainingTailDriftRmsRatio",
            practical_training,
            "final_tail_eval_drift_rms_ratio_ci95_low",
            "final_tail_eval_drift_rms_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailPracticalTrainingTailMarginDiff",
            fmt(practical_training["mean_final_tail_eval_margin_diff_muon_minus_adam"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalTrainingTailMarginDiff",
            practical_training,
            "final_tail_eval_margin_diff_ci95_low",
            "final_tail_eval_margin_diff_ci95_high",
        ),
        macro(
            "EelevenLongTailPracticalTrainingTailAccuracyDiff",
            fmt(practical_training["mean_final_tail_eval_accuracy_diff_muon_minus_adam"]),
        ),
        *ci_macros(
            "EelevenLongTailPracticalTrainingTailAccuracyDiff",
            practical_training,
            "final_tail_eval_accuracy_diff_ci95_low",
            "final_tail_eval_accuracy_diff_ci95_high",
        ),
        "",
        "% Long-tailed practical training LR sensitivity",
        macro("EelevenLongTailPracticalTrainingLrSweepSmallTrainLossRatio", fmt(lr_0003["geomean_final_train_loss_ratio_muon_over_adam"])),
        macro("EelevenLongTailPracticalTrainingLrSweepMediumTrainLossRatio", fmt(lr_001["geomean_final_train_loss_ratio_muon_over_adam"])),
        macro("EelevenLongTailPracticalTrainingLrSweepLargeTailLossRatio", fmt(lr_01["geomean_final_tail_eval_loss_ratio_muon_over_adam"])),
        macro("EelevenLongTailPracticalTrainingLrSweepLargeTailDriftRmsRatio", fmt(lr_01["geomean_final_tail_eval_drift_rms_ratio_muon_over_adam"])),
        "",
        "% Eight-step head-only forgetting diagnostic",
        macro("EelevenLongTailForgettingSteps", int(forgetting["head_only_steps"])),
        macro(
            "EelevenLongTailForgettingFinalDriftRatio",
            fmt(forgetting["geomean_final_tail_output_drift_sq_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailForgettingFinalDriftRatio",
            forgetting,
            "final_tail_output_drift_sq_ratio_ci95_low",
            "final_tail_output_drift_sq_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailForgettingAreaDriftRatio",
            fmt(forgetting["geomean_tail_output_drift_area_ratio_spectral_over_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailForgettingAreaDriftRatio",
            forgetting,
            "tail_output_drift_area_ratio_ci95_low",
            "tail_output_drift_area_ratio_ci95_high",
        ),
        macro(
            "EelevenLongTailForgettingFinalTailLossDiff",
            fmt(forgetting["mean_final_tail_loss_increase_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailForgettingFinalTailLossDiff",
            forgetting,
            "final_tail_loss_increase_diff_ci95_low",
            "final_tail_loss_increase_diff_ci95_high",
        ),
        macro(
            "EelevenLongTailForgettingFinalTailMarginDropDiff",
            fmt(forgetting["mean_final_tail_margin_drop_diff_spectral_minus_fro"]),
        ),
        *ci_macros(
            "EelevenLongTailForgettingFinalTailMarginDropDiff",
            forgetting,
            "final_tail_margin_drop_diff_ci95_low",
            "final_tail_margin_drop_diff_ci95_high",
        ),
        "",
        "% Layerwise mechanism diagnostic",
        macro("EelevenLayerOneActivationConditionScore", fmt(layer_one["mean_condition_score"])),
        macro("EelevenLayerOneLocalOperatorConditionScore", fmt(layer_one["mean_local_operator_condition_score"])),
        macro("EelevenLayerOneLocalOperatorStableRank", fmt(layer_one["mean_tail_local_operator_stable_rank"])),
        macro("EelevenLayerOneUnitJvpDriftRatio", fmt(layer_one["geomean_jvp_tail_drift_sq_ratio_spectral_over_fro"])),
        *ci_macros(
            "EelevenLayerOneUnitJvpDriftRatio",
            layer_one,
            "jvp_tail_drift_sq_ratio_ci95_low",
            "jvp_tail_drift_sq_ratio_ci95_high",
        ),
        macro("EelevenLayerOneScaledJvpDriftRatio", fmt(layer_one["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"])),
        *ci_macros(
            "EelevenLayerOneScaledJvpDriftRatio",
            layer_one,
            "scaled_jvp_tail_drift_sq_ratio_ci95_low",
            "scaled_jvp_tail_drift_sq_ratio_ci95_high",
        ),
        macro("EelevenLayerOneObservedDriftRatio", fmt(layer_one["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"])),
        *ci_macros(
            "EelevenLayerOneObservedDriftRatio",
            layer_one,
            "observed_tail_drift_sq_ratio_ci95_low",
            "observed_tail_drift_sq_ratio_ci95_high",
        ),
        macro("EelevenLayerTwoActivationConditionScore", fmt(layer_two["mean_condition_score"])),
        macro("EelevenLayerTwoSsrankBTA", fmt(layer_two["mean_tail_sandwiched_stable_rank"])),
        macro("EelevenLayerTwoTheoremConditionScore", fmt(layer_two["mean_theorem_condition_score"])),
        macro("EelevenLayerTwoLocalOperatorConditionScore", fmt(layer_two["mean_local_operator_condition_score"])),
        macro("EelevenLayerTwoLocalOperatorStableRank", fmt(layer_two["mean_tail_local_operator_stable_rank"])),
        macro("EelevenLayerTwoUnitJvpDriftRatio", fmt(layer_two["geomean_jvp_tail_drift_sq_ratio_spectral_over_fro"])),
        *ci_macros(
            "EelevenLayerTwoUnitJvpDriftRatio",
            layer_two,
            "jvp_tail_drift_sq_ratio_ci95_low",
            "jvp_tail_drift_sq_ratio_ci95_high",
        ),
        macro("EelevenLayerTwoScaledJvpDriftRatio", fmt(layer_two["geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"])),
        *ci_macros(
            "EelevenLayerTwoScaledJvpDriftRatio",
            layer_two,
            "scaled_jvp_tail_drift_sq_ratio_ci95_low",
            "scaled_jvp_tail_drift_sq_ratio_ci95_high",
        ),
        macro("EelevenLayerTwoObservedDriftRatio", fmt(layer_two["geomean_observed_tail_drift_sq_ratio_spectral_over_fro"])),
        *ci_macros(
            "EelevenLayerTwoObservedDriftRatio",
            layer_two,
            "observed_tail_drift_sq_ratio_ci95_low",
            "observed_tail_drift_sq_ratio_ci95_high",
        ),
        "",
    ]

    text = "\n".join(lines)
    for output_path in (DISCUSSION_OUTPUT_PATH, PAPER_OUTPUT_PATH):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(f"saved paper numbers to {DISCUSSION_OUTPUT_PATH} and {PAPER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
