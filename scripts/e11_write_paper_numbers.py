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
    imbalance = pd.read_csv("results/e11_long_tail_imbalance_ablation/summary.csv")
    checkpoint_sweep = pd.read_csv("results/e11_long_tail_checkpoint_sweep/summary.csv")
    class_partition_sweep = pd.read_csv("results/e11_long_tail_class_partition_sweep/summary.csv")
    rho_sweep = pd.read_csv("results/e11_long_tail_rho_sweep/summary.csv")
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
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
