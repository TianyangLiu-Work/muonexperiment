from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_claim_validity_audit.md")


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])},{fmt(row[high])}]"


def paired_geomean_ratio(step_metrics: pd.DataFrame, numerator: str, denominator: str, column: str) -> float:
    wide = step_metrics.pivot(index="seed", columns="geometry", values=column)
    return math.exp((wide[numerator] / wide[denominator]).map(math.log).mean())


def main() -> None:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    one_step_steps = pd.read_csv("results/e11_long_tail_one_step/step_metrics.csv")
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
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")

    positive = head_tail.loc["high_head_rank_low_tail_srank"]
    negative = head_tail.loc["low_head_rank_high_tail_srank"]
    layer_one = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_two = layerwise[layerwise["layer"].eq(2)].iloc[0]
    alignment_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "alignment")
    update_fro_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "update_fro_norm")
    update_op_ratio = paired_geomean_ratio(one_step_steps, "spectral", "frobenius", "update_op_norm")
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
    cifar_resnet_tail_quality_worst = cifar_resnet_tail_quality.loc[
        cifar_resnet_tail_quality["tail_output_drift_sq_ratio_ci95_high"].idxmax()
    ]
    cifar_resnet_tail_quality_best_tail_accuracy = cifar_resnet_tail_quality.loc[
        cifar_resnet_tail_quality["mean_tail_accuracy_before"].idxmax()
    ]

    claim_status = pd.DataFrame(
        [
            {
                "claim": "The synthetic rank/sensitivity condition has the intended sign.",
                "status": "supported",
                "evidence": (
                    f"Positive condition squared drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(positive, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"negative condition squared drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(negative, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}. "
                    f"On the ResNet checkpoint sweep, mean gradient nuclear rank alone has Pearson correlation "
                    f"{fmt(cifar_resnet_rank_proxy_pearson['estimate'])} "
                    f"CI=[{fmt(cifar_resnet_rank_proxy_pearson['ci95_low'])},{fmt(cifar_resnet_rank_proxy_pearson['ci95_high'])}] "
                    f"with log squared drift ratio, showing that the head-rank proxy alone is not the full downstream-aware condition. "
                    f"The final-layer ResNet condition check measures nrank(G_H)/srank(H_T) directly for fc.weight; "
                    f"the weakest mean score is {fmt(cifar_resnet_fc_condition_weakest['mean_condition_score_nrank_over_tail_srank'])}, "
                    f"all {len(cifar_resnet_fc_condition_points)} seed/checkpoint points favor spectral, and the worst final-layer-only drift ratio is "
                    f"{fmt(cifar_resnet_fc_condition_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(cifar_resnet_fc_condition_worst, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "main_loophole": "The final layer now has a downstream-aware natural-task check, but convolutional blocks still need all-layer tail sensitivity or JVP diagnostics.",
            },
            {
                "claim": "Spectral/polar one-step updates reduce held-out tail-example logit drift at matched head gain.",
                "status": "supported in small digits and CIFAR-100-LT ResNet diagnostic",
                "evidence": (
                    f"Drift-squared ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"head-alignment ratio={fmt(alignment_ratio)}; "
                    f"Frobenius-norm ratio={fmt(update_fro_ratio)}; "
                    f"operator-norm ratio={fmt(update_op_ratio)}; "
                    f"spectral-lower paired fraction={fmt(one_step['spectral_less_tail_output_drift_fraction'])} over "
                    f"{int(one_step['seeds'])} digits seeds. CIFAR-100-LT ResNet18 gives drift-squared ratio="
                    f"{fmt(cifar_resnet['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(cifar_resnet, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')} "
                    f"with spectral-lower fraction={fmt(cifar_resnet['spectral_less_tail_output_drift_fraction'])} "
                    f"over {int(cifar_resnet['seeds'])} seeds. At target head gain 0.002, the ResNet squared drift ratio is "
                    f"{fmt(cifar_resnet_rho002['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(cifar_resnet_rho002, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}. "
                    f"Across the ResNet checkpoint sweep, the worst drift-ratio CI upper endpoint is "
                    f"{fmt(cifar_resnet_checkpoint_worst['tail_output_drift_sq_ratio_ci95_high'])} at "
                    f"{int(cifar_resnet_checkpoint_worst['warmup_steps'])} warmup steps. "
                    f"A tail-rich ResNet control with 300 tail-train examples per class reaches pre-update tail accuracy "
                    f"{fmt(cifar_resnet_tail_quality_best_tail_accuracy['mean_tail_accuracy_before'])} "
                    f"CI={ci(cifar_resnet_tail_quality_best_tail_accuracy, 'tail_accuracy_before_ci95_low', 'tail_accuracy_before_ci95_high')} "
                    f"while preserving worst drift ratio {fmt(cifar_resnet_tail_quality_worst['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(cifar_resnet_tail_quality_worst, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "main_loophole": "The result is about logits/function drift; the matched step is norm-specific, not uniformly smaller, and even the tail-rich control is not a tuned long-tail optimizer benchmark.",
            },
            {
                "claim": "Lower tail-example logit drift implies lower tail loss or better tail accuracy.",
                "status": "not supported",
                "evidence": (
                    f"One-step tail-loss increase difference spectral-minus-Frobenius="
                    f"{fmt(one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={ci(one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}; "
                    f"tail-accuracy-drop difference={fmt(one_step['mean_tail_accuracy_drop_diff_spectral_minus_fro'])} "
                    f"CI={ci(one_step, 'tail_accuracy_drop_diff_ci95_low', 'tail_accuracy_drop_diff_ci95_high')}. "
                    f"CIFAR-100-LT ResNet18 has lower one-step tail-loss increase "
                    f"diff={fmt(cifar_resnet['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={ci(cifar_resnet, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}, "
                    f"and at target head gain 0.002 the tail-loss increase diff is "
                    f"{fmt(cifar_resnet_rho002['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"CI={ci(cifar_resnet_rho002, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}; "
                    f"but tail-accuracy-drop diff still crosses zero "
                    f"CI={ci(cifar_resnet, 'tail_accuracy_drop_diff_ci95_low', 'tail_accuracy_drop_diff_ci95_high')}. "
                    f"The checkpoint sweep's best pre-update tail accuracy is only "
                    f"{fmt(cifar_resnet_checkpoint_best_tail_accuracy['mean_tail_accuracy_before'])}, so it is not "
                    f"evidence for preserving a high-quality tail predictor."
                ),
                "main_loophole": "One-step tail-loss improvement in the ResNet diagnostic is encouraging but still not a retuned long-horizon classification result.",
            },
            {
                "claim": "The tail-drift reduction persists across a short head-only horizon.",
                "status": "supported for eight steps",
                "evidence": (
                    f"Final squared drift ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"CI={ci(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}; "
                    f"drift-area ratio={fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"CI={ci(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}."
                ),
                "main_loophole": "Eight head-only steps are still a diagnostic, not a full optimizer benchmark.",
            },
            {
                "claim": "Spectral/polar directions are intrinsically less tail-sensitive layerwise.",
                "status": "not supported",
                "evidence": (
                    f"Unit-JVP squared drift ratios are above one: layer 1="
                    f"{fmt(layer_one['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"layer 2={fmt(layer_two['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "main_loophole": "The supported mechanism is matched-head-gain scaling, not lower unit-direction sensitivity.",
            },
            {
                "claim": "Matched-head-gain scaling explains the observed lower layerwise drift in a norm-specific way.",
                "status": "supported in the current layerwise diagnostic",
                "evidence": (
                    f"Scaled ratios: layer 1={fmt(layer_one['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"layer 2={fmt(layer_two['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}; "
                    f"observed ratios: layer 1={fmt(layer_one['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}, "
                    f"layer 2={fmt(layer_two['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "main_loophole": "This should be rechecked in larger architectures before claiming generality.",
            },
            {
                "claim": "The clean polar direction has selected-state compatibility with Muon-style momentum directions.",
                "status": "supported as selected-state diagnostic",
                "evidence": (
                    f"polar(M_t) squared drift ratio={fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={ci(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"short-trajectory NS(M_t) squared drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"CI={ci(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "main_loophole": "This is still a small-digits selected-state compatibility check; real long-tail practical Muon training and final performance remain unproven.",
            },
        ]
    )

    evidence_tables = pd.DataFrame(
        [
            {
                "table": "Synthetic boundary",
                "path": "results/e11_head_tail_interference/pair_summary.csv",
                "role": "Positive and negative rank/sensitivity condition check.",
            },
            {
                "table": "Long-tail one-step",
                "path": "results/e11_long_tail_one_step/pair_summary.csv",
                "role": "Matched-head-gain held-out tail-example logit drift and performance caveat.",
            },
            {
                "table": "CIFAR-100-LT ResNet18 one-step",
                "path": "results/e11_cifar100_resnet_one_step/pair_summary.csv",
                "role": "Larger visual-data architecture check for matched-head-gain tail drift under fixed BatchNorm state.",
            },
            {
                "table": "CIFAR-100-LT ResNet18 smaller-head-gain check",
                "path": "results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv",
                "role": "Target-head-gain scale robustness check at 0.002 times head loss.",
            },
            {
                "table": "CIFAR-100-LT ResNet18 checkpoint-quality sweep",
                "path": "results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv",
                "role": "Warmup-checkpoint robustness check across 250, 500, 1000, and 2000 AdamW steps.",
            },
            {
                "table": "CIFAR-100-LT ResNet18 condition-proxy scatter",
                "path": "results/e11_cifar100_resnet_condition_proxy_scatter/summary.csv",
                "role": "Natural-task rank-side proxy check showing that head-gradient rank alone is not the downstream-aware condition.",
            },
            {
                "table": "CIFAR-100-LT ResNet18 final-layer condition scatter",
                "path": "results/e11_cifar100_resnet_fc_condition_scatter/summary.csv",
                "role": "Final-layer downstream-aware condition check comparing head-gradient nuclear rank with tail feature stable rank.",
            },
            {
                "table": "CIFAR-100 ResNet18 tail-quality control",
                "path": "results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv",
                "role": "Tail-rich checkpoint control showing lower matched-gain drift with substantially higher pre-update tail accuracy.",
            },
            {
                "table": "Long-tail Muon-style compatibility",
                "path": "results/e11_long_tail_muon_bridge/pair_summary.csv",
                "role": "Fixed-checkpoint compatibility check for polar(G_t), polar(M_t), and Newton-Schulz directions under matched head gain.",
            },
            {
                "table": "Long-tail practical Muon trajectory compatibility",
                "path": "results/e11_long_tail_practical_muon_bridge/summary.csv",
                "role": "Short trajectory-level matched-head-gain compatibility check for practical momentum/NS directions.",
            },
            {
                "table": "Head-only forgetting",
                "path": "results/e11_long_tail_forgetting/summary.csv",
                "role": "Eight-step drift persistence under matched head-only update schedules.",
            },
            {
                "table": "Layerwise diagnostic",
                "path": "results/e11_long_tail_layerwise/summary.csv",
                "role": "Unit JVP, scaled JVP, and observed drift decomposition.",
            },
        ]
    )

    text = f"""# E11 Claim Validity Audit

This audit separates what the current head-to-tail experiments directly establish from what remains a vulnerability. It is generated from the current result CSVs so that the numerical claims are reproducible.

## Claim Status

{markdown_table(claim_status, ["claim", "status", "evidence", "main_loophole"])}

## Evidence Tables

{markdown_table(evidence_tables, ["table", "path", "role"])}

## Defensible Formulation

The defensible formulation is: **idealized spectral/polar directions can reduce head-to-tail drift on tail-example logits at matched head gain under a measurable rank/sensitivity condition.**

The current data do **not** justify saying that this already proves better tail classification, full practical Muon training behavior, or broad optimizer superiority.

## Strongest Remaining Loopholes

1. The CIFAR-100-LT ResNet evidence is still local diagnostics plus a tail-rich control; it is not a modern long-tail optimizer benchmark.
2. The Muon-style compatibility evidence is local and small-scale; real long-tail practical Muon training remains unchecked.
3. The final-layer ResNet condition scatter is useful, but all-layer convolutional downstream-aware condition diagnostics remain missing.
4. The current performance evidence is weaker than the function-drift evidence.
5. The detailed layerwise JVP mechanism has only been checked in the current small MLP.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved claim audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
