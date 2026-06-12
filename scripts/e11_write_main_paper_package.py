from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.paper_stats import load_paper_stats
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, write_markdown


OUTPUT_PATH = Path("discussion/e11_main_paper_package.md")


def main() -> None:
    stats = load_paper_stats()

    main_items = pd.DataFrame(
        [
            {
                "slot": "Figure 1",
                "artifact": "figures/e11_equal_update/update_spectrum_robustness.png",
                "claim": "Muon has a robust update-spectrum signature.",
                "quantitative_anchor": (
                    f"nrUpdate Muon/Adam={fmt(stats.nr_update['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.nr_update)}; "
                    f"stUpdate={fmt(stats.st_update['geomean_ratio_muon_over_adam'])} {ratio_ci(stats.st_update)}."
                ),
                "reader_takeaway": "The reliable optimizer-intrinsic effect is spectrum shaping, not final-loss superiority.",
            },
            {
                "slot": "Figure 2",
                "artifact": "figures/e11_equal_update/first_order_calibration.png",
                "claim": "One-step progress is locally calibrated by gradient-update alignment.",
                "quantitative_anchor": (
                    f"Spearman(delta_loss, <G,D>)={fmt(stats.calibration_all['spearman_delta_vs_first_order'])} "
                    f"[{fmt(stats.calibration_all['spearman_ci95_low'])}, "
                    f"{fmt(stats.calibration_all['spearman_ci95_high'])}]."
                ),
                "reader_takeaway": "It is meaningful to analyze the positive descent proxy `<G,D>` as a local bridge from geometry to loss decrease.",
            },
            {
                "slot": "Figure 3",
                "artifact": "figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png",
                "claim": "Flat/polar spectral allocation has a norm-geometry boundary.",
                "quantitative_anchor": (
                    f"Frobenius flat/GD={fmt(stats.spectral_fro['geomean_ratio'])} {ratio_ci(stats.spectral_fro)}; "
                    f"operator-norm flat/GD={fmt(stats.spectral_op['geomean_ratio'])} {ratio_ci(stats.spectral_op)}."
                ),
                "reader_takeaway": "Muon-like spectral spreading is locally useful only under the right norm geometry.",
            },
            {
                "slot": "Table 1",
                "artifact": "results/e11_mechanism_boundary/mechanism_boundary_map.csv",
                "claim": "The advantage flips across problem, layer, and control condition.",
                "quantitative_anchor": (
                    f"Boundary rows: {stats.boundary_counts.muon_flat_favorable} Muon/flat favorable, "
                    f"{stats.boundary_counts.own_update_positive_control} own-update positive-control, "
                    f"{stats.boundary_counts.unfavorable} unfavorable, "
                    f"{stats.boundary_counts.mixed} mixed/uncertain."
                ),
                "reader_takeaway": "The paper is a boundary/mechanism paper, not a universal optimizer win paper.",
            },
        ]
    )

    appendix_items = pd.DataFrame(
        [
            {
                "artifact": "discussion/e11_mechanism_theorem_bridge.md",
                "role": "States theorem assumptions, evidence mapping, and safe causal language.",
            },
            {
                "artifact": "discussion/e11_main_figure_captions.md",
                "role": "Paper-safe main figure/table captions with quantitative anchors and interpretation boundaries.",
            },
            {
                "artifact": "discussion/e11_notation_glossary.md",
                "role": "Shared notation source for gradients, updates, ranks, activation products, and recorded diagnostics.",
            },
            {
                "artifact": "discussion/e11_deep_mnist_mlp_probe.md",
                "role": (
                    f"Neural negative control: Deep MNIST nrUpdate={fmt(stats.deep_nr['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.deep_nr)} but first-order="
                    f"{fmt(stats.deep_first['geomean_ratio_muon_over_adam'])} {ratio_ci(stats.deep_first)}."
                ),
            },
            {
                "artifact": "discussion/e11_mnist_patch_probe.md",
                "role": (
                    f"Patch/shared-weight neural sanity check: nrUpdate={fmt(stats.patch_nr['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.patch_nr)} but first-order="
                    f"{fmt(stats.patch_first['geomean_ratio_muon_over_adam'])} {ratio_ci(stats.patch_first)}."
                ),
            },
            {
                "artifact": "discussion/e11_mnist_conv_probe.md",
                "role": (
                    f"True Conv2d neural sanity check: nrUpdate={fmt(stats.conv_nr['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.conv_nr)} but first-order="
                    f"{fmt(stats.conv_first['geomean_ratio_muon_over_adam'])} {ratio_ci(stats.conv_first)}."
                ),
            },
            {
                "artifact": "discussion/e11_stateless_optimizer_trajectory.md",
                "role": (
                    "Trajectory mechanism control: PolarMuon/GD total decrease="
                    f"{fmt(stats.stateless_traj_decrease['geomean_ratio'])} "
                    f"{ratio_ci(stats.stateless_traj_decrease)}."
                ),
            },
            {
                "artifact": "discussion/e11_boundary_predictor_audit.md",
                "role": (
                    "Predictive-boundary gap: best leave-setting-out balanced accuracy="
                    f"{fmt(stats.predictor_best['mean_balanced_accuracy'])} from {stats.predictor_best['feature_set']}; "
                    "chance-filled CI="
                    f"[{fmt(stats.predictor_uncertainty_best['balanced_accuracy_ci95_low'])}, "
                    f"{fmt(stats.predictor_uncertainty_best['balanced_accuracy_ci95_high'])}]."
                ),
            },
            {
                "artifact": "discussion/e11_reviewer_risk_audit.md",
                "role": "Reviewer-risk map and claim discipline.",
            },
            {
                "artifact": "discussion/e11_quantitative_claim_ledger.md",
                "role": "Paper-writing claim ledger with allowed wording, forbidden wording, quantitative anchors, and evidence links.",
            },
            {
                "artifact": "discussion/e11_paper_numbers.tex",
                "role": "LaTeX macros for paper-facing quantitative anchors, generated directly from result CSVs.",
            },
            {
                "artifact": "discussion/e11_reproduction_checklist.md",
                "role": "Minimal main-paper and appendix reproduction map with validation commands.",
            },
        ]
    )

    excluded = pd.DataFrame(
        [
            {
                "artifact": "optimizer switch and LR-sweep figures",
                "reason": "Use as negative controls in appendix; too detailed for the main argument.",
            },
            {
                "artifact": "all individual trajectory/3D figures",
                "reason": "Useful exploratory evidence, but they dilute the main mechanism story.",
            },
            {
                "artifact": "boundary predictor detailed per-setting table",
                "reason": "Keep the main text to the failure summary; detailed rows belong in appendix.",
            },
        ]
    )

    text = f"""# E11 Main Paper Package

This generated note selects the smallest evidence package for a focused manuscript. The point is to keep the main paper readable: three figures plus one table should carry the main claim, with the remaining probes used as safeguards and appendix evidence.

## Main Figure/Table Package

{markdown_table(main_items, ["slot", "artifact", "claim", "quantitative_anchor", "reader_takeaway"])}

## Appendix Allocation

{markdown_table(appendix_items, ["artifact", "role"])}

## Claims To Exclude From Main Text

{markdown_table(excluded, ["artifact", "reason"])}

## Main-Text Claim Order

1. Muon changes update spectra robustly under matched update size.
2. One-step loss decrease is well calibrated by `<G,D>`.
3. The local value of flat/polar spectra depends on the norm geometry.
4. Therefore Muon is a geometry-shaping optimizer with boundary-dependent progress, not a universally better optimizer.

## Drafting Rule

If a sentence cannot be supported by Figure 1, Figure 2, Figure 3, or Table 1, it should probably be in the appendix or discussion rather than in the main result section.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved main paper package to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
