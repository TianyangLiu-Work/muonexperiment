from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.paper_stats import load_paper_stats
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, write_markdown


OUTPUT_PATH = Path("discussion/e11_quantitative_claim_ledger.md")


def main() -> None:
    stats = load_paper_stats()

    ledger = pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "paper_status": "main",
                "claim": "Muon has a reproducible update-spectrum signature under matched update size.",
                "quantitative_evidence": (
                    f"nrUpdate Muon/Adam={fmt(stats.nr_update['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.nr_update)}; "
                    f"stUpdate={fmt(stats.st_update['geomean_ratio_muon_over_adam'])} {ratio_ci(stats.st_update)}."
                ),
                "allowed_wording": "Muon shapes update spectra toward higher numerical/effective rank in the tested settings.",
                "do_not_write": "Do not say this alone proves better optimization or a better final solution.",
                "evidence": "figures/e11_equal_update/update_spectrum_robustness.png; results/e11_equal_update/update_spectrum_summary.csv",
            },
            {
                "claim_id": "C2",
                "paper_status": "main diagnostic",
                "claim": "Observed one-step decrease is locally calibrated by gradient-update alignment.",
                "quantitative_evidence": (
                    f"Spearman(delta_loss, <G, D>)="
                    f"{fmt(stats.calibration_all['spearman_delta_vs_first_order'])} "
                    f"[{fmt(stats.calibration_all['spearman_ci95_low'])}, "
                    f"{fmt(stats.calibration_all['spearman_ci95_high'])}]; "
                    f"within-factor-2={fmt(stats.calibration_all['within_factor_2'])}."
                ),
                "allowed_wording": "The local diagnostic `<G,D>` is a reliable one-step progress proxy here.",
                "do_not_write": "Do not turn this into a long-horizon convergence or final-loss claim.",
                "evidence": "figures/e11_equal_update/first_order_calibration.png; results/e11_equal_update/first_order_calibration_summary.csv",
            },
            {
                "claim_id": "C3",
                "paper_status": "main mechanism",
                "claim": "Flat/polar spectral allocation has a norm-geometry boundary.",
                "quantitative_evidence": (
                    f"flat/GD under Frobenius budget={fmt(stats.spectral_fro['geomean_ratio'])} "
                    f"{ratio_ci(stats.spectral_fro)}; "
                    f"under operator-norm budget={fmt(stats.spectral_op['geomean_ratio'])} "
                    f"{ratio_ci(stats.spectral_op)}."
                ),
                "allowed_wording": "Muon-like spectral spreading is locally useful only when the local norm geometry rewards spreading.",
                "do_not_write": "Do not say high-rank updates generally cause larger progress.",
                "evidence": "figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png; discussion/e11_theory_note.md",
            },
            {
                "claim_id": "C4",
                "paper_status": "main boundary",
                "claim": "The local advantage is conditional across task, layer, and control choices.",
                "quantitative_evidence": (
                    f"Boundary map: {stats.boundary_counts.muon_flat_favorable} Muon/flat favorable rows, "
                    f"{stats.boundary_counts.own_update_positive_control} own-update positive-control row, "
                    f"{stats.boundary_counts.unfavorable} unfavorable rows, "
                    f"{stats.boundary_counts.mixed} mixed/uncertain rows."
                ),
                "allowed_wording": "The evidence supports a boundary/mechanism framing rather than an optimizer leaderboard framing.",
                "do_not_write": "Do not describe the current boundary map as a predictive law.",
                "evidence": "results/e11_mechanism_boundary/mechanism_boundary_map.csv; discussion/e11_mechanism_boundary.md",
            },
            {
                "claim_id": "C5",
                "paper_status": "appendix support",
                "claim": "Neural MNIST variants preserve update-spectrum shaping while showing Adam-favorable first-order progress.",
                "quantitative_evidence": (
                    f"Deep MNIST nrUpdate={fmt(stats.deep_nr['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.deep_nr)}; "
                    f"stUpdate={fmt(stats.deep_st['geomean_ratio_muon_over_adam'])} {ratio_ci(stats.deep_st)}; "
                    f"first-order Muon/Adam={fmt(stats.deep_first['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.deep_first)}. "
                    f"Patch MNIST nrUpdate={fmt(stats.patch_nr['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.patch_nr)}; stUpdate={fmt(stats.patch_st['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.patch_st)}; first-order={fmt(stats.patch_first['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.patch_first)}. "
                    f"ConvNet MNIST nrUpdate={fmt(stats.conv_nr['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.conv_nr)}; stUpdate={fmt(stats.conv_st['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.conv_st)}; first-order={fmt(stats.conv_first['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.conv_first)}."
                ),
                "allowed_wording": "The neural sanity checks support the spectrum-shaping claim and caution against equating it with progress.",
                "do_not_write": "Do not present the current neural evidence as a modern deep-learning or long-horizon benchmark result.",
                "evidence": "discussion/e11_deep_mnist_mlp_probe.md; discussion/e11_mnist_patch_probe.md; discussion/e11_mnist_conv_probe.md; figures/e11_mnist_conv_probe/mnist_conv_first_order_ratios.png",
            },
            {
                "claim_id": "C6",
                "paper_status": "appendix mechanism control",
                "claim": "A stateless polar trajectory keeps higher update rank without guaranteeing larger total decrease.",
                "quantitative_evidence": (
                    f"PolarMuon/GD mean_nrUpdate={fmt(stats.stateless_traj_nr['geomean_ratio'])} "
                    f"{ratio_ci(stats.stateless_traj_nr)}; "
                    f"total_decrease={fmt(stats.stateless_traj_decrease['geomean_ratio'])} "
                    f"{ratio_ci(stats.stateless_traj_decrease)}."
                ),
                "allowed_wording": "Polar direction alone is insufficient to explain progress under Frobenius-matched short trajectories.",
                "do_not_write": "Do not treat this as a fully retuned optimizer comparison.",
                "evidence": "discussion/e11_stateless_optimizer_trajectory.md; results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv",
            },
            {
                "claim_id": "C7",
                "paper_status": "gap",
                "claim": "The current boundary model is descriptive, not predictive out of sample.",
                "quantitative_evidence": (
                    f"Best leave-setting-out balanced accuracy="
                    f"{fmt(stats.predictor_best['mean_balanced_accuracy'])} from {stats.predictor_best['feature_set']} "
                    f"when degenerate settings are skipped; chance-filled best="
                    f"{fmt(stats.predictor_uncertainty_best['mean_balanced_accuracy_chance_filled'])} "
                    f"CI=[{fmt(stats.predictor_uncertainty_best['balanced_accuracy_ci95_low'])}, "
                    f"{fmt(stats.predictor_uncertainty_best['balanced_accuracy_ci95_high'])}]."
                ),
                "allowed_wording": "The current results identify candidate boundary variables but not yet a reliable held-out predictor.",
                "do_not_write": "Do not claim the paper already predicts where Muon will help on unseen tasks.",
                "evidence": "discussion/e11_boundary_predictor_audit.md; results/e11_boundary_predictor/boundary_predictor_summary.csv",
            },
        ]
    )

    priority = pd.DataFrame(
        [
            {
                "priority": "write first",
                "claim_ids": "C1, C2, C3, C4",
                "reason": "These four claims form the shortest coherent paper story.",
            },
            {
                "priority": "appendix",
                "claim_ids": "C5, C6",
                "reason": "These are controls that protect the main story from overclaiming.",
            },
            {
                "priority": "future work / limitation",
                "claim_ids": "C7",
                "reason": "This is the main remaining gap before claiming a predictive theory.",
            },
        ]
    )

    text = f"""# E11 Quantitative Claim Ledger

This generated ledger is the paper-writing guardrail. Each claim below must be stated with its quantitative anchor and caveat. Claims without an entry here should stay out of the main result section until a script adds the corresponding evidence.

## Claim Ledger

{markdown_table(ledger, ["claim_id", "paper_status", "claim", "quantitative_evidence", "allowed_wording", "do_not_write", "evidence"])}

## Writing Priority

{markdown_table(priority, ["priority", "claim_ids", "reason"])}

## Practical Rule

Use `C1 -> C2 -> C3 -> C4` as the main paper sequence. Use `C5` and `C6` to show why the story is not a naive high-rank-implies-progress story. Use `C7` to state the honest limitation: the current boundary is measurable and structured, but it is not yet predictive out of sample.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved quantitative claim ledger to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
