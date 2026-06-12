from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.paper_stats import load_paper_stats
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, write_markdown


OUTPUT_PATH = Path("discussion/e11_main_figure_captions.md")


def main() -> None:
    stats = load_paper_stats()
    captions = pd.DataFrame(
        [
            {
                "slot": "Figure 1",
                "artifact": "figures/e11_equal_update/update_spectrum_robustness.png",
                "caption": (
                    "Muon produces a reproducible update-spectrum signature under matched global update size. "
                    f"Across the current E11 families, Muon/Adam geomean ratios are "
                    f"{fmt(stats.nr_update['geomean_ratio_muon_over_adam'])} {ratio_ci(stats.nr_update)} "
                    f"for `nrUpdate` and {fmt(stats.st_update['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(stats.st_update)} for `stUpdate`."
                ),
                "interpretation": (
                    "This supports the optimizer-intrinsic spectrum-shaping claim, but it should not be read as "
                    "evidence that higher update rank directly improves optimization."
                ),
            },
            {
                "slot": "Figure 2",
                "artifact": "figures/e11_equal_update/first_order_calibration.png",
                "caption": (
                    "Observed one-step loss decrease is well calibrated by the local gradient-update alignment "
                    "`<G,D>`. "
                    f"The Spearman correlation is {fmt(stats.calibration_all['spearman_delta_vs_first_order'])} "
                    f"[{fmt(stats.calibration_all['spearman_ci95_low'])}, "
                    f"{fmt(stats.calibration_all['spearman_ci95_high'])}], with "
                    f"{fmt(stats.calibration_all['within_factor_2'])} of positive decreases within a factor of two."
                ),
                "interpretation": (
                    "This justifies using one-step alignment as the local bridge from update geometry to loss "
                    "decrease, but it is not a convergence or final-performance claim."
                ),
            },
            {
                "slot": "Figure 3",
                "artifact": "figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png",
                "caption": (
                    "The value of flat/polar spectral allocation depends on the norm geometry of the local "
                    "comparison. The flat/GD first-order ratio is "
                    f"{fmt(stats.spectral_fro['geomean_ratio'])} {ratio_ci(stats.spectral_fro)} under a "
                    f"Frobenius budget, but {fmt(stats.spectral_op['geomean_ratio'])} "
                    f"{ratio_ci(stats.spectral_op)} under an operator-norm budget."
                ),
                "interpretation": (
                    "This is the main mechanism boundary: Muon-like spectral spreading is locally useful only "
                    "when the task/layer/norm geometry rewards spreading."
                ),
            },
            {
                "slot": "Table 1",
                "artifact": "results/e11_mechanism_boundary/mechanism_boundary_map.csv",
                "caption": (
                    "Mechanism-boundary map for the current controlled experiments. The map contains "
                    f"{stats.boundary_counts.muon_flat_favorable} Muon/flat favorable rows, "
                    f"{stats.boundary_counts.own_update_positive_control} own-update positive-control row, "
                    f"{stats.boundary_counts.unfavorable} unfavorable rows, and "
                    f"{stats.boundary_counts.mixed} mixed or uncertain row."
                ),
                "interpretation": (
                    "This table frames the paper as a boundary/mechanism study rather than an optimizer "
                    "leaderboard; the current boundary is descriptive, not yet predictive out of sample."
                ),
            },
        ]
    )

    text = f"""# E11 Main Figure Captions

This generated note drafts paper-safe captions for the proposed main figures and table. Each caption includes the quantitative anchor and the intended interpretation boundary.

## Captions

{markdown_table(captions, ["slot", "artifact", "caption", "interpretation"])}

## Caption Discipline

1. Every main caption should include a quantitative anchor.
2. Captions should distinguish the measured geometry effect from optimization or final-performance claims.
3. The wording should follow [e11_quantitative_claim_ledger.md](e11_quantitative_claim_ledger.md) and [e11_notation_glossary.md](e11_notation_glossary.md).
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved main figure captions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
