from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_research_synthesis.md")


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def main() -> None:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    state_control = pd.read_csv(
        "results/e11_long_tail_muon_state_source_control/summary.csv"
    ).set_index(["state_source", "direction"])
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")
    layer_one = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_two = layerwise[layerwise["layer"].eq(2)].iloc[0]

    positive = head_tail.loc["high_head_rank_low_tail_srank"]
    negative = head_tail.loc["low_head_rank_high_tail_srank"]

    claim_cards = pd.DataFrame(
        [
            {
                "claim": "The paper has a precise local mechanism target.",
                "status": "supported as a focused paper scope",
                "evidence": "All main diagnostics compare spectral/polar and Frobenius/GD-style directions at matched head gain.",
                "interpretation": "The paper is about head-to-tail function interference, not an optimizer leaderboard.",
                "caveat": "Full Muon behavior still requires a bridge ablation with momentum and Newton-Schulz approximation.",
            },
            {
                "claim": "The rank/sensitivity condition has the correct synthetic boundary behavior.",
                "status": "supported",
                "evidence": (
                    f"Positive case squared drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(positive, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"negative case squared drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(negative, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                "interpretation": "The constructed examples verify that the theorem condition is not merely decorative.",
                "caveat": "This does not by itself prove predictive power on natural tasks.",
            },
            {
                "claim": "Long-tailed one-step diagnostics show lower tail-example logit drift.",
                "status": "supported",
                "evidence": (
                    f"Drift-squared ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"spectral-lower paired fraction={fmt(one_step['spectral_less_tail_output_drift_fraction'])}."
                ),
                "interpretation": "At the same first-order head progress, spectral/polar updates perturb logits on held-out tail examples less.",
                "caveat": (
                    f"Tail-loss increase diff is {fmt(one_step['mean_tail_loss_increase_diff_spectral_minus_fro'])} "
                    f"{ci(one_step, 'tail_loss_increase_diff_ci95_low', 'tail_loss_increase_diff_ci95_high')}, "
                    "so this is not a tail-performance improvement claim."
                ),
            },
            {
                "claim": "The drift advantage persists in a short head-only forgetting horizon.",
                "status": "supported",
                "evidence": (
                    f"Final squared drift ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}; "
                    f"area ratio={fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}."
                ),
                "interpretation": "The effect survives repeated head-only updates in the small diagnostic.",
                "caveat": "The horizon is eight steps and still not a full long-tailed training benchmark.",
            },
            {
                "claim": "Muon-style directions show selected-state compatibility across a short practical trajectory.",
                "status": "supported as a compatibility diagnostic",
                "evidence": (
                    f"polar(M_t) squared drift ratio={fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"short-trajectory NS(M_t) squared drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"Fro/GD-state NS(M_t) squared drift ratio="
                    f"{fmt(state_control.loc[('fro_gd_trajectory', 'ns_momentum'), 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(state_control.loc[('fro_gd_trajectory', 'ns_momentum')], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                "interpretation": "Sampled Muon-style momentum/NS states have squared drift ratios compatible with the clean polar(G_t) mechanism, including on a Fro/GD-generated state-source control.",
                "caveat": "This is still a local matched-head-gain diagnostic on small digits, not a full long-tail optimizer benchmark.",
            },
            {
                "claim": "Layerwise evidence points to matched-head-gain scaling, not intrinsically safer directions.",
                "status": "supported as a mechanism refinement",
                "evidence": (
                    f"Layer 1 unit/scaled/observed squared drift ratios="
                    f"{fmt(layer_one['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_one['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_one['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}; "
                    f"Layer 2={fmt(layer_two['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_two['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_two['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                "interpretation": "Spectral/polar directions can be more unit-tail-sensitive but still cause lower observed drift because they need less scaling for the same head gain.",
                "caveat": "This caveat should appear anywhere the paper uses intuitive language about being less tail-disruptive.",
            },
        ]
    )

    implications = pd.DataFrame(
        [
            {
                "section": "Main positive finding",
                "use": "Present matched-head-gain spectral/polar updates as reducing head-to-tail function drift.",
            },
            {
                "section": "Mechanism",
                "use": "Use the nrank(G_H) versus ssrank(B_T,A_T) condition and the layerwise scaling diagnostic.",
            },
            {
                "section": "Caveats",
                "use": "Separate drift from tail loss, accuracy, final performance, and full Muon optimizer behavior.",
            },
            {
                "section": "Appendix / guardrails",
                "use": "Use older condition-geometry and Muon/Adam artifacts only as background unless the main paper explicitly needs them.",
            },
        ]
    )

    text = f"""# E11 Research Synthesis

This generated note is the current paper-facing synthesis of the E11 experiments. It is intentionally stricter than the exploratory condition-geometry artifacts: every claim below is tied to a quantitative result and an explicit caveat.

For the figure/CSV source behind each claim, see [E11 evidence index](e11_evidence_index.md). For claim wording constraints, see [E11 quantitative claim ledger](e11_quantitative_claim_ledger.md). For current paper risks, see [E11 reviewer risk audit](e11_reviewer_risk_audit.md).

## Current Thesis

**The current paper is a head-to-tail interference mechanism paper.** In long-tailed small-batch training, head-only updates can perturb logits on held-out tail examples while those examples are absent from the update. The supported claim is that an idealized spectral/polar direction can reduce this tail-example logit drift at matched head gain under a measurable rank/sensitivity condition. Fixed-checkpoint and short-trajectory Muon-style directions show selected-state compatibility with the polar mechanism, but this is still not a claim that full Muon improves final tail accuracy.

## Claim Cards

{markdown_table(claim_cards, ["claim", "status", "evidence", "interpretation", "caveat"])}

## Report Implications

{markdown_table(implications, ["section", "use"])}

## Recommended Main Story

1. Define the head-to-tail interference problem and the matched-head-gain protocol.
2. Prove the local condition using \\(\\operatorname{{nrank}}(G_H)\\) and \\(\\operatorname{{ssrank}}(B_T,A_T)\\).
3. Show the synthetic positive and negative boundary cases.
4. Show the long-tailed one-step, Muon-style compatibility fixed/trajectory, and eight-step forgetting drift diagnostics.
5. Use the layerwise diagnostic to state the correct mechanism: scaled head-gain efficiency rather than intrinsically lower tail sensitivity.
6. State the scope boundary: drift is not performance, and selected-state Muon compatibility is not full Muon training.

## Remaining Open Gaps

1. A real long-tail benchmark is still needed before making broad performance claims.
2. A real long-tail practical Muon benchmark is still needed before claiming the mechanism explains full Muon training outcomes.
3. Larger-architecture layerwise diagnostics are still needed before claiming architecture-level generality.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved research synthesis to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
