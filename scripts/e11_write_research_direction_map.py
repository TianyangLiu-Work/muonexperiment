from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_research_direction_map.md")


def ci(row: pd.Series, low: str, high: str) -> str:
    return f"[{fmt(row[low])}, {fmt(row[high])}]"


def claim_row(
    *,
    question: str,
    claim: str,
    status: str,
    evidence: str,
    interpretation: str,
    next_test: str,
) -> dict[str, str]:
    return {
        "question": question,
        "claim": claim,
        "status": status,
        "evidence": evidence,
        "interpretation": interpretation,
        "next_test": next_test,
    }


def main() -> None:
    head_tail = pd.read_csv("results/e11_head_tail_interference/pair_summary.csv").set_index("setting")
    one_step = pd.read_csv("results/e11_long_tail_one_step/pair_summary.csv").iloc[0]
    muon_bridge = pd.read_csv("results/e11_long_tail_muon_bridge/pair_summary.csv").set_index("direction")
    practical_bridge = pd.read_csv("results/e11_long_tail_practical_muon_bridge/summary.csv").set_index("direction")
    practical_training = pd.read_csv("results/e11_long_tail_practical_training/summary.csv").iloc[0]
    practical_lr_sweep = pd.read_csv("results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv")
    forgetting = pd.read_csv("results/e11_long_tail_forgetting/summary.csv").iloc[0]
    layerwise = pd.read_csv("results/e11_long_tail_layerwise/summary.csv")

    positive = head_tail.loc["high_head_rank_low_tail_srank"]
    negative = head_tail.loc["low_head_rank_high_tail_srank"]
    layer_one = layerwise[layerwise["layer"].eq(1)].iloc[0]
    layer_two = layerwise[layerwise["layer"].eq(2)].iloc[0]
    lr_sweep = practical_lr_sweep.assign(muon_lr_rounded=practical_lr_sweep["muon_lr"].round(3)).set_index("muon_lr_rounded")

    claims = pd.DataFrame(
        [
            claim_row(
                question="What is the current paper's mechanism target?",
                claim="Head-only updates can interfere with tail logits while tail samples are absent.",
                status="paper framing",
                evidence="The current main experiments all use matched-head-gain spectral/polar versus Frobenius/GD-style updates.",
                interpretation="This makes the project a local function-drift mechanism paper, not an optimizer leaderboard.",
                next_test="Keep every main result normalized by matched head gain or explicitly explain why it is not.",
            ),
            claim_row(
                question="When should spectral/polar updates reduce head-to-tail drift?",
                claim="The rank/sensitivity condition predicts the synthetic positive and negative cases.",
                status="supported mechanism boundary",
                evidence=(
                    f"Positive: nrank(G_H)={fmt(positive['mean_head_gradient_nuclear_rank'])}, "
                    f"ssrank(B_T,A_T)={fmt(positive['mean_tail_downstream_aware_stable_rank'])}, "
                    f"drift ratio={fmt(positive['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(positive, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}. "
                    f"Negative: nrank(G_H)={fmt(negative['mean_head_gradient_nuclear_rank'])}, "
                    f"ssrank(B_T,A_T)={fmt(negative['mean_tail_downstream_aware_stable_rank'])}, "
                    f"drift ratio={fmt(negative['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(negative, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}."
                ),
                interpretation="The condition has falsifiable sign content in controlled examples.",
                next_test="Add natural-task measurements of the same condition before presenting it as predictive beyond the constructed probe.",
            ),
            claim_row(
                question="Does the real-data one-step diagnostic support lower tail drift?",
                claim="Yes, for tail logits at matched first-order head gain.",
                status="supported in small long-tailed digits",
                evidence=(
                    f"Drift-squared ratio={fmt(one_step['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(one_step, 'tail_output_drift_sq_ratio_ci95_low', 'tail_output_drift_sq_ratio_ci95_high')}; "
                    f"spectral-lower paired fraction={fmt(one_step['spectral_less_tail_output_drift_fraction'])} over "
                    f"{int(one_step['seeds'])} seeds."
                ),
                interpretation="Spectral/polar updates perturb held-out tail logits less for the same head progress in this diagnostic.",
                next_test="Replicate on a real long-tailed benchmark before claiming final tail-performance relevance.",
            ),
            claim_row(
                question="Does the lower drift persist beyond one update?",
                claim="The 8-step head-only forgetting probe keeps lower spectral tail drift.",
                status="supported short-horizon diagnostic",
                evidence=(
                    f"Final drift ratio={fmt(forgetting['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'final_tail_output_drift_sq_ratio_ci95_low', 'final_tail_output_drift_sq_ratio_ci95_high')}; "
                    f"drift-area ratio={fmt(forgetting['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} "
                    f"{ci(forgetting, 'tail_output_drift_area_ratio_ci95_low', 'tail_output_drift_area_ratio_ci95_high')}."
                ),
                interpretation="The one-step drift effect is not isolated to a single update in the current small diagnostic.",
                next_test="Extend only after deciding whether the paper will include a full training benchmark.",
            ),
            claim_row(
                question="Does the clean polar direction connect to Muon-style state?",
                claim="There is selected-state compatibility through momentum polar and a short practical NS-Muon trajectory.",
                status="supported compatibility check",
                evidence=(
                    f"polar(M_t) drift ratio={fmt(muon_bridge.loc['polar_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(muon_bridge.loc['polar_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}; "
                    f"short-trajectory NS(M_t) drift ratio={fmt(practical_bridge.loc['ns_momentum', 'geomean_tail_output_drift_sq_ratio_vs_fro'])} "
                    f"{ci(practical_bridge.loc['ns_momentum'], 'tail_output_drift_sq_ratio_vs_fro_ci95_low', 'tail_output_drift_sq_ratio_vs_fro_ci95_high')}."
                ),
                interpretation="Sampled Muon-style momentum/NS states have drift ratios compatible with the local polar mechanism.",
                next_test="Test the same compatibility pattern on real long-tail benchmarks and larger models before claiming broad Muon training behavior.",
            ),
            claim_row(
                question="What does the layerwise diagnostic say the mechanism is?",
                claim="The supported mechanism is scaled head-gain efficiency, not lower unit-direction tail sensitivity.",
                status="supported mechanism refinement",
                evidence=(
                    f"Layer 1 unit/scaled/observed ratios="
                    f"{fmt(layer_one['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_one['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_one['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}; "
                    f"Layer 2="
                    f"{fmt(layer_two['geomean_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_two['geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro'])}/"
                    f"{fmt(layer_two['geomean_observed_tail_drift_sq_ratio_spectral_over_fro'])}."
                ),
                interpretation="The spectral unit direction can be more tail-sensitive, but it needs less scaling to achieve the same head gain.",
                next_test="Add a larger-architecture layerwise diagnostic if this mechanism is presented as architecture-level.",
            ),
            claim_row(
                question="What should remain outside the main claim?",
                claim="Broad Muon training behavior, tail accuracy, and real benchmark performance remain open.",
                status="scope boundary",
                evidence=(
                    "One-step tail loss/accuracy do not improve, while the small practical run has "
                    f"tail eval loss ratio={fmt(practical_training['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} "
                    f"but tail accuracy diff={fmt(practical_training['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])}. "
                    f"The LR sweep shows lr=0.1 worsens tail loss/drift ratios="
                    f"{fmt(lr_sweep.loc[0.1, 'geomean_final_tail_eval_loss_ratio_muon_over_adam'])}/"
                    f"{fmt(lr_sweep.loc[0.1, 'geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])}."
                ),
                interpretation="Muon and long-tail performance should be stated as a small sanity check plus future work unless real long-tail benchmark experiments are added.",
                next_test="Run practical Muon training ablations on real long-tail benchmarks if Muon-specific performance claims become central.",
            ),
        ]
    )

    paper_shape = pd.DataFrame(
        [
            {
                "role": "Core positive result",
                "content": "Matched-head-gain spectral/polar updates reduce tail logit drift in synthetic and small long-tailed diagnostics, with fixed-checkpoint and short-trajectory Muon-style compatibility checks.",
            },
            {
                "role": "Mechanistic bridge",
                "content": "Use the nrank(G_H) versus ssrank(B_T,A_T) condition plus layerwise scaling diagnostics.",
            },
            {
                "role": "Negative control",
                "content": "Show the reversed synthetic condition and the layerwise unit-direction caveat.",
            },
            {
                "role": "Scope boundary",
                "content": "Do not infer final tail accuracy, full Muon behavior, or broad optimizer superiority; the small practical training result is a sanity check with LR sensitivity, not a benchmark.",
            },
        ]
    )

    text = f"""# E11 Research Direction Map

This generated note is the compact paper-direction map for E11. It turns the current head-to-tail evidence into testable claims and explicit next experiments.

## Working Research Direction

The current project should be written as a **head-to-tail interference mechanism paper**. The central question is whether an idealized spectral/polar direction can achieve the same head progress while perturbing absent tail classes less than a Frobenius/GD-style direction. Older Muon/Adam condition-geometry results remain useful guardrails, but they are not the main paper thesis.

## Claim Map

{markdown_table(claims, ["question", "claim", "status", "evidence", "interpretation", "next_test"])}

## Paper-Level Hypothesis

The current best hypothesis is:

> In long-tailed small-batch training, spectral/polar updates can reduce head-to-tail function drift at matched head gain when the head-gradient rank is large relative to downstream tail sensitivity. Fixed-checkpoint and short-trajectory Muon-style compatibility checks support Muon-style momentum/NS directions as plausible local implementation paths, but broad Muon training behavior remains a separate claim.

## Current Paper Shape

{markdown_table(paper_shape, ["role", "content"])}

## Immediate Next Experiments

1. Run a real long-tailed benchmark only if the paper wants to discuss final tail performance.
2. Add real long-tail practical Muon benchmarks only if Muon-specific performance claims become central.
3. Add larger-architecture layerwise diagnostics only if architecture-level generality becomes central.

## Sources

- [research synthesis](e11_research_synthesis.md)
- [paper-readiness audit](e11_paper_readiness_audit.md)
- [evidence index](e11_evidence_index.md)
- [quantitative claim ledger](e11_quantitative_claim_ledger.md)
- [head-tail interference note](e11_head_tail_interference.md)
- [long-tail one-step note](e11_long_tail_one_step.md)
- [long-tail Muon-style compatibility note](e11_long_tail_muon_bridge.md)
- [long-tail practical-Muon trajectory compatibility note](e11_long_tail_practical_muon_bridge.md)
- [long-tail forgetting note](e11_long_tail_forgetting.md)
- [long-tail layerwise note](e11_long_tail_layerwise.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved research direction map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
