# E11 Research Synthesis

This generated note is the current paper-facing synthesis of the E11 experiments. It is intentionally stricter than the exploratory condition-geometry artifacts: every claim below is tied to a quantitative result and an explicit caveat.

For the figure/CSV source behind each claim, see [E11 evidence index](e11_evidence_index.md). For claim wording constraints, see [E11 quantitative claim ledger](e11_quantitative_claim_ledger.md). For current paper risks, see [E11 reviewer risk audit](e11_reviewer_risk_audit.md).

## Current Thesis

**The current paper is a head-to-tail interference mechanism paper.** In long-tailed small-batch training, head-only updates can perturb logits on held-out tail examples while those examples are absent from the update. The supported claim is that an idealized spectral/polar direction can reduce this tail-example logit drift at matched head gain under a measurable rank/sensitivity condition. Fixed-checkpoint and short-trajectory Muon-style directions show selected-state compatibility with the polar mechanism, but this is still not a claim that full Muon improves final tail accuracy.

## Claim Cards

| claim                                                                                        | status                                  | evidence                                                                                                                                                                                       | interpretation                                                                                                                                                        | caveat                                                                                                          |
|:---------------------------------------------------------------------------------------------|:----------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
| The paper has a precise local mechanism target.                                              | supported as a focused paper scope      | All main diagnostics compare spectral/polar and Frobenius/GD-style directions at matched head gain.                                                                                            | The paper is about head-to-tail function interference, not an optimizer leaderboard.                                                                                  | Full Muon behavior still requires a bridge ablation with momentum and Newton-Schulz approximation.              |
| The rank/sensitivity condition has the correct synthetic boundary behavior.                  | supported                               | Positive case squared drift ratio=0.3403 [0.3403, 0.3403]; negative case squared drift ratio=7.208 [7.208, 7.208].                                                                             | The constructed examples verify that the theorem condition is not merely decorative.                                                                                  | This does not by itself prove predictive power on natural tasks.                                                |
| Long-tailed one-step diagnostics show lower tail-example logit drift.                        | supported                               | Drift-squared ratio=0.5501 [0.5101, 0.5931]; spectral-lower paired fraction=1.                                                                                                                 | At the same first-order head progress, spectral/polar updates perturb logits on held-out tail examples less.                                                          | Tail-loss increase diff is 0.001399 [0.0002379, 0.002559], so this is not a tail-performance improvement claim. |
| The drift advantage persists in a short head-only forgetting horizon.                        | supported                               | Final squared drift ratio=0.6167 [0.5744, 0.6622]; area ratio=0.7787 [0.7549, 0.8033].                                                                                                         | The effect survives repeated head-only updates in the small diagnostic.                                                                                               | The horizon is eight steps and still not a full long-tailed training benchmark.                                 |
| Muon-style directions show selected-state compatibility across a short practical trajectory. | supported as a compatibility diagnostic | polar(M_t) squared drift ratio=0.8199 [0.6951, 0.9672]; short-trajectory NS(M_t) squared drift ratio=0.8019 [0.7583, 0.848]; Fro/GD-state NS(M_t) squared drift ratio=0.7973 [0.7546, 0.8425]. | Sampled Muon-style momentum/NS states have squared drift ratios compatible with the clean polar(G_t) mechanism, including on a Fro/GD-generated state-source control. | This is still a local matched-head-gain diagnostic on small digits, not a full long-tail optimizer benchmark.   |
| Layerwise evidence points to matched-head-gain scaling, not intrinsically safer directions.  | supported as a mechanism refinement     | Layer 1 unit/scaled/observed squared drift ratios=1.45/0.4766/0.4767; Layer 2=1.476/0.596/0.596.                                                                                               | Spectral/polar directions can be more unit-tail-sensitive but still cause lower observed drift because they need less scaling for the same head gain.                 | This caveat should appear anywhere the paper uses intuitive language about being less tail-disruptive.          |

## Report Implications

| section               | use                                                                                                                  |
|:----------------------|:---------------------------------------------------------------------------------------------------------------------|
| Main positive finding | Present matched-head-gain spectral/polar updates as reducing head-to-tail function drift.                            |
| Mechanism             | Use the nrank(G_H) versus ssrank(B_T,A_T) condition and the layerwise scaling diagnostic.                            |
| Caveats               | Separate drift from tail loss, accuracy, final performance, and full Muon optimizer behavior.                        |
| Appendix / guardrails | Use older condition-geometry and Muon/Adam artifacts only as background unless the main paper explicitly needs them. |

## Recommended Main Story

1. Define the head-to-tail interference problem and the matched-head-gain protocol.
2. Prove the local condition using \(\operatorname{nrank}(G_H)\) and \(\operatorname{ssrank}(B_T,A_T)\).
3. Show the synthetic positive and negative boundary cases.
4. Show the long-tailed one-step, Muon-style compatibility fixed/trajectory, and eight-step forgetting drift diagnostics.
5. Use the layerwise diagnostic to state the correct mechanism: scaled head-gain efficiency rather than intrinsically lower tail sensitivity.
6. State the scope boundary: drift is not performance, and selected-state Muon compatibility is not full Muon training.

## Remaining Open Gaps

1. A real long-tail benchmark is still needed before making broad performance claims.
2. A real long-tail practical Muon benchmark is still needed before claiming the mechanism explains full Muon training outcomes.
3. Larger-architecture layerwise diagnostics are still needed before claiming architecture-level generality.
