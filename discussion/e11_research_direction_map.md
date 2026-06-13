# E11 Research Direction Map

This generated note is the compact paper-direction map for E11. It turns the current head-to-tail evidence into testable claims and explicit next experiments.

## Working Research Direction

The current project should be written as a **head-to-tail interference mechanism paper**. The central question is whether an idealized spectral/polar direction can achieve the same head progress while perturbing absent tail classes less than a Frobenius/GD-style direction. Older Muon/Adam condition-geometry results remain useful guardrails, but they are not the main paper thesis.

## Claim Map

| question                                                         | claim                                                                                                  | status                                | evidence                                                                                                                                                                                                  | interpretation                                                                                                                                  | next_test                                                                                                                        |
|:-----------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:--------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| What is the current paper's mechanism target?                    | Head-only updates can interfere with tail logits while tail samples are absent.                        | paper framing                         | The current main experiments all use matched-head-gain spectral/polar versus Frobenius/GD-style updates.                                                                                                  | This makes the project a local function-drift mechanism paper, not an optimizer leaderboard.                                                    | Keep every main result normalized by matched head gain or explicitly explain why it is not.                                      |
| When should spectral/polar updates reduce head-to-tail drift?    | The rank/sensitivity condition predicts the synthetic positive and negative cases.                     | supported mechanism boundary          | Positive: nrank(G_H)=8.607, ssrank(B_T,A_T)=1, drift ratio=0.3403 [0.3403, 0.3403]. Negative: nrank(G_H)=1.387, ssrank(B_T,A_T)=10, drift ratio=7.208 [7.208, 7.208].                                     | The condition has falsifiable sign content in controlled examples.                                                                              | Add natural-task measurements of the same condition before presenting it as predictive beyond the constructed probe.             |
| Does the real-data one-step diagnostic support lower tail drift? | Yes, for tail logits at matched first-order head gain.                                                 | supported in small long-tailed digits | Drift-squared ratio=0.5501 [0.5101, 0.5931]; spectral-lower paired fraction=1 over 20 seeds.                                                                                                              | Spectral/polar updates perturb held-out tail logits less for the same head progress in this diagnostic.                                         | Replicate on a real long-tailed benchmark before claiming final tail-performance relevance.                                      |
| Does the lower drift persist beyond one update?                  | The 8-step head-only forgetting probe keeps lower spectral tail drift.                                 | supported short-horizon diagnostic    | Final drift ratio=0.6167 [0.5744, 0.6622]; drift-area ratio=0.7787 [0.7549, 0.8033].                                                                                                                      | The one-step drift effect is not isolated to a single update in the current small diagnostic.                                                   | Extend only after deciding whether the paper will include a full training benchmark.                                             |
| Does the clean polar direction connect to Muon-style state?      | There is selected-state compatibility through momentum polar and a short practical NS-Muon trajectory. | supported compatibility check         | polar(M_t) drift ratio=0.8199 [0.6951, 0.9672]; short-trajectory NS(M_t) drift ratio=0.8019 [0.7644, 0.8413].                                                                                             | Sampled Muon-style momentum/NS states have drift ratios compatible with the local polar mechanism.                                              | Test the same compatibility pattern on real long-tail benchmarks and larger models before claiming broad Muon training behavior. |
| What does the layerwise diagnostic say the mechanism is?         | The supported mechanism is scaled head-gain efficiency, not lower unit-direction tail sensitivity.     | supported mechanism refinement        | Layer 1 unit/scaled/observed ratios=1.45/0.4766/0.4767; Layer 2=1.476/0.596/0.596.                                                                                                                        | The spectral unit direction can be more tail-sensitive, but it needs less scaling to achieve the same head gain.                                | Add a larger-architecture layerwise diagnostic if this mechanism is presented as architecture-level.                             |
| What should remain outside the main claim?                       | Broad Muon training behavior, tail accuracy, and real benchmark performance remain open.               | scope boundary                        | One-step tail loss/accuracy do not improve, while the small practical run has tail eval loss ratio=0.8549 but tail accuracy diff=0. The LR sweep shows lr=0.1 worsens tail loss/drift ratios=2.049/1.744. | Muon and long-tail performance should be stated as a small sanity check plus future work unless real long-tail benchmark experiments are added. | Run practical Muon training ablations on real long-tail benchmarks if Muon-specific performance claims become central.           |

## Paper-Level Hypothesis

The current best hypothesis is:

> In long-tailed small-batch training, spectral/polar updates can reduce head-to-tail function drift at matched head gain when the head-gradient rank is large relative to downstream tail sensitivity. Fixed-checkpoint and short-trajectory Muon-style compatibility checks support Muon-style momentum/NS directions as plausible local implementation paths, but broad Muon training behavior remains a separate claim.

## Current Paper Shape

| role                 | content                                                                                                                                                                                      |
|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Core positive result | Matched-head-gain spectral/polar updates reduce tail logit drift in synthetic and small long-tailed diagnostics, with fixed-checkpoint and short-trajectory Muon-style compatibility checks. |
| Mechanistic bridge   | Use the nrank(G_H) versus ssrank(B_T,A_T) condition plus layerwise scaling diagnostics.                                                                                                      |
| Negative control     | Show the reversed synthetic condition and the layerwise unit-direction caveat.                                                                                                               |
| Scope boundary       | Do not infer final tail accuracy, full Muon behavior, or broad optimizer superiority; the small practical training result is a sanity check with LR sensitivity, not a benchmark.            |

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
