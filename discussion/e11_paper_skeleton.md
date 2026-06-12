# E11 Paper Skeleton

This generated skeleton is not a draft paper. It is a maintainable bridge from the current evidence base to a publishable manuscript.

## Working Title

Update-Spectrum Shaping in Muon: Local Geometry, Norm Constraints, and Boundaries of Optimization Progress

## Abstract Sketch

Muon-style optimizers apply a polar-like update that changes the singular-value geometry of parameter updates. In controlled experiments across matrix factorization with input, matrix sensing, and a small neural benchmark, Muon consistently produces flatter and higher-rank update spectra than Adam at matched update size. We show that one-step progress is well captured by gradient-update alignment, and use this diagnostic to identify when Muon's flat/polar spectral bias helps or hurts. The key boundary is not rank itself: flat/polar spreading is favorable under operator-norm-like constraints but unfavorable under Frobenius constraints, and its local advantage flips by task family and layer geometry. These results support viewing Muon as an update-spectrum shaping method rather than as a universally better or more stable optimizer.

## Core Claims

| claim                                                                                  | status                     | evidence                                                                                                                                                                                        | figure_or_table                                        |
|:---------------------------------------------------------------------------------------|:---------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------|
| Muon shapes update spectra.                                                            | main claim                 | `nrUpdate` ratio 2.033 [1.921, 2.151]; `stUpdate` ratio 5.215 [5.008, 5.43].                                                                                                                    | update-spectrum robustness; cross-task signature table |
| One-step progress can be analyzed through gradient-update alignment.                   | diagnostic claim           | Spearman(delta_loss, <G,D>)=0.9206 [0.9145, 0.9263].                                                                                                                                            | first-order calibration                                |
| The same spectral bias has a norm-geometry boundary.                                   | mechanism claim            | flat/GD ratio 0.6071 [0.5846, 0.6304] under Frobenius budget; 1.689 [1.614, 1.768] under operator-norm budget.                                                                                  | spectral-allocation probe                              |
| Muon advantage is conditional, not universal.                                          | boundary claim             | Matrix Sensing ratio 1.209; MF ratio 0.3606; MLP hidden=64 ratio 0.6047.                                                                                                                        | mechanism boundary map                                 |
| The current boundary is not yet a predictive law.                                      | negative/paper-gap claim   | Best leave-setting-out boundary predictor balanced accuracy is 0.6039. Chance-filled best is 0.5866 CI=[0.5, 0.7446].                                                                           | boundary predictor baseline                            |
| Neural MNIST probes preserve update-spectrum shaping but not universal advantage.      | neural supporting evidence | MNIST nrUpdate ratio 1.957; hidden=128 first-order ratio 0.7546; Deep MNIST nrUpdate ratio 2.262; Deep first-order ratio 0.5978; ConvNet nrUpdate ratio 3.718; ConvNet first-order ratio 0.677. | MNIST MLP, Deep MNIST MLP, and MNIST ConvNet probes    |
| Polar direction alone is not enough for progress under Frobenius-matched trajectories. | mechanism negative control | Stateless trajectory PolarMuon/GD total decrease ratio 0.4656 [0.3686, 0.5882].                                                                                                                 | stateless optimizer trajectory ablation                |

## Section Plan

| section               | purpose                                                                                        | must_include                                                                                                                                               |
|:----------------------|:-----------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction          | Motivate optimizer geometry beyond final-loss leaderboards.                                    | State that the paper studies local update geometry, not a universal Muon superiority claim.                                                                |
| Setup and Diagnostics | Define update spectrum, stable/effective rank, one-step decrease, and matched-update controls. | `G_i`, signed update `Delta W_i`, positive descent update `D_i`, `<G,D>`, `nrUpdate`, `stUpdate`, and exact control protocol.                              |
| Cross-Task Signature  | Show the robust optimizer-intrinsic effect.                                                    | Only update-spectrum metrics pass the three-family screen.                                                                                                 |
| Mechanism Probe       | Explain why flat/polar spreading can help or hurt.                                             | Frobenius vs operator-norm spectral-allocation sign flip and the local theory note.                                                                        |
| Boundary Experiments  | Show task/layer conditionality under controls.                                                 | MF, Matrix Sensing, sklearn digits MLP, shallow/deep MNIST MLP, MNIST patch/ConvNet probes, target update, per-layer controls, and optimizer ablation map. |
| Negative Controls     | Prevent overclaiming from local geometry.                                                      | Switch/continuation, stability audit, and final-performance caveats.                                                                                       |
| Discussion            | Frame what would be needed for a predictive theory.                                            | Weak leave-setting-out boundary predictor and modern/long-horizon neural benchmark as future work.                                                         |

## Main Figure/Table Plan

| slot     | artifact                                                             | message                                                                                                                                                         |
|:---------|:---------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Figure 1 | figures/e11_equal_update/update_spectrum_robustness.png              | Muon has a robust update-spectrum signature.                                                                                                                    |
| Figure 2 | figures/e11_equal_update/first_order_calibration.png                 | Observed one-step decrease is locally calibrated by `<G,D>`.                                                                                                    |
| Figure 3 | figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png | Flat/polar allocation wins under operator budget and loses under Frobenius budget; theory note and theorem bridge state the matching local constrained problem. |
| Table 1  | results/e11_mechanism_boundary/mechanism_boundary_map.csv            | The local advantage flips by problem and control condition.                                                                                                     |
| Appendix | discussion/e11_main_paper_package.md                                 | Selects the smallest main figure/table package and assigns the remaining probes to appendix roles.                                                              |
| Appendix | discussion/e11_quantitative_claim_ledger.md                          | Records which claims are paper-ready, which quantitative anchors they require, and which wording to avoid.                                                      |
| Appendix | discussion/e11_paper_numbers.tex                                     | Generated LaTeX macros for the paper-facing ratios, confidence intervals, and calibration numbers.                                                              |
| Appendix | discussion/e11_reproduction_checklist.md                             | Separates minimal main-paper reproduction from appendix/guardrail evidence and validation commands.                                                             |
| Appendix | discussion/e11_main_figure_captions.md                               | Drafts paper-safe captions for the proposed main figures and table.                                                                                             |
| Appendix | discussion/e11_notation_glossary.md                                  | Defines the shared notation for gradients, updates, ranks, activation products, and diagnostics.                                                                |
| Appendix | discussion/e11_mechanism_theorem_bridge.md                           | Maps theorem assumptions to evidence and states safe versus unsafe causal language.                                                                             |
| Appendix | discussion/e11_optimizer_invariance_audit.md                         | Muon is not generally more stable; stability is task/scale dependent.                                                                                           |
| Appendix | discussion/e11_optimizer_ablation_map.md                             | Ablation ladder separates update size, layer allocation, stateless direction/trajectory, spectrum allocation, vector geometry, and continuation effects.        |
| Appendix | discussion/e11_boundary_predictor.md                                 | The current descriptive boundary is not yet a strong leave-setting-out predictive law.                                                                          |
| Appendix | discussion/e11_boundary_predictor_audit.md                           | Explains why the current predictor is not paper-ready and specifies the next held-out test standard.                                                            |
| Appendix | discussion/e11_mnist_mlp_probe.md                                    | MNIST MLP preserves update-spectrum shaping while showing conditional first-order advantage.                                                                    |
| Appendix | discussion/e11_deep_mnist_mlp_probe.md                               | Deep MNIST MLP preserves update-spectrum shaping while becoming more Adam-favorable for first-order progress.                                                   |
| Appendix | discussion/e11_mnist_conv_probe.md                                   | Small true ConvNet preserves update-spectrum shaping while remaining Adam-favorable for first-order progress.                                                   |
| Appendix | discussion/e11_stateless_optimizer_trajectory.md                     | Polar stateless trajectories retain high-rank updates but do not improve total decrease under Frobenius-matched steps.                                          |

## Current Manuscript Gap

The current evidence is strong enough for a focused local-geometry paper, but not yet for a broad optimizer-performance paper. Before writing a full manuscript, the next two paper-critical items are:

1. A modern or longer-horizon neural benchmark with explicit treatment of non-matrix parameters; current MNIST MLP, patch, and ConvNet probes remain short-horizon sanity/negative controls.
2. A stronger predictive boundary model for when Muon's flat/polar update improves `<G,D>`; the current leave-setting-out baseline is weak.

## Linked Evidence

- [paper-readiness audit](e11_paper_readiness_audit.md)
- [reviewer risk audit](e11_reviewer_risk_audit.md)
- [research synthesis](e11_research_synthesis.md)
- [evidence index](e11_evidence_index.md)
- [theory note](e11_theory_note.md)
- [mechanism theorem bridge](e11_mechanism_theorem_bridge.md)
- [optimizer ablation map](e11_optimizer_ablation_map.md)
- [mechanism boundary map](e11_mechanism_boundary.md)
- [boundary predictor baseline](e11_boundary_predictor.md)
- [boundary predictor audit](e11_boundary_predictor_audit.md)
- [MNIST MLP probe](e11_mnist_mlp_probe.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
- [MNIST ConvNet probe](e11_mnist_conv_probe.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
