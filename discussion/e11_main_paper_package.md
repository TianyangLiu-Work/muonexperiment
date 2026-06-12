# E11 Main Paper Package

This generated note selects the smallest evidence package for a focused manuscript. The point is to keep the main paper readable: three figures plus one table should carry the main claim, with the remaining probes used as safeguards and appendix evidence.

## Main Figure/Table Package

| slot     | artifact                                                             | claim                                                                 | quantitative_anchor                                                                                    | reader_takeaway                                                                                                  |
|:---------|:---------------------------------------------------------------------|:----------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| Figure 1 | figures/e11_equal_update/update_spectrum_robustness.png              | Muon has a robust update-spectrum signature.                          | nrUpdate Muon/Adam=2.033 [1.921, 2.151]; stUpdate=5.215 [5.008, 5.43].                                 | The reliable optimizer-intrinsic effect is spectrum shaping, not final-loss superiority.                         |
| Figure 2 | figures/e11_equal_update/first_order_calibration.png                 | One-step progress is locally calibrated by gradient-update alignment. | Spearman(delta_loss, <G,D>)=0.9206 [0.9145, 0.9263].                                                   | It is meaningful to analyze the positive descent proxy `<G,D>` as a local bridge from geometry to loss decrease. |
| Figure 3 | figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png | Flat/polar spectral allocation has a norm-geometry boundary.          | Frobenius flat/GD=0.6071 [0.5846, 0.6304]; operator-norm flat/GD=1.689 [1.614, 1.768].                 | Muon-like spectral spreading is locally useful only under the right norm geometry.                               |
| Table 1  | results/e11_mechanism_boundary/mechanism_boundary_map.csv            | The advantage flips across problem, layer, and control condition.     | Boundary rows: 4 Muon/flat favorable, 1 own-update positive-control, 8 unfavorable, 1 mixed/uncertain. | The paper is a boundary/mechanism paper, not a universal optimizer win paper.                                    |

## Appendix Allocation

| artifact                                         | role                                                                                                                                      |
|:-------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| discussion/e11_mechanism_theorem_bridge.md       | States theorem assumptions, evidence mapping, and safe causal language.                                                                   |
| discussion/e11_main_figure_captions.md           | Paper-safe main figure/table captions with quantitative anchors and interpretation boundaries.                                            |
| discussion/e11_notation_glossary.md              | Shared notation source for gradients, updates, ranks, activation products, and recorded diagnostics.                                      |
| discussion/e11_deep_mnist_mlp_probe.md           | Neural negative control: Deep MNIST nrUpdate=2.262 [2.175, 2.353] but first-order=0.5978 [0.5633, 0.6345].                                |
| discussion/e11_mnist_patch_probe.md              | Patch/shared-weight neural sanity check: nrUpdate=4.235 [3.834, 4.677] but first-order=0.5222 [0.4943, 0.5516].                           |
| discussion/e11_mnist_conv_probe.md               | True Conv2d neural sanity check: nrUpdate=3.718 [3.451, 4.006] but first-order=0.677 [0.6321, 0.7251].                                    |
| discussion/e11_stateless_optimizer_trajectory.md | Trajectory mechanism control: PolarMuon/GD total decrease=0.4656 [0.3686, 0.5882].                                                        |
| discussion/e11_boundary_predictor_audit.md       | Predictive-boundary gap: best leave-setting-out balanced accuracy=0.6039 from state_plus_update_spectrum; chance-filled CI=[0.5, 0.7446]. |
| discussion/e11_reviewer_risk_audit.md            | Reviewer-risk map and claim discipline.                                                                                                   |
| discussion/e11_quantitative_claim_ledger.md      | Paper-writing claim ledger with allowed wording, forbidden wording, quantitative anchors, and evidence links.                             |
| discussion/e11_paper_numbers.tex                 | LaTeX macros for paper-facing quantitative anchors, generated directly from result CSVs.                                                  |
| discussion/e11_reproduction_checklist.md         | Minimal main-paper and appendix reproduction map with validation commands.                                                                |

## Claims To Exclude From Main Text

| artifact                                      | reason                                                                       |
|:----------------------------------------------|:-----------------------------------------------------------------------------|
| optimizer switch and LR-sweep figures         | Use as negative controls in appendix; too detailed for the main argument.    |
| all individual trajectory/3D figures          | Useful exploratory evidence, but they dilute the main mechanism story.       |
| boundary predictor detailed per-setting table | Keep the main text to the failure summary; detailed rows belong in appendix. |

## Main-Text Claim Order

1. Muon changes update spectra robustly under matched update size.
2. One-step loss decrease is well calibrated by `<G,D>`.
3. The local value of flat/polar spectra depends on the norm geometry.
4. Therefore Muon is a geometry-shaping optimizer with boundary-dependent progress, not a universally better optimizer.

## Drafting Rule

If a sentence cannot be supported by Figure 1, Figure 2, Figure 3, or Table 1, it should probably be in the appendix or discussion rather than in the main result section.
