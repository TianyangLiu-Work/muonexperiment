# E11 Main Figure Captions

This generated note drafts paper-safe captions for the proposed main figures and table. Each caption includes the quantitative anchor and the intended interpretation boundary.

## Captions

| slot     | artifact                                                             | caption                                                                                                                                                                                                                                      | interpretation                                                                                                                                                         |
|:---------|:---------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Figure 1 | figures/e11_equal_update/update_spectrum_robustness.png              | Muon produces a reproducible update-spectrum signature under matched global update size. Across the current E11 families, Muon/Adam geomean ratios are 2.033 [1.921, 2.151] for `nrUpdate` and 5.215 [5.008, 5.43] for `stUpdate`.           | This supports the optimizer-intrinsic spectrum-shaping claim, but it should not be read as evidence that higher update rank directly improves optimization.            |
| Figure 2 | figures/e11_equal_update/first_order_calibration.png                 | Observed one-step loss decrease is well calibrated by the local gradient-update alignment `<G,D>`. The Spearman correlation is 0.9206 [0.9145, 0.9263], with 0.9584 of positive decreases within a factor of two.                            | This justifies using one-step alignment as the local bridge from update geometry to loss decrease, but it is not a convergence or final-performance claim.             |
| Figure 3 | figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png | The value of flat/polar spectral allocation depends on the norm geometry of the local comparison. The flat/GD first-order ratio is 0.6071 [0.5846, 0.6304] under a Frobenius budget, but 1.689 [1.614, 1.768] under an operator-norm budget. | This is the main mechanism boundary: Muon-like spectral spreading is locally useful only when the task/layer/norm geometry rewards spreading.                          |
| Table 1  | results/e11_mechanism_boundary/mechanism_boundary_map.csv            | Mechanism-boundary map for the current controlled experiments. The map contains 4 Muon/flat favorable rows, 1 own-update positive-control row, 8 unfavorable rows, and 1 mixed or uncertain row.                                             | This table frames the paper as a boundary/mechanism study rather than an optimizer leaderboard; the current boundary is descriptive, not yet predictive out of sample. |

## Caption Discipline

1. Every main caption should include a quantitative anchor.
2. Captions should distinguish the measured geometry effect from optimization or final-performance claims.
3. The wording should follow [e11_quantitative_claim_ledger.md](e11_quantitative_claim_ledger.md) and [e11_notation_glossary.md](e11_notation_glossary.md).
