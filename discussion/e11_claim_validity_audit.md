# E11 Claim Validity Audit

This audit separates what the current head-to-tail experiments directly establish from what remains a vulnerability. It is generated from the current result CSVs so that the numerical claims are reproducible.

## Claim Status

| claim                                                                                           | status                                        | evidence                                                                                                                                    | main_loophole                                                                                                                                  |
|:------------------------------------------------------------------------------------------------|:----------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| The synthetic rank/sensitivity condition has the intended sign.                                 | supported                                     | Positive condition drift ratio=0.3403 CI=[0.3403,0.3403]; negative condition drift ratio=7.208 CI=[7.208,7.208].                            | The two settings are constructed examples; this validates sign logic, not out-of-sample prediction.                                            |
| Spectral/polar one-step updates reduce held-out tail logit drift at matched head gain.          | supported in small long-tailed digits         | Drift-squared ratio=0.5501 CI=[0.5101,0.5931]; spectral-lower paired fraction=1 over 20 seeds.                                              | The result is about logits/function drift, not tail accuracy or final performance.                                                             |
| Lower tail logit drift implies lower tail loss or better tail accuracy.                         | not supported                                 | One-step tail-loss increase difference spectral-minus-Frobenius=0.001399 CI=[0.0002379,0.002559]; tail-accuracy-drop difference=0 CI=[0,0]. | Performance claims need real long-tail training and class-wise outcome metrics.                                                                |
| The tail-drift reduction persists across a short head-only horizon.                             | supported for eight steps                     | Final drift ratio=0.6167 CI=[0.5744,0.6622]; drift-area ratio=0.7787 CI=[0.7549,0.8033].                                                    | Eight head-only steps are still a diagnostic, not a full optimizer benchmark.                                                                  |
| Spectral/polar directions are intrinsically less tail-sensitive layerwise.                      | not supported                                 | Unit JVP drift ratios are above one: layer 1=1.45, layer 2=1.476.                                                                           | The supported mechanism is matched-head-gain scaling, not lower unit-direction sensitivity.                                                    |
| Matched-head-gain scaling explains the observed lower layerwise drift.                          | supported in the current layerwise diagnostic | Scaled ratios: layer 1=0.4766, layer 2=0.596; observed ratios: layer 1=0.4767, layer 2=0.596.                                               | This should be rechecked in larger architectures before claiming generality.                                                                   |
| The clean polar direction has selected-state compatibility with Muon-style momentum directions. | supported as selected-state diagnostic        | polar(M_t) drift ratio=0.8199 CI=[0.6951,0.9672]; short-trajectory NS(M_t) ratio=0.8019 CI=[0.7644,0.8413].                                 | This is still a small-digits selected-state compatibility check; real long-tail practical Muon training and final performance remain unproven. |

## Evidence Tables

| table                                             | path                                                    | role                                                                                                                   |
|:--------------------------------------------------|:--------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| Synthetic boundary                                | results/e11_head_tail_interference/pair_summary.csv     | Positive and negative rank/sensitivity condition check.                                                                |
| Long-tail one-step                                | results/e11_long_tail_one_step/pair_summary.csv         | Matched-head-gain held-out tail logit drift and performance caveat.                                                    |
| Long-tail Muon-style compatibility                | results/e11_long_tail_muon_bridge/pair_summary.csv      | Fixed-checkpoint compatibility check for polar(G_t), polar(M_t), and Newton-Schulz directions under matched head gain. |
| Long-tail practical Muon trajectory compatibility | results/e11_long_tail_practical_muon_bridge/summary.csv | Short trajectory-level matched-head-gain compatibility check for practical momentum/NS directions.                     |
| Head-only forgetting                              | results/e11_long_tail_forgetting/summary.csv            | Eight-step drift persistence under matched head-only update schedules.                                                 |
| Layerwise diagnostic                              | results/e11_long_tail_layerwise/summary.csv             | Unit JVP, scaled JVP, and observed drift decomposition.                                                                |

## Defensible Formulation

The defensible formulation is: **idealized spectral/polar directions can reduce head-to-tail logit drift at matched head gain under a measurable rank/sensitivity condition.**

The current data do **not** justify saying that this already proves better tail classification, full practical Muon training behavior, or broad optimizer superiority.

## Strongest Remaining Loopholes

1. The real-data evidence is small long-tailed digits, not a modern long-tail benchmark.
2. The Muon-style compatibility evidence is local and small-scale; real long-tail practical Muon training remains unchecked.
3. The current performance evidence is weaker than the function-drift evidence.
4. The layerwise mechanism has only been checked in the current small MLP.
