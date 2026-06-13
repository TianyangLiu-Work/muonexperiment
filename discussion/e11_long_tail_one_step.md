# E11 Long-Tailed One-Step Diagnostic

This probe trains a small MLP checkpoint on an imbalanced sklearn digits split,
then forms a head-only gradient from classes 0-4 and evaluates the same
one-step intervention on held-out tail classes 5-9. Frobenius/GD and
spectral/polar directions are both rescaled to the same head first-order
gain before measuring tail function movement.

- Seeds: 20
- Head classes: (0, 1, 2, 3, 4)
- Tail classes: (5, 6, 7, 8, 9)
- Head train examples per class: 100
- Tail train examples per class: 40
- Tail evaluation examples per class: 40
- Warmup steps: 80
- Hidden dimension: 32
- Target head first-order gain: 0.02 * head loss

Ratio columns are spectral divided by Frobenius/GD. Values below 1 mean the
spectral direction disturbed held-out tail logits less after matching head gain.

| quantity | value |
|---|---:|
| tail drift-sq ratio, spectral/Fro | 0.5501 [0.5101, 0.5931] |
| spectral less tail drift fraction | 1 |
| tail loss increase diff, spectral - Fro | 0.001399 [0.0002379, 0.002559] |
| tail margin drop diff, spectral - Fro | 0.001892 [9.142e-05, 0.003693] |
| tail accuracy drop diff, spectral - Fro | 0 [0, 0] |
| actual head loss decrease diff, spectral - Fro | 2.555e-05 [1.175e-05, 3.936e-05] |
| mean tail diagnostic condition score | 2.373 |

Figure: [figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png](../figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png)

Main readout: this is the first non-synthetic one-step check of the
head-to-tail framing. It does not prove long-tailed generalization, but it
tests the required matched-head-gain protocol on a real data distribution.
The key claim should be stated from the table above, not from optimizer labels
alone.

Caveats:
- This is sklearn digits, not CIFAR-100-LT/ImageNet-LT/iNaturalist.
- The spectral direction is an exact polar intervention, not a full Muon optimizer state.
- Layerwise MLP nonlinearities make `stA_tail` a diagnostic proxy; direct tail drift is the primary measurement.

Artifacts:
- [step_metrics.csv](../results/e11_long_tail_one_step/step_metrics.csv)
- [pair_summary.csv](../results/e11_long_tail_one_step/pair_summary.csv)
- [layer_metrics.csv](../results/e11_long_tail_one_step/layer_metrics.csv)
- [config.json](../results/e11_long_tail_one_step/config.json)
