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
spectral direction disturbed logits on held-out tail examples less after matching head gain.

| quantity | value |
|---|---:|
| squared tail-example logit drift ratio, spectral/Fro | 0.5501 [0.5101, 0.5931] |
| centered-logit squared drift ratio, spectral/Fro | 0.5493 [0.5093, 0.5923] |
| true-class logit delta squared ratio, spectral/Fro | 1.68 [1.16, 2.431] |
| top-competitor logit delta squared ratio, spectral/Fro | 0.6358 [0.5705, 0.7086] |
| margin-delta squared ratio, spectral/Fro | 0.702 [0.6131, 0.8037] |
| spectral lower squared tail-example logit drift fraction | 1 |
| actual head-gain relative error, Fro/GD | 0.02533 [0.02052, 0.03013] |
| actual head-gain relative error, spectral | 0.01662 [0.0148, 0.01845] |
| tail loss increase, Fro/GD | -0.001154 [-0.004135, 0.001827] |
| tail loss increase, spectral | 0.0002447 [-0.001859, 0.002348] |
| tail loss increase diff, spectral - Fro | 0.001399 [0.0002379, 0.002559] |
| tail margin drop, Fro/GD | 0.001372 [-0.002944, 0.005688] |
| tail margin drop, spectral | 0.003264 [0.0004973, 0.006032] |
| tail margin drop diff, spectral - Fro | 0.001892 [9.142e-05, 0.003693] |
| tail accuracy drop, Fro/GD | -0.00025 [-0.00074, 0.00024] |
| tail accuracy drop, spectral | -0.00025 [-0.00074, 0.00024] |
| tail accuracy drop diff, spectral - Fro | 0 [0, 0] |
| actual head loss decrease diff, spectral - Fro | 2.555e-05 [1.175e-05, 3.936e-05] |
| mean tail diagnostic condition score | 2.373 |
| tail CE before | 11.74 [11.5, 11.99] |
| tail CE after Fro/GD | 11.74 [11.5, 11.99] |
| tail CE after spectral | 11.74 [11.5, 11.99] |
| tail margin before | -10.79 [-11.06, -10.52] |
| tail margin after Fro/GD | -10.79 [-11.06, -10.52] |
| tail margin after spectral | -10.79 [-11.06, -10.52] |
| tail accuracy before | 0.1908 [0.1873, 0.1942] |
| positive-margin tail fraction before | 0.1908 [0.1873, 0.1942] |
| certified preserved fraction, Fro/GD | 0.9985 [0.9956, 1] |
| certified preserved fraction, spectral | 0.9985 [0.9956, 1] |
| all-tail prediction changed fraction, Fro/GD | 0.00625 [0.004386, 0.008114] |
| all-tail prediction changed fraction, spectral | 0.00575 [0.003705, 0.007795] |
| positive-margin prediction changed fraction, Fro/GD | 0 [0, 0] |
| positive-margin prediction changed fraction, spectral | 0 [0, 0] |

Figure: [figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png](../figures/e11_long_tail_one_step/long_tail_one_step_tail_response.png)

Main readout: this is the first non-synthetic one-step check of the
head-to-tail framing. It does not prove long-tailed generalization, but it
tests the required matched-head-gain protocol on a real data distribution.
The key claim should be stated from the table above, not from optimizer labels
alone.

Caveats:
- This is sklearn digits, not CIFAR-100-LT/ImageNet-LT/iNaturalist.
- The spectral direction is an exact polar intervention, not a full Muon optimizer state.
- MLP nonlinearities make the closed-form sandwich condition layer-dependent; direct measured tail drift is the primary one-step quantity.

Artifacts:
- [step_metrics.csv](../results/e11_long_tail_one_step/step_metrics.csv)
- [pair_summary.csv](../results/e11_long_tail_one_step/pair_summary.csv)
- [layer_metrics.csv](../results/e11_long_tail_one_step/layer_metrics.csv)
- [config.json](../results/e11_long_tail_one_step/config.json)
