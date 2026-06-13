# E11 Consecutive Head-Only Tail Forgetting Probe

This probe starts from the same imbalanced sklearn digits MLP checkpoint as the
one-step diagnostic, then runs several consecutive head-only updates while
tail classes 5-9 are held out. Frobenius/GD and spectral/polar directions use
the same precomputed head mini-batches and the same target first-order head
gain schedule for each seed.

- Seeds: 20
- Head-only steps: 8
- Head classes: (0, 1, 2, 3, 4)
- Tail classes: (5, 6, 7, 8, 9)
- Head train examples per class: 100
- Tail train examples per class: 40
- Tail evaluation examples per class: 40
- Warmup steps: 80
- Target head first-order gain: 0.02 * reference head-batch loss

| quantity | value |
|---|---:|
| final tail drift-sq ratio, spectral/Fro | 0.6167 [0.5744, 0.6622] |
| spectral lower final tail drift fraction | 0.95 |
| tail-drift area ratio, spectral/Fro | 0.7787 [0.7549, 0.8033] |
| spectral lower tail-drift area fraction | 1 |
| final tail loss increase diff, spectral - Fro | -0.003538 [-0.007721, 0.0006445] |
| final tail margin drop diff, spectral - Fro | -0.003658 [-0.009636, 0.00232] |
| spectral proxy-drift Spearman | 0.9739 [0.9643, 0.9809] |
| Fro/GD proxy-drift Spearman | 0.9503 [0.9324, 0.9635] |

Figure: [figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png](../figures/e11_long_tail_forgetting/long_tail_head_only_forgetting.png)

Main readout: this probe tests whether the one-step tail-drift pattern persists
over a short head-only horizon. It should be read as a controlled intervention
on update geometry, not as a full long-tailed training benchmark.

Caveats:
- The schedule matches first-order head gain, not the realized nonlinear head-loss decrease.
- The cumulative condition proxy is based on MLP layer rank diagnostics, not an exact neural `J_T` coefficient.
- This is still a small sklearn digits probe; the real-data hierarchy should eventually move to CIFAR-100-LT or a similar benchmark.

Artifacts:
- [step_metrics.csv](../results/e11_long_tail_forgetting/step_metrics.csv)
- [summary.csv](../results/e11_long_tail_forgetting/summary.csv)
- [config.json](../results/e11_long_tail_forgetting/config.json)
