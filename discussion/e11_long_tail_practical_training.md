# E11 Long-Tailed Practical Training Diagnostic

This diagnostic compares practical Adam and a finite-Newton-Schulz Muon-style
optimizer on the same imbalanced sklearn digits task, from identical
initialization and with identical mini-batch/noise schedules for each seed.
It is a training-behavior sanity check, not a tuned optimizer leaderboard.

- Seeds: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
- Train steps: 80
- Batch size: 64
- Head classes: (0, 1, 2, 3, 4), 100 train examples/class
- Tail classes: (5, 6, 7, 8, 9), 10 train examples/class
- Tail eval examples/class: 40
- Adam lr: 0.01
- NS-Muon-style lr: 0.03
- Momentum beta: 0.9
- Newton-Schulz steps: 5

| quantity | Muon vs Adam |
|---|---:|
| final train loss ratio | 0.6468 [0.604, 0.6926] |
| final head loss ratio | 0.6677 [0.6271, 0.7108] |
| final tail eval loss ratio | 0.8549 [0.8319, 0.8786] |
| final tail eval accuracy diff | 0 [0, 0] |
| final tail eval margin diff | 1.955 [1.628, 2.281] |
| final tail output-drift RMS ratio | 0.7501 [0.7244, 0.7767] |

Figure: [figures/e11_long_tail_practical_training/long_tail_practical_training.png](../figures/e11_long_tail_practical_training/long_tail_practical_training.png)

Interpretation:

This experiment checks whether the local matched-head-gain drift pattern is
consistent with an actual imbalanced mini-batch training loop. In the
selected lightweight setting, the NS-Muon-style run has lower train/head
loss, lower tail eval loss, and lower tail drift than Adam, while tail
accuracy remains unchanged. This supports a narrow practical sanity check
and reinforces the paper's distinction between continuous logits on tail examples,
tail loss/margin, and discrete classification accuracy.

Caveats:

- The Muon-style optimizer is a small finite-Newton-Schulz implementation for two matrix layers, not a full production Muon stack.
- Hyperparameters are fixed and intentionally lightweight; this is not an optimizer leaderboard.
- The task is sklearn digits, so CIFAR-100-LT or another real long-tail benchmark is still needed for a strong empirical paper.

Artifacts:

- [step_metrics.csv](../results/e11_long_tail_practical_training/step_metrics.csv)
- [summary.csv](../results/e11_long_tail_practical_training/summary.csv)
- [config.json](../results/e11_long_tail_practical_training/config.json)
