# E11 Practical Training LR Sensitivity

This supplemental check tests whether the practical imbalanced-training result is
only a single learning-rate accident. It reruns the same paired Adam vs finite
Newton-Schulz Muon-style training diagnostic for several `muon_lr` values while
keeping Adam lr fixed at `0.01`.

| muon lr | train loss ratio | head loss ratio | tail eval loss ratio | tail drift RMS ratio | tail acc diff |
|---:|---:|---:|---:|---:|---:|
| 0.003 | 17.31 [15.96, 18.77] | 18.42 [17.06, 19.9] | 0.1733 [0.1698, 0.1769] | 0.01154 [0.01116, 0.01194] | 0 [0, 0] |
| 0.01 | 10.34 [9.529, 11.22] | 10.36 [9.59, 11.18] | 0.2404 [0.2357, 0.2452] | 0.1157 [0.1129, 0.1186] | 0 [0, 0] |
| 0.03 | 0.6468 [0.604, 0.6926] | 0.6677 [0.6271, 0.7108] | 0.8549 [0.8319, 0.8786] | 0.7501 [0.7244, 0.7767] | 0 [0, 0] |
| 0.1 | 0.1884 [0.1319, 0.269] | 0.1659 [0.1001, 0.2751] | 2.049 [1.923, 2.184] | 1.744 [1.635, 1.862] | 0 [0, 0] |

Figure: [figures/e11_long_tail_practical_training_lr_sweep/long_tail_practical_training_lr_sweep.png](../figures/e11_long_tail_practical_training_lr_sweep/long_tail_practical_training_lr_sweep.png)

Interpretation:

- Very small `muon_lr` under-trains relative to Adam: train/head loss ratios stay above 1.
- The selected `muon_lr=0.03` is the first tested scale where train/head loss, tail loss, and tail drift are all below Adam with paired confidence intervals below 1.
- Larger `muon_lr` can lower train/head loss further but increases tail loss and tail drift, so the practical benefit is not monotone in step size.

Current selected setting:

- train loss ratio: `0.6468 [0.604, 0.6926]`
- tail eval loss ratio: `0.8549 [0.8319, 0.8786]`
- tail drift RMS ratio: `0.7501 [0.7244, 0.7767]`

Best train-loss setting in this sweep:

- `muon_lr=0.1` with train loss ratio `0.1884`.

Caveat:

This is still a small fixed-grid check on sklearn digits. It reduces the
cherry-picking risk for the selected practical-training setting, but it does not
replace a real long-tailed benchmark with retuned optimizers.

Artifacts:

- [sweep_summary.csv](../results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv)
- [config.json](../results/e11_long_tail_practical_training_lr_sweep/config.json)
