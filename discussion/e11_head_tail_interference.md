# E11 Head-to-Tail Interference Probe

This probe implements a synthetic one-step linear classification experiment for the
head-to-tail interference note. For each seed, it compares a Frobenius-normalized
gradient step and a spectral/polar step scaled to the same first-order head gain.
The tail batch is held out; the reported drift is the change in tail logits after
the head-only step.

- Seeds: 80
- Output dimension: 10
- Input dimension: 24
- Tail batch size: 64
- First-order head gain: 0.25 * ||G_H||_F

Ratio columns are spectral divided by Frobenius. Values below 1 mean the spectral
step disturbed the tail outputs less.

| setting | predicted spectral less drift | nrank(G_H) | srank(A_T) | ssrank(B_T,A_T) | theory ratio | observed drift-sq ratio 95% CI | spectral less drift fraction | CE increase diff 95% CI | margin drop diff 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_head_rank_low_tail_srank | True | 8.607 | 1.002 | 1 | 0.1162 | 0.3403 [0.3403, 0.3403] | 1 | 2.684e-05 [-0.0001661, 0.0002198] | 0.0001355 [-0.001351, 0.001622] |
| low_head_rank_high_tail_srank | False | 1.387 | 24 | 10 | 7.208 | 7.208 [7.208, 7.208] | 0 | -6.28e-05 [-0.000279, 0.0001534] | -9.879e-05 [-0.001074, 0.0008766] |

Figure: [figures/e11_head_tail_interference/head_tail_drift_ratio.png](../figures/e11_head_tail_interference/head_tail_drift_ratio.png)

Main readout: the observed tail-output drift follows the same qualitative boundary
as the theoretical ratio ssrank(B_T,A_T) / nrank(G_H). In the high-head-rank /
low downstream-aware tail-stable-rank setting, the spectral step has lower tail
drift. In the low-head-rank / high downstream-aware tail-stable-rank setting,
it has higher tail drift.

Caveat: this first probe validates the matched-head-gain drift mechanism, not
real long-tailed generalization. Cross-entropy and margin changes are reported as
secondary diagnostics and can be smaller or noisier than direct output drift.

Artifacts:
- [step_metrics.csv](../results/e11_head_tail_interference/step_metrics.csv)
- [pair_summary.csv](../results/e11_head_tail_interference/pair_summary.csv)
- [config.json](../results/e11_head_tail_interference/config.json)
