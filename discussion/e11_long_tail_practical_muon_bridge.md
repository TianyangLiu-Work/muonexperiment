# E11 Long-Tailed Practical-Muon Trajectory Compatibility

This diagnostic extends the fixed-checkpoint Muon-style compatibility check along a short practical
NS-Muon-style head-only trajectory. Starting from the same long-tailed digits
checkpoint, each seed runs `6` practical updates using a
momentum state, finite Newton-Schulz polar approximation, and learning rate
`0.001`. At every trajectory state, the script re-evaluates the
matched-head-gain diagnostic for Fro/GD, `polar(G_t)`, `polar(M_t)`, and
`NS(M_t)`.

## Summary

| direction | comparisons | squared drift ratio vs Fro/GD | cosine to polar(G_t) | gradient-momentum cosine |
|---|---:|---:|---:|---:|
| polar($G_t$) | 120 | 0.5604 [0.5449, 0.5763] | 1 [1, 1] | 0.8589 [0.8328, 0.885] |
| polar($M_t$) | 120 | 0.7292 [0.6891, 0.7717] | 0.4764 [0.466, 0.4868] | 0.8589 [0.8328, 0.885] |
| NS($M_t$) | 120 | 0.8019 [0.7583, 0.848] | 0.4361 [0.4275, 0.4446] | 0.8589 [0.8328, 0.885] |

## Interpretation

Across `120` paired state-step comparisons,
`polar(M_t)` has squared tail-example logit drift ratio
`0.7292`
`[0.6891, 0.7717]`
relative to Fro/GD. The practical finite-step `NS(M_t)` direction has squared drift ratio
`0.8019`
`[0.7583, 0.848]`.

This extends the fixed-checkpoint compatibility check by testing sampled states
along a short practical Muon-like trajectory. It remains a local
matched-head-gain diagnostic: it does not prove final tail accuracy or broad
optimizer superiority.

Figure: [figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png](../figures/e11_long_tail_practical_muon_bridge/long_tail_practical_muon_bridge.png)

Artifacts:

- [step_metrics.csv](../results/e11_long_tail_practical_muon_bridge/step_metrics.csv)
- [step_summary.csv](../results/e11_long_tail_practical_muon_bridge/step_summary.csv)
- [summary.csv](../results/e11_long_tail_practical_muon_bridge/summary.csv)
- [config.json](../results/e11_long_tail_practical_muon_bridge/config.json)
