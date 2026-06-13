# E11 Long-Tailed Muon-Style Compatibility Diagnostic

This diagnostic addresses the gap between the paper's clean direction
`polar(G_t)` and a practical Muon-like update that uses a momentum state and an
approximate polar factor. It reuses the long-tailed digits checkpoint and
matched-head-gain protocol, then compares four directions at the same parameter
state:

1. Frobenius-normalized head gradient, used as the GD-style baseline.
2. Exact `polar(G_t)`, the ideal direction analyzed in the paper.
3. Exact `polar(M_t)`, where `M_t` is an exponential moving average of recent
   head-batch gradients plus the current head gradient.
4. Newton-Schulz `NS(M_t)`, a finite-iteration approximation to `polar(M_t)`.

Settings:

- Seeds: 20
- Momentum beta: 0.9
- Momentum history steps: 8
- Newton-Schulz steps: 5
- Warmup steps: 80
- Hidden dimension: 32
- Target head first-order gain: 0.02 * head loss

## Summary

| direction | squared drift ratio vs Fro/GD | squared drift ratio vs polar(G_t) | cosine to polar(G_t) | alignment ratio to polar(G_t) |
|---|---:|---:|---:|---:|
| polar($G_t$) | 0.5501 [0.5101, 0.5931] | 1 [1, 1] | 1 [1, 1] | 1 [1, 1] |
| polar($M_t$) | 0.8199 [0.6951, 0.9672] | 1.491 [1.281, 1.735] | 0.3785 [0.3541, 0.4029] | 0.8179 [0.7544, 0.8867] |
| NS($M_t$) | 0.9116 [0.7696, 1.08] | 1.657 [1.405, 1.955] | 0.4024 [0.3813, 0.4236] | 0.7506 [0.6879, 0.8191] |

Mean gradient-momentum cosine across the paired seeds is
`0.8055`
`[0.7407, 0.8704]`.

## Interpretation

The ideal `polar(G_t)` row is the paper's current clean mechanism target. The
`polar(M_t)` row asks whether replacing the current gradient by a Muon-style
momentum state gives a compatible local drift signal. The `NS(M_t)` row asks
whether a finite Newton-Schulz approximation still stays close enough to the
momentum polar direction.

If `polar(M_t)` and `NS(M_t)` have tail drift below Fro/GD while maintaining high
cosine and alignment with `polar(G_t)`, the paper can describe the sampled
Muon-style directions as compatible with the ideal spectral direction in this
local diagnostic. If these rows degrade, the safe statement remains only about
spectral-gradient/polar geometry.

Current readout: `polar(M_t)` has squared tail-example logit drift ratio
`0.8199` vs Fro/GD,
and `NS(M_t)` has squared drift ratio
`0.9116`. This supports
only a selected-state compatibility check; it still does not prove full Muon
training performance.

Figure: [figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png](../figures/e11_long_tail_muon_bridge/long_tail_muon_bridge.png)

Artifacts:

- [step_metrics.csv](../results/e11_long_tail_muon_bridge/step_metrics.csv)
- [pair_summary.csv](../results/e11_long_tail_muon_bridge/pair_summary.csv)
- [config.json](../results/e11_long_tail_muon_bridge/config.json)
