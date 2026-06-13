# E11 Long-Tailed Layerwise Diagnostic

This probe uses the same imbalanced sklearn digits checkpoint as the one-step
and head-only forgetting diagnostics. For each layer, it measures the head
gradient rank, tail activation stable rank, downstream-aware tail rank,
finite-difference tail JVP drift, and layer-only tail drift under matched
first-order head gain.

- Seeds: 20
- Layers: 2
- Head classes: (0, 1, 2, 3, 4)
- Tail classes: (5, 6, 7, 8, 9)
- Warmup steps: 80
- Target head first-order gain: 0.02 * head-batch loss
- JVP finite-difference epsilon: 0.0001

Ratio columns are spectral divided by Frobenius/GD. Values below 1 mean the
spectral layer direction disturbs logits on held-out tail examples less.

| layer | activation-only score nr(G)/sr(A) | downstream-aware rank | local operator score | unit-JVP squared drift ratio 95% CI | scaled-JVP squared drift ratio 95% CI | observed squared drift ratio 95% CI | spectral lower observed fraction | tail loss diff 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.352 | n/a | 0.7261 | 1.45 [1.324, 1.588] | 0.4766 [0.4442, 0.5114] | 0.4767 [0.4443, 0.5115] | 1 | 0.0001384 [-0.001391, 0.001667] |
| 2 | 2.206 | 1.136 | 0.2206 | 1.476 [1.383, 1.574] | 0.596 [0.5623, 0.6318] | 0.596 [0.5623, 0.6318] | 1 | 0.0006288 [-0.0003051, 0.001563] |

Figure: [figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png](../figures/e11_long_tail_layerwise/long_tail_layerwise_drift.png)

Main readout: this is a layer-level check of the head-to-tail mechanism.
The strongest supported statement is whether finite-difference tail JVP
squared drift ratios, head-gain scaling, and layer-only observed squared drift ratios agree layer
by layer. Unit JVP drift measures tail sensitivity of the direction; scaled
JVP drift also includes the smaller/larger step required to match head gain.
For layer 2, `downstream-aware rank` is the exact sandwich quantity
`ssrank(B_T,A_T)` because the final linear layer has a single downstream
matrix. For layer 1, ReLU gates vary across tail samples, so the layer is
not a single `B_T D A_T` block; the table instead reports the exact stable
rank of the frozen-gate local linear operator `J_{T,1}` through the
`local operator score`.

Across both layers, the Spearman correlation between local operator score and
the observed spectral/Frobenius squared drift ratio is -0.7634 [-0.8711, -0.5858].

Caveats:
- This is still a small MLP diagnostic, not an exact matrix-block theorem check.
- JVP is estimated by finite difference; the epsilon is fixed and reported.
- Per-layer updates are artificial interventions and should not be interpreted as complete optimizer trajectories.

Artifacts:
- [metrics.csv](../results/e11_long_tail_layerwise/metrics.csv)
- [summary.csv](../results/e11_long_tail_layerwise/summary.csv)
- [config.json](../results/e11_long_tail_layerwise/config.json)
