# E07: Matrix Factorization Rank-Initialization Phase Diagram

## One-sentence purpose

Test how Muon behaves in nonconvex matrix factorization as initialization scale and factor rank vary.

## Research question

Matrix Factorization introduces nonconvex geometry even when the target matrix is simple. The same matrix can be represented by many factor pairs, and the optimization path depends strongly on initialization.

The main question is:

> Is Muon robust to initialization scale and factor-rank overparameterization, or does it become unstable in specific nonconvex regimes?

## Hypotheses

### H1: Muon is sensitive to tiny initialization

When factors are initialized very close to zero, gradients can be small or poorly balanced. Muon may either help by normalizing updates or hurt by taking overly aggressive steps.

### H2: Muon behaves differently under overparameterization

When factor rank is larger than the true rank, the optimizer must decide how to use extra degrees of freedom. Muon may produce different implicit bias from Adam, SGD, or Shampoo.

### H3: Factorization behavior is not equivalent to direct Matrix Sensing

Even if Muon works well on direct matrix variables, it may behave differently when the optimized variables are factors `L` and `R`.

## Proposed notebook name

`E07_mf_rank_init_phase_diagram_torch.ipynb`

## Relationship to existing notebooks

This experiment extends `E04_mf_initialization_ablations_torch.ipynb`. The existing notebook already considers tiny, unbalanced, oversized, and ill-conditioned initialization cases. This experiment turns those cases into a systematic two-dimensional phase diagram.

## Experimental grid

### Fixed parameters

| Parameter | Value |
|---|---|
| Matrix dimension | `d = 60` |
| True target rank | `true_rank = 5` |
| Measurement model | Same as existing Matrix Factorization setup |
| Measurement distribution | Gaussian / normal |
| Measurement multiplier | Use current E02/E04 default; optionally set to `4` |
| Noise level | `0.0` in the main experiment |
| Early stopping | Same patience rule as current notebooks |

### Optimizers

| Optimizer | Role |
|---|---|
| Muon | Main method |
| Muon-Exact | SVD/polar reference |
| Shampoo | Matrix preconditioning baseline |
| Adam | Strong adaptive baseline |
| SGD | Simple baseline |

### Factor ranks

| Factor rank | Interpretation |
|---:|---|
| `3` | Underparameterized |
| `5` | Correct rank |
| `10` | Moderately overparameterized |
| `20` | Strongly overparameterized |

### Initialization scales

| Init scale | Interpretation |
|---:|---|
| `1e-4` | extremely tiny |
| `3e-4` | very tiny |
| `1e-3` | tiny |
| `3e-3` | small |
| `1e-2` | standard small |
| `3e-2` | moderate |
| `1e-1` | large |
| `3e-1` | very large |

### Spectrum cases

First pass:

| Spectrum shape | Condition number | Purpose |
|---|---:|---|
| hard cutoff | `1` | clean nonconvex geometry test |

Second pass:

| Spectrum shape | Condition number | Purpose |
|---|---:|---|
| exponential decay | `100` | hard ill-conditioned target |

### Seeds

Use 5 seeds in the full run.

For the first version, use 3 seeds and only the optimizers Muon, Muon-Exact, Adam, and Shampoo.

## Primary metrics

| Metric | Reason |
|---|---|
| Final loss | Measures objective optimization |
| Minimum loss | Measures best achieved optimization |
| Relative matrix error | Measures target recovery |
| Effective recovered rank | Detects rank collapse or rank inflation |
| Factor norm | Detects unstable scaling of factors |
| Actual steps | Captures early stopping |
| Divergence rate | Captures instability |
| Time per step | Captures optimizer cost |

## Additional diagnostics

Track the following for factorized solutions:

| Diagnostic | Reason |
|---|---|
| Norm of left factor | Detects scale explosion |
| Norm of right factor | Detects scale explosion |
| Product matrix norm | Checks whether only factors or product are unstable |
| Nuclear norm of product | Tracks implicit low-rank bias |
| Effective rank of product | Detects whether overparameterized factors produce unnecessary rank |

## Main plots

### Plot 1: Optimizer-specific heatmaps

For each optimizer, create a heatmap:

| Axis / color | Meaning |
|---|---|
| x-axis | Initialization scale |
| y-axis | Factor rank |
| color | Median relative matrix error |

Use the same color scale across optimizers.

### Plot 2: Muon-vs-Adam heatmap

| Axis / color | Meaning |
|---|---|
| x-axis | Initialization scale |
| y-axis | Factor rank |
| color | `log10(error_Muon) - log10(error_Adam)` |

### Plot 3: Muon-vs-Shampoo heatmap

Same as Plot 2, but compare Muon to Shampoo.

### Plot 4: Divergence heatmap

For each optimizer, plot divergence rate over:

| x-axis | y-axis |
|---|---|
| initialization scale | factor rank |

This is important because aggressive updates may look good only on successful seeds.

### Plot 5: Representative convergence curves

Pick four regimes:

| Regime | Factor rank | Init scale | Purpose |
|---|---:|---:|---|
| Correct rank, standard init | `5` | `1e-2` | baseline |
| Correct rank, tiny init | `5` | `1e-4` | tests tiny-init behavior |
| Overparameterized, standard init | `20` | `1e-2` | tests extra degrees of freedom |
| Overparameterized, large init | `20` | `3e-1` | tests instability |

## Main table

| Regime | Best optimizer | Muon rank | Divergence issue? | Main takeaway |
|---|---|---:|---|---|
| underparameterized | TBD | TBD | TBD | TBD |
| correct rank | TBD | TBD | TBD | TBD |
| overparameterized | TBD | TBD | TBD | TBD |
| tiny init | TBD | TBD | TBD | TBD |
| large init | TBD | TBD | TBD | TBD |

## Expected conclusion patterns

### If Muon is robust across init scales

> Muon is relatively insensitive to factor initialization scale, suggesting that matrix-orthogonalized updates may stabilize some nonconvex factorization dynamics.

### If Muon helps tiny initialization but hurts large initialization

> Muon appears useful when raw gradients are small, but can be too aggressive when factor norms are already large.

### If Muon degrades under overparameterization

> Muon may interact unfavorably with overparameterized factor geometry, possibly because polar-style updates do not preserve the implicit low-rank bias of other methods.

### If Muon improves overparameterized cases

> Muon may provide a useful implicit bias in overparameterized matrix factorization, encouraging faster recovery of the dominant low-rank structure.

## Why this experiment is high priority

This experiment directly connects Muon to the second problem setting. It also tests whether conclusions from direct Matrix Sensing transfer to nonconvex factorized optimization.
