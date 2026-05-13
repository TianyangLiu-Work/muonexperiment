# E09: Muon Geometry Diagnostics

## One-sentence purpose

Diagnose whether Muon's empirical behavior is explained by gradient spectrum, update spectrum, descent alignment, and singular-value recovery.

## Research question

Benchmark plots can show whether Muon wins or loses, but they do not explain why. Since Muon is a matrix optimizer, its behavior should be analyzed through matrix quantities rather than only scalar loss curves.

The main question is:

> When Muon improves performance, what changes in the geometry of gradients, updates, and recovered singular values?

## Hypotheses

### H1: Muon helps when gradients are spectrally skewed

If raw gradients have a few dominant singular directions, Muon may produce updates that better use the matrix structure.

### H2: Muon can hurt when update-gradient alignment becomes poor

Orthogonalized updates may have lower cosine alignment with the negative raw gradient. This may be acceptable if the update has better long-term geometry, but harmful if the alignment becomes too weak.

### H3: Muon advantage should appear in singular-value recovery

If Muon is truly exploiting low-rank matrix structure, it should recover the target singular values faster or more accurately, especially in ill-conditioned regimes.

## Proposed notebook name

`E09_muon_geometry_diagnostics_torch.ipynb`

## Relationship to existing notebooks

This is a small diagnostics notebook, not a large grid benchmark. It should reuse representative settings from the Matrix Sensing and Matrix Factorization experiments.

## Representative settings

Use four to six carefully chosen regimes.

### Matrix Sensing settings

| Setting name | Spectrum | Condition number | Measurement multiplier | Noise | Purpose |
|---|---|---:|---:|---:|---|
| MS-easy | hard cutoff | `1` | `8` | `0.0` | sanity check |
| MS-hard-learnable | exponential decay | `100` | `6` | `0.0` | likely Muon advantage |
| MS-undersampled-hard | exponential decay | `100` | `1` | `0.0` | likely Muon degradation |
| MS-noisy-hard | exponential decay | `100` | `4` | `0.03` | noise robustness |

### Matrix Factorization settings

| Setting name | Factor rank | Init scale / imbalance | Spectrum | Purpose |
|---|---:|---|---|---|
| MF-standard | `5` | standard balanced | hard cutoff | baseline |
| MF-tiny-init | `5` | tiny balanced | hard cutoff | tests normalized update behavior |
| MF-overparam | `20` | standard balanced | hard cutoff | tests overparameterization |
| MF-imbalanced | `5` | left tiny, right large | hard cutoff | tests factor-scale symmetry |

## Optimizers

Use a smaller optimizer set for diagnostics:

| Optimizer | Role |
|---|---|
| Muon | Main method |
| Muon-Exact | Polar/SVD reference |
| Shampoo | Matrix-aware baseline |
| Adam | Strong adaptive baseline |
| SGD | Simple baseline |

Optional extra baselines:

| Optimizer | Purpose |
|---|---|
| NormalizedSGD | Tests whether normalization alone explains Muon |
| SpectralNormSGD | Tests whether operator-norm normalization explains Muon |

## Quantities to record

Record diagnostics every fixed number of steps, such as every 10 or 20 optimizer steps.

### Gradient diagnostics

For each matrix-shaped parameter gradient `G`, record:

| Quantity | Meaning |
|---|---|
| Frobenius norm | Overall gradient size |
| Operator norm | Strength of largest singular direction |
| Stable rank | Effective dimensionality based on norm ratio |
| Effective rank | Entropy-like spread of singular values |
| Top singular value | Dominant gradient direction |
| Top-k singular values | Spectral shape over time |
| Gradient condition proxy | Ratio of large to small singular directions |

### Update diagnostics

For each optimizer update `U`, record:

| Quantity | Meaning |
|---|---|
| Frobenius norm | Overall update size |
| Operator norm | Largest update direction |
| Effective rank | Spectral spread of update |
| Relative step size | Update norm divided by parameter norm |
| Gradient-update cosine | Alignment between `-G` and `U` |
| Update-to-gradient norm ratio | How much the optimizer rescales the raw gradient |

### Recovery diagnostics

For the represented matrix `X_hat`, record:

| Quantity | Meaning |
|---|---|
| Relative matrix error | Main recovery metric |
| Top-k singular value error | Whether dominant spectrum is recovered |
| Tail singular value error | Whether weak components are recovered |
| Effective rank of `X_hat` | Whether solution rank is controlled |
| Subspace overlap | Optional; compares recovered and target singular spaces |

## Main plots

### Plot 1: Gradient effective rank over time

For each representative setting, plot:

| x-axis | y-axis | line |
|---|---|---|
| optimizer step | gradient effective rank | optimizer |

Question:

> Does Muon perform best when raw gradients are low-rank or spectrally concentrated?

### Plot 2: Update effective rank over time

Same as Plot 1, but for optimizer updates.

Question:

> Does Muon produce updates with systematically different spectral structure from Adam, SGD, or Shampoo?

### Plot 3: Gradient-update cosine over time

Plot the cosine alignment between the negative gradient and the actual update.

Interpretation:

| Pattern | Meaning |
|---|---|
| High cosine | update follows gradient direction closely |
| Low positive cosine | update changes direction substantially |
| Negative cosine | update may not be a descent direction |

### Plot 4: Relative step size over time

Plot update norm divided by parameter norm.

This is especially important for tiny initialization and factorization imbalance.

### Plot 5: Singular-value recovery over time

For Matrix Sensing and Matrix Factorization, plot error in top target singular values over training.

Question:

> Does Muon recover weak singular values faster or more accurately?

### Plot 6: Final singular value comparison

For selected settings, plot the target singular values and final recovered singular values for each optimizer.

## Suggested diagnostic summaries

For each representative setting, summarize:

| Setting | Muon performance | Gradient spectrum | Update spectrum | Alignment | Singular-value recovery | Interpretation |
|---|---|---|---|---|---|---|
| MS-easy | TBD | TBD | TBD | TBD | TBD | TBD |
| MS-hard-learnable | TBD | TBD | TBD | TBD | TBD | TBD |
| MS-noisy-hard | TBD | TBD | TBD | TBD | TBD | TBD |
| MF-tiny-init | TBD | TBD | TBD | TBD | TBD | TBD |

## Expected conclusion patterns

### If Muon wins and has distinctive update spectrum

> Muon's advantage appears correlated with a distinct update spectral profile, suggesting that its matrix-orthogonalized update is doing more than scalar learning-rate rescaling.

### If Muon wins but looks like NormalizedSGD

> Muon's advantage may be largely explained by update normalization rather than polar matrix geometry.

### If Muon loses when cosine alignment is poor

> Muon can become harmful when the orthogonalized update has weak alignment with the raw descent direction, especially in noisy or undersampled regimes.

### If Muon recovers singular values better

> Muon's benefit is visible not only in scalar loss but also in structural recovery of the target matrix spectrum.

## Why this experiment is essential

This notebook turns the project from a benchmark into an analysis. It gives mechanistic evidence that can support or reject hypotheses about why Muon works.
