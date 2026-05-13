# E10: Muon Variant and Normalization Ablation

## One-sentence purpose

Test whether Muon's behavior comes from polar matrix geometry, approximate orthogonalization quality, or simple update normalization.

## Research question

Muon modifies the raw gradient using matrix operations. But an empirical improvement could come from several different sources:

1. Matrix polar geometry.
2. Approximate orthogonalization.
3. Frobenius or spectral normalization of the update.
4. A larger or more stable effective learning rate.

The main question is:

> Is Muon better because of matrix polar structure, or because it normalizes updates?

## Hypotheses

### H1: Exact polar Muon is better than approximate Muon in hard regimes

If polar accuracy matters, `Muon-Exact` should outperform low-quality Newton--Schulz or truncated variants in ill-conditioned regimes.

### H2: Approximate Muon is enough in easy regimes

If the task is easy or gradients are well-conditioned, approximate Muon may match exact Muon.

### H3: Normalized SGD should not fully match Muon

If Muon is using matrix geometry, then simple normalized baselines should be weaker than Muon, especially in spectrally structured tasks.

### H4: If NormalizedSGD matches Muon, the story should change

If normalized baselines perform similarly, then the main explanation is likely update scale control rather than matrix polar geometry.

## Proposed notebook name

`E10_muon_variant_ablation_torch.ipynb`

## Relationship to existing notebooks

This is a targeted mechanism ablation. It should use a small number of representative Matrix Sensing and Matrix Factorization settings, not a large exhaustive grid.

## Optimizer variants

### Main baselines

| Method | Purpose |
|---|---|
| Muon | Practical implementation |
| Muon-Exact | Exact polar/SVD reference |
| Adam | Strong adaptive baseline |
| Shampoo | Matrix-aware baseline |
| SGD | Simple baseline |

### Muon variants

| Variant | Purpose |
|---|---|
| Muon with few Newton--Schulz steps | Tests low-quality polar approximation |
| Muon with more Newton--Schulz steps | Tests approximation improvement |
| Exact SVD Muon | Tests idealized polar update |
| Truncated Muon | Tests low-rank polar update |
| Randomized-SVD Muon | Tests cheaper approximate polar update |

### Normalization baselines

| Method | Purpose |
|---|---|
| NormalizedSGD | Raw gradient divided by Frobenius norm |
| SpectralNormSGD | Raw gradient divided by operator norm |
| LayerwiseNormalizedSGD | Parameter-wise normalized gradient update |

These baselines are important because they test whether Muon is merely a normalized gradient method.

## Representative settings

### Matrix Sensing

| Setting | Spectrum | Condition number | Measurement multiplier | Noise | Purpose |
|---|---|---:|---:|---:|---|
| MS-easy | hard cutoff | `1` | `8` | `0.0` | easy baseline |
| MS-hard | exponential decay | `100` | `6` | `0.0` | likely Muon advantage |
| MS-undersampled | exponential decay | `100` | `1` | `0.0` | possible Muon failure |
| MS-noisy | exponential decay | `100` | `4` | `0.03` | noise sensitivity |

### Matrix Factorization

| Setting | Factor rank | Init condition | Purpose |
|---|---:|---|---|
| MF-standard | `5` | standard balanced init | baseline |
| MF-tiny | `5` | tiny balanced init | tests normalization effect |
| MF-overparam | `20` | standard balanced init | tests implicit bias |
| MF-imbalanced | `5` | left tiny, right large | tests factor-scale geometry |

## Primary metrics

| Metric | Reason |
|---|---|
| Relative matrix error | Main recovery metric |
| Final loss | Optimization quality |
| Minimum loss | Best optimization result |
| Time per step | Cost of variant |
| Total runtime | Practicality |
| Divergence rate | Stability |
| Gradient-update cosine | Whether variant remains descent-aligned |
| Update effective rank | Whether variant changes matrix geometry |

## Main plots

### Plot 1: Accuracy-cost scatter plot

For each representative setting, plot:

| Axis / marker | Meaning |
|---|---|
| x-axis | Time per step or total runtime |
| y-axis | Relative matrix error |
| marker | optimizer or variant |

This separates practical methods from oracle methods.

### Plot 2: Muon approximation quality

Plot performance for Muon variants ordered by approximation quality:

| x-axis | y-axis |
|---|---|
| Muon variant | Relative matrix error |

Use separate panels or grouped bars for representative settings.

### Plot 3: Normalization baseline comparison

Compare:

| Method group | Interpretation |
|---|---|
| SGD | raw first-order method |
| NormalizedSGD | scale-normalized first-order method |
| SpectralNormSGD | operator-normalized first-order method |
| Muon | matrix-polar method |
| Muon-Exact | exact matrix-polar method |

If Muon is much better than normalized baselines, the polar matrix geometry matters.

### Plot 4: Gradient-update cosine by method

Compare cosine alignment across variants.

This helps explain whether bad variants fail because they produce poor descent directions.

### Plot 5: Update effective rank by method

Compare update spectral structure across variants.

This helps determine whether the successful variants have a distinctive matrix update geometry.

## Main table

| Setting | Best practical method | Best oracle method | Does NormalizedSGD match Muon? | Does Exact Muon improve over practical Muon? | Takeaway |
|---|---|---|---|---|---|
| MS-easy | TBD | TBD | TBD | TBD | TBD |
| MS-hard | TBD | TBD | TBD | TBD | TBD |
| MS-noisy | TBD | TBD | TBD | TBD | TBD |
| MF-tiny | TBD | TBD | TBD | TBD | TBD |
| MF-overparam | TBD | TBD | TBD | TBD | TBD |

## Expected conclusion patterns

### If Muon beats normalized baselines

> Muon's advantage cannot be explained by update normalization alone. The matrix polar structure appears to contribute meaningfully.

### If normalized baselines match Muon

> Muon's empirical advantage in these settings may be primarily due to normalized update scale rather than matrix-specific polar geometry.

### If Exact Muon beats approximate Muon

> Polar approximation quality matters in hard regimes, especially when the target matrix is ill-conditioned or the gradient spectrum is highly skewed.

### If approximate Muon matches Exact Muon

> The practical approximation is sufficient for these matrix recovery tasks, making exact SVD unnecessary except as an analysis reference.

### If truncated or randomized variants work well

> Muon may not need full-rank polar information. A low-rank or randomized approximation could be enough when the target and gradients are low-rank.

## Why this experiment matters

This experiment protects against an overly broad claim like “Muon works because of matrix geometry.” It forces the analysis to distinguish matrix polar geometry from simpler normalization effects.
