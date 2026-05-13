# E08: Matrix Factorization Left/Right Scale Imbalance

## One-sentence purpose

Test whether Muon is robust to the scale invariance and factor imbalance inherent in matrix factorization.

## Research question

Matrix factorization has a scale symmetry:

`L R^T = (c L) (R / c)^T`

The represented matrix is unchanged, but the gradients with respect to `L` and `R` can change dramatically. This makes factorized optimization sensitive to imbalance between the two factors.

The main question is:

> Does Muon stabilize or destabilize optimization when the left and right factors start at very different scales?

## Hypotheses

### H1: Muon may reduce sensitivity to scale imbalance

Because Muon normalizes or orthogonalizes matrix-shaped updates, it may reduce the effect of raw gradient magnitude differences between `L` and `R`.

### H2: Muon may worsen imbalance

Because Muon changes the gradient geometry separately for each matrix parameter, it may ignore useful scale information and update the two factors in poorly matched directions.

### H3: Balancedness is more informative than loss alone

A run can reduce loss while creating increasingly unbalanced factors. This may lead to instability or poor generalization later.

## Proposed notebook name

`E08_mf_scale_imbalance_torch.ipynb`

## Relationship to existing notebooks

This experiment extends the unbalanced-factor initialization case from `E04_mf_initialization_ablations_torch.ipynb` into a systematic two-dimensional grid.

## Experimental grid

### Fixed parameters

| Parameter | Value |
|---|---|
| Matrix dimension | `d = 60` |
| True target rank | `true_rank = 5` |
| Factor rank | `factor_rank = 5` for main experiment |
| Spectrum shape | hard cutoff |
| Condition number | `1` for main experiment |
| Measurement distribution | Gaussian / normal |
| Measurement multiplier | Use current default; optionally set to `4` |
| Noise level | `0.0` |
| Early stopping | Same as current notebooks |

### Optimizers

| Optimizer | Role |
|---|---|
| Muon | Main method |
| Muon-Exact | SVD/polar reference |
| Shampoo | Matrix preconditioning baseline |
| Adam | Strong adaptive baseline |
| SGD | Simple baseline |

### Initialization scales

Use independent scales for the two factors:

| Left factor scale | Right factor scale |
|---:|---:|
| `1e-4` | `1e-4` |
| `1e-4` | `1e-3` |
| `1e-4` | `1e-2` |
| `1e-4` | `1e-1` |
| `1e-4` | `1.0` |
| `1e-3` | all right scales |
| `1e-2` | all right scales |
| `1e-1` | all right scales |
| `1.0` | all right scales |

This creates a 5-by-5 grid.

### Optional hard version

Repeat the experiment with:

| Spectrum shape | Condition number |
|---|---:|
| exponential decay | `100` |

## Primary metrics

| Metric | Reason |
|---|---|
| Final loss | Optimization quality |
| Relative matrix error | Recovery quality |
| Left factor norm | Detects scale growth |
| Right factor norm | Detects scale growth |
| Product norm | Separates factor explosion from matrix explosion |
| Balancedness | Measures whether factors remain balanced |
| Actual steps | Detects early stopping |
| Divergence rate | Detects unstable regimes |

## Balancedness metric

Track a balancedness score based on the difference between the Gram matrices of the two factors:

`balancedness = ||L^T L - R^T R|| / (||L^T L|| + ||R^T R|| + eps)`

Interpretation:

| Balancedness value | Meaning |
|---|---|
| Near 0 | Left and right factors are balanced |
| Moderate | Some scale mismatch |
| Large | Severe imbalance |

## Main plots

### Plot 1: Recovery heatmap for each optimizer

For each optimizer, create a heatmap:

| Axis / color | Meaning |
|---|---|
| x-axis | Right factor initialization scale |
| y-axis | Left factor initialization scale |
| color | Median relative matrix error |

### Plot 2: Muon-vs-Adam heatmap

| Axis / color | Meaning |
|---|---|
| x-axis | Right factor scale |
| y-axis | Left factor scale |
| color | `log10(error_Muon) - log10(error_Adam)` |

### Plot 3: Balancedness heatmap

For each optimizer:

| Axis / color | Meaning |
|---|---|
| x-axis | Right factor scale |
| y-axis | Left factor scale |
| color | Final balancedness |

### Plot 4: Balancedness over training

Pick three regimes:

| Regime | Left scale | Right scale | Reason |
|---|---:|---:|---|
| Balanced tiny | `1e-4` | `1e-4` | both small |
| Balanced standard | `1e-2` | `1e-2` | normal case |
| Severely imbalanced | `1e-4` | `1.0` | hard imbalance |

Plot balancedness over optimizer steps.

### Plot 5: Factor norms over training

For the same three regimes, plot:

| Quantity | Purpose |
|---|---|
| `||L||_F` | left factor scale |
| `||R||_F` | right factor scale |
| `||LR^T||_F` | represented matrix scale |

## Main table

| Regime | Best optimizer by recovery | Best optimizer by balancedness | Muon issue? | Takeaway |
|---|---|---|---|---|
| balanced tiny | TBD | TBD | TBD | TBD |
| balanced standard | TBD | TBD | TBD | TBD |
| left small, right large | TBD | TBD | TBD | TBD |
| left large, right small | TBD | TBD | TBD | TBD |

## Expected conclusion patterns

### If Muon improves imbalanced cases

> Muon appears to reduce sensitivity to factor scaling, suggesting that its matrix-structured updates partially compensate for poor initialization geometry.

### If Muon worsens imbalanced cases

> Muon appears to interact unfavorably with the scale symmetry of matrix factorization, possibly because it treats the two factor matrices independently and discards useful gradient magnitude information.

### If Adam is more stable

> Adam's coordinatewise adaptivity may be better suited to scale-imbalanced factor dynamics than Muon's matrix-level update normalization.

### If Shampoo is more stable

> Shampoo may better preserve matrix geometry while still adapting to factor-scale differences.

## Why this experiment matters

This experiment is directly tied to nonconvex optimization geometry. It helps explain whether Muon is merely a good optimizer for direct matrix variables or whether it is also appropriate for factorized low-rank models.
