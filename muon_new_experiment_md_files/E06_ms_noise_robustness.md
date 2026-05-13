# E06: Matrix Sensing Noise Robustness

## One-sentence purpose

Test whether Muon is robust to noisy measurements or whether its matrix-orthogonalized update amplifies noisy gradient directions.

## Research question

In noiseless Matrix Sensing, the target matrix can in principle be recovered exactly when the measurement operator is sufficiently informative. With noisy observations, however, the optimizer may reduce training loss while moving away from the true target matrix.

The main question is:

> Does Muon improve true matrix recovery under measurement noise, or does it mainly reduce training loss by fitting noisy observations?

## Hypotheses

### H1: Muon helps at low noise

When noise is small, Muon may still exploit spectral structure in the gradient and recover the target matrix faster than Adam or SGD.

### H2: Muon degrades at high noise

Because Muon changes the singular structure of the raw gradient update, it may overemphasize directions created by noise. In that case, training loss can improve while matrix recovery error gets worse.

### H3: Training loss and recovery error can disagree

The most important diagnostic is the gap between optimization performance and statistical recovery.

Expected pattern:

| Regime | Training loss | Recovery error | Interpretation |
|---|---|---|---|
| Low noise | low | low | good recovery |
| High noise, low loss, high error | low | high | overfitting or wrong inductive bias |
| High noise, high loss, high error | high | high | optimizer cannot fit noisy system |

## Proposed notebook name

`E06_ms_noise_robustness_torch.ipynb`

## Relationship to existing notebooks

This extends the noise ablations in `E03_ms_ablations_torch.ipynb`, but makes noise the central variable and compares training loss against true matrix recovery.

## Experimental grid

### Fixed parameters

| Parameter | Value |
|---|---|
| Matrix dimension | `d = 60` |
| Target rank | `rank = 5` |
| Measurement distribution | Gaussian / normal |
| Measurement multiplier | `4` for main experiment; optionally `2` and `8` for robustness |
| Initialization scale | Current Matrix Sensing default |
| Maximum steps | Same as E03 |
| Early stopping | Same as current notebooks |

### Optimizers

| Optimizer | Role |
|---|---|
| Muon | Main method |
| Muon-Exact | SVD/polar reference |
| Shampoo | Matrix preconditioning baseline |
| Adam | Strong adaptive baseline |
| SGD | Simple baseline |

### Spectrum cases

| Spectrum shape | Condition number | Purpose |
|---|---:|---|
| hard cutoff | `1` | Easy target |
| exponential decay | `10` | Moderate target |
| exponential decay | `100` | Hard ill-conditioned target |

### Noise levels

| Noise standard deviation | Interpretation |
|---:|---|
| `0.0` | noiseless |
| `1e-4` | tiny noise |
| `1e-3` | small noise |
| `1e-2` | moderate noise |
| `3e-2` | significant noise |
| `1e-1` | high noise |

### Seeds

Use 5 seeds for the full run.

Use 3 seeds for the first pass.

## Primary metrics

| Metric | Reason |
|---|---|
| Final training loss | Measures how well observations are fitted |
| Minimum training loss | Captures best optimization result |
| Relative matrix recovery error | Measures statistical recovery |
| Generalization-style measurement error | Optional; evaluate on fresh noiseless measurements |
| Actual steps | Captures early stopping behavior |
| Divergence rate | Captures instability |
| Time per step | Captures computational cost |

## Important distinction

Do not use training loss alone as the conclusion.

In noisy Matrix Sensing, the central comparison is:

| Quantity | What it measures |
|---|---|
| Training loss | Fit to noisy observed measurements |
| Recovery error | Closeness to true target matrix |
| Fresh noiseless measurement error | Generalization to clean measurements |

Muon can look good on training loss while being worse on recovery error.

## Main plots

### Plot 1: Noise vs recovery error

For each spectrum case, plot:

| x-axis | y-axis | line |
|---|---|---|
| noise level | median relative matrix error | optimizer |

Use log scale for both noise and error if appropriate.

### Plot 2: Noise vs training loss

Same format as Plot 1, but y-axis is final training loss.

This plot should be interpreted together with Plot 1.

### Plot 3: Recovery-over-training gap

Plot the difference between normalized recovery error and normalized training loss.

Purpose:

> Identify regimes where an optimizer fits noisy observations but fails to recover the true matrix.

### Plot 4: Muon-vs-Adam noise gap

Create a line plot:

| x-axis | y-axis |
|---|---|
| noise level | `log10(relative_error_Muon) - log10(relative_error_Adam)` |

Repeat for each spectrum case.

### Plot 5: Representative convergence curves

Pick two settings:

| Setting | Reason |
|---|---|
| Exponential decay, kappa 100, noise 0 | clean hard case |
| Exponential decay, kappa 100, noise 0.03 | noisy hard case |

Plot both training loss and recovery error over steps.

## Main table

| Spectrum | Noise | Best training-loss optimizer | Best recovery optimizer | Does Muon overfit? | Takeaway |
|---|---:|---|---|---|---|
| hard cutoff | TBD | TBD | TBD | TBD | TBD |
| exp kappa 10 | TBD | TBD | TBD | TBD | TBD |
| exp kappa 100 | TBD | TBD | TBD | TBD | TBD |

## Expected conclusion patterns

### If Muon is robust

> Muon maintains lower recovery error than Adam and SGD across increasing noise levels, suggesting that its matrix-structured update does not simply overfit noisy measurements.

### If Muon overfits noise

> Muon reduces training loss effectively but loses its recovery advantage at higher noise levels. This suggests that matrix-orthogonalized updates may amplify noisy measurement directions.

### If Shampoo is more robust than Muon

> Shampoo appears more stable under noisy measurements, possibly because its preconditioned updates preserve more information about gradient magnitude than Muon's polar-style update.

## Why this experiment matters

This experiment separates optimization from recovery. It prevents a misleading conclusion where Muon appears better only because it drives down the noisy training objective.
