# E05: Matrix Sensing Sample-Complexity Phase Diagram

## One-sentence purpose

Test whether Muon improves matrix sensing only in statistically well-determined regimes, or whether it also helps when the measurement system is undersampled.

## Research question

Muon changes the geometry of the update by replacing the raw gradient direction with a matrix-orthogonalized direction. In Matrix Sensing, this may help when the target matrix has difficult spectral structure, but it may hurt when the sensing problem is underdetermined or dominated by spurious measurement directions.

The main question is:

> Does Muon help more when the number of measurements is sufficient and the target matrix is ill-conditioned?

## Hypotheses

### H1: Muon helps in sufficiently sampled, spectrally difficult regimes

When the target matrix has a decaying spectrum or large condition number, vanilla first-order methods may struggle to recover weaker singular directions. Muon may help by producing updates with better matrix geometry.

Expected pattern:

| Regime | Expected Muon behavior |
|---|---|
| High measurement count, high condition number | Muon improves recovery error |
| High measurement count, simple spectrum | Muon similar to Adam/Shampoo |
| Low measurement count, high condition number | Muon may degrade or overfit |
| Low measurement count, simple spectrum | Optimizers may be similar |

### H2: Muon is not uniformly better

If Muon helps only because of update normalization, it may appear good across almost all regimes. If it is exploiting matrix spectral geometry, the advantage should be concentrated in specific regimes.

## Proposed notebook name

`E05_ms_sample_complexity_phase_diagram_torch.ipynb`

## Relationship to existing notebooks

This experiment extends the Matrix Sensing ablations already covered by `E03_ms_ablations_torch.ipynb`.

The main new axis is measurement sample complexity. Instead of using a fixed number of measurements, vary the measurement multiplier:

| Symbol | Meaning |
|---|---|
| `d` | Matrix dimension |
| `r` | Target rank |
| `m` | Number of measurements |
| `m_multiplier` | Multiplier in `m = m_multiplier * d * r` |

## Experimental grid

### Fixed parameters

| Parameter | Value |
|---|---|
| Matrix dimension | `d = 60` |
| Target rank | `rank = 5` |
| Measurement distribution | Gaussian / normal |
| Noise level | `0.0` |
| Initialization scale | Use current default from E03 |
| Maximum steps | Same as current Matrix Sensing benchmark |
| Early stopping | Same patience-based rule as current notebooks |

### Optimizers

| Optimizer | Role |
|---|---|
| Muon | Main method |
| Muon-Exact | Polar/SVD oracle reference |
| Shampoo | Matrix preconditioning baseline |
| Adam | Strong adaptive baseline |
| SGD | Simple first-order baseline |

### Measurement multipliers

| `m_multiplier` | Interpretation |
|---|---|
| `1` | Very undersampled |
| `2` | Near minimal sample regime |
| `3` | Mildly sampled |
| `4` | Moderately sampled |
| `6` | Well sampled |
| `8` | Strongly sampled |

### Spectrum cases

| Spectrum shape | Condition number | Interpretation |
|---|---:|---|
| hard cutoff | `1` | Easy low-rank target |
| polynomial decay | `10` | Moderate spectral decay |
| polynomial decay | `100` | Ill-conditioned polynomial spectrum |
| exponential decay | `10` | Moderate exponential decay |
| exponential decay | `100` | Hard ill-conditioned spectrum |

### Seeds

Use at least 5 seeds for the full experiment.

For a first smoke version, use 3 seeds and only the following settings:

| Axis | Smoke values |
|---|---|
| Optimizers | Muon, Muon-Exact, Adam, Shampoo |
| Measurement multipliers | `1`, `2`, `4`, `8` |
| Spectrum cases | hard cutoff kappa 1, exponential decay kappa 100 |

## Primary metrics

| Metric | Reason |
|---|---|
| Final loss | Measures optimization performance |
| Minimum loss | Avoids penalizing late instability only |
| Relative matrix recovery error | Measures whether the target matrix was recovered |
| Time per step | Separates statistical accuracy from cost |
| Actual steps | Detects early-stopping differences |
| Divergence rate | Captures instability across seeds |

The most important metric is relative matrix recovery error, not training loss.

## Main plots

### Plot 1: Muon-vs-Adam phase diagram

Create a heatmap with:

| Axis / value | Meaning |
|---|---|
| x-axis | Measurement multiplier |
| y-axis | Spectrum case |
| color | `log10(relative_error_Muon) - log10(relative_error_Adam)` |

Interpretation:

| Color sign | Meaning |
|---|---|
| Negative | Muon has lower recovery error |
| Positive | Adam has lower recovery error |
| Near zero | No clear difference |

### Plot 2: Muon-vs-Shampoo phase diagram

Same as Plot 1, but compare Muon against Shampoo.

This is important because Shampoo is also a matrix-aware baseline.

### Plot 3: Representative convergence curves

Do not plot every setting. Pick three:

| Regime | Spectrum | Measurement multiplier | Purpose |
|---|---|---:|---|
| Easy | hard cutoff, kappa 1 | `8` | sanity check |
| Learnable but hard | exponential decay, kappa 100 | `6` | likely Muon advantage |
| Undersampled and hard | exponential decay, kappa 100 | `1` | likely Muon degradation |

Plot loss and relative matrix error over steps.

### Plot 4: Success probability

Define success as reaching a relative matrix error below a fixed threshold, such as `1e-2` or `1e-3`.

Plot success probability over seeds as a function of measurement multiplier.

## Main table

Create a table with rows like:

| Setting | Best optimizer | Muon rank | Muon-vs-Adam gap | Muon-vs-Shampoo gap | Takeaway |
|---|---|---:|---:|---:|---|
| Easy, many samples | TBD | TBD | TBD | TBD | Optimizers similar |
| High kappa, many samples | TBD | TBD | TBD | TBD | Muon likely helpful |
| High kappa, few samples | TBD | TBD | TBD | TBD | Muon may degrade |

## Expected conclusion patterns

### If Muon wins only at high sample count

> Muon appears most useful when the sensing operator provides enough information to identify the target matrix. In undersampled regimes, its matrix-orthogonalized updates do not compensate for missing measurements and may even amplify unstable directions.

### If Muon wins even at low sample count

> Muon may provide an implicit regularization effect in Matrix Sensing, improving recovery even when the measurement system is relatively small. This would motivate further diagnostics on singular-value recovery and gradient spectrum.

### If Muon does not win anywhere

> In this Matrix Sensing setup, Muon does not provide a clear advantage over Adam or Shampoo after controlling sample complexity and spectrum. The next step is to test whether the advantage appears only in stochastic or neural-network-like regimes.

## Why this experiment is high priority

This is the cleanest experiment for answering the professor's request:

> Connect theoretical conditions to the specific problem settings and design experiments that vary or control the relevant conditions.

Sample complexity and condition number are interpretable, controllable, and directly connected to matrix recovery theory.
