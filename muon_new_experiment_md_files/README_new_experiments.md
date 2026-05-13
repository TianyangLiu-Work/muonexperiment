# Muon Optimizer: New Experiment Design Notes

This folder contains Markdown-only experiment plans for extending the current `torch-notebook-rewrite` branch.

The goal is not to add another generic benchmark. The goal is to build a clearer research story around when Muon helps, when it fails, and what mechanism explains the behavior.

## Current notebook context

The current branch already has four notebook-first experiments:

| Existing notebook | Current role |
|---|---|
| `E01_ms_benchmark_torch.ipynb` | Matrix Sensing benchmark |
| `E02_matrix_factorization_torch.ipynb` | Matrix Factorization benchmark |
| `E03_ms_ablations_torch.ipynb` | Matrix Sensing ablations over measurement distribution, spectrum, condition number, and noise |
| `E04_mf_initialization_ablations_torch.ipynb` | Matrix Factorization initialization ablations |

The new experiments below are designed to fit this structure. They can be implemented either by extending `E03` and `E04`, or by creating a small number of new notebooks.

## Recommended new notebooks

| Proposed notebook | Main question |
|---|---|
| `E05_ms_sample_complexity_phase_diagram_torch.ipynb` | Does Muon help only in sufficiently sampled matrix sensing regimes? |
| `E06_ms_noise_robustness_torch.ipynb` | Does Muon amplify noisy gradient directions? |
| `E07_mf_rank_init_phase_diagram_torch.ipynb` | How do initialization scale and factor rank affect Muon in nonconvex factorization? |
| `E08_mf_scale_imbalance_torch.ipynb` | Does Muon help or hurt under left/right factor imbalance? |
| `E09_muon_geometry_diagnostics_torch.ipynb` | What spectral mechanism explains Muon behavior? |
| `E10_muon_variant_ablation_torch.ipynb` | Is Muon better because of matrix polar geometry, or just because of normalization? |

## Priority order

The most important experiments are:

1. `E05_ms_sample_complexity_phase_diagram_torch.ipynb`
2. `E07_mf_rank_init_phase_diagram_torch.ipynb`
3. `E09_muon_geometry_diagnostics_torch.ipynb`
4. `E06_ms_noise_robustness_torch.ipynb`
5. `E08_mf_scale_imbalance_torch.ipynb`
6. `E10_muon_variant_ablation_torch.ipynb`

If time is short, do only `E05`, `E07`, and `E09`. That already gives a coherent progress-report story:

> Muon is not uniformly better. Its behavior depends on statistical sample size, target spectrum, factorization geometry, and gradient spectral structure.

## Common optimizer set

Use the same default optimizer set already used in the branch:

| Method | Role |
|---|---|
| `Muon` | Practical Muon implementation |
| `Muon-Exact` | SVD/polar oracle version |
| `Shampoo` | Matrix preconditioning baseline |
| `Adam` | Adaptive first-order baseline |
| `SGD` | Classical first-order baseline |

For some mechanism ablations, add:

| Method | Role |
|---|---|
| `NormalizedSGD` | Tests whether Muon is only step normalization |
| `SpectralNormSGD` | Tests whether operator-norm normalization explains Muon |
| `Muon-NS-k` | Tests Newton--Schulz approximation quality |
| `Muon-Truncated` | Tests low-rank polar direction |
| `Muon-RandSVD` | Tests cheaper approximate polar direction |

## Shared metrics

Every new experiment should report the following whenever possible:

| Metric | Meaning |
|---|---|
| `final_loss` | Training objective at the final step |
| `min_loss` | Best training objective reached during training |
| `relative_matrix_error` | Recovery error compared with the ground-truth target matrix |
| `actual_steps` | Number of optimizer steps actually executed |
| `stopped_early` | Whether patience-based early stopping triggered |
| `stop_reason` | Early-stopping or failure reason |
| `time_s` | Wall-clock runtime |
| `time_per_step` | Runtime cost per optimizer step |
| `diverged` | Whether loss became NaN, infinite, or much worse than initial loss |

For mechanism notebooks, additionally report:

| Metric | Meaning |
|---|---|
| gradient Frobenius norm | Size of raw gradient |
| gradient operator norm | Largest singular direction of raw gradient |
| gradient effective rank | Whether the gradient is low-rank or spread out |
| update Frobenius norm | Size of optimizer update |
| update effective rank | Whether the update is low-rank or full-rank |
| gradient-update cosine | Alignment between descent direction and actual update |
| relative step size | Update size divided by parameter size |
| singular-value recovery error | Whether the target matrix spectrum is recovered |

## Plotting standard

Each notebook should end with a short conclusion cell and a small number of clear figures:

1. One main comparison figure.
2. One Muon-vs-baseline gap figure.
3. One robustness or diagnostics figure.
4. One short table summarizing the best/worst regimes.

Avoid too many convergence curves. Use convergence curves only for representative settings.

## Suggested progress-report structure

A clean report can be organized as:

1. **Question:** When should Muon improve matrix optimization?
2. **Setup:** Matrix Sensing and Matrix Factorization.
3. **Experiment 1:** Sample complexity and condition number phase diagram.
4. **Experiment 2:** Factor-rank and initialization phase diagram.
5. **Experiment 3:** Gradient/update spectral diagnostics.
6. **Takeaway:** Muon seems useful in some spectrally structured regimes, but can degrade under insufficient samples, noise, or unstable factorization geometry.
