# E11 Boundary Predictor

This generated analysis tests whether a pre-specified set of local features predicts when Muon has larger one-step positive descent proxy `<G,D>` than Adam in the target-update sweep.

## Setup

- Data source: `results/e11_target_update_sweep/paired_step_metrics.csv` plus matched step diagnostics.
- Target: `update_grad_inner_muon_higher`.
- Main evaluation: leave-one-base-setting-out logistic regression.
- Feature sets are fixed in the script before fitting: family-only, state-only, update-spectrum-only, and state-plus-update-spectrum.

## Result

Best leave-setting-out mean balanced accuracy is `0.6039` from `state_plus_update_spectrum` when undefined degenerate settings are skipped. With undefined balanced accuracy filled by chance, the best mean is `0.5866` with bootstrap CI `[0.5, 0.7446]` from `state_plus_update_spectrum`. This is an exploratory baseline, not yet a publishable predictive law.

## Aggregate Leave-Setting-Out Metrics

| feature_set                |   mean_balanced_accuracy |   mean_auc |   mean_brier |   mean_baseline_brier |
|:---------------------------|-------------------------:|-----------:|-------------:|----------------------:|
| family_only                |                   0.5    |     0.5    |       0.2048 |               0.09385 |
| state_only                 |                   0.5    |     0.3108 |       0.2989 |               0.09385 |
| state_plus_update_spectrum |                   0.6039 |     0.7526 |       0.2628 |               0.09385 |
| update_spectrum_only       |                   0.5199 |     0.4026 |       0.1084 |               0.09385 |

## Leave-Setting-Out Uncertainty

| feature_set                |   held_out_settings |   mean_balanced_accuracy_chance_filled |   balanced_accuracy_ci95_low |   balanced_accuracy_ci95_high | balanced_accuracy_ci95_above_chance   |   mean_brier_improvement_over_base_rate |   brier_improvement_ci95_low |   brier_improvement_ci95_high | brier_improvement_ci95_above_zero   |
|:---------------------------|--------------------:|---------------------------------------:|-----------------------------:|------------------------------:|:--------------------------------------|----------------------------------------:|-----------------------------:|------------------------------:|:------------------------------------|
| family_only                |                   6 |                                 0.5    |                       0.5    |                        0.5    | no                                    |                                -0.1109  |                     -0.2468  |                    -0.003668  | no                                  |
| state_only                 |                   6 |                                 0.5    |                       0.5    |                        0.5    | no                                    |                                -0.2051  |                     -0.5218  |                    -0.005105  | no                                  |
| state_plus_update_spectrum |                   6 |                                 0.5866 |                       0.5    |                        0.7446 | no                                    |                                -0.169   |                     -0.4211  |                     0.0005075 | no                                  |
| update_spectrum_only       |                   6 |                                 0.5166 |                       0.4914 |                        0.5583 | no                                    |                                -0.01458 |                     -0.02758 |                    -0.00265   | no                                  |

## Per-Setting Results For One-Step First-Order Wins

| feature_set                | held_out_setting           |   test_rows |   test_positive_rate | auc    | balanced_accuracy   |   brier |   baseline_brier |
|:---------------------------|:---------------------------|------------:|---------------------:|:-------|:--------------------|--------:|-----------------:|
| family_only                | MF input kappa=1e+02       |         300 |              0.03667 | 0.5    | 0.5                 | 0.03648 |          0.03532 |
| family_only                | MF input kappa=1e+05       |         300 |              0.03333 | 0.5    | 0.5                 | 0.03406 |          0.03222 |
| family_only                | Matrix sensing kappa=1e+02 |         150 |              0.8533  | 0.5    | 0.5                 | 0.1339  |          0.1252  |
| family_only                | Matrix sensing kappa=1e+05 |         150 |              0.86    | 0.5    | 0.5                 | 0.1276  |          0.1204  |
| family_only                | Small MLP digits hidden=16 |         300 |              0.5     | 0.5    | 0.5                 | 0.4848  |          0.25    |
| family_only                | Small MLP digits hidden=64 |         300 |              0       | n/a    | n/a                 | 0.4119  |          0       |
| state_only                 | MF input kappa=1e+02       |         300 |              0.03667 | 0.1611 | 0.5                 | 0.03762 |          0.03532 |
| state_only                 | MF input kappa=1e+05       |         300 |              0.03333 | 0.1338 | 0.5                 | 0.0365  |          0.03222 |
| state_only                 | Matrix sensing kappa=1e+02 |         150 |              0.8533  | 0.446  | 0.5                 | 0.1371  |          0.1252  |
| state_only                 | Matrix sensing kappa=1e+05 |         150 |              0.86    | 0.4817 | 0.5                 | 0.1259  |          0.1204  |
| state_only                 | Small MLP digits hidden=16 |         300 |              0.5     | 0.3314 | 0.5                 | 0.4997  |          0.25    |
| state_only                 | Small MLP digits hidden=64 |         300 |              0       | n/a    | n/a                 | 0.9567  |          0       |
| update_spectrum_only       | MF input kappa=1e+02       |         300 |              0.03667 | 0.2199 | 0.5                 | 0.04163 |          0.03532 |
| update_spectrum_only       | MF input kappa=1e+05       |         300 |              0.03333 | 0.8093 | 0.4828              | 0.04867 |          0.03222 |
| update_spectrum_only       | Matrix sensing kappa=1e+02 |         150 |              0.8533  | 0.1456 | 0.5                 | 0.1413  |          0.1252  |
| update_spectrum_only       | Matrix sensing kappa=1e+05 |         150 |              0.86    | 0.1554 | 0.5                 | 0.1361  |          0.1204  |
| update_spectrum_only       | Small MLP digits hidden=16 |         300 |              0.5     | 0.6828 | 0.6167              | 0.2406  |          0.25    |
| update_spectrum_only       | Small MLP digits hidden=64 |         300 |              0       | n/a    | n/a                 | 0.0422  |          0       |
| state_plus_update_spectrum | MF input kappa=1e+02       |         300 |              0.03667 | 0.9726 | 0.5455              | 0.0257  |          0.03532 |
| state_plus_update_spectrum | MF input kappa=1e+05       |         300 |              0.03333 | 0.9945 | 0.9741              | 0.03501 |          0.03222 |
| state_plus_update_spectrum | Matrix sensing kappa=1e+02 |         150 |              0.8533  | 0.6783 | 0.5                 | 0.1399  |          0.1252  |
| state_plus_update_spectrum | Matrix sensing kappa=1e+05 |         150 |              0.86    | 0.6264 | 0.5                 | 0.1197  |          0.1204  |
| state_plus_update_spectrum | Small MLP digits hidden=16 |         300 |              0.5     | 0.4914 | 0.5                 | 0.4992  |          0.25    |
| state_plus_update_spectrum | Small MLP digits hidden=64 |         300 |              0       | n/a    | n/a                 | 0.7575  |          0       |

## Interpretation

The current data can support a descriptive boundary map more strongly than a predictive boundary law. If a simple feature set fails leave-setting-out, the paper should keep the boundary claim descriptive. If the best feature set performs well, the next step is to pre-register that rule and test it on a new neural benchmark or new problem family.

## Sources

- [boundary predictor rows](../results/e11_boundary_predictor/boundary_predictor_rows.csv)
- [boundary predictor summary](../results/e11_boundary_predictor/boundary_predictor_summary.csv)
- [target-update paired steps](../results/e11_target_update_sweep/paired_step_metrics.csv)
