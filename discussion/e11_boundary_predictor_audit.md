# E11 Boundary Predictor Audit

This generated audit explains why the current boundary predictor is not yet a publishable predictive law. It is stricter than `e11_boundary_predictor.md`: the goal here is to identify failure modes and the minimum standard for a future held-out test.

## Readiness Summary

| criterion               | status                         | evidence                                                                                                                                                                                                            | paper_implication                                                                      |
|:------------------------|:-------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| In-sample fit           | passes only as exploratory fit | Best in-sample balanced accuracy is 0.8999; best leave-setting-out is 0.6039.                                                                                                                                       | Do not use in-sample performance as predictive evidence.                               |
| Held-out generalization | not paper-ready                | Best feature set `state_plus_update_spectrum` has mean leave-setting-out balanced accuracy 0.6039 when degenerate settings are skipped. Chance-filled best `state_plus_update_spectrum` is 0.5866 CI=[0.5, 0.7446]. | Keep the boundary map descriptive unless a new held-out benchmark validates a rule.    |
| Class balance           | problematic                    | Some held-out settings have zero or near-zero positive rate, making AUC/balanced accuracy unstable or undefined.                                                                                                    | A publishable predictor needs settings with non-degenerate positive/negative examples. |
| Causal interpretation   | unsupported                    | Feature sets mix state geometry, update spectrum, and family labels.                                                                                                                                                | Treat predictor features as descriptive correlates, not mechanism proof.               |

## In-Sample Versus Leave-Setting-Out Gap

| feature_set                |   in_sample |   leave_setting_out |   in_sample_minus_leave_setting_out |
|:---------------------------|------------:|--------------------:|------------------------------------:|
| family_only                |      0.7802 |              0.5    |                              0.2802 |
| state_only                 |      0.7663 |              0.5    |                              0.2663 |
| state_plus_update_spectrum |      0.8999 |              0.6039 |                              0.296  |
| update_spectrum_only       |      0.8943 |              0.5199 |                              0.3744 |

## Leave-Setting-Out Uncertainty

| feature_set                |   held_out_settings |   mean_balanced_accuracy_chance_filled |   balanced_accuracy_ci95_low |   balanced_accuracy_ci95_high | balanced_accuracy_ci95_above_chance   |   mean_brier_improvement_over_base_rate |   brier_improvement_ci95_low |   brier_improvement_ci95_high | brier_improvement_ci95_above_zero   |
|:---------------------------|--------------------:|---------------------------------------:|-----------------------------:|------------------------------:|:--------------------------------------|----------------------------------------:|-----------------------------:|------------------------------:|:------------------------------------|
| family_only                |                   6 |                                 0.5    |                       0.5    |                        0.5    | no                                    |                                -0.1109  |                     -0.2468  |                    -0.003668  | no                                  |
| state_only                 |                   6 |                                 0.5    |                       0.5    |                        0.5    | no                                    |                                -0.2051  |                     -0.5218  |                    -0.005105  | no                                  |
| state_plus_update_spectrum |                   6 |                                 0.5866 |                       0.5    |                        0.7446 | no                                    |                                -0.169   |                     -0.4211  |                     0.0005075 | no                                  |
| update_spectrum_only       |                   6 |                                 0.5166 |                       0.4914 |                        0.5583 | no                                    |                                -0.01458 |                     -0.02758 |                    -0.00265   | no                                  |

## Held-Out Setting Difficulty

| held_out_setting           |   best_balanced_accuracy |   mean_positive_rate | max_auc   |   min_brier |   baseline_brier | class_balance_issue   |
|:---------------------------|-------------------------:|---------------------:|:----------|------------:|-----------------:|:----------------------|
| MF input kappa=1e+02       |                   0.5455 |              0.03667 | 0.9726    |     0.0257  |          0.03532 | highly_imbalanced     |
| MF input kappa=1e+05       |                   0.9741 |              0.03333 | 0.9945    |     0.03406 |          0.03222 | highly_imbalanced     |
| Matrix sensing kappa=1e+02 |                   0.5    |              0.8533  | 0.6783    |     0.1339  |          0.1252  | moderate              |
| Matrix sensing kappa=1e+05 |                   0.5    |              0.86    | 0.6264    |     0.1197  |          0.1204  | moderate              |
| Small MLP digits hidden=16 |                   0.6167 |              0.5     | 0.6828    |     0.2406  |          0.25    | moderate              |
| Small MLP digits hidden=64 |                   0.5    |              0       | n/a       |     0.0422  |          0       | degenerate            |

## Best Feature-Set Failure Modes

Best leave-setting-out feature set: `state_plus_update_spectrum`.

| held_out_setting           |   test_rows |   test_positive_rate | auc    | balanced_accuracy   |   brier |   baseline_brier | failure_mode      |
|:---------------------------|------------:|---------------------:|:-------|:--------------------|--------:|-----------------:|:------------------|
| MF input kappa=1e+02       |         300 |              0.03667 | 0.9726 | 0.5455              | 0.0257  |          0.03532 | near chance       |
| MF input kappa=1e+05       |         300 |              0.03333 | 0.9945 | 0.9741              | 0.03501 |          0.03222 | partial success   |
| Matrix sensing kappa=1e+02 |         150 |              0.8533  | 0.6783 | 0.5                 | 0.1399  |          0.1252  | near chance       |
| Matrix sensing kappa=1e+05 |         150 |              0.86    | 0.6264 | 0.5                 | 0.1197  |          0.1204  | near chance       |
| Small MLP digits hidden=16 |         300 |              0.5     | 0.4914 | 0.5                 | 0.4992  |          0.25    | near chance       |
| Small MLP digits hidden=64 |         300 |              0       | n/a    | n/a                 | 0.7575  |          0       | degenerate target |

## Target Distribution Behind The Failures

| base_setting               |   rows |   positive_rate |   mean_log10_target |   mean_delta_st_update_frac |   mean_delta_update_flatness |
|:---------------------------|-------:|----------------:|--------------------:|----------------------------:|-----------------------------:|
| MF input kappa=1e+02       |    300 |         0.03667 |              -2.761 |                      0.7561 |                       0.4131 |
| MF input kappa=1e+05       |    300 |         0.03333 |              -2.761 |                      0.7883 |                       0.3153 |
| Matrix sensing kappa=1e+02 |    150 |         0.8533  |              -1.761 |                      0.818  |                       0.7429 |
| Matrix sensing kappa=1e+05 |    150 |         0.86    |              -1.761 |                      0.8261 |                       0.7527 |
| Small MLP digits hidden=16 |    300 |         0.5     |              -1.761 |                      0.7555 |                       0.6472 |
| Small MLP digits hidden=64 |    300 |         0       |              -1.761 |                      0.7726 |                       0.6869 |

## Minimum Standard For The Next Predictor

| requirement                                                           | why                                                                            |
|:----------------------------------------------------------------------|:-------------------------------------------------------------------------------|
| Pre-register predictor features                                       | Avoid choosing features after seeing all task families.                        |
| Use a genuinely new held-out task or architecture                     | Leave-setting-out inside the current six settings is too small and imbalanced. |
| Ensure both Muon-win and Adam-win examples within the held-out domain | Degenerate labels make predictor metrics undefined or misleading.              |
| Report calibration as well as ranking                                 | Several models have poor Brier scores despite reasonable in-sample ranking.    |

## Paper Use

The current result should be described as:

> The boundary map is quantitatively documented but not yet predictive out of sample.

It should not be described as:

> We can predict when Muon wins from local rank/spectrum features.

## Sources

- [boundary predictor summary](../results/e11_boundary_predictor/boundary_predictor_summary.csv)
- [boundary predictor rows](../results/e11_boundary_predictor/boundary_predictor_rows.csv)
- [boundary predictor discussion](e11_boundary_predictor.md)
