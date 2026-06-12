# E11 Claim Validity Audit

This audit separates what the current experiments support from what remains a vulnerability. It is generated from the current result CSVs so that the numerical claims are reproducible.

## Claim Status

| claim                                                                                | status                                                  | evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | main_loophole                                                                                                                                                                        |
|:-------------------------------------------------------------------------------------|:--------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Muon has a distinct update-spectrum geometry.                                        | supported                                               | Equal-update nrUpdate ratio=2.033 CI=[1.921,2.151], stUpdate ratio=5.215 CI=[5.008,5.43].                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | This is partly by construction for exact polar Muon, so it is a mechanism fact, not a surprise performance result.                                                                   |
| Muon is globally better than Adam on short-horizon progress.                         | not supported                                           | All-task first-order ratio=0.5342 CI=[0.5063,0.5636]; MF ratio=0.3606, Matrix Sensing ratio=1.209.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | The result is task-dependent; aggregate ratios hide opposite family-level behavior.                                                                                                  |
| One-step progress is explained by gradient-update alignment.                         | supported for these short steps                         | All equal-update Spearman(delta_loss, <G,D>)=0.9206 CI=[0.9145,0.9263], within-factor-2=0.9584.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | This validates a first-order local diagnostic, not long-horizon optimization or generalization.                                                                                      |
| A simple cross-task spectral rule predicts when Muon wins.                           | not yet supported                                       | Leave-family-out tests fail and feature-overlap diagnostics show the held-out families often leave the training support.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Current families are too separated spectrally; a successful in-sample rule may be extrapolating.                                                                                     |
| SmallMLP has a real width-dependent transition.                                      | supported in the current digits setup                   | samples=1024 hidden=8 first-order ratio=1.208 CI=[1.18,1.236], hidden=128 ratio=0.5455 CI=[0.5171,0.5754]. With per-layer update sizes fixed, hidden=16 becomes near-neutral (0.9408 CI=[0.9309,0.9507]), while hidden=64 remains Adam-favorable (0.6882 CI=[0.6761,0.7006]).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | The wide-network failure is robust in this setup, but the narrow-network Muon advantage depends on target scale and is not purely directional.                                       |
| Layerwise Adam/Muon hybrids isolate the bottleneck.                                  | not supported                                           | At hidden=128/samples=1024, AdamFirstMuonSecond layer-2 update-budget fraction=0.003052, efficiency ratio=1.072, inner ratio=0.1486.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Layerwise hybrid changes layer update allocation, so it is not a clean intervention on only the update direction.                                                                    |
| The main conclusions are just a single-learning-rate artifact.                       | weakened by lr and target-norm sweeps, not fully closed | Representative best-lr sweep: raw all-setting final-loss ratio=1.087 CI=[0.8197,1.443], equal-update ratio=0.564 CI=[0.4573,0.6956]; direction-only target sweep gives MatrixSensing kappa=1e2 first-order ratio=1.072 but MLP hidden=64 ratio=0.6047; nrUpdate ratios remain >1 in raw (1.956), equal-update (1.959), and target-norm (1.954).                                                                                                                                                                                                                                                                                                                                                                                                                               | The sweeps are representative and partly oracle-style; per-layer allocation is only controlled for SmallMLP, not for all tasks or broad architectures.                               |
| Flat/polar spectral allocation is universally better than GD allocation.             | not supported; norm-geometry dependent                  | Within-layer probe: flat_polar/GD first-order ratio under Frobenius budget=0.6071 CI=[0.5846,0.6304], but under operator-norm budget=1.689 CI=[1.614,1.768].                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | The probe fixes gradient singular vectors and is one-step artificial; it does not model natural singular-vector trajectories.                                                        |
| Adam and Muon stay in the same singular-vector geometry.                             | not generally supported                                 | Final matched gradient subspace overlap stays high for MF (0.9562), but drops strongly for Matrix Sensing (0.1714) and MLP hidden=64 (0.741). Swap probe mean signed other/own first-order ratio is 0.5053 overall, and -0.1429 for Matrix Sensing. Natural update-vector swap gives all-budget signed other/own ratio=0.4925 CI=[0.4423,0.5427] for first-order progress and observed delta-loss ratio=0.4643 CI=[0.4159,0.5126] across five target scales.                                                                                                                                                                                                                                                                                                                  | The natural-update swap is still one-step and artificial; some setting/budget/eval-state cells favor the other update, so the effect is not a simple universal specialization claim. |
| One-step own-update advantage implies own-optimizer trajectory continuation is best. | not supported                                           | Trajectory switch probe: switched/own total-decrease ratio=5.872 CI=[4.686,7.058]. Switching from Adam checkpoints increases final loss on average (loss ratio=3.221), while switching from Muon checkpoints to Adam often lowers final loss (loss ratio=0.9125). Reset-control gives switched-fresh/own-fresh final-loss ratios 3.142 from Adam checkpoints and 0.9125 from Muon checkpoints; Adam own-fresh/own-preserved ratio=1.145. Horizon sweep final-loss ratios are horizon dependent: all-source horizon-10 ratio=2.659, Muon-source horizon-30 ratio=0.7774, Adam-source horizon-30 ratio=1.786. After a continuation-LR sweep at horizon 30, best switched/own final-loss ratios are 2.304 overall, 2.565 from Adam checkpoints, and 2.043 from Muon checkpoints. | Reset-control, horizon sweep, and a small continuation-LR sweep reduce specific confounds but still do not constitute retuned full training.                                         |

## Evidence Tables

### Equal-Update Spectrum

| metric       | problem_family   |   n_pairs |   muon_higher_pairs |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   |
|:-------------|:-----------------|----------:|--------------------:|-------------------------------:|-----------------:|------------------:|:-----------------------|
| nrUpdate     | All              |       165 |                 165 |                          2.033 |            1.921 |             2.151 | yes                    |
| stUpdate     | All              |       165 |                 165 |                          5.215 |            5.008 |             5.43  | yes                    |
| nrUpdateFrac | All              |       165 |                 165 |                          1.959 |            1.853 |             2.07  | yes                    |
| stUpdateFrac | All              |       165 |                 165 |                          4.862 |            4.749 |             4.978 | yes                    |

### Equal-Update First-Order Progress

| metric            | problem_family           |   n_pairs |   muon_higher_pairs |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   |   mean_delta_muon_minus_adam |
|:------------------|:-------------------------|----------:|--------------------:|-------------------------------:|-----------------:|------------------:|:-----------------------|-----------------------------:|
| delta_loss        | All                      |      1275 |                 420 |                         0.5181 |           0.4905 |            0.5473 | no                     |                   -0.001503  |
| delta_loss        | MatrixFactorizationInput |       750 |                 106 |                         0.3979 |           0.3677 |            0.4307 | no                     |                   -1.331e-06 |
| delta_loss        | MatrixSensing            |       375 |                 314 |                         0.9383 |           0.8857 |            0.994  | no                     |                    0.0004987 |
| delta_loss        | SmallMLPDigits           |       150 |                   0 |                         0.4849 |           0.454  |            0.5178 | no                     |                   -0.01402   |
| update_grad_inner | All                      |      1275 |                 337 |                         0.5342 |           0.5063 |            0.5636 | no                     |                   -0.001362  |
| update_grad_inner | MatrixFactorizationInput |       750 |                  61 |                         0.3606 |           0.3361 |            0.3869 | no                     |                   -3.699e-06 |
| update_grad_inner | MatrixSensing            |       375 |                 276 |                         1.209  |           1.148  |            1.274  | yes                    |                    0.0005758 |
| update_grad_inner | SmallMLPDigits           |       150 |                   0 |                         0.4942 |           0.4639 |            0.5266 | no                     |                   -0.013     |

### First-Order Calibration

| group   |   points |   positive_points |   spearman_delta_vs_first_order |   spearman_ci95_low |   spearman_ci95_high |   within_factor_2 |
|:--------|---------:|------------------:|--------------------------------:|--------------------:|---------------------:|------------------:|
| All     |     2550 |              2406 |                          0.9206 |              0.9145 |               0.9263 |            0.9584 |
| Adam    |     1275 |              1156 |                          0.9266 |              0.9184 |               0.934  |            0.955  |
| Muon    |     1275 |              1250 |                          0.9153 |              0.9059 |               0.9238 |            0.9616 |

### Leave-Family-Out Generalization

| target               | model         | held_out_family          |   train_pairs |   test_pairs |   test_positive_rate | auc    | balanced_accuracy   |   brier |   baseline_brier |
|:---------------------|:--------------|:-------------------------|--------------:|-------------:|---------------------:|:-------|:--------------------|--------:|-----------------:|
| muon_first_order_win | spectrum_only | MatrixFactorizationInput |           525 |          750 |              0.08133 | 0.1749 | 0.5                 |  0.0814 |          0.07472 |
| muon_first_order_win | spectrum_only | MatrixSensing            |           900 |          375 |              0.736   | 0.5    | 0.5                 |  0.736  |          0.1943  |
| muon_first_order_win | spectrum_only | SmallMLPDigits           |          1125 |          150 |              0       | n/a    | n/a                 |  1      |          0       |

### Feature-Overlap Failure

| held_out_family          |   test_pairs |   mean_feature_outside_fraction |   frac_pairs_with_any_feature_outside |   median_min_standardized_distance |   p90_min_standardized_distance |
|:-------------------------|-------------:|--------------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|
| MatrixFactorizationInput |          750 |                          0.4229 |                                     1 |                              14.66 |                           45.61 |
| MatrixSensing            |          375 |                          0.7276 |                                     1 |                             237.1  |                          239.5  |
| SmallMLPDigits           |          150 |                          0.3446 |                                     1 |                              26.5  |                           29.3  |

### SmallMLP Width Transition

|   hidden_dim |   num_samples |   n_pairs |   muon_win_rate |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   | ratio_ci95_below_one   |
|-------------:|--------------:|----------:|----------------:|-------------------------------:|-----------------:|------------------:|:-----------------------|:-----------------------|
|            8 |          1024 |       150 |          0.9467 |                         1.208  |           1.18   |            1.236  | yes                    | no                     |
|           16 |          1024 |       150 |          0.6067 |                         1.005  |           0.9867 |            1.023  | no                     | no                     |
|           32 |          1024 |       150 |          0      |                         0.7196 |           0.6925 |            0.7478 | no                     | yes                    |
|           64 |          1024 |       150 |          0      |                         0.4942 |           0.4639 |            0.5266 | no                     | yes                    |
|          128 |          1024 |       150 |          0      |                         0.5455 |           0.5171 |            0.5754 | no                     | yes                    |

### Hybrid Update Allocation

|   hidden_dim |   num_samples |   layer | algo                |   fro_sq_fraction |   fro_sq_fraction_ratio_over_adam |   efficiency_ratio_over_adam |   inner_ratio_over_adam |
|-------------:|--------------:|--------:|:--------------------|------------------:|----------------------------------:|-----------------------------:|------------------------:|
|            8 |          1024 |       2 | AdamFirstMuonSecond |          0.04109  |                           0.2585  |                       0.8664 |                  0.4483 |
|           16 |          1024 |       2 | AdamFirstMuonSecond |          0.02405  |                           0.1483  |                       0.9539 |                  0.3769 |
|           64 |          1024 |       2 | AdamFirstMuonSecond |          0.006389 |                           0.03935 |                       1.026  |                  0.2066 |
|          128 |          1024 |       2 | AdamFirstMuonSecond |          0.003052 |                           0.01874 |                       1.072  |                  0.1486 |

### Hyperparameter Sweep Robustness

| mode         | problem_family   | base_setting   |   n_seed_pairs |   final_loss_ratio_muon_over_adam |   final_loss_ratio_ci95_low |   final_loss_ratio_ci95_high |   total_decrease_ratio_muon_over_adam |   total_decrease_ratio_ci95_low |   total_decrease_ratio_ci95_high |
|:-------------|:-----------------|:---------------|---------------:|----------------------------------:|----------------------------:|-----------------------------:|--------------------------------------:|--------------------------------:|---------------------------------:|
| equal_update | All              | All            |             30 |                             0.564 |                      0.4573 |                       0.6956 |                                0.8115 |                          0.6582 |                           1      |
| raw          | All              | All            |             30 |                             1.087 |                      0.8197 |                       1.443  |                                0.4633 |                          0.3019 |                           0.7112 |

| mode         | metric       | problem_family   |   n_pairs |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   |
|:-------------|:-------------|:-----------------|----------:|-------------------------------:|-----------------:|------------------:|:-----------------------|
| raw          | nrUpdate     | All              |       180 |                          1.956 |            1.866 |             2.051 | yes                    |
| raw          | stUpdate     | All              |       180 |                          5.357 |            5.107 |             5.619 | yes                    |
| raw          | nrUpdateFrac | All              |       180 |                          1.821 |            1.743 |             1.902 | yes                    |
| raw          | stUpdateFrac | All              |       180 |                          4.658 |            4.557 |             4.761 | yes                    |
| equal_update | nrUpdate     | All              |       180 |                          1.959 |            1.866 |             2.056 | yes                    |
| equal_update | stUpdate     | All              |       180 |                          5.468 |            5.227 |             5.719 | yes                    |
| equal_update | nrUpdateFrac | All              |       180 |                          1.821 |            1.74  |             1.907 | yes                    |
| equal_update | stUpdateFrac | All              |       180 |                          4.723 |            4.619 |             4.828 | yes                    |

### Target-Update-Norm Direction Sweep

| problem_family           | base_setting               |   n_pairs |   muon_win_rate |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   | ratio_ci95_below_one   |
|:-------------------------|:---------------------------|----------:|----------------:|-------------------------------:|-----------------:|------------------:|:-----------------------|:-----------------------|
| MatrixFactorizationInput | MF input kappa=1e+02       |       300 |         0.03667 |                         0.4143 |           0.3868 |            0.4438 | no                     | yes                    |
| MatrixFactorizationInput | MF input kappa=1e+05       |       300 |         0.03333 |                         0.3266 |           0.3035 |            0.3515 | no                     | yes                    |
| MatrixSensing            | Matrix sensing kappa=1e+02 |       150 |         0.8533  |                         1.072  |           1.038  |            1.107  | yes                    | no                     |
| MatrixSensing            | Matrix sensing kappa=1e+05 |       150 |         0.86    |                         1.113  |           1.039  |            1.192  | yes                    | no                     |
| SmallMLPDigits           | Small MLP digits hidden=16 |       300 |         0.5     |                         1.009  |           0.9975 |            1.021  | no                     | no                     |
| SmallMLPDigits           | Small MLP digits hidden=64 |       300 |         0       |                         0.6047 |           0.5818 |            0.6285 | no                     | yes                    |

| problem_family           | base_setting               |   n_seed_pairs |   final_loss_ratio_muon_over_adam |   final_loss_ratio_ci95_low |   final_loss_ratio_ci95_high |   total_decrease_ratio_muon_over_adam |   total_decrease_ratio_ci95_low |   total_decrease_ratio_ci95_high |   mean_best_target_adam |   mean_best_target_muon |
|:-------------------------|:---------------------------|---------------:|----------------------------------:|----------------------------:|-----------------------------:|--------------------------------------:|--------------------------------:|---------------------------------:|------------------------:|------------------------:|
| MatrixFactorizationInput | MF input kappa=1e+02       |              5 |                            0.7288 |                      0.4345 |                       1.222  |                                1.053  |                          0.9636 |                           1.15   |                  0.03   |                    0.03 |
| MatrixFactorizationInput | MF input kappa=1e+05       |              5 |                            1.364  |                      0.8601 |                       2.164  |                                0.9304 |                          0.826  |                           1.048  |                  0.03   |                    0.03 |
| MatrixSensing            | Matrix sensing kappa=1e+02 |              5 |                            0.3379 |                      0.3287 |                       0.3473 |                                1.226  |                          1.21   |                           1.241  |                  0.1    |                    0.3  |
| MatrixSensing            | Matrix sensing kappa=1e+05 |              5 |                            0.3197 |                      0.303  |                       0.3373 |                                1.208  |                          1.196  |                           1.221  |                  0.1    |                    0.3  |
| SmallMLPDigits           | Small MLP digits hidden=16 |              5 |                            1.032  |                      1.021  |                       1.043  |                                0.736  |                          0.6813 |                           0.7952 |                  0.3    |                    0.3  |
| SmallMLPDigits           | Small MLP digits hidden=64 |              5 |                            1.395  |                      1.354  |                       1.437  |                                0.2159 |                          0.2046 |                           0.2278 |                  0.3    |                    0.3  |
| All                      | All                        |             30 |                            0.7326 |                      0.5745 |                       0.9341 |                                0.783  |                          0.6226 |                           0.9848 |                  0.1433 |                    0.21 |

| metric       | problem_family   |   n_pairs |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   |
|:-------------|:-----------------|----------:|-------------------------------:|-----------------:|------------------:|:-----------------------|
| nrUpdate     | All              |       180 |                          1.954 |            1.862 |             2.051 | yes                    |
| stUpdate     | All              |       180 |                          5.482 |            5.241 |             5.734 | yes                    |
| nrUpdateFrac | All              |       180 |                          1.817 |            1.736 |             1.902 | yes                    |
| stUpdateFrac | All              |       180 |                          4.731 |            4.624 |             4.839 | yes                    |

### SmallMLP Per-Layer Update Control

|   hidden_dim |   n_pairs |   muon_win_rate |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   | ratio_ci95_below_one   |
|-------------:|----------:|----------------:|-------------------------------:|-----------------:|------------------:|:-----------------------|:-----------------------|
|           16 |       250 |           0.252 |                         0.9408 |           0.9309 |            0.9507 | no                     | yes                    |
|           64 |       250 |           0     |                         0.6882 |           0.6761 |            0.7006 | no                     | yes                    |

|   hidden_dim |   target_layer_relative_update_norm |   n_pairs |   muon_win_rate |   geomean_ratio_muon_over_adam |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   | ratio_ci95_below_one   |
|-------------:|------------------------------------:|----------:|----------------:|-------------------------------:|-----------------:|------------------:|:-----------------------|:-----------------------|
|           16 |                               0.001 |        50 |            0.3  |                         0.9559 |           0.9366 |            0.9756 | no                     | yes                    |
|           16 |                               0.003 |        50 |            0.32 |                         0.9574 |           0.9374 |            0.9777 | no                     | yes                    |
|           16 |                               0.01  |        50 |            0.3  |                         0.9604 |           0.9422 |            0.9789 | no                     | yes                    |
|           16 |                               0.03  |        50 |            0.2  |                         0.9391 |           0.9174 |            0.9613 | no                     | yes                    |
|           16 |                               0.1   |        50 |            0.14 |                         0.8928 |           0.8674 |            0.919  | no                     | yes                    |
|           64 |                               0.001 |        50 |            0    |                         0.7381 |           0.7276 |            0.7487 | no                     | yes                    |
|           64 |                               0.003 |        50 |            0    |                         0.7358 |           0.7253 |            0.7464 | no                     | yes                    |
|           64 |                               0.01  |        50 |            0    |                         0.7259 |           0.7149 |            0.7371 | no                     | yes                    |
|           64 |                               0.03  |        50 |            0    |                         0.6831 |           0.6657 |            0.7011 | no                     | yes                    |
|           64 |                               0.1   |        50 |            0    |                         0.5733 |           0.5414 |            0.6071 | no                     | yes                    |

### Within-Layer Spectral Allocation Probe

| budget   | comparison                  | metric            |   n_pairs |   geomean_ratio |   ratio_ci95_low |   ratio_ci95_high | ratio_ci95_above_one   | ratio_ci95_below_one   |
|:---------|:----------------------------|:------------------|----------:|----------------:|-----------------:|------------------:|:-----------------------|:-----------------------|
| fro      | flat_polar_over_gd_spectrum | update_grad_inner |       120 |          0.6071 |           0.5846 |            0.6304 | no                     | yes                    |
| op       | flat_polar_over_gd_spectrum | update_grad_inner |       120 |          1.689  |           1.614  |            1.768  | yes                    | no                     |

### Singular-Vector Trajectory Diagnostic

| problem_family           | base_setting               |   step |   initial_grad_overlap |   mean_grad_overlap |   grad_overlap_drop |   mean_param_overlap |   mean_adam_loss |   mean_muon_loss |
|:-------------------------|:---------------------------|-------:|-----------------------:|--------------------:|--------------------:|---------------------:|-----------------:|-----------------:|
| MatrixSensing            | Matrix sensing kappa=1e+02 |      5 |                      1 |              0.1714 |             0.8286  |               0.3094 |        0.002402  |        0.001741  |
| MatrixFactorizationInput | MF input kappa=1e+02       |     10 |                      1 |              0.9562 |             0.04375 |               0.9574 |        1.663e-05 |        0.0001406 |
| SmallMLPDigits           | Small MLP digits hidden=16 |     10 |                      1 |              0.7482 |             0.2518  |               0.5082 |        2.139     |        2.289     |
| SmallMLPDigits           | Small MLP digits hidden=64 |     10 |                      1 |              0.741  |             0.259   |               0.408  |        1.747     |        2.283     |

### Singular-Vector Swap Probe

| problem_family           | base_setting               | eval_algo   |   n_pairs |   other_positive_pairs |   other_better_pairs |   mean_signed_other_over_own |   signed_ratio_ci95_low |   signed_ratio_ci95_high |
|:-------------------------|:---------------------------|:------------|----------:|-----------------------:|---------------------:|-----------------------------:|------------------------:|-------------------------:|
| MatrixFactorizationInput | MF input kappa=1e+02       | All         |        40 |                     40 |                    0 |                       0.7128 |                  0.6343 |                   0.7913 |
| MatrixSensing            | Matrix sensing kappa=1e+02 | All         |        40 |                     10 |                    0 |                      -0.1429 |                 -0.3624 |                   0.0767 |
| SmallMLPDigits           | Small MLP digits hidden=16 | All         |        40 |                     40 |                    0 |                       0.7518 |                  0.6993 |                   0.8043 |
| SmallMLPDigits           | Small MLP digits hidden=64 | All         |        40 |                     40 |                    0 |                       0.6993 |                  0.6392 |                   0.7593 |

### Natural Update-Vector Swap Probe

Reference target `1e-3`, first-order metric:

| problem_family           | base_setting               |   target_layer_relative_norm | budget   | eval_algo   |   n_pairs |   other_positive_pairs |   other_better_pairs |   mean_signed_other_over_own |   signed_ratio_ci95_low |   signed_ratio_ci95_high |
|:-------------------------|:---------------------------|-----------------------------:|:---------|:------------|----------:|-----------------------:|---------------------:|-----------------------------:|------------------------:|-------------------------:|
| MatrixFactorizationInput | MF input kappa=1e+02       |                        0.001 | fro      | All         |        40 |                     35 |                   16 |                       0.8018 |                  0.5625 |                 1.041    |
| MatrixFactorizationInput | MF input kappa=1e+02       |                        0.001 | op       | All         |        40 |                     35 |                   15 |                       1.228  |                  0.639  |                 1.818    |
| MatrixFactorizationInput | MF input kappa=1e+05       |                        0.001 | fro      | All         |        40 |                     34 |                   13 |                       0.7601 |                  0.5332 |                 0.987    |
| MatrixFactorizationInput | MF input kappa=1e+05       |                        0.001 | op       | All         |        40 |                     34 |                   11 |                       1.005  |                  0.5647 |                 1.446    |
| MatrixSensing            | Matrix sensing kappa=1e+02 |                        0.001 | fro      | All         |        40 |                     15 |                    5 |                      -0.2018 |                 -0.4953 |                 0.09169  |
| MatrixSensing            | Matrix sensing kappa=1e+02 |                        0.001 | op       | All         |        40 |                     15 |                    5 |                      -0.4781 |                 -0.9703 |                 0.01415  |
| MatrixSensing            | Matrix sensing kappa=1e+05 |                        0.001 | fro      | All         |        40 |                     15 |                    5 |                      -0.2205 |                 -0.5132 |                 0.07226  |
| MatrixSensing            | Matrix sensing kappa=1e+05 |                        0.001 | op       | All         |        40 |                     15 |                    5 |                      -0.5109 |                 -1.015  |                -0.006458 |
| SmallMLPDigits           | Small MLP digits hidden=16 |                        0.001 | fro      | All         |        40 |                     40 |                    5 |                       0.7792 |                  0.7283 |                 0.8301   |
| SmallMLPDigits           | Small MLP digits hidden=16 |                        0.001 | op       | All         |        40 |                     40 |                   19 |                       0.9704 |                  0.7703 |                 1.171    |
| SmallMLPDigits           | Small MLP digits hidden=64 |                        0.001 | fro      | All         |        40 |                     40 |                   14 |                       0.8283 |                  0.727  |                 0.9296   |
| SmallMLPDigits           | Small MLP digits hidden=64 |                        0.001 | op       | All         |        40 |                     40 |                   19 |                       0.9478 |                  0.7371 |                 1.158    |

Observed delta-loss target sweep:

| group_type        | target_layer_relative_norm   | budget   | eval_algo   |   n_pairs |   other_positive_pairs |   other_better_pairs |   mean_signed_other_over_own |   signed_ratio_ci95_low |   signed_ratio_ci95_high |
|:------------------|:-----------------------------|:---------|:------------|----------:|-----------------------:|---------------------:|-----------------------------:|------------------------:|-------------------------:|
| target_budget_all | 0.0001                       | fro      | All         |       240 |                    179 |                   58 |                       0.4571 |                  0.3496 |                   0.5645 |
| target_budget_all | 0.0001                       | op       | All         |       240 |                    179 |                   74 |                       0.5254 |                  0.3283 |                   0.7224 |
| target_budget_all | 0.0003                       | fro      | All         |       240 |                    179 |                   58 |                       0.4556 |                  0.3485 |                   0.5627 |
| target_budget_all | 0.0003                       | op       | All         |       240 |                    179 |                   74 |                       0.5218 |                  0.326  |                   0.7177 |
| target_budget_all | 0.001                        | fro      | All         |       240 |                    179 |                   58 |                       0.4507 |                  0.3446 |                   0.5568 |
| target_budget_all | 0.001                        | op       | All         |       240 |                    179 |                   74 |                       0.51   |                  0.3178 |                   0.7023 |
| target_budget_all | 0.003                        | fro      | All         |       240 |                    179 |                   58 |                       0.4392 |                  0.3349 |                   0.5435 |
| target_budget_all | 0.003                        | op       | All         |       240 |                    179 |                   75 |                       0.4795 |                  0.295  |                   0.6639 |
| target_budget_all | 0.01                         | fro      | All         |       240 |                    178 |                   58 |                       0.4115 |                  0.3084 |                   0.5147 |
| target_budget_all | 0.01                         | op       | All         |       240 |                    178 |                   73 |                       0.3917 |                  0.218  |                   0.5654 |
| target_all        | 0.0001                       | All      | All         |       480 |                    358 |                  132 |                       0.4912 |                  0.3791 |                   0.6034 |
| target_all        | 0.0003                       | All      | All         |       480 |                    358 |                  132 |                       0.4887 |                  0.3772 |                   0.6003 |
| target_all        | 0.001                        | All      | All         |       480 |                    358 |                  132 |                       0.4804 |                  0.3707 |                   0.5901 |
| target_all        | 0.003                        | All      | All         |       480 |                    358 |                  133 |                       0.4593 |                  0.3535 |                   0.5652 |
| target_all        | 0.01                         | All      | All         |       480 |                    356 |                  131 |                       0.4016 |                  0.3007 |                   0.5025 |
| all               | All                          | All      | All         |      2400 |                   1788 |                  660 |                       0.4643 |                  0.4159 |                   0.5126 |

At reference target `1e-3`, observed Frobenius-budget signed other/own ratio=0.4507 CI=[0.3446,0.5568], operator-budget ratio=0.51 CI=[0.3178,0.7023].

### Optimizer Switch Probe

| group_type                     | problem_family           | base_setting               | source_algo   | checkpoint_step   |   n_pairs |   switched_better_pairs |   mean_signed_switched_over_own |   signed_ratio_ci95_low |   signed_ratio_ci95_high |
|:-------------------------------|:-------------------------|:---------------------------|:--------------|:------------------|----------:|------------------------:|--------------------------------:|------------------------:|-------------------------:|
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+02       | Adam          | All               |        15 |                       0 |                         0.2513  |                 0.1684  |                  0.3343  |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+02       | Muon          | All               |        15 |                      15 |                        10.51    |                 9.937   |                 11.09    |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+05       | Adam          | All               |        15 |                       0 |                         0.2392  |                 0.157   |                  0.3214  |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+05       | Muon          | All               |        15 |                      15 |                        12.99    |                11.96    |                 14.03    |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+02 | Adam          | All               |        15 |                      15 |                         3.646   |                 1.682   |                  5.609   |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+02 | Muon          | All               |        15 |                       0 |                         0.7457  |                 0.6275  |                  0.864   |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+05 | Adam          | All               |        15 |                      15 |                         2.379   |                 1.456   |                  3.303   |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+05 | Muon          | All               |        15 |                       0 |                         0.6894  |                 0.5478  |                  0.831   |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=16 | Adam          | All               |        15 |                       0 |                         0.1119  |                 0.0959  |                  0.128   |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=16 | Muon          | All               |        15 |                      15 |                        11.88    |                10.87    |                 12.88    |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=64 | Adam          | All               |        15 |                       0 |                         0.05751 |                 0.04719 |                  0.06782 |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=64 | Muon          | All               |        15 |                      15 |                        26.97    |                25.64    |                 28.29    |
| source_all                     | All                      | All                        | Adam          | All               |        90 |                      30 |                         1.114   |                 0.6586  |                  1.57    |
| source_all                     | All                      | All                        | Muon          | All               |        90 |                      60 |                        10.63    |                 8.76    |                 12.5     |
| all                            | All                      | All                        | All           | All               |       180 |                      90 |                         5.872   |                 4.686   |                  7.058   |

### Optimizer Switch Reset-Control Probe

| group_type                     | problem_family           | base_setting               | source_algo   | comparison                    |   n_pairs |   numerator_better_pairs |   mean_signed_ratio |   signed_ratio_ci95_low |   signed_ratio_ci95_high |
|:-------------------------------|:-------------------------|:---------------------------|:--------------|:------------------------------|----------:|-------------------------:|--------------------:|------------------------:|-------------------------:|
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+02       | Adam          | switched_fresh_over_own_fresh |        15 |                        0 |             6.779   |                 5.684   |                  7.874   |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+02       | Adam          | own_fresh_over_own_preserved  |        15 |                        5 |             1.014   |                 0.9813  |                  1.047   |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+02       | Muon          | switched_fresh_over_own_fresh |        15 |                       15 |             0.1218  |                 0.1096  |                  0.134   |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+02       | Muon          | own_fresh_over_own_preserved  |        15 |                        0 |             1       |                 1       |                  1       |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+05       | Adam          | switched_fresh_over_own_fresh |        15 |                        0 |             8.96    |                 7.512   |                 10.41    |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+05       | Adam          | own_fresh_over_own_preserved  |        15 |                        5 |             1.021   |                 0.9493  |                  1.093   |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+05       | Muon          | switched_fresh_over_own_fresh |        15 |                       15 |             0.08175 |                 0.07072 |                  0.09278 |
| setting_source_all_checkpoints | MatrixFactorizationInput | MF input kappa=1e+05       | Muon          | own_fresh_over_own_preserved  |        15 |                        0 |             1       |                 1       |                  1       |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+02 | Adam          | switched_fresh_over_own_fresh |        15 |                       15 |             0.3931  |                 0.2395  |                  0.5468  |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+02 | Adam          | own_fresh_over_own_preserved  |        15 |                        0 |             1.393   |                 1.224   |                  1.561   |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+02 | Muon          | switched_fresh_over_own_fresh |        15 |                        0 |             1.636   |                 1.507   |                  1.766   |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+02 | Muon          | own_fresh_over_own_preserved  |        15 |                        0 |             1       |                 1       |                  1       |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+05 | Adam          | switched_fresh_over_own_fresh |        15 |                       15 |             0.3462  |                 0.2138  |                  0.4786  |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+05 | Adam          | own_fresh_over_own_preserved  |        15 |                        0 |             1.458   |                 1.277   |                  1.64    |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+05 | Muon          | switched_fresh_over_own_fresh |        15 |                        0 |             1.899   |                 1.741   |                  2.057   |
| setting_source_all_checkpoints | MatrixSensing            | Matrix sensing kappa=1e+05 | Muon          | own_fresh_over_own_preserved  |        15 |                        0 |             1       |                 1       |                  1       |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=16 | Adam          | switched_fresh_over_own_fresh |        15 |                        0 |             1.071   |                 1.061   |                  1.081   |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=16 | Adam          | own_fresh_over_own_preserved  |        15 |                        9 |             0.9955  |                 0.9928  |                  0.9982  |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=16 | Muon          | switched_fresh_over_own_fresh |        15 |                       15 |             0.9393  |                 0.9326  |                  0.946   |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=16 | Muon          | own_fresh_over_own_preserved  |        15 |                        0 |             1       |                 1       |                  1       |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=64 | Adam          | switched_fresh_over_own_fresh |        15 |                        0 |             1.305   |                 1.285   |                  1.324   |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=64 | Adam          | own_fresh_over_own_preserved  |        15 |                       10 |             0.9884  |                 0.9832  |                  0.9935  |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=64 | Muon          | switched_fresh_over_own_fresh |        15 |                       15 |             0.797   |                 0.7776  |                  0.8164  |
| setting_source_all_checkpoints | SmallMLPDigits           | Small MLP digits hidden=64 | Muon          | own_fresh_over_own_preserved  |        15 |                        0 |             1       |                 1       |                  1       |
| source_all                     | All                      | All                        | Adam          | switched_fresh_over_own_fresh |        90 |                       30 |             3.142   |                 2.373   |                  3.912   |
| source_all                     | All                      | All                        | Adam          | own_fresh_over_own_preserved  |        90 |                       29 |             1.145   |                 1.086   |                  1.204   |
| source_all                     | All                      | All                        | Muon          | switched_fresh_over_own_fresh |        90 |                       60 |             0.9125  |                 0.7661  |                  1.059   |
| source_all                     | All                      | All                        | Muon          | own_fresh_over_own_preserved  |        90 |                        0 |             1       |                 1       |                  1       |
| all                            | All                      | All                        | All           | switched_fresh_over_own_fresh |       180 |                       90 |             2.027   |                 1.604   |                  2.451   |
| all                            | All                      | All                        | All           | own_fresh_over_own_preserved  |       180 |                       29 |             1.073   |                 1.041   |                  1.104   |

### Optimizer Switch Horizon Sweep

| group_type                  | source_algo   | horizon   |   n_pairs |   switched_better_pairs |   mean_signed_switched_over_own |   signed_ratio_ci95_low |   signed_ratio_ci95_high |
|:----------------------------|:--------------|:----------|----------:|------------------------:|--------------------------------:|------------------------:|-------------------------:|
| source_horizon_all_settings | Adam          | 1         |        30 |                       7 |                          1.136  |                  1.055  |                   1.218  |
| source_horizon_all_settings | Adam          | 3         |        30 |                      10 |                          1.332  |                  0.9724 |                   1.691  |
| source_horizon_all_settings | Adam          | 10        |        30 |                      10 |                          2.14   |                  1.363  |                   2.917  |
| source_horizon_all_settings | Adam          | 30        |        30 |                      10 |                          1.786  |                  1.236  |                   2.337  |
| source_horizon_all_settings | Muon          | 1         |        30 |                      20 |                          1.197  |                  1.057  |                   1.336  |
| source_horizon_all_settings | Muon          | 3         |        30 |                      20 |                          1.495  |                  1.078  |                   1.912  |
| source_horizon_all_settings | Muon          | 10        |        30 |                      20 |                          3.179  |                  1.729  |                   4.629  |
| source_horizon_all_settings | Muon          | 30        |        30 |                      19 |                          0.7774 |                  0.6202 |                   0.9345 |
| horizon_all                 | All           | 1         |        60 |                      27 |                          1.166  |                  1.086  |                   1.247  |
| horizon_all                 | All           | 3         |        60 |                      30 |                          1.414  |                  1.14   |                   1.687  |
| horizon_all                 | All           | 10        |        60 |                      30 |                          2.659  |                  1.833  |                   3.485  |
| horizon_all                 | All           | 30        |        60 |                      29 |                          1.282  |                  0.9701 |                   1.594  |
| all                         | All           | All       |       240 |                     116 |                          1.63   |                  1.387  |                   1.873  |

### Optimizer Switch Continuation-LR Sweep

| group_type     | problem_family           | base_setting               | source_algo   |   n_pairs |   switched_better_pairs |   mean_switched_over_own |   ratio_ci95_low |   ratio_ci95_high |   mean_best_lr_own |   mean_best_lr_switched |
|:---------------|:-------------------------|:---------------------------|:--------------|----------:|------------------------:|-------------------------:|-----------------:|------------------:|-------------------:|------------------------:|
| setting_source | MatrixFactorizationInput | MF input kappa=1e+02       | Adam          |         5 |                       5 |                   0.5107 |          0.3309  |            0.6904 |            0.01    |                 0.01    |
| setting_source | MatrixFactorizationInput | MF input kappa=1e+02       | Muon          |         5 |                       2 |                   1.063  |          0.549   |            1.576  |            0.014   |                 0.01    |
| setting_source | MatrixFactorizationInput | MF input kappa=1e+05       | Adam          |         5 |                       2 |                   1.13   |          0.6849  |            1.575  |            0.01    |                 0.01    |
| setting_source | MatrixFactorizationInput | MF input kappa=1e+05       | Muon          |         5 |                       2 |                   0.9949 |          0.5135  |            1.476  |            0.014   |                 0.01    |
| setting_source | MatrixSensing            | Matrix sensing kappa=1e+02 | Adam          |         5 |                       5 |                   0.2922 |          0.1889  |            0.3955 |            0.003   |                 0.003   |
| setting_source | MatrixSensing            | Matrix sensing kappa=1e+02 | Muon          |         5 |                       0 |                   5.596  |          4.153   |            7.038  |            0.003   |                 0.003   |
| setting_source | MatrixSensing            | Matrix sensing kappa=1e+05 | Adam          |         5 |                       5 |                   0.2355 |          0.2163  |            0.2547 |            0.003   |                 0.003   |
| setting_source | MatrixSensing            | Matrix sensing kappa=1e+05 | Muon          |         5 |                       0 |                   4.31   |          2.994   |            5.626  |            0.003   |                 0.003   |
| setting_source | SmallMLPDigits           | Small MLP digits hidden=16 | Adam          |         5 |                       0 |                   4.159  |          3.045   |            5.274  |            0.03    |                 0.03    |
| setting_source | SmallMLPDigits           | Small MLP digits hidden=16 | Muon          |         5 |                       5 |                   0.19   |          0.133   |            0.2471 |            0.03    |                 0.03    |
| setting_source | SmallMLPDigits           | Small MLP digits hidden=64 | Adam          |         5 |                       0 |                   9.063  |          5.631   |           12.49   |            0.03    |                 0.03    |
| setting_source | SmallMLPDigits           | Small MLP digits hidden=64 | Muon          |         5 |                       5 |                   0.1038 |          0.07711 |            0.1305 |            0.03    |                 0.03    |
| source_all     | All                      | All                        | Adam          |        30 |                      17 |                   2.565  |          1.335   |            3.795  |            0.01433 |                 0.01433 |
| source_all     | All                      | All                        | Muon          |        30 |                      14 |                   2.043  |          1.239   |            2.847  |            0.01567 |                 0.01433 |
| all            | All                      | All                        | All           |        60 |                      31 |                   2.304  |          1.573   |            3.035  |            0.015   |                 0.01433 |

## Current Valid Formulation

The defensible formulation is: **Muon is an update-spectrum shaping optimizer; its polar-style update consistently changes the singular-value geometry of the update matrices, but whether that geometry improves one-step progress depends on task and layer conditions.**

The current data do **not** justify saying that Muon is generally more stable, generally better, or that a single rank statistic explains all task families.

The intervention sequence behind this formulation is summarized in [E11 mechanism ladder](e11_mechanism_ladder.md).

## Main Loopholes To Close

1. Hyperparameters and target update norms are not exhaustively searched; the representative sweeps reduce but do not eliminate this concern.
2. The MLP result is a small sklearn digits benchmark, not a broad neural-network result.
3. The cross-task predictor currently fails under leave-family-out evaluation, partly because spectral supports do not overlap enough.
4. Layerwise hybrids are not clean causal interventions because they alter layer update-budget allocation; the per-layer control partially addresses this for SmallMLP but not for other tasks.
5. ExactMuon's update-spectrum flatness is partly algorithmic by construction; the spectral-allocation probe shows that this construction is favorable under an operator-norm budget but not under a Frobenius budget.
6. Singular-vector trajectory divergence is supported by polar-vector and natural-update-vector one-step swap probes.
7. The optimizer-switch, reset-control, horizon-sweep, and continuation-LR probes do not support a simple own-optimizer trajectory-specialization story; practical continuation effects depend on source optimizer, task family, continuation horizon, and tuning.
8. Most evidence is short-horizon one-step or 5-10 step behavior; final training performance remains a separate question.

## Next Experiments With Highest Value

1. Extend the representative hyperparameter and target-update-norm sweeps to a larger grid.
2. Add matched-support task settings where MF, Matrix Sensing, and MLP occupy overlapping `nr(G)/r`, `st(A)`, and update-alignment ranges.
3. Broaden the natural update-vector swap probe across more settings and target scales.
4. Extend the MLP benchmark to at least one external image dataset and one deeper architecture.
5. Report all main claims as paired ratios with confidence intervals, not as visual separations only.
