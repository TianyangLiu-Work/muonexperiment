# E11 Optimizer-Invariance Audit

This generated audit separates optimizer-intrinsic evidence from effects that depend on update scale, task family, or trajectory state. It is meant to answer whether the current results justify a statement like "Muon is more stable" across tasks.

## Short Answer

The current evidence indicates a robust optimizer-intrinsic pattern for **update-spectrum shaping**, but it does **not** support the broad claim that Muon is generally more stable in the state geometry.

Raw trajectories make Muon look smoother: all-task normalized rank-plane speed has Muon/Adam ratio 0.9134 with CI [0.8276, 1.008], and normalized condition speed has ratio 0.7201 with CI [0.6382, 0.8126]. However, after matching global relative update size, those all-task ratios become 1.072 for rank-plane speed and 0.916 for condition speed. In Matrix Sensing under equal-update control, the same ratios are 1.177 and 1.177, both strongly above 1.

Therefore, "Muon is smoother/more stable" is not a reliable optimizer-level conclusion. A safer statement is: **Muon reliably flattens the update spectrum; the induced state-trajectory smoothness is task- and scale-dependent.**

## Update-Spectrum Invariant

| mode         | metric          |   ratio_muon_over_adam | ratio_ci95     | ci_above_one   |
|:-------------|:----------------|-----------------------:|:---------------|:---------------|
| raw          | nrUpdate        |                  1.985 | [1.879, 2.096] | yes            |
| raw          | stUpdate        |                  4.495 | [4.298, 4.702] | yes            |
| raw          | nrUpdateFrac    |                  1.924 | [1.823, 2.031] | yes            |
| raw          | stUpdateFrac    |                  4.212 | [4.128, 4.298] | yes            |
| raw          | update_flatness |                  2.179 | [2.06, 2.305]  | yes            |
| equal_update | nrUpdate        |                  2.017 | [1.905, 2.136] | yes            |
| equal_update | stUpdate        |                  4.765 | [4.582, 4.955] | yes            |
| equal_update | nrUpdateFrac    |                  1.953 | [1.845, 2.068] | yes            |
| equal_update | stUpdateFrac    |                  4.451 | [4.378, 4.525] | yes            |
| equal_update | update_flatness |                  2.268 | [2.131, 2.413] | yes            |

These diagnostics are robust to the equal-update control because rescaling an update does not change normalized singular-value geometry. This is the strongest current cross-task optimizer signature.

## Volatility / Stability Check

| mode         | metric                           | problem_family           |   n_pairs |   muon_lower_pairs |   ratio_muon_over_adam | ratio_ci95        | ci_below_one   |
|:-------------|:---------------------------------|:-------------------------|----------:|-------------------:|-----------------------:|:------------------|:---------------|
| raw          | norm_rank_plane_mean_speed       | All                      |       165 |                 95 |                0.9134  | [0.8276, 1.008]   | no             |
| raw          | norm_rank_plane_mean_speed       | MatrixFactorizationInput |        75 |                 56 |                0.6981  | [0.5993, 0.8133]  | yes            |
| raw          | norm_rank_plane_mean_speed       | MatrixSensing            |        75 |                 25 |                1.292   | [1.16, 1.44]      | no             |
| raw          | norm_rank_plane_mean_speed       | SmallMLPDigits           |        15 |                 14 |                0.6175  | [0.505, 0.7549]   | yes            |
| raw          | norm_condition_mean_speed        | All                      |       165 |                 99 |                0.7201  | [0.6382, 0.8126]  | yes            |
| raw          | norm_condition_mean_speed        | MatrixFactorizationInput |        75 |                 66 |                0.3919  | [0.3391, 0.4528]  | yes            |
| raw          | norm_condition_mean_speed        | MatrixSensing            |        75 |                 25 |                1.292   | [1.16, 1.44]      | no             |
| raw          | norm_condition_mean_speed        | SmallMLPDigits           |        15 |                  8 |                0.8115  | [0.6546, 1.006]   | no             |
| raw          | condition_score_std_speed        | All                      |       165 |                 93 |                0.7307  | [0.6424, 0.8311]  | yes            |
| raw          | condition_score_std_speed        | MatrixFactorizationInput |        75 |                 64 |                0.4076  | [0.3389, 0.4903]  | yes            |
| raw          | condition_score_std_speed        | MatrixSensing            |        75 |                 18 |                1.286   | [1.156, 1.432]    | no             |
| raw          | condition_score_std_speed        | SmallMLPDigits           |        15 |                 11 |                0.8004  | [0.6594, 0.9716]  | yes            |
| raw          | loss_mean_rel_speed              | All                      |       165 |                153 |                0.3736  | [0.3213, 0.4344]  | yes            |
| raw          | loss_mean_rel_speed              | MatrixFactorizationInput |        75 |                 64 |                0.6733  | [0.6164, 0.7355]  | yes            |
| raw          | loss_mean_rel_speed              | MatrixSensing            |        75 |                 74 |                0.3205  | [0.2668, 0.385]   | yes            |
| raw          | loss_mean_rel_speed              | SmallMLPDigits           |        15 |                 15 |                0.04227 | [0.0371, 0.04817] | yes            |
| raw          | per_update_rank_plane_mean_speed | All                      |       165 |                 34 |                2.121   | [1.857, 2.422]    | no             |
| raw          | per_update_rank_plane_mean_speed | MatrixFactorizationInput |        75 |                 23 |                1.652   | [1.36, 2.007]     | no             |
| raw          | per_update_rank_plane_mean_speed | MatrixSensing            |        75 |                  6 |                3.021   | [2.562, 3.562]    | no             |
| raw          | per_update_rank_plane_mean_speed | SmallMLPDigits           |        15 |                  5 |                1.263   | [0.7351, 2.168]   | no             |
| raw          | per_update_condition_mean_speed  | All                      |       165 |                 60 |                1.645   | [1.421, 1.905]    | no             |
| raw          | per_update_condition_mean_speed  | MatrixFactorizationInput |        75 |                 49 |                0.9179  | [0.7725, 1.091]   | no             |
| raw          | per_update_condition_mean_speed  | MatrixSensing            |        75 |                  6 |                3.021   | [2.562, 3.562]    | no             |
| raw          | per_update_condition_mean_speed  | SmallMLPDigits           |        15 |                  5 |                1.461   | [0.8114, 2.632]   | no             |
| equal_update | norm_rank_plane_mean_speed       | All                      |       165 |                 77 |                1.072   | [1.009, 1.138]    | no             |
| equal_update | norm_rank_plane_mean_speed       | MatrixFactorizationInput |        75 |                 41 |                0.9904  | [0.9039, 1.085]   | no             |
| equal_update | norm_rank_plane_mean_speed       | MatrixSensing            |        75 |                 29 |                1.177   | [1.073, 1.29]     | no             |
| equal_update | norm_rank_plane_mean_speed       | SmallMLPDigits           |        15 |                  7 |                0.9972  | [0.9129, 1.089]   | no             |
| equal_update | norm_condition_mean_speed        | All                      |       165 |                 91 |                0.916   | [0.849, 0.9884]   | yes            |
| equal_update | norm_condition_mean_speed        | MatrixFactorizationInput |        75 |                 56 |                0.7026  | [0.6275, 0.7868]  | yes            |
| equal_update | norm_condition_mean_speed        | MatrixSensing            |        75 |                 29 |                1.177   | [1.073, 1.29]     | no             |
| equal_update | norm_condition_mean_speed        | SmallMLPDigits           |        15 |                  6 |                0.9854  | [0.9048, 1.073]   | no             |
| equal_update | condition_score_std_speed        | All                      |       165 |                 82 |                0.9866  | [0.915, 1.064]    | no             |
| equal_update | condition_score_std_speed        | MatrixFactorizationInput |        75 |                 49 |                0.805   | [0.7199, 0.9002]  | yes            |
| equal_update | condition_score_std_speed        | MatrixSensing            |        75 |                 25 |                1.215   | [1.095, 1.349]    | no             |
| equal_update | condition_score_std_speed        | SmallMLPDigits           |        15 |                  8 |                0.9612  | [0.8731, 1.058]   | no             |
| equal_update | loss_mean_rel_speed              | All                      |       165 |                127 |                0.795   | [0.7552, 0.837]   | yes            |
| equal_update | loss_mean_rel_speed              | MatrixFactorizationInput |        75 |                 52 |                0.853   | [0.804, 0.9051]   | yes            |
| equal_update | loss_mean_rel_speed              | MatrixSensing            |        75 |                 60 |                0.8532  | [0.8135, 0.8948]  | yes            |
| equal_update | loss_mean_rel_speed              | SmallMLPDigits           |        15 |                 15 |                0.3927  | [0.3172, 0.4862]  | yes            |
| equal_update | per_update_rank_plane_mean_speed | All                      |       165 |                 75 |                1.073   | [1.009, 1.14]     | no             |
| equal_update | per_update_rank_plane_mean_speed | MatrixFactorizationInput |        75 |                 41 |                0.9897  | [0.9022, 1.086]   | no             |
| equal_update | per_update_rank_plane_mean_speed | MatrixSensing            |        75 |                 28 |                1.179   | [1.073, 1.295]    | no             |
| equal_update | per_update_rank_plane_mean_speed | SmallMLPDigits           |        15 |                  6 |                1.002   | [0.9112, 1.101]   | no             |
| equal_update | per_update_condition_mean_speed  | All                      |       165 |                 91 |                0.9184  | [0.8507, 0.9914]  | yes            |
| equal_update | per_update_condition_mean_speed  | MatrixFactorizationInput |        75 |                 56 |                0.7046  | [0.6293, 0.789]   | yes            |
| equal_update | per_update_condition_mean_speed  | MatrixSensing            |        75 |                 28 |                1.179   | [1.073, 1.295]    | no             |
| equal_update | per_update_condition_mean_speed  | SmallMLPDigits           |        15 |                  7 |                0.9926  | [0.906, 1.088]    | no             |

## Interpretation

1. Raw lower volatility is partly confounded by Muon's smaller effective update scale in several settings.
2. Equal-update control removes that explanation and shows that state-geometry speed can flip direction by task family.
3. The update-spectrum difference is optimizer-intrinsic; the state-trajectory difference is an interaction between optimizer, task, and scale.
4. This audit argues against using "more stable" as the main claim unless stability is defined narrowly as a specific measured quantity in a specific controlled setting.

## Recommended Wording

Use:

> Muon has a robust update-spectrum signature: at matched update size, its update matrices have larger effective/stable rank and flatter spectra than Adam. The downstream trajectory smoothness is not invariant across task families.

Avoid:

> Muon is generally more stable than Adam.

## Sources

- [raw volatility summary](../results/e11/volatility_summary.csv)
- [equal-update volatility summary](../results/e11_equal_update/volatility_summary.csv)
- [raw update-spectrum summary](../results/e11/update_spectrum_summary.csv)
- [equal-update update-spectrum summary](../results/e11_equal_update/update_spectrum_summary.csv)
- [raw volatility figure](../figures/e11/volatility_robustness.png)
- [equal-update volatility figure](../figures/e11_equal_update/volatility_robustness.png)
- [equal-update update-spectrum figure](../figures/e11_equal_update/update_spectrum_robustness.png)
