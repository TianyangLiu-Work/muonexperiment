# E11 Optimizer-Invariance Audit

This generated audit separates optimizer-intrinsic evidence from effects that depend on update scale, task family, or trajectory state. It is meant to answer whether the current results justify a statement like "Muon is more stable" across tasks.

## Short Answer

The current evidence supports a robust optimizer-intrinsic claim for **update-spectrum shaping**, but it does **not** support the broad claim that Muon is generally more stable in the state geometry.

Raw trajectories make Muon look smoother: all-task normalized rank-plane speed has Muon/Adam ratio 0.6352 with CI [0.5584, 0.7226], and normalized condition speed has ratio 0.5324 with CI [0.4542, 0.6241]. However, after matching global relative update size, those all-task ratios become 1.749 for rank-plane speed and 1.562 for condition speed. In Matrix Sensing under equal-update control, the same ratios are 6.096 and 6.096, both strongly above 1.

Therefore, "Muon is smoother/more stable" is not a reliable optimizer-level conclusion. A safer statement is: **Muon reliably flattens the update spectrum; the induced state-trajectory smoothness is task- and scale-dependent.**

## Update-Spectrum Invariant

| mode         | metric          |   ratio_muon_over_adam | ratio_ci95     | ci_above_one   |
|:-------------|:----------------|-----------------------:|:---------------|:---------------|
| raw          | nrUpdate        |                  1.977 | [1.875, 2.085] | yes            |
| raw          | stUpdate        |                  4.856 | [4.637, 5.085] | yes            |
| raw          | nrUpdateFrac    |                  1.907 | [1.812, 2.006] | yes            |
| raw          | stUpdateFrac    |                  4.538 | [4.428, 4.651] | yes            |
| raw          | update_flatness |                  2.377 | [2.226, 2.538] | yes            |
| equal_update | nrUpdate        |                  2.033 | [1.921, 2.151] | yes            |
| equal_update | stUpdate        |                  5.215 | [5.008, 5.43]  | yes            |
| equal_update | nrUpdateFrac    |                  1.959 | [1.853, 2.07]  | yes            |
| equal_update | stUpdateFrac    |                  4.862 | [4.749, 4.978] | yes            |
| equal_update | update_flatness |                  2.473 | [2.3, 2.66]    | yes            |

These diagnostics are robust to the equal-update control because rescaling an update does not change normalized singular-value geometry. This is the strongest current cross-task optimizer signature.

## Volatility / Stability Check

| mode         | metric                           | problem_family           |   n_pairs |   muon_lower_pairs |   ratio_muon_over_adam | ratio_ci95        | ci_below_one   |
|:-------------|:---------------------------------|:-------------------------|----------:|-------------------:|-----------------------:|:------------------|:---------------|
| raw          | norm_rank_plane_mean_speed       | All                      |       165 |                137 |                0.6352  | [0.5584, 0.7226]  | yes            |
| raw          | norm_rank_plane_mean_speed       | MatrixFactorizationInput |        75 |                 75 |                0.3778  | [0.3379, 0.4224]  | yes            |
| raw          | norm_rank_plane_mean_speed       | MatrixSensing            |        75 |                 48 |                1.072   | [0.8785, 1.307]   | no             |
| raw          | norm_rank_plane_mean_speed       | SmallMLPDigits           |        15 |                 14 |                0.6247  | [0.5096, 0.7659]  | yes            |
| raw          | norm_condition_mean_speed        | All                      |       165 |                131 |                0.5324  | [0.4542, 0.6241]  | yes            |
| raw          | norm_condition_mean_speed        | MatrixFactorizationInput |        75 |                 75 |                0.2421  | [0.2082, 0.2817]  | yes            |
| raw          | norm_condition_mean_speed        | MatrixSensing            |        75 |                 48 |                1.072   | [0.8785, 1.307]   | no             |
| raw          | norm_condition_mean_speed        | SmallMLPDigits           |        15 |                  8 |                0.8279  | [0.6707, 1.022]   | no             |
| raw          | condition_score_std_speed        | All                      |       165 |                135 |                0.2204  | [0.172, 0.2824]   | yes            |
| raw          | condition_score_std_speed        | MatrixFactorizationInput |        75 |                 74 |                0.2445  | [0.207, 0.2888]   | yes            |
| raw          | condition_score_std_speed        | MatrixSensing            |        75 |                 50 |                0.1532  | [0.09293, 0.2527] | yes            |
| raw          | condition_score_std_speed        | SmallMLPDigits           |        15 |                 11 |                0.8074  | [0.667, 0.9774]   | yes            |
| raw          | loss_mean_rel_speed              | All                      |       165 |                164 |                0.1663  | [0.1332, 0.2076]  | yes            |
| raw          | loss_mean_rel_speed              | MatrixFactorizationInput |        75 |                 74 |                0.08061 | [0.057, 0.114]    | yes            |
| raw          | loss_mean_rel_speed              | MatrixSensing            |        75 |                 75 |                0.4508  | [0.3881, 0.5236]  | yes            |
| raw          | loss_mean_rel_speed              | SmallMLPDigits           |        15 |                 15 |                0.04235 | [0.0372, 0.04821] | yes            |
| raw          | per_update_rank_plane_mean_speed | All                      |       165 |                 45 |                1.663   | [1.458, 1.897]    | no             |
| raw          | per_update_rank_plane_mean_speed | MatrixFactorizationInput |        75 |                 40 |                0.9038  | [0.8, 1.021]      | no             |
| raw          | per_update_rank_plane_mean_speed | MatrixSensing            |        75 |                  0 |                3.222   | [2.821, 3.68]     | no             |
| raw          | per_update_rank_plane_mean_speed | SmallMLPDigits           |        15 |                  5 |                1.285   | [0.7539, 2.189]   | no             |
| raw          | per_update_condition_mean_speed  | All                      |       165 |                 71 |                1.379   | [1.175, 1.617]    | no             |
| raw          | per_update_condition_mean_speed  | MatrixFactorizationInput |        75 |                 66 |                0.58    | [0.5057, 0.6652]  | yes            |
| raw          | per_update_condition_mean_speed  | MatrixSensing            |        75 |                  0 |                3.222   | [2.821, 3.68]     | no             |
| raw          | per_update_condition_mean_speed  | SmallMLPDigits           |        15 |                  5 |                1.502   | [0.8443, 2.671]   | no             |
| equal_update | norm_rank_plane_mean_speed       | All                      |       165 |                 81 |                1.749   | [1.434, 2.132]    | no             |
| equal_update | norm_rank_plane_mean_speed       | MatrixFactorizationInput |        75 |                 74 |                0.5619  | [0.5156, 0.6124]  | yes            |
| equal_update | norm_rank_plane_mean_speed       | MatrixSensing            |        75 |                  0 |                6.096   | [5.106, 7.277]    | no             |
| equal_update | norm_rank_plane_mean_speed       | SmallMLPDigits           |        15 |                  7 |                0.9922  | [0.9188, 1.071]   | no             |
| equal_update | norm_condition_mean_speed        | All                      |       165 |                 79 |                1.562   | [1.258, 1.941]    | no             |
| equal_update | norm_condition_mean_speed        | MatrixFactorizationInput |        75 |                 72 |                0.4386  | [0.3895, 0.494]   | yes            |
| equal_update | norm_condition_mean_speed        | MatrixSensing            |        75 |                  0 |                6.096   | [5.106, 7.277]    | no             |
| equal_update | norm_condition_mean_speed        | SmallMLPDigits           |        15 |                  7 |                0.9911  | [0.9192, 1.069]   | no             |
| equal_update | condition_score_std_speed        | All                      |       165 |                 98 |                0.9507  | [0.8187, 1.104]   | no             |
| equal_update | condition_score_std_speed        | MatrixFactorizationInput |        75 |                 69 |                0.501   | [0.4376, 0.5736]  | yes            |
| equal_update | condition_score_std_speed        | MatrixSensing            |        75 |                 21 |                1.799   | [1.443, 2.243]    | no             |
| equal_update | condition_score_std_speed        | SmallMLPDigits           |        15 |                  8 |                0.9641  | [0.8849, 1.05]    | no             |
| equal_update | loss_mean_rel_speed              | All                      |       165 |                 92 |                0.5583  | [0.4978, 0.6262]  | yes            |
| equal_update | loss_mean_rel_speed              | MatrixFactorizationInput |        75 |                 69 |                0.327   | [0.2783, 0.3841]  | yes            |
| equal_update | loss_mean_rel_speed              | MatrixSensing            |        75 |                  8 |                1.034   | [1.018, 1.05]     | no             |
| equal_update | loss_mean_rel_speed              | SmallMLPDigits           |        15 |                 15 |                0.3721  | [0.2989, 0.4632]  | yes            |
| equal_update | per_update_rank_plane_mean_speed | All                      |       165 |                 79 |                1.752   | [1.438, 2.136]    | no             |
| equal_update | per_update_rank_plane_mean_speed | MatrixFactorizationInput |        75 |                 73 |                0.5633  | [0.5164, 0.6145]  | yes            |
| equal_update | per_update_rank_plane_mean_speed | MatrixSensing            |        75 |                  0 |                6.101   | [5.116, 7.275]    | no             |
| equal_update | per_update_rank_plane_mean_speed | SmallMLPDigits           |        15 |                  6 |                0.998   | [0.9188, 1.084]   | no             |
| equal_update | per_update_condition_mean_speed  | All                      |       165 |                 79 |                1.567   | [1.262, 1.946]    | no             |
| equal_update | per_update_condition_mean_speed  | MatrixFactorizationInput |        75 |                 72 |                0.4407  | [0.3911, 0.4967]  | yes            |
| equal_update | per_update_condition_mean_speed  | MatrixSensing            |        75 |                  0 |                6.101   | [5.116, 7.275]    | no             |
| equal_update | per_update_condition_mean_speed  | SmallMLPDigits           |        15 |                  7 |                0.9971  | [0.92, 1.081]     | no             |

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
