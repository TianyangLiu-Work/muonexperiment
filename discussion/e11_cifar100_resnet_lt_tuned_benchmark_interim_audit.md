# E11 CIFAR-100-LT Tuned Benchmark Interim Validation Audit

This generated audit is an interim, no-peeking progress readout for the tuned validation grid. It reads only `validation_tuning` outputs and the selection audit tables. It does not authorize final seed runs, recipe-family selection for incomplete families, or benchmark-level optimizer wording.

Current progress: `17/164` validation settings complete. The current completed-setting leader is `TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0` with few balanced accuracy `0.1121` and all balanced accuracy `0.4151`.

## Claim Boundary Gates

| gate_id                              | status    | evidence                                                                         | allowed_wording                                           | blocked_wording                           |
|:-------------------------------------|:----------|:---------------------------------------------------------------------------------|:----------------------------------------------------------|:------------------------------------------|
| IVA-1-validation-readout-scope       | pass      | reads validation_selection/run_registry.csv and validation_tuning summaries only | interim validation progress readout                       | final performance benchmark result        |
| IVA-2-partial-grid-blocks-selection  | not_ready | 17/164 validation settings complete                                              | completed-setting leaderboard with no selection authority | selection from incomplete recipe families |
| IVA-3-familywise-selection-authority | not_ready | 1/6 recipe families complete                                                     | family complete/partial status                            | cross-family optimizer claim              |
| IVA-4-occupancy-coverage             | not_ready | 17/164 occupancy traces complete                                                 | interim occupancy coverage                                | trajectory state-distribution claim       |
| IVA-5-final-seed-quarantine          | pass      | mirrors TVS-3-final-seed-quarantine                                              | final seeds remain untouched                              | final seed result or retuned final claim  |

## Family Progress

| recipe_family          |   expected_settings |   complete_settings |   complete_fraction | family_status   | best_completed_setting_id                     |   best_completed_few_balanced_accuracy |   best_completed_all_balanced_accuracy | ci_relation_to_current_global_best                   | selection_status           |
|:-----------------------|--------------------:|--------------------:|--------------------:|:----------------|:----------------------------------------------|---------------------------------------:|---------------------------------------:|:-----------------------------------------------------|:---------------------------|
| adamw_cb_loss_tuned    |                  24 |                   0 |              0      | not_started     |                                               |                                        |                                        | not_observed                                         | not_allowed_partial_family |
| adamw_cb_sampler_tuned |                   8 |                   0 |              0      | not_started     |                                               |                                        |                                        | not_observed                                         | not_allowed_partial_family |
| adamw_ce_tuned         |                  12 |                  12 |              1      | complete        | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0        |                                 0.1121 |                                 0.4151 | few_ci_intervals_overlap_or_family_is_global_best    | eligible_family_complete   |
| ns_muon_cb_tuned       |                  36 |                   0 |              0      | not_started     |                                               |                                        |                                        | not_observed                                         | not_allowed_partial_family |
| ns_muon_matrix_tuned   |                  72 |                   0 |              0      | not_started     |                                               |                                        |                                        | not_observed                                         | not_allowed_partial_family |
| sgd_momentum_ce_tuned  |                  12 |                   5 |              0.4167 | partial         | TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm0 |                                 0.0844 |                                 0.3789 | current_global_best_ci_low_above_family_best_ci_high | not_allowed_partial_family |

## Completed-Setting Leaderboard

|   rank |   array_index | setting_id                                      | recipe_family         |   few_balanced_accuracy |   few_balanced_accuracy_ci95_low |   few_balanced_accuracy_ci95_high |   all_balanced_accuracy | selection_allowed    |
|-------:|--------------:|:------------------------------------------------|:----------------------|------------------------:|---------------------------------:|----------------------------------:|------------------------:|:---------------------|
|      1 |            10 | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0          | adamw_ce_tuned        |                 0.1121  |                          0.104   |                           0.1203  |                  0.4151 | no_partial_grid_only |
|      2 |             9 | TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm500        | adamw_ce_tuned        |                 0.1093  |                          0.1015  |                           0.1172  |                  0.4139 | no_partial_grid_only |
|      3 |            11 | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm500        | adamw_ce_tuned        |                 0.1092  |                          0.1019  |                           0.1165  |                  0.4133 | no_partial_grid_only |
|      4 |             8 | TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm0          | adamw_ce_tuned        |                 0.1054  |                          0.0973  |                           0.1135  |                  0.4112 | no_partial_grid_only |
|      5 |             4 | TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm0          | adamw_ce_tuned        |                 0.0866  |                          0.08456 |                           0.08864 |                  0.3669 | no_partial_grid_only |
|      6 |            12 | TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm0   | sgd_momentum_ce_tuned |                 0.0844  |                          0.07847 |                           0.09033 |                  0.3789 | no_partial_grid_only |
|      7 |             6 | TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0          | adamw_ce_tuned        |                 0.08367 |                          0.07954 |                           0.08779 |                  0.3669 | no_partial_grid_only |
|      8 |            13 | TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm500 | sgd_momentum_ce_tuned |                 0.08307 |                          0.07457 |                           0.09156 |                  0.3793 | no_partial_grid_only |
|      9 |             7 | TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm500        | adamw_ce_tuned        |                 0.08253 |                          0.07424 |                           0.09083 |                  0.3587 | no_partial_grid_only |
|     10 |             5 | TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm500        | adamw_ce_tuned        |                 0.08047 |                          0.07858 |                           0.08235 |                  0.3595 | no_partial_grid_only |
|     11 |            16 | TBV-sgd_momentum_ce_tuned-lr0p1-wd5e-4-warm0    | sgd_momentum_ce_tuned |                 0.07093 |                          0.06386 |                           0.07801 |                  0.3991 | no_partial_grid_only |
|     12 |            14 | TBV-sgd_momentum_ce_tuned-lr0p03-wd1e-3-warm0   | sgd_momentum_ce_tuned |                 0.06793 |                          0.06224 |                           0.07363 |                  0.38   | no_partial_grid_only |

## Interim Occupancy Summary

| recipe_family          |   expected_settings |   occupancy_complete_settings |   occupancy_probe_rows |   mean_batch_few_fraction |   mean_tail_probe_loss |   mean_ns_tail_output_drift_sq_ratio_vs_fro | occupancy_claim_status       |
|:-----------------------|--------------------:|------------------------------:|-----------------------:|--------------------------:|-----------------------:|--------------------------------------------:|:-----------------------------|
| adamw_cb_loss_tuned    |                  24 |                             0 |                      0 |                           |                        |                                             | not_ready_full_grid_required |
| adamw_cb_sampler_tuned |                   8 |                             0 |                      0 |                           |                        |                                             | not_ready_full_grid_required |
| adamw_ce_tuned         |                  12 |                            12 |                    360 |                   0.02786 |                  7.225 |                                      0.5321 | family_occupancy_complete    |
| ns_muon_cb_tuned       |                  36 |                             0 |                      0 |                           |                        |                                             | not_ready_full_grid_required |
| ns_muon_matrix_tuned   |                  72 |                             0 |                      0 |                           |                        |                                             | not_ready_full_grid_required |
| sgd_momentum_ce_tuned  |                  12 |                             5 |                    150 |                   0.02786 |                  5.007 |                                      0.5629 | not_ready_full_grid_required |

## Operating Rule

This audit can be cited only as progress accounting and reviewer-facing no-peeking evidence. The final claim seed set `20..29` remains blocked until `TVS-1`, `TVS-2`, and `TVS-5` pass in the selection audit.
