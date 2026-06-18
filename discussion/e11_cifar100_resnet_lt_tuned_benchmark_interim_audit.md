# E11 CIFAR-100-LT Tuned Benchmark Interim Validation Audit

This generated audit is an interim, no-peeking progress readout for the tuned validation grid. It reads only `validation_tuning` outputs and the selection audit tables. It does not authorize final seed runs, recipe-family selection for incomplete families, or benchmark-level optimizer wording.

Current progress: `32/164` validation settings complete. The current completed-setting leader is `TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0` with few balanced accuracy `0.1121` and all balanced accuracy `0.4151`.

## Claim Boundary Gates

| gate_id                              | status    | evidence                                                                         | allowed_wording                                           | blocked_wording                           |
|:-------------------------------------|:----------|:---------------------------------------------------------------------------------|:----------------------------------------------------------|:------------------------------------------|
| IVA-1-validation-readout-scope       | pass      | reads validation_selection/run_registry.csv and validation_tuning summaries only | interim validation progress readout                       | final performance benchmark result        |
| IVA-2-partial-grid-blocks-selection  | not_ready | 32/164 validation settings complete                                              | completed-setting leaderboard with no selection authority | selection from incomplete recipe families |
| IVA-3-familywise-selection-authority | not_ready | 2/6 recipe families complete                                                     | family complete/partial status                            | cross-family optimizer claim              |
| IVA-4-occupancy-coverage             | not_ready | 32/164 occupancy traces complete                                                 | interim occupancy coverage                                | trajectory state-distribution claim       |
| IVA-5-final-seed-quarantine          | pass      | mirrors TVS-3-final-seed-quarantine                                              | final seeds remain untouched                              | final seed result or retuned final claim  |

## Partial Grid Guardrail

| guard_id                                | status    | evidence                                                                                   | claim_authority                                         | blocked_action                                    |
|:----------------------------------------|:----------|:-------------------------------------------------------------------------------------------|:--------------------------------------------------------|:--------------------------------------------------|
| IPG-1-completed-prefix-auditable        | pass      | completed array indices span 0..31; next missing array index 32                            | progress accounting only                                | cherry-picking non-contiguous validation settings |
| IPG-2-family-coverage-incomplete        | not_ready | 3/6 recipe families observed; 2/6 recipe families complete                                 | within-family progress only until all families complete | cross-family optimizer selection claim            |
| IPG-3-occupancy-paired-with-validation  | pass      | 32/32 completed settings have occupancy traces with positive probe rows; min probe rows 30 | paired occupancy progress readout                       | trajectory occupancy claim before full grid       |
| IPG-4-missing-work-blocks-final-readout | not_ready | 132/164 settings still lack validation summaries                                           | no final-performance benchmark result                   | final seed launch or benchmark result wording     |
| IPG-5-final-seed-quarantine-mirrored    | pass      | TVS-3-final-seed-quarantine=pass                                                           | final seeds remain untouched                            | retuned final seed claim                          |

## Family Progress

| recipe_family          |   expected_settings |   complete_settings |   complete_fraction | family_status   | best_completed_setting_id                             |   best_completed_few_balanced_accuracy |   best_completed_all_balanced_accuracy | ci_relation_to_current_global_best                   | selection_status           |
|:-----------------------|--------------------:|--------------------:|--------------------:|:----------------|:------------------------------------------------------|---------------------------------------:|---------------------------------------:|:-----------------------------------------------------|:---------------------------|
| adamw_cb_loss_tuned    |                  24 |                   8 |              0.3333 | partial         | TBV-adamw_cb_loss_tuned-beta0p999-lr3e-4-wd1e-4-warm0 |                                 0.076  |                                 0.3297 | current_global_best_ci_low_above_family_best_ci_high | not_allowed_partial_family |
| adamw_cb_sampler_tuned |                   8 |                   0 |              0      | not_started     |                                                       |                                        |                                        | not_observed                                         | not_allowed_partial_family |
| adamw_ce_tuned         |                  12 |                  12 |              1      | complete        | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0                |                                 0.1121 |                                 0.4151 | few_ci_intervals_overlap_or_family_is_global_best    | eligible_family_complete   |
| ns_muon_cb_tuned       |                  36 |                   0 |              0      | not_started     |                                                       |                                        |                                        | not_observed                                         | not_allowed_partial_family |
| ns_muon_matrix_tuned   |                  72 |                   0 |              0      | not_started     |                                                       |                                        |                                        | not_observed                                         | not_allowed_partial_family |
| sgd_momentum_ce_tuned  |                  12 |                  12 |              1      | complete        | TBV-sgd_momentum_ce_tuned-lr0p3-wd1e-3-warm500        |                                 0.1008 |                                 0.4469 | few_ci_intervals_overlap_or_family_is_global_best    | eligible_family_complete   |

## Completed-Setting Leaderboard

|   rank |   array_index | setting_id                                      | recipe_family         |   few_balanced_accuracy |   few_balanced_accuracy_ci95_low |   few_balanced_accuracy_ci95_high |   all_balanced_accuracy | selection_allowed    |
|-------:|--------------:|:------------------------------------------------|:----------------------|------------------------:|---------------------------------:|----------------------------------:|------------------------:|:---------------------|
|      1 |            10 | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0          | adamw_ce_tuned        |                 0.1121  |                          0.104   |                           0.1203  |                  0.4151 | no_partial_grid_only |
|      2 |             9 | TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm500        | adamw_ce_tuned        |                 0.1093  |                          0.1015  |                           0.1172  |                  0.4139 | no_partial_grid_only |
|      3 |            11 | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm500        | adamw_ce_tuned        |                 0.1092  |                          0.1019  |                           0.1165  |                  0.4133 | no_partial_grid_only |
|      4 |             8 | TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm0          | adamw_ce_tuned        |                 0.1054  |                          0.0973  |                           0.1135  |                  0.4112 | no_partial_grid_only |
|      5 |            23 | TBV-sgd_momentum_ce_tuned-lr0p3-wd1e-3-warm500  | sgd_momentum_ce_tuned |                 0.1008  |                          0.09266 |                           0.1089  |                  0.4469 | no_partial_grid_only |
|      6 |            21 | TBV-sgd_momentum_ce_tuned-lr0p3-wd5e-4-warm500  | sgd_momentum_ce_tuned |                 0.09573 |                          0.08526 |                           0.1062  |                  0.4382 | no_partial_grid_only |
|      7 |            19 | TBV-sgd_momentum_ce_tuned-lr0p1-wd1e-3-warm500  | sgd_momentum_ce_tuned |                 0.08753 |                          0.07676 |                           0.09831 |                  0.4302 | no_partial_grid_only |
|      8 |             4 | TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm0          | adamw_ce_tuned        |                 0.0866  |                          0.08456 |                           0.08864 |                  0.3669 | no_partial_grid_only |
|      9 |            12 | TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm0   | sgd_momentum_ce_tuned |                 0.0844  |                          0.07847 |                           0.09033 |                  0.3789 | no_partial_grid_only |
|     10 |             6 | TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0          | adamw_ce_tuned        |                 0.08367 |                          0.07954 |                           0.08779 |                  0.3669 | no_partial_grid_only |
|     11 |            13 | TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm500 | sgd_momentum_ce_tuned |                 0.08307 |                          0.07457 |                           0.09156 |                  0.3793 | no_partial_grid_only |
|     12 |             7 | TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm500        | adamw_ce_tuned        |                 0.08253 |                          0.07424 |                           0.09083 |                  0.3587 | no_partial_grid_only |

## Interim Occupancy Summary

| recipe_family          |   expected_settings |   occupancy_complete_settings |   occupancy_probe_rows |   mean_batch_few_fraction |   mean_tail_probe_loss |   mean_ns_tail_output_drift_sq_ratio_vs_fro | occupancy_claim_status       |
|:-----------------------|--------------------:|------------------------------:|-----------------------:|--------------------------:|-----------------------:|--------------------------------------------:|:-----------------------------|
| adamw_cb_loss_tuned    |                  24 |                             8 |                    240 |                   0.02786 |                  6.828 |                                      0.4782 | not_ready_full_grid_required |
| adamw_cb_sampler_tuned |                   8 |                             0 |                      0 |                           |                        |                                             | not_ready_full_grid_required |
| adamw_ce_tuned         |                  12 |                            12 |                    360 |                   0.02786 |                  7.225 |                                      0.5321 | family_occupancy_complete    |
| ns_muon_cb_tuned       |                  36 |                             0 |                      0 |                           |                        |                                             | not_ready_full_grid_required |
| ns_muon_matrix_tuned   |                  72 |                             0 |                      0 |                           |                        |                                             | not_ready_full_grid_required |
| sgd_momentum_ce_tuned  |                  12 |                            12 |                    360 |                   0.02786 |                  5.645 |                                      0.5471 | family_occupancy_complete    |

## Operating Rule

This audit can be cited only as progress accounting and reviewer-facing no-peeking evidence. The final claim seed set `20..29` remains blocked until `TVS-1`, `TVS-2`, and `TVS-5` pass in the selection audit.
