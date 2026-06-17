# E11 Condition-Score V5 Final Evaluation

This generated evaluator applies the validation-frozen v5 score
`condition_score_v5_transport_normalized_amplitude_minus_direction` to the two unspent final splits. It does not refit
coefficients, does not reselect features, does not inspect v2/v3/v4 final rows,
and does not change the validation-selected score after final outputs exist.

## Final Score Summary

| split_role                      | score                                                             | score_role          |   final_transfer_pairs |   mean_spearman_score_vs_target_residual |   spearman_ci95_low |   spearman_ci95_high |   mean_top5_residual_risk_overlap_fraction | mean_threshold_below_one_accuracy   |
|:--------------------------------|:------------------------------------------------------------------|:--------------------|-----------------------:|-----------------------------------------:|--------------------:|---------------------:|-------------------------------------------:|:------------------------------------|
| v5_final_heldout_architecture   | condition_score_v5_transport_normalized_amplitude_minus_direction | primary_candidate   |                      9 |                                  0.43    |             0.2246  |              0.6354  |                                     0.2222 | n/a                                 |
| v5_final_heldout_architecture   | condition_score_v5_direction_axis_scaled_jvp_ratio                | direction_guardrail |                      9 |                                 -0.4431  |            -0.6581  |             -0.228   |                                     0.1333 | 0.8395                              |
| v5_final_heldout_architecture   | early_layer_prior                                                 | baseline            |                      9 |                                 -0.08853 |            -0.3971  |              0.2201  |                                     0      | n/a                                 |
| v5_final_heldout_architecture   | source_observed_drift_positive_control                            | positive_control    |                      9 |                                  0.1789  |             0.03999 |              0.3179  |                                     0.1778 | 0.8571                              |
| v5_final_heldout_data_partition | condition_score_v5_transport_normalized_amplitude_minus_direction | primary_candidate   |                      9 |                                 -0.6851  |            -0.718   |             -0.6523  |                                     0      | n/a                                 |
| v5_final_heldout_data_partition | condition_score_v5_direction_axis_scaled_jvp_ratio                | direction_guardrail |                      9 |                                  0.6548  |             0.6379  |              0.6718  |                                     0.8222 | 1                                   |
| v5_final_heldout_data_partition | early_layer_prior                                                 | baseline            |                      9 |                                 -0.882   |            -0.9117  |             -0.8522  |                                     0      | n/a                                 |
| v5_final_heldout_data_partition | source_observed_drift_positive_control                            | positive_control    |                      9 |                                 -0.03492 |            -0.08863 |              0.01879 |                                     0.1333 | 1                                   |

## Gate Report

| gate_id                                                      | scope                                                 | status    | evidence                                                                                             |
|:-------------------------------------------------------------|:------------------------------------------------------|:----------|:-----------------------------------------------------------------------------------------------------|
| v5_final_heldout_architecture_residual_spearman              | ResNeXt50-32x4d CIFAR-100-LT final architecture split | pass      | primary residual Spearman=0.43 CI=[0.2246, 0.6354]                                                   |
| v5_final_heldout_architecture_direction_threshold_accuracy   | ResNeXt50-32x4d CIFAR-100-LT final architecture split | fail      | direction-axis below-one threshold accuracy=0.8395, CI low=0.6822                                    |
| v5_final_heldout_architecture_baseline_dominance             | ResNeXt50-32x4d CIFAR-100-LT final architecture split | pass      | primary score must beat early_layer_prior on residual Spearman and top-k overlap                     |
| v5_final_heldout_architecture_controls_reported              | ResNeXt50-32x4d CIFAR-100-LT final architecture split | pass      | direction-axis, early-prior, and source-observed controls reported                                   |
| v5_final_heldout_data_partition_residual_spearman            | CIFAR-10-LT cross-partition final data split          | fail      | primary residual Spearman=-0.6851 CI=[-0.718, -0.6523]                                               |
| v5_final_heldout_data_partition_direction_threshold_accuracy | CIFAR-10-LT cross-partition final data split          | pass      | direction-axis below-one threshold accuracy=1, CI low=1                                              |
| v5_final_heldout_data_partition_baseline_dominance           | CIFAR-10-LT cross-partition final data split          | pass      | primary score must beat early_layer_prior on residual Spearman and top-k overlap                     |
| v5_final_heldout_data_partition_controls_reported            | CIFAR-10-LT cross-partition final data split          | pass      | direction-axis, early-prior, and source-observed controls reported                                   |
| v5_p0_predictive_condition_claim                             | v5 condition-score final evaluation                   | not_ready | Both unspent v5 final splits must pass residual, direction, baseline-dominance, and reporting gates. |

## Claim Boundary

`not_run` is the expected state only for final splits whose layer/metrics
outputs have not arrived yet. `not_ready` remains the correct P0 state unless
both unspent final splits pass residual Spearman, direction-threshold,
baseline-dominance, and control-reporting gates.

Artifacts:
- [final_score_pairs.csv](../results/e11_condition_score_v5_protocol/final_score_evaluation/final_score_pairs.csv)
- [final_score_summary.csv](../results/e11_condition_score_v5_protocol/final_score_evaluation/final_score_summary.csv)
- [final_gate_report.csv](../results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv)
- [config.json](../results/e11_condition_score_v5_protocol/final_score_evaluation/config.json)
