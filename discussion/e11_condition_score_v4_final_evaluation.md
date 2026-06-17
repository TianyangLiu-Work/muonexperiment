# E11 Condition-Score V4 Final Evaluation

This generated evaluator applies the validation-frozen v4 score
`condition_score_v4_two_axis_amplitude_minus_direction` to the two unspent final splits. It does not refit
coefficients, does not use v2/v3 final split outcomes, and reports direction,
early-layer, and source-observed controls beside the primary residual score.

## Final Score Summary

| split_role                         | score                                                 | score_role          |   final_transfer_pairs |   mean_spearman_score_vs_target_residual |   spearman_ci95_low |   spearman_ci95_high | mean_threshold_below_one_accuracy   |
|:-----------------------------------|:------------------------------------------------------|:--------------------|-----------------------:|-----------------------------------------:|--------------------:|---------------------:|:------------------------------------|
| fresh_final_heldout_architecture   | condition_score_v4_two_axis_amplitude_minus_direction | primary_candidate   |                      9 |                                  0.6449  |             0.5122  |               0.7776 | n/a                                 |
| fresh_final_heldout_architecture   | condition_score_v4_direction_axis_scaled_jvp_ratio    | direction_guardrail |                      9 |                                 -0.7114  |            -0.776   |              -0.6469 | 1                                   |
| fresh_final_heldout_architecture   | early_layer_prior                                     | baseline            |                      9 |                                  0.2134  |             0.05471 |               0.372  | n/a                                 |
| fresh_final_heldout_architecture   | source_observed_drift_positive_control                | positive_control    |                      9 |                                  0.4557  |             0.4131  |               0.4983 | 1                                   |
| fresh_final_heldout_data_partition | condition_score_v4_two_axis_amplitude_minus_direction | primary_candidate   |                      9 |                                 -0.6937  |            -0.7129  |              -0.6744 | n/a                                 |
| fresh_final_heldout_data_partition | condition_score_v4_direction_axis_scaled_jvp_ratio    | direction_guardrail |                      9 |                                  0.611   |             0.5886  |               0.6334 | 1                                   |
| fresh_final_heldout_data_partition | early_layer_prior                                     | baseline            |                      9 |                                 -0.7939  |            -0.8086  |              -0.7793 | n/a                                 |
| fresh_final_heldout_data_partition | source_observed_drift_positive_control                | positive_control    |                      9 |                                  0.08369 |             0.0636  |               0.1038 | 1                                   |

## Gate Report

| gate_id                                                         | scope                                                | status    | evidence                                                                      |
|:----------------------------------------------------------------|:-----------------------------------------------------|:----------|:------------------------------------------------------------------------------|
| fresh_final_heldout_architecture_residual_spearman              | WideResNet50-2 CIFAR-100-LT final architecture split | pass      | primary residual Spearman=0.6449 CI=[0.5122, 0.7776]                          |
| fresh_final_heldout_architecture_direction_threshold_accuracy   | WideResNet50-2 CIFAR-100-LT final architecture split | pass      | direction-axis below-one threshold accuracy=1                                 |
| fresh_final_heldout_architecture_baselines_reported             | WideResNet50-2 CIFAR-100-LT final architecture split | pass      | direction-axis, early-prior, and source-observed controls reported            |
| fresh_final_heldout_data_partition_residual_spearman            | CIFAR-10-LT mixed-partition final data split         | fail      | primary residual Spearman=-0.6937 CI=[-0.7129, -0.6744]                       |
| fresh_final_heldout_data_partition_direction_threshold_accuracy | CIFAR-10-LT mixed-partition final data split         | pass      | direction-axis below-one threshold accuracy=1                                 |
| fresh_final_heldout_data_partition_baselines_reported           | CIFAR-10-LT mixed-partition final data split         | pass      | direction-axis, early-prior, and source-observed controls reported            |
| v4_p0_predictive_condition_claim                                | v4 condition-score final evaluation                  | not_ready | Both unspent final splits must pass residual, direction, and reporting gates. |

## Claim Boundary

`not_run` is the expected state until both final Slurm jobs finish. `not_ready`
remains the correct P0 state unless both unspent final splits pass all gates.

Artifacts:
- [final_score_pairs.csv](../results/e11_condition_score_v4_protocol/final_score_evaluation/final_score_pairs.csv)
- [final_score_summary.csv](../results/e11_condition_score_v4_protocol/final_score_evaluation/final_score_summary.csv)
- [final_gate_report.csv](../results/e11_condition_score_v4_protocol/final_score_evaluation/final_gate_report.csv)
- [config.json](../results/e11_condition_score_v4_protocol/final_score_evaluation/config.json)
