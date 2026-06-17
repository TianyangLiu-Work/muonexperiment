# E11 Condition-Score V4 Final Evaluation

This generated evaluator applies the validation-frozen v4 score
`condition_score_v4_two_axis_amplitude_minus_direction` to the two unspent final splits. It does not refit
coefficients, does not use v2/v3 final split outcomes, and reports direction,
early-layer, and source-observed controls beside the primary residual score.

## Final Score Summary

No v4 final score rows are available yet.

## Gate Report

| gate_id                                      | scope                                                | status    | evidence                                                                                                                    |
|:---------------------------------------------|:-----------------------------------------------------|:----------|:----------------------------------------------------------------------------------------------------------------------------|
| fresh_final_heldout_architecture_generated   | WideResNet50-2 CIFAR-100-LT final architecture split | not_run   | missing final layer summary at results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/layer_summary.csv |
| fresh_final_heldout_data_partition_generated | CIFAR-10-LT mixed-partition final data split         | not_run   | missing final layer summary at results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/layer_summary.csv           |
| v4_p0_predictive_condition_claim             | v4 condition-score final evaluation                  | not_ready | Both unspent final splits must pass residual, direction, and reporting gates.                                               |

## Claim Boundary

`not_run` is the expected state until both final Slurm jobs finish. `not_ready`
remains the correct P0 state unless both unspent final splits pass all gates.

Artifacts:
- [final_score_pairs.csv](../results/e11_condition_score_v4_protocol/final_score_evaluation/final_score_pairs.csv)
- [final_score_summary.csv](../results/e11_condition_score_v4_protocol/final_score_evaluation/final_score_summary.csv)
- [final_gate_report.csv](../results/e11_condition_score_v4_protocol/final_score_evaluation/final_gate_report.csv)
- [config.json](../results/e11_condition_score_v4_protocol/final_score_evaluation/config.json)
