# E11 Condition-Score V5 Final Evaluation

This generated evaluator applies the validation-frozen v5 score
`condition_score_v5_transport_normalized_amplitude_minus_direction` to the two unspent final splits. It does not refit
coefficients, does not reselect features, does not inspect v2/v3/v4 final rows,
and does not change the validation-selected score after final outputs exist.

## Final Score Summary

No v5 final score rows are available yet.

## Gate Report

| gate_id                                   | scope                                                 | status    | evidence                                                                                                                            |
|:------------------------------------------|:------------------------------------------------------|:----------|:------------------------------------------------------------------------------------------------------------------------------------|
| v5_final_heldout_architecture_generated   | ResNeXt50-32x4d CIFAR-100-LT final architecture split | not_run   | missing final layer/metrics summary at results/e11_condition_score_v5_protocol/final_architecture_resnext50_32x4d/layer_summary.csv |
| v5_final_heldout_data_partition_generated | CIFAR-10-LT cross-partition final data split          | not_run   | missing final layer/metrics summary at results/e11_condition_score_v5_protocol/final_data_cifar10_cross/layer_summary.csv           |
| v5_p0_predictive_condition_claim          | v5 condition-score final evaluation                   | not_ready | Both unspent v5 final splits must pass residual, direction, baseline-dominance, and reporting gates.                                |

## Claim Boundary

`not_run` is the expected state until both final Slurm jobs finish. `not_ready`
remains the correct P0 state unless both unspent final splits pass residual
Spearman, direction-threshold, baseline-dominance, and control-reporting gates.

Artifacts:
- [final_score_pairs.csv](../results/e11_condition_score_v5_protocol/final_score_evaluation/final_score_pairs.csv)
- [final_score_summary.csv](../results/e11_condition_score_v5_protocol/final_score_evaluation/final_score_summary.csv)
- [final_gate_report.csv](../results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv)
- [config.json](../results/e11_condition_score_v5_protocol/final_score_evaluation/config.json)
