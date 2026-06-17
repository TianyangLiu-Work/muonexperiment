# E11 Fresh Condition-Score Evaluation

This generated evaluator is frozen before the fresh final Slurm outputs exist. It applies the registered `condition_score_v3_zero_fit_scaled_jvp` primary score, reports the retired v2 calibrated residual baseline, and keeps the spent ResNet34/original-CIFAR-10 held-outs out of fitting and final evidence.

Fresh score figure is not generated until at least one fresh final split exists.

## Fresh Score Summary

No fresh final score rows are available yet.

## Gate Report

| gate_id                                      | scope                                            | status    | evidence                                                                                                                |
|:---------------------------------------------|:-------------------------------------------------|:----------|:------------------------------------------------------------------------------------------------------------------------|
| fresh_final_heldout_architecture_generated   | Fresh ResNet50 CIFAR-100-LT architecture split   | not_run   | missing fresh layer summary at results/e11_condition_score_fresh_protocol/fresh_architecture_resnet50/layer_summary.csv |
| fresh_final_heldout_data_partition_generated | Fresh CIFAR-10-LT alternate-partition data split | not_run   | missing fresh layer summary at results/e11_condition_score_fresh_protocol/fresh_data_cifar10_alt/layer_summary.csv      |
| fresh_p0_predictive_condition_claim          | fresh condition-score protocol                   | not_ready | All fresh final splits must pass residual, threshold, baseline-comparison, and reporting gates.                         |

## Boundary

`not_run` or `not_ready` is the expected state until both fresh final layer summaries exist. A P0 predictive-condition claim requires both fresh final splits to pass residual-Spearman, threshold-direction, baseline-comparison, and baseline-reporting gates.

Artifacts:
- [fresh_score_pairs.csv](../results/e11_condition_score_fresh_protocol/fresh_score_evaluation/fresh_score_pairs.csv)
- [fresh_score_summary.csv](../results/e11_condition_score_fresh_protocol/fresh_score_evaluation/fresh_score_summary.csv)
- [fresh_gate_report.csv](../results/e11_condition_score_fresh_protocol/fresh_score_evaluation/fresh_gate_report.csv)
- [config.json](../results/e11_condition_score_fresh_protocol/fresh_score_evaluation/config.json)
