# E11 Condition-Score V4 Validation Freeze

This generated artifact is the missing commit boundary between the registered
v4 protocol and any unspent v4 final split. It is intentionally allowed to look
at the validation-only rotated CIFAR-100-LT split, but it keeps all v2/v3 final
splits quarantined and refuses to mark a final-ready score unless the validation
residual-ranking and direction gates pass before final outputs exist.

## Freeze Registry

| item                       | status     | evidence                                                                                                                                                                         |
|:---------------------------|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v4 validation split output | generated  | results/e11_condition_score_v4_protocol/validation_cifar100_rotated/layer_summary.csv                                                                                            |
| v4 candidate pool          | registered | candidate formulas are fixed in scripts/e11_freeze_condition_score_v4_validation.py                                                                                              |
| v4 selected residual score | frozen     | condition_score_v4_two_axis_amplitude_minus_direction                                                                                                                            |
| v4 final split outputs     | not_run    | results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/layer_summary.csv; results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/layer_summary.csv |

## Candidate Formula Registry

| score_id                                              | role                | formula                                                                             | selected_for_final_evaluation   |
|:------------------------------------------------------|:--------------------|:------------------------------------------------------------------------------------|:--------------------------------|
| condition_score_v4_direction_axis_scaled_jvp_ratio    | direction_guardrail | raw log scaled-JVP spectral/Frobenius squared ratio                                 | no                              |
| condition_score_v4_fro_amplitude_axis                 | residual_candidate  | source-standardized Frobenius matched-head-gain scaled-JVP amplitude                | no                              |
| condition_score_v4_two_axis_positive                  | residual_candidate  | source-standardized direction ratio plus Frobenius amplitude                        | no                              |
| condition_score_v4_two_axis_amplitude_minus_direction | residual_candidate  | source-standardized Frobenius amplitude minus direction ratio                       | yes                             |
| condition_score_v4_two_axis_transport                 | residual_candidate  | two-axis score with generic downsample/classifier transport tags                    | no                              |
| early_layer_prior                                     | baseline            | raw log early-layer prior                                                           | no                              |
| source_observed_drift_positive_control                | positive_control    | source observed residual under source-fit depth baseline, matched by parameter name | no                              |
| condition_score_v4_validation_selected                | primary_alias       | condition_score_v4_two_axis_amplitude_minus_direction                               | yes                             |

## Validation Score Summary

| score                                                 | score_role          |   validation_transfer_pairs |   mean_spearman_score_vs_target_residual |   spearman_ci95_low |   spearman_ci95_high | mean_threshold_below_one_accuracy   |
|:------------------------------------------------------|:--------------------|----------------------------:|-----------------------------------------:|--------------------:|---------------------:|:------------------------------------|
| condition_score_v4_direction_axis_scaled_jvp_ratio    | direction_guardrail |                           9 |                                 -0.2468  |            -0.3085  |              -0.185  | 1                                   |
| condition_score_v4_fro_amplitude_axis                 | residual_candidate  |                           9 |                                  0.2     |             0.09707 |               0.3029 | n/a                                 |
| condition_score_v4_two_axis_positive                  | residual_candidate  |                           9 |                                  0.1879  |             0.1104  |               0.2654 | n/a                                 |
| condition_score_v4_two_axis_amplitude_minus_direction | residual_candidate  |                           9 |                                  0.3758  |             0.2759  |               0.4756 | n/a                                 |
| condition_score_v4_two_axis_transport                 | residual_candidate  |                           9 |                                  0.2203  |             0.1321  |               0.3086 | n/a                                 |
| early_layer_prior                                     | baseline            |                           9 |                                  0.04242 |            -0.05664 |               0.1415 | n/a                                 |
| source_observed_drift_positive_control                | positive_control    |                           9 |                                  0.8586  |             0.8135  |               0.9037 | 1                                   |

## Gate Report

| gate_id                                | scope                          | status   | evidence                                                                                                    |
|:---------------------------------------|:-------------------------------|:---------|:------------------------------------------------------------------------------------------------------------|
| V4F-1-validation-output                | validation-only split          | pass     | results/e11_condition_score_v4_protocol/validation_cifar100_rotated/layer_summary.csv                       |
| V4F-2-no-final-before-freeze           | unspent final splits           | pass     | no v4 final layer_summary.csv exists before a frozen validation score                                       |
| V4F-3-residual-score-freeze            | primary residual-ranking score | pass     | condition_score_v4_two_axis_amplitude_minus_direction                                                       |
| V4F-4-direction-threshold-guardrail    | direction axis                 | pass     | validation direction-axis threshold accuracy=1 CI=[1, 1]                                                    |
| V4F-5-selected-score-residual-spearman | validation residual ranking    | pass     | selected=condition_score_v4_two_axis_amplitude_minus_direction; Spearman=0.3758 CI=[0.2759, 0.4756]         |
| V4F-6-final-claim-readiness            | P0 predictive-condition claim  | pass     | unspent final split jobs may run after this pass artifact is committed; P0 claim still requires final gates |

## Claim Boundary

Current status: validation output exists.

Allowed now: commit this pass validation-freeze artifact, then submit the unspent WideResNet50-2 and CIFAR-10 mixed final split Slurm jobs with the frozen selected score.

Blocked now: making a v4 P0 predictive-condition claim before both unspent final split evaluations pass their residual-ranking, direction, baseline-reporting, and claim-boundary gates.

Artifacts:
- [score_formula_registry.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/score_formula_registry.csv)
- [freeze_status.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/freeze_status.csv)
- [validation_gate_report.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/validation_gate_report.csv)
- [validation_score_pairs.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/validation_score_pairs.csv)
- [validation_score_summary.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/validation_score_summary.csv)
- [config.json](../results/e11_condition_score_v4_protocol/validation_score_freeze/config.json)
