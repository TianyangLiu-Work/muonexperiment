# E11 Condition-Score V4 Validation Freeze

This generated artifact is the missing commit boundary between the registered
v4 protocol and any unspent v4 final split. It is intentionally allowed to look
at the validation-only rotated CIFAR-100-LT split, but it keeps all v2/v3 final
splits quarantined and refuses to mark a final-ready score unless the validation
residual-ranking and direction gates pass before final outputs exist.

## Freeze Registry

| item                       | status     | evidence                                                                                                                                                                         |
|:---------------------------|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v4 validation split output | not_run    | results/e11_condition_score_v4_protocol/validation_cifar100_rotated/layer_summary.csv                                                                                            |
| v4 candidate pool          | registered | candidate formulas are fixed in scripts/e11_freeze_condition_score_v4_validation.py                                                                                              |
| v4 selected residual score | not_ready  | pending_validation_output                                                                                                                                                        |
| v4 final split outputs     | not_run    | results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/layer_summary.csv; results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/layer_summary.csv |

## Candidate Formula Registry

| score_id                                              | role                | formula                                                                             | selected_for_final_evaluation   |
|:------------------------------------------------------|:--------------------|:------------------------------------------------------------------------------------|:--------------------------------|
| condition_score_v4_direction_axis_scaled_jvp_ratio    | direction_guardrail | raw log scaled-JVP spectral/Frobenius squared ratio                                 | no                              |
| condition_score_v4_fro_amplitude_axis                 | residual_candidate  | source-standardized Frobenius matched-head-gain scaled-JVP amplitude                | no                              |
| condition_score_v4_two_axis_positive                  | residual_candidate  | source-standardized direction ratio plus Frobenius amplitude                        | no                              |
| condition_score_v4_two_axis_amplitude_minus_direction | residual_candidate  | source-standardized Frobenius amplitude minus direction ratio                       | no                              |
| condition_score_v4_two_axis_transport                 | residual_candidate  | two-axis score with generic downsample/classifier transport tags                    | no                              |
| early_layer_prior                                     | baseline            | raw log early-layer prior                                                           | no                              |
| source_observed_drift_positive_control                | positive_control    | source observed residual under source-fit depth baseline, matched by parameter name | no                              |
| condition_score_v4_validation_selected                | primary_alias       | pending_validation_output                                                           | no                              |

## Validation Score Summary

Validation score rows are not generated yet because the v4 validation Slurm output is missing.

## Gate Report

| gate_id                                | scope                          | status    | evidence                                                                              |
|:---------------------------------------|:-------------------------------|:----------|:--------------------------------------------------------------------------------------|
| V4F-1-validation-output                | validation-only split          | not_run   | results/e11_condition_score_v4_protocol/validation_cifar100_rotated/layer_summary.csv |
| V4F-2-no-final-before-freeze           | unspent final splits           | pass      | no v4 final layer_summary.csv exists before a frozen validation score                 |
| V4F-3-residual-score-freeze            | primary residual-ranking score | not_ready | pending_validation_output                                                             |
| V4F-4-direction-threshold-guardrail    | direction axis                 | not_run   | validation score rows missing                                                         |
| V4F-5-selected-score-residual-spearman | validation residual ranking    | not_run   | no selected validation summary row                                                    |
| V4F-6-final-claim-readiness            | P0 predictive-condition claim  | not_ready | final splits remain blocked until this gate is pass in a committed artifact           |

## Claim Boundary

Current status: validation output is not run.

Allowed now: commit the v4 validation-freeze machinery and, if needed, submit
the validation-only Slurm job.

Blocked now: running or interpreting the unspent WideResNet50-2 and CIFAR-10
mixed final splits as P0 evidence before `V4F-6-final-claim-readiness` passes in
a committed artifact.

Artifacts:
- [score_formula_registry.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/score_formula_registry.csv)
- [freeze_status.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/freeze_status.csv)
- [validation_gate_report.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/validation_gate_report.csv)
- [validation_score_pairs.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/validation_score_pairs.csv)
- [validation_score_summary.csv](../results/e11_condition_score_v4_protocol/validation_score_freeze/validation_score_summary.csv)
- [config.json](../results/e11_condition_score_v4_protocol/validation_score_freeze/config.json)
