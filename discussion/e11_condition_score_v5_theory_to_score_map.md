# E11 Condition-Score V5 Theory-to-Score Map

This generated artifact is a theory-to-measurement bridge for the v5
condition-score attempt. It does not make the P0 predictive-condition claim.
It states the mathematical quantity, the measurable proxy, the leakage
boundary, and the falsifiable validation/final gates that would be needed
before such a claim is supportable.

## Proposition: Transport-Stable Sandwich Residual

For a target partition or architecture `p`, write the matched-head-gain
layerwise tail risk as
`R_l^p(D)=||B_T,p,l D_l A_T,p,l||_F^2`. If a frozen transport term
`T_l(p,s)` satisfies
`|log R_l^p(D_fro)-log R_l^s(D_fro)-T_l(p,s)| <= epsilon_l` without using
final residual labels, and if the direction ratio
`log R_l^p(D_spectral)-log R_l^p(D_fro)` is evaluated as a separate guardrail,
then a source-standardized amplitude-minus-direction score can preserve
residual-risk ordering only up to the unresolved transport error
`epsilon_l`. When those errors exceed the residual pairwise gaps, sign
reversal is an expected failure mode rather than a statistical accident.

Proof sketch. The matrix-block theorem controls the sandwiched quantity
`B_T D A_T` after head-gain matching. Source-to-target transfer changes the
outer and inner tail operators, so raw amplitude and depth terms acquire a
transport error. Subtracting the direction ratio removes a different target:
the below-one direction comparison. Therefore residual ranking requires a
frozen transport correction and a separate direction guardrail.

## Theorem Proxy Map

| map_id                    | theorem_quantity                                                                                | measured_proxy                                                                     | score_feature                                                                        | transport_term                                               | validation_gate                                                                  | claim_boundary                                                                                   |
|:--------------------------|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| M1-sandwiched-tail-risk   | R_l^p(D)=||B_T,p,l D_l A_T,p,l||_F^2 after matched head gain                                    | geomean observed and finite-difference JVP tail-drift ratios                       | log_scaled_jvp_fro_amplitude                                                         | partition_transport_defect and architecture_transport_defect | residual-candidate Spearman CI lower endpoint above zero on the validation split | cannot claim residual-risk prediction from rank-only or direction-only evidence                  |
| M2-direction-ratio        | log R_l^p(D_spectral)-log R_l^p(D_frobenius)                                                    | log scaled-JVP spectral/Frobenius squared ratio                                    | log_scaled_jvp_ratio                                                                 | none for threshold sign; separate from residual ranking      | below-one threshold accuracy lower endpoint at least 0.8                         | a passing direction guardrail does not support a residual-risk claim if ranking fails            |
| M3-source-depth-residual  | residual layer risk after subtracting a source depth baseline                                   | target residual from source-fit depth intercept and slope                          | log_early_layer_prior and frozen_depth_slope                                         | early_depth_nuisance                                         | primary score beats early_layer_prior on residual ranking                        | failure to beat the early-layer baseline blocks the predictive-condition claim                   |
| M4-partition-transport    | change in class-conditioned B_T,p,l and A_T,p,l weighting across partitions                     | pre-update partition tags and validation-frozen transport penalties                | transport_is_downsample and transport_is_classifier plus future partition statistics | partition_transport_defect                                   | validation-freeze artifact exists before final output directories                | spent CIFAR-10 final rows may motivate this claim boundary but cannot tune it                    |
| M5-architecture-transport | change in layer shape, bottleneck/downsample map, and classifier transport across architectures | architecture and parameterization tags registered before the ResNeXt50 final split | transport_is_downsample and transport_is_classifier                                  | architecture_transport_defect                                | fresh ResNeXt50-32x4d final is evaluated only after validation freeze            | architecture final failure narrows the claim to the local mechanism or fixed architecture family |

## Score Lineage

| score_id                                                          | score_role                | theory_terms_used                                                                                                                  | claim_role                   | leakage_status      | current_validation_status   | restriction                                                                   |
|:------------------------------------------------------------------|:--------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------|:--------------------|:----------------------------|:------------------------------------------------------------------------------|
| condition_score_v5_direction_axis_scaled_jvp_ratio                | direction_guardrail       | direction_ratio_guardrail                                                                                                          | direction guardrail          | no spent final rows | no                          | not eligible as a residual-risk scalar                                        |
| condition_score_v5_raw_fro_amplitude_axis                         | diagnostic_residual_axis  | raw_residual_amplitude                                                                                                             | diagnostic residual axis     | no spent final rows | no                          | spent v4 data evidence shows raw amplitude can reverse                        |
| condition_score_v5_transport_normalized_amplitude_minus_direction | residual_candidate        | raw_residual_amplitude; direction_ratio_guardrail; partition_transport_defect; architecture_transport_defect; early_depth_nuisance | primary residual candidate   | no spent final rows | yes                         | eligible only if validation residual Spearman CI lower endpoint is above zero |
| condition_score_v5_transport_defect_penalty                       | diagnostic_transport_axis | partition_transport_defect; architecture_transport_defect                                                                          | diagnostic transport axis    | no spent final rows | no                          | reported to localize transport failures, not a final residual score           |
| early_layer_prior                                                 | baseline                  | early_depth_nuisance                                                                                                               | baseline                     | no spent final rows | no                          | primary must beat this baseline                                               |
| source_observed_drift_positive_control                            | positive_control          | sandwiched_tail_drift                                                                                                              | upper-bound positive control | no spent final rows | no                          | uses observed source drift and is not claim-eligible                          |
| condition_score_v5_validation_selected                            | primary_alias             | validation-freeze alias                                                                                                            | primary alias                | no spent final rows | frozen                      | frozen / condition_score_v5_transport_normalized_amplitude_minus_direction    |

## Transport Normalization Contract

| step_id               | operation                                                                                            | allowed_inputs                                                      | forbidden_inputs                                         | output                                                         | freeze_point                                                   | gate                                                                        |
|:----------------------|:-----------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:---------------------------------------------------------|:---------------------------------------------------------------|:---------------------------------------------------------------|:----------------------------------------------------------------------------|
| T1-source-calibration | Compute source feature means, variances, and depth baseline on the source checkpoint-transfer split. | source layer_summary.csv and source metrics.csv                     | v2/v3/v4 final residual labels; v5 final residual labels | source-standardized amplitude, ratio, and early-depth features | before validation scoring                                      | source statistics are deterministic and regenerated by the freeze script    |
| T2-transport-tags     | Apply pre-registered downsample/classifier transport penalties.                                      | parameter names and architecture tags visible before final outcomes | final split residual ranking                             | transport_is_downsample and transport_is_classifier penalties  | in score_formula_registry.csv before final split output exists | score_formula_registry.csv reports uses_spent_final_rows=no                 |
| T3-validation-freeze  | Use only the v5 validation split to freeze or reject the residual candidate.                         | v5 validation layer_summary.csv and metrics.csv                     | v5 final architecture/data outputs                       | frozen or validation_failed residual score status              | discussion/e11_condition_score_v5_validation_freeze.md         | V5F-4 residual-score-freeze and V5F-5 direction-threshold-guardrail         |
| T4-final-evaluation   | Evaluate the frozen score without changing weights, thresholds, features, or claim boundary.         | new ResNeXt50-32x4d and CIFAR-10 cross-partition final outputs      | any post-hoc final coefficient or feature selection      | final residual and direction gate report                       | after T3 passes                                                | both final splits must pass residual Spearman and direction threshold gates |
| T5-negative-path      | If validation or final gates fail, preserve the failure as boundary evidence.                        | failed validation/final reports                                     | reranking scores on visible failed final rows            | narrow fixed-partition/local-mechanism claim                   | immediately after a failed gate                                | README and gap register keep predictive-condition claim not_ready           |

## Falsifiable Predictions

| prediction_id             | target                                   | claim_tested                                                                                            | pass_rule                                                                            | fail_interpretation                                            | claim_effect                                                               |
|:--------------------------|:-----------------------------------------|:--------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|:---------------------------------------------------------------------------|
| V5-P1-direction-guardrail | validation and final splits              | The below-one spectral/Frobenius direction remains a separate positive guardrail.                       | threshold accuracy lower endpoint is at least 0.8                                    | the local direction mechanism does not transport to this split | blocks even the direction-guardrail extension                              |
| V5-P2-residual-validation | validation-only mod-4 CIFAR-100-LT split | Transport-normalized amplitude-minus-direction predicts residual layer risk before final outputs exist. | residual Spearman CI lower endpoint is above zero                                    | the current transport terms are still insufficient             | do not run final splits for P0 predictive-condition evidence               |
| V5-P3-architecture-final  | ResNeXt50-32x4d CIFAR-100-LT final split | The frozen score survives a new architecture family and implementation path.                            | residual Spearman CI lower endpoint above zero and direction gate passes             | architecture transport defect remains unmodeled                | narrow to fixed architecture or local mechanism                            |
| V5-P4-data-final          | CIFAR-10 cross-partition final split     | The frozen score survives a new data-family partition after prior CIFAR-10 finals are spent.            | residual Spearman CI lower endpoint above zero and direction gate passes             | partition transport defect remains unmodeled                   | no broad predictive-condition claim                                        |
| V5-P5-baseline-dominance  | validation and final residual summaries  | The score contains information beyond the early-depth nuisance baseline.                                | primary residual score beats early_layer_prior on residual ranking and top-k overlap | the measurable score is mostly a depth/stage proxy             | paper reports a nuisance-baseline failure, not a theorem-derived predictor |

## Required Ablation Matrix

| ablation_id                       | isolated_or_removed_term                                   | spent_v4_observation                                                                                                                                          | expected_v5_failure_mode                                                | required_v5_report                                                 |
|:----------------------------------|:-----------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------|:-------------------------------------------------------------------|
| A1-direction-only                 | direction_ratio_guardrail only                             | CIFAR-10 mixed direction axis stays positive: 0.611 [0.5886, 0.6334]; direction-axis Spearman 0.611; direction threshold accuracy is 1 in the final evaluator | passes below-one direction while failing residual-risk ranking          | direction threshold reported separately from residual Spearman     |
| A2-raw-amplitude-only             | raw_residual_amplitude without transport                   | CIFAR-10 mixed amplitude reverses: -0.6766 [-0.6903, -0.663]                                                                                                  | partition changes flip amplitude/depth ordering                         | raw amplitude remains diagnostic-only                              |
| A3-early-depth-only               | early_depth_nuisance baseline                              | CIFAR-10 mixed early-layer prior reverses: -0.7939 [-0.8086, -0.7793]                                                                                         | score is only a layer-depth proxy                                       | primary score must beat early_layer_prior                          |
| A4-v4-amplitude-minus-direction   | partition_transport_defect omitted from scalar aggregation | WideResNet50-2 passes at 0.6449 [0.5122, 0.7776] but CIFAR-10 mixed fails at -0.6937 [-0.7129, -0.6744]                                                       | architecture transfer may pass while data-partition transfer fails      | architecture and data finals are separate gates                    |
| A5-transport-normalized-candidate | all v5 terms included before final                         | spent rows motivate the term but are quarantined from fitting                                                                                                 | validation_failed or final not_ready if transport is still insufficient | preserve negative validation/final outcomes without score retuning |

## Claim Readiness Ledger

| item                       | status    | evidence                                                                                                                                                                          | blocks_p0_if_missing   |
|:---------------------------|:----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|
| theory-to-score map        | generated | discussion/e11_condition_score_v5_theory_to_score_map.md                                                                                                                          | yes                    |
| v5 validation output       | generated | results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/layer_summary.csv                                                                                      | yes                    |
| v5 residual score freeze   | frozen    | results/e11_condition_score_v5_protocol/validation_score_freeze/freeze_status.csv                                                                                                 | yes                    |
| no final before freeze     | pass      | results/e11_condition_score_v5_protocol/final_* directories absent before freeze                                                                                                  | yes                    |
| predictive-condition claim | not_ready | frozen validation-selected score became eligible for final evaluation runs; completed final gates are evaluated separately and P0 remains not_ready after the registered failures | yes                    |

## Boundary

Allowed now: cite this map as the pre-final theory-to-score bridge, use spent
v4 evidence only as diagnostic motivation, and run the v5 final splits with the
frozen validation-selected residual score.

Blocked now: fitting, selecting, or reweighting any v5 score on v2/v3/v4 final
rows; claiming that the transport-normalized residual score predicts held-out
layer risk before both v5 final gates pass.

Artifacts:
- [theorem_proxy_map.csv](../results/e11_condition_score_v5_theory_to_score_map/theorem_proxy_map.csv)
- [score_lineage.csv](../results/e11_condition_score_v5_theory_to_score_map/score_lineage.csv)
- [transport_normalization_contract.csv](../results/e11_condition_score_v5_theory_to_score_map/transport_normalization_contract.csv)
- [falsifiable_predictions.csv](../results/e11_condition_score_v5_theory_to_score_map/falsifiable_predictions.csv)
- [ablation_matrix.csv](../results/e11_condition_score_v5_theory_to_score_map/ablation_matrix.csv)
- [claim_readiness_ledger.csv](../results/e11_condition_score_v5_theory_to_score_map/claim_readiness_ledger.csv)
