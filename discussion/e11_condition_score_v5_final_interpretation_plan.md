# E11 Condition-Score V5 Final Interpretation Plan

This generated artifact is the post-output interpretation lock for the v5 final
splits. The outcome-to-claim state machine was fixed before final rows were used;
now that both registered final split tables are generated, this artifact records
the current completed negative boundary while it still forbids changing the
validation-frozen score `condition_score_v5_transport_normalized_amplitude_minus_direction`, split set, thresholds, or baseline
comparisons. The locked ladder explicitly separates positive P0 eligibility from
data/architecture transport boundaries, direction-guardrail failures,
baseline-dominance failures, and local-mechanism-only outcomes.

## Current Interpretation Summary

| output_state   | current_claim_state             | active_ladder_states                                                       | blocking_gate_ids                                                                                                                               | allowed_current_interpretation                                                | forbidden_current_interpretation                           | required_paper_action                                                                                      |
|:---------------|:--------------------------------|:---------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|:-----------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|
| generated      | completed_final_failed_boundary | data_transport_boundary; direction_guardrail_failure; local_mechanism_only | v5_final_heldout_architecture_direction_threshold_accuracy; v5_final_heldout_data_partition_residual_spearman; v5_p0_predictive_condition_claim | completed final negative boundary; local mechanism only under the v5 protocol | the v5 frozen score is an unseen-task predictive condition | preserve failed gates, do not repair on final rows, and open a new unspent protocol for any score revision |

## Final Split Status

| split_id                                         | split_role                      | current_output_status   | claim_gate_group   | can_be_replaced   | can_be_dropped_after_result   |
|:-------------------------------------------------|:--------------------------------|:------------------------|:-------------------|:------------------|:------------------------------|
| v5_final_architecture_resnext50_32x4d_cifar100lt | v5_final_heldout_architecture   | generated               | required_for_p0    | no                | no                            |
| v5_final_data_cifar10lt_cross_partition          | v5_final_heldout_data_partition | generated               | required_for_p0    | no                | no                            |

## Gate Contract

| gate_id                          | required_for                  | pass_rule                                                                                         | failure_effect                                                                  |
|:---------------------------------|:------------------------------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|
| V5-FINAL-G1-frozen-primary-score | all final interpretations     | primary score remains condition_score_v5_transport_normalized_amplitude_minus_direction           | all final interpretations are invalid until the committed evaluator is restored |
| V5-FINAL-G2-output-completeness  | P0 predictive-condition claim | both registered final splits have layer_summary.csv and metrics.csv                               | P0 remains not_ready; no split may be substituted                               |
| V5-FINAL-G3-residual-ranking     | P0 predictive-condition claim | primary residual Spearman CI lower endpoint is above zero on each final split                     | the failed split becomes a transport-boundary result, not a tuning target       |
| V5-FINAL-G4-direction-guardrail  | P0 predictive-condition claim | direction-axis below-one threshold accuracy and CI lower endpoint are both at least 0.8           | the local spectral/Frobenius direction guardrail failed to transport            |
| V5-FINAL-G5-baseline-dominance   | P0 predictive-condition claim | primary score beats early_layer_prior on residual Spearman and top-k overlap                      | the score is treated as a depth/stage nuisance proxy                            |
| V5-FINAL-G6-control-reporting    | paper reporting               | direction, early-layer, and source-observed controls are reported for every generated final split | the final evaluator output is incomplete for review                             |

## Outcome Ladder

| outcome_pattern                                              | claim_state                     | allowed_interpretation                                                                                            | forbidden_interpretation                                                      | required_paper_action                                                                                                 |
|:-------------------------------------------------------------|:--------------------------------|:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| both final splits missing or incomplete (pre-output branch)  | not_ready                       | registered final evaluation is pending only while outputs are missing                                             | any predictive-condition or generality claim                                  | before outputs exist, report pending Slurm/output status; after outputs exist, use current_interpretation_summary.csv |
| both final splits pass all P0 gates                          | p0_claim_eligible               | the frozen transport-normalized score predicts residual layer risk on the registered architecture and data finals | claiming final optimizer performance or broader dataset/architecture coverage | report both split summaries, controls, confidence intervals, and the no-retuning boundary                             |
| architecture split passes, data split fails residual ranking | data_transport_boundary         | architecture transfer survived, but data-partition transport remains unresolved                                   | broad data-family predictive-condition claim                                  | preserve the negative CIFAR-10 cross-partition result and analyze partition transport terms                           |
| data split passes, architecture split fails residual ranking | architecture_transport_boundary | data-family transfer survived, but ResNeXt architecture transport remains unresolved                              | broad architecture-family predictive-condition claim                          | preserve the negative ResNeXt50-32x4d result and analyze parameterization transport terms                             |
| either final split fails direction guardrail                 | direction_guardrail_failure     | the local spectral/Frobenius direction comparison failed to transport to the final split                          | residual-risk predictor claim, even if residual ranking is positive elsewhere | separate direction failure from scalar residual-score failure                                                         |
| either final split fails baseline dominance                  | nuisance_proxy_boundary         | the frozen score did not add enough information beyond the early-layer nuisance baseline                          | theory-derived measurable score claim                                         | report early_layer_prior comparison as a failed ablation gate                                                         |
| both final splits fail any P0 gate                           | local_mechanism_only            | the paper retains local drift-mechanism and falsification evidence only                                           | predictive-condition claim on unseen real tasks                               | keep failed finals in the main evidence ledger and do not retune on them                                              |

## Leakage Lock

| locked_item    | locked_value                                                          | forbidden_after_final_outputs                                 | allowed_after_final_outputs                               |
|:---------------|:----------------------------------------------------------------------|:--------------------------------------------------------------|:----------------------------------------------------------|
| primary_score  | condition_score_v5_transport_normalized_amplitude_minus_direction     | changing score weights, features, signs, or aliases           | rerun the same evaluator and report generated rows        |
| final_splits   | ResNeXt50-32x4d CIFAR-100-LT; CIFAR-10 cross partition                | dropping, replacing, or adding splits to rescue the claim     | add future splits only under a new preregistered protocol |
| residual_gate  | Spearman CI lower endpoint above zero on each final split             | using mean-only positivity or a one-split pass as P0 evidence | report partial positives as boundary evidence             |
| direction_gate | threshold accuracy and CI lower endpoint at least 0.8                 | lowering the threshold or merging it into residual ranking    | report a direction failure as a separate obstruction      |
| baseline_gate  | primary beats early_layer_prior on residual ranking and top-k overlap | omitting the early-layer nuisance comparison                  | downgrade to nuisance-proxy boundary if baseline wins     |

Artifacts:
- [current_interpretation_summary.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/current_interpretation_summary.csv)
- [final_split_status.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/final_split_status.csv)
- [final_gate_contract.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/final_gate_contract.csv)
- [outcome_interpretation_ladder.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/outcome_interpretation_ladder.csv)
- [leakage_lock.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/leakage_lock.csv)
- [config.json](../results/e11_condition_score_v5_protocol/final_interpretation_plan/config.json)
