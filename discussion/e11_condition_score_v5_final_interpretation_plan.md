# E11 Condition-Score V5 Final Interpretation Plan

This generated artifact is a pre-output interpretation lock for the v5 final
splits. It fixes the outcome-to-claim state machine for the submitted
ResNeXt50-32x4d architecture final and CIFAR-10 cross-partition final before
their layer tables are available. It uses the validation-frozen score `condition_score_v5_transport_normalized_amplitude_minus_direction`
and forbids changing the score, split set, thresholds, or baseline comparisons
after final outputs exist. The locked ladder explicitly separates positive P0
eligibility from data/architecture transport boundaries, direction-guardrail failures,
baseline-dominance failures, and local-mechanism-only outcomes.

## Final Split Status

| split_id                                         | split_role                      | current_output_status   | claim_gate_group   | can_be_replaced   | can_be_dropped_after_result   |
|:-------------------------------------------------|:--------------------------------|:------------------------|:-------------------|:------------------|:------------------------------|
| v5_final_architecture_resnext50_32x4d_cifar100lt | v5_final_heldout_architecture   | not_run                 | required_for_p0    | no                | no                            |
| v5_final_data_cifar10lt_cross_partition          | v5_final_heldout_data_partition | not_run                 | required_for_p0    | no                | no                            |

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

| outcome_pattern                                              | claim_state                     | allowed_interpretation                                                                                            | forbidden_interpretation                                                      | required_paper_action                                                                       |
|:-------------------------------------------------------------|:--------------------------------|:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|
| both final splits missing or incomplete                      | not_ready                       | registered final evaluation is pending                                                                            | any predictive-condition or generality claim                                  | report pending Slurm/output status and rerun the frozen evaluator after outputs exist       |
| both final splits pass all P0 gates                          | p0_claim_eligible               | the frozen transport-normalized score predicts residual layer risk on the registered architecture and data finals | claiming final optimizer performance or broader dataset/architecture coverage | report both split summaries, controls, confidence intervals, and the no-retuning boundary   |
| architecture split passes, data split fails residual ranking | data_transport_boundary         | architecture transfer survived, but data-partition transport remains unresolved                                   | broad data-family predictive-condition claim                                  | preserve the negative CIFAR-10 cross-partition result and analyze partition transport terms |
| data split passes, architecture split fails residual ranking | architecture_transport_boundary | data-family transfer survived, but ResNeXt architecture transport remains unresolved                              | broad architecture-family predictive-condition claim                          | preserve the negative ResNeXt50-32x4d result and analyze parameterization transport terms   |
| either final split fails direction guardrail                 | direction_guardrail_failure     | the local spectral/Frobenius direction comparison failed to transport to the final split                          | residual-risk predictor claim, even if residual ranking is positive elsewhere | separate direction failure from scalar residual-score failure                               |
| either final split fails baseline dominance                  | nuisance_proxy_boundary         | the frozen score did not add enough information beyond the early-layer nuisance baseline                          | theory-derived measurable score claim                                         | report early_layer_prior comparison as a failed ablation gate                               |
| both final splits fail any P0 gate                           | local_mechanism_only            | the paper retains local drift-mechanism and falsification evidence only                                           | predictive-condition claim on unseen real tasks                               | keep failed finals in the main evidence ledger and do not retune on them                    |

## Leakage Lock

| locked_item    | locked_value                                                          | forbidden_after_final_outputs                                 | allowed_after_final_outputs                               |
|:---------------|:----------------------------------------------------------------------|:--------------------------------------------------------------|:----------------------------------------------------------|
| primary_score  | condition_score_v5_transport_normalized_amplitude_minus_direction     | changing score weights, features, signs, or aliases           | rerun the same evaluator and report generated rows        |
| final_splits   | ResNeXt50-32x4d CIFAR-100-LT; CIFAR-10 cross partition                | dropping, replacing, or adding splits to rescue the claim     | add future splits only under a new preregistered protocol |
| residual_gate  | Spearman CI lower endpoint above zero on each final split             | using mean-only positivity or a one-split pass as P0 evidence | report partial positives as boundary evidence             |
| direction_gate | threshold accuracy and CI lower endpoint at least 0.8                 | lowering the threshold or merging it into residual ranking    | report a direction failure as a separate obstruction      |
| baseline_gate  | primary beats early_layer_prior on residual ranking and top-k overlap | omitting the early-layer nuisance comparison                  | downgrade to nuisance-proxy boundary if baseline wins     |

Artifacts:
- [final_split_status.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/final_split_status.csv)
- [final_gate_contract.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/final_gate_contract.csv)
- [outcome_interpretation_ladder.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/outcome_interpretation_ladder.csv)
- [leakage_lock.csv](../results/e11_condition_score_v5_protocol/final_interpretation_plan/leakage_lock.csv)
- [config.json](../results/e11_condition_score_v5_protocol/final_interpretation_plan/config.json)
