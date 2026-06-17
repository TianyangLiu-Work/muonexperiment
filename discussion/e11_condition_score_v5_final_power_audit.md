# E11 Condition-Score V5 Final Power Audit

This generated audit is a pre-output detectable-effect contract for the v5
final condition-score test. It reads the validation-frozen score `condition_score_v5_transport_normalized_amplitude_minus_direction` and
the registered final split paths, but it does not inspect, refit, reselect, or
retune on final rows. Its purpose is to define how much residual-Spearman signal
the submitted ResNeXt50-32x4d and CIFAR-10 cross-partition final splits can
resolve before their outputs exist. The audit records Fisher-z resolution,
mean-Spearman MDE, and negative transport boundary wording before either final
split is interpreted.

Current final split outputs generated: 0/2.

Under the reference design, each transfer correlation has about 21
layer points and the final summary averages 9 source/target
transfer-pair correlations. A single transfer correlation needs observed
Spearman at least 0.4317
for its Fisher-z CI lower endpoint to exceed zero. With across-pair Spearman SD
0.2, the final mean Spearman needs to be at least
0.1307 before the
evaluator's summary CI lower endpoint can exclude zero.

## Split Power Design

| split_id                                         | split_role                      | current_output_status   |   reference_points_per_transfer |   reference_transfer_pairs | primary_residual_gate                      |
|:-------------------------------------------------|:--------------------------------|:------------------------|--------------------------------:|---------------------------:|:-------------------------------------------|
| v5_final_architecture_resnext50_32x4d_cifar100lt | v5_final_heldout_architecture   | not_run                 |                              21 |                          9 | mean Spearman CI lower endpoint above zero |
| v5_final_data_cifar10lt_cross_partition          | v5_final_heldout_data_partition | not_run                 |                              21 |                          9 | mean Spearman CI lower endpoint above zero |

## Pairwise Fisher-Z Resolution

|   points |   minimum_observed_spearman_for_ci_low_above_zero |   validation_reference_spearman |   validation_reference_ci_low_at_points | claim_interpretation                                                                                                           |
|---------:|--------------------------------------------------:|--------------------------------:|----------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------|
|       15 |                                            0.5123 |                           0.645 |                                  0.1982 | a generated final split needs an observed residual Spearman at least this large before a Fisher-z pairwise CI can exclude zero |
|       21 |                                            0.4317 |                           0.645 |                                  0.2956 | a generated final split needs an observed residual Spearman at least this large before a Fisher-z pairwise CI can exclude zero |
|       30 |                                            0.3603 |                           0.645 |                                  0.3709 | a generated final split needs an observed residual Spearman at least this large before a Fisher-z pairwise CI can exclude zero |
|       40 |                                            0.3115 |                           0.645 |                                  0.4173 | a generated final split needs an observed residual Spearman at least this large before a Fisher-z pairwise CI can exclude zero |

## Final Mean Spearman MDE

|   transfer_pairs |   assumed_across_pair_spearman_sd |   minimum_mean_spearman_for_ci_low_above_zero | claim_interpretation                                                                                                                              |
|-----------------:|----------------------------------:|----------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------|
|                9 |                              0.05 |                                       0.03267 | if the final mean Spearman is below this row, a non-positive summary CI is not a strong negative result under the assumed across-pair variability |
|                9 |                              0.1  |                                       0.06533 | if the final mean Spearman is below this row, a non-positive summary CI is not a strong negative result under the assumed across-pair variability |
|                9 |                              0.15 |                                       0.098   | if the final mean Spearman is below this row, a non-positive summary CI is not a strong negative result under the assumed across-pair variability |
|                9 |                              0.2  |                                       0.1307  | if the final mean Spearman is below this row, a non-positive summary CI is not a strong negative result under the assumed across-pair variability |
|                9 |                              0.25 |                                       0.1633  | if the final mean Spearman is below this row, a non-positive summary CI is not a strong negative result under the assumed across-pair variability |
|                9 |                              0.3  |                                       0.196   | if the final mean Spearman is below this row, a non-positive summary CI is not a strong negative result under the assumed across-pair variability |

## Interpretation Ladder

| case_id                                  | evidence_condition                                                                                 | allowed_wording                                                                | blocked_wording                                                            |
|:-----------------------------------------|:---------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| V5-PWR-1-pending-outputs                 | one or both registered final split layer tables are absent                                         | registered final power audit only; P0 remains not_ready                        | positive or negative final predictive-condition result                     |
| V5-PWR-2-positive-above-zero-ci          | both splits generated; primary mean Spearman CI lower endpoint > 0; other final gates pass         | narrow frozen-score residual-risk predictor on the registered final splits     | optimizer-performance claim or broader architecture/data-family generality |
| V5-PWR-3-null-below-detectable-scale     | generated split has non-positive CI but mean effect is below audited MDE for observed variability  | underpowered or small-effect final boundary                                    | the frozen score is disproven for all natural tasks                        |
| V5-PWR-4-negative-above-detectable-scale | generated split fails residual ranking with effect scale above audited detectable threshold        | negative architecture/data transport boundary under the frozen protocol        | post-hoc score repair under the same v5 final protocol                     |
| V5-PWR-5-high-heterogeneity              | across-pair Spearman SD is high enough that summary CI is dominated by transfer-pair heterogeneity | transport heterogeneity diagnostic requiring split-specific mechanism analysis | single scalar score success/failure without heterogeneity reporting        |

## Outcome State Machine

| state_id                      | trigger                                                                                              | claim_state                          | required_action                                                                                 |
|:------------------------------|:-----------------------------------------------------------------------------------------------------|:-------------------------------------|:------------------------------------------------------------------------------------------------|
| V5-PWR-S1-not-run             | 0/2 final split summaries generated                                                                  | not_ready                            | wait for Slurm outputs; do not interpret missing rows as evidence                               |
| V5-PWR-S2-partial             | 1/2 final split summaries generated                                                                  | not_ready_partial_family             | report the generated split as progress only; keep P0 closed                                     |
| V5-PWR-S3-complete-positive   | 2/2 final split summaries generated and both pass residual, direction, baseline, and reporting gates | p0_claim_eligible_with_power_context | report observed points, transfer-pair variability, controls, and no-retuning boundary           |
| V5-PWR-S4-complete-small-null | 2/2 generated; residual gate fails but observed scale is below audited MDE                           | underpowered_final_boundary          | downgrade to small-effect/underpowered boundary and add a new preregistered larger final family |
| V5-PWR-S5-complete-negative   | 2/2 generated; residual gate fails at or above audited detectable scale                              | negative_transport_boundary          | preserve failed final as main evidence and do not retune on final rows                          |

Artifacts:
- [split_power_design.csv](../results/e11_condition_score_v5_protocol/final_power_audit/split_power_design.csv)
- [fisher_z_resolution.csv](../results/e11_condition_score_v5_protocol/final_power_audit/fisher_z_resolution.csv)
- [mean_spearman_mde.csv](../results/e11_condition_score_v5_protocol/final_power_audit/mean_spearman_mde.csv)
- [interpretation_ladder.csv](../results/e11_condition_score_v5_protocol/final_power_audit/interpretation_ladder.csv)
- [outcome_state_machine.csv](../results/e11_condition_score_v5_protocol/final_power_audit/outcome_state_machine.csv)
- [config.json](../results/e11_condition_score_v5_protocol/final_power_audit/config.json)
