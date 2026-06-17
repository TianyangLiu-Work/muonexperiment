# E11 CIFAR-100-LT Tuned Benchmark Selection

This generated selection report is the no-peeking bridge between the tuned benchmark validation grid and the untouched final claim seeds. It reads only validation summaries under `results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/*`.

Current status: `not_ready` with `0/164` validation settings complete.

## Selection Rule

For each recipe family, select the validation setting with maximum few-class balanced accuracy. Ties are broken by all-class balanced accuracy and then by setting id. The final claim seed set is always `20..29`, and no final output may exist before validation family selection is complete.

## Gate Report

| gate_id                        | status    | evidence                                                              | blocks_final_claim   |
|:-------------------------------|:----------|:----------------------------------------------------------------------|:---------------------|
| TVS-1-validation-grid-complete | not_ready | 0/164 validation settings complete                                    | yes                  |
| TVS-2-family-selection         | not_ready | 0/6 recipe families selected                                          | yes                  |
| TVS-3-final-seed-quarantine    | pass      | no final_claim outputs exist before validation selection              | yes                  |
| TVS-4-final-run-plan           | not_ready | final seed set 20..29 assigned only after validation family selection | yes                  |

## Family Selection

| recipe_family          | selection_status   |   expected_settings |   complete_settings | selected_setting_id   | selected_recipe_name   | primary_few_balanced_accuracy   | tie_break_all_balanced_accuracy   | selection_rule                                             |
|:-----------------------|:-------------------|--------------------:|--------------------:|:----------------------|:-----------------------|:--------------------------------|:----------------------------------|:-----------------------------------------------------------|
| adamw_cb_loss_tuned    | not_ready          |                  24 |                   0 |                       |                        |                                 |                                   | wait for every registered validation setting in the family |
| adamw_cb_sampler_tuned | not_ready          |                   8 |                   0 |                       |                        |                                 |                                   | wait for every registered validation setting in the family |
| adamw_ce_tuned         | not_ready          |                  12 |                   0 |                       |                        |                                 |                                   | wait for every registered validation setting in the family |
| ns_muon_cb_tuned       | not_ready          |                  36 |                   0 |                       |                        |                                 |                                   | wait for every registered validation setting in the family |
| ns_muon_matrix_tuned   | not_ready          |                  72 |                   0 |                       |                        |                                 |                                   | wait for every registered validation setting in the family |
| sgd_momentum_ce_tuned  | not_ready          |                  12 |                   0 |                       |                        |                                 |                                   | wait for every registered validation setting in the family |

## Final Claim Plan

| recipe_family          | final_status   | selected_setting_id   | selected_recipe_name   | final_seed_set   | planned_output_dir   | tuning_allowed   |
|:-----------------------|:---------------|:----------------------|:-----------------------|:-----------------|:---------------------|:-----------------|
| adamw_cb_loss_tuned    | not_ready      |                       |                        |                  |                      | no               |
| adamw_cb_sampler_tuned | not_ready      |                       |                        |                  |                      | no               |
| adamw_ce_tuned         | not_ready      |                       |                        |                  |                      | no               |
| ns_muon_cb_tuned       | not_ready      |                       |                        |                  |                      | no               |
| ns_muon_matrix_tuned   | not_ready      |                       |                        |                  |                      | no               |
| sgd_momentum_ce_tuned  | not_ready      |                       |                        |                  |                      | no               |

## Claim Boundary

The current result is not a final-performance benchmark result. It is a selection audit. A practical optimizer claim remains blocked until the validation grid is complete, one recipe per family is selected by the frozen rule, and the final seed set `20..29` is run pairwise without tuning.
