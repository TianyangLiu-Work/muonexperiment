# E11 CIFAR-100-LT Tuned Benchmark Validation Refresh Firewall

This generated firewall defines what can happen after each partial validation refresh. It is a sequential optional-stopping firewall: new validation outputs can update progress accounting, but they cannot change the registry, selection objective, array order, family selection, final seed set, or benchmark wording.

## Refresh State

| state_id                      | status                       |   completed_validation_settings |   total_validation_settings |   completed_occupancy_traces |   completed_recipe_families |   total_recipe_families | completed_array_prefix   |   next_missing_array_index | prefix_contiguous   | refresh_authority        | final_seed_authority               |
|:------------------------------|:-----------------------------|--------------------------------:|----------------------------:|-----------------------------:|----------------------------:|------------------------:|:-------------------------|---------------------------:|:--------------------|:-------------------------|:-----------------------------------|
| VRF-current-validation-prefix | partial_grid_no_claim_change |                              22 |                         164 |                           22 |                           1 |                       6 | 0..21                    |                         22 | yes                 | progress_accounting_only | blocked_until_TVS_FEP_TFE_FLA_pass |

## Allowed Transition Matrix

| transition_id                      | status    | trigger                                                                 | required_commands                                                                                            | allowed_output                                                      |
|:-----------------------------------|:----------|:------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|
| VRF-A1-prefix-result-refresh       | pass      | new validation summary and occupancy trace for array index 22           | rerun selection, interim audit, leakage audit, and refresh firewall                                          | updated progress accounting with no selection-authority change      |
| VRF-A2-nonprefix-result-quarantine | not_ready | a completed validation row appears beyond the current contiguous prefix | audit Slurm launch history and keep the row out of selection until order is explained                        | diagnostic anomaly report only                                      |
| VRF-A3-full-validation-refresh     | not_ready | TVS-1 and TVS-5 pass with leakage guards still passing                  | rerun selection, power, variance-prior, final-analysis, final-execution, final-eval, and final-launch audits | complete-family selection and final-gate refresh, not final results |
| VRF-A4-final-submit-handoff        | not_ready | TVS-1, TVS-2, TVS-5, leakage guards, FEP, TFE, and FLA all pass         | submit final_claim jobs only through the final-safe-submit path                                              | untouched final seed execution under seed set 20..29                |
| VRF-A5-leakage-guard-repair        | pass      | any TLA guard changes from pass/ready                                   | stop refresh, document violation, and open a fresh preregistered protocol if needed                          | no stronger claim while guard drift remains                         |

## Forbidden Action Matrix

| forbidden_id                                    | status   | forbidden_action                                                                                | violation_response                                                                 |
|:------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| VRF-F1-change-selection-objective               | active   | rewrite the validation objective after seeing partial validation metrics                        | discard current validation/final split and open a fresh preregistered protocol     |
| VRF-F2-metric-dependent-launch-order            | active   | change array order, queue priority, or resubmission policy based on interim leaderboard rank    | quarantine affected rows from selection and audit launch history                   |
| VRF-F3-partial-family-selection                 | active   | select a cross-family winner or baseline from an incomplete recipe family grid                  | rerun only progress audits; do not run final_claim jobs                            |
| VRF-F4-premature-final-submit                   | active   | submit final_claim seed jobs before TVS, FEP, TFE, and FLA gates pass                           | mark final outputs contaminated and require fresh final seeds                      |
| VRF-F5-validation-leaderboard-performance-claim | active   | describe validation leaderboard rows as benchmark performance results                           | downgrade wording to progress accounting and rerun claim-decision audit            |
| VRF-F6-in-place-protocol-repair-after-violation | active   | repair a selection-leakage or final-unblinding violation inside the same validation/final split | register new unspent validation and final splits before renewed performance claims |

## Operating Rule

Every validation refresh must preserve progress accounting only until `TVS-1`, `TVS-2`, and `TVS-5` pass and the leakage guards remain pass/ready. Any non-prefix completion, metric-dependent launch change, partial-family selection, premature final submit, or final-seed contamination requires quarantine and a fresh preregistered protocol before renewed benchmark-performance claims.
