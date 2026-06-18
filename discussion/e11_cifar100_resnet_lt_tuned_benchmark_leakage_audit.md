# E11 CIFAR-100-LT Tuned Benchmark Leakage Audit

This generated audit addresses selection leakage and optional-stopping risk during the long-running tuned validation grid. It is stricter than the interim leaderboard: partial validation observations are visible for progress accounting, but they cannot change the registry, selection rule, Slurm array order, or final seed quarantine.

## Leakage Guard Matrix

| guard_id                            | status   | evidence                                                                                                                                                                                           | leakage_risk_controlled                                               | claim_effect                                                    |
|:------------------------------------|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------|:----------------------------------------------------------------|
| TLA-1-selection-rule-frozen         | pass     | selection_rules.csv contains `Primary validation objective is few balanced accuracy`                                                                                                               | metric-dependent rule rewriting after partial observations            | blocks any stronger selection wording if this fails             |
| TLA-2-validation-only-input-surface | pass     | settings_registry.csv has phase=validation_tuning and seed_set=10..14 for every array row                                                                                                          | final-seed information entering validation selection                  | final seed set 20..29 remains unavailable to selection          |
| TLA-3-array-order-contiguous        | pass     | 164 registry rows with contiguous array_index 0..163                                                                                                                                               | post-hoc array reordering to prioritize favorable settings            | Slurm array order remains auditable against the frozen registry |
| TLA-4-partial-grid-selection-block  | pass     | 30/164 settings complete and 2/6 families selected                                                                                                                                                 | turning partial leaderboard observations into recipe-family selection | partial leaderboard is progress accounting only                 |
| TLA-5-final-output-quarantine       | pass     | no final_claim outputs exist                                                                                                                                                                       | validation rules influenced by final seed outcomes                    | final-performance wording remains blocked                       |
| TLA-6-launch-history-auditable      | pass     | launch_history.csv records planned_array_indices=0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16 / 0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16 / 28;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;47 | unrecorded GPU launches after seeing partial metrics                  | new validation launches must remain queue-audited               |

## Observed Surface

| observed_surface                                           |   observed_rows |   completed_rows | contains_final_seed_data   | allowed_use                                            | blocked_use                                                      |
|:-----------------------------------------------------------|----------------:|-----------------:|:---------------------------|:-------------------------------------------------------|:-----------------------------------------------------------------|
| validation_selection/run_registry.csv                      |             164 |               30 | no                         | progress accounting and complete-family selection only | final-performance claim or partial-family selection              |
| interim_validation_audit/completed_setting_leaderboard.csv |              30 |               30 | no                         | reviewer-facing partial-grid readout                   | changing validation grid, launch order, or frozen selection rule |
| slurm_launch_audit/latest_queue_snapshot.csv               |               0 |                0 | no                         | queue-capacity and inflight-array audit                | metric-dependent scheduling decisions                            |

## Immutability Contract

| contract_id                 | artifact                                                                        | required_anchor                                       | status   | evidence                                                                                                              |
|:----------------------------|:--------------------------------------------------------------------------------|:------------------------------------------------------|:---------|:----------------------------------------------------------------------------------------------------------------------|
| IMM-1-registry-cardinality  | results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv            | 164 validation_tuning rows                            | pass     | 164 rows                                                                                                              |
| IMM-2-family-grid-coverage  | results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv            | six preregistered recipe families                     | pass     | adamw_cb_loss_tuned;adamw_cb_sampler_tuned;adamw_ce_tuned;ns_muon_cb_tuned;ns_muon_matrix_tuned;sgd_momentum_ce_tuned |
| IMM-3-selection-rule-anchor | results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/selection_rules.csv     | Primary validation objective is few balanced accuracy | pass     | rule text present                                                                                                     |
| IMM-4-final-seed-anchor     | results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/seed_split_contract.csv | final_claim seed_set 20..29 tuning_allowed no         | pass     | validated by selection and protocol gates                                                                             |

## Operating Rule

Seeing validation summaries before `164/164` complete does not authorize a new recipe grid, a changed selection rule, a changed final seed set, or a benchmark-level optimizer claim. Any repaired or expanded tuned benchmark must open a new preregistered protocol with fresh unspent validation/final splits.
