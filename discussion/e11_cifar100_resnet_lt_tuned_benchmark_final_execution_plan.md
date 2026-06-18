# E11 CIFAR-100-LT Tuned Benchmark Final Execution Plan

This generated plan is a no-side-effect execution contract for the untouched
final-claim seeds. It exists so the final benchmark can be launched without
hand-written commands after validation selection completes. It does not submit
jobs and it does not inspect or create final output files.

Current final-submit status: `not_ready`.

## Gate Matrix

| gate_id                          | status    | evidence                                                                                                                                                                                 | blocks_final_submit   |
|:---------------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|
| FEP-1-selection-gates-ready      | not_ready | TVS-1-validation-grid-complete=not_ready; TVS-2-family-selection=not_ready; TVS-5-occupancy-logging-complete=not_ready; TVS-3-final-seed-quarantine=pass; TVS-4-final-run-plan=not_ready | yes                   |
| FEP-2-final-analysis-plan-linked | pass      | FAP-1-final-quarantine=pass; primary_family_size=4                                                                                                                                       | yes                   |
| FEP-3-runner-contract-present    | pass      | runner=scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py; sbatch=scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch                                                   | yes                   |
| FEP-4-final-output-quarantine    | pass      | no final_claim files exist                                                                                                                                                               | yes                   |
| FEP-5-all-families-ready         | not_ready | 2/6 families ready_for_final_run                                                                                                                                                         | yes                   |

## Final Family Run Plan

|   final_index | recipe_family          | run_status                         | selected_setting_id                            | selected_recipe_name                       | final_seed_set   | planned_output_dir                                                               | runner_command                                                                                                                                                                                        | tuning_allowed   |
|--------------:|:-----------------------|:-----------------------------------|:-----------------------------------------------|:-------------------------------------------|:-----------------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
|             0 | adamw_cb_loss_tuned    | blocked_until_family_selected      |                                                |                                            |                  |                                                                                  |                                                                                                                                                                                                       | no               |
|             1 | adamw_cb_sampler_tuned | blocked_until_family_selected      |                                                |                                            |                  |                                                                                  |                                                                                                                                                                                                       | no               |
|             2 | adamw_ce_tuned         | blocked_until_selection_gates_pass | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0         | adamw_ce_tuned_lr1e-3_wd5e-4_warm0         | 20..29           | results/e11_cifar100_resnet_lt_tuned_benchmark/final_claim/adamw_ce_tuned        | /data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --phase final_claim --final-family adamw_ce_tuned --device cuda --no-download --progress        | no               |
|             3 | ns_muon_cb_tuned       | blocked_until_family_selected      |                                                |                                            |                  |                                                                                  |                                                                                                                                                                                                       | no               |
|             4 | ns_muon_matrix_tuned   | blocked_until_family_selected      |                                                |                                            |                  |                                                                                  |                                                                                                                                                                                                       | no               |
|             5 | sgd_momentum_ce_tuned  | blocked_until_selection_gates_pass | TBV-sgd_momentum_ce_tuned-lr0p3-wd1e-3-warm500 | sgd_momentum_ce_tuned_lr0p3_wd1e-3_warm500 | 20..29           | results/e11_cifar100_resnet_lt_tuned_benchmark/final_claim/sgd_momentum_ce_tuned | /data/conda_envs/SpatialQuantization/bin/python scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --phase final_claim --final-family sgd_momentum_ce_tuned --device cuda --no-download --progress | no               |

## Slurm Submit Plan

| plan_id                        | submission_status   | array_expression   | submit_command   | expected_outputs                                                                                                                             | post_run_action                                                          | submits_jobs   |
|:-------------------------------|:--------------------|:-------------------|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|:---------------|
| FEP-SLURM-1-final-family-array | not_ready           |                    |                  | train_trace.csv; class_metrics.csv; group_metrics.csv; summary.csv; pair_summary.csv; occupancy_trace.csv; config.json; setting_metadata.csv | run final statistical evaluator and e11-check before any benchmark claim | no             |

## Operating Rule

Do not run `scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch`
until `FEP-1`, `FEP-2`, `FEP-3`, `FEP-4`, and `FEP-5` all pass. The runner also
checks the validation-selection gates at execution time, so premature final submission fails closed. It will not produce final-claim rows from a premature
submit. After final jobs finish, the only allowed analysis path is the pre-registered paired-seed,
Holm-adjusted final-analysis plan.
