# E11 CIFAR-100-LT Tuned Benchmark Final Evaluation

This generated evaluator is the fixed post-final analysis path for the
registered tuned benchmark. It reads only the selected final-claim recipe
directories, keeps all four Muon-vs-baseline primary comparisons in the
Holm family, and reports the all-class noninferiority guardrail before any
benchmark wording is allowed.

Current claim state: `not_ready`.
Current final family coverage: `0/6` complete recipe families.

## Gate Report

| gate_id                         | status    | evidence                                                                                                                                                                                 | claim_state   |
|:--------------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|
| TFE-1-evaluator-implemented     | pass      | scripts/e11_evaluate_cifar100_resnet_lt_tuned_benchmark_final.py writes the fixed final evaluator tables                                                                                 | not_ready     |
| TFE-2-selection-gates-ready     | not_ready | TVS-1-validation-grid-complete=not_ready; TVS-2-family-selection=not_ready; TVS-5-occupancy-logging-complete=not_ready; TVS-3-final-seed-quarantine=pass; TVS-4-final-run-plan=not_ready | not_ready     |
| TFE-3-final-output-completeness | not_ready | 0/6 selected recipe families have complete 10-seed final outputs                                                                                                                         | not_ready     |
| TFE-4-primary-holm-family       | not_ready | 0/4 primary comparisons pass Holm                                                                                                                                                        | not_ready     |
| TFE-5-all-class-guardrail       | not_ready | 0/4 all-class guardrails pass                                                                                                                                                            | not_ready     |
| TFE-6-final-claim-state         | not_ready | selected=1/6; final_complete=0/6                                                                                                                                                         | not_ready     |

## Final Run Registry

| recipe_family          | selection_status   | final_output_status   |   group_seed_count |   occupancy_seed_count | missing_files                                                                                                                         |
|:-----------------------|:-------------------|:----------------------|-------------------:|-----------------------:|:--------------------------------------------------------------------------------------------------------------------------------------|
| adamw_cb_loss_tuned    | not_selected       | missing_or_incomplete |                  0 |                      0 | train_trace.csv;class_metrics.csv;group_metrics.csv;summary.csv;pair_summary.csv;occupancy_trace.csv;config.json;setting_metadata.csv |
| adamw_cb_sampler_tuned | not_selected       | missing_or_incomplete |                  0 |                      0 | train_trace.csv;class_metrics.csv;group_metrics.csv;summary.csv;pair_summary.csv;occupancy_trace.csv;config.json;setting_metadata.csv |
| adamw_ce_tuned         | selected           | missing_or_incomplete |                  0 |                      0 | train_trace.csv;class_metrics.csv;group_metrics.csv;summary.csv;pair_summary.csv;occupancy_trace.csv;config.json;setting_metadata.csv |
| ns_muon_cb_tuned       | not_selected       | missing_or_incomplete |                  0 |                      0 | train_trace.csv;class_metrics.csv;group_metrics.csv;summary.csv;pair_summary.csv;occupancy_trace.csv;config.json;setting_metadata.csv |
| ns_muon_matrix_tuned   | not_selected       | missing_or_incomplete |                  0 |                      0 | train_trace.csv;class_metrics.csv;group_metrics.csv;summary.csv;pair_summary.csv;occupancy_trace.csv;config.json;setting_metadata.csv |
| sgd_momentum_ce_tuned  | not_selected       | missing_or_incomplete |                  0 |                      0 | train_trace.csv;class_metrics.csv;group_metrics.csv;summary.csv;pair_summary.csv;occupancy_trace.csv;config.json;setting_metadata.csv |

## Primary Decisions

| comparison_id                                 |   paired_seed_count | mean_few_bacc_diff   | holm_adjusted_one_sided_p   | primary_decision   | mean_all_bacc_diff   | all_class_guardrail   |
|:----------------------------------------------|--------------------:|:---------------------|:----------------------------|:-------------------|:---------------------|:----------------------|
| ns_muon_matrix_tuned_vs_adamw_ce_tuned        |                   0 | n/a                  | n/a                         | not_ready          | n/a                  | not_ready             |
| ns_muon_matrix_tuned_vs_sgd_momentum_ce_tuned |                   0 | n/a                  | n/a                         | not_ready          | n/a                  | not_ready             |
| ns_muon_cb_tuned_vs_adamw_ce_tuned            |                   0 | n/a                  | n/a                         | not_ready          | n/a                  | not_ready             |
| ns_muon_cb_tuned_vs_sgd_momentum_ce_tuned     |                   0 | n/a                  | n/a                         | not_ready          | n/a                  | not_ready             |

## Final Summary Preview

| recipe_family   | frequency_group   | mean_bacc   | ci95_low   | ci95_high   |
|-----------------|-------------------|-------------|------------|-------------|

## Occupancy Summary Preview

| recipe_family   | seed   | occupancy_rows   | batch_few_fraction   | tail_probe_loss   | gradient_momentum_cosine   | drift_ratio   |
|-----------------|--------|------------------|----------------------|-------------------|----------------------------|---------------|

Artifacts:
- [run_registry.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/run_registry.csv)
- [per_seed_final_metrics.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/per_seed_final_metrics.csv)
- [paired_primary_comparisons.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/paired_primary_comparisons.csv)
- [primary_decisions.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/primary_decisions.csv)
- [final_summary.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/final_summary.csv)
- [occupancy_summary.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/occupancy_summary.csv)
- [claim_gate_report.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/claim_gate_report.csv)
- [config.json](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/config.json)

Blocked now: benchmark-performance wording remains unavailable until the
validation-selection gates pass, all six selected recipe families have 10
paired final seeds, all four Holm-adjusted primary comparisons are evaluated,
and the all-class guardrail is reported.
