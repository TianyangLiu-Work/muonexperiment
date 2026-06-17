# E11 Artifact Manifest

This generated manifest documents the current reproducibility boundary for E11. It separates the head-to-tail paper evidence from background condition-geometry guardrails, records what should be committed as evidence, and lists which key CSV tables back the paper-facing claims.

## Artifact Directories

| path                   | role                                                                                  | commit_policy                            | size     |
|:-----------------------|:--------------------------------------------------------------------------------------|:-----------------------------------------|:---------|
| e11_condition_geometry | Source package for E11 problems, optimizers, diagnostics, reporting, and plotting.    | commit                                   | 396.9 KB |
| scripts                | Experiment runners, artifact writers, and validation scripts.                         | commit E11 scripts                       | 2.0 MB   |
| tests                  | Smoke tests, diagnostics tests, reporting tests, and validator guardrail tests.       | commit                                   | 64.8 KB  |
| discussion             | Generated head-to-tail paper evidence plus legacy condition-geometry guardrail notes. | commit                                   | 1.1 MB   |
| results                | Generated CSV evidence used by the paper table, discussion artifacts, and validator.  | commit current E11 evidence set          | 267.5 MB |
| figures                | Static figures used by the paper draft and paper-facing Markdown artifacts.           | commit static E11 figures                | 31.5 MB  |
| configs                | Experiment configuration snapshots.                                                   | commit                                   | 316 B    |
| paper                  | Head-to-tail LaTeX paper draft, generated paper table, and experiment triage notes.   | commit source and selected rendered PDFs | 5.6 MB   |
| Makefile               | Thin reproducibility entrypoint for E11 artifact generation and validation commands.  | commit                                   | 13.5 KB  |
| .gitignore             | Keeps local caches, datasets, and generated videos out of default commits.            | commit                                   | 479 B    |
| .gitattributes         | Marks generated evidence artifacts and binary files for cleaner GitHub review.        | commit                                   | 667 B    |

## Key Quantitative Tables

| path                                                                                                       |   rows | size      |
|:-----------------------------------------------------------------------------------------------------------|-------:|:----------|
| results/e11/step_metrics.csv                                                                               |   2880 | 20.9 MB   |
| results/e11/activation_perturbation_summary.csv                                                            |     60 | 12.6 KB   |
| results/e11_head_tail_interference/step_metrics.csv                                                        |    320 | 142.9 KB  |
| results/e11_head_tail_interference/pair_summary.csv                                                        |      2 | 2.0 KB    |
| results/e11_long_tail_one_step/step_metrics.csv                                                            |     40 | 26.4 KB   |
| results/e11_long_tail_one_step/pair_summary.csv                                                            |      1 | 5.9 KB    |
| results/e11_long_tail_one_step/layer_metrics.csv                                                           |     80 | 7.0 KB    |
| results/e11_cifar100_lt_one_step/step_metrics.csv                                                          |     10 | 10.3 KB   |
| results/e11_cifar100_lt_one_step/pair_summary.csv                                                          |      1 | 6.0 KB    |
| results/e11_cifar100_lt_one_step/layer_metrics.csv                                                         |     20 | 2.0 KB    |
| results/e11_cifar100_resnet_one_step/step_metrics.csv                                                      |     20 | 19.5 KB   |
| results/e11_cifar100_resnet_one_step/pair_summary.csv                                                      |      1 | 6.1 KB    |
| results/e11_cifar100_resnet_one_step/layer_metrics.csv                                                     |    420 | 33.5 KB   |
| results/e11_cifar100_resnet_one_step_rho002/step_metrics.csv                                               |     20 | 19.6 KB   |
| results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv                                               |      1 | 6.1 KB    |
| results/e11_cifar100_resnet_one_step_rho002/layer_metrics.csv                                              |    420 | 33.5 KB   |
| results/e11_cifar100_resnet_checkpoint_sweep/step_metrics.csv                                              |     80 | 75.5 KB   |
| results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv                                              |      4 | 12.0 KB   |
| results/e11_cifar100_resnet_checkpoint_sweep/layer_metrics.csv                                             |   1680 | 141.4 KB  |
| results/e11_cifar100_resnet_condition_proxy_scatter/scatter_points.csv                                     |     40 | 23.6 KB   |
| results/e11_cifar100_resnet_condition_proxy_scatter/summary.csv                                            |      6 | 784 B     |
| results/e11_cifar100_resnet_condition_proxy_scatter/layer_summary.csv                                      |     84 | 9.5 KB    |
| results/e11_cifar100_resnet_fc_condition_scatter/step_metrics.csv                                          |     80 | 53.6 KB   |
| results/e11_cifar100_resnet_fc_condition_scatter/summary.csv                                               |      4 | 12.3 KB   |
| results/e11_cifar100_resnet_fc_condition_scatter/condition_metrics.csv                                     |     40 | 5.2 KB    |
| results/e11_cifar100_resnet_tail_quality_control/step_metrics.csv                                          |     60 | 57.3 KB   |
| results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv                                          |      3 | 10.1 KB   |
| results/e11_cifar100_resnet_tail_quality_control/layer_metrics.csv                                         |   1260 | 107.0 KB  |
| results/e11_cifar100_resnet_imbalance_sweep/step_metrics.csv                                               |     24 | 23.8 KB   |
| results/e11_cifar100_resnet_imbalance_sweep/pair_summary.csv                                               |      4 | 11.7 KB   |
| results/e11_cifar100_resnet_imbalance_sweep/layer_metrics.csv                                              |    504 | 44.2 KB   |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/metrics.csv                                             |    420 | 405.8 KB  |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/paired_metrics.csv                                      |    210 | 70.4 KB   |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/summary.csv                                             |     21 | 10.9 KB   |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv                                     |      1 | 1.4 KB    |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/metrics.csv                                    |   1260 | 1.2 MB    |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/paired_metrics.csv                             |    630 | 209.1 KB  |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/layer_summary.csv                              |     63 | 18.2 KB   |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/checkpoint_summary.csv                         |      3 | 1.1 KB    |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/prediction_pairs.csv                           |     36 | 7.2 KB    |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/prediction_summary.csv                         |      6 | 1.6 KB    |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/residual_prediction_pairs.csv                  |     24 | 5.6 KB    |
| results/e11_cifar100_resnet_layer_jvp_checkpoint_prediction/residual_prediction_summary.csv                |      4 | 1.2 KB    |
| results/e11_cifar100_resnet_condition_score_audit/raw_score_pairs.csv                                      |     60 | 10.6 KB   |
| results/e11_cifar100_resnet_condition_score_audit/raw_score_summary.csv                                    |     10 | 2.0 KB    |
| results/e11_cifar100_resnet_condition_score_audit/residual_score_pairs.csv                                 |     30 | 6.8 KB    |
| results/e11_cifar100_resnet_condition_score_audit/residual_score_summary.csv                               |      5 | 1.3 KB    |
| results/e11_cifar100_resnet_condition_score_protocol/score_registry.csv                                    |      5 | 1.9 KB    |
| results/e11_cifar100_resnet_condition_score_protocol/split_registry.csv                                    |      6 | 1.5 KB    |
| results/e11_cifar100_resnet_condition_score_protocol/acceptance_gates.csv                                  |      6 | 1.7 KB    |
| results/e11_cifar100_resnet_condition_score_next/score_pairs.csv                                           |     30 | 5.8 KB    |
| results/e11_cifar100_resnet_condition_score_next/score_summary.csv                                         |      5 | 1.5 KB    |
| results/e11_cifar100_resnet_condition_score_next/calibration_coefficients.csv                              |     27 | 2.1 KB    |
| results/e11_cifar100_resnet_condition_score_next/gate_report.csv                                           |      6 | 874 B     |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/metrics.csv                          |   1110 | 1.0 MB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/paired_metrics.csv                   |    555 | 184.8 KB  |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/layer_summary.csv                    |    111 | 32.4 KB   |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/checkpoint_summary.csv               |      3 | 1.2 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/prediction_pairs.csv                 |     36 | 7.2 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/prediction_summary.csv               |      6 | 1.7 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/residual_prediction_pairs.csv        |     24 | 5.6 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_architecture/residual_prediction_summary.csv      |      4 | 1.2 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/metrics.csv                                  |   1260 | 885.5 KB  |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/paired_metrics.csv                           |    630 | 206.5 KB  |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/layer_summary.csv                            |     63 | 18.0 KB   |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/checkpoint_summary.csv                       |      3 | 1.1 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/prediction_pairs.csv                         |     36 | 7.2 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/prediction_summary.csv                       |      6 | 1.7 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/residual_prediction_pairs.csv                |     24 | 5.6 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_data/residual_prediction_summary.csv              |      4 | 1.2 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_score_pairs.csv          |     90 | 29.5 KB   |
| results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_score_summary.csv        |     10 | 3.6 KB    |
| results/e11_cifar100_resnet_condition_score_next/heldout_score_evaluation/heldout_gate_report.csv          |      9 | 1.4 KB    |
| results/e11_condition_score_theory_bridge/score_target_register.csv                                        |      4 | 1.9 KB    |
| results/e11_condition_score_theory_bridge/fresh_protocol_requirements.csv                                  |      6 | 2.0 KB    |
| results/e11_condition_score_fresh_protocol/quarantine_register.csv                                         |      2 | 846 B     |
| results/e11_condition_score_fresh_protocol/score_freeze_registry.csv                                       |      4 | 1.5 KB    |
| results/e11_condition_score_fresh_protocol/fresh_split_registry.csv                                        |      4 | 1.3 KB    |
| results/e11_condition_score_fresh_protocol/acceptance_gates.csv                                            |      6 | 1.5 KB    |
| results/e11_condition_score_fresh_protocol/protocol_status.csv                                             |      4 | 552 B     |
| results/e11_condition_score_fresh_protocol/fresh_score_evaluation/fresh_gate_report.csv                    |      9 | 1.5 KB    |
| results/e11_condition_score_failure_mechanism_audit/score_outcome_matrix.csv                               |     18 | 5.7 KB    |
| results/e11_condition_score_failure_mechanism_audit/split_obstruction_taxonomy.csv                         |      4 | 1.5 KB    |
| results/e11_condition_score_failure_mechanism_audit/resnet50_stage_reversal.csv                            |     18 | 1.5 KB    |
| results/e11_condition_score_v4_protocol/spent_split_register.csv                                           |      4 | 1.7 KB    |
| results/e11_condition_score_v4_protocol/score_axis_registry.csv                                            |      6 | 2.5 KB    |
| results/e11_condition_score_v4_protocol/unspent_split_registry.csv                                         |      4 | 1.3 KB    |
| results/e11_condition_score_v4_protocol/acceptance_gates.csv                                               |      7 | 2.0 KB    |
| results/e11_condition_score_v4_protocol/protocol_status.csv                                                |      4 | 567 B     |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/metrics.csv                            |   1260 | 1.2 MB    |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/paired_metrics.csv                     |    630 | 210.5 KB  |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/layer_summary.csv                      |     63 | 18.5 KB   |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/checkpoint_summary.csv                 |      3 | 1.2 KB    |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/prediction_pairs.csv                   |     36 | 7.2 KB    |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/prediction_summary.csv                 |      6 | 1.6 KB    |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/residual_prediction_pairs.csv          |     24 | 5.6 KB    |
| results/e11_condition_score_v4_protocol/validation_cifar100_rotated/residual_prediction_summary.csv        |      4 | 1.2 KB    |
| results/e11_condition_score_v4_protocol/validation_score_freeze/score_formula_registry.csv                 |      8 | 2.5 KB    |
| results/e11_condition_score_v4_protocol/validation_score_freeze/freeze_status.csv                          |      4 | 553 B     |
| results/e11_condition_score_v4_protocol/validation_score_freeze/validation_gate_report.csv                 |      6 | 870 B     |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/metrics.csv                     |    648 | 632.6 KB  |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/paired_metrics.csv              |    324 | 108.2 KB  |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/layer_summary.csv               |    162 | 46.8 KB   |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/checkpoint_summary.csv          |      3 | 1.2 KB    |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/prediction_pairs.csv            |     36 | 7.2 KB    |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/prediction_summary.csv          |      6 | 1.7 KB    |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/residual_prediction_pairs.csv   |     24 | 5.5 KB    |
| results/e11_condition_score_v4_protocol/final_architecture_wide_resnet50_2/residual_prediction_summary.csv |      4 | 1.2 KB    |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/metrics.csv                               |   1260 | 892.8 KB  |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/paired_metrics.csv                        |    630 | 208.3 KB  |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/layer_summary.csv                         |     63 | 18.3 KB   |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/checkpoint_summary.csv                    |      3 | 1.2 KB    |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/prediction_pairs.csv                      |     36 | 7.2 KB    |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/prediction_summary.csv                    |      6 | 1.6 KB    |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/residual_prediction_pairs.csv             |     24 | 5.6 KB    |
| results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/residual_prediction_summary.csv           |      4 | 1.1 KB    |
| results/e11_condition_score_v4_protocol/final_score_evaluation/final_score_pairs.csv                       |     72 | 26.3 KB   |
| results/e11_condition_score_v4_protocol/final_score_evaluation/final_score_summary.csv                     |      8 | 3.2 KB    |
| results/e11_condition_score_v4_protocol/final_score_evaluation/final_gate_report.csv                       |      7 | 1.2 KB    |
| results/e11_condition_score_v4_failure_mechanism_audit/final_outcome_matrix.csv                            |      9 | 2.0 KB    |
| results/e11_condition_score_v4_failure_mechanism_audit/axis_all_layer_summary.csv                          |      8 | 2.1 KB    |
| results/e11_condition_score_v4_failure_mechanism_audit/axis_transfer_pair_scores.csv                       |     72 | 13.4 KB   |
| results/e11_condition_score_v4_failure_mechanism_audit/axis_pair_summary.csv                               |      8 | 1.5 KB    |
| results/e11_condition_score_v4_failure_mechanism_audit/top5_stage_summary.csv                              |     31 | 5.7 KB    |
| results/e11_condition_score_v4_failure_mechanism_audit/obstruction_summary.csv                             |      4 | 1.4 KB    |
| results/e11_condition_score_v5_theory_protocol/theory_term_register.csv                                    |      6 | 2.8 KB    |
| results/e11_condition_score_v5_theory_protocol/score_contract.csv                                          |      6 | 1.8 KB    |
| results/e11_condition_score_v5_theory_protocol/spent_evidence_policy.csv                                   |      6 | 1.9 KB    |
| results/e11_condition_score_v5_theory_protocol/unspent_split_requirements.csv                              |      3 | 1.3 KB    |
| results/e11_condition_score_v5_theory_protocol/acceptance_gates.csv                                        |      6 | 1.6 KB    |
| results/e11_condition_score_v5_theory_to_score_map/theorem_proxy_map.csv                                   |      5 | 2.5 KB    |
| results/e11_condition_score_v5_theory_to_score_map/score_lineage.csv                                       |      7 | 1.5 KB    |
| results/e11_condition_score_v5_theory_to_score_map/transport_normalization_contract.csv                    |      5 | 1.8 KB    |
| results/e11_condition_score_v5_theory_to_score_map/falsifiable_predictions.csv                             |      5 | 1.6 KB    |
| results/e11_condition_score_v5_theory_to_score_map/ablation_matrix.csv                                     |      5 | 1.4 KB    |
| results/e11_condition_score_v5_theory_to_score_map/claim_readiness_ledger.csv                              |      5 | 610 B     |
| results/e11_submission_repro_audit/toolchain_status.csv                                                    |      7 | 580 B     |
| results/e11_submission_repro_audit/pdf_artifact_checks.csv                                                 |      2 | 313 B     |
| results/e11_submission_repro_audit/source_package_manifest.csv                                             |     11 | 1.8 KB    |
| results/e11_submission_repro_audit/build_gate_summary.csv                                                  |      6 | 975 B     |
| results/e11_condition_score_v5_protocol/validation_score_freeze/score_formula_registry.csv                 |      7 | 2.6 KB    |
| results/e11_condition_score_v5_protocol/validation_score_freeze/freeze_status.csv                          |      5 | 653 B     |
| results/e11_condition_score_v5_protocol/validation_score_freeze/validation_gate_report.csv                 |      6 | 814 B     |
| results/e11_natural_head_tail_boundary/search_registry.csv                                                 |     11 | 2.5 KB    |
| results/e11_natural_head_tail_boundary/primary_drift_scan.csv                                              |    169 | 88.7 KB   |
| results/e11_natural_head_tail_boundary/secondary_outcome_scan.csv                                          |    111 | 60.3 KB   |
| results/e11_natural_head_tail_boundary/boundary_summary.csv                                                |      4 | 839 B     |
| results/e11_natural_head_tail_boundary/candidate_negative_cases.csv                                        |     58 | 19.5 KB   |
| results/e11_natural_negative_search_protocol/audit_baseline.csv                                            |      3 | 846 B     |
| results/e11_natural_negative_search_protocol/search_space_registry.csv                                     |      4 | 2.1 KB    |
| results/e11_natural_negative_search_protocol/metric_contract.csv                                           |      5 | 2.0 KB    |
| results/e11_natural_negative_search_protocol/stopping_rules.csv                                            |      6 | 1.5 KB    |
| results/e11_natural_negative_search_protocol/acceptance_gates.csv                                          |      6 | 1.5 KB    |
| results/e11_natural_negative_search_protocol/claim_ladder.csv                                              |      5 | 1.3 KB    |
| results/e11_natural_negative_search_protocol/protocol_status.csv                                           |      7 | 1.1 KB    |
| results/e11_natural_negative_search_protocol/phase1_power_audit/power_grid.csv                             |     72 | 6.3 KB    |
| results/e11_natural_negative_search_protocol/phase1_power_audit/minimum_detectable_effect.csv              |     36 | 7.1 KB    |
| results/e11_natural_negative_search_protocol/phase1_power_audit/interpretation_ladder.csv                  |      4 | 919 B     |
| results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/settings_registry.csv              |     12 | 10.2 KB   |
| results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/settings_registry.csv               |      8 | 4.7 KB    |
| results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/settings_registry.csv            |      6 | 4.9 KB    |
| results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/run_registry.csv               |      3 | 1.1 KB    |
| results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/primary_decisions.csv          |     26 | 8.6 KB    |
| results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/gate_report.csv                |      5 | 751 B     |
| results/e11_top_conference_gap_register/gap_register.csv                                                   |      7 | 13.7 KB   |
| results/e11_cifar100_resnet_lt_standard_eval/train_trace.csv                                               |    110 | 5.7 KB    |
| results/e11_cifar100_resnet_lt_standard_eval/class_metrics.csv                                             |   1000 | 58.6 KB   |
| results/e11_cifar100_resnet_lt_standard_eval/group_metrics.csv                                             |     40 | 4.2 KB    |
| results/e11_cifar100_resnet_lt_standard_eval/summary.csv                                                   |      4 | 1.3 KB    |
| results/e11_cifar100_resnet_lt_standard_eval/class_summary.csv                                             |    100 | 17.3 KB   |
| results/e11_cifar100_resnet_lt_recipe_benchmark/train_trace.csv                                            |     90 | 7.7 KB    |
| results/e11_cifar100_resnet_lt_recipe_benchmark/class_metrics.csv                                          |   1500 | 110.1 KB  |
| results/e11_cifar100_resnet_lt_recipe_benchmark/group_metrics.csv                                          |     60 | 7.0 KB    |
| results/e11_cifar100_resnet_lt_recipe_benchmark/summary.csv                                                |     12 | 3.4 KB    |
| results/e11_cifar100_resnet_lt_recipe_benchmark/pair_summary.csv                                           |      8 | 2.5 KB    |
| results/e11_cifar100_resnet_lt_muon_final_benchmark/train_trace.csv                                        |     54 | 4.8 KB    |
| results/e11_cifar100_resnet_lt_muon_final_benchmark/class_metrics.csv                                      |    900 | 67.9 KB   |
| results/e11_cifar100_resnet_lt_muon_final_benchmark/group_metrics.csv                                      |     36 | 4.4 KB    |
| results/e11_cifar100_resnet_lt_muon_final_benchmark/summary.csv                                            |     12 | 3.4 KB    |
| results/e11_cifar100_resnet_lt_muon_final_benchmark/pair_summary.csv                                       |      8 | 2.5 KB    |
| results/e11_cifar100_resnet_practical_muon_bridge/metrics.csv                                              |    240 | 257.6 KB  |
| results/e11_cifar100_resnet_practical_muon_bridge/paired_metrics.csv                                       |    180 | 69.1 KB   |
| results/e11_cifar100_resnet_practical_muon_bridge/summary.csv                                              |      6 | 5.2 KB    |
| results/e11_cifar100_resnet_practical_muon_bridge/step_summary.csv                                         |     18 | 5.2 KB    |
| results/e11_long_tail_imbalance_ablation/step_metrics.csv                                                  |    160 | 81.7 KB   |
| results/e11_long_tail_imbalance_ablation/summary.csv                                                       |      4 | 6.8 KB    |
| results/e11_long_tail_checkpoint_sweep/step_metrics.csv                                                    |    200 | 121.5 KB  |
| results/e11_long_tail_checkpoint_sweep/summary.csv                                                         |      5 | 10.0 KB   |
| results/e11_long_tail_class_partition_sweep/step_metrics.csv                                               |    200 | 124.6 KB  |
| results/e11_long_tail_class_partition_sweep/summary.csv                                                    |      5 | 10.8 KB   |
| results/e11_long_tail_rho_sweep/step_metrics.csv                                                           |    200 | 123.2 KB  |
| results/e11_long_tail_rho_sweep/summary.csv                                                                |      5 | 10.1 KB   |
| results/e11_long_tail_muon_bridge/step_metrics.csv                                                         |     80 | 42.2 KB   |
| results/e11_long_tail_muon_bridge/pair_summary.csv                                                         |      3 | 2.1 KB    |
| results/e11_local_linearization/summary.csv                                                                |      4 | 1.0 KB    |
| results/e11_long_tail_practical_muon_bridge/step_metrics.csv                                               |    480 | 253.2 KB  |
| results/e11_long_tail_practical_muon_bridge/step_summary.csv                                               |     18 | 3.3 KB    |
| results/e11_long_tail_practical_muon_bridge/summary.csv                                                    |      3 | 2.2 KB    |
| results/e11_long_tail_muon_state_source_control/step_metrics.csv                                           |    960 | 522.9 KB  |
| results/e11_long_tail_muon_state_source_control/summary.csv                                                |      6 | 3.5 KB    |
| results/e11_long_tail_practical_training/step_metrics.csv                                                  |   3240 | 1.0 MB    |
| results/e11_long_tail_practical_training/summary.csv                                                       |      1 | 1.1 KB    |
| results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv                                        |      4 | 2.0 KB    |
| results/e11_long_tail_forgetting/step_metrics.csv                                                          |    360 | 93.7 KB   |
| results/e11_long_tail_forgetting/summary.csv                                                               |      1 | 1.6 KB    |
| results/e11_long_tail_layerwise/metrics.csv                                                                |     80 | 36.7 KB   |
| results/e11_long_tail_layerwise/summary.csv                                                                |      2 | 2.2 KB    |
| results/e11_equal_update/step_metrics.csv                                                                  |   2880 | 21.0 MB   |
| results/e11_equal_update/activation_perturbation_summary.csv                                               |     60 | 12.6 KB   |
| results/e11_overlap_followup/step_metrics.csv                                                              |   1530 | 8.2 MB    |
| results/e11_hyperparam_sweep/equal_step_metrics.csv                                                        |   3360 | 21.0 MB   |
| results/e11_target_update_sweep/step_metrics.csv                                                           |   3360 | 21.1 MB   |
| results/e11_mlp_width_sweep/step_metrics.csv                                                               |   3300 | 14.3 MB   |
| results/e11_mlp_per_layer_control/step_metrics.csv                                                         |   1100 | 4.8 MB    |
| results/e11_mlp_layer_hybrid/step_metrics.csv                                                              |   2200 | 9.6 MB    |
| results/e11_mnist_mlp_probe/step_metrics.csv                                                               |    216 | 4.4 MB    |
| results/e11_deep_mnist_mlp_probe/step_metrics.csv                                                          |    432 | 12.4 MB   |
| results/e11_mnist_patch_probe/step_metrics.csv                                                             |    288 | 839.6 KB  |
| results/e11_mnist_conv_probe/step_metrics.csv                                                              |    288 | 837.9 KB  |
| results/e11_stateless_direction_ablation/stateless_direction_rows.csv                                      |   1890 | 2.8 MB    |
| results/e11_stateless_optimizer_trajectory/stateless_optimizer_step_metrics.csv                            |    840 | 5.3 MB    |
| results/e11_spectral_allocation_probe/probe_rows.csv                                                       |    720 | 192.6 KB  |
| results/e11_singular_vector_trajectory/update_subspace_rows.csv                                            |    725 | 76.8 KB   |
| results/e11_singular_vector_swap_probe/swap_probe_rows.csv                                                 |    320 | 63.4 KB   |
| results/e11_natural_update_swap_probe/natural_update_swap_rows.csv                                         |   4800 | 968.1 KB  |
| results/e11_optimizer_switch_probe/optimizer_switch_rows.csv                                               |    360 | 72.2 KB   |
| results/e11_optimizer_switch_reset_control/optimizer_switch_reset_rows.csv                                 |    540 | 92.0 KB   |
| results/e11_optimizer_switch_horizon_sweep/optimizer_switch_horizon_rows.csv                               |    480 | 99.0 KB   |
| results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_rows.csv                                         |    360 | 64.6 KB   |
| results/e11_boundary_predictor/boundary_predictor_rows.csv                                                 |   1500 | 1017.5 KB |
| results/e11_mechanism_boundary/mechanism_boundary_map.csv                                                  |     14 | 5.2 KB    |
| results/e11_cross_task_signature/cross_task_signature_summary.csv                                          |     14 | 3.6 KB    |
| results/e11_boundary_predictor/boundary_predictor_uncertainty.csv                                          |      8 | 1.6 KB    |
| results/e11_head_tail_alignment_ablation/step_metrics.csv                                                  |   1000 | 149.9 KB  |
| results/e11_head_tail_alignment_ablation/summary.csv                                                       |      2 | 1.2 KB    |

## Key Paper Documents

| path                                                                        | size    |
|:----------------------------------------------------------------------------|:--------|
| README_E11.md                                                               | 65.7 KB |
| discussion/e11_paper_skeleton.md                                            | 10.9 KB |
| discussion/e11_main_paper_package.md                                        | 22.8 KB |
| discussion/e11_main_figure_captions.md                                      | 6.5 KB  |
| discussion/e11_notation_glossary.md                                         | 8.7 KB  |
| discussion/e11_quantitative_claim_ledger.md                                 | 11.0 KB |
| discussion/e11_reproduction_checklist.md                                    | 39.8 KB |
| discussion/e11_reviewer_risk_audit.md                                       | 24.1 KB |
| discussion/e11_pasted_review_audit.md                                       | 8.3 KB  |
| discussion/e11_completion_audit.md                                          | 4.8 KB  |
| discussion/e11_end_of_draft_self_review.md                                  | 6.4 KB  |
| discussion/e11_reference_audit.md                                           | 6.1 KB  |
| discussion/e11_submission_repro_audit.md                                    | 8.2 KB  |
| discussion/e11_paper_readiness_audit.md                                     | 34.9 KB |
| discussion/e11_top_conference_plan.md                                       | 14.1 KB |
| discussion/e11_top_conference_gap_register.md                               | 39.4 KB |
| discussion/e11_natural_head_tail_boundary.md                                | 17.5 KB |
| discussion/e11_natural_negative_search_protocol.md                          | 16.9 KB |
| discussion/e11_natural_negative_search_phase1_power_audit.md                | 5.1 KB  |
| discussion/e11_natural_negative_search_phase1_evaluation.md                 | 7.3 KB  |
| discussion/e11_evidence_index.md                                            | 29.3 KB |
| discussion/e11_research_synthesis.md                                        | 7.8 KB  |
| discussion/e11_research_direction_map.md                                    | 9.8 KB  |
| discussion/e11_claim_validity_audit.md                                      | 22.7 KB |
| discussion/e11_cifar100_lt_one_step.md                                      | 2.2 KB  |
| discussion/e11_cifar100_resnet_one_step.md                                  | 2.5 KB  |
| discussion/e11_cifar100_resnet_one_step_rho002.md                           | 2.5 KB  |
| discussion/e11_cifar100_resnet_checkpoint_sweep.md                          | 5.1 KB  |
| discussion/e11_cifar100_resnet_condition_proxy_scatter.md                   | 3.4 KB  |
| discussion/e11_cifar100_resnet_fc_condition_scatter.md                      | 3.4 KB  |
| discussion/e11_cifar100_resnet_tail_quality_control.md                      | 4.7 KB  |
| discussion/e11_cifar100_resnet_imbalance_sweep.md                           | 4.5 KB  |
| discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md                    | 13.8 KB |
| discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md           | 8.1 KB  |
| discussion/e11_cifar100_resnet_condition_score_audit.md                     | 5.5 KB  |
| discussion/e11_cifar100_resnet_condition_score_protocol.md                  | 11.0 KB |
| discussion/e11_cifar100_resnet_condition_score_next.md                      | 8.9 KB  |
| discussion/e11_cifar100_resnet_condition_score_next_heldout_architecture.md | 8.2 KB  |
| discussion/e11_cifar100_resnet_condition_score_next_heldout_data.md         | 8.1 KB  |
| discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md   | 7.2 KB  |
| discussion/e11_condition_score_heldout_failure_theory_note.md               | 2.8 KB  |
| discussion/e11_condition_score_theory_bridge.md                             | 10.6 KB |
| discussion/e11_condition_score_fresh_protocol.md                            | 13.1 KB |
| discussion/e11_condition_score_fresh_evaluation.md                          | 6.7 KB  |
| discussion/e11_condition_score_failure_mechanism_audit.md                   | 13.9 KB |
| discussion/e11_condition_score_v4_protocol.md                               | 16.5 KB |
| discussion/e11_condition_score_v4_validation_cifar100_rotated.md            | 8.2 KB  |
| discussion/e11_condition_score_v4_validation_freeze.md                      | 8.5 KB  |
| discussion/e11_condition_score_v4_architecture_wide_resnet50_2.md           | 8.3 KB  |
| discussion/e11_condition_score_v4_data_cifar10_mixed.md                     | 8.1 KB  |
| discussion/e11_condition_score_v4_final_evaluation.md                       | 5.5 KB  |
| discussion/e11_condition_score_v4_failure_mechanism_audit.md                | 8.7 KB  |
| discussion/e11_condition_score_v5_theory_protocol.md                        | 17.8 KB |
| discussion/e11_condition_score_v5_theory_to_score_map.md                    | 19.8 KB |
| discussion/e11_condition_score_v5_validation_freeze.md                      | 9.4 KB  |
| discussion/e11_cifar100_resnet_lt_standard_eval.md                          | 3.3 KB  |
| discussion/e11_cifar100_resnet_lt_recipe_benchmark.md                       | 6.6 KB  |
| discussion/e11_cifar100_resnet_lt_muon_final_benchmark.md                   | 6.9 KB  |
| discussion/e11_cifar100_resnet_practical_muon_bridge.md                     | 5.7 KB  |

## Ignored Local Artifacts

| path_or_pattern   | reason                                                                                                           |
|:------------------|:-----------------------------------------------------------------------------------------------------------------|
| data/             | Local torchvision dataset cache; downloaded by MNIST/CIFAR probes and not part of the evidence set.              |
| outputs/          | Local Slurm stdout/stderr logs; final CSV/Markdown/figure artifacts are committed separately.                    |
| .pytest_cache/    | Local test runner cache.                                                                                         |
| *.gif, *.mp4      | Generated animations/videos are not part of the default E11 evidence set; force-add only when explicitly needed. |
| slidev/           | Local presentation workspace, intentionally excluded from the repo.                                              |

## Validation Command

```bash
make e11-check
```

Machine-readable copy: `results/e11_artifact_manifest.json`.
