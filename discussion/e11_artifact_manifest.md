# E11 Artifact Manifest

This generated manifest documents the current reproducibility boundary for E11. It separates the head-to-tail paper evidence from background condition-geometry guardrails, records what should be committed as evidence, and lists which key CSV tables back the paper-facing claims.

## Artifact Directories

| path                   | role                                                                                  | commit_policy                            | size     |
|:-----------------------|:--------------------------------------------------------------------------------------|:-----------------------------------------|:---------|
| e11_condition_geometry | Source package for E11 problems, optimizers, diagnostics, reporting, and plotting.    | commit                                   | 436.6 KB |
| scripts                | Experiment runners, artifact writers, and validation scripts.                         | commit E11 scripts                       | 3.0 MB   |
| tests                  | Smoke tests, diagnostics tests, reporting tests, and validator guardrail tests.       | commit                                   | 66.0 KB  |
| discussion             | Generated head-to-tail paper evidence plus legacy condition-geometry guardrail notes. | commit                                   | 2.1 MB   |
| results                | Generated CSV evidence used by the paper table, discussion artifacts, and validator.  | commit current E11 evidence set          | 278.7 MB |
| figures                | Static figures used by the paper draft and paper-facing Markdown artifacts.           | commit static E11 figures                | 36.3 MB  |
| configs                | Experiment configuration snapshots.                                                   | commit                                   | 316 B    |
| paper                  | Head-to-tail LaTeX paper draft, generated paper table, and experiment triage notes.   | commit source and selected rendered PDFs | 5.6 MB   |
| Makefile               | Thin reproducibility entrypoint for E11 artifact generation and validation commands.  | commit                                   | 24.1 KB  |
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
| results/e11_condition_score_theory_bridge/fresh_protocol_requirements.csv                                  |      6 | 3.3 KB    |
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
| results/e11_matrix_block_theorem_proof/theorem_statement.csv                                               |      4 | 1.8 KB    |
| results/e11_matrix_block_theorem_proof/assumption_ledger.csv                                               |      5 | 1.9 KB    |
| results/e11_matrix_block_theorem_proof/proof_steps.csv                                                     |      6 | 1.8 KB    |
| results/e11_matrix_block_theorem_proof/claim_implications.csv                                              |      4 | 1.2 KB    |
| results/e11_matrix_block_theorem_proof/paper_cross_checks.csv                                              |      6 | 430 B     |
| results/e11_matrix_block_tightness_audit/rank_boundary_cases.csv                                           |      4 | 884 B     |
| results/e11_matrix_block_tightness_audit/formula_checks.csv                                                |      4 | 638 B     |
| results/e11_matrix_block_tightness_audit/caveat_checks.csv                                                 |      3 | 764 B     |
| results/e11_theory_proof_obligation_register/proof_obligations.csv                                         |      6 | 4.8 KB    |
| results/e11_theory_proof_obligation_register/assumption_stress_tests.csv                                   |      5 | 1.7 KB    |
| results/e11_theory_proof_obligation_register/claim_scope_boundaries.csv                                    |      5 | 1.4 KB    |
| results/e11_theory_proof_obligation_register/theorem_to_experiment_queue.csv                               |      7 | 1.7 KB    |
| results/e11_condition_score_v5_theory_protocol/theory_term_register.csv                                    |      6 | 2.8 KB    |
| results/e11_condition_score_v5_theory_protocol/score_contract.csv                                          |      6 | 1.8 KB    |
| results/e11_condition_score_v5_theory_protocol/spent_evidence_policy.csv                                   |      6 | 1.9 KB    |
| results/e11_condition_score_v5_theory_protocol/unspent_split_requirements.csv                              |      3 | 1.3 KB    |
| results/e11_condition_score_v5_theory_protocol/acceptance_gates.csv                                        |      6 | 1.6 KB    |
| results/e11_condition_score_v5_theory_to_score_map/theorem_proxy_map.csv                                   |      5 | 2.5 KB    |
| results/e11_condition_score_v5_theory_to_score_map/score_lineage.csv                                       |      7 | 1.5 KB    |
| results/e11_condition_score_v5_theory_to_score_map/transport_normalization_contract.csv                    |      5 | 1.8 KB    |
| results/e11_condition_score_v5_theory_to_score_map/post_final_transport_obligations.csv                    |      5 | 3.1 KB    |
| results/e11_condition_score_v5_theory_to_score_map/falsifiable_predictions.csv                             |      5 | 1.6 KB    |
| results/e11_condition_score_v5_theory_to_score_map/ablation_matrix.csv                                     |      5 | 1.4 KB    |
| results/e11_condition_score_v5_theory_to_score_map/claim_readiness_ledger.csv                              |      6 | 898 B     |
| results/e11_condition_score_ablation/score_ablation_summary.csv                                            |     40 | 12.9 KB   |
| results/e11_condition_score_ablation/term_failure_ladder.csv                                               |      5 | 1.9 KB    |
| results/e11_condition_score_ablation/leakage_and_claim_boundary.csv                                        |      4 | 986 B     |
| results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/metrics.csv                     |   1260 | 1.2 MB    |
| results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/paired_metrics.csv              |    630 | 209.4 KB  |
| results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/layer_summary.csv               |     63 | 18.2 KB   |
| results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/checkpoint_summary.csv          |      3 | 1.2 KB    |
| results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/prediction_summary.csv          |      6 | 1.6 KB    |
| results/e11_condition_score_v5_protocol/validation_cifar100_mod4_partition/residual_prediction_summary.csv |      4 | 1.2 KB    |
| results/e11_submission_repro_audit/toolchain_status.csv                                                    |      7 | 580 B     |
| results/e11_submission_repro_audit/pdf_artifact_checks.csv                                                 |      2 | 313 B     |
| results/e11_submission_repro_audit/source_package_manifest.csv                                             |     11 | 1.8 KB    |
| results/e11_submission_repro_audit/build_gate_summary.csv                                                  |      6 | 1.1 KB    |
| results/e11_artifact_review_packet/command_matrix.csv                                                      |     14 | 5.2 KB    |
| results/e11_artifact_review_packet/gate_matrix.csv                                                         |      6 | 1.6 KB    |
| results/e11_artifact_review_packet/local_state_contract.csv                                                |     13 | 3.8 KB    |
| results/e11_artifact_review_packet/reviewer_response.csv                                                   |     10 | 3.9 KB    |
| results/e11_camera_ready_package_audit/package_item_matrix.csv                                             |      8 | 2.0 KB    |
| results/e11_camera_ready_package_audit/submission_gate_matrix.csv                                          |      4 | 860 B     |
| results/e11_camera_ready_package_audit/camera_ready_checklist.csv                                          |      4 | 855 B     |
| results/e11_condition_score_v5_protocol/validation_score_freeze/score_formula_registry.csv                 |      7 | 2.7 KB    |
| results/e11_condition_score_v5_protocol/validation_score_freeze/freeze_status.csv                          |      5 | 692 B     |
| results/e11_condition_score_v5_protocol/validation_score_freeze/validation_gate_report.csv                 |      6 | 838 B     |
| results/e11_condition_score_v5_protocol/final_score_evaluation/final_gate_report.csv                       |      9 | 1.6 KB    |
| results/e11_condition_score_v5_protocol/final_power_audit/split_power_design.csv                           |      2 | 1.0 KB    |
| results/e11_condition_score_v5_protocol/final_power_audit/fisher_z_resolution.csv                          |      4 | 979 B     |
| results/e11_condition_score_v5_protocol/final_power_audit/mean_spearman_mde.csv                            |      6 | 1.2 KB    |
| results/e11_condition_score_v5_protocol/final_power_audit/interpretation_ladder.csv                        |      5 | 1.3 KB    |
| results/e11_condition_score_v5_protocol/final_power_audit/outcome_state_machine.csv                        |      5 | 998 B     |
| results/e11_condition_score_v5_protocol/final_interpretation_plan/current_interpretation_summary.csv       |      1 | 965 B     |
| results/e11_condition_score_v5_protocol/final_interpretation_plan/final_split_status.csv                   |      2 | 802 B     |
| results/e11_condition_score_v5_protocol/final_interpretation_plan/final_gate_contract.csv                  |      6 | 1.2 KB    |
| results/e11_condition_score_v5_protocol/final_interpretation_plan/outcome_interpretation_ladder.csv        |      7 | 2.1 KB    |
| results/e11_condition_score_v5_protocol/final_interpretation_plan/leakage_lock.csv                         |      5 | 998 B     |
| results/e11_condition_score_v5_protocol/reviewer_failure_response/final_split_output_status.csv            |      2 | 773 B     |
| results/e11_condition_score_v5_protocol/reviewer_failure_response/current_gate_snapshot.csv                |      9 | 1.6 KB    |
| results/e11_condition_score_v5_protocol/reviewer_failure_response/active_failure_modes.csv                 |      4 | 1.5 KB    |
| results/e11_condition_score_v5_protocol/reviewer_failure_response/failure_mode_register.csv                |      9 | 4.5 KB    |
| results/e11_condition_score_v5_protocol/reviewer_failure_response/reviewer_objection_map.csv               |      5 | 1.2 KB    |
| results/e11_condition_score_v5_protocol/reviewer_failure_response/claim_downgrade_actions.csv              |      6 | 2.0 KB    |
| results/e11_condition_score_v5_protocol/reviewer_failure_response/next_evidence_queue.csv                  |      5 | 1.2 KB    |
| results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/score_axis_contrast.csv          |      8 | 3.1 KB    |
| results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/gate_boundary_summary.csv        |      9 | 1.9 KB    |
| results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/mechanistic_diagnosis.csv        |      6 | 2.1 KB    |
| results/e11_condition_score_v5_protocol/direction_guardrail_failure_audit/next_protocol_requirements.csv   |      5 | 1.5 KB    |
| results/e11_natural_head_tail_boundary/search_registry.csv                                                 |     11 | 2.5 KB    |
| results/e11_natural_head_tail_boundary/primary_drift_scan.csv                                              |    169 | 88.7 KB   |
| results/e11_natural_head_tail_boundary/secondary_outcome_scan.csv                                          |    111 | 60.3 KB   |
| results/e11_natural_head_tail_boundary/boundary_summary.csv                                                |      4 | 839 B     |
| results/e11_natural_head_tail_boundary/candidate_negative_cases.csv                                        |     58 | 19.5 KB   |
| results/e11_natural_negative_search_protocol/audit_baseline.csv                                            |      3 | 846 B     |
| results/e11_natural_negative_search_protocol/search_space_registry.csv                                     |      4 | 2.1 KB    |
| results/e11_natural_negative_search_protocol/metric_contract.csv                                           |      5 | 2.0 KB    |
| results/e11_natural_negative_search_protocol/stopping_rules.csv                                            |      6 | 1.5 KB    |
| results/e11_natural_negative_search_protocol/acceptance_gates.csv                                          |      6 | 1.6 KB    |
| results/e11_natural_negative_search_protocol/claim_ladder.csv                                              |      5 | 1.4 KB    |
| results/e11_natural_negative_search_protocol/protocol_status.csv                                           |      7 | 1.9 KB    |
| results/e11_natural_negative_search_protocol/phase1_power_audit/power_grid.csv                             |     72 | 6.3 KB    |
| results/e11_natural_negative_search_protocol/phase1_power_audit/minimum_detectable_effect.csv              |     36 | 7.1 KB    |
| results/e11_natural_negative_search_protocol/phase1_power_audit/interpretation_ladder.csv                  |      4 | 919 B     |
| results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/settings_registry.csv              |     12 | 10.2 KB   |
| results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/step_metrics.csv                   |    120 | 142.8 KB  |
| results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/pair_summary.csv                   |     12 | 29.5 KB   |
| results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/layer_metrics.csv                  |   2520 | 911.1 KB  |
| results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/decision_template.csv              |     12 | 6.0 KB    |
| results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/settings_registry.csv               |      8 | 4.7 KB    |
| results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/step_metrics.csv                    |     80 | 73.3 KB   |
| results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/pair_summary.csv                    |      8 | 18.9 KB   |
| results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/layer_metrics.csv                   |   1680 | 577.1 KB  |
| results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/decision_template.csv               |      8 | 4.0 KB    |
| results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/settings_registry.csv            |      6 | 4.9 KB    |
| results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/run_registry.csv               |      3 | 1.0 KB    |
| results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/primary_decisions.csv          |     26 | 13.3 KB   |
| results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/gate_report.csv                |      5 | 725 B     |
| results/e11_natural_negative_search_protocol/phase1_interim_synthesis/family_coverage.csv                  |      3 | 361 B     |
| results/e11_natural_negative_search_protocol/phase1_interim_synthesis/observed_primary_summary.csv         |      4 | 919 B     |
| results/e11_natural_negative_search_protocol/phase1_interim_synthesis/claim_boundary.csv                   |      3 | 965 B     |
| results/e11_natural_negative_search_protocol/phase1_interim_synthesis/remaining_work.csv                   |      0 | 86 B      |
| results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv             |      8 | 7.3 KB    |
| results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/run_registry.csv               |      1 | 450 B     |
| results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/seed_level_primary_ratios.csv  |     24 | 7.5 KB    |
| results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv          |      8 | 4.6 KB    |
| results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/gate_report.csv                |      5 | 630 B     |
| results/e11_natural_negative_search_protocol/phase2_power_audit/power_grid.csv                             |     72 | 5.8 KB    |
| results/e11_natural_negative_search_protocol/phase2_power_audit/minimum_detectable_effect.csv              |     36 | 8.3 KB    |
| results/e11_natural_negative_search_protocol/phase2_power_audit/interpretation_ladder.csv                  |      5 | 1.2 KB    |
| results/e11_natural_negative_search_protocol/phase2_power_audit/outcome_state_machine.csv                  |      7 | 1.6 KB    |
| results/e11_heldout_generality_audit/generality_evidence_matrix.csv                                        |      7 | 5.3 KB    |
| results/e11_heldout_generality_audit/generality_claim_gate.csv                                             |      5 | 1.1 KB    |
| results/e11_bold_conjecture_register/conjecture_register.csv                                               |      5 | 4.2 KB    |
| results/e11_bold_conjecture_register/stress_test_matrix.csv                                                |      5 | 2.7 KB    |
| results/e11_bold_conjecture_register/claim_upgrade_ladder.csv                                              |      4 | 1.1 KB    |
| results/e11_muon_state_distribution_contract/state_distribution_terms.csv                                  |      5 | 2.3 KB    |
| results/e11_muon_state_distribution_contract/evidence_link_matrix.csv                                      |      5 | 1.5 KB    |
| results/e11_muon_state_distribution_contract/falsification_tests.csv                                       |      5 | 2.3 KB    |
| results/e11_muon_state_distribution_contract/claim_gate_ladder.csv                                         |      4 | 1.2 KB    |
| results/e11_top_conference_gap_register/gap_register.csv                                                   |      7 | 31.4 KB   |
| results/e11_top_conference_claim_decision_audit/claim_decision_matrix.csv                                  |      6 | 11.5 KB   |
| results/e11_top_conference_claim_decision_audit/reviewer_objection_matrix.csv                              |      5 | 3.4 KB    |
| results/e11_top_conference_claim_decision_audit/rebuttal_response_pack.csv                                 |      5 | 6.4 KB    |
| results/e11_top_conference_claim_decision_audit/manuscript_edit_queue.csv                                  |      6 | 3.5 KB    |
| results/e11_top_conference_claim_decision_audit/paper_sequence.csv                                         |      6 | 1.9 KB    |
| results/e11_top_conference_claim_decision_audit/readiness_summary.csv                                      |      6 | 237 B     |
| results/e11_manuscript_claim_trace/claim_trace.csv                                                         |      6 | 2.6 KB    |
| results/e11_manuscript_claim_trace/blocked_phrase_audit.csv                                                |     17 | 2.0 KB    |
| results/e11_clean_worktree_replay_audit/run_summary.csv                                                    |      1 | 430 B     |
| results/e11_clean_worktree_replay_audit/gate_matrix.csv                                                    |      6 | 810 B     |
| results/e11_pdf_render_boundary_audit/pdf_inspection_tool_status.csv                                       |      8 | 494 B     |
| results/e11_pdf_render_boundary_audit/rendered_pdf_text_checks.csv                                         |      2 | 451 B     |
| results/e11_pdf_render_boundary_audit/pdf_metadata_checks.csv                                              |      2 | 450 B     |
| results/e11_pdf_render_boundary_audit/render_boundary_gates.csv                                            |      6 | 884 B     |
| results/e11_mechanism_referee_audit/alternative_explanation_matrix.csv                                     |      9 | 5.8 KB    |
| results/e11_mechanism_referee_audit/theory_measurement_contract.csv                                        |      6 | 3.5 KB    |
| results/e11_mechanism_referee_audit/falsification_trigger_matrix.csv                                       |      6 | 2.8 KB    |
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
| results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/pilot_context.csv                                  |      9 | 3.1 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/benchmark_scope.csv                                |      2 | 505 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/seed_split_contract.csv                            |      4 | 505 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/recipe_grid.csv                                    |      6 | 1.4 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/selection_rules.csv                                |      6 | 1.2 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/acceptance_gates.csv                               |      7 | 1.8 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv                                       |    164 | 122.2 KB  |
| results/e11_cifar100_resnet_lt_tuned_benchmark/execution_status.csv                                        |      1 | 189 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/run_registry.csv                       |    164 | 82.1 KB   |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/family_selection.csv                   |      6 | 1.1 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/final_claim_plan.csv                   |      6 | 884 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/gate_report.csv                        |      5 | 491 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/final_family_design.csv                   |      6 | 1.8 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/primary_comparison_plan.csv               |      4 | 1.2 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/paired_diff_mde.csv                       |     12 | 3.0 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/all_class_guardrail_mde.csv               |     12 | 2.9 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/interpretation_ladder.csv                 |      6 | 1.4 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/outcome_state_machine.csv                 |      5 | 1.0 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/validation_setting_variance.csv        |    200 | 51.6 KB   |
| results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/spent_pilot_paired_variance.csv        |     16 | 3.6 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/variance_prior_summary.csv             |      8 | 1.6 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/mde_sensitivity_from_empirical_sd.csv  |     40 | 9.6 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/gate_matrix.csv                        |      6 | 1023 B    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/analysis_input_contract.csv             |      6 | 1.7 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/metric_contract.csv                     |      4 | 817 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/primary_comparison_family.csv           |      4 | 1.4 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/multiplicity_and_guardrail_plan.csv     |      3 | 999 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/reporting_schema.csv                    |      4 | 925 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/claim_ladder.csv                        |      5 | 1.0 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/gate_matrix.csv                         |      6 | 853 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/gate_matrix.csv                        |      5 | 676 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/final_family_run_plan.csv              |      6 | 1.6 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_execution_plan/slurm_submit_plan.csv                  |      1 | 364 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/run_registry.csv                           |      6 | 1.9 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/primary_decisions.csv                      |      4 | 865 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_evaluation/claim_gate_report.csv                      |      6 | 767 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_final_launch_decision.csv         |      1 | 498 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_final_family_plan.csv             |      6 | 602 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/latest_gate_snapshot.csv                 |      5 | 602 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_launch_audit/final_launch_history.csv                 |      2 | 633 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_submission_plan/chunk_plan.csv                        |      6 | 2.2 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_submission_plan/queue_policy.csv                      |      4 | 879 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit/latest_launch_decision.csv               |      1 | 838 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit/latest_selected_settings.csv             |      1 | 511 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit/latest_queue_snapshot.csv                |     23 | 1.9 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit/launch_history.csv                       |      9 | 3.0 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit/completed_setting_leaderboard.csv  |     50 | 12.1 KB   |
| results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit/family_progress.csv                |      6 | 1.5 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit/occupancy_interim_summary.csv      |      6 | 749 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit/partial_grid_guardrail.csv         |      5 | 955 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/interim_validation_audit/claim_boundary_gates.csv           |      5 | 832 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit/leakage_guard_matrix.csv           |      6 | 1.3 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit/observed_surface.csv               |      3 | 553 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_leakage_audit/immutability_contract.csv          |      4 | 821 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/refresh_state.csv               |      1 | 398 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/allowed_transition_matrix.csv   |      5 | 1.2 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_refresh_firewall/forbidden_action_matrix.csv     |      6 | 1.2 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/hash_manifest.csv                  |     18 | 4.4 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/exposure_boundary.csv              |      1 | 415 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/immutability_matrix.csv            |      5 | 1.4 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_protocol_seal/seal_gate_matrix.csv               |      5 | 665 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/family_budget_matrix.csv          |      6 | 1.2 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/seed_metric_parity.csv            |      5 | 764 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/fairness_gate_matrix.csv          |      5 | 903 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/validation_fairness_audit/budget_disclosure_contract.csv    |      3 | 657 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_robustness_plan/robustness_test_matrix.csv            |     16 | 5.1 KB    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_robustness_plan/missing_seed_policy.csv               |      4 | 777 B     |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_robustness_plan/claim_sensitivity_ladder.csv          |      5 | 1011 B    |
| results/e11_cifar100_resnet_lt_tuned_benchmark/final_robustness_plan/gate_matrix.csv                       |      5 | 492 B     |
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

## Zero-Row Allowed Key Tables

| path                                                                                                      |
|:----------------------------------------------------------------------------------------------------------|
| results/e11_natural_negative_search_protocol/phase1_interim_synthesis/remaining_work.csv                  |
| results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/seed_level_primary_ratios.csv |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit/latest_selected_settings.csv            |
| results/e11_cifar100_resnet_lt_tuned_benchmark/slurm_launch_audit/latest_queue_snapshot.csv               |

## Key Paper Documents

| path                                                                                        | size     |
|:--------------------------------------------------------------------------------------------|:---------|
| README_E11.md                                                                               | 102.9 KB |
| discussion/e11_paper_skeleton.md                                                            | 10.9 KB  |
| discussion/e11_main_paper_package.md                                                        | 22.8 KB  |
| discussion/e11_main_figure_captions.md                                                      | 6.5 KB   |
| discussion/e11_notation_glossary.md                                                         | 8.7 KB   |
| discussion/e11_quantitative_claim_ledger.md                                                 | 22.1 KB  |
| discussion/e11_reproduction_checklist.md                                                    | 73.2 KB  |
| discussion/e11_reviewer_risk_audit.md                                                       | 24.1 KB  |
| discussion/e11_pasted_review_audit.md                                                       | 8.3 KB   |
| discussion/e11_completion_audit.md                                                          | 5.3 KB   |
| discussion/e11_end_of_draft_self_review.md                                                  | 7.3 KB   |
| discussion/e11_reference_audit.md                                                           | 6.1 KB   |
| discussion/e11_submission_repro_audit.md                                                    | 8.8 KB   |
| discussion/e11_artifact_review_packet.md                                                    | 29.6 KB  |
| discussion/e11_camera_ready_package_audit.md                                                | 7.6 KB   |
| discussion/e11_paper_readiness_audit.md                                                     | 34.9 KB  |
| discussion/e11_top_conference_plan.md                                                       | 14.1 KB  |
| discussion/e11_top_conference_gap_register.md                                               | 81.8 KB  |
| discussion/e11_top_conference_claim_decision_audit.md                                       | 111.2 KB |
| discussion/e11_manuscript_claim_trace.md                                                    | 9.6 KB   |
| discussion/e11_clean_worktree_replay_audit.md                                               | 3.2 KB   |
| discussion/e11_pdf_render_boundary_audit.md                                                 | 5.9 KB   |
| discussion/e11_mechanism_referee_audit.md                                                   | 30.8 KB  |
| discussion/e11_bold_conjecture_register.md                                                  | 14.5 KB  |
| discussion/e11_muon_state_distribution_contract.md                                          | 14.8 KB  |
| discussion/e11_natural_head_tail_boundary.md                                                | 17.5 KB  |
| discussion/e11_natural_negative_search_protocol.md                                          | 19.7 KB  |
| discussion/e11_natural_negative_search_phase1_power_audit.md                                | 5.1 KB   |
| discussion/e11_natural_negative_search_phase1_NNS-P1-cifar100lt-resnet18-new-partitions.md  | 11.1 KB  |
| discussion/e11_natural_negative_search_phase1_NNS-P1-cifar10lt-resnet18-cross-partitions.md | 7.8 KB   |
| discussion/e11_natural_negative_search_phase1_evaluation.md                                 | 7.5 KB   |
| discussion/e11_natural_negative_search_phase1_interim_synthesis.md                          | 6.2 KB   |
| discussion/e11_natural_negative_search_phase2_NNS-P2-heldout-architecture-boundary.md       | 8.9 KB   |
| discussion/e11_natural_negative_search_phase2_evaluation.md                                 | 5.7 KB   |
| discussion/e11_natural_negative_search_phase2_power_audit.md                                | 10.6 KB  |
| discussion/e11_heldout_generality_audit.md                                                  | 10.8 KB  |
| discussion/e11_evidence_index.md                                                            | 29.3 KB  |
| discussion/e11_research_synthesis.md                                                        | 7.8 KB   |
| discussion/e11_research_direction_map.md                                                    | 9.8 KB   |
| discussion/e11_claim_validity_audit.md                                                      | 22.7 KB  |
| discussion/e11_cifar100_lt_one_step.md                                                      | 2.2 KB   |
| discussion/e11_cifar100_resnet_one_step.md                                                  | 2.5 KB   |
| discussion/e11_cifar100_resnet_one_step_rho002.md                                           | 2.5 KB   |
| discussion/e11_cifar100_resnet_checkpoint_sweep.md                                          | 5.1 KB   |
| discussion/e11_cifar100_resnet_condition_proxy_scatter.md                                   | 3.4 KB   |
| discussion/e11_cifar100_resnet_fc_condition_scatter.md                                      | 3.4 KB   |
| discussion/e11_cifar100_resnet_tail_quality_control.md                                      | 4.7 KB   |
| discussion/e11_cifar100_resnet_imbalance_sweep.md                                           | 4.5 KB   |
| discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md                                    | 13.8 KB  |
| discussion/e11_cifar100_resnet_layer_jvp_checkpoint_prediction.md                           | 8.1 KB   |
| discussion/e11_cifar100_resnet_condition_score_audit.md                                     | 5.5 KB   |
| discussion/e11_cifar100_resnet_condition_score_protocol.md                                  | 11.0 KB  |
| discussion/e11_cifar100_resnet_condition_score_next.md                                      | 8.9 KB   |
| discussion/e11_cifar100_resnet_condition_score_next_heldout_architecture.md                 | 8.2 KB   |
| discussion/e11_cifar100_resnet_condition_score_next_heldout_data.md                         | 8.1 KB   |
| discussion/e11_cifar100_resnet_condition_score_next_heldout_evaluation.md                   | 7.2 KB   |
| discussion/e11_condition_score_heldout_failure_theory_note.md                               | 2.8 KB   |
| discussion/e11_condition_score_theory_bridge.md                                             | 14.9 KB  |
| discussion/e11_condition_score_fresh_protocol.md                                            | 13.1 KB  |
| discussion/e11_condition_score_fresh_evaluation.md                                          | 6.7 KB   |
| discussion/e11_condition_score_failure_mechanism_audit.md                                   | 13.9 KB  |
| discussion/e11_condition_score_v4_protocol.md                                               | 16.5 KB  |
| discussion/e11_condition_score_v4_validation_cifar100_rotated.md                            | 8.2 KB   |
| discussion/e11_condition_score_v4_validation_freeze.md                                      | 8.5 KB   |
| discussion/e11_condition_score_v4_architecture_wide_resnet50_2.md                           | 8.3 KB   |
| discussion/e11_condition_score_v4_data_cifar10_mixed.md                                     | 8.1 KB   |
| discussion/e11_condition_score_v4_final_evaluation.md                                       | 5.5 KB   |
| discussion/e11_condition_score_v4_failure_mechanism_audit.md                                | 8.7 KB   |
| discussion/e11_matrix_block_theorem_proof.md                                                | 13.3 KB  |
| discussion/e11_matrix_block_tightness_audit.md                                              | 6.0 KB   |
| discussion/e11_theory_proof_obligation_register.md                                          | 17.3 KB  |
| discussion/e11_condition_score_v5_theory_protocol.md                                        | 17.8 KB  |
| discussion/e11_condition_score_v5_theory_to_score_map.md                                    | 36.3 KB  |
| discussion/e11_condition_score_ablation.md                                                  | 23.2 KB  |
| discussion/e11_condition_score_v5_validation_cifar100_mod4_partition.md                     | 8.2 KB   |
| discussion/e11_condition_score_v5_validation_freeze.md                                      | 13.0 KB  |
| discussion/e11_condition_score_v5_final_evaluation.md                                       | 6.8 KB   |
| discussion/e11_condition_score_v5_final_power_audit.md                                      | 10.3 KB  |
| discussion/e11_condition_score_v5_final_interpretation_plan.md                              | 10.9 KB  |
| discussion/e11_condition_score_v5_reviewer_failure_response.md                              | 21.3 KB  |
| discussion/e11_condition_score_v5_direction_guardrail_failure_audit.md                      | 14.5 KB  |
| discussion/e11_cifar100_resnet_lt_standard_eval.md                                          | 3.3 KB   |
| discussion/e11_cifar100_resnet_lt_recipe_benchmark.md                                       | 6.6 KB   |
| discussion/e11_cifar100_resnet_lt_muon_final_benchmark.md                                   | 6.9 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md                               | 14.8 KB  |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md                              | 75.4 KB  |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_interim_audit.md                          | 12.3 KB  |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md                          | 6.2 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md                       | 5.9 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol_seal.md                          | 12.3 KB  |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_fairness_audit.md                         | 7.9 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md                            | 16.9 KB  |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_variance_prior_audit.md                   | 9.5 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.md                    | 15.3 KB  |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_robustness_plan.md                  | 12.1 KB  |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md                   | 7.7 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md                       | 7.2 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md                     | 7.4 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_plan.md                             | 5.5 KB   |
| discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_launch_audit.md                     | 5.5 KB   |
| discussion/e11_cifar100_resnet_practical_muon_bridge.md                                     | 5.7 KB   |

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
