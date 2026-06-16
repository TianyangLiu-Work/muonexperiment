# E11 Artifact Manifest

This generated manifest documents the current reproducibility boundary for E11. It separates the head-to-tail paper evidence from background condition-geometry guardrails, records what should be committed as evidence, and lists which key CSV tables back the paper-facing claims.

## Artifact Directories

| path                   | role                                                                                  | commit_policy                            | size     |
|:-----------------------|:--------------------------------------------------------------------------------------|:-----------------------------------------|:---------|
| e11_condition_geometry | Source package for E11 problems, optimizers, diagnostics, reporting, and plotting.    | commit                                   | 722.3 KB |
| scripts                | Experiment runners, artifact writers, and validation scripts.                         | commit E11 scripts                       | 1.5 MB   |
| tests                  | Smoke tests, diagnostics tests, reporting tests, and validator guardrail tests.       | commit                                   | 239.8 KB |
| discussion             | Generated head-to-tail paper evidence plus legacy condition-geometry guardrail notes. | commit                                   | 653.2 KB |
| results                | Generated CSV evidence used by the paper table, discussion artifacts, and validator.  | commit current E11 evidence set          | 256.9 MB |
| figures                | Static figures used by the paper draft and paper-facing Markdown artifacts.           | commit static E11 figures                | 28.7 MB  |
| configs                | Experiment configuration snapshots.                                                   | commit                                   | 316 B    |
| paper                  | Head-to-tail LaTeX paper draft, generated paper table, and experiment triage notes.   | commit source and selected rendered PDFs | 3.4 MB   |
| Makefile               | Thin reproducibility entrypoint for E11 artifact generation and validation commands.  | commit                                   | 4.6 KB   |
| .gitignore             | Keeps local caches, datasets, and generated videos out of default commits.            | commit                                   | 479 B    |
| .gitattributes         | Marks generated evidence artifacts and binary files for cleaner GitHub review.        | commit                                   | 667 B    |

## Key Quantitative Tables

| path                                                                            |   rows | size      |
|:--------------------------------------------------------------------------------|-------:|:----------|
| results/e11/step_metrics.csv                                                    |   2880 | 20.9 MB   |
| results/e11/activation_perturbation_summary.csv                                 |     60 | 12.6 KB   |
| results/e11_head_tail_interference/step_metrics.csv                             |    320 | 142.9 KB  |
| results/e11_head_tail_interference/pair_summary.csv                             |      2 | 2.0 KB    |
| results/e11_long_tail_one_step/step_metrics.csv                                 |     40 | 26.4 KB   |
| results/e11_long_tail_one_step/pair_summary.csv                                 |      1 | 5.9 KB    |
| results/e11_long_tail_one_step/layer_metrics.csv                                |     80 | 7.0 KB    |
| results/e11_cifar100_lt_one_step/step_metrics.csv                               |     10 | 10.3 KB   |
| results/e11_cifar100_lt_one_step/pair_summary.csv                               |      1 | 6.0 KB    |
| results/e11_cifar100_lt_one_step/layer_metrics.csv                              |     20 | 2.0 KB    |
| results/e11_cifar100_resnet_one_step/step_metrics.csv                           |     20 | 19.5 KB   |
| results/e11_cifar100_resnet_one_step/pair_summary.csv                           |      1 | 6.1 KB    |
| results/e11_cifar100_resnet_one_step/layer_metrics.csv                          |    420 | 33.5 KB   |
| results/e11_cifar100_resnet_one_step_rho002/step_metrics.csv                    |     20 | 19.6 KB   |
| results/e11_cifar100_resnet_one_step_rho002/pair_summary.csv                    |      1 | 6.1 KB    |
| results/e11_cifar100_resnet_one_step_rho002/layer_metrics.csv                   |    420 | 33.5 KB   |
| results/e11_cifar100_resnet_checkpoint_sweep/step_metrics.csv                   |     80 | 75.5 KB   |
| results/e11_cifar100_resnet_checkpoint_sweep/pair_summary.csv                   |      4 | 12.0 KB   |
| results/e11_cifar100_resnet_checkpoint_sweep/layer_metrics.csv                  |   1680 | 141.4 KB  |
| results/e11_cifar100_resnet_condition_proxy_scatter/scatter_points.csv          |     40 | 23.6 KB   |
| results/e11_cifar100_resnet_condition_proxy_scatter/summary.csv                 |      6 | 784 B     |
| results/e11_cifar100_resnet_condition_proxy_scatter/layer_summary.csv           |     84 | 9.5 KB    |
| results/e11_cifar100_resnet_fc_condition_scatter/step_metrics.csv               |     80 | 53.6 KB   |
| results/e11_cifar100_resnet_fc_condition_scatter/summary.csv                    |      4 | 12.3 KB   |
| results/e11_cifar100_resnet_fc_condition_scatter/condition_metrics.csv          |     40 | 5.2 KB    |
| results/e11_cifar100_resnet_tail_quality_control/step_metrics.csv               |     60 | 57.3 KB   |
| results/e11_cifar100_resnet_tail_quality_control/pair_summary.csv               |      3 | 10.1 KB   |
| results/e11_cifar100_resnet_tail_quality_control/layer_metrics.csv              |   1260 | 107.0 KB  |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/metrics.csv                  |    420 | 405.8 KB  |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/paired_metrics.csv           |    210 | 70.4 KB   |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/summary.csv                  |     21 | 10.9 KB   |
| results/e11_cifar100_resnet_layer_jvp_tail_quality/overall_summary.csv          |      1 | 1.4 KB    |
| results/e11_long_tail_imbalance_ablation/step_metrics.csv                       |    160 | 81.7 KB   |
| results/e11_long_tail_imbalance_ablation/summary.csv                            |      4 | 6.8 KB    |
| results/e11_long_tail_checkpoint_sweep/step_metrics.csv                         |    200 | 121.5 KB  |
| results/e11_long_tail_checkpoint_sweep/summary.csv                              |      5 | 10.0 KB   |
| results/e11_long_tail_class_partition_sweep/step_metrics.csv                    |    200 | 124.6 KB  |
| results/e11_long_tail_class_partition_sweep/summary.csv                         |      5 | 10.8 KB   |
| results/e11_long_tail_rho_sweep/step_metrics.csv                                |    200 | 123.2 KB  |
| results/e11_long_tail_rho_sweep/summary.csv                                     |      5 | 10.1 KB   |
| results/e11_long_tail_muon_bridge/step_metrics.csv                              |     80 | 42.2 KB   |
| results/e11_long_tail_muon_bridge/pair_summary.csv                              |      3 | 2.1 KB    |
| results/e11_local_linearization/summary.csv                                     |      4 | 1.0 KB    |
| results/e11_long_tail_practical_muon_bridge/step_metrics.csv                    |    480 | 253.2 KB  |
| results/e11_long_tail_practical_muon_bridge/step_summary.csv                    |     18 | 3.3 KB    |
| results/e11_long_tail_practical_muon_bridge/summary.csv                         |      3 | 2.2 KB    |
| results/e11_long_tail_muon_state_source_control/step_metrics.csv                |    960 | 522.9 KB  |
| results/e11_long_tail_muon_state_source_control/summary.csv                     |      6 | 3.5 KB    |
| results/e11_long_tail_practical_training/step_metrics.csv                       |   3240 | 1.0 MB    |
| results/e11_long_tail_practical_training/summary.csv                            |      1 | 1.1 KB    |
| results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv             |      4 | 2.0 KB    |
| results/e11_long_tail_forgetting/step_metrics.csv                               |    360 | 93.7 KB   |
| results/e11_long_tail_forgetting/summary.csv                                    |      1 | 1.6 KB    |
| results/e11_long_tail_layerwise/metrics.csv                                     |     80 | 36.7 KB   |
| results/e11_long_tail_layerwise/summary.csv                                     |      2 | 2.2 KB    |
| results/e11_equal_update/step_metrics.csv                                       |   2880 | 21.0 MB   |
| results/e11_equal_update/activation_perturbation_summary.csv                    |     60 | 12.6 KB   |
| results/e11_overlap_followup/step_metrics.csv                                   |   1530 | 8.2 MB    |
| results/e11_hyperparam_sweep/equal_step_metrics.csv                             |   3360 | 21.0 MB   |
| results/e11_target_update_sweep/step_metrics.csv                                |   3360 | 21.1 MB   |
| results/e11_mlp_width_sweep/step_metrics.csv                                    |   3300 | 14.3 MB   |
| results/e11_mlp_per_layer_control/step_metrics.csv                              |   1100 | 4.8 MB    |
| results/e11_mlp_layer_hybrid/step_metrics.csv                                   |   2200 | 9.6 MB    |
| results/e11_mnist_mlp_probe/step_metrics.csv                                    |    216 | 4.4 MB    |
| results/e11_deep_mnist_mlp_probe/step_metrics.csv                               |    432 | 12.4 MB   |
| results/e11_mnist_patch_probe/step_metrics.csv                                  |    288 | 839.6 KB  |
| results/e11_mnist_conv_probe/step_metrics.csv                                   |    288 | 837.9 KB  |
| results/e11_stateless_direction_ablation/stateless_direction_rows.csv           |   1890 | 2.8 MB    |
| results/e11_stateless_optimizer_trajectory/stateless_optimizer_step_metrics.csv |    840 | 5.3 MB    |
| results/e11_spectral_allocation_probe/probe_rows.csv                            |    720 | 192.6 KB  |
| results/e11_singular_vector_trajectory/update_subspace_rows.csv                 |    725 | 76.8 KB   |
| results/e11_singular_vector_swap_probe/swap_probe_rows.csv                      |    320 | 63.4 KB   |
| results/e11_natural_update_swap_probe/natural_update_swap_rows.csv              |   4800 | 968.1 KB  |
| results/e11_optimizer_switch_probe/optimizer_switch_rows.csv                    |    360 | 72.2 KB   |
| results/e11_optimizer_switch_reset_control/optimizer_switch_reset_rows.csv      |    540 | 92.0 KB   |
| results/e11_optimizer_switch_horizon_sweep/optimizer_switch_horizon_rows.csv    |    480 | 99.0 KB   |
| results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_rows.csv              |    360 | 64.6 KB   |
| results/e11_boundary_predictor/boundary_predictor_rows.csv                      |   1500 | 1017.5 KB |
| results/e11_mechanism_boundary/mechanism_boundary_map.csv                       |     14 | 5.2 KB    |
| results/e11_cross_task_signature/cross_task_signature_summary.csv               |     14 | 3.6 KB    |
| results/e11_boundary_predictor/boundary_predictor_uncertainty.csv               |      8 | 1.6 KB    |
| results/e11_head_tail_alignment_ablation/step_metrics.csv                       |   1000 | 149.9 KB  |
| results/e11_head_tail_alignment_ablation/summary.csv                            |      2 | 1.2 KB    |

## Key Paper Documents

| path                                                      | size    |
|:----------------------------------------------------------|:--------|
| README_E11.md                                             | 24.4 KB |
| discussion/e11_paper_skeleton.md                          | 10.9 KB |
| discussion/e11_main_paper_package.md                      | 12.0 KB |
| discussion/e11_main_figure_captions.md                    | 6.5 KB  |
| discussion/e11_notation_glossary.md                       | 8.7 KB  |
| discussion/e11_quantitative_claim_ledger.md               | 11.0 KB |
| discussion/e11_reproduction_checklist.md                  | 19.5 KB |
| discussion/e11_reviewer_risk_audit.md                     | 18.2 KB |
| discussion/e11_pasted_review_audit.md                     | 8.3 KB  |
| discussion/e11_completion_audit.md                        | 4.8 KB  |
| discussion/e11_end_of_draft_self_review.md                | 6.4 KB  |
| discussion/e11_reference_audit.md                         | 6.1 KB  |
| discussion/e11_paper_readiness_audit.md                   | 20.9 KB |
| discussion/e11_top_conference_plan.md                     | 7.7 KB  |
| discussion/e11_evidence_index.md                          | 14.8 KB |
| discussion/e11_research_synthesis.md                      | 7.8 KB  |
| discussion/e11_research_direction_map.md                  | 9.8 KB  |
| discussion/e11_claim_validity_audit.md                    | 15.4 KB |
| discussion/e11_cifar100_lt_one_step.md                    | 2.2 KB  |
| discussion/e11_cifar100_resnet_one_step.md                | 2.5 KB  |
| discussion/e11_cifar100_resnet_one_step_rho002.md         | 2.5 KB  |
| discussion/e11_cifar100_resnet_checkpoint_sweep.md        | 5.1 KB  |
| discussion/e11_cifar100_resnet_condition_proxy_scatter.md | 3.4 KB  |
| discussion/e11_cifar100_resnet_fc_condition_scatter.md    | 3.4 KB  |
| discussion/e11_cifar100_resnet_tail_quality_control.md    | 4.7 KB  |
| discussion/e11_cifar100_resnet_layer_jvp_tail_quality.md  | 13.8 KB |

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
