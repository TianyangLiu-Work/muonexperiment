# E11 Artifact Manifest

This generated manifest documents the current reproducibility boundary for E11. It separates the head-to-tail paper evidence from background condition-geometry guardrails, records what should be committed as evidence, and lists which key CSV tables back the paper-facing claims.

## Artifact Directories

| path                   | role                                                                                  | commit_policy                            | size     |
|:-----------------------|:--------------------------------------------------------------------------------------|:-----------------------------------------|:---------|
| e11_condition_geometry | Source package for E11 problems, optimizers, diagnostics, reporting, and plotting.    | commit                                   | 675.2 KB |
| scripts                | Experiment runners, artifact writers, and validation scripts.                         | commit E11 scripts                       | 1.7 MB   |
| tests                  | Smoke tests, diagnostics tests, reporting tests, and validator guardrail tests.       | commit                                   | 236.1 KB |
| discussion             | Generated head-to-tail paper evidence plus legacy condition-geometry guardrail notes. | commit                                   | 481.8 KB |
| results                | Generated CSV evidence used by the paper table, discussion artifacts, and validator.  | commit current E11 evidence set          | 254.5 MB |
| figures                | Static figures used by the paper draft and paper-facing Markdown artifacts.           | commit static E11 figures                | 27.0 MB  |
| configs                | Experiment configuration snapshots.                                                   | commit                                   | 316 B    |
| paper                  | Head-to-tail LaTeX paper draft, generated paper table, and experiment triage notes.   | commit source and selected rendered PDFs | 2.1 MB   |
| Makefile               | Thin reproducibility entrypoint for E11 artifact generation and validation commands.  | commit                                   | 2.4 KB   |
| .gitignore             | Keeps local caches, datasets, and generated videos out of default commits.            | commit                                   | 470 B    |
| .gitattributes         | Marks generated evidence artifacts and binary files for cleaner GitHub review.        | commit                                   | 667 B    |

## Key Quantitative Tables

| path                                                                            |   rows | size      |
|:--------------------------------------------------------------------------------|-------:|:----------|
| results/e11/step_metrics.csv                                                    |   2880 | 20.9 MB   |
| results/e11/activation_perturbation_summary.csv                                 |     60 | 12.6 KB   |
| results/e11_head_tail_interference/step_metrics.csv                             |    320 | 142.9 KB  |
| results/e11_head_tail_interference/pair_summary.csv                             |      2 | 2.0 KB    |
| results/e11_long_tail_one_step/step_metrics.csv                                 |     40 | 18.1 KB   |
| results/e11_long_tail_one_step/pair_summary.csv                                 |      1 | 1.0 KB    |
| results/e11_long_tail_one_step/layer_metrics.csv                                |     80 | 7.0 KB    |
| results/e11_long_tail_muon_bridge/step_metrics.csv                              |     80 | 37.2 KB   |
| results/e11_long_tail_muon_bridge/pair_summary.csv                              |      3 | 2.1 KB    |
| results/e11_long_tail_practical_muon_bridge/step_metrics.csv                    |    480 | 223.5 KB  |
| results/e11_long_tail_practical_muon_bridge/step_summary.csv                    |     18 | 3.3 KB    |
| results/e11_long_tail_practical_muon_bridge/summary.csv                         |      3 | 2.2 KB    |
| results/e11_long_tail_practical_training/step_metrics.csv                       |   3240 | 1.0 MB    |
| results/e11_long_tail_practical_training/summary.csv                            |      1 | 1.1 KB    |
| results/e11_long_tail_practical_training_lr_sweep/sweep_summary.csv             |      4 | 2.0 KB    |
| results/e11_long_tail_forgetting/step_metrics.csv                               |    360 | 93.7 KB   |
| results/e11_long_tail_forgetting/summary.csv                                    |      1 | 1.6 KB    |
| results/e11_long_tail_layerwise/metrics.csv                                     |     80 | 32.1 KB   |
| results/e11_long_tail_layerwise/summary.csv                                     |      2 | 1.9 KB    |
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

## Ignored Local Artifacts

| path_or_pattern   | reason                                                                                                           |
|:------------------|:-----------------------------------------------------------------------------------------------------------------|
| data/             | Local torchvision/MNIST cache; downloaded by the MNIST probe and not part of the evidence set.                   |
| .pytest_cache/    | Local test runner cache.                                                                                         |
| *.gif, *.mp4      | Generated animations/videos are not part of the default E11 evidence set; force-add only when explicitly needed. |
| slidev/           | Local presentation workspace, intentionally excluded from the repo.                                              |

## Validation Command

```bash
make e11-check
```

Machine-readable copy: `results/e11_artifact_manifest.json`.
