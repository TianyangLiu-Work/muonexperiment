# E11 Artifact Manifest

This generated manifest documents the current reproducibility boundary for E11. It is intentionally practical: it records what should be committed as evidence, what is local cache, and which key CSV tables back the paper-facing claims.

## Artifact Directories

| path                   | role                                                                                 | commit_policy                   | size     |
|:-----------------------|:-------------------------------------------------------------------------------------|:--------------------------------|:---------|
| e11_condition_geometry | Source package for E11 problems, optimizers, diagnostics, reporting, and plotting.   | commit                          | 415.4 KB |
| scripts                | Experiment runners, artifact writers, and validation scripts.                        | commit E11 scripts              | 1.2 MB   |
| tests                  | Smoke tests, diagnostics tests, reporting tests, and validator guardrail tests.      | commit                          | 141.8 KB |
| discussion             | Generated paper-facing Markdown evidence and synthesis.                              | commit                          | 620.6 KB |
| results                | Generated CSV evidence used by discussion artifacts and validator.                   | commit current E11 evidence set | 247.6 MB |
| figures                | Static figures used by paper-facing Markdown artifacts.                              | commit static E11 figures       | 40.9 MB  |
| configs                | Experiment configuration snapshots.                                                  | commit                          | 316 B    |
| Makefile               | Thin reproducibility entrypoint for E11 artifact generation and validation commands. | commit                          | 1.8 KB   |
| .gitignore             | Keeps local caches, datasets, and generated videos out of default commits.           | commit                          | 375 B    |
| .gitattributes         | Marks generated evidence artifacts and binary files for cleaner GitHub review.       | commit                          | 404 B    |

## Key Quantitative Tables

| path                                                                            |   rows | size      |
|:--------------------------------------------------------------------------------|-------:|:----------|
| results/e11/step_metrics.csv                                                    |   2880 | 20.8 MB   |
| results/e11_equal_update/step_metrics.csv                                       |   2880 | 20.8 MB   |
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
