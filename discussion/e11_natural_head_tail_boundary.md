# E11 Natural Head-to-Tail Boundary Audit

This generated audit scans the already committed natural matched-head-gain
diagnostics for settings where the spectral/polar direction is worse than the
Frobenius direction. It is a fixed-rule scan over existing result tables, not a
replacement for a future pre-registered natural negative-search experiment.

## Boundary Rule

- Primary natural counterexample: the full tail-output logit-vector squared
  drift ratio has CI lower endpoint above 1.
- Component boundary candidate: a true-logit, competitor-logit, centered-drift,
  or margin-delta ratio has CI lower endpoint above 1.
- Secondary outcome tradeoff: a spectral-minus-Frobenius tail loss increase,
  margin drop, or accuracy drop has CI lower endpoint above 0.

The current narrow paper claim is about local full tail-output drift at matched
head gain. Component and outcome tradeoffs constrain stronger loss, margin,
accuracy, or predictive-score claims, but they do not by themselves falsify the
primary full-drift statement.

## Search Registry

| source_id                            | family                       |   rows | setting_axes                                              |   ratio_metric_count |   diff_metric_count | status   |
|:-------------------------------------|:-----------------------------|-------:|:----------------------------------------------------------|---------------------:|--------------------:|:---------|
| digits_one_step                      | sklearn_digits_mlp           |      1 | pooled                                                    |                    5 |                   3 | included |
| digits_imbalance_ablation            | sklearn_digits_mlp           |      4 | head_train_per_class,tail_train_per_class,imbalance_ratio |                    1 |                   3 | included |
| digits_checkpoint_sweep              | sklearn_digits_mlp           |      5 | warmup_steps                                              |                    5 |                   3 | included |
| digits_class_partition_sweep         | sklearn_digits_mlp           |      5 | partition_name,head_classes,tail_classes                  |                    5 |                   3 | included |
| digits_rho_sweep                     | sklearn_digits_mlp           |      5 | target_head_gain_fraction                                 |                    5 |                   3 | included |
| cifar100_resnet_one_step_rho005      | cifar100_lt_resnet18         |      1 | dataset,model                                             |                    5 |                   3 | included |
| cifar100_resnet_one_step_rho002      | cifar100_lt_resnet18         |      1 | dataset,model                                             |                    5 |                   3 | included |
| cifar100_resnet_checkpoint_sweep     | cifar100_lt_resnet18         |      4 | dataset,model,warmup_steps                                |                    5 |                   3 | included |
| cifar100_resnet_tail_quality_control | cifar100_resnet18_tail_rich  |      3 | dataset,model,warmup_steps                                |                    5 |                   3 | included |
| cifar100_resnet_imbalance_sweep      | cifar100_lt_resnet18         |      4 | dataset,model,tail_train_per_class,imbalance_ratio        |                    5 |                   3 | included |
| cifar100_resnet_fc_condition_scatter | cifar100_lt_resnet18_fc_only |      4 | warmup_steps                                              |                    5 |                   3 | included |

## Summary

| summary_id                |   row_count |   source_count |   metric_count |   spectral_better_count |   spectral_worse_count |   mixed_or_uncertain_count | claim_status                                               |
|:--------------------------|------------:|---------------:|---------------:|------------------------:|-----------------------:|---------------------------:|:-----------------------------------------------------------|
| primary_tail_output_drift |          37 |             11 |              1 |                      37 |                      0 |                          0 | no_strict_natural_primary_counterexample_in_committed_scan |
| all_ratio_metrics         |         169 |             11 |              5 |                     148 |                     17 |                          4 | ratio_metric_boundary_candidates_found                     |
| component_ratio_metrics   |         132 |             10 |              4 |                     111 |                     17 |                          4 | component_boundary_candidates_found                        |
| secondary_tail_outcomes   |         111 |             11 |              3 |                      20 |                     41 |                         50 | secondary_outcome_tradeoffs_found                          |

Primary status: `no_strict_natural_primary_counterexample_in_committed_scan`.

## Worst Primary Full-Drift Settings

| source_id                            | setting_values                                                                        |   ratio_spectral_over_fro |   ci95_low |   ci95_high | boundary_status   |
|:-------------------------------------|:--------------------------------------------------------------------------------------|--------------------------:|-----------:|------------:|:------------------|
| cifar100_resnet_imbalance_sweep      | dataset=CIFAR100;model=resnet18_cifar_stem;tail_train_per_class=100;imbalance_ratio=3 |                    0.6075 |     0.3943 |      0.936  | spectral_better   |
| cifar100_resnet_one_step_rho002      | dataset=CIFAR100;model=resnet18_cifar_stem                                            |                    0.7761 |     0.7585 |      0.7941 | spectral_better   |
| cifar100_resnet_tail_quality_control | dataset=CIFAR100;model=resnet18_cifar_stem;warmup_steps=2000                          |                    0.7292 |     0.6923 |      0.768  | spectral_better   |
| cifar100_resnet_imbalance_sweep      | dataset=CIFAR100;model=resnet18_cifar_stem;tail_train_per_class=30;imbalance_ratio=10 |                    0.5369 |     0.3943 |      0.7312 | spectral_better   |
| digits_checkpoint_sweep              | warmup_steps=160                                                                      |                    0.6406 |     0.5803 |      0.7071 | spectral_better   |
| cifar100_resnet_imbalance_sweep      | dataset=CIFAR100;model=resnet18_cifar_stem;tail_train_per_class=10;imbalance_ratio=30 |                    0.5355 |     0.4227 |      0.6785 | spectral_better   |
| digits_class_partition_sweep         | partition_name=mixed_c;head_classes=0,2,3,5,9;tail_classes=1,4,6,7,8                  |                    0.6135 |     0.5762 |      0.6532 | spectral_better   |
| cifar100_resnet_tail_quality_control | dataset=CIFAR100;model=resnet18_cifar_stem;warmup_steps=5000                          |                    0.5081 |     0.3986 |      0.6477 | spectral_better   |

## Strict Negative Candidates

| scan   | claim_boundary            | source_id                            | setting_values                                                           | metric_id        |   estimate |   ci95_low |   ci95_high |
|:-------|:--------------------------|:-------------------------------------|:-------------------------------------------------------------------------|:-----------------|-----------:|-----------:|------------:|
| ratio  | component_metric_boundary | cifar100_resnet_fc_condition_scatter | warmup_steps=250                                                         | true_logit_delta |      1.988 |      1.575 |       2.511 |
| ratio  | component_metric_boundary | cifar100_resnet_fc_condition_scatter | warmup_steps=500                                                         | true_logit_delta |      1.521 |      1.103 |       2.096 |
| ratio  | component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=120                                                         | true_logit_delta |      5.455 |      2.465 |      12.07  |
| ratio  | component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=160                                                         | true_logit_delta |      4.986 |      2.355 |      10.56  |
| ratio  | component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=20                                                          | true_logit_delta |      2.69  |      2.188 |       3.308 |
| ratio  | component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=80                                                          | true_logit_delta |      1.68  |      1.16  |       2.431 |
| ratio  | component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=40                                                          | true_logit_delta |      1.569 |      1.162 |       2.117 |
| ratio  | component_metric_boundary | digits_class_partition_sweep         | partition_name=mixed_c;head_classes=0,2,3,5,9;tail_classes=1,4,6,7,8     | true_logit_delta |      7.276 |      3.663 |      14.45  |
| ratio  | component_metric_boundary | digits_class_partition_sweep         | partition_name=even_vs_odd;head_classes=0,2,4,6,8;tail_classes=1,3,5,7,9 | true_logit_delta |      2.02  |      1.393 |       2.931 |
| ratio  | component_metric_boundary | digits_class_partition_sweep         | partition_name=mixed_a;head_classes=0,1,5,6,7;tail_classes=2,3,4,8,9     | true_logit_delta |      1.865 |      1.102 |       3.158 |
| ratio  | component_metric_boundary | digits_class_partition_sweep         | partition_name=low_vs_high;head_classes=0,1,2,3,4;tail_classes=5,6,7,8,9 | true_logit_delta |      1.68  |      1.16  |       2.431 |
| ratio  | component_metric_boundary | digits_one_step                      | pooled                                                                   | true_logit_delta |      1.68  |      1.16  |       2.431 |
| ratio  | component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.08                                           | true_logit_delta |      1.68  |      1.161 |       2.431 |
| ratio  | component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.04                                           | true_logit_delta |      1.68  |      1.16  |       2.431 |
| ratio  | component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.02                                           | true_logit_delta |      1.68  |      1.16  |       2.431 |
| ratio  | component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.005                                          | true_logit_delta |      1.679 |      1.16  |       2.43  |

## Ratio-Metric Boundary Candidates

| claim_boundary            | source_id                            | setting_values                                                           | metric_id        |   estimate |   ci95_low |   ci95_high |
|:--------------------------|:-------------------------------------|:-------------------------------------------------------------------------|:-----------------|-----------:|-----------:|------------:|
| component_metric_boundary | cifar100_resnet_fc_condition_scatter | warmup_steps=250                                                         | true_logit_delta |      1.988 |      1.575 |       2.511 |
| component_metric_boundary | cifar100_resnet_fc_condition_scatter | warmup_steps=500                                                         | true_logit_delta |      1.521 |      1.103 |       2.096 |
| component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=120                                                         | true_logit_delta |      5.455 |      2.465 |      12.07  |
| component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=160                                                         | true_logit_delta |      4.986 |      2.355 |      10.56  |
| component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=20                                                          | true_logit_delta |      2.69  |      2.188 |       3.308 |
| component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=80                                                          | true_logit_delta |      1.68  |      1.16  |       2.431 |
| component_metric_boundary | digits_checkpoint_sweep              | warmup_steps=40                                                          | true_logit_delta |      1.569 |      1.162 |       2.117 |
| component_metric_boundary | digits_class_partition_sweep         | partition_name=mixed_c;head_classes=0,2,3,5,9;tail_classes=1,4,6,7,8     | true_logit_delta |      7.276 |      3.663 |      14.45  |
| component_metric_boundary | digits_class_partition_sweep         | partition_name=even_vs_odd;head_classes=0,2,4,6,8;tail_classes=1,3,5,7,9 | true_logit_delta |      2.02  |      1.393 |       2.931 |
| component_metric_boundary | digits_class_partition_sweep         | partition_name=mixed_a;head_classes=0,1,5,6,7;tail_classes=2,3,4,8,9     | true_logit_delta |      1.865 |      1.102 |       3.158 |
| component_metric_boundary | digits_class_partition_sweep         | partition_name=low_vs_high;head_classes=0,1,2,3,4;tail_classes=5,6,7,8,9 | true_logit_delta |      1.68  |      1.16  |       2.431 |
| component_metric_boundary | digits_one_step                      | pooled                                                                   | true_logit_delta |      1.68  |      1.16  |       2.431 |
| component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.08                                           | true_logit_delta |      1.68  |      1.161 |       2.431 |
| component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.04                                           | true_logit_delta |      1.68  |      1.16  |       2.431 |
| component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.02                                           | true_logit_delta |      1.68  |      1.16  |       2.431 |
| component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.005                                          | true_logit_delta |      1.679 |      1.16  |       2.43  |
| component_metric_boundary | digits_rho_sweep                     | target_head_gain_fraction=0.01                                           | true_logit_delta |      1.679 |      1.16  |       2.43  |

## Secondary Outcome Tradeoffs

| source_id                            | setting_values                                                                        | metric_id          |   estimate |   ci95_low |   ci95_high |
|:-------------------------------------|:--------------------------------------------------------------------------------------|:-------------------|-----------:|-----------:|------------:|
| cifar100_resnet_checkpoint_sweep     | dataset=CIFAR100;model=resnet18_cifar_stem;warmup_steps=2000                          | tail_accuracy_drop |  0.00025   |  3.087e-05 |   0.0004691 |
| cifar100_resnet_checkpoint_sweep     | dataset=CIFAR100;model=resnet18_cifar_stem;warmup_steps=250                           | tail_loss_increase |  0.001142  |  0.0008175 |   0.001467  |
| cifar100_resnet_checkpoint_sweep     | dataset=CIFAR100;model=resnet18_cifar_stem;warmup_steps=500                           | tail_loss_increase |  0.000209  |  7.468e-05 |   0.0003433 |
| cifar100_resnet_checkpoint_sweep     | dataset=CIFAR100;model=resnet18_cifar_stem;warmup_steps=250                           | tail_margin_drop   |  0.001603  |  0.001154  |   0.002051  |
| cifar100_resnet_checkpoint_sweep     | dataset=CIFAR100;model=resnet18_cifar_stem;warmup_steps=500                           | tail_margin_drop   |  0.0002596 |  8.222e-05 |   0.000437  |
| cifar100_resnet_fc_condition_scatter | warmup_steps=250                                                                      | tail_loss_increase |  0.002237  |  0.001867  |   0.002606  |
| cifar100_resnet_fc_condition_scatter | warmup_steps=500                                                                      | tail_loss_increase |  0.0006773 |  0.0004985 |   0.0008561 |
| cifar100_resnet_fc_condition_scatter | warmup_steps=250                                                                      | tail_margin_drop   |  0.003555  |  0.002938  |   0.004172  |
| cifar100_resnet_fc_condition_scatter | warmup_steps=500                                                                      | tail_margin_drop   |  0.001006  |  0.0007545 |   0.001258  |
| cifar100_resnet_imbalance_sweep      | dataset=CIFAR100;model=resnet18_cifar_stem;tail_train_per_class=300;imbalance_ratio=1 | tail_loss_increase |  0.000634  |  0.0004147 |   0.0008533 |
| cifar100_resnet_imbalance_sweep      | dataset=CIFAR100;model=resnet18_cifar_stem;tail_train_per_class=300;imbalance_ratio=1 | tail_margin_drop   |  0.0009501 |  0.0006677 |   0.001232  |
| digits_checkpoint_sweep              | warmup_steps=20                                                                       | tail_loss_increase |  0.005737  |  0.004906  |   0.006567  |

## Claim Boundary

This audit supports only the following scoped statement: among the committed
natural matched-head-gain diagnostics scanned here, the primary full tail-output
drift metric has the status shown above. If the paper needs natural negative
examples, the next step is a pre-registered search over additional natural
architectures, data partitions, and checkpoints using the same CI rules and
reporting table.
