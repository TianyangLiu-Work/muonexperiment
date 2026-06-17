# E11 Natural Negative Search Phase1 Outputs

This generated artifact is the executable phase1 output boundary for the
pre-registered natural negative-search protocol. It preserves every selected
setting in the registry before any multiplicity-adjusted discovery decision.

## Settings

| search_id                                  | setting_id                            | dataset     | partition_id    |   warmup_steps |   target_head_gain_fraction |   seed_count |   planned_family_size |
|:-------------------------------------------|:--------------------------------------|:------------|:----------------|---------------:|----------------------------:|-------------:|----------------------:|
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w500_rho0p002_tail30  | CIFAR-10-LT | cifar10_cross_a |            500 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w500_rho0p005_tail30  | CIFAR-10-LT | cifar10_cross_a |            500 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w5000_rho0p002_tail30 | CIFAR-10-LT | cifar10_cross_a |           5000 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w5000_rho0p005_tail30 | CIFAR-10-LT | cifar10_cross_a |           5000 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w500_rho0p002_tail30  | CIFAR-10-LT | cifar10_cross_b |            500 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w500_rho0p005_tail30  | CIFAR-10-LT | cifar10_cross_b |            500 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w5000_rho0p002_tail30 | CIFAR-10-LT | cifar10_cross_b |           5000 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w5000_rho0p005_tail30 | CIFAR-10-LT | cifar10_cross_b |           5000 |                       0.005 |            5 |                    26 |

## Raw Primary Readout

| search_id                                  | setting_id                            |   geomean_tail_output_drift_sq_ratio_spectral_over_fro |   tail_output_drift_sq_ratio_ci95_low |   tail_output_drift_sq_ratio_ci95_high |   mean_tail_accuracy_before |
|:-------------------------------------------|:--------------------------------------|-------------------------------------------------------:|--------------------------------------:|---------------------------------------:|----------------------------:|
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w500_rho0p002_tail30  |                                                 0.6617 |                                0.6154 |                                 0.7115 |                       0.014 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w500_rho0p005_tail30  |                                                 0.3464 |                                0.3087 |                                 0.3885 |                       0.014 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w5000_rho0p002_tail30 |                                                 0.6667 |                                0.6306 |                                 0.7049 |                       0.008 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w5000_rho0p005_tail30 |                                                 0.3895 |                                0.3484 |                                 0.4355 |                       0.008 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w500_rho0p002_tail30  |                                                 0.7143 |                                0.6575 |                                 0.7761 |                       0.079 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w500_rho0p005_tail30  |                                                 0.4013 |                                0.3601 |                                 0.4471 |                       0.079 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w5000_rho0p002_tail30 |                                                 0.6809 |                                0.6129 |                                 0.7565 |                       0.056 |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w5000_rho0p005_tail30 |                                                 0.4459 |                                0.3997 |                                 0.4974 |                       0.056 |

## Decision Boundary

The table below is intentionally incomplete: adjusted confidence endpoints
must be added by the registered multiplicity evaluator before any natural
primary counterexample or finite-null claim is made.

| search_id                                  | setting_id                            | metric_id                       | raw_worse_ci95_low_above_one   | adjustment_method              | adjusted_primary_decision   |
|:-------------------------------------------|:--------------------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:----------------------------|
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w500_rho0p002_tail30  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w500_rho0p005_tail30  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w5000_rho0p002_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_a_w5000_rho0p005_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w500_rho0p002_tail30  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w500_rho0p005_tail30  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w5000_rho0p002_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar10lt-resnet18-cross-partitions | cifar10_cross_b_w5000_rho0p005_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |

Artifacts:
- [settings_registry.csv](../results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/settings_registry.csv)
- [step_metrics.csv](../results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/step_metrics.csv)
- [pair_summary.csv](../results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/pair_summary.csv)
- [layer_metrics.csv](../results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/layer_metrics.csv)
- [decision_template.csv](../results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/decision_template.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18/config.json)
