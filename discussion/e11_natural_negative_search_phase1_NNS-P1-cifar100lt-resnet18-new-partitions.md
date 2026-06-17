# E11 Natural Negative Search Phase1 Outputs

This generated artifact is the executable phase1 output boundary for the
pre-registered natural negative-search protocol. It preserves every selected
setting in the registry before any multiplicity-adjusted discovery decision.

## Settings

| search_id                                 | setting_id                                        | dataset      | partition_id                |   warmup_steps |   target_head_gain_fraction |   seed_count |   planned_family_size |
|:------------------------------------------|:--------------------------------------------------|:-------------|:----------------------------|---------------:|----------------------------:|-------------:|----------------------:|
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p002_tail30       | CIFAR-100-LT | cifar100_mod5_01_vs_34      |            500 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p005_tail30       | CIFAR-100-LT | cifar100_mod5_01_vs_34      |            500 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      | CIFAR-100-LT | cifar100_mod5_01_vs_34      |           2000 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      | CIFAR-100-LT | cifar100_mod5_01_vs_34      |           2000 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      | CIFAR-100-LT | cifar100_mod5_01_vs_34      |           5000 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      | CIFAR-100-LT | cifar100_mod5_01_vs_34      |           5000 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p002_tail30  | CIFAR-100-LT | cifar100_decade_even_vs_odd |            500 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p005_tail30  | CIFAR-100-LT | cifar100_decade_even_vs_odd |            500 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 | CIFAR-100-LT | cifar100_decade_even_vs_odd |           2000 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 | CIFAR-100-LT | cifar100_decade_even_vs_odd |           2000 |                       0.005 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 | CIFAR-100-LT | cifar100_decade_even_vs_odd |           5000 |                       0.002 |            5 |                    26 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 | CIFAR-100-LT | cifar100_decade_even_vs_odd |           5000 |                       0.005 |            5 |                    26 |

## Raw Primary Readout

| search_id                                 | setting_id                                        |   geomean_tail_output_drift_sq_ratio_spectral_over_fro |   tail_output_drift_sq_ratio_ci95_low |   tail_output_drift_sq_ratio_ci95_high |   mean_tail_accuracy_before |
|:------------------------------------------|:--------------------------------------------------|-------------------------------------------------------:|--------------------------------------:|---------------------------------------:|----------------------------:|
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p002_tail30       |                                                 0.8204 |                                0.8018 |                                 0.8394 |                     0.05325 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p005_tail30       |                                                 0.6799 |                                0.6197 |                                 0.7459 |                     0.05325 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      |                                                 0.7652 |                                0.7354 |                                 0.7962 |                     0.04725 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      |                                                 0.5309 |                                0.5    |                                 0.5638 |                     0.04725 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      |                                                 0.808  |                                0.7709 |                                 0.8469 |                     0.03963 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      |                                                 0.5894 |                                0.5484 |                                 0.6335 |                     0.03963 |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p002_tail30  |                                                 0.7328 |                                0.6987 |                                 0.7685 |                     0.0545  |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p005_tail30  |                                                 0.4682 |                                0.4081 |                                 0.537  |                     0.0545  |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 |                                                 0.7942 |                                0.7704 |                                 0.8187 |                     0.0422  |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 |                                                 0.5608 |                                0.5175 |                                 0.6077 |                     0.0422  |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 |                                                 0.8378 |                                0.7827 |                                 0.8967 |                     0.0363  |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 |                                                 0.6678 |                                0.556  |                                 0.802  |                     0.0363  |

## Decision Boundary

The table below is intentionally incomplete: adjusted confidence endpoints
must be added by the registered multiplicity evaluator before any natural
primary counterexample or finite-null claim is made.

| search_id                                 | setting_id                                        | metric_id                       | raw_worse_ci95_low_above_one   | adjustment_method              | adjusted_primary_decision   |
|:------------------------------------------|:--------------------------------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:----------------------------|
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p002_tail30       | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p005_tail30       | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p002_tail30  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p005_tail30  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |

Artifacts:
- [settings_registry.csv](../results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/settings_registry.csv)
- [step_metrics.csv](../results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/step_metrics.csv)
- [pair_summary.csv](../results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/pair_summary.csv)
- [layer_metrics.csv](../results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/layer_metrics.csv)
- [decision_template.csv](../results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/decision_template.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18/config.json)
