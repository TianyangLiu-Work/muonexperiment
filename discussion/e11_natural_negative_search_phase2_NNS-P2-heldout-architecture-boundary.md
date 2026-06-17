# E11 Natural Negative Search Phase2 Held-Out Architecture Outputs

This generated artifact freezes the phase2 held-out architecture family for
the registered natural negative-search protocol. It uses ResNet34 as the
architecture transport check after the complete phase1 ResNet18 family and
keeps the phase1 partitions fixed before any phase2 metric output is used.

## Settings

| search_id                            | setting_id                                                 | dataset      | architecture        | partition_id                |   warmup_steps |   target_head_gain_fraction |   seed_count |   planned_family_size |
|:-------------------------------------|:-----------------------------------------------------------|:-------------|:--------------------|:----------------------------|---------------:|----------------------------:|-------------:|----------------------:|
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_mod5_01_vs_34      |           2000 |                       0.002 |            3 |                     8 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_mod5_01_vs_34      |           2000 |                       0.005 |            3 |                     8 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_mod5_01_vs_34      |           5000 |                       0.002 |            3 |                     8 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_mod5_01_vs_34      |           5000 |                       0.005 |            3 |                     8 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_decade_even_vs_odd |           2000 |                       0.002 |            3 |                     8 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_decade_even_vs_odd |           2000 |                       0.005 |            3 |                     8 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_decade_even_vs_odd |           5000 |                       0.002 |            3 |                     8 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 | CIFAR-100-LT | ResNet34 CIFAR stem | cifar100_decade_even_vs_odd |           5000 |                       0.005 |            3 |                     8 |

## Raw Primary Readout

| search_id                            | setting_id                                                 | model               |   geomean_tail_output_drift_sq_ratio_spectral_over_fro |   tail_output_drift_sq_ratio_ci95_low |   tail_output_drift_sq_ratio_ci95_high |   mean_tail_accuracy_before |
|:-------------------------------------|:-----------------------------------------------------------|:--------------------|-------------------------------------------------------:|--------------------------------------:|---------------------------------------:|----------------------------:|
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      | resnet34_cifar_stem |                                                 0.8454 |                                0.798  |                                 0.8955 |                     0.04562 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      | resnet34_cifar_stem |                                                 0.7369 |                                0.6383 |                                 0.8508 |                     0.04562 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      | resnet34_cifar_stem |                                                 0.8491 |                                0.7929 |                                 0.9092 |                     0.04708 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      | resnet34_cifar_stem |                                                 0.6642 |                                0.4804 |                                 0.9183 |                     0.04708 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 | resnet34_cifar_stem |                                                 0.8746 |                                0.868  |                                 0.8812 |                     0.04883 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 | resnet34_cifar_stem |                                                 0.7909 |                                0.7485 |                                 0.8356 |                     0.04883 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 | resnet34_cifar_stem |                                                 0.8439 |                                0.8078 |                                 0.8815 |                     0.04333 |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 | resnet34_cifar_stem |                                                 0.7033 |                                0.5738 |                                 0.862  |                     0.04333 |

## Decision Boundary

Phase2 decisions are claim-boundary evidence only until every declared
phase2 row is evaluated with the same paired per-seed primary rule.

| search_id                            | setting_id                                                 | metric_id                       | raw_worse_ci95_low_above_one   | adjustment_method                     | adjusted_primary_decision   |
|:-------------------------------------|:-----------------------------------------------------------|:--------------------------------|:-------------------------------|:--------------------------------------|:----------------------------|
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 | primary_tail_output_drift_ratio | no                             | pending_phase2_multiplicity_evaluator | pending                     |

Artifacts:
- [settings_registry.csv](../results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv)
- [step_metrics.csv](../results/e11_natural_negative_search_protocol/phase2_heldout_architecture/step_metrics.csv)
- [pair_summary.csv](../results/e11_natural_negative_search_protocol/phase2_heldout_architecture/pair_summary.csv)
- [layer_metrics.csv](../results/e11_natural_negative_search_protocol/phase2_heldout_architecture/layer_metrics.csv)
- [decision_template.csv](../results/e11_natural_negative_search_protocol/phase2_heldout_architecture/decision_template.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase2_heldout_architecture/config.json)
