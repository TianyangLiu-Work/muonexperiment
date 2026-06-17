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

Artifacts:
- [settings_registry.csv](../results/e11_natural_negative_search_protocol/phase2_heldout_architecture/settings_registry.csv)

No phase2 metric rows are present in this settings-only freeze.
The Slurm run will write `step_metrics.csv`, `pair_summary.csv`,
`layer_metrics.csv`, `decision_template.csv`, and `config.json`.
