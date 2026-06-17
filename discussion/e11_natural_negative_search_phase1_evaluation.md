# E11 Natural Negative Search Phase1 Evaluation

This generated evaluator is the multiplicity boundary for the registered
natural negative-search phase1 family. It does not claim a natural primary
counterexample until all 26 declared settings have metric rows and the Holm
adjusted primary decision passes the quality gates.

Current primary metric coverage: 0/26 settings.

## Run Registry

| search_id                                  | output_status            |   expected_settings |   settings_registry_rows |   pair_summary_rows | artifact_prefix                                                           |
|:-------------------------------------------|:-------------------------|--------------------:|-------------------------:|--------------------:|:--------------------------------------------------------------------------|
| NNS-P1-cifar100lt-resnet18-new-partitions  | settings_only_no_metrics |                  12 |                       12 |                   0 | results/e11_natural_negative_search_protocol/phase1_cifar100lt_resnet18   |
| NNS-P1-cifar10lt-resnet18-cross-partitions | settings_only_no_metrics |                   8 |                        8 |                   0 | results/e11_natural_negative_search_protocol/phase1_cifar10lt_resnet18    |
| NNS-P1-tail-quality-controls               | settings_only_no_metrics |                   6 |                        6 |                   0 | results/e11_natural_negative_search_protocol/phase1_tail_quality_controls |

## Gate Report

| gate_id                           | status    | evidence                                                                                                                                                                                                                         |
|:----------------------------------|:----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NNS-E1-evaluator-implemented      | pass      | scripts/e11_evaluate_natural_negative_search_phase1.py wrote evaluator artifacts                                                                                                                                                 |
| NNS-E2-phase1-output-completeness | not_ready | NNS-P1-cifar100lt-resnet18-new-partitions=settings_only_no_metrics (0/12 rows); NNS-P1-cifar10lt-resnet18-cross-partitions=settings_only_no_metrics (0/8 rows); NNS-P1-tail-quality-controls=settings_only_no_metrics (0/6 rows) |
| NNS-E3-primary-multiplicity       | not_ready | Holm one-sided p-values use the registered 26-setting phase1 family                                                                                                                                                              |
| NNS-E4-natural-primary-claim      | not_ready | 0/26 phase1 settings have primary metric rows                                                                                                                                                                                    |
| NNS-E5-full-reporting-boundary    | pass      | primary_decisions.csv keeps every registered setting, including not-run rows                                                                                                                                                     |

## Primary Decision Boundary

| search_id                                 | setting_id                                        | output_status   | raw_one_sided_p   | holm_adjusted_one_sided_p   | adjusted_primary_decision   | claim_status   |
|:------------------------------------------|:--------------------------------------------------|:----------------|:------------------|:----------------------------|:----------------------------|:---------------|
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p002_tail30       | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w500_rho0p005_tail30       | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p002_tail30  | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w500_rho0p005_tail30  | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 | not_run         | n/a               | n/a                         | pending_output              | not_ready      |
| NNS-P1-cifar100lt-resnet18-new-partitions | cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 | not_run         | n/a               | n/a                         | pending_output              | not_ready      |

Artifacts:
- [run_registry.csv](../results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/run_registry.csv)
- [primary_decisions.csv](../results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/primary_decisions.csv)
- [gate_report.csv](../results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/gate_report.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation/config.json)
