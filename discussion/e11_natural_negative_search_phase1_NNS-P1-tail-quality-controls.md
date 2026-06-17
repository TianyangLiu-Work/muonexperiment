# E11 Natural Negative Search Phase1 Outputs

This generated artifact is the executable phase1 output boundary for the
pre-registered natural negative-search protocol. It preserves every selected
setting in the registry before any multiplicity-adjusted discovery decision.

## Settings

| search_id                    | setting_id                                                  | dataset   | partition_id                        |   warmup_steps |   target_head_gain_fraction |   seed_count |   planned_family_size |
|:-----------------------------|:------------------------------------------------------------|:----------|:------------------------------------|---------------:|----------------------------:|-------------:|----------------------:|
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w2000_rho0p002_tail300  | CIFAR-100 | cifar100_tail_quality_mod5_12_vs_04 |           2000 |                       0.002 |            5 |                    26 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w2000_rho0p005_tail300  | CIFAR-100 | cifar100_tail_quality_mod5_12_vs_04 |           2000 |                       0.005 |            5 |                    26 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w5000_rho0p002_tail300  | CIFAR-100 | cifar100_tail_quality_mod5_12_vs_04 |           5000 |                       0.002 |            5 |                    26 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w5000_rho0p005_tail300  | CIFAR-100 | cifar100_tail_quality_mod5_12_vs_04 |           5000 |                       0.005 |            5 |                    26 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w10000_rho0p002_tail300 | CIFAR-100 | cifar100_tail_quality_mod5_12_vs_04 |          10000 |                       0.002 |            5 |                    26 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w10000_rho0p005_tail300 | CIFAR-100 | cifar100_tail_quality_mod5_12_vs_04 |          10000 |                       0.005 |            5 |                    26 |

## Raw Primary Readout

| search_id                    | setting_id                                                  |   geomean_tail_output_drift_sq_ratio_spectral_over_fro |   tail_output_drift_sq_ratio_ci95_low |   tail_output_drift_sq_ratio_ci95_high |   mean_tail_accuracy_before |
|:-----------------------------|:------------------------------------------------------------|-------------------------------------------------------:|--------------------------------------:|---------------------------------------:|----------------------------:|
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w2000_rho0p002_tail300  |                                                 0.7951 |                                0.745  |                                 0.8486 |                      0.3279 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w2000_rho0p005_tail300  |                                                 0.5215 |                                0.4398 |                                 0.6184 |                      0.3279 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w5000_rho0p002_tail300  |                                                 0.8139 |                                0.7925 |                                 0.8359 |                      0.3536 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w5000_rho0p005_tail300  |                                                 0.6072 |                                0.5403 |                                 0.6824 |                      0.3536 |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w10000_rho0p002_tail300 |                                                 0.7988 |                                0.7359 |                                 0.8671 |                      0.363  |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w10000_rho0p005_tail300 |                                                 0.528  |                                0.4044 |                                 0.6894 |                      0.363  |

## Decision Boundary

The table below is intentionally incomplete: adjusted confidence endpoints
must be added by the registered multiplicity evaluator before any natural
primary counterexample or finite-null claim is made.

| search_id                    | setting_id                                                  | metric_id                       | raw_worse_ci95_low_above_one   | adjustment_method              | adjusted_primary_decision   |
|:-----------------------------|:------------------------------------------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:----------------------------|
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w2000_rho0p002_tail300  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w2000_rho0p005_tail300  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w5000_rho0p002_tail300  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w5000_rho0p005_tail300  | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w10000_rho0p002_tail300 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |
| NNS-P1-tail-quality-controls | cifar100_tail_quality_mod5_12_vs_04_w10000_rho0p005_tail300 | primary_tail_output_drift_ratio | no                             | pending_multiplicity_evaluator | pending                     |

Artifacts:
- [settings_registry.csv](../results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/settings_registry.csv)
- [step_metrics.csv](../results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/step_metrics.csv)
- [pair_summary.csv](../results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/pair_summary.csv)
- [layer_metrics.csv](../results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/layer_metrics.csv)
- [decision_template.csv](../results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/decision_template.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase1_tail_quality_controls/config.json)
