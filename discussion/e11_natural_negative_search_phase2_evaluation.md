# E11 Natural Negative Search Phase2 Evaluation

This generated evaluator is the multiplicity boundary for the registered
phase2 ResNet34 held-out architecture family. It does not claim a held-out
architecture replication, counterexample, or broader finite null until all
8 declared settings have metric rows and the Holm-adjusted primary decision
from paired per-seed log-ratio tests passes the reporting and quality gates.

Current primary metric coverage: 8/8 settings.
Current seed-level primary rows: 24.
Current phase2 caveat: head_gain_gate fails in 8/8 rows, while tail_quality_gate passes in 8/8 rows.

## Run Registry

| search_id                            | output_status   |   expected_settings |   settings_registry_rows |   pair_summary_rows | artifact_prefix                                                          |
|:-------------------------------------|:----------------|--------------------:|-------------------------:|--------------------:|:-------------------------------------------------------------------------|
| NNS-P2-heldout-architecture-boundary | complete        |                   8 |                        8 |                   8 | results/e11_natural_negative_search_protocol/phase2_heldout_architecture |

## Gate Report

| gate_id                              | status                | evidence                                                                                                |
|:-------------------------------------|:----------------------|:--------------------------------------------------------------------------------------------------------|
| NNS-P2-E1-evaluator-implemented      | pass                  | scripts/e11_evaluate_natural_negative_search_phase2.py wrote evaluator artifacts                        |
| NNS-P2-E2-phase2-output-completeness | pass                  | NNS-P2-heldout-architecture-boundary=complete (8/8 rows)                                                |
| NNS-P2-E3-primary-multiplicity       | pass                  | Holm one-sided p-values use paired per-seed log-ratio tests over the registered 8-setting phase2 family |
| NNS-P2-E4-heldout-architecture-claim | finite_null_candidate | all phase2 settings evaluated with no adjusted primary worse row                                        |
| NNS-P2-E5-full-reporting-boundary    | pass                  | primary_decisions.csv keeps every registered phase2 setting, including not-run rows                     |

## Primary Decision Boundary

| search_id                            | setting_id                                                 | output_status   | inference_source             |   raw_one_sided_p |   holm_adjusted_one_sided_p | adjusted_primary_decision   | claim_status                          |
|:-------------------------------------|:-----------------------------------------------------------|:----------------|:-----------------------------|------------------:|----------------------------:|:----------------------------|:--------------------------------------|
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p002_tail30      | observed        | paired_seed_log_ratio_t_test |            0.9969 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w2000_rho0p005_tail30      | observed        | paired_seed_log_ratio_t_test |            0.9941 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p002_tail30      | observed        | paired_seed_log_ratio_t_test |            0.9953 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_mod5_01_vs_34_w5000_rho0p005_tail30      | observed        | paired_seed_log_ratio_t_test |            0.9839 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p002_tail30 | observed        | paired_seed_log_ratio_t_test |            0.9999 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w2000_rho0p005_tail30 | observed        | paired_seed_log_ratio_t_test |            0.9985 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p002_tail30 | observed        | paired_seed_log_ratio_t_test |            0.9982 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |
| NNS-P2-heldout-architecture-boundary | resnet34_cifar100_decade_even_vs_odd_w5000_rho0p005_tail30 | observed        | paired_seed_log_ratio_t_test |            0.9912 |                           1 | not_primary_worse_adjusted  | no_primary_counterexample_for_setting |

Artifacts:
- [run_registry.csv](../results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/run_registry.csv)
- [seed_level_primary_ratios.csv](../results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/seed_level_primary_ratios.csv)
- [primary_decisions.csv](../results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/primary_decisions.csv)
- [gate_report.csv](../results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/gate_report.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation/config.json)
