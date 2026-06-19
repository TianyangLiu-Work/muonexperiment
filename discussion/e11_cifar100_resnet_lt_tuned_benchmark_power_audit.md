# E11 CIFAR-100-LT Tuned Benchmark Power Audit

This generated audit is a pre-output detectable-effect contract for the tuned
CIFAR-100-LT ResNet18 benchmark path. It reads the registered validation/final
seed split, validation-only selection audit, and acceptance gates, but it does
not inspect final seed outputs. Its purpose is to fix the paired-seed power,
Holm family, all-class-collapse guardrail, and final claim ladder before any
tuned benchmark final rows exist.

Current validation-selected families: 3/6.
The registered final split has 10 paired seeds and the primary
few-class improvement family has 4 Holm-adjusted Muon-vs-baseline
comparisons. With paired-diff SD 0.03, the Holm worst-case detectable few-class
balanced-accuracy gain is 0.02951.
The all-class guardrail with margin -0.01 needs mean all-class diff at least
0.01951 at the same
SD before a clean benchmark win can be claimed.

## Final Family Design

| recipe_family          | optimizer   | validation_selection_status   | final_status        | final_seed_set   |   final_seed_count | final_output_status   |
|:-----------------------|:------------|:------------------------------|:--------------------|:-----------------|-------------------:|:----------------------|
| adamw_ce_tuned         | adamw       | selected                      | ready_for_final_run | 20..29           |                 10 | absent                |
| sgd_momentum_ce_tuned  | sgd         | selected                      | ready_for_final_run | 20..29           |                 10 | absent                |
| adamw_cb_loss_tuned    | adamw       | selected                      | ready_for_final_run | 20..29           |                 10 | absent                |
| adamw_cb_sampler_tuned | adamw       | not_ready                     | not_ready           | 20..29           |                 10 | absent                |
| ns_muon_matrix_tuned   | ns_muon     | not_ready                     | not_ready           | 20..29           |                 10 | absent                |
| ns_muon_cb_tuned       | ns_muon     | not_ready                     | not_ready           | 20..29           |                 10 | absent                |

## Primary Comparison Plan

| comparison_id                                 | candidate_family     | baseline_family       | metric                     |   family_size | pass_rule                                                           |
|:----------------------------------------------|:---------------------|:----------------------|:---------------------------|--------------:|:--------------------------------------------------------------------|
| ns_muon_matrix_tuned_vs_adamw_ce_tuned        | ns_muon_matrix_tuned | adamw_ce_tuned        | few_balanced_accuracy_diff |             4 | paired mean diff CI lower endpoint above zero after Holm adjustment |
| ns_muon_matrix_tuned_vs_sgd_momentum_ce_tuned | ns_muon_matrix_tuned | sgd_momentum_ce_tuned | few_balanced_accuracy_diff |             4 | paired mean diff CI lower endpoint above zero after Holm adjustment |
| ns_muon_cb_tuned_vs_adamw_ce_tuned            | ns_muon_cb_tuned     | adamw_ce_tuned        | few_balanced_accuracy_diff |             4 | paired mean diff CI lower endpoint above zero after Holm adjustment |
| ns_muon_cb_tuned_vs_sgd_momentum_ce_tuned     | ns_muon_cb_tuned     | sgd_momentum_ce_tuned | few_balanced_accuracy_diff |             4 | paired mean diff CI lower endpoint above zero after Holm adjustment |

## Paired Few-Class MDE

| alpha_scope                |   family_size |   final_seed_count |   paired_diff_sd |   minimum_detectable_paired_mean_diff | claim_interpretation                                                                                                                                                |
|:---------------------------|--------------:|-------------------:|-----------------:|--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| raw_single_comparison      |             1 |                 10 |             0.01 |                              0.007154 | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| raw_single_comparison      |             1 |                 10 |             0.02 |                              0.01431  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| raw_single_comparison      |             1 |                 10 |             0.03 |                              0.02146  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| raw_single_comparison      |             1 |                 10 |             0.05 |                              0.03577  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| raw_single_comparison      |             1 |                 10 |             0.08 |                              0.05723  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| raw_single_comparison      |             1 |                 10 |             0.1  |                              0.07154  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| holm_bonferroni_worst_case |             4 |                 10 |             0.01 |                              0.009838 | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| holm_bonferroni_worst_case |             4 |                 10 |             0.02 |                              0.01968  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| holm_bonferroni_worst_case |             4 |                 10 |             0.03 |                              0.02951  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| holm_bonferroni_worst_case |             4 |                 10 |             0.05 |                              0.04919  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| holm_bonferroni_worst_case |             4 |                 10 |             0.08 |                              0.0787   | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |
| holm_bonferroni_worst_case |             4 |                 10 |             0.1  |                              0.09838  | final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy effects at or above this scale under the assumed across-seed paired-diff SD |

## All-Class Guardrail

| alpha_scope                |   family_size |   paired_diff_sd |   noninferiority_margin |   minimum_mean_all_accuracy_diff_to_pass | claim_interpretation                                                                                                                                                  |
|:---------------------------|--------------:|-----------------:|------------------------:|-----------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| raw_single_comparison      |             1 |             0.01 |                   -0.01 |                               -0.002846  | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| raw_single_comparison      |             1 |             0.02 |                   -0.01 |                                0.004307  | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| raw_single_comparison      |             1 |             0.03 |                   -0.01 |                                0.01146   | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| raw_single_comparison      |             1 |             0.05 |                   -0.01 |                                0.02577   | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| raw_single_comparison      |             1 |             0.08 |                   -0.01 |                                0.04723   | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| raw_single_comparison      |             1 |             0.1  |                   -0.01 |                                0.06154   | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| holm_bonferroni_worst_case |             4 |             0.01 |                   -0.01 |                               -0.0001624 | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| holm_bonferroni_worst_case |             4 |             0.02 |                   -0.01 |                                0.009675  | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| holm_bonferroni_worst_case |             4 |             0.03 |                   -0.01 |                                0.01951   | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| holm_bonferroni_worst_case |             4 |             0.05 |                   -0.01 |                                0.03919   | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| holm_bonferroni_worst_case |             4 |             0.08 |                   -0.01 |                                0.0687    | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |
| holm_bonferroni_worst_case |             4 |             0.1  |                   -0.01 |                                0.08838   | the all-class guardrail is passed only when the paired mean all-balanced-accuracy difference is above this value, so few-class gains are not bought by broad collapse |

## Interpretation Ladder

| case_id                               | evidence_condition                                                                                                    | allowed_wording                                                                         | blocked_wording                                                                    |
|:--------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| TB-PWR-1-validation-incomplete        | validation grid incomplete or family selection not_ready                                                              | registered benchmark power audit only; no final-performance result                      | tuned optimizer performance claim                                                  |
| TB-PWR-2-positive-primary             | all final recipes complete; Muon few-class diff beats AdamW and SGD after Holm adjustment; all-class guardrail passes | CIFAR-100-LT ResNet18 practical Muon improvement under the registered tuned protocol    | broad long-tail optimizer superiority across datasets or architectures             |
| TB-PWR-3-underpowered-null            | final mean few-class differences are below the audited MDE for observed paired-diff SD                                | underpowered or small-effect tuned benchmark boundary                                   | Muon has been ruled out as a practical optimizer                                   |
| TB-PWR-4-negative-at-detectable-scale | Muon fails tuned baselines at or above the audited detectable-effect scale                                            | negative tuned-performance boundary under the registered CIFAR-100-LT ResNet18 protocol | suppressing negative tuned outcomes while keeping optimizer-performance motivation |
| TB-PWR-5-tradeoff-only                | few-class gain passes but all-class guardrail fails                                                                   | few/all tradeoff result, not a clean practical optimizer improvement                    | unqualified benchmark win                                                          |
| TB-PWR-6-reporting-incomplete         | per-seed rows, many/medium/few/all metrics, or paired differences are missing                                         | appendix-only incomplete benchmark progress                                             | main benchmark claim                                                               |

## Outcome State Machine

| state_id                                     | trigger                                                                          | claim_state                                  | required_action                                                                 |
|:---------------------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------|:--------------------------------------------------------------------------------|
| TB-PWR-S1-not-ready                          | selection gates TVS-1/TVS-2/TVS-4/TVS-5 are not_ready                            | not_ready                                    | run validation grid, occupancy logging, and selection audit before final seeds  |
| TB-PWR-S2-final-quarantine-broken            | any final output exists before validation family selection is complete           | protocol_violation                           | do not use those final rows for a tuned-performance claim                       |
| TB-PWR-S3-final-family-complete-positive     | all selected recipes have 10 paired final seeds and primary/guardrail gates pass | cifar100lt_resnet18_benchmark_claim_eligible | report adjusted comparisons, per-group metrics, CIs, and exact scope            |
| TB-PWR-S4-final-family-complete-underpowered | primary comparisons fail but observed effect scale is below audited MDE          | underpowered_tuned_boundary                  | state MDE floor and add a larger preregistered final family before broad claims |
| TB-PWR-S5-final-family-complete-negative     | Muon fails baselines at or above audited MDE or all-class guardrail fails        | negative_or_tradeoff_tuned_boundary          | preserve negative result and keep paper on the local mechanism path             |

Artifacts:
- [final_family_design.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/final_family_design.csv)
- [primary_comparison_plan.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/primary_comparison_plan.csv)
- [paired_diff_mde.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/paired_diff_mde.csv)
- [all_class_guardrail_mde.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/all_class_guardrail_mde.csv)
- [interpretation_ladder.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/interpretation_ladder.csv)
- [outcome_state_machine.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/outcome_state_machine.csv)
- [config.json](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_power_audit/config.json)
