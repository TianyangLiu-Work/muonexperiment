# E11 CIFAR-100-LT Tuned Benchmark Final Analysis Plan

This generated audit fixes the statistical analysis surface for the registered tuned benchmark before final seeds are available. It does not inspect final outputs and does not authorize a benchmark claim while validation selection remains incomplete.

## Gate Matrix

| gate_id                          | status    | evidence                                                                         | claim_effect                                                      |
|:---------------------------------|:----------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------|
| FAP-1-final-quarantine           | pass      | no final_claim files exist                                                       | analysis plan remains pre-final                                   |
| FAP-2-family-selection-readiness | not_ready | 1/6 families ready for final run                                                 | final analysis cannot run until one recipe per family is selected |
| FAP-3-power-audit-linked         | pass      | final power audit locks 10 final paired seeds                                    | final analysis uses the pre-output MDE contract                   |
| FAP-4-variance-prior-linked      | pass      | variance-prior audit is pre-final and does not inspect final seed outputs        | MDE sensitivity remains separated from final outcomes             |
| FAP-5-primary-family-fixed       | pass      | four Muon-vs-baseline primary comparisons are fixed before final outputs         | blocks cherry-picked final comparison wording                     |
| FAP-6-reporting-schema-fixed     | pass      | per-seed, paired-comparison, final-summary, and occupancy tables are predeclared | blocks selective metric reporting                                 |

## Analysis Input Contract

| input_id                      | recipe_family          | selection_status    | selected_setting_id                    | final_seed_set   | required_files                                                                          | pairing_key   | claim_boundary                                                               |
|:------------------------------|:-----------------------|:--------------------|:---------------------------------------|:-----------------|:----------------------------------------------------------------------------------------|:--------------|:-----------------------------------------------------------------------------|
| FIN-IN-adamw_cb_loss_tuned    | adamw_cb_loss_tuned    | not_ready           | n/a                                    | 20..29           | summary.csv; group_metrics.csv; class_metrics.csv; train_trace.csv; occupancy_trace.csv | seed          | required for final analysis only after all six families are selected and run |
| FIN-IN-adamw_cb_sampler_tuned | adamw_cb_sampler_tuned | not_ready           | n/a                                    | 20..29           | summary.csv; group_metrics.csv; class_metrics.csv; train_trace.csv; occupancy_trace.csv | seed          | required for final analysis only after all six families are selected and run |
| FIN-IN-adamw_ce_tuned         | adamw_ce_tuned         | ready_for_final_run | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0 | 20..29           | summary.csv; group_metrics.csv; class_metrics.csv; train_trace.csv; occupancy_trace.csv | seed          | required for final analysis only after all six families are selected and run |
| FIN-IN-ns_muon_cb_tuned       | ns_muon_cb_tuned       | not_ready           | n/a                                    | 20..29           | summary.csv; group_metrics.csv; class_metrics.csv; train_trace.csv; occupancy_trace.csv | seed          | required for final analysis only after all six families are selected and run |
| FIN-IN-ns_muon_matrix_tuned   | ns_muon_matrix_tuned   | not_ready           | n/a                                    | 20..29           | summary.csv; group_metrics.csv; class_metrics.csv; train_trace.csv; occupancy_trace.csv | seed          | required for final analysis only after all six families are selected and run |
| FIN-IN-sgd_momentum_ce_tuned  | sgd_momentum_ce_tuned  | not_ready           | n/a                                    | 20..29           | summary.csv; group_metrics.csv; class_metrics.csv; train_trace.csv; occupancy_trace.csv | seed          | required for final analysis only after all six families are selected and run |

## Metric Contract

| metric_id                             | frequency_group   | statistic                                                                         | role                                                         | direction                                                   | claim_gate                        |
|:--------------------------------------|:------------------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------|:------------------------------------------------------------|:----------------------------------|
| MET-1-primary-few-balanced-accuracy   | few               | balanced_accuracy                                                                 | primary superiority metric                                   | candidate_minus_baseline higher is better                   | TB-2-primary-few-accuracy         |
| MET-2-all-balanced-accuracy-guardrail | all               | balanced_accuracy                                                                 | all-class noninferiority guardrail                           | candidate_minus_best_baseline must not collapse below -0.01 | TB-3-no-all-class-collapse        |
| MET-3-reporting-many-medium           | many;medium       | balanced_accuracy;loss;margin                                                     | required full reporting surface                              | descriptive with confidence intervals                       | TB-4-full-reporting-surface       |
| MET-4-state-distribution-occupancy    | all               | batch_few_fraction;tail_probe_loss;gradient_momentum_cosine;ns_vs_fro_drift_ratio | mechanism-transfer context, not a performance gate by itself | descriptive with missingness gate                           | TB-7-state-distribution-occupancy |

## Primary Comparison Family

| comparison_id                                 | candidate_family     | baseline_family       | metric_id                           | paired_unit   | test                                                                                 | adjustment                                         |
|:----------------------------------------------|:---------------------|:----------------------|:------------------------------------|:--------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------|
| ns_muon_matrix_tuned_vs_adamw_ce_tuned        | ns_muon_matrix_tuned | adamw_ce_tuned        | MET-1-primary-few-balanced-accuracy | seed          | paired t-test on per-seed candidate-minus-baseline few balanced-accuracy differences | Holm step-down across the four primary comparisons |
| ns_muon_matrix_tuned_vs_sgd_momentum_ce_tuned | ns_muon_matrix_tuned | sgd_momentum_ce_tuned | MET-1-primary-few-balanced-accuracy | seed          | paired t-test on per-seed candidate-minus-baseline few balanced-accuracy differences | Holm step-down across the four primary comparisons |
| ns_muon_cb_tuned_vs_adamw_ce_tuned            | ns_muon_cb_tuned     | adamw_ce_tuned        | MET-1-primary-few-balanced-accuracy | seed          | paired t-test on per-seed candidate-minus-baseline few balanced-accuracy differences | Holm step-down across the four primary comparisons |
| ns_muon_cb_tuned_vs_sgd_momentum_ce_tuned     | ns_muon_cb_tuned     | sgd_momentum_ce_tuned | MET-1-primary-few-balanced-accuracy | seed          | paired t-test on per-seed candidate-minus-baseline few balanced-accuracy differences | Holm step-down across the four primary comparisons |

## Multiplicity And Guardrail Plan

| adjustment_id           | scope                                                                      | method                                                                                                      | decision_rule                                                                                                                        | forbidden_shortcut                                                            |
|:------------------------|:---------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| ADJ-1-primary-holm      | four Muon-vs-tuned-baseline few-balanced-accuracy comparisons              | Holm step-down familywise error control at alpha=0.05                                                       | claim primary few-class improvement only if both Muon families beat both tuned AdamW and tuned SGD where applicable after adjustment | claiming the best-looking Muon-vs-baseline pair without the full Holm family  |
| ADJ-2-all-guardrail     | all-class balanced-accuracy candidate-minus-best-tuned-baseline difference | paired CI noninferiority check with margin -0.01                                                            | clean benchmark win requires the lower CI endpoint to be at least -0.01                                                              | describing a few-class gain as clean if all-class balanced accuracy collapses |
| ADJ-3-negative-boundary | all completed final recipes                                                | interpret negative, underpowered, or tradeoff outcomes using the pre-output power and variance-prior audits | publish negative or underpowered final outcomes instead of suppressing them                                                          | using negative tuned outcomes only as private tuning feedback                 |

## Reporting Schema

| table_id                         | required_columns                                                                                          | granularity                                    | claim_role                                                                            |
|:---------------------------------|:----------------------------------------------------------------------------------------------------------|:-----------------------------------------------|:--------------------------------------------------------------------------------------|
| TAB-1-per-seed-final-metrics     | seed;recipe_family;recipe_name;many_bacc;medium_bacc;few_bacc;all_bacc;loss;margin                        | one row per final seed and recipe              | audit trail for paired tests and dropped-seed checks                                  |
| TAB-2-paired-primary-comparisons | comparison_id;seed;candidate_family;baseline_family;few_bacc_diff;all_bacc_diff                           | one row per paired seed and primary comparison | input to Holm-adjusted superiority and all-class guardrail decisions                  |
| TAB-3-final-summary              | recipe_family;frequency_group;mean_bacc;ci95_low;ci95_high;mean_loss;mean_margin                          | one row per recipe family and frequency group  | main benchmark reporting surface if all gates pass                                    |
| TAB-4-occupancy-summary          | recipe_family;seed;occupancy_rows;batch_few_fraction;tail_probe_loss;gradient_momentum_cosine;drift_ratio | one row per final seed and recipe              | state-distribution context for why local Muon evidence transfers or fails to transfer |

## Claim Ladder

| claim_state                       | trigger                                                                                             | allowed_wording                                                                         | blocked_wording                                                       |
|:----------------------------------|:----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------------|
| not_ready                         | any validation family, occupancy trace, or final seed recipe is incomplete                          | registered tuned benchmark analysis plan only                                           | final optimizer-performance result                                    |
| clean_cifar100lt_resnet18_win     | primary Holm family passes and all-class guardrail passes with complete reporting                   | CIFAR-100-LT ResNet18 tuned final-performance improvement under the registered protocol | broad long-tail optimizer superiority                                 |
| few_all_tradeoff                  | few-class primary passes but all-class guardrail fails                                              | few/all tradeoff under the registered protocol                                          | clean benchmark win                                                   |
| negative_or_underpowered_boundary | Muon does not beat tuned baselines, or observed effect is below the pre-output MDE sensitivity grid | negative or underpowered tuned benchmark boundary                                       | Muon has been ruled out broadly or silently dropping the tuned result |
| protocol_violation                | final outputs exist before all selection and analysis gates are ready                               | protocol violation; rows excluded from tuned-performance claims                         | using leaked final rows for selection or claim repair                 |

## Boundary

Allowed now: cite this as a pre-final statistical analysis plan for the tuned benchmark.

Blocked now: running final analysis, changing the primary comparison family, or claiming tuned optimizer performance before validation selection, occupancy logging, and final seed rows are complete.

Artifacts:
- [analysis_input_contract.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/analysis_input_contract.csv)
- [metric_contract.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/metric_contract.csv)
- [primary_comparison_family.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/primary_comparison_family.csv)
- [multiplicity_and_guardrail_plan.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/multiplicity_and_guardrail_plan.csv)
- [reporting_schema.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/reporting_schema.csv)
- [claim_ladder.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/claim_ladder.csv)
- [gate_matrix.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/gate_matrix.csv)
- [config.json](../results/e11_cifar100_resnet_lt_tuned_benchmark/final_analysis_plan/config.json)
