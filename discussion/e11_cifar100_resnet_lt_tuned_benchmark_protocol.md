# E11 CIFAR-100-LT Tuned Benchmark Protocol

This generated protocol is not a new final benchmark result. It turns the existing CIFAR-100-LT ResNet18 standard baseline, augmented recipe pilot, and negative NS-Muon pilot into a clean preregistered route for a tuned final-performance claim.

## Pilot Context Quarantine

The existing pilots are useful for risk assessment, but they cannot select final hyperparameters or justify a practical optimizer claim. Their allowed role is fixed here before the tuned final seeds are run.

| context_id                         | frequency_group   | metric                                        |   estimate |   ci95_low |   ci95_high | allowed_use                                                         |
|:-----------------------------------|:------------------|:----------------------------------------------|-----------:|-----------:|------------:|:--------------------------------------------------------------------|
| standard_adamw_all                 | all               | balanced_accuracy                             |    0.1684  |   0.1572   |    0.1796   | context only; not a tuned benchmark comparator                      |
| standard_adamw_few                 | few               | balanced_accuracy                             |    0.0129  |   0.009351 |    0.01645  | context only; not a tuned benchmark comparator                      |
| recipe_pilot_sgd_aug_ce_all        | all               | paired_balanced_accuracy_diff_vs_adamw_aug_ce |    0.05066 |   0.04418  |    0.05714  | pilot context only; cannot choose final hyperparameters             |
| recipe_pilot_sgd_aug_ce_few        | few               | paired_balanced_accuracy_diff_vs_adamw_aug_ce |    0.0194  |   0.005581 |    0.03322  | pilot context only; cannot choose final hyperparameters             |
| recipe_pilot_adamw_aug_cb_loss_few | few               | paired_balanced_accuracy_diff_vs_adamw_aug_ce |   -0.0114  |  -0.0215   |   -0.001304 | pilot context only; cannot choose final hyperparameters             |
| muon_pilot_ns_muon_aug_lr3e-5_all  | all               | paired_balanced_accuracy_diff_vs_adamw_aug_ce |   -0.2845  |  -0.2883   |   -0.2807   | negative pilot boundary only; does not rule out the tuned Muon grid |
| muon_pilot_ns_muon_aug_lr3e-5_few  | few               | paired_balanced_accuracy_diff_vs_adamw_aug_ce |   -0.08856 |  -0.1014   |   -0.07566  | negative pilot boundary only; does not rule out the tuned Muon grid |
| muon_pilot_ns_muon_aug_lr1e-4_all  | all               | paired_balanced_accuracy_diff_vs_adamw_aug_ce |   -0.2348  |  -0.2395   |   -0.23     | negative pilot boundary only; does not rule out the tuned Muon grid |
| muon_pilot_ns_muon_aug_lr1e-4_few  | few               | paired_balanced_accuracy_diff_vs_adamw_aug_ce |   -0.08767 |  -0.09948  |   -0.07585  | negative pilot boundary only; does not rule out the tuned Muon grid |

## Benchmark Scope

| scope_id                          | dataset                     | imbalance_factor   | architecture           | primary_metric        | secondary_metrics                               | claim_scope                                         | status                       |
|:----------------------------------|:----------------------------|:-------------------|:-----------------------|:----------------------|:------------------------------------------------|:----------------------------------------------------|:-----------------------------|
| primary_cifar100lt_resnet18_if100 | CIFAR-100-LT                | 100                | ResNet18               | few balanced accuracy | all/many/medium balanced accuracy; loss; margin | CIFAR-100-LT ResNet18 final-performance claim only  | protocol_registered          |
| broad_long_tail_optimizer_claim   | multiple long-tail datasets | dataset-specific   | multiple architectures | not covered           | not covered                                     | broad optimizer-performance claim remains forbidden | not_covered_by_this_protocol |

## Seed And Split Contract

| split_id                 | seed_set   | role                       | tuning_allowed              | outputs                                                                  |
|:-------------------------|:-----------|:---------------------------|:----------------------------|:-------------------------------------------------------------------------|
| spent_pilot_context      | 0..9       | pilot_context              | no for final claims         | standard, recipe, and NS-Muon pilot result directories                   |
| validation_tuning        | 10..14     | hyperparameter_selection   | yes before final unblinding | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_*              |
| final_claim              | 20..29     | primary_final_evaluation   | no                          | results/e11_cifar100_resnet_lt_tuned_benchmark/final_cifar100lt_resnet18 |
| optional_stability_rerun | 30..34     | post_claim_stability_check | no                          | optional appendix only                                                   |

## Recipe Grid

| recipe_family          | optimizer   | lr_grid             | weight_decay_grid   | class_reweighting                                      | sampler                   | warmup_steps_grid   | newton_schulz_steps_grid   | claim_role                                           |
|:-----------------------|:------------|:--------------------|:--------------------|:-------------------------------------------------------|:--------------------------|:--------------------|:---------------------------|:-----------------------------------------------------|
| adamw_ce_tuned         | adamw       | 1e-4;3e-4;1e-3      | 1e-4;5e-4           | none                                                   | long-tail natural sampler | 0;500               | not_applicable             | tuned AdamW baseline                                 |
| sgd_momentum_ce_tuned  | sgd         | 0.03;0.1;0.3        | 5e-4;1e-3           | none                                                   | long-tail natural sampler | 0;500               | not_applicable             | tuned SGD baseline                                   |
| adamw_cb_loss_tuned    | adamw       | 1e-4;3e-4;1e-3      | 1e-4;5e-4           | effective-number class-balanced loss beta=0.999;0.9999 | long-tail natural sampler | 0;500               | not_applicable             | class-balanced AdamW baseline                        |
| adamw_cb_sampler_tuned | adamw       | 1e-4;3e-4           | 1e-4;5e-4           | cross entropy                                          | class-balanced sampler    | 0;500               | not_applicable             | sampler baseline required before any practical claim |
| ns_muon_matrix_tuned   | ns_muon     | 1e-5;3e-5;1e-4;3e-4 | 1e-4;5e-4           | none                                                   | long-tail natural sampler | 0;500;1000          | 3;5;7                      | candidate practical Muon recipe                      |
| ns_muon_cb_tuned       | ns_muon     | 1e-5;3e-5;1e-4      | 1e-4;5e-4           | effective-number class-balanced loss beta=0.9999       | long-tail natural sampler | 500;1000            | 3;5;7                      | candidate practical Muon class-balanced recipe       |

## Selection Rules

| rule_id                         | rule                                                                                                   | forbidden_action                                                                     |
|:--------------------------------|:-------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
| SEL-1-validation-only-selection | Choose one recipe per family using validation_tuning seeds only.                                       | Selecting hyperparameters from final_claim seeds or from the published pilot deltas. |
| SEL-2-primary-metric            | Primary validation objective is few balanced accuracy; ties break by all balanced accuracy.            | Switching the primary metric after seeing final results.                             |
| SEL-3-paired-final              | Run final_claim seeds pairwise for every selected recipe and tuned baseline.                           | Reporting unpaired recipe seeds or dropping failed seeds.                            |
| SEL-4-familywise-error          | Treat final comparisons to tuned AdamW and tuned SGD as one family and report Holm-adjusted decisions. | Claiming the best-looking pair without multiplicity adjustment.                      |
| SEL-5-negative-results          | Publish the tuned result as a boundary if Muon fails tuned baselines.                                  | Suppressing negative tuned Muon outcomes while keeping local-drift motivation.       |

## Executable Validation Registry

The validation grid is materialized by `scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py --settings-only`, which writes `results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv` and `execution_status.csv`. GPU validation cells are submitted with `scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch`; each Slurm array cell runs one registered validation setting on seeds `10..14`. The final claim split `20..29` remains untouched until validation selects recipes.

The frozen selection rule is materialized by `scripts/e11_write_cifar100_resnet_lt_tuned_benchmark_selection.py`. It writes `results/e11_cifar100_resnet_lt_tuned_benchmark/validation_selection/*` and `discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md`, selecting one recipe per family only after every registered validation setting in that family has a summary.

## Acceptance Gates

| gate_id                     | claim_unblocked                                                             | pass_rule                                                                                                                                          | failure_claim                             |
|:----------------------------|:----------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------|
| TB-1-complete-final-family  | Any CIFAR-100-LT ResNet18 final-performance statement.                      | All selected recipes and tuned baselines have 10 paired final seeds with many/medium/few/all metrics.                                              | not_ready                                 |
| TB-2-primary-few-accuracy   | Practical Muon improves few-class long-tail performance.                    | Muon few balanced-accuracy paired CI lower endpoint exceeds both tuned AdamW and tuned SGD by more than zero with Holm-adjusted final comparisons. | no practical Muon performance advantage   |
| TB-3-no-all-class-collapse  | Few-class gain is not bought by broad collapse.                             | Muon all-class balanced-accuracy paired CI lower endpoint is no worse than -0.01 against the best tuned baseline.                                  | tradeoff only; no clean performance claim |
| TB-4-full-reporting-surface | Benchmark table can enter the main paper.                                   | Report many, medium, few, all balanced accuracy, loss, margin, per-seed rows, and paired differences.                                              | appendix-only pilot context               |
| TB-5-scope-control          | CIFAR-100-LT ResNet18 benchmark claim.                                      | Wording names CIFAR-100-LT ResNet18 only unless additional datasets and architectures are run under a separate preregistered protocol.             | broad optimizer claim forbidden           |
| TB-6-local-drift-separation | Mechanism and final-performance evidence can coexist without contradiction. | Paper separates matched-head-gain local drift from long-horizon final training, including negative tuned outcomes.                                 | mechanism-only paper path remains         |

## Current Claim Status

The current state remains `not_ready` for any tuned final-performance claim. A top-tier mechanism paper can still use the existing pilots as context, but any CIFAR-100-LT ResNet18 optimizer-performance statement must wait for the validation/final split contract above. A broad long-tail optimizer claim remains forbidden by this protocol.
