# E11 CIFAR-100-LT Tuned Benchmark Variance Prior Audit

This generated audit calibrates the tuned benchmark power assumptions against pre-final evidence. It reads completed validation summaries and spent pilot paired comparisons only; it does not inspect or authorize final seed outputs.

Current anchor: few-class spent-pilot paired-diff SD p50/p80/p95 = 0.00666/0.00933/0.01068. The existing power audit uses paired-diff SD `0.03` as a reference. This audit keeps that value as a sensitivity point and adds empirical p50/p80/p95/max references from validation-only within-setting seed variability and spent-pilot paired-diff variability.

## Gate Matrix

| gate_id                              | status   | evidence                                                                           | claim_effect                                                                      |
|:-------------------------------------|:---------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
| EVPA-1-final-seed-quarantine         | pass     | no files under results/e11_cifar100_resnet_lt_tuned_benchmark/final_claim          | variance prior remains pre-final and cannot inspect final seeds                   |
| EVPA-2-validation-sd-surface         | pass     | 216 validation frequency-group variance rows                                       | completed validation settings provide within-setting seed variability             |
| EVPA-3-spent-pilot-paired-sd-surface | pass     | 16 spent-pilot paired-diff variance rows                                           | paired-diff SD prior is anchored to pre-final spent pilot context                 |
| EVPA-4-primary-few-sd-anchor         | pass     | few-class spent-pilot paired-diff SD p80=0.00933                                   | primary few-class MDE can be compared with empirical pre-final variability        |
| EVPA-5-mde-sensitivity-grid          | pass     | MDE grid includes empirical p50/p80/p95/max and assumed_0p03 references            | reviewers can see how final detectability changes under empirical variance priors |
| EVPA-6-claim-boundary                | pass     | audit reads validation-only and spent-pilot summaries, not final benchmark outputs | no benchmark-performance claim is authorized by this audit                        |

## Variance Prior Summary

| source_id                 | frequency_group   |   row_count |   sd_p50 |   sd_p80 |   sd_p95 |   sd_max |   assumed_paired_diff_sd | p80_within_assumed_0p03   | interpretation                                                                  |
|:--------------------------|:------------------|------------:|---------:|---------:|---------:|---------:|-------------------------:|:--------------------------|:--------------------------------------------------------------------------------|
| validation_within_setting | all               |          54 | 0.00624  | 0.009047 | 0.0122   | 0.1826   |                     0.03 | yes                       | within-setting seed SD is a loose variance prior                                |
| validation_within_setting | few               |          54 | 0.006536 | 0.00813  | 0.0124   | 0.03988  |                     0.03 | yes                       | within-setting seed SD is a loose variance prior                                |
| validation_within_setting | many              |          54 | 0.009561 | 0.01231  | 0.01514  | 0.315    |                     0.03 | yes                       | within-setting seed SD is a loose variance prior                                |
| validation_within_setting | medium            |          54 | 0.009124 | 0.01271  | 0.01469  | 0.1733   |                     0.03 | yes                       | within-setting seed SD is a loose variance prior                                |
| spent_pilot_paired_diff   | all               |           4 | 0.003318 | 0.00492  | 0.005144 | 0.005219 |                     0.03 | yes                       | spent pilot paired-diff SD is the closest available pre-final paired-diff prior |
| spent_pilot_paired_diff   | few               |           4 | 0.00666  | 0.00933  | 0.01068  | 0.01113  |                     0.03 | yes                       | spent pilot paired-diff SD is the closest available pre-final paired-diff prior |
| spent_pilot_paired_diff   | many              |           4 | 0.003806 | 0.007168 | 0.01005  | 0.01101  |                     0.03 | yes                       | spent pilot paired-diff SD is the closest available pre-final paired-diff prior |
| spent_pilot_paired_diff   | medium            |           4 | 0.004417 | 0.007421 | 0.008774 | 0.009224 |                     0.03 | yes                       | spent pilot paired-diff SD is the closest available pre-final paired-diff prior |

## MDE Sensitivity Snapshot

| source_id                 | frequency_group   | sd_reference   |   paired_diff_sd |   holm_worst_case_mde |   minimum_mean_all_diff_to_pass_guardrail | claim_effect                                                                                                    |
|:--------------------------|:------------------|:---------------|-----------------:|----------------------:|------------------------------------------:|:----------------------------------------------------------------------------------------------------------------|
| validation_within_setting | all               | p80            |         0.009047 |              0.0089   |                                -0.0011    | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| validation_within_setting | all               | p95            |         0.0122   |              0.01201  |                                 0.002007  | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| validation_within_setting | all               | assumed_0p03   |         0.03     |              0.02951  |                                 0.01951   | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| validation_within_setting | few               | p80            |         0.00813  |              0.007998 |                                -0.002002  | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| validation_within_setting | few               | p95            |         0.0124   |              0.0122   |                                 0.002199  | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| validation_within_setting | few               | assumed_0p03   |         0.03     |              0.02951  |                                 0.01951   | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| spent_pilot_paired_diff   | all               | p80            |         0.00492  |              0.00484  |                                -0.00516   | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| spent_pilot_paired_diff   | all               | p95            |         0.005144 |              0.005061 |                                -0.004939  | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| spent_pilot_paired_diff   | all               | assumed_0p03   |         0.03     |              0.02951  |                                 0.01951   | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| spent_pilot_paired_diff   | few               | p80            |         0.00933  |              0.009179 |                                -0.000821  | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| spent_pilot_paired_diff   | few               | p95            |         0.01068  |              0.01051  |                                 0.0005065 | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |
| spent_pilot_paired_diff   | few               | assumed_0p03   |         0.03     |              0.02951  |                                 0.01951   | MDE sensitivity only; final benchmark claim remains blocked until validation selection and final seeds complete |

## Boundary

Allowed now: use this as a variance-prior and detectable-effect sensitivity audit for the registered tuned benchmark protocol.

Blocked now: using validation variability, spent pilot variability, or the assumed SD grid to choose recipes, inspect final seeds, or claim tuned optimizer performance.

Artifacts:
- [validation_setting_variance.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/validation_setting_variance.csv)
- [spent_pilot_paired_variance.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/spent_pilot_paired_variance.csv)
- [variance_prior_summary.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/variance_prior_summary.csv)
- [mde_sensitivity_from_empirical_sd.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/mde_sensitivity_from_empirical_sd.csv)
- [gate_matrix.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/gate_matrix.csv)
- [config.json](../results/e11_cifar100_resnet_lt_tuned_benchmark/variance_prior_audit/config.json)
