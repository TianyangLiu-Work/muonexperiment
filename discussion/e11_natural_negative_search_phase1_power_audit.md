# E11 Natural Negative Search Phase1 Power Audit

This generated audit records the detectable-effect boundary for the registered
26-setting natural negative-search phase1 family. It is a design audit, not an
outcome analysis: it should be interpreted before inspecting fresh metric rows.

The primary test is the paired across-seed log ratio for
`tail_output_drift_sq_ratio_spectral_over_fro`, with a one-sided worse-than-one
alternative and Holm-adjusted phase-family decision. The conservative
first-rejection threshold is approximated as Bonferroni `0.05 / 26`.

## Adjusted Minimum Detectable Ratio

|   log_ratio_sd |   target_power |   minimum_detectable_ratio |   minimum_detectable_log_ratio | alpha_scope                |
|---------------:|---------------:|---------------------------:|-------------------------------:|:---------------------------|
|           0.1  |            0.8 |                      1.402 |                         0.3378 | holm_bonferroni_worst_case |
|           0.1  |            0.9 |                      1.473 |                         0.3874 | holm_bonferroni_worst_case |
|           0.2  |            0.8 |                      1.965 |                         0.6756 | holm_bonferroni_worst_case |
|           0.2  |            0.9 |                      2.17  |                         0.7748 | holm_bonferroni_worst_case |
|           0.35 |            0.8 |                      3.262 |                         1.182  | holm_bonferroni_worst_case |
|           0.35 |            0.9 |                      3.88  |                         1.356  | holm_bonferroni_worst_case |
|           0.5  |            0.8 |                      5.415 |                         1.689  | holm_bonferroni_worst_case |
|           0.5  |            0.9 |                      6.938 |                         1.937  | holm_bonferroni_worst_case |

## Adjusted Power Grid

|   log_ratio_sd |   true_tail_drift_ratio |   power | alpha_scope                |
|---------------:|------------------------:|--------:|:---------------------------|
|           0.1  |                    1.25 | 0.4048  | holm_bonferroni_worst_case |
|           0.1  |                    1.5  | 0.9249  | holm_bonferroni_worst_case |
|           0.1  |                    2    | 0.9999  | holm_bonferroni_worst_case |
|           0.1  |                    3    | 1       | holm_bonferroni_worst_case |
|           0.2  |                    1.25 | 0.07502 | holm_bonferroni_worst_case |
|           0.2  |                    1.5  | 0.3275  | holm_bonferroni_worst_case |
|           0.2  |                    2    | 0.8213  | holm_bonferroni_worst_case |
|           0.2  |                    3    | 0.9959  | holm_bonferroni_worst_case |
|           0.35 |                    1.25 | 0.02151 | holm_bonferroni_worst_case |
|           0.35 |                    1.5  | 0.08233 | holm_bonferroni_worst_case |
|           0.35 |                    2    | 0.3104  | holm_bonferroni_worst_case |
|           0.35 |                    3    | 0.7337  | holm_bonferroni_worst_case |

## Interpretation Ladder

| case_id                       | evidence_condition                                                                                 | allowed_wording                                                             | blocked_wording                                                             |
|:------------------------------|:---------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|:----------------------------------------------------------------------------|
| PWR-1-adjusted-positive       | complete 26-setting family; Holm-adjusted primary p <= 0.05; quality gates pass                    | fresh natural primary boundary candidate                                    | broad optimizer-performance claim or universal natural counterexample claim |
| PWR-2-complete-null-above-mde | complete 26-setting family; no adjusted primary row; observed SD makes target effect above the MDE | finite registered null for effects at or above the audited detectable scale | absence of smaller natural counterexamples                                  |
| PWR-3-complete-null-below-mde | complete 26-setting family; no adjusted primary row; target effect below MDE                       | underpowered for small natural negative effects                             | convincing null or mechanism generality claim                               |
| PWR-4-incomplete-family       | any declared phase1 metric output missing                                                          | running registered search; no phase1 discovery decision yet                 | primary counterexample, finite null, or selected-subset claim               |

Generated tables:

- [power_grid.csv](../results/e11_natural_negative_search_protocol/phase1_power_audit/power_grid.csv)
- [minimum_detectable_effect.csv](../results/e11_natural_negative_search_protocol/phase1_power_audit/minimum_detectable_effect.csv)
- [interpretation_ladder.csv](../results/e11_natural_negative_search_protocol/phase1_power_audit/interpretation_ladder.csv)
