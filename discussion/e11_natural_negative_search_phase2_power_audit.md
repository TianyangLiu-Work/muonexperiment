# E11 Natural Negative Search Phase2 Power Audit

This generated audit records the detectable-effect and interpretation boundary
for the registered 8-setting ResNet34 held-out architecture phase2 family. It is
a pre-output design audit: it is written before phase2 metric rows exist and
must not be tuned after inspecting phase2 outcomes.

The primary test is the paired across-seed log ratio for
`tail_output_drift_sq_ratio_spectral_over_fro`, with a one-sided worse-than-one
alternative and Holm-adjusted phase-family decision. With only 3 seeds per
setting, the phase2 family is mainly a held-out architecture stress test for
large effects; null results below the audited MDE are explicitly underpowered.
The registered seed count is 3 seeds per setting.

## Adjusted Minimum Detectable Ratio

|   log_ratio_sd |   target_power |   minimum_detectable_ratio |   minimum_detectable_log_ratio | alpha_scope                |
|---------------:|---------------:|---------------------------:|-------------------------------:|:---------------------------|
|           0.1  |            0.8 |                      1.924 |                         0.6546 | holm_bonferroni_worst_case |
|           0.1  |            0.9 |                      2.19  |                         0.7839 | holm_bonferroni_worst_case |
|           0.2  |            0.8 |                      3.703 |                         1.309  | holm_bonferroni_worst_case |
|           0.2  |            0.9 |                      4.796 |                         1.568  | holm_bonferroni_worst_case |
|           0.35 |            0.8 |                      9.886 |                         2.291  | holm_bonferroni_worst_case |
|           0.35 |            0.9 |                     15.54  |                         2.744  | holm_bonferroni_worst_case |
|           0.5  |            0.8 |                     26.39  |                         3.273  | holm_bonferroni_worst_case |
|           0.5  |            0.9 |                     50.38  |                         3.92   | holm_bonferroni_worst_case |

## Adjusted Power Grid

|   log_ratio_sd |   true_tail_drift_ratio |   power | alpha_scope                |
|---------------:|------------------------:|--------:|:---------------------------|
|           0.1  |                    1.25 | 0.1797  | holm_bonferroni_worst_case |
|           0.1  |                    1.5  | 0.4649  | holm_bonferroni_worst_case |
|           0.1  |                    2    | 0.8352  | holm_bonferroni_worst_case |
|           0.1  |                    3    | 0.989   | holm_bonferroni_worst_case |
|           0.2  |                    1.25 | 0.05717 | holm_bonferroni_worst_case |
|           0.2  |                    1.5  | 0.1527  | holm_bonferroni_worst_case |
|           0.2  |                    2    | 0.3688  | holm_bonferroni_worst_case |
|           0.2  |                    3    | 0.6792  | holm_bonferroni_worst_case |
|           0.35 |                    1.25 | 0.02659 | holm_bonferroni_worst_case |
|           0.35 |                    1.5  | 0.0606  | holm_bonferroni_worst_case |
|           0.35 |                    2    | 0.1468  | holm_bonferroni_worst_case |
|           0.35 |                    3    | 0.316   | holm_bonferroni_worst_case |

## Interpretation Ladder

| case_id                              | evidence_condition                                                                                | allowed_wording                                                                                 | blocked_wording                                                                                  |
|:-------------------------------------|:--------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| P2-PWR-1-adjusted-positive           | complete 8-setting ResNet34 family; Holm-adjusted primary p <= 0.05; quality gates pass           | held-out architecture primary boundary candidate                                                | broad natural counterexample claim or final optimizer-performance claim                          |
| P2-PWR-2-complete-null-above-mde     | complete 8-setting family; no adjusted primary row; observed SD makes target effect above the MDE | finite ResNet34 held-out architecture null for effects at or above the audited detectable scale | absence of smaller ResNet34 effects or absence of natural counterexamples in other architectures |
| P2-PWR-3-complete-null-below-mde     | complete 8-setting family; no adjusted primary row; target effect below MDE                       | underpowered phase2 null for small held-out architecture effects                                | convincing held-out architecture finite null                                                     |
| P2-PWR-4-phase1-null-phase2-positive | phase1 finite-null candidate but phase2 has an adjusted primary worse row                         | architecture-transport boundary case requiring mechanism analysis                               | discarding phase2 as outlier or claiming phase1 generality                                       |
| P2-PWR-5-incomplete-family           | any declared phase2 metric output missing                                                         | registered held-out architecture search in progress                                             | phase2 counterexample, phase2 finite null, or selected-subset claim                              |

## Outcome State Machine

| state_id                      | trigger                                                                       | claim_state                                                | required_action                                                          |
|:------------------------------|:------------------------------------------------------------------------------|:-----------------------------------------------------------|:-------------------------------------------------------------------------|
| P2-S1-not-run                 | 0/8 phase2 primary metric rows                                                | not_ready                                                  | wait for Slurm outputs; do not inspect partial settings for claims       |
| P2-S2-partial                 | 1-7/8 phase2 primary metric rows                                              | not_ready_partial_family                                   | report partial rows only as progress; keep claim gate closed             |
| P2-S3-adjusted-positive       | 8/8 rows; at least one Holm-adjusted primary worse row passes quality gates   | heldout_architecture_boundary_candidate                    | run mechanism analysis and forbid broad optimizer-performance wording    |
| P2-S4-complete-null-above-mde | 8/8 rows; no adjusted positive; target effect is above phase2 MDE             | finite_phase2_null_candidate_with_detectable_effect_caveat | state effect-size floor and keep claim within ResNet34 registered family |
| P2-S5-complete-null-below-mde | 8/8 rows; no adjusted positive; target effect is below phase2 MDE             | underpowered_phase2_null                                   | do not use as strong held-out null; add seeds or larger search family    |
| P2-S6-quality-failure         | adjusted positive exists only in rows failing head-gain or tail-quality gates | quality_caveated_boundary                                  | treat as diagnostic failure mode, not as primary natural counterexample  |

Generated tables:

- [power_grid.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/power_grid.csv)
- [minimum_detectable_effect.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/minimum_detectable_effect.csv)
- [interpretation_ladder.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/interpretation_ladder.csv)
- [outcome_state_machine.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/outcome_state_machine.csv)
