# E11 Natural Negative Search Phase1 Interim Synthesis

This generated artifact summarizes the Holm-family state for the
pre-registered natural negative-search phase1 family. It does not add a new
search after looking at outcomes. The current evaluator has
26/26 observed primary rows and 0
missing declared rows. The registered family is complete. The evaluator reports no adjusted primary full-drift counterexample, so the allowed wording is a finite-null candidate for the registered phase1 primary family, with tail-quality and detectable-effect caveats.

Observed rows have raw_worse_rows=0 and
max raw CI high 0.8967. This is stronger than the previous partial-family readout, but it is still not a universal natural-task theorem.

## Family Coverage

| search_id                                  |   expected_settings |   observed_primary_rows |   missing_primary_rows |   coverage_fraction | output_status   | claim_use                |
|:-------------------------------------------|--------------------:|------------------------:|-----------------------:|--------------------:|:----------------|:-------------------------|
| NNS-P1-cifar100lt-resnet18-new-partitions  |                  12 |                      12 |                      0 |                   1 | complete        | complete_family_evidence |
| NNS-P1-cifar10lt-resnet18-cross-partitions |                   8 |                       8 |                      0 |                   1 | complete        | complete_family_evidence |
| NNS-P1-tail-quality-controls               |                   6 |                       6 |                      0 |                   1 | complete        | complete_family_evidence |

## Observed Primary Summary

| scope                                      |   observed_primary_rows |   missing_primary_rows |   raw_worse_rows |   quality_gate_pass_rows |   quality_gate_fail_rows |   estimate_min |   estimate_max |   raw_ci95_high_max | claim_statuses                        | interpretation                                |
|:-------------------------------------------|------------------------:|-----------------------:|-----------------:|-------------------------:|-------------------------:|---------------:|---------------:|--------------------:|:--------------------------------------|:----------------------------------------------|
| all_observed                               |                      26 |                      0 |                0 |                        3 |                       23 |         0.3464 |         0.8378 |              0.8967 | no_primary_counterexample_for_setting | complete_family_ready_for_registered_decision |
| NNS-P1-cifar100lt-resnet18-new-partitions  |                      12 |                      0 |                0 |                        2 |                       10 |         0.4682 |         0.8378 |              0.8967 | no_primary_counterexample_for_setting | complete_family_ready_for_registered_decision |
| NNS-P1-cifar10lt-resnet18-cross-partitions |                       8 |                      0 |                0 |                        0 |                        8 |         0.3464 |         0.7143 |              0.7761 | no_primary_counterexample_for_setting | complete_family_ready_for_registered_decision |
| NNS-P1-tail-quality-controls               |                       6 |                      0 |                0 |                        1 |                        5 |         0.5215 |         0.8139 |              0.8671 | no_primary_counterexample_for_setting | complete_family_ready_for_registered_decision |

## Claim Boundary

| claim_id                             | current_status                     | evidence                                                                                       | allowed_wording                                                                                                            | blocked_wording                                                      |
|:-------------------------------------|:-----------------------------------|:-----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|
| NNI-1-primary-natural-counterexample | no_adjusted_primary_counterexample | 26/26 observed; raw_worse_rows=0; adjusted_worse_rows=0; NNS-E2=pass                           | complete registered phase1 family found no adjusted primary full-drift counterexample                                      | fresh natural primary counterexample                                 |
| NNI-2-finite-null-search             | finite_null_candidate              | 0 declared settings missing; adjusted_worse_rows=0; NNS-E4=finite_null_candidate               | finite-null candidate for the registered 26-setting phase1 primary family, with tail-quality and detectable-effect caveats | unqualified finite null over all natural settings                    |
| NNI-3-quality-gated-interpretation   | quality_caveated_complete_family   | NNS-E3=pass; NNS-E4=finite_null_candidate; quality_gate_pass_rows=3; quality_gate_fail_rows=23 | quality and multiplicity gates remain the claim boundary; low-tail-quality rows limit interpretation                       | using low-tail-quality rows as mechanism validation or falsification |

## Remaining Work

| search_id   | missing_primary_rows   | current_status   | required_action   | claim_unblocked_if_done   |
|-------------|------------------------|------------------|-------------------|---------------------------|

Artifacts:
- [family_coverage.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/family_coverage.csv)
- [observed_primary_summary.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/observed_primary_summary.csv)
- [claim_boundary.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/claim_boundary.csv)
- [remaining_work.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/remaining_work.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/config.json)
