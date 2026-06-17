# E11 Natural Negative Search Phase1 Interim Synthesis

This generated artifact summarizes the partial Holm-family state for the
pre-registered natural negative-search phase1 family. It is not a discovery
claim. The current evaluator has 20/26 observed
primary rows and 6 missing declared rows. Natural claims remain
blocked because the tail-quality controls are not complete and the evaluator
gates remain `not_ready`.

Observed rows have raw_worse_rows=0 and
max raw CI high 0.8967. This is useful
interim evidence, but no adjusted primary decision is claimable until the full
26-setting family is complete.

## Family Coverage

| search_id                                  |   expected_settings |   observed_primary_rows |   missing_primary_rows |   coverage_fraction | output_status            | claim_use                    |
|:-------------------------------------------|--------------------:|------------------------:|-----------------------:|--------------------:|:-------------------------|:-----------------------------|
| NNS-P1-cifar100lt-resnet18-new-partitions  |                  12 |                      12 |                      0 |                   1 | complete                 | interim_evidence_only        |
| NNS-P1-cifar10lt-resnet18-cross-partitions |                   8 |                       8 |                      0 |                   1 | complete                 | interim_evidence_only        |
| NNS-P1-tail-quality-controls               |                   6 |                       0 |                      6 |                   0 | settings_only_no_metrics | missing_required_family_rows |

## Observed Primary Summary

| scope                                      |   observed_primary_rows |   missing_primary_rows |   raw_worse_rows |   quality_gate_pass_rows |   quality_gate_fail_rows | estimate_min   | estimate_max   | raw_ci95_high_max   | claim_statuses   | interpretation                                |
|:-------------------------------------------|------------------------:|-----------------------:|-----------------:|-------------------------:|-------------------------:|:---------------|:---------------|:--------------------|:-----------------|:----------------------------------------------|
| all_observed                               |                      20 |                      6 |                0 |                        2 |                       18 | 0.3464         | 0.8378         | 0.8967              | not_ready        | no_claim_until_full_family_complete           |
| NNS-P1-cifar100lt-resnet18-new-partitions  |                      12 |                      0 |                0 |                        2 |                       10 | 0.4682         | 0.8378         | 0.8967              | not_ready        | complete_family_ready_for_registered_decision |
| NNS-P1-cifar10lt-resnet18-cross-partitions |                       8 |                      0 |                0 |                        0 |                        8 | 0.3464         | 0.7143         | 0.7761              | not_ready        | complete_family_ready_for_registered_decision |
| NNS-P1-tail-quality-controls               |                       0 |                      6 |                0 |                        0 |                        0 | n/a            | n/a            | n/a                 | not_ready        | no_claim_until_full_family_complete           |

## Claim Boundary

| claim_id                             | current_status                              | evidence                                                                  | allowed_wording                                                | blocked_wording                                                      |
|:-------------------------------------|:--------------------------------------------|:--------------------------------------------------------------------------|:---------------------------------------------------------------|:---------------------------------------------------------------------|
| NNI-1-primary-natural-counterexample | blocked_partial_family                      | 20/26 observed; raw_worse_rows=0; adjusted_worse_rows=0; NNS-E2=not_ready | partial phase1 readout only; no natural primary counterexample | fresh natural primary counterexample                                 |
| NNI-2-finite-null-search             | blocked_partial_family                      | 6 declared settings are still missing, including tail-quality controls    | 20/26 observed rows are reported under the frozen family       | finite null over the 26-setting phase1 family                        |
| NNI-3-quality-gated-interpretation   | blocked_until_quality_and_completeness_pass | NNS-E3=not_ready; NNS-E4=not_ready                                        | quality and multiplicity gates remain the claim boundary       | using low-tail-quality or incomplete rows as mechanism falsification |

## Remaining Work

| search_id                    |   missing_primary_rows | current_status           | required_action                                                                | claim_unblocked_if_done                                                          |
|:-----------------------------|-----------------------:|:-------------------------|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
| NNS-P1-tail-quality-controls |                      6 | settings_only_no_metrics | wait for Slurm output, then rerun make e11-natural-negative-search-phase1-eval | fresh primary candidate or finite-null boundary, depending on adjusted decisions |

Artifacts:
- [family_coverage.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/family_coverage.csv)
- [observed_primary_summary.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/observed_primary_summary.csv)
- [claim_boundary.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/claim_boundary.csv)
- [remaining_work.csv](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/remaining_work.csv)
- [config.json](../results/e11_natural_negative_search_protocol/phase1_interim_synthesis/config.json)
