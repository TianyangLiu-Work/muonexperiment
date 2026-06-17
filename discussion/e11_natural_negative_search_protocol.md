# E11 Natural Negative Search Protocol

This generated protocol upgrades the natural-boundary work from a committed
audit to a pre-registered fresh search. It does not claim a new natural counterexample.
Its purpose is to prevent post-hoc selection when searching for natural settings
where the matched-head-gain spectral/polar direction is worse than Frobenius on
the primary full tail-output drift metric.

## Audit Baseline

| baseline_id                               |   row_count |   source_count |   strict_worse_count |   max_ci95_high | status                                                     | claim_use                                                                                 |
|:------------------------------------------|------------:|---------------:|---------------------:|----------------:|:-----------------------------------------------------------|:------------------------------------------------------------------------------------------|
| committed_natural_primary_full_drift_scan |          37 |             11 |                    0 |          0.936  | no_strict_natural_primary_counterexample_in_committed_scan | baseline null scan only; not a fresh negative-search result                               |
| committed_component_ratio_boundaries      |         132 |             10 |                   17 |         14.45   | component_boundary_candidates_found                        | claim-boundary evidence for true-logit/margin components, not primary full-drift failures |
| committed_secondary_outcome_tradeoffs     |         111 |             11 |                   41 |          0.0138 | secondary_outcome_tradeoffs_found                          | claim-boundary evidence for loss, margin, and accuracy outcomes                           |

## Fresh Search Space

| search_id                                  | phase                          | dataset      | architecture                                             | checkpoint_steps   | target_head_gain_fraction   |   max_settings | compute_mode   | entrypoint                                              | entrypoint_status                    |
|:-------------------------------------------|:-------------------------------|:-------------|:---------------------------------------------------------|:-------------------|:----------------------------|---------------:|:---------------|:--------------------------------------------------------|:-------------------------------------|
| NNS-P1-cifar100lt-resnet18-new-partitions  | phase1_fresh_primary_search    | CIFAR-100-LT | ResNet18 CIFAR stem                                      | 500/2000/5000      | 0.002/0.005                 |             12 | GPU via Slurm  | scripts/slurm/e11_natural_negative_search_phase1.sbatch | implemented_sbatch                   |
| NNS-P1-cifar10lt-resnet18-cross-partitions | phase1_fresh_primary_search    | CIFAR-10-LT  | ResNet18 CIFAR stem                                      | 500/5000           | 0.002/0.005                 |              8 | GPU via Slurm  | scripts/slurm/e11_natural_negative_search_phase1.sbatch | implemented_sbatch                   |
| NNS-P1-tail-quality-controls               | phase1_fresh_primary_search    | CIFAR-100    | ResNet18 CIFAR stem                                      | 2000/5000/10000    | 0.002/0.005                 |              6 | GPU via Slurm  | scripts/slurm/e11_natural_negative_search_phase1.sbatch | implemented_sbatch                   |
| NNS-P2-heldout-architecture-boundary       | phase2_fresh_generality_search | CIFAR-100-LT | ResNet34, WideResNet28-10, or ResNeXt50-32x4d CIFAR stem | 2000/5000          | 0.002/0.005                 |              8 | GPU via Slurm  | not_registered_until_phase1_archive                     | blocked_until_phase1_registry_commit |

## Metric Contract

| metric_id                        | claim_role                     | worse_rule                                                                            | multiplicity_family                       | claim_boundary                                                                           |
|:---------------------------------|:-------------------------------|:--------------------------------------------------------------------------------------|:------------------------------------------|:-----------------------------------------------------------------------------------------|
| primary_tail_output_drift_ratio  | primary natural counterexample | simultaneous or Holm-adjusted 95% lower confidence endpoint is above 1                | all fresh primary settings within a phase | only this metric can support a natural primary full-drift counterexample                 |
| centered_tail_output_drift_ratio | primary robustness check       | same adjusted CI rule as primary, but interpreted as robustness not primary discovery | reported alongside primary family         | cannot replace the full tail-output drift metric                                         |
| true_logit_and_margin_components | component boundary             | raw CI lower endpoint above 1, labeled component-only unless primary also fails       | component exploratory family              | constrains stronger logit-component or margin claims                                     |
| tail_loss_margin_accuracy_diffs  | secondary outcome tradeoff     | raw CI lower endpoint above 0, labeled secondary-only unless primary also fails       | secondary exploratory family              | cannot support a primary drift counterexample or final optimizer-performance claim       |
| head_gain_and_quality_controls   | validity control               | setting is excluded from primary claim if matched head-gain or quality gates fail     | not a discovery metric                    | prevents degenerate negative cases from mismatched head progress or unusable checkpoints |

## Stopping Rules

| rule_id                                | phase            | rule                                                                                                                                     | decision                                                                                                           |
|:---------------------------------------|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| NNS-S1-freeze-before-fresh-runs        | all              | Search space, metric contract, multiplicity procedure, and exclusion list are committed before any fresh outputs are generated.          | outputs generated before this protocol are audit baselines only and cannot support fresh natural-negative claims   |
| NNS-S2-complete-phase-before-discovery | phase1           | Run every declared phase1 setting, except for logged infrastructure failures, before declaring a natural primary counterexample.         | no early stopping on the first negative-looking setting                                                            |
| NNS-S3-primary-success                 | phase1_or_phase2 | At least one fresh primary_tail_output_drift_ratio row has adjusted CI lower endpoint above 1 and passes head-gain and quality controls. | label as fresh natural primary boundary candidate; require replication or held-out architecture before broad claim |
| NNS-S4-finite-null                     | phase1_or_phase2 | All declared settings run, no adjusted primary lower endpoint exceeds 1, and all rows are reported.                                      | report finite null search; do not claim absence of natural counterexamples outside the registered search space     |
| NNS-S5-component-only                  | phase1_or_phase2 | Component or secondary metrics are worse but primary full-drift metric does not pass NNS-S3.                                             | report as claim-boundary evidence only                                                                             |
| NNS-S6-phase2-trigger                  | phase2           | Phase2 architecture search is allowed only after phase1 results are archived without choosing architectures from phase1 outcomes.        | phase2 can support generality or replication, not retroactive phase1 selection                                     |

## Acceptance Gates

| gate_id                   | scope                         | requirement                                                                                                                   | pass_condition                                                                                                                                      |
|:--------------------------|:------------------------------|:------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| NNS-1-protocol-freeze     | fresh natural negative search | Protocol tables and discussion are committed before fresh search outputs exist.                                               | protocol_status.csv marks fresh metric outputs as metric_outputs_not_run and phase prefixes contain at most settings_registry.csv until jobs finish |
| NNS-2-freshness-exclusion | search space                  | No committed-audit setting_id or spent condition-score final partition is reused for selection.                               | fresh evaluator writes an exclusion audit with no overlap against the baseline audit and spent-final registries                                     |
| NNS-3-multiplicity        | statistical decision          | Primary discovery uses simultaneous confidence intervals or Holm-adjusted one-sided bootstrap tests over the phase family.    | every primary decision row includes raw CI, adjusted CI or adjusted p-value, and phase family size                                                  |
| NNS-4-full-reporting      | artifact reporting            | All declared settings are reported, including nulls, component-only reversals, quality failures, and infrastructure failures. | registry row count equals the declared search-space count minus logged infrastructure failures                                                      |
| NNS-5-quality-controls    | claim validity                | Primary negative candidates pass matched head-gain and pre-update tail-quality controls.                                      | candidate rows include head-gain relative error within the registered tolerance and nondegenerate pre-update tail metrics                           |
| NNS-6-claim-boundary      | paper claim                   | Component and secondary tradeoffs are not described as primary full-drift counterexamples.                                    | claim ledger separates primary, component, secondary, and final-performance claims                                                                  |

## Claim Ladder

| claim_id                             | current_status               | unlock_condition                                                                                                    | blocked_if                                                                             |
|:-------------------------------------|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| fresh_natural_primary_counterexample | not_ready                    | NNS-S3 passes on a fresh phase and NNS-1 through NNS-6 pass                                                         | only component or secondary metrics reverse                                            |
| finite_natural_null_search           | not_ready                    | NNS-S4 passes after all declared fresh phase settings run                                                           | fresh outputs are incomplete or selected post hoc                                      |
| component_boundary_cases             | supported_by_committed_audit | already supported as claim-boundary evidence by results/e11_natural_head_tail_boundary/candidate_negative_cases.csv | used as a primary drift or final-performance counterexample                            |
| local_primary_full_drift_mechanism   | unchanged                    | existing positive diagnostics plus current caveats                                                                  | paper claims final loss, accuracy, or universal optimizer superiority from local drift |
| practical_optimizer_performance      | blocked                      | separate tuned benchmark with final metrics                                                                         | only local negative-search evidence exists                                             |

## Protocol Status

| item                                     | status                      | evidence                                                                                                                                                                                                | blocks_stronger_claim_if_missing   |
|:-----------------------------------------|:----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------|
| committed natural audit baseline         | loaded                      | 37 primary rows; strict worse count 0; max CI high 0.936                                                                                                                                                | yes                                |
| fresh natural search protocol            | generated                   | search space, metric contract, stopping rules, gates, and claim ladder written                                                                                                                          | yes                                |
| fresh natural search entrypoints         | implemented                 | scripts/e11_run_natural_negative_search_phase1.py and scripts/slurm/e11_natural_negative_search_phase1.sbatch are registered for phase1                                                                 | yes                                |
| fresh natural search settings registries | locked                      | phase1_cifar100lt_resnet18, phase1_cifar10lt_resnet18, and phase1_tail_quality_controls settings_registry.csv files declare 26 total settings                                                           | yes                                |
| fresh natural search outputs             | metric_outputs_not_run      | phase1 metric files are absent until the submitted GPU jobs finish                                                                                                                                      | yes                                |
| multiplicity-adjusted evaluator          | implemented_pending_outputs | scripts/e11_evaluate_natural_negative_search_phase1.py writes per-seed log-ratio rows and Holm-adjusted decision rows under results/e11_natural_negative_search_protocol/phase1_multiplicity_evaluation | yes                                |
| natural negative claim                   | not_ready                   | no fresh search outputs exist under this protocol                                                                                                                                                       | yes                                |

## Claim Boundary

Allowed now: cite the committed natural-boundary audit as a finite baseline null
for primary full-drift rows and as component/outcome claim-boundary evidence.

Blocked now: claiming a fresh natural primary counterexample, a finite
pre-registered null search, or a practical optimizer-performance result from
this protocol. The phase1 Slurm entrypoint and multiplicity evaluator are
implemented, but those claims still require complete fresh metric outputs and
Holm-adjusted decisions from paired per-seed log-ratio tests that satisfy the
acceptance gates above.

Generated tables:

- [audit_baseline.csv](../results/e11_natural_negative_search_protocol/audit_baseline.csv)
- [search_space_registry.csv](../results/e11_natural_negative_search_protocol/search_space_registry.csv)
- [metric_contract.csv](../results/e11_natural_negative_search_protocol/metric_contract.csv)
- [stopping_rules.csv](../results/e11_natural_negative_search_protocol/stopping_rules.csv)
- [acceptance_gates.csv](../results/e11_natural_negative_search_protocol/acceptance_gates.csv)
- [claim_ladder.csv](../results/e11_natural_negative_search_protocol/claim_ladder.csv)
- [protocol_status.csv](../results/e11_natural_negative_search_protocol/protocol_status.csv)
