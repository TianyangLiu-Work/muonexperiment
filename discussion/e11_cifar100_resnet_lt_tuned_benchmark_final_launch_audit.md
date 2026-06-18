# E11 CIFAR-100-LT Tuned Benchmark Final Launch Audit

This audit records a queue-aware launch decision for the untouched final-claim
seed jobs. It is separate from the final execution plan: this file records the
actual queue state, the final-launch gates, and whether `sbatch` was called.

Current final launch status: `blocked_final_launch_gates_not_ready`.

## Launch Gate Snapshot

| gate_id                           | status    | evidence                                                                                                                                                                                 | blocks_submit   |
|:----------------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------|
| FLA-1-final-execution-gates-ready | not_ready | FEP-1-selection-gates-ready=not_ready; FEP-2-final-analysis-plan-linked=pass; FEP-3-runner-contract-present=pass; FEP-4-final-output-quarantine=pass; FEP-5-all-families-ready=not_ready | yes             |
| FLA-2-final-evaluator-implemented | pass      | TFE-1-evaluator-implemented=pass                                                                                                                                                         | yes             |
| FLA-3-no-final-job-duplicate      | pass      | no final Slurm array elements inflight                                                                                                                                                   | yes             |
| FLA-4-queue-capacity              | not_ready | available_submit_slots=1; required_final_array_elements=6; current_queue_elements=23                                                                                                     | yes             |
| FLA-5-submit-flag                 | dry_run   | no --submit flag; audit only                                                                                                                                                             | no              |

## Launch Decision

| timestamp_utc             | submission_status                    | submit_command   | slurm_job_id   |   current_queue_elements_before_submit |   available_submit_slots_before_submit |   planned_final_array_elements | array_expression   | planned_recipe_families   | inflight_final_indices_before_submit   | final_seed_status                  | phase_guard      |
|:--------------------------|:-------------------------------------|:-----------------|:---------------|---------------------------------------:|---------------------------------------:|-------------------------------:|:-------------------|:--------------------------|:---------------------------------------|:-----------------------------------|:-----------------|
| 2026-06-18T15:11:14+00:00 | blocked_final_launch_gates_not_ready |                  |                |                                     23 |                                      1 |                              0 |                    |                           |                                        | eligible_only_after_fep_gates_pass | final_claim_only |

## Final Family Plan

|   final_index | recipe_family          | run_status                         | selected_setting_id                    | selected_recipe_name               | final_seed_set   | planned_output_dir                                                        |
|--------------:|:-----------------------|:-----------------------------------|:---------------------------------------|:-----------------------------------|:-----------------|:--------------------------------------------------------------------------|
|             0 | adamw_cb_loss_tuned    | blocked_until_family_selected      | n/a                                    | n/a                                | n/a              | n/a                                                                       |
|             1 | adamw_cb_sampler_tuned | blocked_until_family_selected      | n/a                                    | n/a                                | n/a              | n/a                                                                       |
|             2 | adamw_ce_tuned         | blocked_until_selection_gates_pass | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0 | adamw_ce_tuned_lr1e-3_wd5e-4_warm0 | 20..29           | results/e11_cifar100_resnet_lt_tuned_benchmark/final_claim/adamw_ce_tuned |
|             3 | ns_muon_cb_tuned       | blocked_until_family_selected      | n/a                                    | n/a                                | n/a              | n/a                                                                       |
|             4 | ns_muon_matrix_tuned   | blocked_until_family_selected      | n/a                                    | n/a                                | n/a              | n/a                                                                       |
|             5 | sgd_momentum_ce_tuned  | blocked_until_family_selected      | n/a                                    | n/a                                | n/a              | n/a                                                                       |

## Queue Snapshot Before Submit

|   job_id | state   | job_name                 | reason_or_node    |
|---------:|:--------|:-------------------------|:------------------|
|     1324 | PD      | score-bank-seed4v1       | (Resources)       |
|  1342_17 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_18 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_19 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_20 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_21 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_22 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_23 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_24 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_25 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_26 | PD      | e11-lt-tuned-val         | (Priority)        |
|  1342_27 | PD      | e11-lt-tuned-val         | (Priority)        |
|     1327 | PD      | score-bank-seed5v1       | (Priority)        |
|     1334 | PD      | cont-bank-seed7v1        | (Dependency)      |
|     1333 | PD      | score-bank-seed7v1       | (Dependency)      |
|     1331 | PD      | cont-bank-seed6v1        | (Dependency)      |
|     1330 | PD      | score-bank-seed6v1       | (Dependency)      |
|     1328 | PD      | cont-bank-seed5v1        | (Dependency)      |
|     1325 | PD      | cont-bank-seed4v1        | (Dependency)      |
|     1246 | PD      | verl-branchgrpo-qwen3-4b | (Dependency)      |
|     1332 | R       | online-bank-seed7v1      | yumingz5-linux-ml |
|     1329 | R       | online-bank-seed6v1      | yumingz5-linux-ml |
|     1245 | R       | verl-branchgrpo-qwen3-4b | yumingz5-linux-ml |

## Operating Rule

The launch guard preserves the final seed quarantine by requiring every FEP gate
to pass before a final Slurm command is emitted. The submitted command, when
non-empty, is:

```bash

```

Only `final_claim` seed set `20..29` is eligible, and the evaluator contract
must exist before final jobs are launched. Current `not_ready` rows authorize no benchmark-performance wording and no final submission.
