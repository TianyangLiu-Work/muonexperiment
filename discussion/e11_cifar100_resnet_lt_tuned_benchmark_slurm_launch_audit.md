# E11 CIFAR-100-LT Tuned Benchmark Slurm Launch Audit

This audit records a queue-aware launch decision for the tuned validation grid.
It is separate from the static no-side-effect chunk plan: this file records the
actual queue state, the guarded capacity calculation, and whether `sbatch` was
called.

Current launch status: `blocked_no_queue_capacity`.

## Launch Decision

| timestamp_utc             | submission_status         | submit_command   | slurm_job_id   |   current_queue_elements_before_submit |   available_submit_slots_before_submit |   planned_setting_count | array_expression   | planned_recipe_families   |   completed_settings_before_submit |   missing_settings_not_inflight_before_submit | inflight_validation_indices_before_submit   | final_seed_status   | phase_guard            |
|:--------------------------|:--------------------------|:-----------------|:---------------|---------------------------------------:|---------------------------------------:|------------------------:|:-------------------|:--------------------------|-----------------------------------:|----------------------------------------------:|:--------------------------------------------|:--------------------|:-----------------------|
| 2026-06-18T13:24:05+00:00 | blocked_no_queue_capacity |                  |                |                                     24 |                                      0 |                       0 |                    |                           |                                 17 |                                           136 | 17;18;19;20;21;22;23;24;25;26;27            | not_touched         | validation_tuning_only |

## Selected Validation Settings

| array_index   | setting_id   | phase   | seed_set   | recipe_family   | recipe_name   | planned_output_dir   | planned_occupancy_trace_path   |
|---------------|--------------|---------|------------|-----------------|---------------|----------------------|--------------------------------|

## Queue Snapshot Before Submit

|   job_id | state   | job_name                 | reason_or_node    |
|---------:|:--------|:-------------------------|:------------------|
|     1332 | PD      | online-bank-seed7v1      | (Resources)       |
|     1324 | PD      | score-bank-seed4v1       | (Priority)        |
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
|     1334 | PD      | cont-bank-seed7v1        | (Dependency)      |
|     1333 | PD      | score-bank-seed7v1       | (Dependency)      |
|     1331 | PD      | cont-bank-seed6v1        | (Dependency)      |
|     1330 | PD      | score-bank-seed6v1       | (Dependency)      |
|     1328 | PD      | cont-bank-seed5v1        | (Dependency)      |
|     1327 | PD      | score-bank-seed5v1       | (Dependency)      |
|     1325 | PD      | cont-bank-seed4v1        | (Dependency)      |
|     1246 | PD      | verl-branchgrpo-qwen3-4b | (Dependency)      |
|     1329 | R       | online-bank-seed6v1      | yumingz5-linux-ml |
|     1326 | R       | online-bank-seed5v1      | yumingz5-linux-ml |
|     1245 | R       | verl-branchgrpo-qwen3-4b | yumingz5-linux-ml |

## Operating Rule

The launch guard preserves `MaxSubmitJobsPerUser` by enforcing
`current_queue_elements + planned_setting_count + reserved_submit_slots <=
max_submit_jobs_per_user`. The submitted command, when non-empty, is:

```bash

```

Only `validation_tuning` rows with seed set `10..14` are eligible. The untouched
`final_claim` seeds `20..29` remain blocked until validation selection passes
`TVS-1`, `TVS-2`, and `TVS-5`.
