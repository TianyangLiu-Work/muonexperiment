# E11 CIFAR-100-LT Tuned Benchmark Slurm Launch Audit

This audit records a queue-aware launch decision for the tuned validation grid.
It is separate from the static no-side-effect chunk plan: this file records the
actual queue state, the guarded capacity calculation, and whether `sbatch` was
called.

Current launch status: `submitted`.

## Launch Decision

| timestamp_utc             | submission_status   | submit_command                                                                                        |   slurm_job_id |   current_queue_elements_before_submit |   available_submit_slots_before_submit |   planned_setting_count |   array_expression | planned_recipe_families   |   completed_settings_before_submit |   missing_settings_not_inflight_before_submit | inflight_validation_indices_before_submit                         | final_seed_status   | phase_guard            |
|:--------------------------|:--------------------|:------------------------------------------------------------------------------------------------------|---------------:|---------------------------------------:|---------------------------------------:|------------------------:|-------------------:|:--------------------------|-----------------------------------:|----------------------------------------------:|:------------------------------------------------------------------|:--------------------|:-----------------------|
| 2026-06-19T05:40:30+00:00 | submitted           | sbatch --parsable --array=68%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch |           1407 |                                     23 |                                      1 |                       1 |                 68 | ns_muon_matrix_tuned      |                                 46 |                                            96 | 46;47;48;49;50;51;52;53;54;55;56;57;58;59;60;61;62;63;64;65;66;67 | not_touched         | validation_tuning_only |

## Selected Validation Settings

|   array_index | setting_id                                         | phase             | seed_set   | recipe_family        | recipe_name                                    | planned_output_dir                                                                                                  | planned_occupancy_trace_path                                                                                                            |
|--------------:|:---------------------------------------------------|:------------------|:-----------|:---------------------|:-----------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|
|            68 | TBV-ns_muon_matrix_tuned-lr1e-5-wd5e-4-warm500-ns3 | validation_tuning | 10..14     | ns_muon_matrix_tuned | ns_muon_matrix_tuned_lr1e-5_wd5e-4_warm500_ns3 | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-ns_muon_matrix_tuned-lr1e-5-wd5e-4-warm500-ns3 | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-ns_muon_matrix_tuned-lr1e-5-wd5e-4-warm500-ns3/occupancy_trace.csv |

## Queue Snapshot Before Submit

|   job_id | state   | job_name                 | reason_or_node      |
|---------:|:--------|:-------------------------|:--------------------|
|  1359_47 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_49 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_50 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_51 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_52 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_53 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_54 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_55 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_56 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_57 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_58 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_59 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_60 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_61 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_62 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_63 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_64 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_65 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_66 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1404_67 | PD      | e11-lt-tuned-val         | (JobArrayTaskLimit) |
|  1359_46 | R       | e11-lt-tuned-val         | yumingz5-linux-ml   |
|  1404_48 | R       | e11-lt-tuned-val         | yumingz5-linux-ml   |
|     1403 | R       | verl-branchgrpo-qwen3-4b | yumingz5-linux-ml   |

## Operating Rule

The launch guard preserves `MaxSubmitJobsPerUser` by enforcing
`current_queue_elements + planned_setting_count + reserved_submit_slots <=
max_submit_jobs_per_user`. The submitted command, when non-empty, is:

```bash
sbatch --parsable --array=68%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch
```

Only `validation_tuning` rows with seed set `10..14` are eligible. The untouched
`final_claim` seeds `20..29` remain blocked until validation selection passes
`TVS-1`, `TVS-2`, and `TVS-5`.
