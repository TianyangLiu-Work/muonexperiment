# E11 CIFAR-100-LT Tuned Benchmark Slurm Launch Audit

This audit records a queue-aware launch decision for the tuned validation grid.
It is separate from the static no-side-effect chunk plan: this file records the
actual queue state, the guarded capacity calculation, and whether `sbatch` was
called.

Current launch status: `submitted`.

## Launch Decision

| timestamp_utc             | submission_status   | submit_command                                                                                          |   slurm_job_id |   current_queue_elements_before_submit |   available_submit_slots_before_submit |   planned_setting_count | array_expression   | planned_recipe_families              | final_seed_status   | phase_guard            |
|:--------------------------|:--------------------|:--------------------------------------------------------------------------------------------------------|---------------:|---------------------------------------:|---------------------------------------:|------------------------:|:-------------------|:-------------------------------------|:--------------------|:-----------------------|
| 2026-06-18T01:04:04+00:00 | submitted           | sbatch --parsable --array=0-16%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch |           1289 |                                      7 |                                     17 |                      17 | 0-16               | adamw_ce_tuned;sgd_momentum_ce_tuned | not_touched         | validation_tuning_only |

## Selected Validation Settings

|   array_index | setting_id                                      | phase             | seed_set   | recipe_family         | recipe_name                                 | planned_output_dir                                                                                               | planned_occupancy_trace_path                                                                                                         |
|--------------:|:------------------------------------------------|:------------------|:-----------|:----------------------|:--------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
|             0 | TBV-adamw_ce_tuned-lr1e-4-wd1e-4-warm0          | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-4_wd1e-4_warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd1e-4-warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd1e-4-warm0/occupancy_trace.csv          |
|             1 | TBV-adamw_ce_tuned-lr1e-4-wd1e-4-warm500        | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-4_wd1e-4_warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd1e-4-warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd1e-4-warm500/occupancy_trace.csv        |
|             2 | TBV-adamw_ce_tuned-lr1e-4-wd5e-4-warm0          | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-4_wd5e-4_warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd5e-4-warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd5e-4-warm0/occupancy_trace.csv          |
|             3 | TBV-adamw_ce_tuned-lr1e-4-wd5e-4-warm500        | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-4_wd5e-4_warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd5e-4-warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-4-wd5e-4-warm500/occupancy_trace.csv        |
|             4 | TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm0          | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr3e-4_wd1e-4_warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm0/occupancy_trace.csv          |
|             5 | TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm500        | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr3e-4_wd1e-4_warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd1e-4-warm500/occupancy_trace.csv        |
|             6 | TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0          | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr3e-4_wd5e-4_warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/occupancy_trace.csv          |
|             7 | TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm500        | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr3e-4_wd5e-4_warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm500/occupancy_trace.csv        |
|             8 | TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm0          | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-3_wd1e-4_warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm0/occupancy_trace.csv          |
|             9 | TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm500        | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-3_wd1e-4_warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd1e-4-warm500/occupancy_trace.csv        |
|            10 | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0          | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-3_wd5e-4_warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0          | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm0/occupancy_trace.csv          |
|            11 | TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm500        | validation_tuning | 10..14     | adamw_ce_tuned        | adamw_ce_tuned_lr1e-3_wd5e-4_warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm500        | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr1e-3-wd5e-4-warm500/occupancy_trace.csv        |
|            12 | TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm0   | validation_tuning | 10..14     | sgd_momentum_ce_tuned | sgd_momentum_ce_tuned_lr0p03_wd5e-4_warm0   | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm0   | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm0/occupancy_trace.csv   |
|            13 | TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm500 | validation_tuning | 10..14     | sgd_momentum_ce_tuned | sgd_momentum_ce_tuned_lr0p03_wd5e-4_warm500 | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm500 | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd5e-4-warm500/occupancy_trace.csv |
|            14 | TBV-sgd_momentum_ce_tuned-lr0p03-wd1e-3-warm0   | validation_tuning | 10..14     | sgd_momentum_ce_tuned | sgd_momentum_ce_tuned_lr0p03_wd1e-3_warm0   | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd1e-3-warm0   | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd1e-3-warm0/occupancy_trace.csv   |
|            15 | TBV-sgd_momentum_ce_tuned-lr0p03-wd1e-3-warm500 | validation_tuning | 10..14     | sgd_momentum_ce_tuned | sgd_momentum_ce_tuned_lr0p03_wd1e-3_warm500 | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd1e-3-warm500 | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p03-wd1e-3-warm500/occupancy_trace.csv |
|            16 | TBV-sgd_momentum_ce_tuned-lr0p1-wd5e-4-warm0    | validation_tuning | 10..14     | sgd_momentum_ce_tuned | sgd_momentum_ce_tuned_lr0p1_wd5e-4_warm0    | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p1-wd5e-4-warm0    | results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-sgd_momentum_ce_tuned-lr0p1-wd5e-4-warm0/occupancy_trace.csv    |

## Queue Snapshot Before Submit

|   job_id | state   | job_name                 | reason_or_node    |
|---------:|:--------|:-------------------------|:------------------|
|     1285 | PD      | cont-bank-seed2v1        | (Dependency)      |
|     1284 | PD      | score-bank-seed2v1       | (Dependency)      |
|     1246 | PD      | verl-branchgrpo-qwen3-4b | (Dependency)      |
|     1245 | PD      | verl-branchgrpo-qwen3-4b | (Dependency)      |
|     1286 | R       | llmc-quip-tinyllama      | yumingz5-linux-ml |
|     1283 | R       | online-bank-seed2v1      | yumingz5-linux-ml |
|     1244 | R       | verl-branchgrpo-qwen3-4b | yumingz5-linux-ml |

## Operating Rule

The launch guard preserves `MaxSubmitJobsPerUser` by enforcing
`current_queue_elements + planned_setting_count + reserved_submit_slots <=
max_submit_jobs_per_user`. The submitted command, when non-empty, is:

```bash
sbatch --parsable --array=0-16%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch
```

Only `validation_tuning` rows with seed set `10..14` are eligible. The untouched
`final_claim` seeds `20..29` remain blocked until validation selection passes
`TVS-1`, `TVS-2`, and `TVS-5`.
