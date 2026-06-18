# E11 CIFAR-100-LT Tuned Benchmark Slurm Plan

This generated plan is a no-side-effect launch contract for the registered
164-setting validation grid. It does not submit jobs. Its role is to keep the
top-conference benchmark path executable on the current server without violating
the documented GPU policy or `MaxSubmitJobsPerUser` constraint.

Current missing validation cells: `161`. Planned chunks: `9`.
Each completed cell must produce both `summary.csv` and `occupancy_trace.csv`;
the selection audit remains blocked until both artifacts exist for every
registered setting in a recipe family.

## Queue Policy

| policy_id                      | rule                                                                                                         | evidence                                                                                                          |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|
| SLURM-1-gpu-only-through-slurm | GPU validation jobs are submitted through Slurm, never as direct local CUDA processes.                       | serverREADME.md states gpu_rogue_kill enforces Slurm-only GPU use.                                                |
| SLURM-2-submit-limit           | Submit at most 20 validation array elements per chunk, leaving 6 slots below MaxSubmitJobsPerUser=30.        | The static plan chunks 164 settings into bounded sbatch arrays.                                                   |
| SLURM-3-concurrency            | Use array concurrency %1 for this validation grid unless the queue is explicitly checked before submission.  | The server allows limited concurrent GPU jobs per user; serialized chunks avoid starving the queue.               |
| SLURM-4-no-final-unblinding    | Only validation_tuning seed set 10..14 is eligible for this plan; final_claim seeds 20..29 remain untouched. | The sbatch wrapper calls the validation runner by array index and every registry row has phase=validation_tuning. |

## Chunk Plan

| chunk_id     | array_expression   |   setting_count | recipe_families                                                 | submit_command                                                                                  | expected_outputs                                                       | post_chunk_action                                                              | submission_status         |
|:-------------|:-------------------|----------------:|:----------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|:-------------------------------------------------------------------------------|:--------------------------|
| TBV-SLURM-01 | 3-22               |              20 | adamw_ce_tuned;sgd_momentum_ce_tuned                            | sbatch --array=3-22%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch    | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-02 | 23-42              |              20 | adamw_cb_loss_tuned;sgd_momentum_ce_tuned                       | sbatch --array=23-42%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch   | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-03 | 43-62              |              20 | adamw_cb_loss_tuned;adamw_cb_sampler_tuned;ns_muon_matrix_tuned | sbatch --array=43-62%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch   | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-04 | 63-82              |              20 | ns_muon_matrix_tuned                                            | sbatch --array=63-82%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch   | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-05 | 83-102             |              20 | ns_muon_matrix_tuned                                            | sbatch --array=83-102%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch  | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-06 | 103-122            |              20 | ns_muon_matrix_tuned                                            | sbatch --array=103-122%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-07 | 123-142            |              20 | ns_muon_cb_tuned;ns_muon_matrix_tuned                           | sbatch --array=123-142%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-08 | 143-162            |              20 | ns_muon_cb_tuned                                                | sbatch --array=143-162%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |
| TBV-SLURM-09 | 163                |               1 | ns_muon_cb_tuned                                                | sbatch --array=163%1 scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch     | summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv | rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates | not_submitted_static_plan |

## Operating Rule

Before running a command from this table, check `squeue -u "$USER"` and submit
only one chunk when the current number of submitted jobs plus the chunk size
stays below the server's user submit limit. After each chunk finishes, rerun
`make e11-cifar-resnet-lt-tuned-benchmark-selection`; do not run final seeds
until `TVS-1`, `TVS-2`, and `TVS-5` all pass.
