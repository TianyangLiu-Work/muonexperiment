# E11 CIFAR-100-LT ResNet18 Tuned Benchmark Validation Setting

This run is one cell of the preregistered 164-setting tuned validation grid.
It uses validation seeds only and cannot select or report the untouched
final-claim seeds by itself.

- Setting id: `TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0`
- Array index: 6

- Recipe family: `adamw_ce_tuned`
- Recipe: `adamw_ce_tuned_lr3e-4_wd5e-4_warm0`
- Phase / seed set: `validation_tuning` / `10..14`
- Protocol reference: `discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md`
- Number of classes: 100
- Max train examples per class: 500
- Imbalance factor: 100.0
- Train steps: 10000
- Batch size: 256
- Augmentation: reflect-padded random crop with padding 4 and horizontal flip probability 0.5
- Device/dtype request: cuda/float32

![CIFAR-100-LT tuned validation setting](../figures/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/cifar100_resnet_lt_recipe_benchmark.png)

## Summary

| recipe                             | frequency_group   |   seeds |   classes |   mean_balanced_accuracy |   balanced_accuracy_ci95_low |   balanced_accuracy_ci95_high |   mean_loss |   mean_margin |
|:-----------------------------------|:------------------|--------:|----------:|-------------------------:|-----------------------------:|------------------------------:|------------:|--------------:|
| adamw_ce_tuned_lr3e-4_wd5e-4_warm0 | many              |       5 |        35 |                  0.6338  |                      0.6262  |                       0.6415  |       2.153 |         3.325 |
| adamw_ce_tuned_lr3e-4_wd5e-4_warm0 | medium            |       5 |        35 |                  0.3428  |                      0.3323  |                       0.3533  |       4.656 |        -2.57  |
| adamw_ce_tuned_lr3e-4_wd5e-4_warm0 | few               |       5 |        30 |                  0.08367 |                      0.07954 |                       0.08779 |       7.72  |        -7.159 |
| adamw_ce_tuned_lr3e-4_wd5e-4_warm0 | all               |       5 |       100 |                  0.3669  |                      0.3615  |                       0.3724  |       4.699 |        -1.883 |

## Paired Differences

| recipe   | frequency_group   | seeds   | mean_balanced_accuracy_diff   | balanced_accuracy_diff_ci95_low   | balanced_accuracy_diff_ci95_high   | mean_loss_diff   | mean_margin_diff   |
|----------|-------------------|---------|-------------------------------|-----------------------------------|------------------------------------|------------------|--------------------|

## Occupancy Trace

| setting_id                             |   occupancy_probe_rows |   occupancy_seed_count |   occupancy_eval_step_count |   mean_batch_few_fraction |   mean_tail_probe_loss |   mean_ns_tail_output_drift_sq_ratio_vs_fro |
|:---------------------------------------|-----------------------:|-----------------------:|----------------------------:|--------------------------:|-----------------------:|--------------------------------------------:|
| TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0 |                     30 |                      5 |                           6 |                   0.02786 |                   6.92 |                                      0.5643 |

## Readout

- `adamw_ce_tuned_lr3e-4_wd5e-4_warm0` many/medium/few balanced accuracy: 0.6338 [0.6262, 0.6415] / 0.3428 [0.3323, 0.3533] / 0.08367 [0.07954, 0.08779].

Interpretation: this is validation evidence for the frozen tuned benchmark
selection rule only. It may update `TVS-1` and `TVS-5`, but it does not
unblock a final-performance or broad optimizer claim until all registered
validation settings complete, one recipe per family is selected without
peeking, and final seeds `20..29` are run pairwise.

Artifacts:
- [train_trace.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/train_trace.csv)
- [class_metrics.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/class_metrics.csv)
- [group_metrics.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/group_metrics.csv)
- [summary.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/summary.csv)
- [pair_summary.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/pair_summary.csv)
- [occupancy_trace.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/occupancy_trace.csv)
- [setting_metadata.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/setting_metadata.csv)
- [config.json](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_ce_tuned-lr3e-4-wd5e-4-warm0/config.json)
