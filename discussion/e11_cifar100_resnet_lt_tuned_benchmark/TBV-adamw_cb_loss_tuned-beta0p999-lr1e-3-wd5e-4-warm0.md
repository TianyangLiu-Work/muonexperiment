# E11 CIFAR-100-LT ResNet18 Tuned Benchmark Validation Setting

This run is one cell of the preregistered 164-setting tuned validation grid.
It uses validation seeds only and cannot select or report the untouched
final-claim seeds by itself.

- Setting id: `TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0`
- Array index: 34

- Recipe family: `adamw_cb_loss_tuned`
- Recipe: `adamw_cb_loss_tuned_beta0p999_lr1e-3_wd5e-4_warm0`
- Phase / seed set: `validation_tuning` / `10..14`
- Protocol reference: `discussion/e11_cifar100_resnet_lt_tuned_benchmark_protocol.md`
- Number of classes: 100
- Max train examples per class: 500
- Imbalance factor: 100.0
- Train steps: 10000
- Batch size: 256
- Augmentation: reflect-padded random crop with padding 4 and horizontal flip probability 0.5
- Device/dtype request: cuda/float32

![CIFAR-100-LT tuned validation setting](../figures/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/cifar100_resnet_lt_recipe_benchmark.png)

## Summary

| recipe                                            | frequency_group   |   seeds |   classes |   mean_balanced_accuracy |   balanced_accuracy_ci95_low |   balanced_accuracy_ci95_high |   mean_loss |   mean_margin |
|:--------------------------------------------------|:------------------|--------:|----------:|-------------------------:|-----------------------------:|------------------------------:|------------:|--------------:|
| adamw_cb_loss_tuned_beta0p999_lr1e-3_wd5e-4_warm0 | many              |       5 |        35 |                   0.5931 |                      0.5804  |                        0.6057 |       2.476 |         1.974 |
| adamw_cb_loss_tuned_beta0p999_lr1e-3_wd5e-4_warm0 | medium            |       5 |        35 |                   0.3201 |                      0.3023  |                        0.3378 |       5.492 |        -3.497 |
| adamw_cb_loss_tuned_beta0p999_lr1e-3_wd5e-4_warm0 | few               |       5 |        30 |                   0.085  |                      0.06644 |                        0.1036 |       9.553 |        -8.922 |
| adamw_cb_loss_tuned_beta0p999_lr1e-3_wd5e-4_warm0 | all               |       5 |       100 |                   0.3451 |                      0.3295  |                        0.3607 |       5.655 |        -3.209 |

## Paired Differences

| recipe   | frequency_group   | seeds   | mean_balanced_accuracy_diff   | balanced_accuracy_diff_ci95_low   | balanced_accuracy_diff_ci95_high   | mean_loss_diff   | mean_margin_diff   |
|----------|-------------------|---------|-------------------------------|-----------------------------------|------------------------------------|------------------|--------------------|

## Occupancy Trace

| setting_id                                            |   occupancy_probe_rows |   occupancy_seed_count |   occupancy_eval_step_count |   mean_batch_few_fraction |   mean_tail_probe_loss |   mean_ns_tail_output_drift_sq_ratio_vs_fro |
|:------------------------------------------------------|-----------------------:|-----------------------:|----------------------------:|--------------------------:|-----------------------:|--------------------------------------------:|
| TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0 |                     30 |                      5 |                           6 |                   0.02786 |                   10.3 |                                        0.37 |

## Readout

- `adamw_cb_loss_tuned_beta0p999_lr1e-3_wd5e-4_warm0` many/medium/few balanced accuracy: 0.5931 [0.5804, 0.6057] / 0.3201 [0.3023, 0.3378] / 0.085 [0.06644, 0.1036].

Interpretation: this is validation evidence for the frozen tuned benchmark
selection rule only. It may update `TVS-1` and `TVS-5`, but it does not
unblock a final-performance or broad optimizer claim until all registered
validation settings complete, one recipe per family is selected without
peeking, and final seeds `20..29` are run pairwise.

Artifacts:
- [train_trace.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/train_trace.csv)
- [class_metrics.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/class_metrics.csv)
- [group_metrics.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/group_metrics.csv)
- [summary.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/summary.csv)
- [pair_summary.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/pair_summary.csv)
- [occupancy_trace.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/occupancy_trace.csv)
- [setting_metadata.csv](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/setting_metadata.csv)
- [config.json](../results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/TBV-adamw_cb_loss_tuned-beta0p999-lr1e-3-wd5e-4-warm0/config.json)
