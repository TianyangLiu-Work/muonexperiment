# E11 CIFAR-10-LT ResNet18 All-Layer JVP Checkpoint-Prediction Benchmark

This diagnostic turns the single-checkpoint all-layer JVP bridge into a
checkpoint-transfer prediction test. For each tail-rich ResNet18 checkpoint,
the script probes every Conv/Linear matrix weight, computes unit-JVP,
matched-gain scaled-JVP, and observed layer-only drift ratios, then asks
whether layer scores measured at one checkpoint predict observed layer risk
at held-out checkpoints without fitting a new model. The original
pre-registered predictors are retained, and two positive controls are
added: source-checkpoint observed drift and an early-layer architecture
prior. These controls test whether held-out layer-risk ordering is
predictable at all, rather than attributing every failure to target noise.

The same artifact also reports an architecture-adjusted residual test.
For each source checkpoint, it fits source observed log drift from
log early-layer prior, applies that source fit to the held-out target
checkpoint, and asks which source residual scores predict target
residual risk. This avoids fitting the depth correction on the target
checkpoint itself.

- Warmup checkpoints: 2000, 5000, 10000
- Dataset/model: CIFAR-10-LT / ResNet18
- Seeds per checkpoint: 10
- Head train examples per class: 500
- Tail train examples per class: 500
- Tail eval examples per class: 200
- Target head first-order gain: 0.005 * head-batch loss
- Finite-difference JVP epsilon: 0.0001
- Device/dtype request: cuda/float32

![CIFAR ResNet all-layer JVP checkpoint prediction](../figures/e11_condition_score_v4_protocol/final_data_cifar10_mixed/cifar100_resnet_layer_jvp_checkpoint_prediction.png)

## Checkpoint Summary

|   warmup_steps |   seeds |   parameters |   paired_points |   mean_tail_accuracy_before |   geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro |   scaled_jvp_tail_drift_sq_ratio_ci95_low |   scaled_jvp_tail_drift_sq_ratio_ci95_high |   geomean_observed_tail_drift_sq_ratio_spectral_over_fro |   observed_tail_drift_sq_ratio_ci95_low |   observed_tail_drift_sq_ratio_ci95_high |
|---------------:|--------:|-------------:|----------------:|----------------------------:|-----------------------------------------------------------:|------------------------------------------:|-------------------------------------------:|---------------------------------------------------------:|----------------------------------------:|-----------------------------------------:|
|           2000 |      10 |           21 |             210 |                      0.4053 |                                                     0.2187 |                                    0.203  |                                     0.2357 |                                                   0.3871 |                                  0.3681 |                                   0.407  |
|           5000 |      10 |           21 |             210 |                      0.4031 |                                                     0.2458 |                                    0.2279 |                                     0.2652 |                                                   0.4389 |                                  0.417  |                                   0.4618 |
|          10000 |      10 |           21 |             210 |                      0.3982 |                                                     0.2481 |                                    0.2285 |                                     0.2693 |                                                   0.4753 |                                  0.4514 |                                   0.5004 |

## Held-Out Checkpoint Prediction Summary

| predictor                      | predictor_family   |   checkpoint_transfer_pairs |   mean_spearman_log_predictor_vs_log_target_observed |   spearman_ci95_low |   spearman_ci95_high |   mean_top5_risk_overlap_fraction |   top5_risk_overlap_ci95_low |   top5_risk_overlap_ci95_high |
|:-------------------------------|:-------------------|----------------------------:|-----------------------------------------------------:|--------------------:|---------------------:|----------------------------------:|-----------------------------:|------------------------------:|
| source_observed_drift_ratio    | positive_control   |                           6 |                                               0.9922 |              0.9889 |               0.9956 |                               1   |                          1   |                           1   |
| source_scaled_jvp_ratio        | pre_registered     |                           6 |                                               0.234  |              0.1947 |               0.2733 |                               0.6 |                          0.6 |                           0.6 |
| source_unit_jvp_ratio          | pre_registered     |                           6 |                                              -0.2552 |             -0.2827 |              -0.2277 |                               0.4 |                          0.4 |                           0.4 |
| source_alignment_ratio         | pre_registered     |                           6 |                                              -0.61   |             -0.639  |              -0.5809 |                               0   |                          0   |                           0   |
| source_gradient_nuclear_rank   | pre_registered     |                           6 |                                              -0.61   |             -0.639  |              -0.5809 |                               0   |                          0   |                           0   |
| architecture_early_layer_prior | positive_control   |                           6 |                                               0.1952 |              0.1803 |               0.2101 |                               0.6 |                          0.6 |                           0.6 |

## Readout

- Source observed-drift positive control: Spearman 0.9922 [0.9889, 0.9956], top-5 risk overlap 1.
- Early-layer architecture prior: Spearman 0.1952 [0.1803, 0.2101], top-5 risk overlap 0.6.
- Pre-registered scaled-JVP transfer predictor: Spearman 0.234 [0.1947, 0.2733] over 6 directed checkpoint-transfer pairs.
- Rank-only source predictor: Spearman -0.61 [-0.639, -0.5809].
- Architecture-adjusted source observed residual: Spearman 0.9907 [0.9858, 0.9956], top-5 residual overlap 1.
- Architecture-adjusted scaled-JVP residual: Spearman 0.605 [0.5833, 0.6266].
- Architecture-adjusted gradient-rank residual: Spearman -0.6535 [-0.6802, -0.6267].

Interpretation: this is a checkpoint-transfer mechanism benchmark. The
positive controls show that layer-risk ordering is stable enough to
transfer across the tested tail-rich checkpoints. The current scaled-JVP
readout transfers the below-one direction but not the layer ranking, so
the missing ingredient is in the measurable condition score rather than
only in target-checkpoint noise. The residual benchmark further shows
that observed source residuals transfer after removing the early-layer
prior, while the scaled-JVP residual remains inverted. It is still not a standard long-tailed
classification benchmark or a final optimizer-performance result.

Artifacts:
- [metrics.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/metrics.csv)
- [paired_metrics.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/paired_metrics.csv)
- [layer_summary.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/layer_summary.csv)
- [checkpoint_summary.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/checkpoint_summary.csv)
- [prediction_pairs.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/prediction_pairs.csv)
- [prediction_summary.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/prediction_summary.csv)
- [residual_prediction_pairs.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/residual_prediction_pairs.csv)
- [residual_prediction_summary.csv](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/residual_prediction_summary.csv)
- [config.json](../results/e11_condition_score_v4_protocol/final_data_cifar10_mixed/config.json)
