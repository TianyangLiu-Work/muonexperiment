# E11 CIFAR-100-LT ResNet Condition-Score v2 Retrospective Analysis

This generated analysis is the first executable step after the registered condition-score protocol. It uses the existing ResNet18 checkpoint-transfer `layer_summary.csv` as a locked retrospective split: each source checkpoint fits the calibrated residual score, then the frozen score predicts residual observed layer risk on the other checkpoints.

![CIFAR-100-LT ResNet condition-score v2 retrospective analysis](../figures/e11_cifar100_resnet_condition_score_next/cifar100_resnet_condition_score_next.png)

## Summary

| score                                  | score_family      |   checkpoint_transfer_pairs |   mean_spearman_score_vs_target_residual |   spearman_ci95_low |   spearman_ci95_high |   mean_top5_residual_risk_overlap_fraction | mean_threshold_below_one_accuracy   |
|:---------------------------------------|:------------------|----------------------------:|-----------------------------------------:|--------------------:|---------------------:|-------------------------------------------:|:------------------------------------|
| condition_score_v2_calibrated_residual | primary_candidate |                           6 |                                   0.5942 |             0.5428  |               0.6455 |                                     0.5333 | 0.9841                              |
| early_layer_prior                      | baseline          |                           6 |                                   0.2671 |             0.2222  |               0.312  |                                     0.2    | n/a                                 |
| legacy_scaled_jvp_ratio                | baseline          |                           6 |                                  -0.4667 |            -0.5532  |              -0.3802 |                                     0.1333 | 1                                   |
| theory_sign_composite                  | baseline          |                           6 |                                   0.2394 |             0.09677 |               0.382  |                                     0.4    | n/a                                 |
| source_observed_drift_positive_control | positive_control  |                           6 |                                   0.9403 |             0.9216  |               0.959  |                                     0.8667 | 1                                   |

## Gate Report

| gate_id                              | scope                                   | status    | evidence                                                                                                                     |
|:-------------------------------------|:----------------------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------|
| legacy_checkpoint_residual_spearman  | retrospective ResNet18 checkpoint split | pass      | primary residual Spearman=0.5942 CI=[0.5428, 0.6455]                                                                         |
| legacy_checkpoint_threshold_accuracy | retrospective ResNet18 checkpoint split | pass      | primary below-one threshold accuracy=0.9841                                                                                  |
| baseline_comparison                  | retrospective ResNet18 checkpoint split | pass      | primary Spearman=0.5942; early-layer prior Spearman=0.2671                                                                   |
| primary_heldout_architecture         | P0 protocol                             | not_run   | ResNet34 held-out architecture split is registered but not generated.                                                        |
| primary_heldout_data                 | P0 protocol                             | not_run   | CIFAR-10-LT held-out data split is registered but not generated.                                                             |
| p0_predictive_condition_claim        | paper claim                             | not_ready | The retrospective checkpoint split is promising, but the registered architecture and data held-out splits are still missing. |

## Readout

- Primary `condition_score_v2_calibrated_residual` residual Spearman is 0.5942 [0.5428, 0.6455] on the retrospective checkpoint split.
- The early-layer prior residual Spearman is 0.2671 [0.2222, 0.312].
- The legacy scaled-JVP residual score remains negative at -0.4667 [-0.5532, -0.3802].

## Boundary

This is not yet the P0 predictive-condition result. It is a locked ResNet18 checkpoint-split analysis showing that the registered v2 score is worth running on the protocol's held-out architecture and held-out data splits. The paper must still call the P0 condition-score claim incomplete until those GPU/Slurm splits exist and pass the registered gates.

Artifacts:
- [score_pairs.csv](../results/e11_cifar100_resnet_condition_score_next/score_pairs.csv)
- [score_summary.csv](../results/e11_cifar100_resnet_condition_score_next/score_summary.csv)
- [calibration_coefficients.csv](../results/e11_cifar100_resnet_condition_score_next/calibration_coefficients.csv)
- [gate_report.csv](../results/e11_cifar100_resnet_condition_score_next/gate_report.csv)
- [config.json](../results/e11_cifar100_resnet_condition_score_next/config.json)
