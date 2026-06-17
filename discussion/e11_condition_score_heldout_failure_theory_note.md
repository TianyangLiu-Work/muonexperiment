# E11 Held-Out Condition-Score Failure Theory Note

This note records the scientific consequence of the registered held-out condition-score evaluation. The result should be treated as a boundary condition for the theory, not as an implementation nuisance.

## Registered Result

The frozen `condition_score_v2_calibrated_residual` score passes the locked retrospective ResNet18 checkpoint split, but it fails the registered no-tuning held-out residual-ranking gates:

- ResNet34/CIFAR-100-LT held-out architecture: primary residual Spearman `0.1992 [-0.07515, 0.4735]`.
- ResNet18/CIFAR-10-LT held-out data: primary residual Spearman `-0.6771 [-0.7011, -0.653]`.
- Below-one threshold direction still passes on both splits: `0.991` and `0.9841`.
- On CIFAR-10-LT, the legacy scaled-JVP ratio has residual Spearman `0.6219 [0.6013, 0.6425]`, so the v2 calibrated residual score is not uniformly stronger than the older JVP readout.

## Theory Readout

The failure separates two objects that should not be collapsed in the paper:

1. A direction-threshold diagnostic: can a score preserve the sign of whether matched-head-gain spectral drift is below Frobenius drift?
2. A residual-ranking diagnostic: can a score rank which layers carry the largest unexplained observed tail-drift risk after depth adjustment?

The current v2 score preserves the first object but not the second. That means the theorem-to-measurement bridge cannot be stated as a single scalar condition without specifying which target it predicts.

The held-out architecture split and held-out data split fail differently. Source-observed residuals still transfer on the ResNet34 architecture split, but source-observed controls do not transfer on the CIFAR-10-LT data split. This points to two distinct obstructions: architecture perturbation weakens the learned calibrated residual score, while data-family transfer changes the residual risk structure itself.

## Protocol Consequence

These held-out splits are now spent. A stronger condition score cannot be tuned on these failures and then reported as a clean P0 held-out result. The next acceptable protocol needs:

- A score derived from the theorem quantities or a stated falsifiable conjecture before seeing fresh held-out targets.
- A nested calibration plan that keeps any feature or coefficient choice away from the final held-out architecture and data splits.
- Fresh held-out splits beyond ResNet34/CIFAR-100-LT and ResNet18/CIFAR-10-LT.
- A required report of residual-ranking, threshold direction, early-layer prior, source-observed positive control, and legacy scaled-JVP behavior, including negative outcomes.

## Claim Boundary

The present evidence supports a local matched-head-gain drift mechanism and a directional threshold guardrail. It does not support the claim that the current v2 condition score predicts held-out layer-risk ranking across architecture and data families.
