# E11 Optimizer Ablation Map

This generated map organizes the current E11 controls by what each ablation actually isolates. Its purpose is to prevent the paper from treating every Adam/Muon difference as evidence for the same mechanism.

## Current Ablation Coverage

| control_level                       | isolates                                                                                             | main_evidence                                                                                                  | paper_role                                                                                                               |
|:------------------------------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| Natural Adam vs Muon                | Nothing; optimizer identity, update scale, direction, state, and layer allocation are all entangled. | Raw all-task first-order ratio=0.1511 [0.1373, 0.1664]; family signs split: MS=0.4287, MF=0.1092.              | Shows why the paper should avoid a global Muon-is-better claim.                                                          |
| Matched global update size          | Direction plus layer allocation after removing global relative update-norm scale.                    | `nrUpdate` ratio=2.033 [1.921, 2.151]; `stUpdate` ratio=5.215 [5.008, 5.43].                                   | Core optimizer signature: Muon robustly shapes update spectra at matched step size.                                      |
| Target global update norm           | Direction effects across a sweep of shared update sizes.                                             | MatrixSensing kappa=1e2 first-order ratio=1.072 [1.038, 1.107]; MLP hidden=64 ratio=0.6047 [0.5818, 0.6285].   | Shows direction advantage is task/layer conditional, not an LR artifact.                                                 |
| Matched per-layer update size       | Layerwise direction after removing layer allocation for SmallMLP.                                    | Hidden=16 ratio=0.9408 [0.9309, 0.9507]; hidden=64 ratio=0.6882 [0.6761, 0.7006].                              | Separates global direction effects from per-layer update-budget allocation.                                              |
| Stateless direction choice          | Candidate update direction at the same checkpoint state, gradient, and global Frobenius update size. | PolarMuon/GD `nrUpdate` ratio=2.779 [2.668, 2.895], but first-order ratio=0.5032 [0.4857, 0.5213].             | Shows polar direction alone creates the high-rank update spectrum, while Frobenius-matched progress remains conditional. |
| Stateless optimizer trajectories    | Short multi-step training with the same target update size and no optimizer state.                   | PolarMuon/GD mean_nrUpdate=2.704 [2.236, 3.269], but total_decrease=0.4656 [0.3686, 0.5882].                   | Extends the stateless direction result beyond a single step while preserving the same boundary conclusion.               |
| Synthetic singular-value allocation | Singular values with gradient singular vectors fixed.                                                | flat/GD ratio=0.6071 [0.5846, 0.6304] under Frobenius budget; 1.689 [1.614, 1.768] under operator-norm budget. | Mechanism probe linking Muon's polar spectrum to the local theory note.                                                  |
| Synthetic singular-vector swap      | Singular vectors at matched spectrum/norm in one-step probes.                                        | Overall other/own first-order ratio=0.5053 [0.4209, 0.5896].                                                   | Shows optimizer trajectories can move into different vector geometries, not just different spectra.                      |
| Natural update-vector swap          | Each optimizer's actual proposed update vector, including state and spectrum.                        | Overall other/own first-order ratio=0.4925 [0.4423, 0.5427].                                                   | Checks whether actual optimizer-specific updates are locally consequential.                                              |
| Fresh continuation and LR sweep     | Whether local one-step specialization predicts longer-horizon optimizer continuation.                | Best switched/own final-loss ratio=2.304 [1.573, 3.035].                                                       | Negative control: local geometry does not imply global optimizer superiority.                                            |

## Interpretation

The current ablations support a layered claim:

1. Muon robustly changes update spectra at matched update size.
2. Removing global update-size differences does not create a universal Muon advantage.
3. Stateless direction and short-trajectory controls show that polar direction alone creates the high-rank update spectrum under matched Frobenius update size.
4. Per-layer controls and synthetic spectral-allocation probes show that layer allocation, norm budget, and singular-value allocation are distinct mechanisms.
5. Singular-vector and natural-update swaps show that optimizer-specific trajectories matter locally.
6. Continuation probes show that local one-step geometry does not automatically predict longer-horizon optimizer superiority.

## Still Missing For A Stronger Variant Claim

| missing_variant                                                   | why_it_matters                                                                                                       |
|:------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| Momentum-free Muon and momentum-free Adam-like training baselines | Would separate polar spectrum shaping from optimizer state accumulation over a full trajectory.                      |
| GD-spectrum matched natural optimizer                             | Would compare natural training with gradient-spectrum updates rather than only one-step synthetic probes.            |
| Flat/polar update with Adam-style state removed or standardized   | The stateless ablation addresses one-step direction choice; this would test the same question over natural training. |
| Broader neural architecture control                               | Would test whether the ablation map survives beyond shallow MLPs.                                                    |

## Safe Paper Wording

Use:

> The current ablation ladder separates update scale, layer allocation, stateless direction and trajectory choice, singular-value allocation, singular-vector geometry, and continuation effects. The robust optimizer-intrinsic signal is update-spectrum shaping; its optimization benefit is conditional.

Avoid:

> The ablations prove that Muon's performance differences are caused only by polar spectrum shaping.
