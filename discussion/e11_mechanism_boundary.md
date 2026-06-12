# E11 Mechanism Boundary Map

This generated note summarizes when Muon's update-spectrum shaping does or does not translate into one-step progress. It uses only controlled summaries already generated in the current experiment suite.

## Short Answer

The mechanism boundary is not "higher rank is better." The current evidence is:

1. Muon reliably changes the update spectrum.
2. One-step progress follows the positive descent proxy `<G,D>`, where `D=W-W^+`.
3. Whether the flat/polar update spectrum improves `<G,D>` depends on task family, layer/width, update-size control, and norm budget.

## Boundary Evidence

| boundary_axis            | condition                  | metric            | ratio_label             |   ratio | ci95             | direction             | controlled_question                                                                               |
|:-------------------------|:---------------------------|:------------------|:------------------------|--------:|:-----------------|:----------------------|:--------------------------------------------------------------------------------------------------|
| task_family              | MatrixFactorizationInput   | update_grad_inner | Muon/Adam               |  0.3432 | [0.3194, 0.3687] | Adam/GD-favorable     | At matched global update size, does Muon's natural update improve <G,D>?                          |
| task_family              | MatrixSensing              | update_grad_inner | Muon/Adam               |  1.236  | [1.215, 1.258]   | Muon-favorable        | At matched global update size, does Muon's natural update improve <G,D>?                          |
| task_family              | SmallMLPDigits             | update_grad_inner | Muon/Adam               |  0.5232 | [0.4909, 0.5576] | Adam/GD-favorable     | At matched global update size, does Muon's natural update improve <G,D>?                          |
| target_update_size       | MF input kappa=1e+02       | update_grad_inner | Muon/Adam               |  0.4143 | [0.3868, 0.4438] | Adam/GD-favorable     | After sweeping matched target update norms, does Muon's direction remain favorable?               |
| target_update_size       | MF input kappa=1e+05       | update_grad_inner | Muon/Adam               |  0.3266 | [0.3035, 0.3515] | Adam/GD-favorable     | After sweeping matched target update norms, does Muon's direction remain favorable?               |
| target_update_size       | Matrix sensing kappa=1e+02 | update_grad_inner | Muon/Adam               |  1.072  | [1.038, 1.107]   | Muon-favorable        | After sweeping matched target update norms, does Muon's direction remain favorable?               |
| target_update_size       | Matrix sensing kappa=1e+05 | update_grad_inner | Muon/Adam               |  1.113  | [1.039, 1.192]   | Muon-favorable        | After sweeping matched target update norms, does Muon's direction remain favorable?               |
| target_update_size       | Small MLP digits hidden=16 | update_grad_inner | Muon/Adam               |  1.009  | [0.9975, 1.021]  | mixed_or_uncertain    | After sweeping matched target update norms, does Muon's direction remain favorable?               |
| target_update_size       | Small MLP digits hidden=64 | update_grad_inner | Muon/Adam               |  0.6047 | [0.5818, 0.6285] | Adam/GD-favorable     | After sweeping matched target update norms, does Muon's direction remain favorable?               |
| per_layer_update_size    | Small MLP digits hidden=16 | update_grad_inner | Muon/Adam               |  0.9408 | [0.9309, 0.9507] | Adam/GD-favorable     | In MLP, after matching per-layer relative update norms, does Muon's direction remain favorable?   |
| per_layer_update_size    | Small MLP digits hidden=64 | update_grad_inner | Muon/Adam               |  0.6882 | [0.6761, 0.7006] | Adam/GD-favorable     | In MLP, after matching per-layer relative update norms, does Muon's direction remain favorable?   |
| norm_budget              | fro budget                 | update_grad_inner | flat_polar/GD_spectrum  |  0.6071 | [0.5846, 0.6304] | GD-spectrum-favorable | If only singular-value allocation changes, when is a flat/polar spectrum better than GD spectrum? |
| norm_budget              | op budget                  | update_grad_inner | flat_polar/GD_spectrum  |  1.689  | [1.614, 1.768]   | flat/polar-favorable  | If only singular-value allocation changes, when is a flat/polar spectrum better than GD spectrum? |
| state_specific_direction | All budget                 | update_grad_inner | other_update/own_update |  0.4925 | [0.4423, 0.5427] | own-update-favorable  | Does replacing the natural update by the other trajectory's update preserve local progress?       |

## Muon / Flat-Polar Favorable Cases

| boundary_axis            | condition                  | ratio_label             |   ratio | ci95             | direction            | interpretation                                                                                                         |
|:-------------------------|:---------------------------|:------------------------|--------:|:-----------------|:---------------------|:-----------------------------------------------------------------------------------------------------------------------|
| task_family              | MatrixSensing              | Muon/Adam               |  1.236  | [1.215, 1.258]   | Muon-favorable       | Task family alone can flip whether Muon's update spectrum improves local first-order progress.                         |
| target_update_size       | Matrix sensing kappa=1e+02 | Muon/Adam               |  1.072  | [1.038, 1.107]   | Muon-favorable       | The sign persists across target update sizes in some settings, but not uniformly across problem geometry.              |
| target_update_size       | Matrix sensing kappa=1e+05 | Muon/Adam               |  1.113  | [1.039, 1.192]   | Muon-favorable       | The sign persists across target update sizes in some settings, but not uniformly across problem geometry.              |
| norm_budget              | op budget                  | flat_polar/GD_spectrum  |  1.689  | [1.614, 1.768]   | flat/polar-favorable | Flat/polar allocation is beneficial under operator-norm budget and harmful under Frobenius budget.                     |
| state_specific_direction | All budget                 | other_update/own_update |  0.4925 | [0.4423, 0.5427] | own-update-favorable | The optimizer's natural update vector is locally consequential, but the effect is still one-step and budget-dependent. |

## Adam / GD-Spectrum Favorable Cases

| boundary_axis         | condition                  | ratio_label            |   ratio | ci95             | direction             | interpretation                                                                                            |
|:----------------------|:---------------------------|:-----------------------|--------:|:-----------------|:----------------------|:----------------------------------------------------------------------------------------------------------|
| task_family           | MatrixFactorizationInput   | Muon/Adam              |  0.3432 | [0.3194, 0.3687] | Adam/GD-favorable     | Task family alone can flip whether Muon's update spectrum improves local first-order progress.            |
| task_family           | SmallMLPDigits             | Muon/Adam              |  0.5232 | [0.4909, 0.5576] | Adam/GD-favorable     | Task family alone can flip whether Muon's update spectrum improves local first-order progress.            |
| target_update_size    | MF input kappa=1e+02       | Muon/Adam              |  0.4143 | [0.3868, 0.4438] | Adam/GD-favorable     | The sign persists across target update sizes in some settings, but not uniformly across problem geometry. |
| target_update_size    | MF input kappa=1e+05       | Muon/Adam              |  0.3266 | [0.3035, 0.3515] | Adam/GD-favorable     | The sign persists across target update sizes in some settings, but not uniformly across problem geometry. |
| target_update_size    | Small MLP digits hidden=64 | Muon/Adam              |  0.6047 | [0.5818, 0.6285] | Adam/GD-favorable     | The sign persists across target update sizes in some settings, but not uniformly across problem geometry. |
| per_layer_update_size | Small MLP digits hidden=16 | Muon/Adam              |  0.9408 | [0.9309, 0.9507] | Adam/GD-favorable     | The MLP sign depends strongly on width even after per-layer update-size control.                          |
| per_layer_update_size | Small MLP digits hidden=64 | Muon/Adam              |  0.6882 | [0.6761, 0.7006] | Adam/GD-favorable     | The MLP sign depends strongly on width even after per-layer update-size control.                          |
| norm_budget           | fro budget                 | flat_polar/GD_spectrum |  0.6071 | [0.5846, 0.6304] | GD-spectrum-favorable | Flat/polar allocation is beneficial under operator-norm budget and harmful under Frobenius budget.        |

## Mixed Or Uncertain Cases

| boundary_axis      | condition                  | ratio_label   |   ratio | ci95            | direction          | interpretation                                                                                            |
|:-------------------|:---------------------------|:--------------|--------:|:----------------|:-------------------|:----------------------------------------------------------------------------------------------------------|
| target_update_size | Small MLP digits hidden=16 | Muon/Adam     |   1.009 | [0.9975, 1.021] | mixed_or_uncertain | The sign persists across target update sizes in some settings, but not uniformly across problem geometry. |

## Interpretation For The Research Question

Muon's robust cross-task effect is an update-spectrum intervention. The current boundary map says this intervention helps local progress when the task/layer/norm geometry makes a flat/polar positive descent update align well with the gradient. It hurts or becomes ambiguous when the same flat allocation spends update budget in directions that do not improve `<G,D>`.

The cleanest mechanistic statement is therefore:

> Muon is not simply "better because rank is higher"; it applies a flat/polar update-spectrum bias, and that bias is useful only when the local gradient geometry rewards operator-norm-like spectral spreading.

## Sources

- [mechanism boundary map](../results/e11_mechanism_boundary/mechanism_boundary_map.csv)
- [equal-update first-order pair summary](../results/e11_equal_update/first_order_pair_summary.csv)
- [target-update sweep summary](../results/e11_target_update_sweep/target_pair_summary.csv)
- [MLP per-layer control summary](../results/e11_mlp_per_layer_control/pair_summary.csv)
- [spectral allocation probe summary](../results/e11_spectral_allocation_probe/spectral_allocation_summary.csv)
- [natural update swap summary](../results/e11_natural_update_swap_probe/natural_update_swap_summary.csv)
