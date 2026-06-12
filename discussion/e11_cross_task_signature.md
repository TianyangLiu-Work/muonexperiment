# E11 Cross-Task Optimizer Signature

This generated note asks which candidate optimizer-level statements survive a simple cross-task screen across Matrix Factorization with input, Matrix Sensing, and Small MLP digits.

Screening rule: a candidate passes only if all three task families have the same Muon/Adam ratio direction and all three family-level 95% CIs support the expected direction. This is deliberately conservative.

## Passing Candidates

| source                        | metric              | expected_direction   | family_ratios                                                               | family_ci95                                                                                               |   families_support_expected | passes_cross_task_screen   |
|:------------------------------|:--------------------|:---------------------|:----------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|----------------------------:|:---------------------------|
| equal_update:state_volatility | loss_mean_rel_speed | below_one            | MatrixFactorizationInput=0.853; MatrixSensing=0.8532; SmallMLPDigits=0.3927 | MatrixFactorizationInput=[0.804, 0.9051]; MatrixSensing=[0.8135, 0.8948]; SmallMLPDigits=[0.3172, 0.4862] |                           3 | yes                        |
| equal_update:update_spectrum  | nrUpdate            | above_one            | MatrixFactorizationInput=2.8; MatrixSensing=1.405; SmallMLPDigits=2.39      | MatrixFactorizationInput=[2.648, 2.96]; MatrixSensing=[1.404, 1.406]; SmallMLPDigits=[2.335, 2.445]       |                           3 | yes                        |
| equal_update:update_spectrum  | nrUpdateFrac        | above_one            | MatrixFactorizationInput=2.8; MatrixSensing=1.405; SmallMLPDigits=1.674     | MatrixFactorizationInput=[2.648, 2.96]; MatrixSensing=[1.404, 1.406]; SmallMLPDigits=[1.632, 1.718]       |                           3 | yes                        |
| equal_update:update_spectrum  | stUpdate            | above_one            | MatrixFactorizationInput=4.335; MatrixSensing=4.525; SmallMLPDigits=9.903   | MatrixFactorizationInput=[4.206, 4.468]; MatrixSensing=[4.46, 4.592]; SmallMLPDigits=[9.383, 10.45]       |                           3 | yes                        |
| equal_update:update_spectrum  | stUpdateFrac        | above_one            | MatrixFactorizationInput=4.335; MatrixSensing=4.525; SmallMLPDigits=4.68    | MatrixFactorizationInput=[4.206, 4.468]; MatrixSensing=[4.46, 4.592]; SmallMLPDigits=[4.382, 4.998]       |                           3 | yes                        |
| equal_update:update_spectrum  | update_flatness     | above_one            | MatrixFactorizationInput=1.479; MatrixSensing=3.224; SmallMLPDigits=3.308   | MatrixFactorizationInput=[1.432, 1.528]; MatrixSensing=[3.179, 3.269]; SmallMLPDigits=[3.173, 3.447]      |                           3 | yes                        |

## Failed Or Mixed Candidates

| source                         | metric                           | expected_direction   | family_ratios                                                               | family_ci95                                                                                              |   families_support_expected |   families_oppose_expected | all_families_same_sign   |
|:-------------------------------|:---------------------------------|:---------------------|:----------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|----------------------------:|---------------------------:|:-------------------------|
| equal_update:one_step_progress | delta_loss                       | above_one            | MatrixFactorizationInput=0.3823; MatrixSensing=1.162; SmallMLPDigits=0.5128 | MatrixFactorizationInput=[0.3537, 0.4133]; MatrixSensing=[1.127, 1.197]; SmallMLPDigits=[0.48, 0.5478]   |                           1 |                          2 | no                       |
| equal_update:one_step_progress | update_grad_cosine               | above_one            | MatrixFactorizationInput=0.4758; MatrixSensing=1.335; SmallMLPDigits=0.7647 | MatrixFactorizationInput=[0.4543, 0.4983]; MatrixSensing=[1.311, 1.359]; SmallMLPDigits=[0.7578, 0.7716] |                           1 |                          2 | no                       |
| equal_update:one_step_progress | update_grad_inner                | above_one            | MatrixFactorizationInput=0.3432; MatrixSensing=1.236; SmallMLPDigits=0.5232 | MatrixFactorizationInput=[0.3194, 0.3687]; MatrixSensing=[1.215, 1.258]; SmallMLPDigits=[0.4909, 0.5576] |                           1 |                          2 | no                       |
| equal_update:state_volatility  | condition_score_std_speed        | below_one            | MatrixFactorizationInput=0.805; MatrixSensing=1.215; SmallMLPDigits=0.9612  | MatrixFactorizationInput=[0.7199, 0.9002]; MatrixSensing=[1.095, 1.349]; SmallMLPDigits=[0.8731, 1.058]  |                           1 |                          1 | no                       |
| equal_update:state_volatility  | norm_condition_mean_speed        | below_one            | MatrixFactorizationInput=0.7026; MatrixSensing=1.177; SmallMLPDigits=0.9854 | MatrixFactorizationInput=[0.6275, 0.7868]; MatrixSensing=[1.073, 1.29]; SmallMLPDigits=[0.9048, 1.073]   |                           1 |                          1 | no                       |
| equal_update:state_volatility  | norm_rank_plane_mean_speed       | below_one            | MatrixFactorizationInput=0.9904; MatrixSensing=1.177; SmallMLPDigits=0.9972 | MatrixFactorizationInput=[0.9039, 1.085]; MatrixSensing=[1.073, 1.29]; SmallMLPDigits=[0.9129, 1.089]    |                           0 |                          1 | no                       |
| equal_update:state_volatility  | per_update_condition_mean_speed  | below_one            | MatrixFactorizationInput=0.7046; MatrixSensing=1.179; SmallMLPDigits=0.9926 | MatrixFactorizationInput=[0.6293, 0.789]; MatrixSensing=[1.073, 1.295]; SmallMLPDigits=[0.906, 1.088]    |                           1 |                          1 | no                       |
| equal_update:state_volatility  | per_update_rank_plane_mean_speed | below_one            | MatrixFactorizationInput=0.9897; MatrixSensing=1.179; SmallMLPDigits=1.002  | MatrixFactorizationInput=[0.9022, 1.086]; MatrixSensing=[1.073, 1.295]; SmallMLPDigits=[0.9112, 1.101]   |                           0 |                          1 | no                       |

## Interpretation

The only candidates that cleanly pass this cross-task screen are update-spectrum quantities: `nrUpdate`, `stUpdate`, normalized rank fractions, and update flatness. This supports the narrow optimizer-intrinsic claim that Muon changes the singular-value geometry of update matrices.

State-trajectory volatility and one-step progress do not pass the same screen. Their signs or CI support change by task family. This means they should be presented as downstream interactions between optimizer and problem geometry, not as optimizer-intrinsic invariants.

## Implication For The Research Question

The cross-task invariant we can currently defend is:

> Muon consistently produces flatter, higher-rank update spectra than Adam at matched update size.

The mechanism question should then be:

> Under what task/layer/norm geometry does this update-spectrum shaping improve the positive descent proxy `<G,D>` and observed loss decrease?

## Sources

- [cross-task signature summary](../results/e11_cross_task_signature/cross_task_signature_summary.csv)
- [cross-task signature rows](../results/e11_cross_task_signature/cross_task_signature_rows.csv)
- [equal-update update-spectrum summary](../results/e11_equal_update/update_spectrum_summary.csv)
- [equal-update volatility summary](../results/e11_equal_update/volatility_summary.csv)
- [equal-update first-order pair summary](../results/e11_equal_update/first_order_pair_summary.csv)
