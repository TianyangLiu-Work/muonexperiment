# E11 Cross-Task Optimizer Signature

This generated note asks which candidate optimizer-level statements survive a simple cross-task screen across Matrix Factorization with input, Matrix Sensing, and Small MLP digits.

Screening rule: a candidate passes only if all three task families have the same Muon/Adam ratio direction and all three family-level 95% CIs support the expected direction. This is deliberately conservative.

## Passing Candidates

| source                       | metric          | expected_direction   | family_ratios                                                             | family_ci95                                                                                          |   families_support_expected | passes_cross_task_screen   |
|:-----------------------------|:----------------|:---------------------|:--------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|----------------------------:|:---------------------------|
| equal_update:update_spectrum | nrUpdate        | above_one            | MatrixFactorizationInput=2.766; MatrixSensing=1.421; SmallMLPDigits=2.611 | MatrixFactorizationInput=[2.615, 2.926]; MatrixSensing=[1.419, 1.423]; SmallMLPDigits=[2.544, 2.679] |                           3 | yes                        |
| equal_update:update_spectrum | nrUpdateFrac    | above_one            | MatrixFactorizationInput=2.766; MatrixSensing=1.421; SmallMLPDigits=1.737 | MatrixFactorizationInput=[2.615, 2.926]; MatrixSensing=[1.419, 1.423]; SmallMLPDigits=[1.69, 1.784]  |                           3 | yes                        |
| equal_update:update_spectrum | stUpdate        | above_one            | MatrixFactorizationInput=4.344; MatrixSensing=5.468; SmallMLPDigits=10.26 | MatrixFactorizationInput=[4.223, 4.468]; MatrixSensing=[5.363, 5.575]; SmallMLPDigits=[9.704, 10.86] |                           3 | yes                        |
| equal_update:update_spectrum | stUpdateFrac    | above_one            | MatrixFactorizationInput=4.344; MatrixSensing=5.468; SmallMLPDigits=4.75  | MatrixFactorizationInput=[4.223, 4.468]; MatrixSensing=[5.363, 5.575]; SmallMLPDigits=[4.441, 5.082] |                           3 | yes                        |
| equal_update:update_spectrum | update_flatness | above_one            | MatrixFactorizationInput=1.503; MatrixSensing=3.85; SmallMLPDigits=3.266  | MatrixFactorizationInput=[1.454, 1.554]; MatrixSensing=[3.777, 3.924]; SmallMLPDigits=[3.136, 3.402] |                           3 | yes                        |

## Failed Or Mixed Candidates

| source                         | metric                           | expected_direction   | family_ratios                                                                | family_ci95                                                                                              |   families_support_expected |   families_oppose_expected | all_families_same_sign   |
|:-------------------------------|:---------------------------------|:---------------------|:-----------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|----------------------------:|---------------------------:|:-------------------------|
| equal_update:one_step_progress | delta_loss                       | above_one            | MatrixFactorizationInput=0.3979; MatrixSensing=0.9383; SmallMLPDigits=0.4849 | MatrixFactorizationInput=[0.3677, 0.4307]; MatrixSensing=[0.8857, 0.994]; SmallMLPDigits=[0.454, 0.5178] |                           0 |                          3 | yes                      |
| equal_update:one_step_progress | update_grad_cosine               | above_one            | MatrixFactorizationInput=0.4807; MatrixSensing=1.364; SmallMLPDigits=0.7418  | MatrixFactorizationInput=[0.46, 0.5023]; MatrixSensing=[1.265, 1.471]; SmallMLPDigits=[0.736, 0.7477]    |                           1 |                          2 | no                       |
| equal_update:one_step_progress | update_grad_inner                | above_one            | MatrixFactorizationInput=0.3606; MatrixSensing=1.209; SmallMLPDigits=0.4942  | MatrixFactorizationInput=[0.3361, 0.3869]; MatrixSensing=[1.148, 1.274]; SmallMLPDigits=[0.4639, 0.5266] |                           1 |                          2 | no                       |
| equal_update:state_volatility  | condition_score_std_speed        | below_one            | MatrixFactorizationInput=0.501; MatrixSensing=1.799; SmallMLPDigits=0.9641   | MatrixFactorizationInput=[0.4376, 0.5736]; MatrixSensing=[1.443, 2.243]; SmallMLPDigits=[0.8849, 1.05]   |                           1 |                          1 | no                       |
| equal_update:state_volatility  | loss_mean_rel_speed              | below_one            | MatrixFactorizationInput=0.327; MatrixSensing=1.034; SmallMLPDigits=0.3721   | MatrixFactorizationInput=[0.2783, 0.3841]; MatrixSensing=[1.018, 1.05]; SmallMLPDigits=[0.2989, 0.4632]  |                           2 |                          1 | no                       |
| equal_update:state_volatility  | norm_condition_mean_speed        | below_one            | MatrixFactorizationInput=0.4386; MatrixSensing=6.096; SmallMLPDigits=0.9911  | MatrixFactorizationInput=[0.3895, 0.494]; MatrixSensing=[5.106, 7.277]; SmallMLPDigits=[0.9192, 1.069]   |                           1 |                          1 | no                       |
| equal_update:state_volatility  | norm_rank_plane_mean_speed       | below_one            | MatrixFactorizationInput=0.5619; MatrixSensing=6.096; SmallMLPDigits=0.9922  | MatrixFactorizationInput=[0.5156, 0.6124]; MatrixSensing=[5.106, 7.277]; SmallMLPDigits=[0.9188, 1.071]  |                           1 |                          1 | no                       |
| equal_update:state_volatility  | per_update_condition_mean_speed  | below_one            | MatrixFactorizationInput=0.4407; MatrixSensing=6.101; SmallMLPDigits=0.9971  | MatrixFactorizationInput=[0.3911, 0.4967]; MatrixSensing=[5.116, 7.275]; SmallMLPDigits=[0.92, 1.081]    |                           1 |                          1 | no                       |
| equal_update:state_volatility  | per_update_rank_plane_mean_speed | below_one            | MatrixFactorizationInput=0.5633; MatrixSensing=6.101; SmallMLPDigits=0.998   | MatrixFactorizationInput=[0.5164, 0.6145]; MatrixSensing=[5.116, 7.275]; SmallMLPDigits=[0.9188, 1.084]  |                           1 |                          1 | no                       |

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
