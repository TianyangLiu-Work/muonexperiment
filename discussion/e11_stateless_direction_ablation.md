# E11 Stateless Direction Ablation

This generated note tests whether the polar/Muon direction itself is the key spectral intervention. For each problem state, it evaluates three one-step candidate directions under the same gradient and the same global Frobenius update size:

| candidate | definition |
|:--|:--|
| GD | raw gradient direction, globally rescaled |
| FreshAdamSign | first-step Adam-style elementwise normalized direction, globally rescaled |
| PolarMuon | per-layer polar factor `U V^T` of the gradient, globally rescaled |

The states come from Init, Adam checkpoints, and Muon checkpoints. This separates the checkpoint geometry from the candidate direction.

## Main Takeaway

PolarMuon strongly increases update rank statistics even without momentum or optimizer state: nrUpdate PolarMuon/GD=2.779 CI=[2.668, 2.895], stUpdate=6.923 CI=[6.679, 7.177].

However, the same stateless polar direction is not globally better for one-step progress: update_grad_inner PolarMuon/GD=0.5032 CI=[0.4857, 0.5213]. FreshAdamSign/GD gives update_grad_inner=0.6703 CI=[0.6622, 0.6785].

This supports the paper framing: **polar spectrum shaping is a robust direction-level mechanism, but progress remains boundary-dependent.**

## Polar vs GD By Family

| problem_family           | comparison   | metric            |   n_pairs |   geomean_ratio |   ratio_ci95_low |   ratio_ci95_high |   numerator_higher_rate |
|:-------------------------|:-------------|:------------------|----------:|----------------:|-----------------:|------------------:|------------------------:|
| MatrixFactorizationInput | PolarMuon/GD | update_grad_inner |       210 |          0.3004 |           0.2908 |            0.3104 |                       0 |
| MatrixSensing            | PolarMuon/GD | update_grad_inner |       210 |          0.8202 |           0.8171 |            0.8233 |                       0 |
| SmallMLPDigits           | PolarMuon/GD | update_grad_inner |       210 |          0.517  |           0.5017 |            0.5328 |                       0 |

## Polar Update Spectrum By Family

| problem_family           | comparison   | metric   |   n_pairs |   geomean_ratio |   ratio_ci95_low |   ratio_ci95_high |   numerator_higher_rate |
|:-------------------------|:-------------|:---------|----------:|----------------:|-----------------:|------------------:|------------------------:|
| MatrixFactorizationInput | PolarMuon/GD | nrUpdate |       210 |           4.037 |            3.936 |             4.141 |                       1 |
| MatrixFactorizationInput | PolarMuon/GD | stUpdate |       210 |           4.882 |            4.857 |             4.908 |                       1 |
| MatrixSensing            | PolarMuon/GD | nrUpdate |       210 |           1.486 |            1.475 |             1.498 |                       1 |
| MatrixSensing            | PolarMuon/GD | stUpdate |       210 |           6.79  |            6.457 |             7.14  |                       1 |
| SmallMLPDigits           | PolarMuon/GD | nrUpdate |       210 |           3.577 |            3.374 |             3.792 |                       1 |
| SmallMLPDigits           | PolarMuon/GD | stUpdate |       210 |          10.01  |            9.368 |            10.7   |                       1 |

## Evidence

- [stateless direction rows](../results/e11_stateless_direction_ablation/stateless_direction_rows.csv)
- [paired direction rows](../results/e11_stateless_direction_ablation/stateless_direction_pairs.csv)
- [summary table](../results/e11_stateless_direction_ablation/stateless_direction_summary.csv)
- [ratio figure](../figures/e11_stateless_direction_ablation/stateless_direction_ratios.png)

## Caveat

This is a one-step stateless intervention. It isolates the direction-level spectral bias, but it does not replace natural optimizer trajectories, momentum/state ablations, or longer-horizon retuned training.
