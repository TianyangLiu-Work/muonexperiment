# E11 Stateless Optimizer Trajectory Ablation

This generated note runs short trajectories using only stateless directions. At every step, `GD`, `FreshAdamSign`, and `PolarMuon` use the same target relative Frobenius update norm within each problem family.

## Main Takeaway

Across these short trajectories, PolarMuon keeps the direction-level update-spectrum signature: mean nrUpdate PolarMuon/GD=2.704 CI=[2.236, 3.269], mean stUpdate=7.443 CI=[6.378, 8.685].

But this does not convert into uniformly better optimization progress: total_decrease PolarMuon/GD=0.4656 CI=[0.3686, 0.5882], and final_loss PolarMuon/GD=1.012 CI=[1.008, 1.016].

This is the trajectory-level analogue of the one-step stateless direction ablation: **polar direction is sufficient for high-rank updates, but not sufficient for a general progress advantage under Frobenius-matched steps.**

## Polar vs GD Total Decrease By Family

| problem_family           | comparison   | metric         |   n_pairs |   geomean_ratio |   ratio_ci95_low |   ratio_ci95_high |   numerator_higher_rate |
|:-------------------------|:-------------|:---------------|----------:|----------------:|-----------------:|------------------:|------------------------:|
| MatrixFactorizationInput | PolarMuon/GD | total_decrease |        10 |          0.2047 |           0.1865 |            0.2247 |                       0 |
| MatrixSensing            | PolarMuon/GD | total_decrease |        10 |          0.8301 |           0.8283 |            0.832  |                       0 |
| SmallMLPDigits           | PolarMuon/GD | total_decrease |        10 |          0.5939 |           0.5001 |            0.7053 |                       0 |

## Polar Update Spectrum By Family

| problem_family           | comparison   | metric        |   n_pairs |   geomean_ratio |   ratio_ci95_low |   ratio_ci95_high |   numerator_higher_rate |
|:-------------------------|:-------------|:--------------|----------:|----------------:|-----------------:|------------------:|------------------------:|
| MatrixFactorizationInput | PolarMuon/GD | mean_nrUpdate |        10 |           4.009 |            3.622 |             4.438 |                       1 |
| MatrixFactorizationInput | PolarMuon/GD | mean_stUpdate |        10 |           4.92  |            4.84  |             5.001 |                       1 |
| MatrixSensing            | PolarMuon/GD | mean_nrUpdate |        10 |           1.466 |            1.459 |             1.473 |                       1 |
| MatrixSensing            | PolarMuon/GD | mean_stUpdate |        10 |           8.229 |            7.912 |             8.559 |                       1 |
| SmallMLPDigits           | PolarMuon/GD | mean_nrUpdate |        10 |           3.362 |            2.499 |             4.523 |                       1 |
| SmallMLPDigits           | PolarMuon/GD | mean_stUpdate |        10 |          10.18  |            7.195 |            14.41  |                       1 |

## Evidence

- [trajectory step rows](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_step_metrics.csv)
- [trajectory outcomes](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_outcomes.csv)
- [paired trajectory rows](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_pairs.csv)
- [summary table](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv)
- [ratio figure](../figures/e11_stateless_optimizer_trajectory/stateless_optimizer_trajectory_ratios.png)

## Caveat

This still does not model Adam's state or a retuned full optimizer comparison. It is a controlled trajectory-level mechanism probe.
