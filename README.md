# Muon Optimizer — Matrix Sensing Benchmark

## where is the result
If you don't want to verify the experiment code implementation ( or not yet), please go directly to the analysis_report folder

## Experiment Matrix

| ID  | Experiment          | Rows | Description                     |
|-----|---------------------|------|---------------------------------|
| E01 | MS Benchmark        | 60   | d=50-200, Dimension Scan        |
| E02 | MF Benchmark        | 40   | Matrix Factorization Scenario   |
| E03 | LR Calibration      | 100  | Learning Rate Calibration       |
| E04 | Init Noise          | 120  | Initialization Noise Robustness |
| E05 | FLOPs               | 8    | Theoretical FLOPs Calculation   |
| E06 | Observation Noise   | 80   | Observation Noise Robustness    |
| E07 | Rank Ratio          | 80   | Effect of Rank Ratio            |
| E08 | Oversampling        | 80   | Oversampling Factor             |
| E09 | Weight Decay        | 80   | Weight Decay                    |
| E10 | Rectangular         | 64   | Rectangular Matrices            |
| E11 | Baselines           | 50   | Adam/SGD/Muon/RMSprop           |
| E12 | Hessian             | 50   | Hessian Analysis                |
| E13 | Wallclock           | 40   | Per-step Timing                 |
| E14 | RandSVD             | 70   | Randomized SVD                  |
| E15 | Scalability         | 12   | d=100-200 Scalability           |
| E16 | Init Scale          | 80   | Initialization Scale            |
| E17 | Init Type           | 48   | Initialization Distribution     |
| E18 | Condition Number    | 64   | κ=10-10000                      |
| E19 | Spectrum Distribution | 80   | Spectral Distribution           |
| E20 | Power Scheduling    | 100  | 50-Seed Statistics              |

## Data Format

### `results_v3/E*_detailed_results.csv` — One Summary Row Per Run

| Column Name | Type | Description |
|------|------|------|
| `algo` | str | Algorithm: `SGD`, `Muon-Exact`, `Adam`, `Momentum-SGD`, `RMSprop`, `Muon-NS`, `Muon-RandSVD`, `Muon-Trunc` |
| `d` | int | Dimension of the square matrix |
| `r` | int | Rank of the target matrix |
| `lr` | float | Learning rate |
| `noise` | float | Standard deviation of observation noise (0 = no noise) |
| `dist` | str | Distribution of the measurement matrix: `normal`, `uniform`, `rademacher`, `sparse`, `sphere` |
| `spectrum` | str | Spectral type of the target matrix: `hard-cutoff`, `exponential-decay`, `polynomial-decay` |
| `kappa` | float | Condition number |
| `init_scale` | float | Standard deviation of initial weights |
| `seed` | int | Random seed |
| `iters` | int | Total number of iteration steps |
| `final_loss` | float | Loss at the final step |
| `min_loss` | float | **Minimum loss achieved during the run** (Core Metric) |
| `K_epsilon` | int | Iteration step at which ε=1e-10 was first reached; 2001 if not reached |
| `time_s` | float | Wall-clock runtime (seconds) |
| `I_conv` | int | Convergence flag (1 = ε reached) |
| `F_eps` | int | Total FLOPs accumulated when ε was reached; FLOPs corresponding to the maximum iterations if not reached |

`K_epsilon` and `min_loss` measure optimizer performance from different perspectives, and each holds independent significance.

---

`logs_v2/E*_detailed/*/trajectory.jsonl` — One trajectory entry per step

Each JSON line contains: `step`, `loss`, `grad_norm`, `singular_values` (list), `spectral_gap`, `iteration_time_s`. The trajectory logs are managed in Git LFS; for first-time use, please run `git lfs pull`.
