# Muon Optimizer — Matrix Sensing Benchmark

> 实验结果 v3 | 20/20 实验完成 | 2026-04-27

## 实验矩阵

| 编号 | 实验 | 行数 | 描述 |
|------|------|------|------|
| E01 | MS Benchmark | 60 | d=50-200, 维度扫描 |
| E02 | MF Benchmark | 40 | 矩阵分解场景 |
| E03 | LR Calibration | 100 | 学习率标定 |
| E04 | Init Noise | 120 | 初始化噪声鲁棒性 |
| E05 | FLOPs | 8 | 理论 FLOPs 计算 |
| E06 | Observation Noise | 80 | 观测噪声鲁棒性 |
| E07 | Rank Ratio | 80 | 秩比率影响 |
| E08 | Oversampling | 80 | 过采样因子 |
| E09 | Weight Decay | 80 | 权重衰减 |
| E10 | Rectangular | 64 | 矩形矩阵 |
| E11 | Baselines | 50 | Adam/SGD/Muon/RMSprop |
| E12 | Hessian | 50 | Hessian 分析 |
| E13 | Wallclock | 40 | 每步计时 |
| E14 | RandSVD | 70 | 随机化 SVD |
| E15 | Scalability | 12 | d=100-200 扩展性 |
| E16 | Init Scale | 80 | 初始化尺度 |
| E17 | Init Type | 48 | 初始化分布 |
| E18 | Condition Number | 64 | κ=10-10000 |
| E19 | Spectrum Distribution | 80 | 谱分布 |
| E20 | Power Scheduling | 100 | 50 种子统计 |

## 数据格式

### `results_v3/E*_detailed_results.csv` — 每 run 一行汇总

| 列名 | 类型 | 说明 |
|------|------|------|
| `algo` | str | 算法：`SGD`, `Muon-Exact`, `Adam`, `Momentum-SGD`, `RMSprop`, `Muon-NS`, `Muon-RandSVD`, `Muon-Trunc` |
| `d` | int | 方阵维度 |
| `r` | int | 目标矩阵秩 |
| `lr` | float | 学习率 |
| `noise` | float | 观测噪声标准差 (0 = 无噪声) |
| `dist` | str | 测量矩阵分布：`normal`, `uniform`, `rademacher`, `sparse`, `sphere` |
| `spectrum` | str | 目标矩阵谱类型：`hard-cutoff`, `exponential-decay`, `polynomial-decay` |
| `kappa` | float | 条件数 |
| `init_scale` | float | 初始权重标准差 |
| `seed` | int | 随机种子 |
| `iters` | int | 总迭代步数 |
| `final_loss` | float | 最后一步的损失 |
| `min_loss` | float | **运行中达到的最小损失**（核心指标） |
| `K_epsilon` | int | 首次达到 ε=1e-10 的迭代步；若未达到则为 2001 |
| `time_s` | float | 墙钟运行时间 (秒) |
| `I_conv` | int | 收敛标志 (1=达到 ε) |
| `F_eps` | int | 达到 ε 时的总 FLOPs；若未达到则为最大迭代对应的 FLOPs |

`K_epsilon` 和 `min_loss` 从不同维度衡量优化器性能，两者各有独立意义。

### `logs_v2/E*_detailed/*/trajectory.jsonl` — 每步一行轨迹

每行 JSON 包含：`step`, `loss`, `grad_norm`, `singular_values` (列表), `spectral_gap`, `iteration_time_s`。

轨迹日志在 Git LFS 中管理，首次使用需 `git lfs pull`。
