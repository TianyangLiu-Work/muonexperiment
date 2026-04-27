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

## 图表

### E01 Benchmark
![E01](plots_v3/E01_benchmark.png)

### 加速比
![Speedup](plots_v3/speedup_summary.png)

### 理论 FLOPs (E05)
![FLOPs](plots_v3/E05_flops.png)

### 敏感性分析
![Sensitivity](plots_v3/sensitivity.png)

## 数据格式

- `results_v3/E*_detailed_results.csv` — 每 run 一行汇总 (algo, d, final_loss, time_s, K_epsilon, ...)
- `logs_v2/E*_detailed/*/trajectory.jsonl` — 每步一行轨迹 (step, loss, singular_values, grad_norm, ...)
