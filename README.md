# Muon Experiment

> 矩阵优化中 **Muon（谱归一化）vs SGD** 的严格受控实验框架。

## 一句话

Muon 把梯度做 SVD 后丢掉奇异值只保留旋转方向 `U·V^T`。这个项目系统性地回答：**在各种条件下，这个操作到底有没有用？**

## 项目结构

```
muonexperiment/
├── notebooks/          # 原始实验（19 个 notebook，E01-E20）
├── notebooks_v2/       # v2 实验（每步详细 log 过程数据）
├── muonlib/            # 共享库（优化器、数据生成、指标、日志）
├── results/            # 原始实验结果 CSV
├── results_v2/         # v2 实验结果 CSV
├── logs/               # 原始实验日志
├── logs_v2/            # v2 详细轨迹日志（每步 loss/grad_norm/SVD）
├── docs/               # 设计文档（task.md, ENGINEERING_COMPROMISES.md）
├── results_backup/     # 历史结果备份
└── ANALYSIS.md         # 分析笔记
```

## 实验矩阵

| ID | 实验 | 核心问题 |
|----|------|---------|
| E01 | Matrix Sensing Benchmark | Muon vs SGD 基础性能 |
| E02 | Matrix Factorization | 深度分解中非凸优化的表现 |
| E03 | LR Calibration | 不同学习率下的公平对比 |
| E04 | Init + Noise | 初始化和噪声的敏感性 |
| E05 | FLOPs | 纯计算量分析（无训练） |
| E06 | Noise | 噪声鲁棒性扫描 |
| E07 | Rank Ratio | 不同秩比的影响 |
| E08 | Oversampling | 测量数对收敛的影响 |
| E09 | Weight Decay | 权重衰减正则化效果 |
| E10 | Rectangular | 非方阵上的表现 |
| E11 | Baselines | Adam/RMSprop/Momentum-SGD 对比 |
| E12 | Hessian | 曲率 landscape 分析 |
| E13 | Wallclock | 真实耗时对比 |
| E14 | RandSVD | 随机 SVD 精度-效率权衡 |
| E15 | Scalability | 大维度下的扩展性（d=100-200） |
| E16 | Init Scale | 初始化尺度敏感度 |
| E17 | Init Type | 初始化分布类型 |
| E18 | Condition | 条件数控制 |
| E19 | Distribution | 测量矩阵分布泛化 |
| E20 | Power | 统计功效和样本量 |

## 快速开始

```bash
# 克隆
git clone https://github.com/TianyangLiu-Work/muonexperiment.git
cd muonexperiment

# 运行单个实验（Matrix Sensing）
cd notebooks
OMP_NUM_THREADS=1 python3 E01_ms_benchmark.py

# 运行 v2 实验（详细日志）
cd notebooks_v2
OMP_NUM_THREADS=1 python3 E01_ms_benchmark_detailed.py
```

**重要**: 必须在 `import numpy` 之前通过 `os.environ` 设置 `OMP_NUM_THREADS=1`，否则 numpy BLAS 会静默多线程导致 CPU 争抢。

## Muon 核心公式

```python
G = U · Σ · V^T        # SVD of gradient
D = U · V^T             # all σ → 1, pure rotation
X_new = X - η·D - η·λ·X # update + decoupled weight decay
```

详见 `docs/task.md` §2.2.3 和 `muonlib/algorithms.py`。

## v2 详细日志

v2 实验每步记录到 `logs_v2/{实验ID}/{algo}_d{D}_seed{SEED}/trajectory.jsonl`：

```json
{"step": 0, "loss": 123.4, "elapsed_s": 0.001, "grad_norm": 56.7,
 "singular_values": [12.3, 8.9, ...], "grad_max": 5.2, "X_norm": 1.3}
```

包含：loss、梯度范数、梯度最大绝对值、参数范数、动量范数（如有）、SVD 奇异值（Muon）、Muon 更新范数。

## 结果

所有实验结果在 `results/` 和 `results_v2/` 下，每个 CSV 包含 K_epsilon（首次达到 ε 的迭代步）、final_loss、min_loss、耗时等。

## 许可

MIT
