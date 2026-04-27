"""
E02 — Matrix Factorization Benchmark: Muon vs SGD
===================================================
问题: 深度矩阵分解 W_L...W_1 ≈ X* 中，Muon(谱归一化) vs SGD 的收敛性能。
与 E01(MS) 的区别: 目标函数非凸、存在尺度等价类、鞍点丰富。

控制变量:
  - d = 50               矩阵维度
  - L ∈ {2, 3}          分解深度
  - r = 5               目标秩
  - lr = 0.01           学习率
  - 2000 迭代, 10 个随机种子

输出:
  - results/E02_results.csv
  - 终端输出: K_epsilon, 配对检验, 效应量, FLOPs
"""
import os as _os
_os.environ['OMP_NUM_THREADS'] = '1'
_os.environ['MKL_NUM_THREADS'] = '1'
_os.environ['OPENBLAS_NUM_THREADS'] = '1'
_os.environ['NUMEXPR_NUM_THREADS'] = '1'
_os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys, csv, time
from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT = Path(_os.environ.get("MUON_PROJECT", "/data/home/tyliu/muonexperiment"))
sys.path.insert(0, str(PROJECT))
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

from muonlib.algorithms import (MuonOptimizer, SGDOptimizer)
from muonlib.data import (generate_target_matrix, compute_gradient_mf,
                           compute_loss_mf)

# ═══════════════════════════════════════════════════════
# 实验参数
# ═══════════════════════════════════════════════════════
ALGOS       = ["Muon-Exact", "SGD"]
D           = 50
LAYERS      = [2, 3]
R           = 5
LR          = 0.01
INIT_SCALE  = 0.01
ITERS       = 2000
SEEDS       = list(range(10))
EPSILON = 0.01

def make_opt(name):
    opts = {
        "Muon-Exact":   lambda: MuonOptimizer(variant="exact", mu=0.9),
        "SGD":          lambda: SGDOptimizer(momentum=0.9),
    }
    return opts[name]()

# ═══════════════════════════════════════════════════════
# 单次运行: Matrix Factorization
# ═══════════════════════════════════════════════════════
def run_mf(algo, d, L, r, lr, init_scale, seed, iters):
    """运行一次深度矩阵分解实验。

    优化: min_{W_1,...,W_L} 0.5 * ||W_L...W_1 - X*||_F^2
    每层 W_i 是 d×d 方阵。
    """
    rng = np.random.RandomState(seed)
    X_star = generate_target_matrix(d, r=r, seed=seed)

    # 初始化各层
    W = [rng.randn(d, d) * init_scale for _ in range(L)]
    opts = [make_opt(algo) for _ in range(L)]

    losses = []
    t_start = time.time()
    k_epsilon = -1

    for step in range(iters):
        # 计算各层梯度
        grads = compute_gradient_mf(W, X_star)

        # 更新各层
        for i in range(L):
            W[i] = opts[i].step(W[i], grads[i], lr)

        loss = float(compute_loss_mf(W, X_star))
        losses.append(loss)

        if k_epsilon < 0 and loss <= EPSILON:
            k_epsilon = step + 1

    elapsed = time.time() - t_start
    if k_epsilon < 0:
        k_epsilon = iters + 1

    return {
        "algo": algo, "d": d, "L": L, "r": r, "lr": lr,
        "init_scale": init_scale, "seed": seed, "iters": iters,
        "final_loss": losses[-1], "min_loss": min(losses),
        "K_epsilon": k_epsilon, "time_s": elapsed
    }

def save_csv(eid, rows):
    if not rows:
        return
    fname = RESULTS / f"{eid}_results.csv"
    write_header = not fname.exists()
    with open(fname, "a", newline="") as f:
        from muonlib.metrics import enrich_result_row
        rows = [enrich_result_row(r) for r in rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {eid}: {len(rows)} rows → {fname}")

# ═══════════════════════════════════════════════════════
# 主实验循环
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    total = len(ALGOS) * len(LAYERS) * len(SEEDS)
    print(f"E02: {len(ALGOS)} algos × {len(LAYERS)} layers × {len(SEEDS)} seeds = {total} runs")
    print(f"    d={D}, r={R}, lr={LR}, iters={ITERS}")
    print()
    rows = []
    idx = 0
    for algo in tqdm(ALGOS, desc="E02 algo"):
        for L in LAYERS:
            for seed in SEEDS:
                idx += 1
                print(f"[E02 {idx}/{total}] {algo} L={L} seed={seed}", flush=True)
                row = run_mf(algo, D, L, R, LR, INIT_SCALE, seed, ITERS)
                rows.append(row)

    save_csv("E02", rows)

    # ── 统计输出 ──────────────────────────────────────
    from muonlib.metrics import paired_ttest, wilcoxon_test, compute_effect_size

    print(f"\n{'='*60}")
    print("E02 统计汇总: K_epsilon (到达 eps=1e-10 的迭代数)")
    print(f"{'='*60}")
    for algo in ALGOS:
        for L in LAYERS:
            subset = [r for r in rows if r["algo"] == algo and r["L"] == L]
            n = len(subset)
            avg_k = np.mean([r["K_epsilon"] for r in subset])
            avg_loss = np.mean([r["final_loss"] for r in subset])
            avg_time = np.mean([r["time_s"] for r in subset])
            print(f"  {algo:<14s} L={L}  n={n}  avg_K={avg_k:>8.1f}  "
                  f"avg_loss={avg_loss:>10.2e}  avg_t={avg_time:.2f}s")

    print(f"\n{'='*60}")
    print("配对检验: Muon-Exact vs SGD (K_epsilon)")
    print(f"{'='*60}")
    for L in LAYERS:
        muon_k = [r["K_epsilon"] for r in rows if r["algo"] == "Muon-Exact" and r["L"] == L]
        sgd_k  = [r["K_epsilon"] for r in rows if r["algo"] == "SGD" and r["L"] == L]
        if len(muon_k) == len(sgd_k) and len(muon_k) > 1:
            t_stat, p_val = paired_ttest(muon_k, sgd_k)
            try:
                _, w_p = wilcoxon_test(muon_k, sgd_k)
            except:
                w_p = 1.0
            d_effect = compute_effect_size(muon_k, sgd_k, paired=True)
            sign = "✓ SIG" if p_val < 0.05 else "✗ n.s."
            print(f"  L={L}: t={t_stat:+.3f} p={p_val:.4f} {sign}  "
                  f"Wilcoxon p={w_p:.4f}  Cohen's d={d_effect:+.2f}")
        else:
            print(f"  L={L}: insufficient data")

    print(f"\nE02 完成")
