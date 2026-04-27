"""
E01 — Matrix Sensing Benchmark: Muon vs SGD
=============================================
问题: 矩阵感知问题中，Muon(谱归一化) vs SGD(动量0.9) 的收敛性能对比。

控制变量:
  - d ∈ {50, 60, 70}    矩阵维度 (详见 ENGINEERING_COMPROMISES.md)
  - r = 5               目标秩
  - m = 2*d*r           测量数 (matching RIP bound)
  - lr = 0.01           学习率
  - noise = 0           无噪声
  - 2000 迭代, 10 个随机种子

输出:
  - results/E01_results.csv  所有运行结果
  - 终端输出: 按算法汇总的 K_epsilon, FLOPs, 配对t检验, 效应量
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

# ── Project setup ──────────────────────────────────────
PROJECT = Path(_os.environ.get("MUON_PROJECT", "/data/home/tyliu/muonexperiment"))
sys.path.insert(0, str(PROJECT))
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

from muonlib.algorithms import (MuonOptimizer, SGDOptimizer, AdamOptimizer,
                                 RMSpropOptimizer, MomentumSGDOptimizer)

# ── Detailed logging (v2) ───────────────────────────
from muonlib.detailed_logger import DetailedLogger
LOG_DIR = PROJECT / "logs_v2"
RESULTS_V2 = PROJECT / "results_v3"
RESULTS_V2.mkdir(parents=True, exist_ok=True)
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import (compute_flops_matrix_sensing, paired_ttest,
                               wilcoxon_test, compute_effect_size)

# ═══════════════════════════════════════════════════════
# 实验参数 — 在此修改
# ═══════════════════════════════════════════════════════
ALGOS       = ["Muon-Exact", "SGD"]
DIMS        = [50, 60, 70]
R           = 5
LR          = 0.01
NOISE       = 0.0
DIST        = "normal"
SPECTRUM    = "hard-cutoff"
KAPPA       = 1.0
INIT_SCALE  = 0.01
ITERS       = 2000
SEEDS       = list(range(10))
EPSILON = 0.01          # convergence threshold for K_epsilon

# ═══════════════════════════════════════════════════════
# 优化器工厂
# ═══════════════════════════════════════════════════════
def make_opt(name):
    """创建优化器实例。"""
    opts = {
        "Muon-Exact":    lambda: MuonOptimizer(variant="exact", mu=0.9, nesterov=False),
        "Muon-RandSVD":  lambda: MuonOptimizer(variant="randsvd", p=10, q=2),
        "Muon-Trunc":    lambda: MuonOptimizer(variant="truncated", r=5, mu=0.9),
        "SGD":           lambda: SGDOptimizer(momentum=0.9),
        "Adam":          lambda: AdamOptimizer(),
        "RMSprop":       lambda: RMSpropOptimizer(),
        "Momentum-SGD":  lambda: MomentumSGDOptimizer(mu=0.9),
    }
    return opts[name]()

# ═══════════════════════════════════════════════════════
# 单次运行: Matrix Sensing
# ═══════════════════════════════════════════════════════
def run_ms(algo, d, r, lr, noise, dist, spectrum, kappa, init_scale, seed, iters):

    # ── Detailed logger (v2) ─────────────────────────
    logger = DetailedLogger(LOG_DIR, "E01_detailed", algo, {k: v for k, v in locals().items()
                             if k in ("d", "seed", "lr", "r", "noise", "dist",
                                      "spectrum", "kappa", "init_scale", "iters",
                                      "L", "m", "n", "wd", "gamma", "p", "q")})
    """运行一次 Matrix Sensing 实验，返回结果字典。"""
    # 数据生成
    X_star = generate_target_matrix(d, r=r, spectrum=spectrum, kappa=kappa, seed=seed)
    m = int(2 * d * r)
    A_matrices = generate_measurement_matrices(d, m, dist=dist, seed=seed+1000)
    y = np.array([np.trace(Ai.T @ X_star) for Ai in A_matrices])
    if noise > 0:
        y += np.random.RandomState(seed+2000).randn(m) * noise

    # 初始化
    X = np.random.RandomState(seed).randn(d, d) * init_scale
    opt = make_opt(algo)

    losses = []
    t_start = time.time()
    k_epsilon = -1  # 首次达到 epsilon 的迭代步

    try:
        for step in range(iters):
            loss = float(compute_loss_matrix_sensing(X, A_matrices, y))
            losses.append(loss)
            G = compute_gradient_matrix_sensing(X, A_matrices, y)
            X = opt.step(X, G, lr)
            # Detailed logging (v2)
            grad_norm = float(np.linalg.norm(G, 'fro'))
            grad_max = float(np.max(np.abs(G)))
            X_norm = float(np.linalg.norm(X, 'fro'))
            extra = {"grad_max": grad_max, "X_norm": X_norm}
            if hasattr(opt, "momentum") and opt.momentum is not None:
                extra["momentum_norm"] = float(np.linalg.norm(opt.momentum, 'fro'))
            if algo.startswith("Muon") and step % 10 == 0:
                sv_log = opt._last_singular_values
                extra["update_norm"] = opt._last_update_norm
            else:
                sv_log = None
            logger.log_step(step, loss, grad_norm=grad_norm, sv=sv_log, **extra)

            if k_epsilon < 0 and loss <= EPSILON:
                k_epsilon = step + 1

        elapsed = time.time() - t_start
        if k_epsilon < 0:
            k_epsilon = iters + 1  # 未收敛
    finally:
        # ── Close detailed log ───────────────────────────
        logger.close()


    return {
        "algo": algo, "d": d, "r": r, "lr": lr, "noise": noise,
        "dist": dist, "spectrum": spectrum, "kappa": kappa,
        "init_scale": init_scale, "seed": seed, "iters": iters,
        "final_loss": losses[-1], "min_loss": min(losses),
        "K_epsilon": k_epsilon, "time_s": elapsed
    }

# ═══════════════════════════════════════════════════════
# 保存 CSV
# ═══════════════════════════════════════════════════════
def save_csv(eid, rows):
    fname = RESULTS_V2 / f"{eid}_results.csv"
    if not rows:
        return
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
    total = len(ALGOS) * len(DIMS) * len(SEEDS)
    print(f"E01: {len(ALGOS)} algos × {len(DIMS)} dims × {len(SEEDS)} seeds = {total} runs")
    print(f"    epsilon={EPSILON}, iters={ITERS}, lr={LR}")
    print()
    rows = []
    idx = 0
    for algo in tqdm(ALGOS, desc="E01 algo"):
        for d in DIMS:
            for seed in SEEDS:
                idx += 1
                print(f"[E01 {idx}/{total}] {algo} d={d} seed={seed}", flush=True)
                row = run_ms(algo, d, R, LR, NOISE, DIST, SPECTRUM, KAPPA,
                             INIT_SCALE, seed, ITERS)
                rows.append(row)

    save_csv("E01_detailed", rows)

    # ── 统计输出 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("E01 统计汇总")
    print(f"{'='*60}")

    # 按 (algo, d) 分组计算统计量
    print(f"\n{'algo':<16s} {'d':>4s} {'n':>4s} {'avg K_ε':>10s} {'avg loss':>12s} {'avg time_s':>10s}")
    print("-" * 60)
    for algo in ALGOS:
        for d in DIMS:
            subset = [r for r in rows if r["algo"] == algo and r["d"] == d]
            n = len(subset)
            avg_k = np.mean([r["K_epsilon"] for r in subset])
            avg_loss = np.mean([r["final_loss"] for r in subset])
            avg_time = np.mean([r["time_s"] for r in subset])
            print(f"{algo:<16s} {d:>4d} {n:>4d} {avg_k:>10.1f} {avg_loss:>12.2e} {avg_time:>10.2f}")

    # ── 配对检验 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("配对检验: Muon-Exact vs SGD (K_epsilon)")
    print(f"{'='*60}")
    for d in DIMS:
        muon_k = [r["K_epsilon"] for r in rows if r["algo"] == "Muon-Exact" and r["d"] == d]
        sgd_k  = [r["K_epsilon"] for r in rows if r["algo"] == "SGD" and r["d"] == d]
        if len(muon_k) == len(sgd_k) and len(muon_k) > 1:
            t_stat, p_val = paired_ttest(muon_k, sgd_k)
            try:
                w_stat, w_p = wilcoxon_test(muon_k, sgd_k)
            except:
                w_stat, w_p = 0, 1.0
            d_effect = compute_effect_size(muon_k, sgd_k, paired=True)
            sign = "✓" if p_val < 0.05 else "✗"
            print(f"  d={d:>3d}: t={t_stat:+.3f} p={p_val:.4f} {sign}  "
                  f"Wilcoxon p={w_p:.4f}  Cohen's d={d_effect:+.2f}")
        else:
            print(f"  d={d:>3d}: insufficient data for paired test")

    # ── FLOPs 对比 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("FLOPs 对比 (per docs §4.1)")
    print(f"{'='*60}")
    for d in DIMS:
        m = int(2 * d * R)
        sgd_flops = compute_flops_matrix_sensing(d, m, ITERS, "SGD")
        muon_flops = compute_flops_matrix_sensing(d, m, ITERS, "Muon-Exact")
        ratio = muon_flops / sgd_flops if sgd_flops > 0 else 0
        print(f"  d={d:>3d}: SGD={sgd_flops:.2e}  Muon={muon_flops:.2e}  ratio={ratio:.2f}")

    print(f"\n{'='*60}")
    print("E01 完成")
    print(f"{'='*60}")
