"""
E03 — LR calibration: Muon vs SGD at different learning rates
==================================================
MS only. ALGOS=["Muon-Exact","SGD"]. DIMS=[50,70]. LRS=[0.001,0.003,0.01,0.03,0.1]. SEEDS=5.
"""
import os as _os
_os.environ['OMP_NUM_THREADS'] = '1'
_os.environ['MKL_NUM_THREADS'] = '1'
_os.environ['OPENBLAS_NUM_THREADS'] = '1'
_os.environ['NUMEXPR_NUM_THREADS'] = '1'
_os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys, csv, time, json
from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT = Path(_os.environ.get("MUON_PROJECT", "/data/home/tyliu/muonexperiment"))
sys.path.insert(0, str(PROJECT))
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
from muonlib.algorithms import (MuonOptimizer, SGDOptimizer, AdamOptimizer,
                                 RMSpropOptimizer, MomentumSGDOptimizer)
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import (compute_flops_matrix_sensing, paired_ttest,
                              wilcoxon_test, compute_effect_size, compute_statistics)
# ═══════════════════════════════════════════════════════
# 实验参数
# ═══════════════════════════════════════════════════════
ALGOS       = ["Muon-Exact", "SGD"]
DIMS        = [50, 70]
R           = 5
LRS         = [0.001, 0.003, 0.01, 0.03, 0.1]
NOISE       = 0.0
DIST        = "normal"
SPECTRUM    = "hard-cutoff"
KAPPA       = 1.0
INIT_SCALE  = 0.01
ITERS       = 2000
SEEDS       = list(range(5))
EPSILON = 0.01

def make_opt(name, weight_decay=0.0):
    """创建优化器实例。"""
    opts = {
        "Muon-Exact":    lambda: MuonOptimizer(variant="exact", mu=0.9, nesterov=False, weight_decay=weight_decay),
        "Muon-RandSVD":  lambda: MuonOptimizer(variant="randsvd", p=10, q=2, weight_decay=weight_decay),
        "Muon-Trunc":    lambda: MuonOptimizer(variant="truncated", r=5, mu=0.9, weight_decay=weight_decay),
        "SGD":           lambda: SGDOptimizer(momentum=0.9, weight_decay=weight_decay),
        "Adam":          lambda: AdamOptimizer(),
        "RMSprop":       lambda: RMSpropOptimizer(),
        "Momentum-SGD":  lambda: MomentumSGDOptimizer(mu=0.9, weight_decay=weight_decay),
    }
    return opts[name]()

def run_ms(algo, d, r, lr, noise, dist, spectrum, kappa, init_scale, seed, iters):
    """运行一次 Matrix Sensing 实验，返回结果字典。"""
    X_star = generate_target_matrix(d, r=r, spectrum=spectrum, kappa=kappa, seed=seed)
    m = int(2 * d * r)
    A_matrices = generate_measurement_matrices(d, m, dist=dist, seed=seed+1000)
    y = np.array([np.trace(Ai.T @ X_star) for Ai in A_matrices])
    if noise > 0:
        y += np.random.RandomState(seed+2000).randn(m) * noise

    X = np.random.RandomState(seed).randn(d, d) * init_scale
    opt = make_opt(algo)

    losses = []
    t_start = time.time()
    k_epsilon = -1

    for step in range(iters):
        G = compute_gradient_matrix_sensing(X, A_matrices, y)
        X = opt.step(X, G, lr)
        loss = float(compute_loss_matrix_sensing(X, A_matrices, y))
        losses.append(loss)

        if k_epsilon < 0 and loss <= EPSILON:
            k_epsilon = step + 1

    elapsed = time.time() - t_start
    if k_epsilon < 0:
        k_epsilon = iters + 1

    return {
        "algo": algo, "d": d, "r": r, "lr": lr, "noise": noise,
        "dist": dist, "spectrum": spectrum, "kappa": kappa,
        "init_scale": init_scale, "seed": seed, "iters": iters,
        "final_loss": losses[-1], "min_loss": min(losses),
        "K_epsilon": k_epsilon, "time_s": elapsed
    }

def save_csv(eid, rows):
    fname = RESULTS / f"{eid}_results.csv"
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
    print(f"Saved {eid}: {len(rows)} rows \u2192 {fname}")

if __name__ == "__main__":
    total = len(ALGOS) * len(DIMS) * len(LRS) * len(SEEDS)
    print(f"E03: {len(ALGOS)} algos x {len(DIMS)} dims x {len(LRS)} lrs x {len(SEEDS)} seeds = {total} runs")
    print()
    rows = []
    idx = 0
    for algo in ALGOS:
        for d in DIMS:
            for lr in LRS:
                for seed in SEEDS:
                    idx += 1
                    print(f"[E03 {idx}/{total}] {algo} d={d} lr={lr} seed={seed}", flush=True)
                    row = run_ms(algo, d, R, lr, NOISE, DIST, SPECTRUM, KAPPA,
                                 INIT_SCALE, seed, ITERS)
                    rows.append(row)

    save_csv("E03", rows)

    # ── 统计输出 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("E03 统计汇总")
    print(f"{'='*60}")

    print(f"\n{'algo':<16s} {'d':>4s} {'lr':>8s} {'n':>4s} {'avg K_eps':>10s} {'avg loss':>12s}")
    print("-" * 60)
    for algo in ALGOS:
        for d in DIMS:
            for lr in LRS:
                subset = [r for r in rows if r["algo"] == algo and r["d"] == d and abs(r["lr"] - lr) < 1e-9]
                n = len(subset)
                if n == 0: continue
                avg_k = np.mean([r["K_epsilon"] for r in subset])
                avg_loss = np.mean([r["final_loss"] for r in subset])
                print(f"{algo:<16s} {d:>4d} {lr:>8.4f} {n:>4d} {avg_k:>10.1f} {avg_loss:>12.2e}")

    print(f"\n{'='*60}")
    print("配对检验: Muon-Exact vs SGD (K_epsilon)")
    print(f"{'='*60}")
    for d in DIMS:
        for lr in LRS:
            muon_k = [r["K_epsilon"] for r in rows if r["algo"] == "Muon-Exact" and r["d"] == d and abs(r["lr"] - lr) < 1e-9]
            sgd_k  = [r["K_epsilon"] for r in rows if r["algo"] == "SGD" and r["d"] == d and abs(r["lr"] - lr) < 1e-9]
            if len(muon_k) == len(sgd_k) and len(muon_k) > 1:
                t_stat, p_val = paired_ttest(muon_k, sgd_k)
                d_effect = compute_effect_size(muon_k, sgd_k, paired=True)
                sign = "\u2713" if p_val < 0.05 else "\u2717"
                print(f"  d={d:>3d} lr={lr:.4f}: t={t_stat:+.3f} p={p_val:.4f} {sign}  Cohen's d={d_effect:+.2f}")

    print(f"\n{'='*60}")
    print("FLOPs 对比")
    print(f"{'='*60}")
    for d in DIMS:
        m = int(2 * d * R)
        sgd_flops = compute_flops_matrix_sensing(d, m, ITERS, "SGD")
        muon_flops = compute_flops_matrix_sensing(d, m, ITERS, "Muon-Exact")
        ratio = muon_flops / sgd_flops if sgd_flops > 0 else 0
        print(f"  d={d:>3d}: SGD={sgd_flops:.2e}  Muon={muon_flops:.2e}  ratio={ratio:.2f}")

    print(f"\n{'='*60}")
    print("E03 完成")
    print(f"{'='*60}")
