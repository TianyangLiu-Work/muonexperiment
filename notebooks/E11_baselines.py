"""
E11 — Algorithm baseline comparison: Muon vs SGD vs Adam vs RMSprop vs Momentum-SGD
==================================================
MS only. ALGOS=["Muon-Exact","SGD","Adam","RMSprop","Momentum-SGD"]. D=50. SEEDS=10.
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
ALGOS       = ["Muon-Exact", "SGD", "Adam", "RMSprop", "Momentum-SGD"]
DIMS        = [50]
R           = 5
LR          = 0.01
NOISE       = 0.0
DIST        = "normal"
SPECTRUM    = "hard-cutoff"
KAPPA       = 1.0
INIT_SCALE  = 0.01
ITERS       = 2000
SEEDS       = list(range(10))
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
    total = len(ALGOS) * len(SEEDS)
    print(f"E11: {len(ALGOS)} algos x {len(SEEDS)} seeds = {total} runs")
    print()
    rows = []
    idx = 0
    d = DIMS[0]
    for algo in ALGOS:
        for seed in SEEDS:
            idx += 1
            print(f"[E11 {idx}/{total}] {algo} seed={seed}", flush=True)
            row = run_ms(algo, d, R, LR, NOISE, DIST, SPECTRUM, KAPPA,
                         INIT_SCALE, seed, ITERS)
            rows.append(row)

    save_csv("E11", rows)

    print(f"\n{'='*60}")
    print("E11 统计汇总")
    print(f"{'='*60}")
    print(f"\n{'algo':<16s} {'n':>4s} {'avg K_eps':>10s} {'avg loss':>12s} {'avg time_s':>10s}")
    print("-" * 60)
    for algo in ALGOS:
        subset = [r for r in rows if r["algo"] == algo]
        nn = len(subset)
        avg_k = np.mean([r["K_epsilon"] for r in subset])
        avg_loss = np.mean([r["final_loss"] for r in subset])
        avg_time = np.mean([r["time_s"] for r in subset])
        print(f"{algo:<16s} {nn:>4d} {avg_k:>10.1f} {avg_loss:>12.2e} {avg_time:>10.2f}")

    print(f"\n{'='*60}")
    print("配对检验 (K_epsilon vs Muon-Exact)")
    print(f"{'='*60}")
    muon_k = [r["K_epsilon"] for r in rows if r["algo"] == "Muon-Exact"]
    for algo in [a for a in ALGOS if a != "Muon-Exact"]:
        other_k = [r["K_epsilon"] for r in rows if r["algo"] == algo]
        if len(muon_k) == len(other_k) and len(muon_k) > 1:
            t_stat, p_val = paired_ttest(muon_k, other_k)
            d_effect = compute_effect_size(muon_k, other_k, paired=True)
            sign = "\u2713" if p_val < 0.05 else "\u2717"
            print(f"  Muon vs {algo:<16s}: t={t_stat:+.3f} p={p_val:.4f} {sign}  Cohen's d={d_effect:+.2f}")

    print(f"\n{'='*60}")
    print("FLOPs 对比")
    print(f"{'='*60}")
    d = DIMS[0]
    m = int(2 * d * R)
    for algo in ALGOS:
        algo_map = {"Muon-Exact": "Muon-Exact", "SGD": "SGD", "Adam": "SGD",
                    "RMSprop": "SGD", "Momentum-SGD": "SGD"}
        flops = compute_flops_matrix_sensing(d, m, ITERS, algo_map.get(algo, "SGD"))
        print(f"  {algo:<16s}: {flops:.2e} FLOPs")
    print(f"\n{'='*60}")
    print("E11 完成")
    print(f"{'='*60}")
