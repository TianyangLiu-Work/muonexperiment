"""
E10 — Rectangular matrices: Muon vs SGD
==================================================
MS on rectangular matrices. SHAPES=[(50,100),(100,50),(100,200),(200,100)]. SEEDS=8.
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

# ── Detailed logging (v2) ───────────────────────────
from muonlib.detailed_logger import DetailedLogger
LOG_DIR = PROJECT / "logs_v2"
RESULTS_V2 = PROJECT / "results_v3"
RESULTS_V2.mkdir(parents=True, exist_ok=True)
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           generate_rectangular_target, generate_rectangular_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import (compute_flops_matrix_sensing, paired_ttest,
                              wilcoxon_test, compute_effect_size, compute_statistics)
ALGOS       = ["Muon-Exact", "SGD"]
SHAPES      = [(50, 100), (100, 50), (100, 200), (200, 100)]
LR          = 0.01
NOISE       = 0.0
DIST        = "normal"
INIT_SCALE  = 0.01
ITERS       = 2000
SEEDS       = list(range(8))
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

def run_ms_rect(algo, m, n, lr, noise, dist, init_scale, seed, iters):

    # ── Detailed logger (v2) ─────────────────────────
    logger = DetailedLogger(LOG_DIR, "E10_detailed", algo, {k: v for k, v in locals().items()
                             if k in ("d", "seed", "lr", "r", "noise", "dist",
                                      "spectrum", "kappa", "init_scale", "iters",
                                      "L", "m", "n", "wd", "gamma", "p", "q")})
    r = min(m, n) // 10
    X_star = generate_rectangular_target(m, n, r=r, seed=seed)
    m_meas = int(2 * min(m, n) * r)
    A_matrices = generate_rectangular_measurement_matrices(m, n, m_meas, dist=dist, seed=seed+1000)
    y = np.array([np.trace(Ai.T @ X_star) for Ai in A_matrices])
    if noise > 0:
        y += np.random.RandomState(seed+2000).randn(m_meas) * noise

    X = np.random.RandomState(seed).randn(m, n) * init_scale
    opt = make_opt(algo)

    losses = []
    t_start = time.time()
    k_epsilon = -1

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
            k_epsilon = iters + 1
    finally:
        # ── Close detailed log ───────────────────────────
        logger.close()


    return {
        "algo": algo, "d": m, "r": r, "lr": lr, "noise": noise,
        "dist": dist, "init_scale": init_scale, "seed": seed, "iters": iters,
        "m": m, "n": n,
        "final_loss": losses[-1], "min_loss": min(losses),
        "K_epsilon": k_epsilon, "time_s": elapsed
    }

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
    print(f"Saved {eid}: {len(rows)} rows \u2192 {fname}")

if __name__ == "__main__":
    total = len(ALGOS) * len(SHAPES) * len(SEEDS)
    print(f"E10: {len(ALGOS)} algos x {len(SHAPES)} shapes x {len(SEEDS)} seeds = {total} runs")
    print()
    rows = []
    idx = 0
    for algo in ALGOS:
        for (m, n) in SHAPES:
            for seed in SEEDS:
                idx += 1
                print(f"[E10 {idx}/{total}] {algo} ({m}x{n}) seed={seed}", flush=True)
                row = run_ms_rect(algo, m, n, LR, NOISE, DIST, INIT_SCALE, seed, ITERS)
                rows.append(row)

    save_csv("E10_detailed", rows)

    print(f"\n{'='*60}")
    print("E10 统计汇总")
    print(f"{'='*60}")
    print(f"\n{'algo':<16s} {'shape':>12s} {'n':>4s} {'avg K_eps':>10s} {'avg loss':>12s} {'avg time_s':>10s}")
    print("-" * 70)
    for algo in ALGOS:
        for (m, n) in SHAPES:
            subset = [r for r in rows if r["algo"] == algo and r["m"] == m and r["n"] == n]
            nn = len(subset)
            if nn == 0: continue
            avg_k = np.mean([r["K_epsilon"] for r in subset])
            avg_loss = np.mean([r["final_loss"] for r in subset])
            avg_time = np.mean([r["time_s"] for r in subset])
            print(f"{algo:<16s} ({m:>4d},{n:>4d}) {nn:>4d} {avg_k:>10.1f} {avg_loss:>12.2e} {avg_time:>10.2f}")

    print(f"\n{'='*60}")
    print("配对检验: Muon-Exact vs SGD (K_epsilon)")
    print(f"{'='*60}")
    for (m, n) in SHAPES:
        muon_k = [r["K_epsilon"] for r in rows if r["algo"] == "Muon-Exact" and r["m"] == m and r["n"] == n]
        sgd_k  = [r["K_epsilon"] for r in rows if r["algo"] == "SGD" and r["m"] == m and r["n"] == n]
        if len(muon_k) == len(sgd_k) and len(muon_k) > 1:
            t_stat, p_val = paired_ttest(muon_k, sgd_k)
            d_effect = compute_effect_size(muon_k, sgd_k, paired=True)
            sign = "\u2713" if p_val < 0.05 else "\u2717"
            print(f"  ({m:>4d}x{n:<4d}): t={t_stat:+.3f} p={p_val:.4f} {sign}  Cohen's d={d_effect:+.2f}")
    print(f"\n{'='*60}")
    print("E10 完成")
    print(f"{'='*60}")
