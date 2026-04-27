"""
E17 — Init Type Comparison: random vs orthogonal vs spectral
=============================================================
问题: 不同初始化策略下 Muon vs SGD 的表现差异。
"""
import os as _os
_os.environ['OMP_NUM_THREADS'] = '1'
_os.environ['MKL_NUM_THREADS'] = '1'
_os.environ['OPENBLAS_NUM_THREADS'] = '1'
_os.environ['NUMEXPR_NUM_THREADS'] = '1'
_os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import sys, csv, time; from pathlib import Path; import numpy as np
from numpy.linalg import svd
from tqdm import tqdm
PROJECT = Path(_os.environ.get("MUON_PROJECT", "/data/home/tyliu/muonexperiment"))
sys.path.insert(0, str(PROJECT))
RESULTS = PROJECT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
from muonlib.algorithms import MuonOptimizer, SGDOptimizer

# ── Detailed logging (v2) ───────────────────────────
from muonlib.detailed_logger import DetailedLogger
LOG_DIR = PROJECT / "logs_v2"
RESULTS_V2 = PROJECT / "results_v2"
RESULTS_V2.mkdir(parents=True, exist_ok=True)
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import paired_ttest, compute_effect_size

ALGOS       = ["Muon-Exact", "SGD"]
D, R, LR    = 50, 5, 0.01
INIT_TYPES  = ["random", "orthogonal", "spectral"]
ITERS       = 2000
SEEDS       = list(range(8))
EPSILON = 0.01

def init_matrix(d, init_type, rng):
    if init_type == "random":
        return rng.randn(d, d) * 0.01
    elif init_type == "orthogonal":
        Q, _ = np.linalg.qr(rng.randn(d, d))
        return Q * 0.01
    elif init_type == "spectral":
        U, _ = np.linalg.qr(rng.randn(d, d))
        V, _ = np.linalg.qr(rng.randn(d, d))
        return (U @ V.T) * 0.01
    return rng.randn(d, d) * 0.01

def make_opt(name):
    opts = {"Muon-Exact": lambda: MuonOptimizer(variant="exact", mu=0.9),
            "SGD": lambda: SGDOptimizer(momentum=0.9)}
    return opts[name]()

def run_ms(algo, d, r, lr, init_type, seed, iters):

    # ── Detailed logger (v2) ─────────────────────────
    logger = DetailedLogger(LOG_DIR, "E17_detailed", algo, {k: v for k, v in locals().items()
                             if k in ("d", "seed", "lr", "r", "noise", "dist",
                                      "spectrum", "kappa", "init_scale", "iters",
                                      "L", "m", "n", "wd", "gamma", "p", "q")}, svd_interval=10)
    rng = np.random.RandomState(seed)
    X_star = generate_target_matrix(d, r=r, seed=seed)
    m = int(2 * d * r)
    A = generate_measurement_matrices(d, m, dist="normal", seed=seed+1000)
    y = np.array([np.trace(Ai.T @ X_star) for Ai in A])
    X = init_matrix(d, init_type, rng)
    opt = make_opt(algo)
    losses = []; k_epsilon = -1; t0 = time.time()
    for step in range(iters):
        G = compute_gradient_matrix_sensing(X, A, y)
        X = opt.step(X, G, lr)
        loss = float(compute_loss_matrix_sensing(X, A, y))
        losses.append(loss)
                # Detailed logging (v2)
        grad_norm = float(np.linalg.norm(G, 'fro'))
        grad_max = float(np.max(np.abs(G)))
        X_norm = float(np.linalg.norm(X, 'fro'))
        extra = {"grad_max": grad_max, "X_norm": X_norm}
        if hasattr(opt, "momentum") and opt.momentum is not None:
            extra["momentum_norm"] = float(np.linalg.norm(opt.momentum, 'fro'))
        if algo.startswith("Muon") and step % 10 == 0:
            U_svd, s_svd, Vt_svd = svd(G, full_matrices=False)
            D = U_svd @ Vt_svd
            sv_log = s_svd
            extra["update_norm"] = float(np.linalg.norm(D, 'fro'))
        else:
            sv_log = None
        logger.log_step(step, loss, grad_norm=grad_norm, sv=sv_log, **extra)
        if k_epsilon < 0 and loss <= EPSILON: k_epsilon = step + 1
    if k_epsilon < 0: k_epsilon = iters + 1
    # ── Close detailed log ───────────────────────────
    logger.close()

    return {"algo": algo, "d": d, "r": r, "lr": lr, "init_type": init_type,
            "seed": seed, "iters": iters, "final_loss": losses[-1],
            "min_loss": min(losses), "K_epsilon": k_epsilon, "time_s": time.time() - t0}

def save_csv(eid, rows):
    if not rows: return
    fname = RESULTS_V2 / f"{eid}_results.csv"
    wh = not fname.exists()
    with open(fname, "a", newline="") as f:
        from muonlib.metrics import enrich_result_row
        rows = [enrich_result_row(r) for r in rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if wh: w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Saved {eid}: {len(rows)} rows")

if __name__ == "__main__":
    total = len(ALGOS) * len(INIT_TYPES) * len(SEEDS)
    print(f"E17: {len(ALGOS)} algos x {len(INIT_TYPES)} inits x {len(SEEDS)} seeds = {total}")
    rows = []; idx = 0
    for algo in tqdm(ALGOS, desc="E17 algo"):
        for it in INIT_TYPES:
            for seed in SEEDS:
                idx += 1
                print(f"[E17 {idx}/{total}] {algo} init={it} seed={seed}", flush=True)
                rows.append(run_ms(algo, D, R, LR, it, seed, ITERS))
    save_csv("E17_detailed", rows)

    print(f"\n{'='*60}")
    print(f"{'algo':<14s} {'init':>12s} {'avg_K':>8s} {'avg_loss':>12s}")
    print("-"*50)
    for algo in ALGOS:
        for it in INIT_TYPES:
            sub = [r for r in rows if r["algo"]==algo and r["init_type"]==it]
            print(f"{algo:<14s} {it:>12s} {np.mean([r['K_epsilon'] for r in sub]):>8.1f} {np.mean([r['final_loss'] for r in sub]):>12.2e}")
    print(f"\nE17 完成")
