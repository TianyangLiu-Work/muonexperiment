"""
E18 — Condition Number Control: Muon vs SGD
=============================================
问题: 病态问题下 Muon 谱归一化是否比 SGD 更鲁棒？

扫描: kappa ∈ {10, 100, 1000, 10000}
"""
import os as _os
_os.environ['OMP_NUM_THREADS'] = '1'
_os.environ['MKL_NUM_THREADS'] = '1'
_os.environ['OPENBLAS_NUM_THREADS'] = '1'
_os.environ['NUMEXPR_NUM_THREADS'] = '1'
_os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import sys, csv, time; from pathlib import Path; import numpy as np
from tqdm import tqdm
PROJECT = Path(_os.environ.get("MUON_PROJECT", "/data/home/tyliu/muonexperiment"))
sys.path.insert(0, str(PROJECT))
RESULTS = PROJECT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
from muonlib.algorithms import MuonOptimizer, SGDOptimizer

# ── Detailed logging (v2) ───────────────────────────
from muonlib.detailed_logger import DetailedLogger
LOG_DIR = PROJECT / "logs_v2"
RESULTS_V2 = PROJECT / "results_v3"
RESULTS_V2.mkdir(parents=True, exist_ok=True)
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import paired_ttest, compute_effect_size

ALGOS    = ["Muon-Exact", "SGD"]
D, R, LR = 50, 5, 0.01
KAPPAS   = [10, 100, 1000, 10000]
ITERS    = 2000
SEEDS    = list(range(8))
EPSILON = 0.01

def make_opt(name):
    return {"Muon-Exact": lambda: MuonOptimizer(variant="exact", mu=0.9),
            "SGD": lambda: SGDOptimizer(momentum=0.9)}[name]()

def run_ms(algo, d, r, lr, kappa, seed, iters):

    # ── Detailed logger (v2) ─────────────────────────
    logger = DetailedLogger(LOG_DIR, "E18_detailed", algo, {k: v for k, v in locals().items()
                             if k in ("d", "seed", "lr", "r", "noise", "dist",
                                      "spectrum", "kappa", "init_scale", "iters",
                                      "L", "m", "n", "wd", "gamma", "p", "q")})
    X_star = generate_target_matrix(d, r=r, kappa=kappa, seed=seed)
    m = int(2 * d * r)
    A = generate_measurement_matrices(d, m, dist="normal", seed=seed+1000)
    y = np.array([np.trace(Ai.T @ X_star) for Ai in A])
    X = np.random.RandomState(seed).randn(d, d) * 0.01
    opt = make_opt(algo)
    losses = []; k_epsilon = -1; t0 = time.time()
    try:
        for step in range(iters):
            loss = float(compute_loss_matrix_sensing(X, A, y))
            losses.append(loss)
            G = compute_gradient_matrix_sensing(X, A, y)
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
            if k_epsilon < 0 and loss <= EPSILON: k_epsilon = step + 1
        if k_epsilon < 0: k_epsilon = iters + 1
    finally:
        logger.close()

    return {"algo": algo, "d": d, "r": r, "lr": lr, "kappa": kappa,
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
    total = len(ALGOS) * len(KAPPAS) * len(SEEDS)
    print(f"E18: {len(ALGOS)} algos x {len(KAPPAS)} kappas x {len(SEEDS)} seeds = {total}")
    rows = []; idx = 0
    for algo in tqdm(ALGOS, desc="E18 algo"):
        for k in KAPPAS:
            for seed in SEEDS:
                idx += 1
                print(f"[E18 {idx}/{total}] {algo} kappa={k} seed={seed}", flush=True)
                rows.append(run_ms(algo, D, R, LR, k, seed, ITERS))
    save_csv("E18_detailed", rows)

    print(f"\n{'='*60}")
    print(f"{'algo':<14s} {'kappa':>7s} {'avg_K':>8s} {'avg_loss':>12s}")
    print("-"*48)
    for algo in ALGOS:
        for k in KAPPAS:
            sub = [r for r in rows if r["algo"]==algo and r["kappa"]==k]
            print(f"{algo:<14s} {k:>7d} {np.mean([r['K_epsilon'] for r in sub]):>8.1f} {np.mean([r['final_loss'] for r in sub]):>12.2e}")
    print(f"\nE18 完成")
