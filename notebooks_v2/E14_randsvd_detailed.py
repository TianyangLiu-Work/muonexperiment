"""
E14 — RandSVD precision-efficiency
==================================================
MS. ALGOS=["Muon-Exact","Muon-RandSVD"]. D=50. For RandSVD: p in [5,10,20], q in [1,2]. SEEDS=10.
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
from numpy.linalg import svd
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
RESULTS_V2 = PROJECT / "results_v2"
RESULTS_V2.mkdir(parents=True, exist_ok=True)
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import (compute_flops_matrix_sensing, paired_ttest,
                              wilcoxon_test, compute_effect_size, compute_statistics)
ALGOS       = ["Muon-Exact", "Muon-RandSVD"]
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

# RandSVD hyperparameters
RSVD_P_VALS = [5, 10, 20]
RSVD_Q_VALS = [1, 2]

def make_opt(name, p=10, q=2):
    """创建优化器实例，支持 RandSVD 的 p,q 参数。"""
    opts = {
        "Muon-Exact":    lambda: MuonOptimizer(variant="exact", mu=0.9, nesterov=False),
        "Muon-RandSVD":  lambda: MuonOptimizer(variant="randsvd", p=p, q=q),
        "SGD":           lambda: SGDOptimizer(momentum=0.9),
    }
    return opts[name]()
def run_ms(algo, d, r, lr, noise, dist, spectrum, kappa, init_scale, seed, iters, p=10, q=2):

    # ── Detailed logger (v2) ─────────────────────────
    logger = DetailedLogger(LOG_DIR, "E14_detailed", algo, {k: v for k, v in locals().items()
                             if k in ("d", "seed", "lr", "r", "noise", "dist",
                                      "spectrum", "kappa", "init_scale", "iters",
                                      "L", "m", "n", "wd", "gamma", "p", "q")}, svd_interval=10)
    X_star = generate_target_matrix(d, r=r, spectrum=spectrum, kappa=kappa, seed=seed)
    m = int(2 * d * r)
    A_matrices = generate_measurement_matrices(d, m, dist=dist, seed=seed+1000)
    y = np.array([np.trace(Ai.T @ X_star) for Ai in A_matrices])
    if noise > 0:
        y += np.random.RandomState(seed+2000).randn(m) * noise

    X = np.random.RandomState(seed).randn(d, d) * init_scale
    opt = make_opt(algo, p=p, q=q)

    losses = []
    t_start = time.time()
    k_epsilon = -1

    for step in range(iters):
        G = compute_gradient_matrix_sensing(X, A_matrices, y)
        X = opt.step(X, G, lr)
        loss = float(compute_loss_matrix_sensing(X, A_matrices, y))
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

        if k_epsilon < 0 and loss <= EPSILON:
            k_epsilon = step + 1

    elapsed = time.time() - t_start
    if k_epsilon < 0:
        k_epsilon = iters + 1
    # ── Close detailed log ───────────────────────────
    logger.close()


    return {
        "algo": algo, "d": d, "r": r, "lr": lr, "noise": noise,
        "dist": dist, "spectrum": spectrum, "kappa": kappa,
        "init_scale": init_scale, "seed": seed, "iters": iters,
        "p": p, "q": q,
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
    # Muon-Exact (no p,q)
    n_exact = len(SEEDS)
    n_rsvd = len(RSVD_P_VALS) * len(RSVD_Q_VALS) * len(SEEDS)
    total = n_exact + n_rsvd
    print(f"E14: 1 exact x {len(SEEDS)} + {len(RSVD_P_VALS)} p x {len(RSVD_Q_VALS)} q x {len(SEEDS)} = {total} runs")
    print()
    rows = []
    idx = 0
    d = DIMS[0]

    for seed in SEEDS:
        idx += 1
        print(f"[E14 {idx}/{total}] Muon-Exact seed={seed}", flush=True)
        row = run_ms("Muon-Exact", d, R, LR, NOISE, DIST, SPECTRUM, KAPPA,
                     INIT_SCALE, seed, ITERS)
        rows.append(row)

    for p in RSVD_P_VALS:
        for q in RSVD_Q_VALS:
            for seed in SEEDS:
                idx += 1
                print(f"[E14 {idx}/{total}] Muon-RandSVD p={p} q={q} seed={seed}", flush=True)
                row = run_ms("Muon-RandSVD", d, R, LR, NOISE, DIST, SPECTRUM, KAPPA,
                             INIT_SCALE, seed, ITERS, p=p, q=q)
                rows.append(row)

    save_csv("E14_detailed", rows)

    print(f"\n{'='*60}")
    print("E14 统计汇总")
    print(f"{'='*60}")
    print(f"\n{'algo':<18s} {'p':>4s} {'q':>4s} {'n':>4s} {'avg time_s':>10s} {'avg K_eps':>10s} {'avg loss':>12s}")
    print("-" * 70)
    # Exact
    exact_rows = [r for r in rows if r["algo"] == "Muon-Exact"]
    avg_t = np.mean([r["time_s"] for r in exact_rows])
    avg_k = np.mean([r["K_epsilon"] for r in exact_rows])
    avg_l = np.mean([r["final_loss"] for r in exact_rows])
    print(f"{'Muon-Exact':<18s} {'-':>4s} {'-':>4s} {len(exact_rows):>4d} {avg_t:>10.3f} {avg_k:>10.1f} {avg_l:>12.2e}")
    # RandSVD
    for p in RSVD_P_VALS:
        for q in RSVD_Q_VALS:
            subset = [r for r in rows if r["algo"] == "Muon-RandSVD" and r["p"] == p and r["q"] == q]
            nn = len(subset)
            if nn == 0: continue
            avg_t = np.mean([r["time_s"] for r in subset])
            avg_k = np.mean([r["K_epsilon"] for r in subset])
            avg_l = np.mean([r["final_loss"] for r in subset])
            print(f"{'Muon-RandSVD':<18s} {p:>4d} {q:>4d} {nn:>4d} {avg_t:>10.3f} {avg_k:>10.1f} {avg_l:>12.2e}")

    print(f"\n{'='*60}")
    print("配对检验: Exact vs RandSVD(p,q) (K_epsilon)")
    print(f"{'='*60}")
    exact_k = [r["K_epsilon"] for r in exact_rows]
    for p in RSVD_P_VALS:
        for q in RSVD_Q_VALS:
            rsvd_k = [r["K_epsilon"] for r in rows if r["algo"] == "Muon-RandSVD" and r["p"] == p and r["q"] == q]
            if len(exact_k) == len(rsvd_k) and len(exact_k) > 1:
                t_stat, p_val = paired_ttest(exact_k, rsvd_k)
                d_effect = compute_effect_size(exact_k, rsvd_k, paired=True)
                sign = "\u2713" if p_val < 0.05 else "\u2717"
                print(f"  Exact vs RandSVD(p={p},q={q}): t={t_stat:+.3f} p={p_val:.4f} {sign}  Cohen's d={d_effect:+.2f}")
    print(f"\n{'='*60}")
    print("E14 完成")
    print(f"{'='*60}")
