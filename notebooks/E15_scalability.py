"""
E15 — Scalability: Muon vs SGD at Larger Dimensions
=====================================================
问题: d 增大时 Muon SVD O(d³) 的开销是否会抵消其收敛优势？

控制变量:
  - d ∈ {100, 200}    (500 skipped per ENGINEERING_COMPROMISES.md)
  - r = 5
  - 1000 iters, 3 seeds (reduced for runtime)
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

from muonlib.algorithms import MuonOptimizer, SGDOptimizer
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import (compute_flops_matrix_sensing, paired_ttest,
                              compute_effect_size)

ALGOS       = ["Muon-Exact", "SGD"]
DIMS        = [100, 200]
R           = 5
LR          = 0.01
NOISE       = 0.0
DIST        = "normal"
SPECTRUM    = "hard-cutoff"
KAPPA       = 1.0
INIT_SCALE  = 0.01
ITERS       = 1000
SEEDS       = list(range(3))
EPSILON = 0.01

def make_opt(name):
    opts = {
        "Muon-Exact": lambda: MuonOptimizer(variant="exact", mu=0.9),
        "SGD": lambda: SGDOptimizer(momentum=0.9),
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
    print(f"Saved {eid}: {len(rows)} rows -> {fname}")

if __name__ == "__main__":
    total = len(ALGOS) * len(DIMS) * len(SEEDS)
    print(f"E15: {len(ALGOS)} algos x {len(DIMS)} dims x {len(SEEDS)} seeds = {total} runs")
    print(f"    iters={ITERS}, d up to {max(DIMS)}")
    print()
    rows = []
    idx = 0
    for algo in tqdm(ALGOS, desc="E15 algo"):
        for d in DIMS:
            for seed in SEEDS:
                idx += 1
                print(f"[E15 {idx}/{total}] {algo} d={d} seed={seed}", flush=True)
                row = run_ms(algo, d, R, LR, NOISE, DIST, SPECTRUM, KAPPA,
                             INIT_SCALE, seed, ITERS)
                rows.append(row)
    save_csv("E15", rows)

    print(f"\n{'='*60}")
    print("E15 统计: K_epsilon + Wall-clock")
    print(f"{'='*60}")
    print(f"{'algo':<14s} {'d':>4s} {'time_s':>8s} {'avg_K':>8s} {'avg_loss':>12s}")
    print("-" * 55)
    for algo in ALGOS:
        for d in DIMS:
            subset = [r for r in rows if r["algo"] == algo and r["d"] == d]
            avg_k = np.mean([r["K_epsilon"] for r in subset])
            avg_t = np.mean([r["time_s"] for r in subset])
            avg_loss = np.mean([r["final_loss"] for r in subset])
            print(f"{algo:<14s} {d:>4d} {avg_t:>8.2f} {avg_k:>8.1f} {avg_loss:>12.2e}")

    print(f"\n{'='*60}")
    print("FLOPs per run (K_iters x flops/iter)")
    print(f"{'='*60}")
    for d in DIMS:
        m = int(2 * d * R)
        sgd_flops = compute_flops_matrix_sensing(d, m, ITERS, "SGD")
        muon_flops = compute_flops_matrix_sensing(d, m, ITERS, "Muon-Exact")
        print(f"  d={d:>3d}: SGD={sgd_flops:.2e}  Muon={muon_flops:.2e}  "
              f"ratio={muon_flops/sgd_flops:.2f}")
    print(f"\nE15 完成")
