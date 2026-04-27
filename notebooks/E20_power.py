"""
E20 — Statistical Power & Sample Size Determination
=====================================================
问题: 多少随机种子才够检测 Muon vs SGD 的显著差异？

方法: 对每个样本量 n ∈ {5,10,20,30,50}，从 50 个 runs 中
      重采样 n 个计算配对 t 检验的 p 值和效应量。
      展示随着 n 增大，统计显著性如何变化。
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
from muonlib.data import (generate_target_matrix, generate_measurement_matrices,
                           compute_gradient_matrix_sensing, compute_loss_matrix_sensing)
from muonlib.metrics import paired_ttest, compute_effect_size
from scipy import stats

ALGOS          = ["Muon-Exact", "SGD"]
D, R, LR       = 50, 5, 0.01
ITERS          = 2000
TOTAL_SEEDS    = 50           # 跑满 50 个种子作为总体
SAMPLE_SIZES   = [5, 10, 20, 30, 50]  # 要评估的样本量
BOOTSTRAP_REPS = 200          # 每个样本量重采样次数
EPSILON = 0.01

def make_opt(name):
    return {"Muon-Exact": lambda: MuonOptimizer(variant="exact", mu=0.9),
            "SGD": lambda: SGDOptimizer(momentum=0.9)}[name]()

def run_ms(algo, d, r, lr, seed, iters):
    X_star = generate_target_matrix(d, r=r, seed=seed)
    m = int(2 * d * r)
    A = generate_measurement_matrices(d, m, dist="normal", seed=seed+1000)
    y = np.array([np.trace(Ai.T @ X_star) for Ai in A])
    X = np.random.RandomState(seed).randn(d, d) * 0.01
    opt = make_opt(algo)
    losses = []; k_epsilon = -1; t0 = time.time()
    for step in range(iters):
        G = compute_gradient_matrix_sensing(X, A, y)
        X = opt.step(X, G, lr)
        loss = float(compute_loss_matrix_sensing(X, A, y))
        losses.append(loss)
        if k_epsilon < 0 and loss <= EPSILON: k_epsilon = step + 1
    if k_epsilon < 0: k_epsilon = iters + 1
    return {"algo": algo, "d": d, "r": r, "lr": lr, "seed": seed,
            "iters": iters, "final_loss": losses[-1],
            "min_loss": min(losses), "K_epsilon": k_epsilon,
            "time_s": time.time() - t0}

def save_csv(eid, rows):
    if not rows: return
    fname = RESULTS / f"{eid}_results.csv"
    wh = not fname.exists()
    with open(fname, "a", newline="") as f:
        from muonlib.metrics import enrich_result_row
        rows = [enrich_result_row(r) for r in rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if wh: w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Saved {eid}: {len(rows)} rows")

if __name__ == "__main__":
    # Phase 1: 获取 50 个种子的完整数据
    total_runs = len(ALGOS) * TOTAL_SEEDS
    print(f"E20 Phase 1: {total_runs} runs ({TOTAL_SEEDS} seeds x {len(ALGOS)} algos)")
    rows = []; idx = 0
    for algo in tqdm(ALGOS, desc="E20 algo"):
        for seed in range(TOTAL_SEEDS):
            idx += 1
            print(f"[E20 {idx}/{total_runs}] {algo} seed={seed}", flush=True)
            rows.append(run_ms(algo, D, R, LR, seed, ITERS))
    save_csv("E20", rows)

    # Phase 2: 功效分析
    print(f"\n{'='*60}")
    print("E20 统计功效分析")
    print(f"{'='*60}")
    muon_k = np.array([r["K_epsilon"] for r in rows if r["algo"] == "Muon-Exact"])
    sgd_k  = np.array([r["K_epsilon"] for r in rows if r["algo"] == "SGD"])
    rng = np.random.RandomState(42)

    print(f"\n全样本 (N={TOTAL_SEEDS}):")
    t_full, p_full = paired_ttest(muon_k, sgd_k)
    d_full = compute_effect_size(muon_k, sgd_k, paired=True)
    try: _, w_p = stats.wilcoxon(muon_k, sgd_k); 
    except: w_p = 1.0
    print(f"  Muon K_eps = {muon_k.mean():.1f} +/- {muon_k.std():.1f}")
    print(f"  SGD  K_eps = {sgd_k.mean():.1f} +/- {sgd_k.std():.1f}")
    print(f"  Paired t: t={t_full:+.3f} p={p_full:.6f}")
    print(f"  Wilcoxon p={w_p:.6f}")
    print(f"  Cohen's d={d_full:+.3f}")

    print(f"\n{'n':>4s}  {'p(t-test)':>10s}  {'Cohen d':>9s}  {'sig@0.05':>10s}  {'power':>8s}")
    print("-" * 55)
    for n in SAMPLE_SIZES:
        p_vals = []; d_vals = []
        for _ in range(BOOTSTRAP_REPS):
            idx = rng.choice(TOTAL_SEEDS, n, replace=False)
            _, p = paired_ttest(muon_k[idx], sgd_k[idx])
            d = compute_effect_size(muon_k[idx], sgd_k[idx], paired=True)
            p_vals.append(p)
            d_vals.append(d)
        p_vals = np.array(p_vals)
        d_vals = np.array(d_vals)
        sig_rate = np.mean(p_vals < 0.05)
        power = sig_rate  # 经验功效
        print(f"{n:>4d}  {np.median(p_vals):>10.4f}  {np.median(d_vals):>+9.3f}  "
              f"{sig_rate:>10.1%}  {power:>8.1%}")

    print(f"\nE20 完成")
