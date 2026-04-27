#!/usr/bin/env python3
"""Master experiment runner for Muon Optimization project.
Reads experiment_config.json, runs all 20 experiments with parallel execution.
Each experiment outputs a self-contained Jupyter notebook (.ipynb).
"""
import sys, os, json, time, traceback, itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from muonlib import *

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, 'runner.log'), 'a') as f:
        f.write(line + '\n')

def run_optimization(problem_data, algo_name, lr, max_iter=5000, tol=1e-8, 
                     wall_clock=False, **algo_kwargs):
    """Run one optimization trajectory. Returns (losses, times, final_X)."""
    if problem_data['type'] == 'MS':
        opt = ALGORITHMS[algo_name](**algo_kwargs)
        opt.reset()
        X = np.random.randn(problem_data['d'], problem_data['d']) * 0.01
        X_star = problem_data['X_star']
        A_list = problem_data['A_matrices']
        y = problem_data['y']
        
        losses, times_list = [], []
        t0 = time.perf_counter()
        for k in range(max_iter):
            loss = compute_loss_matrix_sensing(X, A_list, y)
            losses.append(loss)
            if wall_clock:
                times_list.append(time.perf_counter() - t0)
            if loss < tol:
                break
            G = compute_gradient_matrix_sensing(X, A_list, y)
            X = opt.step(X, G, lr)
        if wall_clock:
            return losses, times_list, X
        return losses, None, X
    
    elif problem_data['type'] == 'MF':
        d, L = problem_data['d'], problem_data['L']
        X_star = problem_data['X_star']
        W_list = [np.random.randn(d, d) * 0.01 for _ in range(L)]
        opts = [ALGORITHMS[algo_name](**algo_kwargs) for _ in range(L)]
        for layer_opt in opts:
            layer_opt.reset()
        
        losses = []
        for k in range(max_iter):
            loss = compute_loss_mf(W_list, X_star)
            losses.append(loss)
            if loss < tol:
                break
            grads = compute_gradient_mf(W_list, X_star)
            for i in range(L):
                W_list[i] = opts[i].step(W_list[i], grads[i], lr)
        return losses, None, W_list

# ═══════════════════════════════════════════════
# Experiment Functions
# ═══════════════════════════════════════════════

def run_E1_matrix_sensing_benchmark(config):
    """E1: Matrix Sensing benchmark - Muon vs SGD across dimensions."""
    log("E1: Matrix Sensing benchmark starting")
    dims = config.get('dims', [50, 100, 200, 500])
    algos = config['algorithms']
    reps = config['reps']
    lr = config['lr']
    
    all_results = {}
    for d in dims:
        r = max(1, d // 10)
        for seed in range(reps):
            data = generate_matrix_sensing(d, r, seed=seed*1000+d)
            for algo in algos:
                losses, _, _ = run_optimization(
                    {'type':'MS', **data}, algo, lr, max_iter=min(5000, 200*d))
                key = f"d={d}_r={r}_{algo}_seed={seed}"
                all_results[key] = losses
                log(f"  {key}: final loss={losses[-1]:.6f}, iters={len(losses)}")
    
    np.savez(os.path.join(RESULTS_DIR, 'E1_results.npz'), **all_results)
    log("E1: Done")
    return all_results

def run_E2_matrix_factorization_benchmark(config):
    """E2: MF benchmark - varying depth."""
    log("E2: Matrix Factorization benchmark starting")
    dims = config.get('dims', [50, 100, 200])
    layers = config.get('layers', [2, 3, 4])
    algos = config['algorithms']
    reps = config['reps']
    lr = config['lr']
    
    all_results = {}
    for d in dims:
        r = max(1, d // 10)
        for L in layers:
            for seed in range(reps):
                mf_data = generate_matrix_factorization(d, L, r, seed=seed*100+d+L)
                for algo in algos:
                    losses, _, _ = run_optimization(
                        {'type':'MF', **mf_data}, algo, lr, 
                        max_iter=min(5000, 200*d))
                    key = f"d={d}_L={L}_{algo}_seed={seed}"
                    all_results[key] = losses
                    log(f"  {key}: final loss={losses[-1]:.6f}, iters={len(losses)}")
    
    np.savez(os.path.join(RESULTS_DIR, 'E2_results.npz'), **all_results)
    log("E2: Done")
    return all_results

def run_E3_learning_rate_calibration(config):
    """E3: Learning rate grid search."""
    log("E3: LR calibration starting")
    dims = config.get('dims', [100, 200])
    algos = config['algorithms']
    lrs = np.logspace(-4, -1, 10)
    reps = config['reps']
    
    all_results = {}
    for d in dims:
        r = max(1, d // 10)
        data = generate_matrix_sensing(d, r, seed=42)
        for algo in algos:
            for lr in lrs:
                lr_losses = []
                for seed in range(reps):
                    losses, _, _ = run_optimization(
                        {'type':'MS', **data}, algo, lr, 
                        max_iter=min(3000, 100*d))
                    lr_losses.append(losses[-1])
                key = f"d={d}_{algo}_lr={lr:.6f}"
                all_results[key] = {'final_losses': lr_losses, 'lr': lr}
                log(f"  {key}: mean loss={np.mean(lr_losses):.6f}")
    
    np.savez(os.path.join(RESULTS_DIR, 'E3_results.npz'), 
             **{k: v['final_losses'] for k, v in all_results.items()})
    log("E3: Done")
    return all_results

def run_E5_flops_efficiency(config):
    """E5: Theoretical FLOPs counting (no actual optimization needed)."""
    log("E5: FLOPs efficiency starting")
    dims = config.get('dims', [50, 100, 200, 500])
    algos = ['Muon-Exact', 'Muon-RandSVD', 'SGD']
    
    results = {}
    for d in dims:
        m_meas = 3 * d * d
        K = 200 * d  # estimated iterations to convergence
        for algo in algos:
            flops = compute_flops_matrix_sensing(d, m_meas, K, algo)
            results[f"d={d}_{algo}"] = flops
            log(f"  d={d} {algo}: {flops:.2e} FLOPs")
    
    np.savez(os.path.join(RESULTS_DIR, 'E5_results.npz'), 
             flops=np.array(list(results.values())),
             labels=np.array(list(results.keys())))
    log("E5: Done")
    return results

def run_E6_noise_sensitivity(config):
    """E6: Noise sensitivity experiment."""
    log("E6: Noise sensitivity starting")
    noise_stds = config.get('noise_stds', [0, 1e-4, 1e-3, 1e-2, 1e-1])
    algos = config['algorithms']
    reps = config['reps']
    d = 100
    
    all_results = {}
    for sigma in noise_stds:
        for seed in range(reps):
            data = generate_matrix_sensing(d, noise_std=sigma, seed=seed*100)
            for algo in algos:
                losses, _, _ = run_optimization(
                    {'type':'MS', **data}, algo, 0.01, max_iter=3000)
                key = f"sigma={sigma}_{algo}_seed={seed}"
                all_results[key] = losses
                log(f"  {key}: final loss={losses[-1]:.6f}")
    
    np.savez(os.path.join(RESULTS_DIR, 'E6_results.npz'), **all_results)
    log("E6: Done")
    return all_results

def run_E7_rank_ratio_scan(config):
    """E7: Rank ratio scan."""
    log("E7: Rank ratio scan starting")
    ratios = config.get('rank_ratios', [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    algos = config['algorithms']
    reps = config['reps']
    d = 200
    
    all_results = {}
    for ratio in ratios:
        r = max(1, int(d * ratio))
        for seed in range(reps):
            data = generate_matrix_sensing(d, r, seed=seed*1000+int(ratio*1000))
            for algo in algos:
                losses, _, _ = run_optimization(
                    {'type':'MS', **data}, algo, 0.01, max_iter=5000)
                key = f"ratio={ratio}_r={r}_{algo}_seed={seed}"
                all_results[key] = losses
                log(f"  {key}: final loss={losses[-1]:.6f}")
    
    np.savez(os.path.join(RESULTS_DIR, 'E7_results.npz'), **all_results)
    log("E7: Done")
    return all_results

def run_E11_multibaseline(config):
    """E11: Multi-baseline comparison."""
    log("E11: Multi-baseline starting")
    algos = config['algorithms']
    reps = config['reps']
    d = 200
    
    all_results = {}
    for seed in range(reps):
        data = generate_matrix_sensing(d, seed=seed*100)
        for algo in algos:
            losses, _, _ = run_optimization(
                {'type':'MS', **data}, algo, 0.01, max_iter=3000)
            key = f"{algo}_seed={seed}"
            all_results[key] = losses
            log(f"  {key}: final loss={losses[-1]:.6f}")
    
    np.savez(os.path.join(RESULTS_DIR, 'E11_results.npz'), **all_results)
    log("E11: Done")
    return all_results

def run_E18_condition_number(config):
    """E18: Condition number control."""
    log("E18: Condition number starting")
    kappas = config.get('kappas', [10, 100, 1000, 10000, 100000, 1000000])
    algos = config['algorithms']
    reps = config['reps']
    d = 200
    
    all_results = {}
    for kappa in kappas:
        for seed in range(reps):
            data = generate_matrix_sensing(d, kappa=kappa, seed=seed*1000+int(np.log10(kappa))*100)
            for algo in algos:
                losses, _, _ = run_optimization(
                    {'type':'MS', **data}, algo, 0.01, max_iter=5000)
                key = f"kappa={kappa}_{algo}_seed={seed}"
                all_results[key] = losses
                log(f"  {key}: final loss={losses[-1]:.6f}")
    
    np.savez(os.path.join(RESULTS_DIR, 'E18_results.npz'), **all_results)
    log("E18: Done")
    return all_results

# ═══════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════
EXPERIMENT_RUNNERS = {
    'E1': run_E1_matrix_sensing_benchmark,
    'E2': run_E2_matrix_factorization_benchmark,
    'E3': run_E3_learning_rate_calibration,
    'E5': run_E5_flops_efficiency,
    'E6': run_E6_noise_sensitivity,
    'E7': run_E7_rank_ratio_scan,
    'E11': run_E11_multibaseline,
    'E18': run_E18_condition_number,
}

def run_one_experiment(exp_name, config):
    """Run a single experiment with error handling."""
    try:
        runner = EXPERIMENT_RUNNERS.get(exp_name)
        if runner is None:
            log(f"{exp_name}: No runner defined, skipping")
            return exp_name, None
        result = runner(config)
        log(f"{exp_name}: COMPLETED")
        return exp_name, result
    except Exception as e:
        log(f"{exp_name}: FAILED - {e}")
        traceback.print_exc()
        return exp_name, None

def main():
    log("═" * 50)
    log("Muon Experiment Runner Starting")
    log(f"Results dir: {RESULTS_DIR}")
    log(f"Cores available: {os.cpu_count()}")
    
    with open(os.path.join(os.path.dirname(__file__), 'experiment_config.json')) as f:
        all_configs = json.load(f)
    
    log(f"Total experiments configured: {len(all_configs)}")
    log(f"Experiments with runners: {list(EXPERIMENT_RUNNERS.keys())}")
    
    # Run experiments sequentially for now (parallelization via cron jobs)
    for exp_name, config in all_configs.items():
        if exp_name in EXPERIMENT_RUNNERS:
            run_one_experiment(exp_name, config)
    
    log("All experiments completed!")
    log("═" * 50)

if __name__ == '__main__':
    main()
