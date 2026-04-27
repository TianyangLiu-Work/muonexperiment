"""
E05 — FLOPs computation only
==================================================
Compute theoretical FLOPs for MS. No optimization runs. No CSV output.
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
from muonlib.metrics import compute_flops_matrix_sensing

# ═══════════════════════════════════════════════════════
# 实验参数
# ═══════════════════════════════════════════════════════
DIMS = [50, 100, 200, 500]
R = 5
ITERS = 2000
ALGOS = ["SGD", "Muon-Exact"]

if __name__ == "__main__":
    print("=" * 80)
    print("E05 — FLOPs 计算 (Matrix Sensing)")
    print("=" * 80)
    print()
    print(f"{'d':>6s} {'m':>8s} {'alg':<16s} {'total FLOPs':>20s}")
    print("-" * 60)
    for d in DIMS:
        m = int(2 * d * R)
        for algo in ALGOS:
            flops = compute_flops_matrix_sensing(d, m, ITERS, algo)
            print(f"{d:>6d} {m:>8d} {algo:<16s} {flops:>20.2e}")
        print()
    print("=" * 80)
    print("E05 完成 (无 CSV 输出)")
    print("=" * 80)
