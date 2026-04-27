"""
E05 — FLOPs computation (detailed)
==================================================
Compute theoretical FLOPs for Matrix Sensing. Save to CSV.
"""
import os as _os
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["NUMEXPR_NUM_THREADS"] = "1"
_os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys, csv, time
from pathlib import Path
import numpy as np

PROJECT = Path(_os.environ.get("MUON_PROJECT", "/data/home/tyliu/muonexperiment"))
sys.path.insert(0, str(PROJECT))
RESULTS_V3 = PROJECT / "results_v3"
RESULTS_V3.mkdir(parents=True, exist_ok=True)
from muonlib.metrics import compute_flops_matrix_sensing

DIMS = [50, 100, 200, 500]
R = 5
ITERS = 2000
ALGOS = ["SGD", "Muon-Exact"]

def save_csv(eid, rows):
    fname = RESULTS_V3 / f"{eid}_results.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {fname}")

if __name__ == "__main__":
    print("=" * 80)
    print("E05 — FLOPs 计算 (Matrix Sensing)")
    print("=" * 80)
    rows = []
    for d in DIMS:
        m = int(2 * d * R)
        for algo in ALGOS:
            flops = compute_flops_matrix_sensing(d, m, ITERS, algo)
            print(f"  d={d:>4d} m={m:>6d} {algo:<16s} FLOPs={flops:>20.2e}")
            rows.append({"d": d, "m": m, "algo": algo, "iters": ITERS, "total_flops": flops})
    save_csv("E05_detailed", rows)
    print("=" * 80)
    print("E05 完成")
    print("=" * 80)

