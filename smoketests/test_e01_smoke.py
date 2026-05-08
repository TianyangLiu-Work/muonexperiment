"""Quick execution check for the PyTorch E01 optimizer set."""

from __future__ import annotations

import sys
from math import isfinite
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from muonlib_torch.e01_matrix_sensing import run_spec


ALGOS = ["Muon", "Muon-Exact", "Shampoo", "Adam", "SGD"]


def make_spec(algo: str) -> dict:
    return {
        "algo": algo,
        "d": 20,
        "rank": 3,
        "lr": 0.01,
        "noise": 0.0,
        "dist": "normal",
        "spectrum": "hard-cutoff",
        "kappa": 1.0,
        "init_scale": 0.01,
        "seed": 0,
        "iters": 20,
        "epsilon": 0.01,
        "device_type": "cpu",
        "dtype_name": "float64",
    }


def main() -> None:
    print(f"torch={torch.__version__}, official_muon={hasattr(torch.optim, 'Muon')}")
    for algo in ALGOS:
        _, row, _ = run_spec(make_spec(algo))
        final_loss = float(row["final_loss"])
        if not isfinite(final_loss):
            raise AssertionError(f"{algo} produced non-finite loss: {final_loss}")
        print(f"{algo:10s} final_loss={final_loss:.6e}")


if __name__ == "__main__":
    main()
