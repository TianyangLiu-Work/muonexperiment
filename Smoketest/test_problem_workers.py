"""Quick execution checks for the PyTorch problem workers."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import isfinite
from multiprocessing import get_context
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from problems.MatrixFactorization import run_spec as run_matrix_factorization
from problems.MatrixSensing import run_spec as run_matrix_sensing
from problems.MatrixSensing import step as matrix_sensing_step


ALGOS = ["Muon", "Muon-Exact", "Shampoo", "Adam", "SGD"]


def make_sensing_spec(algo: str) -> dict:
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


def make_factorization_spec(algo: str) -> dict:
    return {
        "algo": algo,
        "d": 20,
        "rank": 3,
        "lr": 0.01,
        "spectrum": "hard-cutoff",
        "kappa": 1.0,
        "init_scale": 0.01,
        "seed": 0,
        "iters": 20,
        "epsilon": 0.01,
        "device_type": "cpu",
        "dtype_name": "float64",
    }


def check_problem(name: str, runner, spec_factory) -> None:
    print(f"\n{name}")
    for algo in ALGOS:
        _, row, _ = runner(spec_factory(algo))
        final_loss = float(row["final_loss"])
        if not isfinite(final_loss):
            raise AssertionError(f"{name} {algo} produced non-finite loss: {final_loss}")
        print(f"{algo:10s} final_loss={final_loss:.6e}")


def check_spawn_step_binding() -> None:
    print("\nProcessPool step binding")
    spec = make_sensing_spec("SGD")
    spec["iters"] = 3
    run_one = partial(run_matrix_sensing, step_fn=matrix_sensing_step)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
        key, row, traj = executor.submit(run_one, spec).result(timeout=60)
    if key != ("SGD", spec["d"], spec["seed"]):
        raise AssertionError(f"unexpected key from worker: {key}")
    if len(traj["loss"]) != spec["iters"]:
        raise AssertionError("trajectory length does not match requested iterations")
    print(f"bound step final_loss={row['final_loss']:.6e}")


def main() -> None:
    print(f"torch={torch.__version__}, official_muon={hasattr(torch.optim, 'Muon')}")
    check_problem("MatrixSensing", run_matrix_sensing, make_sensing_spec)
    check_problem("MatrixFactorization", run_matrix_factorization, make_factorization_spec)
    check_spawn_step_binding()


if __name__ == "__main__":
    main()
