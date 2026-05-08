"""Quick execution checks for autograd problems and optimizers."""

from __future__ import annotations

import sys
from functools import partial
from math import isfinite
from pathlib import Path

from joblib import Parallel, delayed
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from optimizers import MuonExact, Shampoo
from problems.MatrixConstruction import randn
from problems.MatrixFactorization import make_matrix_factorization_problem
from problems.MatrixSensing import make_matrix_sensing_problem


ALGOS = ["Muon", "Muon-Exact", "Shampoo", "Adam", "SGD"]


def torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {dtype_name}")
    return dtype


def configure_torch(dtype: torch.dtype) -> None:
    torch.set_default_dtype(dtype)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def make_optimizer(algo: str, params, lr: float, rank: int) -> torch.optim.Optimizer:
    if algo == "Muon":
        if hasattr(torch.optim, "Muon"):
            return torch.optim.Muon(
                params,
                lr=lr,
                weight_decay=0.0,
                momentum=0.9,
                nesterov=False,
                ns_steps=5,
            )
        return MuonExact(params, lr=lr, momentum=0.9, variant="newton_schulz", ns_steps=5)
    if algo in {"Muon-Exact", "MuonExact"}:
        return MuonExact(params, lr=lr, momentum=0.9, variant="exact")
    if algo == "Shampoo":
        return Shampoo(params, lr=lr, beta2=0.9, epsilon=1e-8)
    if algo == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if algo == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown algo: {algo}")


def matrix_sensing_step(state: dict) -> tuple[float, float]:
    problem = state["problem"]
    x = state["x"]
    opt = state["optimizer"]
    opt.zero_grad(set_to_none=True)
    loss = problem.loss(x)
    loss.backward()
    grad_norm = float(x.grad.detach().norm().cpu())
    opt.step()
    return float(loss.detach().cpu()), grad_norm


def matrix_factorization_step(state: dict) -> tuple[float, float]:
    problem = state["problem"]
    left = state["left"]
    right = state["right"]
    opt = state["optimizer"]
    opt.zero_grad(set_to_none=True)
    loss = problem.loss(left, right)
    loss.backward()
    grad_norm = float(
        torch.linalg.vector_norm(
            torch.stack([left.grad.detach().norm(), right.grad.detach().norm()])
        ).cpu()
    )
    opt.step()
    return float(loss.detach().cpu()), grad_norm


def run_sensing_spec(spec: dict, *, step_fn=matrix_sensing_step):
    device = torch.device(spec["device_type"])
    dtype = torch_dtype(spec["dtype_name"])
    configure_torch(dtype)
    problem = make_matrix_sensing_problem(
        spec["d"],
        spec["rank"],
        noise=spec["noise"],
        dist=spec["dist"],
        spectrum=spec["spectrum"],
        kappa=spec["kappa"],
        seed=spec["seed"],
        device=device,
        dtype=dtype,
    )
    x = torch.nn.Parameter(
        randn((spec["d"], spec["d"]), spec["seed"] + 3000, device=device, dtype=dtype)
        * spec["init_scale"]
    )
    opt = make_optimizer(spec["algo"], [x], spec["lr"], rank=spec["rank"])
    state = {"problem": problem, "x": x, "optimizer": opt}
    losses = []
    for _ in range(spec["iters"]):
        loss_value, _ = step_fn(state)
        losses.append(loss_value)
    row = {"final_loss": losses[-1]}
    return (spec["algo"], spec["d"], spec["seed"]), row, {"loss": losses}


def run_factorization_spec(spec: dict, *, step_fn=matrix_factorization_step):
    device = torch.device(spec["device_type"])
    dtype = torch_dtype(spec["dtype_name"])
    configure_torch(dtype)
    problem = make_matrix_factorization_problem(
        spec["d"],
        spec["rank"],
        spectrum=spec["spectrum"],
        kappa=spec["kappa"],
        seed=spec["seed"],
        device=device,
        dtype=dtype,
    )
    left = torch.nn.Parameter(
        randn((spec["d"], spec["rank"]), spec["seed"] + 3000, device=device, dtype=dtype)
        * spec["init_scale"]
    )
    right = torch.nn.Parameter(
        randn((spec["d"], spec["rank"]), spec["seed"] + 4000, device=device, dtype=dtype)
        * spec["init_scale"]
    )
    opt = make_optimizer(spec["algo"], [left, right], spec["lr"], rank=spec["rank"])
    state = {"problem": problem, "left": left, "right": right, "optimizer": opt}
    losses = []
    for _ in range(spec["iters"]):
        loss_value, _ = step_fn(state)
        losses.append(loss_value)
    row = {"final_loss": losses[-1]}
    return (spec["algo"], spec["d"], spec["seed"]), row, {"loss": losses}


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
        "device_type": "cpu",
        "dtype_name": "float64",
    }


def check_problem(name: str, run_one, spec_factory) -> None:
    print(f"\n{name}")
    for algo in ALGOS:
        _, row, _ = run_one(spec_factory(algo))
        final_loss = float(row["final_loss"])
        if not isfinite(final_loss):
            raise AssertionError(f"{name} {algo} produced non-finite loss: {final_loss}")
        print(f"{algo:10s} final_loss={final_loss:.6e}")


def check_joblib_step_binding() -> None:
    print("\njoblib.Parallel step binding")
    spec = make_sensing_spec("SGD")
    spec["iters"] = 3
    run_one = partial(run_sensing_spec, step_fn=matrix_sensing_step)
    [(key, row, traj)] = Parallel(n_jobs=2, backend="loky")(
        delayed(run_one)(spec) for spec in [spec]
    )
    if key != ("SGD", spec["d"], spec["seed"]):
        raise AssertionError(f"unexpected key from worker: {key}")
    if len(traj["loss"]) != spec["iters"]:
        raise AssertionError("trajectory length does not match requested iterations")
    print(f"bound step final_loss={row['final_loss']:.6e}")


def main() -> None:
    print(f"torch={torch.__version__}, official_muon={hasattr(torch.optim, 'Muon')}")
    check_problem("MatrixSensing", run_sensing_spec, make_sensing_spec)
    check_problem("MatrixFactorization", run_factorization_spec, make_factorization_spec)
    check_joblib_step_binding()


if __name__ == "__main__":
    main()
