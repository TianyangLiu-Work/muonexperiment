"""Runner for Matrix Factorization experiments."""

from __future__ import annotations

from typing import Any, Callable

import torch

from problems.MatrixConstruction import randn
from problems.MatrixFactorization import (
    MatrixFactorizationProblem,
    make_matrix_factorization_problem,
)

from .runtime import configure_torch, make_optimizer, run_steps, torch_dtype


StepFn = Callable[[dict[str, Any]], tuple[float, float]]


def run_spec(
    spec: dict[str, Any],
    *,
    step_fn: StepFn | None = None,
) -> tuple[tuple[str, int, int], dict[str, Any], dict[str, list[float]]]:
    row, trajectory = run_once(**spec, step_fn=step_fn or step)
    key = (spec["algo"], spec["d"], spec["seed"])
    return key, row, trajectory


def run_once(
    algo: str,
    d: int,
    rank: int,
    lr: float,
    spectrum: str,
    kappa: float,
    init_scale: float,
    seed: int,
    iters: int,
    epsilon: float,
    step_fn: StepFn | None = None,
    factor_rank: int | None = None,
    device_type: str = "cpu",
    dtype_name: str = "float64",
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    device = torch.device(device_type)
    dtype = torch_dtype(dtype_name)
    configure_torch(dtype)
    step_fn = step_fn or step

    problem = make_matrix_factorization_problem(
        d,
        rank,
        spectrum=spectrum,
        kappa=kappa,
        seed=seed,
        device=device,
        dtype=dtype,
        factor_rank=factor_rank,
    )
    left = torch.nn.Parameter(
        randn((d, problem.factor_rank), seed + 3000, device=device, dtype=dtype) * init_scale
    )
    right = torch.nn.Parameter(
        randn((d, problem.factor_rank), seed + 4000, device=device, dtype=dtype) * init_scale
    )
    opt = make_optimizer(algo, [left, right], lr, rank=problem.factor_rank)
    state = {"problem": problem, "left": left, "right": right, "optimizer": opt}

    losses, grad_norms, k_epsilon, elapsed = run_steps(state, step_fn, iters, epsilon, device)
    row = {
        "problem": "MatrixFactorization",
        "algo": algo,
        "d": d,
        "r": rank,
        "factor_rank": problem.factor_rank,
        "lr": lr,
        "spectrum": spectrum,
        "kappa": kappa,
        "init_scale": init_scale,
        "seed": seed,
        "iters": iters,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "K_epsilon": k_epsilon,
        "time_s": elapsed,
    }
    return row, {"loss": losses, "grad_norm": grad_norms}


def step(state: dict[str, Any]) -> tuple[float, float]:
    """One visible optimization step for MatrixFactorization."""

    problem: MatrixFactorizationProblem = state["problem"]
    opt = state["optimizer"]
    left = state["left"]
    right = state["right"]
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
