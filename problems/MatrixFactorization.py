"""Matrix Factorization problem worker."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from .MatrixConstruction import (
    configure_torch,
    generate_target_matrix,
    make_optimizer,
    randn,
    sync_device,
    torch_dtype,
)


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
    factor_rank = rank if factor_rank is None else factor_rank

    x_star = generate_target_matrix(
        d,
        rank,
        spectrum=spectrum,
        kappa=kappa,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    left = torch.nn.Parameter(
        randn((d, factor_rank), seed + 3000, device=device, dtype=dtype) * init_scale
    )
    right = torch.nn.Parameter(
        randn((d, factor_rank), seed + 4000, device=device, dtype=dtype) * init_scale
    )
    opt = make_optimizer(algo, [left, right], lr, rank=factor_rank)
    state = {"left": left, "right": right, "target": x_star, "optimizer": opt}

    losses: list[float] = []
    grad_norms: list[float] = []
    k_epsilon = None

    sync_device(device)
    t0 = time.time()
    for step in range(iters):
        loss_value, grad_norm = step_fn(state)
        losses.append(loss_value)
        grad_norms.append(grad_norm)

        if k_epsilon is None and loss_value <= epsilon:
            k_epsilon = step + 1

    sync_device(device)
    elapsed = time.time() - t0
    if k_epsilon is None:
        k_epsilon = iters + 1

    row = {
        "problem": "MatrixFactorization",
        "algo": algo,
        "d": d,
        "r": rank,
        "factor_rank": factor_rank,
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
    trajectory = {"loss": losses, "grad_norm": grad_norms}
    return row, trajectory


def step(state: dict[str, Any]) -> tuple[float, float]:
    """One visible optimization step for MatrixFactorization."""

    opt = state["optimizer"]
    left = state["left"]
    right = state["right"]
    opt.zero_grad(set_to_none=True)
    loss = matrix_factorization_loss(left, right, state["target"])
    loss.backward()

    grad_norm = float(
        torch.linalg.vector_norm(
            torch.stack([left.grad.detach().norm(), right.grad.detach().norm()])
        ).cpu()
    )
    opt.step()
    return float(loss.detach().cpu()), grad_norm


matrix_factorization_step = step


def matrix_factorization_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    residual = left @ right.T - target
    return 0.5 * torch.mean(residual.square())
