"""Matrix Sensing problem worker."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from .MatrixConstruction import (
    configure_torch,
    generate_target_matrix,
    make_generator,
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
    noise: float,
    dist: str,
    spectrum: str,
    kappa: float,
    init_scale: float,
    seed: int,
    iters: int,
    epsilon: float,
    step_fn: StepFn | None = None,
    device_type: str = "cpu",
    dtype_name: str = "float64",
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    device = torch.device(device_type)
    dtype = torch_dtype(dtype_name)
    configure_torch(dtype)
    step_fn = step_fn or step

    x_star = generate_target_matrix(
        d,
        rank,
        spectrum=spectrum,
        kappa=kappa,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    m_meas = int(2 * d * rank)
    a = generate_measurements(d, m_meas, dist=dist, seed=seed + 1000, device=device, dtype=dtype)
    y = torch.einsum("mij,ij->m", a, x_star)
    if noise > 0:
        y = y + randn((m_meas,), seed + 2000, device=device, dtype=dtype) * noise

    x = torch.nn.Parameter(randn((d, d), seed + 3000, device=device, dtype=dtype) * init_scale)
    opt = make_optimizer(algo, [x], lr, rank=rank)
    state = {"a": a, "y": y, "x": x, "optimizer": opt}

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
        "problem": "MatrixSensing",
        "algo": algo,
        "d": d,
        "r": rank,
        "m_meas": m_meas,
        "lr": lr,
        "noise": noise,
        "dist": dist,
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
    """One visible optimization step for MatrixSensing."""

    opt = state["optimizer"]
    x = state["x"]
    opt.zero_grad(set_to_none=True)
    loss = matrix_sensing_loss(state["a"], state["y"], x)
    loss.backward()

    grad_norm = float(x.grad.detach().norm().cpu())
    opt.step()
    return float(loss.detach().cpu()), grad_norm


matrix_sensing_step = step


def generate_measurements(
    d: int,
    m_meas: int,
    dist: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    sparsity: float = 0.1,
) -> torch.Tensor:
    g = make_generator(seed, device)
    shape = (m_meas, d, d)

    if dist == "normal":
        a = torch.randn(shape, generator=g, device=device, dtype=dtype)
    elif dist == "uniform":
        a = 2.0 * torch.rand(shape, generator=g, device=device, dtype=dtype) - 1.0
    elif dist == "rademacher":
        a = torch.randint(0, 2, shape, generator=g, device=device).to(dtype)
        a = a.mul(2.0).sub(1.0)
    elif dist == "sparse":
        dense = torch.randn(shape, generator=g, device=device, dtype=dtype)
        mask = torch.rand(shape, generator=g, device=device, dtype=dtype) < sparsity
        a = dense * mask / (sparsity * d * d) ** 0.5
    elif dist == "sphere":
        a = torch.randn(shape, generator=g, device=device, dtype=dtype)
        a = a / a.flatten(1).norm(dim=1).clamp_min(1e-12).view(m_meas, 1, 1)
    else:
        raise ValueError(f"unknown measurement distribution: {dist}")
    return a


def matrix_sensing_loss(a: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    pred = torch.einsum("mij,ij->m", a, x)
    return 0.5 * torch.mean((pred - y) ** 2)
