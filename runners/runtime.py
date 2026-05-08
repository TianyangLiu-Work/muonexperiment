"""Runtime helpers for matrix experiment runners."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any, Callable

import torch

from optimizers import MuonExact, Shampoo


StepFn = Callable[[dict[str, Any]], tuple[float, float]]


THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def set_single_thread_env() -> None:
    for thread_env in THREAD_ENV_VARS:
        os.environ.setdefault(thread_env, "1")


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


def make_optimizer(
    algo: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    rank: int,
) -> torch.optim.Optimizer:
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
    if algo == "Muon-RandSVD":
        return MuonExact(params, lr=lr, momentum=0.9, variant="randsvd", rank=rank)
    if algo == "Muon-Trunc":
        return MuonExact(params, lr=lr, momentum=0.9, variant="truncated", rank=rank)
    if algo == "Shampoo":
        return Shampoo(params, lr=lr, beta2=0.9, epsilon=1e-8)
    if algo == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if algo == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown algo: {algo}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_steps(
    state: dict[str, Any],
    step_fn: StepFn,
    iters: int,
    epsilon: float,
    device: torch.device,
) -> tuple[list[float], list[float], int, float]:
    losses: list[float] = []
    grad_norms: list[float] = []
    k_epsilon = None

    sync_device(device)
    t0 = time.time()
    for step_idx in range(iters):
        loss_value, grad_norm = step_fn(state)
        losses.append(loss_value)
        grad_norms.append(grad_norm)
        if k_epsilon is None and loss_value <= epsilon:
            k_epsilon = step_idx + 1

    sync_device(device)
    elapsed = time.time() - t0
    return losses, grad_norms, k_epsilon or iters + 1, elapsed


set_single_thread_env()
