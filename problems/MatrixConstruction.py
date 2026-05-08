"""Reusable construction and runtime helpers for matrix problems."""

from __future__ import annotations

import os
from typing import Iterable

import torch

from optimizers import MuonExact, Shampoo


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


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        return torch.Generator().manual_seed(int(seed))


def randn(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, generator=make_generator(seed, device), device=device, dtype=dtype)


def generate_target_matrix(
    d: int,
    rank: int,
    spectrum: str,
    kappa: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    u, _ = torch.linalg.qr(randn((d, d), seed, device=device, dtype=dtype))
    v, _ = torch.linalg.qr(randn((d, d), seed + 17, device=device, dtype=dtype))

    s = torch.zeros(d, device=device, dtype=dtype)
    if spectrum == "hard-cutoff":
        s[:rank] = 1.0
    elif spectrum == "polynomial-decay":
        idx = torch.arange(1, rank + 1, device=device, dtype=dtype)
        s[:rank] = idx.pow(-1.0)
        s[:rank] /= s[0]
    elif spectrum == "exponential-decay":
        idx = torch.arange(rank, device=device, dtype=dtype)
        s[:rank] = torch.exp(-0.5 * idx)
    else:
        raise ValueError(f"unknown spectrum: {spectrum}")

    if kappa > 1.0 and rank > 1:
        s[:rank] = torch.linspace(1.0, 1.0 / kappa, rank, device=device, dtype=dtype)

    return u @ torch.diag(s) @ v.T


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


set_single_thread_env()

