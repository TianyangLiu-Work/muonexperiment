"""Importable single-run worker for the E01 matrix sensing benchmark.

The notebook owns the experiment grid, metrics tables, and plots. This module
exists because multiprocessing with the spawn start method requires worker
functions to be importable from a normal Python module.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from .optimizers import MuonTorch, ShampooTorch


for thread_env in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(thread_env, "1")


def run_spec(
    spec: dict[str, Any],
) -> tuple[tuple[str, int, int], dict[str, Any], dict[str, list[float]]]:
    """Run one `(algo, d, seed)` matrix sensing job."""

    row, trajectory = run_ms_once(**spec)
    key = (spec["algo"], spec["d"], spec["seed"])
    return key, row, trajectory


def run_ms_once(
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
    device_type: str = "cpu",
    dtype_name: str = "float64",
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    device = torch.device(device_type)
    dtype = getattr(torch, dtype_name)
    torch.set_default_dtype(dtype)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

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
    a = generate_measurements(
        d,
        m_meas,
        dist=dist,
        seed=seed + 1000,
        device=device,
        dtype=dtype,
    )
    y = torch.einsum("mij,ij->m", a, x_star)
    if noise > 0:
        y = y + randn((m_meas,), seed + 2000, device=device, dtype=dtype) * noise

    x0 = randn((d, d), seed + 3000, device=device, dtype=dtype) * init_scale
    x = torch.nn.Parameter(x0)
    opt = make_optimizer(algo, [x], lr, rank=rank)

    losses: list[float] = []
    grad_norms: list[float] = []
    k_epsilon = None

    sync_device(device)
    t0 = time.time()
    for step in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = matrix_sensing_loss(a, y, x)
        loss.backward()

        grad_norm = float(x.grad.detach().norm().cpu())
        opt.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        grad_norms.append(grad_norm)

        if k_epsilon is None and loss_value <= epsilon:
            k_epsilon = step + 1

    sync_device(device)
    elapsed = time.time() - t0
    if k_epsilon is None:
        k_epsilon = iters + 1

    row = {
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


def make_optimizer(
    algo: str,
    params,
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
        return MuonTorch(params, lr=lr, momentum=0.9, variant="newton_schulz", ns_steps=5)
    if algo in {"Muon-Exact", "MuonExact"}:
        return MuonTorch(params, lr=lr, momentum=0.9, variant="exact")
    if algo == "Muon-RandSVD":
        return MuonTorch(params, lr=lr, momentum=0.9, variant="randsvd", rank=rank)
    if algo == "Muon-Trunc":
        return MuonTorch(params, lr=lr, momentum=0.9, variant="truncated", rank=rank)
    if algo == "Shampoo":
        return ShampooTorch(params, lr=lr, beta2=0.9, epsilon=1e-8)
    if algo == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if algo == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown algo: {algo}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
