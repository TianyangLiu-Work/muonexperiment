"""Tensor construction helpers for matrix problems."""

from __future__ import annotations

import torch


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

    singular_values = torch.zeros(d, device=device, dtype=dtype)
    if spectrum == "hard-cutoff":
        singular_values[:rank] = 1.0
    elif spectrum == "polynomial-decay":
        idx = torch.arange(1, rank + 1, device=device, dtype=dtype)
        singular_values[:rank] = idx.pow(-1.0)
        singular_values[:rank] /= singular_values[0]
    elif spectrum == "exponential-decay":
        idx = torch.arange(rank, device=device, dtype=dtype)
        singular_values[:rank] = torch.exp(-0.5 * idx)
    else:
        raise ValueError(f"unknown spectrum: {spectrum}")

    if kappa > 1.0 and rank > 1:
        singular_values[:rank] = torch.linspace(
            1.0,
            1.0 / kappa,
            rank,
            device=device,
            dtype=dtype,
        )

    return u @ torch.diag(singular_values) @ v.T
