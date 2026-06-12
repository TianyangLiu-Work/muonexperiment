from __future__ import annotations

from typing import Protocol

import torch


class TrainProblem(Protocol):
    family: str
    setting: str
    diagnostic_a_definition: str

    def parameters(self) -> list[torch.nn.Parameter]:
        ...

    def loss(self) -> torch.Tensor:
        ...

    def set_train_step(self, step: int) -> None:
        ...

    def recovery_error(self) -> float:
        ...

    def activation_matrices(self) -> list[torch.Tensor]:
        ...


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def randn(shape: tuple[int, ...], seed: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, generator=make_generator(seed, device), device=device, dtype=dtype)


def make_low_rank_target(
    d: int,
    rank: int,
    kappa: float,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    left, _ = torch.linalg.qr(randn((d, rank), seed, device=device, dtype=dtype), mode="reduced")
    right, _ = torch.linalg.qr(randn((d, rank), seed + 17, device=device, dtype=dtype), mode="reduced")
    singular = torch.logspace(
        0.0,
        -float(torch.log10(torch.tensor(float(kappa), dtype=dtype))),
        rank,
        device=device,
        dtype=dtype,
    )
    return left @ torch.diag(singular) @ right.T
