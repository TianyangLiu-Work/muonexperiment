"""Autograd definition for the Matrix Sensing problem."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .MatrixConstruction import generate_target_matrix, make_generator, randn


@dataclass(frozen=True)
class MatrixSensingProblem:
    target: torch.Tensor
    measurements: torch.Tensor
    observations: torch.Tensor

    @property
    def d(self) -> int:
        return int(self.target.shape[0])

    @property
    def m_meas(self) -> int:
        return int(self.measurements.shape[0])

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return matrix_sensing_loss(self.measurements, self.observations, x)


def make_matrix_sensing_problem(
    d: int,
    rank: int,
    *,
    noise: float,
    dist: str,
    spectrum: str,
    kappa: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    m_multiplier: float | None = None,
    m_meas: int | None = None,
) -> MatrixSensingProblem:
    target = generate_target_matrix(
        d,
        rank,
        spectrum=spectrum,
        kappa=kappa,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    if m_meas is None:
        multiplier = 2.0 if m_multiplier is None else float(m_multiplier)
        m_meas = int(multiplier * d * rank)
    m_meas = max(1, int(m_meas))
    measurements = generate_measurements(
        d,
        m_meas,
        dist=dist,
        seed=seed + 1000,
        device=device,
        dtype=dtype,
    )
    observations = torch.einsum("mij,ij->m", measurements, target)
    if noise > 0:
        observations = observations + randn(
            (m_meas,),
            seed + 2000,
            device=device,
            dtype=dtype,
        ) * noise
    return MatrixSensingProblem(target, measurements, observations)


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
        measurements = torch.randn(shape, generator=g, device=device, dtype=dtype)
    elif dist == "uniform":
        measurements = 2.0 * torch.rand(shape, generator=g, device=device, dtype=dtype) - 1.0
    elif dist == "rademacher":
        measurements = torch.randint(0, 2, shape, generator=g, device=device).to(dtype)
        measurements = measurements.mul(2.0).sub(1.0)
    elif dist == "sparse":
        dense = torch.randn(shape, generator=g, device=device, dtype=dtype)
        mask = torch.rand(shape, generator=g, device=device, dtype=dtype) < sparsity
        measurements = dense * mask / (sparsity * d * d) ** 0.5
    elif dist == "sphere":
        measurements = torch.randn(shape, generator=g, device=device, dtype=dtype)
        measurements = measurements / measurements.flatten(1).norm(dim=1).clamp_min(
            1e-12
        ).view(m_meas, 1, 1)
    else:
        raise ValueError(f"unknown measurement distribution: {dist}")
    return measurements


def matrix_sensing_loss(
    measurements: torch.Tensor,
    observations: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    pred = torch.einsum("mij,ij->m", measurements, x)
    return 0.5 * torch.mean((pred - observations) ** 2)
