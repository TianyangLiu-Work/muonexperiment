"""Autograd definition for the Matrix Factorization problem."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .MatrixConstruction import generate_target_matrix


@dataclass(frozen=True)
class MatrixFactorizationProblem:
    target: torch.Tensor
    rank: int
    factor_rank: int

    @property
    def d(self) -> int:
        return int(self.target.shape[0])

    def loss(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return matrix_factorization_loss(left, right, self.target)


def make_matrix_factorization_problem(
    d: int,
    rank: int,
    *,
    spectrum: str,
    kappa: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    factor_rank: int | None = None,
) -> MatrixFactorizationProblem:
    target = generate_target_matrix(
        d,
        rank,
        spectrum=spectrum,
        kappa=kappa,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    return MatrixFactorizationProblem(
        target=target,
        rank=rank,
        factor_rank=rank if factor_rank is None else factor_rank,
    )


def matrix_factorization_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    residual = left @ right.T - target
    return 0.5 * torch.mean(residual.square())
