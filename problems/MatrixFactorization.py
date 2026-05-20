"""Autograd definition for the Matrix Factorization problem."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .MatrixConstruction import generate_target_matrix, randn


DEFAULT_NUM_FACTORS = 10


@dataclass(frozen=True)
class MatrixFactorizationProblem:
    target: torch.Tensor
    rank: int
    factor_rank: int
    num_factors: int = DEFAULT_NUM_FACTORS

    @property
    def d(self) -> int:
        return int(self.target.shape[0])

    def estimate(self, *factors: torch.Tensor) -> torch.Tensor:
        return matrix_factorization_product(*factors)

    def loss(self, *factors: torch.Tensor) -> torch.Tensor:
        return matrix_factorization_loss(*factors, self.target)


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
    num_factors: int = DEFAULT_NUM_FACTORS,
) -> MatrixFactorizationProblem:
    if num_factors < 2:
        raise ValueError("num_factors must be at least 2")
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
        num_factors=int(num_factors),
    )


def factor_chain_shapes(
    d: int,
    factor_rank: int,
    *,
    num_factors: int = DEFAULT_NUM_FACTORS,
) -> list[tuple[int, int]]:
    if num_factors < 2:
        raise ValueError("num_factors must be at least 2")
    if num_factors == 2:
        return [(int(d), int(factor_rank)), (int(factor_rank), int(d))]
    return (
        [(int(d), int(factor_rank))]
        + [(int(factor_rank), int(factor_rank)) for _ in range(int(num_factors) - 2)]
        + [(int(factor_rank), int(d))]
    )


def initialize_factor_chain(
    d: int,
    factor_rank: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    num_factors: int = DEFAULT_NUM_FACTORS,
    left_scale: float | torch.Tensor = 1e-2,
    right_scale: float | torch.Tensor = 1e-2,
    middle_noise_scale: float = 0.0,
) -> list[torch.Tensor]:
    """Initialise a deep low-rank factor chain.

    The first and last factors preserve the old two-factor scale semantics.
    Middle factors start near identity so moving from two to ten trainable
    matrices does not collapse the initial product scale.
    """
    shapes = factor_chain_shapes(d, factor_rank, num_factors=num_factors)
    factors: list[torch.Tensor] = []

    first = randn(shapes[0], seed, device=device, dtype=dtype)
    factors.append(first * _column_scale(left_scale, device, dtype))

    for idx, shape in enumerate(shapes[1:-1], start=1):
        eye = torch.eye(shape[0], shape[1], device=device, dtype=dtype)
        if middle_noise_scale > 0:
            noise = randn(shape, seed + 100 * idx, device=device, dtype=dtype)
            eye = eye + float(middle_noise_scale) * noise
        factors.append(eye)

    final_source = randn((int(d), int(factor_rank)), seed + 17, device=device, dtype=dtype)
    final = final_source * _column_scale(right_scale, device, dtype)
    factors.append(final.T.contiguous())
    return factors


def matrix_factorization_product(*factors: torch.Tensor) -> torch.Tensor:
    if len(factors) < 2:
        raise ValueError("matrix factorization requires at least two factors")
    if (
        len(factors) == 2
        and factors[0].ndim == 2
        and factors[1].ndim == 2
        and factors[0].shape == factors[1].shape
    ):
        return factors[0] @ factors[1].T

    result = factors[0]
    for factor in factors[1:]:
        result = result @ factor
    return result


def matrix_factorization_loss(*factors_and_target: torch.Tensor) -> torch.Tensor:
    if len(factors_and_target) < 3:
        raise ValueError("pass at least two factors followed by target")
    *factors, target = factors_and_target
    residual = matrix_factorization_product(*factors) - target
    return 0.5 * torch.mean(residual.square())


def _column_scale(
    scale: float | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float | torch.Tensor:
    if torch.is_tensor(scale):
        return scale.to(device=device, dtype=dtype).view(1, -1)
    return float(scale)
