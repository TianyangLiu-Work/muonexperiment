from __future__ import annotations

import torch

from ..config import ProblemSpec
from .base import TrainProblem, make_generator, make_low_rank_target, randn


class MatrixFactorizationInputProblem(TrainProblem):
    family = "MatrixFactorizationInput"
    diagnostic_a_definition = "downstream_activation_product"

    def __init__(self, spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype):
        self.spec = spec
        self.setting = spec.setting
        self.target = make_low_rank_target(spec.d, spec.rank, spec.kappa, seed, device=device, dtype=dtype)
        input_cols = int(spec.input_columns_multiplier * spec.d)
        self.num_train_examples = input_cols
        self.input_matrix = randn((spec.d, input_cols), seed + 1009, device=device, dtype=dtype) / (spec.d**0.5)
        shapes = self._factor_shapes(spec.d, spec.rank, spec.num_factors)
        factors: list[torch.nn.Parameter] = []
        for layer, shape in enumerate(shapes):
            if layer == 0:
                value = 1e-2 * randn(shape, seed + 10 * layer, device=device, dtype=dtype)
            elif layer == len(shapes) - 1:
                value = 1e-2 * randn(shape, seed + 10 * layer, device=device, dtype=dtype)
            else:
                value = torch.eye(shape[0], shape[1], device=device, dtype=dtype)
            factors.append(torch.nn.Parameter(value))
        self.factors = factors
        self.clean_target_output = self.target @ self.input_matrix
        target_scale = self.clean_target_output.square().mean().sqrt().clamp_min(torch.finfo(dtype).eps)
        noise = float(spec.noise_std) * target_scale * randn(
            self.clean_target_output.shape,
            seed + 2017,
            device=device,
            dtype=dtype,
        )
        self.target_output = self.clean_target_output + noise
        self._batch_seed = seed + 7100
        self._train_indices = torch.arange(self.input_matrix.shape[1], device=device)
        self.set_train_step(0)

    @staticmethod
    def _factor_shapes(d: int, rank: int, num_factors: int) -> list[tuple[int, int]]:
        return [(d, rank)] + [(rank, rank) for _ in range(num_factors - 2)] + [(rank, d)]

    def parameters(self) -> list[torch.nn.Parameter]:
        return self.factors

    def product(self) -> torch.Tensor:
        result = self.factors[0]
        for factor in self.factors[1:]:
            result = result @ factor
        return result

    def output(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        input_matrix = self.input_matrix if indices is None else self.input_matrix[:, indices]
        return self.product() @ input_matrix

    def loss(self) -> torch.Tensor:
        residual = self.output(self._train_indices) - self.target_output[:, self._train_indices]
        return 0.5 * torch.mean(residual.square())

    def set_train_step(self, step: int) -> None:
        sample_count = self.input_matrix.shape[1]
        batch_size = sample_count if self.spec.batch_size is None else max(1, min(int(self.spec.batch_size), sample_count))
        generator = make_generator(self._batch_seed + int(step), self.input_matrix.device)
        self._train_indices = torch.randperm(sample_count, generator=generator, device=self.input_matrix.device)[:batch_size]

    def recovery_error(self) -> float:
        numerator = torch.linalg.norm(self.product().detach() - self.target)
        denominator = torch.linalg.norm(self.target).clamp_min(torch.finfo(self.target.dtype).eps)
        return float((numerator / denominator).detach().cpu())

    def activation_matrices(self) -> list[torch.Tensor]:
        activations: list[torch.Tensor] = []
        for idx in range(len(self.factors)):
            downstream = self.input_matrix
            for factor in reversed(self.factors[idx + 1 :]):
                downstream = factor.detach() @ downstream
            activations.append(downstream)
        return activations
