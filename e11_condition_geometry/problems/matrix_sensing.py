from __future__ import annotations

import torch

from ..config import ProblemSpec
from .base import TrainProblem, make_generator, make_low_rank_target, randn


class MatrixSensingProblem(TrainProblem):
    family = "MatrixSensing"
    diagnostic_a_definition = "measurement_operator_proxy"

    def __init__(self, spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype):
        self.spec = spec
        self.setting = spec.setting
        self.target = make_low_rank_target(spec.d, spec.rank, spec.kappa, seed, device=device, dtype=dtype)
        m_meas = int(spec.measurement_multiplier * spec.d * spec.rank)
        self.num_train_examples = m_meas
        measurements = randn((m_meas, spec.d, spec.d), seed + 1000, device=device, dtype=dtype)
        self.measurements = measurements / (spec.d**0.5)
        self.clean_observations = torch.einsum("mij,ij->m", self.measurements, self.target)
        observation_scale = self.clean_observations.square().mean().sqrt().clamp_min(torch.finfo(dtype).eps)
        noise = float(spec.noise_std) * observation_scale * randn(
            self.clean_observations.shape,
            seed + 2017,
            device=device,
            dtype=dtype,
        )
        self.observations = self.clean_observations + noise
        self.variable = torch.nn.Parameter(1e-2 * randn((spec.d, spec.d), seed + 23, device=device, dtype=dtype))
        self.measurement_matrix = self.measurements.reshape(m_meas, spec.d * spec.d)
        self._batch_seed = seed + 7100
        self._train_indices = torch.arange(self.measurements.shape[0], device=device)
        self.set_train_step(0)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.variable]

    def loss(self) -> torch.Tensor:
        measurements = self.measurements[self._train_indices]
        observations = self.observations[self._train_indices]
        pred = torch.einsum("mij,ij->m", measurements, self.variable)
        return 0.5 * torch.mean((pred - observations).square())

    def set_train_step(self, step: int) -> None:
        sample_count = self.measurements.shape[0]
        batch_size = sample_count if self.spec.batch_size is None else max(1, min(int(self.spec.batch_size), sample_count))
        generator = make_generator(self._batch_seed + int(step), self.measurements.device)
        self._train_indices = torch.randperm(sample_count, generator=generator, device=self.measurements.device)[:batch_size]

    def recovery_error(self) -> float:
        numerator = torch.linalg.norm(self.variable.detach() - self.target)
        denominator = torch.linalg.norm(self.target).clamp_min(torch.finfo(self.target.dtype).eps)
        return float((numerator / denominator).detach().cpu())

    def activation_matrices(self) -> list[torch.Tensor]:
        return [self.measurement_matrix.detach()]
