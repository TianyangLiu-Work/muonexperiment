from __future__ import annotations

import torch

from ..config import ProblemSpec
from .base import TrainProblem, make_low_rank_target, randn


class MatrixSensingProblem(TrainProblem):
    family = "MatrixSensing"
    diagnostic_a_definition = "measurement_operator_proxy"

    def __init__(self, spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype):
        self.spec = spec
        self.setting = spec.setting
        self.target = make_low_rank_target(spec.d, spec.rank, spec.kappa, seed, device=device, dtype=dtype)
        m_meas = int(spec.measurement_multiplier * spec.d * spec.rank)
        measurements = randn((m_meas, spec.d, spec.d), seed + 1000, device=device, dtype=dtype)
        self.measurements = measurements / (spec.d**0.5)
        self.observations = torch.einsum("mij,ij->m", self.measurements, self.target)
        self.variable = torch.nn.Parameter(1e-2 * randn((spec.d, spec.d), seed + 23, device=device, dtype=dtype))
        self.measurement_matrix = self.measurements.reshape(m_meas, spec.d * spec.d)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.variable]

    def loss(self) -> torch.Tensor:
        pred = torch.einsum("mij,ij->m", self.measurements, self.variable)
        return 0.5 * torch.mean((pred - self.observations).square())

    def set_train_step(self, step: int) -> None:
        return None

    def recovery_error(self) -> float:
        numerator = torch.linalg.norm(self.variable.detach() - self.target)
        denominator = torch.linalg.norm(self.target).clamp_min(torch.finfo(self.target.dtype).eps)
        return float((numerator / denominator).detach().cpu())

    def activation_matrices(self) -> list[torch.Tensor]:
        return [self.measurement_matrix.detach()]
