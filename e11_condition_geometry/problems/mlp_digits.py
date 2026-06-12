from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

from ..config import ProblemSpec
from .base import TrainProblem, make_generator


class SmallMLPDigitsProblem(TrainProblem):
    family = "SmallMLPDigits"
    diagnostic_a_definition = "full_layer_input_activation"

    def __init__(self, spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype):
        self.spec = spec
        self.setting = spec.setting
        data = load_digits()
        x = torch.tensor(data.data, device=device, dtype=dtype) / 16.0
        y = torch.tensor(data.target, device=device, dtype=torch.long)
        generator = make_generator(seed + 3000, device)
        perm = torch.randperm(x.shape[0], generator=generator, device=device)[: spec.num_samples]
        self.x = x[perm]
        self.y = y[perm]
        self.w1 = torch.nn.Parameter(1e-2 * torch.randn((spec.hidden_dim, spec.input_dim), generator=generator, device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(1e-2 * torch.randn((spec.output_dim, spec.hidden_dim), generator=generator, device=device, dtype=dtype))
        self._last_hidden: torch.Tensor | None = None
        self._batch_seed = seed + 7100
        self._train_indices = torch.arange(self.x.shape[0], device=device)
        self._train_noise = torch.zeros_like(self.x)
        self.set_train_step(0)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.w1, self.w2]

    def set_train_step(self, step: int) -> None:
        batch_size = self.x.shape[0] if self.spec.batch_size is None else max(1, min(int(self.spec.batch_size), self.x.shape[0]))
        generator = make_generator(self._batch_seed + int(step), self.x.device)
        self._train_indices = torch.randperm(self.x.shape[0], generator=generator, device=self.x.device)[:batch_size]
        if self.spec.noise_std > 0.0:
            scale = self.x[self._train_indices].square().mean().sqrt().clamp_min(torch.finfo(self.x.dtype).eps)
            self._train_noise = float(self.spec.noise_std) * scale * torch.randn(
                (batch_size, self.x.shape[1]),
                generator=generator,
                device=self.x.device,
                dtype=self.x.dtype,
            )
        else:
            self._train_noise = torch.zeros((batch_size, self.x.shape[1]), device=self.x.device, dtype=self.x.dtype)

    def logits_for(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(x @ self.w1.T)
        self._last_hidden = hidden
        return hidden @ self.w2.T

    def logits(self) -> torch.Tensor:
        return self.logits_for(self.x[self._train_indices] + self._train_noise)

    def loss(self) -> torch.Tensor:
        return F.cross_entropy(self.logits(), self.y[self._train_indices])

    def recovery_error(self) -> float:
        with torch.no_grad():
            pred = self.logits_for(self.x).argmax(dim=1)
            return float((pred != self.y).to(torch.float64).mean().cpu())

    def activation_matrices(self) -> list[torch.Tensor]:
        with torch.no_grad():
            hidden = F.relu(self.x @ self.w1.detach().T)
        return [self.x.detach(), hidden.detach()]
