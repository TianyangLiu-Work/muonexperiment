from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

from ..config import ProblemSpec
from .base import TrainProblem, make_generator


class DeepMNISTMLPProblem(TrainProblem):
    family = "DeepMNISTMLP"
    diagnostic_a_definition = "full_layer_input_activation"

    def __init__(self, spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype):
        if spec.num_factors < 3:
            raise ValueError("DeepMNISTMLP requires at least three matrix factors")
        self.spec = spec
        self.setting = spec.setting
        dataset = MNIST(root="data/torchvision", train=True, download=True, transform=transforms.ToTensor())
        generator = make_generator(seed + 5000, device)
        indices = torch.randperm(len(dataset), generator=generator)[: spec.num_samples].tolist()
        xs = []
        ys = []
        for index in indices:
            image, label = dataset[index]
            xs.append(image.reshape(-1))
            ys.append(int(label))
        self.x = torch.stack(xs).to(device=device, dtype=dtype)
        self.y = torch.tensor(ys, device=device, dtype=torch.long)

        shapes = [(spec.hidden_dim, spec.input_dim)]
        shapes.extend((spec.hidden_dim, spec.hidden_dim) for _ in range(spec.num_factors - 2))
        shapes.append((spec.output_dim, spec.hidden_dim))
        self.weights = [
            torch.nn.Parameter(1e-2 * torch.randn(shape, generator=generator, device=device, dtype=dtype))
            for shape in shapes
        ]
        self._batch_seed = seed + 9100
        self._train_indices = torch.arange(self.x.shape[0], device=device)
        self.set_train_step(0)

    def parameters(self) -> list[torch.nn.Parameter]:
        return self.weights

    def set_train_step(self, step: int) -> None:
        batch_size = self.x.shape[0] if self.spec.batch_size is None else max(1, min(int(self.spec.batch_size), self.x.shape[0]))
        generator = make_generator(self._batch_seed + int(step), self.x.device)
        self._train_indices = torch.randperm(self.x.shape[0], generator=generator, device=self.x.device)[:batch_size]

    def hidden_activations_for(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = [x]
        current = x
        for weight in self.weights[:-1]:
            current = F.relu(current @ weight.T)
            activations.append(current)
        return activations

    def hidden_activations(self) -> list[torch.Tensor]:
        return self.hidden_activations_for(self.x[self._train_indices])

    def logits_for(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        for weight in self.weights[:-1]:
            current = F.relu(current @ weight.T)
        return current @ self.weights[-1].T

    def logits(self) -> torch.Tensor:
        return self.logits_for(self.x[self._train_indices])

    def loss(self) -> torch.Tensor:
        return F.cross_entropy(self.logits(), self.y[self._train_indices])

    def recovery_error(self) -> float:
        with torch.no_grad():
            pred = self.logits_for(self.x).argmax(dim=1)
            return float((pred != self.y).to(torch.float64).mean().cpu())

    def activation_matrices(self) -> list[torch.Tensor]:
        with torch.no_grad():
            return [activation.detach() for activation in self.hidden_activations_for(self.x)]
