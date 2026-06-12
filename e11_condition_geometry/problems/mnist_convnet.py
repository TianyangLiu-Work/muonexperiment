from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

from ..config import ProblemSpec
from .base import TrainProblem, make_generator


class MNISTConvNetProblem(TrainProblem):
    """Small true Conv2d MNIST model with matrix-view Muon diagnostics."""

    family = "MNISTConvNet"
    diagnostic_a_definition = "full_conv_patch_and_classifier_activation"

    def __init__(self, spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype):
        self.spec = spec
        self.setting = spec.setting
        self.kernel_size = int(spec.rank)
        if self.kernel_size <= 0 or self.kernel_size > 28:
            raise ValueError("MNISTConvNet uses spec.rank as a kernel size in [1, 28]")
        self.stride = max(1, int(spec.d))
        dataset = MNIST(root="data/torchvision", train=True, download=True, transform=transforms.ToTensor())
        generator = make_generator(seed + 8000, device)
        indices = torch.randperm(len(dataset), generator=generator)[: spec.num_samples].tolist()
        images = []
        labels = []
        for index in indices:
            image, label = dataset[index]
            images.append(image)
            labels.append(int(label))
        self.x = torch.stack(images).to(device=device, dtype=dtype)
        self.y = torch.tensor(labels, device=device, dtype=torch.long)

        self.conv_weight = torch.nn.Parameter(
            1e-2
            * torch.randn(
                (spec.hidden_dim, 1, self.kernel_size, self.kernel_size),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
        self.classifier_weight = torch.nn.Parameter(
            1e-2 * torch.randn((spec.output_dim, spec.hidden_dim), generator=generator, device=device, dtype=dtype)
        )
        self._batch_seed = seed + 9400
        self._train_indices = torch.arange(self.x.shape[0], device=device)
        self.set_train_step(0)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.conv_weight, self.classifier_weight]

    def set_train_step(self, step: int) -> None:
        batch_size = self.x.shape[0] if self.spec.batch_size is None else max(1, min(int(self.spec.batch_size), self.x.shape[0]))
        generator = make_generator(self._batch_seed + int(step), self.x.device)
        self._train_indices = torch.randperm(self.x.shape[0], generator=generator, device=self.x.device)[:batch_size]

    def conv_features_for(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(F.conv2d(x, self.conv_weight, stride=self.stride))

    def pooled_features_for(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_features_for(x)
        return features.mean(dim=(2, 3))

    def logits_for(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooled_features_for(x) @ self.classifier_weight.T

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
            patches = F.unfold(self.x, kernel_size=self.kernel_size, stride=self.stride).transpose(1, 2)
            pooled = self.pooled_features_for(self.x)
            return [patches.reshape(-1, patches.shape[-1]).detach(), pooled.detach()]
