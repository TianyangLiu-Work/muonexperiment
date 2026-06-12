from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

from ..config import ProblemSpec
from .base import TrainProblem, make_generator


class MNISTPatchClassifierProblem(TrainProblem):
    """Matrix-only convolutional surrogate using shared patch features.

    The first parameter is a matrix applied to every image patch produced by
    `unfold`; the second parameter is a matrix classifier on mean-pooled patch
    features. This preserves matrix-shaped parameters for ExactMuon while
    testing a local-receptive-field neural objective.
    """

    family = "MNISTPatchClassifier"
    diagnostic_a_definition = "full_patch_and_classifier_activation"

    def __init__(self, spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype):
        self.spec = spec
        self.setting = spec.setting
        self.kernel_size = int(spec.rank)
        if self.kernel_size <= 0 or self.kernel_size > 28:
            raise ValueError("MNISTPatchClassifier uses spec.rank as a kernel size in [1, 28]")
        self.stride = max(1, int(spec.d))
        dataset = MNIST(root="data/torchvision", train=True, download=True, transform=transforms.ToTensor())
        generator = make_generator(seed + 7000, device)
        indices = torch.randperm(len(dataset), generator=generator)[: spec.num_samples].tolist()
        images = []
        labels = []
        for index in indices:
            image, label = dataset[index]
            images.append(image)
            labels.append(int(label))
        self.x = torch.stack(images).to(device=device, dtype=dtype)
        self.y = torch.tensor(labels, device=device, dtype=torch.long)

        patch_dim = self.kernel_size * self.kernel_size
        self.patch_weight = torch.nn.Parameter(
            1e-2 * torch.randn((spec.hidden_dim, patch_dim), generator=generator, device=device, dtype=dtype)
        )
        self.classifier_weight = torch.nn.Parameter(
            1e-2 * torch.randn((spec.output_dim, spec.hidden_dim), generator=generator, device=device, dtype=dtype)
        )
        self._batch_seed = seed + 9300
        self._train_indices = torch.arange(self.x.shape[0], device=device)
        self.set_train_step(0)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.patch_weight, self.classifier_weight]

    def set_train_step(self, step: int) -> None:
        batch_size = self.x.shape[0] if self.spec.batch_size is None else max(1, min(int(self.spec.batch_size), self.x.shape[0]))
        generator = make_generator(self._batch_seed + int(step), self.x.device)
        self._train_indices = torch.randperm(self.x.shape[0], generator=generator, device=self.x.device)[:batch_size]

    def patches_for(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        return patches.transpose(1, 2)

    def features_for(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.patches_for(x)
        patch_features = F.relu(patches @ self.patch_weight.T)
        pooled = patch_features.mean(dim=1)
        return patches, pooled

    def logits_for(self, x: torch.Tensor) -> torch.Tensor:
        _, pooled = self.features_for(x)
        return pooled @ self.classifier_weight.T

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
            patches, pooled = self.features_for(self.x)
            return [patches.reshape(-1, patches.shape[-1]).detach(), pooled.detach()]
