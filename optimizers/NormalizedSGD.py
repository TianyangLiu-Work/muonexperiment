"""Normalized gradient baselines for matrix experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class NormalizedSGD(torch.optim.Optimizer):
    """SGD with per-parameter gradient normalization.

    `norm_type="fro"` divides each matrix gradient by its Frobenius norm.
    `norm_type="spectral"` divides each matrix gradient by its operator norm.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        norm_type: str = "fro",
        epsilon: float = 1e-12,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if norm_type not in {"fro", "spectral"}:
            raise ValueError("norm_type must be 'fro' or 'spectral'")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "norm_type": norm_type,
            "epsilon": epsilon,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            norm_type = group["norm_type"]
            eps = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError("NormalizedSGD expects 2-D matrix parameters")

                grad = p.grad
                if wd > 0:
                    grad = grad.add(p, alpha=wd)
                scale = _gradient_scale(grad, norm_type=norm_type, epsilon=eps)
                update = grad / scale

                state = self.state[p]
                if mu > 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = state["momentum_buffer"].mul_(mu).add_(update)

                p.add_(update, alpha=-lr)
                state["last_update_norm"] = torch.linalg.norm(update).detach()

        return loss


def _gradient_scale(grad: torch.Tensor, *, norm_type: str, epsilon: float) -> torch.Tensor:
    if norm_type == "fro":
        return torch.linalg.norm(grad).clamp_min(epsilon)
    return torch.linalg.svdvals(grad)[0].clamp_min(epsilon)
