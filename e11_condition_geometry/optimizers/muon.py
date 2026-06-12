from __future__ import annotations

from collections.abc import Iterable

import torch

from ..diagnostics import matrix_view


class ExactMuon(torch.optim.Optimizer):
    """Exact polar-direction Muon for matrix-shaped parameter views."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-2):
        if lr <= 0:
            raise ValueError("lr must be positive")
        super().__init__(params, {"lr": float(lr)})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad_matrix = matrix_view(param.grad)
                u, _, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
                polar_update = (u @ vh).reshape_as(param)
                param.add_(polar_update, alpha=-lr)
        return loss
