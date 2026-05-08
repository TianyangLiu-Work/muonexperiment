"""Small matrix Shampoo optimizer."""

from __future__ import annotations

from typing import Iterable

import torch


class Shampoo(torch.optim.Optimizer):
    """Matrix-only Shampoo optimizer.

    This is scoped to 2-D parameters, which is sufficient for the matrix
    experiments in this repository.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        beta2: float = 0.9,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        defaults = {
            "lr": lr,
            "beta2": beta2,
            "epsilon": epsilon,
            "weight_decay": weight_decay,
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
            beta2 = group["beta2"]
            eps = group["epsilon"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError("Shampoo expects 2-D matrix parameters")

                grad = p.grad
                rows, cols = grad.shape
                state = self.state[p]
                if "row_precond" not in state:
                    state["row_precond"] = (
                        torch.eye(rows, dtype=grad.dtype, device=grad.device) * eps
                    )
                    state["col_precond"] = (
                        torch.eye(cols, dtype=grad.dtype, device=grad.device) * eps
                    )

                row_precond = state["row_precond"]
                col_precond = state["col_precond"]
                row_precond.mul_(beta2).add_(grad @ grad.T, alpha=1.0 - beta2)
                col_precond.mul_(beta2).add_(grad.T @ grad, alpha=1.0 - beta2)

                row_inv_quarter = _matrix_inverse_root(row_precond, root=4, epsilon=eps)
                col_inv_quarter = _matrix_inverse_root(col_precond, root=4, epsilon=eps)
                update = row_inv_quarter @ grad @ col_inv_quarter

                if wd > 0:
                    old_p = p.detach().clone()
                p.add_(update, alpha=-lr)
                if wd > 0:
                    p.add_(old_p, alpha=-lr * wd)

                state["last_update_norm"] = torch.linalg.norm(update).detach()

        return loss


def _matrix_inverse_root(
    matrix: torch.Tensor,
    *,
    root: int,
    epsilon: float,
) -> torch.Tensor:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    inv_root = eigvals.clamp_min(epsilon).pow(-1.0 / root)
    return (eigvecs * inv_root.unsqueeze(0)) @ eigvecs.T

