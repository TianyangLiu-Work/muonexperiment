"""PyTorch implementation of the Muon optimizer.

This module intentionally stays small. Experiment logic, data generation,
metrics, plotting, and result tables belong in notebooks_torch/*.ipynb so the
experiment remains readable from the notebook itself.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


class MuonTorch(torch.optim.Optimizer):
    """Muon optimizer for matrix parameters.

    Muon replaces a gradient matrix G with its spectral-normalized direction:

        G = U S Vh
        D = U Vh

    Then it applies optional momentum and decoupled weight decay.

    Parameters
    ----------
    params:
        Iterable of matrix parameters. Each optimized parameter must be 2-D.
    lr:
        Learning rate.
    momentum:
        Momentum coefficient for the spectral-normalized update direction.
    weight_decay:
        Decoupled weight decay coefficient.
    variant:
        "exact", "randsvd", or "truncated".
    rank:
        Rank used by "truncated" and target rank used by "randsvd".
    oversample:
        Oversampling dimension for randomized SVD.
    power_iters:
        Power iterations for randomized SVD.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        variant: str = "exact",
        rank: Optional[int] = None,
        oversample: int = 10,
        power_iters: int = 2,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if variant not in {"exact", "randsvd", "truncated"}:
            raise ValueError("variant must be 'exact', 'randsvd', or 'truncated'")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "variant": variant,
            "rank": rank,
            "oversample": oversample,
            "power_iters": power_iters,
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
            variant = group["variant"]
            rank = group["rank"]
            oversample = group["oversample"]
            power_iters = group["power_iters"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError("MuonTorch expects 2-D matrix parameters")

                direction, singular_values = _spectral_direction(
                    p.grad,
                    variant=variant,
                    rank=rank,
                    oversample=oversample,
                    power_iters=power_iters,
                )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(direction)

                if wd > 0:
                    old_p = p.detach().clone()
                p.add_(buf, alpha=-lr)
                if wd > 0:
                    p.add_(old_p, alpha=-lr * wd)

                state["last_singular_values"] = singular_values.detach()
                state["last_update_norm"] = torch.linalg.norm(direction).detach()

        return loss


def _spectral_direction(
    grad: torch.Tensor,
    *,
    variant: str,
    rank: Optional[int],
    oversample: int,
    power_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "randsvd":
        return _randomized_svd_direction(grad, rank, oversample, power_iters)

    U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
    if variant == "truncated":
        k = _effective_rank(grad, rank)
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
    return U @ Vh, S


def _randomized_svd_direction(
    grad: torch.Tensor,
    rank: Optional[int],
    oversample: int,
    power_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = grad.shape
    min_dim = min(rows, cols)
    k = _effective_rank(grad, rank)
    sketch_dim = min(min_dim, k + max(0, oversample))

    omega = torch.randn(
        cols,
        sketch_dim,
        dtype=grad.dtype,
        device=grad.device,
    )
    Y = grad @ omega
    for _ in range(power_iters):
        Y = grad @ (grad.T @ Y)

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ grad
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k] @ Vh[:k, :], S[:k]


def _effective_rank(grad: torch.Tensor, rank: Optional[int]) -> int:
    min_dim = min(grad.shape)
    if rank is None:
        return min_dim
    return max(1, min(int(rank), min_dim))
