"""Muon optimizer variants for matrix parameters."""

from __future__ import annotations

from typing import Iterable, Optional

import torch


class MuonExact(torch.optim.Optimizer):
    """Muon optimizer with exact and approximate polar directions.

    The `exact` variant replaces a matrix gradient `G = U S V^T` with the polar
    direction `U V^T`. The other variants are kept for benchmark parity.
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
        ns_steps: int = 5,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if variant not in {"newton_schulz", "exact", "randsvd", "truncated"}:
            raise ValueError(
                "variant must be 'newton_schulz', 'exact', 'randsvd', or 'truncated'"
            )

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "variant": variant,
            "rank": rank,
            "oversample": oversample,
            "power_iters": power_iters,
            "ns_steps": ns_steps,
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
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError("MuonExact expects 2-D matrix parameters")

                direction, singular_values = _spectral_direction(
                    p.grad,
                    variant=variant,
                    rank=rank,
                    oversample=oversample,
                    power_iters=power_iters,
                    ns_steps=ns_steps,
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
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "newton_schulz":
        return _newton_schulz_direction(grad, ns_steps), grad.new_empty(0)
    if variant == "randsvd":
        return _randomized_svd_direction(grad, rank, oversample, power_iters)

    u, s, vh = torch.linalg.svd(grad, full_matrices=False)
    if variant == "truncated":
        k = _effective_rank(grad, rank)
        u = u[:, :k]
        s = s[:k]
        vh = vh[:k, :]
    return u @ vh, s


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

    omega = torch.randn(cols, sketch_dim, dtype=grad.dtype, device=grad.device)
    y = grad @ omega
    for _ in range(power_iters):
        y = grad @ (grad.T @ y)

    q, _ = torch.linalg.qr(y, mode="reduced")
    b = q.T @ grad
    ub, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ ub
    return u[:, :k] @ vh[:k, :], s[:k]


def _effective_rank(grad: torch.Tensor, rank: Optional[int]) -> int:
    min_dim = min(grad.shape)
    if rank is None:
        return min_dim
    return max(1, min(int(rank), min_dim))


def _newton_schulz_direction(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate `U V^T` from `grad = U S V^T`."""
    if steps <= 0:
        raise ValueError("ns_steps must be positive")

    x = grad / torch.linalg.norm(grad).clamp_min(torch.finfo(grad.dtype).eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        correction = b * gram + c * (gram @ gram)
        x = a * x + correction @ x

    if transposed:
        x = x.T
    return x

