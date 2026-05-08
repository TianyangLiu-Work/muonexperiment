"""PyTorch optimizer implementations for notebook-first matrix experiments.

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
        "newton_schulz", "exact", "randsvd", or "truncated".
    rank:
        Rank used by "truncated" and target rank used by "randsvd".
    oversample:
        Oversampling dimension for randomized SVD.
    power_iters:
        Power iterations for randomized SVD.
    ns_steps:
        Number of Newton-Schulz iterations for the standard approximate Muon.
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
                    raise ValueError("MuonTorch expects 2-D matrix parameters")

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


class ShampooTorch(torch.optim.Optimizer):
    """Small matrix Shampoo optimizer.

    This is intentionally scoped to 2-D matrix parameters, which is what the
    matrix sensing notebooks optimize. It keeps row and column second-moment
    preconditioners and applies:

        update = L^{-1/4} G R^{-1/4}

    where L accumulates G G^T and R accumulates G^T G.
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
                    raise ValueError("ShampooTorch expects 2-D matrix parameters")

                grad = p.grad
                rows, cols = grad.shape
                state = self.state[p]
                if "row_precond" not in state:
                    state["row_precond"] = torch.eye(
                        rows, dtype=grad.dtype, device=grad.device
                    ) * eps
                    state["col_precond"] = torch.eye(
                        cols, dtype=grad.dtype, device=grad.device
                    ) * eps

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


def _newton_schulz_direction(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate UV^T from grad = U S V^T with Newton-Schulz iterations."""
    if steps <= 0:
        raise ValueError("ns_steps must be positive")

    X = grad / torch.linalg.norm(grad).clamp_min(torch.finfo(grad.dtype).eps)
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    # Coefficients used by practical Muon implementations. The iteration is a
    # quintic Newton-Schulz map tuned for fast polar-factor approximation.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


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
