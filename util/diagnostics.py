"""Numerical diagnostics used by notebook experiments."""

from __future__ import annotations

import torch


def relative_matrix_error(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return ||estimate - target||_F / max(||target||_F, epsilon)."""
    denom = torch.linalg.norm(target).clamp_min(epsilon)
    return float((torch.linalg.norm(estimate - target) / denom).detach().cpu())


def effective_rank(
    matrix: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return entropy effective rank exp(-sum_i p_i log p_i).

    Here p_i = sigma_i / sum_j sigma_j and sigma_i are singular values.
    This is a soft rank: it is high when singular mass is spread out.
    """
    singular_values = torch.linalg.svdvals(matrix.detach())
    total = singular_values.sum()
    if float(total.detach().cpu()) <= epsilon:
        return 0.0
    probabilities = (singular_values / total).clamp_min(epsilon)
    entropy = -(probabilities * probabilities.log()).sum()
    return float(torch.exp(entropy).detach().cpu())


def stable_rank(
    matrix: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return stable rank srank(M) = ||M||_F^2 / max(||M||_op^2, epsilon)."""
    singular_values = torch.linalg.svdvals(matrix.detach())
    if singular_values.numel() == 0:
        return 0.0
    largest = singular_values[0].square().clamp_min(epsilon)
    return float((singular_values.square().sum() / largest).detach().cpu())


def matrix_norms(
    matrix: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Return Frobenius, operator, effective-rank, and stable-rank diagnostics."""
    singular_values = torch.linalg.svdvals(matrix.detach())
    if singular_values.numel() == 0:
        return {"fro_norm": 0.0, "op_norm": 0.0, "effective_rank": 0.0, "stable_rank": 0.0}
    fro_norm = torch.linalg.norm(singular_values)
    op_norm = singular_values[0]
    total = singular_values.sum()
    if float(total.detach().cpu()) <= epsilon:
        eff_rank = 0.0
    else:
        probabilities = (singular_values / total).clamp_min(epsilon)
        eff_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()).detach().cpu())
    stable = float((singular_values.square().sum() / op_norm.square().clamp_min(epsilon)).detach().cpu())
    return {
        "fro_norm": float(fro_norm.detach().cpu()),
        "op_norm": float(op_norm.detach().cpu()),
        "effective_rank": eff_rank,
        "stable_rank": stable,
    }


def aggregate_matrix_diagnostics(
    matrices: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    prefix: str,
) -> dict[str, float]:
    """Aggregate matrix diagnostics over a list of parameter blocks."""
    if not matrices:
        return {
            f"{prefix}_fro_norm": 0.0,
            f"{prefix}_op_norm": 0.0,
            f"{prefix}_effective_rank": 0.0,
            f"{prefix}_stable_rank": 0.0,
        }
    stats = [matrix_norms(matrix) for matrix in matrices]
    fro_norm = sum(item["fro_norm"] ** 2 for item in stats) ** 0.5
    return {
        f"{prefix}_fro_norm": fro_norm,
        f"{prefix}_op_norm": sum(item["op_norm"] for item in stats) / len(stats),
        f"{prefix}_effective_rank": sum(item["effective_rank"] for item in stats) / len(stats),
        f"{prefix}_stable_rank": sum(item["stable_rank"] for item in stats) / len(stats),
    }


def descent_alignment(
    gradients: list[torch.Tensor] | tuple[torch.Tensor, ...],
    updates: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return cosine alignment between the update and negative gradient.

    A value near 1 means the update follows steepest descent direction, 0 means
    it is nearly orthogonal, and negative means it locally points uphill.
    """
    numerator = sum(float(((-grad.detach()) * update.detach()).sum().cpu()) for grad, update in zip(gradients, updates))
    grad_norm = sum(float(grad.detach().square().sum().cpu()) for grad in gradients) ** 0.5
    update_norm = sum(float(update.detach().square().sum().cpu()) for update in updates) ** 0.5
    return numerator / max(grad_norm * update_norm, epsilon)


def relative_step_size(
    params_before: list[torch.Tensor] | tuple[torch.Tensor, ...],
    updates: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return ||update||_F / max(||params_before||_F, epsilon), aggregated over blocks."""
    update_norm = sum(float(update.detach().square().sum().cpu()) for update in updates) ** 0.5
    param_norm = sum(float(param.detach().square().sum().cpu()) for param in params_before) ** 0.5
    return update_norm / max(param_norm, epsilon)


def top_singular_value_error(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int,
    epsilon: float = 1e-12,
) -> float:
    """Return relative l2 error between the top-k singular value vectors."""
    estimate_s = torch.linalg.svdvals(estimate.detach())[:k]
    target_s = torch.linalg.svdvals(target.detach())[:k]
    if estimate_s.numel() < k:
        estimate_s = torch.nn.functional.pad(estimate_s, (0, k - estimate_s.numel()))
    denom = torch.linalg.norm(target_s).clamp_min(epsilon)
    return float((torch.linalg.norm(estimate_s - target_s) / denom).detach().cpu())


def balancedness(
    *factors: torch.Tensor,
    epsilon: float = 1e-12,
) -> float:
    """Return factor-chain Gram mismatch.

    For legacy two-factor inputs with equal shapes this is
    ||L^T L - R^T R||_F / (||L^T L||_F + ||R^T R||_F + epsilon).
    For a factor chain it averages the adjacent mismatch between
    W_i^T W_i and W_{i+1} W_{i+1}^T.
    """
    if len(factors) < 2:
        return 0.0
    if len(factors) == 2 and factors[0].shape == factors[1].shape:
        left_gram = factors[0].detach().T @ factors[0].detach()
        right_gram = factors[1].detach().T @ factors[1].detach()
        denom = torch.linalg.norm(left_gram) + torch.linalg.norm(right_gram) + epsilon
        return float((torch.linalg.norm(left_gram - right_gram) / denom).detach().cpu())

    values = []
    for current, following in zip(factors, factors[1:]):
        current_gram = current.detach().T @ current.detach()
        following_gram = following.detach() @ following.detach().T
        denom = torch.linalg.norm(current_gram) + torch.linalg.norm(following_gram) + epsilon
        values.append(torch.linalg.norm(current_gram - following_gram) / denom)
    return float(torch.stack(values).mean().detach().cpu())
