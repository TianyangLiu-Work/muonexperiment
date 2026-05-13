"""Numerical diagnostics used by notebook experiments."""

from __future__ import annotations

import torch


def relative_matrix_error(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    denom = torch.linalg.norm(target).clamp_min(epsilon)
    return float((torch.linalg.norm(estimate - target) / denom).detach().cpu())


def effective_rank(
    matrix: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
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
    estimate_s = torch.linalg.svdvals(estimate.detach())[:k]
    target_s = torch.linalg.svdvals(target.detach())[:k]
    if estimate_s.numel() < k:
        estimate_s = torch.nn.functional.pad(estimate_s, (0, k - estimate_s.numel()))
    denom = torch.linalg.norm(target_s).clamp_min(epsilon)
    return float((torch.linalg.norm(estimate_s - target_s) / denom).detach().cpu())


def balancedness(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    left_gram = left.detach().T @ left.detach()
    right_gram = right.detach().T @ right.detach()
    denom = torch.linalg.norm(left_gram) + torch.linalg.norm(right_gram) + epsilon
    return float((torch.linalg.norm(left_gram - right_gram) / denom).detach().cpu())
