from __future__ import annotations

import torch


def global_fro_norm(tensors: list[torch.Tensor]) -> float:
    return float(sum(float(torch.linalg.norm(tensor).detach().cpu()) ** 2 for tensor in tensors) ** 0.5)


def parameter_fro_norm(params: list[torch.nn.Parameter]) -> float:
    return float(sum(float(torch.linalg.norm(param.detach()).cpu()) ** 2 for param in params) ** 0.5)


def polar_direction(grad: torch.Tensor) -> torch.Tensor:
    if grad.ndim != 2:
        raise ValueError("polar direction expects a matrix gradient")
    if float(torch.linalg.norm(grad).detach().cpu()) <= 0.0:
        return torch.zeros_like(grad)
    u, _, vh = torch.linalg.svd(grad, full_matrices=False)
    return u @ vh


def candidate_directions(grads: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    return {
        "GD": [grad.detach().clone() for grad in grads],
        "FreshAdamSign": [grad.detach() / (grad.detach().abs() + 1e-8) for grad in grads],
        "PolarMuon": [polar_direction(grad.detach()) for grad in grads],
    }


def scaled_direction(
    direction: list[torch.Tensor],
    *,
    target_relative_update_norm: float,
    params: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    target = target_relative_update_norm * parameter_fro_norm(params)
    norm = global_fro_norm(direction)
    if norm <= 0.0:
        return [torch.zeros_like(tensor) for tensor in direction]
    scale = target / norm
    return [scale * tensor for tensor in direction]
