import math

import pytest
import torch

from e11_condition_geometry.stateless import candidate_directions, global_fro_norm, polar_direction, scaled_direction


def test_polar_direction_has_unit_singular_values_on_gradient_support() -> None:
    grad = torch.diag(torch.tensor([3.0, 2.0, 0.5], dtype=torch.float64))

    direction = polar_direction(grad)
    singular_values = torch.linalg.svdvals(direction)

    assert torch.allclose(singular_values, torch.ones_like(singular_values), atol=1e-12)
    assert torch.sum(grad * direction).item() == pytest.approx(float(torch.linalg.svdvals(grad).sum()))


def test_polar_direction_rejects_non_matrix_gradient() -> None:
    with pytest.raises(ValueError, match="matrix gradient"):
        polar_direction(torch.ones(3, dtype=torch.float64))


def test_candidate_directions_are_detached_and_named() -> None:
    grad = torch.tensor([[1.0, -2.0], [0.0, 4.0]], dtype=torch.float64, requires_grad=True)

    candidates = candidate_directions([grad])

    assert set(candidates) == {"GD", "FreshAdamSign", "PolarMuon"}
    assert all(not tensor.requires_grad for tensors in candidates.values() for tensor in tensors)
    assert torch.allclose(candidates["GD"][0], grad.detach())
    assert candidates["FreshAdamSign"][0][0, 0].item() == pytest.approx(1.0 / (1.0 + 1e-8))
    assert candidates["FreshAdamSign"][0][0, 1].item() == pytest.approx(-2.0 / (2.0 + 1e-8))


def test_scaled_direction_matches_target_relative_update_norm() -> None:
    params = [
        torch.nn.Parameter(torch.tensor([[3.0, 4.0]], dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor([[0.0, 12.0]], dtype=torch.float64)),
    ]
    direction = [
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        torch.tensor([[2.0, 1.0]], dtype=torch.float64),
    ]
    target_relative = 0.125

    scaled = scaled_direction(direction, target_relative_update_norm=target_relative, params=params)
    actual_relative = global_fro_norm(scaled) / math.sqrt(3.0**2 + 4.0**2 + 12.0**2)

    assert actual_relative == pytest.approx(target_relative)


def test_scaled_direction_keeps_zero_direction_zero() -> None:
    params = [torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))]
    direction = [torch.zeros((2, 2), dtype=torch.float64)]

    scaled = scaled_direction(direction, target_relative_update_norm=0.1, params=params)

    assert global_fro_norm(scaled) == 0.0
