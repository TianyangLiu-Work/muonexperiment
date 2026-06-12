import math

import torch

from e11_condition_geometry.diagnostics import derive_step_metrics, matrix_effective_rank, singular_values, stable_rank
from e11_condition_geometry.optimizers import ExactMuon


def test_rank_formulas_from_singular_values():
    sigmas = [3.0, 4.0]
    assert math.isclose(matrix_effective_rank(sigmas), 49.0 / 25.0)
    assert math.isclose(stable_rank(sigmas), 25.0 / 16.0)


def test_derived_prediction_quantities():
    metrics = derive_step_metrics([[3.0, 4.0]], [[2.0, 0.0]])
    assert math.isclose(metrics["nrG"], 49.0 / 25.0)
    assert math.isclose(metrics["stA"], 1.0)
    assert math.isclose(metrics["condition_score"], 49.0 / 25.0)
    assert math.isclose(metrics["delta_gd_pred"], 25.0 / 4.0)
    assert math.isclose(metrics["delta_spec_pred"], 49.0 / 4.0)


def test_singular_values_use_flattened_matrix_view_for_conv_kernels():
    kernel = torch.arange(2 * 1 * 3 * 3, dtype=torch.float64).reshape(2, 1, 3, 3)
    expected = torch.linalg.svdvals(kernel.reshape(2, -1))

    assert torch.allclose(torch.tensor(singular_values(kernel), dtype=expected.dtype), expected)


def test_exact_muon_updates_conv_kernel_matrix_view():
    param = torch.nn.Parameter(torch.zeros((2, 1, 3, 3), dtype=torch.float64))
    param.grad = torch.arange(1, 2 * 1 * 3 * 3 + 1, dtype=torch.float64).reshape_as(param)
    optimizer = ExactMuon([param], lr=0.1)

    optimizer.step()

    assert param.shape == (2, 1, 3, 3)
    assert torch.linalg.norm(param).item() > 0
