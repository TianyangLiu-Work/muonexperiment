from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np
import torch


def matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    """View a trainable tensor as the matrix used for spectral diagnostics."""

    if tensor.ndim < 2:
        raise ValueError(f"spectral diagnostics expect at least 2-D tensors, got shape {tuple(tensor.shape)}")
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def singular_values(matrix: torch.Tensor) -> list[float]:
    values = torch.linalg.svdvals(matrix_view(matrix.detach()))
    return [float(value) for value in values.cpu()]


def matrix_effective_rank(sigmas: Sequence[float]) -> float:
    values = np.asarray(sigmas, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return math.nan
    fro_sq = float(np.square(values).sum())
    if fro_sq <= 0:
        return math.nan
    return float(np.square(values.sum()) / fro_sq)


def stable_rank(sigmas: Sequence[float]) -> float:
    values = np.asarray(sigmas, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return math.nan
    op_sq = float(np.square(values.max()))
    fro_sq = float(np.square(values).sum())
    if op_sq <= 0:
        return math.nan
    return float(fro_sq / op_sq)


def fro_sq(sigmas: Sequence[float]) -> float:
    values = np.asarray(sigmas, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.square(values).sum())


def op_sq(sigmas: Sequence[float]) -> float:
    values = np.asarray(sigmas, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.square(values.max()))


def nuclear_sq(sigmas: Sequence[float]) -> float:
    values = np.asarray(sigmas, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.square(values.sum()))


def json_dumps_nested(values: Sequence[Sequence[float]]) -> str:
    return json.dumps([[float(x) for x in row] for row in values], separators=(",", ":"))


def json_loads_nested(text: str) -> list[list[float]]:
    if not isinstance(text, str) or not text:
        return []
    return json.loads(text)


def derive_step_metrics(sigma_g: Sequence[Sequence[float]], sigma_a: Sequence[Sequence[float]]) -> dict:
    layer_rows = []
    for layer, (g_values, a_values) in enumerate(zip(sigma_g, sigma_a), start=1):
        nr_g = matrix_effective_rank(g_values)
        st_a = stable_rank(a_values)
        g_fro = fro_sq(g_values)
        g_nuc = nuclear_sq(g_values)
        a_fro = fro_sq(a_values)
        a_op = op_sq(a_values)
        delta_gd = g_fro / a_op if np.isfinite(a_op) and a_op > 0 else math.nan
        delta_spec = g_nuc / a_fro if np.isfinite(a_fro) and a_fro > 0 else math.nan
        condition = nr_g / st_a if np.isfinite(st_a) and st_a > 0 else math.nan
        layer_rows.append(
            {
                "layer": layer,
                "nrG": nr_g,
                "stA": st_a,
                "condition_score": condition,
                "delta_gd_pred": delta_gd,
                "delta_spec_pred": delta_spec,
            }
        )
    finite = lambda key: [row[key] for row in layer_rows if np.isfinite(row[key])]
    mean_or_nan = lambda vals: float(np.mean(vals)) if vals else math.nan
    return {
        "layer_metrics": layer_rows,
        "nrG": mean_or_nan(finite("nrG")),
        "stA": mean_or_nan(finite("stA")),
        "condition_score": mean_or_nan(finite("condition_score")),
        "delta_gd_pred": float(np.nansum(finite("delta_gd_pred"))) if finite("delta_gd_pred") else math.nan,
        "delta_spec_pred": float(np.nansum(finite("delta_spec_pred"))) if finite("delta_spec_pred") else math.nan,
    }


def derive_update_metrics(sigma_update: Sequence[Sequence[float]]) -> dict:
    layer_rows = []
    for layer, values in enumerate(sigma_update, start=1):
        nr_update = matrix_effective_rank(values)
        st_update = stable_rank(values)
        rank_ceiling = len(values)
        flatness = st_update / nr_update if np.isfinite(nr_update) and nr_update > 0 else math.nan
        nr_fraction = nr_update / rank_ceiling if rank_ceiling > 0 and np.isfinite(nr_update) else math.nan
        st_fraction = st_update / rank_ceiling if rank_ceiling > 0 and np.isfinite(st_update) else math.nan
        layer_rows.append(
            {
                "layer": layer,
                "update_rank_ceiling": float(rank_ceiling),
                "nrUpdate": nr_update,
                "stUpdate": st_update,
                "nrUpdateFrac": nr_fraction,
                "stUpdateFrac": st_fraction,
                "update_flatness": flatness,
            }
        )
    finite = lambda key: [row[key] for row in layer_rows if np.isfinite(row[key])]
    mean_or_nan = lambda vals: float(np.mean(vals)) if vals else math.nan
    return {
        "layer_metrics": layer_rows,
        "nrUpdate": mean_or_nan(finite("nrUpdate")),
        "stUpdate": mean_or_nan(finite("stUpdate")),
        "nrUpdateFrac": mean_or_nan(finite("nrUpdateFrac")),
        "stUpdateFrac": mean_or_nan(finite("stUpdateFrac")),
        "update_flatness": mean_or_nan(finite("update_flatness")),
    }
