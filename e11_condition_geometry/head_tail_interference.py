from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch

from .statistics import ci95, log_ratio_ci95


@dataclass(frozen=True)
class HeadTailSetting:
    name: str
    gradient_singular_values: tuple[float, ...]
    tail_singular_values: tuple[float, ...]
    downstream_singular_values: tuple[float, ...]


@dataclass(frozen=True)
class HeadTailConfig:
    seeds: tuple[int, ...] = tuple(range(80))
    output_dim: int = 10
    input_dim: int = 24
    tail_output_dim: int | None = None
    tail_batch_size: int = 64
    first_order_gain_per_grad_fro: float = 0.25
    dtype: str = "float64"
    device: str = "cpu"


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return dtype


def _orthonormal(rows: int, cols: int, *, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.randn(rows, cols, generator=generator, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :cols]


def matrix_with_singular_values(
    rows: int,
    cols: int,
    singular_values: tuple[float, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rank = len(singular_values)
    if rank > min(rows, cols):
        raise ValueError(f"rank {rank} exceeds matrix shape {(rows, cols)}")
    u = _orthonormal(rows, rank, generator=generator, device=device, dtype=dtype)
    v = _orthonormal(cols, rank, generator=generator, device=device, dtype=dtype)
    sigma = torch.tensor(singular_values, device=device, dtype=dtype)
    return (u * sigma.unsqueeze(0)) @ v.T


def default_head_tail_settings(output_dim: int = 10, input_dim: int = 24) -> tuple[HeadTailSetting, ...]:
    grad_rank = min(output_dim, input_dim)
    tail_rank = input_dim
    downstream_rank = output_dim
    high_head = tuple(1.0 / math.sqrt(float(index + 1)) for index in range(grad_rank))
    low_head = (1.0,) + tuple(0.02 for _ in range(grad_rank - 1))
    low_tail = (math.sqrt(float(tail_rank)),) + tuple(0.05 for _ in range(tail_rank - 1))
    high_tail = tuple(1.0 for _ in range(tail_rank))
    low_downstream = (1.0,) + tuple(0.05 for _ in range(downstream_rank - 1))
    high_downstream = tuple(1.0 for _ in range(downstream_rank))
    return (
        HeadTailSetting(
            name="high_head_rank_low_tail_srank",
            gradient_singular_values=high_head,
            tail_singular_values=low_tail,
            downstream_singular_values=low_downstream,
        ),
        HeadTailSetting(
            name="low_head_rank_high_tail_srank",
            gradient_singular_values=low_head,
            tail_singular_values=high_tail,
            downstream_singular_values=high_downstream,
        ),
    )


def _complete_orthonormal_basis(
    prefix: torch.Tensor,
    total_cols: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if total_cols <= prefix.shape[1]:
        return prefix[:, :total_cols]
    remaining = total_cols - prefix.shape[1]
    random_part = torch.randn(prefix.shape[0], remaining, generator=generator, device=device, dtype=dtype)
    random_part = random_part - prefix @ (prefix.T @ random_part)
    q, _ = torch.linalg.qr(random_part, mode="reduced")
    return torch.cat([prefix, q[:, :remaining]], dim=1)


def _tail_inputs_from_head_basis(
    head_gradient: torch.Tensor,
    singular_values: tuple[float, ...],
    tail_batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rank = len(singular_values)
    _, _, vh = torch.linalg.svd(head_gradient, full_matrices=False)
    left_basis = _complete_orthonormal_basis(
        vh.T,
        rank,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    right_basis = _orthonormal(tail_batch_size, rank, generator=generator, device=device, dtype=dtype)
    sigma = torch.tensor(singular_values, device=device, dtype=dtype)
    return (left_basis * sigma.unsqueeze(0)) @ right_basis.T


def _downstream_from_head_basis(
    head_gradient: torch.Tensor,
    singular_values: tuple[float, ...],
    tail_output_dim: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rank = len(singular_values)
    u, _, _ = torch.linalg.svd(head_gradient, full_matrices=False)
    right_basis = _complete_orthonormal_basis(
        u,
        rank,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    left_basis = _orthonormal(tail_output_dim, rank, generator=generator, device=device, dtype=dtype)
    sigma = torch.tensor(singular_values, device=device, dtype=dtype)
    return (left_basis * sigma.unsqueeze(0)) @ right_basis.T


def stable_rank(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix)
    fro_sq = float(torch.sum(singular_values.square()).cpu())
    op_sq = float(singular_values[0].square().cpu())
    return fro_sq / max(op_sq, 1e-300)


def nuclear_rank(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix)
    nuclear = float(torch.sum(singular_values).cpu())
    fro_sq = float(torch.sum(singular_values.square()).cpu())
    return nuclear**2 / max(fro_sq, 1e-300)


def sandwiched_stable_rank(downstream: torch.Tensor, activation: torch.Tensor) -> float:
    downstream_s = torch.linalg.svdvals(downstream)
    activation_s = torch.linalg.svdvals(activation)
    rank = min(downstream_s.numel(), activation_s.numel())
    if rank == 0:
        return math.nan
    paired = downstream_s[:rank].square() * activation_s[:rank].square()
    numerator = float(torch.sum(paired).cpu())
    denominator = float((downstream_s[0].square() * activation_s[0].square()).cpu())
    return numerator / max(denominator, 1e-300)


def _classification_state(
    tail_inputs: torch.Tensor,
    downstream: torch.Tensor,
    *,
    parameter_output_dim: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(parameter_output_dim, tail_inputs.shape[0], generator=generator, device=device, dtype=dtype)
    logits = downstream @ weight @ tail_inputs
    labels = torch.argmax(logits, dim=0)
    margin = _margins(logits, labels)
    mean_margin = float(margin.mean().cpu())
    if mean_margin > 1e-12:
        weight = weight * (3.0 / mean_margin)
    return weight, labels


def _margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits[labels, torch.arange(logits.shape[1], device=logits.device)]
    masked = logits.clone()
    masked[labels, torch.arange(logits.shape[1], device=logits.device)] = -torch.inf
    return true_logits - torch.max(masked, dim=0).values


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.T, labels, reduction="mean")


def _head_loss(weight: torch.Tensor, base_weight: torch.Tensor, head_gradient: torch.Tensor) -> torch.Tensor:
    n = weight.shape[1]
    head_inputs = torch.eye(n, device=weight.device, dtype=weight.dtype)
    head_targets = base_weight @ head_inputs - float(n) * head_gradient
    residual = weight @ head_inputs - head_targets
    return 0.5 * torch.sum(residual.square()) / float(n)


def _update_directions(head_gradient: torch.Tensor) -> dict[str, torch.Tensor]:
    u, _, vh = torch.linalg.svd(head_gradient, full_matrices=False)
    return {
        "frobenius": head_gradient / torch.linalg.norm(head_gradient),
        "spectral": u @ vh,
    }


def run_head_tail_interference(config: HeadTailConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = _dtype_from_name(config.dtype)
    tail_output_dim = config.output_dim if config.tail_output_dim is None else int(config.tail_output_dim)
    settings = default_head_tail_settings(config.output_dim, config.input_dim)
    rows: list[dict] = []

    for setting in settings:
        for seed in config.seeds:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            head_gradient = matrix_with_singular_values(
                config.output_dim,
                config.input_dim,
                setting.gradient_singular_values,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            tail_inputs = _tail_inputs_from_head_basis(
                head_gradient,
                setting.tail_singular_values,
                config.tail_batch_size,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            downstream = _downstream_from_head_basis(
                head_gradient,
                setting.downstream_singular_values,
                tail_output_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            base_weight, tail_labels = _classification_state(
                tail_inputs,
                downstream,
                parameter_output_dim=config.output_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            base_tail_logits = downstream @ base_weight @ tail_inputs
            base_tail_ce = _cross_entropy(base_tail_logits, tail_labels)
            base_tail_margin = _margins(base_tail_logits, tail_labels).mean()
            base_tail_accuracy = (torch.argmax(base_tail_logits, dim=0) == tail_labels).double().mean()
            base_head_loss = _head_loss(base_weight, base_weight, head_gradient)

            grad_fro = float(torch.linalg.norm(head_gradient).cpu())
            grad_nuclear = float(torch.linalg.matrix_norm(head_gradient, ord="nuc").cpu())
            first_order_gain = config.first_order_gain_per_grad_fro * grad_fro
            tail_srank = stable_rank(tail_inputs)
            downstream_srank = stable_rank(downstream)
            tail_ssrank = sandwiched_stable_rank(downstream, tail_inputs)
            grad_nrank = nuclear_rank(head_gradient)
            theory_ratio = tail_ssrank / max(grad_nrank, 1e-300)

            for geometry, direction in _update_directions(head_gradient).items():
                alignment = float(torch.sum(head_gradient * direction).cpu())
                step_size = first_order_gain / max(alignment, 1e-300)
                weight_after = base_weight - step_size * direction
                tail_logits_after = downstream @ weight_after @ tail_inputs
                output_delta = tail_logits_after - base_tail_logits
                tail_ce_after = _cross_entropy(tail_logits_after, tail_labels)
                tail_margin_after = _margins(tail_logits_after, tail_labels).mean()
                tail_accuracy_after = (torch.argmax(tail_logits_after, dim=0) == tail_labels).double().mean()
                head_loss_after = _head_loss(weight_after, base_weight, head_gradient)

                rows.append(
                    {
                        "setting": setting.name,
                        "seed": int(seed),
                        "geometry": geometry,
                        "first_order_head_gain": float(first_order_gain),
                        "actual_head_loss_decrease": float((base_head_loss - head_loss_after).cpu()),
                        "head_gradient_fro_norm": grad_fro,
                        "head_gradient_nuclear_norm": grad_nuclear,
                        "head_gradient_nuclear_rank": grad_nrank,
                        "tail_activation_stable_rank": tail_srank,
                        "tail_downstream_stable_rank": downstream_srank,
                        "tail_downstream_aware_stable_rank": tail_ssrank,
                        "theory_ratio_spectral_over_fro": theory_ratio,
                        "predicted_spectral_less_drift": bool(theory_ratio < 1.0),
                        "step_size": float(step_size),
                        "parameter_update_fro_norm": float((step_size * torch.linalg.norm(direction)).cpu()),
                        "parameter_update_op_norm": float((step_size * torch.linalg.matrix_norm(direction, ord=2)).cpu()),
                        "tail_output_drift_fro": float(torch.linalg.norm(output_delta).cpu()),
                        "tail_output_drift_rms": float(torch.sqrt(torch.mean(output_delta.square())).cpu()),
                        "tail_self_mse_increase": float((0.5 * torch.mean(output_delta.square())).cpu()),
                        "tail_ce_before": float(base_tail_ce.cpu()),
                        "tail_ce_after": float(tail_ce_after.cpu()),
                        "tail_ce_increase": float((tail_ce_after - base_tail_ce).cpu()),
                        "tail_margin_before": float(base_tail_margin.cpu()),
                        "tail_margin_after": float(tail_margin_after.cpu()),
                        "tail_margin_drop": float((base_tail_margin - tail_margin_after).cpu()),
                        "tail_accuracy_before": float(base_tail_accuracy.cpu()),
                        "tail_accuracy_after": float(tail_accuracy_after.cpu()),
                    }
                )

    step_metrics = pd.DataFrame(rows)
    pair_summary = summarize_head_tail_pairs(step_metrics)
    return step_metrics, pair_summary


def summarize_head_tail_pairs(step_metrics: pd.DataFrame) -> pd.DataFrame:
    paired_rows = []
    for (setting, seed), group in step_metrics.groupby(["setting", "seed"], observed=True, sort=False):
        by_geometry = group.set_index("geometry")
        fro = by_geometry.loc["frobenius"]
        spectral = by_geometry.loc["spectral"]
        paired_rows.append(
            {
                "setting": setting,
                "seed": int(seed),
                "predicted_spectral_less_drift": bool(spectral["predicted_spectral_less_drift"]),
                "head_gradient_nuclear_rank": float(spectral["head_gradient_nuclear_rank"]),
                "tail_activation_stable_rank": float(spectral["tail_activation_stable_rank"]),
                "tail_downstream_stable_rank": float(spectral["tail_downstream_stable_rank"]),
                "tail_downstream_aware_stable_rank": float(spectral["tail_downstream_aware_stable_rank"]),
                "theory_ratio_spectral_over_fro": float(spectral["theory_ratio_spectral_over_fro"]),
                "tail_output_drift_sq_ratio_spectral_over_fro": float(
                    spectral["tail_output_drift_fro"] ** 2 / max(fro["tail_output_drift_fro"] ** 2, 1e-300)
                ),
                "tail_self_mse_ratio_spectral_over_fro": float(
                    spectral["tail_self_mse_increase"] / max(fro["tail_self_mse_increase"], 1e-300)
                ),
                "actual_head_loss_decrease_diff_spectral_minus_fro": float(
                    spectral["actual_head_loss_decrease"] - fro["actual_head_loss_decrease"]
                ),
                "tail_ce_increase_diff_spectral_minus_fro": float(
                    spectral["tail_ce_increase"] - fro["tail_ce_increase"]
                ),
                "tail_margin_drop_diff_spectral_minus_fro": float(
                    spectral["tail_margin_drop"] - fro["tail_margin_drop"]
                ),
                "spectral_less_tail_output_drift": bool(spectral["tail_output_drift_fro"] < fro["tail_output_drift_fro"]),
                "spectral_less_tail_self_mse": bool(spectral["tail_self_mse_increase"] < fro["tail_self_mse_increase"]),
            }
        )
    pairs = pd.DataFrame(paired_rows)
    grouped = pairs.groupby("setting", observed=True, sort=False)
    summary = grouped.agg(
        predicted_spectral_less_drift=("predicted_spectral_less_drift", "first"),
        seeds=("seed", "nunique"),
        mean_head_gradient_nuclear_rank=("head_gradient_nuclear_rank", "mean"),
        mean_tail_activation_stable_rank=("tail_activation_stable_rank", "mean"),
        mean_tail_downstream_stable_rank=("tail_downstream_stable_rank", "mean"),
        mean_tail_downstream_aware_stable_rank=("tail_downstream_aware_stable_rank", "mean"),
        mean_theory_ratio_spectral_over_fro=("theory_ratio_spectral_over_fro", "mean"),
        mean_tail_output_drift_sq_ratio_spectral_over_fro=(
            "tail_output_drift_sq_ratio_spectral_over_fro",
            "mean",
        ),
        median_tail_output_drift_sq_ratio_spectral_over_fro=(
            "tail_output_drift_sq_ratio_spectral_over_fro",
            "median",
        ),
        spectral_less_tail_output_drift_fraction=("spectral_less_tail_output_drift", "mean"),
        mean_tail_self_mse_ratio_spectral_over_fro=("tail_self_mse_ratio_spectral_over_fro", "mean"),
        mean_actual_head_loss_decrease_diff_spectral_minus_fro=(
            "actual_head_loss_decrease_diff_spectral_minus_fro",
            "mean",
        ),
        mean_tail_ce_increase_diff_spectral_minus_fro=("tail_ce_increase_diff_spectral_minus_fro", "mean"),
        mean_tail_margin_drop_diff_spectral_minus_fro=("tail_margin_drop_diff_spectral_minus_fro", "mean"),
    )
    summary = summary.reset_index()
    interval_rows = []
    for setting, group in pairs.groupby("setting", observed=True, sort=False):
        drift_mean, drift_low, drift_high = log_ratio_ci95(group["tail_output_drift_sq_ratio_spectral_over_fro"])
        mse_mean, mse_low, mse_high = log_ratio_ci95(group["tail_self_mse_ratio_spectral_over_fro"])
        head_diff_mean, head_diff_low, head_diff_high = ci95(group["actual_head_loss_decrease_diff_spectral_minus_fro"])
        ce_diff_mean, ce_diff_low, ce_diff_high = ci95(group["tail_ce_increase_diff_spectral_minus_fro"])
        margin_diff_mean, margin_diff_low, margin_diff_high = ci95(group["tail_margin_drop_diff_spectral_minus_fro"])
        interval_rows.append(
            {
                "setting": setting,
                "geomean_tail_output_drift_sq_ratio_spectral_over_fro": drift_mean,
                "tail_output_drift_sq_ratio_ci95_low": drift_low,
                "tail_output_drift_sq_ratio_ci95_high": drift_high,
                "geomean_tail_self_mse_ratio_spectral_over_fro": mse_mean,
                "tail_self_mse_ratio_ci95_low": mse_low,
                "tail_self_mse_ratio_ci95_high": mse_high,
                "actual_head_loss_decrease_diff_ci95_low": head_diff_low,
                "actual_head_loss_decrease_diff_ci95_high": head_diff_high,
                "tail_ce_increase_diff_ci95_low": ce_diff_low,
                "tail_ce_increase_diff_ci95_high": ce_diff_high,
                "tail_margin_drop_diff_ci95_low": margin_diff_low,
                "tail_margin_drop_diff_ci95_high": margin_diff_high,
            }
        )
    intervals = pd.DataFrame(interval_rows)
    return summary.merge(intervals, on="setting", how="left")
