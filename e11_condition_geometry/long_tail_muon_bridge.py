from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch
import torch.nn.functional as F

from .diagnostics import matrix_view
from .long_tail_digits import (
    TinyMLP,
    add_noise,
    alignment,
    apply_direction,
    batch,
    dtype_from_name,
    full_metrics,
    make_generator,
    restore,
    snapshot,
    split_long_tail_digits,
    train_digits_checkpoint,
    update_norms,
)
from .statistics import ci95, log_ratio_ci95


@dataclass(frozen=True)
class LongTailMuonBridgeConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    head_classes: tuple[int, ...] = (0, 1, 2, 3, 4)
    tail_classes: tuple[int, ...] = (5, 6, 7, 8, 9)
    head_train_per_class: int = 100
    tail_train_per_class: int = 40
    tail_eval_per_class: int = 40
    hidden_dim: int = 32
    warmup_steps: int = 80
    warmup_batch_size: int = 64
    head_batch_size: int = 64
    lr: float = 1e-2
    train_noise_std: float = 0.05
    target_head_gain_fraction: float = 0.02
    momentum_beta: float = 0.9
    momentum_history_steps: int = 8
    newton_schulz_steps: int = 5
    dtype: str = "float64"
    device: str = "cpu"


@dataclass(frozen=True)
class LongTailPracticalMuonBridgeConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    head_classes: tuple[int, ...] = (0, 1, 2, 3, 4)
    tail_classes: tuple[int, ...] = (5, 6, 7, 8, 9)
    head_train_per_class: int = 100
    tail_train_per_class: int = 40
    tail_eval_per_class: int = 40
    hidden_dim: int = 32
    warmup_steps: int = 80
    warmup_batch_size: int = 64
    head_batch_size: int = 64
    lr: float = 1e-2
    train_noise_std: float = 0.05
    target_head_gain_fraction: float = 0.02
    momentum_beta: float = 0.9
    trajectory_steps: int = 6
    trajectory_lr: float = 1e-3
    newton_schulz_steps: int = 5
    dtype: str = "float64"
    device: str = "cpu"


def polar_factor(matrix: torch.Tensor) -> torch.Tensor:
    matrix_2d = matrix_view(matrix)
    u, _, vh = torch.linalg.svd(matrix_2d, full_matrices=False)
    return (u @ vh).reshape_as(matrix)


def newton_schulz_polar(matrix: torch.Tensor, *, steps: int) -> torch.Tensor:
    matrix_2d = matrix_view(matrix)
    op_norm = torch.linalg.matrix_norm(matrix_2d, ord=2)
    if float(op_norm.detach().cpu()) <= 0.0:
        return torch.zeros_like(matrix)
    x = matrix_2d / op_norm.clamp_min(torch.finfo(matrix_2d.dtype).eps)
    for _ in range(int(steps)):
        x = 1.5 * x - 0.5 * x @ x.T @ x
    return x.reshape_as(matrix)


def tensor_fro_norm(tensors: list[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(torch.sum(tensor.detach().square()).cpu())
    return math.sqrt(total)


def direction_cosine(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    numerator = 0.0
    for left_tensor, right_tensor in zip(left, right):
        numerator += float(torch.sum(left_tensor.detach() * right_tensor.detach()).cpu())
    denominator = tensor_fro_norm(left) * tensor_fro_norm(right)
    return numerator / max(denominator, 1e-300)


def frobenius_grad_direction(grads: list[torch.Tensor]) -> list[torch.Tensor]:
    grad_norm = tensor_fro_norm(grads)
    return [grad.detach() / max(grad_norm, 1e-300) for grad in grads]


def polar_directions(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [polar_factor(tensor.detach()) for tensor in tensors]


def newton_schulz_directions(tensors: list[torch.Tensor], *, steps: int) -> list[torch.Tensor]:
    return [newton_schulz_polar(tensor.detach(), steps=steps) for tensor in tensors]


def collect_gradients(
    model: TinyMLP,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> tuple[float, list[torch.Tensor]]:
    params = model.parameters()
    for param in params:
        param.grad = None
    loss = F.cross_entropy(model.logits(x_batch), y_batch)
    loss.backward()
    grads = [param.grad.detach().clone() for param in params]
    return float(loss.detach().cpu()), grads


def build_momentum_buffer(
    model: TinyMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    head_train: torch.Tensor,
    config: LongTailMuonBridgeConfig,
    *,
    seed: int,
    current_grads: list[torch.Tensor],
) -> list[torch.Tensor]:
    beta = float(config.momentum_beta)
    buffers = [torch.zeros_like(grad) for grad in current_grads]
    for step in range(config.momentum_history_steps):
        generator = make_generator(seed + 15000 + step, x.device)
        indices = batch(head_train, config.head_batch_size, generator=generator)
        batch_x = add_noise(x[indices], config.train_noise_std, generator=generator)
        _, grads = collect_gradients(model, batch_x, y[indices])
        buffers = [beta * buffer + (1.0 - beta) * grad for buffer, grad in zip(buffers, grads)]
    return [beta * buffer + (1.0 - beta) * grad for buffer, grad in zip(buffers, current_grads)]


def assign_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor]) -> None:
    for param, grad in zip(params, grads):
        param.grad = grad.detach().clone()


def momentum_update(
    previous: list[torch.Tensor],
    grads: list[torch.Tensor],
    *,
    beta: float,
) -> list[torch.Tensor]:
    if not previous:
        previous = [torch.zeros_like(grad) for grad in grads]
    return [float(beta) * buffer + (1.0 - float(beta)) * grad for buffer, grad in zip(previous, grads)]


def evaluate_bridge_direction(
    model: TinyMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    head_batch: torch.Tensor,
    tail_eval: torch.Tensor,
    before: list[torch.Tensor],
    base_head: dict[str, float],
    base_tail: dict[str, float],
    base_tail_logits: torch.Tensor,
    directions: list[torch.Tensor],
    *,
    target_gain: float,
) -> dict[str, float]:
    params = model.parameters()
    alignment_value = alignment(params, directions)
    step_size = target_gain / max(alignment_value, 1e-300)
    update_fro, update_op = update_norms(directions, step_size)
    apply_direction(params, before, directions, step_size)
    after_head = full_metrics(model, x, y, head_batch)
    after_tail = full_metrics(model, x, y, tail_eval)
    output_delta = model.logits(x[tail_eval]).detach() - base_tail_logits
    restore(params, before)
    return {
        "head_loss_before": base_head["loss"],
        "head_loss_after": after_head["loss"],
        "actual_head_loss_decrease": base_head["loss"] - after_head["loss"],
        "matched_first_order_head_gain": float(target_gain),
        "tail_loss_before": base_tail["loss"],
        "tail_loss_after": after_tail["loss"],
        "tail_loss_increase": after_tail["loss"] - base_tail["loss"],
        "tail_accuracy_before": base_tail["accuracy"],
        "tail_accuracy_after": after_tail["accuracy"],
        "tail_accuracy_drop": base_tail["accuracy"] - after_tail["accuracy"],
        "tail_margin_before": base_tail["mean_margin"],
        "tail_margin_after": after_tail["mean_margin"],
        "tail_margin_drop": base_tail["mean_margin"] - after_tail["mean_margin"],
        "tail_output_drift_fro": float(torch.linalg.norm(output_delta).cpu()),
        "tail_output_drift_rms": float(torch.sqrt(torch.mean(output_delta.square())).cpu()),
        "update_fro_norm": update_fro,
        "update_op_norm": update_op,
        "step_size": float(step_size),
        "alignment": alignment_value,
    }


def apply_practical_muon_step(
    params: list[torch.nn.Parameter],
    direction: list[torch.Tensor],
    *,
    lr: float,
) -> None:
    with torch.no_grad():
        for param, update in zip(params, direction):
            param.add_(update, alpha=-float(lr))


def run_long_tail_muon_bridge(config: LongTailMuonBridgeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []

    for seed in config.seeds:
        x, y, train_indices, head_train, _tail_train, tail_eval = split_long_tail_digits(
            config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        generator = make_generator(seed + 13000, device)
        model = TinyMLP(input_dim=x.shape[1], hidden_dim=config.hidden_dim, output_dim=10, generator=generator, device=device, dtype=dtype)
        train_digits_checkpoint(model, x, y, train_indices, config, seed=seed, seed_offset=12000)

        head_generator = make_generator(seed + 14000, device)
        head_batch = batch(head_train, config.head_batch_size, generator=head_generator)
        params = model.parameters()
        head_loss_value, current_grads = collect_gradients(model, x[head_batch], y[head_batch])
        assign_grads(params, current_grads)

        before = snapshot(params)
        base_tail_logits = model.logits(x[tail_eval]).detach()
        base_head = full_metrics(model, x, y, head_batch)
        base_tail = full_metrics(model, x, y, tail_eval)
        target_gain = config.target_head_gain_fraction * head_loss_value

        momentum = build_momentum_buffer(model, x, y, head_train, config, seed=seed, current_grads=current_grads)
        assign_grads(params, current_grads)

        directions = {
            "frobenius_grad": frobenius_grad_direction(current_grads),
            "polar_grad": polar_directions(current_grads),
            "polar_momentum": polar_directions(momentum),
            "ns_momentum": newton_schulz_directions(momentum, steps=config.newton_schulz_steps),
        }
        polar_grad = directions["polar_grad"]
        grad_norm = tensor_fro_norm(current_grads)
        momentum_norm = tensor_fro_norm(momentum)
        grad_momentum_cosine = direction_cosine(current_grads, momentum)

        for direction_name, direction in directions.items():
            metrics = evaluate_bridge_direction(
                model,
                x,
                y,
                head_batch,
                tail_eval,
                before,
                base_head,
                base_tail,
                base_tail_logits,
                direction,
                target_gain=target_gain,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "direction": direction_name,
                    "head_classes": ",".join(str(label) for label in config.head_classes),
                    "tail_classes": ",".join(str(label) for label in config.tail_classes),
                    "warmup_steps": int(config.warmup_steps),
                    "hidden_dim": int(config.hidden_dim),
                    "momentum_beta": float(config.momentum_beta),
                    "momentum_history_steps": int(config.momentum_history_steps),
                    "newton_schulz_steps": int(config.newton_schulz_steps),
                    "head_gradient_fro_norm": grad_norm,
                    "momentum_fro_norm": momentum_norm,
                    "gradient_momentum_cosine": grad_momentum_cosine,
                    "direction_cosine_to_polar_grad": direction_cosine(direction, polar_grad),
                    **metrics,
                }
            )
        restore(params, before)

    step_metrics = pd.DataFrame(rows)
    pair_summary = summarize_long_tail_muon_bridge(step_metrics)
    return step_metrics, pair_summary


def run_long_tail_practical_muon_bridge(
    config: LongTailPracticalMuonBridgeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []

    for seed in config.seeds:
        x, y, train_indices, head_train, _tail_train, tail_eval = split_long_tail_digits(
            config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        generator = make_generator(seed + 13000, device)
        model = TinyMLP(input_dim=x.shape[1], hidden_dim=config.hidden_dim, output_dim=10, generator=generator, device=device, dtype=dtype)
        train_digits_checkpoint(model, x, y, train_indices, config, seed=seed, seed_offset=12000)
        params = model.parameters()
        momentum: list[torch.Tensor] = []

        for trajectory_step in range(config.trajectory_steps):
            head_generator = make_generator(seed + 17000 + trajectory_step, device)
            head_batch = batch(head_train, config.head_batch_size, generator=head_generator)
            head_x = add_noise(x[head_batch], config.train_noise_std, generator=head_generator)
            head_loss_value, current_grads = collect_gradients(model, head_x, y[head_batch])
            momentum = momentum_update(momentum, current_grads, beta=config.momentum_beta)
            assign_grads(params, current_grads)

            before = snapshot(params)
            base_tail_logits = model.logits(x[tail_eval]).detach()
            base_head = full_metrics(model, x, y, head_batch)
            base_tail = full_metrics(model, x, y, tail_eval)
            target_gain = config.target_head_gain_fraction * head_loss_value

            directions = {
                "frobenius_grad": frobenius_grad_direction(current_grads),
                "polar_grad": polar_directions(current_grads),
                "polar_momentum": polar_directions(momentum),
                "ns_momentum": newton_schulz_directions(momentum, steps=config.newton_schulz_steps),
            }
            polar_grad = directions["polar_grad"]
            grad_norm = tensor_fro_norm(current_grads)
            momentum_norm = tensor_fro_norm(momentum)
            grad_momentum_cosine = direction_cosine(current_grads, momentum)

            for direction_name, direction in directions.items():
                metrics = evaluate_bridge_direction(
                    model,
                    x,
                    y,
                    head_batch,
                    tail_eval,
                    before,
                    base_head,
                    base_tail,
                    base_tail_logits,
                    direction,
                    target_gain=target_gain,
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "trajectory_step": int(trajectory_step),
                        "direction": direction_name,
                        "head_classes": ",".join(str(label) for label in config.head_classes),
                        "tail_classes": ",".join(str(label) for label in config.tail_classes),
                        "warmup_steps": int(config.warmup_steps),
                        "hidden_dim": int(config.hidden_dim),
                        "momentum_beta": float(config.momentum_beta),
                        "trajectory_steps": int(config.trajectory_steps),
                        "trajectory_lr": float(config.trajectory_lr),
                        "newton_schulz_steps": int(config.newton_schulz_steps),
                        "head_gradient_fro_norm": grad_norm,
                        "momentum_fro_norm": momentum_norm,
                        "gradient_momentum_cosine": grad_momentum_cosine,
                        "direction_cosine_to_polar_grad": direction_cosine(direction, polar_grad),
                        **metrics,
                    }
                )

            restore(params, before)
            apply_practical_muon_step(params, directions["ns_momentum"], lr=config.trajectory_lr)

    step_metrics = pd.DataFrame(rows)
    summary = summarize_long_tail_practical_muon_bridge(step_metrics)
    return step_metrics, summary


def summarize_long_tail_muon_bridge(step_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, group in step_metrics.groupby("direction", observed=True, sort=False):
        if direction == "frobenius_grad":
            continue
        ratio_rows = []
        for seed, seed_group in step_metrics.groupby("seed", observed=True, sort=False):
            by_direction = seed_group.set_index("direction")
            baseline = by_direction.loc["frobenius_grad"]
            polar_grad = by_direction.loc["polar_grad"]
            variant = by_direction.loc[direction]
            ratio_rows.append(
                {
                    "seed": int(seed),
                    "tail_output_drift_sq_ratio_vs_fro": float(
                        variant["tail_output_drift_fro"] ** 2 / max(baseline["tail_output_drift_fro"] ** 2, 1e-300)
                    ),
                    "tail_output_drift_sq_ratio_vs_polar_grad": float(
                        variant["tail_output_drift_fro"] ** 2 / max(polar_grad["tail_output_drift_fro"] ** 2, 1e-300)
                    ),
                    "actual_head_loss_decrease_diff_vs_fro": float(
                        variant["actual_head_loss_decrease"] - baseline["actual_head_loss_decrease"]
                    ),
                    "tail_loss_increase_diff_vs_fro": float(variant["tail_loss_increase"] - baseline["tail_loss_increase"]),
                    "direction_cosine_to_polar_grad": float(variant["direction_cosine_to_polar_grad"]),
                    "alignment_ratio_to_polar_grad": float(variant["alignment"] / max(polar_grad["alignment"], 1e-300)),
                    "less_tail_drift_than_fro": bool(variant["tail_output_drift_fro"] < baseline["tail_output_drift_fro"]),
                    "less_tail_drift_than_polar_grad": bool(variant["tail_output_drift_fro"] < polar_grad["tail_output_drift_fro"]),
                    "gradient_momentum_cosine": float(variant["gradient_momentum_cosine"]),
                }
            )
        paired = pd.DataFrame(ratio_rows)
        drift_fro, drift_fro_low, drift_fro_high = log_ratio_ci95(paired["tail_output_drift_sq_ratio_vs_fro"])
        drift_polar, drift_polar_low, drift_polar_high = log_ratio_ci95(paired["tail_output_drift_sq_ratio_vs_polar_grad"])
        head_diff, head_diff_low, head_diff_high = ci95(paired["actual_head_loss_decrease_diff_vs_fro"])
        loss_diff, loss_diff_low, loss_diff_high = ci95(paired["tail_loss_increase_diff_vs_fro"])
        cosine, cosine_low, cosine_high = ci95(paired["direction_cosine_to_polar_grad"])
        align_ratio, align_ratio_low, align_ratio_high = log_ratio_ci95(paired["alignment_ratio_to_polar_grad"])
        momentum_cosine, momentum_cosine_low, momentum_cosine_high = ci95(paired["gradient_momentum_cosine"])
        rows.append(
            {
                "direction": direction,
                "seeds": int(paired["seed"].nunique()),
                "geomean_tail_output_drift_sq_ratio_vs_fro": drift_fro,
                "tail_output_drift_sq_ratio_vs_fro_ci95_low": drift_fro_low,
                "tail_output_drift_sq_ratio_vs_fro_ci95_high": drift_fro_high,
                "geomean_tail_output_drift_sq_ratio_vs_polar_grad": drift_polar,
                "tail_output_drift_sq_ratio_vs_polar_grad_ci95_low": drift_polar_low,
                "tail_output_drift_sq_ratio_vs_polar_grad_ci95_high": drift_polar_high,
                "less_tail_drift_than_fro_fraction": float(paired["less_tail_drift_than_fro"].mean()),
                "less_tail_drift_than_polar_grad_fraction": float(paired["less_tail_drift_than_polar_grad"].mean()),
                "mean_actual_head_loss_decrease_diff_vs_fro": head_diff,
                "actual_head_loss_decrease_diff_vs_fro_ci95_low": head_diff_low,
                "actual_head_loss_decrease_diff_vs_fro_ci95_high": head_diff_high,
                "mean_tail_loss_increase_diff_vs_fro": loss_diff,
                "tail_loss_increase_diff_vs_fro_ci95_low": loss_diff_low,
                "tail_loss_increase_diff_vs_fro_ci95_high": loss_diff_high,
                "mean_direction_cosine_to_polar_grad": cosine,
                "direction_cosine_to_polar_grad_ci95_low": cosine_low,
                "direction_cosine_to_polar_grad_ci95_high": cosine_high,
                "geomean_alignment_ratio_to_polar_grad": align_ratio,
                "alignment_ratio_to_polar_grad_ci95_low": align_ratio_low,
                "alignment_ratio_to_polar_grad_ci95_high": align_ratio_high,
                "mean_gradient_momentum_cosine": momentum_cosine,
                "gradient_momentum_cosine_ci95_low": momentum_cosine_low,
                "gradient_momentum_cosine_ci95_high": momentum_cosine_high,
            }
        )
    return pd.DataFrame(rows)


def summarize_long_tail_practical_muon_bridge(step_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, group in step_metrics.groupby("direction", observed=True, sort=False):
        if direction == "frobenius_grad":
            continue
        ratio_rows = []
        for (seed, trajectory_step), step_group in step_metrics.groupby(
            ["seed", "trajectory_step"],
            observed=True,
            sort=False,
        ):
            by_direction = step_group.set_index("direction")
            baseline = by_direction.loc["frobenius_grad"]
            polar_grad = by_direction.loc["polar_grad"]
            variant = by_direction.loc[direction]
            ratio_rows.append(
                {
                    "seed": int(seed),
                    "trajectory_step": int(trajectory_step),
                    "tail_output_drift_sq_ratio_vs_fro": float(
                        variant["tail_output_drift_fro"] ** 2 / max(baseline["tail_output_drift_fro"] ** 2, 1e-300)
                    ),
                    "tail_output_drift_sq_ratio_vs_polar_grad": float(
                        variant["tail_output_drift_fro"] ** 2 / max(polar_grad["tail_output_drift_fro"] ** 2, 1e-300)
                    ),
                    "actual_head_loss_decrease_diff_vs_fro": float(
                        variant["actual_head_loss_decrease"] - baseline["actual_head_loss_decrease"]
                    ),
                    "tail_loss_increase_diff_vs_fro": float(variant["tail_loss_increase"] - baseline["tail_loss_increase"]),
                    "direction_cosine_to_polar_grad": float(variant["direction_cosine_to_polar_grad"]),
                    "alignment_ratio_to_polar_grad": float(variant["alignment"] / max(polar_grad["alignment"], 1e-300)),
                    "less_tail_drift_than_fro": bool(variant["tail_output_drift_fro"] < baseline["tail_output_drift_fro"]),
                    "less_tail_drift_than_polar_grad": bool(variant["tail_output_drift_fro"] < polar_grad["tail_output_drift_fro"]),
                    "gradient_momentum_cosine": float(variant["gradient_momentum_cosine"]),
                }
            )
        paired = pd.DataFrame(ratio_rows)
        drift_fro, drift_fro_low, drift_fro_high = log_ratio_ci95(paired["tail_output_drift_sq_ratio_vs_fro"])
        drift_polar, drift_polar_low, drift_polar_high = log_ratio_ci95(paired["tail_output_drift_sq_ratio_vs_polar_grad"])
        head_diff, head_diff_low, head_diff_high = ci95(paired["actual_head_loss_decrease_diff_vs_fro"])
        loss_diff, loss_diff_low, loss_diff_high = ci95(paired["tail_loss_increase_diff_vs_fro"])
        cosine, cosine_low, cosine_high = ci95(paired["direction_cosine_to_polar_grad"])
        align_ratio, align_ratio_low, align_ratio_high = log_ratio_ci95(paired["alignment_ratio_to_polar_grad"])
        momentum_cosine, momentum_cosine_low, momentum_cosine_high = ci95(paired["gradient_momentum_cosine"])
        rows.append(
            {
                "direction": direction,
                "seeds": int(paired["seed"].nunique()),
                "trajectory_steps": int(paired["trajectory_step"].nunique()),
                "comparisons": int(len(paired)),
                "geomean_tail_output_drift_sq_ratio_vs_fro": drift_fro,
                "tail_output_drift_sq_ratio_vs_fro_ci95_low": drift_fro_low,
                "tail_output_drift_sq_ratio_vs_fro_ci95_high": drift_fro_high,
                "geomean_tail_output_drift_sq_ratio_vs_polar_grad": drift_polar,
                "tail_output_drift_sq_ratio_vs_polar_grad_ci95_low": drift_polar_low,
                "tail_output_drift_sq_ratio_vs_polar_grad_ci95_high": drift_polar_high,
                "less_tail_drift_than_fro_fraction": float(paired["less_tail_drift_than_fro"].mean()),
                "less_tail_drift_than_polar_grad_fraction": float(paired["less_tail_drift_than_polar_grad"].mean()),
                "mean_actual_head_loss_decrease_diff_vs_fro": head_diff,
                "actual_head_loss_decrease_diff_vs_fro_ci95_low": head_diff_low,
                "actual_head_loss_decrease_diff_vs_fro_ci95_high": head_diff_high,
                "mean_tail_loss_increase_diff_vs_fro": loss_diff,
                "tail_loss_increase_diff_vs_fro_ci95_low": loss_diff_low,
                "tail_loss_increase_diff_vs_fro_ci95_high": loss_diff_high,
                "mean_direction_cosine_to_polar_grad": cosine,
                "direction_cosine_to_polar_grad_ci95_low": cosine_low,
                "direction_cosine_to_polar_grad_ci95_high": cosine_high,
                "geomean_alignment_ratio_to_polar_grad": align_ratio,
                "alignment_ratio_to_polar_grad_ci95_low": align_ratio_low,
                "alignment_ratio_to_polar_grad_ci95_high": align_ratio_high,
                "mean_gradient_momentum_cosine": momentum_cosine,
                "gradient_momentum_cosine_ci95_low": momentum_cosine_low,
                "gradient_momentum_cosine_ci95_high": momentum_cosine_high,
            }
        )
    return pd.DataFrame(rows)
