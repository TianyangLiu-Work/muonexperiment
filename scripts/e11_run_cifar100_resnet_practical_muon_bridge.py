from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    alignment,
    apply_direction,
    batch,
    build_cifar_resnet18,
    dtype_from_name,
    full_metrics,
    matrix_named_parameters,
    resolve_device,
    restore,
    snapshot,
    split_long_tail_cifar100,
    train_checkpoint,
    update_norms,
)
from e11_condition_geometry.diagnostics import matrix_effective_rank, matrix_view, singular_values
from e11_condition_geometry.long_tail_digits import make_generator, margins
from e11_condition_geometry.long_tail_muon_bridge import (
    direction_cosine,
    frobenius_grad_direction,
    newton_schulz_directions,
    polar_directions,
    tensor_fro_norm,
)
from e11_condition_geometry.reporting import fmt, markdown_table
from e11_condition_geometry.statistics import ci95, log_ratio_ci95


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_practical_muon_bridge")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_practical_muon_bridge")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_practical_muon_bridge.md")

STATE_SOURCES = ("adamw_matrix_trajectory", "ns_muon_matrix_trajectory")
DIRECTIONS = ("frobenius_grad", "polar_grad", "polar_momentum", "ns_momentum")
VARIANT_DIRECTIONS = ("polar_grad", "polar_momentum", "ns_momentum")
DISPLAY_DIRECTIONS = {
    "frobenius_grad": "Fro/GD",
    "polar_grad": "polar(G_t)",
    "polar_momentum": "polar(M_t)",
    "ns_momentum": "NS(M_t)",
}
DISPLAY_STATE_SOURCES = {
    "adamw_matrix_trajectory": "AdamW-state",
    "ns_muon_matrix_trajectory": "NS-Muon-state",
}
COLORS = {
    "polar_grad": "#0072B2",
    "polar_momentum": "#009E73",
    "ns_momentum": "#CC79A7",
}
LINESTYLES = {
    "adamw_matrix_trajectory": "-",
    "ns_muon_matrix_trajectory": "--",
}


@dataclass(frozen=True)
class ResNetPracticalBridgeConfig:
    trajectory_steps: int = 3
    momentum_beta: float = 0.9
    adamw_trajectory_lr: float = 3e-4
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    trajectory_weight_decay: float = 1e-4
    muon_trajectory_lr: float = 1e-4
    newton_schulz_steps: int = 5
    max_matrix_parameters: int | None = None


def _finite_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-300))


def _tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def collect_matrix_gradients(
    model: torch.nn.Module,
    named_params: list[tuple[str, torch.nn.Parameter]],
    head_x: torch.Tensor,
    head_y: torch.Tensor,
) -> tuple[float, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(head_x), head_y)
    loss.backward()
    grads = [param.grad.detach().clone() for _name, param in named_params]
    return float(loss.detach().cpu()), grads


def assign_matrix_grads(
    named_params: list[tuple[str, torch.nn.Parameter]],
    grads: list[torch.Tensor],
) -> None:
    for (_name, param), grad in zip(named_params, grads):
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


def update_rank_summary(
    named_params: list[tuple[str, torch.nn.Parameter]],
    directions: list[torch.Tensor],
) -> dict[str, float]:
    grad_ranks = []
    update_ranks = []
    for (_name, param), direction in zip(named_params, directions):
        if param.grad is None:
            continue
        grad_ranks.append(matrix_effective_rank(singular_values(param.grad.detach())))
        update_ranks.append(matrix_effective_rank(singular_values(direction.detach())))
    return {
        "mean_matrix_gradient_nuclear_rank": float(sum(grad_ranks) / len(grad_ranks)) if grad_ranks else math.nan,
        "mean_matrix_update_nuclear_rank": float(sum(update_ranks) / len(update_ranks)) if update_ranks else math.nan,
    }


def evaluate_bridge_direction(
    model: torch.nn.Module,
    named_params: list[tuple[str, torch.nn.Parameter]],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    head_batch: torch.Tensor,
    tail_eval: torch.Tensor,
    before: list[torch.Tensor],
    base_head: dict[str, float],
    base_tail: dict[str, float],
    base_tail_logits: torch.Tensor,
    base_tail_margins: torch.Tensor,
    base_tail_pred: torch.Tensor,
    directions: list[torch.Tensor],
    *,
    target_gain: float,
) -> dict[str, float]:
    alignment_value = alignment(named_params, directions)
    if alignment_value <= 0.0:
        raise RuntimeError(f"cannot match positive head gain with non-positive alignment={alignment_value:.6g}")
    step_size = target_gain / max(alignment_value, 1e-300)
    update_fro, update_op = update_norms(directions, step_size)
    rank_summary = update_rank_summary(named_params, directions)

    apply_direction(named_params, before, directions, step_size)
    with torch.no_grad():
        tail_logits = model(test_x[tail_eval]).detach()
        tail_pred = torch.argmax(tail_logits, dim=1)
    after_head = full_metrics(model, train_x, train_y, head_batch)
    after_tail = full_metrics(model, test_x, test_y, tail_eval)
    output_delta = tail_logits - base_tail_logits
    centered_output_delta = output_delta - output_delta.mean(dim=1, keepdim=True)
    sample_indices = torch.arange(output_delta.shape[0], device=output_delta.device)
    tail_y = test_y[tail_eval]
    true_logit_delta = output_delta[sample_indices, tail_y]
    base_competitor_logits = base_tail_logits.clone()
    base_competitor_logits[sample_indices, tail_y] = -torch.inf
    competitor_indices = torch.argmax(base_competitor_logits, dim=1)
    competitor_logit_delta = output_delta[sample_indices, competitor_indices]
    margin_delta = margins(tail_logits, tail_y) - base_tail_margins
    per_sample_delta_inf = torch.max(torch.abs(output_delta), dim=1).values
    positive_margin_mask = base_tail_margins > 0
    positive_margin_count = int(positive_margin_mask.sum().cpu())
    positive_margin_fraction = float(positive_margin_mask.to(torch.float64).mean().cpu())
    if positive_margin_count > 0:
        certified_positive = per_sample_delta_inf[positive_margin_mask] < 0.5 * base_tail_margins[positive_margin_mask]
        certified_preserved_fraction = float(certified_positive.to(torch.float64).mean().cpu())
        positive_changed = tail_pred[positive_margin_mask] != base_tail_pred[positive_margin_mask]
        positive_prediction_changed_fraction = float(positive_changed.to(torch.float64).mean().cpu())
    else:
        certified_preserved_fraction = math.nan
        positive_prediction_changed_fraction = math.nan
    prediction_changed_fraction = float((tail_pred != base_tail_pred).to(torch.float64).mean().cpu())
    actual_head_loss_decrease = base_head["loss"] - after_head["loss"]
    restore(named_params, before)

    return {
        "head_loss_before": base_head["loss"],
        "head_loss_after": after_head["loss"],
        "actual_head_loss_decrease": actual_head_loss_decrease,
        "actual_head_gain_relative_error": abs(actual_head_loss_decrease - target_gain) / max(target_gain, 1e-300),
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
        "tail_positive_margin_fraction_before": positive_margin_fraction,
        "tail_margin_certified_preserved_fraction": certified_preserved_fraction,
        "tail_prediction_changed_fraction": prediction_changed_fraction,
        "tail_positive_margin_prediction_changed_fraction": positive_prediction_changed_fraction,
        "tail_output_drift_fro": _tensor_float(torch.linalg.norm(output_delta)),
        "tail_output_drift_rms": _tensor_float(torch.sqrt(torch.mean(output_delta.square()))),
        "centered_tail_output_drift_fro": _tensor_float(torch.linalg.norm(centered_output_delta)),
        "true_logit_delta_fro": _tensor_float(torch.linalg.norm(true_logit_delta)),
        "competitor_logit_delta_fro": _tensor_float(torch.linalg.norm(competitor_logit_delta)),
        "margin_delta_fro": _tensor_float(torch.linalg.norm(margin_delta)),
        "margin_delta_rms": _tensor_float(torch.sqrt(torch.mean(margin_delta.square()))),
        "tail_output_jvp_fro": math.nan,
        "tail_output_linearization_residual_fro": math.nan,
        "tail_output_linearization_relative_error": math.nan,
        "update_fro_norm": update_fro,
        "update_op_norm": update_op,
        "step_size": float(step_size),
        "alignment": alignment_value,
        "mean_matrix_gradient_nuclear_rank": rank_summary["mean_matrix_gradient_nuclear_rank"],
        "mean_matrix_update_nuclear_rank": rank_summary["mean_matrix_update_nuclear_rank"],
    }


def apply_adamw_matrix_step(
    named_params: list[tuple[str, torch.nn.Parameter]],
    grads: list[torch.Tensor],
    exp_avg: list[torch.Tensor],
    exp_avg_sq: list[torch.Tensor],
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if not exp_avg:
        exp_avg = [torch.zeros_like(grad) for grad in grads]
        exp_avg_sq = [torch.zeros_like(grad) for grad in grads]
    bias_correction1 = 1.0 - float(beta1) ** int(step)
    bias_correction2 = 1.0 - float(beta2) ** int(step)
    with torch.no_grad():
        for (_name, param), grad, avg, avg_sq in zip(named_params, grads, exp_avg, exp_avg_sq):
            if weight_decay:
                param.mul_(1.0 - float(lr) * float(weight_decay))
            avg.mul_(float(beta1)).add_(grad, alpha=1.0 - float(beta1))
            avg_sq.mul_(float(beta2)).addcmul_(grad, grad, value=1.0 - float(beta2))
            denom = avg_sq.sqrt() / math.sqrt(max(bias_correction2, 1e-300)) + float(eps)
            param.addcdiv_(avg, denom, value=-float(lr) / max(bias_correction1, 1e-300))
    return exp_avg, exp_avg_sq


def apply_muon_matrix_step(
    named_params: list[tuple[str, torch.nn.Parameter]],
    directions: list[torch.Tensor],
    *,
    lr: float,
) -> None:
    with torch.no_grad():
        for (_name, param), direction in zip(named_params, directions):
            param.add_(direction, alpha=-float(lr))


def run_practical_bridge(
    base_config: Cifar100ResNetOneStepConfig,
    bridge_config: ResNetPracticalBridgeConfig,
    *,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(0)
    torch.manual_seed(0)
    device = resolve_device(base_config.device)
    dtype = dtype_from_name(base_config.dtype)
    rows: list[dict] = []

    for seed in base_config.seeds:
        if progress:
            print(f"[resnet-practical-bridge] seed={seed}: loading data", flush=True)
        train_x, train_y, test_x, test_y, train_indices, head_train, tail_train, tail_eval = split_long_tail_cifar100(
            base_config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        torch.manual_seed(seed + 33000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 33000)
        model = build_cifar_resnet18(device=device, dtype=dtype)
        if progress:
            print(
                f"[resnet-practical-bridge] seed={seed}: training {base_config.warmup_steps} warmup steps",
                flush=True,
            )
        train_checkpoint(model, train_x, train_y, train_indices, base_config, seed=seed)
        model.eval()
        named_params = matrix_named_parameters(model)
        if bridge_config.max_matrix_parameters is not None:
            named_params = named_params[: int(bridge_config.max_matrix_parameters)]
        updated_parameter_subset = (
            "conv_and_linear_weights_only"
            if bridge_config.max_matrix_parameters is None
            else f"first_{bridge_config.max_matrix_parameters}_conv_and_linear_weights_only"
        )
        checkpoint = snapshot(named_params)

        for state_source in STATE_SOURCES:
            restore(named_params, checkpoint)
            momentum: list[torch.Tensor] = []
            adam_exp_avg: list[torch.Tensor] = []
            adam_exp_avg_sq: list[torch.Tensor] = []
            if progress:
                print(f"[resnet-practical-bridge] seed={seed}: source={state_source}", flush=True)

            for trajectory_step in range(bridge_config.trajectory_steps):
                head_generator = make_generator(seed + 43000 + trajectory_step, device)
                head_batch = batch(head_train, base_config.head_batch_size, generator=head_generator)
                head_x = train_x[head_batch]
                head_y = train_y[head_batch]
                head_loss_value, current_grads = collect_matrix_gradients(model, named_params, head_x, head_y)
                momentum = momentum_update(momentum, current_grads, beta=bridge_config.momentum_beta)
                assign_matrix_grads(named_params, current_grads)

                before = snapshot(named_params)
                with torch.no_grad():
                    base_tail_logits = model(test_x[tail_eval]).detach()
                    base_tail_pred = torch.argmax(base_tail_logits, dim=1)
                tail_y = test_y[tail_eval]
                base_tail_margins = margins(base_tail_logits, tail_y)
                base_head = full_metrics(model, train_x, train_y, head_batch)
                base_tail = full_metrics(model, test_x, test_y, tail_eval)
                target_gain = base_config.target_head_gain_fraction * head_loss_value

                directions = {
                    "frobenius_grad": frobenius_grad_direction(current_grads),
                    "polar_grad": polar_directions(current_grads),
                    "polar_momentum": polar_directions(momentum),
                    "ns_momentum": newton_schulz_directions(momentum, steps=bridge_config.newton_schulz_steps),
                }
                polar_grad = directions["polar_grad"]
                grad_norm = tensor_fro_norm(current_grads)
                momentum_norm = tensor_fro_norm(momentum)
                grad_momentum_cosine = direction_cosine(current_grads, momentum)

                for direction_name in DIRECTIONS:
                    direction = directions[direction_name]
                    metrics = evaluate_bridge_direction(
                        model,
                        named_params,
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        head_batch,
                        tail_eval,
                        before,
                        base_head,
                        base_tail,
                        base_tail_logits,
                        base_tail_margins,
                        base_tail_pred,
                        direction,
                        target_gain=target_gain,
                    )
                    rows.append(
                        {
                            "seed": int(seed),
                            "state_source": state_source,
                            "trajectory_step": int(trajectory_step),
                            "direction": direction_name,
                            "dataset": "CIFAR100",
                            "model": "resnet18_cifar_stem",
                            "updated_parameter_subset": updated_parameter_subset,
                            "head_classes": ",".join(str(x) for x in base_config.head_classes),
                            "tail_classes": ",".join(str(x) for x in base_config.tail_classes),
                            "head_train_examples": int(head_train.numel()),
                            "tail_train_examples": int(tail_train.numel()),
                            "tail_eval_examples": int(tail_eval.numel()),
                            "warmup_steps": int(base_config.warmup_steps),
                            "momentum_beta": float(bridge_config.momentum_beta),
                            "trajectory_steps": int(bridge_config.trajectory_steps),
                            "adamw_trajectory_lr": float(bridge_config.adamw_trajectory_lr),
                            "adamw_beta2": float(bridge_config.adamw_beta2),
                            "adamw_eps": float(bridge_config.adamw_eps),
                            "trajectory_weight_decay": float(bridge_config.trajectory_weight_decay),
                            "muon_trajectory_lr": float(bridge_config.muon_trajectory_lr),
                            "newton_schulz_steps": int(bridge_config.newton_schulz_steps),
                            "matrix_parameter_count": int(len(named_params)),
                            "head_gradient_fro_norm": grad_norm,
                            "momentum_fro_norm": momentum_norm,
                            "gradient_momentum_cosine": grad_momentum_cosine,
                            "direction_cosine_to_polar_grad": direction_cosine(direction, polar_grad),
                            "device": str(device),
                            "dtype": str(dtype).replace("torch.", ""),
                            **metrics,
                        }
                    )

                restore(named_params, before)
                if state_source == "adamw_matrix_trajectory":
                    adam_exp_avg, adam_exp_avg_sq = apply_adamw_matrix_step(
                        named_params,
                        current_grads,
                        adam_exp_avg,
                        adam_exp_avg_sq,
                        step=trajectory_step + 1,
                        lr=bridge_config.adamw_trajectory_lr,
                        beta1=bridge_config.momentum_beta,
                        beta2=bridge_config.adamw_beta2,
                        eps=bridge_config.adamw_eps,
                        weight_decay=bridge_config.trajectory_weight_decay,
                    )
                elif state_source == "ns_muon_matrix_trajectory":
                    apply_muon_matrix_step(named_params, directions["ns_momentum"], lr=bridge_config.muon_trajectory_lr)
                else:
                    raise ValueError(f"unknown state source: {state_source}")

                if progress:
                    print(
                        f"[resnet-practical-bridge] seed={seed}: source={state_source} "
                        f"step={trajectory_step} done",
                        flush=True,
                    )

        del model, train_x, train_y, test_x, test_y
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)
    paired = paired_bridge_metrics(metrics)
    summary = summarize_bridge(paired)
    step_summary = summarize_bridge_by_step(paired)
    return metrics, paired, summary, step_summary


def paired_bridge_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    paired_rows = []
    for (seed, state_source, trajectory_step), group in metrics.groupby(
        ["seed", "state_source", "trajectory_step"],
        observed=True,
        sort=False,
    ):
        by_direction = group.set_index("direction")
        fro = by_direction.loc["frobenius_grad"]
        polar_grad = by_direction.loc["polar_grad"]
        for direction in VARIANT_DIRECTIONS:
            variant = by_direction.loc[direction]
            paired_rows.append(
                {
                    "seed": int(seed),
                    "state_source": state_source,
                    "trajectory_step": int(trajectory_step),
                    "direction": direction,
                    "warmup_steps": int(variant["warmup_steps"]),
                    "tail_eval_examples": int(variant["tail_eval_examples"]),
                    "tail_accuracy_before": float(variant["tail_accuracy_before"]),
                    "tail_positive_margin_fraction_before": float(variant["tail_positive_margin_fraction_before"]),
                    "tail_output_drift_sq_ratio_vs_fro": _finite_ratio(
                        float(variant["tail_output_drift_fro"]) ** 2,
                        float(fro["tail_output_drift_fro"]) ** 2,
                    ),
                    "centered_tail_output_drift_sq_ratio_vs_fro": _finite_ratio(
                        float(variant["centered_tail_output_drift_fro"]) ** 2,
                        float(fro["centered_tail_output_drift_fro"]) ** 2,
                    ),
                    "margin_delta_sq_ratio_vs_fro": _finite_ratio(
                        float(variant["margin_delta_fro"]) ** 2,
                        float(fro["margin_delta_fro"]) ** 2,
                    ),
                    "tail_output_drift_sq_ratio_vs_polar_grad": _finite_ratio(
                        float(variant["tail_output_drift_fro"]) ** 2,
                        float(polar_grad["tail_output_drift_fro"]) ** 2,
                    ),
                    "actual_head_loss_decrease_diff_vs_fro": float(variant["actual_head_loss_decrease"])
                    - float(fro["actual_head_loss_decrease"]),
                    "tail_loss_increase_diff_vs_fro": float(variant["tail_loss_increase"])
                    - float(fro["tail_loss_increase"]),
                    "tail_accuracy_drop_diff_vs_fro": float(variant["tail_accuracy_drop"]) - float(fro["tail_accuracy_drop"]),
                    "tail_margin_drop_diff_vs_fro": float(variant["tail_margin_drop"]) - float(fro["tail_margin_drop"]),
                    "actual_head_gain_relative_error_diff_vs_fro": float(variant["actual_head_gain_relative_error"])
                    - float(fro["actual_head_gain_relative_error"]),
                    "direction_cosine_to_polar_grad": float(variant["direction_cosine_to_polar_grad"]),
                    "gradient_momentum_cosine": float(variant["gradient_momentum_cosine"]),
                    "alignment_ratio_to_polar_grad": _finite_ratio(float(variant["alignment"]), float(polar_grad["alignment"])),
                    "update_fro_norm_ratio_vs_fro": _finite_ratio(float(variant["update_fro_norm"]), float(fro["update_fro_norm"])),
                    "update_op_norm_ratio_vs_fro": _finite_ratio(float(variant["update_op_norm"]), float(fro["update_op_norm"])),
                    "mean_matrix_gradient_nuclear_rank": float(variant["mean_matrix_gradient_nuclear_rank"]),
                    "mean_matrix_update_nuclear_rank": float(variant["mean_matrix_update_nuclear_rank"]),
                    "less_tail_drift_than_fro": bool(variant["tail_output_drift_fro"] < fro["tail_output_drift_fro"]),
                    "less_tail_drift_than_polar_grad": bool(
                        variant["tail_output_drift_fro"] < polar_grad["tail_output_drift_fro"]
                    ),
                    "tail_prediction_changed_fraction": float(variant["tail_prediction_changed_fraction"]),
                    "tail_margin_certified_preserved_fraction": float(
                        variant["tail_margin_certified_preserved_fraction"]
                    ),
                }
            )
    return pd.DataFrame(paired_rows)


def _seed_clustered_log_ratio_ci95(paired: pd.DataFrame, column: str) -> tuple[float, float, float]:
    positive = paired[paired[column] > 0].copy()
    if positive.empty:
        return math.nan, math.nan, math.nan
    seed_log_means = positive.assign(log_ratio=positive[column].map(math.log)).groupby(
        "seed",
        observed=True,
    )["log_ratio"].mean()
    return log_ratio_ci95(seed_log_means.map(math.exp))


def _seed_clustered_ci95(paired: pd.DataFrame, column: str) -> tuple[float, float, float]:
    seed_means = paired.groupby("seed", observed=True)[column].mean()
    return ci95(seed_means)


def summarize_bridge(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_source, direction), group in paired.groupby(
        ["state_source", "direction"],
        observed=True,
        sort=False,
    ):
        drift, drift_low, drift_high = _seed_clustered_log_ratio_ci95(group, "tail_output_drift_sq_ratio_vs_fro")
        centered, centered_low, centered_high = _seed_clustered_log_ratio_ci95(
            group,
            "centered_tail_output_drift_sq_ratio_vs_fro",
        )
        margin, margin_low, margin_high = _seed_clustered_log_ratio_ci95(group, "margin_delta_sq_ratio_vs_fro")
        polar, polar_low, polar_high = _seed_clustered_log_ratio_ci95(
            group,
            "tail_output_drift_sq_ratio_vs_polar_grad",
        )
        loss_diff, loss_low, loss_high = _seed_clustered_ci95(group, "tail_loss_increase_diff_vs_fro")
        acc_diff, acc_low, acc_high = _seed_clustered_ci95(group, "tail_accuracy_drop_diff_vs_fro")
        cosine, cosine_low, cosine_high = _seed_clustered_ci95(group, "direction_cosine_to_polar_grad")
        momentum_cosine, momentum_low, momentum_high = _seed_clustered_ci95(group, "gradient_momentum_cosine")
        head_error, head_error_low, head_error_high = _seed_clustered_ci95(
            group,
            "actual_head_gain_relative_error_diff_vs_fro",
        )
        seed_flags = group.groupby("seed", observed=True).agg(
            less_tail_drift_than_fro=("less_tail_drift_than_fro", "mean"),
            less_tail_drift_than_polar_grad=("less_tail_drift_than_polar_grad", "mean"),
        )
        rows.append(
            {
                "state_source": state_source,
                "direction": direction,
                "seeds": int(group["seed"].nunique()),
                "trajectory_steps": int(group["trajectory_step"].nunique()),
                "comparisons": int(len(group)),
                "mean_tail_accuracy_before": float(group["tail_accuracy_before"].mean()),
                "mean_tail_positive_margin_fraction_before": float(
                    group["tail_positive_margin_fraction_before"].mean()
                ),
                "geomean_tail_output_drift_sq_ratio_vs_fro": drift,
                "tail_output_drift_sq_ratio_vs_fro_ci95_low": drift_low,
                "tail_output_drift_sq_ratio_vs_fro_ci95_high": drift_high,
                "geomean_centered_tail_output_drift_sq_ratio_vs_fro": centered,
                "centered_tail_output_drift_sq_ratio_vs_fro_ci95_low": centered_low,
                "centered_tail_output_drift_sq_ratio_vs_fro_ci95_high": centered_high,
                "geomean_margin_delta_sq_ratio_vs_fro": margin,
                "margin_delta_sq_ratio_vs_fro_ci95_low": margin_low,
                "margin_delta_sq_ratio_vs_fro_ci95_high": margin_high,
                "geomean_tail_output_drift_sq_ratio_vs_polar_grad": polar,
                "tail_output_drift_sq_ratio_vs_polar_grad_ci95_low": polar_low,
                "tail_output_drift_sq_ratio_vs_polar_grad_ci95_high": polar_high,
                "less_tail_drift_than_fro_fraction": float(seed_flags["less_tail_drift_than_fro"].mean()),
                "less_tail_drift_than_polar_grad_fraction": float(
                    seed_flags["less_tail_drift_than_polar_grad"].mean()
                ),
                "mean_tail_loss_increase_diff_vs_fro": loss_diff,
                "tail_loss_increase_diff_vs_fro_ci95_low": loss_low,
                "tail_loss_increase_diff_vs_fro_ci95_high": loss_high,
                "mean_tail_accuracy_drop_diff_vs_fro": acc_diff,
                "tail_accuracy_drop_diff_vs_fro_ci95_low": acc_low,
                "tail_accuracy_drop_diff_vs_fro_ci95_high": acc_high,
                "mean_actual_head_gain_relative_error_diff_vs_fro": head_error,
                "actual_head_gain_relative_error_diff_vs_fro_ci95_low": head_error_low,
                "actual_head_gain_relative_error_diff_vs_fro_ci95_high": head_error_high,
                "mean_direction_cosine_to_polar_grad": cosine,
                "direction_cosine_to_polar_grad_ci95_low": cosine_low,
                "direction_cosine_to_polar_grad_ci95_high": cosine_high,
                "mean_gradient_momentum_cosine": momentum_cosine,
                "gradient_momentum_cosine_ci95_low": momentum_low,
                "gradient_momentum_cosine_ci95_high": momentum_high,
                "mean_matrix_gradient_nuclear_rank": float(group["mean_matrix_gradient_nuclear_rank"].mean()),
                "mean_matrix_update_nuclear_rank": float(group["mean_matrix_update_nuclear_rank"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_bridge_by_step(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_source, trajectory_step, direction), group in paired.groupby(
        ["state_source", "trajectory_step", "direction"],
        observed=True,
        sort=True,
    ):
        drift, drift_low, drift_high = log_ratio_ci95(group["tail_output_drift_sq_ratio_vs_fro"])
        margin, margin_low, margin_high = log_ratio_ci95(group["margin_delta_sq_ratio_vs_fro"])
        cosine, cosine_low, cosine_high = ci95(group["direction_cosine_to_polar_grad"])
        momentum_cosine, momentum_low, momentum_high = ci95(group["gradient_momentum_cosine"])
        rows.append(
            {
                "state_source": state_source,
                "trajectory_step": int(trajectory_step),
                "direction": direction,
                "seeds": int(group["seed"].nunique()),
                "comparisons": int(len(group)),
                "geomean_tail_output_drift_sq_ratio_vs_fro": drift,
                "tail_output_drift_sq_ratio_vs_fro_ci95_low": drift_low,
                "tail_output_drift_sq_ratio_vs_fro_ci95_high": drift_high,
                "geomean_margin_delta_sq_ratio_vs_fro": margin,
                "margin_delta_sq_ratio_vs_fro_ci95_low": margin_low,
                "margin_delta_sq_ratio_vs_fro_ci95_high": margin_high,
                "mean_direction_cosine_to_polar_grad": cosine,
                "direction_cosine_to_polar_grad_ci95_low": cosine_low,
                "direction_cosine_to_polar_grad_ci95_high": cosine_high,
                "mean_gradient_momentum_cosine": momentum_cosine,
                "gradient_momentum_cosine_ci95_low": momentum_low,
                "gradient_momentum_cosine_ci95_high": momentum_high,
            }
        )
    return pd.DataFrame(rows)


def write_figure(summary: pd.DataFrame, step_summary: pd.DataFrame, paired: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    for state_source in STATE_SOURCES:
        for direction in VARIANT_DIRECTIONS:
            sub = step_summary[
                step_summary["state_source"].eq(state_source) & step_summary["direction"].eq(direction)
            ].sort_values("trajectory_step")
            x = sub["trajectory_step"].to_numpy(dtype=float)
            y = sub["geomean_tail_output_drift_sq_ratio_vs_fro"].to_numpy(dtype=float)
            low = sub["tail_output_drift_sq_ratio_vs_fro_ci95_low"].to_numpy(dtype=float)
            high = sub["tail_output_drift_sq_ratio_vs_fro_ci95_high"].to_numpy(dtype=float)
            axes[0].errorbar(
                x,
                y,
                yerr=[y - low, high - y],
                marker="o",
                linewidth=1.6,
                capsize=3,
                color=COLORS[direction],
                linestyle=LINESTYLES[state_source],
                label=f"{DISPLAY_STATE_SOURCES[state_source]} {DISPLAY_DIRECTIONS[direction]}",
            )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("trajectory step")
    axes[0].set_ylabel("squared tail-logit drift ratio vs Fro/GD")
    axes[0].set_title("Matched-head-gain tail drift")
    axes[0].legend(frameon=False, fontsize=7)

    momentum = (
        paired.groupby(["seed", "state_source", "trajectory_step"], observed=True)["gradient_momentum_cosine"]
        .first()
        .reset_index()
    )
    for state_source, color in [
        ("adamw_matrix_trajectory", "#D55E00"),
        ("ns_muon_matrix_trajectory", "#009E73"),
    ]:
        sub = momentum[momentum["state_source"].eq(state_source)]
        for _seed, seed_group in sub.groupby("seed", observed=True, sort=False):
            axes[1].plot(
                seed_group["trajectory_step"],
                seed_group["gradient_momentum_cosine"],
                color=color,
                alpha=0.18,
                linewidth=1,
                linestyle=LINESTYLES[state_source],
            )
        mean = sub.groupby("trajectory_step", as_index=False, observed=True)["gradient_momentum_cosine"].mean()
        axes[1].plot(
            mean["trajectory_step"],
            mean["gradient_momentum_cosine"],
            color=color,
            linewidth=2.2,
            marker="o",
            linestyle=LINESTYLES[state_source],
            label=DISPLAY_STATE_SOURCES[state_source],
        )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("trajectory step")
    axes[1].set_ylabel("cosine(G_t, M_t)")
    axes[1].set_title("Momentum-gradient alignment")
    axes[1].legend(frameon=False)

    fig.suptitle("CIFAR-100-LT ResNet18 practical trajectory Muon bridge")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_practical_muon_bridge.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    base_config: Cifar100ResNetOneStepConfig,
    bridge_config: ResNetPracticalBridgeConfig,
    summary: pd.DataFrame,
    step_summary: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
) -> None:
    ordered = summary.sort_values(["state_source", "direction"]).copy()
    adam_ns = ordered[
        ordered["state_source"].eq("adamw_matrix_trajectory") & ordered["direction"].eq("ns_momentum")
    ].iloc[0]
    muon_ns = ordered[
        ordered["state_source"].eq("ns_muon_matrix_trajectory") & ordered["direction"].eq("ns_momentum")
    ].iloc[0]
    table_columns = [
        "state_source",
        "direction",
        "seeds",
        "trajectory_steps",
        "comparisons",
        "mean_tail_accuracy_before",
        "geomean_tail_output_drift_sq_ratio_vs_fro",
        "tail_output_drift_sq_ratio_vs_fro_ci95_low",
        "tail_output_drift_sq_ratio_vs_fro_ci95_high",
        "geomean_margin_delta_sq_ratio_vs_fro",
        "margin_delta_sq_ratio_vs_fro_ci95_low",
        "margin_delta_sq_ratio_vs_fro_ci95_high",
        "less_tail_drift_than_fro_fraction",
        "mean_gradient_momentum_cosine",
    ]
    lines = [
        "# E11 CIFAR-100-LT ResNet18 Practical Muon Trajectory Bridge",
        "",
        "This diagnostic moves the Muon-style bridge from a fixed toy checkpoint to",
        "sampled CIFAR-100-LT ResNet18 trajectory states. From the same tail-quality",
        "warmup checkpoint, each seed follows two matrix-weight-only state sources:",
        "AdamW and finite-step Newton-Schulz Muon. At every sampled state, the script",
        "matches the same first-order head-batch gain for Fro/GD, exact polar of the",
        "current gradient, exact polar of the momentum buffer, and finite-step",
        "Newton-Schulz momentum.",
        "",
        f"- Seeds: {len(base_config.seeds)}",
        f"- Trajectory steps per state source: {bridge_config.trajectory_steps}",
        f"- Warmup steps: {base_config.warmup_steps}",
        f"- Head train examples per class: {base_config.head_train_per_class}",
        f"- Tail train examples per class: {base_config.tail_train_per_class}",
        f"- Tail eval examples per class: {base_config.tail_eval_per_class}",
        f"- Target head first-order gain: {base_config.target_head_gain_fraction} * head-batch loss",
        f"- AdamW trajectory lr: {bridge_config.adamw_trajectory_lr}",
        f"- NS-Muon trajectory lr: {bridge_config.muon_trajectory_lr}",
        f"- Newton-Schulz steps: {bridge_config.newton_schulz_steps}",
        f"- Matrix-parameter limit: {bridge_config.max_matrix_parameters}",
        f"- Device/dtype request: {base_config.device}/{base_config.dtype}",
        "",
        f"![CIFAR-100-LT ResNet18 practical Muon bridge](../{figure_path.as_posix()})",
        "",
        "## Summary",
        "",
        markdown_table(ordered, table_columns),
        "",
        "## Readout",
        "",
        f"- On AdamW-sampled states, practical `NS(M_t)` has squared tail-logit drift ratio "
        f"{fmt(adam_ns['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
        f"[{fmt(adam_ns['tail_output_drift_sq_ratio_vs_fro_ci95_low'])}, "
        f"{fmt(adam_ns['tail_output_drift_sq_ratio_vs_fro_ci95_high'])}] relative to Fro/GD.",
        f"- On NS-Muon-sampled states, practical `NS(M_t)` has squared tail-logit drift ratio "
        f"{fmt(muon_ns['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
        f"[{fmt(muon_ns['tail_output_drift_sq_ratio_vs_fro_ci95_low'])}, "
        f"{fmt(muon_ns['tail_output_drift_sq_ratio_vs_fro_ci95_high'])}] relative to Fro/GD.",
        "",
        "Interpretation: this remains a local matched-head-gain mechanism check, not",
        "a full training benchmark or an optimizer-dominance claim. Its purpose is to",
        "test whether the spectral/polar tail-drift mechanism survives practical",
        "momentum, finite Newton-Schulz approximation, and sampled non-toy ResNet",
        "trajectory states.",
        "",
        "Artifacts:",
        f"- [metrics.csv](../{(output_dir / 'metrics.csv').as_posix()})",
        f"- [paired_metrics.csv](../{(output_dir / 'paired_metrics.csv').as_posix()})",
        f"- [summary.csv](../{(output_dir / 'summary.csv').as_posix()})",
        f"- [step_summary.csv](../{(output_dir / 'step_summary.csv').as_posix()})",
        f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
    ]
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--trajectory-steps", type=int, default=3)
    parser.add_argument("--target-head-gain-fraction", type=float, default=0.005)
    parser.add_argument("--head-train-per-class", type=int, default=None)
    parser.add_argument("--tail-train-per-class", type=int, default=None)
    parser.add_argument("--tail-eval-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adamw-trajectory-lr", type=float, default=3e-4)
    parser.add_argument("--muon-trajectory-lr", type=float, default=1e-4)
    parser.add_argument("--momentum-beta", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--trajectory-weight-decay", type=float, default=None)
    parser.add_argument("--newton-schulz-steps", type=int, default=5)
    parser.add_argument("--max-matrix-parameters", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--discussion-path", type=Path, default=DEFAULT_DISCUSSION_PATH)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> tuple[Cifar100ResNetOneStepConfig, ResNetPracticalBridgeConfig]:
    base_config = Cifar100ResNetOneStepConfig()
    if args.smoke:
        base_config = replace(
            base_config,
            seeds=(0,),
            head_classes=tuple(range(5)),
            tail_classes=tuple(range(5, 10)),
            head_train_per_class=20,
            tail_train_per_class=5,
            tail_eval_per_class=5,
            warmup_steps=2,
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
        bridge_config = ResNetPracticalBridgeConfig(
            trajectory_steps=1,
            momentum_beta=args.momentum_beta,
            adamw_trajectory_lr=args.adamw_trajectory_lr,
            adamw_beta2=args.adamw_beta2,
            adamw_eps=args.adamw_eps,
            trajectory_weight_decay=0.0 if args.trajectory_weight_decay is None else args.trajectory_weight_decay,
            muon_trajectory_lr=args.muon_trajectory_lr,
            newton_schulz_steps=args.newton_schulz_steps,
            max_matrix_parameters=2 if args.max_matrix_parameters is None else args.max_matrix_parameters,
        )
    else:
        base_config = replace(base_config, seeds=tuple(range(args.seeds)), warmup_steps=args.warmup_steps)
        bridge_config = ResNetPracticalBridgeConfig(
            trajectory_steps=args.trajectory_steps,
            momentum_beta=args.momentum_beta,
            adamw_trajectory_lr=args.adamw_trajectory_lr,
            adamw_beta2=args.adamw_beta2,
            adamw_eps=args.adamw_eps,
            trajectory_weight_decay=(
                base_config.weight_decay if args.trajectory_weight_decay is None else args.trajectory_weight_decay
            ),
            muon_trajectory_lr=args.muon_trajectory_lr,
            newton_schulz_steps=args.newton_schulz_steps,
            max_matrix_parameters=args.max_matrix_parameters,
        )
    if args.device is not None:
        base_config = replace(base_config, device=args.device)
    if args.target_head_gain_fraction is not None:
        base_config = replace(base_config, target_head_gain_fraction=args.target_head_gain_fraction)
    if args.head_train_per_class is not None:
        base_config = replace(base_config, head_train_per_class=args.head_train_per_class)
    if args.tail_train_per_class is not None:
        base_config = replace(base_config, tail_train_per_class=args.tail_train_per_class)
    if args.tail_eval_per_class is not None:
        base_config = replace(base_config, tail_eval_per_class=args.tail_eval_per_class)
    if args.batch_size is not None:
        base_config = replace(base_config, warmup_batch_size=args.batch_size)
    if args.head_batch_size is not None:
        base_config = replace(base_config, head_batch_size=args.head_batch_size)
    if args.lr is not None:
        base_config = replace(base_config, lr=args.lr)
    if args.weight_decay is not None:
        base_config = replace(base_config, weight_decay=args.weight_decay)
    if args.download is not None:
        base_config = replace(base_config, download=args.download)
    return base_config, bridge_config


def main() -> None:
    args = parse_args()
    base_config, bridge_config = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, paired, summary, step_summary = run_practical_bridge(base_config, bridge_config, progress=args.progress)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    step_summary.to_csv(args.output_dir / "step_summary.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "base_config": asdict(base_config),
                "bridge_config": asdict(bridge_config),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(summary, step_summary, paired, args.figure_dir)
    write_discussion(base_config, bridge_config, summary, step_summary, figure_path, args.output_dir, args.discussion_path)
    print(f"saved CIFAR-100-LT ResNet18 practical Muon bridge to {args.output_dir}")
    print(f"metric rows={len(metrics)}, paired rows={len(paired)}, summary rows={len(summary)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {args.discussion_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
