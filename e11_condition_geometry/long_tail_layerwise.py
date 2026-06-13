from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch
import torch.nn.functional as F

from .diagnostics import matrix_effective_rank, singular_values, stable_rank
from .long_tail_digits import (
    TinyMLP,
    activation_for_layer,
    add_noise,
    apply_layer_update,
    batch,
    dtype_from_name,
    full_metrics,
    jvp_tail_drift_sq,
    layer_direction,
    make_generator,
    restore_layer,
    split_long_tail_digits,
    tail_downstream_rank_metrics,
    train_digits_checkpoint,
)
from .statistics import ci95, corr_ci95, log_ratio_ci95


@dataclass(frozen=True)
class LongTailLayerwiseConfig:
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
    jvp_epsilon: float = 1e-4
    dtype: str = "float64"
    device: str = "cpu"


def _margin_metrics(logits: torch.Tensor, labels: torch.Tensor) -> float:
    true_logits = logits[torch.arange(logits.shape[0], device=logits.device), labels]
    masked = logits.clone()
    masked[torch.arange(logits.shape[0], device=logits.device), labels] = -torch.inf
    return float((true_logits - torch.max(masked, dim=1).values).mean().cpu())


def run_long_tail_layerwise(config: LongTailLayerwiseConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        generator = make_generator(seed + 33000, device)
        model = TinyMLP(input_dim=x.shape[1], hidden_dim=config.hidden_dim, output_dim=10, generator=generator, device=device, dtype=dtype)
        train_digits_checkpoint(model, x, y, train_indices, config, seed=seed, seed_offset=32000)

        batch_generator = make_generator(seed + 34000, device)
        head_batch = batch(head_train, config.head_batch_size, generator=batch_generator)
        head_x = add_noise(x[head_batch], config.train_noise_std, generator=batch_generator)
        head_y = y[head_batch]
        tail_x = x[tail_eval]
        tail_y = y[tail_eval]
        params = model.parameters()

        for param in params:
            param.grad = None
        head_loss = F.cross_entropy(model.logits(head_x), head_y)
        head_loss.backward()
        base_tail_logits = model.logits(tail_x).detach()
        base_tail = full_metrics(model, x, y, tail_eval)
        base_margin = _margin_metrics(base_tail_logits, tail_y)
        target_gain = config.target_head_gain_fraction * float(head_loss.detach().cpu())

        for layer, param in enumerate(params, start=1):
            activation = activation_for_layer(model, tail_x, layer)
            grad_sigmas = singular_values(param.grad.detach())
            activation_sigmas = singular_values(activation)
            nr_g = matrix_effective_rank(grad_sigmas)
            st_a = stable_rank(activation_sigmas)
            condition_score = nr_g / st_a if math.isfinite(st_a) and st_a > 0 else math.nan
            downstream_rank = tail_downstream_rank_metrics(model, tail_x, layer)
            tail_ssrank = downstream_rank["tail_sandwiched_stable_rank"]
            tail_operator_srank = downstream_rank["tail_local_operator_stable_rank"]
            theorem_condition_score = nr_g / tail_ssrank if math.isfinite(tail_ssrank) and tail_ssrank > 0 else math.nan
            local_operator_condition_score = (
                nr_g / tail_operator_srank if math.isfinite(tail_operator_srank) and tail_operator_srank > 0 else math.nan
            )
            for geometry in ["frobenius", "spectral"]:
                direction = layer_direction(param, geometry)
                alignment = float(torch.sum(param.grad.detach() * direction).cpu())
                step_size = target_gain / max(alignment, 1e-300)
                jvp_tail_drift_sq_value = jvp_tail_drift_sq(
                    model,
                    param,
                    direction,
                    tail_x,
                    base_tail_logits,
                    config.jvp_epsilon,
                )
                old = apply_layer_update(param, direction, step_size)
                tail_logits = model.logits(tail_x).detach()
                tail_delta = tail_logits - base_tail_logits
                tail = full_metrics(model, x, y, tail_eval)
                head_after = float(F.cross_entropy(model.logits(head_x), head_y).detach().cpu())
                rows.append(
                    {
                        "seed": int(seed),
                        "layer": int(layer),
                        "geometry": geometry,
                        "head_loss_before": float(head_loss.detach().cpu()),
                        "head_loss_after": head_after,
                        "actual_head_loss_decrease": float(head_loss.detach().cpu()) - head_after,
                        "matched_first_order_head_gain": float(target_gain),
                        "alignment": alignment,
                        "step_size": float(step_size),
                        "head_gradient_nuclear_rank": nr_g,
                        "tail_activation_stable_rank": st_a,
                        "condition_score": condition_score,
                        "tail_sandwiched_stable_rank": tail_ssrank,
                        "tail_local_operator_stable_rank": tail_operator_srank,
                        "theorem_condition_score": theorem_condition_score,
                        "local_operator_condition_score": local_operator_condition_score,
                        "jvp_tail_drift_sq": jvp_tail_drift_sq_value,
                        "scaled_jvp_tail_drift_sq": float(step_size**2 * jvp_tail_drift_sq_value),
                        "tail_output_drift_fro": float(torch.linalg.norm(tail_delta).cpu()),
                        "tail_output_drift_rms": float(torch.sqrt(torch.mean(tail_delta.square())).cpu()),
                        "tail_loss_before": base_tail["loss"],
                        "tail_loss_after": tail["loss"],
                        "tail_loss_increase": tail["loss"] - base_tail["loss"],
                        "tail_margin_before": base_margin,
                        "tail_margin_after": tail["mean_margin"],
                        "tail_margin_drop": base_margin - tail["mean_margin"],
                        "tail_accuracy_before": base_tail["accuracy"],
                        "tail_accuracy_after": tail["accuracy"],
                        "tail_accuracy_drop": base_tail["accuracy"] - tail["accuracy"],
                    }
                )
                restore_layer(param, old)

    metrics = pd.DataFrame(rows)
    summary = summarize_long_tail_layerwise(metrics)
    return metrics, summary


def summarize_long_tail_layerwise(metrics: pd.DataFrame) -> pd.DataFrame:
    paired_rows = []
    for (seed, layer), group in metrics.groupby(["seed", "layer"], observed=True, sort=False):
        by_geometry = group.set_index("geometry")
        fro = by_geometry.loc["frobenius"]
        spectral = by_geometry.loc["spectral"]
        paired_rows.append(
            {
                "seed": int(seed),
                "layer": int(layer),
                "condition_score": float(spectral["condition_score"]),
                "tail_sandwiched_stable_rank": float(spectral["tail_sandwiched_stable_rank"]),
                "tail_local_operator_stable_rank": float(spectral["tail_local_operator_stable_rank"]),
                "theorem_condition_score": float(spectral["theorem_condition_score"]),
                "local_operator_condition_score": float(spectral["local_operator_condition_score"]),
                "jvp_tail_drift_sq_ratio_spectral_over_fro": float(
                    spectral["jvp_tail_drift_sq"] / max(fro["jvp_tail_drift_sq"], 1e-300)
                ),
                "scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": float(
                    spectral["scaled_jvp_tail_drift_sq"] / max(fro["scaled_jvp_tail_drift_sq"], 1e-300)
                ),
                "observed_tail_drift_sq_ratio_spectral_over_fro": float(
                    spectral["tail_output_drift_fro"] ** 2 / max(fro["tail_output_drift_fro"] ** 2, 1e-300)
                ),
                "tail_loss_increase_diff_spectral_minus_fro": float(
                    spectral["tail_loss_increase"] - fro["tail_loss_increase"]
                ),
                "tail_margin_drop_diff_spectral_minus_fro": float(
                    spectral["tail_margin_drop"] - fro["tail_margin_drop"]
                ),
                "actual_head_loss_decrease_diff_spectral_minus_fro": float(
                    spectral["actual_head_loss_decrease"] - fro["actual_head_loss_decrease"]
                ),
                "spectral_less_jvp_tail_drift": bool(spectral["jvp_tail_drift_sq"] < fro["jvp_tail_drift_sq"]),
                "spectral_less_observed_tail_drift": bool(spectral["tail_output_drift_fro"] < fro["tail_output_drift_fro"]),
            }
        )
    paired = pd.DataFrame(paired_rows)
    rows = []
    for layer, group in paired.groupby("layer", observed=True, sort=False):
        jvp_mean, jvp_low, jvp_high = log_ratio_ci95(group["jvp_tail_drift_sq_ratio_spectral_over_fro"])
        scaled_jvp_mean, scaled_jvp_low, scaled_jvp_high = log_ratio_ci95(
            group["scaled_jvp_tail_drift_sq_ratio_spectral_over_fro"]
        )
        observed_mean, observed_low, observed_high = log_ratio_ci95(group["observed_tail_drift_sq_ratio_spectral_over_fro"])
        loss_mean, loss_low, loss_high = ci95(group["tail_loss_increase_diff_spectral_minus_fro"])
        margin_mean, margin_low, margin_high = ci95(group["tail_margin_drop_diff_spectral_minus_fro"])
        head_mean, head_low, head_high = ci95(group["actual_head_loss_decrease_diff_spectral_minus_fro"])
        rows.append(
            {
                "layer": int(layer),
                "seeds": int(group["seed"].nunique()),
                "mean_condition_score": float(group["condition_score"].mean()),
                "mean_tail_sandwiched_stable_rank": float(group["tail_sandwiched_stable_rank"].mean()),
                "mean_tail_local_operator_stable_rank": float(group["tail_local_operator_stable_rank"].mean()),
                "mean_theorem_condition_score": float(group["theorem_condition_score"].mean()),
                "mean_local_operator_condition_score": float(group["local_operator_condition_score"].mean()),
                "geomean_jvp_tail_drift_sq_ratio_spectral_over_fro": jvp_mean,
                "jvp_tail_drift_sq_ratio_ci95_low": jvp_low,
                "jvp_tail_drift_sq_ratio_ci95_high": jvp_high,
                "spectral_less_jvp_tail_drift_fraction": float(group["spectral_less_jvp_tail_drift"].mean()),
                "geomean_scaled_jvp_tail_drift_sq_ratio_spectral_over_fro": scaled_jvp_mean,
                "scaled_jvp_tail_drift_sq_ratio_ci95_low": scaled_jvp_low,
                "scaled_jvp_tail_drift_sq_ratio_ci95_high": scaled_jvp_high,
                "geomean_observed_tail_drift_sq_ratio_spectral_over_fro": observed_mean,
                "observed_tail_drift_sq_ratio_ci95_low": observed_low,
                "observed_tail_drift_sq_ratio_ci95_high": observed_high,
                "spectral_less_observed_tail_drift_fraction": float(group["spectral_less_observed_tail_drift"].mean()),
                "mean_tail_loss_increase_diff_spectral_minus_fro": loss_mean,
                "tail_loss_increase_diff_ci95_low": loss_low,
                "tail_loss_increase_diff_ci95_high": loss_high,
                "mean_tail_margin_drop_diff_spectral_minus_fro": margin_mean,
                "tail_margin_drop_diff_ci95_low": margin_low,
                "tail_margin_drop_diff_ci95_high": margin_high,
                "mean_actual_head_loss_decrease_diff_spectral_minus_fro": head_mean,
                "actual_head_loss_decrease_diff_ci95_low": head_low,
                "actual_head_loss_decrease_diff_ci95_high": head_high,
            }
        )
    overall = paired.copy()
    corr, corr_low, corr_high, points = corr_ci95(
        overall["local_operator_condition_score"],
        overall["observed_tail_drift_sq_ratio_spectral_over_fro"],
    )
    result = pd.DataFrame(rows)
    result["local_operator_score_observed_ratio_spearman_all_layers"] = corr
    result["local_operator_score_observed_ratio_spearman_ci95_low"] = corr_low
    result["local_operator_score_observed_ratio_spearman_ci95_high"] = corr_high
    result["local_operator_score_observed_ratio_spearman_points"] = points
    return result
