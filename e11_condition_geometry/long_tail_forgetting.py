from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch
import torch.nn.functional as F

from .long_tail_digits import (
    TinyMLP,
    add_noise,
    alignment,
    apply_direction,
    batch,
    direction_list,
    dtype_from_name,
    full_metrics,
    layer_rank_rows,
    make_generator,
    restore,
    split_long_tail_digits,
    snapshot,
    train_digits_checkpoint,
)
from .statistics import ci95, corr_ci95, log_ratio_ci95


@dataclass(frozen=True)
class LongTailForgettingConfig:
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
    head_only_steps: int = 8
    lr: float = 1e-2
    train_noise_std: float = 0.05
    target_head_gain_fraction: float = 0.02
    dtype: str = "float64"
    device: str = "cpu"


def _scheduled_head_batches(
    model: TinyMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    head_train: torch.Tensor,
    config: LongTailForgettingConfig,
    *,
    seed: int,
) -> list[dict]:
    schedule = []
    for step in range(1, config.head_only_steps + 1):
        generator = make_generator(seed + 23000 + step, x.device)
        batch_indices = batch(head_train, config.head_batch_size, generator=generator)
        batch_x = add_noise(x[batch_indices], config.train_noise_std, generator=generator)
        batch_y = y[batch_indices]
        with torch.no_grad():
            reference_head_loss = F.cross_entropy(model.logits(batch_x), batch_y)
        schedule.append(
            {
                "step": step,
                "batch_x": batch_x.detach().clone(),
                "batch_y": batch_y.detach().clone(),
                "target_gain": float(config.target_head_gain_fraction * float(reference_head_loss.cpu())),
            }
        )
    return schedule


def _tail_logit_drift(model: TinyMLP, tail_x: torch.Tensor, base_tail_logits: torch.Tensor) -> tuple[float, float]:
    current = model.logits(tail_x).detach()
    delta = current - base_tail_logits
    return float(torch.linalg.norm(delta).cpu()), float(torch.sqrt(torch.mean(delta.square())).cpu())


def _record_row(
    *,
    rows: list[dict],
    model: TinyMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    head_batch_x: torch.Tensor | None,
    head_batch_y: torch.Tensor | None,
    tail_eval: torch.Tensor,
    tail_x: torch.Tensor,
    tail_y: torch.Tensor,
    base_tail_logits: torch.Tensor,
    base_tail: dict[str, float],
    seed: int,
    geometry: str,
    step: int,
    target_gain: float,
    actual_head_loss_decrease: float,
    cumulative_condition_proxy: float,
    condition_score_tail: float,
    nrG: float,
    stA_tail: float,
) -> None:
    tail = full_metrics(model, x, y, tail_eval)
    drift_fro, drift_rms = _tail_logit_drift(model, tail_x, base_tail_logits)
    if head_batch_x is None or head_batch_y is None:
        head_loss = math.nan
    else:
        with torch.no_grad():
            head_loss = float(F.cross_entropy(model.logits(head_batch_x), head_batch_y).cpu())
    rows.append(
        {
            "seed": int(seed),
            "geometry": geometry,
            "step": int(step),
            "target_first_order_head_gain": float(target_gain),
            "actual_head_loss_decrease": float(actual_head_loss_decrease),
            "head_batch_loss_after": head_loss,
            "tail_loss": tail["loss"],
            "tail_loss_increase": tail["loss"] - base_tail["loss"],
            "tail_accuracy": tail["accuracy"],
            "tail_accuracy_drop": base_tail["accuracy"] - tail["accuracy"],
            "tail_margin": tail["mean_margin"],
            "tail_margin_drop": base_tail["mean_margin"] - tail["mean_margin"],
            "tail_output_drift_fro": drift_fro,
            "tail_output_drift_rms": drift_rms,
            "cumulative_condition_proxy": float(cumulative_condition_proxy),
            "condition_score_tail": float(condition_score_tail),
            "nrG": float(nrG),
            "stA_tail": float(stA_tail),
            "tail_eval_examples": int(tail_y.numel()),
        }
    )


def run_long_tail_forgetting(config: LongTailForgettingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        generator = make_generator(seed + 24000, device)
        model = TinyMLP(input_dim=x.shape[1], hidden_dim=config.hidden_dim, output_dim=10, generator=generator, device=device, dtype=dtype)
        train_digits_checkpoint(model, x, y, train_indices, config, seed=seed, seed_offset=22000)
        initial_params = snapshot(model.parameters())
        tail_x = x[tail_eval]
        tail_y = y[tail_eval]
        base_tail_logits = model.logits(tail_x).detach()
        base_tail = full_metrics(model, x, y, tail_eval)
        schedule = _scheduled_head_batches(model, x, y, head_train, config, seed=seed)

        for geometry in ["frobenius", "spectral"]:
            restore(model.parameters(), initial_params)
            cumulative_condition_proxy = 0.0
            _record_row(
                rows=rows,
                model=model,
                x=x,
                y=y,
                head_batch_x=None,
                head_batch_y=None,
                tail_eval=tail_eval,
                tail_x=tail_x,
                tail_y=tail_y,
                base_tail_logits=base_tail_logits,
                base_tail=base_tail,
                seed=seed,
                geometry=geometry,
                step=0,
                target_gain=0.0,
                actual_head_loss_decrease=0.0,
                cumulative_condition_proxy=0.0,
                condition_score_tail=math.nan,
                nrG=math.nan,
                stA_tail=math.nan,
            )
            for item in schedule:
                params = model.parameters()
                for param in params:
                    param.grad = None
                batch_x = item["batch_x"]
                batch_y = item["batch_y"]
                head_loss_before = F.cross_entropy(model.logits(batch_x), batch_y)
                head_loss_before.backward()
                layer_rank, derived = layer_rank_rows(params, model.activation_matrices(tail_x))
                condition_score_tail = float(derived["condition_score"])
                if math.isfinite(condition_score_tail) and condition_score_tail > 0:
                    cumulative_condition_proxy += math.sqrt(condition_score_tail)
                before = snapshot(params)
                directions = direction_list(params, geometry)
                alignment_value = alignment(params, directions)
                target_gain = float(item["target_gain"])
                step_size = target_gain / max(alignment_value, 1e-300)
                apply_direction(params, before, directions, step_size)
                with torch.no_grad():
                    actual_decrease = float((head_loss_before - F.cross_entropy(model.logits(batch_x), batch_y)).cpu())
                _record_row(
                    rows=rows,
                    model=model,
                    x=x,
                    y=y,
                    head_batch_x=batch_x,
                    head_batch_y=batch_y,
                    tail_eval=tail_eval,
                    tail_x=tail_x,
                    tail_y=tail_y,
                    base_tail_logits=base_tail_logits,
                    base_tail=base_tail,
                    seed=seed,
                    geometry=geometry,
                    step=int(item["step"]),
                    target_gain=target_gain,
                    actual_head_loss_decrease=actual_decrease,
                    cumulative_condition_proxy=cumulative_condition_proxy,
                    condition_score_tail=condition_score_tail,
                    nrG=float(derived["nrG"]),
                    stA_tail=float(derived["stA"]),
                )

    step_metrics = pd.DataFrame(rows)
    summary = summarize_long_tail_forgetting(step_metrics, config)
    return step_metrics, summary


def summarize_long_tail_forgetting(step_metrics: pd.DataFrame, config: LongTailForgettingConfig) -> pd.DataFrame:
    final = step_metrics[step_metrics["step"] == config.head_only_steps]
    area = (
        step_metrics.groupby(["seed", "geometry"], observed=True, sort=False)
        .agg(
            tail_output_drift_area=("tail_output_drift_fro", "sum"),
            tail_loss_increase_area=("tail_loss_increase", "sum"),
        )
        .reset_index()
    )
    paired_rows = []
    for seed in sorted(step_metrics["seed"].unique()):
        final_seed = final[final["seed"] == seed].set_index("geometry")
        area_seed = area[area["seed"] == seed].set_index("geometry")
        fro = final_seed.loc["frobenius"]
        spectral = final_seed.loc["spectral"]
        fro_area = area_seed.loc["frobenius"]
        spectral_area = area_seed.loc["spectral"]
        paired_rows.append(
            {
                "seed": int(seed),
                "final_tail_output_drift_sq_ratio_spectral_over_fro": float(
                    spectral["tail_output_drift_fro"] ** 2 / max(fro["tail_output_drift_fro"] ** 2, 1e-300)
                ),
                "tail_output_drift_area_ratio_spectral_over_fro": float(
                    spectral_area["tail_output_drift_area"] / max(fro_area["tail_output_drift_area"], 1e-300)
                ),
                "final_tail_loss_increase_diff_spectral_minus_fro": float(
                    spectral["tail_loss_increase"] - fro["tail_loss_increase"]
                ),
                "final_tail_margin_drop_diff_spectral_minus_fro": float(
                    spectral["tail_margin_drop"] - fro["tail_margin_drop"]
                ),
                "final_tail_accuracy_drop_diff_spectral_minus_fro": float(
                    spectral["tail_accuracy_drop"] - fro["tail_accuracy_drop"]
                ),
                "final_actual_head_loss_decrease_diff_spectral_minus_fro": float(
                    spectral["actual_head_loss_decrease"] - fro["actual_head_loss_decrease"]
                ),
                "spectral_less_final_tail_output_drift": bool(spectral["tail_output_drift_fro"] < fro["tail_output_drift_fro"]),
                "spectral_less_tail_output_drift_area": bool(
                    spectral_area["tail_output_drift_area"] < fro_area["tail_output_drift_area"]
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    final_drift_mean, final_drift_low, final_drift_high = log_ratio_ci95(
        paired["final_tail_output_drift_sq_ratio_spectral_over_fro"]
    )
    area_mean, area_low, area_high = log_ratio_ci95(paired["tail_output_drift_area_ratio_spectral_over_fro"])
    loss_mean, loss_low, loss_high = ci95(paired["final_tail_loss_increase_diff_spectral_minus_fro"])
    margin_mean, margin_low, margin_high = ci95(paired["final_tail_margin_drop_diff_spectral_minus_fro"])
    accuracy_mean, accuracy_low, accuracy_high = ci95(paired["final_tail_accuracy_drop_diff_spectral_minus_fro"])
    head_mean, head_low, head_high = ci95(paired["final_actual_head_loss_decrease_diff_spectral_minus_fro"])
    post_step = step_metrics[step_metrics["step"] > 0]
    corr_rows = {}
    for geometry, group in post_step.groupby("geometry", observed=True, sort=False):
        corr, corr_low, corr_high, points = corr_ci95(group["cumulative_condition_proxy"], group["tail_output_drift_fro"])
        corr_rows[f"{geometry}_proxy_drift_spearman"] = corr
        corr_rows[f"{geometry}_proxy_drift_spearman_ci95_low"] = corr_low
        corr_rows[f"{geometry}_proxy_drift_spearman_ci95_high"] = corr_high
        corr_rows[f"{geometry}_proxy_drift_points"] = points
    return pd.DataFrame(
        [
            {
                "seeds": int(step_metrics["seed"].nunique()),
                "head_only_steps": int(config.head_only_steps),
                "geomean_final_tail_output_drift_sq_ratio_spectral_over_fro": final_drift_mean,
                "final_tail_output_drift_sq_ratio_ci95_low": final_drift_low,
                "final_tail_output_drift_sq_ratio_ci95_high": final_drift_high,
                "spectral_less_final_tail_output_drift_fraction": float(
                    paired["spectral_less_final_tail_output_drift"].mean()
                ),
                "geomean_tail_output_drift_area_ratio_spectral_over_fro": area_mean,
                "tail_output_drift_area_ratio_ci95_low": area_low,
                "tail_output_drift_area_ratio_ci95_high": area_high,
                "spectral_less_tail_output_drift_area_fraction": float(
                    paired["spectral_less_tail_output_drift_area"].mean()
                ),
                "mean_final_tail_loss_increase_diff_spectral_minus_fro": loss_mean,
                "final_tail_loss_increase_diff_ci95_low": loss_low,
                "final_tail_loss_increase_diff_ci95_high": loss_high,
                "mean_final_tail_margin_drop_diff_spectral_minus_fro": margin_mean,
                "final_tail_margin_drop_diff_ci95_low": margin_low,
                "final_tail_margin_drop_diff_ci95_high": margin_high,
                "mean_final_tail_accuracy_drop_diff_spectral_minus_fro": accuracy_mean,
                "final_tail_accuracy_drop_diff_ci95_low": accuracy_low,
                "final_tail_accuracy_drop_diff_ci95_high": accuracy_high,
                "mean_final_actual_head_loss_decrease_diff_spectral_minus_fro": head_mean,
                "final_actual_head_loss_decrease_diff_ci95_low": head_low,
                "final_actual_head_loss_decrease_diff_ci95_high": head_high,
                **corr_rows,
            }
        ]
    )
