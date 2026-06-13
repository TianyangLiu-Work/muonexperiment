from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from .long_tail_digits import (
    TinyMLP,
    alignment,
    apply_direction,
    batch,
    direction_list,
    dtype_from_name,
    full_metrics,
    layer_rank_rows,
    make_generator,
    restore,
    snapshot,
    split_long_tail_digits,
    train_digits_checkpoint,
    update_norms,
)
from .statistics import ci95, log_ratio_ci95


@dataclass(frozen=True)
class LongTailOneStepConfig:
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
    dtype: str = "float64"
    device: str = "cpu"


def run_long_tail_one_step(config: LongTailOneStepConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []
    layer_rows: list[dict] = []

    for seed in config.seeds:
        x, y, train_indices, head_train, tail_train, tail_eval = split_long_tail_digits(
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
        head_x = x[head_batch]
        head_y = y[head_batch]
        tail_x = x[tail_eval]
        tail_y = y[tail_eval]
        params = model.parameters()

        for param in params:
            param.grad = None
        head_loss = F.cross_entropy(model.logits(head_x), head_y)
        head_loss.backward()
        before = snapshot(params)
        base_tail_logits = model.logits(tail_x).detach()
        base_head = full_metrics(model, x, y, head_batch)
        base_tail = full_metrics(model, x, y, tail_eval)
        layer_rank, derived = layer_rank_rows(params, model.activation_matrices(tail_x))
        target_gain = config.target_head_gain_fraction * float(head_loss.detach().cpu())

        for geometry in ["frobenius", "spectral"]:
            directions = direction_list(params, geometry)
            alignment_value = alignment(params, directions)
            step_size = target_gain / max(alignment_value, 1e-300)
            update_fro, update_op = update_norms(directions, step_size)
            apply_direction(params, before, directions, step_size)
            after_tail_logits = model.logits(tail_x).detach()
            after_head = full_metrics(model, x, y, head_batch)
            after_tail = full_metrics(model, x, y, tail_eval)
            output_delta = after_tail_logits - base_tail_logits
            rows.append(
                {
                    "seed": int(seed),
                    "geometry": geometry,
                    "head_classes": ",".join(str(x) for x in config.head_classes),
                    "tail_classes": ",".join(str(x) for x in config.tail_classes),
                    "head_train_examples": int(head_train.numel()),
                    "tail_train_examples": int(tail_train.numel()),
                    "tail_eval_examples": int(tail_eval.numel()),
                    "warmup_steps": int(config.warmup_steps),
                    "hidden_dim": int(config.hidden_dim),
                    "head_loss_before": base_head["loss"],
                    "head_loss_after": after_head["loss"],
                    "actual_head_loss_decrease": base_head["loss"] - after_head["loss"],
                    "matched_first_order_head_gain": target_gain,
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
                    "nrG": derived["nrG"],
                    "stA_tail": derived["stA"],
                    "condition_score_tail": derived["condition_score"],
                }
            )
            for layer_record in layer_rank:
                layer_record = dict(layer_record)
                layer_record.update(
                    {
                        "seed": int(seed),
                        "geometry": geometry,
                        "tail_output_drift_fro": float(torch.linalg.norm(output_delta).cpu()),
                    }
                )
                layer_rows.append(layer_record)
            restore(params, before)

    step_metrics = pd.DataFrame(rows)
    layer_metrics = pd.DataFrame(layer_rows)
    pair_summary = summarize_long_tail_one_step(step_metrics)
    return step_metrics, pair_summary, layer_metrics


def summarize_long_tail_one_step(step_metrics: pd.DataFrame) -> pd.DataFrame:
    paired_rows = []
    for seed, group in step_metrics.groupby("seed", observed=True, sort=False):
        by_geometry = group.set_index("geometry")
        fro = by_geometry.loc["frobenius"]
        spectral = by_geometry.loc["spectral"]
        paired_rows.append(
            {
                "seed": int(seed),
                "tail_output_drift_sq_ratio_spectral_over_fro": float(
                    spectral["tail_output_drift_fro"] ** 2 / max(fro["tail_output_drift_fro"] ** 2, 1e-300)
                ),
                "tail_loss_increase_diff_spectral_minus_fro": float(
                    spectral["tail_loss_increase"] - fro["tail_loss_increase"]
                ),
                "tail_margin_drop_diff_spectral_minus_fro": float(
                    spectral["tail_margin_drop"] - fro["tail_margin_drop"]
                ),
                "tail_accuracy_drop_diff_spectral_minus_fro": float(
                    spectral["tail_accuracy_drop"] - fro["tail_accuracy_drop"]
                ),
                "actual_head_loss_decrease_diff_spectral_minus_fro": float(
                    spectral["actual_head_loss_decrease"] - fro["actual_head_loss_decrease"]
                ),
                "spectral_less_tail_output_drift": bool(spectral["tail_output_drift_fro"] < fro["tail_output_drift_fro"]),
                "spectral_less_tail_loss_increase": bool(spectral["tail_loss_increase"] < fro["tail_loss_increase"]),
                "nrG": float(spectral["nrG"]),
                "stA_tail": float(spectral["stA_tail"]),
                "condition_score_tail": float(spectral["condition_score_tail"]),
            }
        )
    paired = pd.DataFrame(paired_rows)
    drift_mean, drift_low, drift_high = log_ratio_ci95(paired["tail_output_drift_sq_ratio_spectral_over_fro"])
    loss_mean, loss_low, loss_high = ci95(paired["tail_loss_increase_diff_spectral_minus_fro"])
    margin_mean, margin_low, margin_high = ci95(paired["tail_margin_drop_diff_spectral_minus_fro"])
    accuracy_mean, accuracy_low, accuracy_high = ci95(paired["tail_accuracy_drop_diff_spectral_minus_fro"])
    head_mean, head_low, head_high = ci95(paired["actual_head_loss_decrease_diff_spectral_minus_fro"])
    return pd.DataFrame(
        [
            {
                "seeds": int(paired["seed"].nunique()),
                "geomean_tail_output_drift_sq_ratio_spectral_over_fro": drift_mean,
                "tail_output_drift_sq_ratio_ci95_low": drift_low,
                "tail_output_drift_sq_ratio_ci95_high": drift_high,
                "spectral_less_tail_output_drift_fraction": float(paired["spectral_less_tail_output_drift"].mean()),
                "mean_tail_loss_increase_diff_spectral_minus_fro": loss_mean,
                "tail_loss_increase_diff_ci95_low": loss_low,
                "tail_loss_increase_diff_ci95_high": loss_high,
                "mean_tail_margin_drop_diff_spectral_minus_fro": margin_mean,
                "tail_margin_drop_diff_ci95_low": margin_low,
                "tail_margin_drop_diff_ci95_high": margin_high,
                "mean_tail_accuracy_drop_diff_spectral_minus_fro": accuracy_mean,
                "tail_accuracy_drop_diff_ci95_low": accuracy_low,
                "tail_accuracy_drop_diff_ci95_high": accuracy_high,
                "mean_actual_head_loss_decrease_diff_spectral_minus_fro": head_mean,
                "actual_head_loss_decrease_diff_ci95_low": head_low,
                "actual_head_loss_decrease_diff_ci95_high": head_high,
                "mean_nrG": float(paired["nrG"].mean()),
                "mean_stA_tail": float(paired["stA_tail"].mean()),
                "mean_condition_score_tail": float(paired["condition_score_tail"].mean()),
            }
        ]
    )
