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
    local_linearization_metrics,
    make_generator,
    margins,
    restore,
    snapshot,
    split_long_tail_digits,
    train_digits_checkpoint,
    update_norms,
)
from .statistics import ci95, log_ratio_ci95


def _fraction_ci95(values: pd.Series) -> tuple[float, float, float]:
    mean, low, high = ci95(values)
    return mean, max(0.0, low), min(1.0, high)


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
        base_tail_pred = torch.argmax(base_tail_logits, dim=1)
        base_tail_margins = margins(base_tail_logits, tail_y)
        positive_margin_mask = base_tail_margins > 0
        positive_margin_count = int(positive_margin_mask.sum().cpu())
        positive_margin_fraction = float(positive_margin_mask.to(torch.float64).mean().cpu())
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
            after_tail_pred = torch.argmax(after_tail_logits, dim=1)
            after_head = full_metrics(model, x, y, head_batch)
            after_tail = full_metrics(model, x, y, tail_eval)
            output_delta = after_tail_logits - base_tail_logits
            centered_output_delta = output_delta - output_delta.mean(dim=1, keepdim=True)
            sample_indices = torch.arange(output_delta.shape[0], device=output_delta.device)
            true_logit_delta = output_delta[sample_indices, tail_y]
            base_competitor_logits = base_tail_logits.clone()
            base_competitor_logits[sample_indices, tail_y] = -torch.inf
            base_competitor_indices = torch.argmax(base_competitor_logits, dim=1)
            competitor_logit_delta = output_delta[sample_indices, base_competitor_indices]
            margin_delta = margins(after_tail_logits, tail_y) - base_tail_margins
            per_sample_delta_inf = torch.max(torch.abs(output_delta), dim=1).values
            actual_head_loss_decrease = base_head["loss"] - after_head["loss"]
            actual_head_gain_relative_error = abs(actual_head_loss_decrease - target_gain) / max(target_gain, 1e-300)
            if positive_margin_count > 0:
                certified_positive = per_sample_delta_inf[positive_margin_mask] < 0.5 * base_tail_margins[positive_margin_mask]
                certified_preserved_fraction = float(certified_positive.to(torch.float64).mean().cpu())
                positive_changed = after_tail_pred[positive_margin_mask] != base_tail_pred[positive_margin_mask]
                positive_prediction_changed_fraction = float(positive_changed.to(torch.float64).mean().cpu())
            else:
                certified_preserved_fraction = float("nan")
                positive_prediction_changed_fraction = float("nan")
            prediction_changed_fraction = float((after_tail_pred != base_tail_pred).to(torch.float64).mean().cpu())
            linearization = local_linearization_metrics(
                tail_x,
                base_tail_logits,
                after_tail_logits,
                before,
                directions,
                step_size,
            )
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
                    "actual_head_loss_decrease": actual_head_loss_decrease,
                    "actual_head_gain_relative_error": actual_head_gain_relative_error,
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
                    "tail_positive_margin_fraction_before": positive_margin_fraction,
                    "tail_margin_certified_preserved_fraction": certified_preserved_fraction,
                    "tail_prediction_changed_fraction": prediction_changed_fraction,
                    "tail_positive_margin_prediction_changed_fraction": positive_prediction_changed_fraction,
                    "tail_output_drift_fro": float(torch.linalg.norm(output_delta).cpu()),
                    "tail_output_drift_rms": float(torch.sqrt(torch.mean(output_delta.square())).cpu()),
                    "centered_tail_output_drift_fro": float(torch.linalg.norm(centered_output_delta).cpu()),
                    "true_logit_delta_fro": float(torch.linalg.norm(true_logit_delta).cpu()),
                    "competitor_logit_delta_fro": float(torch.linalg.norm(competitor_logit_delta).cpu()),
                    "margin_delta_fro": float(torch.linalg.norm(margin_delta).cpu()),
                    "margin_delta_rms": float(torch.sqrt(torch.mean(margin_delta.square())).cpu()),
                    **linearization,
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
                "centered_tail_output_drift_sq_ratio_spectral_over_fro": float(
                    spectral["centered_tail_output_drift_fro"] ** 2
                    / max(fro["centered_tail_output_drift_fro"] ** 2, 1e-300)
                ),
                "true_logit_delta_sq_ratio_spectral_over_fro": float(
                    spectral["true_logit_delta_fro"] ** 2 / max(fro["true_logit_delta_fro"] ** 2, 1e-300)
                ),
                "competitor_logit_delta_sq_ratio_spectral_over_fro": float(
                    spectral["competitor_logit_delta_fro"] ** 2
                    / max(fro["competitor_logit_delta_fro"] ** 2, 1e-300)
                ),
                "margin_delta_sq_ratio_spectral_over_fro": float(
                    spectral["margin_delta_fro"] ** 2 / max(fro["margin_delta_fro"] ** 2, 1e-300)
                ),
                "tail_loss_increase_diff_spectral_minus_fro": float(
                    spectral["tail_loss_increase"] - fro["tail_loss_increase"]
                ),
                "tail_loss_increase_frobenius": float(fro["tail_loss_increase"]),
                "tail_loss_increase_spectral": float(spectral["tail_loss_increase"]),
                "tail_margin_drop_diff_spectral_minus_fro": float(
                    spectral["tail_margin_drop"] - fro["tail_margin_drop"]
                ),
                "tail_margin_drop_frobenius": float(fro["tail_margin_drop"]),
                "tail_margin_drop_spectral": float(spectral["tail_margin_drop"]),
                "tail_accuracy_drop_diff_spectral_minus_fro": float(
                    spectral["tail_accuracy_drop"] - fro["tail_accuracy_drop"]
                ),
                "tail_accuracy_drop_frobenius": float(fro["tail_accuracy_drop"]),
                "tail_accuracy_drop_spectral": float(spectral["tail_accuracy_drop"]),
                "tail_loss_before": float(spectral["tail_loss_before"]),
                "tail_loss_after_frobenius": float(fro["tail_loss_after"]),
                "tail_loss_after_spectral": float(spectral["tail_loss_after"]),
                "tail_margin_before": float(spectral["tail_margin_before"]),
                "tail_margin_after_frobenius": float(fro["tail_margin_after"]),
                "tail_margin_after_spectral": float(spectral["tail_margin_after"]),
                "tail_accuracy_before": float(spectral["tail_accuracy_before"]),
                "tail_accuracy_after_frobenius": float(fro["tail_accuracy_after"]),
                "tail_accuracy_after_spectral": float(spectral["tail_accuracy_after"]),
                "tail_positive_margin_fraction_before": float(spectral["tail_positive_margin_fraction_before"]),
                "tail_margin_certified_preserved_fraction_frobenius": float(
                    fro["tail_margin_certified_preserved_fraction"]
                ),
                "tail_margin_certified_preserved_fraction_spectral": float(
                    spectral["tail_margin_certified_preserved_fraction"]
                ),
                "tail_prediction_changed_fraction_frobenius": float(fro["tail_prediction_changed_fraction"]),
                "tail_prediction_changed_fraction_spectral": float(spectral["tail_prediction_changed_fraction"]),
                "tail_positive_margin_prediction_changed_fraction_frobenius": float(
                    fro["tail_positive_margin_prediction_changed_fraction"]
                ),
                "tail_positive_margin_prediction_changed_fraction_spectral": float(
                    spectral["tail_positive_margin_prediction_changed_fraction"]
                ),
                "actual_head_loss_decrease_diff_spectral_minus_fro": float(
                    spectral["actual_head_loss_decrease"] - fro["actual_head_loss_decrease"]
                ),
                "actual_head_gain_relative_error_frobenius": float(fro["actual_head_gain_relative_error"]),
                "actual_head_gain_relative_error_spectral": float(spectral["actual_head_gain_relative_error"]),
                "spectral_less_tail_output_drift": bool(spectral["tail_output_drift_fro"] < fro["tail_output_drift_fro"]),
                "spectral_less_tail_loss_increase": bool(spectral["tail_loss_increase"] < fro["tail_loss_increase"]),
                "nrG": float(spectral["nrG"]),
                "stA_tail": float(spectral["stA_tail"]),
                "condition_score_tail": float(spectral["condition_score_tail"]),
            }
        )
    paired = pd.DataFrame(paired_rows)
    drift_mean, drift_low, drift_high = log_ratio_ci95(paired["tail_output_drift_sq_ratio_spectral_over_fro"])
    centered_mean, centered_low, centered_high = log_ratio_ci95(
        paired["centered_tail_output_drift_sq_ratio_spectral_over_fro"]
    )
    true_mean, true_low, true_high = log_ratio_ci95(paired["true_logit_delta_sq_ratio_spectral_over_fro"])
    competitor_mean, competitor_low, competitor_high = log_ratio_ci95(
        paired["competitor_logit_delta_sq_ratio_spectral_over_fro"]
    )
    margin_delta_mean, margin_delta_low, margin_delta_high = log_ratio_ci95(
        paired["margin_delta_sq_ratio_spectral_over_fro"]
    )
    loss_mean, loss_low, loss_high = ci95(paired["tail_loss_increase_diff_spectral_minus_fro"])
    loss_fro_mean, loss_fro_low, loss_fro_high = ci95(paired["tail_loss_increase_frobenius"])
    loss_spectral_mean, loss_spectral_low, loss_spectral_high = ci95(paired["tail_loss_increase_spectral"])
    margin_mean, margin_low, margin_high = ci95(paired["tail_margin_drop_diff_spectral_minus_fro"])
    margin_fro_mean, margin_fro_low, margin_fro_high = ci95(paired["tail_margin_drop_frobenius"])
    margin_spectral_mean, margin_spectral_low, margin_spectral_high = ci95(paired["tail_margin_drop_spectral"])
    accuracy_mean, accuracy_low, accuracy_high = ci95(paired["tail_accuracy_drop_diff_spectral_minus_fro"])
    accuracy_fro_mean, accuracy_fro_low, accuracy_fro_high = ci95(paired["tail_accuracy_drop_frobenius"])
    accuracy_spectral_mean, accuracy_spectral_low, accuracy_spectral_high = ci95(
        paired["tail_accuracy_drop_spectral"]
    )
    head_mean, head_low, head_high = ci95(paired["actual_head_loss_decrease_diff_spectral_minus_fro"])
    head_gain_error_fro_mean, head_gain_error_fro_low, head_gain_error_fro_high = ci95(
        paired["actual_head_gain_relative_error_frobenius"]
    )
    head_gain_error_spectral_mean, head_gain_error_spectral_low, head_gain_error_spectral_high = ci95(
        paired["actual_head_gain_relative_error_spectral"]
    )
    loss_before_mean, loss_before_low, loss_before_high = ci95(paired["tail_loss_before"])
    loss_after_fro_mean, loss_after_fro_low, loss_after_fro_high = ci95(paired["tail_loss_after_frobenius"])
    loss_after_spectral_mean, loss_after_spectral_low, loss_after_spectral_high = ci95(
        paired["tail_loss_after_spectral"]
    )
    margin_before_mean, margin_before_low, margin_before_high = ci95(paired["tail_margin_before"])
    margin_after_fro_mean, margin_after_fro_low, margin_after_fro_high = ci95(paired["tail_margin_after_frobenius"])
    margin_after_spectral_mean, margin_after_spectral_low, margin_after_spectral_high = ci95(
        paired["tail_margin_after_spectral"]
    )
    accuracy_before_mean, accuracy_before_low, accuracy_before_high = _fraction_ci95(paired["tail_accuracy_before"])
    accuracy_after_fro_mean, accuracy_after_fro_low, accuracy_after_fro_high = _fraction_ci95(
        paired["tail_accuracy_after_frobenius"]
    )
    accuracy_after_spectral_mean, accuracy_after_spectral_low, accuracy_after_spectral_high = _fraction_ci95(
        paired["tail_accuracy_after_spectral"]
    )
    positive_margin_mean, positive_margin_low, positive_margin_high = _fraction_ci95(
        paired["tail_positive_margin_fraction_before"]
    )
    certified_fro_mean, certified_fro_low, certified_fro_high = _fraction_ci95(
        paired["tail_margin_certified_preserved_fraction_frobenius"]
    )
    certified_spectral_mean, certified_spectral_low, certified_spectral_high = _fraction_ci95(
        paired["tail_margin_certified_preserved_fraction_spectral"]
    )
    changed_fro_mean, changed_fro_low, changed_fro_high = _fraction_ci95(paired["tail_prediction_changed_fraction_frobenius"])
    changed_spectral_mean, changed_spectral_low, changed_spectral_high = _fraction_ci95(
        paired["tail_prediction_changed_fraction_spectral"]
    )
    positive_changed_fro_mean, positive_changed_fro_low, positive_changed_fro_high = _fraction_ci95(
        paired["tail_positive_margin_prediction_changed_fraction_frobenius"]
    )
    positive_changed_spectral_mean, positive_changed_spectral_low, positive_changed_spectral_high = _fraction_ci95(
        paired["tail_positive_margin_prediction_changed_fraction_spectral"]
    )
    return pd.DataFrame(
        [
            {
                "seeds": int(paired["seed"].nunique()),
                "geomean_tail_output_drift_sq_ratio_spectral_over_fro": drift_mean,
                "tail_output_drift_sq_ratio_ci95_low": drift_low,
                "tail_output_drift_sq_ratio_ci95_high": drift_high,
                "geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro": centered_mean,
                "centered_tail_output_drift_sq_ratio_ci95_low": centered_low,
                "centered_tail_output_drift_sq_ratio_ci95_high": centered_high,
                "geomean_true_logit_delta_sq_ratio_spectral_over_fro": true_mean,
                "true_logit_delta_sq_ratio_ci95_low": true_low,
                "true_logit_delta_sq_ratio_ci95_high": true_high,
                "geomean_competitor_logit_delta_sq_ratio_spectral_over_fro": competitor_mean,
                "competitor_logit_delta_sq_ratio_ci95_low": competitor_low,
                "competitor_logit_delta_sq_ratio_ci95_high": competitor_high,
                "geomean_margin_delta_sq_ratio_spectral_over_fro": margin_delta_mean,
                "margin_delta_sq_ratio_ci95_low": margin_delta_low,
                "margin_delta_sq_ratio_ci95_high": margin_delta_high,
                "spectral_less_tail_output_drift_fraction": float(paired["spectral_less_tail_output_drift"].mean()),
                "mean_tail_loss_increase_diff_spectral_minus_fro": loss_mean,
                "tail_loss_increase_diff_ci95_low": loss_low,
                "tail_loss_increase_diff_ci95_high": loss_high,
                "mean_tail_loss_increase_frobenius": loss_fro_mean,
                "tail_loss_increase_frobenius_ci95_low": loss_fro_low,
                "tail_loss_increase_frobenius_ci95_high": loss_fro_high,
                "mean_tail_loss_increase_spectral": loss_spectral_mean,
                "tail_loss_increase_spectral_ci95_low": loss_spectral_low,
                "tail_loss_increase_spectral_ci95_high": loss_spectral_high,
                "mean_tail_margin_drop_diff_spectral_minus_fro": margin_mean,
                "tail_margin_drop_diff_ci95_low": margin_low,
                "tail_margin_drop_diff_ci95_high": margin_high,
                "mean_tail_margin_drop_frobenius": margin_fro_mean,
                "tail_margin_drop_frobenius_ci95_low": margin_fro_low,
                "tail_margin_drop_frobenius_ci95_high": margin_fro_high,
                "mean_tail_margin_drop_spectral": margin_spectral_mean,
                "tail_margin_drop_spectral_ci95_low": margin_spectral_low,
                "tail_margin_drop_spectral_ci95_high": margin_spectral_high,
                "mean_tail_accuracy_drop_diff_spectral_minus_fro": accuracy_mean,
                "tail_accuracy_drop_diff_ci95_low": accuracy_low,
                "tail_accuracy_drop_diff_ci95_high": accuracy_high,
                "mean_tail_accuracy_drop_frobenius": accuracy_fro_mean,
                "tail_accuracy_drop_frobenius_ci95_low": accuracy_fro_low,
                "tail_accuracy_drop_frobenius_ci95_high": accuracy_fro_high,
                "mean_tail_accuracy_drop_spectral": accuracy_spectral_mean,
                "tail_accuracy_drop_spectral_ci95_low": accuracy_spectral_low,
                "tail_accuracy_drop_spectral_ci95_high": accuracy_spectral_high,
                "mean_tail_loss_before": loss_before_mean,
                "tail_loss_before_ci95_low": loss_before_low,
                "tail_loss_before_ci95_high": loss_before_high,
                "mean_tail_loss_after_frobenius": loss_after_fro_mean,
                "tail_loss_after_frobenius_ci95_low": loss_after_fro_low,
                "tail_loss_after_frobenius_ci95_high": loss_after_fro_high,
                "mean_tail_loss_after_spectral": loss_after_spectral_mean,
                "tail_loss_after_spectral_ci95_low": loss_after_spectral_low,
                "tail_loss_after_spectral_ci95_high": loss_after_spectral_high,
                "mean_tail_margin_before": margin_before_mean,
                "tail_margin_before_ci95_low": margin_before_low,
                "tail_margin_before_ci95_high": margin_before_high,
                "mean_tail_margin_after_frobenius": margin_after_fro_mean,
                "tail_margin_after_frobenius_ci95_low": margin_after_fro_low,
                "tail_margin_after_frobenius_ci95_high": margin_after_fro_high,
                "mean_tail_margin_after_spectral": margin_after_spectral_mean,
                "tail_margin_after_spectral_ci95_low": margin_after_spectral_low,
                "tail_margin_after_spectral_ci95_high": margin_after_spectral_high,
                "mean_tail_accuracy_before": accuracy_before_mean,
                "tail_accuracy_before_ci95_low": accuracy_before_low,
                "tail_accuracy_before_ci95_high": accuracy_before_high,
                "mean_tail_accuracy_after_frobenius": accuracy_after_fro_mean,
                "tail_accuracy_after_frobenius_ci95_low": accuracy_after_fro_low,
                "tail_accuracy_after_frobenius_ci95_high": accuracy_after_fro_high,
                "mean_tail_accuracy_after_spectral": accuracy_after_spectral_mean,
                "tail_accuracy_after_spectral_ci95_low": accuracy_after_spectral_low,
                "tail_accuracy_after_spectral_ci95_high": accuracy_after_spectral_high,
                "mean_tail_positive_margin_fraction_before": positive_margin_mean,
                "tail_positive_margin_fraction_ci95_low": positive_margin_low,
                "tail_positive_margin_fraction_ci95_high": positive_margin_high,
                "mean_tail_margin_certified_preserved_fraction_frobenius": certified_fro_mean,
                "tail_margin_certified_preserved_fraction_frobenius_ci95_low": certified_fro_low,
                "tail_margin_certified_preserved_fraction_frobenius_ci95_high": certified_fro_high,
                "mean_tail_margin_certified_preserved_fraction_spectral": certified_spectral_mean,
                "tail_margin_certified_preserved_fraction_spectral_ci95_low": certified_spectral_low,
                "tail_margin_certified_preserved_fraction_spectral_ci95_high": certified_spectral_high,
                "mean_tail_prediction_changed_fraction_frobenius": changed_fro_mean,
                "tail_prediction_changed_fraction_frobenius_ci95_low": changed_fro_low,
                "tail_prediction_changed_fraction_frobenius_ci95_high": changed_fro_high,
                "mean_tail_prediction_changed_fraction_spectral": changed_spectral_mean,
                "tail_prediction_changed_fraction_spectral_ci95_low": changed_spectral_low,
                "tail_prediction_changed_fraction_spectral_ci95_high": changed_spectral_high,
                "mean_tail_positive_margin_prediction_changed_fraction_frobenius": positive_changed_fro_mean,
                "tail_positive_margin_prediction_changed_fraction_frobenius_ci95_low": positive_changed_fro_low,
                "tail_positive_margin_prediction_changed_fraction_frobenius_ci95_high": positive_changed_fro_high,
                "mean_tail_positive_margin_prediction_changed_fraction_spectral": positive_changed_spectral_mean,
                "tail_positive_margin_prediction_changed_fraction_spectral_ci95_low": positive_changed_spectral_low,
                "tail_positive_margin_prediction_changed_fraction_spectral_ci95_high": positive_changed_spectral_high,
                "mean_actual_head_loss_decrease_diff_spectral_minus_fro": head_mean,
                "actual_head_loss_decrease_diff_ci95_low": head_low,
                "actual_head_loss_decrease_diff_ci95_high": head_high,
                "mean_actual_head_gain_relative_error_frobenius": head_gain_error_fro_mean,
                "actual_head_gain_relative_error_frobenius_ci95_low": head_gain_error_fro_low,
                "actual_head_gain_relative_error_frobenius_ci95_high": head_gain_error_fro_high,
                "mean_actual_head_gain_relative_error_spectral": head_gain_error_spectral_mean,
                "actual_head_gain_relative_error_spectral_ci95_low": head_gain_error_spectral_low,
                "actual_head_gain_relative_error_spectral_ci95_high": head_gain_error_spectral_high,
                "mean_nrG": float(paired["nrG"].mean()),
                "mean_stA_tail": float(paired["stA_tail"].mean()),
                "mean_condition_score_tail": float(paired["condition_score_tail"].mean()),
            }
        ]
    )
