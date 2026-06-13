from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
import torch
import torch.nn.functional as F

from .long_tail_digits import (
    TinyMLP,
    add_noise,
    batch,
    dtype_from_name,
    full_metrics,
    make_generator,
    restore,
    snapshot,
    split_long_tail_digits,
)
from .long_tail_muon_bridge import (
    apply_practical_muon_step,
    collect_gradients,
    momentum_update,
    newton_schulz_directions,
)
from .statistics import ci95, log_ratio_ci95


@dataclass(frozen=True)
class LongTailPracticalTrainingConfig:
    seeds: tuple[int, ...] = tuple(range(20))
    head_classes: tuple[int, ...] = (0, 1, 2, 3, 4)
    tail_classes: tuple[int, ...] = (5, 6, 7, 8, 9)
    head_train_per_class: int = 100
    tail_train_per_class: int = 10
    tail_eval_per_class: int = 40
    hidden_dim: int = 32
    train_steps: int = 80
    train_batch_size: int = 64
    adam_lr: float = 1e-2
    muon_lr: float = 3e-2
    momentum_beta: float = 0.9
    newton_schulz_steps: int = 5
    train_noise_std: float = 0.05
    dtype: str = "float64"
    device: str = "cpu"


@dataclass(frozen=True)
class LongTailPracticalTrainingLRSweepConfig:
    muon_lrs: tuple[float, ...] = (3e-3, 1e-2, 3e-2, 1e-1)
    base_config: LongTailPracticalTrainingConfig = LongTailPracticalTrainingConfig()


def _is_tail(labels: torch.Tensor, tail_classes: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros(labels.shape, device=labels.device, dtype=torch.bool)
    for label in tail_classes:
        mask |= labels == int(label)
    return mask


def _tail_logit_drift(model: TinyMLP, tail_x: torch.Tensor, base_tail_logits: torch.Tensor) -> tuple[float, float]:
    current = model.logits(tail_x).detach()
    delta = current - base_tail_logits
    return float(torch.linalg.norm(delta).cpu()), float(torch.sqrt(torch.mean(delta.square())).cpu())


def _record_training_row(
    *,
    rows: list[dict],
    model: TinyMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: torch.Tensor,
    head_train: torch.Tensor,
    tail_train: torch.Tensor,
    tail_eval: torch.Tensor,
    base_tail_logits: torch.Tensor,
    seed: int,
    optimizer_name: str,
    step: int,
    batch_loss_before: float,
    batch_loss_after: float,
    batch_tail_examples: int,
    update_fro_norm: float,
    update_op_norm: float,
    gradient_momentum_cosine: float,
) -> None:
    train = full_metrics(model, x, y, train_indices)
    head = full_metrics(model, x, y, head_train)
    tail_train_metrics = full_metrics(model, x, y, tail_train)
    tail_eval_metrics = full_metrics(model, x, y, tail_eval)
    drift_fro, drift_rms = _tail_logit_drift(model, x[tail_eval], base_tail_logits)
    rows.append(
        {
            "seed": int(seed),
            "optimizer": optimizer_name,
            "step": int(step),
            "batch_loss_before": float(batch_loss_before),
            "batch_loss_after": float(batch_loss_after),
            "batch_loss_decrease": float(batch_loss_before - batch_loss_after),
            "batch_tail_examples": int(batch_tail_examples),
            "update_fro_norm": float(update_fro_norm),
            "update_op_norm": float(update_op_norm),
            "gradient_momentum_cosine": float(gradient_momentum_cosine),
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "head_loss": head["loss"],
            "head_accuracy": head["accuracy"],
            "head_margin": head["mean_margin"],
            "tail_train_loss": tail_train_metrics["loss"],
            "tail_train_accuracy": tail_train_metrics["accuracy"],
            "tail_train_margin": tail_train_metrics["mean_margin"],
            "tail_eval_loss": tail_eval_metrics["loss"],
            "tail_eval_accuracy": tail_eval_metrics["accuracy"],
            "tail_eval_margin": tail_eval_metrics["mean_margin"],
            "tail_eval_output_drift_fro": drift_fro,
            "tail_eval_output_drift_rms": drift_rms,
        }
    )


def run_long_tail_practical_training(
    config: LongTailPracticalTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []

    for seed in config.seeds:
        x, y, train_indices, head_train, tail_train, tail_eval = split_long_tail_digits(
            config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        generator = make_generator(seed + 31000, device)
        model = TinyMLP(
            input_dim=x.shape[1],
            hidden_dim=config.hidden_dim,
            output_dim=10,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        initial_params = snapshot(model.parameters())
        base_tail_logits = model.logits(x[tail_eval]).detach()

        for optimizer_name in ["adam", "ns_muon"]:
            restore(model.parameters(), initial_params)
            params = model.parameters()
            adam = torch.optim.Adam(params, lr=config.adam_lr) if optimizer_name == "adam" else None
            momentum: list[torch.Tensor] = []
            _record_training_row(
                rows=rows,
                model=model,
                x=x,
                y=y,
                train_indices=train_indices,
                head_train=head_train,
                tail_train=tail_train,
                tail_eval=tail_eval,
                base_tail_logits=base_tail_logits,
                seed=seed,
                optimizer_name=optimizer_name,
                step=0,
                batch_loss_before=math.nan,
                batch_loss_after=math.nan,
                batch_tail_examples=0,
                update_fro_norm=0.0,
                update_op_norm=0.0,
                gradient_momentum_cosine=math.nan,
            )
            for step in range(1, config.train_steps + 1):
                batch_generator = make_generator(seed + 32000 + step, device)
                batch_indices = batch(train_indices, config.train_batch_size, generator=batch_generator)
                batch_x = add_noise(x[batch_indices], config.train_noise_std, generator=batch_generator)
                batch_y = y[batch_indices]
                batch_tail_examples = int(_is_tail(batch_y, config.tail_classes).sum().cpu())

                if optimizer_name == "adam":
                    assert adam is not None
                    adam.zero_grad(set_to_none=True)
                    loss_before = F.cross_entropy(model.logits(batch_x), batch_y)
                    loss_before.backward()
                    before = snapshot(params)
                    adam.step()
                    with torch.no_grad():
                        loss_after = F.cross_entropy(model.logits(batch_x), batch_y)
                    update_tensors = [before_value - param.detach() for before_value, param in zip(before, params)]
                    update_fro_norm = math.sqrt(sum(float(torch.sum(update.square()).cpu()) for update in update_tensors))
                    update_op_norm = max(
                        float(torch.linalg.matrix_norm(update.reshape(update.shape[0], -1), ord=2).cpu())
                        for update in update_tensors
                    )
                    gradient_momentum_cosine = math.nan
                else:
                    loss_value, grads = collect_gradients(model, batch_x, batch_y)
                    momentum = momentum_update(momentum, grads, beta=config.momentum_beta)
                    directions = newton_schulz_directions(momentum, steps=config.newton_schulz_steps)
                    update_fro_norm = math.sqrt(sum(float(torch.sum((config.muon_lr * direction).square()).cpu()) for direction in directions))
                    update_op_norm = max(
                        float(torch.linalg.matrix_norm((config.muon_lr * direction).reshape(direction.shape[0], -1), ord=2).cpu())
                        for direction in directions
                    )
                    grad_norm = math.sqrt(sum(float(torch.sum(grad.square()).cpu()) for grad in grads))
                    momentum_norm = math.sqrt(sum(float(torch.sum(buffer.square()).cpu()) for buffer in momentum))
                    dot = sum(float(torch.sum(grad * buffer).cpu()) for grad, buffer in zip(grads, momentum))
                    gradient_momentum_cosine = dot / max(grad_norm * momentum_norm, 1e-300)
                    apply_practical_muon_step(params, directions, lr=config.muon_lr)
                    with torch.no_grad():
                        loss_before = torch.tensor(loss_value, device=device, dtype=dtype)
                        loss_after = F.cross_entropy(model.logits(batch_x), batch_y)

                _record_training_row(
                    rows=rows,
                    model=model,
                    x=x,
                    y=y,
                    train_indices=train_indices,
                    head_train=head_train,
                    tail_train=tail_train,
                    tail_eval=tail_eval,
                    base_tail_logits=base_tail_logits,
                    seed=seed,
                    optimizer_name=optimizer_name,
                    step=step,
                    batch_loss_before=float(loss_before.detach().cpu()),
                    batch_loss_after=float(loss_after.detach().cpu()),
                    batch_tail_examples=batch_tail_examples,
                    update_fro_norm=update_fro_norm,
                    update_op_norm=update_op_norm,
                    gradient_momentum_cosine=gradient_momentum_cosine,
                )

    step_metrics = pd.DataFrame(rows)
    summary = summarize_long_tail_practical_training(step_metrics, config)
    return step_metrics, summary


def summarize_long_tail_practical_training(
    step_metrics: pd.DataFrame,
    config: LongTailPracticalTrainingConfig,
) -> pd.DataFrame:
    final = step_metrics[step_metrics["step"].eq(config.train_steps)]
    paired_rows = []
    for seed in sorted(final["seed"].unique()):
        seed_final = final[final["seed"].eq(seed)].set_index("optimizer")
        adam = seed_final.loc["adam"]
        muon = seed_final.loc["ns_muon"]
        paired_rows.append(
            {
                "seed": int(seed),
                "final_train_loss_ratio_muon_over_adam": float(muon["train_loss"] / max(adam["train_loss"], 1e-300)),
                "final_head_loss_ratio_muon_over_adam": float(muon["head_loss"] / max(adam["head_loss"], 1e-300)),
                "final_tail_eval_loss_ratio_muon_over_adam": float(
                    muon["tail_eval_loss"] / max(adam["tail_eval_loss"], 1e-300)
                ),
                "final_tail_eval_accuracy_diff_muon_minus_adam": float(
                    muon["tail_eval_accuracy"] - adam["tail_eval_accuracy"]
                ),
                "final_tail_eval_margin_diff_muon_minus_adam": float(muon["tail_eval_margin"] - adam["tail_eval_margin"]),
                "final_tail_eval_drift_rms_ratio_muon_over_adam": float(
                    muon["tail_eval_output_drift_rms"] / max(adam["tail_eval_output_drift_rms"], 1e-300)
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    train_ratio, train_low, train_high = log_ratio_ci95(paired["final_train_loss_ratio_muon_over_adam"])
    head_ratio, head_low, head_high = log_ratio_ci95(paired["final_head_loss_ratio_muon_over_adam"])
    tail_loss_ratio, tail_loss_low, tail_loss_high = log_ratio_ci95(
        paired["final_tail_eval_loss_ratio_muon_over_adam"]
    )
    drift_ratio, drift_low, drift_high = log_ratio_ci95(paired["final_tail_eval_drift_rms_ratio_muon_over_adam"])
    acc_diff, acc_low, acc_high = ci95(paired["final_tail_eval_accuracy_diff_muon_minus_adam"])
    margin_diff, margin_low, margin_high = ci95(paired["final_tail_eval_margin_diff_muon_minus_adam"])
    return pd.DataFrame(
        [
            {
                "seeds": int(paired["seed"].nunique()),
                "train_steps": int(config.train_steps),
                "train_batch_size": int(config.train_batch_size),
                "adam_lr": float(config.adam_lr),
                "muon_lr": float(config.muon_lr),
                "geomean_final_train_loss_ratio_muon_over_adam": train_ratio,
                "final_train_loss_ratio_ci95_low": train_low,
                "final_train_loss_ratio_ci95_high": train_high,
                "geomean_final_head_loss_ratio_muon_over_adam": head_ratio,
                "final_head_loss_ratio_ci95_low": head_low,
                "final_head_loss_ratio_ci95_high": head_high,
                "geomean_final_tail_eval_loss_ratio_muon_over_adam": tail_loss_ratio,
                "final_tail_eval_loss_ratio_ci95_low": tail_loss_low,
                "final_tail_eval_loss_ratio_ci95_high": tail_loss_high,
                "mean_final_tail_eval_accuracy_diff_muon_minus_adam": acc_diff,
                "final_tail_eval_accuracy_diff_ci95_low": acc_low,
                "final_tail_eval_accuracy_diff_ci95_high": acc_high,
                "mean_final_tail_eval_margin_diff_muon_minus_adam": margin_diff,
                "final_tail_eval_margin_diff_ci95_low": margin_low,
                "final_tail_eval_margin_diff_ci95_high": margin_high,
                "geomean_final_tail_eval_drift_rms_ratio_muon_over_adam": drift_ratio,
                "final_tail_eval_drift_rms_ratio_ci95_low": drift_low,
                "final_tail_eval_drift_rms_ratio_ci95_high": drift_high,
            }
        ]
    )


def run_long_tail_practical_training_lr_sweep(
    config: LongTailPracticalTrainingLRSweepConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    for muon_lr in config.muon_lrs:
        run_config = LongTailPracticalTrainingConfig(
            seeds=config.base_config.seeds,
            head_classes=config.base_config.head_classes,
            tail_classes=config.base_config.tail_classes,
            head_train_per_class=config.base_config.head_train_per_class,
            tail_train_per_class=config.base_config.tail_train_per_class,
            tail_eval_per_class=config.base_config.tail_eval_per_class,
            hidden_dim=config.base_config.hidden_dim,
            train_steps=config.base_config.train_steps,
            train_batch_size=config.base_config.train_batch_size,
            adam_lr=config.base_config.adam_lr,
            muon_lr=float(muon_lr),
            momentum_beta=config.base_config.momentum_beta,
            newton_schulz_steps=config.base_config.newton_schulz_steps,
            train_noise_std=config.base_config.train_noise_std,
            dtype=config.base_config.dtype,
            device=config.base_config.device,
        )
        _step_metrics, summary = run_long_tail_practical_training(run_config)
        row = summary.iloc[0].to_dict()
        row["muon_lr"] = float(muon_lr)
        rows.append(row)
    return pd.DataFrame(rows)
