from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

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
    sample_indices,
    snapshot,
    restore,
    update_norms,
)
from .long_tail_one_step import summarize_long_tail_one_step


@dataclass(frozen=True)
class Cifar100LongTailOneStepConfig:
    seeds: tuple[int, ...] = tuple(range(5))
    head_classes: tuple[int, ...] = tuple(range(50))
    tail_classes: tuple[int, ...] = tuple(range(50, 100))
    head_train_per_class: int = 200
    tail_train_per_class: int = 20
    tail_eval_per_class: int = 20
    hidden_dim: int = 256
    warmup_steps: int = 600
    warmup_batch_size: int = 256
    head_batch_size: int = 256
    lr: float = 1e-3
    train_noise_std: float = 0.0
    target_head_gain_fraction: float = 0.01
    dtype: str = "float32"
    device: str = "auto"
    data_root: str = "data/torchvision"
    download: bool = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _normalization_tensors(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor((0.5071, 0.4867, 0.4408), device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.2675, 0.2565, 0.2761), device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def load_cifar100_tensors(
    *,
    root: str | Path,
    train: bool,
    device: torch.device,
    dtype: torch.dtype,
    download: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = CIFAR100(root=str(root), train=train, download=download)
    x = torch.tensor(dataset.data, device=device, dtype=dtype).permute(0, 3, 1, 2) / 255.0
    mean, std = _normalization_tensors(device, dtype)
    x = ((x - mean) / std).reshape(x.shape[0], -1)
    y = torch.tensor(dataset.targets, device=device, dtype=torch.long)
    return x, y


def split_long_tail_cifar100(
    config: Cifar100LongTailOneStepConfig,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x, train_y = load_cifar100_tensors(
        root=config.data_root,
        train=True,
        device=device,
        dtype=dtype,
        download=config.download,
    )
    test_x, test_y = load_cifar100_tensors(
        root=config.data_root,
        train=False,
        device=device,
        dtype=dtype,
        download=config.download,
    )
    generator = make_generator(seed + 21000, device)
    head_train = sample_indices(
        train_y,
        classes=config.head_classes,
        count_per_class=config.head_train_per_class,
        generator=generator,
    )
    tail_train = sample_indices(
        train_y,
        classes=config.tail_classes,
        count_per_class=config.tail_train_per_class,
        generator=generator,
    )
    tail_eval = sample_indices(
        test_y,
        classes=config.tail_classes,
        count_per_class=config.tail_eval_per_class,
        generator=generator,
    )
    train_indices = torch.cat([head_train, tail_train])
    return train_x, train_y, test_x, test_y, train_indices, head_train, tail_train, tail_eval


def add_noise(x: torch.Tensor, noise_std: float, *, generator: torch.Generator) -> torch.Tensor:
    if noise_std <= 0.0:
        return x
    scale = x.square().mean().sqrt().clamp_min(torch.finfo(x.dtype).eps)
    return x + float(noise_std) * scale * torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def train_cifar100_checkpoint(
    model: TinyMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: torch.Tensor,
    config: Cifar100LongTailOneStepConfig,
    *,
    seed: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for step in range(config.warmup_steps):
        generator = make_generator(seed + 22000 + step, x.device)
        batch_indices = batch(train_indices, config.warmup_batch_size, generator=generator)
        batch_x = add_noise(x[batch_indices], config.train_noise_std, generator=generator)
        loss = F.cross_entropy(model.logits(batch_x), y[batch_indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def run_cifar100_long_tail_one_step(
    config: Cifar100LongTailOneStepConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []
    layer_rows: list[dict] = []

    for seed in config.seeds:
        train_x, train_y, test_x, test_y, train_indices, head_train, tail_train, tail_eval = split_long_tail_cifar100(
            config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        generator = make_generator(seed + 23000, device)
        model = TinyMLP(
            input_dim=train_x.shape[1],
            hidden_dim=config.hidden_dim,
            output_dim=100,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        train_cifar100_checkpoint(model, train_x, train_y, train_indices, config, seed=seed)

        head_generator = make_generator(seed + 24000, device)
        head_batch = batch(head_train, config.head_batch_size, generator=head_generator)
        head_x = train_x[head_batch]
        head_y = train_y[head_batch]
        tail_x = test_x[tail_eval]
        tail_y = test_y[tail_eval]
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
        base_head = full_metrics(model, train_x, train_y, head_batch)
        base_tail = full_metrics(model, test_x, test_y, tail_eval)
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
            after_head = full_metrics(model, train_x, train_y, head_batch)
            after_tail = full_metrics(model, test_x, test_y, tail_eval)
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
                    "dataset": "CIFAR100",
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
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                }
            )
            for layer_record in layer_rank:
                layer_record = dict(layer_record)
                layer_record.update(
                    {
                        "seed": int(seed),
                        "geometry": geometry,
                        "dataset": "CIFAR100",
                        "tail_output_drift_fro": float(torch.linalg.norm(output_delta).cpu()),
                    }
                )
                layer_rows.append(layer_record)
            restore(params, before)

    step_metrics = pd.DataFrame(rows)
    layer_metrics = pd.DataFrame(layer_rows)
    pair_summary = summarize_long_tail_one_step(step_metrics)
    pair_summary.insert(0, "dataset", "CIFAR100")
    return step_metrics, pair_summary, layer_metrics
