from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18, resnet34

from .diagnostics import matrix_effective_rank, matrix_view, singular_values
from .long_tail_digits import make_generator, margins, sample_indices
from .long_tail_one_step import summarize_long_tail_one_step


@dataclass(frozen=True)
class Cifar100ResNetOneStepConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    dataset_name: str = "CIFAR100"
    model_arch: str = "resnet18"
    head_classes: tuple[int, ...] = tuple(range(50))
    tail_classes: tuple[int, ...] = tuple(range(50, 100))
    head_train_per_class: int = 300
    tail_train_per_class: int = 30
    tail_eval_per_class: int = 40
    warmup_steps: int = 1000
    warmup_batch_size: int = 256
    head_batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    target_head_gain_fraction: float = 0.005
    dtype: str = "float32"
    device: str = "auto"
    data_root: str = "data/torchvision"
    download: bool = True
    num_workers: int = 2


def dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return dtype


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalized_dataset_name(name: str) -> str:
    normalized = name.upper().replace("-", "")
    if normalized not in {"CIFAR100", "CIFAR10"}:
        raise ValueError(f"unknown CIFAR dataset: {name}")
    return normalized


def dataset_num_classes(name: str) -> int:
    return 100 if normalized_dataset_name(name) == "CIFAR100" else 10


def dataset_display_name(name: str) -> str:
    return "CIFAR-100-LT" if normalized_dataset_name(name) == "CIFAR100" else "CIFAR-10-LT"


def model_display_name(name: str) -> str:
    normalized = name.lower()
    if normalized == "resnet18":
        return "ResNet18"
    if normalized == "resnet34":
        return "ResNet34"
    raise ValueError(f"unknown CIFAR ResNet architecture: {name}")


def _normalization_tensors(
    device: torch.device,
    dtype: torch.dtype,
    *,
    dataset_name: str = "CIFAR100",
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalized_dataset_name(dataset_name) == "CIFAR100":
        mean_values = (0.5071, 0.4867, 0.4408)
        std_values = (0.2675, 0.2565, 0.2761)
    else:
        mean_values = (0.4914, 0.4822, 0.4465)
        std_values = (0.2470, 0.2435, 0.2616)
    mean = torch.tensor(mean_values, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def load_cifar_images(
    *,
    root: str | Path,
    train: bool,
    device: torch.device,
    dtype: torch.dtype,
    download: bool,
    dataset_name: str = "CIFAR100",
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_class = CIFAR100 if normalized_dataset_name(dataset_name) == "CIFAR100" else CIFAR10
    dataset = dataset_class(root=str(root), train=train, download=download)
    x = torch.tensor(dataset.data, device=device, dtype=dtype).permute(0, 3, 1, 2) / 255.0
    mean, std = _normalization_tensors(device, dtype, dataset_name=dataset_name)
    x = (x - mean) / std
    y = torch.tensor(dataset.targets, device=device, dtype=torch.long)
    return x, y


def load_cifar100_images(
    *,
    root: str | Path,
    train: bool,
    device: torch.device,
    dtype: torch.dtype,
    download: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return load_cifar_images(
        root=root,
        train=train,
        device=device,
        dtype=dtype,
        download=download,
        dataset_name="CIFAR100",
    )


def split_long_tail_cifar100(
    config: Cifar100ResNetOneStepConfig,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_classes = dataset_num_classes(config.dataset_name)
    requested_classes = set(config.head_classes) | set(config.tail_classes)
    invalid_classes = sorted(label for label in requested_classes if label < 0 or label >= num_classes)
    if invalid_classes:
        raise ValueError(f"{config.dataset_name} class split contains invalid labels: {invalid_classes}")
    train_x, train_y = load_cifar_images(
        root=config.data_root,
        train=True,
        device=device,
        dtype=dtype,
        download=config.download,
        dataset_name=config.dataset_name,
    )
    test_x, test_y = load_cifar_images(
        root=config.data_root,
        train=False,
        device=device,
        dtype=dtype,
        download=config.download,
        dataset_name=config.dataset_name,
    )
    generator = make_generator(seed + 31000, device)
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


def build_cifar_resnet18(*, device: torch.device, dtype: torch.dtype) -> nn.Module:
    model = resnet18(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device=device, dtype=dtype)


def build_cifar_resnet_model(config: Cifar100ResNetOneStepConfig, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
    arch = config.model_arch.lower()
    if arch == "resnet18":
        model = resnet18(weights=None, num_classes=dataset_num_classes(config.dataset_name))
    elif arch == "resnet34":
        model = resnet34(weights=None, num_classes=dataset_num_classes(config.dataset_name))
    else:
        raise ValueError(f"unknown CIFAR ResNet architecture: {config.model_arch}")
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device=device, dtype=dtype)


def batch(indices: torch.Tensor, batch_size: int, *, generator: torch.Generator) -> torch.Tensor:
    if batch_size >= indices.numel():
        return indices
    perm = torch.randperm(indices.numel(), generator=generator, device=indices.device)[:batch_size]
    return indices[perm]


def matrix_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad and param.ndim >= 2]


def snapshot(named_params: list[tuple[str, nn.Parameter]]) -> list[torch.Tensor]:
    return [param.detach().clone() for _name, param in named_params]


def restore(named_params: list[tuple[str, nn.Parameter]], before: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for (_name, param), value in zip(named_params, before):
            param.copy_(value)


def global_matrix_grad_fro(named_params: list[tuple[str, nn.Parameter]]) -> float:
    total = 0.0
    for _name, param in named_params:
        if param.grad is not None:
            total += float(torch.linalg.norm(param.grad.detach()).cpu()) ** 2
    return total**0.5


def direction_list(named_params: list[tuple[str, nn.Parameter]], geometry: str) -> list[torch.Tensor]:
    if geometry == "frobenius":
        grad_fro = global_matrix_grad_fro(named_params)
        return [param.grad.detach() / max(grad_fro, 1e-300) for _name, param in named_params]
    if geometry == "spectral":
        directions = []
        for _name, param in named_params:
            grad_matrix = matrix_view(param.grad.detach())
            u, _s, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
            directions.append((u @ vh).reshape_as(param))
        return directions
    raise ValueError(f"unknown geometry: {geometry}")


def alignment(named_params: list[tuple[str, nn.Parameter]], directions: list[torch.Tensor]) -> float:
    total = 0.0
    for (_name, param), direction in zip(named_params, directions):
        total += float(torch.sum(param.grad.detach() * direction).cpu())
    return total


def update_norms(directions: list[torch.Tensor], step_size: float) -> tuple[float, float]:
    fro_sq = 0.0
    op_values = []
    for direction in directions:
        update = float(step_size) * direction
        fro_sq += float(torch.linalg.norm(update).cpu()) ** 2
        op_values.append(float(torch.linalg.matrix_norm(matrix_view(update), ord=2).cpu()))
    return fro_sq**0.5, max(op_values) if op_values else math.nan


def apply_direction(
    named_params: list[tuple[str, nn.Parameter]],
    before: list[torch.Tensor],
    directions: list[torch.Tensor],
    step_size: float,
) -> None:
    with torch.no_grad():
        for (_name, param), value, direction in zip(named_params, before, directions):
            param.copy_(value - float(step_size) * direction)


def full_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x[indices])
        loss = F.cross_entropy(logits, y[indices])
        pred = logits.argmax(dim=1)
        return {
            "loss": float(loss.detach().cpu()),
            "accuracy": float((pred == y[indices]).to(torch.float64).mean().cpu()),
            "mean_margin": float(margins(logits.detach(), y[indices]).mean().cpu()),
        }


def update_rank_summary(named_params: list[tuple[str, nn.Parameter]], directions: list[torch.Tensor]) -> dict[str, float]:
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


def train_checkpoint(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: torch.Tensor,
    config: Cifar100ResNetOneStepConfig,
    *,
    seed: int,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.train()
    for step in range(config.warmup_steps):
        generator = make_generator(seed + 32000 + step, x.device)
        batch_indices = batch(train_indices, config.warmup_batch_size, generator=generator)
        loss = F.cross_entropy(model(x[batch_indices]), y[batch_indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def run_cifar100_resnet_one_step(
    config: Cifar100ResNetOneStepConfig,
    *,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(0)
    torch.manual_seed(0)
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict] = []
    layer_rows: list[dict] = []

    for seed in config.seeds:
        if progress:
            print(f"[resnet] seed {seed}: loading data", flush=True)
        train_x, train_y, test_x, test_y, train_indices, head_train, tail_train, tail_eval = split_long_tail_cifar100(
            config,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        torch.manual_seed(seed + 33000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 33000)
        model = build_cifar_resnet_model(config, device=device, dtype=dtype)
        if progress:
            print(f"[resnet] seed {seed}: training {config.warmup_steps} warmup steps", flush=True)
        train_checkpoint(model, train_x, train_y, train_indices, config, seed=seed)

        model.eval()
        named_params = matrix_named_parameters(model)
        head_generator = make_generator(seed + 34000, device)
        head_batch = batch(head_train, config.head_batch_size, generator=head_generator)
        head_x = train_x[head_batch]
        head_y = train_y[head_batch]
        tail_x = test_x[tail_eval]
        tail_y = test_y[tail_eval]

        model.zero_grad(set_to_none=True)
        head_loss = F.cross_entropy(model(head_x), head_y)
        head_loss.backward()
        before = snapshot(named_params)
        with torch.no_grad():
            base_tail_logits = model(tail_x).detach()
            base_tail_pred = torch.argmax(base_tail_logits, dim=1)
        base_tail_margins = margins(base_tail_logits, tail_y)
        positive_margin_mask = base_tail_margins > 0
        positive_margin_count = int(positive_margin_mask.sum().cpu())
        positive_margin_fraction = float(positive_margin_mask.to(torch.float64).mean().cpu())
        base_head = full_metrics(model, train_x, train_y, head_batch)
        base_tail = full_metrics(model, test_x, test_y, tail_eval)
        target_gain = config.target_head_gain_fraction * float(head_loss.detach().cpu())

        for geometry in ["frobenius", "spectral"]:
            directions = direction_list(named_params, geometry)
            alignment_value = alignment(named_params, directions)
            step_size = target_gain / max(alignment_value, 1e-300)
            update_fro, update_op = update_norms(directions, step_size)
            rank_summary = update_rank_summary(named_params, directions)
            apply_direction(named_params, before, directions, step_size)
            with torch.no_grad():
                after_tail_logits = model(tail_x).detach()
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
            rows.append(
                {
                    "seed": int(seed),
                    "geometry": geometry,
                    "dataset": normalized_dataset_name(config.dataset_name),
                    "model": f"{config.model_arch.lower()}_cifar_stem",
                    "updated_parameter_subset": "conv_and_linear_weights_only",
                    "head_classes": ",".join(str(x) for x in config.head_classes),
                    "tail_classes": ",".join(str(x) for x in config.tail_classes),
                    "head_train_examples": int(head_train.numel()),
                    "tail_train_examples": int(tail_train.numel()),
                    "tail_eval_examples": int(tail_eval.numel()),
                    "warmup_steps": int(config.warmup_steps),
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
                    "tail_output_jvp_fro": float("nan"),
                    "tail_output_linearization_residual_fro": float("nan"),
                    "tail_output_linearization_relative_error": float("nan"),
                    "update_fro_norm": update_fro,
                    "update_op_norm": update_op,
                    "step_size": float(step_size),
                    "alignment": alignment_value,
                    "nrG": rank_summary["mean_matrix_gradient_nuclear_rank"],
                    "stA_tail": float("nan"),
                    "condition_score_tail": float("nan"),
                    "mean_matrix_update_nuclear_rank": rank_summary["mean_matrix_update_nuclear_rank"],
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                }
            )
            for (name, param), direction in zip(named_params, directions):
                grad_rank = matrix_effective_rank(singular_values(param.grad.detach()))
                update_rank = matrix_effective_rank(singular_values(direction.detach()))
                layer_rows.append(
                    {
                        "seed": int(seed),
                        "geometry": geometry,
                        "parameter": name,
                        "shape": "x".join(str(dim) for dim in param.shape),
                        "gradient_nuclear_rank": grad_rank,
                        "update_nuclear_rank": update_rank,
                    }
                )
            restore(named_params, before)

        if progress:
            print(f"[resnet] seed {seed}: done", flush=True)

    step_metrics = pd.DataFrame(rows)
    layer_metrics = pd.DataFrame(layer_rows)
    pair_summary = summarize_long_tail_one_step(step_metrics)
    pair_summary.insert(0, "model", "resnet18_cifar_stem")
    pair_summary.insert(0, "dataset", "CIFAR100")
    return step_metrics, pair_summary, layer_metrics
