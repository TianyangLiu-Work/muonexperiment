from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    batch,
    build_cifar_resnet18,
    dtype_from_name,
    load_cifar100_images,
    resolve_device,
)
from e11_condition_geometry.long_tail_digits import make_generator, margins, sample_indices
from e11_condition_geometry.reporting import fmt, markdown_table
from e11_condition_geometry.statistics import ci95


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_lt_recipe_benchmark")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_lt_recipe_benchmark")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_recipe_benchmark.md")
GROUP_ORDER = ("many", "medium", "few", "all")
DEFAULT_RECIPES = ("adamw_aug_ce", "adamw_aug_cb_loss", "sgd_aug_ce")
RECIPE_COLORS = {
    "adamw_aug_ce": "#0072B2",
    "adamw_aug_cb_loss": "#009E73",
    "sgd_aug_ce": "#D55E00",
}


@dataclass(frozen=True)
class Recipe:
    name: str
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float = 0.0
    nesterov: bool = False
    class_balanced_loss: bool = False
    augmentation: bool = True
    cosine_lr: bool = True


@dataclass(frozen=True)
class RecipeBenchmarkConfig:
    seeds: tuple[int, ...] = tuple(range(5))
    recipe_names: tuple[str, ...] = DEFAULT_RECIPES
    baseline_recipe: str = "adamw_aug_ce"
    num_classes: int = 100
    max_train_count: int = 500
    imbalance_factor: float = 100.0
    many_threshold: int = 100
    medium_threshold: int = 20
    train_steps: int = 5000
    train_batch_size: int = 256
    eval_interval: int = 1000
    crop_padding: int = 4
    hflip_probability: float = 0.5
    class_balance_beta: float = 0.9999
    dtype: str = "float32"
    device: str = "auto"
    data_root: str = "data/torchvision"
    download: bool = True


def recipe_from_name(name: str) -> Recipe:
    recipes = {
        "adamw_aug_ce": Recipe(
            name="adamw_aug_ce",
            optimizer="adamw",
            lr=3e-4,
            weight_decay=1e-4,
            class_balanced_loss=False,
            augmentation=True,
        ),
        "adamw_aug_cb_loss": Recipe(
            name="adamw_aug_cb_loss",
            optimizer="adamw",
            lr=3e-4,
            weight_decay=1e-4,
            class_balanced_loss=True,
            augmentation=True,
        ),
        "sgd_aug_ce": Recipe(
            name="sgd_aug_ce",
            optimizer="sgd",
            lr=0.1,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
            class_balanced_loss=False,
            augmentation=True,
        ),
    }
    if name not in recipes:
        raise ValueError(f"unknown recipe {name!r}; available recipes: {sorted(recipes)}")
    return recipes[name]


def long_tail_counts(config: RecipeBenchmarkConfig) -> dict[int, int]:
    if config.num_classes <= 0 or config.num_classes > 100:
        raise ValueError("num_classes must be in [1, 100]")
    if config.num_classes == 1:
        return {0: int(config.max_train_count)}
    counts = {}
    for rank, class_id in enumerate(range(config.num_classes)):
        exponent = -rank / max(config.num_classes - 1, 1)
        count = int(round(config.max_train_count * (config.imbalance_factor**exponent)))
        counts[class_id] = max(1, min(500, count))
    return counts


def frequency_group(train_count: int, config: RecipeBenchmarkConfig) -> str:
    if train_count >= config.many_threshold:
        return "many"
    if train_count >= config.medium_threshold:
        return "medium"
    return "few"


def make_train_indices(
    labels: torch.Tensor,
    counts: dict[int, int],
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = make_generator(seed + 61000, device)
    sampled = [
        sample_indices(labels, classes=(class_id,), count_per_class=count, generator=generator)
        for class_id, count in counts.items()
    ]
    return torch.cat(sampled)


def class_balanced_weights(
    counts: dict[int, int],
    *,
    beta: float,
    num_outputs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.zeros(num_outputs, device=device, dtype=dtype)
    values = []
    for class_id, count in counts.items():
        effective = (1.0 - float(beta) ** int(count)) / max(1.0 - float(beta), 1e-12)
        values.append((int(class_id), 1.0 / max(effective, 1e-12)))
    normalizer = sum(value for _class_id, value in values) / max(len(values), 1)
    for class_id, value in values:
        weights[class_id] = float(value / max(normalizer, 1e-12))
    return weights


def augment_batch(
    x: torch.Tensor,
    *,
    generator: torch.Generator,
    padding: int,
    hflip_probability: float,
) -> torch.Tensor:
    if padding > 0:
        padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        batch_size, _channels, height, width = x.shape
        max_offset = 2 * int(padding) + 1
        top = torch.randint(max_offset, (batch_size,), generator=generator, device=x.device)
        left = torch.randint(max_offset, (batch_size,), generator=generator, device=x.device)
        rows = top[:, None] + torch.arange(height, device=x.device)[None, :]
        cols = left[:, None] + torch.arange(width, device=x.device)[None, :]
        batch_index = torch.arange(batch_size, device=x.device)[:, None, None]
        x = padded[batch_index, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2).contiguous()
    if hflip_probability > 0.0:
        flips = torch.rand(x.shape[0], generator=generator, device=x.device) < float(hflip_probability)
        if bool(flips.any().detach().cpu()):
            x = x.clone()
            x[flips] = torch.flip(x[flips], dims=(3,))
    return x


def make_optimizer(model: torch.nn.Module, recipe: Recipe) -> torch.optim.Optimizer:
    if recipe.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    if recipe.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=recipe.lr,
            momentum=recipe.momentum,
            weight_decay=recipe.weight_decay,
            nesterov=recipe.nesterov,
        )
    raise ValueError(f"unknown optimizer: {recipe.optimizer}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def scheduled_lr(recipe: Recipe, *, step: int, total_steps: int) -> float:
    if not recipe.cosine_lr:
        return float(recipe.lr)
    progress = min(max((int(step) - 1) / max(int(total_steps), 1), 0.0), 1.0)
    return float(recipe.lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _class_test_indices(labels: torch.Tensor, class_id: int) -> torch.Tensor:
    return torch.nonzero(labels == int(class_id), as_tuple=False).flatten()


def evaluate_class_metrics(
    model: torch.nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    counts: dict[int, int],
    config: RecipeBenchmarkConfig,
    *,
    seed: int,
    recipe: Recipe,
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for class_id, train_count in counts.items():
            indices = _class_test_indices(test_y, class_id)
            logits = model(test_x[indices])
            y = test_y[indices]
            loss = F.cross_entropy(logits, y)
            prediction = logits.argmax(dim=1)
            margin_values = margins(logits, y)
            rows.append(
                {
                    "seed": int(seed),
                    "recipe": recipe.name,
                    "class_id": int(class_id),
                    "train_count": int(train_count),
                    "frequency_group": frequency_group(int(train_count), config),
                    "test_examples": int(indices.numel()),
                    "loss": float(loss.detach().cpu()),
                    "accuracy": float((prediction == y).to(torch.float64).mean().cpu()),
                    "mean_margin": float(margin_values.mean().detach().cpu()),
                }
            )
    return pd.DataFrame(rows)


def summarize_groups(class_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, recipe_name), seed_metrics in class_metrics.groupby(["seed", "recipe"], observed=True, sort=True):
        for group_name in GROUP_ORDER:
            if group_name == "all":
                group = seed_metrics
            else:
                group = seed_metrics[seed_metrics["frequency_group"].eq(group_name)]
            if group.empty:
                continue
            total_examples = int(group["test_examples"].sum())
            weighted_loss = float((group["loss"] * group["test_examples"]).sum() / max(total_examples, 1))
            weighted_accuracy = float((group["accuracy"] * group["test_examples"]).sum() / max(total_examples, 1))
            rows.append(
                {
                    "seed": int(seed),
                    "recipe": str(recipe_name),
                    "frequency_group": group_name,
                    "classes": int(group["class_id"].nunique()),
                    "test_examples": total_examples,
                    "min_train_count": int(group["train_count"].min()),
                    "max_train_count": int(group["train_count"].max()),
                    "mean_train_count": float(group["train_count"].mean()),
                    "loss": weighted_loss,
                    "accuracy": weighted_accuracy,
                    "balanced_accuracy": float(group["accuracy"].mean()),
                    "mean_margin": float(group["mean_margin"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_over_seeds(group_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (recipe_name, group_name), group in group_metrics.groupby(["recipe", "frequency_group"], observed=True, sort=False):
        loss, loss_low, loss_high = ci95(group["loss"])
        accuracy, accuracy_low, accuracy_high = ci95(group["accuracy"])
        balanced, balanced_low, balanced_high = ci95(group["balanced_accuracy"])
        margin, margin_low, margin_high = ci95(group["mean_margin"])
        rows.append(
            {
                "recipe": str(recipe_name),
                "frequency_group": str(group_name),
                "seeds": int(group["seed"].nunique()),
                "classes": int(group["classes"].iloc[0]),
                "min_train_count": int(group["min_train_count"].iloc[0]),
                "max_train_count": int(group["max_train_count"].iloc[0]),
                "mean_train_count": float(group["mean_train_count"].mean()),
                "mean_loss": loss,
                "loss_ci95_low": loss_low,
                "loss_ci95_high": loss_high,
                "mean_accuracy": accuracy,
                "accuracy_ci95_low": accuracy_low,
                "accuracy_ci95_high": accuracy_high,
                "mean_balanced_accuracy": balanced,
                "balanced_accuracy_ci95_low": balanced_low,
                "balanced_accuracy_ci95_high": balanced_high,
                "mean_margin": margin,
                "margin_ci95_low": margin_low,
                "margin_ci95_high": margin_high,
            }
        )
    order = {name: index for index, name in enumerate(GROUP_ORDER)}
    return (
        pd.DataFrame(rows)
        .sort_values(["recipe", "frequency_group"], key=lambda s: s.map(order) if s.name == "frequency_group" else s)
        .reset_index(drop=True)
    )


def summarize_recipe_differences(group_metrics: pd.DataFrame, *, baseline_recipe: str) -> pd.DataFrame:
    baseline = group_metrics[group_metrics["recipe"].eq(baseline_recipe)].set_index(["seed", "frequency_group"])
    rows = []
    for (recipe_name, group_name), group in group_metrics.groupby(["recipe", "frequency_group"], observed=True, sort=False):
        if recipe_name == baseline_recipe:
            continue
        diffs = []
        for row in group.itertuples(index=False):
            key = (int(row.seed), str(row.frequency_group))
            if key not in baseline.index:
                continue
            base = baseline.loc[key]
            diffs.append(
                {
                    "recipe": str(recipe_name),
                    "baseline_recipe": baseline_recipe,
                    "frequency_group": str(group_name),
                    "seed": int(row.seed),
                    "balanced_accuracy_diff": float(row.balanced_accuracy - base["balanced_accuracy"]),
                    "accuracy_diff": float(row.accuracy - base["accuracy"]),
                    "loss_diff": float(row.loss - base["loss"]),
                    "margin_diff": float(row.mean_margin - base["mean_margin"]),
                }
            )
        diff_frame = pd.DataFrame(diffs)
        if diff_frame.empty:
            continue
        balanced, balanced_low, balanced_high = ci95(diff_frame["balanced_accuracy_diff"])
        accuracy, accuracy_low, accuracy_high = ci95(diff_frame["accuracy_diff"])
        loss, loss_low, loss_high = ci95(diff_frame["loss_diff"])
        margin, margin_low, margin_high = ci95(diff_frame["margin_diff"])
        rows.append(
            {
                "recipe": str(recipe_name),
                "baseline_recipe": baseline_recipe,
                "frequency_group": str(group_name),
                "seeds": int(diff_frame["seed"].nunique()),
                "mean_balanced_accuracy_diff": balanced,
                "balanced_accuracy_diff_ci95_low": balanced_low,
                "balanced_accuracy_diff_ci95_high": balanced_high,
                "mean_accuracy_diff": accuracy,
                "accuracy_diff_ci95_low": accuracy_low,
                "accuracy_diff_ci95_high": accuracy_high,
                "mean_loss_diff": loss,
                "loss_diff_ci95_low": loss_low,
                "loss_diff_ci95_high": loss_high,
                "mean_margin_diff": margin,
                "margin_diff_ci95_low": margin_low,
                "margin_diff_ci95_high": margin_high,
            }
        )
    order = {name: index for index, name in enumerate(GROUP_ORDER)}
    return (
        pd.DataFrame(rows)
        .sort_values(["recipe", "frequency_group"], key=lambda s: s.map(order) if s.name == "frequency_group" else s)
        .reset_index(drop=True)
    )


def run_recipe_benchmark(
    config: RecipeBenchmarkConfig,
    *,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(0)
    torch.manual_seed(0)
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    counts = long_tail_counts(config)
    recipes = [recipe_from_name(name) for name in config.recipe_names]
    trace_rows: list[dict] = []
    class_frames: list[pd.DataFrame] = []

    train_x, train_y = load_cifar100_images(
        root=config.data_root,
        train=True,
        device=device,
        dtype=dtype,
        download=config.download,
    )
    test_x, test_y = load_cifar100_images(
        root=config.data_root,
        train=False,
        device=device,
        dtype=dtype,
        download=config.download,
    )
    loss_weight = class_balanced_weights(
        counts,
        beta=config.class_balance_beta,
        num_outputs=100,
        device=device,
        dtype=dtype,
    )

    for seed in config.seeds:
        train_indices = make_train_indices(train_y, counts, seed=seed, device=device)
        for recipe in recipes:
            if progress:
                print(f"[resnet-lt-recipe] seed={seed} recipe={recipe.name}: training", flush=True)
            torch.manual_seed(seed + 62000)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 62000)
            model = build_cifar_resnet18(device=device, dtype=dtype)
            optimizer = make_optimizer(model, recipe)
            model.train()

            for step in range(1, config.train_steps + 1):
                generator = make_generator(seed + 63000 + step, device)
                batch_indices = batch(train_indices, config.train_batch_size, generator=generator)
                batch_x = train_x[batch_indices]
                if recipe.augmentation:
                    batch_x = augment_batch(
                        batch_x,
                        generator=generator,
                        padding=config.crop_padding,
                        hflip_probability=config.hflip_probability,
                    )
                current_lr = scheduled_lr(recipe, step=step, total_steps=config.train_steps)
                set_lr(optimizer, current_lr)
                logits = model(batch_x)
                batch_y = train_y[batch_indices]
                loss = F.cross_entropy(logits, batch_y, weight=loss_weight if recipe.class_balanced_loss else None)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step == 1 or step == config.train_steps or step % max(config.eval_interval, 1) == 0:
                    with torch.no_grad():
                        prediction = logits.argmax(dim=1)
                        batch_accuracy = float((prediction == batch_y).to(torch.float64).mean().cpu())
                    trace_rows.append(
                        {
                            "seed": int(seed),
                            "recipe": recipe.name,
                            "step": int(step),
                            "lr": float(current_lr),
                            "batch_loss": float(loss.detach().cpu()),
                            "batch_accuracy": batch_accuracy,
                            "train_examples": int(train_indices.numel()),
                            "device": str(device),
                            "dtype": str(dtype).replace("torch.", ""),
                        }
                    )
                    if progress:
                        print(
                            f"[resnet-lt-recipe] seed={seed} recipe={recipe.name} "
                            f"step={step}/{config.train_steps} loss={float(loss.detach().cpu()):.4g} "
                            f"lr={current_lr:.3g}",
                            flush=True,
                        )

            if progress:
                print(f"[resnet-lt-recipe] seed={seed} recipe={recipe.name}: evaluating", flush=True)
            class_frames.append(evaluate_class_metrics(model, test_x, test_y, counts, config, seed=seed, recipe=recipe))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    class_metrics = pd.concat(class_frames, ignore_index=True)
    group_metrics = summarize_groups(class_metrics)
    summary = summarize_over_seeds(group_metrics)
    pair_summary = summarize_recipe_differences(group_metrics, baseline_recipe=config.baseline_recipe)
    trace = pd.DataFrame(trace_rows)
    return trace, class_metrics, group_metrics, summary, pair_summary


def write_figure(summary: pd.DataFrame, pair_summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    groups = ["many", "medium", "few"]
    recipes = list(summary["recipe"].drop_duplicates())
    width = 0.8 / max(len(recipes), 1)
    x = torch.arange(len(groups), dtype=torch.float64).numpy()
    for idx, recipe_name in enumerate(recipes):
        sub = summary[summary["recipe"].eq(recipe_name)].set_index("frequency_group")
        y = [float(sub.loc[group, "mean_balanced_accuracy"]) for group in groups]
        low = [float(sub.loc[group, "balanced_accuracy_ci95_low"]) for group in groups]
        high = [float(sub.loc[group, "balanced_accuracy_ci95_high"]) for group in groups]
        offset = (idx - (len(recipes) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            y,
            width=width,
            color=RECIPE_COLORS.get(recipe_name, "#555555"),
            alpha=0.9,
            label=recipe_name,
        )
        axes[0].errorbar(x + offset, y, yerr=[torch.tensor(y).numpy() - torch.tensor(low).numpy(), torch.tensor(high).numpy() - torch.tensor(y).numpy()], fmt="none", color="black", capsize=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title("Final many/medium/few accuracy")
    axes[0].legend(frameon=False, fontsize=8)

    diff = pair_summary[pair_summary["frequency_group"].isin(groups)].copy()
    if not diff.empty:
        recipes_diff = list(diff["recipe"].drop_duplicates())
        width = 0.8 / max(len(recipes_diff), 1)
        for idx, recipe_name in enumerate(recipes_diff):
            sub = diff[diff["recipe"].eq(recipe_name)].set_index("frequency_group")
            y = [float(sub.loc[group, "mean_balanced_accuracy_diff"]) for group in groups]
            low = [float(sub.loc[group, "balanced_accuracy_diff_ci95_low"]) for group in groups]
            high = [float(sub.loc[group, "balanced_accuracy_diff_ci95_high"]) for group in groups]
            offset = (idx - (len(recipes_diff) - 1) / 2) * width
            axes[1].bar(
                x + offset,
                y,
                width=width,
                color=RECIPE_COLORS.get(recipe_name, "#555555"),
                alpha=0.9,
                label=recipe_name,
            )
            axes[1].errorbar(x + offset, y, yerr=[torch.tensor(y).numpy() - torch.tensor(low).numpy(), torch.tensor(high).numpy() - torch.tensor(y).numpy()], fmt="none", color="black", capsize=2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylabel("balanced accuracy diff")
    axes[1].set_title("Paired diff vs. adamw_aug_ce")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("CIFAR-100-LT ResNet18 recipe benchmark pilot")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_lt_recipe_benchmark.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _group_line(summary: pd.DataFrame, recipe_name: str, group: str) -> str:
    row = summary.set_index(["recipe", "frequency_group"]).loc[(recipe_name, group)]
    return (
        f"{fmt(row['mean_balanced_accuracy'])} "
        f"[{fmt(row['balanced_accuracy_ci95_low'])}, {fmt(row['balanced_accuracy_ci95_high'])}]"
    )


def write_discussion(
    config: RecipeBenchmarkConfig,
    summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
) -> None:
    lines = [
        "# E11 CIFAR-100-LT ResNet18 Recipe Benchmark Pilot",
        "",
        "This run upgrades the standard reporting surface from an AdamW-only no-augmentation",
        "baseline to a small recipe benchmark with data augmentation, a class-balanced",
        "loss baseline, and SGD-momentum. It is a pilot benchmark, not a final tuned",
        "leaderboard or a Muon comparison.",
        "",
        f"- Seeds: {config.seeds}",
        f"- Recipes: {config.recipe_names}",
        f"- Baseline recipe for paired differences: {config.baseline_recipe}",
        f"- Number of classes: {config.num_classes}",
        f"- Max train examples per class: {config.max_train_count}",
        f"- Imbalance factor: {config.imbalance_factor}",
        f"- Train steps: {config.train_steps}",
        f"- Batch size: {config.train_batch_size}",
        f"- Augmentation: reflect-padded random crop with padding {config.crop_padding} and horizontal flip probability {config.hflip_probability}",
        f"- Device/dtype request: {config.device}/{config.dtype}",
        "",
        f"![CIFAR-100-LT ResNet18 recipe benchmark](../{figure_path.as_posix()})",
        "",
        "## Summary",
        "",
        markdown_table(
            summary,
            [
                "recipe",
                "frequency_group",
                "seeds",
                "classes",
                "mean_balanced_accuracy",
                "balanced_accuracy_ci95_low",
                "balanced_accuracy_ci95_high",
                "mean_loss",
                "mean_margin",
            ],
        ),
        "",
        "## Paired Differences vs. AdamW Augmented CE",
        "",
        markdown_table(
            pair_summary,
            [
                "recipe",
                "frequency_group",
                "seeds",
                "mean_balanced_accuracy_diff",
                "balanced_accuracy_diff_ci95_low",
                "balanced_accuracy_diff_ci95_high",
                "mean_loss_diff",
                "mean_margin_diff",
            ],
        ),
        "",
        "## Readout",
        "",
    ]
    for recipe_name in config.recipe_names:
        if (summary["recipe"].eq(recipe_name)).any():
            lines.append(
                f"- `{recipe_name}` many/medium/few balanced accuracy: "
                f"{_group_line(summary, recipe_name, 'many')} / "
                f"{_group_line(summary, recipe_name, 'medium')} / "
                f"{_group_line(summary, recipe_name, 'few')}."
            )
    lines.extend(
        [
            "",
            "Interpretation: this pilot closes part of the standard-protocol gap by",
            "adding augmentation and common long-tail baselines. It should be used as",
            "benchmark-readout context only; a top-tier optimizer claim still needs a",
            "wider hyperparameter grid, longer training schedules, larger datasets, and",
            "a practical Muon/AdamW final-performance comparison.",
            "",
            "Artifacts:",
            f"- [train_trace.csv](../{(output_dir / 'train_trace.csv').as_posix()})",
            f"- [class_metrics.csv](../{(output_dir / 'class_metrics.csv').as_posix()})",
            f"- [group_metrics.csv](../{(output_dir / 'group_metrics.csv').as_posix()})",
            f"- [summary.csv](../{(output_dir / 'summary.csv').as_posix()})",
            f"- [pair_summary.csv](../{(output_dir / 'pair_summary.csv').as_posix()})",
            f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
        ]
    )
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--recipes", nargs="+", default=list(DEFAULT_RECIPES))
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--max-train-count", type=int, default=500)
    parser.add_argument("--imbalance-factor", type=float, default=100.0)
    parser.add_argument("--many-threshold", type=int, default=100)
    parser.add_argument("--medium-threshold", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--discussion-path", type=Path, default=DEFAULT_DISCUSSION_PATH)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RecipeBenchmarkConfig:
    if args.smoke:
        return RecipeBenchmarkConfig(
            seeds=(0,),
            recipe_names=tuple(args.recipes),
            num_classes=10,
            max_train_count=20,
            imbalance_factor=10.0,
            many_threshold=12,
            medium_threshold=5,
            train_steps=2,
            train_batch_size=16,
            eval_interval=1,
            device=args.device or "cpu",
            download=False if args.download is None else bool(args.download),
        )
    return RecipeBenchmarkConfig(
        seeds=tuple(range(args.seeds)),
        recipe_names=tuple(args.recipes),
        num_classes=args.num_classes,
        max_train_count=args.max_train_count,
        imbalance_factor=args.imbalance_factor,
        many_threshold=args.many_threshold,
        medium_threshold=args.medium_threshold,
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        eval_interval=args.eval_interval,
        device=args.device or "auto",
        download=True if args.download is None else bool(args.download),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace, class_metrics, group_metrics, summary, pair_summary = run_recipe_benchmark(
        config,
        progress=args.progress,
    )
    trace.to_csv(args.output_dir / "train_trace.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    group_metrics.to_csv(args.output_dir / "group_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pair_summary.to_csv(args.output_dir / "pair_summary.csv", index=False)
    config_payload = asdict(config)
    config_payload["recipes"] = [asdict(recipe_from_name(name)) for name in config.recipe_names]
    (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    figure_path = write_figure(summary, pair_summary, args.figure_dir)
    write_discussion(config, summary, pair_summary, figure_path, args.output_dir, args.discussion_path)
    print(f"saved CIFAR-100-LT ResNet18 recipe benchmark to {args.output_dir}")
    print(
        f"trace rows={len(trace)}, class rows={len(class_metrics)}, "
        f"group rows={len(group_metrics)}, summary rows={len(summary)}, pair rows={len(pair_summary)}"
    )


if __name__ == "__main__":
    main()
