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


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_lt_standard_eval")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_lt_standard_eval")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_standard_eval.md")
GROUP_ORDER = ("many", "medium", "few", "all")
GROUP_COLORS = {
    "many": "#0072B2",
    "medium": "#009E73",
    "few": "#D55E00",
    "all": "#555555",
}


@dataclass(frozen=True)
class StandardEvalConfig:
    seeds: tuple[int, ...] = tuple(range(10))
    num_classes: int = 100
    max_train_count: int = 500
    imbalance_factor: float = 100.0
    many_threshold: int = 100
    medium_threshold: int = 20
    train_steps: int = 10000
    train_batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    eval_interval: int = 1000
    dtype: str = "float32"
    device: str = "auto"
    data_root: str = "data/torchvision"
    download: bool = True


def long_tail_counts(config: StandardEvalConfig) -> dict[int, int]:
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


def frequency_group(train_count: int, config: StandardEvalConfig) -> str:
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
    generator = make_generator(seed + 51000, device)
    sampled = [
        sample_indices(labels, classes=(class_id,), count_per_class=count, generator=generator)
        for class_id, count in counts.items()
    ]
    return torch.cat(sampled)


def _class_test_indices(labels: torch.Tensor, class_id: int) -> torch.Tensor:
    return torch.nonzero(labels == int(class_id), as_tuple=False).flatten()


def evaluate_class_metrics(
    model: torch.nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    counts: dict[int, int],
    config: StandardEvalConfig,
    *,
    seed: int,
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
    for seed, seed_metrics in class_metrics.groupby("seed", observed=True, sort=True):
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
    for group_name, group in group_metrics.groupby("frequency_group", observed=True, sort=False):
        loss, loss_low, loss_high = ci95(group["loss"])
        accuracy, accuracy_low, accuracy_high = ci95(group["accuracy"])
        balanced, balanced_low, balanced_high = ci95(group["balanced_accuracy"])
        margin, margin_low, margin_high = ci95(group["mean_margin"])
        rows.append(
            {
                "frequency_group": group_name,
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
    return pd.DataFrame(rows).sort_values("frequency_group", key=lambda s: s.map(order)).reset_index(drop=True)


def summarize_class_metrics(class_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_id, group in class_metrics.groupby("class_id", observed=True, sort=True):
        accuracy, accuracy_low, accuracy_high = ci95(group["accuracy"])
        loss, loss_low, loss_high = ci95(group["loss"])
        margin, margin_low, margin_high = ci95(group["mean_margin"])
        rows.append(
            {
                "class_id": int(class_id),
                "train_count": int(group["train_count"].iloc[0]),
                "frequency_group": str(group["frequency_group"].iloc[0]),
                "seeds": int(group["seed"].nunique()),
                "mean_accuracy": accuracy,
                "accuracy_ci95_low": accuracy_low,
                "accuracy_ci95_high": accuracy_high,
                "mean_loss": loss,
                "loss_ci95_low": loss_low,
                "loss_ci95_high": loss_high,
                "mean_margin": margin,
                "margin_ci95_low": margin_low,
                "margin_ci95_high": margin_high,
            }
        )
    return pd.DataFrame(rows)


def run_standard_eval(
    config: StandardEvalConfig,
    *,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(0)
    torch.manual_seed(0)
    device = resolve_device(config.device)
    dtype = dtype_from_name(config.dtype)
    counts = long_tail_counts(config)
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

    for seed in config.seeds:
        if progress:
            print(f"[resnet-lt-standard] seed={seed}: building split", flush=True)
        train_indices = make_train_indices(train_y, counts, seed=seed, device=device)
        torch.manual_seed(seed + 52000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 52000)
        model = build_cifar_resnet18(device=device, dtype=dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        model.train()

        for step in range(1, config.train_steps + 1):
            generator = make_generator(seed + 53000 + step, device)
            batch_indices = batch(train_indices, config.train_batch_size, generator=generator)
            logits = model(train_x[batch_indices])
            loss = F.cross_entropy(logits, train_y[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step == config.train_steps or step % max(config.eval_interval, 1) == 0:
                with torch.no_grad():
                    prediction = logits.argmax(dim=1)
                    batch_accuracy = float((prediction == train_y[batch_indices]).to(torch.float64).mean().cpu())
                trace_rows.append(
                    {
                        "seed": int(seed),
                        "step": int(step),
                        "batch_loss": float(loss.detach().cpu()),
                        "batch_accuracy": batch_accuracy,
                        "train_examples": int(train_indices.numel()),
                        "device": str(device),
                        "dtype": str(dtype).replace("torch.", ""),
                    }
                )
                if progress:
                    print(
                        f"[resnet-lt-standard] seed={seed} step={step}/{config.train_steps} "
                        f"batch_loss={float(loss.detach().cpu()):.4g}",
                        flush=True,
                    )

        if progress:
            print(f"[resnet-lt-standard] seed={seed}: evaluating class groups", flush=True)
        class_frames.append(evaluate_class_metrics(model, test_x, test_y, counts, config, seed=seed))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    class_metrics = pd.concat(class_frames, ignore_index=True)
    group_metrics = summarize_groups(class_metrics)
    summary = summarize_over_seeds(group_metrics)
    class_summary = summarize_class_metrics(class_metrics)
    trace = pd.DataFrame(trace_rows)
    return trace, class_metrics, group_metrics, summary, class_summary


def write_figure(summary: pd.DataFrame, class_summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    grouped = summary[summary["frequency_group"].isin(["many", "medium", "few"])].copy()
    grouped["order"] = grouped["frequency_group"].map({name: index for index, name in enumerate(GROUP_ORDER)})
    grouped = grouped.sort_values("order")
    x = list(range(len(grouped)))
    y = grouped["mean_balanced_accuracy"].to_numpy(dtype=float)
    low = grouped["balanced_accuracy_ci95_low"].to_numpy(dtype=float)
    high = grouped["balanced_accuracy_ci95_high"].to_numpy(dtype=float)
    colors = [GROUP_COLORS[name] for name in grouped["frequency_group"]]
    axes[0].bar(x, y, color=colors, alpha=0.9)
    axes[0].errorbar(x, y, yerr=[y - low, high - y], fmt="none", color="black", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grouped["frequency_group"].tolist())
    axes[0].set_ylabel("balanced accuracy")
    axes[0].set_title("Many/medium/few final accuracy")
    axes[0].set_ylim(bottom=0.0, top=max(0.1, min(1.0, float(max(high) + 0.05))))

    ordered_classes = class_summary.sort_values("train_count", ascending=False)
    for group_name in ["many", "medium", "few"]:
        sub = ordered_classes[ordered_classes["frequency_group"].eq(group_name)]
        axes[1].scatter(
            sub["train_count"],
            sub["mean_accuracy"],
            color=GROUP_COLORS[group_name],
            label=group_name,
            alpha=0.85,
            s=24,
        )
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("train examples per class")
    axes[1].set_ylabel("per-class accuracy")
    axes[1].set_title("Class accuracy vs. long-tail count")
    axes[1].legend(frameon=False)

    fig.suptitle("CIFAR-100-LT ResNet18 standard many/medium/few reporting baseline")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_lt_standard_eval.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    config: StandardEvalConfig,
    summary: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
) -> None:
    by_group = summary.set_index("frequency_group")
    lines = [
        "# E11 CIFAR-100-LT ResNet18 Standard Many/Medium/Few Evaluation",
        "",
        "This baseline adds standard long-tail reporting to the local head-to-tail",
        "paper package. It trains an AdamW ResNet18 on an exponential CIFAR-100-LT",
        "split and reports final many/medium/few class metrics. It is a reporting",
        "protocol baseline, not a tuned long-tail benchmark or a Muon comparison.",
        "",
        f"- Seeds: {config.seeds}",
        f"- Number of classes: {config.num_classes}",
        f"- Max train examples per class: {config.max_train_count}",
        f"- Imbalance factor: {config.imbalance_factor}",
        f"- Many threshold: train count >= {config.many_threshold}",
        f"- Medium threshold: {config.medium_threshold} <= train count < {config.many_threshold}",
        f"- Few threshold: train count < {config.medium_threshold}",
        f"- Train steps: {config.train_steps}",
        f"- Batch size: {config.train_batch_size}",
        f"- Optimizer: AdamW, lr={config.lr}, weight_decay={config.weight_decay}",
        f"- Device/dtype request: {config.device}/{config.dtype}",
        "",
        f"![CIFAR-100-LT ResNet18 standard eval](../{figure_path.as_posix()})",
        "",
        "## Group Summary",
        "",
        markdown_table(
            summary,
            [
                "frequency_group",
                "seeds",
                "classes",
                "min_train_count",
                "max_train_count",
                "mean_balanced_accuracy",
                "balanced_accuracy_ci95_low",
                "balanced_accuracy_ci95_high",
                "mean_loss",
                "loss_ci95_low",
                "loss_ci95_high",
                "mean_margin",
            ],
        ),
        "",
        "## Readout",
        "",
    ]
    if {"many", "medium", "few"}.issubset(set(by_group.index)):
        lines.extend(
            [
                f"- Many-class balanced accuracy: {fmt(by_group.loc['many', 'mean_balanced_accuracy'])} "
                f"[{fmt(by_group.loc['many', 'balanced_accuracy_ci95_low'])}, "
                f"{fmt(by_group.loc['many', 'balanced_accuracy_ci95_high'])}].",
                f"- Medium-class balanced accuracy: {fmt(by_group.loc['medium', 'mean_balanced_accuracy'])} "
                f"[{fmt(by_group.loc['medium', 'balanced_accuracy_ci95_low'])}, "
                f"{fmt(by_group.loc['medium', 'balanced_accuracy_ci95_high'])}].",
                f"- Few-class balanced accuracy: {fmt(by_group.loc['few', 'mean_balanced_accuracy'])} "
                f"[{fmt(by_group.loc['few', 'balanced_accuracy_ci95_low'])}, "
                f"{fmt(by_group.loc['few', 'balanced_accuracy_ci95_high'])}].",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation: this result gives the paper an explicit standard",
            "long-tail reporting surface. It should be used to separate local",
            "matched-head-gain drift claims from benchmark-level classification",
            "claims. A top-tier optimizer claim would still require tuned baselines,",
            "data augmentation, class-balanced methods or loss baselines, and final",
            "many/medium/few metrics for the optimizer under study.",
            "",
            "Artifacts:",
            f"- [train_trace.csv](../{(output_dir / 'train_trace.csv').as_posix()})",
            f"- [class_metrics.csv](../{(output_dir / 'class_metrics.csv').as_posix()})",
            f"- [group_metrics.csv](../{(output_dir / 'group_metrics.csv').as_posix()})",
            f"- [summary.csv](../{(output_dir / 'summary.csv').as_posix()})",
            f"- [class_summary.csv](../{(output_dir / 'class_summary.csv').as_posix()})",
            f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
        ]
    )
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--max-train-count", type=int, default=500)
    parser.add_argument("--imbalance-factor", type=float, default=100.0)
    parser.add_argument("--many-threshold", type=int, default=100)
    parser.add_argument("--medium-threshold", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--discussion-path", type=Path, default=DEFAULT_DISCUSSION_PATH)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StandardEvalConfig:
    if args.smoke:
        return StandardEvalConfig(
            seeds=(0,),
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
    return StandardEvalConfig(
        seeds=tuple(range(args.seeds)),
        num_classes=args.num_classes,
        max_train_count=args.max_train_count,
        imbalance_factor=args.imbalance_factor,
        many_threshold=args.many_threshold,
        medium_threshold=args.medium_threshold,
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        device=args.device or "auto",
        download=True if args.download is None else bool(args.download),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace, class_metrics, group_metrics, summary, class_summary = run_standard_eval(
        config,
        progress=args.progress,
    )
    trace.to_csv(args.output_dir / "train_trace.csv", index=False)
    class_metrics.to_csv(args.output_dir / "class_metrics.csv", index=False)
    group_metrics.to_csv(args.output_dir / "group_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    class_summary.to_csv(args.output_dir / "class_summary.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    figure_path = write_figure(summary, class_summary, args.figure_dir)
    write_discussion(config, summary, figure_path, args.output_dir, args.discussion_path)
    print(f"saved CIFAR-100-LT ResNet18 standard eval to {args.output_dir}")
    print(
        f"trace rows={len(trace)}, class rows={len(class_metrics)}, "
        f"group rows={len(group_metrics)}, summary rows={len(summary)}"
    )
    print(f"figure: {figure_path}")
    print(f"discussion: {args.discussion_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
