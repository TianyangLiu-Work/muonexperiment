from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    run_cifar100_resnet_one_step,
)
from e11_condition_geometry.reporting import fmt, markdown_table


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_checkpoint_sweep")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_checkpoint_sweep")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_checkpoint_sweep.md")
DEFAULT_WARMUP_STEPS = (250, 500, 1000, 2000)


def parse_warmup_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not steps:
        raise argparse.ArgumentTypeError("expected at least one warmup step")
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("warmup steps must be positive")
    return steps


def run_sweep(
    base_config: Cifar100ResNetOneStepConfig,
    *,
    warmup_steps: tuple[int, ...],
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    layer_frames: list[pd.DataFrame] = []

    for warmup_step in warmup_steps:
        config = replace(base_config, warmup_steps=warmup_step)
        if progress:
            print(f"[resnet-checkpoint-sweep] warmup_steps={warmup_step}", flush=True)
        step_metrics, pair_summary, layer_metrics = run_cifar100_resnet_one_step(config, progress=progress)
        for frame in (step_metrics, pair_summary, layer_metrics):
            frame["warmup_steps"] = int(warmup_step)
        step_frames.append(step_metrics)
        pair_frames.append(pair_summary)
        layer_frames.append(layer_metrics)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        pd.concat(step_frames, ignore_index=True),
        pd.concat(pair_frames, ignore_index=True),
        pd.concat(layer_frames, ignore_index=True),
    )


def write_figure(pair_summary: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = pair_summary.sort_values("warmup_steps")
    x = ordered["warmup_steps"].to_numpy(dtype=float)
    drift = ordered["geomean_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    drift_low = ordered["tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    drift_high = ordered["tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
    margin = ordered["geomean_margin_delta_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    margin_low = ordered["margin_delta_sq_ratio_ci95_low"].to_numpy(dtype=float)
    margin_high = ordered["margin_delta_sq_ratio_ci95_high"].to_numpy(dtype=float)
    tail_acc = ordered["mean_tail_accuracy_before"].to_numpy(dtype=float)
    positive_margin = ordered["mean_tail_positive_margin_fraction_before"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].errorbar(
        x,
        drift,
        yerr=[drift - drift_low, drift_high - drift],
        marker="o",
        capsize=4,
        linewidth=2,
        color="#0072B2",
        label="tail logit drift",
    )
    axes[0].errorbar(
        x,
        margin,
        yerr=[margin - margin_low, margin_high - margin],
        marker="s",
        capsize=4,
        linewidth=2,
        color="#D55E00",
        label="margin delta",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("warmup checkpoint steps")
    axes[0].set_ylabel("spectral / Fro squared ratio")
    axes[0].set_yscale("log")
    axes[0].set_title("Matched-gain tail drift")
    axes[0].legend(frameon=False)

    axes[1].plot(x, tail_acc, marker="o", linewidth=2, color="#009E73", label="tail accuracy")
    axes[1].plot(x, positive_margin, marker="s", linewidth=2, color="#CC79A7", label="positive margin")
    axes[1].set_xlabel("warmup checkpoint steps")
    axes[1].set_ylabel("pre-update tail quality")
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title("Checkpoint quality")
    axes[1].legend(frameon=False)

    fig.suptitle("CIFAR-100-LT ResNet18 checkpoint-quality sweep")
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_checkpoint_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    base_config: Cifar100ResNetOneStepConfig,
    pair_summary: pd.DataFrame,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
) -> None:
    ordered = pair_summary.sort_values("warmup_steps")
    worst_drift = ordered.loc[ordered["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    strongest_tail = ordered.loc[ordered["mean_tail_accuracy_before"].idxmax()]
    table_columns = [
        "warmup_steps",
        "seeds",
        "mean_tail_accuracy_before",
        "mean_tail_positive_margin_fraction_before",
        "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
        "tail_output_drift_sq_ratio_ci95_low",
        "tail_output_drift_sq_ratio_ci95_high",
        "geomean_margin_delta_sq_ratio_spectral_over_fro",
        "margin_delta_sq_ratio_ci95_low",
        "margin_delta_sq_ratio_ci95_high",
        "mean_tail_loss_increase_diff_spectral_minus_fro",
        "tail_loss_increase_diff_ci95_low",
        "tail_loss_increase_diff_ci95_high",
        "mean_tail_accuracy_drop_diff_spectral_minus_fro",
        "tail_accuracy_drop_diff_ci95_low",
        "tail_accuracy_drop_diff_ci95_high",
    ]
    lines = [
        "# E11 CIFAR-100-LT ResNet18 Checkpoint-Quality Sweep",
        "",
        "This generated sweep repeats the matched-head-gain ResNet18 diagnostic across",
        "several warmup checkpoints. It targets the main reviewer objection to the",
        "one-step architecture check: lower drift is most useful if it persists when",
        "the pre-update tail function is not trivially weak.",
        "",
        f"- Seeds per checkpoint: {len(base_config.seeds)}",
        f"- Target head first-order gain: {base_config.target_head_gain_fraction} * head loss",
        f"- Head train examples per class: {base_config.head_train_per_class}",
        f"- Tail train examples per class: {base_config.tail_train_per_class}",
        f"- Tail eval examples per class: {base_config.tail_eval_per_class}",
        f"- Dtype/device request: {base_config.dtype}/{base_config.device}",
        "",
        f"![CIFAR-100-LT ResNet18 checkpoint sweep](../{figure_path.as_posix()})",
        "",
        "## Summary",
        "",
        markdown_table(ordered, table_columns),
        "",
        "## Readout",
        "",
        "The acceptance gate for a top-conference version is not just that the drift",
        "ratio stays below one. The sweep should also show the checkpoint quality",
        "columns improving enough that the preserved tail function is meaningful.",
        "",
        "Current generated readout template:",
        f"- Worst drift-ratio CI upper endpoint: {fmt(worst_drift['tail_output_drift_sq_ratio_ci95_high'])} "
        f"at {int(worst_drift['warmup_steps'])} warmup steps.",
        f"- Strongest pre-update tail accuracy: {fmt(strongest_tail['mean_tail_accuracy_before'])} "
        f"at {int(strongest_tail['warmup_steps'])} warmup steps.",
        "",
        "Artifacts:",
        f"- [step_metrics.csv](../{(output_dir / 'step_metrics.csv').as_posix()})",
        f"- [pair_summary.csv](../{(output_dir / 'pair_summary.csv').as_posix()})",
        f"- [layer_metrics.csv](../{(output_dir / 'layer_metrics.csv').as_posix()})",
        f"- [config.json](../{(output_dir / 'config.json').as_posix()})",
    ]
    discussion_path.parent.mkdir(parents=True, exist_ok=True)
    discussion_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup-steps", type=parse_warmup_steps, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--target-head-gain-fraction", type=float, default=0.005)
    parser.add_argument("--head-train-per-class", type=int, default=None)
    parser.add_argument("--tail-train-per-class", type=int, default=None)
    parser.add_argument("--tail-eval-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--discussion-path", type=Path, default=DEFAULT_DISCUSSION_PATH)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Cifar100ResNetOneStepConfig:
    config = Cifar100ResNetOneStepConfig()
    if args.smoke:
        config = replace(
            config,
            seeds=(0,),
            head_classes=tuple(range(5)),
            tail_classes=tuple(range(5, 10)),
            head_train_per_class=20,
            tail_train_per_class=5,
            tail_eval_per_class=5,
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
    else:
        config = replace(config, seeds=tuple(range(args.seeds)))
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.target_head_gain_fraction is not None:
        config = replace(config, target_head_gain_fraction=args.target_head_gain_fraction)
    if args.head_train_per_class is not None:
        config = replace(config, head_train_per_class=args.head_train_per_class)
    if args.tail_train_per_class is not None:
        config = replace(config, tail_train_per_class=args.tail_train_per_class)
    if args.tail_eval_per_class is not None:
        config = replace(config, tail_eval_per_class=args.tail_eval_per_class)
    if args.batch_size is not None:
        config = replace(config, warmup_batch_size=args.batch_size)
    if args.head_batch_size is not None:
        config = replace(config, head_batch_size=args.head_batch_size)
    if args.lr is not None:
        config = replace(config, lr=args.lr)
    if args.download is not None:
        config = replace(config, download=args.download)
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    warmup_steps = (2, 4) if args.smoke else args.warmup_steps
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    step_metrics, pair_summary, layer_metrics = run_sweep(config, warmup_steps=warmup_steps, progress=args.progress)
    step_metrics.to_csv(output_dir / "step_metrics.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    layer_metrics.to_csv(output_dir / "layer_metrics.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps({"base_config": asdict(config), "warmup_steps": list(warmup_steps)}, indent=2),
        encoding="utf-8",
    )
    figure_path = write_figure(pair_summary, args.figure_dir)
    write_discussion(config, pair_summary, figure_path, output_dir, args.discussion_path)
    print(f"saved CIFAR-100-LT ResNet18 checkpoint sweep to {output_dir}")
    print(f"step rows={len(step_metrics)}, settings={len(pair_summary)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {args.discussion_path}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
