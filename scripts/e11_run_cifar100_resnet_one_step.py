from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    run_cifar100_resnet_one_step,
)


DEFAULT_OUTPUT_DIR = Path("results/e11_cifar100_resnet_one_step")
DEFAULT_FIGURE_DIR = Path("figures/e11_cifar100_resnet_one_step")
DEFAULT_DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_one_step.md")


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(step_metrics, pair_summary, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    order = ["frobenius", "spectral"]
    colors = {"frobenius": "#D55E00", "spectral": "#0072B2"}
    labels = {"frobenius": "Fro/GD", "spectral": "Spectral"}

    frame = step_metrics.copy()
    frame["tail_output_drift_rms_sq"] = frame["tail_output_drift_rms"] ** 2
    values = [
        frame.loc[frame["geometry"] == geometry, "tail_output_drift_rms_sq"].to_numpy(dtype=float)
        for geometry in order
    ]
    axes[0].boxplot(values, tick_labels=[labels[geometry] for geometry in order], patch_artist=True)
    for patch, geometry in zip(axes[0].artists, order):
        patch.set_facecolor(colors[geometry])
        patch.set_alpha(0.35)
    axes[0].set_ylabel("mean squared tail-example logit drift")
    axes[0].set_title("ResNet18 CIFAR-100-LT tail drift")

    for _seed, group in frame.groupby("seed", observed=True, sort=False):
        group = group.set_index("geometry").loc[order]
        axes[1].plot([0, 1], group["tail_loss_increase"], color="#666666", alpha=0.45, linewidth=1)
        axes[1].scatter([0, 1], group["tail_loss_increase"], color=[colors[g] for g in order], s=24, zorder=3)
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([labels[geometry] for geometry in order])
    axes[1].set_ylabel("tail loss increase")
    axes[1].set_title("Paired one-step tail loss response")

    row = pair_summary.iloc[0]
    fig.suptitle(
        "ResNet18 CIFAR-100-LT one-step diagnostic: "
        f"squared drift ratio={row['geomean_tail_output_drift_sq_ratio_spectral_over_fro']:.3g} "
        f"[{row['tail_output_drift_sq_ratio_ci95_low']:.3g}, {row['tail_output_drift_sq_ratio_ci95_high']:.3g}]"
    )
    fig.tight_layout()
    path = figure_dir / "cifar100_resnet_one_step_tail_response.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    config: Cifar100ResNetOneStepConfig,
    pair_summary,
    figure_path: Path,
    output_dir: Path,
    discussion_path: Path,
) -> None:
    row = pair_summary.iloc[0].to_dict()
    lines = [
        "# E11 CIFAR-100-LT ResNet18 One-Step Diagnostic",
        "",
        "This generated probe repeats the matched-head-gain head-to-tail diagnostic on",
        "CIFAR-100-LT using a ResNet18 architecture with a CIFAR-style stem. The",
        "checkpoint is warmed up on the imbalanced train split, then BatchNorm is fixed",
        "with `eval()` for the one-step diagnostic. The intervention updates only",
        "Conv/Linear matrix weights: Conv kernels are flattened to",
        "`out_channels x (in_channels * kernel_height * kernel_width)` for the",
        "spectral/polar direction; BatchNorm and bias parameters are left unchanged.",
        "",
        f"- Seeds: {len(config.seeds)}",
        f"- Head classes: {config.head_classes[0]}--{config.head_classes[-1]} ({len(config.head_classes)} classes)",
        f"- Tail classes: {config.tail_classes[0]}--{config.tail_classes[-1]} ({len(config.tail_classes)} classes)",
        f"- Head train examples per class: {config.head_train_per_class}",
        f"- Tail train examples per class: {config.tail_train_per_class}",
        f"- Tail eval examples per class: {config.tail_eval_per_class} from CIFAR-100 test split",
        f"- Warmup steps: {config.warmup_steps}",
        f"- Warmup/head batch sizes: {config.warmup_batch_size}/{config.head_batch_size}",
        f"- AdamW lr/weight decay: {config.lr}/{config.weight_decay}",
        f"- Dtype/device request: {config.dtype}/{config.device}",
        f"- Target head first-order gain: {config.target_head_gain_fraction} * head loss",
        "",
        f"![ResNet18 CIFAR-100-LT tail response](../{figure_path.as_posix()})",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| squared tail-example logit drift ratio, spectral/Fro | {_fmt(row['geomean_tail_output_drift_sq_ratio_spectral_over_fro'])} [{_fmt(row['tail_output_drift_sq_ratio_ci95_low'])}, {_fmt(row['tail_output_drift_sq_ratio_ci95_high'])}] |",
        f"| centered-logit squared drift ratio, spectral/Fro | {_fmt(row['geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro'])} [{_fmt(row['centered_tail_output_drift_sq_ratio_ci95_low'])}, {_fmt(row['centered_tail_output_drift_sq_ratio_ci95_high'])}] |",
        f"| margin-delta squared ratio, spectral/Fro | {_fmt(row['geomean_margin_delta_sq_ratio_spectral_over_fro'])} [{_fmt(row['margin_delta_sq_ratio_ci95_low'])}, {_fmt(row['margin_delta_sq_ratio_ci95_high'])}] |",
        f"| spectral lower squared tail-example logit drift fraction | {_fmt(row['spectral_less_tail_output_drift_fraction'])} |",
        f"| tail loss increase diff, spectral - Fro | {_fmt(row['mean_tail_loss_increase_diff_spectral_minus_fro'])} [{_fmt(row['tail_loss_increase_diff_ci95_low'])}, {_fmt(row['tail_loss_increase_diff_ci95_high'])}] |",
        f"| tail margin drop diff, spectral - Fro | {_fmt(row['mean_tail_margin_drop_diff_spectral_minus_fro'])} [{_fmt(row['tail_margin_drop_diff_ci95_low'])}, {_fmt(row['tail_margin_drop_diff_ci95_high'])}] |",
        f"| tail accuracy drop diff, spectral - Fro | {_fmt(row['mean_tail_accuracy_drop_diff_spectral_minus_fro'])} [{_fmt(row['tail_accuracy_drop_diff_ci95_low'])}, {_fmt(row['tail_accuracy_drop_diff_ci95_high'])}] |",
        f"| actual head-gain relative error, Fro/GD | {_fmt(row['mean_actual_head_gain_relative_error_frobenius'])} [{_fmt(row['actual_head_gain_relative_error_frobenius_ci95_low'])}, {_fmt(row['actual_head_gain_relative_error_frobenius_ci95_high'])}] |",
        f"| actual head-gain relative error, spectral | {_fmt(row['mean_actual_head_gain_relative_error_spectral'])} [{_fmt(row['actual_head_gain_relative_error_spectral_ci95_low'])}, {_fmt(row['actual_head_gain_relative_error_spectral_ci95_high'])}] |",
        f"| tail CE before | {_fmt(row['mean_tail_loss_before'])} [{_fmt(row['tail_loss_before_ci95_low'])}, {_fmt(row['tail_loss_before_ci95_high'])}] |",
        f"| tail accuracy before | {_fmt(row['mean_tail_accuracy_before'])} [{_fmt(row['tail_accuracy_before_ci95_low'])}, {_fmt(row['tail_accuracy_before_ci95_high'])}] |",
        f"| mean matrix gradient nuclear rank | {_fmt(row['mean_nrG'])} |",
        "",
        "Interpretation should remain conservative. This is a larger architecture",
        "diagnostic for local function drift, not a full practical Muon optimizer",
        "benchmark. It is stronger than the two-layer MLP check because it tests the",
        "same matched-gain readout with Conv blocks and BatchNorm state fixed during",
        "measurement.",
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
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--head-train-per-class", type=int, default=None)
    parser.add_argument("--tail-train-per-class", type=int, default=None)
    parser.add_argument("--tail-eval-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target-head-gain-fraction", type=float, default=None)
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
            warmup_steps=2,
            warmup_batch_size=16,
            head_batch_size=16,
            target_head_gain_fraction=0.001,
        )
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.seeds is not None:
        config = replace(config, seeds=tuple(range(args.seeds)))
    if args.warmup_steps is not None:
        config = replace(config, warmup_steps=args.warmup_steps)
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
    if args.target_head_gain_fraction is not None:
        config = replace(config, target_head_gain_fraction=args.target_head_gain_fraction)
    if args.download is not None:
        config = replace(config, download=args.download)
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    output_dir = args.output_dir
    figure_dir = args.figure_dir
    discussion_path = args.discussion_path
    output_dir.mkdir(parents=True, exist_ok=True)
    step_metrics, pair_summary, layer_metrics = run_cifar100_resnet_one_step(config, progress=args.progress)
    step_metrics.to_csv(output_dir / "step_metrics.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    layer_metrics.to_csv(output_dir / "layer_metrics.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    figure_path = write_figure(step_metrics, pair_summary, figure_dir)
    write_discussion(config, pair_summary, figure_path, output_dir, discussion_path)
    print(f"saved CIFAR-100-LT ResNet18 one-step results to {output_dir}")
    print(f"step rows={len(step_metrics)}, seeds={len(config.seeds)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {discussion_path}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
