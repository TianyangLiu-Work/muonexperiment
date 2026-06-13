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

from e11_condition_geometry.cifar100_long_tail import (
    Cifar100LongTailOneStepConfig,
    run_cifar100_long_tail_one_step,
)


OUTPUT_DIR = Path("results/e11_cifar100_lt_one_step")
FIGURE_DIR = Path("figures/e11_cifar100_lt_one_step")
DISCUSSION_PATH = Path("discussion/e11_cifar100_lt_one_step.md")


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(step_metrics, pair_summary) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    order = ["frobenius", "spectral"]
    colors = {"frobenius": "#D55E00", "spectral": "#0072B2"}
    labels = {"frobenius": "Fro/GD", "spectral": "Spectral"}

    frame = step_metrics.copy()
    frame["tail_output_drift_rms_sq"] = frame["tail_output_drift_rms"] ** 2
    drift_values = [
        frame.loc[frame["geometry"] == geometry, "tail_output_drift_rms_sq"].to_numpy(dtype=float)
        for geometry in order
    ]
    axes[0].boxplot(drift_values, tick_labels=[labels[geometry] for geometry in order], patch_artist=True)
    for patch, geometry in zip(axes[0].artists, order):
        patch.set_facecolor(colors[geometry])
        patch.set_alpha(0.35)
    axes[0].set_ylabel("mean squared tail-example logit drift")
    axes[0].set_title("CIFAR-100-LT tail function movement")

    for seed, group in frame.groupby("seed", observed=True, sort=False):
        group = group.set_index("geometry").loc[order]
        axes[1].plot([0, 1], group["tail_loss_increase"], color="#666666", alpha=0.45, linewidth=1)
        axes[1].scatter(
            [0, 1],
            group["tail_loss_increase"],
            color=[colors["frobenius"], colors["spectral"]],
            s=24,
            zorder=3,
        )
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([labels[geometry] for geometry in order])
    axes[1].set_ylabel("tail loss increase")
    axes[1].set_title("Paired one-step tail loss response")

    row = pair_summary.iloc[0]
    fig.suptitle(
        "CIFAR-100-LT one-step diagnostic: "
        f"squared drift ratio={row['geomean_tail_output_drift_sq_ratio_spectral_over_fro']:.3g} "
        f"[{row['tail_output_drift_sq_ratio_ci95_low']:.3g}, {row['tail_output_drift_sq_ratio_ci95_high']:.3g}]"
    )
    fig.tight_layout()
    path = FIGURE_DIR / "cifar100_lt_one_step_tail_response.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: Cifar100LongTailOneStepConfig, pair_summary, figure_path: Path) -> None:
    row = pair_summary.iloc[0].to_dict()
    lines = [
        "# E11 CIFAR-100-LT One-Step Diagnostic",
        "",
        "This generated probe repeats the matched-head-gain head-to-tail diagnostic on",
        "CIFAR-100 instead of sklearn digits. It uses a deliberately simple two-layer",
        "MLP so the spectral/polar intervention remains exactly defined at each matrix",
        "block, while the dataset and class count are substantially larger.",
        "",
        f"- Seeds: {len(config.seeds)}",
        f"- Head classes: {config.head_classes[0]}--{config.head_classes[-1]} ({len(config.head_classes)} classes)",
        f"- Tail classes: {config.tail_classes[0]}--{config.tail_classes[-1]} ({len(config.tail_classes)} classes)",
        f"- Head train examples per class: {config.head_train_per_class}",
        f"- Tail train examples per class: {config.tail_train_per_class}",
        f"- Tail eval examples per class: {config.tail_eval_per_class} from CIFAR-100 test split",
        f"- Hidden dimension: {config.hidden_dim}",
        f"- Warmup steps: {config.warmup_steps}",
        f"- Warmup/head batch sizes: {config.warmup_batch_size}/{config.head_batch_size}",
        f"- Dtype/device request: {config.dtype}/{config.device}",
        f"- Target head first-order gain: {config.target_head_gain_fraction} * head loss",
        "",
        f"![CIFAR-100-LT tail response](../{figure_path.as_posix()})",
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
        f"| mean tail diagnostic condition score | {_fmt(row['mean_condition_score_tail'])} |",
        "",
        "Interpretation should remain conservative. This is a larger visual-data local",
        "diagnostic, not a full long-tailed optimizer benchmark: it tests whether the",
        "matched-head-gain drift readout survives moving from sklearn digits to",
        "CIFAR-100-LT under an exactly controlled two-matrix intervention.",
        "",
        "Artifacts:",
        "- [step_metrics.csv](../results/e11_cifar100_lt_one_step/step_metrics.csv)",
        "- [pair_summary.csv](../results/e11_cifar100_lt_one_step/pair_summary.csv)",
        "- [layer_metrics.csv](../results/e11_cifar100_lt_one_step/layer_metrics.csv)",
        "- [config.json](../results/e11_cifar100_lt_one_step/config.json)",
    ]
    DISCUSSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny configuration for code validation.")
    parser.add_argument("--device", default=None, help="Device override: cpu, cuda, or auto.")
    parser.add_argument("--seeds", type=int, default=None, help="Use seeds 0..N-1.")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--tail-eval-per-class", type=int, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Cifar100LongTailOneStepConfig:
    config = Cifar100LongTailOneStepConfig()
    if args.smoke:
        config = replace(
            config,
            seeds=(0,),
            head_classes=tuple(range(5)),
            tail_classes=tuple(range(5, 10)),
            head_train_per_class=20,
            tail_train_per_class=5,
            tail_eval_per_class=5,
            hidden_dim=32,
            warmup_steps=2,
            warmup_batch_size=32,
            head_batch_size=32,
            target_head_gain_fraction=0.005,
        )
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.seeds is not None:
        config = replace(config, seeds=tuple(range(args.seeds)))
    if args.warmup_steps is not None:
        config = replace(config, warmup_steps=args.warmup_steps)
    if args.hidden_dim is not None:
        config = replace(config, hidden_dim=args.hidden_dim)
    if args.tail_eval_per_class is not None:
        config = replace(config, tail_eval_per_class=args.tail_eval_per_class)
    if args.download is not None:
        config = replace(config, download=args.download)
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics, pair_summary, layer_metrics = run_cifar100_long_tail_one_step(config)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    pair_summary.to_csv(OUTPUT_DIR / "pair_summary.csv", index=False)
    layer_metrics.to_csv(OUTPUT_DIR / "layer_metrics.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    figure_path = write_figure(step_metrics, pair_summary)
    write_discussion(config, pair_summary, figure_path)
    print(f"saved CIFAR-100-LT one-step results to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, seeds={len(config.seeds)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
