from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_practical_training import (
    LongTailPracticalTrainingConfig,
    run_long_tail_practical_training,
)


OUTPUT_DIR = Path("results/e11_long_tail_practical_training")
FIGURE_DIR = Path("figures/e11_long_tail_practical_training")
DISCUSSION_PATH = Path("discussion/e11_long_tail_practical_training.md")

COLORS = {"adam": "#D55E00", "ns_muon": "#0072B2"}
LABELS = {"adam": "Adam", "ns_muon": "NS-Muon-style"}


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(step_metrics) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grouped = (
        step_metrics.groupby(["optimizer", "step"], observed=True, sort=False)
        .agg(
            tail_eval_loss=("tail_eval_loss", "mean"),
            tail_eval_accuracy=("tail_eval_accuracy", "mean"),
            tail_eval_margin=("tail_eval_margin", "mean"),
            head_loss=("head_loss", "mean"),
            train_loss=("train_loss", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    panels = [
        ("tail_eval_loss", "tail eval loss", "Tail loss"),
        ("tail_eval_margin", "tail eval margin", "Tail margin"),
        ("head_loss", "head loss", "Head loss"),
    ]
    for optimizer in ["adam", "ns_muon"]:
        sub = grouped[grouped["optimizer"].eq(optimizer)]
        for axis, (column, ylabel, title) in zip(axes, panels):
            axis.plot(
                sub["step"],
                sub[column],
                color=COLORS[optimizer],
                linewidth=1.8,
                label=LABELS[optimizer],
            )
            axis.set_xlabel("training step")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].legend(frameon=False)
    fig.suptitle("Long-tailed digits practical training diagnostic")
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_practical_training.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: LongTailPracticalTrainingConfig, summary, figure_path: Path) -> None:
    row = summary.iloc[0].to_dict()
    lines = [
        "# E11 Long-Tailed Practical Training Diagnostic",
        "",
        "This diagnostic compares practical Adam and a finite-Newton-Schulz Muon-style",
        "optimizer on the same imbalanced sklearn digits task, from identical",
        "initialization and with identical mini-batch/noise schedules for each seed.",
        "It is a training-behavior sanity check, not a tuned optimizer leaderboard.",
        "",
        f"- Seeds: {config.seeds}",
        f"- Train steps: {config.train_steps}",
        f"- Batch size: {config.train_batch_size}",
        f"- Head classes: {config.head_classes}, {config.head_train_per_class} train examples/class",
        f"- Tail classes: {config.tail_classes}, {config.tail_train_per_class} train examples/class",
        f"- Tail eval examples/class: {config.tail_eval_per_class}",
        f"- Adam lr: {config.adam_lr}",
        f"- NS-Muon-style lr: {config.muon_lr}",
        f"- Momentum beta: {config.momentum_beta}",
        f"- Newton-Schulz steps: {config.newton_schulz_steps}",
        "",
        "| quantity | Muon vs Adam |",
        "|---|---:|",
        f"| final train loss ratio | {_fmt(row['geomean_final_train_loss_ratio_muon_over_adam'])} [{_fmt(row['final_train_loss_ratio_ci95_low'])}, {_fmt(row['final_train_loss_ratio_ci95_high'])}] |",
        f"| final head loss ratio | {_fmt(row['geomean_final_head_loss_ratio_muon_over_adam'])} [{_fmt(row['final_head_loss_ratio_ci95_low'])}, {_fmt(row['final_head_loss_ratio_ci95_high'])}] |",
        f"| final tail eval loss ratio | {_fmt(row['geomean_final_tail_eval_loss_ratio_muon_over_adam'])} [{_fmt(row['final_tail_eval_loss_ratio_ci95_low'])}, {_fmt(row['final_tail_eval_loss_ratio_ci95_high'])}] |",
        f"| final tail eval accuracy diff | {_fmt(row['mean_final_tail_eval_accuracy_diff_muon_minus_adam'])} [{_fmt(row['final_tail_eval_accuracy_diff_ci95_low'])}, {_fmt(row['final_tail_eval_accuracy_diff_ci95_high'])}] |",
        f"| final tail eval margin diff | {_fmt(row['mean_final_tail_eval_margin_diff_muon_minus_adam'])} [{_fmt(row['final_tail_eval_margin_diff_ci95_low'])}, {_fmt(row['final_tail_eval_margin_diff_ci95_high'])}] |",
        f"| final tail output-drift RMS ratio | {_fmt(row['geomean_final_tail_eval_drift_rms_ratio_muon_over_adam'])} [{_fmt(row['final_tail_eval_drift_rms_ratio_ci95_low'])}, {_fmt(row['final_tail_eval_drift_rms_ratio_ci95_high'])}] |",
        "",
        f"Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})",
        "",
        "Interpretation:",
        "",
        "This experiment checks whether the local matched-head-gain drift pattern is",
        "consistent with an actual imbalanced mini-batch training loop. In the",
        "selected lightweight setting, the NS-Muon-style run has lower train/head",
        "loss, lower tail eval loss, and lower tail drift than Adam, while tail",
        "accuracy remains unchanged. This supports a narrow practical sanity check",
        "and reinforces the paper's distinction between continuous logits on tail examples,",
        "tail loss/margin, and discrete classification accuracy.",
        "",
        "Caveats:",
        "",
        "- The Muon-style optimizer is a small finite-Newton-Schulz implementation for two matrix layers, not a full production Muon stack.",
        "- Hyperparameters are fixed and intentionally lightweight; this is not an optimizer leaderboard.",
        "- The task is sklearn digits, so CIFAR-100-LT or another real long-tail benchmark is still needed for a strong empirical paper.",
        "",
        "Artifacts:",
        "",
        "- [step_metrics.csv](../results/e11_long_tail_practical_training/step_metrics.csv)",
        "- [summary.csv](../results/e11_long_tail_practical_training/summary.csv)",
        "- [config.json](../results/e11_long_tail_practical_training/config.json)",
    ]
    DISCUSSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = LongTailPracticalTrainingConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics, summary = run_long_tail_practical_training(config)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(step_metrics)
    write_discussion(config, summary, figure_path)
    print(f"saved long-tail practical training results to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, seeds={len(config.seeds)}, train_steps={config.train_steps}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
