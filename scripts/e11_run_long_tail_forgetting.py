from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_forgetting import LongTailForgettingConfig, run_long_tail_forgetting


OUTPUT_DIR = Path("results/e11_long_tail_forgetting")
FIGURE_DIR = Path("figures/e11_long_tail_forgetting")
DISCUSSION_PATH = Path("discussion/e11_long_tail_forgetting.md")


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def write_figure(step_metrics, summary) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"frobenius": "#D55E00", "spectral": "#0072B2"}
    labels = {"frobenius": "Fro/GD", "spectral": "Spectral"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    grouped = step_metrics.groupby(["geometry", "step"], observed=True, sort=False).agg(
        mean_tail_drift=("tail_output_drift_rms", "mean"),
        mean_tail_loss_increase=("tail_loss_increase", "mean"),
    )
    for geometry in ["frobenius", "spectral"]:
        data = grouped.loc[geometry].reset_index()
        axes[0].plot(data["step"], data["mean_tail_drift"], marker="o", color=colors[geometry], label=labels[geometry])
        axes[1].plot(
            data["step"],
            data["mean_tail_loss_increase"],
            marker="o",
            color=colors[geometry],
            label=labels[geometry],
        )
    axes[0].set_xlabel("head-only step")
    axes[0].set_ylabel("tail output drift RMS")
    axes[0].set_title("Tail function drift accumulates")
    axes[1].set_xlabel("head-only step")
    axes[1].set_ylabel("tail loss increase")
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_title("Tail loss response")
    axes[0].legend(frameon=False)
    row = summary.iloc[0]
    fig.suptitle(
        "Head-only forgetting probe: "
        f"final drift-sq ratio={row['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro']:.3g} "
        f"[{row['final_tail_output_drift_sq_ratio_ci95_low']:.3g}, "
        f"{row['final_tail_output_drift_sq_ratio_ci95_high']:.3g}]"
    )
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_head_only_forgetting.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(config: LongTailForgettingConfig, summary, figure_path: Path) -> None:
    row = summary.iloc[0].to_dict()
    lines = [
        "# E11 Consecutive Head-Only Tail Forgetting Probe",
        "",
        "This probe starts from the same imbalanced sklearn digits MLP checkpoint as the",
        "one-step diagnostic, then runs several consecutive head-only updates while",
        "tail classes 5-9 are held out. Frobenius/GD and spectral/polar directions use",
        "the same precomputed head mini-batches and the same target first-order head",
        "gain schedule for each seed.",
        "",
        f"- Seeds: {len(config.seeds)}",
        f"- Head-only steps: {config.head_only_steps}",
        f"- Head classes: {config.head_classes}",
        f"- Tail classes: {config.tail_classes}",
        f"- Head train examples per class: {config.head_train_per_class}",
        f"- Tail train examples per class: {config.tail_train_per_class}",
        f"- Tail evaluation examples per class: {config.tail_eval_per_class}",
        f"- Warmup steps: {config.warmup_steps}",
        f"- Target head first-order gain: {config.target_head_gain_fraction} * reference head-batch loss",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| final tail drift-sq ratio, spectral/Fro | {_fmt(row['geomean_final_tail_output_drift_sq_ratio_spectral_over_fro'])} [{_fmt(row['final_tail_output_drift_sq_ratio_ci95_low'])}, {_fmt(row['final_tail_output_drift_sq_ratio_ci95_high'])}] |",
        f"| spectral lower final tail drift fraction | {_fmt(row['spectral_less_final_tail_output_drift_fraction'])} |",
        f"| tail-drift area ratio, spectral/Fro | {_fmt(row['geomean_tail_output_drift_area_ratio_spectral_over_fro'])} [{_fmt(row['tail_output_drift_area_ratio_ci95_low'])}, {_fmt(row['tail_output_drift_area_ratio_ci95_high'])}] |",
        f"| spectral lower tail-drift area fraction | {_fmt(row['spectral_less_tail_output_drift_area_fraction'])} |",
        f"| final tail loss increase diff, spectral - Fro | {_fmt(row['mean_final_tail_loss_increase_diff_spectral_minus_fro'])} [{_fmt(row['final_tail_loss_increase_diff_ci95_low'])}, {_fmt(row['final_tail_loss_increase_diff_ci95_high'])}] |",
        f"| final tail margin drop diff, spectral - Fro | {_fmt(row['mean_final_tail_margin_drop_diff_spectral_minus_fro'])} [{_fmt(row['final_tail_margin_drop_diff_ci95_low'])}, {_fmt(row['final_tail_margin_drop_diff_ci95_high'])}] |",
        f"| spectral proxy-drift Spearman | {_fmt(row.get('spectral_proxy_drift_spearman', float('nan')))} [{_fmt(row.get('spectral_proxy_drift_spearman_ci95_low', float('nan')))}, {_fmt(row.get('spectral_proxy_drift_spearman_ci95_high', float('nan')))}] |",
        f"| Fro/GD proxy-drift Spearman | {_fmt(row.get('frobenius_proxy_drift_spearman', float('nan')))} [{_fmt(row.get('frobenius_proxy_drift_spearman_ci95_low', float('nan')))}, {_fmt(row.get('frobenius_proxy_drift_spearman_ci95_high', float('nan')))}] |",
        "",
        f"Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})",
        "",
        "Main readout: this probe tests whether the one-step tail-drift pattern persists",
        "over a short head-only horizon. It should be read as a controlled intervention",
        "on update geometry, not as a full long-tailed training benchmark.",
        "",
        "Caveats:",
        "- The schedule matches first-order head gain, not the realized nonlinear head-loss decrease.",
        "- The cumulative condition proxy is based on MLP layer rank diagnostics, not an exact neural `J_T` coefficient.",
        "- This is still a small sklearn digits probe; the real-data hierarchy should eventually move to CIFAR-100-LT or a similar benchmark.",
        "",
        "Artifacts:",
        "- [step_metrics.csv](../results/e11_long_tail_forgetting/step_metrics.csv)",
        "- [summary.csv](../results/e11_long_tail_forgetting/summary.csv)",
        "- [config.json](../results/e11_long_tail_forgetting/config.json)",
    ]
    DISCUSSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = LongTailForgettingConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics, summary = run_long_tail_forgetting(config)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(step_metrics, summary)
    write_discussion(config, summary, figure_path)
    print(f"saved long-tail forgetting results to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, seeds={len(config.seeds)}, head_only_steps={config.head_only_steps}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
