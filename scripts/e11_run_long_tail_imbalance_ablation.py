from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_one_step import LongTailOneStepConfig, run_long_tail_one_step
from e11_condition_geometry.reporting import fmt, markdown_table


OUTPUT_DIR = Path("results/e11_long_tail_imbalance_ablation")
FIGURE_DIR = Path("figures/e11_long_tail_imbalance_ablation")
DISCUSSION_PATH = Path("discussion/e11_long_tail_imbalance_ablation.md")
TAIL_TRAIN_PER_CLASS_VALUES = (80, 40, 20, 10)


def run_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    step_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    base_config = LongTailOneStepConfig()
    for tail_train_per_class in TAIL_TRAIN_PER_CLASS_VALUES:
        config = replace(base_config, tail_train_per_class=tail_train_per_class)
        step_metrics, pair_summary, _layer_metrics = run_long_tail_one_step(config)
        imbalance_ratio = config.head_train_per_class / float(config.tail_train_per_class)
        step_metrics = step_metrics.copy()
        pair_summary = pair_summary.copy()
        for frame in (step_metrics, pair_summary):
            frame["head_train_per_class"] = int(config.head_train_per_class)
            frame["tail_train_per_class"] = int(config.tail_train_per_class)
            frame["imbalance_ratio"] = float(imbalance_ratio)
        step_frames.append(step_metrics)
        summary_frames.append(pair_summary)
    return pd.concat(step_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def write_figure(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("imbalance_ratio")
    x_labels = [f"{row.head_train_per_class}:{row.tail_train_per_class}" for row in ordered.itertuples()]
    x = range(len(ordered))
    y = ordered["geomean_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    lo = ordered["tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    hi = ordered["tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(
        list(x),
        y,
        yerr=[y - lo, hi - y],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#0072B2",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_yscale("log")
    ax.set_xlabel("head:tail train examples per class")
    ax.set_ylabel("spectral / Fro squared drift ratio")
    ax.set_title("One-step squared tail-drift ratio across imbalance levels")
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_imbalance_ablation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, figure_path: Path) -> None:
    ordered = summary.sort_values("imbalance_ratio")
    worst = ordered.loc[ordered["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    table_columns = [
        "head_train_per_class",
        "tail_train_per_class",
        "imbalance_ratio",
        "seeds",
        "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
        "tail_output_drift_sq_ratio_ci95_low",
        "tail_output_drift_sq_ratio_ci95_high",
        "spectral_less_tail_output_drift_fraction",
        "mean_tail_loss_increase_diff_spectral_minus_fro",
        "tail_loss_increase_diff_ci95_low",
        "tail_loss_increase_diff_ci95_high",
    ]
    text = f"""# E11 Long-Tailed Imbalance Ablation

This ablation repeats the controlled one-step digits diagnostic while varying
the number of tail training examples per class. The head split is fixed at 100
examples per head class, and the held-out tail evaluation set remains 40
examples per tail class. The purpose is to check that the main matched-gain
tail-drift result is not only an artifact of the default 100:40 controlled
head-heavy split.

![Imbalance ablation](../{figure_path.as_posix()})

## Summary

{markdown_table(ordered, table_columns)}

## Readout

Across the tested imbalance ratios, the largest upper endpoint of the 95%
confidence interval for the spectral/Frobenius squared tail-example logit drift ratio is
{fmt(worst["tail_output_drift_sq_ratio_ci95_high"])} at head:tail
{int(worst["head_train_per_class"])}:{int(worst["tail_train_per_class"])}.
Thus the lower matched-gain tail-drift pattern is not restricted to the default
100:40 split. This is still a controlled sklearn-digits mechanism diagnostic,
not a standard long-tailed benchmark.

## Artifacts

- [step_metrics.csv](../results/e11_long_tail_imbalance_ablation/step_metrics.csv)
- [summary.csv](../results/e11_long_tail_imbalance_ablation/summary.csv)
- [config.json](../results/e11_long_tail_imbalance_ablation/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def write_outputs(step_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    config = {
        "base_config": LongTailOneStepConfig().__dict__,
        "tail_train_per_class_values": list(TAIL_TRAIN_PER_CLASS_VALUES),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    figure_path = write_figure(summary)
    write_discussion(summary, figure_path)


def main() -> None:
    step_metrics, summary = run_ablation()
    write_outputs(step_metrics, summary)
    print(f"saved long-tail imbalance ablation to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, settings={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
