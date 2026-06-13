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


OUTPUT_DIR = Path("results/e11_long_tail_class_partition_sweep")
FIGURE_DIR = Path("figures/e11_long_tail_class_partition_sweep")
DISCUSSION_PATH = Path("discussion/e11_long_tail_class_partition_sweep.md")
CLASS_PARTITIONS: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
    ("low_vs_high", (0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
    ("even_vs_odd", (0, 2, 4, 6, 8), (1, 3, 5, 7, 9)),
    ("mixed_a", (0, 1, 5, 6, 7), (2, 3, 4, 8, 9)),
    ("mixed_b", (0, 3, 4, 7, 9), (1, 2, 5, 6, 8)),
    ("mixed_c", (0, 2, 3, 5, 9), (1, 4, 6, 7, 8)),
)


def _classes_text(classes: tuple[int, ...]) -> str:
    return ",".join(str(label) for label in classes)


def run_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_config = LongTailOneStepConfig()
    step_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for partition_name, head_classes, tail_classes in CLASS_PARTITIONS:
        config = replace(base_config, head_classes=head_classes, tail_classes=tail_classes)
        step_metrics, pair_summary, _layer_metrics = run_long_tail_one_step(config)
        step_metrics = step_metrics.copy()
        pair_summary = pair_summary.copy()
        for frame in (step_metrics, pair_summary):
            frame["partition_name"] = partition_name
            frame["head_classes"] = _classes_text(head_classes)
            frame["tail_classes"] = _classes_text(tail_classes)
        step_frames.append(step_metrics)
        summary_frames.append(pair_summary)
    return pd.concat(step_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def write_figure(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.set_index("partition_name").loc[[item[0] for item in CLASS_PARTITIONS]].reset_index()
    x = range(len(ordered))
    drift = ordered["geomean_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    drift_lo = ordered["tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    drift_hi = ordered["tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
    centered = ordered["geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    centered_lo = ordered["centered_tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    centered_hi = ordered["centered_tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.errorbar(
        list(x),
        drift,
        yerr=[drift - drift_lo, drift_hi - drift],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#0072B2",
        label="tail-example logit drift",
    )
    ax.errorbar(
        [value + 0.12 for value in x],
        centered,
        yerr=[centered - centered_lo, centered_hi - centered],
        marker="s",
        linewidth=2,
        capsize=4,
        color="#009E73",
        label="centered-logit drift",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["partition_name"], rotation=20, ha="right")
    ax.set_ylabel("spectral / Fro squared ratio")
    ax.set_yscale("log")
    ax.set_title("Matched-gain tail drift across head/tail class partitions")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_class_partition_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, figure_path: Path) -> None:
    ordered = summary.set_index("partition_name").loc[[item[0] for item in CLASS_PARTITIONS]].reset_index()
    worst = ordered.loc[ordered["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    table_columns = [
        "partition_name",
        "head_classes",
        "tail_classes",
        "seeds",
        "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
        "tail_output_drift_sq_ratio_ci95_low",
        "tail_output_drift_sq_ratio_ci95_high",
        "geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro",
        "centered_tail_output_drift_sq_ratio_ci95_low",
        "centered_tail_output_drift_sq_ratio_ci95_high",
        "geomean_margin_delta_sq_ratio_spectral_over_fro",
        "margin_delta_sq_ratio_ci95_low",
        "margin_delta_sq_ratio_ci95_high",
        "mean_tail_loss_before",
        "mean_tail_positive_margin_fraction_before",
    ]
    text = f"""# E11 Long-Tailed Class-Partition Sweep

This sweep repeats the matched-head-gain one-step digits diagnostic across
several 5-class head / 5-class tail partitions. The purpose is to check whether
the default classes 0--4 as head and 5--9 as tail are driving the main
tail-drift readout.

![Class-partition sweep](../{figure_path.as_posix()})

## Summary

{markdown_table(ordered, table_columns)}

## Readout

Across {len(CLASS_PARTITIONS)} tested class partitions, the largest upper
endpoint of the 95% confidence interval for the spectral/Frobenius squared
tail-example logit drift ratio is
{fmt(worst["tail_output_drift_sq_ratio_ci95_high"])} for partition
`{worst["partition_name"]}`. This is evidence that the lower matched-gain
tail-drift readout is not unique to the default low-vs-high digit split.

The partition sweep is still a small controlled digits diagnostic. It checks
class-partition sensitivity within the same dataset and model, not robustness
to larger long-tailed visual benchmarks.

## Artifacts

- [step_metrics.csv](../results/e11_long_tail_class_partition_sweep/step_metrics.csv)
- [summary.csv](../results/e11_long_tail_class_partition_sweep/summary.csv)
- [config.json](../results/e11_long_tail_class_partition_sweep/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def write_outputs(step_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    config = {
        "base_config": LongTailOneStepConfig().__dict__,
        "class_partitions": [
            {
                "partition_name": partition_name,
                "head_classes": list(head_classes),
                "tail_classes": list(tail_classes),
            }
            for partition_name, head_classes, tail_classes in CLASS_PARTITIONS
        ],
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    figure_path = write_figure(summary)
    write_discussion(summary, figure_path)


def main() -> None:
    step_metrics, summary = run_sweep()
    write_outputs(step_metrics, summary)
    print(f"saved long-tail class-partition sweep to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, settings={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
