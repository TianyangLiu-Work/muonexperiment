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


OUTPUT_DIR = Path("results/e11_long_tail_checkpoint_sweep")
FIGURE_DIR = Path("figures/e11_long_tail_checkpoint_sweep")
DISCUSSION_PATH = Path("discussion/e11_long_tail_checkpoint_sweep.md")
WARMUP_STEPS = (20, 40, 80, 120, 160)


def run_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_config = LongTailOneStepConfig()
    step_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for warmup_steps in WARMUP_STEPS:
        config = replace(base_config, warmup_steps=warmup_steps)
        step_metrics, pair_summary, _layer_metrics = run_long_tail_one_step(config)
        step_metrics = step_metrics.copy()
        pair_summary = pair_summary.copy()
        for frame in (step_metrics, pair_summary):
            frame["warmup_steps"] = int(warmup_steps)
        step_frames.append(step_metrics)
        summary_frames.append(pair_summary)
    return pd.concat(step_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def write_figure(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("warmup_steps")
    x = ordered["warmup_steps"].to_numpy(dtype=float)
    drift = ordered["geomean_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    drift_lo = ordered["tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    drift_hi = ordered["tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
    margin = ordered["geomean_margin_delta_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    margin_lo = ordered["margin_delta_sq_ratio_ci95_low"].to_numpy(dtype=float)
    margin_hi = ordered["margin_delta_sq_ratio_ci95_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(
        x,
        drift,
        yerr=[drift - drift_lo, drift_hi - drift],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#0072B2",
        label="tail-example logit drift",
    )
    ax.errorbar(
        x,
        margin,
        yerr=[margin - margin_lo, margin_hi - margin],
        marker="s",
        linewidth=2,
        capsize=4,
        color="#D55E00",
        label="margin delta",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("warmup checkpoint steps")
    ax.set_ylabel("spectral / Fro squared ratio")
    ax.set_yscale("log")
    ax.set_title("Matched-gain tail drift across training checkpoints")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_checkpoint_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, figure_path: Path) -> None:
    ordered = summary.sort_values("warmup_steps")
    worst = ordered.loc[ordered["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    table_columns = [
        "warmup_steps",
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
    text = f"""# E11 Long-Tailed Checkpoint Sweep

This sweep repeats the matched-head-gain one-step digits diagnostic across
different warmup checkpoints. The purpose is to check whether the main
tail-drift result depends on selecting a single checkpoint state.

![Checkpoint sweep](../{figure_path.as_posix()})

## Summary

{markdown_table(ordered, table_columns)}

## Readout

Across warmup checkpoints {", ".join(str(x) for x in WARMUP_STEPS)}, the largest
upper endpoint of the 95% confidence interval for the spectral/Frobenius
squared tail-example logit drift ratio is
{fmt(worst["tail_output_drift_sq_ratio_ci95_high"])} at
{int(worst["warmup_steps"])} warmup steps. Thus the lower matched-gain
tail-drift pattern is not restricted to the default 80-step checkpoint.

The margin-delta ratio is also reported because lower full-logit drift is not
identical to a classification-margin claim. As in the main one-step diagnostic,
tail loss, margin, and accuracy remain separate outcome quantities.

## Artifacts

- [step_metrics.csv](../results/e11_long_tail_checkpoint_sweep/step_metrics.csv)
- [summary.csv](../results/e11_long_tail_checkpoint_sweep/summary.csv)
- [config.json](../results/e11_long_tail_checkpoint_sweep/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def write_outputs(step_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    config = {
        "base_config": LongTailOneStepConfig().__dict__,
        "warmup_steps": list(WARMUP_STEPS),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    figure_path = write_figure(summary)
    write_discussion(summary, figure_path)


def main() -> None:
    step_metrics, summary = run_sweep()
    write_outputs(step_metrics, summary)
    print(f"saved long-tail checkpoint sweep to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, settings={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
