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


OUTPUT_DIR = Path("results/e11_long_tail_rho_sweep")
FIGURE_DIR = Path("figures/e11_long_tail_rho_sweep")
DISCUSSION_PATH = Path("discussion/e11_long_tail_rho_sweep.md")
TARGET_HEAD_GAIN_FRACTIONS = (0.005, 0.01, 0.02, 0.04, 0.08)


def run_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_config = LongTailOneStepConfig()
    step_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for target_fraction in TARGET_HEAD_GAIN_FRACTIONS:
        config = replace(base_config, target_head_gain_fraction=target_fraction)
        step_metrics, pair_summary, _layer_metrics = run_long_tail_one_step(config)
        step_metrics = step_metrics.copy()
        pair_summary = pair_summary.copy()
        for frame in (step_metrics, pair_summary):
            frame["target_head_gain_fraction"] = float(target_fraction)
        step_frames.append(step_metrics)
        summary_frames.append(pair_summary)
    return pd.concat(step_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def write_figure(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("target_head_gain_fraction")
    x = ordered["target_head_gain_fraction"].to_numpy(dtype=float)
    drift = ordered["geomean_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    drift_lo = ordered["tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    drift_hi = ordered["tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
    centered = ordered["geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro"].to_numpy(dtype=float)
    centered_lo = ordered["centered_tail_output_drift_sq_ratio_ci95_low"].to_numpy(dtype=float)
    centered_hi = ordered["centered_tail_output_drift_sq_ratio_ci95_high"].to_numpy(dtype=float)
    head_error_fro = ordered["mean_actual_head_gain_relative_error_frobenius"].to_numpy(dtype=float)
    head_error_spectral = ordered["mean_actual_head_gain_relative_error_spectral"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].errorbar(
        x,
        drift,
        yerr=[drift - drift_lo, drift_hi - drift],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#0072B2",
        label="tail-example logit drift",
    )
    axes[0].errorbar(
        x,
        centered,
        yerr=[centered - centered_lo, centered_hi - centered],
        marker="s",
        linewidth=2,
        capsize=4,
        color="#009E73",
        label="centered-logit drift",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"target head gain fraction $\rho/L_H$")
    axes[0].set_ylabel("spectral / Fro squared ratio")
    axes[0].set_title("Tail drift ratio")
    axes[0].legend(frameon=False)

    axes[1].plot(x, head_error_fro, marker="o", linewidth=2, color="#0072B2", label="Fro/GD")
    axes[1].plot(x, head_error_spectral, marker="s", linewidth=2, color="#D55E00", label="spectral/polar")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"target head gain fraction $\rho/L_H$")
    axes[1].set_ylabel("mean actual head-gain relative error")
    axes[1].set_title("Matched-gain linearity check")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_rho_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, figure_path: Path) -> None:
    ordered = summary.sort_values("target_head_gain_fraction")
    worst_drift = ordered.loc[ordered["tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    worst_centered = ordered.loc[ordered["centered_tail_output_drift_sq_ratio_ci95_high"].idxmax()]
    worst_head_error = ordered.loc[
        ordered[
            [
                "actual_head_gain_relative_error_frobenius_ci95_high",
                "actual_head_gain_relative_error_spectral_ci95_high",
            ]
        ]
        .max(axis=1)
        .idxmax()
    ]
    table_columns = [
        "target_head_gain_fraction",
        "seeds",
        "geomean_tail_output_drift_sq_ratio_spectral_over_fro",
        "tail_output_drift_sq_ratio_ci95_low",
        "tail_output_drift_sq_ratio_ci95_high",
        "geomean_centered_tail_output_drift_sq_ratio_spectral_over_fro",
        "centered_tail_output_drift_sq_ratio_ci95_low",
        "centered_tail_output_drift_sq_ratio_ci95_high",
        "mean_actual_head_gain_relative_error_frobenius",
        "actual_head_gain_relative_error_frobenius_ci95_high",
        "mean_actual_head_gain_relative_error_spectral",
        "actual_head_gain_relative_error_spectral_ci95_high",
        "mean_tail_loss_increase_diff_spectral_minus_fro",
    ]
    text = f"""# E11 Long-Tailed Matched-Gain Rho Sweep

This sweep repeats the controlled one-step digits diagnostic while varying the
target first-order head gain \\(\\rho\\) as a fraction of the head-batch loss.
The purpose is to check whether the lower matched-gain tail-drift readout is
an artifact of the default \\(\\rho = 0.02 L_H\\) scale.

![Rho sweep](../{figure_path.as_posix()})

## Summary

{markdown_table(ordered, table_columns)}

## Readout

Across target head-gain fractions {", ".join(fmt(x) for x in TARGET_HEAD_GAIN_FRACTIONS)}, the largest upper endpoint of the 95% confidence interval for the spectral/Frobenius squared tail-example logit drift ratio is {fmt(worst_drift["tail_output_drift_sq_ratio_ci95_high"])} at \\(\\rho/L_H={fmt(worst_drift["target_head_gain_fraction"])}\\). For centered-logit drift, the largest upper endpoint is {fmt(worst_centered["centered_tail_output_drift_sq_ratio_ci95_high"])} at \\(\\rho/L_H={fmt(worst_centered["target_head_gain_fraction"])}\\).

The largest actual-head-gain relative-error upper endpoint across both geometries is {fmt(max(worst_head_error["actual_head_gain_relative_error_frobenius_ci95_high"], worst_head_error["actual_head_gain_relative_error_spectral_ci95_high"]))} at \\(\\rho/L_H={fmt(worst_head_error["target_head_gain_fraction"])}\\). This records the local-linearity cost of increasing the matched-gain scale; it should be read together with the drift ratios rather than as a final optimizer-performance result.

## Artifacts

- [step_metrics.csv](../results/e11_long_tail_rho_sweep/step_metrics.csv)
- [summary.csv](../results/e11_long_tail_rho_sweep/summary.csv)
- [config.json](../results/e11_long_tail_rho_sweep/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def write_outputs(step_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    config = {
        "base_config": LongTailOneStepConfig().__dict__,
        "target_head_gain_fractions": list(TARGET_HEAD_GAIN_FRACTIONS),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    figure_path = write_figure(summary)
    write_discussion(summary, figure_path)


def main() -> None:
    step_metrics, summary = run_sweep()
    write_outputs(step_metrics, summary)
    print(f"saved long-tail rho sweep to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, settings={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
