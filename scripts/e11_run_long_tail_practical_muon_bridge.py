from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.long_tail_muon_bridge import (
    LongTailPracticalMuonBridgeConfig,
    run_long_tail_practical_muon_bridge,
)
from e11_condition_geometry.statistics import ci95, log_ratio_ci95


OUTPUT_DIR = Path("results/e11_long_tail_practical_muon_bridge")
FIGURE_DIR = Path("figures/e11_long_tail_practical_muon_bridge")
DISCUSSION_PATH = Path("discussion/e11_long_tail_practical_muon_bridge.md")

DISPLAY_NAMES = {
    "polar_grad": r"polar($G_t$)",
    "polar_momentum": r"polar($M_t$)",
    "ns_momentum": r"NS($M_t$)",
}
COLORS = {"polar_grad": "#0072B2", "polar_momentum": "#009E73", "ns_momentum": "#CC79A7"}


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _per_step_summary(step_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trajectory_step, step_group in step_metrics.groupby("trajectory_step", observed=True, sort=True):
        for direction in ["polar_grad", "polar_momentum", "ns_momentum"]:
            ratios = []
            cosines = []
            momentum_cosines = []
            for _seed, seed_group in step_group.groupby("seed", observed=True, sort=False):
                by_direction = seed_group.set_index("direction")
                baseline = by_direction.loc["frobenius_grad"]
                variant = by_direction.loc[direction]
                ratios.append(
                    float(
                        variant["tail_output_drift_fro"] ** 2
                        / max(baseline["tail_output_drift_fro"] ** 2, 1e-300)
                    )
                )
                cosines.append(float(variant["direction_cosine_to_polar_grad"]))
                momentum_cosines.append(float(variant["gradient_momentum_cosine"]))
            ratio_mean, ratio_low, ratio_high = log_ratio_ci95(pd.Series(ratios))
            cosine_mean, cosine_low, cosine_high = ci95(pd.Series(cosines))
            momentum_mean, momentum_low, momentum_high = ci95(pd.Series(momentum_cosines))
            rows.append(
                {
                    "trajectory_step": int(trajectory_step),
                    "direction": direction,
                    "geomean_tail_output_drift_sq_ratio_vs_fro": ratio_mean,
                    "tail_output_drift_sq_ratio_vs_fro_ci95_low": ratio_low,
                    "tail_output_drift_sq_ratio_vs_fro_ci95_high": ratio_high,
                    "mean_direction_cosine_to_polar_grad": cosine_mean,
                    "direction_cosine_to_polar_grad_ci95_low": cosine_low,
                    "direction_cosine_to_polar_grad_ci95_high": cosine_high,
                    "mean_gradient_momentum_cosine": momentum_mean,
                    "gradient_momentum_cosine_ci95_low": momentum_low,
                    "gradient_momentum_cosine_ci95_high": momentum_high,
                }
            )
    return pd.DataFrame(rows)


def write_figure(step_metrics: pd.DataFrame, step_summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for direction in ["polar_grad", "polar_momentum", "ns_momentum"]:
        sub = step_summary[step_summary["direction"].eq(direction)].sort_values("trajectory_step")
        x = sub["trajectory_step"].to_numpy(dtype=float)
        y = sub["geomean_tail_output_drift_sq_ratio_vs_fro"].to_numpy(dtype=float)
        low = sub["tail_output_drift_sq_ratio_vs_fro_ci95_low"].to_numpy(dtype=float)
        high = sub["tail_output_drift_sq_ratio_vs_fro_ci95_high"].to_numpy(dtype=float)
        axes[0].errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            marker="o",
            linewidth=1.6,
            capsize=3,
            color=COLORS[direction],
            label=DISPLAY_NAMES[direction],
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("practical NS-Muon trajectory step")
    axes[0].set_ylabel("squared tail-example logit drift ratio vs Fro/GD")
    axes[0].set_title("Matched-head-gain squared tail drift")
    axes[0].legend(frameon=False)

    momentum = (
        step_metrics.groupby(["seed", "trajectory_step"], observed=True)["gradient_momentum_cosine"]
        .first()
        .reset_index()
    )
    for seed, group in momentum.groupby("seed", observed=True, sort=False):
        axes[1].plot(
            group["trajectory_step"],
            group["gradient_momentum_cosine"],
            color="#777777",
            alpha=0.25,
            linewidth=1,
        )
    mean = momentum.groupby("trajectory_step", as_index=False, observed=True)["gradient_momentum_cosine"].mean()
    axes[1].plot(
        mean["trajectory_step"],
        mean["gradient_momentum_cosine"],
        color="#D55E00",
        linewidth=2.2,
        marker="o",
        label="mean",
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("practical NS-Muon trajectory step")
    axes[1].set_ylabel(r"cosine($G_t$, $M_t$)")
    axes[1].set_title("Momentum-gradient alignment")
    axes[1].legend(frameon=False)

    fig.suptitle("Long-tailed digits practical-Muon trajectory compatibility")
    fig.tight_layout()
    path = FIGURE_DIR / "long_tail_practical_muon_bridge.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(
    config: LongTailPracticalMuonBridgeConfig,
    summary: pd.DataFrame,
    step_summary: pd.DataFrame,
    figure_path: Path,
) -> None:
    by_direction = summary.set_index("direction")
    polar_momentum = by_direction.loc["polar_momentum"]
    ns_momentum = by_direction.loc["ns_momentum"]

    table = [
        "| direction | comparisons | squared drift ratio vs Fro/GD | cosine to polar(G_t) | gradient-momentum cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction in ["polar_grad", "polar_momentum", "ns_momentum"]:
        row = by_direction.loc[direction]
        table.append(
            "| "
            + DISPLAY_NAMES[direction]
            + " | "
            + f"{int(row['comparisons'])}"
            + " | "
            + f"{_fmt(row['geomean_tail_output_drift_sq_ratio_vs_fro'])} "
            + f"[{_fmt(row['tail_output_drift_sq_ratio_vs_fro_ci95_low'])}, "
            + f"{_fmt(row['tail_output_drift_sq_ratio_vs_fro_ci95_high'])}]"
            + " | "
            + f"{_fmt(row['mean_direction_cosine_to_polar_grad'])} "
            + f"[{_fmt(row['direction_cosine_to_polar_grad_ci95_low'])}, "
            + f"{_fmt(row['direction_cosine_to_polar_grad_ci95_high'])}]"
            + " | "
            + f"{_fmt(row['mean_gradient_momentum_cosine'])} "
            + f"[{_fmt(row['gradient_momentum_cosine_ci95_low'])}, "
            + f"{_fmt(row['gradient_momentum_cosine_ci95_high'])}]"
            + " |"
        )

    text = f"""# E11 Long-Tailed Practical-Muon Trajectory Compatibility

This diagnostic extends the fixed-checkpoint Muon-style compatibility check along a short practical
NS-Muon-style head-only trajectory. Starting from the same long-tailed digits
checkpoint, each seed runs `{config.trajectory_steps}` practical updates using a
momentum state, finite Newton-Schulz polar approximation, and learning rate
`{config.trajectory_lr}`. At every trajectory state, the script re-evaluates the
matched-head-gain diagnostic for Fro/GD, `polar(G_t)`, `polar(M_t)`, and
`NS(M_t)`.

## Summary

{chr(10).join(table)}

## Interpretation

Across `{int(polar_momentum['comparisons'])}` paired state-step comparisons,
`polar(M_t)` has squared tail-example logit drift ratio
`{_fmt(polar_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])}`
`[{_fmt(polar_momentum['tail_output_drift_sq_ratio_vs_fro_ci95_low'])}, {_fmt(polar_momentum['tail_output_drift_sq_ratio_vs_fro_ci95_high'])}]`
relative to Fro/GD. The practical finite-step `NS(M_t)` direction has squared drift ratio
`{_fmt(ns_momentum['geomean_tail_output_drift_sq_ratio_vs_fro'])}`
`[{_fmt(ns_momentum['tail_output_drift_sq_ratio_vs_fro_ci95_low'])}, {_fmt(ns_momentum['tail_output_drift_sq_ratio_vs_fro_ci95_high'])}]`.

This extends the fixed-checkpoint compatibility check by testing sampled states
along a short practical Muon-like trajectory. It remains a local
matched-head-gain diagnostic: it does not prove final tail accuracy or broad
optimizer superiority.

Figure: [{figure_path.as_posix()}](../{figure_path.as_posix()})

Artifacts:

- [step_metrics.csv](../results/e11_long_tail_practical_muon_bridge/step_metrics.csv)
- [step_summary.csv](../results/e11_long_tail_practical_muon_bridge/step_summary.csv)
- [summary.csv](../results/e11_long_tail_practical_muon_bridge/summary.csv)
- [config.json](../results/e11_long_tail_practical_muon_bridge/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    config = LongTailPracticalMuonBridgeConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_metrics, summary = run_long_tail_practical_muon_bridge(config)
    step_summary = _per_step_summary(step_metrics)
    step_metrics.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    step_summary.to_csv(OUTPUT_DIR / "step_summary.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    figure_path = write_figure(step_metrics, step_summary)
    write_discussion(config, summary, step_summary, figure_path)
    print(f"saved long-tail practical Muon bridge results to {OUTPUT_DIR}")
    print(f"step rows={len(step_metrics)}, comparisons={summary['comparisons'].max()}, seeds={len(config.seeds)}")
    print(f"figure: {figure_path}")
    print(f"discussion: {DISCUSSION_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
