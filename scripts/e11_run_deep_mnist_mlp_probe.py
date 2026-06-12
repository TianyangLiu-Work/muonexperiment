from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.config import ExperimentConfig, ProblemSpec, default_config
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown
from e11_condition_geometry.runner import run_equal_update_experiment
from e11_condition_geometry.statistics import first_order_calibration_summary, log_ratio_ci95, update_spectrum_summary


OUTPUT_DIR = Path("results/e11_deep_mnist_mlp_probe")
FIGURE_DIR = Path("figures/e11_deep_mnist_mlp_probe")
DISCUSSION_PATH = Path("discussion/e11_deep_mnist_mlp_probe.md")


def make_specs() -> tuple[ProblemSpec, ...]:
    specs = []
    for hidden_dim in [64, 128]:
        for num_factors in [3, 4]:
            for target in [1e-4, 3e-4, 1e-3]:
                specs.append(
                    ProblemSpec(
                        family="DeepMNISTMLP",
                        setting=f"Deep MNIST MLP hidden={hidden_dim} factors={num_factors} target={target:.0e}",
                        steps=5,
                        input_dim=784,
                        hidden_dim=hidden_dim,
                        output_dim=10,
                        num_samples=1024,
                        num_factors=num_factors,
                        lr=1e-2,
                    )
                )
    return tuple(specs)


def config() -> ExperimentConfig:
    base = default_config()
    return replace(
        base,
        seeds=(0, 1, 2),
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        specs=make_specs(),
    )


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hidden_dim"] = pd.to_numeric(result["setting"].str.extract(r"hidden=(\d+)")[0], errors="coerce")
    result["num_factors"] = pd.to_numeric(result["setting"].str.extract(r"factors=(\d+)")[0], errors="coerce")
    result["target_relative_update_norm"] = pd.to_numeric(result["setting"].str.extract(r"target=([0-9e\\-]+)")[0], errors="coerce")
    return result


def paired_frame(steps: pd.DataFrame) -> pd.DataFrame:
    source = steps[steps["delta_loss"].notna()].copy()
    pivot = source.pivot_table(
        index=["hidden_dim", "num_factors", "target_relative_update_norm", "seed", "step"],
        columns="algo",
        values=["delta_loss", "update_grad_inner", "update_grad_cosine", "relative_update_fro_norm", "recovery_error"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    for metric in ["delta_loss", "update_grad_inner", "update_grad_cosine"]:
        paired[f"{metric}_ratio"] = paired[f"{metric}_Muon"] / paired[f"{metric}_Adam"].abs().clip(lower=1e-300)
        paired[f"{metric}_muon_higher"] = paired[f"{metric}_Muon"] > paired[f"{metric}_Adam"]
    return paired


def pair_summary(paired: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups: list[tuple[str, object, object, float, pd.DataFrame]] = []
    for (hidden_dim, num_factors, target), group in paired.groupby(
        ["hidden_dim", "num_factors", "target_relative_update_norm"], observed=True, sort=True
    ):
        groups.append(("target_arch", int(hidden_dim), int(num_factors), float(target), group))
    for (hidden_dim, num_factors), group in paired.groupby(["hidden_dim", "num_factors"], observed=True, sort=True):
        groups.append(("arch_all_targets", int(hidden_dim), int(num_factors), np.nan, group))
    for num_factors, group in paired.groupby("num_factors", observed=True, sort=True):
        groups.append(("depth_all_targets", "All", int(num_factors), np.nan, group))
    groups.append(("all", "All", "All", np.nan, paired))
    for group_type, hidden_dim, num_factors, target, group in groups:
        for metric in ["delta_loss", "update_grad_inner", "update_grad_cosine"]:
            ratio, lo, hi = log_ratio_ci95(group[f"{metric}_ratio"])
            records.append(
                {
                    "group_type": group_type,
                    "hidden_dim": hidden_dim,
                    "num_factors": num_factors,
                    "target_relative_update_norm": target,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "muon_win_rate": float(group[f"{metric}_muon_higher"].mean()),
                    "geomean_ratio_muon_over_adam": ratio,
                    "ratio_ci95_low": lo,
                    "ratio_ci95_high": hi,
                    "ratio_ci95_above_one": bool(lo > 1.0),
                    "ratio_ci95_below_one": bool(hi < 1.0),
                }
            )
    return pd.DataFrame(records)


def plot_summary(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = summary[(summary["group_type"] == "target_arch") & (summary["metric"] == "update_grad_inner")].copy()
    path = FIGURE_DIR / "deep_mnist_mlp_first_order_ratios.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, num_factors in zip(axes, [3, 4]):
        subset = rows[rows["num_factors"].astype(int) == num_factors]
        for hidden_dim, group in subset.groupby("hidden_dim", sort=True):
            group = group.sort_values("target_relative_update_norm")
            x = group["target_relative_update_norm"].to_numpy(dtype=float)
            y = group["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
            lo = group["ratio_ci95_low"].to_numpy(dtype=float)
            hi = group["ratio_ci95_high"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3, label=f"hidden={int(float(hidden_dim))}")
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"{num_factors} matrix factors")
        ax.set_xlabel("target relative update norm")
    axes[0].set_ylabel("Muon / Adam first-order ratio")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, calibration: pd.DataFrame, spectrum: pd.DataFrame, figure: Path) -> None:
    arch = summary[(summary["group_type"] == "arch_all_targets") & (summary["metric"] == "update_grad_inner")].copy()
    depth = summary[(summary["group_type"] == "depth_all_targets") & (summary["metric"] == "update_grad_inner")].copy()
    calibration_all = require_one(calibration, group="All")
    nr = require_one(spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate")
    st = require_one(spectrum, problem_family="DeepMNISTMLP", metric="stUpdate")
    text = f"""# E11 Deep MNIST MLP Probe

This generated probe extends the neural sanity check from a two-layer MNIST MLP to deeper all-matrix MLPs while preserving the same Adam-vs-Muon matched-update diagnostics.

## Setup

- Dataset: torchvision MNIST train split, deterministic subset of 1024 examples.
- Model: ReLU MLP with 3 or 4 matrix factors, hidden widths 64 and 128.
- Control: equal global relative update norm at each matched step.
- Horizon: 5 steps, 3 seeds, target relative update norms `1e-4`, `3e-4`, and `1e-3`.

## Main Result

![Deep MNIST MLP first-order ratios](../{figure})

{markdown_table(arch, ["hidden_dim", "num_factors", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

## Depth Summary

{markdown_table(depth, ["num_factors", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

## Calibration And Update Spectrum

First-order calibration on this probe has Spearman `{fmt(calibration_all['spearman_delta_vs_first_order'])}` and within-factor-2 `{fmt(calibration_all['within_factor_2'])}`.

Update-spectrum shaping remains strong: nrUpdate Muon/Adam={fmt(nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(nr)}, stUpdate={fmt(st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(st)}.

## Interpretation

This is still a short-horizon probe, but it is a stronger neural sanity check than a single hidden-layer MLP. It tests whether the update-spectrum signature survives depth while keeping the parameter tensors compatible with exact polar Muon.

## Sources

- [Deep MNIST probe pair summary](../results/e11_deep_mnist_mlp_probe/pair_summary.csv)
- [Deep MNIST probe step metrics](../results/e11_deep_mnist_mlp_probe/step_metrics.csv)
- [Deep MNIST probe figure](../{figure})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    cfg = config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    steps, layers = run_equal_update_experiment(cfg)
    steps = annotate(steps)
    layers = annotate(layers)
    paired = paired_frame(steps)
    summary = pair_summary(paired)
    calibration = first_order_calibration_summary(steps)
    spectrum = update_spectrum_summary(steps)
    steps.to_csv(OUTPUT_DIR / "step_metrics.csv", index=False)
    layers.to_csv(OUTPUT_DIR / "layer_metrics.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "paired_step_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "pair_summary.csv", index=False)
    calibration.to_csv(OUTPUT_DIR / "first_order_calibration_summary.csv", index=False)
    spectrum.to_csv(OUTPUT_DIR / "update_spectrum_summary.csv", index=False)
    figure = plot_summary(summary)
    write_discussion(summary, calibration, spectrum, figure)
    print(f"saved Deep MNIST MLP probe to {OUTPUT_DIR}")
    print(f"runs={steps['run_id'].nunique()}, step rows={len(steps)}")


if __name__ == "__main__":
    main()
