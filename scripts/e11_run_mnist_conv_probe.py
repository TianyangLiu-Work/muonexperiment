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


OUTPUT_DIR = Path("results/e11_mnist_conv_probe")
FIGURE_DIR = Path("figures/e11_mnist_conv_probe")
DISCUSSION_PATH = Path("discussion/e11_mnist_conv_probe.md")


def make_specs() -> tuple[ProblemSpec, ...]:
    specs = []
    for hidden_dim in [16, 32]:
        for kernel_size in [5, 7]:
            for lr in [3e-3, 1e-2]:
                specs.append(
                    ProblemSpec(
                        family="MNISTConvNet",
                        setting=f"MNIST conv filters={hidden_dim} kernel={kernel_size} stride=2 lr={lr:.0e}",
                        steps=5,
                        d=2,
                        rank=kernel_size,
                        hidden_dim=hidden_dim,
                        output_dim=10,
                        num_samples=512,
                        batch_size=64,
                        lr=lr,
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
    result["filters"] = pd.to_numeric(result["setting"].str.extract(r"filters=(\d+)")[0], errors="coerce")
    result["kernel_size"] = pd.to_numeric(result["setting"].str.extract(r"kernel=(\d+)")[0], errors="coerce")
    result["stride"] = pd.to_numeric(result["setting"].str.extract(r"stride=(\d+)")[0], errors="coerce")
    return result


def paired_frame(steps: pd.DataFrame) -> pd.DataFrame:
    source = steps[steps["delta_loss"].notna()].copy()
    pivot = source.pivot_table(
        index=["filters", "kernel_size", "stride", "lr", "seed", "step"],
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
    groups: list[tuple[str, object, object, object, pd.DataFrame]] = []
    for (filters, kernel_size, lr), group in paired.groupby(["filters", "kernel_size", "lr"], observed=True, sort=True):
        groups.append(("setting", int(filters), int(kernel_size), float(lr), group))
    for (filters, kernel_size), group in paired.groupby(["filters", "kernel_size"], observed=True, sort=True):
        groups.append(("arch_all_lr", int(filters), int(kernel_size), "All", group))
    for filters, group in paired.groupby("filters", observed=True, sort=True):
        groups.append(("filters_all", int(filters), "All", "All", group))
    groups.append(("all", "All", "All", "All", paired))
    for group_type, filters, kernel_size, lr, group in groups:
        for metric in ["delta_loss", "update_grad_inner", "update_grad_cosine"]:
            ratio, lo, hi = log_ratio_ci95(group[f"{metric}_ratio"])
            records.append(
                {
                    "group_type": group_type,
                    "filters": filters,
                    "kernel_size": kernel_size,
                    "lr": lr,
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
    rows = summary[(summary["group_type"] == "setting") & (summary["metric"] == "update_grad_inner")].copy()
    path = FIGURE_DIR / "mnist_conv_first_order_ratios.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, kernel_size in zip(axes, [5, 7]):
        subset = rows[rows["kernel_size"].astype(int) == kernel_size]
        for filters, group in subset.groupby("filters", sort=True):
            group = group.sort_values("lr")
            x = group["lr"].to_numpy(dtype=float)
            y = group["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
            lo = group["ratio_ci95_low"].to_numpy(dtype=float)
            hi = group["ratio_ci95_high"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3, label=f"filters={int(float(filters))}")
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"{kernel_size}x{kernel_size} kernels")
        ax.set_xlabel("base optimizer learning rate")
    axes[0].set_ylabel("Muon / Adam first-order ratio")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, calibration: pd.DataFrame, spectrum: pd.DataFrame, figure: Path) -> None:
    arch = summary[(summary["group_type"] == "arch_all_lr") & (summary["metric"] == "update_grad_inner")].copy()
    all_first = require_one(summary, group_type="all", metric="update_grad_inner")
    calibration_all = require_one(calibration, group="All")
    nr = require_one(spectrum, problem_family="MNISTConvNet", metric="nrUpdate")
    st = require_one(spectrum, problem_family="MNISTConvNet", metric="stUpdate")
    text = f"""# E11 MNIST ConvNet Probe

This generated probe adds a true Conv2d neural benchmark. It uses a true `Conv2d` kernel plus a matrix classifier. ExactMuon is applied to the conv kernel through its flattened `(out_channels, in_channels * kernel_height * kernel_width)` matrix view.

## Setup

- Dataset: torchvision MNIST train split, deterministic subset of 512 examples.
- Model: Conv2d -> ReLU -> mean pooling -> matrix classifier.
- Parameters: one 4D Conv2d kernel and one matrix classifier; spectral diagnostics and ExactMuon use the flattened conv-kernel matrix view.
- Training: mini-batch size 64.
- Diagnostics: full sampled dataset for conv-patch and classifier activation matrices.
- Control: Adam and Muon are matched to the same global relative update norm at each step.
- Horizon: 5 steps, 3 seeds, filters 16/32, conv kernel sizes 5/7, learning rates `3e-3` and `1e-2`.

## Main Result

![MNIST conv first-order ratios](../{figure})

{markdown_table(arch, ["filters", "kernel_size", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

## Quantitative Anchors

- All-setting first-order Muon/Adam ratio: {fmt(all_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(all_first)}.
- First-order calibration: Spearman `{fmt(calibration_all['spearman_delta_vs_first_order'])}`, within-factor-2 `{fmt(calibration_all['within_factor_2'])}`.
- Update-spectrum shaping: nrUpdate Muon/Adam={fmt(nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(nr)}, stUpdate={fmt(st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(st)}.

## Interpretation

This probe is a small CNN benchmark. It removes the pure-MLP concern by using a true Conv2d kernel while preserving a clear matrix-view definition for ExactMuon and spectral diagnostics. It should be read as a neural architecture sanity check for the update-spectrum claim, with broad modern-architecture performance claims still out of scope.

## Sources

- [MNIST conv pair summary](../results/e11_mnist_conv_probe/pair_summary.csv)
- [MNIST conv step metrics](../results/e11_mnist_conv_probe/step_metrics.csv)
- [MNIST conv figure](../{figure})
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
    print(f"saved MNIST ConvNet probe to {OUTPUT_DIR}")
    print(f"runs={steps['run_id'].nunique()}, step rows={len(steps)}")


if __name__ == "__main__":
    main()
