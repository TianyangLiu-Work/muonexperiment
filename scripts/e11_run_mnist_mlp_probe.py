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
from e11_condition_geometry.runner import run_equal_update_experiment
from e11_condition_geometry.statistics import first_order_calibration_summary, log_ratio_ci95, update_spectrum_summary
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, write_markdown


OUTPUT_DIR = Path("results/e11_mnist_mlp_probe")
FIGURE_DIR = Path("figures/e11_mnist_mlp_probe")
DISCUSSION_PATH = Path("discussion/e11_mnist_mlp_probe.md")


def make_specs() -> tuple[ProblemSpec, ...]:
    specs = []
    for hidden_dim in [64, 128]:
        for target in [1e-4, 3e-4, 1e-3]:
            specs.append(
                ProblemSpec(
                    family="MNISTMLP",
                    setting=f"MNIST MLP hidden={hidden_dim} target={target:.0e}",
                    steps=5,
                    input_dim=784,
                    hidden_dim=hidden_dim,
                    output_dim=10,
                    num_samples=1024,
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


def annotate(steps: pd.DataFrame) -> pd.DataFrame:
    result = steps.copy()
    result["hidden_dim"] = pd.to_numeric(result["setting"].str.extract(r"hidden=(\d+)")[0], errors="coerce")
    result["target_relative_update_norm"] = pd.to_numeric(result["setting"].str.extract(r"target=([0-9e\\-]+)")[0], errors="coerce")
    return result


def paired_frame(steps: pd.DataFrame) -> pd.DataFrame:
    source = steps[steps["delta_loss"].notna()].copy()
    pivot = source.pivot_table(
        index=["hidden_dim", "target_relative_update_norm", "seed", "step"],
        columns="algo",
        values=["delta_loss", "update_grad_inner", "update_grad_cosine", "relative_update_fro_norm"],
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
    groups: list[tuple[str, object, float, pd.DataFrame]] = []
    for (hidden_dim, target), group in paired.groupby(["hidden_dim", "target_relative_update_norm"], observed=True, sort=True):
        groups.append(("target_hidden", int(hidden_dim), float(target), group))
    for hidden_dim, group in paired.groupby("hidden_dim", observed=True, sort=True):
        groups.append(("hidden_all_targets", int(hidden_dim), np.nan, group))
    groups.append(("all", "All", np.nan, paired))
    for group_type, hidden_dim, target, group in groups:
        for metric in ["delta_loss", "update_grad_inner", "update_grad_cosine"]:
            ratios = group[f"{metric}_ratio"]
            ratio, lo, hi = log_ratio_ci95(ratios)
            records.append(
                {
                    "group_type": group_type,
                    "hidden_dim": hidden_dim,
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
    rows = summary[(summary["group_type"] == "target_hidden") & (summary["metric"] == "update_grad_inner")].copy()
    path = FIGURE_DIR / "mnist_mlp_first_order_ratios.png"
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for hidden_dim, group in rows.groupby("hidden_dim", sort=True):
        group = group.sort_values("target_relative_update_norm")
        x = group["target_relative_update_norm"].to_numpy(dtype=float)
        y = group["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
        lo = group["ratio_ci95_low"].to_numpy(dtype=float)
        hi = group["ratio_ci95_high"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3, label=f"hidden={int(float(hidden_dim))}")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("target relative update norm")
    ax.set_ylabel("Muon / Adam first-order ratio")
    ax.set_title("MNIST MLP matched-update one-step progress")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(summary: pd.DataFrame, calibration: pd.DataFrame, spectrum: pd.DataFrame, figure: Path) -> None:
    core = summary[(summary["group_type"] == "hidden_all_targets") & (summary["metric"] == "update_grad_inner")].copy()
    calibration_all = calibration[calibration["group"] == "All"].iloc[0]
    spectrum_core = spectrum[(spectrum["problem_family"] == "MNISTMLP") & (spectrum["metric"].isin(["nrUpdate", "stUpdate"]))]
    text = f"""# E11 MNIST MLP Probe

This generated probe adds an MNIST neural sanity check beyond the sklearn digits MLP while keeping the experiment small and matched-update controlled.

## Setup

- Dataset: torchvision MNIST train split, deterministic subset of 1024 examples.
- Model: two-layer MLP with hidden widths 64 and 128.
- Optimizers: Adam and ExactMuon.
- Control: equal global relative update norm at each matched step.
- Horizon: 5 steps, 3 seeds, target relative update norms `1e-4`, `3e-4`, and `1e-3`.

## Main Result

![MNIST MLP first-order ratios](../{figure})

{markdown_table(core, ["hidden_dim", "n_pairs", "muon_win_rate", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

## Calibration And Update Spectrum

First-order calibration on this probe has Spearman `{fmt(calibration_all['spearman_delta_vs_first_order'])}` and within-factor-2 `{fmt(calibration_all['within_factor_2'])}`.

{markdown_table(spectrum_core, ["metric", "problem_family", "n_pairs", "muon_higher_pairs", "geomean_ratio_muon_over_adam", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one"])}

## Interpretation

This probe is closer to a real neural benchmark than sklearn digits, but it is still intentionally small. It should be used as a sanity check for the paper story, not as a final neural-network benchmark. If the update-spectrum claim remains strong while first-order advantage is conditional or weak, that supports the main framing: Muon robustly shapes update spectra, but performance impact depends on local geometry.

## Sources

- [MNIST probe pair summary](../results/e11_mnist_mlp_probe/pair_summary.csv)
- [MNIST probe step metrics](../results/e11_mnist_mlp_probe/step_metrics.csv)
- [MNIST probe figure](../{figure})
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
    print(f"saved MNIST MLP probe to {OUTPUT_DIR}")
    print(f"runs={steps['run_id'].nunique()}, step rows={len(steps)}")


if __name__ == "__main__":
    main()
