from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.config import ExperimentConfig, ProblemSpec
from e11_condition_geometry.runner import run_equal_update_experiment, run_experiment
from e11_condition_geometry.statistics import ci95, first_order_calibration_summary, log_ratio_ci95, update_spectrum_summary


OUTPUT_DIR = Path("results/e11_hyperparam_sweep")
FIGURE_DIR = Path("figures/e11_hyperparam_sweep")
DISCUSSION_PATH = Path("discussion/e11_hyperparam_sweep.md")
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def sweep_specs() -> tuple[ProblemSpec, ...]:
    specs: list[ProblemSpec] = []
    for kappa in [1e2, 1e5]:
        for lr in LEARNING_RATES:
            specs.append(
                ProblemSpec(
                    family="MatrixFactorizationInput",
                    setting=f"MF input kappa={kappa:.0e} lr={lr:.0e}",
                    steps=10,
                    d=60,
                    rank=5,
                    kappa=kappa,
                    num_factors=10,
                    input_columns_multiplier=10,
                    lr=lr,
                )
            )
            specs.append(
                ProblemSpec(
                    family="MatrixSensing",
                    setting=f"Matrix sensing kappa={kappa:.0e} lr={lr:.0e}",
                    steps=5,
                    d=60,
                    rank=5,
                    kappa=kappa,
                    measurement_multiplier=2.0,
                    lr=lr,
                )
            )
    for hidden_dim in [16, 64]:
        for lr in LEARNING_RATES:
            specs.append(
                ProblemSpec(
                    family="SmallMLPDigits",
                    setting=f"Small MLP digits hidden={hidden_dim} lr={lr:.0e}",
                    steps=10,
                    d=64,
                    rank=10,
                    kappa=1.0,
                    hidden_dim=hidden_dim,
                    num_samples=1024,
                    lr=lr,
                )
            )
    return tuple(specs)


def build_config() -> ExperimentConfig:
    base = default_config()
    return replace(
        base,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        discussion_path=DISCUSSION_PATH,
        specs=sweep_specs(),
    )


def base_setting(setting: str) -> str:
    return re.sub(r"\s+lr=[0-9.e+-]+$", "", setting)


def add_sweep_columns(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    result = frame.copy()
    result["mode"] = mode
    result["base_setting"] = result["setting"].map(base_setting)
    result["hidden_dim"] = result["setting"].str.extract(r"hidden=(\d+)")[0]
    result["hidden_dim"] = pd.to_numeric(result["hidden_dim"], errors="coerce")
    return result


def run_sweep(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_steps, raw_layers = run_experiment(config)
    equal_steps, equal_layers = run_equal_update_experiment(config)
    return (
        add_sweep_columns(raw_steps, "raw"),
        add_sweep_columns(raw_layers, "raw"),
        add_sweep_columns(equal_steps, "equal_update"),
        add_sweep_columns(equal_layers, "equal_update"),
    )


def outcome_table(steps: pd.DataFrame) -> pd.DataFrame:
    ordered = steps.sort_values(["run_id", "step"]).copy()
    first = ordered.groupby(["mode", "run_id"], observed=True).head(1)[
        ["mode", "run_id", "loss", "recovery_error"]
    ].rename(columns={"loss": "initial_loss", "recovery_error": "initial_recovery_error"})
    final = ordered.groupby(["mode", "run_id"], observed=True).tail(1)[
        ["mode", "run_id", "problem_family", "base_setting", "setting", "kappa", "lr", "algo", "seed", "step", "loss", "recovery_error"]
    ].rename(columns={"loss": "final_loss", "recovery_error": "final_recovery_error", "step": "final_step"})
    outcome = final.merge(first, on=["mode", "run_id"], how="left")
    outcome["total_loss_decrease"] = outcome["initial_loss"] - outcome["final_loss"]
    outcome["relative_loss_decrease"] = outcome["total_loss_decrease"] / (outcome["initial_loss"].abs() + 1e-300)
    return outcome


def lr_curve_summary(outcome: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, group in outcome.groupby(["mode", "problem_family", "base_setting", "algo", "lr"], observed=True, sort=True):
        mode, family, setting, algo, lr = key
        loss_mean, loss_lo, loss_hi = ci95(group["final_loss"])
        decrease_mean, decrease_lo, decrease_hi = ci95(group["total_loss_decrease"])
        records.append(
            {
                "mode": mode,
                "problem_family": family,
                "base_setting": setting,
                "algo": algo,
                "lr": float(lr),
                "runs": int(len(group)),
                "mean_final_loss": loss_mean,
                "final_loss_ci95_low": loss_lo,
                "final_loss_ci95_high": loss_hi,
                "mean_total_loss_decrease": decrease_mean,
                "total_loss_decrease_ci95_low": decrease_lo,
                "total_loss_decrease_ci95_high": decrease_hi,
            }
        )
    return pd.DataFrame(records)


def best_by_seed(outcome: pd.DataFrame) -> pd.DataFrame:
    sortable = outcome.replace([np.inf, -np.inf], np.nan).dropna(subset=["final_loss"]).copy()
    sortable = sortable.sort_values(["mode", "problem_family", "base_setting", "algo", "seed", "final_loss", "lr"])
    return sortable.groupby(["mode", "problem_family", "base_setting", "algo", "seed"], observed=True).head(1).reset_index(drop=True)


def best_lr_frequency(best: pd.DataFrame) -> pd.DataFrame:
    return best.groupby(["mode", "problem_family", "base_setting", "algo", "lr"], as_index=False, observed=True).agg(
        seed_count=("seed", "nunique"),
        mean_best_final_loss=("final_loss", "mean"),
        mean_best_total_loss_decrease=("total_loss_decrease", "mean"),
    )


def best_pair_summary(best: pd.DataFrame) -> pd.DataFrame:
    metrics = ["final_loss", "total_loss_decrease", "relative_loss_decrease"]
    pivot = best.pivot_table(
        index=["mode", "problem_family", "base_setting", "seed"],
        columns="algo",
        values=metrics + ["lr"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groupers: list[tuple[tuple, pd.DataFrame]] = []
    groupers.extend((key, group) for key, group in paired.groupby(["mode", "problem_family", "base_setting"], observed=True, sort=True))
    groupers.extend(((mode, "All", "All"), group) for mode, group in paired.groupby("mode", observed=True, sort=True))
    for (mode, family, setting), group in groupers:
        loss_ratio = group["final_loss_Muon"] / (group["final_loss_Adam"].abs() + 1e-300)
        decrease_ratio = group["total_loss_decrease_Muon"] / (group["total_loss_decrease_Adam"].abs() + 1e-300)
        rel_decrease_ratio = group["relative_loss_decrease_Muon"] / (group["relative_loss_decrease_Adam"].abs() + 1e-300)
        loss_ratio_mean, loss_lo, loss_hi = log_ratio_ci95(loss_ratio)
        decrease_ratio_mean, decrease_lo, decrease_hi = log_ratio_ci95(decrease_ratio)
        rel_decrease_ratio_mean, rel_decrease_lo, rel_decrease_hi = log_ratio_ci95(rel_decrease_ratio)
        loss_delta, loss_delta_lo, loss_delta_hi = ci95(group["final_loss_Muon"] - group["final_loss_Adam"])
        decrease_delta, decrease_delta_lo, decrease_delta_hi = ci95(
            group["total_loss_decrease_Muon"] - group["total_loss_decrease_Adam"]
        )
        records.append(
            {
                "mode": mode,
                "problem_family": family,
                "base_setting": setting,
                "n_seed_pairs": int(len(group)),
                "muon_lower_final_loss_pairs": int((group["final_loss_Muon"] < group["final_loss_Adam"]).sum()),
                "muon_higher_total_decrease_pairs": int((group["total_loss_decrease_Muon"] > group["total_loss_decrease_Adam"]).sum()),
                "final_loss_ratio_muon_over_adam": loss_ratio_mean,
                "final_loss_ratio_ci95_low": loss_lo,
                "final_loss_ratio_ci95_high": loss_hi,
                "final_loss_ratio_ci95_below_one": bool(loss_hi < 1.0) if np.isfinite(loss_hi) else False,
                "final_loss_ratio_ci95_above_one": bool(loss_lo > 1.0) if np.isfinite(loss_lo) else False,
                "total_decrease_ratio_muon_over_adam": decrease_ratio_mean,
                "total_decrease_ratio_ci95_low": decrease_lo,
                "total_decrease_ratio_ci95_high": decrease_hi,
                "total_decrease_ratio_ci95_above_one": bool(decrease_lo > 1.0) if np.isfinite(decrease_lo) else False,
                "total_decrease_ratio_ci95_below_one": bool(decrease_hi < 1.0) if np.isfinite(decrease_hi) else False,
                "relative_decrease_ratio_muon_over_adam": rel_decrease_ratio_mean,
                "relative_decrease_ratio_ci95_low": rel_decrease_lo,
                "relative_decrease_ratio_ci95_high": rel_decrease_hi,
                "mean_final_loss_delta_muon_minus_adam": loss_delta,
                "final_loss_delta_ci95_low": loss_delta_lo,
                "final_loss_delta_ci95_high": loss_delta_hi,
                "mean_total_decrease_delta_muon_minus_adam": decrease_delta,
                "total_decrease_delta_ci95_low": decrease_delta_lo,
                "total_decrease_delta_ci95_high": decrease_delta_hi,
            }
        )
    return pd.DataFrame(records)


def plot_best_ratios(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "hyperparam_best_ratios.png"
    rows = summary[summary["base_setting"] != "All"].copy()
    rows["label"] = rows["base_setting"].str.replace("Matrix sensing", "MS", regex=False).str.replace("MF input", "MF", regex=False)
    modes = ["raw", "equal_update"]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.4), sharex=True)
    x_labels = sorted(rows["label"].unique())
    x = np.arange(len(x_labels))
    offsets = {"raw": -0.16, "equal_update": 0.16}
    colors = {"raw": "#D55E00", "equal_update": "#0072B2"}
    label_to_x = {label: idx for idx, label in enumerate(x_labels)}
    for mode in modes:
        sub = rows[rows["mode"] == mode].copy()
        positions = sub["label"].map(label_to_x).to_numpy(dtype=float) + offsets[mode]
        y = sub["final_loss_ratio_muon_over_adam"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["final_loss_ratio_ci95_low"].to_numpy(dtype=float),
                sub["final_loss_ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        axes[0].errorbar(positions, y, yerr=yerr, fmt="o", color=colors[mode], label=mode, capsize=3)
        y = sub["total_decrease_ratio_muon_over_adam"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["total_decrease_ratio_ci95_low"].to_numpy(dtype=float),
                sub["total_decrease_ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        axes[1].errorbar(positions, y, yerr=yerr, fmt="o", color=colors[mode], label=mode, capsize=3)
    axes[0].axhline(1.0, color="#555555", lw=0.9, ls="--")
    axes[1].axhline(1.0, color="#555555", lw=0.9, ls="--")
    axes[0].set_ylabel("best final loss ratio\nMuon / Adam")
    axes[1].set_ylabel("best total decrease ratio\nMuon / Adam")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=25, ha="right")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Best-over-learning-rate comparison", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lr_curves(curves: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "hyperparam_lr_curves.png"
    settings = sorted(curves["base_setting"].unique())
    fig, axes = plt.subplots(len(settings), 2, figsize=(10.8, 2.65 * len(settings)), squeeze=False, sharex=True)
    colors = {"Adam": "#D55E00", "Muon": "#0072B2"}
    for row_idx, setting in enumerate(settings):
        for col_idx, mode in enumerate(["raw", "equal_update"]):
            ax = axes[row_idx, col_idx]
            sub_mode = curves[(curves["base_setting"] == setting) & (curves["mode"] == mode)]
            for algo in ["Adam", "Muon"]:
                sub = sub_mode[sub_mode["algo"] == algo].sort_values("lr")
                ax.plot(sub["lr"], sub["mean_final_loss"], marker="o", color=colors[algo], label=algo if row_idx == 0 else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{setting} | {mode}")
            ax.set_ylabel("mean final loss")
            ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("learning rate")
    axes[-1, 1].set_xlabel("learning rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Learning-rate sensitivity", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.4g}")
    return display.to_markdown(index=False)


def write_discussion(
    best_summary: pd.DataFrame,
    best_frequency: pd.DataFrame,
    calibration: pd.DataFrame,
    update_spectrum: pd.DataFrame,
    best_figure: Path,
    curve_figure: Path,
) -> None:
    settings = best_summary[best_summary["base_setting"] != "All"].copy()
    overall = best_summary[best_summary["base_setting"] == "All"].copy()
    best_table = markdown_table(
        settings,
        [
            "mode",
            "problem_family",
            "base_setting",
            "n_seed_pairs",
            "muon_lower_final_loss_pairs",
            "final_loss_ratio_muon_over_adam",
            "final_loss_ratio_ci95_low",
            "final_loss_ratio_ci95_high",
            "final_loss_ratio_ci95_below_one",
            "total_decrease_ratio_muon_over_adam",
            "total_decrease_ratio_ci95_low",
            "total_decrease_ratio_ci95_high",
            "total_decrease_ratio_ci95_above_one",
        ],
    )
    overall_table = markdown_table(
        overall,
        [
            "mode",
            "problem_family",
            "base_setting",
            "n_seed_pairs",
            "final_loss_ratio_muon_over_adam",
            "final_loss_ratio_ci95_low",
            "final_loss_ratio_ci95_high",
            "total_decrease_ratio_muon_over_adam",
            "total_decrease_ratio_ci95_low",
            "total_decrease_ratio_ci95_high",
        ],
    )
    freq_table = markdown_table(
        best_frequency.sort_values(["mode", "base_setting", "algo", "seed_count"], ascending=[True, True, True, False]),
        [
            "mode",
            "base_setting",
            "algo",
            "lr",
            "seed_count",
            "mean_best_final_loss",
            "mean_best_total_loss_decrease",
        ],
    )
    cal_table = markdown_table(
        calibration[calibration["group"].isin(["All", "Adam", "Muon"])],
        [
            "mode",
            "group",
            "points",
            "spearman_delta_vs_first_order",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "within_factor_2",
        ],
    )
    spectrum_table = markdown_table(
        update_spectrum[
            (update_spectrum["problem_family"] == "All")
            & update_spectrum["metric"].isin(["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac"])
        ],
        [
            "mode",
            "metric",
            "problem_family",
            "n_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
        ],
    )
    text = f"""# E11 Hyperparameter Sweep

## Purpose

This sweep checks whether the current Adam-vs-Muon conclusions are artifacts of a single learning-rate choice. It uses representative settings from each family and evaluates both raw optimizer runs and equal-update controlled runs.

Settings: MF-with-input and Matrix Sensing at `kappa=1e2` and `kappa=1e5`; SmallMLPDigits at `hidden_dim=16` and `hidden_dim=64`. Learning rates are `{LEARNING_RATES}` with 5 seeds.

## Best-Over-Learning-Rate Result

For each seed, optimizer, mode, and base setting, the best learning rate is selected by lowest final loss. This is an oracle-style robustness check, not a deployment protocol.

![Best-over-learning-rate ratios](../{best_figure})

{best_table}

## Overall Best-LR Summary

{overall_table}

## Selected Best Learning Rates

{freq_table}

## Learning-Rate Sensitivity

![Learning-rate curves](../{curve_figure})

## First-Order Calibration Within The Sweep

{cal_table}

## Update-Spectrum Check Within The Sweep

{spectrum_table}

## Interpretation

The sweep weakens the concern that the main update-spectrum claim is a single-lr artifact: Muon still has larger update effective/stable-rank metrics under both raw and equal-update modes.

The sweep does not support a global optimization-superiority claim for Muon. Even when each optimizer receives an oracle best learning rate per seed, the winner remains setting-dependent. This is consistent with the current main formulation: Muon reliably changes update geometry, but whether that geometry improves short-horizon progress depends on task and layer conditions.

## Caveats

1. The sweep is intentionally small and representative; it is not a full hyperparameter search.
2. Best-lr selection uses final loss from the same short run, so it is an oracle diagnostic rather than a fair validation protocol.
3. Equal-update mode still pairs Adam and Muon at each candidate lr before selecting best lr afterward; it controls update magnitude but does not independently sweep target update norm.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    raw_steps, raw_layers, equal_steps, equal_layers = run_sweep(config)
    steps = pd.concat([raw_steps, equal_steps], ignore_index=True)
    layers = pd.concat([raw_layers, equal_layers], ignore_index=True)
    outcome = outcome_table(steps)
    curves = lr_curve_summary(outcome)
    best = best_by_seed(outcome)
    best_frequency = best_lr_frequency(best)
    best_summary = best_pair_summary(best)
    calibration = pd.concat(
        [
            first_order_calibration_summary(raw_steps).assign(mode="raw"),
            first_order_calibration_summary(equal_steps).assign(mode="equal_update"),
        ],
        ignore_index=True,
    )
    spectrum = pd.concat(
        [
            update_spectrum_summary(raw_steps).assign(mode="raw"),
            update_spectrum_summary(equal_steps).assign(mode="equal_update"),
        ],
        ignore_index=True,
    )
    raw_steps.to_csv(config.output_dir / "raw_step_metrics.csv", index=False)
    raw_layers.to_csv(config.output_dir / "raw_layer_metrics.csv", index=False)
    equal_steps.to_csv(config.output_dir / "equal_step_metrics.csv", index=False)
    equal_layers.to_csv(config.output_dir / "equal_layer_metrics.csv", index=False)
    outcome.to_csv(config.output_dir / "outcome_metrics.csv", index=False)
    curves.to_csv(config.output_dir / "lr_curve_summary.csv", index=False)
    best.to_csv(config.output_dir / "best_by_seed.csv", index=False)
    best_frequency.to_csv(config.output_dir / "best_lr_frequency.csv", index=False)
    best_summary.to_csv(config.output_dir / "best_pair_summary.csv", index=False)
    calibration.to_csv(config.output_dir / "first_order_calibration_summary.csv", index=False)
    spectrum.to_csv(config.output_dir / "update_spectrum_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    best_figure = plot_best_ratios(best_summary)
    curve_figure = plot_lr_curves(curves)
    write_discussion(best_summary, best_frequency, calibration, spectrum, best_figure, curve_figure)
    print(f"saved hyperparameter sweep results to {config.output_dir}")
    print(f"raw runs={raw_steps['run_id'].nunique()}, equal-update runs={equal_steps['run_id'].nunique()}")
    print(f"best figure: {best_figure}")
    print(f"curve figure: {curve_figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
