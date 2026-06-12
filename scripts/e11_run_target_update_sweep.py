from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.config import ExperimentConfig, ProblemSpec
from e11_condition_geometry.runner import (
    append_step_diagnostics,
    apply_update_diagnostics,
    build_optimizer,
    build_problem,
    dtype_from_name,
    rescale_update,
    snapshot_parameters,
)
from e11_condition_geometry.statistics import ci95, first_order_calibration_summary, log_ratio_ci95, update_spectrum_summary


OUTPUT_DIR = Path("results/e11_target_update_sweep")
FIGURE_DIR = Path("figures/e11_target_update_sweep")
DISCUSSION_PATH = Path("discussion/e11_target_update_sweep.md")


def target_grid(family: str) -> tuple[float, ...]:
    if family == "MatrixFactorizationInput":
        return (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    return (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)


def target_specs() -> tuple[ProblemSpec, ...]:
    base_specs: list[ProblemSpec] = []
    for kappa in [1e2, 1e5]:
        base_specs.append(
            ProblemSpec(
                family="MatrixFactorizationInput",
                setting=f"MF input kappa={kappa:.0e}",
                steps=10,
                d=60,
                rank=5,
                kappa=kappa,
                num_factors=10,
                input_columns_multiplier=10,
                lr=1e-2,
            )
        )
        base_specs.append(
            ProblemSpec(
                family="MatrixSensing",
                setting=f"Matrix sensing kappa={kappa:.0e}",
                steps=5,
                d=60,
                rank=5,
                kappa=kappa,
                measurement_multiplier=2.0,
                lr=1e-2,
            )
        )
    for hidden_dim in [16, 64]:
        base_specs.append(
            ProblemSpec(
                family="SmallMLPDigits",
                setting=f"Small MLP digits hidden={hidden_dim}",
                steps=10,
                d=64,
                rank=10,
                hidden_dim=hidden_dim,
                num_samples=1024,
                lr=1e-2,
            )
        )

    specs: list[ProblemSpec] = []
    for spec in base_specs:
        for target in target_grid(spec.family):
            specs.append(replace(spec, setting=f"{spec.setting} target={target:.0e}"))
    return tuple(specs)


def build_config() -> ExperimentConfig:
    base = default_config()
    return replace(
        base,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        discussion_path=DISCUSSION_PATH,
        specs=target_specs(),
    )


def parse_target(setting: str) -> float:
    match = re.search(r"target=([0-9.e+-]+)$", setting)
    if not match:
        raise ValueError(f"missing target in setting: {setting}")
    return float(match.group(1))


def base_setting(setting: str) -> str:
    return re.sub(r"\s+target=[0-9.e+-]+$", "", setting)


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target_relative_update_norm"] = result["setting"].map(parse_target)
    result["base_setting"] = result["setting"].map(base_setting)
    result["hidden_dim"] = pd.to_numeric(result["setting"].str.extract(r"hidden=(\d+)")[0], errors="coerce")
    return result


def run_target_pair(
    spec: ProblemSpec,
    seed: int,
    start_run_id: int,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    target = parse_target(spec.setting)
    problems = {
        "Adam": build_problem(spec, seed, device=device, dtype=dtype),
        "Muon": build_problem(spec, seed, device=device, dtype=dtype),
    }
    params = {algo: problem.parameters() for algo, problem in problems.items()}
    optimizers = {algo: build_optimizer(algo, params[algo], spec.lr) for algo in ["Adam", "Muon"]}
    run_ids = {"Adam": start_run_id, "Muon": start_run_id + 1}
    step_rows: list[dict] = []
    layer_rows: list[dict] = []
    current_row_indices: dict[str, int] = {"Adam": 0, "Muon": 0}
    current_layer_starts: dict[str, int] = {"Adam": 0, "Muon": 0}
    started = time.perf_counter()

    for step in range(spec.steps + 1):
        for algo in ["Adam", "Muon"]:
            optimizers[algo].zero_grad(set_to_none=True)
            current_row_index, layer_start = append_step_diagnostics(
                problem=problems[algo],
                spec=spec,
                algo=algo,
                seed=seed,
                run_id=run_ids[algo],
                step=step,
                started=started,
                step_rows=step_rows,
                layer_rows=layer_rows,
            )
            current_row_indices[algo] = current_row_index
            current_layer_starts[algo] = layer_start

        if step < spec.steps:
            before = {algo: snapshot_parameters(params[algo]) for algo in ["Adam", "Muon"]}
            for algo in ["Adam", "Muon"]:
                optimizers[algo].step()
                rescale_update(before[algo], params[algo], target)
                post_update_loss = float(problems[algo].loss().detach().cpu())
                step_rows[current_row_indices[algo]]["delta_loss"] = (
                    step_rows[current_row_indices[algo]]["loss"] - post_update_loss
                )
                apply_update_diagnostics(
                    before=before[algo],
                    after=params[algo],
                    step_row=step_rows[current_row_indices[algo]],
                    layer_rows=layer_rows,
                    layer_start=current_layer_starts[algo],
                )

    return pd.DataFrame(step_rows), pd.DataFrame(layer_rows)


def run_target_sweep(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_frames = []
    layer_frames = []
    run_id = 0
    for spec in config.specs:
        for seed in config.seeds:
            step_frame, layer_frame = run_target_pair(spec, seed, run_id, config)
            step_frames.append(step_frame)
            layer_frames.append(layer_frame)
            run_id += 2
    return pd.concat(step_frames, ignore_index=True), pd.concat(layer_frames, ignore_index=True)


def target_pair_frame(steps: pd.DataFrame) -> pd.DataFrame:
    source = annotate(steps[np.isfinite(steps["delta_loss"])].copy())
    metrics = ["delta_loss", "update_grad_inner", "update_grad_cosine", "loss", "recovery_error"]
    pivot = source.pivot_table(
        index=["problem_family", "base_setting", "setting", "target_relative_update_norm", "seed", "step"],
        columns="algo",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    for metric in ["delta_loss", "update_grad_inner", "update_grad_cosine"]:
        paired[f"{metric}_ratio"] = paired[f"{metric}_Muon"] / (paired[f"{metric}_Adam"].abs() + 1e-300)
        paired[f"{metric}_delta"] = paired[f"{metric}_Muon"] - paired[f"{metric}_Adam"]
        paired[f"{metric}_muon_higher"] = paired[f"{metric}_delta"] > 0
    return paired


def target_pair_summary(steps: pd.DataFrame) -> pd.DataFrame:
    paired = target_pair_frame(steps)
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("target_setting", key, group)
        for key, group in paired.groupby(
            ["problem_family", "base_setting", "target_relative_update_norm"], observed=True, sort=True
        )
    )
    groups.extend(
        ("setting_all_targets", (family, setting, np.nan), group)
        for (family, setting), group in paired.groupby(["problem_family", "base_setting"], observed=True, sort=True)
    )
    groups.extend(("all", ("All", "All", np.nan), group) for _, group in paired.groupby(lambda _: "all"))
    for group_type, key, group in groups:
        family, setting, target = key
        for metric in ["update_grad_inner", "delta_loss", "update_grad_cosine"]:
            ratio, lo, hi = log_ratio_ci95(group[f"{metric}_ratio"])
            delta, delta_lo, delta_hi = ci95(group[f"{metric}_delta"])
            records.append(
                {
                    "group_type": group_type,
                    "problem_family": family,
                    "base_setting": setting,
                    "target_relative_update_norm": target,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "muon_higher_pairs": int(group[f"{metric}_muon_higher"].sum()),
                    "muon_win_rate": float(group[f"{metric}_muon_higher"].mean()),
                    "geomean_ratio_muon_over_adam": ratio,
                    "ratio_ci95_low": lo,
                    "ratio_ci95_high": hi,
                    "ratio_ci95_above_one": bool(lo > 1.0) if np.isfinite(lo) else False,
                    "ratio_ci95_below_one": bool(hi < 1.0) if np.isfinite(hi) else False,
                    "mean_delta_muon_minus_adam": delta,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                }
            )
    return pd.DataFrame(records)


def final_outcomes(steps: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate(steps)
    ordered = annotated.sort_values(["run_id", "step"])
    first = ordered.groupby("run_id", observed=True).head(1)[["run_id", "loss"]].rename(columns={"loss": "initial_loss"})
    final = ordered.groupby("run_id", observed=True).tail(1)[
        [
            "run_id",
            "problem_family",
            "base_setting",
            "setting",
            "target_relative_update_norm",
            "algo",
            "seed",
            "step",
            "loss",
            "recovery_error",
        ]
    ].rename(columns={"loss": "final_loss", "step": "final_step"})
    result = final.merge(first, on="run_id", how="left")
    result["total_loss_decrease"] = result["initial_loss"] - result["final_loss"]
    return result


def best_target_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    best = outcomes.sort_values(["problem_family", "base_setting", "algo", "seed", "final_loss"]).groupby(
        ["problem_family", "base_setting", "algo", "seed"], observed=True
    ).head(1)
    pivot = best.pivot_table(
        index=["problem_family", "base_setting", "seed"],
        columns="algo",
        values=["final_loss", "total_loss_decrease", "target_relative_update_norm"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups = list(paired.groupby(["problem_family", "base_setting"], observed=True, sort=True))
    groups.append((("All", "All"), paired))
    for (family, setting), group in groups:
        loss_ratio, loss_lo, loss_hi = log_ratio_ci95(group["final_loss_Muon"] / (group["final_loss_Adam"].abs() + 1e-300))
        decrease_ratio, decrease_lo, decrease_hi = log_ratio_ci95(
            group["total_loss_decrease_Muon"] / (group["total_loss_decrease_Adam"].abs() + 1e-300)
        )
        records.append(
            {
                "problem_family": family,
                "base_setting": setting,
                "n_seed_pairs": int(len(group)),
                "muon_lower_final_loss_pairs": int((group["final_loss_Muon"] < group["final_loss_Adam"]).sum()),
                "final_loss_ratio_muon_over_adam": loss_ratio,
                "final_loss_ratio_ci95_low": loss_lo,
                "final_loss_ratio_ci95_high": loss_hi,
                "final_loss_ratio_ci95_below_one": bool(loss_hi < 1.0) if np.isfinite(loss_hi) else False,
                "total_decrease_ratio_muon_over_adam": decrease_ratio,
                "total_decrease_ratio_ci95_low": decrease_lo,
                "total_decrease_ratio_ci95_high": decrease_hi,
                "total_decrease_ratio_ci95_above_one": bool(decrease_lo > 1.0) if np.isfinite(decrease_lo) else False,
                "mean_best_target_adam": float(group["target_relative_update_norm_Adam"].mean()),
                "mean_best_target_muon": float(group["target_relative_update_norm_Muon"].mean()),
            }
        )
    return pd.DataFrame(records)


def plot_target_ratios(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "target_update_first_order_ratios.png"
    rows = summary[
        (summary["group_type"] == "target_setting") & (summary["metric"].isin(["update_grad_inner", "delta_loss"]))
    ].copy()
    settings = sorted(rows["base_setting"].unique())
    fig, axes = plt.subplots(len(settings), 1, figsize=(8.0, 2.7 * len(settings)), sharex=False, squeeze=False)
    colors = {"update_grad_inner": "#0072B2", "delta_loss": "#D55E00"}
    labels = {"update_grad_inner": "first-order", "delta_loss": "delta loss"}
    for ax, setting in zip(axes.ravel(), settings):
        sub_setting = rows[rows["base_setting"] == setting]
        for metric in ["update_grad_inner", "delta_loss"]:
            sub = sub_setting[sub_setting["metric"] == metric].sort_values("target_relative_update_norm")
            x = sub["target_relative_update_norm"].to_numpy(dtype=float)
            y = sub["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
            lo = sub["ratio_ci95_low"].to_numpy(dtype=float)
            hi = sub["ratio_ci95_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", color=colors[metric], label=labels[metric])
            ax.fill_between(x, lo, hi, color=colors[metric], alpha=0.15)
        ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(setting)
        ax.set_ylabel("Muon / Adam ratio")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("target relative update Frobenius norm")
    handles, label_values = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, label_values, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Direction-only target-update sweep", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_best_target(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "target_update_best_ratios.png"
    rows = summary[summary["base_setting"] != "All"].copy()
    rows["label"] = rows["base_setting"].str.replace("Matrix sensing", "MS", regex=False).str.replace("MF input", "MF", regex=False)
    rows = rows.sort_values("label")
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    y = rows["final_loss_ratio_muon_over_adam"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - rows["final_loss_ratio_ci95_low"].to_numpy(dtype=float),
            rows["final_loss_ratio_ci95_high"].to_numpy(dtype=float) - y,
        ]
    )
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#0072B2", capsize=3)
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"], rotation=25, ha="right")
    ax.set_ylabel("best target final loss ratio\nMuon / Adam")
    ax.set_title("Best-over-target update norm")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.4g}")
    return display.to_markdown(index=False)


def write_discussion(
    pair_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    spectrum: pd.DataFrame,
    ratio_figure: Path,
    best_figure: Path,
) -> None:
    setting_all = pair_summary[
        (pair_summary["group_type"] == "setting_all_targets")
        & pair_summary["metric"].isin(["update_grad_inner", "delta_loss"])
    ].copy()
    target_table = markdown_table(
        setting_all,
        [
            "problem_family",
            "base_setting",
            "metric",
            "n_pairs",
            "muon_win_rate",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
            "ratio_ci95_below_one",
        ],
    )
    best_table = markdown_table(
        best_summary,
        [
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
            "mean_best_target_adam",
            "mean_best_target_muon",
        ],
    )
    cal_table = markdown_table(
        calibration[calibration["group"].isin(["All", "Adam", "Muon"])],
        [
            "group",
            "points",
            "spearman_delta_vs_first_order",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "within_factor_2",
        ],
    )
    spectrum_table = markdown_table(
        spectrum[
            (spectrum["problem_family"] == "All")
            & spectrum["metric"].isin(["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac"])
        ],
        [
            "metric",
            "problem_family",
            "n_pairs",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
        ],
    )
    text = f"""# E11 Target-Update-Norm Sweep

## Purpose

The equal-update experiments force Adam and Muon to have the same global update norm, but the chosen norm is inherited from each candidate learning rate. This sweep decouples update direction from update magnitude by directly setting the global relative Frobenius norm of each proposed update.

MF targets are `{target_grid("MatrixFactorizationInput")}`. Matrix Sensing and SmallMLP targets are `{target_grid("MatrixSensing")}`. Each setting uses 5 seeds and the same diagnostic pipeline as the main experiments.

## Direction-Only First-Order Ratios

Ratios above 1 mean Muon's rescaled update direction produces larger one-step progress than Adam's rescaled update direction at the same target update norm.

![Target update first-order ratios](../{ratio_figure})

{target_table}

## Best-Over-Target Result

For each seed and optimizer, the best target update norm is selected by lowest final loss. This is an oracle diagnostic for direction-plus-step-size robustness.

![Best target ratios](../{best_figure})

{best_table}

## First-Order Calibration

{cal_table}

## Update-Spectrum Check

{spectrum_table}

## Interpretation

This experiment asks a cleaner direction question than the learning-rate sweep. The main thing to inspect is whether a setting remains Muon-favorable across target norms, or only after selecting a favorable step size. If Muon wins only at particular target norms, then the update-spectrum story is still real but not sufficient: step-size scale is part of the mechanism.

The update-spectrum claim should remain true under this sweep because rescaling an update changes singular values by a scalar but does not change effective rank, stable rank, or normalized rank fractions.

## Caveats

1. This sweep controls global update norm, not per-layer update allocation.
2. The best-target rows are oracle diagnostics and should not be read as a validation protocol.
3. Very large target update norms can leave the local first-order regime, so the target-wise curves are more informative than the single best-target number.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    steps, layers = run_target_sweep(config)
    annotated_steps = annotate(steps)
    annotated_layers = annotate(layers)
    pair_summary = target_pair_summary(steps)
    outcomes = final_outcomes(steps)
    best_summary = best_target_summary(outcomes)
    calibration = first_order_calibration_summary(steps)
    spectrum = update_spectrum_summary(steps)
    annotated_steps.to_csv(config.output_dir / "step_metrics.csv", index=False)
    annotated_layers.to_csv(config.output_dir / "layer_metrics.csv", index=False)
    target_pair_frame(steps).to_csv(config.output_dir / "paired_step_metrics.csv", index=False)
    pair_summary.to_csv(config.output_dir / "target_pair_summary.csv", index=False)
    outcomes.to_csv(config.output_dir / "final_outcomes.csv", index=False)
    best_summary.to_csv(config.output_dir / "best_target_summary.csv", index=False)
    calibration.to_csv(config.output_dir / "first_order_calibration_summary.csv", index=False)
    spectrum.to_csv(config.output_dir / "update_spectrum_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    ratio_figure = plot_target_ratios(pair_summary)
    best_figure = plot_best_target(best_summary)
    write_discussion(pair_summary, best_summary, calibration, spectrum, ratio_figure, best_figure)
    print(f"saved target-update sweep results to {config.output_dir}")
    print(f"runs={steps['run_id'].nunique()}, step rows={len(steps)}, layer rows={len(layers)}")
    print(f"ratio figure: {ratio_figure}")
    print(f"best figure: {best_figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
