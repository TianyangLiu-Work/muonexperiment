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
    snapshot_parameters,
)
from e11_condition_geometry.statistics import ci95, first_order_calibration_summary, log_ratio_ci95, update_spectrum_summary


OUTPUT_DIR = Path("results/e11_mlp_per_layer_control")
FIGURE_DIR = Path("figures/e11_mlp_per_layer_control")
DISCUSSION_PATH = Path("discussion/e11_mlp_per_layer_control.md")
TARGETS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def control_specs() -> tuple[ProblemSpec, ...]:
    specs: list[ProblemSpec] = []
    for hidden_dim in [16, 64]:
        for target in TARGETS:
            specs.append(
                ProblemSpec(
                    family="SmallMLPDigits",
                    setting=f"Small MLP per-layer hidden={hidden_dim} target={target:.0e}",
                    steps=10,
                    d=64,
                    rank=10,
                    hidden_dim=hidden_dim,
                    num_samples=1024,
                    lr=1e-2,
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
        specs=control_specs(),
    )


def parse_target(setting: str) -> float:
    match = re.search(r"target=([0-9.e+-]+)$", setting)
    if not match:
        raise ValueError(f"missing target in setting: {setting}")
    return float(match.group(1))


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hidden_dim"] = pd.to_numeric(result["setting"].str.extract(r"hidden=(\d+)")[0], errors="coerce")
    result["target_layer_relative_update_norm"] = result["setting"].map(parse_target)
    result["base_setting"] = result["setting"].str.replace(r"\s+target=[0-9.e+-]+$", "", regex=True)
    return result


def rescale_each_layer(before: list[torch.Tensor], after: list[torch.nn.Parameter], target_relative_update: float) -> None:
    with torch.no_grad():
        for old, param in zip(before, after):
            diff = param.detach() - old
            diff_norm = float(torch.linalg.norm(diff).cpu())
            if diff_norm <= 0.0:
                continue
            layer_target = target_relative_update * max(float(torch.linalg.norm(old).cpu()), 1e-300)
            param.copy_(old + (layer_target / diff_norm) * diff)


def add_layer_relative_update_columns(
    before: list[torch.Tensor],
    after: list[torch.nn.Parameter],
    layer_rows: list[dict],
    layer_start: int,
    target_relative_update: float,
) -> None:
    for offset, (old, param) in enumerate(zip(before, after)):
        diff_fro = float(torch.linalg.norm(param.detach() - old).cpu())
        param_fro = float(torch.linalg.norm(old).cpu())
        row = layer_rows[layer_start + offset]
        row["layer_update_fro_norm"] = diff_fro
        row["layer_param_fro_norm"] = param_fro
        row["layer_relative_update_norm"] = diff_fro / max(param_fro, 1e-300)
        row["target_layer_relative_update_norm"] = target_relative_update


def run_control_pair(
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
                rescale_each_layer(before[algo], params[algo], target)
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
                add_layer_relative_update_columns(
                    before=before[algo],
                    after=params[algo],
                    layer_rows=layer_rows,
                    layer_start=current_layer_starts[algo],
                    target_relative_update=target,
                )

    return pd.DataFrame(step_rows), pd.DataFrame(layer_rows)


def run_control_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_frames = []
    layer_frames = []
    run_id = 0
    for spec in config.specs:
        for seed in config.seeds:
            step_frame, layer_frame = run_control_pair(spec, seed, run_id, config)
            step_frames.append(step_frame)
            layer_frames.append(layer_frame)
            run_id += 2
    return pd.concat(step_frames, ignore_index=True), pd.concat(layer_frames, ignore_index=True)


def paired_steps(steps: pd.DataFrame) -> pd.DataFrame:
    source = annotate(steps[np.isfinite(steps["delta_loss"])].copy())
    metrics = ["delta_loss", "update_grad_inner", "update_grad_cosine", "loss", "recovery_error"]
    pivot = source.pivot_table(
        index=["hidden_dim", "target_layer_relative_update_norm", "seed", "step"],
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


def pair_summary(steps: pd.DataFrame) -> pd.DataFrame:
    paired = paired_steps(steps)
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("target_hidden", key, group)
        for key, group in paired.groupby(["hidden_dim", "target_layer_relative_update_norm"], observed=True, sort=True)
    )
    groups.extend(("hidden_all_targets", (hidden, np.nan), group) for hidden, group in paired.groupby("hidden_dim", observed=True, sort=True))
    groups.append(("all", ("All", np.nan), paired))
    for group_type, key, group in groups:
        hidden_dim, target = key
        for metric in ["update_grad_inner", "delta_loss", "update_grad_cosine"]:
            ratio, lo, hi = log_ratio_ci95(group[f"{metric}_ratio"])
            delta, delta_lo, delta_hi = ci95(group[f"{metric}_delta"])
            records.append(
                {
                    "group_type": group_type,
                    "hidden_dim": hidden_dim,
                    "target_layer_relative_update_norm": target,
                    "metric": metric,
                    "n_pairs": int(len(group)),
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
    source = annotate(steps)
    ordered = source.sort_values(["run_id", "step"])
    first = ordered.groupby("run_id", observed=True).head(1)[["run_id", "loss"]].rename(columns={"loss": "initial_loss"})
    final = ordered.groupby("run_id", observed=True).tail(1)[
        ["run_id", "hidden_dim", "target_layer_relative_update_norm", "algo", "seed", "loss", "recovery_error"]
    ].rename(columns={"loss": "final_loss", "recovery_error": "final_recovery_error"})
    result = final.merge(first, on="run_id", how="left")
    result["total_loss_decrease"] = result["initial_loss"] - result["final_loss"]
    return result


def best_target_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    best = outcomes.sort_values(["hidden_dim", "algo", "seed", "final_loss"]).groupby(
        ["hidden_dim", "algo", "seed"], observed=True
    ).head(1)
    pivot = best.pivot_table(
        index=["hidden_dim", "seed"],
        columns="algo",
        values=["final_loss", "total_loss_decrease", "target_layer_relative_update_norm"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups = list(paired.groupby("hidden_dim", observed=True, sort=True))
    groups.append(("All", paired))
    for hidden_dim, group in groups:
        loss_ratio, loss_lo, loss_hi = log_ratio_ci95(group["final_loss_Muon"] / (group["final_loss_Adam"].abs() + 1e-300))
        decrease_ratio, decrease_lo, decrease_hi = log_ratio_ci95(
            group["total_loss_decrease_Muon"] / (group["total_loss_decrease_Adam"].abs() + 1e-300)
        )
        records.append(
            {
                "hidden_dim": hidden_dim,
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
                "mean_best_target_adam": float(group["target_layer_relative_update_norm_Adam"].mean()),
                "mean_best_target_muon": float(group["target_layer_relative_update_norm_Muon"].mean()),
            }
        )
    return pd.DataFrame(records)


def plot_ratios(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "per_layer_control_first_order_ratios.png"
    rows = summary[(summary["group_type"] == "target_hidden") & summary["metric"].isin(["update_grad_inner", "delta_loss"])].copy()
    hidden_values = sorted(rows["hidden_dim"].unique())
    colors = {"update_grad_inner": "#0072B2", "delta_loss": "#D55E00"}
    labels = {"update_grad_inner": "first-order", "delta_loss": "delta loss"}
    fig, axes = plt.subplots(1, len(hidden_values), figsize=(5.2 * len(hidden_values), 4.0), sharey=True, squeeze=False)
    for ax, hidden_dim in zip(axes.ravel(), hidden_values):
        sub_hidden = rows[rows["hidden_dim"] == hidden_dim]
        for metric in ["update_grad_inner", "delta_loss"]:
            sub = sub_hidden[sub_hidden["metric"] == metric].sort_values("target_layer_relative_update_norm")
            x = sub["target_layer_relative_update_norm"].to_numpy(dtype=float)
            y = sub["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
            lo = sub["ratio_ci95_low"].to_numpy(dtype=float)
            hi = sub["ratio_ci95_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", color=colors[metric], label=labels[metric])
            ax.fill_between(x, lo, hi, color=colors[metric], alpha=0.15)
        ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(f"hidden={int(hidden_dim)}")
        ax.set_xlabel("per-layer relative update norm")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Muon / Adam ratio")
    handles, label_values = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, label_values, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("SmallMLP direction comparison with per-layer update norms fixed", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.4g}")
    return display.to_markdown(index=False)


def write_discussion(
    summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    spectrum: pd.DataFrame,
    figure: Path,
) -> None:
    hidden_all = summary[
        (summary["group_type"] == "hidden_all_targets") & summary["metric"].isin(["update_grad_inner", "delta_loss"])
    ].copy()
    target_rows = summary[(summary["group_type"] == "target_hidden") & (summary["metric"] == "update_grad_inner")].copy()
    hidden_table = markdown_table(
        hidden_all,
        [
            "hidden_dim",
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
    target_table = markdown_table(
        target_rows,
        [
            "hidden_dim",
            "target_layer_relative_update_norm",
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
            "hidden_dim",
            "n_seed_pairs",
            "muon_lower_final_loss_pairs",
            "final_loss_ratio_muon_over_adam",
            "final_loss_ratio_ci95_low",
            "final_loss_ratio_ci95_high",
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
    text = f"""# E11 SmallMLP Per-Layer Update Control

## Purpose

The target-update-norm sweep controls only the global update norm. This follow-up fixes each layer's relative update norm separately, so layer allocation is no longer available as an explanation for Adam/Muon differences.

The experiment uses SmallMLP hidden widths 16 and 64, target per-layer relative update norms `{TARGETS}`, 5 seeds, and 10 steps.

## Direction Result

Ratios above 1 mean Muon's direction gives larger progress than Adam's direction after both have the same per-layer update size.

![Per-layer control first-order ratios](../{figure})

{hidden_table}

## Target-Wise First-Order Ratios

{target_table}

## Best-Over-Target Result

{best_table}

## First-Order Calibration

{cal_table}

## Update-Spectrum Check

{spectrum_table}

## Interpretation

This is a cleaner direction-only test for the MLP width transition. The hidden-64 result remains strongly Adam-favorable even after every layer update has the same relative size, so the wide-network disadvantage is not just a layer-budget allocation artifact.

The hidden-16 result is more delicate: Muon is favorable at the smallest per-layer targets, nearly tied around the middle target, and unfavorable at larger targets. This weakens the claim that the narrow-network advantage is purely directional. A better reading is that the narrow case depends on both Muon's direction and the step-size regime, while the wide-network failure is robust to per-layer update-size control.

## Caveats

1. This only tests a two-layer SmallMLP on sklearn digits.
2. Fixing per-layer relative update norm is still an intervention, not the natural optimizer trajectory.
3. It controls update size per layer but not finer within-layer spectral allocation.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    steps, layers = run_control_experiment(config)
    steps = annotate(steps)
    layers = annotate(layers)
    paired = paired_steps(steps)
    summary = pair_summary(steps)
    outcomes = final_outcomes(steps)
    best = best_target_summary(outcomes)
    calibration = first_order_calibration_summary(steps)
    spectrum = update_spectrum_summary(steps)
    steps.to_csv(config.output_dir / "step_metrics.csv", index=False)
    layers.to_csv(config.output_dir / "layer_metrics.csv", index=False)
    paired.to_csv(config.output_dir / "paired_step_metrics.csv", index=False)
    summary.to_csv(config.output_dir / "pair_summary.csv", index=False)
    outcomes.to_csv(config.output_dir / "final_outcomes.csv", index=False)
    best.to_csv(config.output_dir / "best_target_summary.csv", index=False)
    calibration.to_csv(config.output_dir / "first_order_calibration_summary.csv", index=False)
    spectrum.to_csv(config.output_dir / "update_spectrum_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_ratios(summary)
    write_discussion(summary, best, calibration, spectrum, figure)
    print(f"saved MLP per-layer control results to {config.output_dir}")
    print(f"runs={steps['run_id'].nunique()}, step rows={len(steps)}, layer rows={len(layers)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
