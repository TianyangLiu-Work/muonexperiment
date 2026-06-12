from __future__ import annotations

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
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown
from e11_condition_geometry.runner import (
    append_step_diagnostics,
    apply_update_diagnostics,
    build_problem,
    dtype_from_name,
    snapshot_parameters,
)
from e11_condition_geometry.stateless import candidate_directions, scaled_direction
from e11_condition_geometry.statistics import log_ratio_ci95


OUTPUT_DIR = Path("results/e11_stateless_optimizer_trajectory")
FIGURE_DIR = Path("figures/e11_stateless_optimizer_trajectory")
DISCUSSION_PATH = Path("discussion/e11_stateless_optimizer_trajectory.md")


def trajectory_specs() -> tuple[ProblemSpec, ...]:
    specs: list[ProblemSpec] = []
    for kappa in [1e2, 1e5]:
        specs.append(
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
        specs.append(
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
        specs.append(
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
    return tuple(specs)


def build_config() -> ExperimentConfig:
    base = default_config()
    return replace(
        base,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        discussion_path=DISCUSSION_PATH,
        specs=trajectory_specs(),
    )


def target_relative_update_norm(family: str) -> float:
    if family == "MatrixFactorizationInput":
        return 1e-3
    return 1e-2


def apply_descent(params: list[torch.nn.Parameter], direction: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, descent in zip(params, direction):
            param.add_(descent, alpha=-1.0)


def run_single_candidate(
    spec: ProblemSpec,
    candidate: str,
    seed: int,
    run_id: int,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    params = problem.parameters()
    step_rows: list[dict] = []
    layer_rows: list[dict] = []
    started = time.perf_counter()
    target = target_relative_update_norm(spec.family)

    for step in range(spec.steps + 1):
        for param in params:
            param.grad = None
        current_row_index, layer_start = append_step_diagnostics(
            problem=problem,
            spec=spec,
            algo=candidate,
            seed=seed,
            run_id=run_id,
            step=step,
            started=started,
            step_rows=step_rows,
            layer_rows=layer_rows,
        )
        step_rows[current_row_index]["target_relative_update_norm"] = target

        if step < spec.steps:
            before = snapshot_parameters(params)
            grads = [param.grad.detach().clone() for param in params]
            direction = candidate_directions(grads)[candidate]
            scaled = scaled_direction(direction, target_relative_update_norm=target, params=params)
            apply_descent(params, scaled)
            post_update_loss = float(problem.loss().detach().cpu())
            step_rows[current_row_index]["delta_loss"] = step_rows[current_row_index]["loss"] - post_update_loss
            apply_update_diagnostics(
                before=before,
                after=params,
                step_row=step_rows[current_row_index],
                layer_rows=layer_rows,
                layer_start=layer_start,
            )

    return pd.DataFrame(step_rows), pd.DataFrame(layer_rows)


def run_trajectories(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_frames = []
    layer_frames = []
    run_id = 0
    for spec in config.specs:
        for seed in config.seeds:
            for candidate in ["GD", "FreshAdamSign", "PolarMuon"]:
                step_frame, layer_frame = run_single_candidate(spec, candidate, seed, run_id, config)
                step_frames.append(step_frame)
                layer_frames.append(layer_frame)
                run_id += 1
    return pd.concat(step_frames, ignore_index=True), pd.concat(layer_frames, ignore_index=True)


def trajectory_outcomes(steps: pd.DataFrame) -> pd.DataFrame:
    records = []
    for run_id, group in steps.sort_values(["run_id", "step"]).groupby("run_id", observed=True):
        finite = group[np.isfinite(group["delta_loss"])].copy()
        first = group.iloc[0]
        final = group.iloc[-1]
        records.append(
            {
                "run_id": int(run_id),
                "problem_family": first["problem_family"],
                "setting": first["setting"],
                "kappa": float(first["kappa"]),
                "seed": int(first["seed"]),
                "candidate": first["algo"],
                "steps": int(final["step"]),
                "target_relative_update_norm": float(first["target_relative_update_norm"]),
                "initial_loss": float(first["loss"]),
                "final_loss": float(final["loss"]),
                "total_decrease": float(first["loss"] - final["loss"]),
                "final_recovery_error": float(final["recovery_error"]),
                "mean_delta_loss": float(finite["delta_loss"].mean()),
                "mean_update_grad_inner": float(finite["update_grad_inner"].mean()),
                "mean_update_grad_cosine": float(finite["update_grad_cosine"].mean()),
                "mean_nrUpdate": float(finite["nrUpdate"].mean()),
                "mean_stUpdate": float(finite["stUpdate"].mean()),
            }
        )
    return pd.DataFrame(records)


def paired_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_loss",
        "total_decrease",
        "mean_delta_loss",
        "mean_update_grad_inner",
        "mean_update_grad_cosine",
        "mean_nrUpdate",
        "mean_stUpdate",
    ]
    pivot = outcomes.pivot_table(
        index=["problem_family", "setting", "kappa", "seed"],
        columns="candidate",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{candidate}" for metric, candidate in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    comparisons = [("PolarMuon", "GD"), ("FreshAdamSign", "GD"), ("PolarMuon", "FreshAdamSign")]
    rows = []
    for _, row in paired.iterrows():
        base = row[["problem_family", "setting", "kappa", "seed"]].to_dict()
        for numerator, denominator in comparisons:
            rec = {**base, "comparison": f"{numerator}/{denominator}"}
            for metric in metrics:
                top = float(row[f"{metric}_{numerator}"])
                bottom = float(row[f"{metric}_{denominator}"])
                rec[f"{metric}_ratio"] = top / (abs(bottom) + 1e-300)
                rec[f"{metric}_delta"] = top - bottom
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_specs: list[tuple[str, list[str]]] = [
        ("all", ["comparison"]),
        ("family", ["problem_family", "comparison"]),
        ("setting", ["problem_family", "setting", "comparison"]),
    ]
    for group_type, columns in group_specs:
        for key, group in paired.groupby(columns, observed=True, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            values = dict(zip(columns, key))
            for metric in [
                "final_loss",
                "total_decrease",
                "mean_delta_loss",
                "mean_update_grad_inner",
                "mean_update_grad_cosine",
                "mean_nrUpdate",
                "mean_stUpdate",
            ]:
                ratio, lo, hi = log_ratio_ci95(group[f"{metric}_ratio"])
                delta = group[f"{metric}_delta"]
                records.append(
                    {
                        "group_type": group_type,
                        "problem_family": values.get("problem_family", "All"),
                        "setting": values.get("setting", "All"),
                        "comparison": values["comparison"],
                        "metric": metric,
                        "n_pairs": int(len(group)),
                        "geomean_ratio": ratio,
                        "ratio_ci95_low": lo,
                        "ratio_ci95_high": hi,
                        "ratio_ci95_above_one": bool(lo > 1.0) if np.isfinite(lo) else False,
                        "ratio_ci95_below_one": bool(hi < 1.0) if np.isfinite(hi) else False,
                        "numerator_higher_rate": float((delta > 0).mean()),
                    }
                )
    return pd.DataFrame(records)


def plot_summary(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = summary[
        (summary["group_type"] == "family")
        & (summary["comparison"].isin(["PolarMuon/GD", "FreshAdamSign/GD"]))
        & (summary["metric"].isin(["total_decrease", "mean_nrUpdate", "mean_stUpdate"]))
    ].copy()
    labels = []
    values = []
    lows = []
    highs = []
    colors = []
    palette = {"PolarMuon/GD": "#2C7BB6", "FreshAdamSign/GD": "#D7191C"}
    for _, row in source.sort_values(["metric", "problem_family", "comparison"]).iterrows():
        labels.append(f"{row['metric']}\n{row['problem_family']}\n{row['comparison']}")
        values.append(float(row["geomean_ratio"]))
        lows.append(float(row["ratio_ci95_low"]))
        highs.append(float(row["ratio_ci95_high"]))
        colors.append(palette[row["comparison"]])
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(max(10, 0.42 * len(values)), 5.5))
    ax.bar(x, values, color=colors, alpha=0.86)
    ax.errorbar(
        x,
        values,
        yerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)],
        fmt="none",
        ecolor="black",
        capsize=2,
    )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("geomean ratio")
    ax.set_title("Stateless optimizer trajectories at matched per-step update size")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "stateless_optimizer_trajectory_ratios.png", dpi=180)
    plt.close(fig)


def write_discussion(summary: pd.DataFrame) -> None:
    polar_nr = require_one(summary, group_type="all", problem_family="All", comparison="PolarMuon/GD", metric="mean_nrUpdate")
    polar_st = require_one(summary, group_type="all", problem_family="All", comparison="PolarMuon/GD", metric="mean_stUpdate")
    polar_decrease = require_one(summary, group_type="all", problem_family="All", comparison="PolarMuon/GD", metric="total_decrease")
    polar_loss = require_one(summary, group_type="all", problem_family="All", comparison="PolarMuon/GD", metric="final_loss")
    family_decrease = summary[
        (summary["group_type"] == "family") & (summary["comparison"] == "PolarMuon/GD") & (summary["metric"] == "total_decrease")
    ].copy()
    family_spectrum = summary[
        (summary["group_type"] == "family")
        & (summary["comparison"] == "PolarMuon/GD")
        & (summary["metric"].isin(["mean_nrUpdate", "mean_stUpdate"]))
    ].copy()
    text = f"""# E11 Stateless Optimizer Trajectory Ablation

This generated note runs short trajectories using only stateless directions. At every step, `GD`, `FreshAdamSign`, and `PolarMuon` use the same target relative Frobenius update norm within each problem family.

## Main Takeaway

Across these short trajectories, PolarMuon keeps the direction-level update-spectrum signature: mean nrUpdate PolarMuon/GD={fmt(polar_nr['geomean_ratio'])} CI={ratio_ci(polar_nr)}, mean stUpdate={fmt(polar_st['geomean_ratio'])} CI={ratio_ci(polar_st)}.

But this does not convert into uniformly better optimization progress: total_decrease PolarMuon/GD={fmt(polar_decrease['geomean_ratio'])} CI={ratio_ci(polar_decrease)}, and final_loss PolarMuon/GD={fmt(polar_loss['geomean_ratio'])} CI={ratio_ci(polar_loss)}.

This is the trajectory-level analogue of the one-step stateless direction ablation: **polar direction is sufficient for high-rank updates, but not sufficient for a general progress advantage under Frobenius-matched steps.**

## Polar vs GD Total Decrease By Family

{markdown_table(family_decrease, ["problem_family", "comparison", "metric", "n_pairs", "geomean_ratio", "ratio_ci95_low", "ratio_ci95_high", "numerator_higher_rate"])}

## Polar Update Spectrum By Family

{markdown_table(family_spectrum, ["problem_family", "comparison", "metric", "n_pairs", "geomean_ratio", "ratio_ci95_low", "ratio_ci95_high", "numerator_higher_rate"])}

## Evidence

- [trajectory step rows](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_step_metrics.csv)
- [trajectory outcomes](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_outcomes.csv)
- [paired trajectory rows](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_pairs.csv)
- [summary table](../results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv)
- [ratio figure](../figures/e11_stateless_optimizer_trajectory/stateless_optimizer_trajectory_ratios.png)

## Caveat

This still does not model Adam's state or a retuned full optimizer comparison. It is a controlled trajectory-level mechanism probe.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    config = build_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps, layers = run_trajectories(config)
    outcomes = trajectory_outcomes(steps)
    paired = paired_outcomes(outcomes)
    summary = summarize_pairs(paired)
    steps.to_csv(OUTPUT_DIR / "stateless_optimizer_step_metrics.csv", index=False)
    layers.to_csv(OUTPUT_DIR / "stateless_optimizer_layer_metrics.csv", index=False)
    outcomes.to_csv(OUTPUT_DIR / "stateless_optimizer_outcomes.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "stateless_optimizer_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "stateless_optimizer_summary.csv", index=False)
    plot_summary(summary)
    write_discussion(summary)
    print(f"saved stateless optimizer trajectories to {OUTPUT_DIR}")
    print(f"saved stateless optimizer trajectory discussion to {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
