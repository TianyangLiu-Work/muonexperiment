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
from e11_condition_geometry.diagnostics import derive_update_metrics, json_dumps_nested, singular_values
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown
from e11_condition_geometry.runner import build_optimizer, build_problem, dtype_from_name, snapshot_parameters
from e11_condition_geometry.statistics import log_ratio_ci95
from e11_condition_geometry.stateless import candidate_directions, global_fro_norm, scaled_direction


OUTPUT_DIR = Path("results/e11_stateless_direction_ablation")
FIGURE_DIR = Path("figures/e11_stateless_direction_ablation")
DISCUSSION_PATH = Path("discussion/e11_stateless_direction_ablation.md")


def base_specs() -> tuple[ProblemSpec, ...]:
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
        specs=base_specs(),
    )


def target_grid(family: str) -> tuple[float, ...]:
    if family == "MatrixFactorizationInput":
        return (3e-4, 1e-3, 3e-3)
    return (3e-3, 1e-2, 3e-2)


def checkpoint_plan(spec: ProblemSpec) -> tuple[tuple[str, int], ...]:
    max_step = int(spec.steps)
    steps = sorted({0, min(1, max_step), min(3, max_step), max_step})
    plan = [("Init", 0)]
    for source in ["Adam", "Muon"]:
        for step in steps:
            if step > 0:
                plan.append((source, step))
    return tuple(plan)


def advance_to_checkpoint(problem, source_algo: str, checkpoint_step: int, lr: float) -> None:
    if checkpoint_step == 0:
        problem.set_train_step(0)
        return
    optimizer = build_optimizer(source_algo, problem.parameters(), lr)
    for step in range(checkpoint_step):
        problem.set_train_step(step)
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss()
        loss.backward()
        optimizer.step()
    problem.set_train_step(checkpoint_step)


def evaluate_direction(problem, direction: list[torch.Tensor], old_loss: float) -> dict[str, float | str]:
    params = problem.parameters()
    before = snapshot_parameters(params)
    grads = [param.grad.detach().clone() for param in params]
    with torch.no_grad():
        for old, param, descent in zip(before, params, direction):
            param.copy_(old - descent)
    new_loss = float(problem.loss().detach().cpu())
    with torch.no_grad():
        for old, param in zip(before, params):
            param.copy_(old)

    update_fro = global_fro_norm(direction)
    update_op = max(float(torch.linalg.matrix_norm(tensor, ord=2).detach().cpu()) for tensor in direction)
    grad_fro = global_fro_norm(grads)
    inner = float(sum(float(torch.sum(grad * descent).detach().cpu()) for grad, descent in zip(grads, direction)))
    sigma_update = [singular_values(tensor) for tensor in direction]
    update_metrics = derive_update_metrics(sigma_update)
    return {
        "delta_loss": old_loss - new_loss,
        "new_loss": new_loss,
        "update_grad_inner": inner,
        "update_grad_cosine": inner / max(grad_fro * update_fro, 1e-300),
        "update_grad_per_update_norm": inner / max(update_fro, 1e-300),
        "update_fro_norm": update_fro,
        "update_op_norm": update_op,
        "sigma_update": json_dumps_nested(sigma_update),
        "nrUpdate": update_metrics["nrUpdate"],
        "stUpdate": update_metrics["stUpdate"],
        "nrUpdateFrac": update_metrics["nrUpdateFrac"],
        "stUpdateFrac": update_metrics["stUpdateFrac"],
        "update_flatness": update_metrics["update_flatness"],
    }


def run_ablation(config: ExperimentConfig) -> pd.DataFrame:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for spec in config.specs:
        for seed in config.seeds:
            for checkpoint_source, checkpoint_step in checkpoint_plan(spec):
                problem = build_problem(spec, seed, device=device, dtype=dtype)
                advance_to_checkpoint(problem, checkpoint_source, checkpoint_step, spec.lr)
                params = problem.parameters()
                for param in params:
                    param.grad = None
                problem.set_train_step(checkpoint_step)
                loss = problem.loss()
                loss.backward()
                old_loss = float(loss.detach().cpu())
                grads = [param.grad.detach().clone() for param in params]
                directions = candidate_directions(grads)
                for target in target_grid(spec.family):
                    for candidate, direction in directions.items():
                        scaled = scaled_direction(direction, target_relative_update_norm=target, params=params)
                        metrics = evaluate_direction(problem, scaled, old_loss)
                        rows.append(
                            {
                                "problem_family": spec.family,
                                "setting": spec.setting,
                                "kappa": float(spec.kappa),
                                "seed": int(seed),
                                "checkpoint_source": checkpoint_source,
                                "checkpoint_step": int(checkpoint_step),
                                "target_relative_update_norm": float(target),
                                "candidate": candidate,
                                "old_loss": old_loss,
                                "recovery_error": problem.recovery_error(),
                                "elapsed_s": time.perf_counter() - started,
                                **metrics,
                            }
                        )
    return pd.DataFrame(rows)


def pair_frame(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_loss",
        "update_grad_inner",
        "update_grad_cosine",
        "nrUpdate",
        "stUpdate",
        "update_op_norm",
    ]
    pivot = rows.pivot_table(
        index=[
            "problem_family",
            "setting",
            "kappa",
            "seed",
            "checkpoint_source",
            "checkpoint_step",
            "target_relative_update_norm",
        ],
        columns="candidate",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{candidate}" for metric, candidate in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    comparisons = [("PolarMuon", "GD"), ("FreshAdamSign", "GD"), ("PolarMuon", "FreshAdamSign")]
    out = []
    for _, row in paired.iterrows():
        base = row[
            [
                "problem_family",
                "setting",
                "kappa",
                "seed",
                "checkpoint_source",
                "checkpoint_step",
                "target_relative_update_norm",
            ]
        ].to_dict()
        for numerator, denominator in comparisons:
            rec = {**base, "comparison": f"{numerator}/{denominator}"}
            for metric in metrics:
                top = float(row[f"{metric}_{numerator}"])
                bottom = float(row[f"{metric}_{denominator}"])
                rec[f"{metric}_ratio"] = top / (abs(bottom) + 1e-300)
                rec[f"{metric}_delta"] = top - bottom
                rec[f"{metric}_{numerator}"] = top
                rec[f"{metric}_{denominator}"] = bottom
            out.append(rec)
    return pd.DataFrame(out)


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_specs: list[tuple[str, list[str]]] = [
        ("all", ["comparison"]),
        ("family", ["problem_family", "comparison"]),
        ("family_source", ["problem_family", "checkpoint_source", "comparison"]),
        ("setting_source", ["problem_family", "setting", "checkpoint_source", "comparison"]),
    ]
    for group_type, columns in group_specs:
        for key, group in paired.groupby(columns, observed=True, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            values = dict(zip(columns, key))
            for metric in ["delta_loss", "update_grad_inner", "update_grad_cosine", "nrUpdate", "stUpdate"]:
                ratio, lo, hi = log_ratio_ci95(group[f"{metric}_ratio"])
                records.append(
                    {
                        "group_type": group_type,
                        "problem_family": values.get("problem_family", "All"),
                        "setting": values.get("setting", "All"),
                        "checkpoint_source": values.get("checkpoint_source", "All"),
                        "comparison": values["comparison"],
                        "metric": metric,
                        "n_pairs": int(len(group)),
                        "geomean_ratio": ratio,
                        "ratio_ci95_low": lo,
                        "ratio_ci95_high": hi,
                        "ratio_ci95_above_one": bool(lo > 1.0) if np.isfinite(lo) else False,
                        "ratio_ci95_below_one": bool(hi < 1.0) if np.isfinite(hi) else False,
                        "numerator_higher_rate": float((group[f"{metric}_delta"] > 0).mean()),
                    }
                )
    return pd.DataFrame(records)


def plot_summary(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source = summary[
        (summary["group_type"] == "family")
        & (summary["metric"].isin(["update_grad_inner", "nrUpdate", "stUpdate"]))
        & (summary["comparison"].isin(["PolarMuon/GD", "FreshAdamSign/GD"]))
    ].copy()
    if source.empty:
        return
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
    ax.errorbar(x, values, yerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)], fmt="none", ecolor="black", capsize=2)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("geomean ratio")
    ax.set_title("Stateless direction ablation under matched update Frobenius size")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "stateless_direction_ratios.png", dpi=180)
    plt.close(fig)


def write_discussion(summary: pd.DataFrame) -> None:
    polar_gd_inner = require_one(
        summary, group_type="all", problem_family="All", checkpoint_source="All", comparison="PolarMuon/GD", metric="update_grad_inner"
    )
    polar_gd_nr = require_one(summary, group_type="all", problem_family="All", checkpoint_source="All", comparison="PolarMuon/GD", metric="nrUpdate")
    polar_gd_st = require_one(summary, group_type="all", problem_family="All", checkpoint_source="All", comparison="PolarMuon/GD", metric="stUpdate")
    adam_gd_inner = require_one(
        summary, group_type="all", problem_family="All", checkpoint_source="All", comparison="FreshAdamSign/GD", metric="update_grad_inner"
    )
    family_inner = summary[
        (summary["group_type"] == "family") & (summary["metric"] == "update_grad_inner") & (summary["comparison"] == "PolarMuon/GD")
    ].copy()
    family_spectrum = summary[
        (summary["group_type"] == "family")
        & (summary["metric"].isin(["nrUpdate", "stUpdate"]))
        & (summary["comparison"] == "PolarMuon/GD")
    ].copy()
    text = f"""# E11 Stateless Direction Ablation

This generated note tests whether the polar/Muon direction itself is the key spectral intervention. For each problem state, it evaluates three one-step candidate directions under the same gradient and the same global Frobenius update size:

| candidate | definition |
|:--|:--|
| GD | raw gradient direction, globally rescaled |
| FreshAdamSign | first-step Adam-style elementwise normalized direction, globally rescaled |
| PolarMuon | per-layer polar factor `U V^T` of the gradient, globally rescaled |

The states come from Init, Adam checkpoints, and Muon checkpoints. This separates the checkpoint geometry from the candidate direction.

## Main Takeaway

PolarMuon strongly increases update rank statistics even without momentum or optimizer state: nrUpdate PolarMuon/GD={fmt(polar_gd_nr['geomean_ratio'])} CI={ratio_ci(polar_gd_nr)}, stUpdate={fmt(polar_gd_st['geomean_ratio'])} CI={ratio_ci(polar_gd_st)}.

However, the same stateless polar direction is not globally better for one-step progress: update_grad_inner PolarMuon/GD={fmt(polar_gd_inner['geomean_ratio'])} CI={ratio_ci(polar_gd_inner)}. FreshAdamSign/GD gives update_grad_inner={fmt(adam_gd_inner['geomean_ratio'])} CI={ratio_ci(adam_gd_inner)}.

This supports the paper framing: **polar spectrum shaping is a robust direction-level mechanism, but progress remains boundary-dependent.**

## Polar vs GD By Family

{markdown_table(family_inner, ["problem_family", "comparison", "metric", "n_pairs", "geomean_ratio", "ratio_ci95_low", "ratio_ci95_high", "numerator_higher_rate"])}

## Polar Update Spectrum By Family

{markdown_table(family_spectrum, ["problem_family", "comparison", "metric", "n_pairs", "geomean_ratio", "ratio_ci95_low", "ratio_ci95_high", "numerator_higher_rate"])}

## Evidence

- [stateless direction rows](../results/e11_stateless_direction_ablation/stateless_direction_rows.csv)
- [paired direction rows](../results/e11_stateless_direction_ablation/stateless_direction_pairs.csv)
- [summary table](../results/e11_stateless_direction_ablation/stateless_direction_summary.csv)
- [ratio figure](../figures/e11_stateless_direction_ablation/stateless_direction_ratios.png)

## Caveat

This is a one-step stateless intervention. It isolates the direction-level spectral bias, but it does not replace natural optimizer trajectories, momentum/state ablations, or longer-horizon retuned training.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    config = build_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_ablation(config)
    paired = pair_frame(rows)
    summary = summarize_pairs(paired)
    rows.to_csv(OUTPUT_DIR / "stateless_direction_rows.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "stateless_direction_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "stateless_direction_summary.csv", index=False)
    plot_summary(summary)
    write_discussion(summary)
    print(f"saved stateless direction ablation to {OUTPUT_DIR}")
    print(f"saved stateless direction discussion to {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
