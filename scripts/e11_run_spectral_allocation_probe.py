from __future__ import annotations

import json
import sys
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
from e11_condition_geometry.diagnostics import matrix_effective_rank, stable_rank, singular_values
from e11_condition_geometry.runner import build_optimizer, build_problem, dtype_from_name, snapshot_parameters
from e11_condition_geometry.statistics import ci95, log_ratio_ci95


OUTPUT_DIR = Path("results/e11_spectral_allocation_probe")
FIGURE_DIR = Path("figures/e11_spectral_allocation_probe")
DISCUSSION_PATH = Path("discussion/e11_spectral_allocation_probe.md")
TARGETS = (1e-3, 1e-2)
DIRECTIONS = ("gd_spectrum", "flat_polar", "top_singular")
BUDGETS = ("fro", "op")
STATE_SOURCES = ("initial", "adam_step3", "muon_step3")


def probe_specs() -> tuple[ProblemSpec, ...]:
    return (
        ProblemSpec(
            family="MatrixFactorizationInput",
            setting="MF input kappa=1e+02",
            steps=10,
            d=60,
            rank=5,
            kappa=1e2,
            num_factors=10,
            input_columns_multiplier=10,
            lr=1e-2,
        ),
        ProblemSpec(
            family="MatrixSensing",
            setting="Matrix sensing kappa=1e+02",
            steps=5,
            d=60,
            rank=5,
            kappa=1e2,
            measurement_multiplier=2.0,
            lr=1e-2,
        ),
        ProblemSpec(
            family="SmallMLPDigits",
            setting="Small MLP digits hidden=16",
            steps=10,
            d=64,
            rank=10,
            hidden_dim=16,
            num_samples=1024,
            lr=1e-2,
        ),
        ProblemSpec(
            family="SmallMLPDigits",
            setting="Small MLP digits hidden=64",
            steps=10,
            d=64,
            rank=10,
            hidden_dim=64,
            num_samples=1024,
            lr=1e-2,
        ),
    )


def build_config() -> ExperimentConfig:
    base = default_config()
    return replace(
        base,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        discussion_path=DISCUSSION_PATH,
        specs=probe_specs(),
    )


def advance_problem(problem, algo: str, steps: int, lr: float) -> None:
    params = problem.parameters()
    optimizer = build_optimizer(algo, params, lr)
    for step in range(steps):
        problem.set_train_step(step)
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss()
        loss.backward()
        optimizer.step()


def direction_from_gradient(grad: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "gd_spectrum":
        return grad.detach().clone()
    u, _, vh = torch.linalg.svd(grad.detach(), full_matrices=False)
    if direction == "flat_polar":
        return u @ vh
    if direction == "top_singular":
        return u[:, :1] @ vh[:1, :]
    raise ValueError(f"unknown direction: {direction}")


def matrix_norm(matrix: torch.Tensor, budget: str) -> float:
    if budget == "fro":
        return float(torch.linalg.norm(matrix).cpu())
    if budget == "op":
        return float(torch.linalg.matrix_norm(matrix, ord=2).cpu())
    raise ValueError(f"unknown budget: {budget}")


def param_norm(matrix: torch.Tensor, budget: str) -> float:
    value = matrix_norm(matrix, budget)
    return max(value, 1e-300)


def scaled_descent_updates(
    before: list[torch.Tensor],
    grads: list[torch.Tensor],
    direction: str,
    budget: str,
    target: float,
) -> list[torch.Tensor]:
    updates = []
    for old, grad in zip(before, grads):
        base = direction_from_gradient(grad, direction)
        base_norm = matrix_norm(base, budget)
        if base_norm <= 0.0 or not np.isfinite(base_norm):
            updates.append(torch.zeros_like(base))
            continue
        update_norm = target * param_norm(old, budget)
        updates.append((update_norm / base_norm) * base)
    return updates


def evaluate_update(problem, before: list[torch.Tensor], updates: list[torch.Tensor]) -> float:
    params = problem.parameters()
    with torch.no_grad():
        for param, old, update in zip(params, before, updates):
            param.copy_(old - update)
    loss_after = float(problem.loss().detach().cpu())
    with torch.no_grad():
        for param, old in zip(params, before):
            param.copy_(old)
    return loss_after


def update_stats(grads: list[torch.Tensor], updates: list[torch.Tensor]) -> dict:
    inner = 0.0
    grad_fro_sq = 0.0
    update_fro_sq = 0.0
    update_op_values = []
    nr_values = []
    st_values = []
    grad_rank_values = []
    for grad, update in zip(grads, updates):
        inner += float(torch.sum(grad.detach() * update.detach()).cpu())
        grad_fro_sq += float(torch.linalg.norm(grad.detach()).cpu()) ** 2
        update_fro = float(torch.linalg.norm(update.detach()).cpu())
        update_fro_sq += update_fro**2
        update_op_values.append(float(torch.linalg.matrix_norm(update.detach(), ord=2).cpu()))
        update_sigmas = singular_values(update)
        grad_sigmas = singular_values(grad)
        nr_values.append(matrix_effective_rank(update_sigmas))
        st_values.append(stable_rank(update_sigmas))
        rank_ceiling = len(grad_sigmas)
        grad_rank_values.append(matrix_effective_rank(grad_sigmas) / rank_ceiling if rank_ceiling else np.nan)
    update_fro_total = update_fro_sq**0.5
    grad_fro_total = grad_fro_sq**0.5
    return {
        "update_grad_inner": inner,
        "update_grad_cosine": inner / max(update_fro_total * grad_fro_total, 1e-300),
        "update_fro_norm": update_fro_total,
        "update_op_norm": max(update_op_values) if update_op_values else np.nan,
        "mean_nr_update": float(np.nanmean(nr_values)),
        "mean_st_update": float(np.nanmean(st_values)),
        "mean_grad_rank_fraction": float(np.nanmean(grad_rank_values)),
    }


def probe_state(spec: ProblemSpec, seed: int, state_source: str, run_id: int, config: ExperimentConfig) -> list[dict]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    if state_source == "adam_step3":
        advance_problem(problem, "Adam", 3, spec.lr)
        problem.set_train_step(3)
    elif state_source == "muon_step3":
        advance_problem(problem, "Muon", 3, spec.lr)
        problem.set_train_step(3)
    elif state_source != "initial":
        raise ValueError(f"unknown state source: {state_source}")
    else:
        problem.set_train_step(0)

    params = problem.parameters()
    for param in params:
        param.grad = None
    loss = problem.loss()
    loss.backward()
    loss_before = float(loss.detach().cpu())
    before = snapshot_parameters(params)
    grads = [param.grad.detach().clone() for param in params]

    rows = []
    for budget in BUDGETS:
        for target in TARGETS:
            for direction in DIRECTIONS:
                updates = scaled_descent_updates(before, grads, direction, budget, target)
                loss_after = evaluate_update(problem, before, updates)
                stats = update_stats(grads, updates)
                rows.append(
                    {
                        "run_id": run_id,
                        "problem_family": spec.family,
                        "base_setting": spec.setting,
                        "seed": seed,
                        "state_source": state_source,
                        "budget": budget,
                        "target_relative_norm": target,
                        "direction": direction,
                        "loss_before": loss_before,
                        "loss_after": loss_after,
                        "delta_loss": loss_before - loss_after,
                        **stats,
                    }
                )
    return rows


def run_probe(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    run_id = 0
    for spec in config.specs:
        for seed in config.seeds:
            for state_source in STATE_SOURCES:
                rows.extend(probe_state(spec, seed, state_source, run_id, config))
                run_id += 1
    return pd.DataFrame(rows)


def ratio_summary(rows: pd.DataFrame) -> pd.DataFrame:
    pivot = rows.pivot_table(
        index=["problem_family", "base_setting", "state_source", "budget", "target_relative_norm", "seed"],
        columns="direction",
        values=["delta_loss", "update_grad_inner"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{direction}" for metric, direction in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("setting_budget", key, group)
        for key, group in paired.groupby(["problem_family", "base_setting", "budget"], observed=True, sort=True)
    )
    groups.extend(("budget_all", ("All", "All", budget), group) for budget, group in paired.groupby("budget", observed=True, sort=True))
    for group_type, key, group in groups:
        family, setting, budget = key
        for numerator in ["flat_polar", "top_singular"]:
            for metric in ["update_grad_inner", "delta_loss"]:
                ratio_values = group[f"{metric}_{numerator}"] / (group[f"{metric}_gd_spectrum"].abs() + 1e-300)
                delta_values = group[f"{metric}_{numerator}"] - group[f"{metric}_gd_spectrum"]
                ratio, lo, hi = log_ratio_ci95(ratio_values)
                delta, delta_lo, delta_hi = ci95(delta_values)
                records.append(
                    {
                        "group_type": group_type,
                        "problem_family": family,
                        "base_setting": setting,
                        "budget": budget,
                        "comparison": f"{numerator}_over_gd_spectrum",
                        "metric": metric,
                        "n_pairs": int(len(group)),
                        "geomean_ratio": ratio,
                        "ratio_ci95_low": lo,
                        "ratio_ci95_high": hi,
                        "ratio_ci95_above_one": bool(lo > 1.0) if np.isfinite(lo) else False,
                        "ratio_ci95_below_one": bool(hi < 1.0) if np.isfinite(hi) else False,
                        "mean_delta": delta,
                        "delta_ci95_low": delta_lo,
                        "delta_ci95_high": delta_hi,
                    }
                )
    return pd.DataFrame(records)


def direction_level_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["problem_family", "base_setting", "budget", "direction"], as_index=False, observed=True).agg(
        points=("run_id", "size"),
        mean_delta_loss=("delta_loss", "mean"),
        mean_update_grad_inner=("update_grad_inner", "mean"),
        mean_update_grad_cosine=("update_grad_cosine", "mean"),
        mean_nr_update=("mean_nr_update", "mean"),
        mean_st_update=("mean_st_update", "mean"),
        mean_grad_rank_fraction=("mean_grad_rank_fraction", "mean"),
    )


def plot_ratio_summary(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "spectral_allocation_ratios.png"
    rows = summary[
        (summary["group_type"] == "setting_budget")
        & (summary["comparison"] == "flat_polar_over_gd_spectrum")
        & (summary["metric"] == "update_grad_inner")
    ].copy()
    settings = sorted(rows["base_setting"].unique())
    x_positions = {setting: idx for idx, setting in enumerate(settings)}
    offsets = {"fro": -0.13, "op": 0.13}
    colors = {"fro": "#0072B2", "op": "#D55E00"}
    labels = {"fro": "fixed Frobenius budget", "op": "fixed operator budget"}
    fig, ax = plt.subplots(figsize=(9.8, 4.1))
    for budget in ["fro", "op"]:
        sub = rows[rows["budget"] == budget].copy()
        x = np.array([x_positions[setting] + offsets[budget] for setting in sub["base_setting"]], dtype=float)
        y = sub["geomean_ratio"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["ratio_ci95_low"].to_numpy(dtype=float),
                sub["ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=colors[budget], label=labels[budget])
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(
        [label.replace("Matrix sensing", "MS").replace("MF input", "MF").replace("Small MLP digits ", "MLP ") for label in settings],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("flat polar / GD first-order ratio")
    ax.set_title("Within-layer singular-value allocation probe")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.4g}")
    return display.to_markdown(index=False)


def write_discussion(summary: pd.DataFrame, direction_summary: pd.DataFrame, figure: Path) -> None:
    flat_rows = summary[
        (summary["comparison"] == "flat_polar_over_gd_spectrum")
        & (summary["metric"] == "update_grad_inner")
    ].copy()
    top_rows = summary[
        (summary["comparison"] == "top_singular_over_gd_spectrum")
        & (summary["metric"] == "update_grad_inner")
        & (summary["group_type"] == "budget_all")
    ].copy()
    level_rows = direction_summary[
        direction_summary["direction"].isin(["gd_spectrum", "flat_polar"])
    ].copy()
    text = f"""# E11 Within-Layer Spectral Allocation Probe

## Purpose

This probe addresses the remaining gap in the mechanism ladder: per-layer update-size control fixes how much each layer moves, but not how each layer's update singular values are allocated.

For each current gradient matrix \\(G_i = U_i \\Sigma_i V_i^\\top\\), the probe constructs three descent directions with the same gradient singular vectors:

- `gd_spectrum`: \\(U_i \\Sigma_i V_i^\\top\\), the gradient-descent singular-value allocation.
- `flat_polar`: \\(U_i V_i^\\top\\), the Muon/polar singular-value allocation.
- `top_singular`: \\(u_{{i,1}} v_{{i,1}}^\\top\\), a rank-one allocation.

Each direction is evaluated under either a fixed per-layer Frobenius-norm budget or a fixed per-layer operator-norm budget. The probe is run at initial, Adam-step-3, and Muon-step-3 states, with targets `{TARGETS}` and 5 seeds.

## Flat Polar Versus GD Spectrum

Ratios above 1 mean the flat/polar singular-value allocation gives larger one-step first-order progress than the GD singular-value allocation under the same norm budget.

![Spectral allocation ratios](../{figure})

{markdown_table(flat_rows, ["group_type", "problem_family", "base_setting", "budget", "n_pairs", "geomean_ratio", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

## Rank-One Control

{markdown_table(top_rows, ["group_type", "budget", "comparison", "metric", "n_pairs", "geomean_ratio", "ratio_ci95_low", "ratio_ci95_high", "ratio_ci95_above_one", "ratio_ci95_below_one"])}

## Direction-Level Diagnostics

{markdown_table(level_rows, ["problem_family", "base_setting", "budget", "direction", "points", "mean_delta_loss", "mean_update_grad_inner", "mean_update_grad_cosine", "mean_nr_update", "mean_st_update", "mean_grad_rank_fraction"])}

## Interpretation

The norm constraint matters. Under a fixed Frobenius budget, the GD singular-value allocation is expected to be first-order optimal by Cauchy-Schwarz, so flat/polar should not beat it. Under a fixed operator-norm budget, flat/polar can use more singular directions at the same operator norm, so it can be favorable when the gradient has useful multi-directional spectral mass.

This gives a more precise version of the Muon mechanism: Muon's flat polar update is not universally better as a direction; it is a particular answer to an operator-norm-like geometry. Whether that helps depends on whether the task/layer gradient spectrum rewards spreading update mass across singular directions.

## Caveats

1. These are artificial one-step probes, not natural optimizer trajectories.
2. The probe reuses gradient singular vectors, so it isolates singular-value allocation rather than singular-vector mismatch.
3. Adam's actual direction is not modeled here; this only compares gradient-spectrum, flat-polar, and rank-one allocations.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = run_probe(config)
    summary = ratio_summary(rows)
    direction_summary = direction_level_summary(rows)
    rows.to_csv(config.output_dir / "probe_rows.csv", index=False)
    summary.to_csv(config.output_dir / "spectral_allocation_summary.csv", index=False)
    direction_summary.to_csv(config.output_dir / "direction_level_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_ratio_summary(summary)
    write_discussion(summary, direction_summary, figure)
    print(f"saved spectral allocation probe to {config.output_dir}")
    print(f"rows={len(rows)}, summary rows={len(summary)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
