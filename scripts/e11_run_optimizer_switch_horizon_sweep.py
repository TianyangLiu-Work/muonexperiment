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
from e11_condition_geometry.runner import build_optimizer, build_problem, dtype_from_name, snapshot_parameters
from e11_condition_geometry.statistics import ci95, log_ratio_ci95


OUTPUT_DIR = Path("results/e11_optimizer_switch_horizon_sweep")
FIGURE_DIR = Path("figures/e11_optimizer_switch_horizon_sweep")
DISCUSSION_PATH = Path("discussion/e11_optimizer_switch_horizon_sweep.md")
CONTINUATION_HORIZONS = (1, 3, 10, 30)


def switch_specs() -> tuple[ProblemSpec, ...]:
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
            family="MatrixFactorizationInput",
            setting="MF input kappa=1e+05",
            steps=10,
            d=60,
            rank=5,
            kappa=1e5,
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
            family="MatrixSensing",
            setting="Matrix sensing kappa=1e+05",
            steps=5,
            d=60,
            rank=5,
            kappa=1e5,
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
        specs=switch_specs(),
    )


def checkpoint_step(spec: ProblemSpec) -> int:
    return min(3, spec.steps - 1)


def copy_parameters(params: list[torch.nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)


def train_steps(problem, optimizer: torch.optim.Optimizer, steps: int, start_step: int = 0) -> None:
    params = problem.parameters()
    for offset in range(steps):
        problem.set_train_step(start_step + offset)
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss()
        loss.backward()
        optimizer.step()
        for param in params:
            if not torch.isfinite(param.detach()).all():
                raise FloatingPointError("non-finite parameter during horizon sweep")


def checkpoint_params(
    spec: ProblemSpec,
    seed: int,
    source_algo: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], float, float]:
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    optimizer = build_optimizer(source_algo, problem.parameters(), spec.lr)
    step = checkpoint_step(spec)
    train_steps(problem, optimizer, step, start_step=0)
    problem.set_train_step(step)
    return snapshot_parameters(problem.parameters()), float(problem.loss().detach().cpu()), float(problem.recovery_error())


def continue_from_checkpoint(
    spec: ProblemSpec,
    seed: int,
    params_at_checkpoint: list[torch.Tensor],
    source_algo: str,
    continuation_variant: str,
    horizon: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    if continuation_variant == "own_fresh":
        continuation_algo = source_algo
    elif continuation_variant == "switched_fresh":
        continuation_algo = "Muon" if source_algo == "Adam" else "Adam"
    else:
        raise ValueError(f"unknown continuation variant: {continuation_variant}")
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    copy_parameters(problem.parameters(), params_at_checkpoint)
    optimizer = build_optimizer(continuation_algo, problem.parameters(), spec.lr)
    start_step = checkpoint_step(spec)
    problem.set_train_step(start_step)
    loss_before = float(problem.loss().detach().cpu())
    recovery_before = float(problem.recovery_error())
    try:
        train_steps(problem, optimizer, horizon, start_step=start_step)
        problem.set_train_step(start_step + horizon)
        loss_after = float(problem.loss().detach().cpu())
        recovery_after = float(problem.recovery_error())
        finite = np.isfinite(loss_after) and np.isfinite(recovery_after)
    except FloatingPointError:
        loss_after = float("inf")
        recovery_after = float("inf")
        finite = False
    return {
        "continuation_variant": continuation_variant,
        "continuation_algo": continuation_algo,
        "horizon": horizon,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "total_decrease": loss_before - loss_after if finite else float("-inf"),
        "recovery_before": recovery_before,
        "recovery_after": recovery_after,
        "finite": bool(finite),
    }


def run_probe(config: ExperimentConfig) -> pd.DataFrame:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows = []
    for spec in config.specs:
        for seed in config.seeds:
            for source_algo in ["Adam", "Muon"]:
                params, loss, recovery = checkpoint_params(
                    spec,
                    seed,
                    source_algo,
                    device=device,
                    dtype=dtype,
                )
                for horizon in CONTINUATION_HORIZONS:
                    for variant in ["own_fresh", "switched_fresh"]:
                        result = continue_from_checkpoint(
                            spec,
                            seed,
                            params,
                            source_algo,
                            variant,
                            horizon,
                            device=device,
                            dtype=dtype,
                        )
                        rows.append(
                            {
                                "problem_family": spec.family,
                                "base_setting": spec.setting,
                                "seed": seed,
                                "source_algo": source_algo,
                                "checkpoint_step": checkpoint_step(spec),
                                "checkpoint_loss": loss,
                                "checkpoint_recovery": recovery,
                                **result,
                            }
                        )
    return pd.DataFrame(rows)


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    finite_rows = rows[rows["finite"]].copy()
    pivot = finite_rows.pivot_table(
        index=["problem_family", "base_setting", "source_algo", "horizon", "seed"],
        columns="continuation_variant",
        values=["loss_after", "total_decrease", "recovery_after"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{variant}" for metric, variant in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("setting_source_horizon", key, group)
        for key, group in paired.groupby(["problem_family", "base_setting", "source_algo", "horizon"], observed=True, sort=True)
    )
    groups.extend(
        ("source_horizon_all_settings", ("All", "All", source_algo, horizon), group)
        for (source_algo, horizon), group in paired.groupby(["source_algo", "horizon"], observed=True, sort=True)
    )
    groups.extend(
        ("horizon_all", ("All", "All", "All", horizon), group)
        for horizon, group in paired.groupby("horizon", observed=True, sort=True)
    )
    groups.append(("all", ("All", "All", "All", "All"), paired))
    for group_type, key, group in groups:
        family, setting, source_algo, horizon = key
        for metric in ["loss_after", "total_decrease", "recovery_after"]:
            switched = group[f"{metric}_switched_fresh"]
            own = group[f"{metric}_own_fresh"]
            ratio_values = switched / (own.abs() + 1e-300)
            if metric == "total_decrease":
                switched_better = switched > own
            else:
                switched_better = switched < own
            signed_mean, signed_lo, signed_hi = ci95(ratio_values)
            ratio, lo, hi = log_ratio_ci95(ratio_values)
            delta, delta_lo, delta_hi = ci95(switched - own)
            records.append(
                {
                    "group_type": group_type,
                    "problem_family": family,
                    "base_setting": setting,
                    "source_algo": source_algo,
                    "horizon": horizon,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "switched_better_pairs": int(switched_better.sum()),
                    "mean_signed_switched_over_own": signed_mean,
                    "signed_ratio_ci95_low": signed_lo,
                    "signed_ratio_ci95_high": signed_hi,
                    "geomean_switched_over_own": ratio,
                    "ratio_ci95_low": lo,
                    "ratio_ci95_high": hi,
                    "positive_ratio_pairs": int((ratio_values > 0).sum()),
                    "mean_delta_switched_minus_own": delta,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                }
            )
    return pd.DataFrame(records)


def plot_horizon_sweep(summary_frame: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "optimizer_switch_horizon_sweep.png"
    rows = summary_frame[
        (summary_frame["group_type"] == "source_horizon_all_settings")
        & (summary_frame["metric"] == "loss_after")
    ].copy()
    colors = {"Adam": "#D55E00", "Muon": "#0072B2"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for source_algo in ["Adam", "Muon"]:
        sub = rows[rows["source_algo"] == source_algo].sort_values("horizon")
        x = sub["horizon"].to_numpy(dtype=float)
        y = sub["mean_signed_switched_over_own"].to_numpy(dtype=float)
        lo = sub["signed_ratio_ci95_low"].to_numpy(dtype=float)
        hi = sub["signed_ratio_ci95_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=colors[source_algo], label=f"checkpoint from {source_algo}")
        ax.fill_between(x, lo, hi, color=colors[source_algo], alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("fresh continuation horizon")
    ax.set_ylabel("final-loss ratio: switched / own")
    ax.set_title("Fresh optimizer switch horizon sweep")
    ax.grid(alpha=0.25)
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


def write_discussion(summary_frame: pd.DataFrame, figure: Path) -> None:
    horizon_core = summary_frame[
        (summary_frame["metric"] == "loss_after")
        & (summary_frame["group_type"].isin(["source_horizon_all_settings", "horizon_all", "all"]))
    ].copy()
    setting_core = summary_frame[
        (summary_frame["metric"] == "loss_after")
        & (summary_frame["group_type"] == "setting_source_horizon")
        & (summary_frame["horizon"] == 10)
    ].copy()
    text = f"""# E11 Optimizer Switch Horizon Sweep

## Purpose

The reset-control probe compares fresh-own and fresh-switched continuations over the original short remaining horizon. This sweep asks whether that pattern persists when the fresh continuation horizon changes.

All runs start from the same checkpoint step (`3`, or the last non-final step for shorter problems). Both compared continuations use fresh optimizer state, so the main comparison is optimizer type rather than preserved optimizer memory.

## Horizon-Level Final Loss Ratio

Ratios below 1 mean the switched fresh optimizer gives lower final loss than the source optimizer type with fresh state.

![Optimizer switch horizon sweep](../{figure})

{markdown_table(horizon_core, ["group_type", "source_algo", "horizon", "metric", "n_pairs", "switched_better_pairs", "mean_signed_switched_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Setting-Level Snapshot At Horizon 10

{markdown_table(setting_core, ["problem_family", "base_setting", "source_algo", "horizon", "metric", "n_pairs", "switched_better_pairs", "mean_signed_switched_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Interpretation

This is a stronger trajectory-level check than the original switch probe because it varies the continuation horizon while keeping both continuation optimizers fresh. It directly tests whether the observed switch pattern is only a one-window artifact.

## Caveats

1. Fresh continuations are not natural optimizer histories.
2. Longer horizons are still short relative to full training.
3. The same learning rates are reused for all horizons, so this is not a retuned long-horizon comparison.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = run_probe(config)
    summary_frame = summary(rows)
    rows.to_csv(config.output_dir / "optimizer_switch_horizon_rows.csv", index=False)
    summary_frame.to_csv(config.output_dir / "optimizer_switch_horizon_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_horizon_sweep(summary_frame)
    write_discussion(summary_frame, figure)
    print(f"saved optimizer switch horizon sweep to {config.output_dir}")
    print(f"rows={len(rows)}, summary rows={len(summary_frame)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
