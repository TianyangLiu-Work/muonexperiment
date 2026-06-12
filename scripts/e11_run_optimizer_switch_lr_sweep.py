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


OUTPUT_DIR = Path("results/e11_optimizer_switch_lr_sweep")
FIGURE_DIR = Path("figures/e11_optimizer_switch_lr_sweep")
DISCUSSION_PATH = Path("discussion/e11_optimizer_switch_lr_sweep.md")
CONTINUATION_HORIZON = 30
CONTINUATION_LRS = (3e-3, 1e-2, 3e-2)


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
    for offset in range(steps):
        problem.set_train_step(start_step + offset)
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss()
        loss.backward()
        optimizer.step()
        for param in problem.parameters():
            if not torch.isfinite(param.detach()).all():
                raise FloatingPointError("non-finite parameter during continuation lr sweep")


def checkpoint_params(spec: ProblemSpec, seed: int, source_algo: str, *, device: torch.device, dtype: torch.dtype) -> tuple[list[torch.Tensor], float]:
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    optimizer = build_optimizer(source_algo, problem.parameters(), spec.lr)
    step = checkpoint_step(spec)
    train_steps(problem, optimizer, step, start_step=0)
    problem.set_train_step(step)
    return snapshot_parameters(problem.parameters()), float(problem.loss().detach().cpu())


def build_continuation_optimizer(algo: str, params: list[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
    return build_optimizer(algo, params, lr)


def continue_run(
    spec: ProblemSpec,
    seed: int,
    params_at_checkpoint: list[torch.Tensor],
    source_algo: str,
    continuation_variant: str,
    continuation_lr: float,
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
    optimizer = build_continuation_optimizer(continuation_algo, problem.parameters(), continuation_lr)
    start_step = checkpoint_step(spec)
    problem.set_train_step(start_step)
    loss_before = float(problem.loss().detach().cpu())
    try:
        train_steps(problem, optimizer, CONTINUATION_HORIZON, start_step=start_step)
        problem.set_train_step(start_step + CONTINUATION_HORIZON)
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
        "continuation_lr": continuation_lr,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "total_decrease": loss_before - loss_after if finite else float("-inf"),
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
                params, checkpoint_loss = checkpoint_params(
                    spec,
                    seed,
                    source_algo,
                    device=device,
                    dtype=dtype,
                )
                for variant in ["own_fresh", "switched_fresh"]:
                    for lr in CONTINUATION_LRS:
                        result = continue_run(
                            spec,
                            seed,
                            params,
                            source_algo,
                            variant,
                            lr,
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
                                "checkpoint_loss": checkpoint_loss,
                                "continuation_horizon": CONTINUATION_HORIZON,
                                **result,
                            }
                        )
    return pd.DataFrame(rows)


def best_by_variant(rows: pd.DataFrame) -> pd.DataFrame:
    finite = rows[rows["finite"]].copy()
    idx = finite.groupby(
        ["problem_family", "base_setting", "source_algo", "seed", "continuation_variant"],
        observed=True,
    )["loss_after"].idxmin()
    return finite.loc[idx].reset_index(drop=True)


def pair_summary(best_rows: pd.DataFrame) -> pd.DataFrame:
    pivot = best_rows.pivot_table(
        index=["problem_family", "base_setting", "source_algo", "seed"],
        columns="continuation_variant",
        values=["loss_after", "recovery_after", "continuation_lr"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{variant}" for metric, variant in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("setting_source", key, group)
        for key, group in paired.groupby(["problem_family", "base_setting", "source_algo"], observed=True, sort=True)
    )
    groups.extend(
        ("source_all", ("All", "All", source_algo), group)
        for source_algo, group in paired.groupby("source_algo", observed=True, sort=True)
    )
    groups.append(("all", ("All", "All", "All"), paired))
    for group_type, key, group in groups:
        family, setting, source_algo = key
        for metric in ["loss_after", "recovery_after"]:
            switched = group[f"{metric}_switched_fresh"]
            own = group[f"{metric}_own_fresh"]
            ratio_values = switched / (own.abs() + 1e-300)
            switched_better = switched < own
            mean_ratio, lo, hi = ci95(ratio_values)
            geo, geo_lo, geo_hi = log_ratio_ci95(ratio_values)
            delta, delta_lo, delta_hi = ci95(switched - own)
            records.append(
                {
                    "group_type": group_type,
                    "problem_family": family,
                    "base_setting": setting,
                    "source_algo": source_algo,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "switched_better_pairs": int(switched_better.sum()),
                    "mean_switched_over_own": mean_ratio,
                    "ratio_ci95_low": lo,
                    "ratio_ci95_high": hi,
                    "geomean_switched_over_own": geo,
                    "geomean_ci95_low": geo_lo,
                    "geomean_ci95_high": geo_hi,
                    "mean_delta_switched_minus_own": delta,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                    "mean_best_lr_own": float(group["continuation_lr_own_fresh"].mean()),
                    "mean_best_lr_switched": float(group["continuation_lr_switched_fresh"].mean()),
                }
            )
    return pd.DataFrame(records)


def lr_frequency(best_rows: pd.DataFrame) -> pd.DataFrame:
    return best_rows.groupby(
        ["source_algo", "continuation_variant", "continuation_lr"],
        as_index=False,
        observed=True,
    ).agg(count=("seed", "size"))


def plot_lr_sweep(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "optimizer_switch_lr_sweep.png"
    rows = summary[
        (summary["group_type"] == "setting_source")
        & (summary["metric"] == "loss_after")
    ].copy()
    settings = sorted(rows["base_setting"].unique())
    colors = {"Adam": "#D55E00", "Muon": "#0072B2"}
    offsets = {"Adam": -0.14, "Muon": 0.14}
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    for source_algo in ["Adam", "Muon"]:
        sub = rows[rows["source_algo"] == source_algo]
        x = np.array([settings.index(setting) + offsets[source_algo] for setting in sub["base_setting"]], dtype=float)
        y = sub["mean_switched_over_own"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["ratio_ci95_low"].to_numpy(dtype=float),
                sub["ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=colors[source_algo], label=f"checkpoint from {source_algo}")
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(
        [label.replace("Matrix sensing", "MS").replace("MF input", "MF").replace("Small MLP digits ", "MLP ") for label in settings],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("best-final-loss ratio: switched / own")
    ax.set_title("Fresh optimizer switch after continuation LR sweep")
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


def write_discussion(summary: pd.DataFrame, frequency: pd.DataFrame, figure: Path) -> None:
    core = summary[
        (summary["metric"] == "loss_after")
        & (summary["group_type"].isin(["setting_source", "source_all", "all"]))
    ].copy()
    text = f"""# E11 Optimizer Switch Continuation-LR Sweep

## Purpose

The fresh optimizer switch horizon sweep reused the same learning rate for both continuation optimizers. This probe checks whether the horizon-30 switch pattern survives a small continuation-learning-rate sweep.

From the same checkpoint state, `own_fresh` and `switched_fresh` are each run for `{CONTINUATION_HORIZON}` steps with continuation learning rates `{", ".join(f"{lr:g}" for lr in CONTINUATION_LRS)}`. Each variant is evaluated by its best final loss over that small LR grid.

## Best-LR Final Loss Ratio

Ratios below 1 mean the switched optimizer type reaches lower final loss than the source optimizer type after each gets its own best continuation LR.

![Optimizer switch LR sweep](../{figure})

{markdown_table(core, ["group_type", "problem_family", "base_setting", "source_algo", "metric", "n_pairs", "switched_better_pairs", "mean_switched_over_own", "ratio_ci95_low", "ratio_ci95_high", "mean_best_lr_own", "mean_best_lr_switched"])}

## Best-LR Frequencies

{markdown_table(frequency, ["source_algo", "continuation_variant", "continuation_lr", "count"])}

## Interpretation

This probe weakens the fixed-continuation-LR concern for the horizon-30 switch comparison. It is still a small LR grid and a short continuation, so it should be treated as a robustness check rather than a full hyperparameter search.

## Caveats

1. The LR grid is small and shared across all task families.
2. The checkpoint was produced with the original source optimizer at `lr=1e-2`; only continuation LR is swept.
3. Best-LR selection is oracle-style and should not be interpreted as an online switching policy.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = run_probe(config)
    best = best_by_variant(rows)
    summary = pair_summary(best)
    frequency = lr_frequency(best)
    rows.to_csv(config.output_dir / "optimizer_switch_lr_rows.csv", index=False)
    best.to_csv(config.output_dir / "optimizer_switch_lr_best.csv", index=False)
    summary.to_csv(config.output_dir / "optimizer_switch_lr_summary.csv", index=False)
    frequency.to_csv(config.output_dir / "optimizer_switch_lr_frequency.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_lr_sweep(summary)
    write_discussion(summary, frequency, figure)
    print(f"saved optimizer switch continuation-lr sweep to {config.output_dir}")
    print(f"rows={len(rows)}, best rows={len(best)}, summary rows={len(summary)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
