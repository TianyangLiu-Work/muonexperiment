from __future__ import annotations

import copy
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


OUTPUT_DIR = Path("results/e11_optimizer_switch_reset_control")
FIGURE_DIR = Path("figures/e11_optimizer_switch_reset_control")
DISCUSSION_PATH = Path("discussion/e11_optimizer_switch_reset_control.md")


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


def selected_checkpoints(spec: ProblemSpec) -> tuple[int, ...]:
    return (0, 1, min(3, spec.steps - 1))


def copy_parameters(params: list[torch.nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)


def train_steps(problem, params: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer, steps: int, start_step: int = 0) -> None:
    for offset in range(steps):
        problem.set_train_step(start_step + offset)
        optimizer.zero_grad(set_to_none=True)
        loss = problem.loss()
        loss.backward()
        optimizer.step()


def checkpoint_state(spec, seed: int, source_algo: str, checkpoint_step: int, *, device: torch.device, dtype: torch.dtype):
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    params = problem.parameters()
    optimizer = build_optimizer(source_algo, params, spec.lr)
    train_steps(problem, params, optimizer, checkpoint_step, start_step=0)
    return snapshot_parameters(params), copy.deepcopy(optimizer.state_dict())


def continue_variant(
    spec,
    seed: int,
    checkpoint_params: list[torch.Tensor],
    optimizer_state: dict,
    source_algo: str,
    variant: str,
    checkpoint_step: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    if variant == "own_preserved":
        continuation_algo = source_algo
        load_state = True
    elif variant == "own_fresh":
        continuation_algo = source_algo
        load_state = False
    elif variant == "switched_fresh":
        continuation_algo = "Muon" if source_algo == "Adam" else "Adam"
        load_state = False
    else:
        raise ValueError(f"unknown continuation variant: {variant}")

    problem = build_problem(spec, seed, device=device, dtype=dtype)
    params = problem.parameters()
    copy_parameters(params, checkpoint_params)
    optimizer = build_optimizer(continuation_algo, params, spec.lr)
    if load_state:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))

    problem.set_train_step(checkpoint_step)
    loss_before = float(problem.loss().detach().cpu())
    recovery_before = float(problem.recovery_error())
    train_steps(problem, params, optimizer, spec.steps - checkpoint_step, start_step=checkpoint_step)
    problem.set_train_step(spec.steps)
    loss_after = float(problem.loss().detach().cpu())
    recovery_after = float(problem.recovery_error())
    return {
        "continuation_variant": variant,
        "continuation_algo": continuation_algo,
        "loaded_source_optimizer_state": load_state,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "total_decrease": loss_before - loss_after,
        "recovery_before": recovery_before,
        "recovery_after": recovery_after,
    }


def run_probe(config: ExperimentConfig) -> pd.DataFrame:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows = []
    for spec in config.specs:
        for seed in config.seeds:
            for source_algo in ["Adam", "Muon"]:
                for checkpoint_step in selected_checkpoints(spec):
                    params, optimizer_state = checkpoint_state(
                        spec,
                        seed,
                        source_algo,
                        checkpoint_step,
                        device=device,
                        dtype=dtype,
                    )
                    for variant in ["own_preserved", "own_fresh", "switched_fresh"]:
                        result = continue_variant(
                            spec,
                            seed,
                            params,
                            optimizer_state,
                            source_algo,
                            variant,
                            checkpoint_step,
                            device=device,
                            dtype=dtype,
                        )
                        rows.append(
                            {
                                "problem_family": spec.family,
                                "base_setting": spec.setting,
                                "seed": seed,
                                "source_algo": source_algo,
                                "checkpoint_step": checkpoint_step,
                                "remaining_steps": spec.steps - checkpoint_step,
                                **result,
                            }
                        )
    return pd.DataFrame(rows)


def pairwise_summary(rows: pd.DataFrame) -> pd.DataFrame:
    pivot = rows.pivot_table(
        index=["problem_family", "base_setting", "source_algo", "checkpoint_step", "seed"],
        columns="continuation_variant",
        values=["total_decrease", "loss_after", "recovery_after"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{variant}" for metric, variant in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    comparisons = {
        "switched_fresh_over_own_fresh": ("switched_fresh", "own_fresh"),
        "own_fresh_over_own_preserved": ("own_fresh", "own_preserved"),
    }
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("setting_source_all_checkpoints", key, group)
        for key, group in paired.groupby(["problem_family", "base_setting", "source_algo"], observed=True, sort=True)
    )
    groups.extend(
        ("source_all", ("All", "All", source_algo), group)
        for source_algo, group in paired.groupby("source_algo", observed=True, sort=True)
    )
    groups.append(("all", ("All", "All", "All"), paired))
    for group_type, key, group in groups:
        family, setting, source_algo = key
        for comparison, (numerator_variant, denominator_variant) in comparisons.items():
            for metric in ["total_decrease", "loss_after", "recovery_after"]:
                numerator = group[f"{metric}_{numerator_variant}"]
                denominator = group[f"{metric}_{denominator_variant}"]
                ratio_values = numerator / (denominator.abs() + 1e-300)
                if metric == "total_decrease":
                    numerator_better = numerator > denominator
                else:
                    numerator_better = numerator < denominator
                signed_mean, signed_lo, signed_hi = ci95(ratio_values)
                ratio, lo, hi = log_ratio_ci95(ratio_values)
                delta, delta_lo, delta_hi = ci95(numerator - denominator)
                records.append(
                    {
                        "group_type": group_type,
                        "problem_family": family,
                        "base_setting": setting,
                        "source_algo": source_algo,
                        "comparison": comparison,
                        "metric": metric,
                        "n_pairs": int(len(group)),
                        "numerator_better_pairs": int(numerator_better.sum()),
                        "mean_signed_ratio": signed_mean,
                        "signed_ratio_ci95_low": signed_lo,
                        "signed_ratio_ci95_high": signed_hi,
                        "geomean_ratio": ratio,
                        "ratio_ci95_low": lo,
                        "ratio_ci95_high": hi,
                        "positive_ratio_pairs": int((ratio_values > 0).sum()),
                        "mean_delta": delta,
                        "delta_ci95_low": delta_lo,
                        "delta_ci95_high": delta_hi,
                    }
                )
    return pd.DataFrame(records)


def plot_reset_control(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "optimizer_switch_reset_control.png"
    rows = summary[
        (summary["group_type"] == "setting_source_all_checkpoints")
        & (summary["metric"] == "loss_after")
        & (summary["comparison"] == "switched_fresh_over_own_fresh")
    ].copy()
    settings = sorted(rows["base_setting"].unique())
    colors = {"Adam": "#D55E00", "Muon": "#0072B2"}
    offsets = {"Adam": -0.14, "Muon": 0.14}
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    for source_algo in ["Adam", "Muon"]:
        sub = rows[rows["source_algo"] == source_algo]
        x = np.array([settings.index(setting) + offsets[source_algo] for setting in sub["base_setting"]], dtype=float)
        y = sub["mean_signed_ratio"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["signed_ratio_ci95_low"].to_numpy(dtype=float),
                sub["signed_ratio_ci95_high"].to_numpy(dtype=float) - y,
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
    ax.set_ylabel("final-loss ratio: switched fresh / own fresh")
    ax.set_title("Optimizer switch reset control")
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


def write_discussion(summary: pd.DataFrame, figure: Path) -> None:
    switch_core = summary[
        (summary["metric"] == "loss_after")
        & (summary["comparison"] == "switched_fresh_over_own_fresh")
        & (summary["group_type"].isin(["setting_source_all_checkpoints", "source_all", "all"]))
    ].copy()
    reset_core = summary[
        (summary["metric"] == "loss_after")
        & (summary["comparison"] == "own_fresh_over_own_preserved")
        & (summary["group_type"].isin(["source_all", "all"]))
    ].copy()
    text = f"""# E11 Optimizer Switch Reset-Control Probe

## Purpose

The optimizer-switch probe showed that Muon checkpoints often continue better with Adam, but switching to Adam also starts Adam with fresh moment state. This reset-control probe separates two effects at the same checkpoint state:

1. `own_preserved`: continue with the original optimizer and its optimizer state.
2. `own_fresh`: continue with the original optimizer type but reset optimizer state.
3. `switched_fresh`: continue with the other optimizer type with fresh optimizer state.

The main comparison is `switched_fresh / own_fresh`, which controls for optimizer-state reset as much as possible.

## Switched Fresh Vs Own Fresh

For final loss, ratios below 1 mean switching optimizer type with fresh state gives lower final loss than using the source optimizer type with fresh state.

![Optimizer switch reset control](../{figure})

{markdown_table(switch_core, ["group_type", "problem_family", "base_setting", "source_algo", "comparison", "metric", "n_pairs", "numerator_better_pairs", "mean_signed_ratio", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Reset Effect

`own_fresh / own_preserved` measures the effect of resetting the source optimizer state while keeping the optimizer type fixed. For final loss, ratios above 1 mean reset hurts.

{markdown_table(reset_core, ["group_type", "source_algo", "comparison", "metric", "n_pairs", "numerator_better_pairs", "mean_signed_ratio", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Interpretation

This probe is cleaner than the raw optimizer switch for separating optimizer identity from optimizer-state reset. It still uses short continuations, so it should be read as a local trajectory intervention rather than a final training claim.

## Caveats

1. `own_fresh` is not a natural continuation; it intentionally resets optimizer state.
2. Muon has no moment state, so reset control is more important for Adam checkpoints than for Muon checkpoints.
3. The result is still limited to the representative short-horizon settings.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = run_probe(config)
    summary = pairwise_summary(rows)
    rows.to_csv(config.output_dir / "optimizer_switch_reset_rows.csv", index=False)
    summary.to_csv(config.output_dir / "optimizer_switch_reset_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_reset_control(summary)
    write_discussion(summary, figure)
    print(f"saved optimizer switch reset-control probe to {config.output_dir}")
    print(f"rows={len(rows)}, summary rows={len(summary)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
