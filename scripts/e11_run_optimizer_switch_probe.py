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


OUTPUT_DIR = Path("results/e11_optimizer_switch_probe")
FIGURE_DIR = Path("figures/e11_optimizer_switch_probe")
DISCUSSION_PATH = Path("discussion/e11_optimizer_switch_probe.md")


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


def checkpoint_state(
    spec: ProblemSpec,
    seed: int,
    source_algo: str,
    checkpoint_step: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], dict, float, float]:
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    params = problem.parameters()
    optimizer = build_optimizer(source_algo, params, spec.lr)
    train_steps(problem, params, optimizer, checkpoint_step, start_step=0)
    problem.set_train_step(checkpoint_step)
    loss = float(problem.loss().detach().cpu())
    recovery = float(problem.recovery_error())
    return snapshot_parameters(params), copy.deepcopy(optimizer.state_dict()), loss, recovery


def continue_from_checkpoint(
    spec: ProblemSpec,
    seed: int,
    checkpoint_params: list[torch.Tensor],
    optimizer_state: dict,
    source_algo: str,
    continuation_algo: str,
    checkpoint_step: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    params = problem.parameters()
    copy_parameters(params, checkpoint_params)
    optimizer = build_optimizer(continuation_algo, params, spec.lr)
    loaded_source_state = source_algo == continuation_algo
    if loaded_source_state:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    problem.set_train_step(checkpoint_step)
    loss_before = float(problem.loss().detach().cpu())
    recovery_before = float(problem.recovery_error())
    train_steps(problem, params, optimizer, spec.steps - checkpoint_step, start_step=checkpoint_step)
    problem.set_train_step(spec.steps)
    loss_after = float(problem.loss().detach().cpu())
    recovery_after = float(problem.recovery_error())
    return {
        "continuation_algo": continuation_algo,
        "loaded_source_optimizer_state": loaded_source_state,
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
                    params, optimizer_state, checkpoint_loss, checkpoint_recovery = checkpoint_state(
                        spec,
                        seed,
                        source_algo,
                        checkpoint_step,
                        device=device,
                        dtype=dtype,
                    )
                    for continuation_algo in ["Adam", "Muon"]:
                        result = continue_from_checkpoint(
                            spec,
                            seed,
                            params,
                            optimizer_state,
                            source_algo,
                            continuation_algo,
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
                                "checkpoint_loss": checkpoint_loss,
                                "checkpoint_recovery": checkpoint_recovery,
                                "continuation_type": "own" if continuation_algo == source_algo else "switched",
                                **result,
                            }
                        )
    return pd.DataFrame(rows)


def ratio_summary(rows: pd.DataFrame) -> pd.DataFrame:
    pivot = rows.pivot_table(
        index=["problem_family", "base_setting", "source_algo", "checkpoint_step", "seed"],
        columns="continuation_type",
        values=["total_decrease", "loss_after", "recovery_after"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{kind}" for metric, kind in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("setting_source_checkpoint", key, group)
        for key, group in paired.groupby(["problem_family", "base_setting", "source_algo", "checkpoint_step"], observed=True, sort=True)
    )
    groups.extend(
        ("setting_source_all_checkpoints", (family, setting, source_algo, "All"), group)
        for (family, setting, source_algo), group in paired.groupby(["problem_family", "base_setting", "source_algo"], observed=True, sort=True)
    )
    groups.extend(
        ("source_all", ("All", "All", source_algo, "All"), group)
        for source_algo, group in paired.groupby("source_algo", observed=True, sort=True)
    )
    groups.append(("all", ("All", "All", "All", "All"), paired))
    for group_type, key, group in groups:
        family, setting, source_algo, checkpoint_step = key
        for metric in ["total_decrease", "loss_after", "recovery_after"]:
            own = group[f"{metric}_own"]
            switched = group[f"{metric}_switched"]
            if metric == "total_decrease":
                ratio_values = switched / (own.abs() + 1e-300)
                switched_better = switched > own
            else:
                ratio_values = switched / (own.abs() + 1e-300)
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
                    "checkpoint_step": checkpoint_step,
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


def plot_switch_summary(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "optimizer_switch_total_decrease.png"
    rows = summary[
        (summary["group_type"] == "setting_source_all_checkpoints")
        & (summary["metric"] == "total_decrease")
    ].copy()
    settings = sorted(rows["base_setting"].unique())
    colors = {"Adam": "#D55E00", "Muon": "#0072B2"}
    offsets = {"Adam": -0.14, "Muon": 0.14}
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    for source_algo in ["Adam", "Muon"]:
        sub = rows[rows["source_algo"] == source_algo]
        x = np.array([settings.index(setting) + offsets[source_algo] for setting in sub["base_setting"]], dtype=float)
        y = sub["mean_signed_switched_over_own"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["signed_ratio_ci95_low"].to_numpy(dtype=float),
                sub["signed_ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=colors[source_algo], label=f"checkpoint from {source_algo}")
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.axhline(0.0, color="#999999", ls=":", lw=0.9)
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(
        [label.replace("Matrix sensing", "MS").replace("MF input", "MF").replace("Small MLP digits ", "MLP ") for label in settings],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("ratio: switched continuation / own continuation")
    ax.set_title("Trajectory-level optimizer switch: total loss decrease")
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
    core = summary[
        (summary["metric"] == "total_decrease")
        & (summary["group_type"].isin(["setting_source_all_checkpoints", "source_all", "all"]))
    ].copy()
    final_loss = summary[
        (summary["metric"] == "loss_after")
        & (summary["group_type"].isin(["source_all", "all"]))
    ].copy()
    text = f"""# E11 Optimizer Switch Probe

## Purpose

The natural update-vector swap is a one-step intervention. This probe asks a trajectory-level question: after Adam or Muon has produced a checkpoint state, what happens if the remaining short horizon is continued with the other optimizer instead of the original optimizer?

For own continuations, the original optimizer state is preserved. For switched continuations, the other optimizer is initialized fresh at the checkpoint state. This is intentionally closer to an actual optimizer switch, but it means the Adam state reset is part of the intervention.

## Total Decrease Ratio

Ratios below 1 mean the switched continuation produces less remaining-horizon loss decrease than continuing with the checkpoint optimizer.

![Optimizer switch total decrease](../{figure})

{markdown_table(core, ["group_type", "problem_family", "base_setting", "source_algo", "checkpoint_step", "metric", "n_pairs", "switched_better_pairs", "mean_signed_switched_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Final Loss Ratio

For final loss, ratios below 1 mean switching gives lower final loss.

{markdown_table(final_loss, ["group_type", "source_algo", "metric", "n_pairs", "switched_better_pairs", "mean_signed_switched_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Interpretation

This probe is more trajectory-level than the one-step update-vector swap, but also less clean because switching optimizer changes optimizer state. It is best read as a practical intervention test: does a state produced by one optimizer continue better with that same optimizer or with the other optimizer?

## Caveats

1. Switching to Adam starts Adam with fresh moment state, so Adam switches are not only direction changes.
2. The horizons are still short and use the same representative setting set as the natural update-vector swap.
3. A negative own or switched remaining decrease makes signed ratios harder to interpret; pair counts and final-loss ratios should be read alongside total-decrease ratios.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = run_probe(config)
    summary = ratio_summary(rows)
    rows.to_csv(config.output_dir / "optimizer_switch_rows.csv", index=False)
    summary.to_csv(config.output_dir / "optimizer_switch_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_switch_summary(summary)
    write_discussion(summary, figure)
    print(f"saved optimizer switch probe to {config.output_dir}")
    print(f"rows={len(rows)}, summary rows={len(summary)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
