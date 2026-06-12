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


OUTPUT_DIR = Path("results/e11_singular_vector_trajectory")
FIGURE_DIR = Path("figures/e11_singular_vector_trajectory")
DISCUSSION_PATH = Path("discussion/e11_singular_vector_trajectory.md")
TOP_K = 5


def trajectory_specs() -> tuple[ProblemSpec, ...]:
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
        specs=trajectory_specs(),
    )


def zero_grad(params: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    for param in params:
        param.grad = None


def top_subspaces(matrix: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    values = matrix.detach()
    u, _, vh = torch.linalg.svd(values, full_matrices=False)
    k = min(top_k, u.shape[1], vh.shape[0])
    return u[:, :k], vh[:k, :].T, k


def subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    k = min(a.shape[1], b.shape[1])
    if k == 0:
        return float("nan")
    value = torch.linalg.norm(a[:, :k].T @ b[:, :k], ord="fro") ** 2 / k
    return float(value.detach().cpu())


def matrix_pair_overlap(a: torch.Tensor, b: torch.Tensor, top_k: int = TOP_K) -> dict:
    ua, va, ka = top_subspaces(a, top_k)
    ub, vb, kb = top_subspaces(b, top_k)
    k = min(ka, kb)
    return {
        "top_k": int(k),
        "left_overlap": subspace_overlap(ua[:, :k], ub[:, :k]),
        "right_overlap": subspace_overlap(va[:, :k], vb[:, :k]),
    }


def collect_grads(problem, params: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer) -> tuple[float, list[torch.Tensor]]:
    zero_grad(params, optimizer)
    loss = problem.loss()
    loss.backward()
    return float(loss.detach().cpu()), [param.grad.detach().clone() for param in params]


def run_pair(spec: ProblemSpec, seed: int, config: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    problems = {algo: build_problem(spec, seed, device=device, dtype=dtype) for algo in ["Adam", "Muon"]}
    params = {algo: problems[algo].parameters() for algo in ["Adam", "Muon"]}
    optimizers = {algo: build_optimizer(algo, params[algo], spec.lr) for algo in ["Adam", "Muon"]}
    gradient_rows: list[dict] = []
    update_rows: list[dict] = []

    for step in range(spec.steps + 1):
        losses: dict[str, float] = {}
        grads: dict[str, list[torch.Tensor]] = {}
        for algo in ["Adam", "Muon"]:
            problems[algo].set_train_step(step)
            loss, grad_list = collect_grads(problems[algo], params[algo], optimizers[algo])
            losses[algo] = loss
            grads[algo] = grad_list

        for layer, (grad_adam, grad_muon, param_adam, param_muon) in enumerate(
            zip(grads["Adam"], grads["Muon"], params["Adam"], params["Muon"]),
            start=1,
        ):
            grad_overlap = matrix_pair_overlap(grad_adam, grad_muon)
            param_overlap = matrix_pair_overlap(param_adam.detach(), param_muon.detach())
            gradient_rows.append(
                {
                    "problem_family": spec.family,
                    "base_setting": spec.setting,
                    "seed": seed,
                    "step": step,
                    "layer": layer,
                    "adam_loss": losses["Adam"],
                    "muon_loss": losses["Muon"],
                    "top_k": grad_overlap["top_k"],
                    "grad_left_overlap": grad_overlap["left_overlap"],
                    "grad_right_overlap": grad_overlap["right_overlap"],
                    "grad_mean_overlap": 0.5 * (grad_overlap["left_overlap"] + grad_overlap["right_overlap"]),
                    "param_left_overlap": param_overlap["left_overlap"],
                    "param_right_overlap": param_overlap["right_overlap"],
                    "param_mean_overlap": 0.5 * (param_overlap["left_overlap"] + param_overlap["right_overlap"]),
                }
            )

        if step < spec.steps:
            before = {algo: snapshot_parameters(params[algo]) for algo in ["Adam", "Muon"]}
            for algo in ["Adam", "Muon"]:
                optimizers[algo].step()
            for layer, (old_adam, old_muon, param_adam, param_muon) in enumerate(
                zip(before["Adam"], before["Muon"], params["Adam"], params["Muon"]),
                start=1,
            ):
                update_adam = old_adam - param_adam.detach()
                update_muon = old_muon - param_muon.detach()
                update_overlap = matrix_pair_overlap(update_adam, update_muon)
                update_rows.append(
                    {
                        "problem_family": spec.family,
                        "base_setting": spec.setting,
                        "seed": seed,
                        "step": step,
                        "layer": layer,
                        "top_k": update_overlap["top_k"],
                        "update_left_overlap": update_overlap["left_overlap"],
                        "update_right_overlap": update_overlap["right_overlap"],
                        "update_mean_overlap": 0.5 * (update_overlap["left_overlap"] + update_overlap["right_overlap"]),
                    }
                )

    return gradient_rows, update_rows


def run_trajectory(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    gradient_rows: list[dict] = []
    update_rows: list[dict] = []
    for spec in config.specs:
        for seed in config.seeds:
            grad_part, update_part = run_pair(spec, seed, config)
            gradient_rows.extend(grad_part)
            update_rows.extend(update_part)
    return pd.DataFrame(gradient_rows), pd.DataFrame(update_rows)


def step_summary(gradient_rows: pd.DataFrame, update_rows: pd.DataFrame) -> pd.DataFrame:
    grad_summary = gradient_rows.groupby(["problem_family", "base_setting", "step"], as_index=False, observed=True).agg(
        layers=("layer", "size"),
        mean_grad_overlap=("grad_mean_overlap", "mean"),
        mean_grad_left_overlap=("grad_left_overlap", "mean"),
        mean_grad_right_overlap=("grad_right_overlap", "mean"),
        mean_param_overlap=("param_mean_overlap", "mean"),
        mean_adam_loss=("adam_loss", "mean"),
        mean_muon_loss=("muon_loss", "mean"),
    )
    update_summary = update_rows.groupby(["problem_family", "base_setting", "step"], as_index=False, observed=True).agg(
        mean_update_overlap=("update_mean_overlap", "mean"),
        mean_update_left_overlap=("update_left_overlap", "mean"),
        mean_update_right_overlap=("update_right_overlap", "mean"),
    )
    return grad_summary.merge(update_summary, on=["problem_family", "base_setting", "step"], how="left")


def final_summary(steps: pd.DataFrame) -> pd.DataFrame:
    final = steps.sort_values("step").groupby(["problem_family", "base_setting"], observed=True).tail(1).copy()
    initial = steps[steps["step"] == 0][["problem_family", "base_setting", "mean_grad_overlap"]].rename(
        columns={"mean_grad_overlap": "initial_grad_overlap"}
    )
    result = final.merge(initial, on=["problem_family", "base_setting"], how="left")
    result["grad_overlap_drop"] = result["initial_grad_overlap"] - result["mean_grad_overlap"]
    return result[
        [
            "problem_family",
            "base_setting",
            "step",
            "initial_grad_overlap",
            "mean_grad_overlap",
            "grad_overlap_drop",
            "mean_param_overlap",
            "mean_adam_loss",
            "mean_muon_loss",
        ]
    ]


def plot_overlap_trajectory(steps: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "singular_vector_overlap_trajectory.png"
    settings = sorted(steps["base_setting"].unique())
    fig, axes = plt.subplots(len(settings), 1, figsize=(7.4, 2.6 * len(settings)), sharex=False, squeeze=False)
    for ax, setting in zip(axes.ravel(), settings):
        sub = steps[steps["base_setting"] == setting].sort_values("step")
        ax.plot(sub["step"], sub["mean_grad_overlap"], marker="o", color="#0072B2", label="gradient subspace")
        ax.plot(sub["step"], sub["mean_param_overlap"], marker="s", color="#009E73", label="parameter subspace")
        ax.plot(sub["step"], sub["mean_update_overlap"], marker="^", color="#D55E00", label="update subspace")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(setting)
        ax.set_ylabel("top-k overlap")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Adam/Muon singular-vector trajectory overlap", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.4g}")
    return display.to_markdown(index=False)


def write_discussion(steps: pd.DataFrame, final: pd.DataFrame, figure: Path) -> None:
    final_table = markdown_table(
        final,
        [
            "problem_family",
            "base_setting",
            "step",
            "initial_grad_overlap",
            "mean_grad_overlap",
            "grad_overlap_drop",
            "mean_param_overlap",
            "mean_adam_loss",
            "mean_muon_loss",
        ],
    )
    selected_steps = steps[steps["step"].isin([0, 1, 3, 5, 10])].copy()
    step_table = markdown_table(
        selected_steps,
        [
            "base_setting",
            "step",
            "mean_grad_overlap",
            "mean_param_overlap",
            "mean_update_overlap",
            "mean_adam_loss",
            "mean_muon_loss",
        ],
    )
    text = f"""# E11 Singular-Vector Trajectory Diagnostic

## Purpose

The spectral-allocation probe fixes gradient singular vectors, so it isolates singular-value allocation but not natural trajectory effects. This diagnostic asks whether Adam and Muon move into different gradient, parameter, and update singular subspaces over time.

For each matched seed, both optimizers start from the same initialization. At each step and layer, the diagnostic computes top-{TOP_K} left/right singular subspace overlap between Adam and Muon. A value near 1 means the subspaces match; a value near 0 means they are nearly orthogonal.

## Trajectory Plot

![Singular-vector overlap trajectory](../{figure})

## Final Overlap Summary

{final_table}

## Selected Step Summary

{step_table}

## Interpretation

This diagnostic separates two mechanisms. If gradient subspace overlap stays high, then Adam and Muon mostly differ by singular-value allocation within a shared geometry. If gradient subspace overlap decays, then the optimizers also move into different singular-vector geometries, and the one-step spectral-allocation probe is only part of the story.

The current representative settings split into two regimes. MF-with-input keeps high gradient and parameter subspace overlap through the short horizon, so the Adam/Muon difference there is closer to a within-geometry singular-value allocation difference. Matrix Sensing and both SmallMLP widths show much larger subspace divergence, so their Adam/Muon differences include trajectory-level singular-vector geometry, not just singular-value allocation at a fixed state.

The update subspace overlap is expected to be lower because Adam and Muon implement different update maps even at the same state. The more important signal is whether the gradient and parameter subspaces diverge after several natural steps.

## Caveats

1. Top-k subspace overlap is a coarse diagnostic and ignores lower singular directions.
2. It measures matched Adam/Muon divergence, not causality.
3. It is still short-horizon and uses representative settings only.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    gradient_rows, update_rows = run_trajectory(config)
    steps = step_summary(gradient_rows, update_rows)
    final = final_summary(steps)
    gradient_rows.to_csv(config.output_dir / "gradient_subspace_rows.csv", index=False)
    update_rows.to_csv(config.output_dir / "update_subspace_rows.csv", index=False)
    steps.to_csv(config.output_dir / "subspace_step_summary.csv", index=False)
    final.to_csv(config.output_dir / "subspace_final_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_overlap_trajectory(steps)
    write_discussion(steps, final, figure)
    print(f"saved singular-vector trajectory diagnostic to {config.output_dir}")
    print(f"gradient rows={len(gradient_rows)}, update rows={len(update_rows)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
