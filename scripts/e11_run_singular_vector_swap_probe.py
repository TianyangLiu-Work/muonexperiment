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


OUTPUT_DIR = Path("results/e11_singular_vector_swap_probe")
FIGURE_DIR = Path("figures/e11_singular_vector_swap_probe")
DISCUSSION_PATH = Path("discussion/e11_singular_vector_swap_probe.md")
TARGET_OP_RELATIVE_NORM = 1e-3


def swap_specs() -> tuple[ProblemSpec, ...]:
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
        specs=swap_specs(),
    )


def zero_grad(params: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    for param in params:
        param.grad = None


def collect_loss_grads(problem, params: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer) -> tuple[float, list[torch.Tensor]]:
    zero_grad(params, optimizer)
    loss = problem.loss()
    loss.backward()
    return float(loss.detach().cpu()), [param.grad.detach().clone() for param in params]


def polar_direction(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix.detach(), full_matrices=False)
    return u @ vh


def polar_updates_from_source(
    evaluation_params: list[torch.nn.Parameter],
    source_grads: list[torch.Tensor],
    target_relative_op_norm: float,
) -> list[torch.Tensor]:
    updates = []
    for param, source_grad in zip(evaluation_params, source_grads):
        direction = polar_direction(source_grad)
        direction_op = float(torch.linalg.matrix_norm(direction, ord=2).cpu())
        param_op = max(float(torch.linalg.matrix_norm(param.detach(), ord=2).cpu()), 1e-300)
        update_op = target_relative_op_norm * param_op
        updates.append((update_op / max(direction_op, 1e-300)) * direction)
    return updates


def evaluate_updates(problem, params: list[torch.nn.Parameter], before: list[torch.Tensor], updates: list[torch.Tensor]) -> float:
    with torch.no_grad():
        for param, old, update in zip(params, before, updates):
            param.copy_(old - update)
    loss_after = float(problem.loss().detach().cpu())
    with torch.no_grad():
        for param, old in zip(params, before):
            param.copy_(old)
    return loss_after


def update_alignment(eval_grads: list[torch.Tensor], updates: list[torch.Tensor]) -> dict:
    inner = 0.0
    grad_fro_sq = 0.0
    update_fro_sq = 0.0
    for grad, update in zip(eval_grads, updates):
        inner += float(torch.sum(grad.detach() * update.detach()).cpu())
        grad_fro_sq += float(torch.linalg.norm(grad.detach()).cpu()) ** 2
        update_fro_sq += float(torch.linalg.norm(update.detach()).cpu()) ** 2
    return {
        "update_grad_inner": inner,
        "update_grad_cosine": inner / max((grad_fro_sq * update_fro_sq) ** 0.5, 1e-300),
        "update_fro_norm": update_fro_sq**0.5,
    }


def selected_steps(spec: ProblemSpec) -> set[int]:
    return {0, 1, min(3, spec.steps), spec.steps}


def run_pair(spec: ProblemSpec, seed: int, config: ExperimentConfig) -> list[dict]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    problems = {algo: build_problem(spec, seed, device=device, dtype=dtype) for algo in ["Adam", "Muon"]}
    params = {algo: problems[algo].parameters() for algo in ["Adam", "Muon"]}
    optimizers = {algo: build_optimizer(algo, params[algo], spec.lr) for algo in ["Adam", "Muon"]}
    rows: list[dict] = []

    for step in range(spec.steps + 1):
        losses: dict[str, float] = {}
        grads: dict[str, list[torch.Tensor]] = {}
        before: dict[str, list[torch.Tensor]] = {}
        for algo in ["Adam", "Muon"]:
            problems[algo].set_train_step(step)
            loss, grad_list = collect_loss_grads(problems[algo], params[algo], optimizers[algo])
            losses[algo] = loss
            grads[algo] = grad_list
            before[algo] = snapshot_parameters(params[algo])

        if step in selected_steps(spec):
            for eval_algo in ["Adam", "Muon"]:
                other_algo = "Muon" if eval_algo == "Adam" else "Adam"
                for source_label, source_algo in [("own_vectors", eval_algo), ("other_vectors", other_algo)]:
                    updates = polar_updates_from_source(
                        params[eval_algo],
                        grads[source_algo],
                        TARGET_OP_RELATIVE_NORM,
                    )
                    loss_after = evaluate_updates(problems[eval_algo], params[eval_algo], before[eval_algo], updates)
                    alignment = update_alignment(grads[eval_algo], updates)
                    rows.append(
                        {
                            "problem_family": spec.family,
                            "base_setting": spec.setting,
                            "seed": seed,
                            "step": step,
                            "eval_algo": eval_algo,
                            "vector_source": source_label,
                            "source_algo": source_algo,
                            "target_relative_op_norm": TARGET_OP_RELATIVE_NORM,
                            "loss_before": losses[eval_algo],
                            "loss_after": loss_after,
                            "delta_loss": losses[eval_algo] - loss_after,
                            **alignment,
                        }
                    )

        if step < spec.steps:
            for algo in ["Adam", "Muon"]:
                optimizers[algo].step()

    return rows


def run_probe(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for spec in config.specs:
        for seed in config.seeds:
            rows.extend(run_pair(spec, seed, config))
    return pd.DataFrame(rows)


def ratio_summary(rows: pd.DataFrame) -> pd.DataFrame:
    pivot = rows.pivot_table(
        index=["problem_family", "base_setting", "eval_algo", "step", "seed"],
        columns="vector_source",
        values=["delta_loss", "update_grad_inner", "update_grad_cosine"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{source}" for metric, source in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("setting_eval", key, group)
        for key, group in paired.groupby(["problem_family", "base_setting", "eval_algo"], observed=True, sort=True)
    )
    groups.extend(
        ("setting_all_eval", (family, setting, "All"), group)
        for (family, setting), group in paired.groupby(["problem_family", "base_setting"], observed=True, sort=True)
    )
    groups.append(("all", ("All", "All", "All"), paired))
    for group_type, key, group in groups:
        family, setting, eval_algo = key
        for metric in ["update_grad_inner", "delta_loss", "update_grad_cosine"]:
            ratio_values = group[f"{metric}_other_vectors"] / (group[f"{metric}_own_vectors"].abs() + 1e-300)
            delta_values = group[f"{metric}_other_vectors"] - group[f"{metric}_own_vectors"]
            signed_ratio, signed_lo, signed_hi = ci95(ratio_values)
            ratio, lo, hi = log_ratio_ci95(ratio_values)
            delta, delta_lo, delta_hi = ci95(delta_values)
            records.append(
                {
                    "group_type": group_type,
                    "problem_family": family,
                    "base_setting": setting,
                    "eval_algo": eval_algo,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "other_positive_pairs": int((group[f"{metric}_other_vectors"] > 0).sum()),
                    "other_better_pairs": int((group[f"{metric}_other_vectors"] > group[f"{metric}_own_vectors"]).sum()),
                    "mean_signed_other_over_own": signed_ratio,
                    "signed_ratio_ci95_low": signed_lo,
                    "signed_ratio_ci95_high": signed_hi,
                    "geomean_other_over_own": ratio,
                    "ratio_ci95_low": lo,
                    "ratio_ci95_high": hi,
                    "positive_ratio_pairs": int((ratio_values > 0).sum()),
                    "ratio_ci95_above_one": bool(lo > 1.0) if np.isfinite(lo) else False,
                    "ratio_ci95_below_one": bool(hi < 1.0) if np.isfinite(hi) else False,
                    "mean_delta_other_minus_own": delta,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                }
            )
    return pd.DataFrame(records)


def step_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["problem_family", "base_setting", "eval_algo", "step", "vector_source"], as_index=False, observed=True).agg(
        points=("seed", "size"),
        mean_delta_loss=("delta_loss", "mean"),
        mean_update_grad_inner=("update_grad_inner", "mean"),
        mean_update_grad_cosine=("update_grad_cosine", "mean"),
    )


def plot_swap_ratios(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "singular_vector_swap_ratios.png"
    rows = summary[
        (summary["group_type"] == "setting_eval")
        & (summary["metric"] == "update_grad_inner")
    ].copy()
    settings = sorted(rows["base_setting"].unique())
    x_positions = {setting: idx for idx, setting in enumerate(settings)}
    offsets = {"Adam": -0.13, "Muon": 0.13}
    colors = {"Adam": "#D55E00", "Muon": "#0072B2"}
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    for eval_algo in ["Adam", "Muon"]:
        sub = rows[rows["eval_algo"] == eval_algo].copy()
        x = np.array([x_positions[setting] + offsets[eval_algo] for setting in sub["base_setting"]], dtype=float)
        y = sub["mean_signed_other_over_own"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["signed_ratio_ci95_low"].to_numpy(dtype=float),
                sub["signed_ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=colors[eval_algo], label=f"evaluated at {eval_algo} state")
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(
        [label.replace("Matrix sensing", "MS").replace("MF input", "MF").replace("Small MLP digits ", "MLP ") for label in settings],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("signed ratio: other vectors / own vectors")
    ax.set_title("One-step effect of swapping polar singular vectors")
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


def write_discussion(summary: pd.DataFrame, steps: pd.DataFrame, figure: Path) -> None:
    core = summary[
        (summary["metric"] == "update_grad_inner")
        & (summary["group_type"].isin(["setting_eval", "setting_all_eval", "all"]))
    ].copy()
    selected_steps = steps[
        (steps["vector_source"].isin(["own_vectors", "other_vectors"]))
        & (steps["step"].isin([0, 1, 3, 5, 10]))
    ].copy()
    text = f"""# E11 Singular-Vector Swap Probe

## Purpose

The singular-vector trajectory diagnostic shows that Adam and Muon can move into different gradient subspaces, but that is only diagnostic. This probe asks a more causal one-step question: at a fixed Adam or Muon state, does replacing the state's own gradient polar singular vectors with the other optimizer's matched-state gradient polar singular vectors reduce first-order progress?

For each layer, both candidate updates use a flat polar spectrum and the same per-layer operator-norm budget `{TARGET_OP_RELATIVE_NORM}`. Only the singular vectors are swapped.

## Swap Ratio

Signed ratios below 1 mean that using the other optimizer's singular vectors gives less one-step first-order progress than using the state's own singular vectors. Negative ratios mean the swapped-vector update is locally ascent while the own-vector update is descent.

![Singular-vector swap ratios](../{figure})

{markdown_table(core, ["group_type", "problem_family", "base_setting", "eval_algo", "metric", "n_pairs", "other_positive_pairs", "other_better_pairs", "mean_signed_other_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Step-Level Means

{markdown_table(selected_steps, ["problem_family", "base_setting", "eval_algo", "step", "vector_source", "points", "mean_delta_loss", "mean_update_grad_inner", "mean_update_grad_cosine"])}

## Interpretation

This is still an artificial intervention, but it is stronger than measuring subspace overlap alone. If other-vector updates are worse than own-vector updates at the same state and norm budget, then the natural Adam/Muon singular-vector divergence is relevant to one-step descent, not merely a visual trajectory difference.

The result should be read together with the spectral-allocation probe. The allocation probe fixes singular vectors and changes singular values; this swap probe fixes the flat/polar singular values and changes singular vectors.

## Caveats

1. This is a one-step artificial intervention, not a natural optimizer.
2. It swaps gradient polar singular vectors, not full Adam update singular vectors.
3. It uses a small representative setting set and a single operator-norm target.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = run_probe(config)
    summary = ratio_summary(rows)
    steps = step_summary(rows)
    rows.to_csv(config.output_dir / "swap_probe_rows.csv", index=False)
    summary.to_csv(config.output_dir / "swap_ratio_summary.csv", index=False)
    steps.to_csv(config.output_dir / "swap_step_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_swap_ratios(summary)
    write_discussion(summary, steps, figure)
    print(f"saved singular-vector swap probe to {config.output_dir}")
    print(f"rows={len(rows)}, summary rows={len(summary)}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
