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


OUTPUT_DIR = Path("results/e11_natural_update_swap_probe")
FIGURE_DIR = Path("figures/e11_natural_update_swap_probe")
DISCUSSION_PATH = Path("discussion/e11_natural_update_swap_probe.md")
REFERENCE_TARGET_LAYER_RELATIVE_NORM = 1e-3
TARGET_LAYER_RELATIVE_NORMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
BUDGETS = ("fro", "op")


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


def proposed_optimizer_update(
    params: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> list[torch.Tensor]:
    before = snapshot_parameters(params)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    updates = [old - param.detach().clone() for old, param in zip(before, params)]
    with torch.no_grad():
        for param, old in zip(params, before):
            param.copy_(old)
    optimizer.load_state_dict(optimizer_state)
    return updates


def matrix_norm(matrix: torch.Tensor, budget: str) -> float:
    if budget == "fro":
        return float(torch.linalg.norm(matrix.detach()).cpu())
    if budget == "op":
        return float(torch.linalg.matrix_norm(matrix.detach(), ord=2).cpu())
    raise ValueError(f"unknown budget: {budget}")


def normalize_source_updates(
    evaluation_params: list[torch.nn.Parameter],
    source_updates: list[torch.Tensor],
    budget: str,
    target_relative_norm: float,
) -> list[torch.Tensor]:
    updates = []
    for param, source_update in zip(evaluation_params, source_updates):
        source_norm = max(matrix_norm(source_update, budget), 1e-300)
        param_norm = max(matrix_norm(param.detach(), budget), 1e-300)
        target_norm = target_relative_norm * param_norm
        updates.append((target_norm / source_norm) * source_update.detach())
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

        natural_updates = {
            algo: proposed_optimizer_update(params[algo], optimizers[algo])
            for algo in ["Adam", "Muon"]
        }

        if step in selected_steps(spec):
            for eval_algo in ["Adam", "Muon"]:
                other_algo = "Muon" if eval_algo == "Adam" else "Adam"
                for source_label, source_algo in [("own_update", eval_algo), ("other_update", other_algo)]:
                    for target in TARGET_LAYER_RELATIVE_NORMS:
                        for budget in BUDGETS:
                            updates = normalize_source_updates(
                                params[eval_algo],
                                natural_updates[source_algo],
                                budget,
                                target,
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
                                    "update_source": source_label,
                                    "source_algo": source_algo,
                                    "budget": budget,
                                    "target_layer_relative_norm": target,
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
        index=["problem_family", "base_setting", "target_layer_relative_norm", "eval_algo", "budget", "step", "seed"],
        columns="update_source",
        values=["delta_loss", "update_grad_inner", "update_grad_cosine"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{source}" for metric, source in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    groups: list[tuple[str, tuple, pd.DataFrame]] = []
    groups.extend(
        ("setting_target_budget_eval", key, group)
        for key, group in paired.groupby(
            ["problem_family", "base_setting", "target_layer_relative_norm", "budget", "eval_algo"],
            observed=True,
            sort=True,
        )
    )
    groups.extend(
        ("setting_target_budget_all_eval", (family, setting, target, budget, "All"), group)
        for (family, setting, target, budget), group in paired.groupby(
            ["problem_family", "base_setting", "target_layer_relative_norm", "budget"],
            observed=True,
            sort=True,
        )
    )
    groups.extend(
        ("target_budget_all", ("All", "All", target, budget, "All"), group)
        for (target, budget), group in paired.groupby(["target_layer_relative_norm", "budget"], observed=True, sort=True)
    )
    groups.extend(
        ("target_all", ("All", "All", target, "All", "All"), group)
        for target, group in paired.groupby("target_layer_relative_norm", observed=True, sort=True)
    )
    groups.append(("all", ("All", "All", "All", "All", "All"), paired))
    for group_type, key, group in groups:
        family, setting, target, budget, eval_algo = key
        for metric in ["update_grad_inner", "delta_loss", "update_grad_cosine"]:
            own = group[f"{metric}_own_update"]
            other = group[f"{metric}_other_update"]
            ratio_values = other / (own.abs() + 1e-300)
            delta_values = other - own
            signed_ratio, signed_lo, signed_hi = ci95(ratio_values)
            ratio, lo, hi = log_ratio_ci95(ratio_values)
            delta, delta_lo, delta_hi = ci95(delta_values)
            records.append(
                {
                    "group_type": group_type,
                    "problem_family": family,
                    "base_setting": setting,
                    "target_layer_relative_norm": target,
                    "budget": budget,
                    "eval_algo": eval_algo,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "other_positive_pairs": int((other > 0).sum()),
                    "other_better_pairs": int((other > own).sum()),
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
    return rows.groupby(
        ["problem_family", "base_setting", "eval_algo", "target_layer_relative_norm", "budget", "step", "update_source"],
        as_index=False,
        observed=True,
    ).agg(
        points=("seed", "size"),
        mean_delta_loss=("delta_loss", "mean"),
        mean_update_grad_inner=("update_grad_inner", "mean"),
        mean_update_grad_cosine=("update_grad_cosine", "mean"),
    )


def plot_swap_ratios(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "natural_update_swap_ratios.png"
    rows = summary[
        (summary["group_type"] == "setting_target_budget_all_eval")
        & (summary["metric"] == "update_grad_inner")
        & (summary["target_layer_relative_norm"] == REFERENCE_TARGET_LAYER_RELATIVE_NORM)
    ].copy()
    settings = sorted(rows["base_setting"].unique())
    budgets = list(BUDGETS)
    x_positions = {(setting, budget): idx * 3 + j for idx, setting in enumerate(settings) for j, budget in enumerate(budgets)}
    colors = {"fro": "#009E73", "op": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    for budget in budgets:
        sub = rows[rows["budget"] == budget].copy()
        x = np.array([x_positions[(setting, budget)] for setting in sub["base_setting"]], dtype=float)
        y = sub["mean_signed_other_over_own"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["signed_ratio_ci95_low"].to_numpy(dtype=float),
                sub["signed_ratio_ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=colors[budget], label=f"{budget} layer budget")
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.axhline(0.0, color="#999999", ls=":", lw=0.9)
    xticks = [idx * 3 + 0.5 for idx in range(len(settings))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [label.replace("Matrix sensing", "MS").replace("MF input", "MF").replace("Small MLP digits ", "MLP ") for label in settings],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("signed ratio: other update / own update")
    ax.set_title("One-step effect of swapping natural optimizer update vectors")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_sweep(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "natural_update_swap_target_sweep.png"
    rows = summary[
        (summary["group_type"] == "target_budget_all")
        & (summary["metric"] == "delta_loss")
    ].copy()
    colors = {"fro": "#009E73", "op": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for budget in BUDGETS:
        sub = rows[rows["budget"] == budget].sort_values("target_layer_relative_norm")
        x = sub["target_layer_relative_norm"].to_numpy(dtype=float)
        y = sub["mean_signed_other_over_own"].to_numpy(dtype=float)
        lo = sub["signed_ratio_ci95_low"].to_numpy(dtype=float)
        hi = sub["signed_ratio_ci95_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=colors[budget], label=f"{budget} layer budget")
        ax.fill_between(x, lo, hi, color=colors[budget], alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
    ax.axhline(0.0, color="#999999", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("target layer relative update norm")
    ax.set_ylabel("signed ratio: other update / own update")
    ax.set_title("Observed loss decrease under natural update-vector swaps")
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


def write_discussion(summary: pd.DataFrame, steps: pd.DataFrame, figure: Path, target_figure: Path) -> None:
    reference_core = summary[
        (summary["metric"] == "update_grad_inner")
        & (summary["group_type"] == "setting_target_budget_all_eval")
        & (summary["target_layer_relative_norm"] == REFERENCE_TARGET_LAYER_RELATIVE_NORM)
    ].copy()
    target_core = summary[
        (summary["metric"].isin(["update_grad_inner", "delta_loss"]))
        & (summary["group_type"].isin(["target_budget_all", "target_all", "all"]))
    ].copy()
    selected_steps = steps[
        (steps["update_source"].isin(["own_update", "other_update"]))
        & (steps["step"].isin([0, 1, 3, 5, 10]))
        & (steps["target_layer_relative_norm"] == REFERENCE_TARGET_LAYER_RELATIVE_NORM)
    ].copy()
    text = f"""# E11 Natural Update-Vector Swap Probe

## Purpose

The previous singular-vector swap probe swaps gradient-polar singular vectors. This probe asks the more direct optimizer-level question: at a fixed Adam or Muon state, what happens if we apply the other optimizer trajectory's natural proposed update vector instead of the state's own natural proposed update vector?

For each selected matched step, the script temporarily takes one optimizer step to extract the natural proposed update, restores both parameters and optimizer state, then evaluates own-update and other-update directions at the same evaluation state. Each layer is normalized under either Frobenius norm or operator norm across target relative update norms `{", ".join(f"{target:g}" for target in TARGET_LAYER_RELATIVE_NORMS)}`.

## Swap Ratio

Signed ratios below 1 mean that using the other optimizer's natural update vector gives less one-step first-order progress than using the evaluation state's own natural update vector. Negative ratios mean the swapped update is locally ascent while the own update is descent.

![Natural update-vector swap ratios](../{figure})

The setting-level table below uses the reference target `{REFERENCE_TARGET_LAYER_RELATIVE_NORM:g}`.

{markdown_table(reference_core, ["group_type", "problem_family", "base_setting", "target_layer_relative_norm", "budget", "eval_algo", "metric", "n_pairs", "other_positive_pairs", "other_better_pairs", "mean_signed_other_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Target-Scale Sweep

![Natural update-vector swap target sweep](../{target_figure})

The first-order metric `update_grad_inner` is nearly scale-invariant under this normalization, so the target sweep is most useful for checking the observed nonlinear `delta_loss` ratios.

{markdown_table(target_core, ["group_type", "target_layer_relative_norm", "budget", "eval_algo", "metric", "n_pairs", "other_positive_pairs", "other_better_pairs", "mean_signed_other_over_own", "signed_ratio_ci95_low", "signed_ratio_ci95_high"])}

## Step-Level Means

{markdown_table(selected_steps, ["problem_family", "base_setting", "eval_algo", "target_layer_relative_norm", "budget", "step", "update_source", "points", "mean_delta_loss", "mean_update_grad_inner", "mean_update_grad_cosine"])}

## Interpretation

This is a stronger intervention than the polar singular-vector swap because it uses each optimizer's actual proposed update vector, including its singular-value allocation and optimizer state. Across target scales, the aggregate other-update ratio stays below 1, so optimizer-specific update vectors are locally consequential beyond a single target choice. The setting-level rows are not monotone, however: some budget/evaluation-state cells favor the other update. The defensible reading is therefore not universal own-update dominance, but budget- and state-dependent trajectory specialization.

The result should be read as a one-step local direction test. It does not prove that a full optimizer trajectory would remain better or worse after a swap.

## Caveats

1. The intervention is still artificial: the other update is evaluated at a state where that optimizer did not naturally produce it.
2. Layerwise norm normalization removes update magnitude, so this tests direction and within-layer spectral shape rather than raw step size.
3. The target-scale sweep reduces but does not eliminate the representative-setting limitation.
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
    rows.to_csv(config.output_dir / "natural_update_swap_rows.csv", index=False)
    summary.to_csv(config.output_dir / "natural_update_swap_summary.csv", index=False)
    steps.to_csv(config.output_dir / "natural_update_swap_step_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_swap_ratios(summary)
    target_figure = plot_target_sweep(summary)
    write_discussion(summary, steps, figure, target_figure)
    print(f"saved natural update-vector swap probe to {config.output_dir}")
    print(f"rows={len(rows)}, summary rows={len(summary)}")
    print(f"figure: {figure}")
    print(f"target figure: {target_figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
