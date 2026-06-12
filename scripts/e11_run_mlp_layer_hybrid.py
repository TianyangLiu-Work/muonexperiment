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
from e11_condition_geometry.optimizers import ExactMuon
from e11_condition_geometry.runner import (
    append_step_diagnostics,
    apply_update_diagnostics,
    build_problem,
    dtype_from_name,
    relative_update_norm,
    rescale_update,
    snapshot_parameters,
)
from e11_condition_geometry.diagnostics import json_loads_nested
from e11_condition_geometry.statistics import ci95, first_order_calibration_summary, log_ratio_ci95


OUTPUT_DIR = Path("results/e11_mlp_layer_hybrid")
FIGURE_DIR = Path("figures/e11_mlp_layer_hybrid")
DISCUSSION_PATH = Path("discussion/e11_mlp_layer_hybrid.md")
ALGOS = ("Adam", "Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond")


def hybrid_specs() -> tuple[ProblemSpec, ...]:
    specs = []
    for num_samples in [128, 1024]:
        for hidden_dim in [8, 16, 32, 64, 128]:
            specs.append(
                ProblemSpec(
                    family="SmallMLPDigits",
                    setting=f"MLP hybrid hidden={hidden_dim} samples={num_samples} lr=1e-02",
                    steps=10,
                    d=64,
                    rank=10,
                    hidden_dim=hidden_dim,
                    num_samples=num_samples,
                    batch_size=32 if num_samples == 128 else 128,
                    lr=1e-2,
                )
            )
    return tuple(specs)


def build_config() -> ExperimentConfig:
    base = default_config()
    return replace(
        base,
        algos=ALGOS,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        discussion_path=DISCUSSION_PATH,
        specs=hybrid_specs(),
    )


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hidden_dim"] = result["setting"].str.extract(r"hidden=(\d+)")[0].astype(int)
    result["num_samples"] = result["setting"].str.extract(r"samples=(\d+)")[0].astype(int)
    return result


def build_hybrid_optimizers(algo: str, params: list[torch.nn.Parameter], lr: float) -> list[torch.optim.Optimizer]:
    if algo == "Adam":
        return [torch.optim.Adam(params, lr=lr)]
    if algo == "Muon":
        return [ExactMuon(params, lr=lr)]
    if algo == "AdamFirstMuonSecond":
        return [torch.optim.Adam([params[0]], lr=lr), ExactMuon([params[1]], lr=lr)]
    if algo == "MuonFirstAdamSecond":
        return [ExactMuon([params[0]], lr=lr), torch.optim.Adam([params[1]], lr=lr)]
    raise ValueError(f"unknown hybrid algo: {algo}")


def zero_grad(params: list[torch.nn.Parameter], optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    for param in params:
        param.grad = None


def run_hybrid_group(spec: ProblemSpec, seed: int, start_run_id: int, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    problems = {algo: build_problem(spec, seed, device=device, dtype=dtype) for algo in ALGOS}
    params = {algo: problem.parameters() for algo, problem in problems.items()}
    optimizers = {algo: build_hybrid_optimizers(algo, params[algo], spec.lr) for algo in ALGOS}
    run_ids = {algo: start_run_id + offset for offset, algo in enumerate(ALGOS)}
    step_rows: list[dict] = []
    layer_rows: list[dict] = []
    current_row_indices: dict[str, int] = {algo: 0 for algo in ALGOS}
    current_layer_starts: dict[str, int] = {algo: 0 for algo in ALGOS}
    started = __import__("time").perf_counter()

    for step in range(spec.steps + 1):
        for algo in ALGOS:
            zero_grad(params[algo], optimizers[algo])
            current_row_index, layer_start = append_step_diagnostics(
                problem=problems[algo],
                spec=spec,
                algo=algo,
                seed=seed,
                run_id=run_ids[algo],
                step=step,
                started=started,
                step_rows=step_rows,
                layer_rows=layer_rows,
            )
            current_row_indices[algo] = current_row_index
            current_layer_starts[algo] = layer_start

        if step < spec.steps:
            before = {algo: snapshot_parameters(params[algo]) for algo in ALGOS}
            for algo in ALGOS:
                for optimizer in optimizers[algo]:
                    optimizer.step()
            proposed = {algo: relative_update_norm(before[algo], params[algo]) for algo in ALGOS}
            target = min(value for value in proposed.values() if value > 0.0)
            for algo in ALGOS:
                rescale_update(before[algo], params[algo], target)
                post_update_loss = float(problems[algo].loss().detach().cpu())
                step_rows[current_row_indices[algo]]["delta_loss"] = (
                    step_rows[current_row_indices[algo]]["loss"] - post_update_loss
                )
                apply_update_diagnostics(
                    before=before[algo],
                    after=params[algo],
                    step_row=step_rows[current_row_indices[algo]],
                    layer_rows=layer_rows,
                    layer_start=current_layer_starts[algo],
                )

    return pd.DataFrame(step_rows), pd.DataFrame(layer_rows)


def run_hybrid_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_frames = []
    layer_frames = []
    run_id = 0
    for spec in config.specs:
        for seed in config.seeds:
            step_frame, layer_frame = run_hybrid_group(spec, seed, run_id, config)
            step_frames.append(step_frame)
            layer_frames.append(layer_frame)
            run_id += len(ALGOS)
    return pd.concat(step_frames, ignore_index=True), pd.concat(layer_frames, ignore_index=True)


def hybrid_ratio_summary(steps: pd.DataFrame) -> pd.DataFrame:
    source = annotate(steps[np.isfinite(steps["delta_loss"])].copy())
    metrics = ["delta_loss", "update_grad_inner", "update_grad_cosine"]
    pivot = source.pivot_table(
        index=["setting", "hidden_dim", "num_samples", "seed", "step"],
        columns="algo",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    records = []
    for (hidden_dim, num_samples), group in paired.groupby(["hidden_dim", "num_samples"], observed=True, sort=True):
        for algo in ["Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond"]:
            for metric in metrics:
                ratio_values = group[f"{metric}_{algo}"] / (group[f"{metric}_Adam"].abs() + 1e-300)
                delta_values = group[f"{metric}_{algo}"] - group[f"{metric}_Adam"]
                ratio, lo, hi = log_ratio_ci95(ratio_values)
                delta, delta_lo, delta_hi = ci95(delta_values)
                records.append(
                    {
                        "hidden_dim": int(hidden_dim),
                        "num_samples": int(num_samples),
                        "algo": algo,
                        "baseline": "Adam",
                        "metric": metric,
                        "n_pairs": int(len(group)),
                        "higher_than_adam_pairs": int((delta_values > 0).sum()),
                        "win_rate_vs_adam": float((delta_values > 0).mean()),
                        "geomean_ratio_over_adam": ratio,
                        "ratio_ci95_low": lo,
                        "ratio_ci95_high": hi,
                        "ratio_ci95_above_one": bool(lo > 1.0) if np.isfinite(lo) else False,
                        "ratio_ci95_below_one": bool(hi < 1.0) if np.isfinite(hi) else False,
                        "mean_delta_vs_adam": delta,
                        "delta_ci95_low": delta_lo,
                        "delta_ci95_high": delta_hi,
                    }
                )
    return pd.DataFrame(records)


def layer_inner_summary(layers: pd.DataFrame) -> pd.DataFrame:
    source = annotate(layers[np.isfinite(layers["update_grad_inner"])].copy())
    return source.groupby(["hidden_dim", "num_samples", "algo", "layer"], as_index=False, observed=True).agg(
        points=("run_id", "size"),
        mean_update_grad_inner=("update_grad_inner", "mean"),
        mean_update_grad_cosine=("update_grad_cosine", "mean"),
        mean_nrG=("nrG", "mean"),
        mean_stA=("stA", "mean"),
    )


def hybrid_layer_ratio_summary(layer_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = layer_summary.pivot_table(
        index=["hidden_dim", "num_samples", "layer"],
        columns="algo",
        values=["mean_update_grad_inner", "mean_update_grad_cosine", "mean_nrG"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    rows = pivot.reset_index().dropna().copy()
    records = []
    for _, row in rows.iterrows():
        for algo in ["Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond"]:
            records.append(
                {
                    "hidden_dim": int(row["hidden_dim"]),
                    "num_samples": int(row["num_samples"]),
                    "layer": int(row["layer"]),
                    "algo": algo,
                    "inner_ratio_over_adam": row[f"mean_update_grad_inner_{algo}"]
                    / (abs(row["mean_update_grad_inner_Adam"]) + 1e-300),
                    "cosine_ratio_over_adam": row[f"mean_update_grad_cosine_{algo}"]
                    / (abs(row["mean_update_grad_cosine_Adam"]) + 1e-300),
                    "nrG_ratio_over_adam": row[f"mean_nrG_{algo}"] / (abs(row["mean_nrG_Adam"]) + 1e-300),
                }
            )
    return pd.DataFrame(records)


def update_allocation_summary(layers: pd.DataFrame) -> pd.DataFrame:
    source = annotate(layers[np.isfinite(layers["update_grad_inner"])].copy())

    def update_fro_norm(text: str) -> float:
        values = json_loads_nested(text)
        if not values:
            return np.nan
        flat = np.asarray([value for row in values for value in row], dtype=float)
        return float(np.sqrt(np.square(flat).sum()))

    source["layer_update_fro_norm"] = source["sigma_update"].map(update_fro_norm)
    total_fro_sq = source.groupby(["run_id", "step"], observed=True)["layer_update_fro_norm"].transform(
        lambda values: float(np.square(values).sum())
    )
    source["layer_update_fro_fraction"] = source["layer_update_fro_norm"] / np.sqrt(total_fro_sq)
    source["layer_update_fro_sq_fraction"] = np.square(source["layer_update_fro_norm"]) / total_fro_sq
    source["layer_update_efficiency"] = source["update_grad_inner"] / (source["layer_update_fro_norm"] + 1e-300)

    summary = source.groupby(["hidden_dim", "num_samples", "algo", "layer"], as_index=False, observed=True).agg(
        points=("run_id", "size"),
        mean_layer_update_fro_fraction=("layer_update_fro_fraction", "mean"),
        mean_layer_update_fro_sq_fraction=("layer_update_fro_sq_fraction", "mean"),
        mean_layer_update_efficiency=("layer_update_efficiency", "mean"),
        mean_update_grad_inner=("update_grad_inner", "mean"),
        mean_update_grad_cosine=("update_grad_cosine", "mean"),
    )
    pivot = summary.pivot_table(
        index=["hidden_dim", "num_samples", "layer"],
        columns="algo",
        values=[
            "mean_layer_update_fro_fraction",
            "mean_layer_update_fro_sq_fraction",
            "mean_layer_update_efficiency",
            "mean_update_grad_inner",
        ],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    rows = pivot.reset_index().dropna().copy()
    records = []
    for _, row in rows.iterrows():
        for algo in ["Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond"]:
            records.append(
                {
                    "hidden_dim": int(row["hidden_dim"]),
                    "num_samples": int(row["num_samples"]),
                    "layer": int(row["layer"]),
                    "algo": algo,
                    "fro_fraction": row[f"mean_layer_update_fro_fraction_{algo}"],
                    "fro_sq_fraction": row[f"mean_layer_update_fro_sq_fraction_{algo}"],
                    "efficiency": row[f"mean_layer_update_efficiency_{algo}"],
                    "inner": row[f"mean_update_grad_inner_{algo}"],
                    "fro_fraction_ratio_over_adam": row[f"mean_layer_update_fro_fraction_{algo}"]
                    / (abs(row["mean_layer_update_fro_fraction_Adam"]) + 1e-300),
                    "fro_sq_fraction_ratio_over_adam": row[f"mean_layer_update_fro_sq_fraction_{algo}"]
                    / (abs(row["mean_layer_update_fro_sq_fraction_Adam"]) + 1e-300),
                    "efficiency_ratio_over_adam": row[f"mean_layer_update_efficiency_{algo}"]
                    / (abs(row["mean_layer_update_efficiency_Adam"]) + 1e-300),
                    "inner_ratio_over_adam": row[f"mean_update_grad_inner_{algo}"]
                    / (abs(row["mean_update_grad_inner_Adam"]) + 1e-300),
                }
            )
    return pd.DataFrame(records)


def plot_hybrid_ratios(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "hybrid_first_order_ratios.png"
    rows = summary[summary["metric"] == "update_grad_inner"].copy()
    sample_values = sorted(rows["num_samples"].unique())
    colors = {
        "Muon": "#0072B2",
        "AdamFirstMuonSecond": "#009E73",
        "MuonFirstAdamSecond": "#CC79A7",
    }
    labels = {
        "Muon": "Muon/Muon",
        "AdamFirstMuonSecond": "Adam first + Muon second",
        "MuonFirstAdamSecond": "Muon first + Adam second",
    }
    fig, axes = plt.subplots(1, len(sample_values), figsize=(5.6 * len(sample_values), 4.1), sharey=True, squeeze=False)
    for ax, samples in zip(axes.ravel(), sample_values):
        sub_samples = rows[rows["num_samples"] == samples]
        for algo in ["Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond"]:
            sub = sub_samples[sub_samples["algo"] == algo].sort_values("hidden_dim")
            x = sub["hidden_dim"].to_numpy(dtype=float)
            y = sub["geomean_ratio_over_adam"].to_numpy(dtype=float)
            lo = sub["ratio_ci95_low"].to_numpy(dtype=float)
            hi = sub["ratio_ci95_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", color=colors[algo], label=labels[algo])
            ax.fill_between(x, lo, hi, color=colors[algo], alpha=0.14)
        ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128])
        ax.set_xticklabels(["8", "16", "32", "64", "128"])
        ax.set_xlabel("hidden_dim")
        ax.set_title(f"num_samples={samples}")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("first-order ratio over Adam")
    handles, labels_values = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_values, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Layerwise hybrid test of the SmallMLP first-layer bottleneck", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hybrid_layer_contributions(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "hybrid_layer_contributions.png"
    rows = summary.copy()
    sample_values = sorted(rows["num_samples"].unique())
    colors = {
        "Muon": "#0072B2",
        "AdamFirstMuonSecond": "#009E73",
        "MuonFirstAdamSecond": "#CC79A7",
    }
    labels = {
        "Muon": "Muon/Muon",
        "AdamFirstMuonSecond": "Adam first + Muon second",
        "MuonFirstAdamSecond": "Muon first + Adam second",
    }
    fig, axes = plt.subplots(2, len(sample_values), figsize=(5.6 * len(sample_values), 6.2), sharey=True, squeeze=False)
    for col, samples in enumerate(sample_values):
        for row_idx, layer in enumerate([1, 2]):
            ax = axes[row_idx, col]
            sub_layer = rows[(rows["num_samples"] == samples) & (rows["layer"] == layer)]
            for algo in ["Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond"]:
                sub = sub_layer[sub_layer["algo"] == algo].sort_values("hidden_dim")
                ax.plot(
                    sub["hidden_dim"],
                    sub["inner_ratio_over_adam"],
                    marker="o",
                    color=colors[algo],
                    label=labels[algo] if row_idx == 0 and col == 0 else None,
                )
            ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128])
            ax.set_xticklabels(["8", "16", "32", "64", "128"])
            ax.set_title(f"samples={samples}, layer={layer}")
            ax.set_xlabel("hidden_dim")
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel("layer inner ratio over Adam")
    handles, label_values = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, label_values, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Layerwise first-order contributions in hybrid optimizers", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hybrid_update_allocation(summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "hybrid_update_allocation.png"
    rows = summary.copy()
    sample_values = sorted(rows["num_samples"].unique())
    colors = {
        "Muon": "#0072B2",
        "AdamFirstMuonSecond": "#009E73",
        "MuonFirstAdamSecond": "#CC79A7",
    }
    labels = {
        "Muon": "Muon/Muon",
        "AdamFirstMuonSecond": "Adam first + Muon second",
        "MuonFirstAdamSecond": "Muon first + Adam second",
    }
    fig, axes = plt.subplots(2, len(sample_values), figsize=(5.6 * len(sample_values), 6.2), sharey=True, squeeze=False)
    for col, samples in enumerate(sample_values):
        for row_idx, layer in enumerate([1, 2]):
            ax = axes[row_idx, col]
            sub_layer = rows[(rows["num_samples"] == samples) & (rows["layer"] == layer)]
            for algo in ["Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond"]:
                sub = sub_layer[sub_layer["algo"] == algo].sort_values("hidden_dim")
                ax.plot(
                    sub["hidden_dim"],
                    sub["fro_sq_fraction"],
                    marker="o",
                    color=colors[algo],
                    label=labels[algo] if row_idx == 0 and col == 0 else None,
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128])
            ax.set_xticklabels(["8", "16", "32", "64", "128"])
            ax.set_ylim(0.0, 1.02)
            ax.set_title(f"samples={samples}, layer={layer}")
            ax.set_xlabel("hidden_dim")
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel("fraction of global update Frobenius^2")
    handles, label_values = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, label_values, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Layerwise update-budget allocation in hybrid optimizers", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.4g}")
    return display.to_markdown(index=False)


def write_discussion(
    config: ExperimentConfig,
    ratio_summary: pd.DataFrame,
    layer_ratio: pd.DataFrame,
    allocation_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    figure: Path,
    layer_figure: Path,
    allocation_figure: Path,
) -> None:
    first_order = ratio_summary[ratio_summary["metric"] == "update_grad_inner"].copy()
    table = markdown_table(
        first_order,
        [
            "hidden_dim",
            "num_samples",
            "algo",
            "n_pairs",
            "win_rate_vs_adam",
            "geomean_ratio_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
            "ratio_ci95_below_one",
        ],
    )
    cal_table = markdown_table(
        calibration[calibration["group"].isin(["All", "Adam", "Muon", "AdamFirstMuonSecond", "MuonFirstAdamSecond"])],
        [
            "group",
            "points",
            "positive_points",
            "spearman_delta_vs_first_order",
            "spearman_ci95_low",
            "spearman_ci95_high",
            "within_factor_2",
        ],
    )
    layer_table = markdown_table(
        layer_ratio,
        [
            "hidden_dim",
            "num_samples",
            "layer",
            "algo",
            "inner_ratio_over_adam",
            "cosine_ratio_over_adam",
            "nrG_ratio_over_adam",
        ],
    )
    allocation_table = markdown_table(
        allocation_summary,
        [
            "hidden_dim",
            "num_samples",
            "layer",
            "algo",
            "fro_sq_fraction",
            "fro_sq_fraction_ratio_over_adam",
            "efficiency_ratio_over_adam",
            "inner_ratio_over_adam",
        ],
    )
    text = f"""# E11 SmallMLP Layerwise Hybrid Test

## Purpose

The width sweep suggests a first-layer bottleneck: Muon's first-layer gradient-update alignment collapses as the hidden layer widens. This experiment tests that mechanism directly by assigning Adam or Muon separately to the first and second linear layers.

All runs use equal-update control across four optimizers: `Adam`, `Muon`, `AdamFirstMuonSecond`, and `MuonFirstAdamSecond`.

## Hybrid Result

Ratios are measured against the Adam/Adam baseline. Values above 1 mean larger one-step first-order progress than Adam.

![Hybrid first-order ratios](../{figure})

{table}

## Layer Contribution Breakdown

The mixed optimizers fail for different reasons. `AdamFirstMuonSecond` keeps the first-layer contribution closer to Adam, but its second-layer first-order contribution collapses far below Adam. `MuonFirstAdamSecond` often preserves more second-layer contribution, but its first-layer contribution drops. This explains why neither hybrid recovers pure Muon's narrow-width behavior.

![Hybrid layer contributions](../{layer_figure})

{layer_table}

## Update-Budget Allocation

The layer contribution collapse is partly an update-allocation effect, not only a directional-efficiency effect. The table reports each layer's fraction of the global equal-update Frobenius-squared budget, plus the efficiency ratio `update_grad_inner / layer_update_fro_norm` against Adam.

![Hybrid update allocation](../{allocation_figure})

{allocation_table}

## First-Order Calibration

{cal_table}

## Interpretation

The simple layer-replacement causal hypothesis is not supported. If the only issue were "Muon is bad on the first layer at large width," then `AdamFirstMuonSecond` should recover progress at large widths. It does not: it is below Adam across the sweep and is often worse than `MuonFirstAdamSecond`.

The stronger current reading is that the width transition is tied to layerwise alignment, but not in an independently swappable way. Pure Muon appears to rely on a coupled two-layer update geometry: it is favorable at hidden widths 8 and 16, but mixing Adam and Muon layerwise disrupts that geometry rather than cleanly isolating a better first-layer direction.

The update-allocation diagnostic makes this more precise. When the second layer is updated by Muon while the first layer is updated by Adam, the second layer often receives a much smaller share of the equalized global update budget. When the first layer is updated by Muon while the second layer is updated by Adam, the first-layer contribution is directionally weak even if the second layer remains useful. This supports a coupled-geometry explanation rather than a single-layer replacement rule.
"""
    config.discussion_path.parent.mkdir(parents=True, exist_ok=True)
    config.discussion_path.write_text(text, encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    steps, layers = run_hybrid_experiment(config)
    ratio_summary = hybrid_ratio_summary(steps)
    layer_summary = layer_inner_summary(layers)
    layer_ratio = hybrid_layer_ratio_summary(layer_summary)
    allocation_summary = update_allocation_summary(layers)
    calibration = first_order_calibration_summary(steps)
    steps.to_csv(config.output_dir / "step_metrics.csv", index=False)
    layers.to_csv(config.output_dir / "layer_metrics.csv", index=False)
    ratio_summary.to_csv(config.output_dir / "hybrid_ratio_summary.csv", index=False)
    layer_summary.to_csv(config.output_dir / "layer_inner_summary.csv", index=False)
    layer_ratio.to_csv(config.output_dir / "hybrid_layer_ratio_summary.csv", index=False)
    allocation_summary.to_csv(config.output_dir / "hybrid_update_allocation_summary.csv", index=False)
    calibration.to_csv(config.output_dir / "first_order_calibration_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    figure = plot_hybrid_ratios(ratio_summary)
    layer_figure = plot_hybrid_layer_contributions(layer_ratio)
    allocation_figure = plot_hybrid_update_allocation(allocation_summary)
    write_discussion(config, ratio_summary, layer_ratio, allocation_summary, calibration, figure, layer_figure, allocation_figure)
    print(f"saved MLP layer hybrid results to {config.output_dir}")
    print(f"step rows={len(steps)}, layer rows={len(layers)}, runs={steps['run_id'].nunique()}")
    print(f"figure: {figure}")
    print(f"layer figure: {layer_figure}")
    print(f"allocation figure: {allocation_figure}")
    print(f"discussion: {config.discussion_path}")


if __name__ == "__main__":
    main()
