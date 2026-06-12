from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.config import ExperimentConfig, ProblemSpec
from e11_condition_geometry.runner import run_equal_update_experiment
from e11_condition_geometry.statistics import (
    ci95,
    final_performance_summary,
    first_order_calibration_summary,
    first_order_pair_summary,
    first_order_win_condition_summary,
    log_ratio_ci95,
    prediction_summary,
    run_dynamics_summary,
    update_spectrum_summary,
    update_transmission_summary,
    volatility_summary,
    win_feature_overlap_summary,
    win_prediction_generalization_summary,
)


OUTPUT_DIR = Path("results/e11_mlp_width_sweep")
FIGURE_DIR = Path("figures/e11_mlp_width_sweep")
DISCUSSION_PATH = Path("discussion/e11_mlp_width_sweep.md")


def width_specs() -> tuple[ProblemSpec, ...]:
    learning_rates = (3e-3, 1e-2, 3e-2)
    specs: list[ProblemSpec] = []
    for num_samples in [128, 1024]:
        for hidden_dim in [8, 16, 32, 64, 128]:
            for lr in learning_rates:
                specs.append(
                    ProblemSpec(
                        family="SmallMLPDigits",
                        setting=f"MLP width sweep hidden={hidden_dim} samples={num_samples} lr={lr:.0e}",
                        steps=10,
                        d=64,
                        rank=10,
                        hidden_dim=hidden_dim,
                        num_samples=num_samples,
                        batch_size=32 if num_samples == 128 else 128,
                        lr=lr,
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
        specs=width_specs(),
    )


def annotate_width_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    hidden = result["setting"].str.extract(r"hidden=(\d+)")[0].astype(int)
    samples = result["setting"].str.extract(r"samples=(\d+)")[0].astype(int)
    result["hidden_dim"] = hidden
    result["num_samples"] = samples
    return result


def pair_frame(steps: pd.DataFrame) -> pd.DataFrame:
    source = annotate_width_columns(steps[np.isfinite(steps["delta_loss"])].copy())
    metrics = ["delta_loss", "update_grad_inner", "update_grad_cosine", "loss", "recovery_error", "nrG", "stA", "condition_score"]
    pivot = source.pivot_table(
        index=["setting", "hidden_dim", "num_samples", "lr", "seed", "step"],
        columns="algo",
        values=metrics,
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{algo}" for metric, algo in pivot.columns]
    paired = pivot.reset_index().dropna().copy()
    for metric in ["delta_loss", "update_grad_inner", "update_grad_cosine"]:
        paired[f"{metric}_ratio"] = paired[f"{metric}_Muon"] / (paired[f"{metric}_Adam"].abs() + 1e-300)
        paired[f"{metric}_delta"] = paired[f"{metric}_Muon"] - paired[f"{metric}_Adam"]
        paired[f"{metric}_muon_higher"] = paired[f"{metric}_delta"] > 0
    return paired


def width_pair_summary(steps: pd.DataFrame) -> pd.DataFrame:
    paired = pair_frame(steps)
    records = []
    groupers: list[tuple[str, tuple, pd.DataFrame]] = []
    groupers.extend(
        ("hidden_samples", key, group)
        for key, group in paired.groupby(["hidden_dim", "num_samples"], observed=True, sort=True)
    )
    groupers.extend(("hidden_all_samples", (hidden, "All"), group) for hidden, group in paired.groupby("hidden_dim", observed=True, sort=True))
    for group_type, key, group in groupers:
        hidden_dim, num_samples = key
        for metric in ["update_grad_inner", "update_grad_cosine", "delta_loss"]:
            ratio, lo, hi = log_ratio_ci95(group[f"{metric}_ratio"])
            delta, delta_lo, delta_hi = ci95(group[f"{metric}_delta"])
            records.append(
                {
                    "group_type": group_type,
                    "hidden_dim": int(hidden_dim),
                    "num_samples": num_samples,
                    "metric": metric,
                    "n_pairs": int(len(group)),
                    "muon_higher_pairs": int(group[f"{metric}_muon_higher"].sum()),
                    "muon_win_rate": float(group[f"{metric}_muon_higher"].mean()),
                    "geomean_ratio_muon_over_adam": ratio,
                    "ratio_ci95_low": lo,
                    "ratio_ci95_high": hi,
                    "ratio_ci95_above_one": bool(lo > 1.0) if np.isfinite(lo) else False,
                    "ratio_ci95_below_one": bool(hi < 1.0) if np.isfinite(hi) else False,
                    "mean_delta_muon_minus_adam": delta,
                    "delta_ci95_low": delta_lo,
                    "delta_ci95_high": delta_hi,
                }
            )
    return pd.DataFrame(records)


def width_layer_geometry_summary(layers: pd.DataFrame) -> pd.DataFrame:
    source = annotate_width_columns(layers.copy())
    source = source[
        np.isfinite(source["nrG"])
        & np.isfinite(source["stA"])
        & np.isfinite(source["condition_score"])
        & np.isfinite(source["update_rank_ceiling"])
        & (source["update_rank_ceiling"] > 0)
        & np.isfinite(source["update_grad_cosine"])
        & np.isfinite(source["update_grad_inner"])
    ].copy()
    source["grad_rank_fraction"] = source["nrG"] / source["update_rank_ceiling"]
    return source.groupby(["hidden_dim", "num_samples", "algo", "layer"], as_index=False, observed=True).agg(
        points=("run_id", "size"),
        mean_nrG=("nrG", "mean"),
        mean_stA=("stA", "mean"),
        mean_condition_score=("condition_score", "mean"),
        mean_grad_rank_fraction=("grad_rank_fraction", "mean"),
        mean_update_grad_cosine=("update_grad_cosine", "mean"),
        mean_update_grad_inner=("update_grad_inner", "mean"),
    )


def width_mechanism_coupling_summary(width_pair: pd.DataFrame, layer_summary: pd.DataFrame) -> pd.DataFrame:
    pair_rows = width_pair[
        (width_pair["group_type"] == "hidden_samples")
        & (width_pair["metric"].isin(["update_grad_inner", "delta_loss"]))
    ].copy()
    pair_rows["hidden_dim"] = pair_rows["hidden_dim"].astype(int)
    pair_rows["num_samples"] = pair_rows["num_samples"].astype(int)
    layer_summary = layer_summary.copy()
    layer_summary["hidden_dim"] = layer_summary["hidden_dim"].astype(int)
    layer_summary["num_samples"] = layer_summary["num_samples"].astype(int)
    layer_wide = layer_summary[layer_summary["layer"].isin([1, 2])].pivot_table(
        index=["hidden_dim", "num_samples"],
        columns=["layer", "algo"],
        values=["mean_grad_rank_fraction", "mean_update_grad_cosine", "mean_update_grad_inner"],
        aggfunc="mean",
    )
    layer_wide.columns = [f"layer{layer}_{metric}_{algo}" for metric, layer, algo in layer_wide.columns]
    layer_wide = layer_wide.reset_index()
    for layer in [1, 2]:
        layer_wide[f"layer{layer}_cosine_ratio_muon_over_adam"] = layer_wide[
            f"layer{layer}_mean_update_grad_cosine_Muon"
        ] / (layer_wide[f"layer{layer}_mean_update_grad_cosine_Adam"].abs() + 1e-300)
        layer_wide[f"layer{layer}_rank_fraction_ratio_muon_over_adam"] = layer_wide[
            f"layer{layer}_mean_grad_rank_fraction_Muon"
        ] / (layer_wide[f"layer{layer}_mean_grad_rank_fraction_Adam"].abs() + 1e-300)
        layer_wide[f"layer{layer}_inner_ratio_muon_over_adam"] = layer_wide[f"layer{layer}_mean_update_grad_inner_Muon"] / (
            layer_wide[f"layer{layer}_mean_update_grad_inner_Adam"].abs() + 1e-300
        )
    result = pair_rows.merge(layer_wide, on=["hidden_dim", "num_samples"], how="left")
    result["first_order_or_loss_ratio"] = result["geomean_ratio_muon_over_adam"]
    result["same_side_of_one_as_layer1_cosine"] = (
        (result["first_order_or_loss_ratio"] > 1.0)
        == (result["layer1_cosine_ratio_muon_over_adam"] > 1.0)
    )
    result["same_side_of_one_as_layer2_cosine"] = (
        (result["first_order_or_loss_ratio"] > 1.0)
        == (result["layer2_cosine_ratio_muon_over_adam"] > 1.0)
    )
    return result


def save_standard_summaries(config: ExperimentConfig, steps: pd.DataFrame, layers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dynamics = run_dynamics_summary(steps)
    summaries = {
        "performance": final_performance_summary(steps),
        "prediction": prediction_summary(steps),
        "volatility": volatility_summary(dynamics),
        "update_spectrum": update_spectrum_summary(steps),
        "update_transmission": update_transmission_summary(steps),
        "first_order_calibration": first_order_calibration_summary(steps),
        "first_order_pair": first_order_pair_summary(steps),
        "win_condition": first_order_win_condition_summary(steps, layers),
        "win_generalization": win_prediction_generalization_summary(steps, layers),
        "win_overlap": win_feature_overlap_summary(steps, layers),
        "width_pair": width_pair_summary(steps),
        "width_layer_geometry": width_layer_geometry_summary(layers),
        "run_dynamics": dynamics,
    }
    summaries["width_mechanism_coupling"] = width_mechanism_coupling_summary(
        summaries["width_pair"],
        summaries["width_layer_geometry"],
    )
    steps.to_csv(config.output_dir / "step_metrics.csv", index=False)
    layers.to_csv(config.output_dir / "layer_metrics.csv", index=False)
    for name, frame in summaries.items():
        frame.to_csv(config.output_dir / f"{name}_summary.csv", index=False)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return summaries


def plot_width_transition(width_summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "width_first_order_transition.png"
    rows = width_summary[
        (width_summary["group_type"] == "hidden_samples")
        & width_summary["metric"].isin(["update_grad_inner", "delta_loss"])
    ].copy()
    rows["num_samples"] = rows["num_samples"].astype(int)
    sample_values = sorted(rows["num_samples"].unique())
    fig, axes = plt.subplots(1, len(sample_values), figsize=(5.3 * len(sample_values), 4.0), sharey=True, squeeze=False)
    metric_styles = {
        "update_grad_inner": ("#0072B2", "first-order"),
        "delta_loss": ("#D55E00", "delta loss"),
    }
    for ax, samples in zip(axes.ravel(), sample_values):
        sub_samples = rows[rows["num_samples"] == samples]
        for metric, (color, label) in metric_styles.items():
            sub = sub_samples[sub_samples["metric"] == metric].sort_values("hidden_dim")
            x = sub["hidden_dim"].to_numpy(dtype=float)
            y = sub["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
            lo = sub["ratio_ci95_low"].to_numpy(dtype=float)
            hi = sub["ratio_ci95_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", color=color, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.16)
        ax.axhline(1.0, color="#555555", ls="--", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128])
        ax.set_xticklabels(["8", "16", "32", "64", "128"])
        ax.set_title(f"num_samples={samples}")
        ax.set_xlabel("hidden_dim")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Muon / Adam geometric mean ratio")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("SmallMLP width sweep under equal-update control", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_width_layer_mechanism(layer_summary: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "width_layer_mechanism.png"
    rows = layer_summary.copy()
    rows["num_samples"] = rows["num_samples"].astype(int)
    sample_values = sorted(rows["num_samples"].unique())
    fig, axes = plt.subplots(2, len(sample_values), figsize=(5.4 * len(sample_values), 6.4), sharex=True, squeeze=False)
    colors = {"Adam": "#D55E00", "Muon": "#0072B2"}
    for col, samples in enumerate(sample_values):
        sub_samples = rows[rows["num_samples"] == samples]
        for layer, linestyle in [(1, "-"), (2, "--")]:
            for algo in ["Adam", "Muon"]:
                sub = sub_samples[(sub_samples["layer"] == layer) & (sub_samples["algo"] == algo)].sort_values("hidden_dim")
                label = f"{algo} layer {layer}" if col == 0 else None
                axes[0, col].plot(
                    sub["hidden_dim"],
                    sub["mean_update_grad_cosine"],
                    marker="o",
                    color=colors[algo],
                    linestyle=linestyle,
                    label=label,
                )
                axes[1, col].plot(
                    sub["hidden_dim"],
                    sub["mean_grad_rank_fraction"],
                    marker="o",
                    color=colors[algo],
                    linestyle=linestyle,
                    label=label,
                )
        axes[0, col].set_title(f"num_samples={samples}")
        axes[0, col].set_ylabel("mean gradient-update cosine")
        axes[1, col].set_ylabel(r"mean $nr(G_i)/r_i$")
        axes[1, col].set_xlabel("hidden_dim")
        for ax in axes[:, col]:
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128])
            ax.set_xticklabels(["8", "16", "32", "64", "128"])
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Layerwise mechanism behind the SmallMLP width transition", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_width_mechanism_coupling(coupling: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "width_mechanism_coupling.png"
    rows = coupling[coupling["metric"].isin(["update_grad_inner", "delta_loss"])].copy()
    sample_values = sorted(rows["num_samples"].astype(int).unique())
    metric_styles = {
        "update_grad_inner": ("#0072B2", "first-order"),
        "delta_loss": ("#D55E00", "delta loss"),
    }
    fig, axes = plt.subplots(1, len(sample_values), figsize=(5.0 * len(sample_values), 4.0), sharey=True, squeeze=False)
    for ax, samples in zip(axes.ravel(), sample_values):
        sub_samples = rows[rows["num_samples"].astype(int) == samples]
        for metric, (color, label) in metric_styles.items():
            sub = sub_samples[sub_samples["metric"] == metric].sort_values("hidden_dim")
            ax.scatter(
                sub["layer1_cosine_ratio_muon_over_adam"],
                sub["first_order_or_loss_ratio"],
                s=58,
                color=color,
                label=label,
            )
            for _, row in sub.iterrows():
                ax.text(
                    row["layer1_cosine_ratio_muon_over_adam"],
                    row["first_order_or_loss_ratio"],
                    str(int(row["hidden_dim"])),
                    fontsize=7,
                    ha="center",
                    va="center",
                    color="white",
                )
        ax.axhline(1.0, color="#555555", ls="--", lw=0.8)
        ax.axvline(1.0, color="#555555", ls="--", lw=0.8)
        ax.set_title(f"num_samples={samples}")
        ax.set_xlabel("layer-1 cosine ratio: Muon / Adam")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("progress ratio: Muon / Adam")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Does first-layer alignment explain the width transition?", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    summaries: dict[str, pd.DataFrame],
    figure_path: Path,
    mechanism_path: Path,
    coupling_path: Path,
) -> None:
    width_rows = summaries["width_pair"][
        (summaries["width_pair"]["group_type"] == "hidden_samples")
        & summaries["width_pair"]["metric"].isin(["update_grad_inner", "delta_loss"])
    ].copy()
    table = markdown_table(
        width_rows,
        [
            "hidden_dim",
            "num_samples",
            "metric",
            "n_pairs",
            "muon_higher_pairs",
            "muon_win_rate",
            "geomean_ratio_muon_over_adam",
            "ratio_ci95_low",
            "ratio_ci95_high",
            "ratio_ci95_above_one",
            "ratio_ci95_below_one",
        ],
    )
    calibration = markdown_table(
        summaries["first_order_calibration"][
            summaries["first_order_calibration"]["group"].isin(["All", "Adam", "Muon"])
        ],
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
    mechanism_rows = summaries["width_layer_geometry"][
        (summaries["width_layer_geometry"]["layer"] == 1)
        & (summaries["width_layer_geometry"]["algo"].isin(["Adam", "Muon"]))
    ].copy()
    mechanism_table = markdown_table(
        mechanism_rows,
        [
            "hidden_dim",
            "num_samples",
            "algo",
            "layer",
            "mean_grad_rank_fraction",
            "mean_update_grad_cosine",
            "mean_update_grad_inner",
        ],
    )
    coupling_rows = summaries["width_mechanism_coupling"][
        summaries["width_mechanism_coupling"]["metric"].isin(["update_grad_inner", "delta_loss"])
    ].copy()
    coupling_table = markdown_table(
        coupling_rows,
        [
            "hidden_dim",
            "num_samples",
            "metric",
            "geomean_ratio_muon_over_adam",
            "layer1_cosine_ratio_muon_over_adam",
            "layer2_cosine_ratio_muon_over_adam",
            "layer1_rank_fraction_ratio_muon_over_adam",
            "layer2_rank_fraction_ratio_muon_over_adam",
            "layer1_inner_ratio_muon_over_adam",
            "layer2_inner_ratio_muon_over_adam",
            "same_side_of_one_as_layer1_cosine",
            "same_side_of_one_as_layer2_cosine",
        ],
    )
    text = f"""# E11 SmallMLP Width Sweep

## Purpose

The overlap follow-up showed that `SmallMLPDigits` becomes Muon-favorable when the hidden layer is reduced to 16 units. This sweep tests whether that flip is a width-dependent pattern or a single-setting accident.

All runs use equal-update control. The sweep varies `hidden_dim in {{8, 16, 32, 64, 128}}`, `num_samples in {{128, 1024}}`, `lr in {{3e-3, 1e-2, 3e-2}}`, and 5 seeds.

## Width Transition

Ratios above 1 mean Muon has larger matched one-step progress than Adam.

![Width transition](../{figure_path})

{table}

## First-Order Calibration

{calibration}

## Layerwise Mechanism

The transition is mainly visible in the first layer. As `hidden_dim` grows, Muon's first-layer gradient-rank fraction and gradient-update cosine drop sharply, while Adam's first-layer cosine stays comparatively flat. This is consistent with the ExactMuon identity \\(\\cos^2(G_i, \\Delta W_i)=nr(G_i)/r_i\\): widening the first layer increases the rank ceiling faster than the effective gradient rank.

![Layerwise mechanism](../{mechanism_path})

{mechanism_table}

## Mechanism Coupling

This table directly compares the progress ratio with layerwise alignment ratios. The second layer stays mildly Muon-favorable across the sweep, but the first layer changes sharply with width. Hidden width 16 is the boundary case: layer 1 is already slightly Adam-favorable, but the disadvantage is small enough that the layer-2 advantage keeps the total first-order progress ratio above 1. At widths 32 and above, the first-layer disadvantage becomes too large to compensate.

![Mechanism coupling](../{coupling_path})

{coupling_table}

## Interpretation

The sweep gives a clear width transition. Muon is consistently first-order favorable at hidden widths 8 and 16 for both sample regimes, while Adam is consistently first-order favorable at widths 32, 64, and 128. The transition is visible in both `update_grad_inner` and realized `delta_loss`, so the hidden=16 overlap-followup result is not a single-setting accident.

This points to network width as a concrete task-structure variable that changes whether Muon's polar update geometry is useful. The main quantity remains `update_grad_inner = <G, W_t-W_(t+1)>`; observed `delta_loss` is included as the realized one-step decrease.
"""
    config.discussion_path.parent.mkdir(parents=True, exist_ok=True)
    config.discussion_path.write_text(text, encoding="utf-8")


def main() -> None:
    config = build_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    steps, layers = run_equal_update_experiment(config)
    summaries = save_standard_summaries(config, steps, layers)
    figure = plot_width_transition(summaries["width_pair"])
    mechanism = plot_width_layer_mechanism(summaries["width_layer_geometry"])
    coupling = plot_width_mechanism_coupling(summaries["width_mechanism_coupling"])
    write_discussion(config, summaries, figure, mechanism, coupling)
    print(f"saved MLP width sweep to {config.output_dir}")
    print(f"step rows={len(steps)}, layer rows={len(layers)}, runs={steps['run_id'].nunique()}")
    print(f"figure: {figure}")
    print(f"mechanism: {mechanism}")
    print(f"coupling: {coupling}")
    print(f"discussion: {config.discussion_path}")


if __name__ == "__main__":
    main()
