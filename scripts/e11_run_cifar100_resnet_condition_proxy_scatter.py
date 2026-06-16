from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table


INPUT_DIR = Path("results/e11_cifar100_resnet_checkpoint_sweep")
OUTPUT_DIR = Path("results/e11_cifar100_resnet_condition_proxy_scatter")
FIGURE_DIR = Path("figures/e11_cifar100_resnet_condition_proxy_scatter")
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_condition_proxy_scatter.md")


def paired_step_points(step_metrics: pd.DataFrame) -> pd.DataFrame:
    index = ["warmup_steps", "seed"]
    value_columns = [
        "tail_output_drift_rms",
        "centered_tail_output_drift_fro",
        "margin_delta_rms",
        "tail_loss_increase",
        "tail_accuracy_drop",
        "tail_accuracy_before",
        "tail_positive_margin_fraction_before",
        "matched_first_order_head_gain",
        "actual_head_gain_relative_error",
        "update_fro_norm",
        "update_op_norm",
        "nrG",
    ]
    wide = step_metrics.pivot(index=index, columns="geometry", values=value_columns)
    rows = []
    for warmup_steps, seed in wide.index:
        row = {"warmup_steps": int(warmup_steps), "seed": int(seed)}
        for column in value_columns:
            row[f"{column}_frobenius"] = float(wide.loc[(warmup_steps, seed), (column, "frobenius")])
            row[f"{column}_spectral"] = float(wide.loc[(warmup_steps, seed), (column, "spectral")])
        row["tail_output_drift_sq_ratio_spectral_over_fro"] = (
            row["tail_output_drift_rms_spectral"] / row["tail_output_drift_rms_frobenius"]
        ) ** 2
        row["centered_tail_output_drift_sq_ratio_spectral_over_fro"] = (
            row["centered_tail_output_drift_fro_spectral"] / row["centered_tail_output_drift_fro_frobenius"]
        ) ** 2
        row["margin_delta_sq_ratio_spectral_over_fro"] = (
            row["margin_delta_rms_spectral"] / row["margin_delta_rms_frobenius"]
        ) ** 2
        row["tail_loss_increase_diff_spectral_minus_fro"] = (
            row["tail_loss_increase_spectral"] - row["tail_loss_increase_frobenius"]
        )
        row["tail_accuracy_drop_diff_spectral_minus_fro"] = (
            row["tail_accuracy_drop_spectral"] - row["tail_accuracy_drop_frobenius"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(index).reset_index(drop=True)


def layer_rank_summary(layer_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gradient_rows = layer_metrics[layer_metrics["geometry"].eq("frobenius")].copy()
    point_summary = (
        gradient_rows.groupby(["warmup_steps", "seed"], as_index=False)
        .agg(
            mean_gradient_nuclear_rank=("gradient_nuclear_rank", "mean"),
            median_gradient_nuclear_rank=("gradient_nuclear_rank", "median"),
            max_gradient_nuclear_rank=("gradient_nuclear_rank", "max"),
            min_gradient_nuclear_rank=("gradient_nuclear_rank", "min"),
            layer_count=("parameter", "count"),
        )
        .sort_values(["warmup_steps", "seed"])
        .reset_index(drop=True)
    )
    layer_summary = (
        gradient_rows.groupby(["warmup_steps", "parameter", "shape"], as_index=False)
        .agg(
            mean_gradient_nuclear_rank=("gradient_nuclear_rank", "mean"),
            std_gradient_nuclear_rank=("gradient_nuclear_rank", "std"),
            min_gradient_nuclear_rank=("gradient_nuclear_rank", "min"),
            max_gradient_nuclear_rank=("gradient_nuclear_rank", "max"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["warmup_steps", "parameter"])
        .reset_index(drop=True)
    )
    return point_summary, layer_summary


def correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    value = x.corr(y, method=method)
    return float(value) if pd.notna(value) else float("nan")


def bootstrap_correlation(frame: pd.DataFrame, x_col: str, y_col: str, method: str, *, samples: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(0)
    values = []
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=float)
    for _ in range(samples):
        indices = rng.integers(0, len(frame), size=len(frame))
        sample = pd.DataFrame({"x": x[indices], "y": y[indices]})
        value = sample["x"].corr(sample["y"], method=method)
        if pd.notna(value):
            values.append(float(value))
    if not values:
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {
        "estimate": correlation(frame[x_col], frame[y_col], method),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def write_figure(points: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {250: "#0072B2", 500: "#009E73", 1000: "#D55E00", 2000: "#CC79A7"}
    for warmup_steps, group in points.groupby("warmup_steps", sort=True):
        axes[0].scatter(
            group["mean_gradient_nuclear_rank"],
            group["tail_output_drift_sq_ratio_spectral_over_fro"],
            s=34,
            alpha=0.8,
            color=colors.get(int(warmup_steps), "#666666"),
            label=f"{int(warmup_steps)} steps",
        )
        axes[1].scatter(
            group["tail_accuracy_before_frobenius"],
            group["tail_output_drift_sq_ratio_spectral_over_fro"],
            s=34,
            alpha=0.8,
            color=colors.get(int(warmup_steps), "#666666"),
            label=f"{int(warmup_steps)} steps",
        )
    for ax in axes:
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylabel("spectral / Fro squared tail drift")
    axes[0].set_xlabel("mean matrix-gradient nuclear rank")
    axes[0].set_title("Rank-side proxy")
    axes[1].set_xlabel("pre-update tail accuracy")
    axes[1].set_title("Tail quality")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("CIFAR-100-LT ResNet18 condition-proxy scatter")
    fig.tight_layout()
    path = FIGURE_DIR / "cifar100_resnet_condition_proxy_scatter.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_discussion(points: pd.DataFrame, summary: pd.DataFrame, figure_path: Path) -> None:
    rank_pearson = summary[
        summary["comparison"].eq("mean_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio")
        & summary["correlation"].eq("pearson")
    ].iloc[0]
    rank_spearman = summary[
        summary["comparison"].eq("mean_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio")
        & summary["correlation"].eq("spearman")
    ].iloc[0]
    tail_accuracy = summary[
        summary["comparison"].eq("tail_accuracy_before_vs_log_tail_drift_sq_ratio")
        & summary["correlation"].eq("spearman")
    ].iloc[0]
    table_columns = [
        "comparison",
        "correlation",
        "estimate",
        "ci95_low",
        "ci95_high",
        "n_points",
    ]
    readout = pd.DataFrame(
        [
            {
                "quantity": "points",
                "value": len(points),
            },
            {
                "quantity": "warmup checkpoints",
                "value": ",".join(str(int(x)) for x in sorted(points["warmup_steps"].unique())),
            },
            {
                "quantity": "drift ratio range",
                "value": f"{fmt(points['tail_output_drift_sq_ratio_spectral_over_fro'].min())} to {fmt(points['tail_output_drift_sq_ratio_spectral_over_fro'].max())}",
            },
            {
                "quantity": "tail accuracy range",
                "value": f"{fmt(points['tail_accuracy_before_frobenius'].min())} to {fmt(points['tail_accuracy_before_frobenius'].max())}",
            },
            {
                "quantity": "rank-proxy Pearson",
                "value": f"{fmt(rank_pearson['estimate'])} [{fmt(rank_pearson['ci95_low'])}, {fmt(rank_pearson['ci95_high'])}]",
            },
            {
                "quantity": "rank-proxy Spearman",
                "value": f"{fmt(rank_spearman['estimate'])} [{fmt(rank_spearman['ci95_low'])}, {fmt(rank_spearman['ci95_high'])}]",
            },
            {
                "quantity": "tail-accuracy Spearman",
                "value": f"{fmt(tail_accuracy['estimate'])} [{fmt(tail_accuracy['ci95_low'])}, {fmt(tail_accuracy['ci95_high'])}]",
            },
        ]
    )
    text = f"""# E11 CIFAR-100-LT ResNet18 Condition-Proxy Scatter

This generated diagnostic uses the completed CIFAR-100-LT ResNet18 checkpoint
sweep to ask whether a natural-task rank-side proxy tracks the observed
matched-head-gain drift ratio. Each point is one seed/checkpoint pair. The
x-axis proxy is the mean matrix-gradient nuclear rank across Conv/Linear
weights, while the y-axis is the paired squared tail-example logit drift ratio,
spectral over Frobenius.

This is intentionally weaker than the theorem's full downstream-aware condition
`nrank(G_H) > ssrank(B_T,A_T)`: it does not measure the tail downstream
sensitivity term. It is a falsification-oriented proxy check, not a replacement
for the sandwich condition.

![CIFAR-100-LT ResNet18 condition-proxy scatter](../{figure_path.as_posix()})

## Readout

{markdown_table(readout, ["quantity", "value"])}

## Correlations

{markdown_table(summary, table_columns)}

## Interpretation

The rank-side proxy alone is not a complete natural-task predictor. Across the
tested checkpoints, higher mean gradient nuclear rank co-varies with larger
observed drift ratios, while all observed ratios still remain below one. This
is consistent with the paper's boundary discipline: the theorem depends on a
downstream-aware tail sensitivity quantity, not only on head-gradient rank.

The diagnostic therefore strengthens the paper by making a reviewer-facing
caveat explicit. It reduces the risk of over-reading `nrank(G_H)` by itself and
sets up the next experiment: measure a real downstream-aware tail sensitivity
proxy for ResNet layers/checkpoints.

## Artifacts

- [scatter_points.csv](../results/e11_cifar100_resnet_condition_proxy_scatter/scatter_points.csv)
- [summary.csv](../results/e11_cifar100_resnet_condition_proxy_scatter/summary.csv)
- [layer_summary.csv](../results/e11_cifar100_resnet_condition_proxy_scatter/layer_summary.csv)
- [config.json](../results/e11_cifar100_resnet_condition_proxy_scatter/config.json)
"""
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    step_metrics = pd.read_csv(INPUT_DIR / "step_metrics.csv")
    layer_metrics = pd.read_csv(INPUT_DIR / "layer_metrics.csv")
    points = paired_step_points(step_metrics)
    rank_points, layer_summary = layer_rank_summary(layer_metrics)
    points = points.merge(rank_points, on=["warmup_steps", "seed"], validate="one_to_one")
    points["log_tail_output_drift_sq_ratio_spectral_over_fro"] = np.log(
        points["tail_output_drift_sq_ratio_spectral_over_fro"]
    )
    comparisons = [
        (
            "mean_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio",
            "mean_gradient_nuclear_rank",
            "log_tail_output_drift_sq_ratio_spectral_over_fro",
        ),
        (
            "median_gradient_nuclear_rank_vs_log_tail_drift_sq_ratio",
            "median_gradient_nuclear_rank",
            "log_tail_output_drift_sq_ratio_spectral_over_fro",
        ),
        (
            "tail_accuracy_before_vs_log_tail_drift_sq_ratio",
            "tail_accuracy_before_frobenius",
            "log_tail_output_drift_sq_ratio_spectral_over_fro",
        ),
    ]
    summary_rows = []
    for comparison, x_col, y_col in comparisons:
        for method in ["pearson", "spearman"]:
            row = bootstrap_correlation(points, x_col, y_col, method)
            row.update({"comparison": comparison, "correlation": method, "n_points": len(points)})
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)[
        ["comparison", "correlation", "estimate", "ci95_low", "ci95_high", "n_points"]
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUTPUT_DIR / "scatter_points.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    layer_summary.to_csv(OUTPUT_DIR / "layer_summary.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "source": str(INPUT_DIR),
                "point_unit": "seed_checkpoint_pair",
                "rank_proxy": "mean Conv/Linear matrix-gradient nuclear rank across layers",
                "bootstrap_samples": 2000,
                "bootstrap_seed": 0,
                "claim_boundary": "rank-side proxy only; downstream-aware tail sensitivity is not measured",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = write_figure(points)
    write_discussion(points, summary, figure_path)
    print(f"saved condition-proxy scatter to {OUTPUT_DIR}")
    print(f"points={len(points)}, summary rows={len(summary)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
