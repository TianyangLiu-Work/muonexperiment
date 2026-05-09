"""Stateless metric plot functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .colors import algorithm_color, algorithm_dimension_color, dimension_color, dimension_linestyle
from .data import ordered_algos_in, ordered_dims_in, summary_table


def empty_figure(title: str = "No data") -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_axis_off()
    ax.set_title(title)
    return fig, ax


def plot_color_key(
    df: pd.DataFrame,
    algorithm_colors: Mapping[str, str] | None = None,
) -> tuple[Figure, Sequence[Axes]]:
    dims = ordered_dims_in(df)
    algos = ordered_algos_in(df)
    if not dims or not algos:
        fig, ax = empty_figure()
        return fig, (ax,)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.2))
    for idx, algo in enumerate(algos):
        axes[0].plot(
            [0, 1],
            [idx, idx],
            color=algorithm_color(algo, algorithm_colors),
            linewidth=5,
            label=algo,
        )
    axes[0].set_title("Algorithm hue")
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].legend(loc="center", ncol=min(3, len(algos)), frameon=False)

    demo_algo = algos[0]
    for idx, d in enumerate(dims):
        axes[1].plot(
            [0, 1],
            [idx, idx],
            color=algorithm_dimension_color(demo_algo, d, dims, algorithm_colors),
            linestyle=dimension_linestyle(d, dims),
            linewidth=5,
            label=f"d={d}",
        )
    axes[1].set_title("Dimension shade/line style")
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[1].legend(loc="center", ncol=max(1, len(dims)), frameon=False)
    fig.tight_layout()
    return fig, axes


def plot_metric_overview(
    df: pd.DataFrame,
    algorithm_colors: Mapping[str, str] | None = None,
) -> tuple[Figure, Sequence[Axes]]:
    if df.empty:
        fig, ax = empty_figure()
        return fig, (ax,)

    summary = summary_table(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics = [
        ("actual_steps_mean", "Mean executed steps", "steps", False),
        ("time_s_mean", "Mean wall-clock", "seconds", False),
        ("min_loss_mean", "Mean min loss", "loss", True),
    ]
    for ax, (metric, title, ylabel, log_y) in zip(axes, metrics):
        for algo in ordered_algos_in(summary):
            sub = summary[summary["algo"] == algo]
            ax.plot(
                sub["d"],
                sub[metric],
                marker="o",
                linewidth=2.4,
                color=algorithm_color(algo, algorithm_colors),
                label=algo,
            )
        ax.set_title(title)
        ax.set_xlabel("d")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    log_y: bool = False,
    dimension_base_colors: Sequence[str] | None = None,
) -> tuple[Figure, Axes]:
    if df.empty:
        return empty_figure()

    summary = summary_table(df)
    if metric not in summary and metric.endswith("_mean"):
        source_metric = metric.removesuffix("_mean")
        if source_metric in df:
            summary = summary.merge(
                df.groupby(["algo", "d"], as_index=False)[source_metric]
                .mean()
                .rename(columns={source_metric: metric}),
                on=["algo", "d"],
                how="left",
            )
    if metric not in summary:
        return empty_figure(f"Missing metric: {metric}")

    dims = ordered_dims_in(summary)
    bar_colors = [dimension_color(d, dims, dimension_base_colors) for d in dims]
    pivot = summary.pivot(index="algo", columns="d", values=metric).reindex(
        ordered_algos_in(summary)
    )
    ax = pivot.plot(kind="bar", figsize=(9.5, 4.5), width=0.82, color=bar_colors)
    fig = ax.figure
    ax.set_title(title)
    ax.set_xlabel("algorithm")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="d", fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    fig.tight_layout()
    return fig, ax
