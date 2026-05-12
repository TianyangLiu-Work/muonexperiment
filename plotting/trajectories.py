"""Stateless trajectory plot functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .colors import algorithm_color, algorithm_dimension_color, dimension_linestyle
from .data import mean_trajectory_frame, ordered_algos_in, ordered_dims_in
from .metrics import empty_figure

TrajectoryDict = Mapping[tuple[str, int, int], Mapping[str, Sequence[float]]]


def loss_scale_name(log_y: bool) -> str:
    return "log loss" if log_y else "loss"


def apply_loss_axis(ax: Axes, *, log_y: bool, ylabel: str = "mean loss") -> None:
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(f"{ylabel} (log scale)" if log_y else ylabel)
    ax.grid(alpha=0.3)


def plot_algorithms_for_dimension(
    trajectories: TrajectoryDict,
    d: int,
    algorithm_colors: Mapping[str, str] | None = None,
    *,
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    mean_tf = mean_trajectory_frame(trajectories)
    sub_d = mean_tf[mean_tf["d"] == int(d)]
    if sub_d.empty:
        return empty_figure(f"No data for d={d}")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for algo in ordered_algos_in(sub_d):
        sub = sub_d[sub_d["algo"] == algo]
        ax.plot(
            sub["step"],
            sub["loss"],
            color=algorithm_color(algo, algorithm_colors),
            linewidth=2.6,
            label=algo,
        )
    apply_loss_axis(ax, log_y=log_y)
    ax.set_title(f"Same dimension comparison: d={d} ({loss_scale_name(log_y)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_dimensions_for_algorithm(
    trajectories: TrajectoryDict,
    algo: str,
    algorithm_colors: Mapping[str, str] | None = None,
    *,
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    mean_tf = mean_trajectory_frame(trajectories)
    sub_algo = mean_tf[mean_tf["algo"] == algo]
    if sub_algo.empty:
        return empty_figure(f"No data for {algo}")

    dims = ordered_dims_in(mean_tf)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for d in dims:
        sub = sub_algo[sub_algo["d"] == d]
        ax.plot(
            sub["step"],
            sub["loss"],
            color=algorithm_dimension_color(algo, d, dims, algorithm_colors),
            linestyle=dimension_linestyle(d, dims),
            linewidth=2.6,
            label=f"d={d}",
        )
    apply_loss_axis(ax, log_y=log_y)
    ax.set_title(f"Same algorithm comparison: {algo} ({loss_scale_name(log_y)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_all_mean_curves_combined(
    trajectories: TrajectoryDict,
    algorithm_colors: Mapping[str, str] | None = None,
    *,
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    mean_tf = mean_trajectory_frame(trajectories)
    if mean_tf.empty:
        return empty_figure()

    dims = ordered_dims_in(mean_tf)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for algo in ordered_algos_in(mean_tf):
        sub_algo = mean_tf[mean_tf["algo"] == algo]
        for d in dims:
            sub = sub_algo[sub_algo["d"] == d]
            ax.plot(
                sub["step"],
                sub["loss"],
                color=algorithm_dimension_color(algo, d, dims, algorithm_colors),
                linestyle=dimension_linestyle(d, dims),
                linewidth=2.1,
                label=f"{algo}, d={d}",
            )
    apply_loss_axis(ax, log_y=log_y)
    ax.set_title(f"All algorithms and all dimensions: mean {loss_scale_name(log_y)} curves")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig, ax


def plot_algorithm_dimension_grid(
    trajectories: TrajectoryDict,
    algorithm_colors: Mapping[str, str] | None = None,
    *,
    log_y: bool = False,
) -> tuple[Figure, Sequence[Sequence[Axes]]]:
    mean_tf = mean_trajectory_frame(trajectories)
    if mean_tf.empty:
        fig, ax = empty_figure()
        return fig, ((ax,),)

    algos = ordered_algos_in(mean_tf)
    dims = ordered_dims_in(mean_tf)
    fig, axes = plt.subplots(
        len(algos),
        len(dims),
        figsize=(4.2 * len(dims), 2.5 * len(algos)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, algo in enumerate(algos):
        for j, d in enumerate(dims):
            ax = axes[i][j]
            sub = mean_tf[(mean_tf["algo"] == algo) & (mean_tf["d"] == d)]
            ax.plot(
                sub["step"],
                sub["loss"],
                color=algorithm_dimension_color(algo, d, dims, algorithm_colors),
                linestyle=dimension_linestyle(d, dims),
                linewidth=2.2,
            )
            apply_loss_axis(ax, log_y=log_y)
            ax.set_title(f"{algo}, d={d}", fontsize=10)
    fig.suptitle(f"Algorithm-dimension grid ({loss_scale_name(log_y)})", y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_seed_variability_for_dimension(
    trajectories: TrajectoryDict,
    d: int,
    algorithm_colors: Mapping[str, str] | None = None,
    *,
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    has_data = any(dd == int(d) for _, dd, _ in trajectories)
    if not has_data:
        return empty_figure(f"No data for d={d}")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    seen: set[str] = set()
    for (algo, dd, seed), traj in trajectories.items():
        if dd != int(d):
            continue
        label = algo if algo not in seen else None
        seen.add(algo)
        ax.plot(
            traj["loss"],
            color=algorithm_color(algo, algorithm_colors),
            linewidth=0.85,
            alpha=0.28,
            label=label,
        )
    apply_loss_axis(ax, log_y=log_y, ylabel="loss")
    ax.set_title(f"All seed trajectories at d={d} ({loss_scale_name(log_y)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax
