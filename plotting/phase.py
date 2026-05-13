"""Stateless plotting helpers for phase-diagram experiments."""

from __future__ import annotations

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .colors import algorithm_color
from .metrics import empty_figure


def plot_metric_heatmap(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    metric: str,
    title: str,
    agg: str = "median",
    log_color: bool = False,
    cmap: str = "viridis",
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Axes]:
    if df.empty:
        return empty_figure()
    missing = [column for column in (x, y, metric) if column not in df]
    if missing:
        return empty_figure(f"Missing column: {', '.join(missing)}")

    pivot = df.pivot_table(index=y, columns=x, values=metric, aggfunc=agg, observed=True)
    values = pivot.to_numpy(dtype=float)
    label = metric
    if log_color:
        values = np.log10(np.clip(values, 1e-300, None))
        label = f"log10({metric})"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig = ax.figure
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    _label_heatmap_axes(ax, pivot, x=x, y=y)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    return fig, ax


def plot_optimizer_heatmaps(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    metric: str,
    title: str,
    optimizer_col: str = "algo",
    agg: str = "median",
    log_color: bool = False,
    cmap: str = "viridis",
) -> tuple[Figure, np.ndarray]:
    if df.empty:
        fig, ax = empty_figure()
        return fig, np.asarray([[ax]])
    missing = [column for column in (x, y, metric, optimizer_col) if column not in df]
    if missing:
        fig, ax = empty_figure(f"Missing column: {', '.join(missing)}")
        return fig, np.asarray([[ax]])

    algos = _ordered_unique(df[optimizer_col])
    panels = []
    for algo in algos:
        pivot = df[df[optimizer_col] == algo].pivot_table(
            index=y,
            columns=x,
            values=metric,
            aggfunc=agg,
            observed=True,
        )
        values = pivot.to_numpy(dtype=float)
        if log_color:
            values = np.log10(np.clip(values, 1e-300, None))
        panels.append((algo, pivot, values))

    finite_values = np.concatenate([values[np.isfinite(values)].ravel() for _, _, values in panels])
    vmin = float(finite_values.min()) if finite_values.size else None
    vmax = float(finite_values.max()) if finite_values.size else None

    cols = min(3, len(algos))
    rows = math.ceil(len(algos) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.0 * rows), squeeze=False)
    image = None
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (algo, pivot, values) in zip(axes.ravel(), panels):
        ax.set_axis_on()
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        _label_heatmap_axes(ax, pivot, x=x, y=y)
        ax.set_title(str(algo), color=algorithm_color(str(algo)))
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist())
        cbar.set_label(f"log10({metric})" if log_color else metric)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_gap_heatmap(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    metric: str,
    left_algo: str,
    right_algo: str,
    title: str,
    optimizer_col: str = "algo",
    agg: str = "median",
    log_gap: bool = True,
    cmap: str = "RdBu_r",
) -> tuple[Figure, Axes]:
    if df.empty:
        return empty_figure()
    missing = [column for column in (x, y, metric, optimizer_col) if column not in df]
    if missing:
        return empty_figure(f"Missing column: {', '.join(missing)}")

    grouped = df.groupby([y, x, optimizer_col], observed=True)[metric].agg(agg).reset_index()
    pivot = grouped.pivot_table(
        index=[y, x],
        columns=optimizer_col,
        values=metric,
        observed=True,
    )
    if left_algo not in pivot or right_algo not in pivot:
        return empty_figure(f"Need both {left_algo} and {right_algo}")

    left = pivot[left_algo].astype(float)
    right = pivot[right_algo].astype(float)
    if log_gap:
        gap = np.log10(np.clip(left, 1e-300, None)) - np.log10(np.clip(right, 1e-300, None))
        label = f"log10({left_algo}) - log10({right_algo})"
    else:
        gap = left - right
        label = f"{left_algo} - {right_algo}"
    gap_frame = gap.reset_index(name="gap").pivot(index=y, columns=x, values="gap")

    values = gap_frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
        limit = float(np.nanmax(np.abs(finite)))
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit) if limit > 0 else None
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    _label_heatmap_axes(ax, gap_frame, x=x, y=y)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    return fig, ax


def plot_metric_lines(
    df: pd.DataFrame,
    *,
    x: str,
    metric: str,
    title: str,
    ylabel: str,
    hue: str = "algo",
    agg: str = "median",
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    if df.empty:
        return empty_figure()
    missing = [column for column in (x, metric, hue) if column not in df]
    if missing:
        return empty_figure(f"Missing column: {', '.join(missing)}")

    summary = df.groupby([x, hue], as_index=False, observed=True)[metric].agg(agg)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for value in _ordered_unique(summary[hue]):
        sub = summary[summary[hue] == value].sort_values(x)
        ax.plot(
            sub[x],
            sub[metric],
            marker="o",
            linewidth=2.4,
            color=algorithm_color(str(value)),
            label=str(value),
        )
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-4)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def _label_heatmap_axes(ax: Axes, pivot: pd.DataFrame, *, x: str, y: str) -> None:
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_format_tick(value) for value in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_format_tick(value) for value in pivot.index])


def _format_tick(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _ordered_unique(values: pd.Series) -> list:
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna().astype(str))
        return [value for value in values.cat.categories if str(value) in present]
    return list(dict.fromkeys(values.dropna().tolist()))
