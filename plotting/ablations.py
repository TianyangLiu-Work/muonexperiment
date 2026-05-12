"""Stateless ablation plot functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .colors import algorithm_color
from .data import ordered_algos_in
from .metrics import empty_figure


def ordered_values_in(df: pd.DataFrame, column: str) -> list:
    values = df[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna().astype(str))
        return [value for value in values.cat.categories if str(value) in present]
    return list(dict.fromkeys(values.dropna().tolist()))


def plot_scenario_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    scenario_col: str = "scenario",
    log_y: bool = False,
) -> tuple[Figure, Axes]:
    if df.empty:
        return empty_figure()
    if scenario_col not in df:
        return empty_figure(f"Missing column: {scenario_col}")
    if metric not in df:
        return empty_figure(f"Missing metric: {metric}")

    summary = df.groupby([scenario_col, "algo"], as_index=False, observed=True)[metric].mean()
    scenarios = ordered_values_in(df, scenario_col)
    if not scenarios:
        return empty_figure()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x_positions = list(range(len(scenarios)))
    for algo in ordered_algos_in(summary):
        sub = summary[summary["algo"] == algo].set_index(scenario_col).reindex(scenarios)
        ax.plot(
            x_positions,
            sub[metric],
            marker="o",
            linewidth=2.4,
            color=algorithm_color(algo),
            label=algo,
        )

    ax.set_title(f"{title} (log scale)" if log_y else title)
    ax.set_xlabel("scenario")
    ax.set_ylabel(f"{ylabel} (log scale)" if log_y else ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax
