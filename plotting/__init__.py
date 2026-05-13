"""Stateless plotting helpers for matrix experiment results."""

from .colors import (
    ALGORITHM_COLORS,
    DEFAULT_ALGORITHM_ORDER,
    DIMENSION_BASE_COLORS,
    DIMENSION_LINESTYLES,
    PLOT_COLORS,
    algorithm_color,
    algorithm_dimension_color,
    dimension_color,
    dimension_linestyle,
)
from .data import (
    mean_trajectory_frame,
    ordered_algos_in,
    ordered_dims_in,
    summary_table,
    trajectory_frame,
)
from .ablations import plot_scenario_metric
from .metrics import (
    plot_color_key,
    plot_metric_bar,
    plot_metric_overview,
)
from .phase import (
    plot_gap_heatmap,
    plot_metric_heatmap,
    plot_metric_lines,
    plot_optimizer_heatmaps,
)
from .trajectories import (
    plot_algorithm_dimension_grid,
    plot_algorithms_for_dimension,
    plot_all_mean_curves_combined,
    plot_dimensions_for_algorithm,
    plot_seed_variability_for_dimension,
)

__all__ = [
    "ALGORITHM_COLORS",
    "DEFAULT_ALGORITHM_ORDER",
    "DIMENSION_BASE_COLORS",
    "DIMENSION_LINESTYLES",
    "PLOT_COLORS",
    "algorithm_color",
    "algorithm_dimension_color",
    "dimension_color",
    "dimension_linestyle",
    "mean_trajectory_frame",
    "ordered_algos_in",
    "ordered_dims_in",
    "plot_algorithm_dimension_grid",
    "plot_algorithms_for_dimension",
    "plot_all_mean_curves_combined",
    "plot_color_key",
    "plot_dimensions_for_algorithm",
    "plot_metric_bar",
    "plot_gap_heatmap",
    "plot_metric_heatmap",
    "plot_metric_lines",
    "plot_metric_overview",
    "plot_optimizer_heatmaps",
    "plot_seed_variability_for_dimension",
    "plot_scenario_metric",
    "summary_table",
    "trajectory_frame",
]
