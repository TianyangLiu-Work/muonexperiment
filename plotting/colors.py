"""Shared color definitions for experiment plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.colors import to_hex, to_rgb


ALGORITHM_COLORS: dict[str, str] = {
    "Muon": "#0072B2",
    "Muon-Exact": "#56B4E9",
    "Muon-NS-1": "#1F9AC7",
    "Muon-NS-10": "#085C83",
    "Muon-Truncated": "#7B92D4",
    "Muon-RandSVD": "#5E3C99",
    "Shampoo": "#009E73",
    "Adam": "#D55E00",
    "SGD": "#CC79A7",
    "NormalizedSGD": "#E69F00",
    "SpectralNormSGD": "#A6761D",
    "LayerwiseNormalizedSGD": "#8C564B",
}
DIMENSION_BASE_COLORS: tuple[str, ...] = ("#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
DIMENSION_LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")
DEFAULT_ALGORITHM_ORDER: tuple[str, ...] = tuple(ALGORITHM_COLORS)

PLOT_COLORS = {
    "algorithms": ALGORITHM_COLORS,
    "dimension_base": DIMENSION_BASE_COLORS,
    "dimension_linestyles": DIMENSION_LINESTYLES,
}


def blend_with_white(color: str, amount: float) -> str:
    rgb = to_rgb(color)
    mixed = tuple((1.0 - amount) * channel + amount for channel in rgb)
    return to_hex(mixed)


def algorithm_color(
    algo: str,
    algorithm_colors: Mapping[str, str] | None = None,
) -> str:
    colors = ALGORITHM_COLORS if algorithm_colors is None else algorithm_colors
    return colors.get(algo, "#4D4D4D")


def dimension_rank(d: int, dims: Sequence[int]) -> float:
    dims = [int(value) for value in dims]
    if len(dims) <= 1:
        return 0.0
    return dims.index(int(d)) / (len(dims) - 1)


def algorithm_dimension_color(
    algo: str,
    d: int,
    dims: Sequence[int],
    algorithm_colors: Mapping[str, str] | None = None,
) -> str:
    rank = dimension_rank(d, dims)
    white_mix = 0.55 * (1.0 - rank)
    return blend_with_white(algorithm_color(algo, algorithm_colors), white_mix)


def dimension_color(
    d: int,
    dims: Sequence[int],
    dimension_base_colors: Sequence[str] | None = None,
) -> str:
    colors = DIMENSION_BASE_COLORS if dimension_base_colors is None else dimension_base_colors
    dims = [int(value) for value in dims]
    idx = dims.index(int(d)) if int(d) in dims else 0
    if len(dims) == 1:
        return colors[-1]
    pos = round(idx * (len(colors) - 1) / (len(dims) - 1))
    return colors[pos]


def dimension_linestyle(
    d: int,
    dims: Sequence[int],
    linestyles: Sequence[str] | None = None,
) -> str:
    styles = DIMENSION_LINESTYLES if linestyles is None else linestyles
    dims = [int(value) for value in dims]
    idx = dims.index(int(d)) if int(d) in dims else 0
    return styles[idx % len(styles)]
