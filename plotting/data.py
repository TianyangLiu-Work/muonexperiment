"""Stateless data transforms used by plotting functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .colors import DEFAULT_ALGORITHM_ORDER


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "runs": ("seed", "count"),
        "min_loss_mean": ("min_loss", "mean"),
        "final_loss_mean": ("final_loss", "mean"),
        "time_s_mean": ("time_s", "mean"),
    }
    if "actual_steps" in df:
        aggregations["actual_steps_mean"] = ("actual_steps", "mean")
    if "stopped_early" in df:
        aggregations["stopped_early_rate"] = ("stopped_early", "mean")
    return df.groupby(["algo", "d"], as_index=False).agg(**aggregations)


def trajectory_frame(
    trajectories: Mapping[tuple[str, int, int], Mapping[str, Sequence[float]]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (algo, d, seed), traj in trajectories.items():
        for step_idx, loss in enumerate(traj["loss"], start=1):
            records.append(
                {
                    "algo": algo,
                    "d": int(d),
                    "seed": int(seed),
                    "step": step_idx,
                    "loss": float(loss),
                }
            )
    return pd.DataFrame(records, columns=["algo", "d", "seed", "step", "loss"])


def mean_trajectory_frame(
    trajectories: Mapping[tuple[str, int, int], Mapping[str, Sequence[float]]],
) -> pd.DataFrame:
    tf = trajectory_frame(trajectories)
    if tf.empty:
        return pd.DataFrame(columns=["algo", "d", "step", "loss"])
    return tf.groupby(["algo", "d", "step"], as_index=False)["loss"].mean()


def ordered_algos_in(
    df: pd.DataFrame,
    algorithm_order: Sequence[str] = DEFAULT_ALGORITHM_ORDER,
) -> list[str]:
    present = set(df["algo"].unique()) if "algo" in df else set()
    ordered = [algo for algo in algorithm_order if algo in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def ordered_dims_in(df: pd.DataFrame) -> list[int]:
    if "d" not in df:
        return []
    return sorted(int(d) for d in df["d"].unique())
