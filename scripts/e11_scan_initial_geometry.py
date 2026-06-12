from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry import default_config
from e11_condition_geometry.config import ProblemSpec
from e11_condition_geometry.diagnostics import derive_step_metrics, matrix_effective_rank, singular_values
from e11_condition_geometry.runner import build_problem, dtype_from_name


SCAN_DIR = Path("results/e11_candidate_scan")
FIGURE_DIR = Path("figures/e11_candidate_scan")
DISCUSSION_PATH = Path("discussion/e11_candidate_overlap_scan.md")
FEATURES = ["nrG", "stA", "condition_score", "mean_grad_rank_fraction"]


def candidate_specs() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []
    for d in [20, 40, 60]:
        for rank in [2, 5, 10]:
            if rank > d:
                continue
            for num_factors in [2, 5, 10]:
                for kappa in [1.0, 10.0, 100.0]:
                    for input_mult in [2, 10]:
                        specs.append(
                            ProblemSpec(
                                family="MatrixFactorizationInput",
                                setting=(
                                    f"scan MF d={d} r={rank} factors={num_factors} "
                                    f"kappa={kappa:.0e} input_mult={input_mult}"
                                ),
                                steps=0,
                                d=d,
                                rank=rank,
                                kappa=kappa,
                                num_factors=num_factors,
                                input_columns_multiplier=input_mult,
                            )
                        )
    for d in [20, 40, 60]:
        for rank in [2, 5, 10]:
            if rank > d:
                continue
            for kappa in [1.0, 10.0, 100.0]:
                for measurement_mult in [0.5, 1.0, 2.0]:
                    specs.append(
                        ProblemSpec(
                            family="MatrixSensing",
                            setting=(
                                f"scan MS d={d} r={rank} kappa={kappa:.0e} "
                                f"measurement_mult={measurement_mult:g}"
                            ),
                            steps=0,
                            d=d,
                            rank=rank,
                            kappa=kappa,
                            measurement_multiplier=measurement_mult,
                        )
                    )
    for hidden_dim in [16, 32, 64, 128]:
        for num_samples in [128, 512, 1024]:
            specs.append(
                ProblemSpec(
                    family="SmallMLPDigits",
                    setting=f"scan MLP hidden={hidden_dim} samples={num_samples}",
                    steps=0,
                    d=64,
                    rank=10,
                    hidden_dim=hidden_dim,
                    num_samples=num_samples,
                )
            )
    return specs


def initial_geometry(spec: ProblemSpec, seed: int, *, device: torch.device, dtype: torch.dtype) -> dict:
    problem = build_problem(spec, seed, device=device, dtype=dtype)
    for param in problem.parameters():
        param.grad = None
    loss = problem.loss()
    loss.backward()
    sigma_g = [singular_values(param.grad) for param in problem.parameters()]
    sigma_a = [singular_values(matrix) for matrix in problem.activation_matrices()]
    derived = derive_step_metrics(sigma_g, sigma_a)
    rank_fractions = []
    for values in sigma_g:
        ceiling = len(values)
        nr_g = matrix_effective_rank(values)
        if ceiling > 0 and np.isfinite(nr_g):
            rank_fractions.append(nr_g / ceiling)
    return {
        "problem_family": spec.family,
        "setting": spec.setting,
        "seed": seed,
        "d": spec.d,
        "rank": spec.rank,
        "kappa": spec.kappa,
        "num_factors": spec.num_factors,
        "input_columns_multiplier": spec.input_columns_multiplier,
        "measurement_multiplier": spec.measurement_multiplier,
        "hidden_dim": spec.hidden_dim,
        "num_samples": spec.num_samples,
        "loss": float(loss.detach().cpu()),
        "recovery_error": problem.recovery_error(),
        "nrG": derived["nrG"],
        "stA": derived["stA"],
        "condition_score": derived["condition_score"],
        "mean_grad_rank_fraction": float(np.mean(rank_fractions)) if rank_fractions else np.nan,
        "min_grad_rank_fraction": float(np.min(rank_fractions)) if rank_fractions else np.nan,
        "max_grad_rank_fraction": float(np.max(rank_fractions)) if rank_fractions else np.nan,
    }


def run_scan() -> pd.DataFrame:
    config = default_config()
    device = torch.device(config.device)
    dtype = dtype_from_name(config.dtype)
    rows = []
    for spec in candidate_specs():
        for seed in [0, 1, 2]:
            rows.append(initial_geometry(spec, seed, device=device, dtype=dtype))
    return pd.DataFrame(rows)


def setting_summary(scan: pd.DataFrame) -> pd.DataFrame:
    grouped = scan.groupby(
        [
            "problem_family",
            "setting",
            "d",
            "rank",
            "kappa",
            "num_factors",
            "input_columns_multiplier",
            "measurement_multiplier",
            "hidden_dim",
            "num_samples",
        ],
        as_index=False,
        observed=True,
    ).agg(
        loss=("loss", "mean"),
        recovery_error=("recovery_error", "mean"),
        nrG=("nrG", "mean"),
        stA=("stA", "mean"),
        condition_score=("condition_score", "mean"),
        mean_grad_rank_fraction=("mean_grad_rank_fraction", "mean"),
        nrG_std=("nrG", "std"),
        stA_std=("stA", "std"),
        condition_score_std=("condition_score", "std"),
        mean_grad_rank_fraction_std=("mean_grad_rank_fraction", "std"),
    )
    return grouped


def current_family_support() -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = pd.read_csv("results/e11_equal_update/step_metrics.csv")
    layers = pd.read_csv("results/e11_equal_update/layer_metrics.csv")
    layer_source = layers[
        np.isfinite(layers["nrG"])
        & np.isfinite(layers["update_rank_ceiling"])
        & (layers["update_rank_ceiling"] > 0)
    ].copy()
    layer_source["grad_rank_fraction"] = layer_source["nrG"] / layer_source["update_rank_ceiling"]
    rank_fraction = layer_source.groupby(
        ["problem_family", "setting", "seed", "step", "algo"],
        as_index=False,
        observed=True,
    ).agg(mean_grad_rank_fraction=("grad_rank_fraction", "mean"))
    current = steps.merge(
        rank_fraction,
        on=["problem_family", "setting", "seed", "step", "algo"],
        how="left",
    )
    current = current[current["algo"] == "Muon"].dropna(subset=FEATURES).copy()
    ranges = current.groupby("problem_family", as_index=False, observed=True).agg(
        **{f"{feature}_min": (feature, "min") for feature in FEATURES},
        **{f"{feature}_max": (feature, "max") for feature in FEATURES},
        **{f"{feature}_mean": (feature, "mean") for feature in FEATURES},
        **{f"{feature}_std": (feature, "std") for feature in FEATURES},
    )
    return current, ranges


def score_candidates(summary: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    current_mean = current[FEATURES].mean()
    current_std = current[FEATURES].std(ddof=0).replace(0.0, 1.0)
    family_centers = current.groupby("problem_family", observed=True)[FEATURES].mean()
    global_min = current[FEATURES].min()
    global_max = current[FEATURES].max()

    scored = summary.copy()
    z = (scored[FEATURES] - current_mean) / current_std
    center_z = (family_centers - current_mean) / current_std
    for family in family_centers.index:
        diff = z.to_numpy(dtype=float) - center_z.loc[family].to_numpy(dtype=float)
        scored[f"distance_to_{family}"] = np.sqrt(np.square(diff).sum(axis=1))
    scored["inside_global_feature_fraction"] = (
        (scored[FEATURES] >= global_min) & (scored[FEATURES] <= global_max)
    ).mean(axis=1)
    distance_cols = [column for column in scored.columns if column.startswith("distance_to_")]
    scored["nearest_current_family_distance"] = scored[distance_cols].min(axis=1)
    scored["second_nearest_current_family_distance"] = np.sort(scored[distance_cols].to_numpy(dtype=float), axis=1)[:, 1]
    scored["bridge_score"] = scored["inside_global_feature_fraction"] / (
        1.0 + scored["second_nearest_current_family_distance"]
    )
    return scored.sort_values(["inside_global_feature_fraction", "bridge_score"], ascending=[False, False])


def write_candidate_markdown(scored: pd.DataFrame, ranges: pd.DataFrame, paths: dict[str, Path]) -> None:
    top = scored.head(20).copy()
    columns = [
        "problem_family",
        "setting",
        "inside_global_feature_fraction",
        "bridge_score",
        "nearest_current_family_distance",
        "second_nearest_current_family_distance",
        "nrG",
        "stA",
        "condition_score",
        "mean_grad_rank_fraction",
    ]
    display = top[columns].copy()
    for column in display.select_dtypes(include=[float]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4g}")
    table = display.to_markdown(index=False)
    range_columns = ["problem_family"]
    for feature in FEATURES:
        range_columns.extend([f"{feature}_min", f"{feature}_max", f"{feature}_mean"])
    ranges_display = ranges[range_columns].copy()
    for column in ranges_display.select_dtypes(include=[float]).columns:
        ranges_display[column] = ranges_display[column].map(lambda value: f"{value:.4g}")
    ranges_table = ranges_display.to_markdown(index=False)
    text = f"""# E11 Candidate Overlap Scan

## Purpose

The leave-family-out predictor currently fails under substantial extrapolation. This scan searches for lightweight candidate settings whose initial spectral geometry lies closer to the existing global support, so the next experiment can test interpolation rather than extrapolation.

## Current Equal-Update Support

{ranges_table}

## Top Candidate Settings

The scan varies MF depth/rank/input width, Matrix Sensing rank/measurement count, and MLP width/sample count. `inside_global_feature_fraction` is the fraction of `nrG`, `stA`, condition score, and mean `nr(G_i)/r_i` that fall inside the current equal-update global range. `bridge_score` favors settings that are inside the global range while not being close to only one existing family center.

{table}

## Evidence

![Initial geometry scan]({Path('..') / paths['scan_figure']})

## Takeaway

The next useful experiment should add a small number of these intermediate settings rather than only adding more seeds to the current three separated families. That directly tests whether a Muon-win rule can interpolate within overlapping spectral support.
"""
    DISCUSSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCUSSION_PATH.write_text(text, encoding="utf-8")


def plot_scan(scored: pd.DataFrame, current: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "initial_geometry_support.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    family_colors = {
        "MatrixFactorizationInput": "#0072B2",
        "MatrixSensing": "#D55E00",
        "SmallMLPDigits": "#009E73",
    }
    for family, group in current.groupby("problem_family", observed=True):
        axes[0].scatter(
            group["mean_grad_rank_fraction"],
            group["condition_score"],
            s=8,
            alpha=0.18,
            color=family_colors.get(family, "#555555"),
            label=f"current {family}",
        )
        axes[1].scatter(
            group["nrG"],
            group["stA"],
            s=8,
            alpha=0.18,
            color=family_colors.get(family, "#555555"),
            label=f"current {family}",
        )
    top = scored.head(25)
    axes[0].scatter(
        top["mean_grad_rank_fraction"],
        top["condition_score"],
        marker="*",
        s=70,
        color="#CC79A7",
        edgecolor="#222222",
        linewidth=0.3,
        label="top candidates",
    )
    axes[1].scatter(
        top["nrG"],
        top["stA"],
        marker="*",
        s=70,
        color="#CC79A7",
        edgecolor="#222222",
        linewidth=0.3,
        label="top candidates",
    )
    axes[0].set_xlabel(r"mean $nr(G_i)/r_i$")
    axes[0].set_ylabel("condition score")
    axes[0].set_yscale("log")
    axes[0].set_title("Candidate support in rank-fraction/condition plane")
    axes[1].set_xlabel("nrG")
    axes[1].set_ylabel("stA")
    axes[1].set_title("Candidate support in nrG/stA plane")
    for ax in axes:
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    scan = run_scan()
    summary = setting_summary(scan)
    current, ranges = current_family_support()
    scored = score_candidates(summary, current)
    scan.to_csv(SCAN_DIR / "initial_geometry_scan.csv", index=False)
    scored.to_csv(SCAN_DIR / "candidate_setting_summary.csv", index=False)
    ranges.to_csv(SCAN_DIR / "current_equal_update_support.csv", index=False)
    figure = plot_scan(scored, current)
    write_candidate_markdown(scored, ranges, {"scan_figure": figure})
    print(f"saved scan rows={len(scan)} settings={len(summary)} to {SCAN_DIR}")
    print(f"figure: {figure}")
    print(f"discussion: {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
