from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ALGO_COLORS = {"Adam": "#D55E00", "Muon": "#0072B2"}


def algo_color(algo: str) -> str:
    return ALGO_COLORS.get(str(algo), "#555555")


def short_title(setting: str) -> str:
    return str(setting).replace(" kappa=", "\n$\\kappa=$")


def panel_grid(count: int, *, width: float = 4.5, height: float = 3.4):
    ncols = min(3, max(1, count))
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    return fig, axes.ravel()


def make_loss_curves(steps: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "loss_curves.png"
    settings = list(dict.fromkeys(steps["setting"]))
    fig, axes = panel_grid(len(settings), width=4.7, height=3.3)
    for ax, setting in zip(axes, settings):
        sub = steps[steps["setting"] == setting]
        for _, group in sub.groupby("run_id", observed=True, sort=False):
            ax.plot(group["step"], group["loss"], color=algo_color(group["algo"].iloc[0]), alpha=0.28, lw=0.9)
        ax.set_title(f"{sub['problem_family'].iloc[0]}\n{short_title(setting)}", fontsize=8.5)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.25)
    for ax in axes[len(settings) :]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=algo_color(algo), lw=2.2, label=algo) for algo in ["Adam", "Muon"]]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Loss trajectories across all three problem families", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_geometry_separation(summary: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "geometry_separation.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    axes = axes.ravel()
    x = np.arange(len(summary))
    labels = [f"{row.problem_family}\n{row.setting}" for row in summary.itertuples()]
    for ax, column, title, ylabel in [
        (axes[0], "mean_delta_nrG", "Matched gradient-rank difference", "Muon - Adam mean nrG"),
        (axes[1], "mean_delta_stA", "Matched activation-rank difference", "Muon - Adam mean stA"),
        (axes[2], "centroid_distance_std", "Geometry centroid distance", "std units"),
        (axes[3], "nearest_centroid_balanced_accuracy", "Optimizer recovery from geometry", "balanced accuracy"),
    ]:
        ax.axhline(0.0 if column != "nearest_centroid_balanced_accuracy" else 0.5, color="#555555", ls="--", lw=0.8)
        ax.plot(x, summary[column], marker="o", color="#0072B2")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.grid(alpha=0.25)
    axes[3].set_ylim(0.0, 1.05)
    fig.suptitle("Muon-vs-Adam geometry separation by problem setting", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_predicted_decrease(steps: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "predicted_decrease_vs_observed.png"
    rows = steps[np.isfinite(steps["delta_loss"])].copy()
    families = list(dict.fromkeys(rows["problem_family"]))
    fig, axes = plt.subplots(len(families), 2, figsize=(11.0, 3.8 * len(families)), squeeze=False)
    for row_idx, family in enumerate(families):
        family_rows = rows[rows["problem_family"] == family]
        for col_idx, predictor in enumerate(["delta_gd_pred", "delta_spec_pred"]):
            ax = axes[row_idx, col_idx]
            for algo, group in family_rows.groupby("algo", observed=True):
                ax.scatter(
                    group[predictor].clip(lower=1e-300),
                    group["delta_loss"],
                    s=12,
                    alpha=0.42,
                    color=algo_color(algo),
                    label=algo if row_idx == 0 and col_idx == 0 else None,
                )
            ax.axhline(0.0, color="#333333", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=max(np.nanquantile(np.abs(rows["delta_loss"].dropna()), 0.05), 1e-16))
            ax.set_title(f"{family}: observed $\\Delta L$ vs {predictor}")
            ax.set_xlabel(predictor)
            ax.set_ylabel("$\\Delta L = L_t - L_{t+1}$")
            ax.grid(alpha=0.25)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=algo_color(algo), label=algo) for algo in ["Adam", "Muon"]]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Paper-style predicted decrease vs observed one-step loss decrease", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_condition_trajectories(steps: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "condition_score_trajectories.png"
    settings = list(dict.fromkeys(steps["setting"]))
    fig, axes = panel_grid(len(settings), width=4.7, height=3.4)
    rows = steps[np.isfinite(steps["condition_score"]) & (steps["condition_score"] > 0)].copy()
    for ax, setting in zip(axes, settings):
        sub = rows[rows["setting"] == setting]
        for _, group in sub.groupby("run_id", observed=True, sort=False):
            ax.plot(group["condition_score"], group["loss"], color=algo_color(group["algo"].iloc[0]), alpha=0.32, lw=0.9)
        ax.set_xscale("log")
        ax.set_title(f"{sub['problem_family'].iloc[0]}\n{short_title(setting)}", fontsize=8.5)
        ax.set_xlabel("condition score")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
    for ax in axes[len(settings) :]:
        ax.axis("off")
    fig.suptitle("Condition-score trajectories vs loss", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_3d_mean(steps: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "mean_3d_condition_loss.png"
    fig = plt.figure(figsize=(9.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    for _, group in steps.groupby("run_id", observed=True, sort=False):
        group = group.sort_values("step")
        ax.plot(group["nrG"], group["stA"], group["loss"], color=algo_color(group["algo"].iloc[0]), alpha=0.24, lw=0.8)
    ax.set_xlabel("mean nr(G)")
    ax.set_ylabel("mean sr(A)")
    ax.set_zlabel("loss")
    ax.set_title("Mean rank-geometry and loss trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_3d_layerwise(layers: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "layerwise_3d_condition_loss.png"
    fig = plt.figure(figsize=(9.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    for _, group in layers.groupby(["run_id", "layer"], observed=True, sort=False):
        group = group.sort_values("step")
        ax.plot(group["nrG"], group["stA"], group["loss"], color=algo_color(group["algo"].iloc[0]), alpha=0.08, lw=0.6)
    ax.set_xlabel("layerwise nr(G_i)")
    ax.set_ylabel("layerwise sr(A_i)")
    ax.set_zlabel("loss")
    ax.set_title("Layerwise rank-geometry and loss trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_volatility_robustness(volatility: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "volatility_robustness.png"
    key_metrics = [
        "norm_rank_plane_mean_speed",
        "norm_condition_mean_speed",
        "per_update_rank_plane_mean_speed",
        "per_update_condition_mean_speed",
        "condition_score_std_speed",
        "delta_spec_pred_mean_rel_speed",
        "loss_mean_rel_speed",
    ]
    labels = {
        "norm_rank_plane_mean_speed": "rank-plane speed",
        "norm_condition_mean_speed": "condition speed",
        "per_update_rank_plane_mean_speed": "rank-plane speed per update",
        "per_update_condition_mean_speed": "condition speed per update",
        "condition_score_std_speed": "condition volatility",
        "delta_spec_pred_mean_rel_speed": "Spec-pred rel speed",
        "loss_mean_rel_speed": "loss rel speed",
    }
    rows = volatility[
        volatility["metric"].isin(key_metrics)
        & volatility["problem_family"].isin(["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"])
    ].copy()
    rows["metric"] = pd.Categorical(rows["metric"], categories=key_metrics, ordered=True)
    rows = rows.sort_values(["metric", "problem_family"])
    families = ["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(9.0, 1.65 * len(key_metrics)), sharex=True)
    if len(key_metrics) == 1:
        axes = [axes]
    y_positions = np.arange(len(families))
    for ax, metric in zip(axes, key_metrics):
        sub = rows[rows["metric"] == metric].set_index("problem_family").reindex(families).reset_index()
        x = sub["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
        lo = sub["ratio_ci95_low"].to_numpy(dtype=float)
        hi = sub["ratio_ci95_high"].to_numpy(dtype=float)
        xerr = np.vstack([x - lo, hi - x])
        ax.axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.errorbar(x, y_positions, xerr=xerr, fmt="o", color="#0072B2", ecolor="#444444", capsize=3)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(families, fontsize=8)
        ax.set_xscale("log")
        ax.set_title(labels[metric], fontsize=9)
        ax.grid(alpha=0.22, axis="x")
    axes[-1].set_xlabel("geometric mean speed ratio: Muon / Adam (95% CI)")
    fig.suptitle("Muon-vs-Adam step-to-step volatility ratios", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_update_spectrum_robustness(update_spectrum: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "update_spectrum_robustness.png"
    key_metrics = ["nrUpdate", "stUpdate", "nrUpdateFrac", "stUpdateFrac", "update_flatness"]
    labels = {
        "nrUpdate": "update effective rank",
        "stUpdate": "update stable rank",
        "nrUpdateFrac": "effective-rank fraction",
        "stUpdateFrac": "stable-rank fraction",
        "update_flatness": "update flatness st/nr",
    }
    families = ["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    rows = update_spectrum[
        update_spectrum["metric"].isin(key_metrics)
        & update_spectrum["problem_family"].isin(families)
    ].copy()
    rows["metric"] = pd.Categorical(rows["metric"], categories=key_metrics, ordered=True)
    rows = rows.sort_values(["metric", "problem_family"])
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(9.0, 1.85 * len(key_metrics)), sharex=True)
    if len(key_metrics) == 1:
        axes = [axes]
    y_positions = np.arange(len(families))
    for ax, metric in zip(axes, key_metrics):
        sub = rows[rows["metric"] == metric].set_index("problem_family").reindex(families).reset_index()
        x = sub["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
        lo = sub["ratio_ci95_low"].to_numpy(dtype=float)
        hi = sub["ratio_ci95_high"].to_numpy(dtype=float)
        xerr = np.vstack([x - lo, hi - x])
        ax.axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.errorbar(x, y_positions, xerr=xerr, fmt="o", color="#009E73", ecolor="#444444", capsize=3)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(families, fontsize=8)
        ax.set_xscale("log")
        ax.set_title(labels[metric], fontsize=9)
        ax.grid(alpha=0.22, axis="x")
    axes[-1].set_xlabel("geometric mean update-spectrum ratio: Muon / Adam (95% CI)")
    fig.suptitle("Muon-vs-Adam update-matrix spectral structure", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_update_transmission_heatmap(transmission: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "update_transmission_heatmap.png"
    pairs = [
        ("stUpdateFrac", "relative_loss_decrease", "stFrac -> rel loss drop"),
        ("update_flatness", "relative_loss_decrease", "flatness -> rel loss drop"),
        ("update_grad_inner", "relative_loss_decrease", "1st-order -> rel loss drop"),
        ("update_grad_cosine", "relative_loss_decrease", "grad cosine -> rel loss drop"),
        ("stUpdateFrac", "norm_rank_movement", "stFrac -> rank move"),
        ("update_flatness", "norm_rank_movement", "flatness -> rank move"),
        ("relative_update_fro_norm", "relative_loss_decrease", "update norm -> rel loss drop"),
    ]
    groups = ["All", "Adam", "Muon", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    matrix = np.full((len(groups), len(pairs)), np.nan)
    labels = []
    for col, (predictor, outcome, label) in enumerate(pairs):
        labels.append(label)
        for row, group in enumerate(groups):
            match = transmission[
                (transmission["group"] == group)
                & (transmission["predictor"] == predictor)
                & (transmission["outcome"] == outcome)
            ]
            if not match.empty:
                matrix[row, col] = float(match["spearman"].iloc[0])

    fig, ax = plt.subplots(figsize=(10.4, 4.5))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e6e6e6")
    image = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "n/a" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=7, color="#111111")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Does update-spectrum geometry transmit to next-step progress or movement?")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_first_order_calibration(steps: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "first_order_calibration.png"
    rows = steps[
        np.isfinite(steps["delta_loss"])
        & np.isfinite(steps["update_grad_inner"])
        & (steps["delta_loss"] > 0)
        & (steps["update_grad_inner"] > 0)
    ].copy()
    families = list(dict.fromkeys(rows["problem_family"]))
    fig, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 3.7), squeeze=False)
    for ax, family in zip(axes.ravel(), families):
        sub = rows[rows["problem_family"] == family]
        for algo, group in sub.groupby("algo", observed=True, sort=False):
            ax.scatter(
                group["update_grad_inner"],
                group["delta_loss"],
                s=13,
                alpha=0.38,
                color=algo_color(algo),
                label=algo,
            )
        bounds = [
            sub["update_grad_inner"].min(),
            sub["update_grad_inner"].max(),
            sub["delta_loss"].min(),
            sub["delta_loss"].max(),
        ]
        lo = max(min(bounds), 1e-300)
        hi = max(bounds)
        ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.8, ls="--", label="y=x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(family, fontsize=9)
        ax.set_xlabel(r"$\langle G, W_t-W_{t+1}\rangle$")
        ax.set_ylabel(r"observed $\Delta L$")
        ax.grid(alpha=0.25)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=algo_color(algo), label=algo) for algo in ["Adam", "Muon"]]
    handles.append(Line2D([0], [0], color="#333333", lw=0.8, ls="--", label="y=x"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Observed one-step decrease vs first-order gradient-update inner product", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_polar_alignment_identity(layers: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "polar_alignment_identity.png"
    rows = layers[
        (layers["algo"] == "Muon")
        & np.isfinite(layers["update_grad_cosine"])
        & np.isfinite(layers["nrG"])
        & np.isfinite(layers["update_rank_ceiling"])
        & (layers["update_rank_ceiling"] > 0)
    ].copy()
    rows["cosine_sq"] = rows["update_grad_cosine"] ** 2
    rows["gradient_rank_fraction"] = rows["nrG"] / rows["update_rank_ceiling"]
    families = list(dict.fromkeys(rows["problem_family"]))
    fig, axes = plt.subplots(1, len(families), figsize=(4.1 * len(families), 3.5), squeeze=False)
    for ax, family in zip(axes.ravel(), families):
        sub = rows[rows["problem_family"] == family]
        ax.scatter(
            sub["gradient_rank_fraction"],
            sub["cosine_sq"],
            s=9,
            alpha=0.22,
            color="#0072B2",
        )
        ax.plot([0, 1], [0, 1], color="#333333", lw=0.8, ls="--")
        ax.set_xlim(0, 1.03)
        ax.set_ylim(0, 1.03)
        ax.set_title(family, fontsize=9)
        ax.set_xlabel(r"$nr(G_i)/r_i$")
        ax.set_ylabel(r"$\cos^2(G_i,\Delta W_i)$")
        ax.grid(alpha=0.25)
    fig.suptitle("ExactMuon polar alignment identity", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_first_order_pair_comparison(first_order_pair: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "first_order_pair_comparison.png"
    key_metrics = [
        "update_grad_inner",
        "update_grad_cosine",
        "update_grad_per_update_norm",
        "delta_loss",
        "relative_update_fro_norm",
    ]
    labels = {
        "update_grad_inner": "first-order inner product",
        "update_grad_cosine": "gradient-update cosine",
        "update_grad_per_update_norm": "first-order per update norm",
        "delta_loss": "observed delta loss",
        "relative_update_fro_norm": "relative update norm",
    }
    families = ["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    rows = first_order_pair[
        first_order_pair["metric"].isin(key_metrics)
        & first_order_pair["problem_family"].isin(families)
    ].copy()
    rows["metric"] = pd.Categorical(rows["metric"], categories=key_metrics, ordered=True)
    rows = rows.sort_values(["metric", "problem_family"])
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(9.0, 1.75 * len(key_metrics)), sharex=True)
    if len(key_metrics) == 1:
        axes = [axes]
    y_positions = np.arange(len(families))
    for ax, metric in zip(axes, key_metrics):
        sub = rows[rows["metric"] == metric].set_index("problem_family").reindex(families).reset_index()
        x = sub["geomean_ratio_muon_over_adam"].to_numpy(dtype=float)
        lo = sub["ratio_ci95_low"].to_numpy(dtype=float)
        hi = sub["ratio_ci95_high"].to_numpy(dtype=float)
        xerr = np.vstack([x - lo, hi - x])
        ax.axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.errorbar(x, y_positions, xerr=xerr, fmt="o", color="#CC79A7", ecolor="#444444", capsize=3)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(families, fontsize=8)
        ax.set_xscale("log")
        ax.set_title(labels[metric], fontsize=9)
        ax.grid(alpha=0.22, axis="x")
    axes[-1].set_xlabel("geometric mean ratio: Muon / Adam (95% CI)")
    fig.suptitle("Matched Muon-vs-Adam first-order descent comparison", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_win_condition_summary(win_condition: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "win_condition_summary.png"
    if win_condition.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.text(0.5, 0.5, "No win-condition rows available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        return path

    key_conditions = ["muon_grad_rank_fraction", "delta_mean_grad_rank_fraction"]
    labels = {
        "muon_grad_rank_fraction": r"Muon mean $nr(G_i)/r_i$",
        "delta_mean_grad_rank_fraction": r"Muon - Adam mean $nr(G_i)/r_i$",
    }
    families = ["All", "MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    bin_order = ["low", "middle", "high"]
    fig, axes = plt.subplots(len(key_conditions), len(families), figsize=(3.35 * len(families), 3.15 * len(key_conditions)), squeeze=False)
    for row_idx, condition in enumerate(key_conditions):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            sub = win_condition[
                (win_condition["condition"] == condition)
                & (win_condition["problem_family"] == family)
                & (win_condition["bin"].isin(bin_order))
            ].copy()
            if sub.empty:
                ax.axis("off")
                continue
            sub["bin"] = pd.Categorical(sub["bin"], categories=bin_order, ordered=True)
            sub = sub.sort_values("bin")
            x = np.arange(len(sub))
            ax.axhline(0.5, color="#555555", lw=0.8, ls="--")
            ax.plot(x, sub["muon_first_order_win_rate"], marker="o", color="#0072B2", lw=1.7, label="first-order")
            ax.plot(x, sub["muon_delta_loss_win_rate"], marker="s", color="#D55E00", lw=1.3, ls="--", label="delta loss")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(x)
            ax.set_xticklabels([str(value) for value in sub["bin"]], fontsize=8)
            ax.set_title(f"{family}\n{labels[condition]}", fontsize=8.2)
            if col_idx == 0:
                ax.set_ylabel("Muon win rate")
            ax.set_xlabel("tertile bin", fontsize=8)
            ax.grid(alpha=0.22)
    handles = [
        Line2D([0], [0], color="#0072B2", marker="o", lw=1.7, label="first-order win"),
        Line2D([0], [0], color="#D55E00", marker="s", lw=1.3, ls="--", label="delta-loss win"),
        Line2D([0], [0], color="#555555", lw=0.8, ls="--", label="chance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("When does Muon beat Adam in matched one-step progress?", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_win_generalization_summary(win_generalization: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "win_prediction_generalization.png"
    if win_generalization.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.text(0.5, 0.5, "No win-prediction rows available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        return path

    targets = ["muon_first_order_win", "muon_delta_loss_win"]
    target_labels = {
        "muon_first_order_win": "first-order win",
        "muon_delta_loss_win": "delta-loss win",
    }
    families = ["MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    models = ["spectrum_only", "spectrum_plus_family"]
    model_labels = {
        "spectrum_only": "spectrum only",
        "spectrum_plus_family": "spectrum + family",
    }
    model_colors = {"spectrum_only": "#0072B2", "spectrum_plus_family": "#CC79A7"}
    rows = win_generalization[
        (win_generalization["evaluation"] == "leave_family_out")
        & (win_generalization["held_out_family"].isin(families))
        & (win_generalization["model"].isin(models))
    ].copy()

    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 3.9), squeeze=False)
    x = np.arange(len(families))
    offsets = {"spectrum_only": -0.14, "spectrum_plus_family": 0.14}
    for ax, target in zip(axes.ravel(), targets):
        sub_target = rows[rows["target"] == target]
        truth = (
            sub_target.groupby("held_out_family", observed=True)["test_positive_rate"]
            .first()
            .reindex(families)
            .to_numpy(dtype=float)
        )
        ax.scatter(x, truth, marker="x", color="#111111", s=64, label="true win rate", zorder=4)
        for model in models:
            sub = sub_target[sub_target["model"] == model].set_index("held_out_family").reindex(families)
            ax.scatter(
                x + offsets[model],
                sub["mean_predicted_probability"].to_numpy(dtype=float),
                color=model_colors[model],
                s=42,
                label=model_labels[model],
                alpha=0.9,
            )
            for idx, value in enumerate(sub["mean_predicted_probability"].to_numpy(dtype=float)):
                if np.isfinite(value) and np.isfinite(truth[idx]):
                    ax.plot([x[idx] + offsets[model], x[idx]], [value, truth[idx]], color=model_colors[model], lw=0.7, alpha=0.45)
        ax.set_title(f"Leave-family-out prediction: {target_labels[target]}")
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel("held-out win probability")
        ax.grid(alpha=0.22, axis="y")
    handles = [
        Line2D([0], [0], marker="x", color="#111111", lw=0, label="true win rate"),
        Line2D([0], [0], marker="o", color=model_colors["spectrum_only"], lw=0, label=model_labels["spectrum_only"]),
        Line2D(
            [0],
            [0],
            marker="o",
            color=model_colors["spectrum_plus_family"],
            lw=0,
            label=model_labels["spectrum_plus_family"],
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Do rank/spectral win rules generalize across problem families?", y=1.12, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_win_feature_overlap(win_overlap: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "win_feature_overlap.png"
    if win_overlap.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.text(0.5, 0.5, "No feature-overlap rows available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        return path

    feature_order = [
        "muon_grad_rank_fraction",
        "adam_grad_rank_fraction",
        "pair_mean_grad_rank_fraction",
        "delta_mean_grad_rank_fraction",
        "condition_score_Muon",
        "condition_score_Adam",
        "delta_condition_score",
        "nrG_Muon",
        "nrG_Adam",
        "delta_nrG",
        "stA_Muon",
        "stA_Adam",
        "delta_stA",
    ]
    feature_labels = [
        r"Muon $nr(G)/r$",
        r"Adam $nr(G)/r$",
        r"pair mean $nr(G)/r$",
        r"$\Delta nr(G)/r$",
        "Muon condition",
        "Adam condition",
        r"$\Delta$ condition",
        "Muon nrG",
        "Adam nrG",
        r"$\Delta$ nrG",
        "Muon stA",
        "Adam stA",
        r"$\Delta$ stA",
    ]
    families = ["MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]
    rows = win_overlap[win_overlap["row_type"] == "feature"].copy()
    matrix = np.full((len(feature_order), len(families)), np.nan)
    for row_idx, feature in enumerate(feature_order):
        for col_idx, family in enumerate(families):
            match = rows[(rows["feature"] == feature) & (rows["held_out_family"] == family)]
            if not match.empty:
                matrix[row_idx, col_idx] = float(match["frac_test_outside_train_range"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.0, 6.7))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#111111")
    summary = win_overlap[win_overlap["row_type"] == "family_summary"].set_index("held_out_family").reindex(families)
    subtitle = " | ".join(
        f"{family}: any-outside={summary.loc[family, 'frac_pairs_with_any_feature_outside']:.2f}, "
        f"NNdist={summary.loc[family, 'median_min_standardized_distance']:.2f}"
        for family in families
        if family in summary.index and np.isfinite(summary.loc[family, "frac_pairs_with_any_feature_outside"])
    )
    fig.colorbar(image, ax=ax, label="fraction outside train-family range")
    ax.set_title("Feature support overlap in leave-family-out prediction\n" + subtitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all_figures(steps: pd.DataFrame, layers: pd.DataFrame, summaries: dict[str, pd.DataFrame], figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss_curves": make_loss_curves(steps, figure_dir),
        "geometry_separation": make_geometry_separation(summaries["geometry"], figure_dir),
        "predicted_decrease_vs_observed": make_predicted_decrease(steps, figure_dir),
        "condition_score_trajectories": make_condition_trajectories(steps, figure_dir),
        "mean_3d_condition_loss": make_3d_mean(steps, figure_dir),
        "layerwise_3d_condition_loss": make_3d_layerwise(layers, figure_dir),
        "volatility_robustness": make_volatility_robustness(summaries["volatility"], figure_dir),
        "update_spectrum_robustness": make_update_spectrum_robustness(summaries["update_spectrum"], figure_dir),
        "update_transmission_heatmap": make_update_transmission_heatmap(summaries["update_transmission"], figure_dir),
        "first_order_calibration": make_first_order_calibration(steps, figure_dir),
        "polar_alignment_identity": make_polar_alignment_identity(layers, figure_dir),
        "first_order_pair_comparison": make_first_order_pair_comparison(summaries["first_order_pair"], figure_dir),
    }
    if "win_condition" in summaries:
        paths["win_condition_summary"] = make_win_condition_summary(summaries["win_condition"], figure_dir)
    if "win_generalization" in summaries:
        paths["win_prediction_generalization"] = make_win_generalization_summary(summaries["win_generalization"], figure_dir)
    if "win_overlap" in summaries:
        paths["win_feature_overlap"] = make_win_feature_overlap(summaries["win_overlap"], figure_dir)
    return paths
