from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from e11_condition_geometry.reporting import require_one


@dataclass(frozen=True)
class BoundaryCounts:
    muon_flat_favorable: int
    own_update_positive_control: int
    favorable: int
    unfavorable: int
    mixed: int


@dataclass(frozen=True)
class PaperStats:
    nr_update: pd.Series
    st_update: pd.Series
    calibration_all: pd.Series
    spectral_fro: pd.Series
    spectral_op: pd.Series
    deep_nr: pd.Series
    deep_st: pd.Series
    deep_first: pd.Series
    patch_nr: pd.Series
    patch_st: pd.Series
    patch_first: pd.Series
    conv_nr: pd.Series
    conv_st: pd.Series
    conv_first: pd.Series
    stateless_traj_nr: pd.Series
    stateless_traj_decrease: pd.Series
    predictor_best: pd.Series
    predictor_uncertainty_best: pd.Series
    boundary_counts: BoundaryCounts


def _read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(root / relative_path)


def load_paper_stats(root: Path | str = ".") -> PaperStats:
    """Load the paper-facing quantitative anchors from generated E11 results."""

    root = Path(root)
    update_spectrum = _read_csv(root, "results/e11_equal_update/update_spectrum_summary.csv")
    calibration = _read_csv(root, "results/e11_equal_update/first_order_calibration_summary.csv")
    spectral = _read_csv(root, "results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    boundary = _read_csv(root, "results/e11_mechanism_boundary/mechanism_boundary_map.csv")
    predictor = _read_csv(root, "results/e11_boundary_predictor/boundary_predictor_summary.csv")
    predictor_uncertainty = _read_csv(root, "results/e11_boundary_predictor/boundary_predictor_uncertainty.csv")
    deep_pair = _read_csv(root, "results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    deep_spectrum = _read_csv(root, "results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")
    patch_pair = _read_csv(root, "results/e11_mnist_patch_probe/pair_summary.csv")
    patch_spectrum = _read_csv(root, "results/e11_mnist_patch_probe/update_spectrum_summary.csv")
    conv_pair = _read_csv(root, "results/e11_mnist_conv_probe/pair_summary.csv")
    conv_spectrum = _read_csv(root, "results/e11_mnist_conv_probe/update_spectrum_summary.csv")
    stateless_traj = _read_csv(
        root,
        "results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv",
    )

    predictor_focus = predictor[
        (predictor["target"] == "update_grad_inner_muon_higher") & (predictor["evaluation"] == "leave_setting_out")
    ]
    predictor_best = (
        predictor_focus.groupby("feature_set", as_index=False)
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"))
        .sort_values("mean_balanced_accuracy", ascending=False)
        .iloc[0]
    )
    predictor_uncertainty_best = (
        predictor_uncertainty[predictor_uncertainty["target"] == "update_grad_inner_muon_higher"]
        .sort_values("mean_balanced_accuracy_chance_filled", ascending=False)
        .iloc[0]
    )

    boundary_counts = BoundaryCounts(
        muon_flat_favorable=int(boundary["direction"].isin(["Muon-favorable", "flat/polar-favorable"]).sum()),
        own_update_positive_control=int(boundary["direction"].eq("own-update-favorable").sum()),
        favorable=int(
            boundary["direction"].isin(["Muon-favorable", "flat/polar-favorable", "own-update-favorable"]).sum()
        ),
        unfavorable=int(boundary["direction"].isin(["Adam/GD-favorable", "GD-spectrum-favorable"]).sum()),
        mixed=int(boundary["direction"].eq("mixed_or_uncertain").sum()),
    )

    return PaperStats(
        nr_update=require_one(update_spectrum, problem_family="All", metric="nrUpdate"),
        st_update=require_one(update_spectrum, problem_family="All", metric="stUpdate"),
        calibration_all=require_one(calibration, group="All"),
        spectral_fro=require_one(
            spectral,
            group_type="budget_all",
            comparison="flat_polar_over_gd_spectrum",
            metric="update_grad_inner",
            budget="fro",
        ),
        spectral_op=require_one(
            spectral,
            group_type="budget_all",
            comparison="flat_polar_over_gd_spectrum",
            metric="update_grad_inner",
            budget="op",
        ),
        deep_nr=require_one(deep_spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate"),
        deep_st=require_one(deep_spectrum, problem_family="DeepMNISTMLP", metric="stUpdate"),
        deep_first=require_one(deep_pair, group_type="all", hidden_dim="All", num_factors="All", metric="update_grad_inner"),
        patch_nr=require_one(patch_spectrum, problem_family="MNISTPatchClassifier", metric="nrUpdate"),
        patch_st=require_one(patch_spectrum, problem_family="MNISTPatchClassifier", metric="stUpdate"),
        patch_first=require_one(
            patch_pair,
            group_type="all",
            filters="All",
            kernel_size="All",
            metric="update_grad_inner",
        ),
        conv_nr=require_one(conv_spectrum, problem_family="MNISTConvNet", metric="nrUpdate"),
        conv_st=require_one(conv_spectrum, problem_family="MNISTConvNet", metric="stUpdate"),
        conv_first=require_one(
            conv_pair,
            group_type="all",
            filters="All",
            kernel_size="All",
            metric="update_grad_inner",
        ),
        stateless_traj_nr=require_one(
            stateless_traj,
            group_type="all",
            problem_family="All",
            comparison="PolarMuon/GD",
            metric="mean_nrUpdate",
        ),
        stateless_traj_decrease=require_one(
            stateless_traj,
            group_type="all",
            problem_family="All",
            comparison="PolarMuon/GD",
            metric="total_decrease",
        ),
        predictor_best=predictor_best,
        predictor_uncertainty_best=predictor_uncertainty_best,
        boundary_counts=boundary_counts,
    )
