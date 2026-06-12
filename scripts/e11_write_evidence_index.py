from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_evidence_index.md")

def link(path: str) -> str:
    return f"[{path}](../{path})"


def main() -> None:
    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    first_order = pd.read_csv("results/e11_equal_update/first_order_pair_summary.csv")
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    stateless = pd.read_csv("results/e11_stateless_direction_ablation/stateless_direction_summary.csv")
    stateless_traj = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    target_pair = pd.read_csv("results/e11_target_update_sweep/target_pair_summary.csv")
    natural_swap = pd.read_csv("results/e11_natural_update_swap_probe/natural_update_swap_summary.csv")
    switch_lr = pd.read_csv("results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_summary.csv")
    raw_volatility = pd.read_csv("results/e11/volatility_summary.csv")
    equal_volatility = pd.read_csv("results/e11_equal_update/volatility_summary.csv")
    mnist_spectrum = pd.read_csv("results/e11_mnist_mlp_probe/update_spectrum_summary.csv")
    mnist_pair = pd.read_csv("results/e11_mnist_mlp_probe/pair_summary.csv")
    deep_mnist_spectrum = pd.read_csv("results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")
    deep_mnist_pair = pd.read_csv("results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    patch_spectrum = pd.read_csv("results/e11_mnist_patch_probe/update_spectrum_summary.csv")
    patch_pair = pd.read_csv("results/e11_mnist_patch_probe/pair_summary.csv")
    conv_spectrum = pd.read_csv("results/e11_mnist_conv_probe/update_spectrum_summary.csv")
    conv_pair = pd.read_csv("results/e11_mnist_conv_probe/pair_summary.csv")

    nr_update = require_one(update_spectrum, problem_family="All", metric="nrUpdate")
    st_update = require_one(update_spectrum, problem_family="All", metric="stUpdate")
    first_all = require_one(first_order, problem_family="All", metric="update_grad_inner")
    calibration_all = require_one(calibration, group="All")
    spectral_fro = require_one(
        spectral,
        group_type="budget_all",
        comparison="flat_polar_over_gd_spectrum",
        metric="update_grad_inner",
        budget="fro",
    )
    spectral_op = require_one(
        spectral,
        group_type="budget_all",
        comparison="flat_polar_over_gd_spectrum",
        metric="update_grad_inner",
        budget="op",
    )
    stateless_nr = require_one(
        stateless,
        group_type="all",
        problem_family="All",
        checkpoint_source="All",
        comparison="PolarMuon/GD",
        metric="nrUpdate",
    )
    stateless_inner = require_one(
        stateless,
        group_type="all",
        problem_family="All",
        checkpoint_source="All",
        comparison="PolarMuon/GD",
        metric="update_grad_inner",
    )
    traj_nr = require_one(
        stateless_traj,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="mean_nrUpdate",
    )
    traj_decrease = require_one(
        stateless_traj,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="total_decrease",
    )
    target_ms = require_one(
        target_pair,
        group_type="setting_all_targets",
        metric="update_grad_inner",
        base_setting="Matrix sensing kappa=1e+02",
    )
    target_mlp64 = require_one(
        target_pair,
        group_type="setting_all_targets",
        metric="update_grad_inner",
        base_setting="Small MLP digits hidden=64",
    )
    natural_delta = require_one(
        natural_swap,
        group_type="all",
        metric="delta_loss",
        target_layer_relative_norm="All",
        budget="All",
        eval_algo="All",
    )
    switch_lr_all = require_one(switch_lr, group_type="all", metric="loss_after", source_algo="All")
    raw_rank_speed = require_one(raw_volatility, metric="norm_rank_plane_mean_speed", problem_family="All")
    equal_rank_speed = require_one(equal_volatility, metric="norm_rank_plane_mean_speed", problem_family="All")
    equal_ms_rank_speed = require_one(
        equal_volatility,
        metric="norm_rank_plane_mean_speed",
        problem_family="MatrixSensing",
    )
    mnist_nr = require_one(mnist_spectrum, problem_family="MNISTMLP", metric="nrUpdate")
    mnist_h128 = require_one(mnist_pair, group_type="hidden_all_targets", hidden_dim="128", metric="update_grad_inner")
    deep_nr = require_one(deep_mnist_spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate")
    deep_first = require_one(deep_mnist_pair, group_type="all", hidden_dim="All", num_factors="All", metric="update_grad_inner")
    patch_nr = require_one(patch_spectrum, problem_family="MNISTPatchClassifier", metric="nrUpdate")
    patch_first = require_one(
        patch_pair,
        group_type="all",
        filters="All",
        kernel_size="All",
        metric="update_grad_inner",
    )
    conv_nr = require_one(conv_spectrum, problem_family="MNISTConvNet", metric="nrUpdate")
    conv_first = require_one(
        conv_pair,
        group_type="all",
        filters="All",
        kernel_size="All",
        metric="update_grad_inner",
    )

    evidence = pd.DataFrame(
        [
            {
                "claim": "Muon changes update-spectrum geometry.",
                "recommended_figure": link("figures/e11_equal_update/update_spectrum_robustness.png"),
                "source_data": link("results/e11_equal_update/update_spectrum_summary.csv"),
                "quantitative_anchor": (
                    f"nrUpdate ratio={fmt(nr_update['geomean_ratio_muon_over_adam'])}; "
                    f"stUpdate ratio={fmt(st_update['geomean_ratio_muon_over_adam'])}"
                ),
                "how_to_read": "Ratios above 1 mean Muon updates have larger rank/effective-rank diagnostics than Adam under matched update size.",
                "caveat": "ExactMuon's polar update makes this partly construction-level evidence.",
            },
            {
                "claim": "Update-spectrum shaping is the current cross-task optimizer signature.",
                "recommended_figure": link("discussion/e11_cross_task_signature.md"),
                "source_data": link("results/e11_cross_task_signature/cross_task_signature_summary.csv"),
                "quantitative_anchor": "Only update-spectrum metrics pass the conservative three-family screen.",
                "how_to_read": "A candidate passes only when all task families have the same direction and all family-level CIs support it.",
                "caveat": "This is a screen over the current three task families, not a theorem.",
            },
            {
                "claim": "Muon is not globally better on one-step progress.",
                "recommended_figure": link("figures/e11_equal_update/first_order_pair_comparison.png"),
                "source_data": link("results/e11_equal_update/first_order_pair_summary.csv"),
                "quantitative_anchor": f"All-task first-order ratio={fmt(first_all['geomean_ratio_muon_over_adam'])}",
                "how_to_read": "Ratios above 1 favor Muon; ratios below 1 favor Adam. The aggregate is below 1 while some families favor Muon.",
                "caveat": "Use family-level rows rather than only the all-task aggregate.",
            },
            {
                "claim": "One-step loss decrease is locally explained by gradient-update alignment.",
                "recommended_figure": link("figures/e11_equal_update/first_order_calibration.png"),
                "source_data": link("results/e11_equal_update/first_order_calibration_summary.csv"),
                "quantitative_anchor": f"Spearman={fmt(calibration_all['spearman_delta_vs_first_order'])}",
                "how_to_read": "High Spearman correlation means observed one-step decrease tracks the positive descent proxy <G,D>.",
                "caveat": "This validates local one-step diagnostics, not long-horizon performance.",
            },
            {
                "claim": "Polar direction alone creates high-rank updates but not universal Frobenius-budget progress.",
                "recommended_figure": link("figures/e11_stateless_direction_ablation/stateless_direction_ratios.png"),
                "source_data": link("results/e11_stateless_direction_ablation/stateless_direction_summary.csv"),
                "quantitative_anchor": (
                    f"PolarMuon/GD nrUpdate ratio={fmt(stateless_nr['geomean_ratio'])}; "
                    f"update_grad_inner ratio={fmt(stateless_inner['geomean_ratio'])}"
                ),
                "how_to_read": "The stateless candidates use the same checkpoint state, gradient, and global Frobenius update size.",
                "caveat": "This is a one-step direction intervention, not a full optimizer-state training ablation.",
            },
            {
                "claim": "The stateless polar trajectory keeps high-rank updates without guaranteeing larger total decrease.",
                "recommended_figure": link("figures/e11_stateless_optimizer_trajectory/stateless_optimizer_trajectory_ratios.png"),
                "source_data": link("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv"),
                "quantitative_anchor": (
                    f"PolarMuon/GD mean_nrUpdate ratio={fmt(traj_nr['geomean_ratio'])}; "
                    f"total_decrease ratio={fmt(traj_decrease['geomean_ratio'])}"
                ),
                "how_to_read": "All stateless candidates use the same per-family target relative update norm at each step.",
                "caveat": "This is a short controlled trajectory, not a retuned optimizer benchmark.",
            },
            {
                "claim": "Flat/polar spectral allocation is norm-geometry dependent.",
                "recommended_figure": link("figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png"),
                "source_data": link("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv"),
                "quantitative_anchor": (
                    f"flat/GD ratio={fmt(spectral_fro['geomean_ratio'])} under Frobenius budget; "
                    f"{fmt(spectral_op['geomean_ratio'])} under operator budget"
                ),
                "how_to_read": "The same flat/polar allocation loses under Frobenius budget and wins under operator-norm budget.",
                "caveat": "This probe fixes singular vectors and only varies singular-value allocation.",
            },
            {
                "claim": "The norm-geometry boundary has a local first-order explanation.",
                "recommended_figure": link("discussion/e11_mechanism_theorem_bridge.md"),
                "source_data": link("discussion/e11_theory_note.md"),
                "quantitative_anchor": (
                    f"Frobenius flat/GD ratio={fmt(spectral_fro['geomean_ratio'])}; "
                    f"operator-norm flat/GD ratio={fmt(spectral_op['geomean_ratio'])}"
                ),
                "how_to_read": "The bridge maps each theorem component to the evidence it supports and the claims it does not support.",
                "caveat": "It is a local first-order proposition plus mechanism evidence, not a convergence theorem.",
            },
            {
                "claim": "The current mechanism boundary is task/layer/norm dependent.",
                "recommended_figure": link("discussion/e11_mechanism_boundary.md"),
                "source_data": link("results/e11_mechanism_boundary/mechanism_boundary_map.csv"),
                "quantitative_anchor": "Boundary map combines equal-update, target-update, per-layer, spectral-allocation, and swap controls.",
                "how_to_read": "Each row states what was controlled, which ratio was tested, and whether the condition is Muon/flat/polar favorable.",
                "caveat": "This is a structured evidence map, not a fitted predictive law.",
            },
            {
                "claim": "The ablation ladder separates distinct optimizer mechanisms.",
                "recommended_figure": link("discussion/e11_optimizer_ablation_map.md"),
                "source_data": link("discussion/e11_optimizer_ablation_map.md"),
                "quantitative_anchor": (
                    "Controls separate update size, layer allocation, singular-value allocation, "
                    "singular-vector geometry, natural update vectors, and continuation effects."
                ),
                "how_to_read": "Use the map to avoid attributing every Adam/Muon difference to polar spectrum shaping alone.",
                "caveat": "It is an evidence organization layer; momentum-free and state-standardized optimizer variants are still missing.",
            },
            {
                "claim": "MNIST MLP preserves update-spectrum shaping but not universal first-order advantage.",
                "recommended_figure": link("figures/e11_mnist_mlp_probe/mnist_mlp_first_order_ratios.png"),
                "source_data": link("results/e11_mnist_mlp_probe/pair_summary.csv"),
                "quantitative_anchor": (
                    f"MNIST nrUpdate ratio={fmt(mnist_nr['geomean_ratio_muon_over_adam'])}; "
                    f"hidden=128 first-order ratio={fmt(mnist_h128['geomean_ratio_muon_over_adam'])}"
                ),
                "how_to_read": "Muon still strongly changes update spectra, but the wider MNIST MLP setting is Adam-favorable for one-step progress.",
                "caveat": "This is a shallow short-horizon sanity probe, not a final neural benchmark.",
            },
            {
                "claim": "Deep MNIST MLP preserves update-spectrum shaping while becoming more Adam-favorable for first-order progress.",
                "recommended_figure": link("figures/e11_deep_mnist_mlp_probe/deep_mnist_mlp_first_order_ratios.png"),
                "source_data": link("results/e11_deep_mnist_mlp_probe/pair_summary.csv"),
                "quantitative_anchor": (
                    f"Deep MNIST nrUpdate ratio={fmt(deep_nr['geomean_ratio_muon_over_adam'])}; "
                    f"first-order ratio={fmt(deep_first['geomean_ratio_muon_over_adam'])}"
                ),
                "how_to_read": "The probe uses 3- and 4-factor all-matrix MNIST MLPs under matched global update size.",
                "caveat": "It is still short-horizon and MLP-only, but it is a stronger neural negative control.",
            },
            {
                "claim": "MNIST patch/shared-weight surrogate preserves update-spectrum shaping while favoring Adam locally.",
                "recommended_figure": link("figures/e11_mnist_patch_probe/mnist_patch_first_order_ratios.png"),
                "source_data": link("results/e11_mnist_patch_probe/pair_summary.csv"),
                "quantitative_anchor": (
                    f"Patch nrUpdate ratio={fmt(patch_nr['geomean_ratio_muon_over_adam'])}; "
                    f"first-order ratio={fmt(patch_first['geomean_ratio_muon_over_adam'])}"
                ),
                "how_to_read": "The probe uses local patches and shared patch weights while keeping all parameters matrix-shaped for ExactMuon.",
                "caveat": "It is not a true 4D CNN kernel or modern architecture benchmark.",
            },
            {
                "claim": "MNIST ConvNet preserves update-spectrum shaping while favoring Adam locally.",
                "recommended_figure": link("figures/e11_mnist_conv_probe/mnist_conv_first_order_ratios.png"),
                "source_data": link("results/e11_mnist_conv_probe/pair_summary.csv"),
                "quantitative_anchor": (
                    f"ConvNet nrUpdate ratio={fmt(conv_nr['geomean_ratio_muon_over_adam'])}; "
                    f"first-order ratio={fmt(conv_first['geomean_ratio_muon_over_adam'])}"
                ),
                "how_to_read": "The probe uses a true Conv2d kernel; ExactMuon and spectral diagnostics use the flattened conv-kernel matrix view.",
                "caveat": "It is still a small short-horizon benchmark, not a modern architecture result.",
            },
            {
                "claim": "Direction advantage after size control is task/layer conditional.",
                "recommended_figure": link("figures/e11_target_update_sweep/target_update_first_order_ratios.png"),
                "source_data": link("results/e11_target_update_sweep/target_pair_summary.csv"),
                "quantitative_anchor": (
                    f"MatrixSensing kappa=1e2 ratio={fmt(target_ms['geomean_ratio_muon_over_adam'])}; "
                    f"MLP hidden=64 ratio={fmt(target_mlp64['geomean_ratio_muon_over_adam'])}"
                ),
                "how_to_read": "At matched global update size, Muon can win in one setting and lose in another.",
                "caveat": "Global update control does not fully control per-layer allocation except in the MLP follow-up.",
            },
            {
                "claim": "Natural update vectors matter locally.",
                "recommended_figure": link("figures/e11_natural_update_swap_probe/natural_update_swap_target_sweep.png"),
                "source_data": link("results/e11_natural_update_swap_probe/natural_update_swap_summary.csv"),
                "quantitative_anchor": f"observed delta-loss other/own ratio={fmt(natural_delta['mean_signed_other_over_own'])}",
                "how_to_read": "Ratios below 1 mean the other trajectory's natural update gives less local progress than the state-specific update.",
                "caveat": "This is a one-step intervention and has setting-level exceptions.",
            },
            {
                "claim": "Trajectory-level optimizer superiority does not follow from local geometry.",
                "recommended_figure": link("figures/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_sweep.png"),
                "source_data": link("results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_summary.csv"),
                "quantitative_anchor": f"best switched/own final-loss ratio={fmt(switch_lr_all['mean_switched_over_own'])}",
                "how_to_read": "The switch result depends on task family and tuning, so it is a negative control against overclaiming.",
                "caveat": "The LR sweep is small and short-horizon.",
            },
            {
                "claim": "Muon is not generally more stable in state geometry.",
                "recommended_figure": link("figures/e11_equal_update/volatility_robustness.png"),
                "source_data": link("results/e11_equal_update/volatility_summary.csv"),
                "quantitative_anchor": (
                    f"raw all-task rank speed ratio={fmt(raw_rank_speed['geomean_ratio_muon_over_adam'])}; "
                    f"equal-update all-task={fmt(equal_rank_speed['geomean_ratio_muon_over_adam'])}; "
                    f"equal-update MatrixSensing={fmt(equal_ms_rank_speed['geomean_ratio_muon_over_adam'])}"
                ),
                "how_to_read": "Ratios below 1 mean Muon moves more slowly in the state-rank plane; equal-update control removes update-scale confounding.",
                "caveat": "The robust optimizer-intrinsic result is update-spectrum shaping, not general trajectory stability.",
            },
        ]
    )

    text = f"""# E11 Evidence Index

This generated index maps each paper-facing claim to the most relevant figure and CSV source. Use it as the evidence checklist when preparing slides or a final report.

{markdown_table(evidence, ["claim", "recommended_figure", "source_data", "quantitative_anchor", "how_to_read", "caveat"])}
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved evidence index to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
