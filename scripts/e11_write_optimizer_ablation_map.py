from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_optimizer_ablation_map.md")


def main() -> None:
    raw_first_order = pd.read_csv("results/e11/first_order_pair_summary.csv")
    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    target_pair = pd.read_csv("results/e11_target_update_sweep/target_pair_summary.csv")
    layer_pair = pd.read_csv("results/e11_mlp_per_layer_control/pair_summary.csv")
    stateless = pd.read_csv("results/e11_stateless_direction_ablation/stateless_direction_summary.csv")
    stateless_traj = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    swap = pd.read_csv("results/e11_singular_vector_swap_probe/swap_ratio_summary.csv")
    natural = pd.read_csv("results/e11_natural_update_swap_probe/natural_update_swap_summary.csv")
    switch_lr = pd.read_csv("results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_summary.csv")

    nr_update = require_one(update_spectrum, problem_family="All", metric="nrUpdate")
    st_update = require_one(update_spectrum, problem_family="All", metric="stUpdate")
    raw_first_all = require_one(raw_first_order, problem_family="All", metric="update_grad_inner")
    raw_first_ms = require_one(raw_first_order, problem_family="MatrixSensing", metric="update_grad_inner")
    raw_first_mf = require_one(raw_first_order, problem_family="MatrixFactorizationInput", metric="update_grad_inner")
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
    layer_h16 = require_one(layer_pair, group_type="hidden_all_targets", metric="update_grad_inner", hidden_dim="16")
    layer_h64 = require_one(layer_pair, group_type="hidden_all_targets", metric="update_grad_inner", hidden_dim="64")
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
    swap_all = require_one(
        swap[(swap["group_type"] == "all") & (swap["metric"] == "update_grad_inner")],
        eval_algo="All",
    )
    natural_all = require_one(
        natural[(natural["group_type"] == "all") & (natural["metric"] == "update_grad_inner")],
        target_layer_relative_norm="All",
        budget="All",
        eval_algo="All",
    )
    switch_all = require_one(switch_lr, group_type="all", metric="loss_after", source_algo="All")

    ablation_map = pd.DataFrame(
        [
            {
                "control_level": "Natural Adam vs Muon",
                "isolates": "Nothing; optimizer identity, update scale, direction, state, and layer allocation are all entangled.",
                "main_evidence": (
                    f"Raw all-task first-order ratio={fmt(raw_first_all['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(raw_first_all)}; family signs split: MS={fmt(raw_first_ms['geomean_ratio_muon_over_adam'])}, "
                    f"MF={fmt(raw_first_mf['geomean_ratio_muon_over_adam'])}."
                ),
                "paper_role": "Shows why the paper should avoid a global Muon-is-better claim.",
            },
            {
                "control_level": "Matched global update size",
                "isolates": "Direction plus layer allocation after removing global relative update-norm scale.",
                "main_evidence": (
                    f"`nrUpdate` ratio={fmt(nr_update['geomean_ratio_muon_over_adam'])} {ratio_ci(nr_update)}; "
                    f"`stUpdate` ratio={fmt(st_update['geomean_ratio_muon_over_adam'])} {ratio_ci(st_update)}."
                ),
                "paper_role": "Core optimizer signature: Muon robustly shapes update spectra at matched step size.",
            },
            {
                "control_level": "Target global update norm",
                "isolates": "Direction effects across a sweep of shared update sizes.",
                "main_evidence": (
                    f"MatrixSensing kappa=1e2 first-order ratio={fmt(target_ms['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(target_ms)}; MLP hidden=64 ratio={fmt(target_mlp64['geomean_ratio_muon_over_adam'])} "
                    f"{ratio_ci(target_mlp64)}."
                ),
                "paper_role": "Shows direction advantage is task/layer conditional, not an LR artifact.",
            },
            {
                "control_level": "Matched per-layer update size",
                "isolates": "Layerwise direction after removing layer allocation for SmallMLP.",
                "main_evidence": (
                    f"Hidden=16 ratio={fmt(layer_h16['geomean_ratio_muon_over_adam'])} {ratio_ci(layer_h16)}; "
                    f"hidden=64 ratio={fmt(layer_h64['geomean_ratio_muon_over_adam'])} {ratio_ci(layer_h64)}."
                ),
                "paper_role": "Separates global direction effects from per-layer update-budget allocation.",
            },
            {
                "control_level": "Stateless direction choice",
                "isolates": "Candidate update direction at the same checkpoint state, gradient, and global Frobenius update size.",
                "main_evidence": (
                    f"PolarMuon/GD `nrUpdate` ratio={fmt(stateless_nr['geomean_ratio'])} {ratio_ci(stateless_nr)}, "
                    f"but first-order ratio={fmt(stateless_inner['geomean_ratio'])} {ratio_ci(stateless_inner)}."
                ),
                "paper_role": "Shows polar direction alone creates the high-rank update spectrum, while Frobenius-matched progress remains conditional.",
            },
            {
                "control_level": "Stateless optimizer trajectories",
                "isolates": "Short multi-step training with the same target update size and no optimizer state.",
                "main_evidence": (
                    f"PolarMuon/GD mean_nrUpdate={fmt(traj_nr['geomean_ratio'])} {ratio_ci(traj_nr)}, "
                    f"but total_decrease={fmt(traj_decrease['geomean_ratio'])} {ratio_ci(traj_decrease)}."
                ),
                "paper_role": "Extends the stateless direction result beyond a single step while preserving the same boundary conclusion.",
            },
            {
                "control_level": "Synthetic singular-value allocation",
                "isolates": "Singular values with gradient singular vectors fixed.",
                "main_evidence": (
                    f"flat/GD ratio={fmt(spectral_fro['geomean_ratio'])} {ratio_ci(spectral_fro)} under Frobenius budget; "
                    f"{fmt(spectral_op['geomean_ratio'])} {ratio_ci(spectral_op)} under operator-norm budget."
                ),
                "paper_role": "Mechanism probe linking Muon's polar spectrum to the local theory note.",
            },
            {
                "control_level": "Synthetic singular-vector swap",
                "isolates": "Singular vectors at matched spectrum/norm in one-step probes.",
                "main_evidence": (
                    f"Overall other/own first-order ratio={fmt(swap_all['mean_signed_other_over_own'])} "
                    f"[{fmt(swap_all['signed_ratio_ci95_low'])}, {fmt(swap_all['signed_ratio_ci95_high'])}]."
                ),
                "paper_role": "Shows optimizer trajectories can move into different vector geometries, not just different spectra.",
            },
            {
                "control_level": "Natural update-vector swap",
                "isolates": "Each optimizer's actual proposed update vector, including state and spectrum.",
                "main_evidence": (
                    f"Overall other/own first-order ratio={fmt(natural_all['mean_signed_other_over_own'])} "
                    f"[{fmt(natural_all['signed_ratio_ci95_low'])}, {fmt(natural_all['signed_ratio_ci95_high'])}]."
                ),
                "paper_role": "Checks whether actual optimizer-specific updates are locally consequential.",
            },
            {
                "control_level": "Fresh continuation and LR sweep",
                "isolates": "Whether local one-step specialization predicts longer-horizon optimizer continuation.",
                "main_evidence": (
                    f"Best switched/own final-loss ratio={fmt(switch_all['mean_switched_over_own'])} "
                    f"[{fmt(switch_all['ratio_ci95_low'])}, {fmt(switch_all['ratio_ci95_high'])}]."
                ),
                "paper_role": "Negative control: local geometry does not imply global optimizer superiority.",
            },
        ]
    )

    missing_variants = pd.DataFrame(
        [
            {
                "missing_variant": "Momentum-free Muon and momentum-free Adam-like training baselines",
                "why_it_matters": "Would separate polar spectrum shaping from optimizer state accumulation over a full trajectory.",
            },
            {
                "missing_variant": "GD-spectrum matched natural optimizer",
                "why_it_matters": "Would compare natural training with gradient-spectrum updates rather than only one-step synthetic probes.",
            },
            {
                "missing_variant": "Flat/polar update with Adam-style state removed or standardized",
                "why_it_matters": "The stateless ablation addresses one-step direction choice; this would test the same question over natural training.",
            },
            {
                "missing_variant": "Broader neural architecture control",
                "why_it_matters": "Would test whether the ablation map survives beyond shallow MLPs.",
            },
        ]
    )

    text = f"""# E11 Optimizer Ablation Map

This generated map organizes the current E11 controls by what each ablation actually isolates. Its purpose is to prevent the paper from treating every Adam/Muon difference as evidence for the same mechanism.

## Current Ablation Coverage

{markdown_table(ablation_map, ["control_level", "isolates", "main_evidence", "paper_role"])}

## Interpretation

The current ablations support a layered claim:

1. Muon robustly changes update spectra at matched update size.
2. Removing global update-size differences does not create a universal Muon advantage.
3. Stateless direction and short-trajectory controls show that polar direction alone creates the high-rank update spectrum under matched Frobenius update size.
4. Per-layer controls and synthetic spectral-allocation probes show that layer allocation, norm budget, and singular-value allocation are distinct mechanisms.
5. Singular-vector and natural-update swaps show that optimizer-specific trajectories matter locally.
6. Continuation probes show that local one-step geometry does not automatically predict longer-horizon optimizer superiority.

## Still Missing For A Stronger Variant Claim

{markdown_table(missing_variants, ["missing_variant", "why_it_matters"])}

## Safe Paper Wording

Use:

> The current ablation ladder separates update scale, layer allocation, stateless direction and trajectory choice, singular-value allocation, singular-vector geometry, and continuation effects. The robust optimizer-intrinsic signal is update-spectrum shaping; its optimization benefit is conditional.

Avoid:

> The ablations prove that Muon's performance differences are caused only by polar spectrum shaping.
"""

    write_markdown(OUTPUT_PATH, text)
    print(f"saved optimizer ablation map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
