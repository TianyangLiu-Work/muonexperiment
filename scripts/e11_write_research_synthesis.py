from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_research_synthesis.md")

def main() -> None:
    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    first_order = pd.read_csv("results/e11_equal_update/first_order_pair_summary.csv")
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    stateless = pd.read_csv("results/e11_stateless_direction_ablation/stateless_direction_summary.csv")
    stateless_traj = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")
    natural_swap = pd.read_csv("results/e11_natural_update_swap_probe/natural_update_swap_summary.csv")
    target_pair = pd.read_csv("results/e11_target_update_sweep/target_pair_summary.csv")
    switch_lr = pd.read_csv("results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_summary.csv")
    raw_volatility = pd.read_csv("results/e11/volatility_summary.csv")
    equal_volatility = pd.read_csv("results/e11_equal_update/volatility_summary.csv")
    deep_mnist_pair = pd.read_csv("results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    deep_mnist_spectrum = pd.read_csv("results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")
    patch_pair = pd.read_csv("results/e11_mnist_patch_probe/pair_summary.csv")
    patch_spectrum = pd.read_csv("results/e11_mnist_patch_probe/update_spectrum_summary.csv")
    conv_pair = pd.read_csv("results/e11_mnist_conv_probe/pair_summary.csv")
    conv_spectrum = pd.read_csv("results/e11_mnist_conv_probe/update_spectrum_summary.csv")

    nr_update = require_one(update_spectrum, problem_family="All", metric="nrUpdate")
    st_update = require_one(update_spectrum, problem_family="All", metric="stUpdate")
    first_all = require_one(first_order, problem_family="All", metric="update_grad_inner")
    first_ms = require_one(first_order, problem_family="MatrixSensing", metric="update_grad_inner")
    first_mf = require_one(first_order, problem_family="MatrixFactorizationInput", metric="update_grad_inner")
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
    stateless_st = require_one(
        stateless,
        group_type="all",
        problem_family="All",
        checkpoint_source="All",
        comparison="PolarMuon/GD",
        metric="stUpdate",
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
    traj_loss = require_one(
        stateless_traj,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="final_loss",
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
    natural_first = require_one(
        natural_swap,
        group_type="all",
        metric="update_grad_inner",
        target_layer_relative_norm="All",
        budget="All",
        eval_algo="All",
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
    switch_lr_adam = require_one(switch_lr, group_type="source_all", metric="loss_after", source_algo="Adam")
    switch_lr_muon = require_one(switch_lr, group_type="source_all", metric="loss_after", source_algo="Muon")
    raw_rank_speed = require_one(raw_volatility, metric="norm_rank_plane_mean_speed", problem_family="All")
    equal_rank_speed = require_one(equal_volatility, metric="norm_rank_plane_mean_speed", problem_family="All")
    equal_ms_rank_speed = require_one(
        equal_volatility,
        metric="norm_rank_plane_mean_speed",
        problem_family="MatrixSensing",
    )
    deep_nr = require_one(deep_mnist_spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate")
    deep_st = require_one(deep_mnist_spectrum, problem_family="DeepMNISTMLP", metric="stUpdate")
    deep_first = require_one(deep_mnist_pair, group_type="all", hidden_dim="All", num_factors="All", metric="update_grad_inner")
    patch_nr = require_one(patch_spectrum, problem_family="MNISTPatchClassifier", metric="nrUpdate")
    patch_st = require_one(patch_spectrum, problem_family="MNISTPatchClassifier", metric="stUpdate")
    patch_first = require_one(
        patch_pair,
        group_type="all",
        filters="All",
        kernel_size="All",
        metric="update_grad_inner",
    )
    conv_nr = require_one(conv_spectrum, problem_family="MNISTConvNet", metric="nrUpdate")
    conv_st = require_one(conv_spectrum, problem_family="MNISTConvNet", metric="stUpdate")
    conv_first = require_one(
        conv_pair,
        group_type="all",
        filters="All",
        kernel_size="All",
        metric="update_grad_inner",
    )

    claim_cards = pd.DataFrame(
        [
            {
                "claim": "Muon is a strong update-spectrum shaper.",
                "status": "supported",
                "evidence": (
                    f"Equal-update nrUpdate ratio={fmt(nr_update['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(nr_update)}, stUpdate ratio={fmt(st_update['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(st_update)}."
                ),
                "interpretation": "ExactMuon reliably makes flatter and higher-rank update matrices than Adam in this setup.",
                "caveat": "This is partly algorithmic by construction, not a performance claim.",
            },
            {
                "claim": "Muon is not globally better than Adam.",
                "status": "not supported as a global claim",
                "evidence": (
                    f"All-task first-order ratio={fmt(first_all['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(first_all)}, while Matrix Sensing ratio={fmt(first_ms['geomean_ratio_muon_over_adam'])} "
                    f"and MF ratio={fmt(first_mf['geomean_ratio_muon_over_adam'])}."
                ),
                "interpretation": "The same update geometry can help one task family and hurt another.",
                "caveat": "Aggregate ratios hide task-family sign flips.",
            },
            {
                "claim": "One-step progress is well captured by gradient-update alignment.",
                "status": "supported for short steps",
                "evidence": (
                    f"Spearman(delta_loss, <G,D>)={fmt(calibration_all['spearman_delta_vs_first_order'])} "
                    f"CI=[{fmt(calibration_all['spearman_ci95_low'])}, {fmt(calibration_all['spearman_ci95_high'])}], "
                    f"within-factor-2={fmt(calibration_all['within_factor_2'])}."
                ),
                "interpretation": "Local first-order diagnostics are meaningful for the current small steps.",
                "caveat": "This does not imply long-horizon performance or generalization.",
            },
            {
                "claim": "Flat/polar allocation is norm-geometry dependent.",
                "status": "supported by one-step probe",
                "evidence": (
                    f"flat_polar/GD-spectrum ratio={fmt(spectral_fro['geomean_ratio'])} "
                    f"CI={ratio_ci(spectral_fro)} under Frobenius budget, but {fmt(spectral_op['geomean_ratio'])} "
                    f"CI={ratio_ci(spectral_op)} under operator-norm budget."
                ),
                "interpretation": "Polar spreading is useful when the constraint rewards operator-norm-like spreading, not universally.",
                "caveat": "The probe fixes gradient singular vectors and is artificial.",
            },
            {
                "claim": "The norm-geometry boundary has a local first-order explanation.",
                "status": "supported as a narrow proposition",
                "evidence": (
                    "The theory note derives gradient-spectrum optimality under a Frobenius budget and "
                    "flat/polar optimality under an operator-norm budget."
                ),
                "interpretation": "This gives the paper a precise mathematical version of the intuitive spectral-spreading claim.",
                "caveat": "It is not a convergence theorem and does not predict the boundary across unseen tasks by itself.",
            },
            {
                "claim": "The current ablation ladder separates several mechanisms but not all optimizer details.",
                "status": "supported as evidence organization",
                "evidence": (
                    "The optimizer ablation map separates global update size, per-layer allocation, "
                    "singular-value allocation, stateless direction choice, singular-vector swaps, natural update-vector swaps, and continuation probes. "
                    f"Stateless PolarMuon/GD nrUpdate={fmt(stateless_nr['geomean_ratio'])} CI={ratio_ci(stateless_nr)}, "
                    f"stUpdate={fmt(stateless_st['geomean_ratio'])} CI={ratio_ci(stateless_st)}, while "
                    f"update_grad_inner={fmt(stateless_inner['geomean_ratio'])} CI={ratio_ci(stateless_inner)} under matched Frobenius update size. "
                    f"In short stateless trajectories, PolarMuon/GD mean_nrUpdate={fmt(traj_nr['geomean_ratio'])} CI={ratio_ci(traj_nr)}, "
                    f"total_decrease={fmt(traj_decrease['geomean_ratio'])} CI={ratio_ci(traj_decrease)}, final_loss={fmt(traj_loss['geomean_ratio'])} CI={ratio_ci(traj_loss)}."
                ),
                "interpretation": "This makes the paper's causal language more precise: polar direction alone is enough to create the high-rank update spectrum across one-step and short-trajectory controls, while progress still depends on allocation, vectors, state, norm budget, and horizon.",
                "caveat": "The stateless trajectory ablation still does not model Adam's state or retuned full training.",
            },
            {
                "claim": "Muon's direction advantage is task/layer conditional after size control.",
                "status": "supported",
                "evidence": (
                    f"Target-update sweep: Matrix Sensing kappa=1e2 ratio={fmt(target_ms['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(target_ms)}, but SmallMLP hidden=64 ratio={fmt(target_mlp64['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(target_mlp64)}."
                ),
                "interpretation": "Direction, not only raw LR scale, matters; but the sign depends on problem geometry.",
                "caveat": "The target-norm sweep is representative, not exhaustive.",
            },
            {
                "claim": "Natural update vectors are locally consequential.",
                "status": "supported as one-step local evidence",
                "evidence": (
                    f"Natural-update swap all-budget first-order ratio={fmt(natural_first['mean_signed_other_over_own'])} "
                    f"CI=[{fmt(natural_first['signed_ratio_ci95_low'])}, {fmt(natural_first['signed_ratio_ci95_high'])}], "
                    f"observed delta-loss ratio={fmt(natural_delta['mean_signed_other_over_own'])} "
                    f"CI=[{fmt(natural_delta['signed_ratio_ci95_low'])}, {fmt(natural_delta['signed_ratio_ci95_high'])}]."
                ),
                "interpretation": "Using the other trajectory's natural update usually reduces local progress.",
                "caveat": "Some setting/budget/eval-state cells favor the other update.",
            },
            {
                "claim": "Trajectory-level optimizer superiority does not follow from one-step geometry.",
                "status": "not supported",
                "evidence": (
                    f"Continuation-LR sweep best switched/own final-loss ratio={fmt(switch_lr_all['mean_switched_over_own'])} "
                    f"CI=[{fmt(switch_lr_all['ratio_ci95_low'])}, {fmt(switch_lr_all['ratio_ci95_high'])}], "
                    f"with source-Adam ratio={fmt(switch_lr_adam['mean_switched_over_own'])} and "
                    f"source-Muon ratio={fmt(switch_lr_muon['mean_switched_over_own'])}."
                ),
                "interpretation": "Fresh switch behavior depends on source, task family, horizon, and tuning.",
                "caveat": "This is still short-horizon and uses a small continuation-LR grid.",
            },
            {
                "claim": "The update-spectrum story survives neural MNIST variants, but the progress gap remains Adam-favorable.",
                "status": "supporting neural negative control",
                "evidence": (
                    f"Deep MNIST MLP nrUpdate ratio={fmt(deep_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_nr)}, "
                    f"stUpdate ratio={fmt(deep_st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_st)}, "
                    f"but first-order ratio={fmt(deep_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_first)}. "
                    f"MNIST patch nrUpdate ratio={fmt(patch_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(patch_nr)}, "
                    f"stUpdate={fmt(patch_st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(patch_st)}, "
                    f"but first-order ratio={fmt(patch_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(patch_first)}. "
                    f"MNIST ConvNet nrUpdate ratio={fmt(conv_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(conv_nr)}, "
                    f"stUpdate={fmt(conv_st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(conv_st)}, "
                    f"but first-order ratio={fmt(conv_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(conv_first)}."
                ),
                "interpretation": "Depth, patch/shared-weight locality, and a true Conv2d kernel preserve Muon's update-spectrum signature while strengthening the warning that high-rank updates do not imply better neural progress.",
                "caveat": "This is still short-horizon and does not include a modern architecture or tuned long-horizon benchmark.",
            },
            {
                "claim": "Muon is generally more stable than Adam.",
                "status": "not supported as a broad claim",
                "evidence": (
                    f"Raw all-task rank-plane speed ratio={fmt(raw_rank_speed['geomean_ratio_muon_over_adam'])}, "
                    f"but equal-update all-task ratio={fmt(equal_rank_speed['geomean_ratio_muon_over_adam'])} "
                    f"and equal-update Matrix Sensing ratio={fmt(equal_ms_rank_speed['geomean_ratio_muon_over_adam'])}."
                ),
                "interpretation": "The apparent stability gap is partly scale- and task-dependent.",
                "caveat": "Stability can still be claimed only for a specific metric, task family, and control condition.",
            },
        ]
    )

    report_implications = pd.DataFrame(
        [
            {
                "section": "Main positive finding",
                "use": "Present Muon as an update-spectrum shaping method, not as a universally better optimizer.",
            },
            {
                "section": "Mechanism",
                "use": "Explain benefits through norm geometry and task/layer spectral conditions.",
            },
            {
                "section": "Negative controls",
                "use": "Use trajectory-switch probes to explicitly separate local geometry from optimizer-level performance.",
            },
            {
                "section": "Avoid",
                "use": "Do not claim larger rank statistics alone imply better loss, recovery, or classification outcomes.",
            },
        ]
    )

    text = f"""# E11 Research Synthesis

This generated note is the current paper-facing synthesis of the E11 experiments. It is intentionally stricter than the exploratory figures: every claim below is tied to a quantitative result and an explicit caveat.

For the figure/CSV source behind each claim, see [E11 evidence index](e11_evidence_index.md). For the specific stability caveat, see [E11 optimizer-invariance audit](e11_optimizer_invariance_audit.md). For a conservative screen of cross-task optimizer signatures, see [E11 cross-task optimizer signature](e11_cross_task_signature.md). For when the update-spectrum intervention helps or hurts, see [E11 mechanism boundary map](e11_mechanism_boundary.md). For the predictive-boundary gap, see [E11 boundary predictor audit](e11_boundary_predictor_audit.md). For the local first-order proposition behind the norm-geometry boundary, see [E11 theory note](e11_theory_note.md) and [E11 mechanism theorem bridge](e11_mechanism_theorem_bridge.md). For the stateless direction-level control, see [E11 stateless direction ablation](e11_stateless_direction_ablation.md) and [E11 stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md). For what each control isolates, see [E11 optimizer ablation map](e11_optimizer_ablation_map.md).

## Current Thesis

**Muon should be framed as an update-spectrum shaping optimizer.** It reliably changes the singular-value geometry of update matrices. Whether that local geometry improves optimization depends on task, layer, norm constraint, horizon, and tuning. The current evidence supports a local geometry mechanism, not a universal optimizer-superiority claim.

## Claim Cards

{markdown_table(claim_cards, ["claim", "status", "evidence", "interpretation", "caveat"])}

## Report Implications

{markdown_table(report_implications, ["section", "use"])}

## Recommended Main Story

1. Establish that Muon changes update-spectrum geometry, and show that this is the only current candidate that passes the conservative cross-task screen.
2. Show that one-step progress is locally explained by gradient-update alignment.
3. Show the mechanism boundary: polar/flat allocation is beneficial only under the right norm geometry.
4. Use the ablation map to separate update scale, layer allocation, stateless direction choice, singular-value allocation, singular-vector geometry, and continuation effects.
5. Show task/layer conditional wins and losses under global and per-layer update-size controls.
6. Use natural-update swaps as local evidence that optimizer-specific update vectors matter.
7. Use trajectory-switch probes as negative controls: local geometry is not the same as longer-horizon optimizer superiority.

## Remaining Open Gaps

1. The neural evidence now includes MNIST shallow/deeper MLP probes, a matrix-only patch/shared-weight surrogate, and a small true ConvNet, but not modern architectures or long-horizon tuned benchmarks.
2. Cross-task spectral predictors remain weak under leave-family-out evaluation.
3. Long-horizon and fully retuned trajectory-level comparisons remain open.
4. ExactMuon's flat update spectrum is partly by construction; the research value is in when this construction helps or hurts.
5. True momentum-free or state-standardized training variants are still needed to fully separate polar spectrum shaping from optimizer state over a trajectory.
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved research synthesis to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
