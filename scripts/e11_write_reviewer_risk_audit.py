from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_reviewer_risk_audit.md")


def main() -> None:
    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    stateless = pd.read_csv("results/e11_stateless_direction_ablation/stateless_direction_summary.csv")
    trajectory = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")
    deep_pair = pd.read_csv("results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    deep_spectrum = pd.read_csv("results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")
    conv_pair = pd.read_csv("results/e11_mnist_conv_probe/pair_summary.csv")
    conv_spectrum = pd.read_csv("results/e11_mnist_conv_probe/update_spectrum_summary.csv")
    predictor = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_summary.csv")
    predictor_uncertainty = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_uncertainty.csv")
    switch_lr = pd.read_csv("results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_summary.csv")

    nr_update = require_one(update_spectrum, problem_family="All", metric="nrUpdate")
    st_update = require_one(update_spectrum, problem_family="All", metric="stUpdate")
    calibration_all = require_one(calibration, group="All")
    fro = require_one(
        spectral,
        group_type="budget_all",
        comparison="flat_polar_over_gd_spectrum",
        metric="update_grad_inner",
        budget="fro",
    )
    op = require_one(
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
    traj_decrease = require_one(
        trajectory,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="total_decrease",
    )
    deep_nr = require_one(deep_spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate")
    deep_first = require_one(deep_pair, group_type="all", hidden_dim="All", num_factors="All", metric="update_grad_inner")
    conv_nr = require_one(conv_spectrum, problem_family="MNISTConvNet", metric="nrUpdate")
    conv_first = require_one(conv_pair, group_type="all", filters="All", kernel_size="All", metric="update_grad_inner")
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
        predictor_uncertainty[predictor_uncertainty["target"].eq("update_grad_inner_muon_higher")]
        .sort_values("mean_balanced_accuracy_chance_filled", ascending=False)
        .iloc[0]
    )
    switch_all = require_one(switch_lr, group_type="all", metric="loss_after", source_algo="All")

    risks = pd.DataFrame(
        [
            {
                "reviewer_objection": "The update-rank result is trivial because ExactMuon is polar by construction.",
                "risk_level": "medium",
                "current_evidence": (
                    f"Equal-update nrUpdate={fmt(nr_update['geomean_ratio_muon_over_adam'])} CI={ratio_ci(nr_update)}, "
                    f"stUpdate={fmt(st_update['geomean_ratio_muon_over_adam'])} CI={ratio_ci(st_update)}; "
                    f"stateless PolarMuon/GD nrUpdate={fmt(stateless_nr['geomean_ratio'])} CI={ratio_ci(stateless_nr)}."
                ),
                "safe_response": "Frame this as the optimizer's controlled spectral bias, not as a surprising performance result.",
                "remaining_work": "Add non-exact Muon variants or approximate polar iterations if the paper targets broader optimizer implementations.",
            },
            {
                "reviewer_objection": "Higher-rank updates do not explain optimization progress.",
                "risk_level": "low for current claim, high for stronger claims",
                "current_evidence": (
                    f"Stateless PolarMuon/GD update_grad_inner={fmt(stateless_inner['geomean_ratio'])} CI={ratio_ci(stateless_inner)}; "
                    f"trajectory total_decrease={fmt(traj_decrease['geomean_ratio'])} CI={ratio_ci(traj_decrease)}; "
                    f"Deep MNIST first-order ratio={fmt(deep_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_first)} despite "
                    f"nrUpdate={fmt(deep_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_nr)}."
                ),
                "safe_response": "Agree; the paper's claim is boundary-dependent update-spectrum shaping, not rank-implies-progress.",
                "remaining_work": "Keep high-rank progress claims out of the abstract and theorem statements.",
            },
            {
                "reviewer_objection": "The theory is only local and does not prove optimizer superiority.",
                "risk_level": "low if framed correctly",
                "current_evidence": (
                    f"First-order calibration Spearman={fmt(calibration_all['spearman_delta_vs_first_order'])} "
                    f"CI=[{fmt(calibration_all['spearman_ci95_low'])}, {fmt(calibration_all['spearman_ci95_high'])}], "
                    f"Frobenius flat/GD={fmt(fro['geomean_ratio'])} CI={ratio_ci(fro)}, operator flat/GD={fmt(op['geomean_ratio'])} CI={ratio_ci(op)}."
                ),
                "safe_response": "State the theorem as a local constrained linearized result and use it only to explain mechanism probes.",
                "remaining_work": "A convergence theorem would be a different paper and is not currently supported.",
            },
            {
                "reviewer_objection": "The boundary map is descriptive but not predictive.",
                "risk_level": "high for predictive-theory claims",
                "current_evidence": (
                    f"Best leave-setting-out predictor balanced accuracy={fmt(predictor_best['mean_balanced_accuracy'])} "
                    f"from {predictor_best['feature_set']} when degenerate settings are skipped; "
                    f"chance-filled best={fmt(predictor_uncertainty_best['mean_balanced_accuracy_chance_filled'])} "
                    f"CI=[{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_low'])}, "
                    f"{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_high'])}]."
                ),
                "safe_response": "Explicitly present the predictor as a failed/weak baseline and call the boundary map descriptive.",
                "remaining_work": "Pre-register a predictor and test it on a genuinely new balanced held-out task or architecture.",
            },
            {
                "reviewer_objection": "The neural evidence is too small or not representative.",
                "risk_level": "medium",
                "current_evidence": (
                    f"Deep MNIST MLP preserves update-spectrum shaping with nrUpdate={fmt(deep_nr['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(deep_nr)}, but first-order progress is Adam-favorable: {fmt(deep_first['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(deep_first)}. MNIST ConvNet also preserves nrUpdate={fmt(conv_nr['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(conv_nr)} while first-order is Adam-favorable: {fmt(conv_first['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(conv_first)}."
                ),
                "safe_response": "Use neural experiments as sanity/negative controls, not as broad performance benchmarks.",
                "remaining_work": "Add modern or longer-horizon neural benchmarks only if the paper claims broad neural relevance.",
            },
            {
                "reviewer_objection": "Local geometry may not predict longer-horizon behavior.",
                "risk_level": "medium",
                "current_evidence": (
                    f"Continuation-LR sweep switched/own final-loss ratio={fmt(switch_all['mean_switched_over_own'])} "
                    f"CI=[{fmt(switch_all['ratio_ci95_low'])}, {fmt(switch_all['ratio_ci95_high'])}]."
                ),
                "safe_response": "Use continuation/switch probes as negative controls against overclaiming.",
                "remaining_work": "Run longer retuned training only if final-performance claims become central.",
            },
            {
                "reviewer_objection": "There are too many artifacts and the claim may be hard to follow.",
                "risk_level": "medium",
                "current_evidence": "Paper skeleton, evidence index, theorem bridge, and reviewer audit now provide a claim hierarchy.",
                "safe_response": "Keep the main paper to three claims: update-spectrum signature, local calibration, norm/boundary mechanism.",
                "remaining_work": "Before drafting, select 3 main figures and move most ablations to appendix.",
            },
        ]
    )

    claim_decisions = pd.DataFrame(
        [
            {
                "claim": "Muon is an update-spectrum shaping optimizer.",
                "decision": "main-paper claim",
                "reason": "Direct, robust, matched-update evidence supports it.",
            },
            {
                "claim": "Muon's polar direction is locally operator-norm optimal.",
                "decision": "main-paper mechanism claim",
                "reason": "Narrow theorem plus spectral allocation probe support this exactly.",
            },
            {
                "claim": "Muon is generally better than Adam.",
                "decision": "do not claim",
                "reason": "Boundary, deep MNIST, stateless trajectory, and continuation evidence contradict a broad claim.",
            },
            {
                "claim": "Local features predict when Muon wins.",
                "decision": "future-work claim only",
                "reason": "Current leave-setting-out predictor is weak and imbalanced.",
            },
            {
                "claim": "The paper gives a convergence theory.",
                "decision": "do not claim",
                "reason": "Current theorem is local and first-order.",
            },
        ]
    )

    text = f"""# E11 Reviewer Risk Audit

This generated audit lists the likely reviewer objections and the current evidence-backed response. It is meant to keep the manuscript focused on defensible claims.

## Risk Table

{markdown_table(risks, ["reviewer_objection", "risk_level", "current_evidence", "safe_response", "remaining_work"])}

## Claim Decisions

{markdown_table(claim_decisions, ["claim", "decision", "reason"])}

## Recommended Manuscript Discipline

1. Put update-spectrum shaping, first-order calibration, and norm-geometry boundary in the main paper.
2. Put neural probes, stateless trajectories, switch controls, and predictor failure modes in supporting/appendix sections.
3. Do not state or imply that higher update rank generally improves progress.
4. Do not present the current boundary predictor as a predictive theory.

## Sources

- [paper skeleton](e11_paper_skeleton.md)
- [mechanism theorem bridge](e11_mechanism_theorem_bridge.md)
- [boundary predictor audit](e11_boundary_predictor_audit.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
- [MNIST ConvNet probe](e11_mnist_conv_probe.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved reviewer risk audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
