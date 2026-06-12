from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_paper_skeleton.md")


def main() -> None:
    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    boundary = pd.read_csv("results/e11_mechanism_boundary/mechanism_boundary_map.csv")
    boundary_predictor = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_summary.csv")
    mnist_pair = pd.read_csv("results/e11_mnist_mlp_probe/pair_summary.csv")
    mnist_spectrum = pd.read_csv("results/e11_mnist_mlp_probe/update_spectrum_summary.csv")
    deep_mnist_pair = pd.read_csv("results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    deep_mnist_spectrum = pd.read_csv("results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")
    conv_pair = pd.read_csv("results/e11_mnist_conv_probe/pair_summary.csv")
    conv_spectrum = pd.read_csv("results/e11_mnist_conv_probe/update_spectrum_summary.csv")
    boundary_predictor_uncertainty = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_uncertainty.csv")
    stateless_traj = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")

    nr_update = require_one(update_spectrum, problem_family="All", metric="nrUpdate")
    st_update = require_one(update_spectrum, problem_family="All", metric="stUpdate")
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
    matrix_sensing = require_one(boundary, boundary_axis="task_family", condition="MatrixSensing")
    mf_input = require_one(boundary, boundary_axis="task_family", condition="MatrixFactorizationInput")
    mlp64 = require_one(boundary, boundary_axis="target_update_size", condition="Small MLP digits hidden=64")
    predictor_focus = boundary_predictor[
        (boundary_predictor["target"] == "update_grad_inner_muon_higher")
        & (boundary_predictor["evaluation"] == "leave_setting_out")
    ]
    predictor_best = predictor_focus.groupby("feature_set", as_index=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean")
    ).sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
    mnist_nr = require_one(mnist_spectrum, problem_family="MNISTMLP", metric="nrUpdate")
    mnist_h128 = require_one(mnist_pair, group_type="hidden_all_targets", hidden_dim="128", metric="update_grad_inner")
    deep_nr = require_one(deep_mnist_spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate")
    deep_first = require_one(deep_mnist_pair, group_type="all", hidden_dim="All", num_factors="All", metric="update_grad_inner")
    conv_nr = require_one(conv_spectrum, problem_family="MNISTConvNet", metric="nrUpdate")
    conv_first = require_one(conv_pair, group_type="all", filters="All", kernel_size="All", metric="update_grad_inner")
    predictor_uncertainty_best = (
        boundary_predictor_uncertainty[
            boundary_predictor_uncertainty["target"].eq("update_grad_inner_muon_higher")
        ]
        .sort_values("mean_balanced_accuracy_chance_filled", ascending=False)
        .iloc[0]
    )
    traj_decrease = require_one(
        stateless_traj,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="total_decrease",
    )

    paper_claims = pd.DataFrame(
        [
            {
                "claim": "Muon shapes update spectra.",
                "status": "main claim",
                "evidence": (
                    f"`nrUpdate` ratio {fmt(nr_update['geomean_ratio_muon_over_adam'])} {ratio_ci(nr_update)}; "
                    f"`stUpdate` ratio {fmt(st_update['geomean_ratio_muon_over_adam'])} {ratio_ci(st_update)}."
                ),
                "figure_or_table": "update-spectrum robustness; cross-task signature table",
            },
            {
                "claim": "One-step progress can be analyzed through gradient-update alignment.",
                "status": "diagnostic claim",
                "evidence": (
                    f"Spearman(delta_loss, <G,D>)={fmt(calibration_all['spearman_delta_vs_first_order'])} "
                    f"[{fmt(calibration_all['spearman_ci95_low'])}, {fmt(calibration_all['spearman_ci95_high'])}]."
                ),
                "figure_or_table": "first-order calibration",
            },
            {
                "claim": "The same spectral bias has a norm-geometry boundary.",
                "status": "mechanism claim",
                "evidence": (
                    f"flat/GD ratio {fmt(spectral_fro['geomean_ratio'])} {ratio_ci(spectral_fro)} under Frobenius budget; "
                    f"{fmt(spectral_op['geomean_ratio'])} {ratio_ci(spectral_op)} under operator-norm budget."
                ),
                "figure_or_table": "spectral-allocation probe",
            },
            {
                "claim": "Muon advantage is conditional, not universal.",
                "status": "boundary claim",
                "evidence": (
                    f"Matrix Sensing ratio {fmt(matrix_sensing['ratio'])}; "
                    f"MF ratio {fmt(mf_input['ratio'])}; MLP hidden=64 ratio {fmt(mlp64['ratio'])}."
                ),
                "figure_or_table": "mechanism boundary map",
            },
            {
                "claim": "The current boundary is not yet a predictive law.",
                "status": "negative/paper-gap claim",
                "evidence": (
                    f"Best leave-setting-out boundary predictor balanced accuracy is {fmt(predictor_best['mean_balanced_accuracy'])}."
                    f" Chance-filled best is {fmt(predictor_uncertainty_best['mean_balanced_accuracy_chance_filled'])} "
                    f"CI=[{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_low'])}, "
                    f"{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_high'])}]."
                ),
                "figure_or_table": "boundary predictor baseline",
            },
            {
                "claim": "Neural MNIST probes preserve update-spectrum shaping but not universal advantage.",
                "status": "neural supporting evidence",
                "evidence": (
                    f"MNIST nrUpdate ratio {fmt(mnist_nr['geomean_ratio_muon_over_adam'])}; "
                    f"hidden=128 first-order ratio {fmt(mnist_h128['geomean_ratio_muon_over_adam'])}; "
                    f"Deep MNIST nrUpdate ratio {fmt(deep_nr['geomean_ratio_muon_over_adam'])}; "
                    f"Deep first-order ratio {fmt(deep_first['geomean_ratio_muon_over_adam'])}; "
                    f"ConvNet nrUpdate ratio {fmt(conv_nr['geomean_ratio_muon_over_adam'])}; "
                    f"ConvNet first-order ratio {fmt(conv_first['geomean_ratio_muon_over_adam'])}."
                ),
                "figure_or_table": "MNIST MLP, Deep MNIST MLP, and MNIST ConvNet probes",
            },
            {
                "claim": "Polar direction alone is not enough for progress under Frobenius-matched trajectories.",
                "status": "mechanism negative control",
                "evidence": f"Stateless trajectory PolarMuon/GD total decrease ratio {fmt(traj_decrease['geomean_ratio'])} {ratio_ci(traj_decrease)}.",
                "figure_or_table": "stateless optimizer trajectory ablation",
            },
        ]
    )

    section_plan = pd.DataFrame(
        [
            {
                "section": "Introduction",
                "purpose": "Motivate optimizer geometry beyond final-loss leaderboards.",
                "must_include": "State that the paper studies local update geometry, not a universal Muon superiority claim.",
            },
            {
                "section": "Setup and Diagnostics",
                "purpose": "Define update spectrum, stable/effective rank, one-step decrease, and matched-update controls.",
                "must_include": "`G_i`, signed update `Delta W_i`, positive descent update `D_i`, `<G,D>`, `nrUpdate`, `stUpdate`, and exact control protocol.",
            },
            {
                "section": "Cross-Task Signature",
                "purpose": "Show the robust optimizer-intrinsic effect.",
                "must_include": "Only update-spectrum metrics pass the three-family screen.",
            },
            {
                "section": "Mechanism Probe",
                "purpose": "Explain why flat/polar spreading can help or hurt.",
                "must_include": "Frobenius vs operator-norm spectral-allocation sign flip and the local theory note.",
            },
            {
                "section": "Boundary Experiments",
                "purpose": "Show task/layer conditionality under controls.",
                "must_include": "MF, Matrix Sensing, sklearn digits MLP, shallow/deep MNIST MLP, MNIST patch/ConvNet probes, target update, per-layer controls, and optimizer ablation map.",
            },
            {
                "section": "Negative Controls",
                "purpose": "Prevent overclaiming from local geometry.",
                "must_include": "Switch/continuation, stability audit, and final-performance caveats.",
            },
            {
                "section": "Discussion",
                "purpose": "Frame what would be needed for a predictive theory.",
                "must_include": "Weak leave-setting-out boundary predictor and modern/long-horizon neural benchmark as future work.",
            },
        ]
    )

    figure_plan = pd.DataFrame(
        [
            {
                "slot": "Figure 1",
                "artifact": "figures/e11_equal_update/update_spectrum_robustness.png",
                "message": "Muon has a robust update-spectrum signature.",
            },
            {
                "slot": "Figure 2",
                "artifact": "figures/e11_equal_update/first_order_calibration.png",
                "message": "Observed one-step decrease is locally calibrated by `<G,D>`.",
            },
            {
                "slot": "Figure 3",
                "artifact": "figures/e11_spectral_allocation_probe/spectral_allocation_ratios.png",
                "message": "Flat/polar allocation wins under operator budget and loses under Frobenius budget; theory note and theorem bridge state the matching local constrained problem.",
            },
            {
                "slot": "Table 1",
                "artifact": "results/e11_mechanism_boundary/mechanism_boundary_map.csv",
                "message": "The local advantage flips by problem and control condition.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_main_paper_package.md",
                "message": "Selects the smallest main figure/table package and assigns the remaining probes to appendix roles.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_quantitative_claim_ledger.md",
                "message": "Records which claims are paper-ready, which quantitative anchors they require, and which wording to avoid.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_paper_numbers.tex",
                "message": "Generated LaTeX macros for the paper-facing ratios, confidence intervals, and calibration numbers.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_reproduction_checklist.md",
                "message": "Separates minimal main-paper reproduction from appendix/guardrail evidence and validation commands.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_main_figure_captions.md",
                "message": "Drafts paper-safe captions for the proposed main figures and table.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_notation_glossary.md",
                "message": "Defines the shared notation for gradients, updates, ranks, activation products, and diagnostics.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_mechanism_theorem_bridge.md",
                "message": "Maps theorem assumptions to evidence and states safe versus unsafe causal language.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_optimizer_invariance_audit.md",
                "message": "Muon is not generally more stable; stability is task/scale dependent.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_optimizer_ablation_map.md",
                "message": "Ablation ladder separates update size, layer allocation, stateless direction/trajectory, spectrum allocation, vector geometry, and continuation effects.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_boundary_predictor.md",
                "message": "The current descriptive boundary is not yet a strong leave-setting-out predictive law.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_boundary_predictor_audit.md",
                "message": "Explains why the current predictor is not paper-ready and specifies the next held-out test standard.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_mnist_mlp_probe.md",
                "message": "MNIST MLP preserves update-spectrum shaping while showing conditional first-order advantage.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_deep_mnist_mlp_probe.md",
                "message": "Deep MNIST MLP preserves update-spectrum shaping while becoming more Adam-favorable for first-order progress.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_mnist_conv_probe.md",
                "message": "Small true ConvNet preserves update-spectrum shaping while remaining Adam-favorable for first-order progress.",
            },
            {
                "slot": "Appendix",
                "artifact": "discussion/e11_stateless_optimizer_trajectory.md",
                "message": "Polar stateless trajectories retain high-rank updates but do not improve total decrease under Frobenius-matched steps.",
            },
        ]
    )

    text = f"""# E11 Paper Skeleton

This generated skeleton is not a draft paper. It is a maintainable bridge from the current evidence base to a publishable manuscript.

## Working Title

Update-Spectrum Shaping in Muon: Local Geometry, Norm Constraints, and Boundaries of Optimization Progress

## Abstract Sketch

Muon-style optimizers apply a polar-like update that changes the singular-value geometry of parameter updates. In controlled experiments across matrix factorization with input, matrix sensing, and a small neural benchmark, Muon consistently produces flatter and higher-rank update spectra than Adam at matched update size. We show that one-step progress is well captured by gradient-update alignment, and use this diagnostic to identify when Muon's flat/polar spectral bias helps or hurts. The key boundary is not rank itself: flat/polar spreading is favorable under operator-norm-like constraints but unfavorable under Frobenius constraints, and its local advantage flips by task family and layer geometry. These results support viewing Muon as an update-spectrum shaping method rather than as a universally better or more stable optimizer.

## Core Claims

{markdown_table(paper_claims, ["claim", "status", "evidence", "figure_or_table"])}

## Section Plan

{markdown_table(section_plan, ["section", "purpose", "must_include"])}

## Main Figure/Table Plan

{markdown_table(figure_plan, ["slot", "artifact", "message"])}

## Current Manuscript Gap

The current evidence is strong enough for a focused local-geometry paper, but not yet for a broad optimizer-performance paper. Before writing a full manuscript, the next two paper-critical items are:

1. A modern or longer-horizon neural benchmark with explicit treatment of non-matrix parameters; current MNIST MLP, patch, and ConvNet probes remain short-horizon sanity/negative controls.
2. A stronger predictive boundary model for when Muon's flat/polar update improves `<G,D>`; the current leave-setting-out baseline is weak.

## Linked Evidence

- [paper-readiness audit](e11_paper_readiness_audit.md)
- [reviewer risk audit](e11_reviewer_risk_audit.md)
- [research synthesis](e11_research_synthesis.md)
- [evidence index](e11_evidence_index.md)
- [theory note](e11_theory_note.md)
- [mechanism theorem bridge](e11_mechanism_theorem_bridge.md)
- [optimizer ablation map](e11_optimizer_ablation_map.md)
- [mechanism boundary map](e11_mechanism_boundary.md)
- [boundary predictor baseline](e11_boundary_predictor.md)
- [boundary predictor audit](e11_boundary_predictor_audit.md)
- [MNIST MLP probe](e11_mnist_mlp_probe.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
- [MNIST ConvNet probe](e11_mnist_conv_probe.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved paper skeleton to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
