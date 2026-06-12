from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.paper_stats import load_paper_stats
from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_paper_readiness_audit.md")


def main() -> None:
    paper_stats = load_paper_stats()

    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    cross_task = pd.read_csv("results/e11_cross_task_signature/cross_task_signature_summary.csv")
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    stateless = pd.read_csv("results/e11_stateless_direction_ablation/stateless_direction_summary.csv")
    stateless_traj = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")
    switch_lr = pd.read_csv("results/e11_optimizer_switch_lr_sweep/optimizer_switch_lr_summary.csv")
    boundary_predictor = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_summary.csv")
    boundary_predictor_uncertainty = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_uncertainty.csv")
    mnist_pair = pd.read_csv("results/e11_mnist_mlp_probe/pair_summary.csv")
    mnist_spectrum = pd.read_csv("results/e11_mnist_mlp_probe/update_spectrum_summary.csv")
    deep_mnist_pair = pd.read_csv("results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    deep_mnist_spectrum = pd.read_csv("results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")
    patch_pair = pd.read_csv("results/e11_mnist_patch_probe/pair_summary.csv")
    patch_spectrum = pd.read_csv("results/e11_mnist_patch_probe/update_spectrum_summary.csv")
    conv_pair = pd.read_csv("results/e11_mnist_conv_probe/pair_summary.csv")
    conv_spectrum = pd.read_csv("results/e11_mnist_conv_probe/update_spectrum_summary.csv")

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
    switch_all = require_one(switch_lr, group_type="all", metric="loss_after", source_algo="All")
    passed = cross_task[cross_task["passes_cross_task_screen"]]
    boundary_counts = paper_stats.boundary_counts
    predictor_focus = boundary_predictor[
        (boundary_predictor["target"] == "update_grad_inner_muon_higher")
        & (boundary_predictor["evaluation"] == "leave_setting_out")
    ]
    predictor_best = predictor_focus.groupby("feature_set", as_index=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auc=("auc", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False).iloc[0]
    predictor_uncertainty_best = (
        boundary_predictor_uncertainty[
            boundary_predictor_uncertainty["target"].eq("update_grad_inner_muon_higher")
        ]
        .sort_values("mean_balanced_accuracy_chance_filled", ascending=False)
        .iloc[0]
    )
    mnist_nr = require_one(mnist_spectrum, problem_family="MNISTMLP", metric="nrUpdate")
    mnist_st = require_one(mnist_spectrum, problem_family="MNISTMLP", metric="stUpdate")
    mnist_h128 = require_one(mnist_pair, group_type="hidden_all_targets", hidden_dim="128", metric="update_grad_inner")
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

    claim_status = pd.DataFrame(
        [
            {
                "claim": "Muon imposes a distinct update-spectrum bias.",
                "readiness": "paper-ready core claim",
                "evidence": (
                    f"Equal-update nrUpdate ratio={fmt(nr_update['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(nr_update)}; stUpdate ratio={fmt(st_update['geomean_ratio_muon_over_adam'])} CI={ratio_ci(st_update)}. "
                    f"{len(passed)} update-spectrum metrics pass the three-family screen."
                ),
                "why_it_is_ready": "It is directly measured, controlled for global update size, and replicated across the current three task families.",
                "remaining_risk": "ExactMuon's polar update makes this partly construction-level; the paper must frame it as a controlled optimizer property.",
            },
            {
                "claim": "One-step progress is locally explained by gradient-update alignment.",
                "readiness": "paper-ready diagnostic claim",
                "evidence": (
                    f"Spearman(delta_loss, <G,D>)={fmt(calibration_all['spearman_delta_vs_first_order'])} "
                    f"CI=[{fmt(calibration_all['spearman_ci95_low'])}, {fmt(calibration_all['spearman_ci95_high'])}], "
                    f"within-factor-2={fmt(calibration_all['within_factor_2'])}."
                ),
                "why_it_is_ready": "This validates the local analysis tool used in the rest of the paper.",
                "remaining_risk": "It is a one-step claim, not a statement about final optimization quality.",
            },
            {
                "claim": "Flat/polar spectral spreading has a norm-geometry boundary.",
                "readiness": "strong mechanism claim",
                "evidence": (
                    f"flat_polar/GD-spectrum ratio={fmt(spectral_fro['geomean_ratio'])} CI={ratio_ci(spectral_fro)} under Frobenius budget; "
                    f"{fmt(spectral_op['geomean_ratio'])} CI={ratio_ci(spectral_op)} under operator-norm budget. "
                    f"On actual problem states, stateless PolarMuon/GD gives nrUpdate={fmt(stateless_nr['geomean_ratio'])} "
                    f"CI={ratio_ci(stateless_nr)} and stUpdate={fmt(stateless_st['geomean_ratio'])} CI={ratio_ci(stateless_st)}, "
                    f"but Frobenius-matched update_grad_inner={fmt(stateless_inner['geomean_ratio'])} CI={ratio_ci(stateless_inner)}. "
                    f"In short stateless trajectories, mean_nrUpdate={fmt(traj_nr['geomean_ratio'])} CI={ratio_ci(traj_nr)} while total_decrease={fmt(traj_decrease['geomean_ratio'])} CI={ratio_ci(traj_decrease)}."
                ),
                "why_it_is_ready": "The sign flips under a clean intervention that changes the budget geometry, and stateless one-step plus short-trajectory controls confirm that polar direction alone creates the high-rank update spectrum.",
                "remaining_risk": "These controls are mechanism probes; they still do not replace retuned full optimizer experiments.",
            },
            {
                "claim": "Muon's local advantage is task/layer conditional.",
                "readiness": "paper-ready boundary claim",
                "evidence": (
                    f"Boundary map contains {boundary_counts.muon_flat_favorable} Muon/flat favorable rows, "
                    f"{boundary_counts.own_update_positive_control} own-update positive-control row, "
                    f"{boundary_counts.unfavorable} Adam/GD unfavorable rows, and "
                    f"{boundary_counts.mixed} mixed/uncertain row."
                ),
                "why_it_is_ready": "Multiple controls show sign flips after matching global or per-layer update sizes.",
                "remaining_risk": "The boundary is descriptive; it is not yet a predictive law for unseen tasks.",
            },
            {
                "claim": "The update-spectrum story extends to a stronger neural sanity benchmark.",
                "readiness": "supporting evidence, not final neural claim",
                "evidence": (
                    f"MNIST MLP nrUpdate ratio={fmt(mnist_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(mnist_nr)}; "
                    f"stUpdate ratio={fmt(mnist_st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(mnist_st)}; "
                    f"hidden=128 first-order ratio={fmt(mnist_h128['geomean_ratio_muon_over_adam'])} CI={ratio_ci(mnist_h128)}. "
                    f"Deep MNIST MLP nrUpdate={fmt(deep_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_nr)}, "
                    f"stUpdate={fmt(deep_st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_st)}, "
                    f"first-order ratio={fmt(deep_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(deep_first)}. "
                    f"MNIST patch nrUpdate={fmt(patch_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(patch_nr)}, "
                    f"stUpdate={fmt(patch_st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(patch_st)}, "
                    f"first-order ratio={fmt(patch_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(patch_first)}. "
                    f"MNIST ConvNet nrUpdate={fmt(conv_nr['geomean_ratio_muon_over_adam'])} CI={ratio_ci(conv_nr)}, "
                    f"stUpdate={fmt(conv_st['geomean_ratio_muon_over_adam'])} CI={ratio_ci(conv_st)}, "
                    f"first-order ratio={fmt(conv_first['geomean_ratio_muon_over_adam'])} CI={ratio_ci(conv_first)}."
                ),
                "why_it_is_ready": "It checks the main update-spectrum claim on real torchvision MNIST with shallow/deeper MLPs, a matrix-only patch/shared-weight surrogate, and a true Conv2d kernel under matched-update control.",
                "remaining_risk": "It is still short-horizon and does not use a modern architecture or tuned long-horizon benchmark.",
            },
            {
                "claim": "A simple local-feature model predicts the boundary.",
                "readiness": "not yet paper-ready",
                "evidence": (
                    f"Best leave-setting-out balanced accuracy={fmt(predictor_best['mean_balanced_accuracy'])} "
                    f"from {predictor_best['feature_set']} when degenerate settings are skipped; "
                    f"chance-filled best={fmt(predictor_uncertainty_best['mean_balanced_accuracy_chance_filled'])} "
                    f"CI=[{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_low'])}, "
                    f"{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_high'])}]."
                ),
                "why_it_is_ready": "It provides a pre-specified baseline and prevents pretending the descriptive boundary is already predictive.",
                "remaining_risk": "The best current predictor is weak and setting-sensitive; a publishable predictive claim needs a held-out benchmark.",
            },
            {
                "claim": "Muon is globally better or generally more stable.",
                "readiness": "reject as main claim",
                "evidence": (
                    f"Continuation-LR sweep best switched/own final-loss ratio={fmt(switch_all['mean_switched_over_own'])} "
                    f"CI=[{fmt(switch_all['ratio_ci95_low'])}, {fmt(switch_all['ratio_ci95_high'])}]; "
                    "cross-task state-volatility and one-step progress do not pass the conservative screen."
                ),
                "why_it_is_ready": "The negative conclusion is well supported and prevents overclaiming.",
                "remaining_risk": "A larger hyperparameter search could change optimizer-performance comparisons, but not the current caution.",
            },
        ]
    )

    publishable_story = pd.DataFrame(
        [
            {
                "section": "Question",
                "content": "Does Muon act as a geometry-shaping optimizer, and when does that geometry help optimization?",
            },
            {
                "section": "Core finding",
                "content": "Muon consistently produces flatter, higher-rank update spectra than Adam under matched update size.",
            },
            {
                "section": "Mechanism",
                "content": "The flat/polar update spectrum helps only when the local gradient/norm geometry rewards spectral spreading.",
            },
            {
                "section": "Boundary",
                "content": "The advantage flips by task family, MLP width/layer control, and Frobenius versus operator-norm budget.",
            },
            {
                "section": "Negative control",
                "content": "Local geometry does not imply global optimizer superiority, general stability, or better final loss.",
            },
        ]
    )

    next_experiments = pd.DataFrame(
        [
            {
                "priority": "must-have",
                "experiment": "Stronger neural benchmark",
                "purpose": "Show the update-spectrum and boundary story beyond small digits MLP.",
                "minimum_standard": "MNIST shallow/deeper MLP probes, a matrix-only patch surrogate, and a small true ConvNet are now available; next need a modern benchmark with reasonable tuning and per-layer diagnostics.",
            },
            {
                "priority": "must-have",
                "experiment": "Predictive boundary model",
                "purpose": "Turn the descriptive boundary map into a quantitative rule.",
                "minimum_standard": "Current leave-setting-out baseline is weak; define a stronger rule and test it on a genuinely held-out benchmark.",
            },
            {
                "priority": "must-have",
                "experiment": "Optimizer variants / ablations",
                "purpose": "Separate polar spectrum shaping from other optimizer details.",
                "minimum_standard": "Current ablation map now includes stateless one-step and short-trajectory controls in addition to update size, layer allocation, spectral allocation, vector swaps, and continuation effects; next need true Adam-state trajectory variants.",
            },
            {
                "priority": "should-have",
                "experiment": "Longer-horizon retuned runs",
                "purpose": "Clarify whether local geometry ever accumulates into final performance.",
                "minimum_standard": "Report final loss/accuracy with small LR grids and explicitly keep it separate from one-step mechanism claims.",
            },
            {
                "priority": "should-have",
                "experiment": "Theory formalization",
                "purpose": "State mathematically when a flat/polar update beats a gradient-spectrum update.",
                "minimum_standard": "Initial local proposition is in e11_theory_note.md; next step is a polished theorem/proof section with exact assumptions.",
            },
        ]
    )

    next_experiment_checklist = pd.DataFrame(
        [
            {
                "work_item": "CNN or matrix-only convolutional surrogate",
                "implementation_entry": "`e11_condition_geometry/problems/mnist_patch_classifier.py`, `e11_condition_geometry/problems/mnist_convnet.py`, `scripts/e11_run_mnist_patch_probe.py`, and `scripts/e11_run_mnist_conv_probe.py` now implement the patch surrogate and true ConvNet probes.",
                "acceptance_evidence": "Matched-update `nrUpdate`/`stUpdate` CI and first-order Muon/Adam ratio CI for both patch and Conv2d probes.",
                "claim_decision": "Moves neural evidence beyond pure MLP while keeping modern-architecture and long-horizon claims out of scope.",
            },
            {
                "work_item": "Held-out boundary prediction benchmark",
                "implementation_entry": "Extend `scripts/e11_run_boundary_predictor.py` with a pre-specified held-out family or architecture split.",
                "acceptance_evidence": "Balanced accuracy and AUC with confidence intervals on a held-out setting not used to choose features.",
                "claim_decision": "Decides whether the boundary map can be stated as predictive rather than descriptive.",
            },
            {
                "work_item": "Approximate Muon / polar-iteration ablation",
                "implementation_entry": "Add optimizer variants under `e11_condition_geometry/optimizers/` and reuse equal-update plus stateless trajectory summaries.",
                "acceptance_evidence": "Update-spectrum ratio and first-order ratio as a function of polar approximation strength at matched update size.",
                "claim_decision": "Separates ExactMuon construction effects from implementable Muon-like behavior.",
            },
            {
                "work_item": "Retuned longer-horizon training grid",
                "implementation_entry": "Add a small LR/steps grid runner rather than extending the one-step mechanism scripts.",
                "acceptance_evidence": "Final loss/error ratios with CI, reported separately from `update_grad_inner` and same-batch `delta_loss`.",
                "claim_decision": "Decides whether any final-performance statement belongs in the paper or should remain excluded.",
            },
            {
                "work_item": "Polished local theorem/proof",
                "implementation_entry": "Turn `discussion/e11_theory_note.md` and `discussion/e11_mechanism_theorem_bridge.md` into a theorem statement with exact norm constraints.",
                "acceptance_evidence": "A proof that matches the spectral-allocation probe's Frobenius/operator-norm interventions without changing notation.",
                "claim_decision": "Determines whether the mechanism can be presented as a theorem-backed main result.",
            },
        ]
    )

    text = f"""# E11 Paper-Readiness Audit

This generated audit reframes the current E11 results around a publishable paper target. It separates claims that are currently strong enough for a paper from claims that should remain caveats or future work.

## Proposed Paper Thesis

Muon is best understood as an **update-spectrum shaping optimizer**. Its robust effect is not that it is universally better or more stable, but that it imposes a flat/polar spectral bias on update matrices. This bias improves one-step progress only when the local task/layer/norm geometry rewards spectral spreading.

## Claim Readiness

{markdown_table(claim_status, ["claim", "readiness", "evidence", "why_it_is_ready", "remaining_risk"])}

## Publishable Story Arc

{markdown_table(publishable_story, ["section", "content"])}

## Experiments Still Needed For A Strong Paper

{markdown_table(next_experiments, ["priority", "experiment", "purpose", "minimum_standard"])}

## Next Experiment Checklist

{markdown_table(next_experiment_checklist, ["work_item", "implementation_entry", "acceptance_evidence", "claim_decision"])}

## What Not To Claim

1. Do not claim Muon is generally better than Adam.
2. Do not claim higher rank or stable rank directly implies lower loss.
3. Do not claim Muon is generally more stable.
4. Do not claim the current boundary map is already a predictive theory for unseen tasks.

## Current Best Paper Title Direction

> Update-Spectrum Shaping in Muon: Local Geometry, Norm Constraints, and Boundaries of Optimization Progress

## Sources

- [research synthesis](e11_research_synthesis.md)
- [cross-task optimizer signature](e11_cross_task_signature.md)
- [mechanism boundary map](e11_mechanism_boundary.md)
- [boundary predictor baseline](e11_boundary_predictor.md)
- [MNIST MLP probe](e11_mnist_mlp_probe.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
- [stateless direction ablation](e11_stateless_direction_ablation.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
- [optimizer-invariance audit](e11_optimizer_invariance_audit.md)
- [claim validity audit](e11_claim_validity_audit.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved paper-readiness audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
