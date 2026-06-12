from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_research_direction_map.md")


def claim_row(
    *,
    question: str,
    claim: str,
    status: str,
    evidence: str,
    interpretation: str,
    next_test: str,
) -> dict[str, str]:
    return {
        "question": question,
        "claim": claim,
        "status": status,
        "evidence": evidence,
        "interpretation": interpretation,
        "next_test": next_test,
    }


def main() -> None:
    update_spectrum = pd.read_csv("results/e11_equal_update/update_spectrum_summary.csv")
    first_order = pd.read_csv("results/e11_equal_update/first_order_pair_summary.csv")
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    stateless = pd.read_csv("results/e11_stateless_direction_ablation/stateless_direction_summary.csv")
    stateless_traj = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")
    boundary = pd.read_csv("results/e11_mechanism_boundary/mechanism_boundary_map.csv")
    predictor = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_summary.csv")
    mnist_spectrum = pd.read_csv("results/e11_mnist_mlp_probe/update_spectrum_summary.csv")
    mnist_pair = pd.read_csv("results/e11_mnist_mlp_probe/pair_summary.csv")
    deep_mnist_spectrum = pd.read_csv("results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")
    deep_mnist_pair = pd.read_csv("results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    conv_spectrum = pd.read_csv("results/e11_mnist_conv_probe/update_spectrum_summary.csv")
    conv_pair = pd.read_csv("results/e11_mnist_conv_probe/pair_summary.csv")
    predictor_uncertainty = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_uncertainty.csv")

    nr_update = require_one(update_spectrum, problem_family="All", metric="nrUpdate")
    st_update = require_one(update_spectrum, problem_family="All", metric="stUpdate")
    calibration_all = require_one(calibration, group="All")
    spectral_fro = require_one(
        spectral,
        group_type="budget_all",
        budget="fro",
        comparison="flat_polar_over_gd_spectrum",
        metric="update_grad_inner",
    )
    spectral_op = require_one(
        spectral,
        group_type="budget_all",
        budget="op",
        comparison="flat_polar_over_gd_spectrum",
        metric="update_grad_inner",
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
        stateless_traj,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="total_decrease",
    )
    ms_first_order = require_one(first_order, problem_family="MatrixSensing", metric="update_grad_inner")
    mf_first_order = require_one(first_order, problem_family="MatrixFactorizationInput", metric="update_grad_inner")
    mnist_nr_update = require_one(mnist_spectrum, problem_family="MNISTMLP", metric="nrUpdate")
    mnist_st_update = require_one(mnist_spectrum, problem_family="MNISTMLP", metric="stUpdate")
    mnist_h128 = require_one(mnist_pair, group_type="hidden_all_targets", hidden_dim="128", metric="update_grad_inner")
    deep_nr = require_one(deep_mnist_spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate")
    deep_first = require_one(deep_mnist_pair, group_type="all", hidden_dim="All", num_factors="All", metric="update_grad_inner")
    conv_nr = require_one(conv_spectrum, problem_family="MNISTConvNet", metric="nrUpdate")
    conv_first = require_one(conv_pair, group_type="all", filters="All", kernel_size="All", metric="update_grad_inner")
    predictor_uncertainty_best = (
        predictor_uncertainty[predictor_uncertainty["target"].eq("update_grad_inner_muon_higher")]
        .sort_values("mean_balanced_accuracy_chance_filled", ascending=False)
        .iloc[0]
    )

    leave_setting = predictor[predictor["evaluation"].eq("leave_setting_out")].copy()
    predictor_means = (
        leave_setting.groupby("feature_set", as_index=False)["balanced_accuracy"].mean().sort_values("balanced_accuracy", ascending=False)
    )
    best_predictor = predictor_means.iloc[0]
    favorable = int(boundary["direction"].isin(["Muon-favorable", "flat/polar-favorable", "own-update-favorable"]).sum())
    unfavorable = int(
        boundary["direction"].isin(["Adam/GD-favorable", "GD-spectrum-favorable", "weakly Adam/GD-favorable"]).sum()
    )

    claims = pd.DataFrame(
        [
            claim_row(
                question="What is Muon's optimizer-intrinsic signature?",
                claim="Muon is best framed as an update-spectrum shaping optimizer.",
                status="strong core claim",
                evidence=(
                    f"equal-update nrUpdate Muon/Adam={fmt(nr_update['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(nr_update)}; stUpdate={fmt(st_update['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(st_update)}"
                ),
                interpretation=(
                    "The cross-task effect that survives matched update size is spectral flattening/high-rank updates, "
                    "not universal loss improvement."
                ),
                next_test="Add momentum-free/state-standardized optimizer variants to isolate polar spectrum from optimizer state.",
            ),
            claim_row(
                question="Does this signature extend beyond toy matrix problems?",
                claim="The update-spectrum signature appears in MNIST MLP and ConvNet probes too.",
                status="supporting evidence",
                evidence=(
                    f"MNIST nrUpdate Muon/Adam={fmt(mnist_nr_update['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(mnist_nr_update)}; stUpdate={fmt(mnist_st_update['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(mnist_st_update)}. Deep MNIST nrUpdate={fmt(deep_nr['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(deep_nr)} with first-order ratio={fmt(deep_first['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(deep_first)}. ConvNet nrUpdate={fmt(conv_nr['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(conv_nr)} with first-order ratio={fmt(conv_first['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(conv_first)}"
                ),
                interpretation="The spectral-shaping effect is not limited to synthetic matrix objectives, but neural probes make the progress caveat sharper.",
                next_test="Move from short MNIST sanity checks to a modern or longer-horizon benchmark only if neural performance claims become central.",
            ),
            claim_row(
                question="Does higher-rank update geometry automatically imply better progress?",
                claim="No. Local progress is conditional on task and layer geometry.",
                status="strong boundary claim",
                evidence=(
                    f"MatrixSensing first-order Muon/Adam={fmt(ms_first_order['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(ms_first_order)}, but MF-with-input={fmt(mf_first_order['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(mf_first_order)}; boundary map has {favorable} favorable and {unfavorable} unfavorable rows."
                ),
                interpretation="The same optimizer-induced spectral bias can help or hurt depending on the local geometry.",
                next_test="Replace the descriptive boundary map with a held-out predictive rule, then test on a new task family.",
            ),
            claim_row(
                question="What local quantity explains observed one-step decrease?",
                claim="Observed one-step decrease is well calibrated by gradient-update alignment.",
                status="paper-ready diagnostic",
                evidence=(
                    f"Spearman(delta_loss, <G,D>)={fmt(calibration_all['spearman_delta_vs_first_order'])} "
                    f"CI=[{fmt(calibration_all['spearman_ci95_low'])}, {fmt(calibration_all['spearman_ci95_high'])}], "
                    f"within-factor-2={fmt(calibration_all['within_factor_2'])}"
                ),
                interpretation="The paper can use <G,D> as the local bridge between update geometry and loss change.",
                next_test="Keep this claim local; validate separately if longer-horizon progress is discussed.",
            ),
            claim_row(
                question="Why can a flat/polar spectrum help in some regimes and hurt in others?",
                claim="The sign flips with the norm budget, even though polar direction reliably raises update rank.",
                status="mechanistic evidence",
                evidence=(
                    f"flat_polar/GD under Frobenius budget={fmt(spectral_fro['geomean_ratio'])} "
                    f"CI={ratio_ci(spectral_fro)}, but under operator budget={fmt(spectral_op['geomean_ratio'])} "
                    f"CI={ratio_ci(spectral_op)}; stateless PolarMuon/GD nrUpdate={fmt(stateless_nr['geomean_ratio'])} "
                    f"CI={ratio_ci(stateless_nr)} while Frobenius-matched update_grad_inner={fmt(stateless_inner['geomean_ratio'])} "
                    f"CI={ratio_ci(stateless_inner)} and short-trajectory total_decrease={fmt(traj_decrease['geomean_ratio'])} "
                    f"CI={ratio_ci(traj_decrease)}"
                ),
                interpretation="Flat spectral allocation is not intrinsically better; it matches operator-norm-like geometry better than Frobenius geometry.",
                next_test="Formalize this as the main theorem/proposition and add trajectory-level optimizer variants.",
            ),
            claim_row(
                question="Do current features predict where Muon wins?",
                claim="Not yet at a publishable level.",
                status="open gap",
                evidence=(
                    f"best leave-setting-out balanced accuracy={fmt(best_predictor['balanced_accuracy'])} "
                    f"from {best_predictor['feature_set']} when degenerate settings are skipped; "
                    f"chance-filled best={fmt(predictor_uncertainty_best['mean_balanced_accuracy_chance_filled'])} "
                    f"CI=[{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_low'])}, "
                    f"{fmt(predictor_uncertainty_best['balanced_accuracy_ci95_high'])}]"
                ),
                interpretation="The current boundary is descriptive, not a reliable predictive theory for unseen settings.",
                next_test="Use the new benchmark as a held-out test instead of selecting features after seeing all task families.",
            ),
            claim_row(
                question="Does Muon's geometry guarantee neural performance gains?",
                claim="No; MNIST hidden=128 is Adam-favorable in first-order progress.",
                status="negative control",
                evidence=(
                    f"MNIST hidden=128 first-order Muon/Adam={fmt(mnist_h128['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(mnist_h128)}"
                ),
                interpretation="The neural evidence supports geometry shaping but warns against claiming optimizer superiority.",
                next_test="Separate local update geometry, final training loss, and classification error in any neural section.",
            ),
        ]
    )

    text = f"""# E11 Research Direction Map

This generated note is the compact paper-direction map for E11. It turns the current evidence into testable claims and explicit next experiments.

## Working Research Direction

Muon should be studied as an **update-spectrum shaping optimizer**. Its robust effect is to produce flatter, higher-rank update matrices than Adam under matched update size. The open scientific question is not whether Muon is universally better, but when this induced update geometry aligns with the local loss geometry strongly enough to explain one-step decrease or optimization progress.

## Claim Map

{markdown_table(claims, ["question", "claim", "status", "evidence", "interpretation", "next_test"])}

## Paper-Level Hypothesis

The current best hypothesis is:

> Muon imposes a flat/polar update-spectrum bias. This bias is a robust optimizer-level signature, but its optimization benefit is conditional: it helps when the local gradient/norm/task geometry rewards spectrally spread updates, and it hurts or becomes neutral otherwise.

## Current Paper Shape

| role | content |
|:--|:--|
| Core positive result | Muon reliably induces higher-rank, flatter update spectra than Adam under matched update size. |
| Mechanistic bridge | One-step loss decrease is well captured by `<G,D>`. |
| Boundary result | Muon's first-order advantage changes sign across task family, width/layer control, and norm budget. |
| Negative control | Higher rank does not imply lower loss, better final performance, or general stability. |
| Main gap | The current boundary map is descriptive; a stronger paper needs a held-out predictive boundary test and, only for broader neural claims, modern/longer-horizon neural benchmarks. |

## Immediate Next Experiments

1. Add a modern or longer-horizon neural benchmark with per-layer update spectra, first-order alignment, final loss, and classification error if neural performance claims become central.
2. Add optimizer variants that separate polar spectrum shaping from momentum/state details.
3. Turn the descriptive boundary map into a pre-specified predictor and test it on a genuinely held-out task or architecture.

## Sources

- [research synthesis](e11_research_synthesis.md)
- [paper-readiness audit](e11_paper_readiness_audit.md)
- [mechanism boundary map](e11_mechanism_boundary.md)
- [boundary predictor baseline](e11_boundary_predictor.md)
- [MNIST MLP probe](e11_mnist_mlp_probe.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
- [stateless direction ablation](e11_stateless_direction_ablation.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved research direction map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
