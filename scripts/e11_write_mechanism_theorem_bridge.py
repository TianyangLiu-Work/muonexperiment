from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_PATH = Path("discussion/e11_mechanism_theorem_bridge.md")


def main() -> None:
    calibration = pd.read_csv("results/e11_equal_update/first_order_calibration_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    stateless = pd.read_csv("results/e11_stateless_direction_ablation/stateless_direction_summary.csv")
    trajectory = pd.read_csv("results/e11_stateless_optimizer_trajectory/stateless_optimizer_summary.csv")
    deep_mnist = pd.read_csv("results/e11_deep_mnist_mlp_probe/pair_summary.csv")
    deep_spectrum = pd.read_csv("results/e11_deep_mnist_mlp_probe/update_spectrum_summary.csv")

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
    trajectory_nr = require_one(
        trajectory,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="mean_nrUpdate",
    )
    trajectory_decrease = require_one(
        trajectory,
        group_type="all",
        problem_family="All",
        comparison="PolarMuon/GD",
        metric="total_decrease",
    )
    deep_nr = require_one(deep_spectrum, problem_family="DeepMNISTMLP", metric="nrUpdate")
    deep_first = require_one(deep_mnist, group_type="all", hidden_dim="All", num_factors="All", metric="update_grad_inner")

    theorem_bridge = pd.DataFrame(
        [
            {
                "theory_piece": "First-order local analysis is meaningful.",
                "mathematical_content": "`delta_loss` should track `<G, D>` for small positive descent updates `D=W-W^+`.",
                "evidence": (
                    f"Spearman(delta_loss, <G, D>)={fmt(calibration_all['spearman_delta_vs_first_order'])} "
                    f"CI=[{fmt(calibration_all['spearman_ci95_low'])}, {fmt(calibration_all['spearman_ci95_high'])}], "
                    f"within-factor-2={fmt(calibration_all['within_factor_2'])}."
                ),
                "supports": "Using gradient-update alignment as the local progress diagnostic.",
                "does_not_support": "Any claim about final loss, convergence, or generalization by itself.",
            },
            {
                "theory_piece": "Frobenius-constrained linear objective favors gradient-shaped spectra.",
                "mathematical_content": r"`D_F^* = rG/||G||_F` with singular values proportional to those of `G`.",
                "evidence": f"flat_polar/GD first-order ratio under Frobenius budget={fmt(fro['geomean_ratio'])} CI={ratio_ci(fro)}.",
                "supports": "A flat/polar update can be locally worse than GD-spectrum allocation under a Frobenius budget.",
                "does_not_support": "That Muon is bad in every natural optimizer trajectory.",
            },
            {
                "theory_piece": "Operator-norm-constrained linear objective favors flat/polar spectra.",
                "mathematical_content": r"`D_op^* = eta U V^T` with equal active singular values.",
                "evidence": f"flat_polar/GD first-order ratio under operator budget={fmt(op['geomean_ratio'])} CI={ratio_ci(op)}.",
                "supports": "The precise sense in which Muon's polar-like update matches an operator-norm local optimum.",
                "does_not_support": "That higher rank alone is the right explanatory variable.",
            },
            {
                "theory_piece": "Polar direction alone creates the high-rank update signature.",
                "mathematical_content": "The polar factor has flattened active singular values independent of Adam state.",
                "evidence": (
                    f"Stateless PolarMuon/GD nrUpdate={fmt(stateless_nr['geomean_ratio'])} CI={ratio_ci(stateless_nr)}, "
                    f"but Frobenius-matched update_grad_inner={fmt(stateless_inner['geomean_ratio'])} CI={ratio_ci(stateless_inner)}."
                ),
                "supports": "Update-spectrum shaping is a direction-level mechanism, not merely a side effect of optimizer state.",
                "does_not_support": "That polar direction alone improves Frobenius-matched local progress.",
            },
            {
                "theory_piece": "The one-step mechanism persists across short stateless trajectories.",
                "mathematical_content": "Repeated polar directions retain high-rank updates under matched per-step Frobenius size.",
                "evidence": (
                    f"Stateless trajectory PolarMuon/GD mean_nrUpdate={fmt(trajectory_nr['geomean_ratio'])} "
                    f"CI={ratio_ci(trajectory_nr)}, total_decrease={fmt(trajectory_decrease['geomean_ratio'])} "
                    f"CI={ratio_ci(trajectory_decrease)}."
                ),
                "supports": "The mechanism is not just a single-step artifact.",
                "does_not_support": "That stateless polar training is a better optimizer under Frobenius-matched steps.",
            },
            {
                "theory_piece": "Neural negative control: high-rank updates can coexist with worse first-order progress.",
                "mathematical_content": "The theorem predicts a boundary, not a universal benefit of rank spreading.",
                "evidence": (
                    f"Deep MNIST MLP nrUpdate Muon/Adam={fmt(deep_nr['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(deep_nr)}, but first-order ratio={fmt(deep_first['geomean_ratio_muon_over_adam'])} "
                    f"CI={ratio_ci(deep_first)}."
                ),
                "supports": "The boundary framing is necessary even on neural probes.",
                "does_not_support": "A broad neural-performance claim for Muon.",
            },
        ]
    )

    claim_language = pd.DataFrame(
        [
            {
                "safe_wording": "Muon is an update-spectrum shaping method.",
                "reason": "The update-rank effect is direct, replicated, and survives matched-update controls.",
            },
            {
                "safe_wording": "Muon's polar-like direction matches the local operator-norm constrained optimum.",
                "reason": "This is exactly the local proposition and is supported by the operator-budget sign flip.",
            },
            {
                "safe_wording": "The optimization benefit is boundary-dependent.",
                "reason": "Frobenius, stateless trajectory, MF, and deep MNIST controls all show high-rank updates without better progress.",
            },
            {
                "safe_wording": "The current result is local-geometry evidence, not a convergence theorem.",
                "reason": "All core mechanism probes are one-step or short-horizon controls.",
            },
        ]
    )

    avoid_language = pd.DataFrame(
        [
            {
                "unsafe_wording": "Muon is better because it has higher rank updates.",
                "why_wrong": "Deep MNIST and stateless controls show higher rank with worse first-order progress.",
            },
            {
                "unsafe_wording": "The theory proves Muon should beat Adam.",
                "why_wrong": "The theorem compares constrained linearized directions, not full optimizer trajectories.",
            },
            {
                "unsafe_wording": "Operator-norm optimality explains all observed Muon behavior.",
                "why_wrong": "Natural optimizers also differ by state, layer allocation, singular vectors, and horizon.",
            },
        ]
    )

    text = f"""# E11 Mechanism Theorem Bridge

This generated note links the local theorem to the empirical controls. Its purpose is to make the paper's causal language precise: which statements follow from the theorem and probes, and which statements remain unsupported.

## Theorem-To-Evidence Map

{markdown_table(theorem_bridge, ["theory_piece", "mathematical_content", "evidence", "supports", "does_not_support"])}

## Safe Claim Language

{markdown_table(claim_language, ["safe_wording", "reason"])}

## Language To Avoid

{markdown_table(avoid_language, ["unsafe_wording", "why_wrong"])}

## One-Sentence Paper Use

The theorem explains Muon's polar update as the local solution to an operator-norm-constrained linearized problem, while the experiments show that this spectral bias is robust but only useful when the task, layer, and norm geometry reward such spectral spreading.

## Sources

- [theory note](e11_theory_note.md)
- [spectral allocation probe](e11_spectral_allocation_probe.md)
- [stateless direction ablation](e11_stateless_direction_ablation.md)
- [stateless optimizer trajectory ablation](e11_stateless_optimizer_trajectory.md)
- [Deep MNIST MLP probe](e11_deep_mnist_mlp_probe.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved mechanism theorem bridge to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
