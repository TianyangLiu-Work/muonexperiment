from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, ratio_ci, require_one, write_markdown


OUTPUT_DIR = Path("results/e11_mechanism_boundary")
DISCUSSION_PATH = Path("discussion/e11_mechanism_boundary.md")


def classify_ratio(record: pd.Series, *, ratio_column: str = "geomean_ratio_muon_over_adam") -> str:
    ratio = float(record[ratio_column])
    low = float(record["ratio_ci95_low"])
    high = float(record["ratio_ci95_high"])
    if low > 1.0:
        return "Muon-favorable"
    if high < 1.0:
        return "Adam/GD-favorable"
    return "mixed_or_uncertain" if ratio >= 1.0 else "weakly Adam/GD-favorable"


def add_boundary_row(
    rows: list[dict[str, object]],
    *,
    boundary_axis: str,
    controlled_question: str,
    condition: str,
    metric: str,
    ratio_label: str,
    ratio: float,
    ci95: str,
    direction: str,
    interpretation: str,
    source: str,
) -> None:
    rows.append(
        {
            "boundary_axis": boundary_axis,
            "controlled_question": controlled_question,
            "condition": condition,
            "metric": metric,
            "ratio_label": ratio_label,
            "ratio": ratio,
            "ci95": ci95,
            "direction": direction,
            "interpretation": interpretation,
            "source": source,
        }
    )


def main() -> None:
    first_order = pd.read_csv("results/e11_equal_update/first_order_pair_summary.csv")
    target = pd.read_csv("results/e11_target_update_sweep/target_pair_summary.csv")
    layer_control = pd.read_csv("results/e11_mlp_per_layer_control/pair_summary.csv")
    spectral = pd.read_csv("results/e11_spectral_allocation_probe/spectral_allocation_summary.csv")
    natural = pd.read_csv("results/e11_natural_update_swap_probe/natural_update_swap_summary.csv")

    rows: list[dict[str, object]] = []

    for family in ["MatrixFactorizationInput", "MatrixSensing", "SmallMLPDigits"]:
        rec = require_one(first_order, problem_family=family, metric="update_grad_inner")
        add_boundary_row(
            rows,
            boundary_axis="task_family",
            controlled_question="At matched global update size, does Muon's natural update improve <G,D>?",
            condition=family,
            metric="update_grad_inner",
            ratio_label="Muon/Adam",
            ratio=float(rec["geomean_ratio_muon_over_adam"]),
            ci95=ratio_ci(rec),
            direction=classify_ratio(rec),
            interpretation="Task family alone can flip whether Muon's update spectrum improves local first-order progress.",
            source="results/e11_equal_update/first_order_pair_summary.csv",
        )

    for base_setting in [
        "MF input kappa=1e+02",
        "MF input kappa=1e+05",
        "Matrix sensing kappa=1e+02",
        "Matrix sensing kappa=1e+05",
        "Small MLP digits hidden=16",
        "Small MLP digits hidden=64",
    ]:
        rec = require_one(target, group_type="setting_all_targets", base_setting=base_setting, metric="update_grad_inner")
        add_boundary_row(
            rows,
            boundary_axis="target_update_size",
            controlled_question="After sweeping matched target update norms, does Muon's direction remain favorable?",
            condition=base_setting,
            metric="update_grad_inner",
            ratio_label="Muon/Adam",
            ratio=float(rec["geomean_ratio_muon_over_adam"]),
            ci95=ratio_ci(rec),
            direction=classify_ratio(rec),
            interpretation="The sign persists across target update sizes in some settings, but not uniformly across problem geometry.",
            source="results/e11_target_update_sweep/target_pair_summary.csv",
        )

    for hidden_dim in ["16", "64"]:
        rec = require_one(layer_control, group_type="hidden_all_targets", hidden_dim=hidden_dim, metric="update_grad_inner")
        add_boundary_row(
            rows,
            boundary_axis="per_layer_update_size",
            controlled_question="In MLP, after matching per-layer relative update norms, does Muon's direction remain favorable?",
            condition=f"Small MLP digits hidden={hidden_dim}",
            metric="update_grad_inner",
            ratio_label="Muon/Adam",
            ratio=float(rec["geomean_ratio_muon_over_adam"]),
            ci95=ratio_ci(rec),
            direction=classify_ratio(rec),
            interpretation="The MLP sign depends strongly on width even after per-layer update-size control.",
            source="results/e11_mlp_per_layer_control/pair_summary.csv",
        )

    for budget in ["fro", "op"]:
        rec = require_one(
            spectral,
            group_type="budget_all",
            budget=budget,
            comparison="flat_polar_over_gd_spectrum",
            metric="update_grad_inner",
        )
        add_boundary_row(
            rows,
            boundary_axis="norm_budget",
            controlled_question="If only singular-value allocation changes, when is a flat/polar spectrum better than GD spectrum?",
            condition=f"{budget} budget",
            metric="update_grad_inner",
            ratio_label="flat_polar/GD_spectrum",
            ratio=float(rec["geomean_ratio"]),
            ci95=ratio_ci(rec),
            direction="flat/polar-favorable" if rec["ratio_ci95_low"] > 1.0 else "GD-spectrum-favorable",
            interpretation="Flat/polar allocation is beneficial under operator-norm budget and harmful under Frobenius budget.",
            source="results/e11_spectral_allocation_probe/spectral_allocation_summary.csv",
        )

    for budget in ["All"]:
        rec = require_one(
            natural,
            group_type="all",
            target_layer_relative_norm="All",
            budget="All",
            eval_algo="All",
            metric="update_grad_inner",
        )
        add_boundary_row(
            rows,
            boundary_axis="state_specific_direction",
            controlled_question="Does replacing the natural update by the other trajectory's update preserve local progress?",
            condition=f"{budget} budget",
            metric="update_grad_inner",
            ratio_label="other_update/own_update",
            ratio=float(rec["mean_signed_other_over_own"]),
            ci95=f"[{fmt(rec['signed_ratio_ci95_low'])}, {fmt(rec['signed_ratio_ci95_high'])}]",
            direction="own-update-favorable" if rec["signed_ratio_ci95_high"] < 1.0 else "mixed_or_uncertain",
            interpretation="The optimizer's natural update vector is locally consequential, but the effect is still one-step and budget-dependent.",
            source="results/e11_natural_update_swap_probe/natural_update_swap_summary.csv",
        )

    boundary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary.to_csv(OUTPUT_DIR / "mechanism_boundary_map.csv", index=False)

    muon_or_flat_favorable = boundary[
        boundary["direction"].isin(["Muon-favorable", "flat/polar-favorable", "own-update-favorable"])
    ].copy()
    adam_or_gd_favorable = boundary[
        boundary["direction"].isin(["Adam/GD-favorable", "GD-spectrum-favorable", "weakly Adam/GD-favorable"])
    ].copy()
    mixed = boundary[boundary["direction"].eq("mixed_or_uncertain")].copy()

    text = f"""# E11 Mechanism Boundary Map

This generated note summarizes when Muon's update-spectrum shaping does or does not translate into one-step progress. It uses only controlled summaries already generated in the current experiment suite.

## Short Answer

The mechanism boundary is not "higher rank is better." The current evidence is:

1. Muon reliably changes the update spectrum.
2. One-step progress follows the positive descent proxy `<G,D>`, where `D=W-W^+`.
3. Whether the flat/polar update spectrum improves `<G,D>` depends on task family, layer/width, update-size control, and norm budget.

## Boundary Evidence

{markdown_table(boundary, ["boundary_axis", "condition", "metric", "ratio_label", "ratio", "ci95", "direction", "controlled_question"])}

## Muon / Flat-Polar Favorable Cases

{markdown_table(muon_or_flat_favorable, ["boundary_axis", "condition", "ratio_label", "ratio", "ci95", "direction", "interpretation"])}

## Adam / GD-Spectrum Favorable Cases

{markdown_table(adam_or_gd_favorable, ["boundary_axis", "condition", "ratio_label", "ratio", "ci95", "direction", "interpretation"])}

## Mixed Or Uncertain Cases

{markdown_table(mixed, ["boundary_axis", "condition", "ratio_label", "ratio", "ci95", "direction", "interpretation"])}

## Interpretation For The Research Question

Muon's robust cross-task effect is an update-spectrum intervention. The current boundary map says this intervention helps local progress when the task/layer/norm geometry makes a flat/polar positive descent update align well with the gradient. It hurts or becomes ambiguous when the same flat allocation spends update budget in directions that do not improve `<G,D>`.

The cleanest mechanistic statement is therefore:

> Muon is not simply "better because rank is higher"; it applies a flat/polar update-spectrum bias, and that bias is useful only when the local gradient geometry rewards operator-norm-like spectral spreading.

## Sources

- [mechanism boundary map](../results/e11_mechanism_boundary/mechanism_boundary_map.csv)
- [equal-update first-order pair summary](../results/e11_equal_update/first_order_pair_summary.csv)
- [target-update sweep summary](../results/e11_target_update_sweep/target_pair_summary.csv)
- [MLP per-layer control summary](../results/e11_mlp_per_layer_control/pair_summary.csv)
- [spectral allocation probe summary](../results/e11_spectral_allocation_probe/spectral_allocation_summary.csv)
- [natural update swap summary](../results/e11_natural_update_swap_probe/natural_update_swap_summary.csv)
"""
    write_markdown(DISCUSSION_PATH, text)
    print(f"saved mechanism boundary map to {OUTPUT_DIR}")
    print(f"saved mechanism boundary discussion to {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
