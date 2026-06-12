from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


OUTPUT_PATH = Path("discussion/e11_boundary_predictor_audit.md")


def main() -> None:
    summary = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_summary.csv")
    uncertainty = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_uncertainty.csv")
    rows = pd.read_csv("results/e11_boundary_predictor/boundary_predictor_rows.csv")
    target = "update_grad_inner_muon_higher"
    focus = summary[summary["target"].eq(target)].copy()

    aggregate = (
        focus.groupby(["feature_set", "evaluation"], as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_auc=("auc", "mean"),
            mean_brier=("brier", "mean"),
            mean_baseline_brier=("baseline_brier", "mean"),
        )
        .sort_values(["feature_set", "evaluation"])
    )
    wide = aggregate.pivot(index="feature_set", columns="evaluation", values="mean_balanced_accuracy").reset_index()
    wide["in_sample_minus_leave_setting_out"] = wide["in_sample"] - wide["leave_setting_out"]
    best_leave = wide.sort_values("leave_setting_out", ascending=False).iloc[0]
    uncertainty_focus = uncertainty[uncertainty["target"].eq(target)].copy()
    best_uncertainty = uncertainty_focus.sort_values("mean_balanced_accuracy_chance_filled", ascending=False).iloc[0]

    leave = focus[focus["evaluation"].eq("leave_setting_out")].copy()
    leave["balanced_accuracy_filled"] = leave["balanced_accuracy"].fillna(0.5)
    per_setting = (
        leave.groupby("held_out_setting", as_index=False)
        .agg(
            best_balanced_accuracy=("balanced_accuracy_filled", "max"),
            mean_positive_rate=("test_positive_rate", "mean"),
            max_auc=("auc", "max"),
            min_brier=("brier", "min"),
            baseline_brier=("baseline_brier", "mean"),
        )
        .sort_values("held_out_setting")
    )
    per_setting["class_balance_issue"] = per_setting["mean_positive_rate"].map(
        lambda value: "degenerate" if value in {0.0, 1.0} else ("highly_imbalanced" if min(value, 1.0 - value) < 0.1 else "moderate")
    )

    feature_set_failures = leave[
        leave["feature_set"].eq(str(best_leave["feature_set"]))
    ][["held_out_setting", "test_rows", "test_positive_rate", "auc", "balanced_accuracy", "brier", "baseline_brier"]].copy()
    feature_set_failures["failure_mode"] = feature_set_failures.apply(
        lambda row: "degenerate target" if row["test_positive_rate"] in {0.0, 1.0} else (
            "near chance" if pd.isna(row["balanced_accuracy"]) or row["balanced_accuracy"] <= 0.55 else "partial success"
        ),
        axis=1,
    )

    target_distribution = (
        rows.groupby("base_setting", as_index=False)
        .agg(
            rows=("update_grad_inner_muon_higher", "size"),
            positive_rate=("update_grad_inner_muon_higher", "mean"),
            mean_log10_target=("log10_target_relative_update_norm", "mean"),
            mean_delta_st_update_frac=("delta_stUpdateFrac", "mean"),
            mean_delta_update_flatness=("delta_update_flatness", "mean"),
        )
        .sort_values("base_setting")
    )

    readiness = pd.DataFrame(
        [
            {
                "criterion": "In-sample fit",
                "status": "passes only as exploratory fit",
                "evidence": (
                    f"Best in-sample balanced accuracy is {fmt(wide['in_sample'].max())}; "
                    f"best leave-setting-out is {fmt(best_leave['leave_setting_out'])}."
                ),
                "paper_implication": "Do not use in-sample performance as predictive evidence.",
            },
            {
                "criterion": "Held-out generalization",
                "status": "not paper-ready",
                "evidence": (
                    f"Best feature set `{best_leave['feature_set']}` has mean leave-setting-out balanced accuracy "
                    f"{fmt(best_leave['leave_setting_out'])} when degenerate settings are skipped. "
                    f"Chance-filled best `{best_uncertainty['feature_set']}` is "
                    f"{fmt(best_uncertainty['mean_balanced_accuracy_chance_filled'])} "
                    f"CI=[{fmt(best_uncertainty['balanced_accuracy_ci95_low'])}, "
                    f"{fmt(best_uncertainty['balanced_accuracy_ci95_high'])}]."
                ),
                "paper_implication": "Keep the boundary map descriptive unless a new held-out benchmark validates a rule.",
            },
            {
                "criterion": "Class balance",
                "status": "problematic",
                "evidence": "Some held-out settings have zero or near-zero positive rate, making AUC/balanced accuracy unstable or undefined.",
                "paper_implication": "A publishable predictor needs settings with non-degenerate positive/negative examples.",
            },
            {
                "criterion": "Causal interpretation",
                "status": "unsupported",
                "evidence": "Feature sets mix state geometry, update spectrum, and family labels.",
                "paper_implication": "Treat predictor features as descriptive correlates, not mechanism proof.",
            },
        ]
    )

    next_design = pd.DataFrame(
        [
            {
                "requirement": "Pre-register predictor features",
                "why": "Avoid choosing features after seeing all task families.",
            },
            {
                "requirement": "Use a genuinely new held-out task or architecture",
                "why": "Leave-setting-out inside the current six settings is too small and imbalanced.",
            },
            {
                "requirement": "Ensure both Muon-win and Adam-win examples within the held-out domain",
                "why": "Degenerate labels make predictor metrics undefined or misleading.",
            },
            {
                "requirement": "Report calibration as well as ranking",
                "why": "Several models have poor Brier scores despite reasonable in-sample ranking.",
            },
        ]
    )

    text = f"""# E11 Boundary Predictor Audit

This generated audit explains why the current boundary predictor is not yet a publishable predictive law. It is stricter than `e11_boundary_predictor.md`: the goal here is to identify failure modes and the minimum standard for a future held-out test.

## Readiness Summary

{markdown_table(readiness, ["criterion", "status", "evidence", "paper_implication"])}

## In-Sample Versus Leave-Setting-Out Gap

{markdown_table(wide, ["feature_set", "in_sample", "leave_setting_out", "in_sample_minus_leave_setting_out"])}

## Leave-Setting-Out Uncertainty

{markdown_table(uncertainty_focus, ["feature_set", "held_out_settings", "mean_balanced_accuracy_chance_filled", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high", "balanced_accuracy_ci95_above_chance", "mean_brier_improvement_over_base_rate", "brier_improvement_ci95_low", "brier_improvement_ci95_high", "brier_improvement_ci95_above_zero"])}

## Held-Out Setting Difficulty

{markdown_table(per_setting, ["held_out_setting", "best_balanced_accuracy", "mean_positive_rate", "max_auc", "min_brier", "baseline_brier", "class_balance_issue"])}

## Best Feature-Set Failure Modes

Best leave-setting-out feature set: `{best_leave['feature_set']}`.

{markdown_table(feature_set_failures, ["held_out_setting", "test_rows", "test_positive_rate", "auc", "balanced_accuracy", "brier", "baseline_brier", "failure_mode"])}

## Target Distribution Behind The Failures

{markdown_table(target_distribution, ["base_setting", "rows", "positive_rate", "mean_log10_target", "mean_delta_st_update_frac", "mean_delta_update_flatness"])}

## Minimum Standard For The Next Predictor

{markdown_table(next_design, ["requirement", "why"])}

## Paper Use

The current result should be described as:

> The boundary map is quantitatively documented but not yet predictive out of sample.

It should not be described as:

> We can predict when Muon wins from local rank/spectrum features.

## Sources

- [boundary predictor summary](../results/e11_boundary_predictor/boundary_predictor_summary.csv)
- [boundary predictor rows](../results/e11_boundary_predictor/boundary_predictor_rows.csv)
- [boundary predictor discussion](e11_boundary_predictor.md)
"""
    write_markdown(OUTPUT_PATH, text)
    print(f"saved boundary predictor audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
