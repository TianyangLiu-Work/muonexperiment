from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
SELECTION_DIR = RESULT_ROOT / "validation_selection"
OUTPUT_DIR = RESULT_ROOT / "interim_validation_audit"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_interim_audit.md")


def _clean_metric(value: object) -> float | None:
    if pd.isna(value) or value == "":
        return None
    return float(value)


def _fmt(value: object) -> str:
    if pd.isna(value) or value == "":
        return ""
    if isinstance(value, float):
        return fmt(value)
    return str(value)


def build_completed_leaderboard(run_registry: pd.DataFrame) -> pd.DataFrame:
    complete = run_registry[run_registry["validation_status"].eq("complete")].copy()
    if complete.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "array_index",
                "setting_id",
                "recipe_family",
                "recipe_name",
                "few_balanced_accuracy",
                "few_balanced_accuracy_ci95_low",
                "few_balanced_accuracy_ci95_high",
                "all_balanced_accuracy",
                "all_balanced_accuracy_ci95_low",
                "all_balanced_accuracy_ci95_high",
                "selection_allowed",
            ]
        )
    complete = complete.sort_values(
        ["few_balanced_accuracy", "all_balanced_accuracy", "setting_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    complete.insert(0, "rank", complete.index + 1)
    complete["selection_allowed"] = "no_partial_grid_only"
    return complete[
        [
            "rank",
            "array_index",
            "setting_id",
            "recipe_family",
            "recipe_name",
            "few_balanced_accuracy",
            "few_balanced_accuracy_ci95_low",
            "few_balanced_accuracy_ci95_high",
            "all_balanced_accuracy",
            "all_balanced_accuracy_ci95_low",
            "all_balanced_accuracy_ci95_high",
            "selection_allowed",
        ]
    ]


def build_family_progress(run_registry: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    global_best_low = None
    if not leaderboard.empty:
        global_best_low = _clean_metric(leaderboard.iloc[0]["few_balanced_accuracy_ci95_low"])

    rows: list[dict[str, object]] = []
    for family, family_rows in run_registry.groupby("recipe_family", sort=True):
        expected = int(len(family_rows))
        complete = family_rows[family_rows["validation_status"].eq("complete")].copy()
        complete_count = int(len(complete))
        if complete.empty:
            rows.append(
                {
                    "recipe_family": family,
                    "expected_settings": expected,
                    "complete_settings": complete_count,
                    "complete_fraction": complete_count / expected,
                    "family_status": "not_started",
                    "best_completed_setting_id": "",
                    "best_completed_recipe_name": "",
                    "best_completed_few_balanced_accuracy": "",
                    "best_completed_few_ci95_low": "",
                    "best_completed_few_ci95_high": "",
                    "best_completed_all_balanced_accuracy": "",
                    "ci_relation_to_current_global_best": "not_observed",
                    "selection_status": "not_allowed_partial_family",
                }
            )
            continue
        complete = complete.sort_values(
            ["few_balanced_accuracy", "all_balanced_accuracy", "setting_id"],
            ascending=[False, False, True],
        )
        best = complete.iloc[0]
        best_high = _clean_metric(best["few_balanced_accuracy_ci95_high"])
        if global_best_low is None or best_high is None:
            ci_relation = "not_evaluable"
        elif global_best_low > best_high:
            ci_relation = "current_global_best_ci_low_above_family_best_ci_high"
        else:
            ci_relation = "few_ci_intervals_overlap_or_family_is_global_best"
        rows.append(
            {
                "recipe_family": family,
                "expected_settings": expected,
                "complete_settings": complete_count,
                "complete_fraction": complete_count / expected,
                "family_status": "complete" if complete_count == expected else "partial",
                "best_completed_setting_id": best["setting_id"],
                "best_completed_recipe_name": best["recipe_name"],
                "best_completed_few_balanced_accuracy": best["few_balanced_accuracy"],
                "best_completed_few_ci95_low": best["few_balanced_accuracy_ci95_low"],
                "best_completed_few_ci95_high": best["few_balanced_accuracy_ci95_high"],
                "best_completed_all_balanced_accuracy": best["all_balanced_accuracy"],
                "ci_relation_to_current_global_best": ci_relation,
                "selection_status": "eligible_family_complete" if complete_count == expected else "not_allowed_partial_family",
            }
        )
    return pd.DataFrame(rows)


def build_occupancy_interim_summary(run_registry: pd.DataFrame) -> pd.DataFrame:
    complete = run_registry[run_registry["occupancy_status"].eq("complete")].copy()
    rows: list[dict[str, object]] = []
    for family, family_rows in run_registry.groupby("recipe_family", sort=True):
        family_complete = complete[complete["recipe_family"].eq(family)]
        rows.append(
            {
                "recipe_family": family,
                "expected_settings": int(len(family_rows)),
                "occupancy_complete_settings": int(len(family_complete)),
                "occupancy_probe_rows": int(family_complete["occupancy_probe_rows"].fillna(0).sum()),
                "mean_batch_few_fraction": family_complete["mean_batch_few_fraction"].mean()
                if not family_complete.empty
                else "",
                "mean_tail_probe_loss": family_complete["mean_tail_probe_loss"].mean()
                if not family_complete.empty
                else "",
                "mean_ns_tail_output_drift_sq_ratio_vs_fro": family_complete[
                    "mean_ns_tail_output_drift_sq_ratio_vs_fro"
                ].mean()
                if not family_complete.empty
                else "",
                "occupancy_claim_status": "not_ready_full_grid_required"
                if len(family_complete) < len(family_rows)
                else "family_occupancy_complete",
            }
        )
    return pd.DataFrame(rows)


def build_partial_grid_guardrail(
    run_registry: pd.DataFrame,
    family_progress: pd.DataFrame,
    gate_report: pd.DataFrame,
) -> pd.DataFrame:
    gate_status = gate_report.set_index("gate_id")["status"].astype(str).to_dict()
    complete = run_registry[run_registry["validation_status"].eq("complete")].copy()
    completed_count = int(len(complete))
    total = int(len(run_registry))
    completed_indices = sorted(complete["array_index"].astype(int).tolist())
    expected_prefix = list(range(completed_count))
    prefix_is_contiguous = completed_indices == expected_prefix
    missing_indices = sorted(
        set(run_registry["array_index"].astype(int).tolist()).difference(completed_indices)
    )
    completed_span = (
        f"{completed_indices[0]}..{completed_indices[-1]}" if completed_indices else "none"
    )
    next_missing = str(missing_indices[0]) if missing_indices else "none"
    observed_families = int(complete["recipe_family"].nunique()) if not complete.empty else 0
    complete_families = int(family_progress["family_status"].eq("complete").sum())
    total_families = int(run_registry["recipe_family"].nunique())
    occupancy_paired = complete[
        complete["occupancy_status"].eq("complete")
        & complete["occupancy_probe_rows"].fillna(0).astype(float).gt(0)
    ]
    min_probe_rows = (
        int(occupancy_paired["occupancy_probe_rows"].astype(float).min())
        if not occupancy_paired.empty
        else 0
    )
    missing_count = total - completed_count
    return pd.DataFrame(
        [
            {
                "guard_id": "IPG-1-completed-prefix-auditable",
                "status": "pass" if prefix_is_contiguous else "not_ready",
                "evidence": (
                    f"completed array indices span {completed_span}; next missing array index {next_missing}"
                ),
                "claim_authority": "progress accounting only",
                "blocked_action": "cherry-picking non-contiguous validation settings",
            },
            {
                "guard_id": "IPG-2-family-coverage-incomplete",
                "status": "ready" if complete_families == total_families else "not_ready",
                "evidence": (
                    f"{observed_families}/{total_families} recipe families observed; "
                    f"{complete_families}/{total_families} recipe families complete"
                ),
                "claim_authority": "within-family progress only until all families complete",
                "blocked_action": "cross-family optimizer selection claim",
            },
            {
                "guard_id": "IPG-3-occupancy-paired-with-validation",
                "status": "pass" if len(occupancy_paired) == completed_count else "not_ready",
                "evidence": (
                    f"{len(occupancy_paired)}/{completed_count} completed settings have occupancy traces "
                    f"with positive probe rows; min probe rows {min_probe_rows}"
                ),
                "claim_authority": "paired occupancy progress readout",
                "blocked_action": "trajectory occupancy claim before full grid",
            },
            {
                "guard_id": "IPG-4-missing-work-blocks-final-readout",
                "status": "ready" if missing_count == 0 else "not_ready",
                "evidence": f"{missing_count}/{total} settings still lack validation summaries",
                "claim_authority": "no final-performance benchmark result",
                "blocked_action": "final seed launch or benchmark result wording",
            },
            {
                "guard_id": "IPG-5-final-seed-quarantine-mirrored",
                "status": gate_status.get("TVS-3-final-seed-quarantine", "missing"),
                "evidence": (
                    f"TVS-3-final-seed-quarantine="
                    f"{gate_status.get('TVS-3-final-seed-quarantine', 'missing')}"
                ),
                "claim_authority": "final seeds remain untouched",
                "blocked_action": "retuned final seed claim",
            },
        ]
    )


def build_claim_boundary_gates(
    run_registry: pd.DataFrame,
    family_progress: pd.DataFrame,
    gate_report: pd.DataFrame,
) -> pd.DataFrame:
    gate_status = gate_report.set_index("gate_id")["status"].astype(str).to_dict()
    completed = int(run_registry["validation_status"].eq("complete").sum())
    occupancy = int(run_registry["occupancy_status"].eq("complete").sum())
    eligible_families = int(family_progress["selection_status"].eq("eligible_family_complete").sum())
    return pd.DataFrame(
        [
            {
                "gate_id": "IVA-1-validation-readout-scope",
                "status": "pass",
                "evidence": "reads validation_selection/run_registry.csv and validation_tuning summaries only",
                "allowed_wording": "interim validation progress readout",
                "blocked_wording": "final performance benchmark result",
            },
            {
                "gate_id": "IVA-2-partial-grid-blocks-selection",
                "status": "not_ready" if completed < len(run_registry) else "ready",
                "evidence": f"{completed}/{len(run_registry)} validation settings complete",
                "allowed_wording": "completed-setting leaderboard with no selection authority",
                "blocked_wording": "selection from incomplete recipe families",
            },
            {
                "gate_id": "IVA-3-familywise-selection-authority",
                "status": "not_ready" if eligible_families < run_registry["recipe_family"].nunique() else "ready",
                "evidence": f"{eligible_families}/{run_registry['recipe_family'].nunique()} recipe families complete",
                "allowed_wording": "family complete/partial status",
                "blocked_wording": "cross-family optimizer claim",
            },
            {
                "gate_id": "IVA-4-occupancy-coverage",
                "status": "not_ready" if occupancy < len(run_registry) else "ready",
                "evidence": f"{occupancy}/{len(run_registry)} occupancy traces complete",
                "allowed_wording": "interim occupancy coverage",
                "blocked_wording": "trajectory state-distribution claim",
            },
            {
                "gate_id": "IVA-5-final-seed-quarantine",
                "status": gate_status.get("TVS-3-final-seed-quarantine", "missing"),
                "evidence": "mirrors TVS-3-final-seed-quarantine",
                "allowed_wording": "final seeds remain untouched",
                "blocked_wording": "final seed result or retuned final claim",
            },
        ]
    )


def write_discussion(
    leaderboard: pd.DataFrame,
    family_progress: pd.DataFrame,
    occupancy_summary: pd.DataFrame,
    partial_guardrail: pd.DataFrame,
    claim_gates: pd.DataFrame,
) -> None:
    total = int(family_progress["expected_settings"].sum())
    completed = int(family_progress["complete_settings"].sum())
    best = leaderboard.iloc[0] if not leaderboard.empty else None
    best_sentence = (
        f"The current completed-setting leader is `{best['setting_id']}` with few balanced accuracy "
        f"`{fmt(best['few_balanced_accuracy'])}` and all balanced accuracy `{fmt(best['all_balanced_accuracy'])}`."
        if best is not None
        else "No validation setting has completed yet."
    )
    display_leaderboard = leaderboard.head(12).copy()
    for column in display_leaderboard.columns:
        if pd.api.types.is_float_dtype(display_leaderboard[column]):
            display_leaderboard[column] = display_leaderboard[column].map(_fmt)
    display_family = family_progress.copy()
    for column in display_family.columns:
        if pd.api.types.is_float_dtype(display_family[column]):
            display_family[column] = display_family[column].map(_fmt)
    display_occupancy = occupancy_summary.copy()
    for column in display_occupancy.columns:
        if pd.api.types.is_float_dtype(display_occupancy[column]):
            display_occupancy[column] = display_occupancy[column].map(_fmt)

    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Interim Validation Audit

This generated audit is an interim, no-peeking progress readout for the tuned validation grid. It reads only `validation_tuning` outputs and the selection audit tables. It does not authorize final seed runs, recipe-family selection for incomplete families, or benchmark-level optimizer wording.

Current progress: `{completed}/{total}` validation settings complete. {best_sentence}

## Claim Boundary Gates

{markdown_table(claim_gates, ["gate_id", "status", "evidence", "allowed_wording", "blocked_wording"])}

## Partial Grid Guardrail

{markdown_table(partial_guardrail, ["guard_id", "status", "evidence", "claim_authority", "blocked_action"])}

## Family Progress

{markdown_table(display_family, ["recipe_family", "expected_settings", "complete_settings", "complete_fraction", "family_status", "best_completed_setting_id", "best_completed_few_balanced_accuracy", "best_completed_all_balanced_accuracy", "ci_relation_to_current_global_best", "selection_status"])}

## Completed-Setting Leaderboard

{markdown_table(display_leaderboard, ["rank", "array_index", "setting_id", "recipe_family", "few_balanced_accuracy", "few_balanced_accuracy_ci95_low", "few_balanced_accuracy_ci95_high", "all_balanced_accuracy", "selection_allowed"])}

## Interim Occupancy Summary

{markdown_table(display_occupancy, ["recipe_family", "expected_settings", "occupancy_complete_settings", "occupancy_probe_rows", "mean_batch_few_fraction", "mean_tail_probe_loss", "mean_ns_tail_output_drift_sq_ratio_vs_fro", "occupancy_claim_status"])}

## Operating Rule

This audit can be cited only as progress accounting and reviewer-facing no-peeking evidence. The final claim seed set `20..29` remains blocked until `TVS-1`, `TVS-2`, and `TVS-5` pass in the selection audit.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    run_registry = pd.read_csv(SELECTION_DIR / "run_registry.csv")
    gate_report = pd.read_csv(SELECTION_DIR / "gate_report.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = build_completed_leaderboard(run_registry)
    family_progress = build_family_progress(run_registry, leaderboard)
    occupancy_summary = build_occupancy_interim_summary(run_registry)
    partial_guardrail = build_partial_grid_guardrail(run_registry, family_progress, gate_report)
    claim_gates = build_claim_boundary_gates(run_registry, family_progress, gate_report)
    leaderboard.to_csv(OUTPUT_DIR / "completed_setting_leaderboard.csv", index=False)
    family_progress.to_csv(OUTPUT_DIR / "family_progress.csv", index=False)
    occupancy_summary.to_csv(OUTPUT_DIR / "occupancy_interim_summary.csv", index=False)
    partial_guardrail.to_csv(OUTPUT_DIR / "partial_grid_guardrail.csv", index=False)
    claim_gates.to_csv(OUTPUT_DIR / "claim_boundary_gates.csv", index=False)
    config = {
        "run_registry": (SELECTION_DIR / "run_registry.csv").as_posix(),
        "gate_report": (SELECTION_DIR / "gate_report.csv").as_posix(),
        "partial_grid_guardrail": (OUTPUT_DIR / "partial_grid_guardrail.csv").as_posix(),
        "scope": "validation_tuning_only",
        "final_seed_set": "20..29",
        "final_seed_status": "not_touched",
        "claim_authority": "progress_accounting_only",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_discussion(leaderboard, family_progress, occupancy_summary, partial_guardrail, claim_gates)
    print(f"saved tuned benchmark interim audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
