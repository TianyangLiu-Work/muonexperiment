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
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_selection.md")
REGISTRY_PATH = RESULT_ROOT / "settings_registry.csv"


def _empty_metric_payload() -> dict[str, float | None]:
    return {
        "few_balanced_accuracy": None,
        "few_balanced_accuracy_ci95_low": None,
        "few_balanced_accuracy_ci95_high": None,
        "all_balanced_accuracy": None,
        "all_balanced_accuracy_ci95_low": None,
        "all_balanced_accuracy_ci95_high": None,
        "many_balanced_accuracy": None,
        "medium_balanced_accuracy": None,
    }


def read_metric_payload(setting: pd.Series) -> dict[str, float | None]:
    summary_path = Path(str(setting["planned_output_dir"])) / "summary.csv"
    if not summary_path.exists():
        return _empty_metric_payload()
    summary = pd.read_csv(summary_path)
    expected_groups = {"many", "medium", "few", "all"}
    if set(summary["frequency_group"]) != expected_groups:
        raise AssertionError(f"validation summary has wrong frequency groups: {summary_path}")
    by_group = summary.set_index("frequency_group")
    return {
        "few_balanced_accuracy": float(by_group.loc["few", "mean_balanced_accuracy"]),
        "few_balanced_accuracy_ci95_low": float(by_group.loc["few", "balanced_accuracy_ci95_low"]),
        "few_balanced_accuracy_ci95_high": float(by_group.loc["few", "balanced_accuracy_ci95_high"]),
        "all_balanced_accuracy": float(by_group.loc["all", "mean_balanced_accuracy"]),
        "all_balanced_accuracy_ci95_low": float(by_group.loc["all", "balanced_accuracy_ci95_low"]),
        "all_balanced_accuracy_ci95_high": float(by_group.loc["all", "balanced_accuracy_ci95_high"]),
        "many_balanced_accuracy": float(by_group.loc["many", "mean_balanced_accuracy"]),
        "medium_balanced_accuracy": float(by_group.loc["medium", "mean_balanced_accuracy"]),
    }


def build_run_registry(settings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in settings.itertuples(index=False):
        setting_dict = setting._asdict()
        output_dir = Path(str(setting_dict["planned_output_dir"]))
        summary_path = output_dir / "summary.csv"
        metric_payload = read_metric_payload(pd.Series(setting_dict))
        rows.append(
            {
                "array_index": int(setting_dict["array_index"]),
                "setting_id": str(setting_dict["setting_id"]),
                "recipe_family": str(setting_dict["recipe_family"]),
                "recipe_name": str(setting_dict["recipe_name"]),
                "optimizer": str(setting_dict["optimizer"]),
                "seed_set": str(setting_dict["seed_set"]),
                "validation_status": "complete" if summary_path.exists() else "missing_summary",
                "summary_path": summary_path.as_posix(),
                **metric_payload,
            }
        )
    return pd.DataFrame(rows)


def build_family_selection(run_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, family_rows in run_registry.groupby("recipe_family", sort=True):
        complete = family_rows[family_rows["validation_status"].eq("complete")].copy()
        expected = int(len(family_rows))
        if len(complete) != expected:
            rows.append(
                {
                    "recipe_family": family,
                    "selection_status": "not_ready",
                    "expected_settings": expected,
                    "complete_settings": int(len(complete)),
                    "selected_setting_id": "",
                    "selected_recipe_name": "",
                    "primary_few_balanced_accuracy": "",
                    "tie_break_all_balanced_accuracy": "",
                    "selection_rule": "wait for every registered validation setting in the family",
                }
            )
            continue
        complete = complete.sort_values(
            ["few_balanced_accuracy", "all_balanced_accuracy", "setting_id"],
            ascending=[False, False, True],
        )
        winner = complete.iloc[0]
        rows.append(
            {
                "recipe_family": family,
                "selection_status": "selected",
                "expected_settings": expected,
                "complete_settings": int(len(complete)),
                "selected_setting_id": winner["setting_id"],
                "selected_recipe_name": winner["recipe_name"],
                "primary_few_balanced_accuracy": winner["few_balanced_accuracy"],
                "tie_break_all_balanced_accuracy": winner["all_balanced_accuracy"],
                "selection_rule": "maximize validation few balanced accuracy, tie-break by all balanced accuracy",
            }
        )
    return pd.DataFrame(rows).sort_values("recipe_family").reset_index(drop=True)


def build_final_plan(family_selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in family_selection.itertuples(index=False):
        selected = str(row.selection_status) == "selected"
        rows.append(
            {
                "recipe_family": row.recipe_family,
                "final_status": "ready_for_final_run" if selected else "not_ready",
                "selected_setting_id": row.selected_setting_id if selected else "",
                "selected_recipe_name": row.selected_recipe_name if selected else "",
                "final_seed_set": "20..29" if selected else "",
                "planned_output_dir": (
                    f"results/e11_cifar100_resnet_lt_tuned_benchmark/final_claim/{row.recipe_family}"
                    if selected
                    else ""
                ),
                "tuning_allowed": "no",
            }
        )
    return pd.DataFrame(rows)


def build_gate_report(
    run_registry: pd.DataFrame,
    family_selection: pd.DataFrame,
    final_plan: pd.DataFrame,
) -> pd.DataFrame:
    all_validation_complete = bool(run_registry["validation_status"].eq("complete").all())
    all_families_selected = bool(family_selection["selection_status"].eq("selected").all())
    final_outputs = sorted(
        path
        for path in Path("results/e11_cifar100_resnet_lt_tuned_benchmark").glob("final*")
        if path.name != "final_power_audit"
    )
    return pd.DataFrame(
        [
            {
                "gate_id": "TVS-1-validation-grid-complete",
                "status": "pass" if all_validation_complete else "not_ready",
                "evidence": f"{int(run_registry['validation_status'].eq('complete').sum())}/{len(run_registry)} validation settings complete",
                "blocks_final_claim": "yes",
            },
            {
                "gate_id": "TVS-2-family-selection",
                "status": "pass" if all_families_selected else "not_ready",
                "evidence": f"{int(family_selection['selection_status'].eq('selected').sum())}/{len(family_selection)} recipe families selected",
                "blocks_final_claim": "yes",
            },
            {
                "gate_id": "TVS-3-final-seed-quarantine",
                "status": "pass" if not final_outputs else "fail",
                "evidence": "no final_claim outputs exist before validation selection"
                if not final_outputs
                else "; ".join(path.as_posix() for path in final_outputs),
                "blocks_final_claim": "yes",
            },
            {
                "gate_id": "TVS-4-final-run-plan",
                "status": "ready" if bool(final_plan["final_status"].eq("ready_for_final_run").all()) else "not_ready",
                "evidence": "final seed set 20..29 assigned only after validation family selection",
                "blocks_final_claim": "yes",
            },
        ]
    )


def _format_selection(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("primary_few_balanced_accuracy", "tie_break_all_balanced_accuracy"):
        display[column] = display[column].map(lambda value: "" if value == "" else fmt(value))
    return display


def write_discussion(
    run_registry: pd.DataFrame,
    family_selection: pd.DataFrame,
    final_plan: pd.DataFrame,
    gate_report: pd.DataFrame,
) -> None:
    completed = int(run_registry["validation_status"].eq("complete").sum())
    total = int(len(run_registry))
    status = "ready_for_final_selection" if completed == total else "not_ready"
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Selection

This generated selection report is the no-peeking bridge between the tuned benchmark validation grid and the untouched final claim seeds. It reads only validation summaries under `results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning/*`.

Current status: `{status}` with `{completed}/{total}` validation settings complete.

## Selection Rule

For each recipe family, select the validation setting with maximum few-class balanced accuracy. Ties are broken by all-class balanced accuracy and then by setting id. The final claim seed set is always `20..29`, and no final output may exist before validation family selection is complete.

## Gate Report

{markdown_table(gate_report, ["gate_id", "status", "evidence", "blocks_final_claim"])}

## Family Selection

{markdown_table(_format_selection(family_selection), ["recipe_family", "selection_status", "expected_settings", "complete_settings", "selected_setting_id", "selected_recipe_name", "primary_few_balanced_accuracy", "tie_break_all_balanced_accuracy", "selection_rule"])}

## Final Claim Plan

{markdown_table(final_plan, ["recipe_family", "final_status", "selected_setting_id", "selected_recipe_name", "final_seed_set", "planned_output_dir", "tuning_allowed"])}

## Claim Boundary

The current result is not a final-performance benchmark result. It is a selection audit. A practical optimizer claim remains blocked until the validation grid is complete, one recipe per family is selected by the frozen rule, and the final seed set `20..29` is run pairwise without tuning.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"missing tuned benchmark settings registry: {REGISTRY_PATH}")
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    settings = pd.read_csv(REGISTRY_PATH)
    run_registry = build_run_registry(settings)
    family_selection = build_family_selection(run_registry)
    final_plan = build_final_plan(family_selection)
    gate_report = build_gate_report(run_registry, family_selection, final_plan)
    run_registry.to_csv(SELECTION_DIR / "run_registry.csv", index=False)
    family_selection.to_csv(SELECTION_DIR / "family_selection.csv", index=False)
    final_plan.to_csv(SELECTION_DIR / "final_claim_plan.csv", index=False)
    gate_report.to_csv(SELECTION_DIR / "gate_report.csv", index=False)
    config = {
        "settings_registry": REGISTRY_PATH.as_posix(),
        "selection_rule": "maximize few balanced accuracy, tie-break by all balanced accuracy then setting id",
        "final_seed_set": "20..29",
        "validation_output_prefix": "results/e11_cifar100_resnet_lt_tuned_benchmark/validation_tuning",
    }
    (SELECTION_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_discussion(run_registry, family_selection, final_plan, gate_report)
    print(f"saved tuned benchmark validation selection to {SELECTION_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
