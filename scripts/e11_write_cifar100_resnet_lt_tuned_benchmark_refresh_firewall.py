from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
SELECTION_DIR = RESULT_ROOT / "validation_selection"
INTERIM_DIR = RESULT_ROOT / "interim_validation_audit"
LEAKAGE_DIR = RESULT_ROOT / "validation_leakage_audit"
OUTPUT_DIR = RESULT_ROOT / "validation_refresh_firewall"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_refresh_firewall.md")


def _completed_prefix(run_registry: pd.DataFrame) -> tuple[list[int], bool, str]:
    complete = run_registry[run_registry["validation_status"].eq("complete")].copy()
    completed_indices = sorted(complete["array_index"].astype(int).tolist())
    prefix_is_contiguous = completed_indices == list(range(len(completed_indices)))
    missing_indices = sorted(
        set(run_registry["array_index"].astype(int).tolist()).difference(completed_indices)
    )
    next_missing = str(missing_indices[0]) if missing_indices else "none"
    return completed_indices, prefix_is_contiguous, next_missing


def _status(gate_report: pd.DataFrame, gate_id: str) -> str:
    rows = gate_report[gate_report["gate_id"].eq(gate_id)]
    if rows.empty:
        return "missing"
    return str(rows["status"].iloc[0])


def build_refresh_state(
    run_registry: pd.DataFrame,
    gate_report: pd.DataFrame,
    partial_guardrail: pd.DataFrame,
    leakage_guards: pd.DataFrame,
) -> pd.DataFrame:
    completed_indices, prefix_is_contiguous, next_missing = _completed_prefix(run_registry)
    completed_count = len(completed_indices)
    total = int(len(run_registry))
    occupancy_count = int(run_registry["occupancy_status"].eq("complete").sum())
    complete_by_family = run_registry.groupby("recipe_family")["validation_status"].apply(
        lambda values: bool(values.eq("complete").all())
    )
    completed_families = int(complete_by_family.sum())
    total_families = int(run_registry["recipe_family"].nunique())
    leakage_ok = bool(leakage_guards["status"].astype(str).isin({"pass", "ready"}).all())
    ipg_ok = bool(partial_guardrail["status"].astype(str).isin({"pass", "ready", "not_ready"}).all())
    full_validation_ready = (
        _status(gate_report, "TVS-1-validation-grid-complete") == "pass"
        and _status(gate_report, "TVS-2-family-selection") == "pass"
        and _status(gate_report, "TVS-5-occupancy-logging-complete") == "pass"
    )
    if full_validation_ready and leakage_ok:
        refresh_status = "full_grid_ready_for_final_gate_refresh"
    elif prefix_is_contiguous and leakage_ok and ipg_ok:
        refresh_status = "partial_grid_no_claim_change"
    else:
        refresh_status = "quarantine_refresh_until_audit_repaired"
    completed_span = f"0..{completed_indices[-1]}" if completed_indices else "none"
    return pd.DataFrame(
        [
            {
                "state_id": "VRF-current-validation-prefix",
                "status": refresh_status,
                "completed_validation_settings": completed_count,
                "total_validation_settings": total,
                "completed_occupancy_traces": occupancy_count,
                "completed_recipe_families": completed_families,
                "total_recipe_families": total_families,
                "completed_array_prefix": completed_span,
                "next_missing_array_index": next_missing,
                "prefix_contiguous": "yes" if prefix_is_contiguous else "no",
                "refresh_authority": "progress_accounting_only"
                if not full_validation_ready
                else "gate_refresh_only_before_final_submit",
                "final_seed_authority": "blocked_until_TVS_FEP_TFE_FLA_pass",
            }
        ]
    )


def build_allowed_transition_matrix(
    run_registry: pd.DataFrame,
    gate_report: pd.DataFrame,
    leakage_guards: pd.DataFrame,
) -> pd.DataFrame:
    _, prefix_is_contiguous, next_missing = _completed_prefix(run_registry)
    tvs1 = _status(gate_report, "TVS-1-validation-grid-complete")
    tvs2 = _status(gate_report, "TVS-2-family-selection")
    tvs5 = _status(gate_report, "TVS-5-occupancy-logging-complete")
    leakage_ok = bool(leakage_guards["status"].astype(str).isin({"pass", "ready"}).all())
    full_grid_ready = tvs1 == "pass" and tvs5 == "pass"
    final_gate_ready = full_grid_ready and tvs2 == "pass" and leakage_ok
    return pd.DataFrame(
        [
            {
                "transition_id": "VRF-A1-prefix-result-refresh",
                "status": "pass" if prefix_is_contiguous and next_missing != "none" else "not_ready",
                "trigger": f"new validation summary and occupancy trace for array index {next_missing}",
                "required_commands": "rerun selection, interim audit, leakage audit, and refresh firewall",
                "allowed_output": "updated progress accounting with no selection-authority change",
            },
            {
                "transition_id": "VRF-A2-nonprefix-result-quarantine",
                "status": "not_ready",
                "trigger": "a completed validation row appears beyond the current contiguous prefix",
                "required_commands": "audit Slurm launch history and keep the row out of selection until order is explained",
                "allowed_output": "diagnostic anomaly report only",
            },
            {
                "transition_id": "VRF-A3-full-validation-refresh",
                "status": "ready" if full_grid_ready and leakage_ok else "not_ready",
                "trigger": "TVS-1 and TVS-5 pass with leakage guards still passing",
                "required_commands": "rerun selection, power, variance-prior, final-analysis, final-execution, final-eval, and final-launch audits",
                "allowed_output": "complete-family selection and final-gate refresh, not final results",
            },
            {
                "transition_id": "VRF-A4-final-submit-handoff",
                "status": "ready" if final_gate_ready else "not_ready",
                "trigger": "TVS-1, TVS-2, TVS-5, leakage guards, FEP, TFE, and FLA all pass",
                "required_commands": "submit final_claim jobs only through the final-safe-submit path",
                "allowed_output": "untouched final seed execution under seed set 20..29",
            },
            {
                "transition_id": "VRF-A5-leakage-guard-repair",
                "status": "pass" if leakage_ok else "fail",
                "trigger": "any TLA guard changes from pass/ready",
                "required_commands": "stop refresh, document violation, and open a fresh preregistered protocol if needed",
                "allowed_output": "no stronger claim while guard drift remains",
            },
        ]
    )


def build_forbidden_action_matrix(gate_report: pd.DataFrame) -> pd.DataFrame:
    partial_grid = _status(gate_report, "TVS-1-validation-grid-complete") != "pass"
    return pd.DataFrame(
        [
            {
                "forbidden_id": "VRF-F1-change-selection-objective",
                "status": "active",
                "forbidden_action": "rewrite the validation objective after seeing partial validation metrics",
                "violation_response": "discard current validation/final split and open a fresh preregistered protocol",
            },
            {
                "forbidden_id": "VRF-F2-metric-dependent-launch-order",
                "status": "active",
                "forbidden_action": "change array order, queue priority, or resubmission policy based on interim leaderboard rank",
                "violation_response": "quarantine affected rows from selection and audit launch history",
            },
            {
                "forbidden_id": "VRF-F3-partial-family-selection",
                "status": "active" if partial_grid else "retire_after_full_grid",
                "forbidden_action": "select a cross-family winner or baseline from an incomplete recipe family grid",
                "violation_response": "rerun only progress audits; do not run final_claim jobs",
            },
            {
                "forbidden_id": "VRF-F4-premature-final-submit",
                "status": "active",
                "forbidden_action": "submit final_claim seed jobs before TVS, FEP, TFE, and FLA gates pass",
                "violation_response": "mark final outputs contaminated and require fresh final seeds",
            },
            {
                "forbidden_id": "VRF-F5-validation-leaderboard-performance-claim",
                "status": "active",
                "forbidden_action": "describe validation leaderboard rows as benchmark performance results",
                "violation_response": "downgrade wording to progress accounting and rerun claim-decision audit",
            },
            {
                "forbidden_id": "VRF-F6-in-place-protocol-repair-after-violation",
                "status": "active",
                "forbidden_action": "repair a selection-leakage or final-unblinding violation inside the same validation/final split",
                "violation_response": "register new unspent validation and final splits before renewed performance claims",
            },
        ]
    )


def write_discussion(
    refresh_state: pd.DataFrame,
    allowed_transitions: pd.DataFrame,
    forbidden_actions: pd.DataFrame,
) -> None:
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Validation Refresh Firewall

This generated firewall defines what can happen after each partial validation refresh. It is a sequential optional-stopping firewall: new validation outputs can update progress accounting, but they cannot change the registry, selection objective, array order, family selection, final seed set, or benchmark wording.

## Refresh State

{markdown_table(refresh_state, ["state_id", "status", "completed_validation_settings", "total_validation_settings", "completed_occupancy_traces", "completed_recipe_families", "total_recipe_families", "completed_array_prefix", "next_missing_array_index", "prefix_contiguous", "refresh_authority", "final_seed_authority"])}

## Allowed Transition Matrix

{markdown_table(allowed_transitions, ["transition_id", "status", "trigger", "required_commands", "allowed_output"])}

## Forbidden Action Matrix

{markdown_table(forbidden_actions, ["forbidden_id", "status", "forbidden_action", "violation_response"])}

## Operating Rule

Every validation refresh must preserve progress accounting only until `TVS-1`, `TVS-2`, and `TVS-5` pass and the leakage guards remain pass/ready. Any non-prefix completion, metric-dependent launch change, partial-family selection, premature final submit, or final-seed contamination requires quarantine and a fresh preregistered protocol before renewed benchmark-performance claims.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    run_registry = pd.read_csv(SELECTION_DIR / "run_registry.csv")
    gate_report = pd.read_csv(SELECTION_DIR / "gate_report.csv")
    partial_guardrail = pd.read_csv(INTERIM_DIR / "partial_grid_guardrail.csv")
    leakage_guards = pd.read_csv(LEAKAGE_DIR / "leakage_guard_matrix.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    refresh_state = build_refresh_state(run_registry, gate_report, partial_guardrail, leakage_guards)
    allowed_transitions = build_allowed_transition_matrix(run_registry, gate_report, leakage_guards)
    forbidden_actions = build_forbidden_action_matrix(gate_report)
    refresh_state.to_csv(OUTPUT_DIR / "refresh_state.csv", index=False)
    allowed_transitions.to_csv(OUTPUT_DIR / "allowed_transition_matrix.csv", index=False)
    forbidden_actions.to_csv(OUTPUT_DIR / "forbidden_action_matrix.csv", index=False)
    config = {
        "run_registry": (SELECTION_DIR / "run_registry.csv").as_posix(),
        "gate_report": (SELECTION_DIR / "gate_report.csv").as_posix(),
        "partial_grid_guardrail": (INTERIM_DIR / "partial_grid_guardrail.csv").as_posix(),
        "leakage_guard_matrix": (LEAKAGE_DIR / "leakage_guard_matrix.csv").as_posix(),
        "scope": "sequential_validation_refresh_firewall",
        "claim_authority": "progress_accounting_only_until_TVS_FEP_TFE_FLA_pass",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_discussion(refresh_state, allowed_transitions, forbidden_actions)
    print(f"saved tuned benchmark validation refresh firewall to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
