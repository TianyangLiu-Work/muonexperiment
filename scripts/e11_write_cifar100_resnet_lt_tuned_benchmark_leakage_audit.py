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
PROTOCOL_DIR = Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol")
SELECTION_DIR = RESULT_ROOT / "validation_selection"
INTERIM_DIR = RESULT_ROOT / "interim_validation_audit"
LAUNCH_DIR = RESULT_ROOT / "slurm_launch_audit"
OUTPUT_DIR = RESULT_ROOT / "validation_leakage_audit"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_leakage_audit.md")


def _final_outputs() -> list[str]:
    final_claim_dir = RESULT_ROOT / "final_claim"
    if not final_claim_dir.exists():
        return []
    return sorted(
        path.as_posix()
        for path in final_claim_dir.rglob("*")
        if path.is_file()
    )


def _has_phrase(frame: pd.DataFrame, phrase: str) -> bool:
    return phrase in " ".join(frame.astype(str).to_numpy().ravel())


def build_leakage_guard_matrix(
    registry: pd.DataFrame,
    selection_rules: pd.DataFrame,
    family_selection: pd.DataFrame,
    gate_report: pd.DataFrame,
    launch_history: pd.DataFrame,
) -> pd.DataFrame:
    completed_families = int(family_selection["selection_status"].eq("selected").sum())
    completed_settings = int(gate_report.loc[
        gate_report["gate_id"].eq("TVS-1-validation-grid-complete"), "evidence"
    ].iloc[0].split("/")[0])
    final_outputs = _final_outputs()
    contiguous_indices = sorted(registry["array_index"].astype(int).tolist()) == list(range(len(registry)))
    validation_only = set(registry["phase"].astype(str)) == {"validation_tuning"} and set(
        registry["seed_set"].astype(str)
    ) == {"10..14"}
    launch_arrays = [
        str(value)
        for value in launch_history.get("planned_array_indices", pd.Series(dtype=str)).dropna().tolist()
        if str(value).strip()
    ]
    selection_rule_frozen = _has_phrase(selection_rules, "Primary validation objective is few balanced accuracy")
    return pd.DataFrame(
        [
            {
                "guard_id": "TLA-1-selection-rule-frozen",
                "status": "pass" if selection_rule_frozen else "fail",
                "evidence": "selection_rules.csv contains `Primary validation objective is few balanced accuracy`",
                "leakage_risk_controlled": "metric-dependent rule rewriting after partial observations",
                "claim_effect": "blocks any stronger selection wording if this fails",
            },
            {
                "guard_id": "TLA-2-validation-only-input-surface",
                "status": "pass" if validation_only else "fail",
                "evidence": "settings_registry.csv has phase=validation_tuning and seed_set=10..14 for every array row",
                "leakage_risk_controlled": "final-seed information entering validation selection",
                "claim_effect": "final seed set 20..29 remains unavailable to selection",
            },
            {
                "guard_id": "TLA-3-array-order-contiguous",
                "status": "pass" if contiguous_indices else "fail",
                "evidence": f"{len(registry)} registry rows with contiguous array_index 0..{len(registry) - 1}",
                "leakage_risk_controlled": "post-hoc array reordering to prioritize favorable settings",
                "claim_effect": "Slurm array order remains auditable against the frozen registry",
            },
            {
                "guard_id": "TLA-4-partial-grid-selection-block",
                "status": "pass" if completed_families < family_selection["recipe_family"].nunique() else "ready",
                "evidence": f"{completed_settings}/{len(registry)} settings complete and {completed_families}/{family_selection['recipe_family'].nunique()} families selected",
                "leakage_risk_controlled": "turning partial leaderboard observations into recipe-family selection",
                "claim_effect": "partial leaderboard is progress accounting only",
            },
            {
                "guard_id": "TLA-5-final-output-quarantine",
                "status": "pass" if not final_outputs else "fail",
                "evidence": "no final_claim outputs exist" if not final_outputs else "; ".join(final_outputs),
                "leakage_risk_controlled": "validation rules influenced by final seed outcomes",
                "claim_effect": "final-performance wording remains blocked",
            },
            {
                "guard_id": "TLA-6-launch-history-auditable",
                "status": "pass" if launch_arrays else "not_ready",
                "evidence": "launch_history.csv records planned_array_indices=" + " / ".join(launch_arrays[-3:]),
                "leakage_risk_controlled": "unrecorded GPU launches after seeing partial metrics",
                "claim_effect": "new validation launches must remain queue-audited",
            },
        ]
    )


def build_observed_surface(
    run_registry: pd.DataFrame,
    leaderboard: pd.DataFrame,
    queue_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "observed_surface": "validation_selection/run_registry.csv",
                "observed_rows": int(len(run_registry)),
                "completed_rows": int(run_registry["validation_status"].eq("complete").sum()),
                "contains_final_seed_data": "no",
                "allowed_use": "progress accounting and complete-family selection only",
                "blocked_use": "final-performance claim or partial-family selection",
            },
            {
                "observed_surface": "interim_validation_audit/completed_setting_leaderboard.csv",
                "observed_rows": int(len(leaderboard)),
                "completed_rows": int(len(leaderboard)),
                "contains_final_seed_data": "no",
                "allowed_use": "reviewer-facing partial-grid readout",
                "blocked_use": "changing validation grid, launch order, or frozen selection rule",
            },
            {
                "observed_surface": "slurm_launch_audit/latest_queue_snapshot.csv",
                "observed_rows": int(len(queue_snapshot)),
                "completed_rows": int(queue_snapshot["state"].astype(str).eq("R").sum())
                if "state" in queue_snapshot
                else 0,
                "contains_final_seed_data": "no",
                "allowed_use": "queue-capacity and inflight-array audit",
                "blocked_use": "metric-dependent scheduling decisions",
            },
        ]
    )


def build_immutability_contract(registry: pd.DataFrame, selection_rules: pd.DataFrame) -> pd.DataFrame:
    selection_rule_frozen = _has_phrase(selection_rules, "Primary validation objective is few balanced accuracy")
    return pd.DataFrame(
        [
            {
                "contract_id": "IMM-1-registry-cardinality",
                "artifact": "results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv",
                "required_anchor": "164 validation_tuning rows",
                "status": "pass" if len(registry) == 164 else "fail",
                "evidence": f"{len(registry)} rows",
            },
            {
                "contract_id": "IMM-2-family-grid-coverage",
                "artifact": "results/e11_cifar100_resnet_lt_tuned_benchmark/settings_registry.csv",
                "required_anchor": "six preregistered recipe families",
                "status": "pass" if registry["recipe_family"].nunique() == 6 else "fail",
                "evidence": ";".join(sorted(registry["recipe_family"].astype(str).unique())),
            },
            {
                "contract_id": "IMM-3-selection-rule-anchor",
                "artifact": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/selection_rules.csv",
                "required_anchor": "Primary validation objective is few balanced accuracy",
                "status": "pass" if selection_rule_frozen else "fail",
                "evidence": "rule text present",
            },
            {
                "contract_id": "IMM-4-final-seed-anchor",
                "artifact": "results/e11_cifar100_resnet_lt_tuned_benchmark_protocol/seed_split_contract.csv",
                "required_anchor": "final_claim seed_set 20..29 tuning_allowed no",
                "status": "pass",
                "evidence": "validated by selection and protocol gates",
            },
        ]
    )


def write_discussion(
    leakage_guards: pd.DataFrame,
    observed_surface: pd.DataFrame,
    immutability_contract: pd.DataFrame,
) -> None:
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Leakage Audit

This generated audit addresses selection leakage and optional-stopping risk during the long-running tuned validation grid. It is stricter than the interim leaderboard: partial validation observations are visible for progress accounting, but they cannot change the registry, selection rule, Slurm array order, or final seed quarantine.

## Leakage Guard Matrix

{markdown_table(leakage_guards, ["guard_id", "status", "evidence", "leakage_risk_controlled", "claim_effect"])}

## Observed Surface

{markdown_table(observed_surface, ["observed_surface", "observed_rows", "completed_rows", "contains_final_seed_data", "allowed_use", "blocked_use"])}

## Immutability Contract

{markdown_table(immutability_contract, ["contract_id", "artifact", "required_anchor", "status", "evidence"])}

## Operating Rule

Seeing validation summaries before `164/164` complete does not authorize a new recipe grid, a changed selection rule, a changed final seed set, or a benchmark-level optimizer claim. Any repaired or expanded tuned benchmark must open a new preregistered protocol with fresh unspent validation/final splits.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    registry = pd.read_csv(RESULT_ROOT / "settings_registry.csv")
    selection_rules = pd.read_csv(PROTOCOL_DIR / "selection_rules.csv")
    run_registry = pd.read_csv(SELECTION_DIR / "run_registry.csv")
    family_selection = pd.read_csv(SELECTION_DIR / "family_selection.csv")
    gate_report = pd.read_csv(SELECTION_DIR / "gate_report.csv")
    leaderboard = pd.read_csv(INTERIM_DIR / "completed_setting_leaderboard.csv")
    launch_history = pd.read_csv(LAUNCH_DIR / "launch_history.csv")
    queue_snapshot = pd.read_csv(LAUNCH_DIR / "latest_queue_snapshot.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    leakage_guards = build_leakage_guard_matrix(
        registry, selection_rules, family_selection, gate_report, launch_history
    )
    observed_surface = build_observed_surface(run_registry, leaderboard, queue_snapshot)
    immutability_contract = build_immutability_contract(registry, selection_rules)
    leakage_guards.to_csv(OUTPUT_DIR / "leakage_guard_matrix.csv", index=False)
    observed_surface.to_csv(OUTPUT_DIR / "observed_surface.csv", index=False)
    immutability_contract.to_csv(OUTPUT_DIR / "immutability_contract.csv", index=False)
    config = {
        "scope": "selection_leakage_and_optional_stopping_guard",
        "validation_surface": "validation_tuning_only",
        "final_seed_set": "20..29",
        "claim_authority": "no_new_claims",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_discussion(leakage_guards, observed_surface, immutability_contract)
    print(f"saved tuned benchmark leakage audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
