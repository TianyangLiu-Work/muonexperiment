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
ANALYSIS_PLAN_DIR = RESULT_ROOT / "final_analysis_plan"
OUTPUT_DIR = RESULT_ROOT / "final_execution_plan"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_execution_plan.md")
RUNNER_PATH = Path("scripts/e11_run_cifar100_resnet_lt_tuned_benchmark.py")
SBATCH_PATH = Path("scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch")
PYTHON_CMD = "/data/conda_envs/SpatialQuantization/bin/python"


REQUIRED_SELECTION_GATES = {
    "TVS-1-validation-grid-complete": "pass",
    "TVS-2-family-selection": "pass",
    "TVS-5-occupancy-logging-complete": "pass",
    "TVS-3-final-seed-quarantine": "pass",
    "TVS-4-final-run-plan": "ready",
}


def _final_seed_set() -> str:
    seed_split = pd.read_csv(PROTOCOL_DIR / "seed_split_contract.csv")
    final_rows = seed_split[seed_split["split_id"].astype(str).eq("final_claim")]
    if final_rows.empty:
        raise AssertionError("missing final_claim split in seed_split_contract.csv")
    return str(final_rows["seed_set"].iloc[0])


def _selection_gate_status() -> tuple[bool, str]:
    gate_report = pd.read_csv(SELECTION_DIR / "gate_report.csv")
    lookup = gate_report.set_index("gate_id")["status"].astype(str).to_dict()
    evidence = "; ".join(
        f"{gate_id}={lookup.get(gate_id, 'missing')}"
        for gate_id in REQUIRED_SELECTION_GATES
    )
    ready = all(
        lookup.get(gate_id) == status
        for gate_id, status in REQUIRED_SELECTION_GATES.items()
    )
    return ready, evidence


def _analysis_plan_linked() -> tuple[bool, str]:
    gate_path = ANALYSIS_PLAN_DIR / "gate_matrix.csv"
    comparison_path = ANALYSIS_PLAN_DIR / "primary_comparison_family.csv"
    if not gate_path.exists() or not comparison_path.exists():
        return False, "missing final_analysis_plan gate or comparison table"
    gates = pd.read_csv(gate_path)
    comparisons = pd.read_csv(comparison_path)
    lookup = gates.set_index("gate_id")["status"].astype(str).to_dict()
    final_quarantine = lookup.get("FAP-1-final-quarantine", "missing")
    family_size = int(len(comparisons))
    return (
        final_quarantine == "pass" and family_size == 4,
        f"FAP-1-final-quarantine={final_quarantine}; primary_family_size={family_size}",
    )


def _runner_contract_present() -> tuple[bool, str]:
    runner_text = RUNNER_PATH.read_text(encoding="utf-8") if RUNNER_PATH.exists() else ""
    sbatch_text = SBATCH_PATH.read_text(encoding="utf-8") if SBATCH_PATH.exists() else ""
    required_runner_phrases = [
        "--phase",
        "final_claim",
        "--final-index",
        "final_settings_from_selection",
        "validation selection gates pass",
    ]
    required_sbatch_phrases = [
        "--phase final_claim",
        "--final-index",
        "--device cuda",
        "--no-download",
    ]
    missing = [
        phrase for phrase in required_runner_phrases if phrase not in runner_text
    ] + [
        phrase for phrase in required_sbatch_phrases if phrase not in sbatch_text
    ]
    if missing:
        return False, "missing contract phrases: " + "; ".join(missing)
    return True, f"runner={RUNNER_PATH.as_posix()}; sbatch={SBATCH_PATH.as_posix()}"


def _final_outputs_absent() -> tuple[bool, str]:
    final_root = RESULT_ROOT / "final_claim"
    outputs = sorted(path.as_posix() for path in final_root.rglob("*")) if final_root.exists() else []
    return not outputs, "no final_claim files exist" if not outputs else "; ".join(outputs)


def build_gate_matrix(final_plan: pd.DataFrame) -> pd.DataFrame:
    selection_ready, selection_evidence = _selection_gate_status()
    analysis_ready, analysis_evidence = _analysis_plan_linked()
    runner_ready, runner_evidence = _runner_contract_present()
    final_absent, final_evidence = _final_outputs_absent()
    all_families_ready = bool(final_plan["final_status"].astype(str).eq("ready_for_final_run").all())
    return pd.DataFrame(
        [
            {
                "gate_id": "FEP-1-selection-gates-ready",
                "status": "pass" if selection_ready else "not_ready",
                "evidence": selection_evidence,
                "blocks_final_submit": "yes",
            },
            {
                "gate_id": "FEP-2-final-analysis-plan-linked",
                "status": "pass" if analysis_ready else "not_ready",
                "evidence": analysis_evidence,
                "blocks_final_submit": "yes",
            },
            {
                "gate_id": "FEP-3-runner-contract-present",
                "status": "pass" if runner_ready else "fail",
                "evidence": runner_evidence,
                "blocks_final_submit": "yes",
            },
            {
                "gate_id": "FEP-4-final-output-quarantine",
                "status": "pass" if final_absent else "fail",
                "evidence": final_evidence,
                "blocks_final_submit": "yes",
            },
            {
                "gate_id": "FEP-5-all-families-ready",
                "status": "pass" if all_families_ready else "not_ready",
                "evidence": (
                    f"{int(final_plan['final_status'].astype(str).eq('ready_for_final_run').sum())}/"
                    f"{len(final_plan)} families ready_for_final_run"
                ),
                "blocks_final_submit": "yes",
            },
        ]
    )


def build_family_run_plan(final_plan: pd.DataFrame, gate_matrix: pd.DataFrame) -> pd.DataFrame:
    final_seed_set = _final_seed_set()
    gate_ready = bool(gate_matrix["status"].astype(str).eq("pass").all())
    rows: list[dict[str, object]] = []
    for index, row in enumerate(final_plan.sort_values("recipe_family").itertuples(index=False)):
        selected_setting_id = "" if pd.isna(row.selected_setting_id) else str(row.selected_setting_id)
        selected_recipe_name = "" if pd.isna(row.selected_recipe_name) else str(row.selected_recipe_name)
        planned_output_dir = "" if pd.isna(row.planned_output_dir) else str(row.planned_output_dir)
        selected = selected_setting_id != ""
        command = (
            f"{PYTHON_CMD} {RUNNER_PATH.as_posix()} --phase final_claim "
            f"--final-family {row.recipe_family} --device cuda --no-download --progress"
            if selected
            else ""
        )
        rows.append(
            {
                "final_index": int(index),
                "recipe_family": str(row.recipe_family),
                "run_status": (
                    "ready_not_submitted"
                    if gate_ready and selected
                    else "blocked_until_selection_gates_pass"
                    if selected
                    else "blocked_until_family_selected"
                ),
                "selected_setting_id": selected_setting_id,
                "selected_recipe_name": selected_recipe_name,
                "final_seed_set": final_seed_set if selected else "",
                "planned_output_dir": planned_output_dir if selected else "",
                "runner_command": command,
                "tuning_allowed": "no",
            }
        )
    return pd.DataFrame(rows)


def build_sbatch_plan(gate_matrix: pd.DataFrame, family_run_plan: pd.DataFrame) -> pd.DataFrame:
    all_gates_pass = bool(gate_matrix["status"].astype(str).eq("pass").all())
    selected_count = int(family_run_plan["selected_setting_id"].astype(str).ne("").sum())
    return pd.DataFrame(
        [
            {
                "plan_id": "FEP-SLURM-1-final-family-array",
                "submission_status": "ready_not_submitted" if all_gates_pass else "not_ready",
                "array_expression": "0-5%1" if selected_count == 6 else "",
                "submit_command": f"sbatch {SBATCH_PATH.as_posix()}" if all_gates_pass else "",
                "expected_outputs": "train_trace.csv; class_metrics.csv; group_metrics.csv; summary.csv; pair_summary.csv; occupancy_trace.csv; config.json; setting_metadata.csv",
                "post_run_action": "run final statistical evaluator and e11-check before any benchmark claim",
                "submits_jobs": "no",
            }
        ]
    )


def write_discussion(
    gate_matrix: pd.DataFrame,
    family_run_plan: pd.DataFrame,
    sbatch_plan: pd.DataFrame,
) -> None:
    ready = bool(gate_matrix["status"].astype(str).eq("pass").all())
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Final Execution Plan

This generated plan is a no-side-effect execution contract for the untouched
final-claim seeds. It exists so the final benchmark can be launched without
hand-written commands after validation selection completes. It does not submit
jobs and it does not inspect or create final output files.

Current final-submit status: `{"ready_not_submitted" if ready else "not_ready"}`.

## Gate Matrix

{markdown_table(gate_matrix, ["gate_id", "status", "evidence", "blocks_final_submit"])}

## Final Family Run Plan

{markdown_table(family_run_plan, ["final_index", "recipe_family", "run_status", "selected_setting_id", "selected_recipe_name", "final_seed_set", "planned_output_dir", "runner_command", "tuning_allowed"])}

## Slurm Submit Plan

{markdown_table(sbatch_plan, ["plan_id", "submission_status", "array_expression", "submit_command", "expected_outputs", "post_run_action", "submits_jobs"])}

## Operating Rule

Do not run `scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch`
until `FEP-1`, `FEP-2`, `FEP-3`, `FEP-4`, and `FEP-5` all pass. The runner also
checks the validation-selection gates at execution time, so premature final submission fails closed. It will not produce final-claim rows from a premature
submit. After final jobs finish, the only allowed analysis path is the pre-registered paired-seed,
Holm-adjusted final-analysis plan.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_plan = pd.read_csv(SELECTION_DIR / "final_claim_plan.csv")
    gate_matrix = build_gate_matrix(final_plan)
    family_run_plan = build_family_run_plan(final_plan, gate_matrix)
    sbatch_plan = build_sbatch_plan(gate_matrix, family_run_plan)
    gate_matrix.to_csv(OUTPUT_DIR / "gate_matrix.csv", index=False)
    family_run_plan.to_csv(OUTPUT_DIR / "final_family_run_plan.csv", index=False)
    sbatch_plan.to_csv(OUTPUT_DIR / "slurm_submit_plan.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "selection_plan": (SELECTION_DIR / "final_claim_plan.csv").as_posix(),
                "analysis_plan": ANALYSIS_PLAN_DIR.as_posix(),
                "runner": RUNNER_PATH.as_posix(),
                "sbatch_wrapper": SBATCH_PATH.as_posix(),
                "final_seed_set": _final_seed_set(),
                "side_effects": False,
                "submits_jobs": False,
                "inspects_final_outputs": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(gate_matrix, family_run_plan, sbatch_plan)
    print(f"saved tuned benchmark final execution plan to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
