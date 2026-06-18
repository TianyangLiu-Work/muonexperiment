from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown


RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
FINAL_EXECUTION_DIR = RESULT_ROOT / "final_execution_plan"
FINAL_EVALUATION_DIR = RESULT_ROOT / "final_evaluation"
OUTPUT_DIR = RESULT_ROOT / "final_launch_audit"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_launch_audit.md")
SBATCH_PATH = Path("scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_final.sbatch")

MAX_SUBMIT_JOBS_PER_USER = 30
RESERVED_SUBMIT_SLOTS = 6
FINAL_ARRAY_ELEMENTS = 6
SBATCH_JOB_NAME = "e11-lt-tuned-final"


def run_squeue() -> pd.DataFrame:
    result = subprocess.run(
        ["squeue", "-h", "-r", "-u", os.environ.get("USER", ""), "-o", "%i|%t|%j|%R"],
        check=False,
        text=True,
        capture_output=True,
    )
    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = line.split("|", maxsplit=3)
        if len(parts) != 4:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "state": parts[1],
                "job_name": parts[2],
                "reason_or_node": parts[3],
                "source": "squeue -h -r -u $USER -o %i|%t|%j|%R",
            }
        )
    frame = pd.DataFrame(rows, columns=["job_id", "state", "job_name", "reason_or_node", "source"])
    frame.attrs["squeue_returncode"] = result.returncode
    frame.attrs["squeue_stderr"] = result.stderr.strip()
    return frame


def parse_inflight_final_indices(queue: pd.DataFrame) -> list[int]:
    indices: list[int] = []
    if queue.empty:
        return indices
    for _, row in queue[queue["job_name"].astype(str).eq(SBATCH_JOB_NAME)].iterrows():
        match = re.search(r"_(\d+)$", str(row["job_id"]))
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def _status_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return frame.set_index("gate_id")["status"].astype(str).to_dict()


def load_gate_snapshot(queue: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    fep_gates = pd.read_csv(FINAL_EXECUTION_DIR / "gate_matrix.csv")
    tfe_gates = pd.read_csv(FINAL_EVALUATION_DIR / "claim_gate_report.csv")
    fep_lookup = _status_lookup(fep_gates)
    tfe_lookup = _status_lookup(tfe_gates)
    fep_ready = bool(fep_gates["status"].astype(str).eq("pass").all())
    evaluator_ready = tfe_lookup.get("TFE-1-evaluator-implemented") == "pass"
    final_inflight = parse_inflight_final_indices(queue)
    current_queue_elements = int(len(queue))
    available_slots = max(0, args.max_submit_jobs - args.reserved_submit_slots - current_queue_elements)
    queue_ready = available_slots >= FINAL_ARRAY_ELEMENTS
    duplicate_ready = not final_inflight
    rows = [
        {
            "gate_id": "FLA-1-final-execution-gates-ready",
            "status": "pass" if fep_ready else "not_ready",
            "evidence": "; ".join(f"{gate}={status}" for gate, status in sorted(fep_lookup.items())),
            "blocks_submit": "yes",
        },
        {
            "gate_id": "FLA-2-final-evaluator-implemented",
            "status": "pass" if evaluator_ready else "fail",
            "evidence": f"TFE-1-evaluator-implemented={tfe_lookup.get('TFE-1-evaluator-implemented', 'missing')}",
            "blocks_submit": "yes",
        },
        {
            "gate_id": "FLA-3-no-final-job-duplicate",
            "status": "pass" if duplicate_ready else "not_ready",
            "evidence": "no final Slurm array elements inflight"
            if duplicate_ready
            else "inflight final array indices: " + ";".join(str(index) for index in final_inflight),
            "blocks_submit": "yes",
        },
        {
            "gate_id": "FLA-4-queue-capacity",
            "status": "pass" if queue_ready else "not_ready",
            "evidence": (
                f"available_submit_slots={available_slots}; required_final_array_elements={FINAL_ARRAY_ELEMENTS}; "
                f"current_queue_elements={current_queue_elements}"
            ),
            "blocks_submit": "yes",
        },
        {
            "gate_id": "FLA-5-submit-flag",
            "status": "requested" if args.submit else "dry_run",
            "evidence": "--submit provided" if args.submit else "no --submit flag; audit only",
            "blocks_submit": "no",
        },
    ]
    launch_ready = fep_ready and evaluator_ready and duplicate_ready and queue_ready
    return pd.DataFrame(rows), launch_ready


def build_decision(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queue = run_squeue()
    gate_snapshot, launch_ready = load_gate_snapshot(queue, args)
    final_plan = pd.read_csv(FINAL_EXECUTION_DIR / "final_family_run_plan.csv")
    selected = final_plan[["final_index", "recipe_family", "run_status", "selected_setting_id", "selected_recipe_name", "final_seed_set", "planned_output_dir"]].copy()
    current_queue_elements = int(len(queue))
    available_slots = max(0, args.max_submit_jobs - args.reserved_submit_slots - current_queue_elements)
    ready_families = (
        selected["recipe_family"].astype(str).tolist()
        if launch_ready
        else []
    )
    submit_command_parts = ["sbatch", "--parsable", SBATCH_PATH.as_posix()]
    submit_command = " ".join(submit_command_parts) if launch_ready else ""
    status = "dry_run" if launch_ready else "blocked_final_launch_gates_not_ready"
    slurm_job_id = ""
    sbatch_exit_code = ""
    sbatch_stdout = ""
    sbatch_stderr = ""
    if launch_ready and args.submit:
        result = subprocess.run(submit_command_parts, check=False, text=True, capture_output=True)
        sbatch_exit_code = str(result.returncode)
        sbatch_stdout = result.stdout.strip()
        sbatch_stderr = result.stderr.strip()
        if result.returncode == 0:
            status = "submitted"
            slurm_job_id = sbatch_stdout.split(";")[0].strip()
        else:
            status = "sbatch_failed"
    decision = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "submission_status": status,
                "submit_command": submit_command,
                "slurm_job_id": slurm_job_id,
                "sbatch_exit_code": sbatch_exit_code,
                "sbatch_stdout": sbatch_stdout,
                "sbatch_stderr": sbatch_stderr,
                "current_queue_elements_before_submit": current_queue_elements,
                "max_submit_jobs_per_user": args.max_submit_jobs,
                "reserved_submit_slots": args.reserved_submit_slots,
                "available_submit_slots_before_submit": available_slots,
                "planned_final_array_elements": FINAL_ARRAY_ELEMENTS if launch_ready else 0,
                "array_expression": "0-5%1" if launch_ready else "",
                "planned_recipe_families": ";".join(ready_families),
                "inflight_final_indices_before_submit": ";".join(
                    str(index) for index in parse_inflight_final_indices(queue)
                ),
                "final_seed_status": "eligible_only_after_fep_gates_pass",
                "phase_guard": "final_claim_only",
            }
        ]
    )
    return decision, queue, selected, gate_snapshot


def append_history(decision: pd.DataFrame) -> None:
    history_path = OUTPUT_DIR / "final_launch_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, decision], ignore_index=True)
    else:
        history = decision
    history.to_csv(history_path, index=False)


def write_discussion(
    decision: pd.DataFrame,
    queue: pd.DataFrame,
    selected: pd.DataFrame,
    gate_snapshot: pd.DataFrame,
) -> None:
    status = str(decision["submission_status"].iloc[0])
    command = str(decision["submit_command"].iloc[0])
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Final Launch Audit

This audit records a queue-aware launch decision for the untouched final-claim
seed jobs. It is separate from the final execution plan: this file records the
actual queue state, the final-launch gates, and whether `sbatch` was called.

Current final launch status: `{status}`.

## Launch Gate Snapshot

{markdown_table(gate_snapshot, ["gate_id", "status", "evidence", "blocks_submit"])}

## Launch Decision

{markdown_table(decision, ["timestamp_utc", "submission_status", "submit_command", "slurm_job_id", "current_queue_elements_before_submit", "available_submit_slots_before_submit", "planned_final_array_elements", "array_expression", "planned_recipe_families", "inflight_final_indices_before_submit", "final_seed_status", "phase_guard"])}

## Final Family Plan

{markdown_table(selected, ["final_index", "recipe_family", "run_status", "selected_setting_id", "selected_recipe_name", "final_seed_set", "planned_output_dir"])}

## Queue Snapshot Before Submit

{markdown_table(queue, ["job_id", "state", "job_name", "reason_or_node"])}

## Operating Rule

The launch guard preserves the final seed quarantine by requiring every FEP gate
to pass before a final Slurm command is emitted. The submitted command, when
non-empty, is:

```bash
{command}
```

Only `final_claim` seed set `20..29` is eligible, and the evaluator contract
must exist before final jobs are launched. Current `not_ready` rows authorize no benchmark-performance wording and no final submission.
"""
    write_markdown(DISCUSSION_PATH, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue-aware launcher for the CIFAR-100-LT tuned benchmark final-claim jobs."
    )
    parser.add_argument("--submit", action="store_true", help="call sbatch only if every final launch gate passes")
    parser.add_argument("--max-submit-jobs", type=int, default=MAX_SUBMIT_JOBS_PER_USER)
    parser.add_argument("--reserved-submit-slots", type=int, default=RESERVED_SUBMIT_SLOTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decision, queue, selected, gate_snapshot = build_decision(args)
    decision.to_csv(OUTPUT_DIR / "latest_final_launch_decision.csv", index=False)
    queue.to_csv(OUTPUT_DIR / "latest_queue_snapshot.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "latest_final_family_plan.csv", index=False)
    gate_snapshot.to_csv(OUTPUT_DIR / "latest_gate_snapshot.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "final_execution_plan": FINAL_EXECUTION_DIR.as_posix(),
                "final_evaluation": FINAL_EVALUATION_DIR.as_posix(),
                "sbatch_wrapper": SBATCH_PATH.as_posix(),
                "max_submit_jobs_per_user": args.max_submit_jobs,
                "reserved_submit_slots": args.reserved_submit_slots,
                "final_array_elements": FINAL_ARRAY_ELEMENTS,
                "submits_jobs_when_submit_flag_is_set_and_gates_pass": True,
                "submit_flag": bool(args.submit),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    append_history(decision)
    write_discussion(decision, queue, selected, gate_snapshot)
    print(
        "saved tuned benchmark final launch audit to "
        f"{OUTPUT_DIR} and {DISCUSSION_PATH}; status={decision['submission_status'].iloc[0]}"
    )


if __name__ == "__main__":
    main()
