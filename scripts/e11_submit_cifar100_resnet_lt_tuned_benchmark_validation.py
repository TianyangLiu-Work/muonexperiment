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
REGISTRY_PATH = RESULT_ROOT / "settings_registry.csv"
RUN_REGISTRY_PATH = RESULT_ROOT / "validation_selection/run_registry.csv"
OUTPUT_DIR = RESULT_ROOT / "slurm_launch_audit"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_launch_audit.md")
SBATCH_PATH = Path("scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch")

MAX_SUBMIT_JOBS_PER_USER = 30
RESERVED_SUBMIT_SLOTS = 6
CHUNK_SIZE = 20
ARRAY_CONCURRENCY = 1
SBATCH_JOB_NAME = "e11-lt-tuned-val"


def compress_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = value
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


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


def parse_inflight_array_indices(queue: pd.DataFrame) -> list[int]:
    indices: list[int] = []
    if queue.empty:
        return indices
    for _, row in queue[queue["job_name"].astype(str).eq(SBATCH_JOB_NAME)].iterrows():
        match = re.search(r"_(\d+)$", str(row["job_id"]))
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def load_missing_indices(inflight_indices: set[int]) -> tuple[pd.DataFrame, list[int], int]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"missing tuned benchmark settings registry: {REGISTRY_PATH}")
    registry = pd.read_csv(REGISTRY_PATH).sort_values("array_index").reset_index(drop=True)
    if RUN_REGISTRY_PATH.exists():
        run_registry = pd.read_csv(RUN_REGISTRY_PATH)
        merged = registry[["array_index", "setting_id", "recipe_family", "phase", "seed_set"]].merge(
            run_registry[["array_index", "validation_status", "occupancy_status"]],
            on="array_index",
            how="left",
        )
        missing = merged[
            ~(
                merged["validation_status"].astype(str).eq("complete")
                & merged["occupancy_status"].astype(str).eq("complete")
            )
        ]
    else:
        missing = registry
    missing_indices = [
        int(value)
        for value in sorted(missing["array_index"].tolist())
        if int(value) not in inflight_indices
    ]
    completed_settings = int(len(registry) - len(missing))
    return registry, missing_indices, completed_settings


def build_decision(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    queue = run_squeue()
    inflight_indices = set(parse_inflight_array_indices(queue))
    registry, missing_indices, completed_settings = load_missing_indices(inflight_indices)
    current_queue_elements = int(len(queue))
    available_slots = max(0, args.max_submit_jobs - args.reserved_submit_slots - current_queue_elements)
    new_count = min(args.chunk_size, args.max_new_elements, available_slots, len(missing_indices))
    selected_indices = missing_indices[:new_count]
    array_expression = compress_indices(selected_indices)
    submit_command_parts = [
        "sbatch",
        "--parsable",
        f"--array={array_expression}%{args.array_concurrency}",
        SBATCH_PATH.as_posix(),
    ]
    submit_command = " ".join(submit_command_parts) if selected_indices else ""
    status = "dry_run"
    slurm_job_id = ""
    sbatch_exit_code = ""
    sbatch_stdout = ""
    sbatch_stderr = ""
    if not missing_indices:
        status = "blocked_no_missing_settings"
    elif new_count == 0:
        status = "blocked_no_queue_capacity"
    elif args.submit:
        result = subprocess.run(submit_command_parts, check=False, text=True, capture_output=True)
        sbatch_exit_code = str(result.returncode)
        sbatch_stdout = result.stdout.strip()
        sbatch_stderr = result.stderr.strip()
        if result.returncode == 0:
            status = "submitted"
            slurm_job_id = sbatch_stdout.split(";")[0].strip()
        else:
            status = "sbatch_failed"

    selected = registry[registry["array_index"].astype(int).isin(selected_indices)]
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
                "chunk_size": args.chunk_size,
                "max_new_elements": args.max_new_elements,
                "array_concurrency": args.array_concurrency,
                "planned_setting_count": int(len(selected_indices)),
                "array_expression": array_expression,
                "array_start": int(min(selected_indices)) if selected_indices else "",
                "array_end": int(max(selected_indices)) if selected_indices else "",
                "planned_array_indices": ";".join(str(index) for index in selected_indices),
                "planned_recipe_families": ";".join(sorted(selected["recipe_family"].astype(str).unique())),
                "completed_settings_before_submit": completed_settings,
                "missing_settings_not_inflight_before_submit": int(len(missing_indices)),
                "inflight_validation_indices_before_submit": ";".join(str(index) for index in sorted(inflight_indices)),
                "final_seed_status": "not_touched",
                "phase_guard": "validation_tuning_only",
            }
        ]
    )
    selected_rows = selected[
        [
            "array_index",
            "setting_id",
            "phase",
            "seed_set",
            "recipe_family",
            "recipe_name",
            "planned_output_dir",
            "planned_occupancy_trace_path",
        ]
    ].copy()
    return decision, queue, selected_rows, submit_command_parts


def append_history(decision: pd.DataFrame) -> None:
    history_path = OUTPUT_DIR / "launch_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, decision], ignore_index=True)
    else:
        history = decision
    history.to_csv(history_path, index=False)


def write_discussion(decision: pd.DataFrame, queue: pd.DataFrame, selected: pd.DataFrame) -> None:
    status = str(decision["submission_status"].iloc[0])
    command = str(decision["submit_command"].iloc[0])
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Slurm Launch Audit

This audit records a queue-aware launch decision for the tuned validation grid.
It is separate from the static no-side-effect chunk plan: this file records the
actual queue state, the guarded capacity calculation, and whether `sbatch` was
called.

Current launch status: `{status}`.

## Launch Decision

{markdown_table(decision, ["timestamp_utc", "submission_status", "submit_command", "slurm_job_id", "current_queue_elements_before_submit", "available_submit_slots_before_submit", "planned_setting_count", "array_expression", "planned_recipe_families", "completed_settings_before_submit", "missing_settings_not_inflight_before_submit", "inflight_validation_indices_before_submit", "final_seed_status", "phase_guard"])}

## Selected Validation Settings

{markdown_table(selected, ["array_index", "setting_id", "phase", "seed_set", "recipe_family", "recipe_name", "planned_output_dir", "planned_occupancy_trace_path"])}

## Queue Snapshot Before Submit

{markdown_table(queue, ["job_id", "state", "job_name", "reason_or_node"])}

## Operating Rule

The launch guard preserves `MaxSubmitJobsPerUser` by enforcing
`current_queue_elements + planned_setting_count + reserved_submit_slots <=
max_submit_jobs_per_user`. The submitted command, when non-empty, is:

```bash
{command}
```

Only `validation_tuning` rows with seed set `10..14` are eligible. The untouched
`final_claim` seeds `20..29` remain blocked until validation selection passes
`TVS-1`, `TVS-2`, and `TVS-5`.
"""
    write_markdown(DISCUSSION_PATH, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue-aware launcher for the CIFAR-100-LT tuned benchmark validation grid."
    )
    parser.add_argument("--submit", action="store_true", help="call sbatch after writing the guarded decision")
    parser.add_argument("--max-submit-jobs", type=int, default=MAX_SUBMIT_JOBS_PER_USER)
    parser.add_argument("--reserved-submit-slots", type=int, default=RESERVED_SUBMIT_SLOTS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--array-concurrency", type=int, default=ARRAY_CONCURRENCY)
    parser.add_argument("--max-new-elements", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decision, queue, selected, _ = build_decision(args)
    decision.to_csv(OUTPUT_DIR / "latest_launch_decision.csv", index=False)
    queue.to_csv(OUTPUT_DIR / "latest_queue_snapshot.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "latest_selected_settings.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "settings_registry": REGISTRY_PATH.as_posix(),
                "run_registry": RUN_REGISTRY_PATH.as_posix(),
                "sbatch_wrapper": SBATCH_PATH.as_posix(),
                "max_submit_jobs_per_user": args.max_submit_jobs,
                "reserved_submit_slots": args.reserved_submit_slots,
                "chunk_size": args.chunk_size,
                "array_concurrency": args.array_concurrency,
                "max_new_elements": args.max_new_elements,
                "submits_jobs_when_submit_flag_is_set": True,
                "submit_flag": bool(args.submit),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    append_history(decision)
    write_discussion(decision, queue, selected)
    print(
        "saved tuned benchmark Slurm launch audit to "
        f"{OUTPUT_DIR} and {DISCUSSION_PATH}; status={decision['submission_status'].iloc[0]}"
    )


if __name__ == "__main__":
    main()
