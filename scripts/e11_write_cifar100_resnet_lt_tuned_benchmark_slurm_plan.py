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
REGISTRY_PATH = RESULT_ROOT / "settings_registry.csv"
RUN_REGISTRY_PATH = RESULT_ROOT / "validation_selection/run_registry.csv"
OUTPUT_DIR = RESULT_ROOT / "slurm_submission_plan"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_slurm_plan.md")
SBATCH_PATH = Path("scripts/slurm/e11_cifar100_resnet_lt_tuned_benchmark_validation.sbatch")

CHUNK_SIZE = 20
ARRAY_CONCURRENCY = 1
MAX_SUBMIT_JOBS_PER_USER = 30
RESERVED_SUBMIT_SLOTS = 6


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


def chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def load_missing_indices() -> tuple[pd.DataFrame, list[int], int]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"missing tuned benchmark settings registry: {REGISTRY_PATH}")
    registry = pd.read_csv(REGISTRY_PATH).sort_values("array_index").reset_index(drop=True)
    if RUN_REGISTRY_PATH.exists():
        run_registry = pd.read_csv(RUN_REGISTRY_PATH)
        merged = registry[["array_index", "setting_id", "recipe_family"]].merge(
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
    missing_indices = sorted(int(value) for value in missing["array_index"].tolist())
    return registry, missing_indices, int(len(registry) - len(missing_indices))


def build_plan(registry: pd.DataFrame, missing_indices: list[int], completed_settings: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chunk_index, indices in enumerate(chunked(missing_indices, CHUNK_SIZE), start=1):
        expression = compress_indices(indices)
        settings = registry[registry["array_index"].astype(int).isin(indices)]
        rows.append(
            {
                "chunk_id": f"TBV-SLURM-{chunk_index:02d}",
                "array_expression": expression,
                "array_start": int(min(indices)),
                "array_end": int(max(indices)),
                "setting_count": int(len(indices)),
                "recipe_families": ";".join(sorted(settings["recipe_family"].astype(str).unique())),
                "submit_command": f"sbatch --array={expression}%{ARRAY_CONCURRENCY} {SBATCH_PATH.as_posix()}",
                "expected_outputs": "summary.csv; occupancy_trace.csv; class_metrics.csv; group_metrics.csv",
                "post_chunk_action": "rerun make e11-cifar-resnet-lt-tuned-benchmark-selection and inspect TVS gates",
                "submission_status": "not_submitted_static_plan",
            }
        )
    if not rows:
        rows.append(
            {
                "chunk_id": "TBV-SLURM-COMPLETE",
                "array_expression": "",
                "array_start": "",
                "array_end": "",
                "setting_count": 0,
                "recipe_families": "",
                "submit_command": "",
                "expected_outputs": "all validation summaries and occupancy traces are present",
                "post_chunk_action": "rerun selection and power audit",
                "submission_status": "all_settings_complete",
            }
        )
    frame = pd.DataFrame(rows)
    frame["completed_settings_before_plan"] = completed_settings
    frame["missing_settings_before_plan"] = len(missing_indices)
    return frame


def build_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_id": "SLURM-1-gpu-only-through-slurm",
                "rule": "GPU validation jobs are submitted through Slurm, never as direct local CUDA processes.",
                "evidence": "serverREADME.md states gpu_rogue_kill enforces Slurm-only GPU use.",
            },
            {
                "policy_id": "SLURM-2-submit-limit",
                "rule": f"Submit at most {CHUNK_SIZE} validation array elements per chunk, leaving {RESERVED_SUBMIT_SLOTS} slots below MaxSubmitJobsPerUser={MAX_SUBMIT_JOBS_PER_USER}.",
                "evidence": "The static plan chunks 164 settings into bounded sbatch arrays.",
            },
            {
                "policy_id": "SLURM-3-concurrency",
                "rule": f"Use array concurrency %{ARRAY_CONCURRENCY} for this validation grid unless the queue is explicitly checked before submission.",
                "evidence": "The server allows limited concurrent GPU jobs per user; serialized chunks avoid starving the queue.",
            },
            {
                "policy_id": "SLURM-4-no-final-unblinding",
                "rule": "Only validation_tuning seed set 10..14 is eligible for this plan; final_claim seeds 20..29 remain untouched.",
                "evidence": "The sbatch wrapper calls the validation runner by array index and every registry row has phase=validation_tuning.",
            },
        ]
    )


def write_discussion(plan: pd.DataFrame, policy: pd.DataFrame) -> None:
    chunk_count = int(plan["chunk_id"].ne("TBV-SLURM-COMPLETE").sum())
    missing = int(plan["missing_settings_before_plan"].iloc[0])
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Slurm Plan

This generated plan is a no-side-effect launch contract for the registered
164-setting validation grid. It does not submit jobs. Its role is to keep the
top-conference benchmark path executable on the current server without violating
the documented GPU policy or `MaxSubmitJobsPerUser` constraint.

Current missing validation cells: `{missing}`. Planned chunks: `{chunk_count}`.
Each completed cell must produce both `summary.csv` and `occupancy_trace.csv`;
the selection audit remains blocked until both artifacts exist for every
registered setting in a recipe family.

## Queue Policy

{markdown_table(policy, ["policy_id", "rule", "evidence"])}

## Chunk Plan

{markdown_table(plan, ["chunk_id", "array_expression", "setting_count", "recipe_families", "submit_command", "expected_outputs", "post_chunk_action", "submission_status"])}

## Operating Rule

Before running a command from this table, check `squeue -u "$USER"` and submit
only one chunk when the current number of submitted jobs plus the chunk size
stays below the server's user submit limit. After each chunk finishes, rerun
`make e11-cifar-resnet-lt-tuned-benchmark-selection`; do not run final seeds
until `TVS-1`, `TVS-2`, and `TVS-5` all pass.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry, missing_indices, completed_settings = load_missing_indices()
    plan = build_plan(registry, missing_indices, completed_settings)
    policy = build_policy()
    plan.to_csv(OUTPUT_DIR / "chunk_plan.csv", index=False)
    policy.to_csv(OUTPUT_DIR / "queue_policy.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "settings_registry": REGISTRY_PATH.as_posix(),
                "run_registry": RUN_REGISTRY_PATH.as_posix(),
                "sbatch_wrapper": SBATCH_PATH.as_posix(),
                "chunk_size": CHUNK_SIZE,
                "array_concurrency": ARRAY_CONCURRENCY,
                "max_submit_jobs_per_user": MAX_SUBMIT_JOBS_PER_USER,
                "reserved_submit_slots": RESERVED_SUBMIT_SLOTS,
                "side_effects": False,
                "submits_jobs": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(plan, policy)
    print(f"saved tuned benchmark Slurm plan to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
