from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown
from scripts.e11_evaluate_natural_negative_search_phase1 import (
    ALPHA,
    HEAD_GAIN_RELATIVE_ERROR_MAX,
    TAIL_ACCURACY_FLOOR_BY_DATASET,
    build_primary_decisions,
    seed_level_primary_ratios,
)
from scripts.e11_run_natural_negative_search_phase2 import (
    DEFAULT_PHASE2_SEEDS,
    PHASE2_MULTIPLICITY_FAMILY,
    PHASE2_OUTPUT_PREFIX,
    PHASE2_SEARCH_ID,
    PROTOCOL_VERSION,
    phase2_settings,
    settings_registry,
)


RESULT_DIR = Path("results/e11_natural_negative_search_protocol")
OUTPUT_DIR = RESULT_DIR / "phase2_multiplicity_evaluation"
DISCUSSION_PATH = Path("discussion/e11_natural_negative_search_phase2_evaluation.md")


def expected_registry() -> pd.DataFrame:
    registry_path = PHASE2_OUTPUT_PREFIX / "settings_registry.csv"
    if registry_path.exists():
        return pd.read_csv(registry_path)
    return settings_registry(phase2_settings(), seeds=DEFAULT_PHASE2_SEEDS)


def expected_output_files(prefix: Path) -> dict[str, Path]:
    return {
        "settings_registry": prefix / "settings_registry.csv",
        "step_metrics": prefix / "step_metrics.csv",
        "pair_summary": prefix / "pair_summary.csv",
        "layer_metrics": prefix / "layer_metrics.csv",
        "decision_template": prefix / "decision_template.csv",
        "config": prefix / "config.json",
    }


def build_run_registry(expected: pd.DataFrame) -> pd.DataFrame:
    files = expected_output_files(PHASE2_OUTPUT_PREFIX)
    required_metric_files = ["step_metrics", "pair_summary", "layer_metrics", "decision_template", "config"]
    expected_rows = len(expected)
    settings_rows = len(pd.read_csv(files["settings_registry"])) if files["settings_registry"].exists() else 0
    pair_rows = len(pd.read_csv(files["pair_summary"])) if files["pair_summary"].exists() else 0
    metric_files_present = all(files[name].exists() for name in required_metric_files)
    if metric_files_present and pair_rows == expected_rows:
        status = "complete"
    elif files["settings_registry"].exists() and not any(files[name].exists() for name in required_metric_files):
        status = "settings_only_no_metrics"
    elif any(path.exists() for path in files.values()):
        status = "incomplete_metrics"
    else:
        status = "not_run"
    return pd.DataFrame(
        [
            {
                "search_id": PHASE2_SEARCH_ID,
                "artifact_prefix": PHASE2_OUTPUT_PREFIX.as_posix(),
                "expected_settings": expected_rows,
                "settings_registry_rows": settings_rows,
                "pair_summary_rows": pair_rows,
                "output_status": status,
                "settings_registry_path": files["settings_registry"].as_posix(),
                "pair_summary_path": files["pair_summary"].as_posix(),
            }
        ]
    )


def load_observed_pair_summary() -> pd.DataFrame:
    path = PHASE2_OUTPUT_PREFIX / "pair_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "search_id" not in frame.columns:
        frame["search_id"] = PHASE2_SEARCH_ID
    return frame


def load_observed_step_metrics() -> pd.DataFrame:
    path = PHASE2_OUTPUT_PREFIX / "step_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "search_id" not in frame.columns:
        frame["search_id"] = PHASE2_SEARCH_ID
    return frame


def build_gate_report(run_registry: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    complete = set(run_registry["output_status"]) == {"complete"}
    observed_count = int(decisions["output_status"].eq("observed").sum())
    expected_count = len(decisions)
    adjusted_worse_count = int(decisions["adjusted_primary_decision"].eq("primary_worse_adjusted").sum())
    if complete and adjusted_worse_count > 0:
        primary_status = "candidate"
        primary_evidence = f"{adjusted_worse_count} adjusted primary rows pass Holm and quality gates"
    elif complete:
        primary_status = "finite_null_candidate"
        primary_evidence = "all phase2 settings evaluated with no adjusted primary worse row"
    else:
        primary_status = "not_ready"
        primary_evidence = f"{observed_count}/{expected_count} phase2 settings have primary metric rows"
    return pd.DataFrame(
        [
            {
                "gate_id": "NNS-P2-E1-evaluator-implemented",
                "status": "pass",
                "evidence": "scripts/e11_evaluate_natural_negative_search_phase2.py wrote evaluator artifacts",
            },
            {
                "gate_id": "NNS-P2-E2-phase2-output-completeness",
                "status": "pass" if complete else "not_ready",
                "evidence": "; ".join(
                    f"{row.search_id}={row.output_status} ({int(row.pair_summary_rows)}/{int(row.expected_settings)} rows)"
                    for row in run_registry.itertuples()
                ),
            },
            {
                "gate_id": "NNS-P2-E3-primary-multiplicity",
                "status": "pass" if complete else "not_ready",
                "evidence": "Holm one-sided p-values use paired per-seed log-ratio tests over the registered 8-setting phase2 family",
            },
            {
                "gate_id": "NNS-P2-E4-heldout-architecture-claim",
                "status": primary_status,
                "evidence": primary_evidence,
            },
            {
                "gate_id": "NNS-P2-E5-full-reporting-boundary",
                "status": "pass",
                "evidence": "primary_decisions.csv keeps every registered phase2 setting, including not-run rows",
            },
        ]
    )


def write_discussion(
    run_registry: pd.DataFrame,
    decisions: pd.DataFrame,
    gate_report: pd.DataFrame,
    seed_ratios: pd.DataFrame,
) -> None:
    observed = int(decisions["output_status"].eq("observed").sum())
    total = len(decisions)
    head_gain_fail = 0
    tail_quality_pass = 0
    if "head_gain_gate" in decisions.columns:
        head_gain_fail = int(decisions["head_gain_gate"].astype(str).str.lower().eq("false").sum())
    if "tail_quality_gate" in decisions.columns:
        tail_quality_pass = int(decisions["tail_quality_gate"].astype(str).str.lower().eq("true").sum())
    lines = [
        "# E11 Natural Negative Search Phase2 Evaluation",
        "",
        "This generated evaluator is the multiplicity boundary for the registered",
        "phase2 ResNet34 held-out architecture family. It does not claim a held-out",
        "architecture replication, counterexample, or broader finite null until all",
        "8 declared settings have metric rows and the Holm-adjusted primary decision",
        "from paired per-seed log-ratio tests passes the reporting and quality gates.",
        "",
        f"Current primary metric coverage: {observed}/{total} settings.",
        f"Current seed-level primary rows: {len(seed_ratios)}.",
        f"Current phase2 caveat: head_gain_gate fails in {head_gain_fail}/{total} rows, while tail_quality_gate passes in {tail_quality_pass}/{total} rows.",
        "",
        "## Run Registry",
        "",
        markdown_table(
            run_registry,
            [
                "search_id",
                "output_status",
                "expected_settings",
                "settings_registry_rows",
                "pair_summary_rows",
                "artifact_prefix",
            ],
        ),
        "",
        "## Gate Report",
        "",
        markdown_table(gate_report, ["gate_id", "status", "evidence"]),
        "",
        "## Primary Decision Boundary",
        "",
        markdown_table(
            decisions,
            [
                "search_id",
                "setting_id",
                "output_status",
                "inference_source",
                "raw_one_sided_p",
                "holm_adjusted_one_sided_p",
                "adjusted_primary_decision",
                "claim_status",
            ],
        ),
        "",
        "Artifacts:",
        f"- [run_registry.csv](../{(OUTPUT_DIR / 'run_registry.csv').as_posix()})",
        f"- [seed_level_primary_ratios.csv](../{(OUTPUT_DIR / 'seed_level_primary_ratios.csv').as_posix()})",
        f"- [primary_decisions.csv](../{(OUTPUT_DIR / 'primary_decisions.csv').as_posix()})",
        f"- [gate_report.csv](../{(OUTPUT_DIR / 'gate_report.csv').as_posix()})",
        f"- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})",
    ]
    write_markdown(DISCUSSION_PATH, "\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = expected_registry()
    run_registry = build_run_registry(expected)
    observed = load_observed_pair_summary()
    step_metrics = load_observed_step_metrics()
    seed_ratios = seed_level_primary_ratios(step_metrics)
    decisions = build_primary_decisions(expected, observed, seed_ratios, run_registry)
    gate_report = build_gate_report(run_registry, decisions)
    run_registry.to_csv(OUTPUT_DIR / "run_registry.csv", index=False)
    seed_ratios.to_csv(OUTPUT_DIR / "seed_level_primary_ratios.csv", index=False)
    decisions.to_csv(OUTPUT_DIR / "primary_decisions.csv", index=False)
    gate_report.to_csv(OUTPUT_DIR / "gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "search_id": PHASE2_SEARCH_ID,
                "multiplicity_family": PHASE2_MULTIPLICITY_FAMILY,
                "planned_family_size": int(expected["planned_family_size"].max()),
                "alpha": ALPHA,
                "head_gain_relative_error_max": HEAD_GAIN_RELATIVE_ERROR_MAX,
                "tail_accuracy_floor_by_dataset": TAIL_ACCURACY_FLOOR_BY_DATASET,
                "primary_test": "paired per-seed log tail-output-drift ratio t-test with Holm one-sided phase-family adjustment",
                "claim_boundary": "phase2 can support held-out architecture replication or a bounded phase2 null, not retroactive phase1 selection",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(run_registry, decisions, gate_report, seed_ratios)
    print(f"saved natural negative-search phase2 evaluation to {OUTPUT_DIR}")
    print(f"observed primary rows={int(decisions['output_status'].eq('observed').sum())}/{len(decisions)}")
    print(gate_report.to_string(index=False))


if __name__ == "__main__":
    main()
