from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown
from e11_condition_geometry.statistics import ci95


RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
SELECTION_DIR = RESULT_ROOT / "validation_selection"
ANALYSIS_PLAN_DIR = RESULT_ROOT / "final_analysis_plan"
EXECUTION_PLAN_DIR = RESULT_ROOT / "final_execution_plan"
FINAL_ROOT = RESULT_ROOT / "final_claim"
OUTPUT_DIR = RESULT_ROOT / "final_evaluation"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_evaluation.md")
ALPHA = 0.05
ALL_CLASS_NONINFERIORITY_MARGIN = -0.01
EXPECTED_FINAL_SEED_COUNT = 10


PER_SEED_COLUMNS = [
    "seed",
    "recipe_family",
    "recipe_name",
    "many_bacc",
    "medium_bacc",
    "few_bacc",
    "all_bacc",
    "loss",
    "margin",
]
PAIR_COLUMNS = [
    "comparison_id",
    "seed",
    "candidate_family",
    "baseline_family",
    "few_bacc_diff",
    "all_bacc_diff",
]
DECISION_COLUMNS = [
    "comparison_id",
    "candidate_family",
    "baseline_family",
    "paired_seed_count",
    "mean_few_bacc_diff",
    "few_bacc_diff_ci95_low",
    "few_bacc_diff_ci95_high",
    "raw_one_sided_p",
    "holm_adjusted_one_sided_p",
    "primary_decision",
    "mean_all_bacc_diff",
    "all_bacc_diff_ci95_low",
    "all_bacc_diff_ci95_high",
    "all_class_guardrail",
    "claim_effect",
]
SUMMARY_COLUMNS = [
    "recipe_family",
    "frequency_group",
    "mean_bacc",
    "ci95_low",
    "ci95_high",
    "mean_loss",
    "mean_margin",
]
OCCUPANCY_COLUMNS = [
    "recipe_family",
    "seed",
    "occupancy_rows",
    "batch_few_fraction",
    "tail_probe_loss",
    "gradient_momentum_cosine",
    "drift_ratio",
]


def _one_sided_t_p_value(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return math.nan
    sd = float(arr.std(ddof=1))
    if sd <= 0.0:
        return 0.0 if float(arr.mean()) > 0.0 else 1.0
    statistic = float(arr.mean()) / (sd / math.sqrt(arr.size))
    try:
        from scipy.stats import t

        return float(t.sf(statistic, arr.size - 1))
    except Exception:
        return float(0.5 * math.erfc(statistic / math.sqrt(2.0)))


def holm_adjust(raw_p_values: pd.Series, *, family_size: int) -> pd.Series:
    adjusted = pd.Series(np.nan, index=raw_p_values.index, dtype=float)
    finite = raw_p_values.replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    running = 0.0
    for rank, (idx, p_value) in enumerate(finite.items(), start=1):
        multiplier = max(int(family_size) - rank + 1, 1)
        running = max(running, min(float(p_value) * multiplier, 1.0))
        adjusted.loc[idx] = running
    return adjusted


def _selection_gates_ready() -> tuple[bool, str]:
    gate_path = SELECTION_DIR / "gate_report.csv"
    if not gate_path.exists():
        return False, "missing validation_selection/gate_report.csv"
    gates = pd.read_csv(gate_path)
    lookup = gates.set_index("gate_id")["status"].astype(str).to_dict()
    required = {
        "TVS-1-validation-grid-complete": "pass",
        "TVS-2-family-selection": "pass",
        "TVS-5-occupancy-logging-complete": "pass",
        "TVS-3-final-seed-quarantine": "pass",
        "TVS-4-final-run-plan": "ready",
    }
    evidence = "; ".join(f"{gate}={lookup.get(gate, 'missing')}" for gate in required)
    return all(lookup.get(gate) == status for gate, status in required.items()), evidence


def build_run_registry(final_plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required_files = [
        "train_trace.csv",
        "class_metrics.csv",
        "group_metrics.csv",
        "summary.csv",
        "pair_summary.csv",
        "occupancy_trace.csv",
        "config.json",
        "setting_metadata.csv",
    ]
    for row in final_plan.sort_values("recipe_family").itertuples(index=False):
        family = str(row.recipe_family)
        selected_setting_id = "" if pd.isna(row.selected_setting_id) else str(row.selected_setting_id)
        planned_dir = FINAL_ROOT / family
        present = {name: (planned_dir / name).exists() for name in required_files}
        group_rows = 0
        group_seed_count = 0
        occupancy_rows = 0
        occupancy_seed_count = 0
        if present["group_metrics.csv"]:
            group_metrics = pd.read_csv(planned_dir / "group_metrics.csv")
            group_rows = int(len(group_metrics))
            group_seed_count = int(group_metrics["seed"].nunique()) if "seed" in group_metrics else 0
        if present["occupancy_trace.csv"]:
            occupancy = pd.read_csv(planned_dir / "occupancy_trace.csv")
            occupancy_rows = int(len(occupancy))
            occupancy_seed_count = int(occupancy["seed"].nunique()) if "seed" in occupancy else 0
        complete = (
            bool(selected_setting_id)
            and all(present.values())
            and group_seed_count == EXPECTED_FINAL_SEED_COUNT
            and occupancy_seed_count == EXPECTED_FINAL_SEED_COUNT
        )
        rows.append(
            {
                "recipe_family": family,
                "selection_status": "selected" if selected_setting_id else "not_selected",
                "selected_setting_id": selected_setting_id,
                "selected_recipe_name": "" if pd.isna(row.selected_recipe_name) else str(row.selected_recipe_name),
                "planned_output_dir": planned_dir.as_posix(),
                "final_output_status": "complete" if complete else "missing_or_incomplete",
                "present_files": ";".join(name for name, exists in present.items() if exists),
                "missing_files": ";".join(name for name, exists in present.items() if not exists),
                "group_metric_rows": group_rows,
                "group_seed_count": group_seed_count,
                "occupancy_rows": occupancy_rows,
                "occupancy_seed_count": occupancy_seed_count,
            }
        )
    return pd.DataFrame(rows)


def build_per_seed_metrics(run_registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in run_registry[run_registry["final_output_status"].eq("complete")].itertuples(index=False):
        family = str(run.recipe_family)
        group_metrics = pd.read_csv(Path(str(run.planned_output_dir)) / "group_metrics.csv")
        for (seed, recipe_name), group in group_metrics.groupby(["seed", "recipe"], observed=True, sort=True):
            by_group = group.set_index("frequency_group")
            required_groups = {"many", "medium", "few", "all"}
            if not required_groups.issubset(set(by_group.index.astype(str))):
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "recipe_family": family,
                    "recipe_name": str(recipe_name),
                    "many_bacc": float(by_group.loc["many", "balanced_accuracy"]),
                    "medium_bacc": float(by_group.loc["medium", "balanced_accuracy"]),
                    "few_bacc": float(by_group.loc["few", "balanced_accuracy"]),
                    "all_bacc": float(by_group.loc["all", "balanced_accuracy"]),
                    "loss": float(by_group.loc["all", "loss"]),
                    "margin": float(by_group.loc["all", "mean_margin"]),
                }
            )
    return pd.DataFrame(rows, columns=PER_SEED_COLUMNS)


def build_final_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if per_seed.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for family, group in per_seed.groupby("recipe_family", sort=True):
        for frequency_group, column in (
            ("many", "many_bacc"),
            ("medium", "medium_bacc"),
            ("few", "few_bacc"),
            ("all", "all_bacc"),
        ):
            mean, low, high = ci95(group[column])
            rows.append(
                {
                    "recipe_family": str(family),
                    "frequency_group": frequency_group,
                    "mean_bacc": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_loss": float(group["loss"].mean()) if frequency_group == "all" else math.nan,
                    "mean_margin": float(group["margin"].mean()) if frequency_group == "all" else math.nan,
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_occupancy_summary(run_registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in run_registry[run_registry["final_output_status"].eq("complete")].itertuples(index=False):
        family = str(run.recipe_family)
        occupancy = pd.read_csv(Path(str(run.planned_output_dir)) / "occupancy_trace.csv")
        for seed, group in occupancy.groupby("seed", observed=True, sort=True):
            rows.append(
                {
                    "recipe_family": family,
                    "seed": int(seed),
                    "occupancy_rows": int(len(group)),
                    "batch_few_fraction": float(group["batch_few_fraction"].mean()),
                    "tail_probe_loss": float(group["tail_probe_loss"].mean()),
                    "gradient_momentum_cosine": float(group["gradient_momentum_cosine"].mean())
                    if "gradient_momentum_cosine" in group
                    else math.nan,
                    "drift_ratio": float(group["ns_tail_output_drift_sq_ratio_vs_fro"].mean()),
                }
            )
    return pd.DataFrame(rows, columns=OCCUPANCY_COLUMNS)


def build_paired_comparisons(per_seed: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    indexed = per_seed.set_index(["recipe_family", "seed"])
    rows: list[dict[str, object]] = []
    for comparison in comparisons.itertuples(index=False):
        candidate = str(comparison.candidate_family)
        baseline = str(comparison.baseline_family)
        candidate_seeds = set(per_seed[per_seed["recipe_family"].eq(candidate)]["seed"])
        baseline_seeds = set(per_seed[per_seed["recipe_family"].eq(baseline)]["seed"])
        for seed in sorted(candidate_seeds & baseline_seeds):
            c = indexed.loc[(candidate, seed)]
            b = indexed.loc[(baseline, seed)]
            rows.append(
                {
                    "comparison_id": str(comparison.comparison_id),
                    "seed": int(seed),
                    "candidate_family": candidate,
                    "baseline_family": baseline,
                    "few_bacc_diff": float(c["few_bacc"] - b["few_bacc"]),
                    "all_bacc_diff": float(c["all_bacc"] - b["all_bacc"]),
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_primary_decisions(paired: pd.DataFrame, comparisons: pd.DataFrame, *, complete_family: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison in comparisons.itertuples(index=False):
        comparison_id = str(comparison.comparison_id)
        group = paired[paired["comparison_id"].eq(comparison_id)] if not paired.empty else pd.DataFrame()
        few_mean, few_low, few_high = ci95(group["few_bacc_diff"]) if not group.empty else (math.nan, math.nan, math.nan)
        all_mean, all_low, all_high = ci95(group["all_bacc_diff"]) if not group.empty else (math.nan, math.nan, math.nan)
        raw_p = _one_sided_t_p_value(group["few_bacc_diff"]) if len(group) == EXPECTED_FINAL_SEED_COUNT else math.nan
        guardrail = (
            "pass"
            if complete_family and math.isfinite(all_low) and all_low >= ALL_CLASS_NONINFERIORITY_MARGIN
            else "fail"
            if complete_family
            else "not_ready"
        )
        rows.append(
            {
                "comparison_id": comparison_id,
                "candidate_family": str(comparison.candidate_family),
                "baseline_family": str(comparison.baseline_family),
                "paired_seed_count": int(group["seed"].nunique()) if not group.empty else 0,
                "mean_few_bacc_diff": few_mean,
                "few_bacc_diff_ci95_low": few_low,
                "few_bacc_diff_ci95_high": few_high,
                "raw_one_sided_p": raw_p,
                "holm_adjusted_one_sided_p": math.nan,
                "primary_decision": "not_ready",
                "mean_all_bacc_diff": all_mean,
                "all_bacc_diff_ci95_low": all_low,
                "all_bacc_diff_ci95_high": all_high,
                "all_class_guardrail": guardrail,
                "claim_effect": "final paired seed family incomplete",
            }
        )
    frame = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    frame["holm_adjusted_one_sided_p"] = holm_adjust(frame["raw_one_sided_p"], family_size=len(comparisons))
    adjusted_pass = (
        complete_family
        & frame["paired_seed_count"].eq(EXPECTED_FINAL_SEED_COUNT)
        & frame["mean_few_bacc_diff"].gt(0.0)
        & frame["holm_adjusted_one_sided_p"].le(ALPHA)
    )
    frame.loc[adjusted_pass, "primary_decision"] = "primary_superiority_adjusted"
    frame.loc[adjusted_pass, "claim_effect"] = "candidate beats baseline on few balanced accuracy after Holm adjustment"
    complete_not_pass = complete_family & ~adjusted_pass
    frame.loc[complete_not_pass, "primary_decision"] = "not_primary_superiority_adjusted"
    frame.loc[complete_not_pass, "claim_effect"] = "no adjusted few balanced-accuracy superiority for this pair"
    return frame


def build_claim_gate_report(run_registry: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    selection_ready, selection_evidence = _selection_gates_ready()
    selected_count = int(run_registry["selection_status"].eq("selected").sum())
    complete_count = int(run_registry["final_output_status"].eq("complete").sum())
    complete_family = selected_count == 6 and complete_count == 6
    primary_pass = complete_family and decisions["primary_decision"].eq("primary_superiority_adjusted").all()
    guardrail_pass = complete_family and decisions["all_class_guardrail"].eq("pass").all()
    if not selection_ready or not complete_family:
        claim_state = "not_ready"
        claim_evidence = f"selected={selected_count}/6; final_complete={complete_count}/6"
    elif primary_pass and guardrail_pass:
        claim_state = "clean_cifar100lt_resnet18_win"
        claim_evidence = "all four primary comparisons pass Holm and all-class guardrails"
    elif primary_pass:
        claim_state = "few_all_tradeoff"
        claim_evidence = "primary few-class family passes but at least one all-class guardrail fails"
    else:
        claim_state = "negative_or_underpowered_boundary"
        claim_evidence = "primary Holm family does not support a clean tuned Muon improvement"
    return pd.DataFrame(
        [
            {
                "gate_id": "TFE-1-evaluator-implemented",
                "status": "pass",
                "evidence": "scripts/e11_evaluate_cifar100_resnet_lt_tuned_benchmark_final.py writes the fixed final evaluator tables",
                "claim_state": claim_state,
            },
            {
                "gate_id": "TFE-2-selection-gates-ready",
                "status": "pass" if selection_ready else "not_ready",
                "evidence": selection_evidence,
                "claim_state": claim_state,
            },
            {
                "gate_id": "TFE-3-final-output-completeness",
                "status": "pass" if complete_family else "not_ready",
                "evidence": f"{complete_count}/6 selected recipe families have complete 10-seed final outputs",
                "claim_state": claim_state,
            },
            {
                "gate_id": "TFE-4-primary-holm-family",
                "status": "pass" if primary_pass else "not_ready" if not complete_family else "fail",
                "evidence": f"{int(decisions['primary_decision'].eq('primary_superiority_adjusted').sum())}/4 primary comparisons pass Holm",
                "claim_state": claim_state,
            },
            {
                "gate_id": "TFE-5-all-class-guardrail",
                "status": "pass" if guardrail_pass else "not_ready" if not complete_family else "fail",
                "evidence": f"{int(decisions['all_class_guardrail'].eq('pass').sum())}/4 all-class guardrails pass",
                "claim_state": claim_state,
            },
            {
                "gate_id": "TFE-6-final-claim-state",
                "status": claim_state,
                "evidence": claim_evidence,
                "claim_state": claim_state,
            },
        ]
    )


def write_discussion(
    run_registry: pd.DataFrame,
    gate_report: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    occupancy: pd.DataFrame,
) -> None:
    claim_state = str(gate_report[gate_report["gate_id"].eq("TFE-6-final-claim-state")]["claim_state"].iloc[0])
    complete_count = int(run_registry["final_output_status"].eq("complete").sum())
    lines = [
        "# E11 CIFAR-100-LT Tuned Benchmark Final Evaluation",
        "",
        "This generated evaluator is the fixed post-final analysis path for the",
        "registered tuned benchmark. It reads only the selected final-claim recipe",
        "directories, keeps all four Muon-vs-baseline primary comparisons in the",
        "Holm family, and reports the all-class noninferiority guardrail before any",
        "benchmark wording is allowed.",
        "",
        f"Current claim state: `{claim_state}`.",
        f"Current final family coverage: `{complete_count}/6` complete recipe families.",
        "",
        "## Gate Report",
        "",
        markdown_table(gate_report, ["gate_id", "status", "evidence", "claim_state"]),
        "",
        "## Final Run Registry",
        "",
        markdown_table(
            run_registry,
            [
                "recipe_family",
                "selection_status",
                "final_output_status",
                "group_seed_count",
                "occupancy_seed_count",
                "missing_files",
            ],
        ),
        "",
        "## Primary Decisions",
        "",
        markdown_table(
            decisions,
            [
                "comparison_id",
                "paired_seed_count",
                "mean_few_bacc_diff",
                "holm_adjusted_one_sided_p",
                "primary_decision",
                "mean_all_bacc_diff",
                "all_class_guardrail",
            ],
        ),
        "",
        "## Final Summary Preview",
        "",
        markdown_table(summary.head(12), ["recipe_family", "frequency_group", "mean_bacc", "ci95_low", "ci95_high"]),
        "",
        "## Occupancy Summary Preview",
        "",
        markdown_table(occupancy.head(12), ["recipe_family", "seed", "occupancy_rows", "batch_few_fraction", "tail_probe_loss", "gradient_momentum_cosine", "drift_ratio"]),
        "",
        "Artifacts:",
        f"- [run_registry.csv](../{(OUTPUT_DIR / 'run_registry.csv').as_posix()})",
        f"- [per_seed_final_metrics.csv](../{(OUTPUT_DIR / 'per_seed_final_metrics.csv').as_posix()})",
        f"- [paired_primary_comparisons.csv](../{(OUTPUT_DIR / 'paired_primary_comparisons.csv').as_posix()})",
        f"- [primary_decisions.csv](../{(OUTPUT_DIR / 'primary_decisions.csv').as_posix()})",
        f"- [final_summary.csv](../{(OUTPUT_DIR / 'final_summary.csv').as_posix()})",
        f"- [occupancy_summary.csv](../{(OUTPUT_DIR / 'occupancy_summary.csv').as_posix()})",
        f"- [claim_gate_report.csv](../{(OUTPUT_DIR / 'claim_gate_report.csv').as_posix()})",
        f"- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})",
        "",
        "Blocked now: benchmark-performance wording remains unavailable until the",
        "validation-selection gates pass, all six selected recipe families have 10",
        "paired final seeds, all four Holm-adjusted primary comparisons are evaluated,",
        "and the all-class guardrail is reported.",
    ]
    write_markdown(DISCUSSION_PATH, "\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_plan = pd.read_csv(SELECTION_DIR / "final_claim_plan.csv")
    comparisons = pd.read_csv(ANALYSIS_PLAN_DIR / "primary_comparison_family.csv")
    run_registry = build_run_registry(final_plan)
    per_seed = build_per_seed_metrics(run_registry)
    paired = build_paired_comparisons(per_seed, comparisons)
    complete_family = (
        int(run_registry["selection_status"].eq("selected").sum()) == 6
        and int(run_registry["final_output_status"].eq("complete").sum()) == 6
    )
    decisions = build_primary_decisions(paired, comparisons, complete_family=complete_family)
    summary = build_final_summary(per_seed)
    occupancy = build_occupancy_summary(run_registry)
    gate_report = build_claim_gate_report(run_registry, decisions)
    run_registry.to_csv(OUTPUT_DIR / "run_registry.csv", index=False)
    per_seed.to_csv(OUTPUT_DIR / "per_seed_final_metrics.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "paired_primary_comparisons.csv", index=False)
    decisions.to_csv(OUTPUT_DIR / "primary_decisions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "final_summary.csv", index=False)
    occupancy.to_csv(OUTPUT_DIR / "occupancy_summary.csv", index=False)
    gate_report.to_csv(OUTPUT_DIR / "claim_gate_report.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "selection_plan": (SELECTION_DIR / "final_claim_plan.csv").as_posix(),
                "analysis_plan": ANALYSIS_PLAN_DIR.as_posix(),
                "execution_plan": EXECUTION_PLAN_DIR.as_posix(),
                "final_root": FINAL_ROOT.as_posix(),
                "alpha": ALPHA,
                "primary_family_size": int(len(comparisons)),
                "expected_final_seed_count": EXPECTED_FINAL_SEED_COUNT,
                "all_class_noninferiority_margin": ALL_CLASS_NONINFERIORITY_MARGIN,
                "primary_test": "paired one-sided t-test on candidate-minus-baseline few balanced accuracy with Holm adjustment",
                "claim_boundary": "no benchmark-performance wording until TFE gates are complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(run_registry, gate_report, decisions, summary, occupancy)
    print(f"saved tuned benchmark final evaluation to {OUTPUT_DIR} and {DISCUSSION_PATH}")
    print(gate_report.to_string(index=False))


if __name__ == "__main__":
    main()
