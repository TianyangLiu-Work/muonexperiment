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
POWER_DIR = RESULT_ROOT / "final_power_audit"
VARIANCE_DIR = RESULT_ROOT / "variance_prior_audit"
OUTPUT_DIR = RESULT_ROOT / "final_analysis_plan"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_analysis_plan.md")


def final_outputs_present() -> bool:
    final_root = RESULT_ROOT / "final_claim"
    if not final_root.exists():
        return False
    return any(final_root.rglob("*"))


def build_analysis_input_contract() -> pd.DataFrame:
    final_plan = pd.read_csv(SELECTION_DIR / "final_claim_plan.csv")
    seed_split = pd.read_csv(PROTOCOL_DIR / "seed_split_contract.csv")
    final_seed_set = str(seed_split[seed_split["split_id"].eq("final_claim")]["seed_set"].iloc[0])
    rows = []
    for row in final_plan.to_dict("records"):
        rows.append(
            {
                "input_id": f"FIN-IN-{row['recipe_family']}",
                "recipe_family": row["recipe_family"],
                "selection_status": row["final_status"],
                "selected_setting_id": row["selected_setting_id"],
                "final_seed_set": row["final_seed_set"] if str(row["final_seed_set"]) != "nan" else final_seed_set,
                "required_files": "summary.csv; group_metrics.csv; class_metrics.csv; train_trace.csv; occupancy_trace.csv",
                "pairing_key": "seed",
                "claim_boundary": "required for final analysis only after all six families are selected and run",
            }
        )
    return pd.DataFrame(rows)


def build_metric_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_id": "MET-1-primary-few-balanced-accuracy",
                "frequency_group": "few",
                "statistic": "balanced_accuracy",
                "role": "primary superiority metric",
                "direction": "candidate_minus_baseline higher is better",
                "claim_gate": "TB-2-primary-few-accuracy",
            },
            {
                "metric_id": "MET-2-all-balanced-accuracy-guardrail",
                "frequency_group": "all",
                "statistic": "balanced_accuracy",
                "role": "all-class noninferiority guardrail",
                "direction": "candidate_minus_best_baseline must not collapse below -0.01",
                "claim_gate": "TB-3-no-all-class-collapse",
            },
            {
                "metric_id": "MET-3-reporting-many-medium",
                "frequency_group": "many;medium",
                "statistic": "balanced_accuracy;loss;margin",
                "role": "required full reporting surface",
                "direction": "descriptive with confidence intervals",
                "claim_gate": "TB-4-full-reporting-surface",
            },
            {
                "metric_id": "MET-4-state-distribution-occupancy",
                "frequency_group": "all",
                "statistic": "batch_few_fraction;tail_probe_loss;gradient_momentum_cosine;ns_vs_fro_drift_ratio",
                "role": "mechanism-transfer context, not a performance gate by itself",
                "direction": "descriptive with missingness gate",
                "claim_gate": "TB-7-state-distribution-occupancy",
            },
        ]
    )


def build_comparison_family() -> pd.DataFrame:
    comparisons = pd.read_csv(POWER_DIR / "primary_comparison_plan.csv")
    rows = []
    for row in comparisons.to_dict("records"):
        rows.append(
            {
                "comparison_id": row["comparison_id"],
                "candidate_family": row["candidate_family"],
                "baseline_family": row["baseline_family"],
                "metric_id": "MET-1-primary-few-balanced-accuracy",
                "paired_unit": "seed",
                "test": "paired t-test on per-seed candidate-minus-baseline few balanced-accuracy differences",
                "null_hypothesis": "mean_diff <= 0",
                "alternative": "mean_diff > 0",
                "multiplicity_family": row["multiplicity_family"],
                "adjustment": "Holm step-down across the four primary comparisons",
            }
        )
    return pd.DataFrame(rows)


def build_adjustment_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "adjustment_id": "ADJ-1-primary-holm",
                "scope": "four Muon-vs-tuned-baseline few-balanced-accuracy comparisons",
                "method": "Holm step-down familywise error control at alpha=0.05",
                "decision_rule": "claim primary few-class improvement only if both Muon families beat both tuned AdamW and tuned SGD where applicable after adjustment",
                "forbidden_shortcut": "claiming the best-looking Muon-vs-baseline pair without the full Holm family",
            },
            {
                "adjustment_id": "ADJ-2-all-guardrail",
                "scope": "all-class balanced-accuracy candidate-minus-best-tuned-baseline difference",
                "method": "paired CI noninferiority check with margin -0.01",
                "decision_rule": "clean benchmark win requires the lower CI endpoint to be at least -0.01",
                "forbidden_shortcut": "describing a few-class gain as clean if all-class balanced accuracy collapses",
            },
            {
                "adjustment_id": "ADJ-3-negative-boundary",
                "scope": "all completed final recipes",
                "method": "interpret negative, underpowered, or tradeoff outcomes using the pre-output power and variance-prior audits",
                "decision_rule": "publish negative or underpowered final outcomes instead of suppressing them",
                "forbidden_shortcut": "using negative tuned outcomes only as private tuning feedback",
            },
        ]
    )


def build_reporting_schema() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_id": "TAB-1-per-seed-final-metrics",
                "required_columns": "seed;recipe_family;recipe_name;many_bacc;medium_bacc;few_bacc;all_bacc;loss;margin",
                "granularity": "one row per final seed and recipe",
                "claim_role": "audit trail for paired tests and dropped-seed checks",
            },
            {
                "table_id": "TAB-2-paired-primary-comparisons",
                "required_columns": "comparison_id;seed;candidate_family;baseline_family;few_bacc_diff;all_bacc_diff",
                "granularity": "one row per paired seed and primary comparison",
                "claim_role": "input to Holm-adjusted superiority and all-class guardrail decisions",
            },
            {
                "table_id": "TAB-3-final-summary",
                "required_columns": "recipe_family;frequency_group;mean_bacc;ci95_low;ci95_high;mean_loss;mean_margin",
                "granularity": "one row per recipe family and frequency group",
                "claim_role": "main benchmark reporting surface if all gates pass",
            },
            {
                "table_id": "TAB-4-occupancy-summary",
                "required_columns": "recipe_family;seed;occupancy_rows;batch_few_fraction;tail_probe_loss;gradient_momentum_cosine;drift_ratio",
                "granularity": "one row per final seed and recipe",
                "claim_role": "state-distribution context for why local Muon evidence transfers or fails to transfer",
            },
        ]
    )


def build_claim_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_state": "not_ready",
                "trigger": "any validation family, occupancy trace, or final seed recipe is incomplete",
                "allowed_wording": "registered tuned benchmark analysis plan only",
                "blocked_wording": "final optimizer-performance result",
            },
            {
                "claim_state": "clean_cifar100lt_resnet18_win",
                "trigger": "primary Holm family passes and all-class guardrail passes with complete reporting",
                "allowed_wording": "CIFAR-100-LT ResNet18 tuned final-performance improvement under the registered protocol",
                "blocked_wording": "broad long-tail optimizer superiority",
            },
            {
                "claim_state": "few_all_tradeoff",
                "trigger": "few-class primary passes but all-class guardrail fails",
                "allowed_wording": "few/all tradeoff under the registered protocol",
                "blocked_wording": "clean benchmark win",
            },
            {
                "claim_state": "negative_or_underpowered_boundary",
                "trigger": "Muon does not beat tuned baselines, or observed effect is below the pre-output MDE sensitivity grid",
                "allowed_wording": "negative or underpowered tuned benchmark boundary",
                "blocked_wording": "Muon has been ruled out broadly or silently dropping the tuned result",
            },
            {
                "claim_state": "protocol_violation",
                "trigger": "final outputs exist before all selection and analysis gates are ready",
                "allowed_wording": "protocol violation; rows excluded from tuned-performance claims",
                "blocked_wording": "using leaked final rows for selection or claim repair",
            },
        ]
    )


def build_gate_matrix() -> pd.DataFrame:
    final_plan = pd.read_csv(SELECTION_DIR / "final_claim_plan.csv")
    power_config = json.loads((POWER_DIR / "config.json").read_text(encoding="utf-8"))
    variance_config = json.loads((VARIANCE_DIR / "config.json").read_text(encoding="utf-8"))
    final_present = final_outputs_present()
    complete_families = int(final_plan["final_status"].astype(str).eq("ready_for_final_run").sum())
    return pd.DataFrame(
        [
            {
                "gate_id": "FAP-1-final-quarantine",
                "status": "fail" if final_present else "pass",
                "evidence": "no final_claim files exist" if not final_present else "final_claim files exist",
                "claim_effect": "analysis plan remains pre-final",
            },
            {
                "gate_id": "FAP-2-family-selection-readiness",
                "status": "pass" if complete_families == len(final_plan) else "not_ready",
                "evidence": f"{complete_families}/{len(final_plan)} families ready for final run",
                "claim_effect": "final analysis cannot run until one recipe per family is selected",
            },
            {
                "gate_id": "FAP-3-power-audit-linked",
                "status": "pass" if power_config.get("final_seed_count") == 10 else "fail",
                "evidence": "final power audit locks 10 final paired seeds",
                "claim_effect": "final analysis uses the pre-output MDE contract",
            },
            {
                "gate_id": "FAP-4-variance-prior-linked",
                "status": "pass" if not variance_config.get("final_seed_outputs_inspected", True) else "fail",
                "evidence": "variance-prior audit is pre-final and does not inspect final seed outputs",
                "claim_effect": "MDE sensitivity remains separated from final outcomes",
            },
            {
                "gate_id": "FAP-5-primary-family-fixed",
                "status": "pass",
                "evidence": "four Muon-vs-baseline primary comparisons are fixed before final outputs",
                "claim_effect": "blocks cherry-picked final comparison wording",
            },
            {
                "gate_id": "FAP-6-reporting-schema-fixed",
                "status": "pass",
                "evidence": "per-seed, paired-comparison, final-summary, and occupancy tables are predeclared",
                "claim_effect": "blocks selective metric reporting",
            },
        ]
    )


def write_discussion(
    inputs: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    adjustments: pd.DataFrame,
    schema: pd.DataFrame,
    claims: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Final Analysis Plan

This generated audit fixes the statistical analysis surface for the registered tuned benchmark before final seeds are available. It does not inspect final outputs and does not authorize a benchmark claim while validation selection remains incomplete.

## Gate Matrix

{markdown_table(gates, ["gate_id", "status", "evidence", "claim_effect"])}

## Analysis Input Contract

{markdown_table(inputs, ["input_id", "recipe_family", "selection_status", "selected_setting_id", "final_seed_set", "required_files", "pairing_key", "claim_boundary"])}

## Metric Contract

{markdown_table(metrics, ["metric_id", "frequency_group", "statistic", "role", "direction", "claim_gate"])}

## Primary Comparison Family

{markdown_table(comparisons, ["comparison_id", "candidate_family", "baseline_family", "metric_id", "paired_unit", "test", "adjustment"])}

## Multiplicity And Guardrail Plan

{markdown_table(adjustments, ["adjustment_id", "scope", "method", "decision_rule", "forbidden_shortcut"])}

## Reporting Schema

{markdown_table(schema, ["table_id", "required_columns", "granularity", "claim_role"])}

## Claim Ladder

{markdown_table(claims, ["claim_state", "trigger", "allowed_wording", "blocked_wording"])}

## Boundary

Allowed now: cite this as a pre-final statistical analysis plan for the tuned benchmark.

Blocked now: running final analysis, changing the primary comparison family, or claiming tuned optimizer performance before validation selection, occupancy logging, and final seed rows are complete.

Artifacts:
- [analysis_input_contract.csv](../{(OUTPUT_DIR / 'analysis_input_contract.csv').as_posix()})
- [metric_contract.csv](../{(OUTPUT_DIR / 'metric_contract.csv').as_posix()})
- [primary_comparison_family.csv](../{(OUTPUT_DIR / 'primary_comparison_family.csv').as_posix()})
- [multiplicity_and_guardrail_plan.csv](../{(OUTPUT_DIR / 'multiplicity_and_guardrail_plan.csv').as_posix()})
- [reporting_schema.csv](../{(OUTPUT_DIR / 'reporting_schema.csv').as_posix()})
- [claim_ladder.csv](../{(OUTPUT_DIR / 'claim_ladder.csv').as_posix()})
- [gate_matrix.csv](../{(OUTPUT_DIR / 'gate_matrix.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = build_analysis_input_contract()
    metrics = build_metric_contract()
    comparisons = build_comparison_family()
    adjustments = build_adjustment_plan()
    schema = build_reporting_schema()
    claims = build_claim_ladder()
    gates = build_gate_matrix()

    inputs.to_csv(OUTPUT_DIR / "analysis_input_contract.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metric_contract.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "primary_comparison_family.csv", index=False)
    adjustments.to_csv(OUTPUT_DIR / "multiplicity_and_guardrail_plan.csv", index=False)
    schema.to_csv(OUTPUT_DIR / "reporting_schema.csv", index=False)
    claims.to_csv(OUTPUT_DIR / "claim_ladder.csv", index=False)
    gates.to_csv(OUTPUT_DIR / "gate_matrix.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "purpose": "pre-final statistical analysis plan for tuned benchmark final claims",
                "final_outputs_present": final_outputs_present(),
                "final_outputs_inspected": False,
                "primary_family_size": len(comparisons),
                "analysis_scope": "plan only; final benchmark claim remains blocked until validation and final seeds complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(inputs, metrics, comparisons, adjustments, schema, claims, gates)
    invalid = set(gates["status"].astype(str)) - {"pass", "not_ready"}
    if invalid:
        raise AssertionError(f"unexpected final-analysis plan gate status: {invalid}")
    print(f"saved tuned benchmark final analysis plan to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
