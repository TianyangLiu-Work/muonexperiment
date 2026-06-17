from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import fmt, markdown_table, write_markdown


PROTOCOL_DIR = Path("results/e11_cifar100_resnet_lt_tuned_benchmark_protocol")
RESULT_ROOT = Path("results/e11_cifar100_resnet_lt_tuned_benchmark")
SELECTION_DIR = RESULT_ROOT / "validation_selection"
OUTPUT_DIR = RESULT_ROOT / "final_power_audit"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_power_audit.md")


def t_critical(df: int, two_sided_alpha: float) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(1.0 - two_sided_alpha / 2.0, df))
    except Exception:
        return NormalDist().inv_cdf(1.0 - two_sided_alpha / 2.0)


def parse_seed_count(seed_set: str) -> int:
    if ".." not in seed_set:
        raise ValueError(f"unsupported seed_set format: {seed_set}")
    start, end = (int(part) for part in seed_set.split("..", maxsplit=1))
    return end - start + 1


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "seed_split": pd.read_csv(PROTOCOL_DIR / "seed_split_contract.csv"),
        "recipe_grid": pd.read_csv(PROTOCOL_DIR / "recipe_grid.csv"),
        "acceptance_gates": pd.read_csv(PROTOCOL_DIR / "acceptance_gates.csv"),
        "family_selection": pd.read_csv(SELECTION_DIR / "family_selection.csv"),
        "final_plan": pd.read_csv(SELECTION_DIR / "final_claim_plan.csv"),
        "selection_gates": pd.read_csv(SELECTION_DIR / "gate_report.csv"),
    }


def build_final_family_design(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    seed_split = inputs["seed_split"]
    final_seed_row = seed_split[seed_split["split_id"].eq("final_claim")].iloc[0]
    final_seed_set = str(final_seed_row["seed_set"])
    final_seed_count = parse_seed_count(final_seed_set)
    recipe_grid = inputs["recipe_grid"]
    family_selection = inputs["family_selection"]
    final_plan = inputs["final_plan"]
    selection_gate_lookup = inputs["selection_gates"].set_index("gate_id")["status"].astype(str).to_dict()
    final_outputs = sorted(
        path.as_posix()
        for path in RESULT_ROOT.glob("final*")
        if path.name != "final_power_audit"
    )

    rows = []
    for recipe in recipe_grid.itertuples(index=False):
        family = str(recipe.recipe_family)
        selected_row = family_selection[family_selection["recipe_family"].eq(family)].iloc[0]
        final_row = final_plan[final_plan["recipe_family"].eq(family)].iloc[0]
        rows.append(
            {
                "recipe_family": family,
                "optimizer": str(recipe.optimizer),
                "claim_role": str(recipe.claim_role),
                "validation_selection_status": str(selected_row.selection_status),
                "final_status": str(final_row.final_status),
                "final_seed_set": final_seed_set,
                "final_seed_count": final_seed_count,
                "final_output_status": "absent" if not final_outputs else "present",
                "selection_gate_status": "; ".join(
                    f"{gate}={status}" for gate, status in sorted(selection_gate_lookup.items())
                ),
            }
        )
    return pd.DataFrame(rows)


def build_primary_comparison_plan(final_seed_count: int) -> pd.DataFrame:
    rows = []
    for muon_family in ("ns_muon_matrix_tuned", "ns_muon_cb_tuned"):
        for baseline_family in ("adamw_ce_tuned", "sgd_momentum_ce_tuned"):
            rows.append(
                {
                    "comparison_id": f"{muon_family}_vs_{baseline_family}",
                    "candidate_family": muon_family,
                    "baseline_family": baseline_family,
                    "metric": "few_balanced_accuracy_diff",
                    "direction": "candidate_minus_baseline",
                    "final_seed_count": final_seed_count,
                    "multiplicity_family": "primary_few_accuracy_holm_family",
                    "family_size": 4,
                    "claim_gate": "TB-2-primary-few-accuracy",
                    "pass_rule": "paired mean diff CI lower endpoint above zero after Holm adjustment",
                }
            )
    return pd.DataFrame(rows)


def build_paired_diff_mde(final_seed_count: int, family_size: int) -> pd.DataFrame:
    rows = []
    for alpha_scope, family, two_sided_alpha in (
        ("raw_single_comparison", 1, 0.05),
        ("holm_bonferroni_worst_case", family_size, 0.05 / family_size),
    ):
        crit = t_critical(final_seed_count - 1, two_sided_alpha)
        for sd in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1):
            rows.append(
                {
                    "alpha_scope": alpha_scope,
                    "family_size": family,
                    "final_seed_count": final_seed_count,
                    "two_sided_alpha": two_sided_alpha,
                    "paired_diff_sd": sd,
                    "t_critical": crit,
                    "minimum_detectable_paired_mean_diff": crit * sd / math.sqrt(final_seed_count),
                    "claim_interpretation": (
                        "final Muon-vs-baseline improvement is informative only for paired few-balanced-accuracy "
                        "effects at or above this scale under the assumed across-seed paired-diff SD"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_all_class_guardrail(final_seed_count: int, family_size: int) -> pd.DataFrame:
    rows = []
    margin = -0.01
    for alpha_scope, family, two_sided_alpha in (
        ("raw_single_comparison", 1, 0.05),
        ("holm_bonferroni_worst_case", family_size, 0.05 / family_size),
    ):
        crit = t_critical(final_seed_count - 1, two_sided_alpha)
        for sd in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1):
            rows.append(
                {
                    "alpha_scope": alpha_scope,
                    "family_size": family,
                    "final_seed_count": final_seed_count,
                    "two_sided_alpha": two_sided_alpha,
                    "paired_diff_sd": sd,
                    "noninferiority_margin": margin,
                    "minimum_mean_all_accuracy_diff_to_pass": margin + crit * sd / math.sqrt(final_seed_count),
                    "claim_interpretation": (
                        "the all-class guardrail is passed only when the paired mean all-balanced-accuracy "
                        "difference is above this value, so few-class gains are not bought by broad collapse"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_interpretation_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "TB-PWR-1-validation-incomplete",
                "evidence_condition": "validation grid incomplete or family selection not_ready",
                "allowed_wording": "registered benchmark power audit only; no final-performance result",
                "blocked_wording": "tuned optimizer performance claim",
            },
            {
                "case_id": "TB-PWR-2-positive-primary",
                "evidence_condition": "all final recipes complete; Muon few-class diff beats AdamW and SGD after Holm adjustment; all-class guardrail passes",
                "allowed_wording": "CIFAR-100-LT ResNet18 practical Muon improvement under the registered tuned protocol",
                "blocked_wording": "broad long-tail optimizer superiority across datasets or architectures",
            },
            {
                "case_id": "TB-PWR-3-underpowered-null",
                "evidence_condition": "final mean few-class differences are below the audited MDE for observed paired-diff SD",
                "allowed_wording": "underpowered or small-effect tuned benchmark boundary",
                "blocked_wording": "Muon has been ruled out as a practical optimizer",
            },
            {
                "case_id": "TB-PWR-4-negative-at-detectable-scale",
                "evidence_condition": "Muon fails tuned baselines at or above the audited detectable-effect scale",
                "allowed_wording": "negative tuned-performance boundary under the registered CIFAR-100-LT ResNet18 protocol",
                "blocked_wording": "suppressing negative tuned outcomes while keeping optimizer-performance motivation",
            },
            {
                "case_id": "TB-PWR-5-tradeoff-only",
                "evidence_condition": "few-class gain passes but all-class guardrail fails",
                "allowed_wording": "few/all tradeoff result, not a clean practical optimizer improvement",
                "blocked_wording": "unqualified benchmark win",
            },
            {
                "case_id": "TB-PWR-6-reporting-incomplete",
                "evidence_condition": "per-seed rows, many/medium/few/all metrics, or paired differences are missing",
                "allowed_wording": "appendix-only incomplete benchmark progress",
                "blocked_wording": "main benchmark claim",
            },
        ]
    )


def build_outcome_state_machine() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_id": "TB-PWR-S1-not-ready",
                "trigger": "selection gates TVS-1/TVS-2/TVS-4 are not_ready",
                "claim_state": "not_ready",
                "required_action": "run validation grid and selection audit before final seeds",
            },
            {
                "state_id": "TB-PWR-S2-final-quarantine-broken",
                "trigger": "any final output exists before validation family selection is complete",
                "claim_state": "protocol_violation",
                "required_action": "do not use those final rows for a tuned-performance claim",
            },
            {
                "state_id": "TB-PWR-S3-final-family-complete-positive",
                "trigger": "all selected recipes have 10 paired final seeds and primary/guardrail gates pass",
                "claim_state": "cifar100lt_resnet18_benchmark_claim_eligible",
                "required_action": "report adjusted comparisons, per-group metrics, CIs, and exact scope",
            },
            {
                "state_id": "TB-PWR-S4-final-family-complete-underpowered",
                "trigger": "primary comparisons fail but observed effect scale is below audited MDE",
                "claim_state": "underpowered_tuned_boundary",
                "required_action": "state MDE floor and add a larger preregistered final family before broad claims",
            },
            {
                "state_id": "TB-PWR-S5-final-family-complete-negative",
                "trigger": "Muon fails baselines at or above audited MDE or all-class guardrail fails",
                "claim_state": "negative_or_tradeoff_tuned_boundary",
                "required_action": "preserve negative result and keep paper on the local mechanism path",
            },
        ]
    )


def write_discussion(
    final_design: pd.DataFrame,
    comparison_plan: pd.DataFrame,
    paired_mde: pd.DataFrame,
    guardrail: pd.DataFrame,
    ladder: pd.DataFrame,
    states: pd.DataFrame,
) -> None:
    final_seed_count = int(final_design["final_seed_count"].iloc[0])
    family_size = int(comparison_plan["family_size"].iloc[0])
    mde_reference = paired_mde[
        paired_mde["alpha_scope"].eq("holm_bonferroni_worst_case")
        & paired_mde["paired_diff_sd"].eq(0.03)
    ].iloc[0]
    guardrail_reference = guardrail[
        guardrail["alpha_scope"].eq("holm_bonferroni_worst_case")
        & guardrail["paired_diff_sd"].eq(0.03)
    ].iloc[0]
    complete_families = int(final_design["validation_selection_status"].eq("selected").sum())
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Power Audit

This generated audit is a pre-output detectable-effect contract for the tuned
CIFAR-100-LT ResNet18 benchmark path. It reads the registered validation/final
seed split, validation-only selection audit, and acceptance gates, but it does
not inspect final seed outputs. Its purpose is to fix the paired-seed power,
Holm family, all-class-collapse guardrail, and final claim ladder before any
tuned benchmark final rows exist.

Current validation-selected families: {complete_families}/{len(final_design)}.
The registered final split has {final_seed_count} paired seeds and the primary
few-class improvement family has {family_size} Holm-adjusted Muon-vs-baseline
comparisons. With paired-diff SD 0.03, the Holm worst-case detectable few-class
balanced-accuracy gain is {fmt(mde_reference['minimum_detectable_paired_mean_diff'])}.
The all-class guardrail with margin -0.01 needs mean all-class diff at least
{fmt(guardrail_reference['minimum_mean_all_accuracy_diff_to_pass'])} at the same
SD before a clean benchmark win can be claimed.

## Final Family Design

{markdown_table(final_design, ["recipe_family", "optimizer", "validation_selection_status", "final_status", "final_seed_set", "final_seed_count", "final_output_status"])}

## Primary Comparison Plan

{markdown_table(comparison_plan, ["comparison_id", "candidate_family", "baseline_family", "metric", "family_size", "pass_rule"])}

## Paired Few-Class MDE

{markdown_table(paired_mde, ["alpha_scope", "family_size", "final_seed_count", "paired_diff_sd", "minimum_detectable_paired_mean_diff", "claim_interpretation"])}

## All-Class Guardrail

{markdown_table(guardrail, ["alpha_scope", "family_size", "paired_diff_sd", "noninferiority_margin", "minimum_mean_all_accuracy_diff_to_pass", "claim_interpretation"])}

## Interpretation Ladder

{markdown_table(ladder, ["case_id", "evidence_condition", "allowed_wording", "blocked_wording"])}

## Outcome State Machine

{markdown_table(states, ["state_id", "trigger", "claim_state", "required_action"])}

Artifacts:
- [final_family_design.csv](../{(OUTPUT_DIR / 'final_family_design.csv').as_posix()})
- [primary_comparison_plan.csv](../{(OUTPUT_DIR / 'primary_comparison_plan.csv').as_posix()})
- [paired_diff_mde.csv](../{(OUTPUT_DIR / 'paired_diff_mde.csv').as_posix()})
- [all_class_guardrail_mde.csv](../{(OUTPUT_DIR / 'all_class_guardrail_mde.csv').as_posix()})
- [interpretation_ladder.csv](../{(OUTPUT_DIR / 'interpretation_ladder.csv').as_posix()})
- [outcome_state_machine.csv](../{(OUTPUT_DIR / 'outcome_state_machine.csv').as_posix()})
- [config.json](../{(OUTPUT_DIR / 'config.json').as_posix()})
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    final_design = build_final_family_design(inputs)
    final_seed_count = int(final_design["final_seed_count"].iloc[0])
    comparison_plan = build_primary_comparison_plan(final_seed_count)
    family_size = int(comparison_plan["family_size"].iloc[0])
    paired_mde = build_paired_diff_mde(final_seed_count, family_size)
    guardrail = build_all_class_guardrail(final_seed_count, family_size)
    ladder = build_interpretation_ladder()
    states = build_outcome_state_machine()

    final_design.to_csv(OUTPUT_DIR / "final_family_design.csv", index=False)
    comparison_plan.to_csv(OUTPUT_DIR / "primary_comparison_plan.csv", index=False)
    paired_mde.to_csv(OUTPUT_DIR / "paired_diff_mde.csv", index=False)
    guardrail.to_csv(OUTPUT_DIR / "all_class_guardrail_mde.csv", index=False)
    ladder.to_csv(OUTPUT_DIR / "interpretation_ladder.csv", index=False)
    states.to_csv(OUTPUT_DIR / "outcome_state_machine.csv", index=False)
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "protocol_dir": PROTOCOL_DIR.as_posix(),
                "selection_dir": SELECTION_DIR.as_posix(),
                "final_seed_set": str(final_design["final_seed_set"].iloc[0]),
                "final_seed_count": final_seed_count,
                "primary_comparison_family_size": family_size,
                "primary_metric": "few balanced accuracy paired candidate-minus-baseline diff",
                "guardrail_metric": "all balanced accuracy paired diff with noninferiority margin -0.01",
                "analysis_scope": "pre-output tuned benchmark power audit; no final seed outputs inspected",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_discussion(final_design, comparison_plan, paired_mde, guardrail, ladder, states)
    print(f"saved tuned benchmark power audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
