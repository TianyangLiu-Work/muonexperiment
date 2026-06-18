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
FINAL_ANALYSIS_DIR = RESULT_ROOT / "final_analysis_plan"
POWER_DIR = RESULT_ROOT / "final_power_audit"
OUTPUT_DIR = RESULT_ROOT / "final_robustness_plan"
DISCUSSION_PATH = Path("discussion/e11_cifar100_resnet_lt_tuned_benchmark_final_robustness_plan.md")


def final_outputs_present() -> bool:
    final_root = RESULT_ROOT / "final_claim"
    return final_root.exists() and any(path.is_file() for path in final_root.rglob("*"))


def build_robustness_test_matrix(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in comparisons.to_dict("records"):
        comparison_id = str(row["comparison_id"])
        candidate = str(row["candidate_family"])
        baseline = str(row["baseline_family"])
        rows.extend(
            [
                {
                    "robustness_id": f"RBT-{comparison_id}-paired-t",
                    "comparison_id": comparison_id,
                    "candidate_family": candidate,
                    "baseline_family": baseline,
                    "test_family": "primary_parametric",
                    "method": "paired t-test with Holm adjustment",
                    "required_input": "10 paired final seed differences for few balanced accuracy",
                    "claim_use": "primary claim gate",
                },
                {
                    "robustness_id": f"RBT-{comparison_id}-sign-flip",
                    "comparison_id": comparison_id,
                    "candidate_family": candidate,
                    "baseline_family": baseline,
                    "test_family": "nonparametric_sensitivity",
                    "method": "exact paired sign-flip permutation p-value over seed-level differences",
                    "required_input": "same paired seed differences as primary test",
                    "claim_use": "must be directionally compatible with the primary claim",
                },
                {
                    "robustness_id": f"RBT-{comparison_id}-bootstrap-ci",
                    "comparison_id": comparison_id,
                    "candidate_family": candidate,
                    "baseline_family": baseline,
                    "test_family": "interval_sensitivity",
                    "method": "paired bootstrap CI for the mean few-balanced-accuracy difference",
                    "required_input": "same paired seed differences with fixed resampling seed",
                    "claim_use": "downgrade to sensitivity caveat if the interval crosses zero",
                },
                {
                    "robustness_id": f"RBT-{comparison_id}-sign-count",
                    "comparison_id": comparison_id,
                    "candidate_family": candidate,
                    "baseline_family": baseline,
                    "test_family": "direction_sensitivity",
                    "method": "paired sign-count table for positive, zero, and negative seed differences",
                    "required_input": "same paired seed differences as primary test",
                    "claim_use": "expose seed-level heterogeneity next to any aggregate claim",
                },
            ]
        )
    return pd.DataFrame(rows)


def build_missing_seed_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_id": "RBP-1-no-dropped-seed-claim",
                "status": "active",
                "rule": "no primary performance claim if any selected recipe lacks one of final seeds 20..29",
                "violation_response": "report incomplete final progress only and rerun missing seeds before claims",
            },
            {
                "policy_id": "RBP-2-paired-unit-lock",
                "status": "active",
                "rule": "all primary and robustness tests use seed as the pairing unit",
                "violation_response": "block unpaired or partially paired comparisons from the main claim",
            },
            {
                "policy_id": "RBP-3-failed-run-reporting",
                "status": "active",
                "rule": "failed final runs must be counted in the run registry with failure reason before rerun",
                "violation_response": "publish a missingness table and keep the final claim not_ready",
            },
            {
                "policy_id": "RBP-4-fixed-resampling-seed",
                "status": "active",
                "rule": "bootstrap and permutation code must use a fixed evaluator seed recorded in config",
                "violation_response": "rerun robustness artifacts with the fixed seed before citing sensitivity results",
            },
        ]
    )


def build_claim_sensitivity_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sensitivity_state": "robust_positive",
                "trigger": "primary Holm gate passes, sign-flip sensitivity is directionally compatible, bootstrap CI excludes zero, and all-class guardrail passes",
                "allowed_wording": "robust CIFAR-100-LT ResNet18 tuned final-performance improvement under the registered protocol",
                "blocked_wording": "broad long-tail optimizer superiority",
            },
            {
                "sensitivity_state": "parametric_only_positive",
                "trigger": "primary Holm gate passes but sign-flip or bootstrap sensitivity is weak",
                "allowed_wording": "parametric positive with robustness caveat",
                "blocked_wording": "robust optimizer-performance improvement",
            },
            {
                "sensitivity_state": "heterogeneous_seed_effect",
                "trigger": "aggregate mean is positive but sign-count shows mixed seed directions",
                "allowed_wording": "seed-heterogeneous final result with aggregate estimate",
                "blocked_wording": "uniform recipe improvement",
            },
            {
                "sensitivity_state": "negative_or_underpowered",
                "trigger": "primary and robustness tests do not support a positive claim",
                "allowed_wording": "negative or underpowered tuned benchmark boundary",
                "blocked_wording": "suppressing negative tuned outcomes",
            },
            {
                "sensitivity_state": "incomplete_or_unpaired",
                "trigger": "any selected recipe lacks complete paired seed rows",
                "allowed_wording": "incomplete final-run progress only",
                "blocked_wording": "final-performance claim",
            },
        ]
    )


def build_gate_matrix(
    final_plan: pd.DataFrame,
    comparisons: pd.DataFrame,
    robustness_tests: pd.DataFrame,
    missing_policy: pd.DataFrame,
) -> pd.DataFrame:
    selected_families = int(final_plan["final_status"].astype(str).eq("ready_for_final_run").sum())
    final_present = final_outputs_present()
    expected_robustness_rows = len(comparisons) * 4
    return pd.DataFrame(
        [
            {
                "gate_id": "RBP-G1-final-output-quarantine",
                "status": "pass" if not final_present else "fail",
                "evidence": "no final_claim files exist" if not final_present else "final_claim files already exist",
                "blocks_claim": "yes",
            },
            {
                "gate_id": "RBP-G2-primary-family-linked",
                "status": "pass" if len(comparisons) == 4 else "fail",
                "evidence": f"{len(comparisons)} primary comparisons inherited from final_analysis_plan",
                "blocks_claim": "yes",
            },
            {
                "gate_id": "RBP-G3-robustness-tests-registered",
                "status": "pass" if len(robustness_tests) == expected_robustness_rows else "fail",
                "evidence": f"{len(robustness_tests)} robustness rows for {len(comparisons)} primary comparisons",
                "blocks_claim": "yes",
            },
            {
                "gate_id": "RBP-G4-missing-seed-policy-active",
                "status": "pass" if missing_policy["status"].astype(str).eq("active").all() else "fail",
                "evidence": "dropped-seed, pairing, failed-run, and fixed-resampling policies active",
                "blocks_claim": "yes",
            },
            {
                "gate_id": "RBP-G5-final-execution-readiness",
                "status": "ready" if selected_families == 6 else "not_ready",
                "evidence": f"{selected_families}/6 recipe families ready for final run",
                "blocks_claim": "yes",
            },
        ]
    )


def write_discussion(
    robustness_tests: pd.DataFrame,
    missing_policy: pd.DataFrame,
    sensitivity_ladder: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    text = f"""# E11 CIFAR-100-LT Tuned Benchmark Final Robustness Plan

This generated plan pre-registers robustness checks for the final tuned benchmark before final outputs exist. It is a robustness plan, not a result: it links every primary Muon-vs-baseline comparison to parametric, exact sign-flip, bootstrap, and sign-count sensitivity checks, and it blocks any final-performance claim with missing or unpaired final seeds.

## Robustness Test Matrix

{markdown_table(robustness_tests, ["robustness_id", "comparison_id", "candidate_family", "baseline_family", "test_family", "method", "required_input", "claim_use"])}

## Missing Seed Policy

{markdown_table(missing_policy, ["policy_id", "status", "rule", "violation_response"])}

## Claim Sensitivity Ladder

{markdown_table(sensitivity_ladder, ["sensitivity_state", "trigger", "allowed_wording", "blocked_wording"])}

## Gate Matrix

{markdown_table(gate_matrix, ["gate_id", "status", "evidence", "blocks_claim"])}

## Operating Rule

The primary Holm-adjusted paired t-test remains the main gate, but a top-conference benchmark claim must report the nonparametric sensitivity checks. If the robustness checks disagree with the parametric gate, the manuscript must downgrade to a sensitivity caveat or negative/underpowered boundary rather than presenting a robust optimizer-performance improvement.
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    final_plan = pd.read_csv(SELECTION_DIR / "final_claim_plan.csv")
    comparisons = pd.read_csv(FINAL_ANALYSIS_DIR / "primary_comparison_family.csv")
    robustness_tests = build_robustness_test_matrix(comparisons)
    missing_policy = build_missing_seed_policy()
    sensitivity_ladder = build_claim_sensitivity_ladder()
    gate_matrix = build_gate_matrix(final_plan, comparisons, robustness_tests, missing_policy)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    robustness_tests.to_csv(OUTPUT_DIR / "robustness_test_matrix.csv", index=False)
    missing_policy.to_csv(OUTPUT_DIR / "missing_seed_policy.csv", index=False)
    sensitivity_ladder.to_csv(OUTPUT_DIR / "claim_sensitivity_ladder.csv", index=False)
    gate_matrix.to_csv(OUTPUT_DIR / "gate_matrix.csv", index=False)
    config = {
        "scope": "pre_final_statistical_robustness_plan",
        "claim_authority": "robustness_plan_only_until_final_outputs_exist",
        "primary_comparison_family": (FINAL_ANALYSIS_DIR / "primary_comparison_family.csv").as_posix(),
        "final_claim_plan": (SELECTION_DIR / "final_claim_plan.csv").as_posix(),
        "final_outputs_inspected": final_outputs_present(),
        "resampling_seed": 20260211,
        "primary_comparison_count": int(len(comparisons)),
        "robustness_test_rows": int(len(robustness_tests)),
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_discussion(robustness_tests, missing_policy, sensitivity_ladder, gate_matrix)
    print(f"saved tuned benchmark final robustness plan to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
