from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.reporting import markdown_table, write_markdown
from scripts.e11_write_natural_negative_power_audit import (
    ALPHA,
    LOG_RATIO_SDS,
    TARGET_POWERS,
    TRUE_RATIOS,
    minimum_detectable_ratio,
    power_one_sided,
)


OUTPUT_DIR = Path("results/e11_natural_negative_search_protocol/phase2_power_audit")
DISCUSSION_PATH = Path("discussion/e11_natural_negative_search_phase2_power_audit.md")
PHASE2_EVAL_DIR = Path("results/e11_natural_negative_search_protocol/phase2_multiplicity_evaluation")
SEED_COUNT = 3
FAMILY_SIZE = 8


def alpha_rows() -> list[dict[str, object]]:
    return [
        {
            "alpha_scope": "raw_single_setting",
            "one_sided_alpha": ALPHA,
            "family_size": 1,
            "role": "diagnostic raw p-value only; cannot unlock a phase2 held-out architecture claim",
        },
        {
            "alpha_scope": "holm_bonferroni_worst_case",
            "one_sided_alpha": ALPHA / FAMILY_SIZE,
            "family_size": FAMILY_SIZE,
            "role": "conservative first-rejection threshold for the 8-setting phase2 ResNet34 family",
        },
    ]


def build_power_grid() -> pd.DataFrame:
    rows = []
    for alpha_spec in alpha_rows():
        alpha = float(alpha_spec["one_sided_alpha"])
        for log_sd in LOG_RATIO_SDS:
            for true_ratio in TRUE_RATIOS:
                rows.append(
                    {
                        "alpha_scope": alpha_spec["alpha_scope"],
                        "seed_count": SEED_COUNT,
                        "family_size": int(alpha_spec["family_size"]),
                        "one_sided_alpha": alpha,
                        "log_ratio_sd": log_sd,
                        "true_tail_drift_ratio": true_ratio,
                        "true_log_ratio": math.log(true_ratio),
                        "power": power_one_sided(
                            true_log_ratio=math.log(true_ratio),
                            log_ratio_sd=log_sd,
                            seed_count=SEED_COUNT,
                            alpha=alpha,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_mde_table() -> pd.DataFrame:
    rows = []
    for alpha_spec in alpha_rows():
        alpha = float(alpha_spec["one_sided_alpha"])
        for log_sd in LOG_RATIO_SDS:
            for target_power in TARGET_POWERS:
                mde_ratio = minimum_detectable_ratio(
                    log_ratio_sd=log_sd,
                    seed_count=SEED_COUNT,
                    alpha=alpha,
                    target_power=target_power,
                )
                rows.append(
                    {
                        "alpha_scope": alpha_spec["alpha_scope"],
                        "seed_count": SEED_COUNT,
                        "family_size": int(alpha_spec["family_size"]),
                        "one_sided_alpha": alpha,
                        "log_ratio_sd": log_sd,
                        "target_power": target_power,
                        "minimum_detectable_ratio": mde_ratio,
                        "minimum_detectable_log_ratio": math.log(mde_ratio)
                        if math.isfinite(mde_ratio)
                        else math.inf,
                        "claim_interpretation": (
                            "phase2 null is informative only for held-out architecture effects at or above "
                            "this audited detectable scale under the assumed across-seed log-ratio SD"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_interpretation_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "P2-PWR-1-adjusted-positive",
                "evidence_condition": "complete 8-setting ResNet34 family; Holm-adjusted primary p <= 0.05; quality gates pass",
                "allowed_wording": "held-out architecture primary boundary candidate",
                "blocked_wording": "broad natural counterexample claim or final optimizer-performance claim",
            },
            {
                "case_id": "P2-PWR-2-complete-null-above-mde",
                "evidence_condition": "complete 8-setting family; no adjusted primary row; observed SD makes target effect above the MDE",
                "allowed_wording": "finite ResNet34 held-out architecture null for effects at or above the audited detectable scale",
                "blocked_wording": "absence of smaller ResNet34 effects or absence of natural counterexamples in other architectures",
            },
            {
                "case_id": "P2-PWR-3-complete-null-below-mde",
                "evidence_condition": "complete 8-setting family; no adjusted primary row; target effect below MDE",
                "allowed_wording": "underpowered phase2 null for small held-out architecture effects",
                "blocked_wording": "convincing held-out architecture finite null",
            },
            {
                "case_id": "P2-PWR-4-phase1-null-phase2-positive",
                "evidence_condition": "phase1 finite-null candidate but phase2 has an adjusted primary worse row",
                "allowed_wording": "architecture-transport boundary case requiring mechanism analysis",
                "blocked_wording": "discarding phase2 as outlier or claiming phase1 generality",
            },
            {
                "case_id": "P2-PWR-5-incomplete-family",
                "evidence_condition": "any declared phase2 metric output missing",
                "allowed_wording": "registered held-out architecture search in progress",
                "blocked_wording": "phase2 counterexample, phase2 finite null, or selected-subset claim",
            },
        ]
    )


def phase2_current_context() -> dict[str, object]:
    decisions_path = PHASE2_EVAL_DIR / "primary_decisions.csv"
    gates_path = PHASE2_EVAL_DIR / "gate_report.csv"
    if not decisions_path.exists() or not gates_path.exists():
        return {
            "current_state_id": "P2-S1-not-run",
            "observed_count": 0,
            "adjusted_worse_count": 0,
            "head_gain_fail_count": 0,
            "tail_quality_pass_count": 0,
            "gate_status": "not_ready",
            "current_evidence": "phase2 multiplicity evaluation outputs are absent",
        }

    decisions = pd.read_csv(decisions_path)
    gates = pd.read_csv(gates_path)
    gate_status = gates.set_index("gate_id")["status"].astype(str).to_dict()
    observed_count = int(decisions["output_status"].astype(str).eq("observed").sum())
    adjusted_worse_count = int(
        decisions["adjusted_primary_decision"].astype(str).eq("primary_worse_adjusted").sum()
    )
    head_gain_fail_count = int(decisions["head_gain_gate"].astype(str).str.lower().eq("false").sum())
    tail_quality_pass_count = int(decisions["tail_quality_gate"].astype(str).str.lower().eq("true").sum())
    quality_pass_adjusted = int(
        decisions[
            decisions["adjusted_primary_decision"].astype(str).eq("primary_worse_adjusted")
            & decisions["quality_gate"].astype(str).str.lower().eq("true")
        ].shape[0]
    )

    if observed_count == 0:
        current_state_id = "P2-S1-not-run"
    elif observed_count < FAMILY_SIZE:
        current_state_id = "P2-S2-partial"
    elif adjusted_worse_count and quality_pass_adjusted:
        current_state_id = "P2-S3-adjusted-positive"
    elif adjusted_worse_count:
        current_state_id = "P2-S6-quality-failure"
    elif head_gain_fail_count == FAMILY_SIZE and tail_quality_pass_count == FAMILY_SIZE:
        current_state_id = "P2-S7-complete-null-head-gain-caveat"
    else:
        current_state_id = "P2-S4-complete-null-above-mde"

    return {
        "current_state_id": current_state_id,
        "observed_count": observed_count,
        "adjusted_worse_count": adjusted_worse_count,
        "head_gain_fail_count": head_gain_fail_count,
        "tail_quality_pass_count": tail_quality_pass_count,
        "gate_status": gate_status.get("NNS-P2-E4-heldout-architecture-claim", "missing"),
        "current_evidence": (
            f"{observed_count}/{FAMILY_SIZE} observed; adjusted_worse_rows={adjusted_worse_count}; "
            f"head_gain_gate_fail_rows={head_gain_fail_count}; tail_quality_gate_pass_rows={tail_quality_pass_count}; "
            f"NNS-P2-E4={gate_status.get('NNS-P2-E4-heldout-architecture-claim', 'missing')}"
        ),
    }


def build_outcome_state_machine(context: dict[str, object]) -> pd.DataFrame:
    current_state_id = str(context["current_state_id"])
    current_evidence = str(context["current_evidence"])
    rows = [
        {
            "state_id": "P2-S1-not-run",
            "trigger": "0/8 phase2 primary metric rows",
            "claim_state": "not_ready",
            "required_action": "wait for Slurm outputs; do not inspect partial settings for claims",
        },
        {
            "state_id": "P2-S2-partial",
            "trigger": "1-7/8 phase2 primary metric rows",
            "claim_state": "not_ready_partial_family",
            "required_action": "report partial rows only as progress; keep claim gate closed",
        },
        {
            "state_id": "P2-S3-adjusted-positive",
            "trigger": "8/8 rows; at least one Holm-adjusted primary worse row passes quality gates",
            "claim_state": "heldout_architecture_boundary_candidate",
            "required_action": "run mechanism analysis and forbid broad optimizer-performance wording",
        },
        {
            "state_id": "P2-S4-complete-null-above-mde",
            "trigger": "8/8 rows; no adjusted positive; target effect is above phase2 MDE",
            "claim_state": "finite_phase2_null_candidate_with_detectable_effect_caveat",
            "required_action": "state effect-size floor and keep claim within ResNet34 registered family",
        },
        {
            "state_id": "P2-S5-complete-null-below-mde",
            "trigger": "8/8 rows; no adjusted positive; target effect is below phase2 MDE",
            "claim_state": "underpowered_phase2_null",
            "required_action": "do not use as strong held-out null; add seeds or larger search family",
        },
        {
            "state_id": "P2-S6-quality-failure",
            "trigger": "adjusted positive exists only in rows failing head-gain or tail-quality gates",
            "claim_state": "quality_caveated_boundary",
            "required_action": "treat as diagnostic failure mode, not as primary natural counterexample",
        },
        {
            "state_id": "P2-S7-complete-null-head-gain-caveat",
            "trigger": "8/8 rows; no adjusted positive; all rows fail head-gain while tail-quality passes",
            "claim_state": "finite_phase2_null_candidate_with_head_gain_and_detectable_effect_caveats",
            "required_action": "report a bounded ResNet34 finite-null candidate, not mechanism validation or a universal natural null",
        },
    ]
    return pd.DataFrame(
        [
            {
                **row,
                "current_match": "yes" if row["state_id"] == current_state_id else "no",
                "current_evidence": current_evidence if row["state_id"] == current_state_id else "",
            }
            for row in rows
        ]
    )


def write_discussion(
    power_grid: pd.DataFrame,
    mde_table: pd.DataFrame,
    ladder: pd.DataFrame,
    state_machine: pd.DataFrame,
    context: dict[str, object],
) -> None:
    adjusted_mde = mde_table[
        (mde_table["alpha_scope"] == "holm_bonferroni_worst_case")
        & (mde_table["target_power"].isin([0.80, 0.90]))
    ].copy()
    adjusted_mde = adjusted_mde[adjusted_mde["log_ratio_sd"].isin([0.10, 0.20, 0.35, 0.50])]
    adjusted_power = power_grid[
        (power_grid["alpha_scope"] == "holm_bonferroni_worst_case")
        & (power_grid["log_ratio_sd"].isin([0.10, 0.20, 0.35]))
        & (power_grid["true_tail_drift_ratio"].isin([1.25, 1.50, 2.00, 3.00]))
    ].copy()
    text = f"""# E11 Natural Negative Search Phase2 Power Audit

This generated audit records the detectable-effect and interpretation boundary
for the registered 8-setting ResNet34 held-out architecture phase2 family. It is
a frozen design audit plus post-output readout: the detectable-effect grid and
state machine were fixed before phase2 outcomes, and the current row now reads
the completed multiplicity evaluator without changing thresholds or claims.

The primary test is the paired across-seed log ratio for
`tail_output_drift_sq_ratio_spectral_over_fro`, with a one-sided worse-than-one
alternative and Holm-adjusted phase-family decision. With only 3 seeds per
setting, the phase2 family is mainly a held-out architecture stress test for
large effects; null results below the audited MDE are explicitly underpowered.
The registered seed count is 3 seeds per setting.

## Current Post-Output Reading

Current state: `{context["current_state_id"]}`. Evidence:
{context["current_evidence"]}. This supports only a bounded ResNet34
held-out-architecture finite-null candidate with detectable-effect and head-gain
caveats; it is not a natural counterexample, mechanism validation, or universal
natural finite null.

## Adjusted Minimum Detectable Ratio

{markdown_table(adjusted_mde, ["log_ratio_sd", "target_power", "minimum_detectable_ratio", "minimum_detectable_log_ratio", "alpha_scope"])}

## Adjusted Power Grid

{markdown_table(adjusted_power, ["log_ratio_sd", "true_tail_drift_ratio", "power", "alpha_scope"])}

## Interpretation Ladder

{markdown_table(ladder, ["case_id", "evidence_condition", "allowed_wording", "blocked_wording"])}

## Outcome State Machine

{markdown_table(state_machine, ["state_id", "trigger", "claim_state", "required_action", "current_match", "current_evidence"])}

Generated tables:

- [power_grid.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/power_grid.csv)
- [minimum_detectable_effect.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/minimum_detectable_effect.csv)
- [interpretation_ladder.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/interpretation_ladder.csv)
- [outcome_state_machine.csv](../results/e11_natural_negative_search_protocol/phase2_power_audit/outcome_state_machine.csv)
"""
    write_markdown(DISCUSSION_PATH, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    power_grid = build_power_grid()
    mde_table = build_mde_table()
    ladder = build_interpretation_ladder()
    context = phase2_current_context()
    state_machine = build_outcome_state_machine(context)
    power_grid.to_csv(OUTPUT_DIR / "power_grid.csv", index=False)
    mde_table.to_csv(OUTPUT_DIR / "minimum_detectable_effect.csv", index=False)
    ladder.to_csv(OUTPUT_DIR / "interpretation_ladder.csv", index=False)
    state_machine.to_csv(OUTPUT_DIR / "outcome_state_machine.csv", index=False)
    write_discussion(power_grid, mde_table, ladder, state_machine, context)
    print(f"saved natural negative-search phase2 power audit to {OUTPUT_DIR} and {DISCUSSION_PATH}")


if __name__ == "__main__":
    main()
